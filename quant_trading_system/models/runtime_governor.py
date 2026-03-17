"""
Runtime champion/challenger governor for live model allocation.

Turns offline deployment plans into enforceable runtime behavior:
- Resolves champion/challenger from registry + active model pointer
- Routes symbol-level model traffic through canary phases
- Scales challenger capital by deployment-plan canary fractions
- Quarantines challengers on execution/risk guardrail breaches
- Optionally promotes challengers after successful canary completion

This module is intentionally conservative. When routing metadata is missing or
ambiguous it falls back to the current champion, or leaves predictions
unmanaged rather than forcing an unsafe promotion.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_trading_system.core.data_types import ModelPrediction
from quant_trading_system.core.events import EventBus, EventType, create_system_event
from quant_trading_system.monitoring.audit import AuditLogger
from quant_trading_system.models.staleness_detector import ModelHealth, ModelStalenessDetector
from quant_trading_system.models.training_lineage import (
    load_registry_entries,
    persist_active_model_pointer,
    set_registry_active_version,
)

if TYPE_CHECKING:
    from quant_trading_system.trading.signal_generator import EnrichedSignal

logger = logging.getLogger(__name__)


def _parse_datetime(value: Any) -> datetime | None:
    """Parse registry timestamps as timezone-aware UTC datetimes."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float, returning default on parse failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert value to int, returning default on parse failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class CanaryPhase:
    """Single canary rollout phase from the deployment plan."""

    phase: int
    capital_fraction: float
    min_trades: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize phase for metrics/state APIs."""
        return {
            "phase": self.phase,
            "capital_fraction": self.capital_fraction,
            "min_trades": self.min_trades,
        }


@dataclass(frozen=True)
class ManagedModelEntry:
    """Registry-backed model entry used by runtime governance."""

    model_name: str
    version_id: str
    registered_at: datetime | None = None
    deployment_plan: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def model_key(self) -> str:
        """Canonical unique identifier for the deployed model version."""
        return f"{self.model_name}:{self.version_id}" if self.version_id else self.model_name

    @property
    def ready_for_production(self) -> bool:
        """Whether the deployment plan marks this entry as production-ready."""
        return bool(self.deployment_plan.get("ready_for_production", False))

    @classmethod
    def from_registry_entry(cls, payload: dict[str, Any]) -> "ManagedModelEntry | None":
        """Build a managed entry from flattened registry payload."""
        model_name = str(payload.get("model_name", "")).strip()
        version_id = str(payload.get("version_id", "")).strip()
        if not model_name:
            return None

        aliases: list[str] = [model_name]
        if version_id:
            aliases.extend(
                [
                    f"{model_name}:{version_id}",
                    f"{model_name}@{version_id}",
                    version_id,
                ]
            )

        deployment_plan = payload.get("deployment_plan", {})
        metrics = payload.get("metrics", {})
        return cls(
            model_name=model_name,
            version_id=version_id,
            registered_at=_parse_datetime(payload.get("registered_at")),
            deployment_plan=deployment_plan if isinstance(deployment_plan, dict) else {},
            metrics=metrics if isinstance(metrics, dict) else {},
            aliases=tuple(dict.fromkeys(alias for alias in aliases if alias)),
        )


@dataclass(frozen=True)
class RuntimeRoutingDecision:
    """Resolved routing decision for a symbol at runtime."""

    symbol: str
    role: str
    selected_model: str | None
    selected_prediction_name: str | None
    capital_fraction: float
    phase: int | None
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize routing decision for logs/metrics."""
        return {
            "symbol": self.symbol,
            "role": self.role,
            "selected_model": self.selected_model,
            "selected_prediction_name": self.selected_prediction_name,
            "capital_fraction": self.capital_fraction,
            "phase": self.phase,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RuntimeModelStats:
    """Execution telemetry tracked per deployed model version."""

    model_key: str
    signals_routed: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    cumulative_slippage_bps: float = 0.0
    quarantined: bool = False
    quarantine_reason: str | None = None
    last_updated: datetime | None = None

    @property
    def avg_slippage_bps(self) -> float:
        """Average slippage over filled orders."""
        if self.orders_filled <= 0:
            return 0.0
        return self.cumulative_slippage_bps / self.orders_filled

    @property
    def rejection_rate(self) -> float:
        """Rejection rate over submitted orders."""
        if self.orders_submitted <= 0:
            return 0.0
        return self.orders_rejected / self.orders_submitted

    def touch(self) -> None:
        """Refresh last-updated timestamp."""
        self.last_updated = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Serialize runtime model metrics."""
        return {
            "model_key": self.model_key,
            "signals_routed": self.signals_routed,
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "avg_slippage_bps": self.avg_slippage_bps,
            "rejection_rate": self.rejection_rate,
            "quarantined": self.quarantined,
            "quarantine_reason": self.quarantine_reason,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


@dataclass
class RuntimeGovernorConfig:
    """Configuration for runtime governance behavior."""

    models_root: Path = Path("models")
    runtime_mode: str = "paper"
    auto_promote: bool = True
    allow_persistent_promotion: bool = False
    refresh_interval_seconds: int = 30
    min_orders_for_guardrail: int = 5
    max_challenger_rejection_rate: float = 0.25
    default_canary_fraction: float = 0.05
    default_canary_min_trades: int = 25

    @property
    def registry_root(self) -> Path:
        """Registry directory under the models root."""
        return self.models_root / "registry"

    @property
    def active_pointer_path(self) -> Path:
        """Champion pointer compatible with training/dashboard flows."""
        return self.models_root / "active_model.json"


class ModelRuntimeGovernor:
    """Champion/challenger runtime governor for live model routing."""

    def __init__(
        self,
        config: RuntimeGovernorConfig | None = None,
        event_bus: EventBus | None = None,
        staleness_detector: ModelStalenessDetector | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.config = config or RuntimeGovernorConfig()
        self.event_bus = event_bus
        self.staleness_detector = staleness_detector or ModelStalenessDetector()
        self.audit_logger = audit_logger

        self._lock = threading.RLock()
        self._champion_entry: ManagedModelEntry | None = None
        self._challenger_entry: ManagedModelEntry | None = None
        self._canary_phases: list[CanaryPhase] = []
        self._challenger_phase_index: int = 0
        self._model_stats: dict[str, RuntimeModelStats] = {}
        self._last_routing_decisions: dict[str, RuntimeRoutingDecision] = {}
        self._last_refresh: datetime | None = None

    def refresh_deployment_state(self, force: bool = False) -> dict[str, Any]:
        """Reload champion/challenger state from registry and active pointer."""
        with self._lock:
            now = datetime.now(timezone.utc)
            if (
                not force
                and self._last_refresh is not None
                and (now - self._last_refresh).total_seconds() < self.config.refresh_interval_seconds
            ):
                return self.get_statistics()

            rows = load_registry_entries(self.config.registry_root)
            entries = [
                entry
                for row in rows
                if isinstance(row, dict)
                for entry in [ManagedModelEntry.from_registry_entry(row)]
                if entry is not None
            ]

            champion = self._resolve_champion(entries)
            challenger = self._resolve_challenger(entries, champion)
            signature_before = (
                self._champion_entry.model_key if self._champion_entry else None,
                self._challenger_entry.model_key if self._challenger_entry else None,
            )
            signature_after = (
                champion.model_key if champion else None,
                challenger.model_key if challenger else None,
            )

            self._champion_entry = champion
            if signature_before[1] != signature_after[1]:
                self._challenger_phase_index = 0
            self._challenger_entry = challenger
            self._canary_phases = self._extract_canary_phases(challenger)
            self._last_refresh = now

            for entry in (champion, challenger):
                if entry is None:
                    continue
                self._ensure_model_registered(entry)
                self._stats_for(entry.model_key)

            if signature_before != signature_after:
                self._publish_runtime_event(
                    event_type=EventType.MODEL_RELOAD,
                    message="Runtime model deployment state refreshed",
                    details={
                        "champion": champion.model_key if champion else None,
                        "challenger": challenger.model_key if challenger else None,
                    },
                )
                self._audit_deployment_state(champion, challenger)

            return self.get_statistics()

    def route_predictions(
        self,
        predictions: dict[str, ModelPrediction],
    ) -> dict[str, ModelPrediction]:
        """Select champion/challenger predictions per symbol for signal generation."""
        self.refresh_deployment_state()
        if not predictions:
            return {}

        grouped: dict[str, list[tuple[str, ModelPrediction]]] = {}
        for key, prediction in predictions.items():
            grouped.setdefault(prediction.symbol.upper(), []).append((key, prediction))

        routed: dict[str, ModelPrediction] = {}
        with self._lock:
            for symbol, candidates in grouped.items():
                decision = self._route_for_symbol(symbol)
                selected_candidates, resolved_decision = self._select_prediction_candidates(
                    candidates,
                    decision,
                )
                self._last_routing_decisions[symbol] = resolved_decision
                for key, prediction in selected_candidates:
                    routed[key] = prediction

        return routed

    def govern_signals(self, signals: list[EnrichedSignal]) -> list[EnrichedSignal]:
        """Annotate signals with runtime routing metadata and capital scaling."""
        if not signals:
            return signals

        self.refresh_deployment_state()
        with self._lock:
            for signal in signals:
                symbol = signal.signal.symbol.upper()
                decision = self._last_routing_decisions.get(symbol) or self._route_for_symbol(symbol)
                signal_models = self._extract_signal_models(signal)
                if signal_models:
                    decision = self._resolve_signal_decision(decision, signal_models)

                metadata = dict(signal.signal.metadata)
                metadata.update(
                    {
                        "runtime_model_role": decision.role,
                        "runtime_selected_model": decision.selected_model,
                        "runtime_selected_prediction_name": decision.selected_prediction_name,
                        "runtime_position_scale": decision.capital_fraction if decision.role == "challenger" else 1.0,
                        "runtime_canary_phase": decision.phase,
                        "runtime_routing_reason": decision.reason,
                    }
                )
                signal.signal = signal.signal.model_copy(update={"metadata": metadata})

                if decision.selected_model:
                    stats = self._stats_for(decision.selected_model)
                    stats.signals_routed += 1
                    stats.touch()

        return signals

    def record_prediction_outcome(
        self,
        model_name: str,
        prediction: float,
        actual_return: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Feed realized outcomes back into the staleness detector."""
        self.refresh_deployment_state()
        with self._lock:
            model_key = self._resolve_model_key(model_name)
            health = self.staleness_detector.add_observation(
                model_key,
                prediction=prediction,
                actual_return=actual_return,
                timestamp=timestamp,
            )
            if (
                self._challenger_entry is not None
                and model_key == self._challenger_entry.model_key
                and not self.staleness_detector.is_usable(model_key)
            ):
                self._quarantine_challenger_locked(
                    reason=f"staleness_detector:{health.value}",
                    current_value=_safe_float(actual_return),
                    limit_value=0.0,
                )

    def record_order_submission(self, metadata: dict[str, Any] | None) -> None:
        """Record a submitted order for runtime guardrail monitoring."""
        model_key, role = self._extract_model_from_metadata(metadata)
        if not model_key:
            return

        with self._lock:
            stats = self._stats_for(model_key)
            stats.orders_submitted += 1
            stats.touch()
            if role == "challenger":
                self._evaluate_challenger_guardrails_locked(stats)

    def record_order_fill(
        self,
        metadata: dict[str, Any] | None,
        fill_price: Decimal,
        expected_price: Decimal | None = None,
    ) -> None:
        """Record a fill and evaluate challenger execution quality."""
        model_key, role = self._extract_model_from_metadata(metadata)
        if not model_key:
            return

        with self._lock:
            stats = self._stats_for(model_key)
            stats.orders_filled += 1
            stats.touch()

            if expected_price and expected_price > 0:
                slippage_bps = abs(float((fill_price - expected_price) / expected_price * Decimal("10000")))
                stats.cumulative_slippage_bps += slippage_bps

            if role == "challenger":
                self._evaluate_challenger_guardrails_locked(stats)
                self._advance_canary_phase_locked(stats)

    def record_order_rejection(
        self,
        metadata: dict[str, Any] | None,
        reason: str = "",
    ) -> None:
        """Record a rejection and evaluate challenger rejection-rate guardrails."""
        model_key, role = self._extract_model_from_metadata(metadata)
        if not model_key:
            return

        with self._lock:
            stats = self._stats_for(model_key)
            stats.orders_rejected += 1
            stats.touch()
            if role == "challenger":
                self._evaluate_challenger_guardrails_locked(
                    stats,
                    rejection_reason=reason or "broker_rejection",
                )

    def record_engine_risk_snapshot(
        self,
        current_drawdown: float,
        max_drawdown: float,
        daily_pnl_pct: float | None = None,
    ) -> None:
        """Evaluate conservative drawdown guardrails against the challenger."""
        self.refresh_deployment_state()
        with self._lock:
            if self._challenger_entry is None or self._is_quarantined_locked(self._challenger_entry.model_key):
                return

            limit = self._challenger_drawdown_limit_locked()
            if limit <= 0:
                return

            breach_value = max(current_drawdown, max_drawdown)
            if breach_value > limit:
                self._quarantine_challenger_locked(
                    reason="drawdown_guardrail_breach",
                    current_value=breach_value,
                    limit_value=limit,
                    extra_details={"daily_pnl_pct": daily_pnl_pct},
                )

    def get_routing_decision(self, symbol: str) -> RuntimeRoutingDecision | None:
        """Get the most recent routing decision for a symbol."""
        with self._lock:
            return self._last_routing_decisions.get(symbol.upper())

    def get_statistics(self) -> dict[str, Any]:
        """Expose runtime governor state for engine/dashboard observability."""
        with self._lock:
            champion = self._champion_entry.model_key if self._champion_entry else None
            challenger = self._challenger_entry.model_key if self._challenger_entry else None
            phase = None
            if self._challenger_entry is not None and self._canary_phases:
                phase = self._active_phase_locked().to_dict()

            return {
                "mode": self.config.runtime_mode,
                "allow_persistent_promotion": self.config.allow_persistent_promotion,
                "champion": champion,
                "challenger": challenger,
                "active_canary_phase": phase,
                "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
                "models": {
                    model_key: stats.to_dict()
                    for model_key, stats in self._model_stats.items()
                },
                "routes": {
                    symbol: decision.to_dict()
                    for symbol, decision in self._last_routing_decisions.items()
                },
            }

    def _resolve_champion(
        self,
        entries: list[ManagedModelEntry],
    ) -> ManagedModelEntry | None:
        """Resolve the champion from active pointer, active flag, or recency."""
        pointer_payload: dict[str, Any] = {}
        if self.config.active_pointer_path.exists():
            try:
                with self.config.active_pointer_path.open("r", encoding="utf-8") as handle:
                    parsed = json.load(handle)
                    if isinstance(parsed, dict):
                        pointer_payload = parsed
            except Exception as exc:
                logger.warning(f"Failed to read active model pointer: {exc}")

        pointer_model = str(pointer_payload.get("model_name", "")).strip()
        pointer_version = str(pointer_payload.get("version_id", "")).strip()
        for entry in entries:
            if entry.model_name == pointer_model and entry.version_id == pointer_version:
                return entry

        for entry in entries:
            if entry.deployment_plan.get("is_active") or entry.metrics.get("is_active"):
                return entry

        return entries[0] if entries else None

    def _resolve_challenger(
        self,
        entries: list[ManagedModelEntry],
        champion: ManagedModelEntry | None,
    ) -> ManagedModelEntry | None:
        """Resolve the most recent ready challenger distinct from the champion."""
        candidates = [
            entry
            for entry in entries
            if entry.ready_for_production
            and champion is not None
            and entry.model_key != champion.model_key
        ]
        return candidates[0] if candidates else None

    def _extract_canary_phases(
        self,
        challenger: ManagedModelEntry | None,
    ) -> list[CanaryPhase]:
        """Build canary phases from the challenger deployment plan."""
        if challenger is None:
            return []

        raw_phases = challenger.deployment_plan.get("canary_rollout", [])
        phases: list[CanaryPhase] = []
        if isinstance(raw_phases, list):
            for idx, payload in enumerate(raw_phases, start=1):
                if not isinstance(payload, dict):
                    continue
                phases.append(
                    CanaryPhase(
                        phase=_safe_int(payload.get("phase"), idx),
                        capital_fraction=max(0.0, min(1.0, _safe_float(payload.get("capital_fraction"), 0.0))),
                        min_trades=max(0, _safe_int(payload.get("min_trades"), 0)),
                    )
                )

        if not phases:
            phases = [
                CanaryPhase(
                    phase=1,
                    capital_fraction=self.config.default_canary_fraction,
                    min_trades=self.config.default_canary_min_trades,
                )
            ]

        phases.sort(key=lambda item: item.phase)
        return phases

    def _ensure_model_registered(self, entry: ManagedModelEntry) -> None:
        """Register all known aliases with the staleness detector."""
        for alias in dict.fromkeys((entry.model_key, *entry.aliases)):
            self.staleness_detector.register_model(alias)

    def _route_for_symbol(self, symbol: str) -> RuntimeRoutingDecision:
        """Resolve champion/challenger route for a symbol."""
        if self._champion_entry is None:
            return RuntimeRoutingDecision(
                symbol=symbol,
                role="unmanaged",
                selected_model=None,
                selected_prediction_name=None,
                capital_fraction=1.0,
                phase=None,
                reason="no_champion_configured",
            )

        if self._challenger_entry is None or not self._canary_phases:
            return RuntimeRoutingDecision(
                symbol=symbol,
                role="champion",
                selected_model=self._champion_entry.model_key,
                selected_prediction_name=None,
                capital_fraction=1.0,
                phase=None,
                reason="no_ready_challenger",
            )

        if self._is_quarantined_locked(self._challenger_entry.model_key):
            return RuntimeRoutingDecision(
                symbol=symbol,
                role="champion",
                selected_model=self._champion_entry.model_key,
                selected_prediction_name=None,
                capital_fraction=1.0,
                phase=self._active_phase_locked().phase,
                reason="challenger_quarantined",
            )

        phase = self._active_phase_locked()
        if self._is_in_canary_bucket(symbol, phase.capital_fraction):
            return RuntimeRoutingDecision(
                symbol=symbol,
                role="challenger",
                selected_model=self._challenger_entry.model_key,
                selected_prediction_name=None,
                capital_fraction=phase.capital_fraction,
                phase=phase.phase,
                reason="symbol_in_canary_bucket",
            )

        return RuntimeRoutingDecision(
            symbol=symbol,
            role="champion",
            selected_model=self._champion_entry.model_key,
            selected_prediction_name=None,
            capital_fraction=1.0,
            phase=phase.phase,
            reason="symbol_outside_canary_bucket",
        )

    def _select_prediction_candidates(
        self,
        candidates: list[tuple[str, ModelPrediction]],
        decision: RuntimeRoutingDecision,
    ) -> tuple[list[tuple[str, ModelPrediction]], RuntimeRoutingDecision]:
        """Select the routed prediction subset for a symbol."""
        selected = self._match_entry_predictions(candidates, decision.selected_model)
        if selected:
            selected_prediction_name = selected[0][1].model_name
            return selected, replace(decision, selected_prediction_name=selected_prediction_name)

        if decision.role == "challenger" and self._champion_entry is not None:
            champion_fallback = self._match_entry_predictions(candidates, self._champion_entry.model_key)
            if champion_fallback:
                return champion_fallback, RuntimeRoutingDecision(
                    symbol=decision.symbol,
                    role="champion",
                    selected_model=self._champion_entry.model_key,
                    selected_prediction_name=champion_fallback[0][1].model_name,
                    capital_fraction=1.0,
                    phase=decision.phase,
                    reason="challenger_prediction_missing_fallback_to_champion",
                )

        return candidates, replace(
            decision,
            role="unmanaged",
            selected_model=None,
            selected_prediction_name=None,
            capital_fraction=1.0,
            reason=f"{decision.reason}:no_alias_match",
        )

    def _match_entry_predictions(
        self,
        candidates: list[tuple[str, ModelPrediction]],
        model_key: str | None,
    ) -> list[tuple[str, ModelPrediction]]:
        """Match predictions to the champion/challenger alias set."""
        entry = self._entry_from_model_key(model_key)
        if entry is None:
            return []

        aliases = set(entry.aliases) | {entry.model_key}
        return [
            (key, prediction)
            for key, prediction in candidates
            if prediction.model_name in aliases
        ]

    def _resolve_signal_decision(
        self,
        decision: RuntimeRoutingDecision,
        signal_models: set[str],
    ) -> RuntimeRoutingDecision:
        """Reconcile the routing decision with signal-level model metadata."""
        if decision.selected_model is None:
            return decision

        entry = self._entry_from_model_key(decision.selected_model)
        aliases = set(entry.aliases) | {entry.model_key} if entry is not None else {decision.selected_model}
        if signal_models & aliases:
            return replace(
                decision,
                selected_prediction_name=sorted(signal_models & aliases)[0],
            )

        if decision.role == "challenger" and self._champion_entry is not None:
            champion_aliases = set(self._champion_entry.aliases) | {self._champion_entry.model_key}
            if signal_models & champion_aliases:
                return RuntimeRoutingDecision(
                    symbol=decision.symbol,
                    role="champion",
                    selected_model=self._champion_entry.model_key,
                    selected_prediction_name=sorted(signal_models & champion_aliases)[0],
                    capital_fraction=1.0,
                    phase=decision.phase,
                    reason="signal_missing_challenger_alias_fallback_to_champion",
                )

        return replace(
            decision,
            role="unmanaged",
            selected_model=None,
            selected_prediction_name=None,
            capital_fraction=1.0,
            reason=f"{decision.reason}:signal_alias_unmatched",
        )

    def _extract_signal_models(self, signal: EnrichedSignal) -> set[str]:
        """Collect model aliases embedded in signal metadata/features."""
        models: set[str] = set()
        payload = signal.signal.metadata if isinstance(signal.signal.metadata, dict) else {}
        raw_names = payload.get("prediction_model_names", [])
        if isinstance(raw_names, list):
            models.update(str(item) for item in raw_names if item)

        raw_feature_models = signal.signal.features_snapshot.get("model_predictions", {})
        if isinstance(raw_feature_models, dict):
            models.update(str(item) for item in raw_feature_models.keys() if item)

        return models

    def _active_phase_locked(self) -> CanaryPhase:
        """Return the current canary phase under lock."""
        if not self._canary_phases:
            return CanaryPhase(phase=1, capital_fraction=1.0, min_trades=0)
        idx = max(0, min(self._challenger_phase_index, len(self._canary_phases) - 1))
        return self._canary_phases[idx]

    def _advance_canary_phase_locked(self, challenger_stats: RuntimeModelStats) -> None:
        """Advance canary phase or promote challenger after sufficient fills."""
        if self._challenger_entry is None or challenger_stats.quarantined:
            return

        while self._challenger_phase_index < len(self._canary_phases):
            active_phase = self._active_phase_locked()
            if challenger_stats.orders_filled < active_phase.min_trades:
                return

            if self._challenger_phase_index + 1 < len(self._canary_phases):
                self._challenger_phase_index += 1
                next_phase = self._active_phase_locked()
                self._publish_runtime_event(
                    event_type=EventType.MODEL_RELOAD,
                    message="Advanced challenger canary phase",
                    details={
                        "challenger": self._challenger_entry.model_key,
                        "phase": next_phase.phase,
                        "capital_fraction": next_phase.capital_fraction,
                        "filled_orders": challenger_stats.orders_filled,
                    },
                )
                continue

            if self.config.auto_promote:
                self._promote_challenger_locked()
            return

    def _promote_challenger_locked(self) -> None:
        """Persist challenger as champion when canary rollout completes."""
        if self._challenger_entry is None:
            return

        if not self.config.allow_persistent_promotion or self.config.runtime_mode != "live":
            self._publish_runtime_event(
                event_type=EventType.SYSTEM_ALERT,
                message="Challenger completed canary but persistent promotion is disabled",
                details={"challenger": self._challenger_entry.model_key},
            )
            return

        previous_champion = self._champion_entry
        registry_updated = set_registry_active_version(
            self.config.registry_root,
            self._challenger_entry.model_name,
            self._challenger_entry.version_id,
        )
        if not registry_updated:
            logger.warning(
                "Failed to mark challenger active in registry: %s",
                self._challenger_entry.model_key,
            )

        persist_active_model_pointer(
            self.config.models_root,
            model_name=self._challenger_entry.model_name,
            version_id=self._challenger_entry.version_id,
            updated_by="runtime_governor",
            reason="runtime_canary_promotion",
        )

        if self.audit_logger is not None:
            self.audit_logger.log_model_deployed(
                self._challenger_entry.model_name,
                self._challenger_entry.version_id,
                metrics=self._challenger_entry.metrics,
                deployment_mode="runtime_canary_promotion",
                previous_champion=previous_champion.model_key if previous_champion else None,
            )

        self._publish_runtime_event(
            event_type=EventType.MODEL_RELOAD,
            message="Promoted challenger to champion",
            details={
                "promoted_model": self._challenger_entry.model_key,
                "previous_champion": previous_champion.model_key if previous_champion else None,
            },
        )
        self.refresh_deployment_state(force=True)

    def _evaluate_challenger_guardrails_locked(
        self,
        challenger_stats: RuntimeModelStats,
        rejection_reason: str | None = None,
    ) -> None:
        """Evaluate challenger rejection/slippage guardrails."""
        if self._challenger_entry is None or challenger_stats.quarantined:
            return

        min_orders = max(1, self.config.min_orders_for_guardrail)
        if challenger_stats.orders_submitted >= min_orders:
            if challenger_stats.rejection_rate > self.config.max_challenger_rejection_rate:
                self._quarantine_challenger_locked(
                    reason=rejection_reason or "rejection_rate_guardrail_breach",
                    current_value=challenger_stats.rejection_rate,
                    limit_value=self.config.max_challenger_rejection_rate,
                )
                return

        slippage_limit = self._challenger_slippage_limit_locked()
        if slippage_limit > 0 and challenger_stats.orders_filled >= min_orders:
            if challenger_stats.avg_slippage_bps > slippage_limit:
                self._quarantine_challenger_locked(
                    reason="slippage_guardrail_breach",
                    current_value=challenger_stats.avg_slippage_bps,
                    limit_value=slippage_limit,
                )

    def _challenger_drawdown_limit_locked(self) -> float:
        """Resolve challenger drawdown limit from deployment plan."""
        if self._challenger_entry is None:
            return 0.0

        guardrails = self._challenger_entry.deployment_plan.get("kill_switch_guardrails", {})
        if not isinstance(guardrails, dict):
            return 0.0
        return max(0.0, _safe_float(guardrails.get("max_intraday_drawdown"), 0.0))

    def _challenger_slippage_limit_locked(self) -> float:
        """Resolve challenger slippage limit from deployment plan."""
        if self._challenger_entry is None:
            return 0.0

        guardrails = self._challenger_entry.deployment_plan.get("tca_guardrails", {})
        if not isinstance(guardrails, dict):
            return 0.0
        return max(0.0, _safe_float(guardrails.get("max_slippage_bps"), 0.0))

    def _quarantine_challenger_locked(
        self,
        reason: str,
        current_value: float,
        limit_value: float,
        extra_details: dict[str, Any] | None = None,
    ) -> None:
        """Quarantine challenger and fall back to champion routing."""
        if self._challenger_entry is None:
            return

        stats = self._stats_for(self._challenger_entry.model_key)
        if stats.quarantined:
            return

        stats.quarantined = True
        stats.quarantine_reason = reason
        stats.touch()
        self.staleness_detector.force_quarantine(self._challenger_entry.model_key, reason=reason)

        details = {
            "challenger": self._challenger_entry.model_key,
            "reason": reason,
            "current_value": current_value,
            "limit_value": limit_value,
        }
        if extra_details:
            details.update(extra_details)

        self._publish_runtime_event(
            event_type=EventType.SYSTEM_ALERT,
            message="Challenger quarantined by runtime governor",
            details=details,
        )

        if self.audit_logger is not None:
            self.audit_logger.log_risk_limit_breach(
                limit_name="runtime_model_governor",
                current_value=current_value,
                limit_value=limit_value,
                action_taken="challenger_quarantined",
                model_name=self._challenger_entry.model_name,
                model_version=self._challenger_entry.version_id,
                reason=reason,
            )

    def _extract_model_from_metadata(
        self,
        metadata: dict[str, Any] | None,
    ) -> tuple[str | None, str | None]:
        """Resolve runtime model key and role from order metadata."""
        if not isinstance(metadata, dict):
            return None, None

        model_key = metadata.get("runtime_selected_model")
        role = metadata.get("runtime_model_role")
        return (str(model_key), str(role)) if model_key else (None, None)

    def _resolve_model_key(self, model_name: str) -> str:
        """Resolve a prediction alias back to the canonical model key."""
        for entry in (self._champion_entry, self._challenger_entry):
            if entry is None:
                continue
            if model_name == entry.model_key or model_name in entry.aliases:
                return entry.model_key
        return model_name

    def _entry_from_model_key(self, model_key: str | None) -> ManagedModelEntry | None:
        """Resolve managed entry from a canonical model key."""
        if not model_key:
            return None
        for entry in (self._champion_entry, self._challenger_entry):
            if entry is not None and entry.model_key == model_key:
                return entry
        return None

    def _stats_for(self, model_key: str) -> RuntimeModelStats:
        """Return mutable runtime stats for a model key."""
        if model_key not in self._model_stats:
            self._model_stats[model_key] = RuntimeModelStats(model_key=model_key)
        return self._model_stats[model_key]

    def _is_quarantined_locked(self, model_key: str) -> bool:
        """Check quarantine state under lock."""
        stats = self._model_stats.get(model_key)
        if stats and stats.quarantined:
            return True
        return self.staleness_detector.get_health(model_key).health == ModelHealth.STALE

    def _is_in_canary_bucket(self, entity_id: str, capital_fraction: float) -> bool:
        """Stable hash-based canary allocation for symbols."""
        if capital_fraction >= 1.0:
            return True
        if capital_fraction <= 0.0:
            return False

        seed = f"runtime_governor:{entity_id}".encode("utf-8")
        bucket = int(hashlib.sha256(seed).hexdigest()[:12], 16) / float(0xFFFFFFFFFFFF)
        return bucket < capital_fraction

    def _publish_runtime_event(
        self,
        event_type: EventType,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Publish runtime governor events through the shared event bus."""
        if self.event_bus is None:
            return
        self.event_bus.publish(
            create_system_event(
                event_type=event_type,
                message=message,
                details=details or {},
                source="ModelRuntimeGovernor",
            )
        )

    def _audit_deployment_state(
        self,
        champion: ManagedModelEntry | None,
        challenger: ManagedModelEntry | None,
    ) -> None:
        """Audit champion/challenger resolution changes."""
        if self.audit_logger is None:
            return

        if champion is not None:
            self.audit_logger.log_model_deployed(
                champion.model_name,
                champion.version_id,
                metrics=champion.metrics,
                deployment_mode="runtime_champion_resolved",
            )
        if challenger is not None:
            self.audit_logger.log_model_deployed(
                challenger.model_name,
                challenger.version_id,
                metrics=challenger.metrics,
                deployment_mode="runtime_challenger_resolved",
            )
