"""
Deterministic trading-day replay and SLO gate utilities.

This module provides institutional-style replay primitives that run a
deterministic strategy over historical bars and evaluate execution/risk SLOs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable

import pandas as pd

from quant_trading_system.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    ExecutionMode,
    PandasDataHandler,
    Strategy,
)
from quant_trading_system.core.data_types import Direction, TradeSignal
from quant_trading_system.models.trading_costs import TradingCostModel


@dataclass(frozen=True)
class ReplaySignalConfig:
    """Signal-generation parameters for deterministic replay strategy."""

    return_threshold_bps: float = 5.0
    confidence: float = 0.75
    horizon_bars: int = 1
    model_source: str = "deterministic_replay"


@dataclass(frozen=True)
class ReplaySLOGates:
    """SLO thresholds applied to replay outputs."""

    max_drawdown: float = 0.20
    max_rejection_rate: float = 0.20
    max_avg_slippage_bps: float = 35.0
    max_risk_escalations: int = 0
    max_escalation_latency_ms: float = 2_000.0
    min_orders_for_gate: int = 1
    fail_on_kill_switch_active: bool = True
    max_cost_vs_expected_ratio: float = 2.5


@dataclass(frozen=True)
class ReplayScenario:
    """Replay scenario configuration."""

    scenario_id: str
    symbols: list[str]
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal = Decimal("100000")
    execution_mode: ExecutionMode = ExecutionMode.REALISTIC
    slippage_bps: float = 5.0
    commission_bps: float = 1.0
    allow_short: bool = True
    signal: ReplaySignalConfig = field(default_factory=ReplaySignalConfig)
    backtest_overrides: dict[str, Any] = field(default_factory=dict)

    def to_trading_cost_model(self) -> TradingCostModel:
        """Build canonical execution cost assumptions for replay validation."""
        return TradingCostModel(
            spread_bps=float(self.commission_bps),
            slippage_bps=float(self.slippage_bps),
            impact_bps=0.0,
        )


@dataclass(frozen=True)
class ReplayOutcome:
    """Replay result for a single scenario."""

    scenario_id: str
    passed: bool
    violations: list[str]
    bars_processed: int
    trades_closed: int
    total_return: float
    max_drawdown: float
    execution_slo: dict[str, Any]
    started_at: datetime
    completed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert result to JSON-serializable dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "passed": self.passed,
            "violations": self.violations,
            "bars_processed": self.bars_processed,
            "trades_closed": self.trades_closed,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "execution_slo": self.execution_slo,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
        }


@dataclass(frozen=True)
class ReplaySuiteReport:
    """Aggregate report for multiple replay scenarios."""

    outcomes: list[ReplayOutcome]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def passed(self) -> bool:
        """Return True when all scenarios pass."""
        return all(outcome.passed for outcome in self.outcomes)

    def to_dict(self) -> dict[str, Any]:
        """Convert suite report to JSON-serializable dictionary."""
        failed = [outcome.scenario_id for outcome in self.outcomes if not outcome.passed]
        return {
            "passed": self.passed,
            "total_scenarios": len(self.outcomes),
            "failed_scenarios": failed,
            "generated_at": self.generated_at.isoformat(),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


class DeterministicReplayStrategy(Strategy):
    """Deterministic return-threshold strategy for replay workflows."""

    def __init__(
        self,
        symbols: list[str],
        signal_config: ReplaySignalConfig | None = None,
        allow_short: bool = True,
    ) -> None:
        self.symbols = [symbol.upper() for symbol in symbols]
        self.signal_config = signal_config or ReplaySignalConfig()
        self.allow_short = allow_short
        self._last_close: dict[str, Decimal] = {}

    def generate_signals(self, data_handler, portfolio) -> list[TradeSignal]:
        threshold = max(1e-9, self.signal_config.return_threshold_bps / 10_000.0)
        signals: list[TradeSignal] = []

        for symbol in self.symbols:
            bar = data_handler.get_current_bar(symbol)
            if bar is None:
                continue

            previous_close = self._last_close.get(symbol)
            self._last_close[symbol] = bar.close
            if previous_close is None or previous_close <= 0:
                continue

            simple_return = float((bar.close - previous_close) / previous_close)
            abs_return = abs(simple_return)
            if abs_return < threshold:
                continue

            direction = Direction.LONG if simple_return > 0 else Direction.SHORT
            if direction == Direction.SHORT and not self.allow_short:
                continue
            position = portfolio.get_position(symbol)
            if position is not None:
                if direction == Direction.LONG and position.quantity > 0:
                    continue
                if direction == Direction.SHORT and position.quantity < 0:
                    continue

            base_strength = max(0.0, min(1.0, abs_return / threshold))
            strength = base_strength if direction == Direction.LONG else -base_strength
            confidence = min(1.0, max(self.signal_config.confidence, base_strength))

            signals.append(
                TradeSignal(
                    symbol=symbol,
                    direction=direction,
                    strength=strength,
                    confidence=confidence,
                    timestamp=bar.timestamp,
                    horizon=max(1, self.signal_config.horizon_bars),
                    model_source=self.signal_config.model_source,
                    metadata={
                        "strategy": "deterministic_replay",
                        "simple_return": simple_return,
                        "threshold_bps": self.signal_config.return_threshold_bps,
                    },
                )
            )

        return signals


def run_replay_suite(
    scenarios: list[ReplayScenario],
    data_provider: Callable[[ReplayScenario], dict[str, pd.DataFrame]],
    slo_gates: ReplaySLOGates | None = None,
) -> ReplaySuiteReport:
    """Execute a deterministic replay suite across scenarios."""
    outcomes = []
    gates = slo_gates or ReplaySLOGates()
    for scenario in scenarios:
        scenario_data = data_provider(scenario)
        outcomes.append(run_replay_scenario(scenario_data, scenario, gates))
    return ReplaySuiteReport(outcomes=outcomes)


def run_replay_scenario(
    data: dict[str, pd.DataFrame],
    scenario: ReplayScenario,
    slo_gates: ReplaySLOGates | None = None,
) -> ReplayOutcome:
    """Run a deterministic replay scenario and evaluate SLO gates."""
    prepared_data = _prepare_replay_data(data, scenario)
    config = _build_backtest_config(scenario)
    strategy = DeterministicReplayStrategy(
        symbols=scenario.symbols,
        signal_config=scenario.signal,
        allow_short=scenario.allow_short,
    )
    engine = BacktestEngine(
        data_handler=PandasDataHandler(prepared_data),
        strategy=strategy,
        config=config,
    )

    started_at = datetime.now(timezone.utc)
    state = engine.run()
    completed_at = datetime.now(timezone.utc)
    cost_model = scenario.to_trading_cost_model()
    execution_slo = dict(engine.get_execution_slo_snapshot())
    execution_slo["expected_execution_cost_bps"] = float(cost_model.execution_cost_bps)

    max_drawdown = _compute_max_drawdown(state.equity_curve)
    total_return = _compute_total_return(state.equity_curve)
    violations = evaluate_replay_slo(
        execution_slo=execution_slo,
        max_drawdown=max_drawdown,
        slo_gates=slo_gates or ReplaySLOGates(),
        cost_model=cost_model,
    )

    return ReplayOutcome(
        scenario_id=scenario.scenario_id,
        passed=not violations,
        violations=violations,
        bars_processed=state.bars_processed,
        trades_closed=len(state.trades),
        total_return=total_return,
        max_drawdown=max_drawdown,
        execution_slo=execution_slo,
        started_at=started_at,
        completed_at=completed_at,
    )


def evaluate_replay_slo(
    execution_slo: dict[str, Any],
    max_drawdown: float,
    slo_gates: ReplaySLOGates,
    cost_model: TradingCostModel | None = None,
) -> list[str]:
    """Evaluate replay against configured SLO gates and return violations."""
    violations: list[str] = []

    if max_drawdown > slo_gates.max_drawdown:
        violations.append(
            f"max_drawdown={max_drawdown:.4f} > allowed={slo_gates.max_drawdown:.4f}"
        )

    orders_submitted = int(execution_slo.get("orders_submitted", 0) or 0)
    if orders_submitted >= max(0, int(slo_gates.min_orders_for_gate)):
        rejection_rate = float(execution_slo.get("rejection_rate", 0.0) or 0.0)
        if rejection_rate > slo_gates.max_rejection_rate:
            violations.append(
                f"rejection_rate={rejection_rate:.4f} > allowed={slo_gates.max_rejection_rate:.4f}"
            )

        avg_slippage_bps = float(execution_slo.get("avg_slippage_bps", 0.0) or 0.0)
        if avg_slippage_bps > slo_gates.max_avg_slippage_bps:
            violations.append(
                "avg_slippage_bps="
                f"{avg_slippage_bps:.4f} > allowed={slo_gates.max_avg_slippage_bps:.4f}"
            )
        if cost_model is not None:
            expected_cost_bps = max(1e-9, float(cost_model.execution_cost_bps))
            realized_to_expected = avg_slippage_bps / expected_cost_bps
            if realized_to_expected > float(slo_gates.max_cost_vs_expected_ratio):
                violations.append(
                    "execution_cost_ratio="
                    f"{realized_to_expected:.4f} > allowed={slo_gates.max_cost_vs_expected_ratio:.4f} "
                    f"(avg={avg_slippage_bps:.4f}bps expected={expected_cost_bps:.4f}bps)"
                )

    risk_playbook = execution_slo.get("risk_playbook", {})
    if isinstance(risk_playbook, dict):
        escalation_count = int(risk_playbook.get("escalation_count", 0) or 0)
        if escalation_count > slo_gates.max_risk_escalations:
            violations.append(
                f"risk_escalations={escalation_count} > allowed={slo_gates.max_risk_escalations}"
            )

        escalation_latency = risk_playbook.get("last_escalation_latency_ms")
        if escalation_latency is not None:
            latency_value = float(escalation_latency)
            if latency_value > slo_gates.max_escalation_latency_ms:
                violations.append(
                    "last_escalation_latency_ms="
                    f"{latency_value:.2f} > allowed={slo_gates.max_escalation_latency_ms:.2f}"
                )

        kill_switch_active = bool(risk_playbook.get("kill_switch_active", False))
        if kill_switch_active and slo_gates.fail_on_kill_switch_active:
            violations.append("kill_switch_active=True")

    return violations


def _prepare_replay_data(
    raw_data: dict[str, pd.DataFrame],
    scenario: ReplayScenario,
) -> dict[str, pd.DataFrame]:
    """Normalize and date-filter replay data for scenario symbols."""
    prepared: dict[str, pd.DataFrame] = {}
    start = pd.Timestamp(scenario.start_date)
    end = pd.Timestamp(scenario.end_date)

    for symbol in scenario.symbols:
        upper_symbol = symbol.upper()
        frame = raw_data.get(upper_symbol)
        if frame is None:
            frame = raw_data.get(symbol)
        if frame is None:
            raise ValueError(f"Replay data missing for symbol: {upper_symbol}")

        normalized = _normalize_ohlcv_frame(upper_symbol, frame)
        filtered = normalized[(normalized.index >= start) & (normalized.index <= end)]
        if filtered.empty:
            raise ValueError(
                f"Replay data empty after date filter for {upper_symbol} "
                f"between {scenario.start_date.date()} and {scenario.end_date.date()}"
            )
        prepared[upper_symbol] = filtered

    return prepared


def _normalize_ohlcv_frame(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize an OHLCV dataframe to required backtest shape."""
    if frame.empty:
        raise ValueError(f"Replay dataframe is empty for symbol {symbol}")

    result = frame.copy()
    column_map = {str(column).lower(): column for column in result.columns}
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(column_map.keys())
    if missing:
        raise ValueError(
            f"Replay dataframe for {symbol} missing columns: {sorted(missing)}"
        )

    rename_map = {column_map[name]: name for name in required}
    result = result.rename(columns=rename_map)

    if "timestamp" in result.columns:
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
        result = result.dropna(subset=["timestamp"]).set_index("timestamp")
    if not isinstance(result.index, pd.DatetimeIndex):
        raise ValueError(f"Replay dataframe for {symbol} must have DatetimeIndex or timestamp column")

    result.index = pd.to_datetime(result.index, utc=True, errors="coerce")
    result = result[~result.index.isna()].sort_index()
    return result.loc[:, ["open", "high", "low", "close", "volume"]]


def _build_backtest_config(scenario: ReplayScenario) -> BacktestConfig:
    """Create backtest config for deterministic replay scenario."""
    config_kwargs: dict[str, Any] = {
        "initial_capital": scenario.initial_capital,
        "start_date": scenario.start_date,
        "end_date": scenario.end_date,
        "execution_mode": scenario.execution_mode,
        "commission_bps": scenario.commission_bps,
        "slippage_bps": scenario.slippage_bps,
        "allow_short": scenario.allow_short,
        "random_seed": 42,
    }

    if scenario.backtest_overrides:
        allowed = {field_def.name for field_def in fields(BacktestConfig)}
        for key, value in scenario.backtest_overrides.items():
            if key in allowed:
                config_kwargs[key] = value

    return BacktestConfig(**config_kwargs)


def _compute_total_return(equity_curve: list[tuple[datetime, float]]) -> float:
    """Compute total return from equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    start_equity = equity_curve[0][1]
    end_equity = equity_curve[-1][1]
    if start_equity <= 0:
        return 0.0
    return float(end_equity / start_equity - 1.0)


def _compute_max_drawdown(equity_curve: list[tuple[datetime, float]]) -> float:
    """Compute maximum drawdown from equity curve."""
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0][1]
    max_drawdown = 0.0
    for _, equity in equity_curve:
        peak = max(peak, equity)
        if peak <= 0:
            continue
        drawdown = (peak - equity) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return float(max_drawdown)
