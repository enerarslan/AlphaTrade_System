"""OOF-safe expected-edge policy model shared by training and promotion inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

_CONTEXT_TOKENS = (
    "vol",
    "regime",
    "spread",
    "imbalance",
    "flow",
    "sentiment",
    "news",
    "earnings",
    "filing",
    "macro",
    "relative",
    "strength",
    "short",
    "ftd",
    "liquid",
    "vwap",
    "pressure",
    "quality",
    "breadth",
    "momentum",
)
_DERIVED_POLICY_COLUMNS = (
    "policy_probability",
    "policy_probability_centered",
    "policy_probability_confidence",
    "policy_long_margin",
    "policy_short_margin",
    "policy_abs_threshold_margin",
    "policy_signal_strength",
    "policy_signal_direction",
    "policy_is_long",
    "policy_is_short",
    "policy_raw_prediction",
)


def _normalize_regime_values(
    regimes: np.ndarray | list[str] | pd.Series | None,
    *,
    length: int,
) -> np.ndarray:
    """Normalize arbitrary regime labels into stable lowercase keys."""
    if length <= 0:
        return np.array([], dtype=object)
    if regimes is None:
        return np.full(length, "normal_range", dtype=object)
    arr = np.asarray(regimes, dtype=object).reshape(-1)
    if arr.size < length:
        aligned = np.full(length, "normal_range", dtype=object)
        aligned[-arr.size :] = arr.astype(object)
        arr = aligned
    elif arr.size > length:
        arr = arr[-length:]
    normalized = pd.Series(arr, dtype="object").fillna("normal_range").astype(str).str.strip().str.lower()
    normalized = normalized.where(normalized != "", "normal_range")
    return normalized.to_numpy(dtype=object)


def _normalize_symbol_values(
    symbols: np.ndarray | list[str] | pd.Series | None,
    *,
    length: int,
) -> np.ndarray:
    """Normalize arbitrary symbol labels into stable uppercase keys."""
    if length <= 0:
        return np.array([], dtype=object)
    if symbols is None:
        return np.full(length, "__UNKNOWN__", dtype=object)
    arr = np.asarray(symbols, dtype=object).reshape(-1)
    if arr.size < length:
        aligned = np.full(length, "__UNKNOWN__", dtype=object)
        aligned[-arr.size :] = arr.astype(object)
        arr = aligned
    elif arr.size > length:
        arr = arr[-length:]
    normalized = pd.Series(arr, dtype="object").fillna("__UNKNOWN__").astype(str).str.strip().str.upper()
    normalized = normalized.where(normalized != "", "__UNKNOWN__")
    return normalized.to_numpy(dtype=object)


def _neutral_regime_policy() -> dict[str, Any]:
    """Return a no-op regime policy block."""
    return {
        "enabled": True,
        "long_threshold_adjustment": 0.0,
        "short_threshold_adjustment": 0.0,
        "min_pass_probability_adjustment": 0.0,
        "min_expected_edge_adjustment": 0.0,
        "signal_scale": 1.0,
        "confidence_scale": 1.0,
        "trade_count": 0,
        "active_rate": 0.0,
        "mean_trade_return": 0.0,
        "win_rate": 0.5,
        "score": 0.0,
        "sample_confidence": 0.0,
    }


def _coerce_regime_policy_block(raw_policy: Any) -> dict[str, Any]:
    """Normalize arbitrary regime policy payloads into a stable block."""
    base = _neutral_regime_policy()
    if not isinstance(raw_policy, dict):
        return base
    normalized = dict(base)
    normalized["enabled"] = bool(raw_policy.get("enabled", base["enabled"]))
    for key, low, high in (
        ("long_threshold_adjustment", -0.08, 0.08),
        ("short_threshold_adjustment", -0.08, 0.08),
        ("min_pass_probability_adjustment", -0.10, 0.12),
        ("min_expected_edge_adjustment", -0.01, 0.01),
        ("signal_scale", 0.0, 1.25),
        ("confidence_scale", 0.0, 1.20),
        ("active_rate", 0.0, 1.0),
        ("win_rate", 0.0, 1.0),
        ("score", -10.0, 10.0),
        ("sample_confidence", 0.0, 1.0),
    ):
        normalized[key] = float(np.clip(float(raw_policy.get(key, base[key]) or base[key]), low, high))
    for key in ("mean_trade_return",):
        normalized[key] = float(raw_policy.get(key, base[key]) or base[key])
    normalized["trade_count"] = int(max(0, int(float(raw_policy.get("trade_count", base["trade_count"]) or 0))))
    return normalized


def resolve_regime_policy_frame(
    regimes: np.ndarray | list[str] | pd.Series | None,
    regime_policy: dict[str, Any] | None,
    *,
    length: int,
) -> pd.DataFrame:
    """Resolve per-row regime-conditioned policy parameters."""
    normalized_regimes = _normalize_regime_values(regimes, length=length)
    raw_policy = regime_policy if isinstance(regime_policy, dict) else {}
    raw_default = raw_policy.get("default_policy", {})
    default_block = _coerce_regime_policy_block(raw_default)
    policy_by_regime = raw_policy.get("regimes", {})
    if not isinstance(policy_by_regime, dict):
        policy_by_regime = {}

    rows: list[dict[str, Any]] = []
    for regime_name in normalized_regimes.tolist():
        raw_block = policy_by_regime.get(str(regime_name), default_block)
        merged = dict(default_block)
        merged.update(_coerce_regime_policy_block(raw_block))
        merged["regime"] = str(regime_name)
        rows.append(merged)

    if not rows:
        return pd.DataFrame(
            columns=[
                "regime",
                "enabled",
                "long_threshold_adjustment",
                "short_threshold_adjustment",
                "min_pass_probability_adjustment",
                "min_expected_edge_adjustment",
                "signal_scale",
                "confidence_scale",
                "trade_count",
                "active_rate",
                "mean_trade_return",
                "win_rate",
                "score",
                "sample_confidence",
            ]
        )
    return pd.DataFrame(rows)


def derive_regime_conditioned_policy(
    *,
    regimes: np.ndarray | list[str] | pd.Series | None,
    trade_returns: np.ndarray | list[float],
    signal_values: np.ndarray | list[float],
    selected_mask: np.ndarray | list[bool] | pd.Series | None = None,
    min_samples: int = 120,
    edge_reference: float = 0.001,
) -> dict[str, Any]:
    """Derive a compact regime-conditioned policy from realized candidate-trade outcomes."""
    trade_return_values = np.asarray(trade_returns, dtype=float).reshape(-1)
    signal = np.asarray(signal_values, dtype=float).reshape(-1)
    if trade_return_values.size == 0 or signal.size == 0:
        return {
            "enabled": False,
            "reason": "insufficient_rows",
            "default_policy": _neutral_regime_policy(),
            "regimes": {},
        }
    length = int(min(trade_return_values.size, signal.size))
    trade_return_values = trade_return_values[-length:]
    signal = signal[-length:]
    normalized_regimes = _normalize_regime_values(regimes, length=length)
    finite_mask = np.isfinite(trade_return_values)
    candidate_mask = finite_mask & (np.abs(signal) > 1e-8)
    candidate_count = int(np.count_nonzero(candidate_mask))
    if candidate_count < max(30, int(min_samples * 0.5)):
        return {
            "enabled": False,
            "reason": "insufficient_candidates",
            "default_policy": _neutral_regime_policy(),
            "regimes": {},
        }

    selected = None
    if selected_mask is not None:
        selected = np.asarray(selected_mask, dtype=bool).reshape(-1)
        if selected.size < length:
            aligned = np.zeros(length, dtype=bool)
            aligned[-selected.size :] = selected
            selected = aligned
        elif selected.size > length:
            selected = selected[-length:]

    candidate_returns = trade_return_values[candidate_mask]
    global_mean = float(np.mean(candidate_returns)) if candidate_returns.size else 0.0
    global_win_rate = float(np.mean(candidate_returns > 0.0)) if candidate_returns.size else 0.5
    reference = float(max(abs(edge_reference), 1e-4))
    min_regime_observations = max(15, int(round(candidate_count * 0.08)))
    resolved: dict[str, dict[str, Any]] = {}

    for regime_name in sorted(set(normalized_regimes.tolist())):
        regime_mask = candidate_mask & (normalized_regimes == regime_name)
        regime_count = int(np.count_nonzero(regime_mask))
        if regime_count < min_regime_observations:
            continue
        regime_returns = trade_return_values[regime_mask]
        regime_signal = signal[regime_mask]
        mean_trade_return = float(np.mean(regime_returns)) if regime_returns.size else 0.0
        win_rate = float(np.mean(regime_returns > 0.0)) if regime_returns.size else 0.5
        active_rate = float(regime_count / max(candidate_count, 1))

        long_returns = regime_returns[regime_signal > 0.0]
        short_returns = regime_returns[regime_signal < 0.0]
        long_mean = float(np.mean(long_returns)) if long_returns.size else mean_trade_return
        short_mean = float(np.mean(short_returns)) if short_returns.size else mean_trade_return
        directional_bias = float(np.clip((long_mean - short_mean) / reference, -1.25, 1.25))
        edge_bias = float(np.clip((mean_trade_return - global_mean) / reference, -1.25, 1.25))

        selected_rate = 0.0
        selected_lift = 0.0
        if selected is not None:
            regime_selected = regime_mask & selected
            selected_rate = float(np.mean(regime_selected[regime_mask])) if regime_count else 0.0
            selected_returns = trade_return_values[regime_selected & np.isfinite(trade_return_values)]
            if selected_returns.size:
                selected_lift = float(np.mean(selected_returns) - mean_trade_return)
        selected_bias = float(np.clip(selected_lift / reference, -1.0, 1.0))

        downside = max(0.0, -edge_bias)
        upside = max(0.0, edge_bias)
        long_adjustment = float(
            np.clip((0.018 * downside) - (0.018 * upside) - (0.022 * directional_bias), -0.05, 0.05)
        )
        short_adjustment = float(
            np.clip((0.018 * downside) - (0.018 * upside) + (0.022 * directional_bias), -0.05, 0.05)
        )
        pass_adjustment = float(
            np.clip((0.035 * downside) - (0.030 * upside) - (0.020 * selected_bias), -0.05, 0.08)
        )
        edge_adjustment = float(
            np.clip((0.00035 * downside) - (0.00025 * upside) - (0.00015 * selected_bias), -0.0006, 0.0012)
        )
        signal_scale = float(np.clip(1.0 + (0.16 * edge_bias) + (0.08 * selected_bias), 0.70, 1.15))
        confidence_scale = float(np.clip(1.0 + (0.08 * edge_bias) + (0.05 * selected_bias), 0.80, 1.10))
        sample_confidence = float(np.clip(regime_count / max(min_regime_observations * 2.0, 1.0), 0.0, 1.0))
        score = float(
            (0.60 * edge_bias)
            + (0.20 * ((win_rate - global_win_rate) * 4.0))
            + (0.20 * selected_bias)
        )
        enabled = not (
            mean_trade_return < min(global_mean - (0.50 * reference), -0.25 * reference)
            and win_rate < 0.42
            and regime_count >= max(min_regime_observations, 24)
        )
        resolved[str(regime_name)] = {
            "enabled": bool(enabled),
            "long_threshold_adjustment": long_adjustment,
            "short_threshold_adjustment": short_adjustment,
            "min_pass_probability_adjustment": pass_adjustment,
            "min_expected_edge_adjustment": edge_adjustment,
            "signal_scale": signal_scale if enabled else 0.0,
            "confidence_scale": confidence_scale if enabled else 0.0,
            "trade_count": regime_count,
            "active_rate": active_rate,
            "mean_trade_return": mean_trade_return,
            "win_rate": win_rate,
            "score": score,
            "sample_confidence": sample_confidence,
            "selected_rate": selected_rate,
            "selected_edge_lift": selected_lift,
            "long_mean_trade_return": long_mean,
            "short_mean_trade_return": short_mean,
        }

    enabled = bool(resolved)
    return {
        "enabled": enabled,
        "reason": "trained" if enabled else "insufficient_regimes",
        "default_policy": _neutral_regime_policy(),
        "regimes": resolved,
    }


def _safe_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize arbitrary feature frames into finite numeric columns."""
    numeric = (
        frame.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .copy()
    )
    return numeric


def _normalize_vector(values: Any, length: int, fill_value: float = np.nan) -> np.ndarray:
    """Align arbitrary inputs to a dense float vector."""
    if values is None:
        return np.full(length, fill_value, dtype=float)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == length:
        return arr.astype(float)
    if arr.size < length:
        aligned = np.full(length, fill_value, dtype=float)
        aligned[-arr.size :] = arr.astype(float)
        return aligned
    return arr[-length:].astype(float)


def derive_base_signal(
    probabilities: np.ndarray | list[float],
    *,
    long_threshold: float,
    short_threshold: float,
) -> np.ndarray:
    """Convert probabilities into long/short/flat base signals."""
    probability = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), 0.0, 1.0)
    return np.where(
        probability >= float(long_threshold),
        1.0,
        np.where(probability <= float(short_threshold), -1.0, 0.0),
    ).astype(float)


def build_expected_edge_feature_frame(
    feature_frame: pd.DataFrame,
    *,
    probabilities: np.ndarray | list[float],
    long_threshold: float,
    short_threshold: float,
    raw_predictions: np.ndarray | list[float] | None = None,
    signal_values: np.ndarray | list[float] | None = None,
    confidence: np.ndarray | list[float] | None = None,
    context_features: list[str] | tuple[str, ...] | None = None,
    symbols: np.ndarray | list[str] | pd.Series | None = None,
    symbol_priors: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Build the deterministic feature frame consumed by the edge policy model."""
    numeric_frame = _safe_numeric_frame(feature_frame.reset_index(drop=True))
    length = len(numeric_frame)
    probability = np.clip(_normalize_vector(probabilities, length, fill_value=np.nan), 0.0, 1.0)
    if signal_values is None:
        signal = derive_base_signal(
            probability,
            long_threshold=float(long_threshold),
            short_threshold=float(short_threshold),
        )
    else:
        signal = _normalize_vector(signal_values, length, fill_value=0.0)
    if confidence is None:
        resolved_confidence = np.clip(np.abs(probability - 0.5) * 2.0, 0.0, 1.0)
    else:
        resolved_confidence = np.clip(
            _normalize_vector(confidence, length, fill_value=0.0),
            0.0,
            1.0,
        )
    raw_prediction = _normalize_vector(raw_predictions, length, fill_value=np.nan)

    frame = pd.DataFrame(
        {
            "policy_probability": probability,
            "policy_probability_centered": probability - 0.5,
            "policy_probability_confidence": resolved_confidence,
            "policy_long_margin": probability - float(long_threshold),
            "policy_short_margin": float(short_threshold) - probability,
            "policy_abs_threshold_margin": np.maximum(
                probability - float(long_threshold),
                float(short_threshold) - probability,
            ),
            "policy_signal_strength": np.abs(signal),
            "policy_signal_direction": np.sign(signal),
            "policy_is_long": (signal > 0.0).astype(float),
            "policy_is_short": (signal < 0.0).astype(float),
            "policy_raw_prediction": raw_prediction,
        },
        index=numeric_frame.index,
    )

    selected_context = [name for name in list(context_features or []) if name in numeric_frame.columns]
    if selected_context:
        frame = pd.concat([frame, numeric_frame.loc[:, selected_context]], axis=1)

    resolved_symbol_priors = symbol_priors if isinstance(symbol_priors, dict) else {}
    if resolved_symbol_priors:
        normalized_symbols = _normalize_symbol_values(symbols, length=length)
        default_prior = resolved_symbol_priors.get("__GLOBAL__", {})

        def _prior_series(key: str, fallback: float) -> np.ndarray:
            values: list[float] = []
            for symbol in normalized_symbols.tolist():
                prior_block = resolved_symbol_priors.get(str(symbol), default_prior)
                raw_value = prior_block.get(key, default_prior.get(key, fallback))
                values.append(float(raw_value))
            return np.asarray(values, dtype=float)

        frame["policy_symbol_edge_prior"] = _prior_series("edge_prior", 0.0)
        frame["policy_symbol_pass_prior"] = np.clip(_prior_series("pass_prior", 0.5), 0.0, 1.0)
        frame["policy_symbol_tail_prior"] = _prior_series("tail_prior", 0.0)
        frame["policy_symbol_downside_prior"] = np.clip(
            _prior_series("downside_prior", 0.5),
            0.0,
            1.0,
        )
        frame["policy_symbol_sample_confidence"] = np.clip(
            _prior_series("sample_confidence", 0.0),
            0.0,
            1.0,
        )
    return frame.replace([np.inf, -np.inf], np.nan)


@dataclass(slots=True)
class ExpectedEdgePolicyConfig:
    """Configuration for the second-stage expected-edge policy model."""

    min_samples: int = 120
    min_coverage: float = 0.55
    max_context_features: int = 24
    min_pass_probability: float = 0.53
    min_expected_edge: float = 0.0
    min_signal_scale: float = 0.25
    max_signal_scale: float = 1.10
    random_state: int = 42


class ExpectedEdgePolicyModel:
    """Learn a cost-aware trade admission and sizing policy from OOF-safe data."""

    def __init__(self, config: ExpectedEdgePolicyConfig | None = None) -> None:
        self.config = config or ExpectedEdgePolicyConfig()
        self.selected_context_features_: list[str] = []
        self.policy_feature_names_: list[str] = []
        self.feature_scores_: dict[str, float] = {}
        self.edge_reference_: float = 0.001
        self.training_summary_: dict[str, Any] = {}
        self.symbol_priors_: dict[str, dict[str, float]] = {}
        self._edge_model: Any | None = None
        self._pass_model: Any | None = None

    @staticmethod
    def resolve_effective_min_coverage(
        *,
        min_coverage: float,
        min_samples: int,
        row_count: int,
    ) -> float:
        """Scale row-share coverage to dataset size for selective but well-sampled policies."""
        base_threshold = float(np.clip(float(min_coverage), 0.0, 1.0))
        if row_count <= 0:
            return base_threshold

        # Require enough candidate density to exceed the absolute sample floor with buffer,
        # but do not force highly selective strategies to satisfy an unrealistic row-share gate.
        buffered_floor = (3.0 * float(max(1, int(min_samples)))) / float(max(1, int(row_count)))
        return float(min(base_threshold, max(0.10, buffered_floor)))

    @staticmethod
    def _feature_score(series: pd.Series, target: np.ndarray) -> float:
        """Score candidate context features with stable finite correlation."""
        values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() < 40:
            return 0.0
        valid = values.notna().to_numpy(dtype=bool) & np.isfinite(target)
        if int(np.count_nonzero(valid)) < 40:
            return 0.0
        aligned_values = values.to_numpy(dtype=float)[valid]
        aligned_target = target[valid]
        if np.nanstd(aligned_values) < 1e-9:
            return 0.0
        corr = np.corrcoef(aligned_values, aligned_target)[0, 1]
        if not np.isfinite(corr):
            return 0.0
        return float(abs(corr))

    def _select_context_features(
        self,
        feature_frame: pd.DataFrame,
        target: np.ndarray,
    ) -> list[str]:
        """Select a compact high-signal context feature slice for policy learning."""
        numeric_frame = _safe_numeric_frame(feature_frame)
        scored: list[tuple[int, float, str]] = []
        for column in numeric_frame.columns:
            if column in _DERIVED_POLICY_COLUMNS:
                continue
            score = self._feature_score(numeric_frame[column], target)
            token_bonus = int(any(token in column.lower() for token in _CONTEXT_TOKENS))
            if score <= 0.0 and token_bonus <= 0:
                continue
            scored.append((token_bonus, score, str(column)))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        selected = [column for _, _, column in scored[: int(self.config.max_context_features)]]
        self.feature_scores_ = {
            column: float(score)
            for _, score, column in scored[: int(self.config.max_context_features)]
        }
        return selected

    @staticmethod
    def _coerce_sample_weights(
        weights: np.ndarray | pd.Series | None,
        mask: np.ndarray,
    ) -> np.ndarray | None:
        """Align optional sample weights to the active candidate mask."""
        if weights is None:
            return None
        arr = np.asarray(weights, dtype=float).reshape(-1)
        if arr.size != mask.size:
            return None
        selected = arr[mask]
        return None if selected.size == 0 else selected

    @staticmethod
    def _safe_positive_reference(values: np.ndarray) -> float:
        """Derive a stable positive edge anchor for downstream sizing."""
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.001
        positive = np.abs(finite[finite > 0.0])
        if positive.size == 0:
            positive = np.abs(finite)
        reference = float(np.quantile(positive, 0.65)) if positive.size else 0.001
        return float(max(reference, 1e-4))

    def _build_symbol_priors(
        self,
        *,
        symbols: np.ndarray,
        trade_returns: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        """Build shrinkage-based symbol priors for selector alignment."""
        candidate_returns = trade_returns[candidate_mask]
        global_mean = float(np.mean(candidate_returns)) if candidate_returns.size else 0.0
        global_pass = float(np.mean(candidate_returns > 0.0)) if candidate_returns.size else 0.5
        global_tail = float(np.quantile(candidate_returns, 0.25)) if candidate_returns.size else 0.0
        global_downside = float(np.mean(candidate_returns <= 0.0)) if candidate_returns.size else 0.5
        global_prior = {
            "edge_prior": global_mean,
            "pass_prior": global_pass,
            "tail_prior": global_tail,
            "downside_prior": global_downside,
            "sample_confidence": 0.0,
            "candidate_count": 0.0,
        }
        priors: dict[str, dict[str, float]] = {"__GLOBAL__": dict(global_prior)}
        if candidate_returns.size == 0:
            return priors

        prior_strength = max(12.0, float(self.config.min_samples) * 0.35)
        for symbol_name in sorted({str(value) for value in symbols[candidate_mask].tolist()}):
            symbol_mask = candidate_mask & (symbols == symbol_name)
            symbol_returns = trade_returns[symbol_mask]
            count = int(symbol_returns.size)
            if count <= 0:
                continue
            shrink = float(count / (count + prior_strength))
            sym_mean = float(np.mean(symbol_returns))
            sym_pass = float(np.mean(symbol_returns > 0.0))
            sym_tail = float(np.quantile(symbol_returns, 0.25))
            sym_downside = float(np.mean(symbol_returns <= 0.0))
            priors[str(symbol_name)] = {
                "edge_prior": float((shrink * sym_mean) + ((1.0 - shrink) * global_mean)),
                "pass_prior": float((shrink * sym_pass) + ((1.0 - shrink) * global_pass)),
                "tail_prior": float((shrink * sym_tail) + ((1.0 - shrink) * global_tail)),
                "downside_prior": float(
                    (shrink * sym_downside) + ((1.0 - shrink) * global_downside)
                ),
                "sample_confidence": float(np.clip(shrink, 0.0, 1.0)),
                "candidate_count": float(count),
            }
        return priors

    def fit(
        self,
        feature_frame: pd.DataFrame,
        *,
        probabilities: np.ndarray | list[float],
        trade_returns: np.ndarray | list[float],
        long_threshold: float,
        short_threshold: float,
        raw_predictions: np.ndarray | list[float] | None = None,
        signal_values: np.ndarray | list[float] | None = None,
        confidence: np.ndarray | list[float] | None = None,
        sample_weights: np.ndarray | pd.Series | None = None,
        symbols: np.ndarray | list[str] | pd.Series | None = None,
    ) -> dict[str, Any]:
        """Fit the expected-edge admission/sizing policy on candidate trades only."""
        trade_return_values = np.asarray(trade_returns, dtype=float).reshape(-1)
        numeric_frame = _safe_numeric_frame(feature_frame.reset_index(drop=True))
        if len(numeric_frame) != trade_return_values.size:
            raise ValueError("Expected-edge policy feature/target length mismatch.")

        selected_context = self._select_context_features(numeric_frame, trade_return_values)
        resolved_signal = (
            _normalize_vector(signal_values, len(numeric_frame), fill_value=0.0)
            if signal_values is not None
            else derive_base_signal(
                probabilities,
                long_threshold=float(long_threshold),
                short_threshold=float(short_threshold),
            )
        )
        normalized_symbols = _normalize_symbol_values(symbols, length=len(numeric_frame))
        candidate_mask = np.isfinite(trade_return_values) & (np.abs(resolved_signal) > 1e-8)
        self.symbol_priors_ = self._build_symbol_priors(
            symbols=normalized_symbols,
            trade_returns=trade_return_values,
            candidate_mask=candidate_mask,
        )
        policy_frame = build_expected_edge_feature_frame(
            numeric_frame,
            probabilities=probabilities,
            long_threshold=float(long_threshold),
            short_threshold=float(short_threshold),
            raw_predictions=raw_predictions,
            signal_values=signal_values,
            confidence=confidence,
            context_features=selected_context,
            symbols=normalized_symbols,
            symbol_priors=self.symbol_priors_,
        )

        signal = policy_frame["policy_signal_direction"].to_numpy(dtype=float)
        candidate_mask = np.isfinite(trade_return_values) & (np.abs(signal) > 1e-8)
        candidate_count = int(np.count_nonzero(candidate_mask))
        coverage = float(candidate_count / max(len(policy_frame), 1))
        effective_min_coverage = self.resolve_effective_min_coverage(
            min_coverage=float(self.config.min_coverage),
            min_samples=int(self.config.min_samples),
            row_count=int(len(policy_frame)),
        )
        if candidate_count < int(self.config.min_samples):
            raise ValueError(
                f"Expected-edge policy needs at least {self.config.min_samples} candidate trades; "
                f"received {candidate_count}."
            )
        if coverage < effective_min_coverage:
            raise ValueError(
                f"Expected-edge policy coverage too low ({coverage:.2%} < "
                f"effective {effective_min_coverage:.2%}; "
                f"configured {float(self.config.min_coverage):.2%})."
            )

        X_train = (
            policy_frame.loc[candidate_mask]
            .fillna(0.0)
            .reset_index(drop=True)
        )
        y_edge = trade_return_values[candidate_mask]
        y_pass = (y_edge > 0.0).astype(int)
        weights = self._coerce_sample_weights(sample_weights, candidate_mask)

        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        self._edge_model = RandomForestRegressor(
            n_estimators=160,
            max_depth=6,
            min_samples_leaf=24,
            random_state=int(self.config.random_state),
            n_jobs=-1,
        )
        self._edge_model.fit(X_train, y_edge, sample_weight=weights)

        if len(np.unique(y_pass)) <= 1:
            self._pass_model = DummyClassifier(strategy="constant", constant=int(y_pass[0]))
            self._pass_model.fit(X_train, y_pass, sample_weight=weights)
        else:
            self._pass_model = RandomForestClassifier(
                n_estimators=160,
                max_depth=6,
                min_samples_leaf=24,
                random_state=int(self.config.random_state),
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
            self._pass_model.fit(X_train, y_pass, sample_weight=weights)

        self.selected_context_features_ = list(selected_context)
        self.policy_feature_names_ = list(X_train.columns)
        self.edge_reference_ = self._safe_positive_reference(y_edge)

        training_policy = self.predict_policy(
            numeric_frame,
            probabilities=probabilities,
            long_threshold=float(long_threshold),
            short_threshold=float(short_threshold),
            raw_predictions=raw_predictions,
            signal_values=signal_values,
            confidence=confidence,
            symbols=normalized_symbols,
        )
        selected_mask = training_policy["edge_policy_pass"].to_numpy(dtype=bool)
        selected_edge = trade_return_values[selected_mask & np.isfinite(trade_return_values)]
        candidate_edge = trade_return_values[candidate_mask]
        self.training_summary_ = {
            "candidate_count": int(candidate_count),
            "candidate_rate": float(coverage),
            "effective_min_coverage": float(effective_min_coverage),
            "configured_min_coverage": float(self.config.min_coverage),
            "selected_count": int(np.count_nonzero(selected_mask)),
            "selected_rate": float(np.mean(selected_mask)) if len(selected_mask) else 0.0,
            "candidate_mean_trade_return": float(np.mean(candidate_edge)) if candidate_edge.size else 0.0,
            "selected_mean_trade_return": float(np.mean(selected_edge)) if selected_edge.size else 0.0,
            "selected_win_rate": float(np.mean(selected_edge > 0.0)) if selected_edge.size else 0.0,
            "edge_reference": float(self.edge_reference_),
            "selected_context_features": list(self.selected_context_features_),
            "symbol_prior_summary": dict(self.symbol_priors_),
        }
        return dict(self.training_summary_)

    def _predict_pass_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the profitable-trade probability from the fitted classifier."""
        if self._pass_model is None:
            return np.zeros(len(X), dtype=float)
        if hasattr(self._pass_model, "predict_proba"):
            proba = np.asarray(self._pass_model.predict_proba(X), dtype=float)
            if proba.ndim == 2:
                if proba.shape[1] >= 2:
                    return np.clip(proba[:, -1], 0.0, 1.0)
                return np.clip(proba[:, 0], 0.0, 1.0)
        prediction = np.asarray(self._pass_model.predict(X), dtype=float).reshape(-1)
        return np.clip(prediction, 0.0, 1.0)

    def predict_policy(
        self,
        feature_frame: pd.DataFrame,
        *,
        probabilities: np.ndarray | list[float],
        long_threshold: float,
        short_threshold: float,
        raw_predictions: np.ndarray | list[float] | None = None,
        signal_values: np.ndarray | list[float] | None = None,
        confidence: np.ndarray | list[float] | None = None,
        symbols: np.ndarray | list[str] | pd.Series | None = None,
        regimes: np.ndarray | list[str] | pd.Series | None = None,
        regime_policy: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Predict expected edge, trade pass probability, and signal scale."""
        if self._edge_model is None or self._pass_model is None:
            raise RuntimeError("Expected-edge policy model has not been fitted.")

        numeric_frame = _safe_numeric_frame(feature_frame.reset_index(drop=True))
        symbol_priors = getattr(self, "symbol_priors_", {})
        policy_frame = build_expected_edge_feature_frame(
            numeric_frame,
            probabilities=probabilities,
            long_threshold=float(long_threshold),
            short_threshold=float(short_threshold),
            raw_predictions=raw_predictions,
            signal_values=signal_values,
            confidence=confidence,
            context_features=self.selected_context_features_,
            symbols=symbols,
            symbol_priors=symbol_priors,
        )
        for column in self.policy_feature_names_:
            if column not in policy_frame.columns:
                policy_frame[column] = 0.0
        X = policy_frame.loc[:, self.policy_feature_names_].fillna(0.0)
        signal = policy_frame["policy_signal_direction"].to_numpy(dtype=float)
        candidate_mask = np.abs(signal) > 1e-8

        expected_edge = np.zeros(len(X), dtype=float)
        pass_probability = np.zeros(len(X), dtype=float)
        if bool(np.any(candidate_mask)):
            candidate_features = X.loc[candidate_mask]
            expected_edge[candidate_mask] = np.asarray(
                self._edge_model.predict(candidate_features),
                dtype=float,
            ).reshape(-1)
            pass_probability[candidate_mask] = self._predict_pass_probability(candidate_features)

        regime_frame = resolve_regime_policy_frame(regimes, regime_policy, length=len(X))
        regime_enabled = (
            regime_frame["enabled"].to_numpy(dtype=bool)
            if "enabled" in regime_frame.columns
            else np.ones(len(X), dtype=bool)
        )
        min_edge = np.full(len(X), float(self.config.min_expected_edge), dtype=float)
        min_pass_probability = np.full(len(X), float(self.config.min_pass_probability), dtype=float)
        regime_signal_scale = np.ones(len(X), dtype=float)
        regime_confidence_scale = np.ones(len(X), dtype=float)
        if not regime_frame.empty:
            min_edge = np.clip(
                min_edge
                + regime_frame["min_expected_edge_adjustment"].to_numpy(dtype=float),
                -0.02,
                0.02,
            )
            min_pass_probability = np.clip(
                min_pass_probability
                + regime_frame["min_pass_probability_adjustment"].to_numpy(dtype=float),
                0.0,
                0.995,
            )
            regime_signal_scale = regime_frame["signal_scale"].to_numpy(dtype=float)
            regime_confidence_scale = regime_frame["confidence_scale"].to_numpy(dtype=float)

        edge_range = np.maximum(float(self.edge_reference_) - min_edge, 1e-6)
        edge_component = np.clip((expected_edge - min_edge) / edge_range, 0.0, 1.5)
        pass_component = np.clip(
            (pass_probability - min_pass_probability) / np.maximum(1.0 - min_pass_probability, 1e-6),
            0.0,
            1.0,
        )
        scale = float(self.config.min_signal_scale) + (
            float(self.config.max_signal_scale) - float(self.config.min_signal_scale)
        ) * ((0.55 * pass_component) + (0.45 * np.clip(edge_component, 0.0, 1.0)))
        scale = np.clip(scale, float(self.config.min_signal_scale), float(self.config.max_signal_scale))
        policy_pass = (
            candidate_mask
            & regime_enabled
            & (expected_edge >= min_edge)
            & (pass_probability >= min_pass_probability)
        )
        scale = np.where(policy_pass, scale * regime_signal_scale, 0.0)

        result = pd.DataFrame(
            {
                "expected_edge": expected_edge,
                "edge_pass_probability": np.clip(pass_probability, 0.0, 1.0),
                "edge_loss_probability": np.clip(1.0 - pass_probability, 0.0, 1.0),
                "edge_policy_pass": policy_pass.astype(bool),
                "edge_policy_scale": scale,
                "edge_policy_confidence_scale": np.where(
                    policy_pass,
                    np.clip(regime_confidence_scale, 0.0, 1.20),
                    0.0,
                ),
                "runtime_regime": regime_frame.get("regime", pd.Series(["normal_range"] * len(X))).to_numpy(),
                "regime_policy_enabled": regime_enabled.astype(bool),
                "regime_policy_signal_scale": regime_signal_scale,
                "regime_policy_min_pass_probability": min_pass_probability,
                "regime_policy_min_expected_edge": min_edge,
            },
            index=policy_frame.index,
        )
        return result
