"""Institutional target engineering utilities for training pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_trading_system.models.meta_labeling import TripleBarrierLabeler


@dataclass
class TargetEngineeringConfig:
    """Config for multi-horizon, cost-aware label generation."""

    horizons: tuple[int, ...] = (1, 5, 20)
    primary_horizon: int = 5
    profit_taking_threshold: float = 0.015
    stop_loss_threshold: float = 0.010
    max_holding_period: int = 20
    use_volatility_barriers: bool = True
    volatility_lookback: int = 20
    spread_bps: float = 1.0
    slippage_bps: float = 3.0
    impact_bps: float = 2.0
    regime_lookback: int = 30
    temporal_weight_decay: float = 0.999

    def __post_init__(self) -> None:
        if not self.horizons:
            raise ValueError("horizons cannot be empty")
        if self.primary_horizon not in set(self.horizons):
            raise ValueError("primary_horizon must be present in horizons")
        if self.max_holding_period <= 0:
            raise ValueError("max_holding_period must be positive")
        if not 0.0 < self.temporal_weight_decay <= 1.0:
            raise ValueError("temporal_weight_decay must be in (0, 1]")


@dataclass
class TargetEngineeringResult:
    """Outputs from target engineering."""

    frame: pd.DataFrame
    sample_weights: pd.Series
    diagnostics: dict[str, Any]


def _classify_regime(close: pd.Series, lookback: int, vol_lookback: int) -> pd.Series:
    returns = close.pct_change().fillna(0.0)
    trend = returns.rolling(max(2, lookback)).mean()
    vol = returns.rolling(max(2, vol_lookback)).std()
    vol_hi = float(vol.quantile(0.70)) if len(vol.dropna()) > 0 else 0.0
    vol_lo = float(vol.quantile(0.30)) if len(vol.dropna()) > 0 else 0.0

    regime = np.select(
        condlist=[
            vol >= vol_hi,
            (trend > 0) & (vol <= vol_hi),
            (trend < 0) & (vol <= vol_hi),
            vol <= vol_lo,
        ],
        choicelist=[
            "high_vol",
            "trend_up",
            "trend_down",
            "low_vol_range",
        ],
        default="normal_range",
    )
    return pd.Series(regime, index=close.index, dtype="object")


def _compute_temporal_weights(length: int, decay: float) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=float)
    # Newer observations receive higher weights.
    ages = np.arange(length - 1, -1, -1)
    w = np.power(decay, ages)
    w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
    return w.astype(float)


def _class_weights_binary(labels: pd.Series) -> dict[int, float]:
    counts = labels.value_counts().to_dict()
    pos = max(int(counts.get(1, 0)), 1)
    neg = max(int(counts.get(0, 0)), 1)
    total = pos + neg
    return {
        0: float(total / (2.0 * neg)),
        1: float(total / (2.0 * pos)),
    }


def generate_targets(
    frame: pd.DataFrame,
    config: TargetEngineeringConfig,
) -> TargetEngineeringResult:
    """Generate multi-horizon, triple-barrier, cost-aware labels and sample weights."""
    required = {"symbol", "timestamp", "close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Target engineering missing required columns: {missing}")

    work = frame.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["symbol", "timestamp", "close"])
    work = work.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    if work.empty:
        raise ValueError("No valid rows available for target engineering.")

    labeler = TripleBarrierLabeler(
        profit_taking=config.profit_taking_threshold,
        stop_loss=abs(config.stop_loss_threshold),
        max_holding_period=config.max_holding_period,
        use_volatility_barriers=config.use_volatility_barriers,
        volatility_lookback=config.volatility_lookback,
    )

    by_symbol: list[pd.DataFrame] = []
    for symbol, sdf in work.groupby("symbol", sort=True):
        sdf = sdf.sort_values("timestamp").reset_index(drop=True)
        close = sdf["close"].astype(float)

        # Multi-horizon forward returns.
        for horizon in config.horizons:
            sdf[f"forward_return_h{horizon}"] = close.pct_change(horizon).shift(-horizon)

        primary_forward = sdf[f"forward_return_h{config.primary_horizon}"]
        primary_signals = pd.Series(
            np.where(primary_forward > 0, 1.0, np.where(primary_forward < 0, -1.0, 0.0)),
            index=sdf.index,
            dtype=float,
        )
        tb = labeler.generate_labels(prices=close, signals=primary_signals)

        total_cost = (config.spread_bps + config.slippage_bps + config.impact_bps) / 10000.0
        gross_ret = pd.to_numeric(tb["return"], errors="coerce")
        net_ret = gross_ret - total_cost * primary_signals.abs()
        cost_label = np.where(primary_signals != 0, (net_ret > 0).astype(float), np.nan)

        sdf["primary_signal"] = primary_signals
        sdf["triple_barrier_label"] = pd.to_numeric(tb["label"], errors="coerce")
        sdf["triple_barrier_event_return"] = gross_ret
        sdf["triple_barrier_net_return"] = net_ret
        sdf["barrier_touched"] = tb["barrier_touched"].astype("object")
        sdf["holding_period"] = pd.to_numeric(tb["holding_period"], errors="coerce")
        # Final binary label for model training.
        sdf["label"] = np.where(~np.isnan(sdf["triple_barrier_label"]), cost_label, np.nan)
        sdf["regime"] = _classify_regime(
            close=close,
            lookback=config.regime_lookback,
            vol_lookback=config.volatility_lookback,
        ).values

        by_symbol.append(sdf)

    labeled = pd.concat(by_symbol, ignore_index=True)

    # Sample weighting: temporal + class imbalance + mild regime weighting.
    temporal_chunks: list[np.ndarray] = []
    for _, sdf in labeled.groupby("symbol", sort=True):
        temporal_chunks.append(
            _compute_temporal_weights(len(sdf), decay=config.temporal_weight_decay)
        )
    temporal_weight = np.concatenate(temporal_chunks) if temporal_chunks else np.array([])
    if len(temporal_weight) != len(labeled):
        temporal_weight = np.ones(len(labeled), dtype=float)

    label_series = pd.to_numeric(labeled["label"], errors="coerce")
    valid_mask = label_series.isin([0.0, 1.0])
    class_weight_values = np.ones(len(labeled), dtype=float)
    if valid_mask.any():
        cw = _class_weights_binary(label_series[valid_mask].astype(int))
        class_weight_values[valid_mask.to_numpy()] = label_series[valid_mask].astype(int).map(cw).to_numpy()

    regime_boost = {
        "high_vol": 1.10,
        "trend_up": 1.00,
        "trend_down": 1.00,
        "low_vol_range": 1.05,
        "normal_range": 1.00,
    }
    regime_weight_values = labeled["regime"].map(regime_boost).fillna(1.0).to_numpy(dtype=float)
    sample_weights = temporal_weight * class_weight_values * regime_weight_values
    sample_weights = np.nan_to_num(sample_weights, nan=1.0, posinf=1.0, neginf=1.0)
    mean_w = float(np.mean(sample_weights)) if len(sample_weights) else 1.0
    if mean_w > 1e-12:
        sample_weights = sample_weights / mean_w

    # Diagnostics for governance.
    valid_labels = label_series[valid_mask].astype(int)
    pos_rate = float(valid_labels.mean()) if len(valid_labels) else 0.0
    half = max(1, len(valid_labels) // 2)
    first_half_rate = float(valid_labels.iloc[:half].mean()) if len(valid_labels) else 0.0
    second_half_rate = float(valid_labels.iloc[half:].mean()) if len(valid_labels.iloc[half:]) else 0.0
    drift_abs = abs(second_half_rate - first_half_rate)

    regime_distribution = labeled.loc[valid_mask, "regime"].value_counts(normalize=True).to_dict()
    label_counts = valid_labels.value_counts().to_dict()
    class_balance_ratio = (
        float(min(label_counts.get(0, 0), label_counts.get(1, 0)) / max(label_counts.get(0, 1), label_counts.get(1, 1)))
        if label_counts
        else 0.0
    )

    diagnostics: dict[str, Any] = {
        "label_count": int(len(valid_labels)),
        "positive_rate": pos_rate,
        "class_balance_ratio": class_balance_ratio,
        "positive_rate_first_half": first_half_rate,
        "positive_rate_second_half": second_half_rate,
        "label_drift_abs": float(drift_abs),
        "regime_distribution": {str(k): float(v) for k, v in regime_distribution.items()},
        "cost_assumptions_bps": {
            "spread": float(config.spread_bps),
            "slippage": float(config.slippage_bps),
            "impact": float(config.impact_bps),
        },
        "horizons": [int(h) for h in config.horizons],
        "primary_horizon": int(config.primary_horizon),
        "triple_barrier": {
            "profit_taking_threshold": float(config.profit_taking_threshold),
            "stop_loss_threshold": float(config.stop_loss_threshold),
            "max_holding_period": int(config.max_holding_period),
        },
    }

    return TargetEngineeringResult(
        frame=labeled,
        sample_weights=pd.Series(sample_weights, index=labeled.index, dtype=float),
        diagnostics=diagnostics,
    )
