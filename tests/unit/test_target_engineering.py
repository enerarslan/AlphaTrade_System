"""Unit tests for institutional target engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_trading_system.models.target_engineering import (
    TargetEngineeringConfig,
    generate_targets,
)


def _sample_frame(n: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ts = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    base = 100 + np.cumsum(rng.normal(0.01, 0.20, size=n))
    return pd.DataFrame(
        {
            "symbol": ["AAPL"] * n,
            "timestamp": ts,
            "open": base + rng.normal(0, 0.05, size=n),
            "high": base + 0.1,
            "low": base - 0.1,
            "close": base,
            "volume": rng.integers(1000, 5000, size=n),
            "feat_1": rng.normal(size=n),
            "feat_2": rng.normal(size=n),
        }
    )


def test_generate_targets_creates_label_and_horizon_columns() -> None:
    frame = _sample_frame()
    config = TargetEngineeringConfig(
        horizons=(1, 5, 10),
        primary_horizon=5,
        max_holding_period=12,
    )
    result = generate_targets(frame, config)

    assert "label" in result.frame.columns
    assert "forward_return_h1" in result.frame.columns
    assert "forward_return_h5" in result.frame.columns
    assert "forward_return_h10" in result.frame.columns
    assert result.frame["label"].notna().sum() > 0


def test_generate_targets_outputs_finite_normalized_weights() -> None:
    frame = _sample_frame()
    result = generate_targets(frame, TargetEngineeringConfig())

    weights = result.sample_weights.to_numpy(dtype=float)
    assert len(weights) == len(result.frame)
    assert np.isfinite(weights).all()
    assert np.all(weights > 0)
    assert np.mean(weights) == np.mean(weights)  # not NaN
    assert abs(float(np.mean(weights)) - 1.0) < 1e-6


def test_generate_targets_emits_diagnostics_with_regime_distribution() -> None:
    frame = _sample_frame()
    result = generate_targets(frame, TargetEngineeringConfig())

    diagnostics = result.diagnostics
    assert diagnostics["label_count"] >= 0
    assert 0.0 <= diagnostics["positive_rate"] <= 1.0
    assert 0.0 <= diagnostics["label_drift_abs"] <= 1.0
    assert isinstance(diagnostics["regime_distribution"], dict)
