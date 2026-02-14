"""Unit tests for statistical validation helpers used by promotion gates."""

from __future__ import annotations

import numpy as np

from quant_trading_system.models.statistical_validation import (
    calculate_deflated_sharpe_ratio,
    calculate_probability_of_backtest_overfitting,
)


def test_calculate_deflated_sharpe_ratio_returns_finite_values() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0008, 0.01, size=600)

    dsr, p_value = calculate_deflated_sharpe_ratio(
        observed_sharpe=1.2,
        returns=returns,
        n_trials=50,
    )

    assert np.isfinite(dsr)
    assert 0.0 <= p_value <= 1.0


def test_calculate_probability_of_backtest_overfitting_in_range() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0002, 0.015, size=800)

    pbo, interpretation = calculate_probability_of_backtest_overfitting(returns)

    assert 0.0 <= pbo <= 1.0
    assert interpretation != ""
