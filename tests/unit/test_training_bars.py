"""Unit tests for training bar builders."""

from __future__ import annotations

import pandas as pd

from quant_trading_system.data.training_bars import TrainingBarBuilder, TrainingBarConfig


def _sample_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL"] * 6,
            "timestamp": pd.date_range("2025-01-02 09:30:00", periods=6, freq="15min", tz="UTC"),
            "open": [100.0, 101.0, 102.0, 103.0, 102.5, 103.5],
            "high": [101.0, 102.0, 103.0, 104.0, 103.0, 104.5],
            "low": [99.5, 100.5, 101.5, 102.5, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 102.8, 104.0],
            "volume": [500.0, 600.0, 700.0, 800.0, 750.0, 850.0],
        }
    )


def test_training_bar_builder_keeps_time_mode_data_unchanged() -> None:
    frame = _sample_ohlcv()
    builder = TrainingBarBuilder(TrainingBarConfig(mode="time"))

    result = builder.build(frame)

    assert len(result) == len(frame)
    assert result[["symbol", "timestamp", "open", "high", "low", "close", "volume"]].equals(frame)


def test_training_bar_builder_generates_intrinsic_bars_from_time_bars_when_no_trades() -> None:
    frame = _sample_ohlcv()
    builder = TrainingBarBuilder(
        TrainingBarConfig(
            mode="intrinsic",
            intrinsic_bar_type="volume",
            intrinsic_threshold=900.0,
            target_bars_per_day=20,
            use_trade_prints_if_available=False,
        )
    )

    result = builder.build(frame)

    assert not result.empty
    assert {"symbol", "timestamp", "open", "high", "low", "close", "volume"}.issubset(
        result.columns
    )
    assert len(result) < len(frame)
