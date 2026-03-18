"""Unit tests for multi-timeframe feature alignment."""

from __future__ import annotations

import pandas as pd

from quant_trading_system.features.multi_timeframe import (
    MultiTimeframeFeatureEngine,
    normalize_timeframes,
    resample_ohlcv,
)


def test_normalize_timeframes_includes_base_and_sorts() -> None:
    result = normalize_timeframes("15Min", ["1Day", "1Hour", "15Min", "4h"])
    assert result == ["15Min", "1Hour", "4Hour", "1Day"]


def test_resample_ohlcv_aggregates_to_hourly_close() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 4,
            "timestamp": pd.to_datetime(
                [
                    "2025-01-02T09:15:00Z",
                    "2025-01-02T09:30:00Z",
                    "2025-01-02T09:45:00Z",
                    "2025-01-02T10:00:00Z",
                ],
                utc=True,
            ),
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.5, 100.5, 101.5, 102.5],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [10.0, 20.0, 30.0, 40.0],
        }
    )

    resampled = resample_ohlcv(frame, "1Hour")

    assert len(resampled) == 1
    assert resampled.loc[0, "timestamp"] == pd.Timestamp("2025-01-02T10:00:00Z")
    assert resampled.loc[0, "open"] == 100.0
    assert resampled.loc[0, "high"] == 104.0
    assert resampled.loc[0, "low"] == 99.5
    assert resampled.loc[0, "close"] == 103.5
    assert resampled.loc[0, "volume"] == 100.0


def test_align_feature_frames_delays_daily_features_until_next_business_day() -> None:
    base = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "timestamp": pd.to_datetime(
                ["2025-01-02T10:00:00Z", "2025-01-03T10:00:00Z"],
                utc=True,
            ),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000.0, 1100.0],
            "base_feature": [1.0, 2.0],
        }
    )
    daily = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "timestamp": pd.to_datetime(["2025-01-02T00:00:00Z"], utc=True),
            "open": [95.0],
            "high": [105.0],
            "low": [94.0],
            "close": [100.0],
            "volume": [5000.0],
            "daily_feature": [7.0],
        }
    )

    engine = MultiTimeframeFeatureEngine(base_timeframe="15Min", timeframes=["1Day"])
    aligned = engine.align_feature_frames(base_frame=base, feature_frames={"1Day": daily})

    assert pd.isna(aligned.loc[0, "tf1d_daily_feature"])
    assert aligned.loc[1, "tf1d_daily_feature"] == 7.0
