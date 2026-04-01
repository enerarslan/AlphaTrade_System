"""Unit tests for tick/quote microstructure feature augmentation."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_trading_system.features.tick_microstructure import (
    TickMicrostructureFeatureBuilder,
    TickMicrostructureFeatureConfig,
)


def test_tick_microstructure_builder_aggregates_quotes_and_trades(monkeypatch) -> None:
    base = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-02T09:45:00Z",
                    "2025-01-02T10:00:00Z",
                    "2025-01-02T10:15:00Z",
                ],
                utc=True,
            ),
            "open": [100.0, 101.0, 101.2],
            "high": [101.0, 102.0, 101.8],
            "low": [99.0, 100.0, 100.7],
            "close": [100.5, 101.5, 101.0],
            "volume": [1000.0, 1100.0, 1800.0],
        }
    )

    quotes = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-02T09:40:00Z",
                    "2025-01-02T09:58:00Z",
                    "2025-01-02T10:05:00Z",
                ],
                utc=True,
            ),
            "bid_price": [100.0, 101.0, 101.5],
            "bid_size": [10.0, 12.0, 8.0],
            "ask_price": [100.1, 101.2, 101.6],
            "ask_size": [8.0, 9.0, 14.0],
        }
    )
    trades = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-02T09:41:00Z",
                    "2025-01-02T09:44:00Z",
                    "2025-01-02T09:57:00Z",
                    "2025-01-02T10:06:00Z",
                    "2025-01-02T10:10:00Z",
                ],
                utc=True,
            ),
            "price": [100.0, 100.2, 101.0, 101.2, 100.9],
            "size": [100.0, 120.0, 80.0, 2000.0, 500.0],
        }
    )

    builder = TickMicrostructureFeatureBuilder(TickMicrostructureFeatureConfig(timeframe="15Min"))
    monkeypatch.setattr(builder, "_load_quotes", lambda *_args, **_kwargs: quotes)
    monkeypatch.setattr(builder, "_load_trades", lambda *_args, **_kwargs: trades)

    enriched = builder.augment(base)

    assert "tick_quote_spread_bps_mean" in enriched.columns
    assert "tick_trade_count" in enriched.columns
    assert "tick_trade_block_volume_share" in enriched.columns
    assert "tick_trade_signed_volume_ratio_change" in enriched.columns
    assert "tick_trade_flow_imbalance_gap" in enriched.columns
    assert enriched.loc[0, "tick_quote_update_count"] == 1.0
    assert enriched.loc[0, "tick_trade_count"] == 2.0
    assert enriched.loc[1, "tick_trade_count"] == 1.0
    assert enriched.loc[2, "tick_trade_count"] == 2.0
    assert enriched.loc[2, "tick_trade_block_volume_share"] == pytest.approx(0.8)
    assert enriched.loc[2, "tick_trade_signed_volume_ratio_change"] == pytest.approx(-0.4)
    assert enriched.loc[2, "tick_quote_imbalance_change"] < 0.0
    assert enriched.loc[2, "tick_trade_flow_imbalance_gap"] > 0.8
