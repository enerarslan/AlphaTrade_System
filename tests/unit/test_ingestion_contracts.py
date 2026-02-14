"""Unit tests for canonical OHLCV ingestion contracts."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import polars as pl
import pytest

from quant_trading_system.data.ingestion_contracts import (
    OHLCVIngestionContract,
    apply_ohlcv_ingestion_contract,
)


def test_apply_contract_normalizes_naive_timestamps_to_utc_and_deduplicates() -> None:
    df = pl.DataFrame(
        {
            "timestamp": [
                "2025-01-02 09:30:00",
                "2025-01-02 09:30:00",  # duplicate
                "2025-01-02 09:45:00",
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
        }
    )
    contract = OHLCVIngestionContract(source_timezone="America/New_York")
    result = apply_ohlcv_ingestion_contract(df, "aapl", contract)

    assert len(result) == 2
    assert result["symbol"].to_list() == ["AAPL", "AAPL"]
    first_ts = result["timestamp"][0]
    assert first_ts.tzinfo is not None
    assert first_ts.tzinfo.utcoffset(first_ts) == ZoneInfo("UTC").utcoffset(first_ts)


def test_apply_contract_rejects_invalid_ohlc_rows() -> None:
    df = pl.DataFrame(
        {
            "timestamp": ["2025-01-02 09:30:00"],
            "open": [100.0],
            "high": [99.0],  # invalid: high < open
            "low": [98.0],
            "close": [99.5],
            "volume": [1000],
        }
    )
    with pytest.raises(ValueError, match="No valid OHLCV rows remain"):
        apply_ohlcv_ingestion_contract(df, "AAPL")


def test_apply_contract_requires_schema_columns() -> None:
    df = pl.DataFrame(
        {
            "timestamp": ["2025-01-02 09:30:00"],
            "open": [100.0],
            "high": [101.0],
            "close": [100.5],
            "volume": [1000],
        }
    )
    with pytest.raises(ValueError, match="Missing required columns"):
        apply_ohlcv_ingestion_contract(df, "AAPL")
