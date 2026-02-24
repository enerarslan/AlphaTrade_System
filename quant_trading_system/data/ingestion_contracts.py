"""Canonical OHLCV ingestion contracts with strict timezone normalization."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import polars as pl


@dataclass(frozen=True)
class OHLCVIngestionContract:
    """Strict schema contract for institutional OHLCV ingestion."""

    required_columns: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")
    source_timezone: str = "America/New_York"
    target_timezone: str = "UTC"
    allow_negative_prices: bool = False
    allow_zero_volume: bool = True


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def apply_ohlcv_ingestion_contract(
    df: pl.DataFrame,
    symbol: str,
    contract: OHLCVIngestionContract | None = None,
) -> pl.DataFrame:
    """Validate and normalize an OHLCV dataframe to contract-compliant schema."""
    contract = contract or OHLCVIngestionContract()
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError("Symbol must be non-empty for ingestion contract validation.")

    missing = [c for c in contract.required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {symbol}: {missing}")
    if len(df) == 0:
        raise ValueError(f"Empty OHLCV data for {symbol}.")

    # Cast numeric columns deterministically.
    casted = df.with_columns(
        [
            pl.col("open").cast(pl.Float64, strict=False),
            pl.col("high").cast(pl.Float64, strict=False),
            pl.col("low").cast(pl.Float64, strict=False),
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Int64, strict=False),
        ]
    )

    # Timestamp normalization:
    # - if timezone-aware -> convert to UTC
    # - if naive -> treat as source timezone, then convert to UTC
    ts_dtype = casted.schema.get("timestamp")
    if ts_dtype is None:
        raise ValueError(f"Missing timestamp column for {symbol}.")

    if isinstance(ts_dtype, pl.datatypes.Datetime):
        if ts_dtype.time_zone is None:
            casted = casted.with_columns(
                pl.col("timestamp")
                .dt.replace_time_zone(contract.source_timezone)
                .dt.convert_time_zone(contract.target_timezone)
                .alias("timestamp")
            )
        else:
            casted = casted.with_columns(
                pl.col("timestamp")
                .dt.convert_time_zone(contract.target_timezone)
                .alias("timestamp")
            )
    else:
        casted = casted.with_columns(
            pl.col("timestamp")
            .str.to_datetime(strict=False)
            .dt.replace_time_zone(contract.source_timezone)
            .dt.convert_time_zone(contract.target_timezone)
            .alias("timestamp")
        )

    # Remove invalid/null critical values.
    casted = casted.drop_nulls(
        subset=["timestamp", "open", "high", "low", "close", "volume"]
    )
    if len(casted) == 0:
        raise ValueError(f"All OHLCV rows invalid after normalization for {symbol}.")

    # Enforce strict OHLC relationships and price sanity.
    ohlc_ok = (
        (pl.col("high") >= pl.col("low"))
        & (pl.col("high") >= pl.col("open"))
        & (pl.col("high") >= pl.col("close"))
        & (pl.col("low") <= pl.col("open"))
        & (pl.col("low") <= pl.col("close"))
    )
    filtered = casted.filter(ohlc_ok)

    if not contract.allow_negative_prices:
        filtered = filtered.filter(
            (pl.col("open") > 0)
            & (pl.col("high") > 0)
            & (pl.col("low") > 0)
            & (pl.col("close") > 0)
        )
    if not contract.allow_zero_volume:
        filtered = filtered.filter(pl.col("volume") > 0)
    else:
        filtered = filtered.filter(pl.col("volume") >= 0)

    # Deduplicate by (symbol, timestamp), keep latest row deterministically.
    with_symbol = filtered.with_columns(pl.lit(symbol).alias("symbol"))
    deduped = with_symbol.unique(subset=["symbol", "timestamp"], keep="last")
    deduped = deduped.sort("timestamp")

    if len(deduped) == 0:
        raise ValueError(f"No valid OHLCV rows remain after contract checks for {symbol}.")

    return deduped


def quality_summary(df: pl.DataFrame) -> dict[str, Any]:
    """Build a lightweight quality summary for ingestion logs."""
    n = len(df)
    if n == 0:
        return {"rows": 0}

    first_ts = df["timestamp"][0]
    last_ts = df["timestamp"][-1]
    return {
        "rows": n,
        "start": str(first_ts),
        "end": str(last_ts),
        "min_close": float(df["close"].min()),
        "max_close": float(df["close"].max()),
        "min_volume": int(df["volume"].min()),
        "max_volume": int(df["volume"].max()),
    }
