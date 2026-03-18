"""
Tick/quote microstructure feature augmentation for training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import select

from quant_trading_system.data.timeframe import DEFAULT_TIMEFRAME, normalize_timeframe
from quant_trading_system.features.multi_timeframe import timeframe_to_rule
from quant_trading_system.database.connection import DatabaseManager, get_db_manager
from quant_trading_system.database.models import StockQuote, StockTrade

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TickMicrostructureFeatureConfig:
    timeframe: str = DEFAULT_TIMEFRAME

    def normalized_timeframe(self) -> str:
        return normalize_timeframe(self.timeframe, default=DEFAULT_TIMEFRAME)


class TickMicrostructureFeatureBuilder:
    def __init__(
        self,
        config: TickMicrostructureFeatureConfig,
        db_manager: DatabaseManager | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.db_manager = db_manager or get_db_manager()
        self.logger = logger_ or logger

    def augment(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        if feature_matrix is None or feature_matrix.empty:
            return feature_matrix

        base = feature_matrix.copy()
        base["symbol"] = base["symbol"].astype(str).str.upper().str.strip()
        base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True, errors="coerce")
        base = base.dropna(subset=["symbol", "timestamp"]).copy()
        if base.empty:
            return feature_matrix

        quotes = self._aggregate_quotes(base)
        trades = self._aggregate_trades(base)

        if not quotes.empty:
            base = base.merge(quotes, on=["symbol", "timestamp"], how="left")
        if not trades.empty:
            base = base.merge(trades, on=["symbol", "timestamp"], how="left")

        feature_cols = [column for column in base.columns if column.startswith("tick_")]
        for column in feature_cols:
            base[column] = pd.to_numeric(base[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            default = 0.0
            if column.endswith("_days_since_last"):
                default = 9999.0
            base[column] = base[column].fillna(default)
        return base

    def _load_quotes(self, symbols: list[str], min_timestamp: pd.Timestamp, max_timestamp: pd.Timestamp) -> pd.DataFrame:
        try:
            with self.db_manager.session() as session:
                rows = list(
                    session.execute(
                        select(
                            StockQuote.symbol,
                            StockQuote.timestamp,
                            StockQuote.bid_price,
                            StockQuote.bid_size,
                            StockQuote.ask_price,
                            StockQuote.ask_size,
                        ).where(
                            StockQuote.symbol.in_(symbols),
                            StockQuote.timestamp >= min_timestamp,
                            StockQuote.timestamp <= max_timestamp,
                        )
                    ).all()
                )
        except Exception as exc:
            self.logger.warning("Tick quote feature load skipped: %s", exc)
            return pd.DataFrame()
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(
            rows,
            columns=["symbol", "timestamp", "bid_price", "bid_size", "ask_price", "ask_size"],
        )
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        for column in ("bid_price", "bid_size", "ask_price", "ask_size"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.dropna(subset=["symbol", "timestamp"]).copy()

    def _load_trades(self, symbols: list[str], min_timestamp: pd.Timestamp, max_timestamp: pd.Timestamp) -> pd.DataFrame:
        try:
            with self.db_manager.session() as session:
                rows = list(
                    session.execute(
                        select(
                            StockTrade.symbol,
                            StockTrade.timestamp,
                            StockTrade.price,
                            StockTrade.size,
                        ).where(
                            StockTrade.symbol.in_(symbols),
                            StockTrade.timestamp >= min_timestamp,
                            StockTrade.timestamp <= max_timestamp,
                        )
                    ).all()
                )
        except Exception as exc:
            self.logger.warning("Tick trade feature load skipped: %s", exc)
            return pd.DataFrame()
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows, columns=["symbol", "timestamp", "price", "size"])
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        frame["size"] = pd.to_numeric(frame["size"], errors="coerce")
        return frame.dropna(subset=["symbol", "timestamp", "price", "size"]).copy()

    def _aggregate_quotes(self, base: pd.DataFrame) -> pd.DataFrame:
        symbols = sorted(base["symbol"].dropna().unique().tolist())
        min_timestamp = pd.Timestamp(base["timestamp"].min())
        max_timestamp = pd.Timestamp(base["timestamp"].max())
        quotes = self._load_quotes(symbols, min_timestamp, max_timestamp)
        if quotes.empty:
            return pd.DataFrame()

        rule = timeframe_to_rule(self.config.normalized_timeframe())
        mid = (quotes["bid_price"] + quotes["ask_price"]) / 2.0
        spread = quotes["ask_price"] - quotes["bid_price"]
        size_sum = quotes["bid_size"] + quotes["ask_size"]
        quotes["tick_quote_spread_bps"] = (
            (spread / mid.replace(0.0, np.nan)) * 10_000.0
        ).replace([np.inf, -np.inf], np.nan)
        quotes["tick_quote_imbalance"] = (
            (quotes["bid_size"] - quotes["ask_size"]) / size_sum.replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        quotes["timestamp"] = quotes["timestamp"].dt.ceil(rule)
        grouped = (
            quotes.groupby(["symbol", "timestamp"], as_index=False)
            .agg(
                {
                    "tick_quote_spread_bps": ["mean", "last"],
                    "tick_quote_imbalance": ["mean", "last"],
                }
            )
        )
        grouped.columns = [
            "symbol",
            "timestamp",
            "tick_quote_spread_bps_mean",
            "tick_quote_spread_bps_last",
            "tick_quote_imbalance_mean",
            "tick_quote_imbalance_last",
        ]
        counts = (
            quotes.groupby(["symbol", "timestamp"], as_index=False)
            .size()
            .rename(columns={"size": "tick_quote_update_count"})
        )
        return grouped.merge(counts, on=["symbol", "timestamp"], how="left")

    def _aggregate_trades(self, base: pd.DataFrame) -> pd.DataFrame:
        symbols = sorted(base["symbol"].dropna().unique().tolist())
        min_timestamp = pd.Timestamp(base["timestamp"].min())
        max_timestamp = pd.Timestamp(base["timestamp"].max())
        trades = self._load_trades(symbols, min_timestamp, max_timestamp)
        if trades.empty:
            return pd.DataFrame()

        rule = timeframe_to_rule(self.config.normalized_timeframe())
        trades = trades.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        trades["timestamp"] = trades["timestamp"].dt.ceil(rule)
        trades["tick_trade_dollar_volume"] = trades["price"] * trades["size"]
        trades["price_delta"] = trades.groupby("symbol", sort=False)["price"].diff().fillna(0.0)
        signed_size = np.sign(trades["price_delta"]).replace(0.0, np.nan)
        signed_size = signed_size.groupby(trades["symbol"], sort=False).ffill().fillna(1.0)
        trades["signed_size"] = signed_size * trades["size"]

        agg = (
            trades.groupby(["symbol", "timestamp"], as_index=False)
            .agg(
                tick_trade_count=("size", "size"),
                tick_trade_volume=("size", "sum"),
                tick_trade_avg_size=("size", "mean"),
                tick_trade_dollar_volume=("tick_trade_dollar_volume", "sum"),
                tick_trade_first_price=("price", "first"),
                tick_trade_last_price=("price", "last"),
                tick_trade_signed_volume=("signed_size", "sum"),
            )
        )
        agg["tick_trade_signed_volume_ratio"] = (
            agg["tick_trade_signed_volume"]
            / agg["tick_trade_volume"].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        agg["tick_trade_price_impact_bps_per_million"] = (
            (
                (
                    agg["tick_trade_last_price"]
                    / agg["tick_trade_first_price"].replace(0.0, np.nan)
                )
                - 1.0
            ).abs()
            * 10_000.0
            / (agg["tick_trade_dollar_volume"].replace(0.0, np.nan) / 1_000_000.0)
        ).replace([np.inf, -np.inf], np.nan)
        return agg.drop(
            columns=["tick_trade_first_price", "tick_trade_last_price", "tick_trade_signed_volume"],
            errors="ignore",
        )
