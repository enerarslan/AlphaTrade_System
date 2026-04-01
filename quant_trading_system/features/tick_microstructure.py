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
from quant_trading_system.database.connection import DatabaseManager, get_db_manager
from quant_trading_system.database.models import StockQuote, StockTrade
from quant_trading_system.features.multi_timeframe import timeframe_to_rule

logger = logging.getLogger(__name__)
_FLOW_LOOKBACK_BARS_SHORT = 3
_FLOW_LOOKBACK_BARS_LONG = 12
_FLOW_MIN_BASELINE_BARS = 3
_BLOCK_TRADE_SIZE_THRESHOLD = 1000.0


def _relative_to_history(series: pd.Series, window: int) -> pd.Series:
    baseline = series.shift(1).rolling(window, min_periods=_FLOW_MIN_BASELINE_BARS).mean()
    return ((series / baseline.replace(0.0, np.nan)) - 1.0).replace([np.inf, -np.inf], np.nan)


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
        base["__row_id"] = np.arange(len(base), dtype=np.int64)
        base["symbol"] = base["symbol"].astype(str).str.upper().str.strip()
        base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True, errors="coerce")
        base = base.dropna(subset=["symbol", "timestamp"]).copy()
        if base.empty:
            return feature_matrix
        base = base.sort_values(["symbol", "timestamp", "__row_id"]).reset_index(drop=True)

        quotes = self._aggregate_quotes(base)
        trades = self._aggregate_trades(base)

        if not quotes.empty:
            base = base.merge(quotes, on=["symbol", "timestamp"], how="left")
        if not trades.empty:
            base = base.merge(trades, on=["symbol", "timestamp"], how="left")
        base = self._augment_flow_regime_features(base)

        feature_cols = [column for column in base.columns if column.startswith("tick_")]
        for column in feature_cols:
            base[column] = pd.to_numeric(base[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            default = 0.0
            if column.endswith("_days_since_last"):
                default = 9999.0
            base[column] = base[column].fillna(default)
        return (
            base.sort_values("__row_id")
            .drop(columns=["__row_id"], errors="ignore")
            .reset_index(drop=True)
        )

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
        trades["tick_trade_block_volume"] = np.where(
            trades["size"] >= _BLOCK_TRADE_SIZE_THRESHOLD,
            trades["size"],
            0.0,
        )
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
                tick_trade_block_volume=("tick_trade_block_volume", "sum"),
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
        agg["tick_trade_block_volume_share"] = (
            agg["tick_trade_block_volume"]
            / agg["tick_trade_volume"].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        agg["tick_trade_vwap"] = (
            agg["tick_trade_dollar_volume"]
            / agg["tick_trade_volume"].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        agg["tick_trade_return_bps"] = (
            (
                agg["tick_trade_last_price"]
                / agg["tick_trade_first_price"].replace(0.0, np.nan)
                - 1.0
            )
            * 10_000.0
        ).replace([np.inf, -np.inf], np.nan)
        agg["tick_trade_vwap_deviation_bps"] = (
            (
                agg["tick_trade_last_price"]
                / agg["tick_trade_vwap"].replace(0.0, np.nan)
                - 1.0
            )
            * 10_000.0
        ).replace([np.inf, -np.inf], np.nan)
        return agg.drop(
            columns=[
                "tick_trade_first_price",
                "tick_trade_last_price",
                "tick_trade_signed_volume",
                "tick_trade_block_volume",
                "tick_trade_vwap",
            ],
            errors="ignore",
        )

    def _augment_flow_regime_features(self, base: pd.DataFrame) -> pd.DataFrame:
        derived_columns = (
            "tick_quote_spread_bps_change",
            "tick_quote_spread_bps_ema_3",
            "tick_quote_spread_bps_burst_12",
            "tick_quote_imbalance_change",
            "tick_quote_imbalance_ema_3",
            "tick_quote_update_intensity_12",
            "tick_trade_signed_volume_ratio_change",
            "tick_trade_signed_volume_ratio_ema_3",
            "tick_trade_volume_burst_12",
            "tick_trade_count_burst_12",
            "tick_trade_flow_imbalance_gap",
            "tick_trade_pressure_alignment",
        )
        for column in derived_columns:
            base[column] = np.nan

        grouped_rows = base.groupby("symbol", sort=False).groups
        for row_index in grouped_rows.values():
            ordered_index = base.loc[row_index].sort_values("timestamp").index

            if "tick_quote_spread_bps_mean" in base.columns:
                spread = pd.to_numeric(base.loc[ordered_index, "tick_quote_spread_bps_mean"], errors="coerce")
                base.loc[ordered_index, "tick_quote_spread_bps_change"] = spread.diff().to_numpy()
                base.loc[ordered_index, "tick_quote_spread_bps_ema_3"] = (
                    spread.ewm(
                        span=_FLOW_LOOKBACK_BARS_SHORT,
                        adjust=False,
                        min_periods=1,
                    ).mean().to_numpy()
                )
                base.loc[ordered_index, "tick_quote_spread_bps_burst_12"] = (
                    _relative_to_history(spread, _FLOW_LOOKBACK_BARS_LONG).to_numpy()
                )

            if "tick_quote_imbalance_last" in base.columns:
                imbalance = pd.to_numeric(base.loc[ordered_index, "tick_quote_imbalance_last"], errors="coerce")
                base.loc[ordered_index, "tick_quote_imbalance_change"] = imbalance.diff().to_numpy()
                base.loc[ordered_index, "tick_quote_imbalance_ema_3"] = (
                    imbalance.ewm(
                        span=_FLOW_LOOKBACK_BARS_SHORT,
                        adjust=False,
                        min_periods=1,
                    ).mean().to_numpy()
                )

            if "tick_quote_update_count" in base.columns:
                quote_updates = pd.to_numeric(base.loc[ordered_index, "tick_quote_update_count"], errors="coerce")
                base.loc[ordered_index, "tick_quote_update_intensity_12"] = (
                    _relative_to_history(quote_updates, _FLOW_LOOKBACK_BARS_LONG).to_numpy()
                )

            if "tick_trade_signed_volume_ratio" in base.columns:
                signed_ratio = pd.to_numeric(
                    base.loc[ordered_index, "tick_trade_signed_volume_ratio"],
                    errors="coerce",
                )
                base.loc[ordered_index, "tick_trade_signed_volume_ratio_change"] = (
                    signed_ratio.diff().to_numpy()
                )
                base.loc[ordered_index, "tick_trade_signed_volume_ratio_ema_3"] = (
                    signed_ratio.ewm(
                        span=_FLOW_LOOKBACK_BARS_SHORT,
                        adjust=False,
                        min_periods=1,
                    ).mean().to_numpy()
                )

            if "tick_trade_volume" in base.columns:
                trade_volume = pd.to_numeric(base.loc[ordered_index, "tick_trade_volume"], errors="coerce")
                base.loc[ordered_index, "tick_trade_volume_burst_12"] = (
                    _relative_to_history(trade_volume, _FLOW_LOOKBACK_BARS_LONG).to_numpy()
                )

            if "tick_trade_count" in base.columns:
                trade_count = pd.to_numeric(base.loc[ordered_index, "tick_trade_count"], errors="coerce")
                base.loc[ordered_index, "tick_trade_count_burst_12"] = (
                    _relative_to_history(trade_count, _FLOW_LOOKBACK_BARS_LONG).to_numpy()
                )

            if (
                "tick_trade_signed_volume_ratio" in base.columns
                and "tick_quote_imbalance_last" in base.columns
            ):
                signed_ratio = pd.to_numeric(
                    base.loc[ordered_index, "tick_trade_signed_volume_ratio"],
                    errors="coerce",
                )
                imbalance = pd.to_numeric(
                    base.loc[ordered_index, "tick_quote_imbalance_last"],
                    errors="coerce",
                )
                base.loc[ordered_index, "tick_trade_flow_imbalance_gap"] = (
                    signed_ratio - imbalance
                ).to_numpy()
                base.loc[ordered_index, "tick_trade_pressure_alignment"] = (
                    signed_ratio * imbalance
                ).to_numpy()

        return base
