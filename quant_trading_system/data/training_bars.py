"""
Training-dataset bar construction utilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import select

from quant_trading_system.data.intrinsic_bars import (
    AdaptiveThresholdCalculator,
    BarType,
    Trade,
    convert_to_intrinsic_bars,
    create_bar_generator,
)
from quant_trading_system.database.connection import DatabaseManager, get_db_manager
from quant_trading_system.database.models import StockTrade

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingBarConfig:
    mode: str = "time"
    intrinsic_bar_type: str = "volume"
    intrinsic_threshold: float = 0.0
    target_bars_per_day: int = 100
    use_trade_prints_if_available: bool = True

    def normalized_mode(self) -> str:
        return str(self.mode or "time").strip().lower()

    def normalized_intrinsic_bar_type(self) -> BarType:
        raw = str(self.intrinsic_bar_type or "volume").strip().lower()
        return BarType(raw)


class TrainingBarBuilder:
    def __init__(
        self,
        config: TrainingBarConfig,
        db_manager: DatabaseManager | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.db_manager = db_manager or get_db_manager()
        self.logger = logger_ or logger

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])

        working = frame.copy()
        working["symbol"] = working["symbol"].astype(str).str.upper().str.strip()
        working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
        working = working.dropna(subset=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
        working = working.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        if working.empty or self.config.normalized_mode() == "time":
            return working

        if self.config.normalized_mode() != "intrinsic":
            raise ValueError(f"Unsupported training bar mode: {self.config.mode}")

        trades = (
            self._load_trade_prints(working)
            if bool(self.config.use_trade_prints_if_available)
            else pd.DataFrame()
        )
        if not trades.empty:
            self.logger.info(
                "Building intrinsic training bars from stock_trades (%d trade rows)",
                len(trades),
            )
            result = self._build_intrinsic_from_trades(working, trades)
            if not result.empty:
                return result

        self.logger.info(
            "Falling back to intrinsic bars derived from time bars for training mode."
        )
        return self._build_intrinsic_from_time_bars(working)

    def _load_trade_prints(self, frame: pd.DataFrame) -> pd.DataFrame:
        symbols = sorted(frame["symbol"].dropna().unique().tolist())
        min_timestamp = pd.Timestamp(frame["timestamp"].min())
        max_timestamp = pd.Timestamp(frame["timestamp"].max())
        if not symbols:
            return pd.DataFrame()

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
            self.logger.warning("Intrinsic training bar trade load skipped: %s", exc)
            return pd.DataFrame()
        if not rows:
            return pd.DataFrame()

        trades = pd.DataFrame(rows, columns=["symbol", "timestamp", "price", "size"])
        trades["symbol"] = trades["symbol"].astype(str).str.upper().str.strip()
        trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
        trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
        trades["size"] = pd.to_numeric(trades["size"], errors="coerce")
        return trades.dropna(subset=["symbol", "timestamp", "price", "size"]).sort_values(
            ["symbol", "timestamp"]
        ).reset_index(drop=True)

    def _threshold_for_symbol(self, symbol_frame: pd.DataFrame) -> float:
        configured = float(self.config.intrinsic_threshold)
        if configured > 0.0:
            return configured
        calculator = AdaptiveThresholdCalculator(
            target_bars_per_day=max(10, int(self.config.target_bars_per_day))
        )
        thresholds = calculator.calculate_thresholds(symbol_frame)
        return float(thresholds[self.config.normalized_intrinsic_bar_type()])

    def _build_intrinsic_from_time_bars(self, frame: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        bar_type = self.config.normalized_intrinsic_bar_type()
        for symbol, symbol_frame in frame.groupby("symbol", sort=False):
            threshold = self._threshold_for_symbol(symbol_frame)
            converted = convert_to_intrinsic_bars(
                symbol_frame[["timestamp", "open", "high", "low", "close", "volume"]],
                bar_type=bar_type,
                threshold=threshold,
            )
            if converted.empty:
                continue
            converted["symbol"] = symbol
            parts.append(converted)

        if not parts:
            return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
        return pd.concat(parts, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(
            drop=True
        )

    def _build_intrinsic_from_trades(
        self,
        ohlcv_frame: pd.DataFrame,
        trades: pd.DataFrame,
    ) -> pd.DataFrame:
        bar_type = self.config.normalized_intrinsic_bar_type()
        parts: list[pd.DataFrame] = []
        for symbol, symbol_trades in trades.groupby("symbol", sort=False):
            symbol_ohlcv = ohlcv_frame[ohlcv_frame["symbol"] == symbol]
            threshold = self._threshold_for_symbol(symbol_ohlcv)
            generator = create_bar_generator(bar_type, threshold)
            bars: list[dict[str, Any]] = []
            for row in symbol_trades.itertuples(index=False):
                trade = Trade(
                    timestamp=pd.Timestamp(row.timestamp).to_pydatetime(),
                    price=float(row.price),
                    volume=float(row.size),
                )
                bar = generator.process_trade(trade)
                if bar is not None:
                    payload = bar.to_dict()
                    payload["symbol"] = symbol
                    bars.append(payload)
            if not bars:
                continue
            parts.append(pd.DataFrame(bars))

        if not parts:
            return pd.DataFrame()
        result = pd.concat(parts, ignore_index=True)
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
        return result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
