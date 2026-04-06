#!/usr/bin/env python3
"""
Artifact-aware trading entrypoint aligned with the current package APIs.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant_trading_system.backtest.promotion import (
    PromotionPackageContract,
    PromotionSignalAdapter,
    load_promotion_package,
)
from quant_trading_system.config.settings import get_settings
from quant_trading_system.core.data_types import Direction, OHLCVBar, TradeSignal
from quant_trading_system.data.data_access import DataAccessConfig, DataAccessLayer
from quant_trading_system.monitoring.logger import (
    LogCategory,
    LogFormat,
    get_logger,
    setup_logging,
)
from quant_trading_system.trading.portfolio_manager import RebalanceMethod
from quant_trading_system.trading.strategy import StrategyConfig, StrategyType, create_strategy
from quant_trading_system.trading.trading_engine import TradingMode, create_trading_engine

DEFAULT_TRADE_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

logger = get_logger("trade", LogCategory.TRADING)


def _normalize_metadata_value(value: Any) -> Any:
    """Normalize pandas/numpy scalars into JSON-friendly primitives."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


class PromotionPackageSignalSource:
    """Generate live paper-trading signals from a promotion package."""

    def __init__(
        self,
        contract: PromotionPackageContract,
        *,
        use_gpu: bool = False,
        max_bars: int = 4096,
        use_database_liquidity: bool = False,
    ) -> None:
        self.contract = contract
        self.adapter = PromotionSignalAdapter(contract, use_gpu=use_gpu, logger_=logger)
        self.max_bars = max_bars
        self.use_database_liquidity = bool(use_database_liquidity)
        self._history: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.max_bars)
        )
        self._last_bar_timestamp: dict[str, datetime] = {}
        self._last_batch_metrics: dict[str, Any] = {
            "raw_candidates": 0,
            "meta_passed": 0,
            "edge_passed": 0,
        }
        self._data_access = (
            DataAccessLayer(
                DataAccessConfig(
                    use_database=True,
                    fallback_to_files=False,
                )
            )
            if self.use_database_liquidity
            else None
        )

    @staticmethod
    def _bar_to_record(bar: OHLCVBar) -> dict[str, Any]:
        return {
            "timestamp": bar.timestamp,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": int(bar.volume),
        }

    def _append_new_bars(self, bars: dict[str, OHLCVBar]) -> list[str]:
        updated_symbols: list[str] = []
        for raw_symbol, bar in bars.items():
            symbol = str(raw_symbol).strip().upper()
            last_timestamp = self._last_bar_timestamp.get(symbol)
            if last_timestamp is not None and bar.timestamp <= last_timestamp:
                continue
            self._history[symbol].append(self._bar_to_record(bar))
            self._last_bar_timestamp[symbol] = bar.timestamp
            updated_symbols.append(symbol)
        return updated_symbols

    def _build_history_frames(self) -> dict[str, pd.DataFrame]:
        symbols = list(self.contract.symbols or tuple(self._history.keys()))
        frames: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            rows = list(self._history.get(symbol, ()))
            if not rows:
                continue
            frame = pd.DataFrame(rows)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            frame = frame.dropna(subset=["timestamp"]).drop_duplicates(
                subset=["timestamp"],
                keep="last",
            )
            frame = frame.sort_values("timestamp").set_index("timestamp")
            frames[symbol] = frame.loc[:, ["open", "high", "low", "close", "volume"]]
        return frames

    def get_last_batch_metrics(self) -> dict[str, Any]:
        """Expose the latest per-bar candidate telemetry for paper/live parity."""
        return dict(self._last_batch_metrics)

    def _reset_last_batch_metrics(self) -> None:
        self._last_batch_metrics = {
            "raw_candidates": 0,
            "meta_passed": 0,
            "edge_passed": 0,
        }

    def generate(
        self,
        bars: dict[str, OHLCVBar],
        features: dict[str, Any] | None = None,
        predictions: dict[str, Any] | None = None,
        portfolio: Any | None = None,
    ) -> list[TradeSignal]:
        """Generate fresh signals only for genuinely new bars."""
        updated_symbols = self._append_new_bars(bars)
        if not updated_symbols:
            self._reset_last_batch_metrics()
            return []

        history_frames = self._build_history_frames()
        if not history_frames:
            self._reset_last_batch_metrics()
            return []

        try:
            feature_frames = self.adapter.compute_features(history_frames)
            signal_frames = self.adapter.generate_signal_frames(history_frames, feature_frames)
        except ValueError as exc:
            logger.debug("Promotion package warm-up incomplete, skipping signal batch: %s", exc)
            self._reset_last_batch_metrics()
            return []
        except Exception as exc:
            logger.warning("Promotion package signal generation failed: %s", exc)
            self._reset_last_batch_metrics()
            return []

        generated: list[TradeSignal] = []
        batch_metrics = {
            "raw_candidates": 0,
            "meta_passed": 0,
            "edge_passed": 0,
        }
        for symbol in updated_symbols:
            signal_frame = signal_frames.get(symbol)
            current_timestamp = self._last_bar_timestamp.get(symbol)
            if signal_frame is None or signal_frame.empty or current_timestamp is None:
                continue

            matches = signal_frame[
                pd.to_datetime(signal_frame["timestamp"], utc=True, errors="coerce")
                == pd.Timestamp(current_timestamp)
            ]
            if matches.empty:
                continue
            signal_row = matches.iloc[-1].copy()
            batch_metrics["raw_candidates"] += int(bool(signal_row.get("raw_candidate", False)))
            batch_metrics["meta_passed"] += int(bool(signal_row.get("meta_passed", False)))
            batch_metrics["edge_passed"] += int(bool(signal_row.get("edge_passed", False)))

            if self._data_access is not None:
                try:
                    liquidity_metrics = self._data_access.get_trailing_liquidity_metrics(
                        symbol,
                        end_time=current_timestamp,
                        lookback_days=int(getattr(self.contract, "adv_lookback_days", 20)),
                        timeframe=str(getattr(self.contract, "timeframe", "15Min")),
                    )
                except Exception as exc:
                    logger.debug("Database liquidity lookup failed for %s: %s", symbol, exc)
                    liquidity_metrics = {}
                for key in ("avg_daily_volume", "avg_daily_dollar_volume", "spread_bps"):
                    if key in liquidity_metrics:
                        signal_row[key] = liquidity_metrics[key]
            signal_value = float(signal_row.get("signal", 0.0) or 0.0)
            if abs(signal_value) <= 0.0:
                continue

            direction = Direction.LONG if signal_value > 0 else Direction.SHORT
            confidence = float(max(0.0, min(1.0, signal_row.get("confidence", abs(signal_value)))))
            horizon = int(max(1, float(signal_row.get("horizon", 1))))
            metadata = {
                column: _normalize_metadata_value(signal_row[column])
                for column in signal_frame.columns
                if column not in {"signal", "confidence", "horizon", "model_source", "timestamp"}
            }
            metadata["hold_contract_enabled"] = True
            metadata["prediction_horizon_bars"] = horizon
            metadata["max_holding_bars"] = int(
                max(1, float(signal_row.get("max_holding_bars", horizon)))
            )

            generated.append(
                TradeSignal(
                    symbol=symbol,
                    direction=direction,
                    strength=max(-1.0, min(1.0, signal_value)),
                    confidence=confidence,
                    timestamp=current_timestamp,
                    horizon=horizon,
                    model_source=str(signal_row.get("model_source", self.contract.model_source)),
                    metadata=metadata,
                )
            )

        self._last_batch_metrics = batch_metrics
        return generated


def _resolve_trading_mode(raw_mode: str) -> TradingMode:
    normalized = str(raw_mode).strip().lower()
    if normalized == "live":
        return TradingMode.LIVE
    if normalized == "dry-run":
        return TradingMode.DRY_RUN
    return TradingMode.PAPER


def _resolve_strategy_type(raw_strategy: str) -> StrategyType:
    normalized = str(raw_strategy).strip().lower()
    if normalized == "mean_reversion":
        return StrategyType.MEAN_REVERSION
    if normalized == "ml_based":
        return StrategyType.ML_BASED
    return StrategyType.MOMENTUM


def _configure_engine(
    args: argparse.Namespace,
    contract: PromotionPackageContract | None,
):
    settings = get_settings()
    symbols = list(getattr(args, "symbols", []) or [])
    if contract is not None and (not symbols or symbols == DEFAULT_TRADE_SYMBOLS) and contract.symbols:
        symbols = list(contract.symbols)
    if not symbols:
        symbols = list(DEFAULT_TRADE_SYMBOLS)

    mode = _resolve_trading_mode(args.mode)
    engine = create_trading_engine(
        mode=mode,
        symbols=symbols,
        api_key=settings.alpaca.api_key,
        api_secret=settings.alpaca.api_secret,
    )
    engine.config.symbols = symbols
    engine.config.kill_switch_drawdown = float(args.kill_switch_drawdown)
    if contract is not None and contract.timeframe:
        engine.config.bar_interval = str(contract.timeframe)

    position_config = engine.portfolio_manager.position_sizer.config
    position_config.max_total_positions = int(getattr(args, "max_positions", 10))
    if contract is not None:
        position_config.percent_of_equity = float(contract.max_position_pct)
        position_config.max_position_pct = float(contract.max_position_pct)
        position_config.max_total_positions = int(contract.max_total_positions)
        position_config.confidence_position_sizing = bool(contract.confidence_position_sizing)
        position_config.min_confidence_position_scale = float(
            contract.min_confidence_position_scale
        )
        engine.portfolio_manager.rebalance_config.method = RebalanceMethod.SIGNAL_DRIVEN
        source = PromotionPackageSignalSource(contract, use_database_liquidity=True)
        engine._signal_batch_metrics_provider = source
        engine.signal_generator.add_external_source(
            f"promotion_package:{contract.model_name}",
            source.generate,
        )
    else:
        strategy_type = _resolve_strategy_type(getattr(args, "strategy", "momentum"))
        strategy = create_strategy(
            strategy_type,
            StrategyConfig(
                strategy_id=f"{strategy_type.value}_runtime",
                strategy_type=strategy_type,
                symbols=symbols,
                max_positions=int(getattr(args, "max_positions", 10)),
            ),
        )
        engine.signal_generator.add_strategy(strategy)

    return engine, symbols


async def _run_engine_until_stopped(engine, duration_minutes: int | None) -> None:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        loop.call_soon_threadsafe(stop_event.set)

    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, lambda *_: _request_stop())
    signal.signal(signal.SIGTERM, lambda *_: _request_stop())

    auto_stop_task: asyncio.Task | None = None
    try:
        await engine.start()
        if duration_minutes is not None and duration_minutes > 0:
            auto_stop_task = asyncio.create_task(asyncio.sleep(duration_minutes * 60))
            auto_stop_task.add_done_callback(lambda *_: stop_event.set())

        await stop_event.wait()
    finally:
        if auto_stop_task is not None:
            auto_stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await auto_stop_task
        await engine.stop()
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)


def run_trading(args: argparse.Namespace) -> int:
    """Run paper/live trading with either promotion-package or classic strategies."""
    setup_logging(log_level=args.log_level, log_format=LogFormat.TEXT)

    promotion_package_path = getattr(args, "promotion_package", None)
    contract: PromotionPackageContract | None = None
    if promotion_package_path is not None:
        package_path = Path(promotion_package_path).resolve()
        if not package_path.exists():
            logger.error(f"Promotion package not found: {package_path}")
            return 1
        contract = load_promotion_package(package_path)

    try:
        engine, symbols = _configure_engine(args, contract)
    except Exception as exc:
        logger.exception("Failed to configure trading engine: %s", exc)
        return 1

    strategy_name = (
        f"promotion_package:{contract.model_name}" if contract is not None else getattr(args, "strategy", "momentum")
    )
    logger.info("=" * 60)
    logger.info("ALPHATRADE TRADING SESSION")
    logger.info("=" * 60)
    logger.info("Mode: %s", args.mode)
    logger.info("Symbols: %s", symbols)
    logger.info("Capital reference: $%s", f"{float(args.capital):,.2f}")
    logger.info("Strategy: %s", strategy_name)
    if contract is not None:
        logger.info("Promotion package: %s", contract.package_path)
        logger.info("Model: %s (%s)", contract.model_name, contract.model_type)
        logger.info("Bar interval: %s", engine.config.bar_interval)
    logger.info("=" * 60)

    try:
        asyncio.run(_run_engine_until_stopped(engine, getattr(args, "duration", None)))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logger.exception("Trading session failed: %s", exc)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build standalone CLI parser for the trading script."""
    parser = argparse.ArgumentParser(description="AlphaTrade Trading Script")
    parser.add_argument("--mode", "-m", choices=["live", "paper", "dry-run"], default="paper")
    parser.add_argument("--symbols", "-s", nargs="+", default=list(DEFAULT_TRADE_SYMBOLS))
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--strategy", default="momentum")
    parser.add_argument("--promotion-package", type=Path, default=None)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--kill-switch-drawdown", type=float, default=0.10)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_trading(args)


def main_paper(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.mode = "paper"
    return run_trading(args)


def main_live(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.mode = "live"
    return run_trading(args)


if __name__ == "__main__":
    sys.exit(main())
