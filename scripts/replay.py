#!/usr/bin/env python3
"""Deterministic trading-day replay runner with SLO policy gates."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant_trading_system.backtest.engine import ExecutionMode
from quant_trading_system.backtest.replay import (
    ReplaySLOGates,
    ReplayScenario,
    ReplaySignalConfig,
    run_replay_scenario,
)
from quant_trading_system.data.timeframe import DEFAULT_TIMEFRAME, normalize_timeframe
from quant_trading_system.data.loader import DataLoader


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for deterministic replay runs."""
    parser = argparse.ArgumentParser(
        description="Run deterministic trading-day replay with execution/risk SLO gates"
    )

    parser.add_argument("--scenario-id", type=str, default="trading_day_replay")
    parser.add_argument("--start", type=str, required=True, help="Replay start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="Replay end date (YYYY-MM-DD)")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols for replay")

    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument(
        "--execution-mode",
        choices=["realistic", "optimistic", "pessimistic"],
        default="realistic",
    )
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--commission-bps", type=float, default=1.0)

    parser.add_argument("--return-threshold-bps", type=float, default=5.0)
    parser.add_argument("--signal-confidence", type=float, default=0.75)
    parser.add_argument("--signal-horizon", type=int, default=1)
    parser.add_argument("--allow-short", dest="allow_short", action="store_true")
    parser.add_argument("--no-allow-short", dest="allow_short", action="store_false")
    parser.set_defaults(allow_short=True)

    parser.add_argument("--max-drawdown", type=float, default=0.20)
    parser.add_argument("--max-rejection-rate", type=float, default=0.20)
    parser.add_argument("--max-avg-slippage-bps", type=float, default=35.0)
    parser.add_argument("--max-risk-escalations", type=int, default=0)
    parser.add_argument("--max-escalation-latency-ms", type=float, default=2000.0)
    parser.add_argument("--min-orders-for-slo", type=int, default=1)
    parser.add_argument("--no-fail-on-kill-switch", action="store_true")

    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME)
    parser.add_argument("--use-database", action="store_true", default=True)
    parser.add_argument("--no-database", action="store_true")

    parser.add_argument("--output", type=Path, help="Optional output path for replay JSON result")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument(
        "--fail-on-slo-breach",
        action="store_true",
        default=True,
        help="Return exit code 2 when SLO gates fail (default: True)",
    )
    parser.add_argument(
        "--no-fail-on-slo-breach",
        dest="fail_on_slo_breach",
        action="store_false",
        help="Return success code even when SLO gates fail",
    )
    return parser


def _parse_utc_date(date_str: str) -> datetime:
    """Parse YYYY-MM-DD date string into UTC datetime."""
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _to_pandas_frame(frame: Any) -> pd.DataFrame:
    """Convert DataLoader output to pandas DataFrame."""
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    raise TypeError(f"Unsupported frame type from loader: {type(frame)}")


def _load_replay_data(
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    timeframe: str,
) -> dict[str, pd.DataFrame]:
    """Load replay data for all symbols and normalize to pandas."""
    loader = DataLoader(data_dir=Path("data/raw"), use_database=True)
    loaded: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        normalized_symbol = symbol.upper()
        frame = loader.load_symbol(
            symbol=normalized_symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
        )
        loaded[normalized_symbol] = _to_pandas_frame(frame)

    if not loaded:
        raise ValueError("Replay data could not be loaded for any symbol")
    return loaded


def _verify_replay_infra() -> None:
    """Fail-fast institutional infrastructure check for replay runs."""
    from quant_trading_system.database.connection import get_db_manager, get_redis_manager

    if not get_db_manager().health_check():
        raise RuntimeError("PostgreSQL health check failed. Replay requires PostgreSQL.")
    if not get_redis_manager().health_check():
        raise RuntimeError("Redis health check failed. Replay requires Redis.")


def run_replay(args: argparse.Namespace) -> int:
    """Execute replay command from parsed arguments."""
    try:
        start_date = _parse_utc_date(args.start)
        end_date = _parse_utc_date(args.end)
    except ValueError as exc:
        print(f"Invalid date format: {exc}", file=sys.stderr)
        return 1

    if end_date <= start_date:
        print(f"End date ({args.end}) must be after start date ({args.start})", file=sys.stderr)
        return 1

    if not bool(getattr(args, "use_database", True)) or bool(getattr(args, "no_database", False)):
        print("Institutional replay requires PostgreSQL + Redis. CSV mode is disabled.", file=sys.stderr)
        return 1

    try:
        _verify_replay_infra()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    timeframe = normalize_timeframe(getattr(args, "timeframe", DEFAULT_TIMEFRAME))

    try:
        symbols = [symbol.upper() for symbol in args.symbols]
        data = _load_replay_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
        )
        scenario = ReplayScenario(
            scenario_id=str(getattr(args, "scenario_id", "trading_day_replay")),
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=Decimal(str(args.capital)),
            execution_mode=ExecutionMode(args.execution_mode),
            slippage_bps=float(args.slippage_bps),
            commission_bps=float(args.commission_bps),
            allow_short=bool(getattr(args, "allow_short", True)),
            signal=ReplaySignalConfig(
                return_threshold_bps=float(args.return_threshold_bps),
                confidence=float(args.signal_confidence),
                horizon_bars=max(1, int(args.signal_horizon)),
            ),
        )
        slo_gates = ReplaySLOGates(
            max_drawdown=float(args.max_drawdown),
            max_rejection_rate=float(args.max_rejection_rate),
            max_avg_slippage_bps=float(args.max_avg_slippage_bps),
            max_risk_escalations=int(args.max_risk_escalations),
            max_escalation_latency_ms=float(args.max_escalation_latency_ms),
            min_orders_for_gate=max(0, int(args.min_orders_for_slo)),
            fail_on_kill_switch_active=not bool(getattr(args, "no_fail_on_kill_switch", False)),
        )

        outcome = run_replay_scenario(data=data, scenario=scenario, slo_gates=slo_gates)
    except Exception as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 1

    payload = outcome.to_dict()
    if getattr(args, "output", None):
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if getattr(args, "format", "text") == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(f"Replay scenario: {outcome.scenario_id}")
        print(f"Passed: {outcome.passed}")
        print(f"Bars processed: {outcome.bars_processed}")
        print(f"Trades closed: {outcome.trades_closed}")
        print(f"Total return: {outcome.total_return:.4%}")
        print(f"Max drawdown: {outcome.max_drawdown:.4%}")
        if outcome.violations:
            print("Violations:")
            for violation in outcome.violations:
                print(f"  - {violation}")

    if not outcome.passed and bool(getattr(args, "fail_on_slo_breach", True)):
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    """Console entrypoint for quant-replay."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
