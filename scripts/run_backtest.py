#!/usr/bin/env python3
"""
Backtest runner script.

Runs backtests on historical data with configurable parameters.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant_trading_system.config.settings import get_settings
from quant_trading_system.monitoring.logger import setup_logging, LogFormat, get_logger, LogCategory


logger = get_logger("run_backtest", LogCategory.SYSTEM)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run backtest on historical data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Symbols to backtest (default: from config)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000.0,
        help="Initial capital for backtest",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        help="Model to use for backtest",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backtest_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--log-format",
        choices=["json", "text"],
        default="text",
        help="Log format",
    )
    return parser.parse_args()


def validate_dates(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    """Validate and parse date strings.

    Args:
        start_date: Start date string.
        end_date: End date string.

    Returns:
        Tuple of parsed datetime objects.

    Raises:
        ValueError: If dates are invalid.
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

    if start >= end:
        raise ValueError("Start date must be before end date")

    return start, end


def run_backtest(args: argparse.Namespace) -> dict:
    """Run the backtest.

    Args:
        args: Command line arguments.

    Returns:
        Dictionary with backtest results.
    """
    settings = get_settings()
    start_date, end_date = validate_dates(args.start_date, args.end_date)

    symbols = args.symbols or settings.symbols

    logger.info(
        f"Starting backtest",
        extra={
            "extra_data": {
                "start_date": args.start_date,
                "end_date": args.end_date,
                "symbols": symbols,
                "initial_capital": args.initial_capital,
                "model": args.model,
            }
        },
    )

    # Placeholder for actual backtest logic
    # In a real implementation, this would:
    # 1. Load historical data
    # 2. Initialize models
    # 3. Run simulation
    # 4. Calculate metrics
    # 5. Generate reports

    results = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "symbols": symbols,
        "initial_capital": args.initial_capital,
        "final_equity": args.initial_capital,  # Placeholder
        "total_return": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "total_trades": 0,
        "win_rate": 0.0,
    }

    logger.info(
        "Backtest completed",
        extra={"extra_data": results},
    )

    return results


def save_results(results: dict, output_dir: Path) -> None:
    """Save backtest results.

    Args:
        results: Backtest results.
        output_dir: Output directory.
    """
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"backtest_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_format = LogFormat.JSON if args.log_format == "json" else LogFormat.TEXT
    setup_logging(level=args.log_level, log_format=log_format)

    try:
        results = run_backtest(args)
        save_results(results, args.output_dir)
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
