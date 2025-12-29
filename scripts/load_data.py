#!/usr/bin/env python3
"""
Data loading script.

Loads historical data from various sources into the database.
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


logger = get_logger("load_data", LogCategory.DATA)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load historical data into the database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["alpaca", "csv", "parquet"],
        default="alpaca",
        help="Data source",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Data start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="Data end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Symbols to load (default: from config)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15Min",
        help="Bar timeframe",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory for CSV/Parquet files",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate data after loading",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run (don't insert into database)",
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
    """Validate and parse date strings."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

    if start >= end:
        raise ValueError("Start date must be before end date")

    return start, end


def load_from_alpaca(
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    timeframe: str,
    dry_run: bool = False,
) -> dict:
    """Load data from Alpaca API.

    Args:
        symbols: List of symbols to load.
        start_date: Start date.
        end_date: End date.
        timeframe: Bar timeframe.
        dry_run: If True, don't insert into database.

    Returns:
        Loading statistics.
    """
    logger.info(
        f"Loading data from Alpaca",
        extra={
            "extra_data": {
                "symbols": symbols,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "timeframe": timeframe,
            }
        },
    )

    # Placeholder for actual Alpaca data loading
    # In a real implementation, this would:
    # 1. Connect to Alpaca API
    # 2. Fetch historical bars
    # 3. Validate and transform data
    # 4. Insert into database

    stats = {
        "source": "alpaca",
        "symbols_loaded": len(symbols),
        "bars_loaded": 0,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timeframe": timeframe,
        "dry_run": dry_run,
    }

    return stats


def load_from_csv(
    data_dir: Path,
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    dry_run: bool = False,
) -> dict:
    """Load data from CSV files.

    Args:
        data_dir: Directory containing CSV files.
        symbols: List of symbols to load.
        start_date: Start date.
        end_date: End date.
        dry_run: If True, don't insert into database.

    Returns:
        Loading statistics.
    """
    logger.info(
        f"Loading data from CSV files in {data_dir}",
        extra={
            "extra_data": {
                "data_dir": str(data_dir),
                "symbols": symbols,
            }
        },
    )

    # Placeholder for actual CSV loading
    stats = {
        "source": "csv",
        "data_dir": str(data_dir),
        "symbols_loaded": len(symbols),
        "bars_loaded": 0,
        "dry_run": dry_run,
    }

    return stats


def load_from_parquet(
    data_dir: Path,
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    dry_run: bool = False,
) -> dict:
    """Load data from Parquet files.

    Args:
        data_dir: Directory containing Parquet files.
        symbols: List of symbols to load.
        start_date: Start date.
        end_date: End date.
        dry_run: If True, don't insert into database.

    Returns:
        Loading statistics.
    """
    logger.info(
        f"Loading data from Parquet files in {data_dir}",
        extra={
            "extra_data": {
                "data_dir": str(data_dir),
                "symbols": symbols,
            }
        },
    )

    # Placeholder for actual Parquet loading
    stats = {
        "source": "parquet",
        "data_dir": str(data_dir),
        "symbols_loaded": len(symbols),
        "bars_loaded": 0,
        "dry_run": dry_run,
    }

    return stats


def validate_loaded_data(symbols: list[str]) -> dict:
    """Validate loaded data.

    Args:
        symbols: Symbols to validate.

    Returns:
        Validation results.
    """
    logger.info("Validating loaded data")

    # Placeholder for actual validation
    # Would check for:
    # - Missing bars
    # - Invalid OHLC relationships
    # - Data gaps
    # - Outliers

    results = {
        "symbols_validated": len(symbols),
        "issues_found": 0,
        "issues": [],
    }

    return results


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_format = LogFormat.JSON if args.log_format == "json" else LogFormat.TEXT
    setup_logging(level=args.log_level, log_format=log_format)

    settings = get_settings()
    symbols = args.symbols or settings.symbols
    start_date, end_date = validate_dates(args.start_date, args.end_date)

    logger.info(
        "Starting data load",
        extra={
            "extra_data": {
                "source": args.source,
                "symbols": symbols,
                "start_date": args.start_date,
                "end_date": args.end_date,
            }
        },
    )

    try:
        if args.source == "alpaca":
            stats = load_from_alpaca(
                symbols, start_date, end_date, args.timeframe, args.dry_run
            )
        elif args.source == "csv":
            if not args.data_dir:
                raise ValueError("--data-dir is required for CSV source")
            stats = load_from_csv(
                args.data_dir, symbols, start_date, end_date, args.dry_run
            )
        elif args.source == "parquet":
            if not args.data_dir:
                raise ValueError("--data-dir is required for Parquet source")
            stats = load_from_parquet(
                args.data_dir, symbols, start_date, end_date, args.dry_run
            )

        logger.info(
            "Data loading completed",
            extra={"extra_data": stats},
        )

        if args.validate:
            validation_results = validate_loaded_data(symbols)
            logger.info(
                "Data validation completed",
                extra={"extra_data": validation_results},
            )

    except Exception as e:
        logger.error(f"Data loading failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
