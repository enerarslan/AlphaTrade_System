#!/usr/bin/env python3
"""
Export PostgreSQL data to Parquet files for ML training.

This script is part of the hybrid architecture where PostgreSQL + TimescaleDB
is the source of truth, and Parquet files are used for ML training (faster I/O).

Usage:
    python scripts/export_training_data.py --output data/training
    python scripts/export_training_data.py --symbols AAPL MSFT --start-date 2024-01-01
    python scripts/export_training_data.py --features-only --output data/training/features

@agent: @data, @mlquant
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from quant_trading_system.data.db_loader import get_db_loader
from quant_trading_system.data.db_feature_store import get_db_feature_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def export_ohlcv_data(
    output_dir: Path,
    symbols: list[str] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, Path]:
    """
    Export OHLCV data from PostgreSQL to Parquet files.

    Args:
        output_dir: Output directory for Parquet files.
        symbols: List of symbols to export. If None, exports all.
        start_date: Start date filter.
        end_date: End date filter.

    Returns:
        Dictionary mapping symbol to output path.
    """
    db_loader = get_db_loader()
    ohlcv_dir = output_dir / "ohlcv"
    ohlcv_dir.mkdir(parents=True, exist_ok=True)

    if symbols is None:
        symbols = db_loader.get_available_symbols()
        logger.info(f"Found {len(symbols)} symbols in database")

    results = {}
    for symbol in symbols:
        try:
            output_path = ohlcv_dir / f"{symbol}.parquet"
            db_loader.export_to_parquet(
                symbol, output_path, start_date, end_date
            )
            results[symbol] = output_path
            logger.info(f"Exported OHLCV for {symbol}")
        except Exception as e:
            logger.error(f"Failed to export OHLCV for {symbol}: {e}")

    return results


def export_features_data(
    output_dir: Path,
    symbols: list[str] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, Path]:
    """
    Export feature data from PostgreSQL to Parquet files.

    Args:
        output_dir: Output directory for Parquet files.
        symbols: List of symbols to export. If None, exports all.
        start_date: Start date filter.
        end_date: End date filter.

    Returns:
        Dictionary mapping symbol to output path.
    """
    feature_store = get_db_feature_store()
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    if symbols is None:
        symbols = feature_store.get_available_symbols()
        logger.info(f"Found {len(symbols)} symbols with features in database")

    results = {}
    for symbol in symbols:
        try:
            output_path = features_dir / f"{symbol}_features.parquet"
            feature_store.export_to_parquet(
                symbol, output_path, start_date, end_date
            )
            results[symbol] = output_path
            logger.info(f"Exported features for {symbol}")
        except Exception as e:
            logger.error(f"Failed to export features for {symbol}: {e}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export PostgreSQL data to Parquet for ML training"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/training"),
        help="Output directory for Parquet files",
    )
    parser.add_argument(
        "--symbols", "-s",
        nargs="+",
        help="Symbols to export (default: all)",
    )
    parser.add_argument(
        "--start-date",
        type=lambda s: datetime.fromisoformat(s),
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=lambda s: datetime.fromisoformat(s),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--ohlcv-only",
        action="store_true",
        help="Export only OHLCV data",
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Export only feature data",
    )

    args = parser.parse_args()

    logger.info(f"Exporting training data to {args.output}")

    total_exported = 0

    # Export OHLCV data
    if not args.features_only:
        ohlcv_results = export_ohlcv_data(
            args.output, args.symbols, args.start_date, args.end_date
        )
        total_exported += len(ohlcv_results)
        logger.info(f"Exported OHLCV data for {len(ohlcv_results)} symbols")

    # Export feature data
    if not args.ohlcv_only:
        features_results = export_features_data(
            args.output, args.symbols, args.start_date, args.end_date
        )
        total_exported += len(features_results)
        logger.info(f"Exported features for {len(features_results)} symbols")

    logger.info(f"Export complete. Total files: {total_exported}")


if __name__ == "__main__":
    main()
