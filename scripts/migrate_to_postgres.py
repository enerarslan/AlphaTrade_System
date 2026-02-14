#!/usr/bin/env python3
"""
Migrate historical data from CSV files to PostgreSQL + TimescaleDB.

This script performs the initial data migration from file-based storage
to the PostgreSQL database. It should be run once after setting up
the database with the Alembic migrations.

Usage:
    python scripts/migrate_to_postgres.py
    python scripts/migrate_to_postgres.py --source data/raw --batch-size 50000
    python scripts/migrate_to_postgres.py --symbols AAPL MSFT --verify

@agent: @data, @infra
"""

from __future__ import annotations

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import polars as pl

from quant_trading_system.data.db_loader import get_db_loader, DatabaseDataLoader
from quant_trading_system.database.connection import get_db_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_csv_files(data_dir: Path) -> list[tuple[str, Path]]:
    """
    Find all CSV files in the data directory.

    Returns:
        List of (symbol, file_path) tuples.
    """
    files = []

    # Check for CSV files in main directory
    for csv_file in data_dir.glob("*.csv"):
        symbol = csv_file.stem.upper()
        # Remove common suffixes
        for suffix in ["_15MIN", "_15Min", "_1D", "_DAILY"]:
            if symbol.endswith(suffix):
                symbol = symbol[:-len(suffix)]
                break
        files.append((symbol, csv_file))

    # Check subdirectories
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            for csv_file in subdir.glob("*.csv"):
                symbol = csv_file.stem.upper()
                for suffix in ["_15MIN", "_15Min", "_1D", "_DAILY"]:
                    if symbol.endswith(suffix):
                        symbol = symbol[:-len(suffix)]
                        break
                files.append((symbol, csv_file))

    return files


def verify_migration(symbol: str, csv_path: Path, db_loader: DatabaseDataLoader) -> bool:
    """
    Verify that migration was successful for a symbol.

    Args:
        symbol: Symbol to verify.
        csv_path: Original CSV file path.
        db_loader: Database loader instance.

    Returns:
        True if verification passed.
    """
    try:
        # Load from CSV
        csv_df = pl.read_csv(csv_path, try_parse_dates=True)
        csv_count = len(csv_df)

        # Load from database
        db_df = db_loader.load_symbol(symbol)
        db_count = len(db_df)

        if csv_count == db_count:
            logger.info(f"Verification PASSED for {symbol}: {db_count} rows")
            return True
        else:
            logger.warning(
                f"Verification WARNING for {symbol}: "
                f"CSV has {csv_count} rows, DB has {db_count} rows"
            )
            return False

    except Exception as e:
        logger.error(f"Verification FAILED for {symbol}: {e}")
        return False


def migrate_csv_to_postgres(
    source_dir: Path,
    symbols: list[str] | None = None,
    batch_size: int = 50000,
    verify: bool = False,
    source_timezone: str = "America/New_York",
    resume: bool = True,
) -> dict[str, int]:
    """
    Migrate CSV files to PostgreSQL.

    Args:
        source_dir: Directory containing CSV files.
        symbols: Specific symbols to migrate. If None, migrates all.
        batch_size: Batch size for bulk inserts.
        verify: Whether to verify migration after each symbol.

    Returns:
        Dictionary mapping symbol to number of rows migrated.
    """
    db_loader = get_db_loader()
    results = {}

    # Find CSV files
    csv_files = find_csv_files(source_dir)
    logger.info(f"Found {len(csv_files)} CSV files in {source_dir}")

    # Filter by symbols if specified
    if symbols:
        symbols_upper = [s.upper() for s in symbols]
        csv_files = [(s, p) for s, p in csv_files if s in symbols_upper]
        logger.info(f"Filtered to {len(csv_files)} files for specified symbols")

    for symbol, csv_path in csv_files:
        try:
            logger.info(f"Migrating {symbol} from {csv_path}...")

            rows = db_loader.load_csv_to_database(
                csv_path,
                symbol=symbol,
                batch_size=batch_size,
                validate=True,
                source_timezone=source_timezone,
                resume=resume,
            )

            results[symbol] = rows
            logger.info(f"Migrated {rows} rows for {symbol}")

            # Verify if requested
            if verify:
                verify_migration(symbol, csv_path, db_loader)

        except Exception as e:
            logger.error(f"Failed to migrate {symbol}: {e}")
            results[symbol] = -1

    return results


def check_database_connection() -> bool:
    """Check if database is accessible."""
    from sqlalchemy import text
    try:
        db_manager = get_db_manager()
        with db_manager.session() as session:
            session.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def check_timescaledb() -> bool:
    """Check if TimescaleDB extension is enabled."""
    from sqlalchemy import text
    try:
        db_manager = get_db_manager()
        with db_manager.session() as session:
            result = session.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
            )
            if result.fetchone():
                logger.info("TimescaleDB extension is enabled")
                return True
            else:
                logger.warning("TimescaleDB extension is NOT enabled")
                return False
    except Exception as e:
        logger.error(f"Failed to check TimescaleDB: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate CSV data to PostgreSQL + TimescaleDB"
    )
    parser.add_argument(
        "--source", "-s",
        type=Path,
        default=Path("data/raw"),
        help="Source directory containing CSV files",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Specific symbols to migrate (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Batch size for bulk inserts",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify migration after each symbol",
    )
    parser.add_argument(
        "--source-timezone",
        type=str,
        default="America/New_York",
        help="Timezone for naive source timestamps (default: America/New_York)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resumable chunk checkpoints",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check database connection and exit",
    )
    parser.add_argument(
        "--export-after",
        action="store_true",
        help="Export to Parquet after migration",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("data/training"),
        help="Directory for Parquet export",
    )

    args = parser.parse_args()

    # Check database connection
    if not check_database_connection():
        logger.error("Cannot proceed without database connection")
        sys.exit(1)

    # Check TimescaleDB
    check_timescaledb()

    if args.check_only:
        logger.info("Database check complete")
        sys.exit(0)

    # Verify source directory exists
    if not args.source.exists():
        logger.error(f"Source directory does not exist: {args.source}")
        sys.exit(1)

    # Run migration
    logger.info("Starting migration...")
    results = migrate_csv_to_postgres(
        args.source,
        args.symbols,
        args.batch_size,
        args.verify,
        source_timezone=args.source_timezone,
        resume=not args.no_resume,
    )

    # Summary
    successful = sum(1 for v in results.values() if v > 0)
    failed = sum(1 for v in results.values() if v < 0)
    total_rows = sum(v for v in results.values() if v > 0)

    logger.info(f"\nMigration Summary:")
    logger.info(f"  Symbols migrated: {successful}")
    logger.info(f"  Symbols failed: {failed}")
    logger.info(f"  Total rows: {total_rows}")

    # Export to Parquet if requested
    if args.export_after and successful > 0:
        logger.info("\nExporting to Parquet for ML training...")
        from scripts.export_training_data import export_ohlcv_data
        export_results = export_ohlcv_data(args.export_dir, list(results.keys()))
        logger.info(f"Exported {len(export_results)} Parquet files")


if __name__ == "__main__":
    main()
