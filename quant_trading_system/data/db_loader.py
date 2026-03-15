"""
Database-backed data loader for PostgreSQL + TimescaleDB.

Provides the same interface as DataLoader but reads/writes from the database.
Supports bulk loading from CSV files for initial migration and Parquet exports
for ML training (hybrid architecture).

@agent: @data
"""

from __future__ import annotations

import logging
import json
import hashlib
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator

import polars as pl
import pandas as pd

from quant_trading_system.core.data_types import OHLCVBar
from quant_trading_system.core.exceptions import DataError, DataNotFoundError, DataValidationError
from quant_trading_system.data.ingestion_contracts import (
    OHLCVIngestionContract,
    apply_ohlcv_ingestion_contract,
    quality_summary,
)
from quant_trading_system.data.timeframe import (
    DEFAULT_TIMEFRAME,
    infer_symbol_and_timeframe,
    normalize_timeframe,
    timeframe_slug,
)
from quant_trading_system.database.connection import DatabaseManager, get_db_manager
from quant_trading_system.database.models import OHLCVBar as OHLCVBarModel
from quant_trading_system.database.repository import (
    OHLCVRepository,
    get_ohlcv_repository,
)

logger = logging.getLogger(__name__)


class DatabaseDataLoader:
    """
    Load and persist OHLCV data from/to PostgreSQL + TimescaleDB.

    This loader provides database-backed data access while maintaining
    compatibility with the file-based DataLoader interface.

    Key features:
    - Load data from TimescaleDB hypertables
    - Bulk insert from CSV/Parquet for initial migration
    - Export to Parquet for ML training (hybrid architecture)
    - Efficient batch operations using psycopg2 executemany
    """

    def __init__(
        self,
        db_manager: DatabaseManager | None = None,
        batch_size: int = 10000,
    ):
        """
        Initialize the database data loader.

        Args:
            db_manager: Database manager instance. If None, uses default.
            batch_size: Default batch size for bulk operations.
        """
        self._db = db_manager or get_db_manager()
        self._ohlcv_repo = OHLCVRepository(self._db)
        self._batch_size = batch_size

    def load_symbol(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        columns: list[str] | None = None,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> pl.DataFrame:
        """
        Load OHLCV data for a single symbol from database.

        Args:
            symbol: Ticker symbol to load.
            start_date: Start date filter (inclusive).
            end_date: End date filter (inclusive).
            columns: Specific columns to return (None for all).

        Returns:
            Polars DataFrame with OHLCV data.

        Raises:
            DataNotFoundError: If no data found for symbol.
        """
        symbol = symbol.upper()
        timeframe = normalize_timeframe(timeframe)

        with self._db.session() as session:
            # Use a high limit for full data retrieval
            # TimescaleDB handles large queries efficiently
            bars = self._ohlcv_repo.get_bars(
                session,
                symbol,
                timeframe=timeframe,
                start_time=start_date,
                end_time=end_date,
                limit=10_000_000,  # Effectively unlimited
            )

            if not bars:
                raise DataNotFoundError(f"No data found in database for symbol: {symbol}")

            # Convert to Polars DataFrame
            df = self._bars_to_dataframe(bars)

            # Apply column filter if specified
            if columns:
                available_cols = [c for c in columns if c in df.columns]
                if available_cols:
                    df = df.select(available_cols)

            logger.debug(f"Loaded {len(df)} bars for {symbol} ({timeframe}) from database")
            return df

    def load_symbols(
        self,
        symbols: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        columns: list[str] | None = None,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> dict[str, pl.DataFrame]:
        """
        Load OHLCV data for multiple symbols from database.

        Args:
            symbols: List of ticker symbols.
            start_date: Start date filter.
            end_date: End date filter.
            columns: Specific columns to load.

        Returns:
            Dictionary mapping symbol to DataFrame.
        """
        results = {}
        errors = []

        for symbol in symbols:
            try:
                df = self.load_symbol(
                    symbol,
                    start_date,
                    end_date,
                    columns,
                    timeframe=timeframe,
                )
                results[symbol.upper()] = df
            except DataError as e:
                errors.append(f"{symbol}: {str(e)}")
                logger.warning(f"Failed to load {symbol}: {e}")

        if errors:
            logger.warning(f"Failed to load {len(errors)} symbols")

        return results

    def get_available_symbols(self, timeframe: str = DEFAULT_TIMEFRAME) -> list[str]:
        """Get list of available symbols in database."""
        timeframe = normalize_timeframe(timeframe)
        with self._db.session() as session:
            return self._ohlcv_repo.get_symbols(session, timeframe=timeframe)

    def get_date_range(
        self,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> tuple[datetime, datetime] | None:
        """
        Get the date range available for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            Tuple of (start_date, end_date) or None if no data.
        """
        symbol = symbol.upper()
        timeframe = normalize_timeframe(timeframe)

        with self._db.session() as session:
            # Get first bar
            bars = self._ohlcv_repo.get_bars(session, symbol, timeframe=timeframe, limit=1)
            if not bars:
                return None
            min_date = bars[0].timestamp

            # Get last bar
            latest_bar = self._ohlcv_repo.get_latest_bar(session, symbol, timeframe=timeframe)
            if not latest_bar:
                return None
            max_date = latest_bar.timestamp

            return (min_date, max_date)

    def load_csv_to_database(
        self,
        csv_path: Path,
        symbol: str | None = None,
        timeframe: str | None = None,
        batch_size: int | None = None,
        validate: bool = True,
        source_timezone: str = "America/New_York",
        resume: bool = True,
    ) -> int:
        """
        Bulk load CSV data into database.

        This is optimized for initial data migration. Uses batch inserts
        for efficient bulk loading into TimescaleDB.

        Args:
            csv_path: Path to CSV file.
            symbol: Symbol name. If None, inferred from filename.
            batch_size: Batch size for inserts. Defaults to instance batch_size.
            validate: Whether to validate data before insert.

        Returns:
            Number of rows inserted.

        Raises:
            DataError: If file cannot be read or data is invalid.
        """
        csv_path = Path(csv_path)
        batch_size = batch_size or self._batch_size

        if not csv_path.exists():
            raise DataNotFoundError(f"CSV file not found: {csv_path}")

        inferred_symbol, inferred_timeframe = infer_symbol_and_timeframe(csv_path)
        symbol = (symbol or inferred_symbol).upper()
        timeframe = normalize_timeframe(timeframe or inferred_timeframe)
        logger.info(f"Loading {csv_path} for symbol {symbol} ({timeframe})")

        try:
            # Read CSV with Polars for speed
            df = pl.read_csv(csv_path, try_parse_dates=True)

            # Standardize column names
            df = self._standardize_columns(df)

            # Add symbol column if not present
            if "symbol" not in df.columns:
                df = df.with_columns(pl.lit(symbol).alias("symbol"))
            else:
                df = df.with_columns(
                    pl.col("symbol").cast(pl.Utf8).str.to_uppercase().alias("symbol")
                )
            df = df.with_columns(pl.lit(timeframe).alias("timeframe"))

            # Validate if requested
            if validate:
                contract = OHLCVIngestionContract(source_timezone=source_timezone)
                df = apply_ohlcv_ingestion_contract(df, symbol, contract=contract)
                logger.info(f"Ingestion contract passed for {symbol}: {quality_summary(df)}")
            else:
                df = self._validate_data(df, symbol)

            # Sort by timestamp
            df = df.sort("timestamp")

            # Batch insert
            total_inserted = 0
            start_offset = 0
            checkpoint_file: Path | None = None
            if resume:
                checkpoint_file = self._checkpoint_file(csv_path, symbol, batch_size)
                start_offset = self._read_checkpoint_offset(checkpoint_file)
                if start_offset > 0:
                    logger.info(
                        f"Resuming materialization for {symbol} from row offset {start_offset}"
                    )

            for i in range(start_offset, len(df), batch_size):
                batch_df = df.slice(i, batch_size)
                batch_dicts = self._dataframe_to_dicts(batch_df)

                with self._db.session() as session:
                    inserted = self._upsert_batch(session, batch_dicts)
                    total_inserted += inserted

                if checkpoint_file is not None:
                    self._write_checkpoint(
                        checkpoint_file,
                        symbol=symbol,
                        source=str(csv_path),
                        next_offset=i + len(batch_df),
                        total_rows=len(df),
                    )
                logger.debug(f"Inserted batch {i // batch_size + 1}: {len(batch_dicts)} rows")

            if checkpoint_file is not None and checkpoint_file.exists():
                checkpoint_file.unlink()

            logger.info(f"Loaded {total_inserted} bars for {symbol} into database")
            return total_inserted

        except Exception as e:
            raise DataError(f"Failed to load CSV {csv_path}: {e}")

    def load_parquet_to_database(
        self,
        parquet_path: Path,
        symbol: str | None = None,
        batch_size: int | None = None,
        source_timezone: str = "America/New_York",
        resume: bool = True,
        timeframe: str | None = None,
    ) -> int:
        """
        Bulk load Parquet data into database.

        Args:
            parquet_path: Path to Parquet file.
            symbol: Symbol name. If None, inferred from filename.
            batch_size: Batch size for inserts.

        Returns:
            Number of rows inserted.
        """
        parquet_path = Path(parquet_path)
        batch_size = batch_size or self._batch_size

        if not parquet_path.exists():
            raise DataNotFoundError(f"Parquet file not found: {parquet_path}")

        inferred_symbol, inferred_timeframe = infer_symbol_and_timeframe(parquet_path)
        symbol = (symbol or inferred_symbol).upper()
        timeframe = normalize_timeframe(timeframe or inferred_timeframe)
        logger.info(f"Loading {parquet_path} for symbol {symbol} ({timeframe})")

        try:
            df = pl.read_parquet(parquet_path)
            df = self._standardize_columns(df)

            if "symbol" not in df.columns:
                df = df.with_columns(pl.lit(symbol).alias("symbol"))
            else:
                df = df.with_columns(
                    pl.col("symbol").cast(pl.Utf8).str.to_uppercase().alias("symbol")
                )

            if "timeframe" not in df.columns:
                df = df.with_columns(pl.lit(timeframe).alias("timeframe"))
            else:
                df = df.with_columns(
                    pl.col("timeframe")
                    .cast(pl.Utf8)
                    .map_elements(
                        lambda value: normalize_timeframe(value, default=timeframe),
                        return_dtype=pl.Utf8,
                    )
                    .alias("timeframe")
                )

            contract = OHLCVIngestionContract(source_timezone=source_timezone)
            df = apply_ohlcv_ingestion_contract(df, symbol, contract=contract)
            logger.info(f"Ingestion contract passed for {symbol}: {quality_summary(df)}")
            df = df.sort("timestamp")

            total_inserted = 0
            start_offset = 0
            checkpoint_file: Path | None = None
            if resume:
                checkpoint_file = self._checkpoint_file(parquet_path, symbol, batch_size)
                start_offset = self._read_checkpoint_offset(checkpoint_file)
                if start_offset > 0:
                    logger.info(
                        f"Resuming materialization for {symbol} from row offset {start_offset}"
                    )

            for i in range(start_offset, len(df), batch_size):
                batch_df = df.slice(i, batch_size)
                batch_dicts = self._dataframe_to_dicts(batch_df)

                with self._db.session() as session:
                    inserted = self._upsert_batch(session, batch_dicts)
                    total_inserted += inserted

                if checkpoint_file is not None:
                    self._write_checkpoint(
                        checkpoint_file,
                        symbol=symbol,
                        source=str(parquet_path),
                        next_offset=i + len(batch_df),
                        total_rows=len(df),
                    )

            if checkpoint_file is not None and checkpoint_file.exists():
                checkpoint_file.unlink()

            logger.info(f"Loaded {total_inserted} bars for {symbol} ({timeframe}) into database")
            return total_inserted

        except Exception as e:
            raise DataError(f"Failed to load Parquet {parquet_path}: {e}")

    def export_to_parquet(
        self,
        symbol: str,
        output_path: Path,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        timeframe: str = DEFAULT_TIMEFRAME,
        compression: str = "zstd",
    ) -> Path:
        """
        Export database data to Parquet file for ML training.

        This is part of the hybrid architecture where PostgreSQL is the
        source of truth and Parquet files are used for ML training.

        Args:
            symbol: Ticker symbol to export.
            output_path: Output Parquet file path.
            start_date: Start date filter.
            end_date: End date filter.
            compression: Compression codec (zstd, snappy, lz4).

        Returns:
            Path to exported file.
        """
        symbol = symbol.upper()
        timeframe = normalize_timeframe(timeframe)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load from database
        df = self.load_symbol(symbol, start_date, end_date, timeframe=timeframe)

        # Write to Parquet with compression
        df.write_parquet(
            output_path,
            compression=compression,
            statistics=True,
        )

        logger.info(f"Exported {len(df)} bars for {symbol} ({timeframe}) to {output_path}")
        return output_path

    def export_all_to_parquet(
        self,
        output_dir: Path,
        symbols: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        timeframe: str = DEFAULT_TIMEFRAME,
        compression: str = "zstd",
    ) -> dict[str, Path]:
        """
        Export all symbols to Parquet files.

        Args:
            output_dir: Output directory.
            symbols: List of symbols to export. If None, exports all.
            start_date: Start date filter.
            end_date: End date filter.
            compression: Compression codec.

        Returns:
            Dictionary mapping symbol to output path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timeframe = normalize_timeframe(timeframe)

        if symbols is None:
            symbols = self.get_available_symbols(timeframe=timeframe)

        results = {}
        for symbol in symbols:
            try:
                output_path = output_dir / f"{symbol}_{timeframe_slug(timeframe)}.parquet"
                self.export_to_parquet(
                    symbol,
                    output_path,
                    start_date,
                    end_date,
                    timeframe,
                    compression,
                )
                results[symbol] = output_path
            except DataError as e:
                logger.warning(f"Failed to export {symbol}: {e}")

        return results

    def insert_bars(
        self,
        bars: list[OHLCVBar],
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> int:
        """
        Insert OHLCV bars into database.

        Args:
            bars: List of OHLCVBar Pydantic models.

        Returns:
            Number of bars inserted.
        """
        if not bars:
            return 0
        timeframe = normalize_timeframe(timeframe)

        bar_dicts = []
        for bar in bars:
            bar_dicts.append(
                {
                    "symbol": bar.symbol.upper(),
                    "timestamp": bar.timestamp,
                    "timeframe": timeframe,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "vwap": bar.vwap,
                    "trade_count": bar.trade_count,
                }
            )

        with self._db.session() as session:
            return self._ohlcv_repo.bulk_insert(session, bar_dicts)

    def upsert_bar(
        self,
        bar: OHLCVBar,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> None:
        """
        Upsert a single OHLCV bar (insert or update on conflict).

        For live data persistence where bars may need to be updated.

        Args:
            bar: OHLCVBar to upsert.
        """
        from sqlalchemy.dialects.postgresql import insert

        timeframe = normalize_timeframe(timeframe)

        with self._db.session() as session:
            stmt = insert(OHLCVBarModel).values(
                symbol=bar.symbol.upper(),
                timestamp=bar.timestamp,
                timeframe=timeframe,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                vwap=bar.vwap,
                trade_count=bar.trade_count,
            )

            # On conflict (symbol, timestamp, timeframe), update the values
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "timestamp", "timeframe"],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                    "vwap": stmt.excluded.vwap,
                    "trade_count": stmt.excluded.trade_count,
                },
            )

            session.execute(stmt)
            session.commit()

    def upsert_dataframe(
        self,
        df: pl.DataFrame | pd.DataFrame,
        symbol: str | None = None,
        timeframe: str | None = None,
        batch_size: int | None = None,
        source_timezone: str = "UTC",
    ) -> int:
        """
        Upsert OHLCV rows from a dataframe into PostgreSQL/TimescaleDB.

        Args:
            df: Input dataframe (Polars or pandas).
            symbol: Optional symbol override/fallback when dataframe has no symbol column.
            batch_size: Upsert batch size (defaults to loader batch size).
            source_timezone: Source timezone for ingestion contract normalization.

        Returns:
            Number of rows upserted.
        """
        if isinstance(df, pd.DataFrame):
            frame = pl.from_pandas(df)
        else:
            frame = df

        if frame.is_empty():
            return 0

        batch_size = batch_size or self._batch_size
        frame = self._standardize_columns(frame)
        default_timeframe = normalize_timeframe(timeframe or DEFAULT_TIMEFRAME)

        symbol_override = symbol.upper() if symbol else None
        if "symbol" not in frame.columns:
            if symbol_override is None:
                raise DataValidationError(
                    "Missing `symbol` column and no symbol override provided."
                )
            frame = frame.with_columns(pl.lit(symbol_override).alias("symbol"))
        else:
            frame = frame.with_columns(
                pl.col("symbol").cast(pl.Utf8).str.to_uppercase().alias("symbol")
            )
            if symbol_override is not None:
                frame = frame.with_columns(
                    pl.when((pl.col("symbol").is_null()) | (pl.col("symbol") == ""))
                    .then(pl.lit(symbol_override))
                    .otherwise(pl.col("symbol"))
                    .alias("symbol")
                )

        if "timeframe" not in frame.columns:
            frame = frame.with_columns(pl.lit(default_timeframe).alias("timeframe"))
        else:
            frame = frame.with_columns(
                pl.col("timeframe")
                .cast(pl.Utf8)
                .map_elements(
                    lambda value: normalize_timeframe(value, default=default_timeframe),
                    return_dtype=pl.Utf8,
                )
                .alias("timeframe")
            )

        contract = OHLCVIngestionContract(source_timezone=source_timezone)
        unique_symbols = [s for s in frame["symbol"].unique().to_list() if isinstance(s, str) and s]
        if not unique_symbols:
            raise DataValidationError("No valid symbols present after normalization.")

        normalized_frames: list[pl.DataFrame] = []
        for sym in unique_symbols:
            sym_df = frame.filter(pl.col("symbol") == sym)
            normalized_frames.append(apply_ohlcv_ingestion_contract(sym_df, sym, contract=contract))

        normalized = pl.concat(normalized_frames, how="vertical").sort(["symbol", "timestamp"])

        total_upserted = 0
        for offset in range(0, len(normalized), batch_size):
            chunk = normalized.slice(offset, batch_size)
            rows = self._dataframe_to_dicts(chunk)
            if not rows:
                continue
            with self._db.session() as session:
                total_upserted += self._upsert_batch(session, rows)

        return total_upserted

    def _bars_to_dataframe(self, bars: list[OHLCVBarModel]) -> pl.DataFrame:
        """Convert database bar models to Polars DataFrame."""
        if not bars:
            return pl.DataFrame()

        data = {
            "timestamp": [],
            "symbol": [],
            "timeframe": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
            "vwap": [],
            "trade_count": [],
        }

        for bar in bars:
            data["timestamp"].append(bar.timestamp)
            data["symbol"].append(bar.symbol)
            data["timeframe"].append(bar.timeframe)
            data["open"].append(float(bar.open))
            data["high"].append(float(bar.high))
            data["low"].append(float(bar.low))
            data["close"].append(float(bar.close))
            data["volume"].append(bar.volume)
            data["vwap"].append(float(bar.vwap) if bar.vwap else None)
            data["trade_count"].append(bar.trade_count)

        return pl.DataFrame(data)

    def _dataframe_to_dicts(self, df: pl.DataFrame) -> list[dict[str, Any]]:
        """Convert Polars DataFrame to list of dicts for bulk insert."""
        records = []

        for row in df.iter_rows(named=True):
            record = {
                "symbol": row["symbol"].upper() if "symbol" in row else "UNKNOWN",
                "timestamp": row["timestamp"],
                "timeframe": normalize_timeframe(row.get("timeframe"), default=DEFAULT_TIMEFRAME),
                "open": self._to_decimal(row["open"]),
                "high": self._to_decimal(row["high"]),
                "low": self._to_decimal(row["low"]),
                "close": self._to_decimal(row["close"]),
                "volume": int(row["volume"]),
            }

            if "vwap" in row and row["vwap"] is not None:
                record["vwap"] = self._to_decimal(row["vwap"])
            if "trade_count" in row and row["trade_count"] is not None:
                record["trade_count"] = int(row["trade_count"])

            records.append(record)

        return records

    @staticmethod
    def _to_decimal(value: float | int | str | Decimal | None) -> Decimal | None:
        """Convert value to Decimal safely."""
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        if isinstance(value, str):
            return Decimal(value)
        # Convert via string to preserve precision
        return Decimal(str(round(value, 8)))

    def _standardize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize column names."""
        column_mappings = {
            "date": "timestamp",
            "datetime": "timestamp",
            "time": "timestamp",
            "Date": "timestamp",
            "Datetime": "timestamp",
            "Timestamp": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
            "VWAP": "vwap",
            "vwap": "vwap",
            "Trade_Count": "trade_count",
            "trade_count": "trade_count",
            "Symbol": "symbol",
        }

        current_cols = df.columns
        rename_dict = {}

        for old_name, new_name in column_mappings.items():
            if old_name in current_cols and old_name != new_name:
                rename_dict[old_name] = new_name

        if rename_dict:
            df = df.rename(rename_dict)

        return df

    def _validate_data(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """Validate loaded data."""
        errors = []

        # Check required columns
        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise DataValidationError(f"Missing required columns for {symbol}: {missing_cols}")

        # Check for empty data
        if len(df) == 0:
            raise DataValidationError(f"Empty data for {symbol}")

        # Remove duplicates
        if df["timestamp"].n_unique() != len(df):
            dup_count = len(df) - df["timestamp"].n_unique()
            errors.append(f"Removed {dup_count} duplicate timestamps")
            df = df.unique(subset=["timestamp"], keep="last")

        # Sort by timestamp
        if not df["timestamp"].is_sorted():
            df = df.sort("timestamp")

        for error in errors:
            logger.warning(f"Data validation for {symbol}: {error}")

        return df

    def _upsert_batch(self, session, bars: list[dict[str, Any]]) -> int:
        """Idempotent upsert for OHLCV batches."""
        if not bars:
            return 0

        dialect_name = session.bind.dialect.name if session.bind is not None else ""

        if dialect_name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            stmt = pg_insert(OHLCVBarModel).values(bars)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "timestamp", "timeframe"],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                    "vwap": stmt.excluded.vwap,
                    "trade_count": stmt.excluded.trade_count,
                },
            )
            session.execute(stmt)
            return len(bars)

        if dialect_name == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            stmt = sqlite_insert(OHLCVBarModel).values(bars)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "timestamp", "timeframe"],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                    "vwap": stmt.excluded.vwap,
                    "trade_count": stmt.excluded.trade_count,
                },
            )
            session.execute(stmt)
            return len(bars)

        # Generic fallback for other engines: last-write-wins merge.
        for row in bars:
            session.merge(OHLCVBarModel(**row))
        session.flush()
        return len(bars)

    @staticmethod
    def _checkpoint_root() -> Path:
        root = Path("logs") / "feature_materialization_state"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _checkpoint_file(self, source_path: Path, symbol: str, batch_size: int) -> Path:
        source_path = Path(source_path).resolve()
        fingerprint = hashlib.sha256(
            f"{source_path}:{source_path.stat().st_mtime_ns}:{source_path.stat().st_size}:{batch_size}".encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        return self._checkpoint_root() / f"ohlcv_{symbol.upper()}_{fingerprint}.json"

    @staticmethod
    def _read_checkpoint_offset(checkpoint_file: Path) -> int:
        if not checkpoint_file.exists():
            return 0
        try:
            payload = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            offset = int(payload.get("next_offset", 0))
            return max(offset, 0)
        except Exception:
            return 0

    @staticmethod
    def _write_checkpoint(
        checkpoint_file: Path,
        symbol: str,
        source: str,
        next_offset: int,
        total_rows: int,
    ) -> None:
        payload = {
            "symbol": symbol,
            "source": source,
            "next_offset": int(next_offset),
            "total_rows": int(total_rows),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        checkpoint_file.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )


# Singleton instance
_db_loader: DatabaseDataLoader | None = None


def get_db_loader() -> DatabaseDataLoader:
    """Get database data loader singleton instance."""
    global _db_loader
    if _db_loader is None:
        _db_loader = DatabaseDataLoader()
    return _db_loader
