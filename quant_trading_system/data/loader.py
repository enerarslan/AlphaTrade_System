"""
Data loader module using Polars for high-performance data loading.

Provides efficient loading of historical OHLCV data from multiple formats
(CSV, Parquet, HDF5) with validation, caching, and memory-efficient streaming.

Supports dual-source architecture:
- File-based loading (CSV, Parquet, HDF5) for backward compatibility
- Database loading (PostgreSQL + TimescaleDB) when use_database=True

@agent: @data
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import polars as pl

from quant_trading_system.core.data_types import OHLCVBar
from quant_trading_system.core.exceptions import DataError, DataNotFoundError, DataValidationError

if TYPE_CHECKING:
    from quant_trading_system.database.connection import DatabaseManager

logger = logging.getLogger(__name__)


class DataValidationRules:
    """Data validation rules and thresholds."""

    MAX_GAP_BARS = 5  # Maximum allowed gap in bars
    MIN_COVERAGE_PCT = 95.0  # Minimum data coverage percentage
    MAX_PRICE_CHANGE_PCT = 50.0  # Maximum single-bar price change (for outlier detection)
    MIN_VOLUME = 0  # Minimum volume
    TRADING_HOURS_START = 9  # Market open hour (ET)
    TRADING_HOURS_END = 16  # Market close hour (ET)


class DataLoader:
    """
    High-performance data loader using Polars.

    Loads OHLCV data from multiple formats with validation,
    caching, and parallel processing support.

    Supports dual-source architecture:
    - File-based loading (default): Reads from CSV, Parquet, HDF5 files
    - Database loading (use_database=True): Reads from PostgreSQL + TimescaleDB
    """

    SUPPORTED_FORMATS = {".csv", ".parquet", ".hdf5", ".h5"}

    def __init__(
        self,
        data_dir: str | Path,
        cache_size: int = 100,
        validate: bool = True,
        use_database: bool = False,
        db_manager: "DatabaseManager | None" = None,
    ):
        """
        Initialize the data loader.

        Args:
            data_dir: Base directory for data files (used as fallback if use_database=True)
            cache_size: Number of dataframes to cache in memory
            validate: Whether to validate data on load
            use_database: If True, load from PostgreSQL + TimescaleDB instead of files
            db_manager: Database manager instance (required if use_database=True)
        """
        self.data_dir = Path(data_dir)
        self.validate = validate
        self._cache: dict[str, pl.LazyFrame] = {}
        self._cache_size = cache_size
        self.use_database = use_database
        self._db_loader = None

        # Initialize database loader if using database
        if use_database:
            from quant_trading_system.data.db_loader import DatabaseDataLoader
            if db_manager is None:
                from quant_trading_system.database.connection import get_db_manager
                db_manager = get_db_manager()
            self._db_loader = DatabaseDataLoader(db_manager)
            logger.info("DataLoader initialized with database backend")
        elif not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")

    def load_symbol(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        columns: list[str] | None = None,
        timeframe: str | None = None,
    ) -> pl.DataFrame:
        """
        Load OHLCV data for a single symbol.

        Args:
            symbol: Ticker symbol to load
            start_date: Start date filter (inclusive)
            end_date: End date filter (inclusive)
            columns: Specific columns to load (None for all)

        Returns:
            Polars DataFrame with OHLCV data

        Raises:
            DataNotFoundError: If data file not found
            DataValidationError: If data fails validation
        """
        symbol = symbol.upper()

        # Use database loader if configured
        if self.use_database and self._db_loader is not None:
            df = self._db_loader.load_symbol(
                symbol,
                start_date,
                end_date,
                columns,
                timeframe=timeframe or "15Min",
            )
            if self.validate:
                df = self._validate_data(df, symbol)
            logger.debug(f"Loaded {len(df)} bars for {symbol} from database")
            return df

        # File-based loading
        file_path = self._find_data_file(
            symbol,
            start_date=start_date,
            end_date=end_date,
        )

        if file_path is None:
            raise DataNotFoundError(f"No data file found for symbol: {symbol}")

        # Load based on file format
        lf = self._load_file(file_path, columns)
        lf = self._ensure_timestamp_utc(lf)

        # Apply date filters
        if start_date is not None:
            start_utc = self._normalize_filter_datetime(start_date)
            lf = lf.filter(pl.col("timestamp") >= start_utc)
        if end_date is not None:
            end_utc = self._normalize_filter_datetime(end_date)
            lf = lf.filter(pl.col("timestamp") <= end_utc)

        # Collect and validate
        df = lf.collect()

        if self.validate:
            df = self._validate_data(df, symbol)

        logger.debug(f"Loaded {len(df)} bars for {symbol}")
        return df

    def load_symbols(
        self,
        symbols: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        columns: list[str] | None = None,
        parallel: bool = True,
    ) -> dict[str, pl.DataFrame]:
        """
        Load OHLCV data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start date filter
            end_date: End date filter
            columns: Specific columns to load
            parallel: Whether to load in parallel

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        errors: list[str] = []

        for symbol in symbols:
            try:
                df = self.load_symbol(symbol, start_date, end_date, columns)
                results[symbol.upper()] = df
            except DataError as e:
                errors.append(f"{symbol}: {str(e)}")
                logger.warning(f"Failed to load {symbol}: {e}")

        if errors:
            logger.warning(f"Failed to load {len(errors)} symbols: {errors[:5]}")

        return results

    def load_combined(
        self,
        symbols: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pl.DataFrame:
        """
        Load and combine data for multiple symbols into a single DataFrame.

        Args:
            symbols: List of ticker symbols
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Combined DataFrame with symbol column
        """
        dfs = []

        for symbol in symbols:
            try:
                df = self.load_symbol(symbol, start_date, end_date)
                df = df.with_columns(pl.lit(symbol.upper()).alias("symbol"))
                dfs.append(df)
            except DataError as e:
                logger.warning(f"Skipping {symbol}: {e}")

        if not dfs:
            raise DataNotFoundError("No data loaded for any symbol")

        return pl.concat(dfs)

    def stream_symbol(
        self,
        symbol: str,
        chunk_size: int = 10000,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> Iterator[pl.DataFrame]:
        """
        Stream data for a symbol in chunks for memory efficiency.

        Args:
            symbol: Ticker symbol
            chunk_size: Number of rows per chunk
            start_date: Start date filter
            end_date: End date filter

        Yields:
            DataFrame chunks
        """
        symbol = symbol.upper()
        file_path = self._find_data_file(symbol)

        if file_path is None:
            raise DataNotFoundError(f"No data file found for symbol: {symbol}")

        # Use lazy evaluation for streaming
        lf = self._load_file(file_path)

        if start_date is not None:
            lf = lf.filter(pl.col("timestamp") >= start_date)
        if end_date is not None:
            lf = lf.filter(pl.col("timestamp") <= end_date)

        # Collect and yield chunks
        df = lf.collect()
        total_rows = len(df)

        for i in range(0, total_rows, chunk_size):
            chunk = df.slice(i, chunk_size)
            yield chunk

    def to_ohlcv_bars(self, df: pl.DataFrame) -> list[OHLCVBar]:
        """
        Convert DataFrame to list of OHLCVBar objects.

        CRITICAL: Converts Polars float64 values to Decimal using string conversion
        to avoid floating-point precision issues. Direct float-to-Decimal conversion
        (e.g., Decimal(0.1)) preserves binary floating-point imprecision.
        Using str() conversion (e.g., Decimal(str(0.1))) gives exact decimal values.

        Args:
            df: Polars DataFrame with OHLCV data

        Returns:
            List of OHLCVBar Pydantic models

        Raises:
            DataValidationError: If conversion to Decimal fails
        """
        bars = []
        for row in df.iter_rows(named=True):
            try:
                # CRITICAL: Convert floats to Decimal via string to preserve precision
                # Decimal(float) keeps floating-point imprecision
                # Decimal(str(float)) gives exact decimal representation
                open_price = self._float_to_decimal(row["open"])
                high_price = self._float_to_decimal(row["high"])
                low_price = self._float_to_decimal(row["low"])
                close_price = self._float_to_decimal(row["close"])

                # Handle optional vwap field
                vwap_value = row.get("vwap")
                vwap = self._float_to_decimal(vwap_value) if vwap_value is not None else None

                bar = OHLCVBar(
                    symbol=row.get("symbol", "UNKNOWN"),
                    timestamp=row["timestamp"],
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=int(row["volume"]),
                    vwap=vwap,
                    trade_count=row.get("trade_count"),
                )
                bars.append(bar)
            except (InvalidOperation, KeyError, ValueError, TypeError) as e:
                logger.warning(f"Failed to convert row to OHLCVBar: {e}")
                raise DataValidationError(f"Failed to convert data to OHLCVBar: {e}")
        return bars

    @staticmethod
    def _float_to_decimal(value: float | int | str | None) -> Decimal:
        """
        Convert a numeric value to Decimal with proper precision handling.

        CRITICAL: Uses string conversion to avoid floating-point representation issues.
        For example:
            - Decimal(0.1) = Decimal('0.1000000000000000055511151231257827021181583404541015625')
            - Decimal(str(0.1)) = Decimal('0.1')

        Args:
            value: Numeric value to convert (float, int, or string)

        Returns:
            Decimal representation with exact precision

        Raises:
            InvalidOperation: If value cannot be converted to Decimal
        """
        if value is None:
            raise InvalidOperation("Cannot convert None to Decimal")

        if isinstance(value, Decimal):
            return value

        if isinstance(value, str):
            return Decimal(value)

        if isinstance(value, int):
            return Decimal(value)

        # For float: convert via string to preserve exact decimal representation
        # Round to 8 decimal places to match OHLCVBar field constraints
        return Decimal(str(round(value, 8)))

    def get_available_symbols(self, timeframe: str | None = None) -> list[str]:
        """Get list of available symbols based on data source."""
        # Use database if configured
        if self.use_database and self._db_loader is not None:
            db_symbols = self._db_loader.get_available_symbols(timeframe=timeframe or "15Min")
            return sorted(db_symbols)

        # File-based symbol discovery
        symbols = []
        for ext in self.SUPPORTED_FORMATS:
            for file_path in self.data_dir.glob(f"*{ext}"):
                symbol = file_path.stem.upper()
                # Remove common suffixes
                for suffix in ["_15MIN", "_15Min", "_1D", "_DAILY"]:
                    if symbol.endswith(suffix):
                        symbol = symbol[:-len(suffix)]
                        break
                if symbol not in symbols:
                    symbols.append(symbol)
        return sorted(symbols)

    def get_date_range(
        self,
        symbol: str,
        timeframe: str | None = None,
    ) -> tuple[datetime, datetime]:
        """
        Get the date range available for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Tuple of (start_date, end_date)
        """
        df = self.load_symbol(symbol, columns=["timestamp"], timeframe=timeframe)
        min_date = df["timestamp"].min()
        max_date = df["timestamp"].max()
        return (min_date, max_date)

    @staticmethod
    def _normalize_filter_datetime(value: datetime) -> datetime:
        """Normalize datetime filter values to UTC-aware timestamps."""
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _ensure_timestamp_utc(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Normalize timestamp column to UTC-aware datetime when present."""
        schema = lf.collect_schema()
        if "timestamp" not in schema:
            return lf

        timestamp_dtype = schema["timestamp"]

        if timestamp_dtype in (pl.Utf8, pl.String):
            lf = lf.with_columns(
                pl.col("timestamp").str.to_datetime(time_zone="UTC", strict=False)
            )
            schema = lf.collect_schema()
            timestamp_dtype = schema.get("timestamp")

        if timestamp_dtype == pl.Date:
            return lf.with_columns(
                pl.col("timestamp")
                .cast(pl.Datetime("us"))
                .dt.replace_time_zone("UTC")
            )

        tz = getattr(timestamp_dtype, "time_zone", None)
        if tz is None and timestamp_dtype != pl.Utf8:
            return lf.with_columns(
                pl.col("timestamp").dt.replace_time_zone("UTC")
            )
        if tz and tz != "UTC":
            return lf.with_columns(
                pl.col("timestamp").dt.convert_time_zone("UTC")
            )

        return lf

    def _get_file_date_span(self, file_path: Path) -> tuple[datetime, datetime] | None:
        """Return min/max timestamp span for a candidate file."""
        try:
            lf = self._load_file(file_path, columns=["timestamp"])
            lf = self._ensure_timestamp_utc(lf)
            summary = lf.select(
                [
                    pl.col("timestamp").min().alias("_min_ts"),
                    pl.col("timestamp").max().alias("_max_ts"),
                ]
            ).collect()
            min_ts = summary.get_column("_min_ts").item(0)
            max_ts = summary.get_column("_max_ts").item(0)
            if min_ts is None or max_ts is None:
                return None
            if isinstance(min_ts, datetime) and isinstance(max_ts, datetime):
                return min_ts, max_ts
            return None
        except Exception:
            return None

    def _score_candidate_file(
        self,
        file_path: Path,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> tuple[float, float, float, float]:
        """Score candidate files by requested-range coverage and usable history."""
        date_span = self._get_file_date_span(file_path)
        name = file_path.stem.upper()
        intraday_bias = 1.0 if any(s in name for s in ("_1MIN", "_5MIN", "_15MIN", "_30MIN", "_1H")) else 0.0

        if date_span is None:
            return (0.0, 0.0, intraday_bias, 0.0)

        file_start, file_end = date_span
        span_seconds = max(0.0, (file_end - file_start).total_seconds())

        coverage_ratio = 1.0
        if start_date is not None and end_date is not None and end_date > start_date:
            overlap_start = max(file_start, start_date)
            overlap_end = min(file_end, end_date)
            overlap_seconds = max(0.0, (overlap_end - overlap_start).total_seconds())
            requested_seconds = (end_date - start_date).total_seconds()
            coverage_ratio = overlap_seconds / requested_seconds if requested_seconds > 0 else 0.0

        freshness = file_end.timestamp() if isinstance(file_end, datetime) else 0.0
        return (coverage_ratio, span_seconds, intraday_bias, freshness)

    def _find_data_file(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> Path | None:
        """Find data file for a symbol."""
        symbol = symbol.upper()

        # Common file naming patterns to check
        patterns = [
            f"{symbol}",           # AAPL.csv
            f"{symbol.lower()}",   # aapl.csv
            f"{symbol}_15min",     # AAPL_15min.csv
            f"{symbol}_15Min",     # AAPL_15Min.csv
            f"{symbol}_1d",        # AAPL_1d.csv
            f"{symbol}_daily",     # AAPL_daily.csv
        ]

        # Check for symbol-specific files
        candidates: set[Path] = set()
        for pattern in patterns:
            for ext in self.SUPPORTED_FORMATS:
                file_path = self.data_dir / f"{pattern}{ext}"
                if file_path.exists():
                    candidates.add(file_path)

        # Check subdirectories
        if self.data_dir.exists():
            for subdir in self.data_dir.iterdir():
                if subdir.is_dir():
                    for pattern in patterns:
                        for ext in self.SUPPORTED_FORMATS:
                            file_path = subdir / f"{pattern}{ext}"
                            if file_path.exists():
                                candidates.add(file_path)

        if not candidates:
            return None
        if len(candidates) == 1:
            return next(iter(candidates))

        start_utc = self._normalize_filter_datetime(start_date) if start_date is not None else None
        end_utc = self._normalize_filter_datetime(end_date) if end_date is not None else None

        scored = [
            (self._score_candidate_file(path, start_utc, end_utc), path)
            for path in candidates
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _load_file(
        self,
        file_path: Path,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Load a data file based on its format."""
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            return self._load_csv(file_path, columns)
        elif suffix == ".parquet":
            return self._load_parquet(file_path, columns)
        elif suffix in {".hdf5", ".h5"}:
            return self._load_hdf5(file_path, columns)
        else:
            raise DataError(f"Unsupported file format: {suffix}")

    def _load_csv(
        self,
        file_path: Path,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Load CSV file."""
        try:
            lf = pl.scan_csv(
                file_path,
                try_parse_dates=True,
                infer_schema_length=10000,
            )

            # Standardize column names
            lf = self._standardize_columns(lf)

            if columns:
                lf = lf.select(columns)

            return lf
        except Exception as e:
            raise DataError(f"Failed to load CSV {file_path}: {e}")

    def _load_parquet(
        self,
        file_path: Path,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Load Parquet file."""
        try:
            if columns:
                lf = pl.scan_parquet(file_path, columns=columns)
            else:
                lf = pl.scan_parquet(file_path)

            # Standardize column names
            lf = self._standardize_columns(lf)

            return lf
        except Exception as e:
            raise DataError(f"Failed to load Parquet {file_path}: {e}")

    def _load_hdf5(
        self,
        file_path: Path,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Load HDF5 file using pandas as intermediate."""
        try:
            import pandas as pd

            # HDF5 requires pandas
            pdf = pd.read_hdf(file_path)
            df = pl.from_pandas(pdf)

            # Convert to lazy frame
            lf = df.lazy()

            # Standardize column names
            lf = self._standardize_columns(lf)

            if columns:
                lf = lf.select(columns)

            return lf
        except ImportError:
            raise DataError("HDF5 support requires pandas and tables packages")
        except Exception as e:
            raise DataError(f"Failed to load HDF5 {file_path}: {e}")

    def _standardize_columns(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Standardize column names to expected format."""
        # Common column name mappings
        column_mappings = {
            "date": "timestamp",
            "datetime": "timestamp",
            "time": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
            "VWAP": "vwap",
            "Trade_Count": "trade_count",
            "trade_count": "trade_count",
            "Symbol": "symbol",
        }

        # Get current columns
        schema = lf.collect_schema()
        current_cols = list(schema.keys())

        # Apply mappings
        rename_dict = {}
        for old_name, new_name in column_mappings.items():
            if old_name in current_cols:
                rename_dict[old_name] = new_name

        if rename_dict:
            lf = lf.rename(rename_dict)

        return lf

    def _validate_data(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """
        Validate loaded data against quality rules.

        Args:
            df: DataFrame to validate
            symbol: Symbol for error messages

        Returns:
            Validated DataFrame

        Raises:
            DataValidationError: If validation fails
        """
        warnings_list: list[str] = []
        critical_errors: list[str] = []

        # Check required columns
        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise DataValidationError(
                f"Missing required columns for {symbol}: {missing_cols}"
            )

        # Check for empty data
        if len(df) == 0:
            raise DataValidationError(f"Empty data for {symbol}")

        # Check for duplicates
        if df["timestamp"].n_unique() != len(df):
            dup_count = len(df) - df["timestamp"].n_unique()
            warnings_list.append(f"Found {dup_count} duplicate timestamps")
            df = df.unique(subset=["timestamp"], keep="last")

        # Check timestamp ordering
        if not df["timestamp"].is_sorted():
            warnings_list.append("Timestamps not sorted - sorting now")
            df = df.sort("timestamp")

        # Validate OHLC relationships
        invalid_ohlc = df.filter(
            (pl.col("low") > pl.col("high")) |
            (pl.col("low") > pl.col("open")) |
            (pl.col("low") > pl.col("close")) |
            (pl.col("high") < pl.col("open")) |
            (pl.col("high") < pl.col("close"))
        )
        if len(invalid_ohlc) > 0:
            critical_errors.append(
                f"Found {len(invalid_ohlc)} bars with invalid OHLC relationships"
            )

        # Validate volume
        negative_volume = df.filter(pl.col("volume") < 0)
        if len(negative_volume) > 0:
            critical_errors.append(f"Found {len(negative_volume)} bars with negative volume")

        # Validate prices (no negative prices)
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            negative = df.filter(pl.col(col) < 0)
            if len(negative) > 0:
                critical_errors.append(f"Found {len(negative)} bars with negative {col} price")

        # Check for large gaps
        if len(df) > 1:
            timestamps = df["timestamp"].to_list()
            expected_interval = timedelta(minutes=15)  # 15-minute bars
            gap_count = 0
            for i in range(1, len(timestamps)):
                diff = timestamps[i] - timestamps[i - 1]
                if diff > expected_interval * DataValidationRules.MAX_GAP_BARS:
                    gap_count += 1
            if gap_count > 0:
                warnings_list.append(
                    f"Found {gap_count} large gaps (> {DataValidationRules.MAX_GAP_BARS} bars)"
                )

        # Check coverage
        if len(df) > 0:
            date_range = (df["timestamp"].max() - df["timestamp"].min()).days
            if date_range > 0:
                expected_bars = date_range * 26  # ~26 bars per day for 15-min
                coverage = (len(df) / expected_bars) * 100 if expected_bars > 0 else 100
                if coverage < DataValidationRules.MIN_COVERAGE_PCT:
                    critical_errors.append(
                        f"Data coverage ({coverage:.1f}%) below minimum ({DataValidationRules.MIN_COVERAGE_PCT}%)"
                    )

        # Warn on auto-remediated anomalies
        for warning in warnings_list:
            logger.warning(f"Data validation warning for {symbol}: {warning}")

        # Hard fail on critical anomalies so bad data cannot reach live trading.
        if critical_errors:
            for error in critical_errors:
                logger.error(f"Data validation critical for {symbol}: {error}")
            raise DataValidationError(
                f"Critical data validation failed for {symbol}: {'; '.join(critical_errors)}"
            )

        return df

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()

    def fill_gaps(
        self,
        df: pl.DataFrame,
        interval_minutes: int = 15,
        method: str = "forward",
        max_gap_bars: int = 10,
    ) -> pl.DataFrame:
        """
        Fill timestamp gaps in OHLCV data.

        CRITICAL for trading: Gaps in data can cause incorrect calculations.
        This method fills gaps with appropriate values based on the fill method.

        Args:
            df: DataFrame with OHLCV data
            interval_minutes: Expected interval between bars in minutes
            method: Fill method ('forward', 'linear', 'zero_volume')
            max_gap_bars: Maximum number of bars to fill (larger gaps left unfilled)

        Returns:
            DataFrame with gaps filled

        Methods:
            - 'forward': Forward fill OHLC prices, volume=0 for filled bars
            - 'linear': Interpolate OHLC prices linearly, volume=0
            - 'zero_volume': Use previous close for all prices, volume=0
        """
        if len(df) < 2:
            return df

        # Ensure sorted by timestamp
        df = df.sort("timestamp")

        # Get timestamp range
        min_ts = df["timestamp"].min()
        max_ts = df["timestamp"].max()

        # Generate expected timestamps (trading hours only)
        interval = timedelta(minutes=interval_minutes)
        expected_timestamps = []

        current = min_ts
        while current <= max_ts:
            # Check if this is a trading hour
            hour = current.hour
            minute = current.minute
            weekday = current.weekday()

            # Skip weekends
            if weekday < 5:  # Monday=0, Friday=4
                # Trading hours: 9:30 AM to 4:00 PM ET
                if (hour == 9 and minute >= 30) or (10 <= hour < 16) or (hour == 16 and minute == 0):
                    expected_timestamps.append(current)

            current += interval

            # If we crossed midnight, move to next day's trading hours
            if current.hour < 9 or (current.hour == 9 and current.minute < 30):
                current = current.replace(hour=9, minute=30, second=0, microsecond=0)

        if not expected_timestamps:
            return df

        # Create expected timestamps dataframe
        expected_df = pl.DataFrame({"timestamp": expected_timestamps})

        # Find missing timestamps
        existing_timestamps = set(df["timestamp"].to_list())
        missing = [ts for ts in expected_timestamps if ts not in existing_timestamps]

        if not missing:
            logger.debug("No gaps to fill")
            return df

        # Group consecutive gaps
        gap_groups = []
        current_group = []

        for ts in missing:
            if not current_group:
                current_group.append(ts)
            elif (ts - current_group[-1]) == interval:
                current_group.append(ts)
            else:
                if current_group:
                    gap_groups.append(current_group)
                current_group = [ts]

        if current_group:
            gap_groups.append(current_group)

        # Fill gaps that are within max_gap_bars limit
        filled_rows = []
        for gap_group in gap_groups:
            if len(gap_group) > max_gap_bars:
                logger.warning(
                    f"Gap too large to fill: {len(gap_group)} bars from {gap_group[0]} to {gap_group[-1]}"
                )
                continue

            # Find the bar before the gap
            gap_start = gap_group[0]
            prev_bars = df.filter(pl.col("timestamp") < gap_start).sort("timestamp", descending=True).head(1)

            if len(prev_bars) == 0:
                continue

            prev_bar = prev_bars.row(0, named=True)

            for ts in gap_group:
                if method == "forward":
                    # Forward fill: use previous bar's close for all prices
                    filled_row = {
                        "timestamp": ts,
                        "open": prev_bar["close"],
                        "high": prev_bar["close"],
                        "low": prev_bar["close"],
                        "close": prev_bar["close"],
                        "volume": 0,
                    }
                elif method == "zero_volume":
                    # Same as forward but more explicit
                    filled_row = {
                        "timestamp": ts,
                        "open": prev_bar["close"],
                        "high": prev_bar["close"],
                        "low": prev_bar["close"],
                        "close": prev_bar["close"],
                        "volume": 0,
                    }
                else:
                    # Default to forward fill
                    filled_row = {
                        "timestamp": ts,
                        "open": prev_bar["close"],
                        "high": prev_bar["close"],
                        "low": prev_bar["close"],
                        "close": prev_bar["close"],
                        "volume": 0,
                    }

                # Copy other columns if they exist
                for col in df.columns:
                    if col not in filled_row:
                        if col in prev_bar:
                            filled_row[col] = prev_bar[col]

                filled_rows.append(filled_row)
                # Update prev_bar for subsequent fills
                prev_bar = filled_row

        if filled_rows:
            filled_df = pl.DataFrame(filled_rows)
            df = pl.concat([df, filled_df]).sort("timestamp")
            logger.info(f"Filled {len(filled_rows)} missing bars")

        return df


class ParquetDataStore:
    """
    Optimized Parquet-based data store with compression.

    Provides efficient storage and retrieval of OHLCV data
    using Parquet format with zstd compression.
    """

    def __init__(self, base_dir: str | Path, compression: str = "zstd"):
        """
        Initialize the Parquet data store.

        Args:
            base_dir: Base directory for Parquet files
            compression: Compression codec (zstd, snappy, lz4, gzip)
        """
        self.base_dir = Path(base_dir)
        self.compression = compression
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_symbol(
        self,
        symbol: str,
        df: pl.DataFrame,
        partition_by: str | None = None,
    ) -> Path:
        """
        Save DataFrame to Parquet file.

        Args:
            symbol: Ticker symbol
            df: DataFrame to save
            partition_by: Optional column for partitioning

        Returns:
            Path to saved file
        """
        symbol = symbol.upper()
        file_path = self.base_dir / f"{symbol}.parquet"

        # Ensure timestamp column is datetime
        if "timestamp" in df.columns:
            df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))

        # Sort by timestamp before saving
        if "timestamp" in df.columns:
            df = df.sort("timestamp")

        # Write with compression
        df.write_parquet(
            file_path,
            compression=self.compression,
            statistics=True,
        )

        logger.info(f"Saved {len(df)} bars for {symbol} to {file_path}")
        return file_path

    def append_data(self, symbol: str, df: pl.DataFrame) -> None:
        """
        Append data to existing Parquet file.

        Args:
            symbol: Ticker symbol
            df: DataFrame to append
        """
        symbol = symbol.upper()
        file_path = self.base_dir / f"{symbol}.parquet"

        if file_path.exists():
            # Load existing data
            existing_df = pl.read_parquet(file_path)

            # Combine and deduplicate
            combined = pl.concat([existing_df, df])
            combined = combined.unique(subset=["timestamp"], keep="last")
            combined = combined.sort("timestamp")

            # Save
            self.save_symbol(symbol, combined)
        else:
            self.save_symbol(symbol, df)

    def load_symbol(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pl.DataFrame:
        """
        Load symbol data from Parquet.

        Args:
            symbol: Ticker symbol
            start_date: Start date filter
            end_date: End date filter

        Returns:
            DataFrame with OHLCV data
        """
        symbol = symbol.upper()
        file_path = self.base_dir / f"{symbol}.parquet"

        if not file_path.exists():
            raise DataNotFoundError(f"No Parquet file for symbol: {symbol}")

        # Use predicate pushdown for efficiency
        filters = []
        if start_date is not None:
            filters.append(pl.col("timestamp") >= start_date)
        if end_date is not None:
            filters.append(pl.col("timestamp") <= end_date)

        lf = pl.scan_parquet(file_path)

        for f in filters:
            lf = lf.filter(f)

        return lf.collect()

    def get_latest_timestamp(self, symbol: str) -> datetime | None:
        """Get the latest timestamp for a symbol."""
        symbol = symbol.upper()
        file_path = self.base_dir / f"{symbol}.parquet"

        if not file_path.exists():
            return None

        df = pl.scan_parquet(file_path).select("timestamp").max().collect()
        return df["timestamp"][0]

    def list_symbols(self) -> list[str]:
        """List all available symbols."""
        symbols = []
        for file_path in self.base_dir.glob("*.parquet"):
            symbols.append(file_path.stem.upper())
        return sorted(symbols)


def create_sample_data(
    symbols: list[str],
    output_dir: str | Path,
    start_date: datetime,
    end_date: datetime,
    timeframe_minutes: int = 15,
) -> None:
    """
    Create sample OHLCV data for testing.

    Args:
        symbols: List of symbols to create
        output_dir: Output directory
        start_date: Start date
        end_date: End date
        timeframe_minutes: Bar timeframe in minutes
    """
    import random

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        timestamps = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []

        current_date = start_date
        price = random.uniform(50, 500)

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                # Generate bars for trading hours
                for hour in range(9, 16):
                    for minute in range(0, 60, timeframe_minutes):
                        if hour == 9 and minute < 30:
                            continue
                        if hour == 16 and minute > 0:
                            continue

                        timestamp = current_date.replace(hour=hour, minute=minute)
                        timestamps.append(timestamp)

                        # Random walk price movement
                        change = random.gauss(0, price * 0.002)
                        open_price = price
                        close_price = price + change

                        high_price = max(open_price, close_price) + abs(random.gauss(0, price * 0.001))
                        low_price = min(open_price, close_price) - abs(random.gauss(0, price * 0.001))

                        opens.append(round(open_price, 2))
                        highs.append(round(high_price, 2))
                        lows.append(round(low_price, 2))
                        closes.append(round(close_price, 2))
                        volumes.append(random.randint(1000, 100000))

                        price = close_price

            current_date += timedelta(days=1)

        df = pl.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        })

        output_path = output_dir / f"{symbol}.parquet"
        df.write_parquet(output_path, compression="zstd")
        logger.info(f"Created sample data for {symbol}: {len(df)} bars")
