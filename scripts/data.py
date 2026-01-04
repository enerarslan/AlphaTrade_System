"""
================================================================================
ALPHATRADE DATA MANAGEMENT
================================================================================

Institutional-grade data management for the AlphaTrade trading system.

@data: This script implements all data engineering requirements including:
  - Market data loading (CSV, Parquet, HDF5)
  - Data validation and quality checks
  - OHLCV preprocessing and cleaning
  - Database operations (PostgreSQL + TimescaleDB)
  - Alternative data ingestion (P2-A)
  - VIX feed integration (P1-A)
  - Data lineage tracking for regulatory compliance
  - Intrinsic bars generation (P3-3.4: tick, volume, dollar, imbalance)

Commands:
    python main.py data download --symbols AAPL MSFT --start 2024-01-01
    python main.py data validate --path data/raw/
    python main.py data export --format parquet --output data/export/
    python main.py data intrinsic --bar-type volume --threshold 10000
    python main.py data altdata --providers news sentiment

Author: AlphaTrade System
Version: 1.3.0
================================================================================
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data")

# ============================================================================
# DATA CONFIGURATION
# ============================================================================


@dataclass
class DataConfig:
    """Data management configuration."""

    # Data paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    export_dir: str = "data/export"

    # Download settings
    symbols: list[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    timeframe: str = "15Min"  # 1Min, 5Min, 15Min, 1Hour, 1Day

    # Validation settings
    validate_ohlcv: bool = True
    check_gaps: bool = True
    max_gap_hours: int = 24
    min_rows: int = 100

    # Preprocessing settings
    handle_missing: str = "forward_fill"  # forward_fill, interpolate, drop
    remove_outliers: bool = True
    outlier_std: float = 5.0

    # Database settings
    use_database: bool = False
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "alphatrade"

    # Alternative data (P2-A)
    alt_data_providers: list[str] = field(default_factory=list)

    # Intrinsic bars (P3-3.4)
    intrinsic_bar_type: str = ""  # tick, volume, dollar, imbalance, run
    intrinsic_threshold: float = 0.0

    # Export settings
    export_format: str = "parquet"  # csv, parquet, hdf5


# ============================================================================
# DATA LOADER
# ============================================================================


class DataManager:
    """
    Comprehensive data management for the trading system.

    @data: This class handles all data operations including:
      - Loading from multiple formats (CSV, Parquet, HDF5)
      - Validation and quality checks
      - Preprocessing and cleaning
      - Database operations
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logging.getLogger("DataManager")
        self.data_cache: dict[str, pd.DataFrame] = {}

    # ========================================================================
    # DATA LOADING
    # ========================================================================

    def load_symbol(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Load data for a single symbol.

        Args:
            symbol: Stock ticker symbol.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            DataFrame with OHLCV data.
        """
        self.logger.info(f"Loading data for {symbol}")

        # Check cache first
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        # Try different data sources
        df = None

        # Try internal loader first
        try:
            from quant_trading_system.data.loader import DataLoader

            loader = DataLoader()
            df = loader.load_symbol(symbol, start_date, end_date)
        except ImportError:
            pass

        # Fallback to CSV
        if df is None:
            df = self._load_from_csv(symbol)

        # Fallback to Parquet
        if df is None:
            df = self._load_from_parquet(symbol)

        if df is None:
            raise FileNotFoundError(f"No data found for symbol: {symbol}")

        # Filter by date range
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df["timestamp"] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df["timestamp"] <= end_dt]

        # Cache and return
        self.data_cache[cache_key] = df
        return df

    def _load_from_csv(self, symbol: str) -> pd.DataFrame | None:
        """Load data from CSV file."""
        csv_path = Path(self.config.raw_data_dir) / f"{symbol}.csv"

        if not csv_path.exists():
            # Try lowercase
            csv_path = Path(self.config.raw_data_dir) / f"{symbol.lower()}.csv"

        if not csv_path.exists():
            return None

        try:
            df = pd.read_csv(csv_path)

            # Parse timestamp
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            elif "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"])
                df = df.drop(columns=["date"])

            return df

        except Exception as e:
            self.logger.warning(f"Failed to load CSV for {symbol}: {e}")
            return None

    def _load_from_parquet(self, symbol: str) -> pd.DataFrame | None:
        """Load data from Parquet file."""
        parquet_path = Path(self.config.processed_data_dir) / f"{symbol}.parquet"

        if not parquet_path.exists():
            return None

        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            self.logger.warning(f"Failed to load Parquet for {symbol}: {e}")
            return None

    def load_all_symbols(self) -> dict[str, pd.DataFrame]:
        """Load data for all available symbols."""
        symbols = self.get_available_symbols()
        data = {}

        for symbol in symbols:
            try:
                df = self.load_symbol(symbol)
                data[symbol] = df
                self.logger.info(f"Loaded {symbol}: {len(df)} rows")
            except Exception as e:
                self.logger.warning(f"Failed to load {symbol}: {e}")

        return data

    def get_available_symbols(self) -> list[str]:
        """Get list of available symbols."""
        symbols = set()

        # Check CSV files
        raw_dir = Path(self.config.raw_data_dir)
        if raw_dir.exists():
            for csv_file in raw_dir.glob("*.csv"):
                symbols.add(csv_file.stem.upper())

        # Check Parquet files
        processed_dir = Path(self.config.processed_data_dir)
        if processed_dir.exists():
            for pq_file in processed_dir.glob("*.parquet"):
                symbols.add(pq_file.stem.upper())

        return sorted(list(symbols))

    # ========================================================================
    # DATA VALIDATION
    # ========================================================================

    def validate_data(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> dict:
        """
        Validate OHLCV data quality.

        @data: Comprehensive validation including:
          - Required columns check
          - OHLCV relationship validation (high >= low, etc.)
          - Gap detection
          - Outlier detection
          - Missing value analysis

        Args:
            df: DataFrame to validate.
            symbol: Symbol name for logging.

        Returns:
            Validation results dictionary.
        """
        self.logger.info(f"Validating data for {symbol}")

        results = {
            "symbol": symbol,
            "rows": len(df),
            "valid": True,
            "issues": [],
            "warnings": [],
        }

        # Check required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = [c for c in required_cols if c not in df.columns]

        if missing_cols:
            results["valid"] = False
            results["issues"].append(f"Missing columns: {missing_cols}")
            return results

        # Check minimum rows
        if len(df) < self.config.min_rows:
            results["valid"] = False
            results["issues"].append(f"Insufficient data: {len(df)} rows (min: {self.config.min_rows})")

        # OHLCV relationship validation
        if self.config.validate_ohlcv:
            ohlcv_issues = self._validate_ohlcv_relationships(df)
            if ohlcv_issues:
                results["warnings"].extend(ohlcv_issues)

        # Gap detection
        if self.config.check_gaps:
            gaps = self._detect_gaps(df)
            if gaps:
                results["warnings"].extend(gaps)

        # Missing values
        missing_counts = df[required_cols].isnull().sum()
        if missing_counts.any():
            for col, count in missing_counts.items():
                if count > 0:
                    pct = count / len(df) * 100
                    results["warnings"].append(f"Missing values in {col}: {count} ({pct:.1f}%)")

        # Outlier detection
        if self.config.remove_outliers:
            outliers = self._detect_outliers(df)
            if outliers:
                results["warnings"].extend(outliers)

        # Summary
        results["columns"] = list(df.columns)
        results["date_range"] = {
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
        }

        self.logger.info(f"Validation complete: {len(results['issues'])} issues, {len(results['warnings'])} warnings")

        return results

    def _validate_ohlcv_relationships(self, df: pd.DataFrame) -> list[str]:
        """Validate OHLCV relationships."""
        issues = []

        # High must be >= Low
        violations = (df["high"] < df["low"]).sum()
        if violations > 0:
            issues.append(f"High < Low violations: {violations} rows")

        # High must be >= Open and Close
        high_open = (df["high"] < df["open"]).sum()
        high_close = (df["high"] < df["close"]).sum()
        if high_open > 0:
            issues.append(f"High < Open violations: {high_open} rows")
        if high_close > 0:
            issues.append(f"High < Close violations: {high_close} rows")

        # Low must be <= Open and Close
        low_open = (df["low"] > df["open"]).sum()
        low_close = (df["low"] > df["close"]).sum()
        if low_open > 0:
            issues.append(f"Low > Open violations: {low_open} rows")
        if low_close > 0:
            issues.append(f"Low > Close violations: {low_close} rows")

        # Volume must be non-negative
        neg_volume = (df["volume"] < 0).sum()
        if neg_volume > 0:
            issues.append(f"Negative volume: {neg_volume} rows")

        # Prices must be positive
        for col in ["open", "high", "low", "close"]:
            neg_price = (df[col] <= 0).sum()
            if neg_price > 0:
                issues.append(f"Non-positive {col}: {neg_price} rows")

        return issues

    def _detect_gaps(self, df: pd.DataFrame) -> list[str]:
        """Detect gaps in time series."""
        if "timestamp" not in df.columns:
            return []

        df_sorted = df.sort_values("timestamp")
        time_diffs = df_sorted["timestamp"].diff()

        # Find large gaps (more than max_gap_hours)
        max_gap = pd.Timedelta(hours=self.config.max_gap_hours)
        large_gaps = time_diffs[time_diffs > max_gap]

        issues = []
        if len(large_gaps) > 0:
            issues.append(f"Large time gaps detected: {len(large_gaps)} gaps > {self.config.max_gap_hours}h")

        return issues

    def _detect_outliers(self, df: pd.DataFrame) -> list[str]:
        """Detect outliers in price data."""
        issues = []

        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                continue

            mean = df[col].mean()
            std = df[col].std()

            lower = mean - self.config.outlier_std * std
            upper = mean + self.config.outlier_std * std

            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > 0:
                pct = outliers / len(df) * 100
                issues.append(f"Outliers in {col}: {outliers} ({pct:.1f}%)")

        return issues

    # ========================================================================
    # DATA PREPROCESSING
    # ========================================================================

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess OHLCV data.

        @data: Preprocessing pipeline including:
          - Missing value handling
          - Outlier removal
          - Timezone normalization
          - Sorting and deduplication

        Args:
            df: Raw DataFrame.

        Returns:
            Preprocessed DataFrame.
        """
        self.logger.info(f"Preprocessing {len(df)} rows")

        df = df.copy()

        # Sort by timestamp
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        # Remove duplicates
        if "timestamp" in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=["timestamp"], keep="last")
            after = len(df)
            if before > after:
                self.logger.info(f"Removed {before - after} duplicate timestamps")

        # Handle missing values
        df = self._handle_missing(df)

        # Remove outliers
        if self.config.remove_outliers:
            df = self._remove_outliers(df)

        # Ensure timezone (UTC)
        if "timestamp" in df.columns and df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

        self.logger.info(f"Preprocessing complete: {len(df)} rows")

        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        numeric_cols = ["open", "high", "low", "close", "volume"]
        numeric_cols = [c for c in numeric_cols if c in df.columns]

        if self.config.handle_missing == "forward_fill":
            df[numeric_cols] = df[numeric_cols].ffill()
        elif self.config.handle_missing == "interpolate":
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
        elif self.config.handle_missing == "drop":
            df = df.dropna(subset=numeric_cols)

        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outlier rows."""
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                continue

            mean = df[col].mean()
            std = df[col].std()

            lower = mean - self.config.outlier_std * std
            upper = mean + self.config.outlier_std * std

            df = df[(df[col] >= lower) & (df[col] <= upper)]

        return df

    # ========================================================================
    # DATA DOWNLOAD
    # ========================================================================

    async def download_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        """
        Download market data from Alpaca.

        @data: Downloads OHLCV bars from Alpaca Markets API.

        Args:
            symbols: List of ticker symbols.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            Dictionary of DataFrames by symbol.
        """
        self.logger.info(f"Downloading data for {len(symbols)} symbols")
        self.logger.info(f"Date range: {start_date} to {end_date}")

        downloaded = {}

        try:
            from quant_trading_system.execution.alpaca_client import AlpacaClient

            client = AlpacaClient()

            for symbol in symbols:
                try:
                    bars = await client.get_bars(
                        symbol=symbol,
                        start=start_date,
                        end=end_date,
                        timeframe=self.config.timeframe,
                    )

                    if bars is not None and len(bars) > 0:
                        downloaded[symbol] = bars
                        self.logger.info(f"Downloaded {symbol}: {len(bars)} bars")
                    else:
                        self.logger.warning(f"No data for {symbol}")

                except Exception as e:
                    self.logger.error(f"Failed to download {symbol}: {e}")

        except ImportError:
            self.logger.error("AlpacaClient not available. Set ALPACA_API_KEY and ALPACA_API_SECRET.")
            return {}

        return downloaded

    def save_data(
        self,
        data: dict[str, pd.DataFrame],
        output_dir: str | None = None,
        format: str = "csv",
    ) -> list[str]:
        """
        Save data to files.

        Args:
            data: Dictionary of DataFrames by symbol.
            output_dir: Output directory.
            format: Output format (csv, parquet, hdf5).

        Returns:
            List of saved file paths.
        """
        output_dir = output_dir or self.config.raw_data_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = []

        for symbol, df in data.items():
            if format == "csv":
                file_path = output_path / f"{symbol}.csv"
                df.to_csv(file_path, index=False)
            elif format == "parquet":
                file_path = output_path / f"{symbol}.parquet"
                df.to_parquet(file_path, index=False)
            elif format == "hdf5":
                file_path = output_path / f"{symbol}.h5"
                df.to_hdf(file_path, key="data", mode="w")
            else:
                raise ValueError(f"Unknown format: {format}")

            saved_files.append(str(file_path))
            self.logger.info(f"Saved {symbol} to {file_path}")

        return saved_files

    # ========================================================================
    # INTRINSIC BARS (P3-3.4)
    # ========================================================================

    def generate_intrinsic_bars(
        self,
        df: pd.DataFrame,
        bar_type: str,
        threshold: float,
    ) -> pd.DataFrame:
        """
        Generate intrinsic time bars.

        @data P3-3.4: Implements various bar types based on information content:
          - tick: Fixed number of trades
          - volume: Fixed volume
          - dollar: Fixed dollar volume
          - imbalance: Based on signed volume imbalance
          - run: Based on consecutive buy/sell sequences

        Args:
            df: Input tick/bar data.
            bar_type: Type of bar (tick, volume, dollar, imbalance, run).
            threshold: Threshold for bar completion.

        Returns:
            DataFrame with intrinsic bars.
        """
        self.logger.info(f"Generating {bar_type} bars with threshold {threshold}")

        try:
            from quant_trading_system.data.intrinsic_bars import (
                generate_tick_bars,
                generate_volume_bars,
                generate_dollar_bars,
                generate_imbalance_bars,
                generate_run_bars,
            )

            if bar_type == "tick":
                return generate_tick_bars(df, int(threshold))
            elif bar_type == "volume":
                return generate_volume_bars(df, threshold)
            elif bar_type == "dollar":
                return generate_dollar_bars(df, threshold)
            elif bar_type == "imbalance":
                return generate_imbalance_bars(df, threshold)
            elif bar_type == "run":
                return generate_run_bars(df, threshold)
            else:
                raise ValueError(f"Unknown bar type: {bar_type}")

        except ImportError:
            # Fallback implementation
            return self._generate_intrinsic_bars_fallback(df, bar_type, threshold)

    def _generate_intrinsic_bars_fallback(
        self,
        df: pd.DataFrame,
        bar_type: str,
        threshold: float,
    ) -> pd.DataFrame:
        """Fallback intrinsic bar generation."""
        self.logger.info("Using fallback intrinsic bar generation")

        if bar_type == "volume":
            return self._generate_volume_bars(df, threshold)
        elif bar_type == "dollar":
            df["dollar_volume"] = df["close"] * df["volume"]
            return self._generate_cumulative_bars(df, "dollar_volume", threshold)
        else:
            # Default to time bars (passthrough)
            self.logger.warning(f"Bar type {bar_type} not supported in fallback, returning original")
            return df

    def _generate_volume_bars(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Generate volume bars."""
        df = df.copy()
        df["cum_volume"] = df["volume"].cumsum()
        df["bar_idx"] = (df["cum_volume"] / threshold).astype(int)

        bars = df.groupby("bar_idx").agg({
            "timestamp": "first",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).reset_index(drop=True)

        return bars

    def _generate_cumulative_bars(
        self,
        df: pd.DataFrame,
        column: str,
        threshold: float
    ) -> pd.DataFrame:
        """Generate bars based on cumulative threshold."""
        df = df.copy()
        df["cumsum"] = df[column].cumsum()
        df["bar_idx"] = (df["cumsum"] / threshold).astype(int)

        bars = df.groupby("bar_idx").agg({
            "timestamp": "first",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).reset_index(drop=True)

        return bars

    # ========================================================================
    # ALTERNATIVE DATA (P2-A)
    # ========================================================================

    async def fetch_alternative_data(
        self,
        symbols: list[str],
        providers: list[str],
    ) -> dict[str, dict]:
        """
        Fetch alternative data from various providers.

        @data P2-A: Alternative data sources include:
          - news: News sentiment
          - social: Twitter/Reddit sentiment
          - web_traffic: Web analytics
          - satellite: Satellite imagery
          - credit_card: Consumer spending
          - supply_chain: Shipping/inventory data

        Args:
            symbols: List of ticker symbols.
            providers: List of data providers.

        Returns:
            Dictionary of alternative data by symbol.
        """
        self.logger.info(f"Fetching alternative data from: {providers}")

        alt_data = {}

        try:
            from quant_trading_system.data.alternative_data import (
                create_alt_data_aggregator,
            )

            aggregator = create_alt_data_aggregator(providers)

            for symbol in symbols:
                try:
                    data = await aggregator.get_composite_signal(symbol)
                    alt_data[symbol] = data
                    self.logger.info(f"Fetched alt data for {symbol}")
                except Exception as e:
                    self.logger.warning(f"Failed to fetch alt data for {symbol}: {e}")

        except ImportError:
            self.logger.warning("Alternative data module not available")

        return alt_data

    # ========================================================================
    # VIX FEED (P1-A)
    # ========================================================================

    async def fetch_vix_data(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch VIX data for regime detection.

        @data P1-A: VIX data is used for:
          - Market regime detection
          - Dynamic risk adjustment
          - Kill switch triggers

        Args:
            start_date: Start date.
            end_date: End date.

        Returns:
            DataFrame with VIX data.
        """
        self.logger.info("Fetching VIX data")

        try:
            from quant_trading_system.data.vix_feed import VIXFeed

            feed = VIXFeed()
            vix_data = await feed.get_historical(start_date, end_date)

            self.logger.info(f"Fetched {len(vix_data)} VIX records")
            return vix_data

        except ImportError:
            # Fallback: Try to download from Yahoo Finance
            self.logger.info("Using fallback VIX data source")
            return self._fetch_vix_fallback(start_date, end_date)

    def _fetch_vix_fallback(
        self,
        start_date: str | None,
        end_date: str | None,
    ) -> pd.DataFrame:
        """Fallback VIX data from Yahoo Finance."""
        try:
            import yfinance as yf

            vix = yf.Ticker("^VIX")
            df = vix.history(start=start_date, end=end_date)

            if len(df) > 0:
                df = df.reset_index()
                df = df.rename(columns={
                    "Date": "timestamp",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                })
                return df

        except ImportError:
            self.logger.warning("yfinance not installed")

        return pd.DataFrame()

    # ========================================================================
    # DATA LINEAGE
    # ========================================================================

    def register_lineage(
        self,
        source: str,
        transformation: str,
        output: str,
        metadata: dict | None = None,
    ) -> str:
        """
        Register data lineage for regulatory compliance.

        @data: Data lineage tracking provides:
          - Full provenance of all data
          - Transformation history
          - Audit trail for regulators

        Args:
            source: Source data identifier.
            transformation: Transformation applied.
            output: Output data identifier.
            metadata: Additional metadata.

        Returns:
            Lineage record ID.
        """
        try:
            from quant_trading_system.data.data_lineage import DataLineageTracker

            tracker = DataLineageTracker()
            record_id = tracker.register(
                source=source,
                transformation=transformation,
                output=output,
                metadata=metadata or {},
            )

            self.logger.info(f"Registered lineage: {record_id}")
            return record_id

        except ImportError:
            # Fallback: Simple logging
            record_id = f"lineage_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            self.logger.info(f"Lineage (fallback): {source} -> {transformation} -> {output}")
            return record_id

    # ========================================================================
    # DATABASE OPERATIONS
    # ========================================================================

    async def sync_to_database(self, data: dict[str, pd.DataFrame]) -> int:
        """
        Sync data to PostgreSQL/TimescaleDB.

        @data: Database operations include:
          - Upsert with conflict handling
          - TimescaleDB hypertable optimization
          - Batch processing for performance

        Args:
            data: Dictionary of DataFrames by symbol.

        Returns:
            Number of rows synced.
        """
        if not self.config.use_database:
            self.logger.warning("Database sync disabled")
            return 0

        try:
            from quant_trading_system.database.repository import OHLCVRepository

            repo = OHLCVRepository()
            total_rows = 0

            for symbol, df in data.items():
                try:
                    rows = await repo.upsert_bars(symbol, df)
                    total_rows += rows
                    self.logger.info(f"Synced {symbol}: {rows} rows")
                except Exception as e:
                    self.logger.error(f"Failed to sync {symbol}: {e}")

            return total_rows

        except ImportError:
            self.logger.warning("Database module not available")
            return 0


# ============================================================================
# COMMAND HANDLERS
# ============================================================================


def cmd_download(args: argparse.Namespace) -> int:
    """Download market data."""
    logger.info("=" * 80)
    logger.info("DATA DOWNLOAD")
    logger.info("=" * 80)

    config = DataConfig(
        symbols=getattr(args, "symbols", []),
        start_date=getattr(args, "start_date", ""),
        end_date=getattr(args, "end_date", ""),
        timeframe=getattr(args, "timeframe", "15Min"),
    )

    manager = DataManager(config)

    async def _download():
        symbols = config.symbols
        if not symbols:
            logger.error("No symbols specified. Use --symbols AAPL MSFT ...")
            return 1

        start = config.start_date or (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")
        end = config.end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        data = await manager.download_data(symbols, start, end)

        if data:
            manager.save_data(data, format="csv")
            logger.info(f"Downloaded data for {len(data)} symbols")
            return 0
        else:
            logger.error("No data downloaded")
            return 1

    return asyncio.run(_download())


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate data files."""
    logger.info("=" * 80)
    logger.info("DATA VALIDATION")
    logger.info("=" * 80)

    config = DataConfig(
        raw_data_dir=getattr(args, "path", "data/raw"),
    )

    manager = DataManager(config)
    symbols = manager.get_available_symbols()

    logger.info(f"Found {len(symbols)} symbols")

    all_valid = True
    results = []

    for symbol in symbols:
        try:
            df = manager.load_symbol(symbol)
            result = manager.validate_data(df, symbol)
            results.append(result)

            if not result["valid"]:
                all_valid = False
                logger.error(f"{symbol}: INVALID - {result['issues']}")
            else:
                logger.info(f"{symbol}: VALID ({len(result['warnings'])} warnings)")

        except Exception as e:
            logger.error(f"{symbol}: ERROR - {e}")
            all_valid = False

    # Summary
    logger.info("=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info(f"Total symbols: {len(symbols)}")
    logger.info(f"Valid: {sum(1 for r in results if r['valid'])}")
    logger.info(f"Invalid: {sum(1 for r in results if not r['valid'])}")
    logger.info("=" * 80)

    return 0 if all_valid else 1


def cmd_preprocess(args: argparse.Namespace) -> int:
    """Preprocess data files."""
    logger.info("=" * 80)
    logger.info("DATA PREPROCESSING")
    logger.info("=" * 80)

    config = DataConfig(
        raw_data_dir=getattr(args, "input", "data/raw"),
        processed_data_dir=getattr(args, "output", "data/processed"),
        handle_missing=getattr(args, "handle_missing", "forward_fill"),
        remove_outliers=getattr(args, "remove_outliers", True),
    )

    manager = DataManager(config)
    symbols = manager.get_available_symbols()

    for symbol in symbols:
        try:
            df = manager.load_symbol(symbol)
            df_processed = manager.preprocess(df)

            # Save processed data
            output_path = Path(config.processed_data_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            df_processed.to_parquet(output_path / f"{symbol}.parquet", index=False)
            logger.info(f"Processed {symbol}: {len(df)} -> {len(df_processed)} rows")

        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export data to specified format."""
    logger.info("=" * 80)
    logger.info("DATA EXPORT")
    logger.info("=" * 80)

    config = DataConfig(
        raw_data_dir=getattr(args, "input", "data/raw"),
        export_dir=getattr(args, "output", "data/export"),
        export_format=getattr(args, "format", "parquet"),
    )

    manager = DataManager(config)
    data = manager.load_all_symbols()

    if data:
        saved = manager.save_data(data, config.export_dir, config.export_format)
        logger.info(f"Exported {len(saved)} files to {config.export_dir}")
        return 0
    else:
        logger.error("No data to export")
        return 1


def cmd_intrinsic(args: argparse.Namespace) -> int:
    """Generate intrinsic bars."""
    logger.info("=" * 80)
    logger.info("INTRINSIC BAR GENERATION")
    logger.info("=" * 80)

    bar_type = getattr(args, "bar_type", "volume")
    threshold = getattr(args, "threshold", 10000)

    config = DataConfig(
        raw_data_dir=getattr(args, "input", "data/raw"),
        processed_data_dir=getattr(args, "output", "data/intrinsic"),
    )

    manager = DataManager(config)
    symbols = manager.get_available_symbols()

    output_dir = Path(config.processed_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        try:
            df = manager.load_symbol(symbol)
            bars = manager.generate_intrinsic_bars(df, bar_type, threshold)

            bars.to_parquet(output_dir / f"{symbol}_{bar_type}.parquet", index=False)
            logger.info(f"Generated {bar_type} bars for {symbol}: {len(df)} -> {len(bars)} bars")

        except Exception as e:
            logger.error(f"Failed to generate bars for {symbol}: {e}")

    return 0


def cmd_altdata(args: argparse.Namespace) -> int:
    """Fetch alternative data."""
    logger.info("=" * 80)
    logger.info("ALTERNATIVE DATA FETCH")
    logger.info("=" * 80)

    providers = getattr(args, "providers", ["news", "sentiment"])
    symbols = getattr(args, "symbols", [])

    if not symbols:
        logger.error("No symbols specified")
        return 1

    config = DataConfig(alt_data_providers=providers)
    manager = DataManager(config)

    async def _fetch():
        data = await manager.fetch_alternative_data(symbols, providers)

        if data:
            output_dir = Path("data/alternative")
            output_dir.mkdir(parents=True, exist_ok=True)

            for symbol, alt_data in data.items():
                with open(output_dir / f"{symbol}_altdata.json", "w") as f:
                    json.dump(alt_data, f, indent=2, default=str)

            logger.info(f"Fetched alternative data for {len(data)} symbols")
            return 0
        else:
            logger.warning("No alternative data fetched")
            return 1

    return asyncio.run(_fetch())


def cmd_vix(args: argparse.Namespace) -> int:
    """Fetch VIX data."""
    logger.info("=" * 80)
    logger.info("VIX DATA FETCH")
    logger.info("=" * 80)

    start_date = getattr(args, "start_date", None)
    end_date = getattr(args, "end_date", None)

    config = DataConfig()
    manager = DataManager(config)

    async def _fetch():
        vix_data = await manager.fetch_vix_data(start_date, end_date)

        if len(vix_data) > 0:
            output_path = Path("data/raw/VIX.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            vix_data.to_csv(output_path, index=False)

            logger.info(f"Saved VIX data: {len(vix_data)} rows")
            return 0
        else:
            logger.warning("No VIX data fetched")
            return 1

    return asyncio.run(_fetch())


def cmd_list(args: argparse.Namespace) -> int:
    """List available symbols."""
    config = DataConfig(
        raw_data_dir=getattr(args, "path", "data/raw"),
    )

    manager = DataManager(config)
    symbols = manager.get_available_symbols()

    print(f"Available symbols ({len(symbols)}):")
    for symbol in symbols:
        print(f"  {symbol}")

    return 0


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def run_data_command(args: argparse.Namespace) -> int:
    """
    Main entry point for data commands.

    @data: This function routes to the appropriate data command handler.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    command = getattr(args, "data_command", "list")

    commands = {
        "download": cmd_download,
        "validate": cmd_validate,
        "preprocess": cmd_preprocess,
        "export": cmd_export,
        "intrinsic": cmd_intrinsic,
        "altdata": cmd_altdata,
        "vix": cmd_vix,
        "list": cmd_list,
    }

    handler = commands.get(command)
    if handler:
        return handler(args)
    else:
        logger.error(f"Unknown data command: {command}")
        return 1


if __name__ == "__main__":
    # Direct execution for testing
    parser = argparse.ArgumentParser(description="AlphaTrade Data Management")
    subparsers = parser.add_subparsers(dest="data_command")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download market data")
    download_parser.add_argument("--symbols", nargs="+", required=True)
    download_parser.add_argument("--start-date", type=str)
    download_parser.add_argument("--end-date", type=str)

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate data")
    validate_parser.add_argument("--path", type=str, default="data/raw")

    # List command
    list_parser = subparsers.add_parser("list", help="List available symbols")
    list_parser.add_argument("--path", type=str, default="data/raw")

    args = parser.parse_args()
    sys.exit(run_data_command(args))
