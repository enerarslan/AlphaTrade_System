"""
Unified data access layer for AlphaTrade System.

Provides a single interface for accessing data from multiple sources:
- PostgreSQL + TimescaleDB (primary source of truth)
- File-based storage (CSV, Parquet - fallback)
- Redis cache (hot data for live trading)

This layer abstracts the underlying data sources and provides:
- Automatic source selection based on configuration
- Fallback to file-based storage if database unavailable
- Consistent interface for all data access

@agent: @data, @architect
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from quant_trading_system.core.data_types import OHLCVBar, Order, Position
from quant_trading_system.core.exceptions import DataError, DataNotFoundError
from quant_trading_system.database.connection import DatabaseManager, get_db_manager

if TYPE_CHECKING:
    from quant_trading_system.data.db_loader import DatabaseDataLoader
    from quant_trading_system.data.db_feature_store import DatabaseFeatureStore
    from quant_trading_system.data.loader import DataLoader

logger = logging.getLogger(__name__)


class DataAccessConfig:
    """Configuration for data access layer."""

    def __init__(
        self,
        use_database: bool = True,
        fallback_to_files: bool = True,
        data_dir: Path | None = None,
        training_data_dir: Path | None = None,
    ):
        """
        Initialize data access configuration.

        Args:
            use_database: Whether to use PostgreSQL as primary source.
            fallback_to_files: Whether to fall back to files if DB fails.
            data_dir: Directory for raw data files.
            training_data_dir: Directory for training Parquet files.
        """
        self.use_database = use_database
        self.fallback_to_files = fallback_to_files
        self.data_dir = data_dir or Path("data/raw")
        self.training_data_dir = training_data_dir or Path("data/training")


class DataAccessLayer:
    """
    Unified data access layer with database + file fallback.

    Provides consistent interface for accessing OHLCV data, features,
    orders, and positions from the appropriate source.
    """

    def __init__(
        self,
        config: DataAccessConfig | None = None,
        db_manager: DatabaseManager | None = None,
    ):
        """
        Initialize data access layer.

        Args:
            config: Data access configuration.
            db_manager: Database manager instance.
        """
        self.config = config or DataAccessConfig()
        self._db = db_manager

        # Lazy-loaded components
        self._db_loader: "DatabaseDataLoader | None" = None
        self._feature_store: "DatabaseFeatureStore | None" = None
        self._file_loader: "DataLoader | None" = None

        # Connection status
        self._db_available: bool | None = None

    def _get_db_manager(self) -> DatabaseManager:
        """Get database manager, initializing if needed."""
        if self._db is None:
            self._db = get_db_manager()
        return self._db

    def _get_db_loader(self) -> "DatabaseDataLoader":
        """Get database data loader, initializing if needed."""
        if self._db_loader is None:
            from quant_trading_system.data.db_loader import DatabaseDataLoader
            self._db_loader = DatabaseDataLoader(self._get_db_manager())
        return self._db_loader

    def _get_feature_store(self) -> "DatabaseFeatureStore":
        """Get database feature store, initializing if needed."""
        if self._feature_store is None:
            from quant_trading_system.data.db_feature_store import DatabaseFeatureStore
            self._feature_store = DatabaseFeatureStore(self._get_db_manager())
        return self._feature_store

    def _get_file_loader(self) -> "DataLoader":
        """Get file-based data loader, initializing if needed."""
        if self._file_loader is None:
            from quant_trading_system.data.loader import DataLoader
            self._file_loader = DataLoader(self.config.data_dir)
        return self._file_loader

    def _is_db_available(self) -> bool:
        """Check if database is available."""
        if self._db_available is not None:
            return self._db_available

        try:
            db = self._get_db_manager()
            with db.session() as session:
                session.execute("SELECT 1")
            self._db_available = True
        except Exception as e:
            logger.warning(f"Database not available: {e}")
            self._db_available = False

        return self._db_available

    def get_ohlcv_bars(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pl.DataFrame:
        """
        Get OHLCV data from database or files.

        Args:
            symbol: Stock symbol.
            start_date: Start date filter.
            end_date: End date filter.

        Returns:
            Polars DataFrame with OHLCV data.

        Raises:
            DataNotFoundError: If no data found from any source.
        """
        symbol = symbol.upper()

        # Try database first if configured
        if self.config.use_database and self._is_db_available():
            try:
                return self._get_db_loader().load_symbol(
                    symbol, start_date, end_date
                )
            except DataNotFoundError:
                logger.info(f"No database data for {symbol}, trying files")
            except Exception as e:
                logger.warning(f"Database error for {symbol}: {e}")

        # Try file-based fallback
        if self.config.fallback_to_files:
            try:
                return self._get_file_loader().load_symbol(
                    symbol, start_date, end_date
                )
            except DataNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"File loader error for {symbol}: {e}")

        # Try training data Parquet
        if self.config.fallback_to_files:
            parquet_path = self.config.training_data_dir / "ohlcv" / f"{symbol}.parquet"
            if parquet_path.exists():
                try:
                    df = pl.read_parquet(parquet_path)
                    if start_date:
                        df = df.filter(pl.col("timestamp") >= start_date)
                    if end_date:
                        df = df.filter(pl.col("timestamp") <= end_date)
                    return df
                except Exception as e:
                    logger.warning(f"Parquet read error for {symbol}: {e}")

        raise DataNotFoundError(f"No data found for {symbol} from any source")

    def get_ohlcv_bars_multi(
        self,
        symbols: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Get OHLCV data for multiple symbols.

        Args:
            symbols: List of stock symbols.
            start_date: Start date filter.
            end_date: End date filter.

        Returns:
            Dictionary mapping symbol to DataFrame.
        """
        results = {}
        for symbol in symbols:
            try:
                df = self.get_ohlcv_bars(symbol, start_date, end_date)
                results[symbol.upper()] = df
            except DataNotFoundError:
                logger.warning(f"No data found for {symbol}")
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")

        return results

    def get_features(
        self,
        symbol: str,
        timestamp: datetime,
        feature_names: list[str] | None = None,
    ) -> dict[str, float] | None:
        """
        Get features for a symbol at a specific timestamp.

        Args:
            symbol: Stock symbol.
            timestamp: Feature timestamp.
            feature_names: Optional list of feature names.

        Returns:
            Dictionary of feature name to value, or None.
        """
        if self.config.use_database and self._is_db_available():
            try:
                return self._get_feature_store().get_features(
                    symbol, timestamp, feature_names
                )
            except Exception as e:
                logger.warning(f"Feature store error: {e}")

        return None

    def get_latest_features(
        self,
        symbol: str,
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Get latest features for a symbol.

        Args:
            symbol: Stock symbol.
            feature_names: Optional list of feature names.

        Returns:
            Dictionary of feature name to value.
        """
        if self.config.use_database and self._is_db_available():
            try:
                return self._get_feature_store().get_latest_features(
                    symbol, feature_names
                )
            except Exception as e:
                logger.warning(f"Feature store error: {e}")

        return {}

    def save_features(
        self,
        symbol: str,
        timestamp: datetime,
        features: dict[str, float],
    ) -> int:
        """
        Save computed features.

        Args:
            symbol: Stock symbol.
            timestamp: Feature timestamp.
            features: Dictionary of feature name to value.

        Returns:
            Number of features saved.
        """
        if self.config.use_database and self._is_db_available():
            try:
                return self._get_feature_store().save_features(
                    symbol, timestamp, features
                )
            except Exception as e:
                logger.error(f"Failed to save features: {e}")
                return 0

        return 0

    def get_available_symbols(self) -> list[str]:
        """Get list of available symbols from all sources."""
        symbols = set()

        # Try database
        if self.config.use_database and self._is_db_available():
            try:
                db_symbols = self._get_db_loader().get_available_symbols()
                symbols.update(db_symbols)
            except Exception as e:
                logger.warning(f"Database symbol query failed: {e}")

        # Try file-based
        if self.config.fallback_to_files:
            try:
                file_symbols = self._get_file_loader().get_available_symbols()
                symbols.update(file_symbols)
            except Exception as e:
                logger.warning(f"File loader symbol query failed: {e}")

        return sorted(symbols)

    def refresh_db_status(self) -> bool:
        """Refresh database availability status."""
        self._db_available = None
        return self._is_db_available()


# Singleton instance
_data_access: DataAccessLayer | None = None


def get_data_access(config: DataAccessConfig | None = None) -> DataAccessLayer:
    """Get data access layer singleton instance."""
    global _data_access
    if _data_access is None:
        _data_access = DataAccessLayer(config)
    return _data_access


def configure_data_access(
    use_database: bool = True,
    fallback_to_files: bool = True,
    data_dir: Path | str | None = None,
) -> DataAccessLayer:
    """
    Configure and return the data access layer.

    Args:
        use_database: Whether to use PostgreSQL as primary source.
        fallback_to_files: Whether to fall back to files if DB fails.
        data_dir: Directory for raw data files.

    Returns:
        Configured DataAccessLayer instance.
    """
    global _data_access

    config = DataAccessConfig(
        use_database=use_database,
        fallback_to_files=fallback_to_files,
        data_dir=Path(data_dir) if data_dir else None,
    )

    _data_access = DataAccessLayer(config)
    return _data_access
