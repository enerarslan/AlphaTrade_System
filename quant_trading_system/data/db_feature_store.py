"""
Database-backed feature store for PostgreSQL + TimescaleDB.

Stores computed ML features in the database with Redis hot cache.
Part of the hybrid architecture where:
- PostgreSQL is the persistent source of truth
- Redis provides low-latency access for live trading
- Parquet exports are used for ML training

@agent: @data, @mlquant
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from quant_trading_system.core.exceptions import DataError, DataNotFoundError
from quant_trading_system.data.timeframe import DEFAULT_TIMEFRAME, normalize_timeframe
from quant_trading_system.database.connection import DatabaseManager, get_db_manager
from quant_trading_system.database.models import Feature
from quant_trading_system.database.repository import (
    FeatureRepository,
    get_feature_repository,
)

if TYPE_CHECKING:
    from redis import Redis

logger = logging.getLogger(__name__)


class DatabaseFeatureStore:
    """
    Database-backed feature store with Redis hot cache.

    This store implements a multi-tier caching strategy:
    1. Memory cache (LRU, bounded) - fastest
    2. Redis cache (TTL-based) - distributed, persistent
    3. PostgreSQL (TimescaleDB) - permanent storage

    Key features:
    - Batch feature saving for efficiency
    - Point-in-time feature retrieval
    - Parquet export for ML training
    - Feature versioning support
    """

    def __init__(
        self,
        db_manager: DatabaseManager | None = None,
        redis_client: "Redis | None" = None,
        memory_cache_size: int = 1000,
        redis_ttl: int = 3600,
        batch_size: int = 5000,
    ):
        """
        Initialize the database feature store.

        Args:
            db_manager: Database manager instance. If None, uses default.
            redis_client: Optional Redis client for caching.
            memory_cache_size: Maximum entries in memory cache.
            redis_ttl: Redis cache TTL in seconds.
            batch_size: Batch size for bulk operations.
        """
        self._db = db_manager or get_db_manager()
        self._feature_repo = FeatureRepository(self._db)
        self._redis = redis_client
        self._memory_cache: dict[str, dict[str, float]] = {}
        self._memory_cache_size = memory_cache_size
        self._redis_ttl = redis_ttl
        self._batch_size = batch_size

        # LRU tracking for memory cache
        self._cache_access_order: list[str] = []

    def save_features(
        self,
        symbol: str,
        timestamp: datetime,
        features: dict[str, float],
        timeframe: str = DEFAULT_TIMEFRAME,
        feature_set_id: str = "default",
        batch_mode: bool = False,
    ) -> int:
        """
        Save computed features to database and cache.

        Args:
            symbol: Stock symbol.
            timestamp: Feature timestamp.
            features: Dictionary of feature name to value.
            batch_mode: If True, defer commit for batch operations.

        Returns:
            Number of features saved.
        """
        symbol = symbol.upper()
        timeframe = normalize_timeframe(timeframe)
        feature_set_id = str(feature_set_id or "default").strip() or "default"

        # Ensure timezone-aware timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Prepare feature records
        feature_records = []
        for name, value in features.items():
            if value is not None and not (isinstance(value, float) and (value != value)):  # Check for NaN
                feature_records.append({
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "timeframe": timeframe,
                    "feature_name": name,
                    "feature_set_id": feature_set_id,
                    "value": float(value),
                })

        if not feature_records:
            return 0

        # Save to database
        with self._db.session() as session:
            count = self._feature_repo.bulk_insert(session, feature_records)

        # Update caches
        cache_key = self._make_cache_key(symbol, timestamp, timeframe, feature_set_id)

        # Update memory cache
        self._memory_cache[cache_key] = features
        self._update_cache_lru(cache_key)

        # Update Redis cache
        if self._redis is not None:
            try:
                self._redis.setex(
                    cache_key,
                    self._redis_ttl,
                    json.dumps(features),
                )
            except Exception as e:
                logger.warning(f"Failed to update Redis cache: {e}")

        logger.debug(f"Saved {count} features for {symbol} at {timestamp}")
        return count

    def save_features_batch(
        self,
        symbol: str,
        features_df: pl.DataFrame,
        timeframe: str = DEFAULT_TIMEFRAME,
        feature_set_id: str = "default",
    ) -> int:
        """
        Batch save features from a DataFrame.

        Args:
            symbol: Stock symbol.
            features_df: DataFrame with 'timestamp' column and feature columns.

        Returns:
            Total number of features saved.
        """
        symbol = symbol.upper()
        timeframe = normalize_timeframe(timeframe)
        feature_set_id = str(feature_set_id or "default").strip() or "default"

        if "timestamp" not in features_df.columns:
            raise DataError("features_df must have 'timestamp' column")

        # Get feature columns (all except timestamp)
        feature_cols = [c for c in features_df.columns if c != "timestamp"]

        total_saved = 0
        feature_records = []

        for row in features_df.iter_rows(named=True):
            timestamp = row["timestamp"]
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            for col in feature_cols:
                value = row[col]
                if value is not None and not (isinstance(value, float) and (value != value)):
                    feature_records.append({
                        "symbol": symbol,
                        "timestamp": timestamp,
                        "timeframe": timeframe,
                        "feature_name": col,
                        "feature_set_id": feature_set_id,
                        "value": float(value),
                    })

            # Batch insert when buffer is full
            if len(feature_records) >= self._batch_size:
                with self._db.session() as session:
                    count = self._feature_repo.bulk_insert(session, feature_records)
                    total_saved += count
                feature_records = []

        # Insert remaining records
        if feature_records:
            with self._db.session() as session:
                count = self._feature_repo.bulk_insert(session, feature_records)
                total_saved += count

        logger.info(f"Batch saved {total_saved} features for {symbol}")
        return total_saved

    def get_features(
        self,
        symbol: str,
        timestamp: datetime,
        timeframe: str = DEFAULT_TIMEFRAME,
        feature_set_id: str = "default",
        feature_names: list[str] | None = None,
    ) -> dict[str, float] | None:
        """
        Get features for a symbol at a specific timestamp.

        Checks caches first (memory, Redis), then falls back to database.

        Args:
            symbol: Stock symbol.
            timestamp: Feature timestamp.
            feature_names: Optional list of feature names to retrieve.

        Returns:
            Dictionary of feature name to value, or None if not found.
        """
        symbol = symbol.upper()
        timeframe = normalize_timeframe(timeframe)
        feature_set_id = str(feature_set_id or "default").strip() or "default"
        cache_key = self._make_cache_key(symbol, timestamp, timeframe, feature_set_id)

        # L1: Check memory cache
        if cache_key in self._memory_cache:
            features = self._memory_cache[cache_key]
            self._update_cache_lru(cache_key)
            if feature_names:
                return {k: v for k, v in features.items() if k in feature_names}
            return features

        # L2: Check Redis cache
        if self._redis is not None:
            try:
                cached = self._redis.get(cache_key)
                if cached:
                    features = json.loads(cached)
                    # Populate memory cache
                    self._memory_cache[cache_key] = features
                    self._update_cache_lru(cache_key)
                    if feature_names:
                        return {k: v for k, v in features.items() if k in feature_names}
                    return features
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")

        # L3: Query database
        with self._db.session() as session:
            db_features = self._feature_repo.get_features(
                session,
                symbol,
                timeframe=timeframe,
                feature_set_id=feature_set_id,
                feature_names=feature_names,
                start_time=timestamp,
                end_time=timestamp,
            )

            if not db_features:
                return None

            features = {f.feature_name: f.value for f in db_features}

            # Populate caches
            self._memory_cache[cache_key] = features
            self._update_cache_lru(cache_key)

            if self._redis is not None:
                try:
                    self._redis.setex(cache_key, self._redis_ttl, json.dumps(features))
                except Exception as e:
                    logger.warning(f"Failed to populate Redis cache: {e}")

            return features

    def get_latest_features(
        self,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
        feature_set_id: str = "default",
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Get the most recent features for a symbol.

        Args:
            symbol: Stock symbol.
            feature_names: Optional list of feature names.

        Returns:
            Dictionary of feature name to value.
        """
        symbol = symbol.upper()
        timeframe = normalize_timeframe(timeframe)
        feature_set_id = str(feature_set_id or "default").strip() or "default"

        with self._db.session() as session:
            return self._feature_repo.get_latest_features(
                session,
                symbol,
                timeframe=timeframe,
                feature_set_id=feature_set_id,
                feature_names=feature_names,
            )

    def get_features_range(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = DEFAULT_TIMEFRAME,
        feature_set_id: str = "default",
        feature_names: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Get features for a date range as a DataFrame.

        Args:
            symbol: Stock symbol.
            start_date: Start date (inclusive).
            end_date: End date (inclusive).
            feature_names: Optional list of feature names.

        Returns:
            Polars DataFrame with features.
        """
        symbol = symbol.upper()
        timeframe = normalize_timeframe(timeframe)
        feature_set_id = str(feature_set_id or "default").strip() or "default"

        with self._db.session() as session:
            db_features = self._feature_repo.get_features(
                session,
                symbol,
                timeframe=timeframe,
                feature_set_id=feature_set_id,
                feature_names=feature_names,
                start_time=start_date,
                end_time=end_date,
            )

        if not db_features:
            return pl.DataFrame()

        # Convert to wide format (timestamp, feature1, feature2, ...)
        data = {}
        for f in db_features:
            ts_key = f.timestamp.isoformat()
            if ts_key not in data:
                data[ts_key] = {"timestamp": f.timestamp}
            data[ts_key][f.feature_name] = f.value

        rows = list(data.values())
        if not rows:
            return pl.DataFrame()

        return pl.DataFrame(rows).sort("timestamp")

    def export_to_parquet(
        self,
        symbol: str,
        output_path: Path,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        timeframe: str = DEFAULT_TIMEFRAME,
        feature_set_id: str = "default",
        feature_names: list[str] | None = None,
        compression: str = "zstd",
    ) -> Path:
        """
        Export features to Parquet file for ML training.

        Args:
            symbol: Stock symbol.
            output_path: Output file path.
            start_date: Start date filter.
            end_date: End date filter.
            feature_names: Optional list of features to export.
            compression: Compression codec.

        Returns:
            Path to exported file.
        """
        symbol = symbol.upper()
        timeframe = normalize_timeframe(timeframe)
        feature_set_id = str(feature_set_id or "default").strip() or "default"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use very wide date range if not specified
        if start_date is None:
            start_date = datetime(2000, 1, 1, tzinfo=timezone.utc)
        if end_date is None:
            end_date = datetime(2100, 1, 1, tzinfo=timezone.utc)

        df = self.get_features_range(
            symbol,
            start_date,
            end_date,
            timeframe=timeframe,
            feature_set_id=feature_set_id,
            feature_names=feature_names,
        )

        if len(df) == 0:
            raise DataNotFoundError(f"No features found for {symbol}")

        df.write_parquet(output_path, compression=compression, statistics=True)

        logger.info(
            f"Exported {len(df)} feature rows for {symbol} ({timeframe}, {feature_set_id}) to {output_path}"
        )
        return output_path

    def export_all_to_parquet(
        self,
        output_dir: Path,
        symbols: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        timeframe: str = DEFAULT_TIMEFRAME,
        feature_set_id: str = "default",
        compression: str = "zstd",
    ) -> dict[str, Path]:
        """
        Export all symbols' features to Parquet files.

        Args:
            output_dir: Output directory.
            symbols: List of symbols. If None, exports all available.
            start_date: Start date filter.
            end_date: End date filter.
            compression: Compression codec.

        Returns:
            Dictionary mapping symbol to output path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timeframe = normalize_timeframe(timeframe)
        feature_set_id = str(feature_set_id or "default").strip() or "default"

        if symbols is None:
            symbols = self.get_available_symbols(timeframe=timeframe, feature_set_id=feature_set_id)

        results = {}
        for symbol in symbols:
            try:
                output_path = output_dir / f"{symbol}_features.parquet"
                self.export_to_parquet(
                    symbol,
                    output_path,
                    start_date,
                    end_date,
                    timeframe=timeframe,
                    feature_set_id=feature_set_id,
                    compression=compression,
                )
                results[symbol] = output_path
            except DataNotFoundError:
                logger.warning(f"No features found for {symbol}")
            except Exception as e:
                logger.error(f"Failed to export features for {symbol}: {e}")

        return results

    def get_available_symbols(
        self,
        timeframe: str = DEFAULT_TIMEFRAME,
        feature_set_id: str = "default",
    ) -> list[str]:
        """Get list of symbols with features in database."""
        timeframe = normalize_timeframe(timeframe)
        feature_set_id = str(feature_set_id or "default").strip() or "default"
        with self._db.session() as session:
            from sqlalchemy import and_, select
            stmt = (
                select(Feature.symbol)
                .where(
                    and_(
                        Feature.timeframe == timeframe,
                        Feature.feature_set_id == feature_set_id,
                    )
                )
                .distinct()
            )
            result = session.execute(stmt)
            return sorted([row[0] for row in result])

    def get_feature_names(
        self,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
        feature_set_id: str = "default",
    ) -> list[str]:
        """Get list of feature names for a symbol."""
        symbol = symbol.upper()
        timeframe = normalize_timeframe(timeframe)
        feature_set_id = str(feature_set_id or "default").strip() or "default"

        with self._db.session() as session:
            from sqlalchemy import and_, select
            stmt = (
                select(Feature.feature_name)
                .where(
                    and_(
                        Feature.symbol == symbol,
                        Feature.timeframe == timeframe,
                        Feature.feature_set_id == feature_set_id,
                    )
                )
                .distinct()
            )
            result = session.execute(stmt)
            return sorted([row[0] for row in result])

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._memory_cache.clear()
        self._cache_access_order.clear()

        if self._redis is not None:
            try:
                # Clear only feature keys
                keys = self._redis.keys("feature:*")
                if keys:
                    self._redis.delete(*keys)
            except Exception as e:
                logger.warning(f"Failed to clear Redis cache: {e}")

    def _make_cache_key(
        self,
        symbol: str,
        timestamp: datetime,
        timeframe: str,
        feature_set_id: str,
    ) -> str:
        """Generate cache key for symbol and timestamp."""
        ts_str = timestamp.isoformat()
        return f"feature:{symbol}:{timeframe}:{feature_set_id}:{ts_str}"

    def _update_cache_lru(self, key: str) -> None:
        """Update LRU tracking for memory cache."""
        # Remove from current position
        if key in self._cache_access_order:
            self._cache_access_order.remove(key)

        # Add to end (most recently used)
        self._cache_access_order.append(key)

        # Evict oldest entries if cache is full
        while len(self._memory_cache) > self._memory_cache_size:
            if self._cache_access_order:
                oldest_key = self._cache_access_order.pop(0)
                self._memory_cache.pop(oldest_key, None)


# Singleton instance
_db_feature_store: DatabaseFeatureStore | None = None


def get_db_feature_store() -> DatabaseFeatureStore:
    """Get database feature store singleton instance."""
    global _db_feature_store
    if _db_feature_store is None:
        _db_feature_store = DatabaseFeatureStore()
    return _db_feature_store
