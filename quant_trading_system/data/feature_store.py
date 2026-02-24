"""
Feature store for computing, caching, and serving features.

Implements a multi-layer feature store with:
- Computation layer for on-demand feature calculation
- Caching layer with Redis and memory cache
- Storage layer with database and Parquet
- Serving layer for point-in-time queries

SECURITY: Uses JSON-based serialization instead of pickle to prevent
arbitrary code execution vulnerabilities. Numpy arrays are handled
with base64 encoding for efficiency.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl

from quant_trading_system.core.data_types import FeatureVector
from quant_trading_system.core.exceptions import DataError, DataNotFoundError

logger = logging.getLogger(__name__)


def _safe_serialize(value: Any) -> bytes:
    """Safely serialize a value for caching.

    SECURITY FIX: Uses JSON + base64 instead of pickle to prevent
    arbitrary code execution from malicious cache entries.

    Args:
        value: Value to serialize.

    Returns:
        Serialized bytes.
    """
    if isinstance(value, np.ndarray):
        # Encode numpy arrays as base64 for efficiency
        return json.dumps({
            "__numpy__": True,
            "dtype": str(value.dtype),
            "shape": value.shape,
            "data": base64.b64encode(value.tobytes()).decode("ascii"),
        }).encode("utf-8")
    elif isinstance(value, pl.DataFrame):
        # Serialize polars DataFrames as JSON
        return json.dumps({
            "__polars__": True,
            "data": value.to_dicts(),
        }).encode("utf-8")
    elif isinstance(value, pl.Series):
        # Serialize polars Series as JSON
        return json.dumps({
            "__polars_series__": True,
            "name": value.name,
            "data": value.to_list(),
        }).encode("utf-8")
    elif isinstance(value, (datetime,)):
        return json.dumps({
            "__datetime__": True,
            "iso": value.isoformat(),
        }).encode("utf-8")
    elif isinstance(value, FeatureVector):
        # Serialize FeatureVector using Pydantic model_dump
        return json.dumps({
            "__feature_vector__": True,
            "data": value.model_dump(mode="json"),
        }).encode("utf-8")
    elif hasattr(value, 'model_dump'):
        # Generic Pydantic model support
        return json.dumps({
            "__pydantic__": True,
            "class": value.__class__.__name__,
            "data": value.model_dump(mode="json"),
        }).encode("utf-8")
    else:
        # Default JSON serialization
        return json.dumps(value, default=str).encode("utf-8")


def _safe_deserialize(data: bytes) -> Any:
    """Safely deserialize a cached value.

    SECURITY FIX: Uses JSON + base64 instead of pickle.
    Only reconstructs known safe types.

    Args:
        data: Serialized bytes.

    Returns:
        Deserialized value.
    """
    try:
        obj = json.loads(data.decode("utf-8"))

        if isinstance(obj, dict):
            if obj.get("__numpy__"):
                # Reconstruct numpy array
                dtype = np.dtype(obj["dtype"])
                shape = tuple(obj["shape"])
                buffer = base64.b64decode(obj["data"])
                return np.frombuffer(buffer, dtype=dtype).reshape(shape).copy()
            elif obj.get("__polars__"):
                # Reconstruct polars DataFrame
                return pl.DataFrame(obj["data"])
            elif obj.get("__polars_series__"):
                # Reconstruct polars Series
                return pl.Series(obj["name"], obj["data"])
            elif obj.get("__datetime__"):
                # Reconstruct datetime
                return datetime.fromisoformat(obj["iso"])
            elif obj.get("__feature_vector__"):
                # Reconstruct FeatureVector
                return FeatureVector.model_validate(obj["data"])
            elif obj.get("__pydantic__"):
                # Generic Pydantic reconstruction - only for known safe types
                # For security, we only reconstruct FeatureVector here
                class_name = obj.get("class")
                if class_name == "FeatureVector":
                    return FeatureVector.model_validate(obj["data"])
                else:
                    logger.warning(f"Unknown Pydantic class in cache: {class_name}")
                    return obj["data"]  # Return raw data as fallback

        return obj
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Failed to deserialize cached data: {e}")
        return None


class FeatureDefinition:
    """Definition of a feature including computation logic."""

    def __init__(
        self,
        name: str,
        category: str,
        compute_fn: Callable[[pl.DataFrame], pl.Series],
        dependencies: list[str] | None = None,
        description: str = "",
        version: str = "1.0",
    ):
        """
        Initialize a feature definition.

        Args:
            name: Feature name following convention {category}_{indicator}_{window}_{aggregation}
            category: Feature category (tech, stat, micro, cross)
            compute_fn: Function to compute the feature
            dependencies: List of input column dependencies
            description: Human-readable description
            version: Feature version for tracking changes
        """
        self.name = name
        self.category = category
        self.compute_fn = compute_fn
        self.dependencies = dependencies or ["close"]
        self.description = description
        self.version = version

    def compute(self, df: pl.DataFrame) -> pl.Series:
        """Compute the feature from input data."""
        return self.compute_fn(df)


class FeatureRegistry:
    """Registry of available features and their definitions."""

    def __init__(self):
        self._features: dict[str, FeatureDefinition] = {}
        self._categories: dict[str, list[str]] = {}

    def register(self, feature: FeatureDefinition) -> None:
        """Register a feature definition."""
        self._features[feature.name] = feature

        if feature.category not in self._categories:
            self._categories[feature.category] = []
        self._categories[feature.category].append(feature.name)

    def get(self, name: str) -> FeatureDefinition | None:
        """Get a feature definition by name."""
        return self._features.get(name)

    def get_by_category(self, category: str) -> list[FeatureDefinition]:
        """Get all features in a category."""
        names = self._categories.get(category, [])
        return [self._features[n] for n in names]

    def list_features(self) -> list[str]:
        """List all registered feature names."""
        return list(self._features.keys())

    def list_categories(self) -> list[str]:
        """List all feature categories."""
        return list(self._categories.keys())


class RedisCache:
    """
    Redis-based distributed cache for features.

    Provides fast access to frequently-used features across
    multiple processes and machines.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        prefix: str = "feature:",
        default_ttl: int = 3600,
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            prefix: Key prefix for feature store
            default_ttl: Default TTL in seconds
        """
        self.prefix = prefix
        self.default_ttl = default_ttl
        self._client = None
        self._connected = False

        # Connection config
        self._config = {
            "host": host,
            "port": port,
            "db": db,
            "password": password,
            "decode_responses": False,  # Binary for pickle
        }

    def connect(self) -> bool:
        """
        Establish connection to Redis.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            import redis
            self._client = redis.Redis(**self._config)
            self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self._config['host']}:{self._config['port']}")
            return True
        except ImportError:
            logger.warning("redis package not installed, Redis cache disabled")
            return False
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        if not self._connected or self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            self._connected = False
            return False

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Any | None:
        """
        Get value from Redis cache.

        SECURITY FIX: Uses safe JSON deserialization instead of pickle.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self.is_connected:
            return None

        try:
            data = self._client.get(self._make_key(key))
            if data:
                return _safe_deserialize(data)
        except Exception as e:
            logger.debug(f"Redis get error for {key}: {e}")

        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> bool:
        """
        Set value in Redis cache.

        SECURITY FIX: Uses safe JSON serialization instead of pickle.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds

        Returns:
            True if successful
        """
        if not self.is_connected:
            return False

        try:
            ttl = ttl_seconds or self.default_ttl
            data = _safe_serialize(value)
            self._client.setex(self._make_key(key), ttl, data)
            return True
        except Exception as e:
            logger.debug(f"Redis set error for {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.is_connected:
            return False

        try:
            self._client.delete(self._make_key(key))
            return True
        except Exception:
            return False

    def get_many(self, keys: list[str]) -> dict[str, Any]:
        """
        Get multiple values from cache.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary of key -> value for found keys
        """
        if not self.is_connected or not keys:
            return {}

        try:
            prefixed_keys = [self._make_key(k) for k in keys]
            values = self._client.mget(prefixed_keys)

            result = {}
            for key, data in zip(keys, values):
                if data:
                    try:
                        # SECURITY FIX: Use safe JSON deserialization instead of pickle
                        result[key] = _safe_deserialize(data)
                    except Exception:
                        pass
            return result
        except Exception as e:
            logger.debug(f"Redis mget error: {e}")
            return {}

    def set_many(
        self,
        items: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> int:
        """
        Set multiple values in cache.

        Args:
            items: Dictionary of key -> value
            ttl_seconds: Time-to-live in seconds

        Returns:
            Number of successful sets
        """
        if not self.is_connected or not items:
            return 0

        try:
            ttl = ttl_seconds or self.default_ttl
            pipe = self._client.pipeline()

            for key, value in items.items():
                # SECURITY FIX: Use safe JSON serialization instead of pickle
                data = _safe_serialize(value)
                pipe.setex(self._make_key(key), ttl, data)

            pipe.execute()
            return len(items)
        except Exception as e:
            logger.debug(f"Redis mset error: {e}")
            return 0

    def clear_prefix(self, pattern: str = "*") -> int:
        """
        Clear all keys matching pattern.

        Args:
            pattern: Key pattern to match

        Returns:
            Number of keys deleted
        """
        if not self.is_connected:
            return 0

        try:
            full_pattern = f"{self.prefix}{pattern}"
            keys = list(self._client.scan_iter(match=full_pattern))
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.debug(f"Redis clear error: {e}")
            return 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if not self.is_connected:
            return {}

        try:
            info = self._client.info("memory")
            keys_count = len(list(self._client.scan_iter(match=f"{self.prefix}*", count=1000)))

            return {
                "used_memory_mb": info.get("used_memory", 0) / 1e6,
                "keys_count": keys_count,
                "connected": True,
            }
        except Exception:
            return {"connected": False}


class MemoryCache:
    """In-memory LRU cache for features."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of entries
        """
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._max_size = max_size
        self._access_order: list[str] = []

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if key in self._cache:
            value, _ = self._cache[key]
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return value
        return None

    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Set value in cache."""
        # Evict if at capacity
        while len(self._cache) >= self._max_size:
            if self._access_order:
                oldest = self._access_order.pop(0)
                self._cache.pop(oldest, None)

        expiry = datetime.now(tz=None) + timedelta(seconds=ttl_seconds)
        self._cache[key] = (value, expiry)
        self._access_order.append(key)

    def invalidate(self, key: str) -> None:
        """Remove entry from cache."""
        self._cache.pop(key, None)
        if key in self._access_order:
            self._access_order.remove(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        now = datetime.now(tz=None)
        expired = [k for k, (_, exp) in self._cache.items() if exp < now]
        for key in expired:
            self.invalidate(key)
        return len(expired)


class FeatureStore:
    """
    Feature store for computing, caching, and serving features.

    Provides point-in-time correct feature retrieval to prevent
    data leakage in training and inference.

    Multi-layer caching:
    1. Memory cache (L1) - fastest, limited size
    2. Redis cache (L2) - distributed, shared across processes
    3. File cache (L3) - persistent, disk-based
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        redis_config: dict[str, Any] | None = None,
        redis_client: Any | None = None,
        memory_cache_size: int = 1000,
        default_ttl: int = 3600,
        enable_redis: bool = True,
    ):
        """
        Initialize the feature store.

        Args:
            cache_dir: Directory for file-based cache
            redis_config: Redis connection config dict (host, port, db, password)
            redis_client: Redis client for distributed caching (legacy)
            memory_cache_size: Size of in-memory cache
            default_ttl: Default time-to-live for cache entries
            enable_redis: Whether to enable Redis caching
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.memory_cache = MemoryCache(max_size=memory_cache_size)
        self.default_ttl = default_ttl
        self.registry = FeatureRegistry()

        # Initialize Redis cache
        self.redis_cache: RedisCache | None = None
        if enable_redis:
            if redis_config:
                self.redis_cache = RedisCache(
                    host=redis_config.get("host", "localhost"),
                    port=redis_config.get("port", 6379),
                    db=redis_config.get("db", 0),
                    password=redis_config.get("password"),
                    default_ttl=default_ttl,
                )
                self.redis_cache.connect()
            elif redis_client:
                # Legacy support for direct client
                self.redis_client = redis_client

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Register default features
        self._register_default_features()

    def _register_default_features(self) -> None:
        """Register commonly used features."""
        # Returns
        self.registry.register(FeatureDefinition(
            name="returns_1",
            category="stat",
            compute_fn=lambda df: df["close"].pct_change(),
            dependencies=["close"],
            description="1-bar returns",
        ))

        self.registry.register(FeatureDefinition(
            name="log_returns_1",
            category="stat",
            compute_fn=lambda df: (df["close"] / df["close"].shift(1)).log(),
            dependencies=["close"],
            description="1-bar log returns",
        ))

        # Volatility
        # CRITICAL FIX: Use .shift(1) to prevent look-ahead bias
        # Rolling std is computed on PREVIOUS bars only
        self.registry.register(FeatureDefinition(
            name="stat_volatility_20_std",
            category="stat",
            compute_fn=lambda df: df["close"].pct_change().rolling_std(20).shift(1),
            dependencies=["close"],
            description="20-bar rolling volatility (lagged to prevent look-ahead bias)",
        ))

        # Simple moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            self.registry.register(FeatureDefinition(
                name=f"tech_sma_{window}_raw",
                category="tech",
                compute_fn=lambda df, w=window: df["close"].rolling_mean(w),
                dependencies=["close"],
                description=f"{window}-bar simple moving average",
            ))

        # RSI
        self.registry.register(FeatureDefinition(
            name="tech_rsi_14_raw",
            category="tech",
            compute_fn=self._compute_rsi,
            dependencies=["close"],
            description="14-bar RSI",
        ))

        # Volume features
        self.registry.register(FeatureDefinition(
            name="tech_volume_sma_20",
            category="tech",
            compute_fn=lambda df: df["volume"].rolling_mean(20),
            dependencies=["volume"],
            description="20-bar volume moving average",
        ))

        # CRITICAL FIX: Use .shift(1) on rolling_mean to prevent look-ahead bias
        # Current volume is compared to average of PREVIOUS 20 bars
        self.registry.register(FeatureDefinition(
            name="tech_volume_ratio_20",
            category="tech",
            compute_fn=lambda df: df["volume"] / df["volume"].rolling_mean(20).shift(1),
            dependencies=["volume"],
            description="Volume ratio to 20-bar average (lagged to prevent look-ahead bias)",
        ))

    def _compute_rsi(self, df: pl.DataFrame, window: int = 14) -> pl.Series:
        """Compute RSI indicator."""
        delta = df["close"].diff()

        gain = delta.map_elements(lambda x: max(x, 0) if x is not None else 0, return_dtype=pl.Float64)
        loss = delta.map_elements(lambda x: abs(min(x, 0)) if x is not None else 0, return_dtype=pl.Float64)

        avg_gain = gain.rolling_mean(window)
        avg_loss = loss.rolling_mean(window)

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def compute_features(
        self,
        df: pl.DataFrame,
        feature_names: list[str],
        symbol: str | None = None,
    ) -> pl.DataFrame:
        """
        Compute specified features for a DataFrame.

        Args:
            df: Input DataFrame with OHLCV data
            feature_names: List of feature names to compute
            symbol: Symbol for caching

        Returns:
            DataFrame with computed features added
        """
        for name in feature_names:
            feature_def = self.registry.get(name)
            if feature_def is None:
                logger.warning(f"Feature not found: {name}")
                continue

            try:
                feature_values = feature_def.compute(df)
                df = df.with_columns(feature_values.alias(name))
            except Exception as e:
                logger.error(f"Error computing feature {name}: {e}")
                # Add null column
                df = df.with_columns(pl.lit(None).alias(name))

        return df

    def get_features(
        self,
        symbol: str,
        timestamp: datetime,
        feature_names: list[str],
        lookback_bars: int = 1,
        allow_future: bool = False,
    ) -> FeatureVector | None:
        """
        Get features for a symbol at a specific point in time.

        This is the main serving interface for inference.
        Ensures point-in-time correctness to prevent data leakage.

        JPMORGAN FIX: Added strict point-in-time validation to prevent
        querying future timestamps which would cause data leakage.

        Uses multi-layer caching:
        1. L1 Memory cache (fastest)
        2. L2 Redis cache (distributed)
        3. Compute on miss

        Args:
            symbol: Ticker symbol
            timestamp: Point in time for feature retrieval
            feature_names: Features to retrieve
            lookback_bars: Number of bars to look back
            allow_future: If True, bypass future check (FOR BACKTESTING ONLY)

        Returns:
            FeatureVector with requested features

        Raises:
            ValueError: If timestamp is in the future and allow_future=False
        """
        # JPMORGAN FIX: Point-in-time validation to prevent data leakage
        # This is CRITICAL for preventing look-ahead bias in production
        current_time = datetime.now(timezone.utc)

        # Make timestamp timezone-aware if needed for comparison
        ts_for_comparison = timestamp
        if timestamp.tzinfo is None:
            ts_for_comparison = timestamp.replace(tzinfo=timezone.utc)

        if not allow_future and ts_for_comparison > current_time:
            raise ValueError(
                f"POINT-IN-TIME VIOLATION: Cannot query future timestamp. "
                f"Requested: {timestamp.isoformat()}, Current: {current_time.isoformat()}. "
                f"This would cause data leakage. Use allow_future=True for backtesting."
            )

        # Warn if timestamp is very far in the past (potential stale data)
        time_diff = current_time - ts_for_comparison
        if time_diff > timedelta(days=1):
            logger.warning(
                f"Querying features for {symbol} at {timestamp.isoformat()} "
                f"which is {time_diff.days} days in the past"
            )

        cache_key = self._make_cache_key(symbol, timestamp, feature_names)

        # L1: Check memory cache
        cached = self.memory_cache.get(cache_key)
        if cached is not None:
            logger.debug(f"L1 cache hit for {symbol}")
            return cached

        # L2: Check Redis cache
        if self.redis_cache and self.redis_cache.is_connected:
            redis_cached = self.redis_cache.get(cache_key)
            if redis_cached is not None:
                logger.debug(f"L2 Redis cache hit for {symbol}")
                # Promote to L1
                self.memory_cache.set(cache_key, redis_cached)
                return redis_cached

        # Legacy Redis client support
        elif hasattr(self, 'redis_client') and self.redis_client:
            try:
                redis_cached = self.redis_client.get(cache_key)
                if redis_cached:
                    # SECURITY FIX: Use safe JSON deserialization instead of pickle
                    feature_vector = _safe_deserialize(redis_cached)
                    if feature_vector:
                        self.memory_cache.set(cache_key, feature_vector)
                        return feature_vector
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")

        # Cache miss - would need to be computed
        logger.debug(f"Cache miss for {symbol} at {timestamp}")
        return None

    def cache_features(
        self,
        symbol: str,
        timestamp: datetime,
        feature_names: list[str],
        feature_vector: FeatureVector,
    ) -> None:
        """
        Cache computed features in all cache layers.

        Args:
            symbol: Ticker symbol
            timestamp: Point in time
            feature_names: Feature names
            feature_vector: Computed feature vector
        """
        cache_key = self._make_cache_key(symbol, timestamp, feature_names)

        # L1: Memory cache
        self.memory_cache.set(cache_key, feature_vector, self.default_ttl)

        # L2: Redis cache
        if self.redis_cache and self.redis_cache.is_connected:
            self.redis_cache.set(cache_key, feature_vector, self.default_ttl)

    def get_features_batch(
        self,
        df: pl.DataFrame,
        feature_names: list[str],
        symbol: str,
    ) -> list[FeatureVector]:
        """
        Get features for all rows in a DataFrame.

        Used for batch retrieval during training.

        Args:
            df: DataFrame with OHLCV data
            feature_names: Features to compute
            symbol: Symbol name

        Returns:
            List of FeatureVector objects
        """
        # Compute features
        df_with_features = self.compute_features(df, feature_names, symbol)

        # Convert to FeatureVector objects
        vectors = []
        for row in df_with_features.iter_rows(named=True):
            features = {name: row.get(name, 0.0) for name in feature_names}
            vector = FeatureVector(
                symbol=symbol,
                timestamp=row["timestamp"],
                features=features,
                feature_names=feature_names,
            )
            vectors.append(vector)

        return vectors

    def save_features(
        self,
        symbol: str,
        df: pl.DataFrame,
        feature_names: list[str],
    ) -> None:
        """
        Save computed features to persistent storage.

        Args:
            symbol: Ticker symbol
            df: DataFrame with features
            feature_names: List of feature column names
        """
        if self.cache_dir is None:
            return

        # Select only timestamp and feature columns
        columns_to_save = ["timestamp"] + [f for f in feature_names if f in df.columns]
        df_features = df.select(columns_to_save)

        # Save to Parquet
        file_path = self.cache_dir / f"{symbol}_features.parquet"
        df_features.write_parquet(file_path, compression="zstd")
        logger.info(f"Saved {len(df_features)} feature rows for {symbol}")

    def load_features(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pl.DataFrame | None:
        """
        Load features from persistent storage.

        Args:
            symbol: Ticker symbol
            start_date: Start date filter
            end_date: End date filter

        Returns:
            DataFrame with features or None if not found
        """
        if self.cache_dir is None:
            return None

        file_path = self.cache_dir / f"{symbol}_features.parquet"
        if not file_path.exists():
            return None

        lf = pl.scan_parquet(file_path)

        if start_date:
            lf = lf.filter(pl.col("timestamp") >= start_date)
        if end_date:
            lf = lf.filter(pl.col("timestamp") <= end_date)

        return lf.collect()

    def cache_feature_vector(
        self,
        symbol: str,
        timestamp: datetime,
        feature_vector: FeatureVector,
        ttl: int | None = None,
    ) -> None:
        """
        Cache a feature vector for quick retrieval.

        Args:
            symbol: Ticker symbol
            timestamp: Timestamp of the features
            feature_vector: Feature vector to cache
            ttl: Time-to-live in seconds
        """
        ttl = ttl or self.default_ttl
        cache_key = self._make_cache_key(symbol, timestamp, list(feature_vector.features.keys()))

        # Memory cache
        self.memory_cache.set(cache_key, feature_vector, ttl)

        # Redis cache (legacy client)
        if hasattr(self, 'redis_client') and self.redis_client:
            try:
                # SECURITY FIX: Use safe JSON serialization instead of pickle
                self.redis_client.setex(
                    cache_key,
                    ttl,
                    _safe_serialize(feature_vector),
                )
            except Exception as e:
                logger.warning(f"Redis cache write error: {e}")
        # Modern Redis cache
        elif self.redis_cache and self.redis_cache.is_connected:
            self.redis_cache.set(cache_key, feature_vector, ttl)

    def invalidate_cache(self, symbol: str | None = None) -> None:
        """
        Invalidate cached features.

        Args:
            symbol: Symbol to invalidate (None for all)
        """
        if symbol is None:
            self.memory_cache.clear()
            if self.redis_client:
                try:
                    # Use scan to find and delete keys
                    cursor = 0
                    while True:
                        cursor, keys = self.redis_client.scan(cursor, match="feature:*")
                        if keys:
                            self.redis_client.delete(*keys)
                        if cursor == 0:
                            break
                except Exception as e:
                    logger.warning(f"Redis cache clear error: {e}")
        else:
            # Invalidate specific symbol (would need to track keys)
            logger.info(f"Invalidating cache for {symbol}")

    def _make_cache_key(
        self,
        symbol: str,
        timestamp: datetime,
        feature_names: list[str],
    ) -> str:
        """Generate cache key for feature lookup."""
        features_str = ",".join(sorted(feature_names))
        ts_str = timestamp.isoformat()
        key_data = f"{symbol}:{ts_str}:{features_str}"
        return f"feature:{hashlib.md5(key_data.encode()).hexdigest()}"

    def get_feature_metadata(self, feature_name: str) -> dict[str, Any] | None:
        """Get metadata for a feature."""
        feature_def = self.registry.get(feature_name)
        if feature_def is None:
            return None

        return {
            "name": feature_def.name,
            "category": feature_def.category,
            "dependencies": feature_def.dependencies,
            "description": feature_def.description,
            "version": feature_def.version,
        }

    def list_features(self, category: str | None = None) -> list[str]:
        """List available features, optionally filtered by category."""
        if category:
            features = self.registry.get_by_category(category)
            return [f.name for f in features]
        return self.registry.list_features()


class StreamingFeatureUpdater:
    """
    Handles incremental feature updates for streaming data.

    Efficiently updates features when new bars arrive without
    recomputing the entire history.
    """

    def __init__(self, feature_store: FeatureStore):
        """
        Initialize the streaming updater.

        Args:
            feature_store: FeatureStore instance to update
        """
        self.feature_store = feature_store
        self._buffers: dict[str, pl.DataFrame] = {}
        self._buffer_size = 500  # Keep last N bars in buffer

    def update(
        self,
        symbol: str,
        new_bar: dict[str, Any],
        feature_names: list[str],
    ) -> FeatureVector:
        """
        Update features with a new bar.

        Args:
            symbol: Ticker symbol
            new_bar: New OHLCV bar data
            feature_names: Features to compute

        Returns:
            Updated feature vector for the new bar
        """
        # Get or create buffer
        if symbol not in self._buffers:
            self._buffers[symbol] = pl.DataFrame(schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            })

        # Add new bar to buffer
        new_df = pl.DataFrame([new_bar])
        self._buffers[symbol] = pl.concat([self._buffers[symbol], new_df])

        # Trim buffer
        if len(self._buffers[symbol]) > self._buffer_size:
            self._buffers[symbol] = self._buffers[symbol].tail(self._buffer_size)

        # Compute features on buffer
        df_with_features = self.feature_store.compute_features(
            self._buffers[symbol],
            feature_names,
            symbol,
        )

        # Get last row as feature vector
        last_row = df_with_features.tail(1).to_dicts()[0]
        features = {name: last_row.get(name, 0.0) for name in feature_names}

        feature_vector = FeatureVector(
            symbol=symbol,
            timestamp=last_row["timestamp"],
            features=features,
            feature_names=feature_names,
        )

        # Cache the result
        self.feature_store.cache_feature_vector(
            symbol,
            last_row["timestamp"],
            feature_vector,
        )

        return feature_vector

    def clear_buffer(self, symbol: str | None = None) -> None:
        """Clear feature buffers."""
        if symbol:
            self._buffers.pop(symbol, None)
        else:
            self._buffers.clear()
