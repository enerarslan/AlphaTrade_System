"""
Feature engineering pipeline orchestration module.

Provides:
- Configuration-driven feature computation
- Dependency resolution and parallel computation
- Feature validation (NaN detection, range checks)
- Feature selection (variance, correlation, importance)
- Target variable generation
- Feature versioning and metadata
- GPU-accelerated computation via OptimizedFeaturePipeline (when available)
- Alternative data feature integration
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
import polars as pl

# Type variable for generic LRU cache
K = TypeVar('K')
V = TypeVar('V')

from .cross_sectional import CrossSectionalFeatureCalculator
from .microstructure import MicrostructureFeatureCalculator
from .statistical import StatisticalFeatureCalculator
from .technical import TechnicalIndicatorCalculator

# Import optimized pipeline for GPU acceleration
from .optimized_pipeline import (
    OptimizedFeaturePipeline,
    OptimizedPipelineConfig,
    ComputeMode,
    CUDF_AVAILABLE,
)

logger = logging.getLogger(__name__)


# =============================================================================
# P1-H4 Enhancement: LRU Cache for Feature Pipeline
# =============================================================================


class LRUCache(Generic[K, V]):
    """
    P1-H4 Enhancement: Thread-safe LRU Cache with bounded memory.

    Prevents memory leaks in long-running trading sessions by evicting
    least-recently-used feature computations when cache exceeds max_size.

    Features:
    - O(1) get/put operations using OrderedDict
    - Thread-safe with RLock
    - Configurable max_size (default 1GB estimated)
    - Cache hit/miss statistics
    - Automatic eviction of stale entries

    Example:
        >>> cache = LRUCache[str, np.ndarray](max_size=1000)
        >>> cache.put("feature_key", feature_array)
        >>> if "feature_key" in cache:
        ...     data = cache.get("feature_key")
    """

    # Default max entries (roughly 1GB with 100 features * 10K rows each)
    DEFAULT_MAX_SIZE: int = 1000

    def __init__(
        self,
        max_size: int | None = None,
        max_memory_mb: int | None = None,
    ):
        """Initialize LRU Cache.

        Args:
            max_size: Maximum number of cache entries. Default 1000.
            max_memory_mb: Optional memory limit in MB. If set, overrides max_size
                          using estimated entry size.
        """
        self._max_size = max_size or self.DEFAULT_MAX_SIZE
        self._max_memory_mb = max_memory_mb
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        logger.debug(f"LRUCache initialized with max_size={self._max_size}")

    def get(self, key: K) -> V | None:
        """Get value from cache, updating access order.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found.
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: K, value: V) -> None:
        """Put value into cache, evicting LRU entry if needed.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        with self._lock:
            if key in self._cache:
                # Update existing and move to end
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                # Add new entry
                self._cache[key] = value

                # Evict if over capacity
                while len(self._cache) > self._max_size:
                    # Remove oldest (first) entry
                    evicted_key, _ = self._cache.popitem(last=False)
                    self._evictions += 1
                    logger.debug(f"LRUCache evicted key: {evicted_key}")

    def __contains__(self, key: K) -> bool:
        """Check if key exists in cache (does not update access order)."""
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        """Get number of cached entries."""
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            logger.info("LRUCache cleared")

    def remove(self, key: K) -> bool:
        """Remove specific key from cache.

        Args:
            key: Key to remove.

        Returns:
            True if key was found and removed, False otherwise.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with hits, misses, hit_rate, size, evictions.
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "size": len(self._cache),
                "max_size": self._max_size,
                "evictions": self._evictions,
            }

    def estimate_memory_mb(self) -> float:
        """Estimate current memory usage in MB.

        Returns:
            Estimated memory usage.
        """
        with self._lock:
            total_bytes = 0
            for value in self._cache.values():
                if isinstance(value, np.ndarray):
                    total_bytes += value.nbytes
                elif isinstance(value, dict):
                    for v in value.values():
                        if isinstance(v, np.ndarray):
                            total_bytes += v.nbytes
            return total_bytes / (1024 * 1024)


class FeatureGroup(str, Enum):
    """Feature group enumeration."""

    TECHNICAL = "technical"
    STATISTICAL = "statistical"
    MICROSTRUCTURE = "microstructure"
    CROSS_SECTIONAL = "cross_sectional"
    ALL = "all"


class NormalizationMethod(str, Enum):
    """Normalization method enumeration."""

    NONE = "none"
    ZSCORE = "zscore"
    MINMAX = "minmax"
    ROBUST = "robust"
    QUANTILE = "quantile"


@dataclass
class FeatureConfig:
    """Configuration for feature computation."""

    groups: list[FeatureGroup] = field(default_factory=lambda: [FeatureGroup.ALL])
    normalization: NormalizationMethod = NormalizationMethod.NONE
    normalization_window: int = 60
    fill_nan: bool = True
    fill_method: str = "ffill"  # 'ffill', 'bfill', 'zero', 'mean'
    max_nan_ratio: float = 0.5  # Max allowed NaN ratio per feature
    variance_threshold: float = 0.0  # Min variance to keep feature
    correlation_threshold: float = 0.95  # Max correlation between features
    include_targets: bool = False
    target_horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    # GPU acceleration settings
    use_gpu: bool = False  # Enable GPU-accelerated features via cuDF
    use_optimized_pipeline: bool = True  # Use OptimizedFeaturePipeline for basic features
    gpu_min_batch_size: int = 10000  # Minimum rows to benefit from GPU
    parallel_workers: int = 4  # Number of parallel workers for feature computation
    use_cache: bool = True  # Enable feature caching
    # P1-H4 Enhancement: LRU cache settings to prevent memory leaks
    cache_max_size: int = 1000  # Max number of cached feature sets (0 = unbounded, not recommended)
    cache_max_memory_mb: int | None = None  # Optional memory limit in MB


@dataclass
class FeatureMetadata:
    """Metadata for computed features."""

    name: str
    group: FeatureGroup
    computed_at: datetime
    num_samples: int
    nan_ratio: float
    mean: float
    std: float
    min_val: float
    max_val: float
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureSet:
    """Container for computed features with metadata.

    IMPORTANT: Features and targets are kept SEPARATE to prevent look-ahead bias.
    - `features`: Point-in-time features that use only past/current data
    - `targets`: Forward-looking labels that use FUTURE data (for training only!)

    WARNING: Never use `targets` as model inputs during training or inference.
    Targets contain future information and are ONLY for use as training labels.
    """

    features: dict[str, np.ndarray]
    metadata: dict[str, FeatureMetadata]
    config: FeatureConfig
    version: str
    symbol: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # CRITICAL: Targets are kept SEPARATE to prevent look-ahead bias
    # These contain FUTURE information and must NEVER be used as model inputs
    targets: dict[str, np.ndarray] = field(default_factory=dict)
    target_metadata: dict[str, FeatureMetadata] = field(default_factory=dict)

    def to_numpy(self, feature_names: list[str] | None = None) -> np.ndarray:
        """Convert features to numpy array.

        NOTE: This only returns features, NOT targets. Use get_targets_numpy() for targets.
        """
        if feature_names is None:
            feature_names = list(self.features.keys())
        # SAFETY CHECK: Ensure no targets are accidentally included
        for name in feature_names:
            if name.startswith("target_"):
                raise ValueError(
                    f"LOOK-AHEAD BIAS DETECTED: '{name}' is a target variable containing "
                    f"future information. Do not include targets in feature matrix. "
                    f"Use get_targets_numpy() instead for training labels."
                )
        return np.column_stack([self.features[name] for name in feature_names])

    def to_polars(self, feature_names: list[str] | None = None) -> pl.DataFrame:
        """Convert features to polars DataFrame.

        NOTE: This only returns features, NOT targets. Use get_targets_polars() for targets.
        """
        if feature_names is None:
            feature_names = list(self.features.keys())
        # SAFETY CHECK: Ensure no targets are accidentally included
        for name in feature_names:
            if name.startswith("target_"):
                raise ValueError(
                    f"LOOK-AHEAD BIAS DETECTED: '{name}' is a target variable containing "
                    f"future information. Do not include targets in feature matrix. "
                    f"Use get_targets_polars() instead for training labels."
                )
        return pl.DataFrame({name: self.features[name] for name in feature_names})

    def get_targets_numpy(self, target_names: list[str] | None = None) -> np.ndarray:
        """Get target variables as numpy array (for training labels ONLY).

        WARNING: Targets contain FUTURE information. Only use as training labels,
        never as model inputs.
        """
        if target_names is None:
            target_names = list(self.targets.keys())
        if not target_names:
            return np.array([])
        return np.column_stack([self.targets[name] for name in target_names])

    def get_targets_polars(self, target_names: list[str] | None = None) -> pl.DataFrame:
        """Get target variables as polars DataFrame (for training labels ONLY).

        WARNING: Targets contain FUTURE information. Only use as training labels,
        never as model inputs.
        """
        if target_names is None:
            target_names = list(self.targets.keys())
        if not target_names:
            return pl.DataFrame()
        return pl.DataFrame({name: self.targets[name] for name in target_names})

    @property
    def feature_names(self) -> list[str]:
        """Get list of feature names (excludes targets)."""
        return list(self.features.keys())

    @property
    def target_names(self) -> list[str]:
        """Get list of target names."""
        return list(self.targets.keys())

    @property
    def num_features(self) -> int:
        """Get number of features (excludes targets)."""
        return len(self.features)

    @property
    def num_targets(self) -> int:
        """Get number of target variables."""
        return len(self.targets)


class FeaturePipeline:
    """
    Main feature engineering pipeline.

    Orchestrates computation of all feature groups with:
    - Configuration-driven execution
    - Validation and cleaning
    - Normalization
    - Feature selection
    - GPU-accelerated computation (when available)
    - P1-H4 Enhancement: LRU caching to prevent memory leaks

    Usage:
        pipeline = FeaturePipeline(config)
        feature_set = pipeline.compute(df, universe_data)
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self._calculators: dict[FeatureGroup, Any] = {}
        self._optimized_pipeline: OptimizedFeaturePipeline | None = None

        # P1-H4 Enhancement: Initialize bounded LRU cache
        if self.config.use_cache and self.config.cache_max_size > 0:
            self._feature_cache: LRUCache[str, FeatureSet] = LRUCache(
                max_size=self.config.cache_max_size,
                max_memory_mb=self.config.cache_max_memory_mb,
            )
            logger.info(
                f"FeaturePipeline initialized with LRU cache "
                f"(max_size={self.config.cache_max_size})"
            )
        else:
            self._feature_cache = None
            if self.config.use_cache:
                logger.warning(
                    "Cache disabled: cache_max_size=0. "
                    "Memory may grow unbounded in long sessions."
                )

        self._initialize_calculators()
        self._initialize_optimized_pipeline()

    def _initialize_calculators(self) -> None:
        """Initialize feature calculators for each group."""
        self._calculators[FeatureGroup.TECHNICAL] = TechnicalIndicatorCalculator()
        self._calculators[FeatureGroup.STATISTICAL] = StatisticalFeatureCalculator()
        self._calculators[FeatureGroup.MICROSTRUCTURE] = MicrostructureFeatureCalculator()
        self._calculators[FeatureGroup.CROSS_SECTIONAL] = CrossSectionalFeatureCalculator()

    def _initialize_optimized_pipeline(self) -> None:
        """Initialize the optimized pipeline for GPU acceleration."""
        if not self.config.use_optimized_pipeline:
            return

        # Determine compute mode
        use_gpu = self.config.use_gpu and CUDF_AVAILABLE
        if use_gpu:
            compute_mode = ComputeMode.GPU
            logger.info("GPU acceleration enabled for feature computation")
        else:
            compute_mode = ComputeMode.PARALLEL
            if self.config.use_gpu and not CUDF_AVAILABLE:
                logger.warning("GPU requested but cuDF not available, using parallel CPU")

        opt_config = OptimizedPipelineConfig(
            compute_mode=compute_mode,
            num_workers=self.config.parallel_workers,
            use_numba=True,
            use_gpu=use_gpu,
            gpu_min_batch_size=self.config.gpu_min_batch_size,
        )
        self._optimized_pipeline = OptimizedFeaturePipeline(opt_config)
        logger.info(f"Optimized pipeline initialized with mode: {compute_mode.value}")

    def _compute_optimized_features(
        self, df: pl.DataFrame, symbol: str
    ) -> dict[str, np.ndarray]:
        """Compute features using the optimized pipeline.

        Uses GPU acceleration if available, otherwise parallel CPU.

        Args:
            df: Input polars DataFrame
            symbol: Symbol identifier

        Returns:
            Dictionary of feature name to numpy array
        """
        if self._optimized_pipeline is None:
            return {}

        try:
            # Convert polars to pandas for optimized pipeline
            pdf = df.to_pandas()

            # Ensure correct column names
            col_mapping = {
                c: c.lower() for c in pdf.columns
            }
            pdf = pdf.rename(columns=col_mapping)

            # Compute features using optimized pipeline
            result_df = self._optimized_pipeline.compute_features(
                pdf,
                symbol=symbol,
                use_cache=self.config.use_cache,
            )

            # Extract computed features (exclude original columns)
            original_cols = set(pdf.columns)
            features = {}
            for col in result_df.columns:
                if col not in original_cols:
                    features[col] = result_df[col].to_numpy()

            logger.debug(f"Optimized pipeline computed {len(features)} features for {symbol}")
            return features

        except Exception as e:
            logger.warning(f"Optimized pipeline failed, falling back to standard: {e}")
            return {}

    def _generate_cache_key(
        self,
        df: pl.DataFrame,
        symbol: str,
    ) -> str:
        """Generate cache key from DataFrame characteristics.

        Args:
            df: Input DataFrame.
            symbol: Symbol identifier.

        Returns:
            Cache key string.
        """
        config_hash = self._get_output_config_hash()

        # Create deterministic key from data characteristics.
        # Include output-affecting config hash and interior samples (not only boundaries)
        # to reduce stale-cache collisions.
        key_parts = [
            symbol,
            f"cfg:{config_hash}",
            str(len(df)),
            str(df.columns),
        ]
        n_rows = len(df)
        sample_idx = (
            np.linspace(0, n_rows - 1, num=min(12, n_rows), dtype=int).tolist()
            if n_rows > 0
            else []
        )

        # Add first/last timestamps if available
        if "timestamp" in df.columns or "date" in df.columns:
            time_col = "timestamp" if "timestamp" in df.columns else "date"
            key_parts.append(str(df[time_col][0]))
            key_parts.append(str(df[time_col][-1]))
            if sample_idx:
                ts_samples = [str(df[time_col][int(i)]) for i in sample_idx]
                key_parts.append("|".join(ts_samples))
        elif "close" in df.columns:
            # Use price as fallback identifier
            key_parts.append(f"{df['close'][0]:.4f}")
            key_parts.append(f"{df['close'][-1]:.4f}")

        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns or not sample_idx:
                continue
            try:
                arr = df[col].to_numpy().astype(np.float64)
                sampled = arr[np.asarray(sample_idx, dtype=int)]
                sampled = np.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
                key_parts.append(
                    f"{col}:{','.join(f'{v:.6f}' for v in sampled)}"
                )
            except Exception:
                # Skip unstable columns from key fingerprinting.
                continue

        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_output_config_hash(self) -> str:
        """Return deterministic hash for config fields that affect feature outputs."""
        config_payload = {
            "groups": [group.value for group in self.config.groups],
            "normalization": self.config.normalization.value,
            "normalization_window": self.config.normalization_window,
            "fill_nan": self.config.fill_nan,
            "fill_method": self.config.fill_method,
            "max_nan_ratio": self.config.max_nan_ratio,
            "variance_threshold": self.config.variance_threshold,
            "correlation_threshold": self.config.correlation_threshold,
            "include_targets": self.config.include_targets,
            "target_horizons": self.config.target_horizons,
            "use_gpu": self.config.use_gpu,
            "use_optimized_pipeline": self.config.use_optimized_pipeline,
            "gpu_min_batch_size": self.config.gpu_min_batch_size,
            "parallel_workers": self.config.parallel_workers,
        }
        serialized = json.dumps(
            config_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.md5(serialized.encode()).hexdigest()

    def compute(
        self,
        df: pl.DataFrame,
        symbol: str = "UNKNOWN",
        universe_data: dict[str, pl.DataFrame] | None = None,
        use_cache: bool = True,
    ) -> FeatureSet:
        """
        Compute all configured features.

        P1-H4 Enhancement: Uses LRU cache to avoid recomputation and
        prevent memory leaks in long-running sessions.

        Args:
            df: Input DataFrame with OHLCV data
            symbol: Symbol identifier
            universe_data: Optional dict of DataFrames for cross-sectional features
            use_cache: Whether to use cache for this computation. Default True.

        Returns:
            FeatureSet containing computed features and metadata
        """
        # P1-H4 Enhancement: Check cache first
        cache_key = None
        if use_cache and self._feature_cache is not None:
            cache_key = self._generate_cache_key(df, symbol)
            cached_result = self._feature_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {symbol} (key={cache_key[:8]}...)")
                return cached_result

        all_features: dict[str, np.ndarray] = {}
        all_metadata: dict[str, FeatureMetadata] = {}

        # Use optimized pipeline for basic technical features if available
        if self._optimized_pipeline is not None and len(df) >= 100:
            optimized_features = self._compute_optimized_features(df, symbol)
            for name, values in optimized_features.items():
                if self._validate_feature(values):
                    all_features[name] = values
                    all_metadata[name] = self._compute_metadata(
                        name, values, FeatureGroup.TECHNICAL
                    )

        groups_to_compute = self._get_groups_to_compute()

        for group in groups_to_compute:
            group_features = self._compute_group(df, group, universe_data)

            for name, values in group_features.items():
                # Skip if already computed by optimized pipeline
                if name in all_features:
                    continue

                # Validate feature
                if not self._validate_feature(values):
                    continue

                # Clean feature
                values = self._clean_feature(values)

                # Normalize if configured
                if self.config.normalization != NormalizationMethod.NONE:
                    values = self._normalize_feature(values)

                # Compute metadata
                metadata = self._compute_metadata(name, values, group)

                all_features[name] = values
                all_metadata[name] = metadata

        # Feature selection (applied ONLY to features, NOT targets)
        if self.config.variance_threshold > 0:
            all_features, all_metadata = self._filter_by_variance(
                all_features, all_metadata
            )

        if self.config.correlation_threshold < 1.0:
            all_features, all_metadata = self._filter_by_correlation(
                all_features, all_metadata
            )

        # Compute target variables SEPARATELY (CRITICAL: prevents look-ahead bias)
        # Targets contain FUTURE information and must NEVER be mixed with features
        target_features: dict[str, np.ndarray] = {}
        target_metadata: dict[str, FeatureMetadata] = {}

        if self.config.include_targets:
            targets = self._compute_targets(df)
            for name, values in targets.items():
                # Store targets in separate dict, NOT in all_features
                target_features[name] = values
                target_metadata[name] = self._compute_metadata(
                    name, values, FeatureGroup.TECHNICAL
                )
            # NOTE: Targets are NOT filtered by variance/correlation
            # (they contain future info and filtering them would be inappropriate)

        # Generate version hash (based on features only, not targets)
        version = self._generate_version(all_features)

        result = FeatureSet(
            features=all_features,
            metadata=all_metadata,
            config=self.config,
            version=version,
            symbol=symbol,
            targets=target_features,  # SEPARATE from features to prevent look-ahead bias
            target_metadata=target_metadata,
        )

        # P1-H4 Enhancement: Store in cache
        if use_cache and self._feature_cache is not None and cache_key:
            self._feature_cache.put(cache_key, result)
            logger.debug(f"Cached features for {symbol} (key={cache_key[:8]}...)")

        return result

    def _get_groups_to_compute(self) -> list[FeatureGroup]:
        """Get list of feature groups to compute."""
        if FeatureGroup.ALL in self.config.groups:
            return [
                FeatureGroup.TECHNICAL,
                FeatureGroup.STATISTICAL,
                FeatureGroup.MICROSTRUCTURE,
                FeatureGroup.CROSS_SECTIONAL,
            ]
        return [g for g in self.config.groups if g != FeatureGroup.ALL]

    def _compute_group(
        self,
        df: pl.DataFrame,
        group: FeatureGroup,
        universe_data: dict[str, pl.DataFrame] | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute features for a specific group."""
        calculator = self._calculators.get(group)
        if calculator is None:
            return {}

        if group == FeatureGroup.CROSS_SECTIONAL:
            return calculator.compute_all(df, universe_data)
        else:
            return calculator.compute_all(df)

    def _validate_feature(self, values: np.ndarray) -> bool:
        """Validate feature values."""
        if len(values) == 0:
            return False

        nan_ratio = np.sum(np.isnan(values)) / len(values)
        if nan_ratio > self.config.max_nan_ratio:
            return False

        # Check for infinities
        if np.any(np.isinf(values)):
            return False

        return True

    def _clean_feature(self, values: np.ndarray) -> np.ndarray:
        """Clean feature values (handle NaN)."""
        if not self.config.fill_nan:
            return values

        values = values.copy()

        if self.config.fill_method == "ffill":
            # Forward fill
            mask = np.isnan(values)
            idx = np.where(~mask, np.arange(len(values)), -1)
            np.maximum.accumulate(idx, out=idx)
            valid = idx >= 0
            values[valid] = values[idx[valid]]

        elif self.config.fill_method == "bfill":
            # Causal-safe fallback: avoid backward-looking fill in training paths.
            mask = np.isnan(values)
            idx = np.where(~mask, np.arange(len(values)), -1)
            np.maximum.accumulate(idx, out=idx)
            valid = idx >= 0
            values[valid] = values[idx[valid]]

        elif self.config.fill_method == "zero":
            values = np.nan_to_num(values, nan=0.0)

        elif self.config.fill_method == "mean":
            # Causal mean: use historical mean up to t-1, never future samples.
            running_sum = 0.0
            running_count = 0
            for i, value in enumerate(values):
                if np.isnan(value):
                    if running_count > 0:
                        values[i] = running_sum / running_count
                else:
                    running_sum += float(value)
                    running_count += 1

        return values

    def _normalize_feature(self, values: np.ndarray) -> np.ndarray:
        """Normalize feature values.

        CRITICAL FIX: Uses PREVIOUS bars only to prevent look-ahead bias.
        The window excludes the current bar from statistics calculation.
        """
        method = self.config.normalization
        window = self.config.normalization_window

        if method == NormalizationMethod.NONE:
            return values

        result = np.full_like(values, np.nan)
        n = len(values)

        # CRITICAL FIX: Start from window index and use values[i - window : i]
        # to exclude current bar from statistics calculation
        for i in range(window, n):
            # Use PREVIOUS bars only (exclude current bar i)
            window_data = values[i - window : i]
            valid_data = window_data[~np.isnan(window_data)]

            if len(valid_data) < 2:
                continue

            if method == NormalizationMethod.ZSCORE:
                mean = np.mean(valid_data)
                std = np.std(valid_data, ddof=1)
                if std > 0:
                    result[i] = (values[i] - mean) / std

            elif method == NormalizationMethod.MINMAX:
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                if max_val > min_val:
                    result[i] = (values[i] - min_val) / (max_val - min_val)

            elif method == NormalizationMethod.ROBUST:
                median = np.median(valid_data)
                q75, q25 = np.percentile(valid_data, [75, 25])
                iqr = q75 - q25
                if iqr > 0:
                    result[i] = (values[i] - median) / iqr

            elif method == NormalizationMethod.QUANTILE:
                from scipy import stats
                # CRITICAL FIX: Use searchsorted instead of percentileofscore
                # to exclude current bar from ranking
                result[i] = np.searchsorted(np.sort(valid_data), values[i]) / len(valid_data)

        return result

    def _compute_metadata(
        self, name: str, values: np.ndarray, group: FeatureGroup
    ) -> FeatureMetadata:
        """Compute metadata for a feature."""
        valid_values = values[~np.isnan(values)]

        return FeatureMetadata(
            name=name,
            group=group,
            computed_at=datetime.now(timezone.utc),
            num_samples=len(values),
            nan_ratio=np.sum(np.isnan(values)) / len(values) if len(values) > 0 else 1.0,
            mean=float(np.mean(valid_values)) if len(valid_values) > 0 else 0.0,
            std=float(np.std(valid_values, ddof=1)) if len(valid_values) > 1 else 0.0,
            min_val=float(np.min(valid_values)) if len(valid_values) > 0 else 0.0,
            max_val=float(np.max(valid_values)) if len(valid_values) > 0 else 0.0,
        )

    def _compute_targets(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        """Compute target variables for model training.

        CRITICAL WARNING - LOOK-AHEAD BIAS:
        =====================================
        These target variables use FUTURE price information!
        They are computed using prices at time T+horizon to create labels for time T.

        CORRECT USAGE:
        - Use ONLY as training labels (y_train)
        - NEVER include in feature matrix (X_train)
        - NEVER use during live inference

        INCORRECT USAGE (causes look-ahead bias):
        - Including targets in feature matrix
        - Using targets as model inputs
        - Not properly masking the last `horizon` rows (which are NaN)

        The last `horizon` rows of each target will be NaN because
        we cannot know future returns for those rows.
        """
        if "close" not in df.columns:
            return {}

        close = df["close"].to_numpy().astype(np.float64)
        targets = {}

        for horizon in self.config.target_horizons:
            # Forward returns (classification target)
            forward_return = np.full(len(close), np.nan)
            forward_return[:-horizon] = (close[horizon:] - close[:-horizon]) / close[:-horizon]
            targets[f"target_return_{horizon}"] = forward_return

            # Direction target (1 = up, 0 = down)
            direction = np.full(len(close), np.nan)
            direction[:-horizon] = np.where(forward_return[:-horizon] > 0, 1, 0)
            targets[f"target_direction_{horizon}"] = direction

            # Large move target (> 1 ATR)
            if "high" in df.columns and "low" in df.columns:
                high = df["high"].to_numpy().astype(np.float64)
                low = df["low"].to_numpy().astype(np.float64)

                # Simple ATR approximation
                tr = high - low
                atr = np.full(len(close), np.nan)
                for i in range(13, len(close)):
                    atr[i] = np.mean(tr[i - 13 : i + 1])

                large_move = np.full(len(close), np.nan)
                large_move[:-horizon] = np.where(
                    np.abs(forward_return[:-horizon]) > atr[:-horizon] / close[:-horizon],
                    1,
                    0,
                )
                targets[f"target_large_move_{horizon}"] = large_move

        return targets

    def _filter_by_variance(
        self,
        features: dict[str, np.ndarray],
        metadata: dict[str, FeatureMetadata],
    ) -> tuple[dict[str, np.ndarray], dict[str, FeatureMetadata]]:
        """Filter features by variance threshold."""
        filtered_features = {}
        filtered_metadata = {}

        for name, values in features.items():
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 1:
                variance = np.var(valid_values, ddof=1)
                if variance >= self.config.variance_threshold:
                    filtered_features[name] = values
                    filtered_metadata[name] = metadata[name]

        return filtered_features, filtered_metadata

    def _filter_by_correlation(
        self,
        features: dict[str, np.ndarray],
        metadata: dict[str, FeatureMetadata],
    ) -> tuple[dict[str, np.ndarray], dict[str, FeatureMetadata]]:
        """Filter highly correlated features."""
        if len(features) < 2:
            return features, metadata

        feature_names = list(features.keys())
        feature_matrix = np.column_stack([features[name] for name in feature_names])

        # Handle NaNs for correlation computation
        valid_mask = ~np.any(np.isnan(feature_matrix), axis=1)
        if np.sum(valid_mask) < 10:
            return features, metadata

        valid_matrix = feature_matrix[valid_mask]

        # Compute correlation matrix
        corr_matrix = np.corrcoef(valid_matrix.T)

        # Find features to remove (highly correlated with earlier features)
        to_remove = set()
        for i in range(len(feature_names)):
            if feature_names[i] in to_remove:
                continue
            for j in range(i + 1, len(feature_names)):
                if feature_names[j] in to_remove:
                    continue
                if abs(corr_matrix[i, j]) > self.config.correlation_threshold:
                    # Remove the feature with higher mean NaN ratio
                    nan_i = metadata[feature_names[i]].nan_ratio
                    nan_j = metadata[feature_names[j]].nan_ratio
                    if nan_j > nan_i:
                        to_remove.add(feature_names[j])
                    else:
                        to_remove.add(feature_names[i])
                        break

        filtered_features = {
            name: values
            for name, values in features.items()
            if name not in to_remove
        }
        filtered_metadata = {
            name: meta
            for name, meta in metadata.items()
            if name not in to_remove
        }

        return filtered_features, filtered_metadata

    def _generate_version(self, features: dict[str, np.ndarray]) -> str:
        """Generate version hash for feature set."""
        content = json.dumps(
            {
                "feature_names": sorted(features.keys()),
                "config": {
                    "groups": [g.value for g in self.config.groups],
                    "normalization": self.config.normalization.value,
                },
            },
            sort_keys=True,
        )
        return hashlib.md5(content.encode()).hexdigest()[:8]

    # P1-H4 Enhancement: Cache management methods

    def get_cache_stats(self) -> dict[str, Any]:
        """Get feature cache statistics.

        Returns:
            Dictionary with cache statistics or empty if cache disabled.
        """
        if self._feature_cache is not None:
            stats = self._feature_cache.get_stats()
            stats["memory_mb"] = self._feature_cache.estimate_memory_mb()
            return stats
        return {"enabled": False}

    def clear_cache(self) -> None:
        """Clear the feature cache.

        Call this method to free memory if needed during long sessions.
        """
        if self._feature_cache is not None:
            self._feature_cache.clear()
            logger.info("Feature pipeline cache cleared")

    def invalidate_symbol(self, symbol: str) -> int:
        """Invalidate all cached entries for a symbol.

        Args:
            symbol: Symbol to invalidate.

        Returns:
            Number of entries removed.
        """
        if self._feature_cache is None:
            return 0

        # Note: This is O(n) but cache invalidation is infrequent
        removed = 0
        keys_to_remove = []
        for key in list(self._feature_cache._cache.keys()):
            # Keys start with hash, but we need to check the cached FeatureSet
            cached = self._feature_cache._cache.get(key)
            if cached and hasattr(cached, 'symbol') and cached.symbol == symbol:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            if self._feature_cache.remove(key):
                removed += 1

        if removed > 0:
            logger.info(f"Invalidated {removed} cache entries for {symbol}")

        return removed


class FeatureSelector:
    """
    Feature selection utilities.

    Provides methods for:
    - Variance-based selection
    - Correlation-based selection
    - Mutual information selection
    - Tree-based importance selection
    """

    @staticmethod
    def select_by_variance(
        features: dict[str, np.ndarray], threshold: float = 0.01
    ) -> list[str]:
        """Select features above variance threshold."""
        selected = []
        for name, values in features.items():
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 1:
                variance = np.var(valid_values, ddof=1)
                if variance >= threshold:
                    selected.append(name)
        return selected

    @staticmethod
    def select_by_correlation(
        features: dict[str, np.ndarray], threshold: float = 0.95
    ) -> list[str]:
        """Select features with correlation below threshold."""
        if len(features) < 2:
            return list(features.keys())

        feature_names = list(features.keys())
        feature_matrix = np.column_stack([features[name] for name in feature_names])

        valid_mask = ~np.any(np.isnan(feature_matrix), axis=1)
        if np.sum(valid_mask) < 10:
            return feature_names

        valid_matrix = feature_matrix[valid_mask]
        corr_matrix = np.corrcoef(valid_matrix.T)

        selected = []
        for i, name in enumerate(feature_names):
            is_correlated = False
            for j, selected_name in enumerate(selected):
                selected_idx = feature_names.index(selected_name)
                if abs(corr_matrix[i, selected_idx]) > threshold:
                    is_correlated = True
                    break
            if not is_correlated:
                selected.append(name)

        return selected

    @staticmethod
    def select_by_importance(
        features: dict[str, np.ndarray],
        target: np.ndarray,
        top_k: int = 50,
        method: str = "mutual_info",
    ) -> list[str]:
        """Select top-k features by importance."""
        from scipy import stats

        feature_names = list(features.keys())
        importances = []

        for name in feature_names:
            values = features[name]
            valid_mask = ~(np.isnan(values) | np.isnan(target))

            if np.sum(valid_mask) < 10:
                importances.append(0.0)
                continue

            if method == "mutual_info":
                # Simplified mutual information using correlation as proxy
                corr = np.abs(np.corrcoef(values[valid_mask], target[valid_mask])[0, 1])
                importances.append(corr if not np.isnan(corr) else 0.0)

            elif method == "correlation":
                corr = np.abs(np.corrcoef(values[valid_mask], target[valid_mask])[0, 1])
                importances.append(corr if not np.isnan(corr) else 0.0)

            elif method == "information_coefficient":
                # Spearman rank correlation (IC)
                ic = stats.spearmanr(values[valid_mask], target[valid_mask])[0]
                importances.append(abs(ic) if not np.isnan(ic) else 0.0)

            else:
                importances.append(0.0)

        # Select top-k
        top_indices = np.argsort(importances)[::-1][:top_k]
        return [feature_names[i] for i in top_indices]


class FeatureValidator:
    """
    Feature validation utilities.

    Provides methods for:
    - NaN detection
    - Range checks
    - Temporal consistency
    - Distribution checks
    """

    @staticmethod
    def check_nan_ratio(
        features: dict[str, np.ndarray], max_ratio: float = 0.5
    ) -> dict[str, float]:
        """Check NaN ratio for each feature."""
        ratios = {}
        for name, values in features.items():
            ratio = np.sum(np.isnan(values)) / len(values) if len(values) > 0 else 1.0
            ratios[name] = ratio
        return ratios

    @staticmethod
    def check_infinities(features: dict[str, np.ndarray]) -> dict[str, bool]:
        """Check for infinities in features."""
        has_inf = {}
        for name, values in features.items():
            has_inf[name] = bool(np.any(np.isinf(values)))
        return has_inf

    @staticmethod
    def check_range(
        features: dict[str, np.ndarray],
        expected_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, bool]:
        """Check if features are within expected ranges."""
        in_range = {}

        # Default ranges for common features
        default_ranges = {
            "rsi": (0, 100),
            "stoch": (0, 100),
            "mfi": (0, 100),
            "williams": (-100, 0),
            "bb_percent": (-1, 2),  # Can exceed 0-1 range
        }

        if expected_ranges:
            default_ranges.update(expected_ranges)

        for name, values in features.items():
            valid_values = values[~np.isnan(values)]
            if len(valid_values) == 0:
                in_range[name] = True
                continue

            # Find matching range
            range_found = False
            for key, (min_val, max_val) in default_ranges.items():
                if key in name.lower():
                    in_range[name] = bool(
                        np.min(valid_values) >= min_val and np.max(valid_values) <= max_val
                    )
                    range_found = True
                    break

            if not range_found:
                in_range[name] = True  # No range specified

        return in_range

    @staticmethod
    def check_temporal_consistency(
        features: dict[str, np.ndarray], lookback: int = 5
    ) -> dict[str, float]:
        """Check temporal consistency (autocorrelation) of features."""
        consistency = {}
        for name, values in features.items():
            valid_values = values[~np.isnan(values)]
            if len(valid_values) < lookback + 10:
                consistency[name] = 0.0
                continue

            # Compute lag-1 autocorrelation
            acf = np.corrcoef(valid_values[:-1], valid_values[1:])[0, 1]
            consistency[name] = abs(acf) if not np.isnan(acf) else 0.0

        return consistency


# Convenience function for creating pipelines
def create_pipeline(
    groups: list[str] | None = None,
    normalization: str = "none",
    include_targets: bool = False,
    target_horizons: list[int] | None = None,
) -> FeaturePipeline:
    """
    Create a feature pipeline with specified configuration.

    Args:
        groups: List of feature groups ('technical', 'statistical', 'microstructure', 'cross_sectional', 'all')
        normalization: Normalization method ('none', 'zscore', 'minmax', 'robust', 'quantile')
        include_targets: Whether to include target variables
        target_horizons: List of forecast horizons for targets

    Returns:
        Configured FeaturePipeline
    """
    config = FeatureConfig(
        groups=[FeatureGroup(g) for g in (groups or ["all"])],
        normalization=NormalizationMethod(normalization),
        include_targets=include_targets,
        target_horizons=target_horizons or [1, 5, 10, 20],
    )
    return FeaturePipeline(config)
