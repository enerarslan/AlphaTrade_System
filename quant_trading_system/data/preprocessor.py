"""
Data preprocessing module for cleaning, normalizing, and aligning OHLCV data.

Implements the preprocessing pipeline:
1. Cleaning - Remove invalid data, handle outliers
2. Adjustment - Corporate action adjustments
3. Normalization - Multiple scaling schemes
4. Alignment - Common timeline alignment
5. Validation - Final quality checks
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any

import numpy as np
import polars as pl

from quant_trading_system.core.exceptions import DataValidationError

logger = logging.getLogger(__name__)


class NormalizationMethod(str, Enum):
    """Available normalization methods."""

    ZSCORE = "zscore"  # Z-score normalization (mean=0, std=1)
    MINMAX = "minmax"  # Min-max scaling [0, 1]
    ROBUST = "robust"  # Robust scaling (median, IQR)
    LOG_RETURN = "log_return"  # Log returns
    PERCENT_CHANGE = "percent_change"  # Percent change
    QUANTILE = "quantile"  # Quantile transformation


class BarStatus(str, Enum):
    """JPMORGAN FIX: Bar data quality status for institutional trading."""

    REAL = "real"        # Actual market data
    IMPUTED = "imputed"  # Interpolated/estimated data
    MISSING = "missing"  # No data available
    STALE = "stale"      # Data older than expected
    SUSPICIOUS = "suspicious"  # Flagged for quality issues


class DataQualityScore:
    """
    JPMORGAN-LEVEL: Data quality scoring for institutional trading.

    Tracks quality metrics for making decisions about data usability.
    """

    def __init__(
        self,
        completeness: float = 1.0,  # Fraction of fields present
        timeliness: float = 1.0,    # Within latency SLA (0-1)
        accuracy: float = 1.0,      # Matches external sources
        consistency: float = 1.0,   # Consistent with adjacent bars
    ):
        self.completeness = completeness
        self.timeliness = timeliness
        self.accuracy = accuracy
        self.consistency = consistency

    @property
    def overall(self) -> float:
        """Weighted overall quality score."""
        weights = {
            "completeness": 0.3,
            "timeliness": 0.25,
            "accuracy": 0.25,
            "consistency": 0.2,
        }
        return (
            self.completeness * weights["completeness"]
            + self.timeliness * weights["timeliness"]
            + self.accuracy * weights["accuracy"]
            + self.consistency * weights["consistency"]
        )

    def can_use_for_trading(self, threshold: float = 0.95) -> bool:
        """Check if data quality is sufficient for live trading."""
        return self.overall >= threshold

    def can_use_for_training(self, threshold: float = 0.90) -> bool:
        """Check if data quality is sufficient for model training."""
        return self.overall >= threshold

    def to_dict(self) -> dict[str, float]:
        return {
            "completeness": self.completeness,
            "timeliness": self.timeliness,
            "accuracy": self.accuracy,
            "consistency": self.consistency,
            "overall": self.overall,
        }


class DataPreprocessor:
    """
    Data preprocessing pipeline for OHLCV data.

    Handles cleaning, adjustment, normalization, alignment,
    and validation of market data.
    """

    # Configuration defaults
    DEFAULT_OUTLIER_THRESHOLD = 5.0  # Standard deviations
    DEFAULT_MAX_GAP_BARS = 3  # Bars to interpolate
    DEFAULT_FORWARD_FILL_LIMIT = 10  # Max bars to forward fill
    DEFAULT_ZSCORE_WINDOW = 60  # Rolling window for z-score

    def __init__(
        self,
        outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
        max_interpolate_gap: int = DEFAULT_MAX_GAP_BARS,
        forward_fill_limit: int = DEFAULT_FORWARD_FILL_LIMIT,
        zscore_window: int = DEFAULT_ZSCORE_WINDOW,
    ):
        """
        Initialize the preprocessor.

        Args:
            outlier_threshold: Threshold in std devs for outlier detection
            max_interpolate_gap: Maximum gap to interpolate
            forward_fill_limit: Maximum bars to forward fill
            zscore_window: Rolling window for z-score normalization
        """
        self.outlier_threshold = outlier_threshold
        self.max_interpolate_gap = max_interpolate_gap
        self.forward_fill_limit = forward_fill_limit
        self.zscore_window = zscore_window

    def preprocess(
        self,
        df: pl.DataFrame,
        steps: list[str] | None = None,
        symbol: str | None = None,
    ) -> pl.DataFrame:
        """
        Run the full preprocessing pipeline.

        Args:
            df: Input DataFrame with OHLCV data
            steps: List of steps to run (default: all)
            symbol: Symbol name for logging

        Returns:
            Preprocessed DataFrame
        """
        symbol = symbol or "UNKNOWN"
        all_steps = ["clean", "adjust", "validate"]
        steps = steps or all_steps

        logger.info(f"Preprocessing {symbol}: {len(df)} bars, steps={steps}")

        if "clean" in steps:
            df = self.clean_data(df, symbol)

        if "adjust" in steps:
            df = self.adjust_for_splits(df)

        if "validate" in steps:
            df = self.validate_preprocessed(df, symbol)

        logger.info(f"Preprocessing complete for {symbol}: {len(df)} bars")
        return df

    def clean_data(self, df: pl.DataFrame, symbol: str = "UNKNOWN") -> pl.DataFrame:
        """
        Clean data by removing invalid entries and handling outliers.

        Steps:
        1. Remove rows with negative prices
        2. Remove rows with invalid OHLC relationships
        3. Handle outliers (> threshold std moves)
        4. Interpolate small gaps
        5. Forward fill larger gaps

        Args:
            df: Input DataFrame
            symbol: Symbol for logging

        Returns:
            Cleaned DataFrame
        """
        initial_count = len(df)

        # Ensure timestamp column is datetime
        if "timestamp" in df.columns:
            if df["timestamp"].dtype != pl.Datetime:
                df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))

        # Sort by timestamp
        df = df.sort("timestamp")

        # Remove duplicates
        df = df.unique(subset=["timestamp"], keep="last")

        # Remove negative prices
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                df = df.filter(pl.col(col) >= 0)

        # Remove invalid OHLC relationships
        df = df.filter(
            (pl.col("low") <= pl.col("high")) &
            (pl.col("low") <= pl.col("open")) &
            (pl.col("low") <= pl.col("close")) &
            (pl.col("high") >= pl.col("open")) &
            (pl.col("high") >= pl.col("close"))
        )

        # Remove negative volume
        if "volume" in df.columns:
            df = df.filter(pl.col("volume") >= 0)

        # Handle outliers in returns
        df = self._handle_outliers(df)

        # Handle missing data
        df = self._handle_missing_data(df)

        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.info(f"Cleaned {symbol}: removed {removed_count} invalid rows")

        return df

    def _handle_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Handle outliers using rolling z-score.

        Replaces extreme price moves with interpolated values.
        """
        if len(df) < self.zscore_window:
            return df

        # Calculate returns
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("return")
        )

        # Calculate rolling z-score of returns
        # CRITICAL FIX: Use .shift(1) to prevent look-ahead bias
        # The rolling statistics are computed on PREVIOUS bars only
        df = df.with_columns([
            pl.col("return").rolling_mean(self.zscore_window).shift(1).alias("_rolling_mean"),
            pl.col("return").rolling_std(self.zscore_window).shift(1).alias("_rolling_std"),
        ])
        df = df.with_columns(
            (
                (pl.col("return") - pl.col("_rolling_mean"))
                / pl.col("_rolling_std")
            ).alias("return_zscore")
        )
        df = df.drop(["_rolling_mean", "_rolling_std"])

        # Flag outliers
        df = df.with_columns(
            (pl.col("return_zscore").abs() > self.outlier_threshold).alias("is_outlier")
        )

        # Count outliers
        outlier_count = df.filter(pl.col("is_outlier") == True).height
        if outlier_count > 0:
            logger.debug(f"Found {outlier_count} outlier bars")

        # For outliers, interpolate prices
        df = df.with_columns(
            pl.when(pl.col("is_outlier"))
            .then(None)
            .otherwise(pl.col("close"))
            .alias("close_clean")
        )

        # Interpolate
        df = df.with_columns(
            pl.col("close_clean").interpolate().alias("close")
        )

        # Clean up temporary columns
        df = df.drop(["return", "return_zscore", "is_outlier", "close_clean"])

        return df

    def _handle_missing_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        JPMORGAN FIX: Handle missing data with proper status tracking.

        Instead of blindly forward-filling (which creates synthetic data that
        corrupts technical indicators), we:
        1. Track bar status (real, imputed, missing)
        2. Only interpolate very short gaps (1-2 bars)
        3. Flag but don't fill larger gaps - let downstream systems decide
        4. Provide quality scores for decision-making

        This prevents the creation of artificial consecutive bars that can
        cause false signals in trading strategies.
        """
        # Initialize bar_status column if not present
        if "bar_status" not in df.columns:
            df = df.with_columns(pl.lit(BarStatus.REAL.value).alias("bar_status"))

        # Initialize data_quality_score column
        if "data_quality_score" not in df.columns:
            df = df.with_columns(pl.lit(1.0).alias("data_quality_score"))

        price_cols = ["open", "high", "low", "close"]

        # Step 1: Identify missing bars and their gap lengths
        for col in price_cols:
            if col in df.columns:
                # Mark where nulls exist
                df = df.with_columns(
                    pl.col(col).is_null().alias(f"_is_null_{col}")
                )

        # Create combined null indicator
        null_indicators = [f"_is_null_{col}" for col in price_cols if f"_is_null_{col}" in df.columns]
        if null_indicators:
            df = df.with_columns(
                pl.any_horizontal(null_indicators).alias("_any_null")
            )

            # Calculate gap lengths using cumulative sum of non-null runs
            df = df.with_columns(
                (~pl.col("_any_null")).cum_sum().alias("_null_group")
            )

            # Count gap sizes
            df = df.with_columns(
                pl.col("_any_null").sum().over("_null_group").alias("_gap_size")
            )

            # Update bar status based on gap size
            df = df.with_columns(
                pl.when(~pl.col("_any_null"))
                .then(pl.lit(BarStatus.REAL.value))
                .when(pl.col("_gap_size") <= 2)  # Only interpolate 1-2 bar gaps
                .then(pl.lit(BarStatus.IMPUTED.value))
                .otherwise(pl.lit(BarStatus.MISSING.value))
                .alias("bar_status")
            )

            # Update quality score based on status
            df = df.with_columns(
                pl.when(pl.col("bar_status") == BarStatus.REAL.value)
                .then(pl.lit(1.0))
                .when(pl.col("bar_status") == BarStatus.IMPUTED.value)
                .then(pl.lit(0.85))  # Imputed data has reduced quality
                .otherwise(pl.lit(0.0))  # Missing data has zero quality
                .alias("data_quality_score")
            )

            # Step 2: Only interpolate short gaps (1-2 bars) - never forward fill!
            for col in price_cols:
                if col in df.columns:
                    # Interpolate only for IMPUTED status (short gaps)
                    df = df.with_columns(
                        pl.when(
                            (pl.col("bar_status") == BarStatus.IMPUTED.value)
                            & pl.col(col).is_null()
                        )
                        .then(pl.col(col).interpolate())
                        .otherwise(pl.col(col))
                        .alias(col)
                    )

            # Clean up temporary columns
            cleanup_cols = (
                null_indicators + ["_any_null", "_null_group", "_gap_size"]
            )
            df = df.drop([c for c in cleanup_cols if c in df.columns])

        # Step 3: Handle volume - use 0 for missing but flag it
        if "volume" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("volume").is_null())
                .then(pl.lit(0))
                .otherwise(pl.col("volume"))
                .alias("volume")
            )

        # Log summary of data quality
        real_count = df.filter(pl.col("bar_status") == BarStatus.REAL.value).height
        imputed_count = df.filter(pl.col("bar_status") == BarStatus.IMPUTED.value).height
        missing_count = df.filter(pl.col("bar_status") == BarStatus.MISSING.value).height

        if imputed_count > 0 or missing_count > 0:
            logger.info(
                f"Data quality: {real_count} real, {imputed_count} imputed, "
                f"{missing_count} missing bars"
            )

        return df

    def adjust_for_splits(
        self,
        df: pl.DataFrame,
        split_factor: float | None = None,
        split_date: datetime | None = None,
    ) -> pl.DataFrame:
        """
        Adjust prices for stock splits (backward adjustment).

        Args:
            df: Input DataFrame
            split_factor: Split ratio (e.g., 4.0 for 4:1 split)
            split_date: Date of the split

        Returns:
            Split-adjusted DataFrame
        """
        if split_factor is None or split_date is None:
            return df

        # Backward adjustment: multiply pre-split prices by split factor
        df = df.with_columns([
            pl.when(pl.col("timestamp") < split_date)
            .then(pl.col("open") * split_factor)
            .otherwise(pl.col("open"))
            .alias("open"),

            pl.when(pl.col("timestamp") < split_date)
            .then(pl.col("high") * split_factor)
            .otherwise(pl.col("high"))
            .alias("high"),

            pl.when(pl.col("timestamp") < split_date)
            .then(pl.col("low") * split_factor)
            .otherwise(pl.col("low"))
            .alias("low"),

            pl.when(pl.col("timestamp") < split_date)
            .then(pl.col("close") * split_factor)
            .otherwise(pl.col("close"))
            .alias("close"),

            # Adjust volume in opposite direction
            pl.when(pl.col("timestamp") < split_date)
            .then((pl.col("volume") / split_factor).cast(pl.Int64))
            .otherwise(pl.col("volume"))
            .alias("volume"),
        ])

        return df

    def normalize(
        self,
        df: pl.DataFrame,
        columns: list[str],
        method: NormalizationMethod = NormalizationMethod.ZSCORE,
        window: int | None = None,
    ) -> pl.DataFrame:
        """
        Normalize specified columns using the chosen method.

        Args:
            df: Input DataFrame
            columns: Columns to normalize
            method: Normalization method
            window: Rolling window size (for rolling normalization)

        Returns:
            DataFrame with normalized columns
        """
        window = window or self.zscore_window

        for col in columns:
            if col not in df.columns:
                continue

            normalized_col = f"{col}_norm"

            if method == NormalizationMethod.ZSCORE:
                df = self._zscore_normalize(df, col, normalized_col, window)
            elif method == NormalizationMethod.MINMAX:
                df = self._minmax_normalize(df, col, normalized_col, window)
            elif method == NormalizationMethod.ROBUST:
                df = self._robust_normalize(df, col, normalized_col, window)
            elif method == NormalizationMethod.LOG_RETURN:
                df = self._log_return(df, col, normalized_col)
            elif method == NormalizationMethod.PERCENT_CHANGE:
                df = self._percent_change(df, col, normalized_col)
            elif method == NormalizationMethod.QUANTILE:
                df = self._quantile_transform(df, col, normalized_col, window)

        return df

    def _zscore_normalize(
        self,
        df: pl.DataFrame,
        col: str,
        output_col: str,
        window: int,
    ) -> pl.DataFrame:
        """Z-score normalization: (x - mean) / std

        CRITICAL: Uses .shift(1) to prevent look-ahead bias.
        Rolling statistics are computed on PREVIOUS bars only.
        """
        return df.with_columns(
            (
                (pl.col(col) - pl.col(col).rolling_mean(window).shift(1))
                / pl.col(col).rolling_std(window).shift(1)
            ).alias(output_col)
        )

    def _minmax_normalize(
        self,
        df: pl.DataFrame,
        col: str,
        output_col: str,
        window: int,
    ) -> pl.DataFrame:
        """Min-max normalization to [0, 1] range

        CRITICAL: Uses .shift(1) to prevent look-ahead bias.
        Rolling min/max are computed on PREVIOUS bars only.
        """
        rolling_min = pl.col(col).rolling_min(window).shift(1)
        rolling_max = pl.col(col).rolling_max(window).shift(1)
        return df.with_columns(
            (
                (pl.col(col) - rolling_min)
                / (rolling_max - rolling_min)
            ).alias(output_col)
        )

    def _robust_normalize(
        self,
        df: pl.DataFrame,
        col: str,
        output_col: str,
        window: int,
    ) -> pl.DataFrame:
        """Robust normalization using median and IQR

        CRITICAL: Uses .shift(1) to prevent look-ahead bias.
        Rolling median/IQR are computed on PREVIOUS bars only.
        """
        rolling_median = pl.col(col).rolling_median(window).shift(1)
        rolling_q75 = pl.col(col).rolling_quantile(0.75, window=window).shift(1)
        rolling_q25 = pl.col(col).rolling_quantile(0.25, window=window).shift(1)
        return df.with_columns(
            (
                (pl.col(col) - rolling_median)
                / (rolling_q75 - rolling_q25)
            ).alias(output_col)
        )

    def _log_return(
        self,
        df: pl.DataFrame,
        col: str,
        output_col: str,
    ) -> pl.DataFrame:
        """Log returns: ln(x_t / x_{t-1})"""
        return df.with_columns(
            (pl.col(col) / pl.col(col).shift(1)).log().alias(output_col)
        )

    def _percent_change(
        self,
        df: pl.DataFrame,
        col: str,
        output_col: str,
    ) -> pl.DataFrame:
        """Percent change: (x_t - x_{t-1}) / x_{t-1}"""
        return df.with_columns(
            (pl.col(col) / pl.col(col).shift(1) - 1).alias(output_col)
        )

    def _quantile_transform(
        self,
        df: pl.DataFrame,
        col: str,
        output_col: str,
        window: int,
    ) -> pl.DataFrame:
        """Transform to uniform distribution based on rolling quantile rank

        CRITICAL: Uses window+1 and excludes current bar to prevent look-ahead bias.
        The rank is computed relative to PREVIOUS bars only.
        """
        # Use percent rank within rolling window, excluding current bar
        # by using window+1 and then computing rank of second-to-last element
        return df.with_columns(
            pl.col(col).rolling_map(
                lambda x: (np.searchsorted(np.sort(x[:-1]), x[-1])) / (len(x) - 1) if len(x) > 1 else 0.5,
                window_size=window + 1  # +1 because we exclude current bar in the function
            ).alias(output_col)
        )

    def validate_preprocessed(
        self,
        df: pl.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> pl.DataFrame:
        """
        Validate preprocessed data quality.

        Args:
            df: Preprocessed DataFrame
            symbol: Symbol for error messages

        Returns:
            Validated DataFrame

        Raises:
            DataValidationError: If critical validation fails
        """
        errors = []
        warnings = []

        # Check for NaN in critical fields
        critical_cols = ["open", "high", "low", "close"]
        for col in critical_cols:
            if col in df.columns:
                null_count = df[col].null_count()
                if null_count > 0:
                    null_pct = null_count / len(df) * 100
                    if null_pct > 5:
                        errors.append(f"{col} has {null_pct:.1f}% null values")
                    else:
                        warnings.append(f"{col} has {null_count} null values")

        # Check timestamp ordering
        if "timestamp" in df.columns:
            if not df["timestamp"].is_sorted():
                errors.append("Timestamps are not sorted")

        # Check price reasonability
        for col in critical_cols:
            if col in df.columns:
                neg_count = df.filter(pl.col(col) < 0).height
                if neg_count > 0:
                    errors.append(f"{col} has {neg_count} negative values")

        # Log warnings
        for warning in warnings:
            logger.warning(f"Validation warning for {symbol}: {warning}")

        # Raise on critical errors
        if errors:
            error_msg = f"Validation failed for {symbol}: {'; '.join(errors)}"
            raise DataValidationError(error_msg)

        return df


class DataAligner:
    """
    Aligns multiple symbol data to a common timeline.

    Handles different trading hours, market closures,
    and ensures consistent timestamps across all symbols.
    """

    # US Market hours (Eastern Time)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    PRE_MARKET_OPEN = time(4, 0)
    POST_MARKET_CLOSE = time(20, 0)

    def __init__(
        self,
        timeframe_minutes: int = 15,
        include_premarket: bool = False,
        include_postmarket: bool = False,
    ):
        """
        Initialize the data aligner.

        Args:
            timeframe_minutes: Bar timeframe in minutes
            include_premarket: Include pre-market data
            include_postmarket: Include post-market data
        """
        self.timeframe_minutes = timeframe_minutes
        self.include_premarket = include_premarket
        self.include_postmarket = include_postmarket

    def align_symbols(
        self,
        data: dict[str, pl.DataFrame],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Align multiple symbols to a common timeline.

        Args:
            data: Dictionary of symbol -> DataFrame
            start_date: Start of alignment period
            end_date: End of alignment period

        Returns:
            Dictionary of aligned DataFrames
        """
        if not data:
            return {}

        # Generate common timeline
        timeline = self._generate_timeline(data, start_date, end_date)

        aligned_data = {}
        for symbol, df in data.items():
            aligned_df = self._align_to_timeline(df, timeline, symbol)
            aligned_data[symbol] = aligned_df

        return aligned_data

    def _generate_timeline(
        self,
        data: dict[str, pl.DataFrame],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[datetime]:
        """Generate a common timeline from all symbols."""
        # Find date range across all symbols
        all_timestamps = set()
        for df in data.values():
            if "timestamp" in df.columns:
                all_timestamps.update(df["timestamp"].to_list())

        if not all_timestamps:
            return []

        min_ts = min(all_timestamps)
        max_ts = max(all_timestamps)

        if start_date:
            min_ts = max(min_ts, start_date)
        if end_date:
            max_ts = min(max_ts, end_date)

        # Generate regular timeline
        timeline = []
        current = min_ts

        while current <= max_ts:
            if self._is_trading_time(current):
                timeline.append(current)
            current += timedelta(minutes=self.timeframe_minutes)

        return timeline

    def _is_trading_time(self, ts: datetime) -> bool:
        """Check if timestamp is during trading hours."""
        # Skip weekends
        if ts.weekday() >= 5:
            return False

        ts_time = ts.time()

        # Regular hours
        if self.MARKET_OPEN <= ts_time <= self.MARKET_CLOSE:
            return True

        # Pre-market
        if self.include_premarket and self.PRE_MARKET_OPEN <= ts_time < self.MARKET_OPEN:
            return True

        # Post-market
        if self.include_postmarket and self.MARKET_CLOSE < ts_time <= self.POST_MARKET_CLOSE:
            return True

        return False

    def _align_to_timeline(
        self,
        df: pl.DataFrame,
        timeline: list[datetime],
        symbol: str,
    ) -> pl.DataFrame:
        """Align a single DataFrame to the common timeline."""
        if not timeline:
            return df

        # Create timeline DataFrame
        timeline_df = pl.DataFrame({"timestamp": timeline})

        # Left join to get all timeline points
        aligned = timeline_df.join(
            df,
            on="timestamp",
            how="left",
        )

        # Add symbol column
        aligned = aligned.with_columns(pl.lit(symbol).alias("symbol"))

        # Mark pre/post market
        # FIX: hour() returns integer, use proper time comparison
        # Pre-market: before 9:30 AM (hour < 9 OR (hour == 9 AND minute < 30))
        # Post-market: at or after 4:00 PM (hour >= 16)
        aligned = aligned.with_columns([
            (
                (pl.col("timestamp").dt.hour() < 9) |
                ((pl.col("timestamp").dt.hour() == 9) & (pl.col("timestamp").dt.minute() < 30))
            ).alias("is_premarket"),
            (pl.col("timestamp").dt.hour() >= 16).alias("is_postmarket"),
            pl.col("close").is_null().alias("is_missing"),
        ])

        # Forward fill missing prices
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in aligned.columns:
                aligned = aligned.with_columns(
                    pl.col(col).forward_fill(limit=5)
                )

        # Fill missing volume with 0
        if "volume" in aligned.columns:
            aligned = aligned.with_columns(
                pl.col("volume").fill_null(0)
            )

        return aligned

    def create_panel(
        self,
        data: dict[str, pl.DataFrame],
        column: str = "close",
    ) -> pl.DataFrame:
        """
        Create a panel DataFrame with symbols as columns.

        Args:
            data: Dictionary of aligned DataFrames
            column: Column to use for values

        Returns:
            Panel DataFrame with timestamp index and symbol columns
        """
        if not data:
            return pl.DataFrame()

        # Start with first symbol
        symbols = list(data.keys())
        first_df = data[symbols[0]][["timestamp", column]].rename({column: symbols[0]})

        panel = first_df

        # Join remaining symbols
        for symbol in symbols[1:]:
            df = data[symbol][["timestamp", column]].rename({column: symbol})
            panel = panel.join(df, on="timestamp", how="outer")

        # Sort by timestamp
        panel = panel.sort("timestamp")

        return panel


def preprocess_market_data(
    data: dict[str, pl.DataFrame],
    clean: bool = True,
    normalize: bool = False,
    align: bool = True,
    normalization_method: NormalizationMethod = NormalizationMethod.ZSCORE,
) -> dict[str, pl.DataFrame]:
    """
    Convenience function for preprocessing market data.

    Args:
        data: Dictionary of symbol -> DataFrame
        clean: Whether to clean data
        normalize: Whether to normalize prices
        align: Whether to align timestamps
        normalization_method: Method for normalization

    Returns:
        Preprocessed data dictionary
    """
    preprocessor = DataPreprocessor()
    aligner = DataAligner()

    result = {}

    # Clean each symbol
    for symbol, df in data.items():
        processed = df

        if clean:
            processed = preprocessor.clean_data(processed, symbol)

        if normalize:
            processed = preprocessor.normalize(
                processed,
                columns=["close", "volume"],
                method=normalization_method,
            )

        result[symbol] = processed

    # Align all symbols
    if align:
        result = aligner.align_symbols(result)

    return result
