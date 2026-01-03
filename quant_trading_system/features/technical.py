"""
Technical indicators module.

Implements 200+ technical indicators organized by category:
- Trend indicators (moving averages, ADX, Aroon, etc.)
- Momentum indicators (RSI, MACD, Stochastic, etc.)
- Volatility indicators (Bollinger Bands, ATR, Keltner, etc.)
- Volume indicators (OBV, MFI, VWAP, etc.)
- Price action (support/resistance, pivot points)
- Candlestick patterns (40+ patterns)
- Composite/derived indicators

This module contains 50+ core indicators. For the full 200+ indicator set,
use TechnicalIndicatorCalculator with include_extended=True, which combines
this module with technical_extended.py.
"""

from __future__ import annotations

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache, wraps
from typing import Any, Callable, TypeVar

import numpy as np
import polars as pl


# =============================================================================
# MEMOIZATION CACHE FOR INDICATOR CALCULATIONS
# =============================================================================


class IndicatorCache:
    """
    Thread-safe LRU cache for indicator calculations.

    Provides memoization to avoid recomputing indicators on the same data.
    Uses content-based hashing for cache keys.
    """

    _instance: IndicatorCache | None = None
    _lock = threading.Lock()

    def __new__(cls) -> IndicatorCache:
        """Singleton pattern for global cache."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 300.0,  # 5 minutes default TTL
    ) -> None:
        if self._initialized:
            return

        self._cache: dict[str, tuple[Any, float]] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._access_order: list[str] = []
        self._cache_lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._initialized = True

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def _make_key(
        self,
        indicator_name: str,
        df_hash: str,
        params: dict[str, Any]
    ) -> str:
        """Create cache key from indicator name, data hash, and parameters."""
        # Sort params for consistent key generation
        params_str = str(sorted(params.items()))
        key_str = f"{indicator_name}:{df_hash}:{params_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _hash_dataframe(self, df: pl.DataFrame) -> str:
        """Create hash of DataFrame contents for cache key."""
        # Use shape and sample of data for fast hashing
        shape_str = str(df.shape)
        cols_str = str(df.columns)

        # Sample first, middle, last rows for content hash
        n = len(df)
        if n > 0:
            indices = [0, n // 2, n - 1] if n > 2 else list(range(n))
            sample_data = []
            for idx in indices:
                row = df.row(idx)
                sample_data.append(str(row))
            data_str = "|".join(sample_data)
        else:
            data_str = "empty"

        hash_input = f"{shape_str}:{cols_str}:{data_str}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def get(
        self,
        indicator_name: str,
        df: pl.DataFrame,
        params: dict[str, Any]
    ) -> dict[str, np.ndarray] | None:
        """Get cached result if available and not expired."""
        df_hash = self._hash_dataframe(df)
        key = self._make_key(indicator_name, df_hash, params)

        with self._cache_lock:
            if key in self._cache:
                result, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl_seconds:
                    self._hits += 1
                    # Update access order for LRU
                    if key in self._access_order:
                        self._access_order.remove(key)
                    self._access_order.append(key)
                    return result
                else:
                    # Expired, remove from cache
                    del self._cache[key]
                    if key in self._access_order:
                        self._access_order.remove(key)

            self._misses += 1
            return None

    def set(
        self,
        indicator_name: str,
        df: pl.DataFrame,
        params: dict[str, Any],
        result: dict[str, np.ndarray]
    ) -> None:
        """Store result in cache."""
        df_hash = self._hash_dataframe(df)
        key = self._make_key(indicator_name, df_hash, params)

        with self._cache_lock:
            # Evict oldest entries if at capacity
            while len(self._cache) >= self._max_size and self._access_order:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self._cache:
                    del self._cache[oldest_key]

            self._cache[key] = (result, time.time())
            self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._cache_lock:
            self._cache.clear()
            self._access_order.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "ttl_seconds": self._ttl_seconds,
            }


def cached_indicator(method: Callable) -> Callable:
    """
    Decorator to cache indicator compute() results.

    Usage:
        class MyIndicator(TechnicalIndicator):
            @cached_indicator
            def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
                # expensive computation
                ...
    """
    @wraps(method)
    def wrapper(self: TechnicalIndicator, df: pl.DataFrame) -> dict[str, np.ndarray]:
        cache = IndicatorCache()

        # Get indicator parameters for cache key
        params = {}
        for attr in dir(self):
            if not attr.startswith('_') and attr != 'name':
                val = getattr(self, attr)
                if isinstance(val, (int, float, str, bool, list, tuple)):
                    params[attr] = val

        # Check cache
        cached_result = cache.get(self.name, df, params)
        if cached_result is not None:
            return cached_result

        # Compute and cache
        result = method(self, df)
        cache.set(self.name, df, params, result)
        return result

    return wrapper


# Global function to get cache instance
def get_indicator_cache() -> IndicatorCache:
    """Get the global indicator cache instance."""
    return IndicatorCache()


@dataclass
class IndicatorResult:
    """Result container for indicator calculations."""

    name: str
    values: np.ndarray
    params: dict[str, Any]
    metadata: dict[str, Any] | None = None


class TechnicalIndicator(ABC):
    """Abstract base class for technical indicators."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        """Compute the indicator and return named arrays."""
        pass

    def validate_input(
        self,
        df: pl.DataFrame,
        required_columns: list[str],
        min_rows: int = 1,
    ) -> None:
        """Validate DataFrame input for indicator computation.

        Args:
            df: Input DataFrame.
            required_columns: List of required column names.
            min_rows: Minimum number of rows required.

        Raises:
            ValueError: If validation fails.
            TypeError: If df is not a polars DataFrame.
        """
        # Type check
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"Expected pl.DataFrame, got {type(df).__name__}")

        # Empty check
        if df.is_empty():
            raise ValueError("DataFrame is empty")

        # Missing columns check
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Minimum rows check
        if len(df) < min_rows:
            raise ValueError(f"DataFrame has {len(df)} rows, minimum {min_rows} required")

        # Numeric type check for required columns
        for col in required_columns:
            dtype = df[col].dtype
            if dtype not in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                raise ValueError(f"Column '{col}' must be numeric, got {dtype}")


# =============================================================================
# TREND INDICATORS
# =============================================================================


class SMA(TechnicalIndicator):
    """Simple Moving Average."""

    def __init__(self, periods: list[int] | None = None):
        super().__init__("SMA")
        self.periods = periods or [5, 10, 20, 50, 100, 200]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}
        for period in self.periods:
            sma = self._rolling_mean(close, period)
            results[f"sma_{period}"] = sma
        return results

    @staticmethod
    def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling mean."""
        result = np.full(len(arr), np.nan)
        if len(arr) >= window:
            cumsum = np.cumsum(arr)
            cumsum[window:] = cumsum[window:] - cumsum[:-window]
            result[window - 1 :] = cumsum[window - 1 :] / window
        return result


class EMA(TechnicalIndicator):
    """Exponential Moving Average."""

    def __init__(self, periods: list[int] | None = None):
        super().__init__("EMA")
        self.periods = periods or [5, 10, 20, 50, 100, 200]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}
        for period in self.periods:
            ema = self._ema(close, period)
            results[f"ema_{period}"] = ema
        return results

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        """Compute exponential moving average (optimized).

        Uses pandas ewm() which has C-level optimized implementation,
        falling back to scipy.signal.lfilter if pandas unavailable.
        """
        if len(arr) < period:
            return np.full(len(arr), np.nan)

        try:
            # Pandas EWM is highly optimized (C implementation)
            import pandas as pd
            series = pd.Series(arr)
            ema = series.ewm(span=period, min_periods=period, adjust=False).mean()
            return ema.to_numpy()
        except ImportError:
            pass

        try:
            # scipy.signal.lfilter is vectorized
            from scipy.signal import lfilter
            alpha = 2.0 / (period + 1)
            # First, compute SMA for initialization
            result = np.full(len(arr), np.nan)
            result[period - 1] = np.mean(arr[:period])
            # Use lfilter for the recursive part
            b = [alpha]
            a = [1, -(1 - alpha)]
            # Apply filter starting from period
            filtered = lfilter(b, a, arr[period:], zi=[result[period - 1] * (1 - alpha)])[0]
            result[period:] = filtered
            return result
        except ImportError:
            pass

        # Pure NumPy fallback (still faster than pure Python due to array ops)
        alpha = 2.0 / (period + 1)
        decay = 1 - alpha
        result = np.full(len(arr), np.nan)
        result[period - 1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = alpha * arr[i] + decay * result[i - 1]
        return result


class WMA(TechnicalIndicator):
    """Weighted Moving Average."""

    def __init__(self, periods: list[int] | None = None):
        super().__init__("WMA")
        self.periods = periods or [10, 20, 50]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}
        for period in self.periods:
            wma = self._wma(close, period)
            results[f"wma_{period}"] = wma
        return results

    @staticmethod
    def _wma(arr: np.ndarray, period: int) -> np.ndarray:
        """Compute weighted moving average (vectorized using np.convolve).

        Uses numpy's optimized convolution for O(n) performance instead of O(n*period).
        """
        if len(arr) < period:
            return np.full(len(arr), np.nan)

        # Create linearly increasing weights [1, 2, 3, ..., period]
        weights = np.arange(1, period + 1, dtype=np.float64)
        weight_sum = weights.sum()

        # Normalize weights for convolution
        normalized_weights = weights / weight_sum

        # Use convolution for vectorized computation
        # 'valid' mode gives output only where full overlap occurs
        wma_valid = np.convolve(arr, normalized_weights[::-1], mode='valid')

        # Build result with NaN padding for initial period
        result = np.full(len(arr), np.nan)
        result[period - 1:] = wma_valid

        return result


class DEMA(TechnicalIndicator):
    """Double Exponential Moving Average."""

    def __init__(self, periods: list[int] | None = None):
        super().__init__("DEMA")
        self.periods = periods or [10, 20, 50]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}
        for period in self.periods:
            ema1 = EMA._ema(close, period)
            ema2 = EMA._ema(ema1, period)
            dema = 2 * ema1 - ema2
            results[f"dema_{period}"] = dema
        return results


class TEMA(TechnicalIndicator):
    """Triple Exponential Moving Average."""

    def __init__(self, periods: list[int] | None = None):
        super().__init__("TEMA")
        self.periods = periods or [10, 20, 50]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}
        for period in self.periods:
            ema1 = EMA._ema(close, period)
            ema2 = EMA._ema(ema1, period)
            ema3 = EMA._ema(ema2, period)
            tema = 3 * ema1 - 3 * ema2 + ema3
            results[f"tema_{period}"] = tema
        return results


class KAMA(TechnicalIndicator):
    """
    Kaufman Adaptive Moving Average (KAMA).

    KAMA adapts to price volatility by using an Efficiency Ratio (ER)
    that measures the directional movement relative to volatility.

    Algorithm:
        1. Calculate Efficiency Ratio: ER = |Price Change| / Volatility
           - Price Change = |Close[t] - Close[t-period]|
           - Volatility = Sum of |Close[i] - Close[i-1]| over period
        2. Calculate Smoothing Constant: SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
           - fast_sc = 2/(fast+1), typically 2/3
           - slow_sc = 2/(slow+1), typically 2/31
        3. KAMA[t] = KAMA[t-1] + SC * (Close[t] - KAMA[t-1])

    Trading Interpretation:
        - KAMA follows price closely in trending markets (high ER)
        - KAMA moves slowly in ranging markets (low ER)
        - Reduces whipsaws compared to simple moving averages

    Complexity: O(n * period) where n is the number of data points
    """

    def __init__(self, period: int = 10, fast: int = 2, slow: int = 30):
        super().__init__("KAMA")
        self.period = period
        self.fast_sc = 2.0 / (fast + 1)
        self.slow_sc = 2.0 / (slow + 1)

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)

        result = np.full(len(close), np.nan)
        n = len(close)

        if n < self.period:
            return {f"kama_{self.period}": result}

        # Efficiency Ratio: measures trend strength vs noise
        # ER = 1 means perfect trend (all movement in one direction)
        # ER = 0 means pure noise (price went nowhere despite movement)
        change = np.abs(close[self.period :] - close[: -self.period])

        # VECTORIZED: Calculate volatility using rolling sum of absolute differences
        # Instead of loop, use cumsum of absolute differences
        abs_diff = np.abs(np.diff(close))
        # Pad with 0 at start to align indices
        abs_diff_padded = np.concatenate([[0], abs_diff])
        cumsum_abs_diff = np.cumsum(abs_diff_padded)
        # Rolling sum = cumsum[i+period] - cumsum[i]
        volatility = cumsum_abs_diff[self.period:] - cumsum_abs_diff[:-self.period]

        # Avoid division by zero
        volatility = np.where(volatility == 0, 1e-10, volatility)
        er = change / volatility

        # Smoothing constant: adapts based on efficiency ratio
        # High ER -> SC approaches fast_sc^2 (responsive)
        # Low ER -> SC approaches slow_sc^2 (smooth)
        sc = (er * (self.fast_sc - self.slow_sc) + self.slow_sc) ** 2

        # KAMA calculation using exponential smoothing with adaptive alpha
        result[self.period - 1] = close[self.period - 1]
        for i in range(self.period, n):
            result[i] = result[i - 1] + sc[i - self.period] * (close[i] - result[i - 1])

        return {f"kama_{self.period}": result}


class HMA(TechnicalIndicator):
    """Hull Moving Average."""

    def __init__(self, periods: list[int] | None = None):
        super().__init__("HMA")
        self.periods = periods or [9, 16, 25]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}
        for period in self.periods:
            half_period = period // 2
            sqrt_period = int(np.sqrt(period))

            wma_half = WMA._wma(close, half_period)
            wma_full = WMA._wma(close, period)
            raw_hma = 2 * wma_half - wma_full
            hma = WMA._wma(raw_hma, sqrt_period)
            results[f"hma_{period}"] = hma
        return results


class VWMA(TechnicalIndicator):
    """Volume Weighted Moving Average."""

    def __init__(self, periods: list[int] | None = None):
        super().__init__("VWMA")
        self.periods = periods or [10, 20, 50]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close", "volume"])
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        results = {}

        pv = close * volume
        for period in self.periods:
            result = np.full(len(close), np.nan)
            for i in range(period - 1, len(close)):
                pv_sum = np.sum(pv[i - period + 1 : i + 1])
                vol_sum = np.sum(volume[i - period + 1 : i + 1])
                if vol_sum > 0:
                    result[i] = pv_sum / vol_sum
            results[f"vwma_{period}"] = result
        return results


class ADX(TechnicalIndicator):
    """
    Average Directional Index (ADX) with DI+ and DI-.

    ADX measures trend strength regardless of direction, while DI+/DI-
    indicate bullish/bearish directional movement.

    Algorithm:
        1. Calculate Directional Movement:
           - +DM = High[t] - High[t-1] if positive and > |Low[t-1] - Low[t]|
           - -DM = Low[t-1] - Low[t] if positive and > High[t] - High[t-1]
        2. Calculate True Range (TR):
           - TR = max(High-Low, |High-Close[t-1]|, |Low-Close[t-1]|)
        3. Smooth using Wilder's method (like EMA but with 1/period factor)
        4. DI+ = 100 * Smoothed(+DM) / ATR
        5. DI- = 100 * Smoothed(-DM) / ATR
        6. DX = 100 * |DI+ - DI-| / (DI+ + DI-)
        7. ADX = Wilder's smoothed average of DX

    Trading Interpretation:
        - ADX > 25: Strong trend (either direction)
        - ADX < 20: Weak trend or ranging market
        - DI+ > DI-: Bullish trend
        - DI- > DI+: Bearish trend
        - DI crossovers signal potential trend reversals

    Complexity: O(n) where n is the number of data points
    """

    def __init__(self, period: int = 14):
        super().__init__("ADX")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        n = len(close)

        # VECTORIZED: Calculate TR, +DM, -DM without loops
        # True Range calculation
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        hl = high[1:] - low[1:]
        hc = np.abs(high[1:] - close[:-1])
        lc = np.abs(low[1:] - close[:-1])
        tr[1:] = np.maximum(np.maximum(hl, hc), lc)

        # Directional Movement calculation
        h_diff = np.zeros(n)
        l_diff = np.zeros(n)
        h_diff[1:] = high[1:] - high[:-1]
        l_diff[1:] = low[:-1] - low[1:]

        # +DM: h_diff if h_diff > l_diff and h_diff > 0, else 0
        plus_dm = np.where((h_diff > l_diff) & (h_diff > 0), h_diff, 0)
        # -DM: l_diff if l_diff > h_diff and l_diff > 0, else 0
        minus_dm = np.where((l_diff > h_diff) & (l_diff > 0), l_diff, 0)

        # Smooth with Wilder's method
        atr = self._wilder_smooth(tr, self.period)
        plus_di = 100 * self._wilder_smooth(plus_dm, self.period) / np.where(atr == 0, 1, atr)
        minus_di = 100 * self._wilder_smooth(minus_dm, self.period) / np.where(atr == 0, 1, atr)

        dx = 100 * np.abs(plus_di - minus_di) / np.where(plus_di + minus_di == 0, 1, plus_di + minus_di)
        adx = self._wilder_smooth(dx, self.period)

        return {
            f"adx_{self.period}": adx,
            f"plus_di_{self.period}": plus_di,
            f"minus_di_{self.period}": minus_di,
        }

    @staticmethod
    def _wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
        """Wilder's smoothing method."""
        result = np.full(len(arr), np.nan)
        if len(arr) >= period:
            result[period - 1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = (result[i - 1] * (period - 1) + arr[i]) / period
        return result


class Aroon(TechnicalIndicator):
    """Aroon indicator (Up, Down, Oscillator)."""

    def __init__(self, period: int = 25):
        super().__init__("Aroon")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)

        n = len(high)
        aroon_up = np.full(n, np.nan)
        aroon_down = np.full(n, np.nan)

        # VECTORIZED using numpy stride tricks for rolling window operations
        # LOOK-AHEAD BIAS FIX: Window should be exactly 'period' elements
        if n >= self.period:
            from numpy.lib.stride_tricks import sliding_window_view

            # Create rolling windows for high and low
            high_windows = sliding_window_view(high, self.period)
            low_windows = sliding_window_view(low, self.period)

            # Find argmax/argmin for each window (vectorized)
            argmax_high = np.argmax(high_windows, axis=1)
            argmin_low = np.argmin(low_windows, axis=1)

            # Days since highest high / lowest low within the window
            days_since_high = (self.period - 1) - argmax_high
            days_since_low = (self.period - 1) - argmin_low

            # Calculate Aroon values
            aroon_up[self.period - 1:] = ((self.period - days_since_high) / self.period) * 100
            aroon_down[self.period - 1:] = ((self.period - days_since_low) / self.period) * 100

        aroon_osc = aroon_up - aroon_down

        return {
            f"aroon_up_{self.period}": aroon_up,
            f"aroon_down_{self.period}": aroon_down,
            f"aroon_osc_{self.period}": aroon_osc,
        }


class ParabolicSAR(TechnicalIndicator):
    """
    JPMORGAN FIX: Parabolic SAR indicator with proper state machine.

    Fixed issues:
    1. Proper state machine with explicit UPTREND/DOWNTREND states
    2. Correct initialization based on initial price action
    3. Proper boundary conditions for SAR constraints
    4. Fixed AF reset logic on reversals
    5. Correct EP update logic
    6. Added reversal signal output for trading

    The Parabolic SAR (Stop and Reverse) is a trend-following indicator that:
    - Provides potential entry/exit points
    - Trails price during a trend
    - Reverses when price crosses the SAR

    State machine states:
    - UPTREND (1): SAR below price, trailing up
    - DOWNTREND (-1): SAR above price, trailing down
    """

    # State constants for clarity
    UPTREND = 1
    DOWNTREND = -1

    def __init__(self, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2):
        super().__init__("PSAR")
        self.af_start = af_start
        self.af_increment = af_increment
        self.af_max = af_max

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)

        n = len(high)
        if n < 2:
            return {
                "psar": np.full(n, np.nan),
                "psar_trend": np.zeros(n),
                "psar_reversal": np.zeros(n),
            }

        psar = np.zeros(n)
        trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
        ep = np.zeros(n)  # Extreme point
        af = np.zeros(n)  # Acceleration factor
        reversal = np.zeros(n)  # Signal when reversal occurs

        # JPMORGAN FIX: Proper initialization based on initial price action
        # Determine initial trend from first two bars
        if high[1] > high[0] and low[1] >= low[0]:
            # Higher high and higher/equal low = uptrend
            initial_trend = self.UPTREND
        elif low[1] < low[0] and high[1] <= high[0]:
            # Lower low and lower/equal high = downtrend
            initial_trend = self.DOWNTREND
        else:
            # Mixed signals - default to uptrend
            initial_trend = self.UPTREND

        # Initialize first bar
        trend[0] = initial_trend
        af[0] = self.af_start

        if initial_trend == self.UPTREND:
            psar[0] = low[0]  # SAR starts below price
            ep[0] = high[0]   # EP is highest high
        else:
            psar[0] = high[0]  # SAR starts above price
            ep[0] = low[0]     # EP is lowest low

        # State machine processing
        for i in range(1, n):
            prev_trend = trend[i - 1]
            prev_psar = psar[i - 1]
            prev_ep = ep[i - 1]
            prev_af = af[i - 1]

            if prev_trend == self.UPTREND:
                # ===== UPTREND STATE =====

                # Step 1: Calculate new SAR
                new_psar = prev_psar + prev_af * (prev_ep - prev_psar)

                # Step 2: SAR cannot be above prior two lows (support constraint)
                # This prevents SAR from entering the price range prematurely
                prior_low_1 = low[i - 1]
                prior_low_2 = low[i - 2] if i >= 2 else low[i - 1]
                sar_constraint = min(prior_low_1, prior_low_2)
                new_psar = min(new_psar, sar_constraint)

                # Step 3: Check for reversal
                if low[i] < new_psar:
                    # REVERSAL: Switch to downtrend
                    trend[i] = self.DOWNTREND
                    reversal[i] = -1  # Bearish reversal signal

                    # New SAR is the previous EP (highest high of uptrend)
                    psar[i] = prev_ep

                    # New EP is current low
                    ep[i] = low[i]

                    # Reset AF
                    af[i] = self.af_start
                else:
                    # Continue uptrend
                    trend[i] = self.UPTREND
                    psar[i] = new_psar

                    # Update EP if new high made
                    if high[i] > prev_ep:
                        ep[i] = high[i]
                        # Increment AF (capped at max)
                        af[i] = min(prev_af + self.af_increment, self.af_max)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af

            else:
                # ===== DOWNTREND STATE =====

                # Step 1: Calculate new SAR
                new_psar = prev_psar - prev_af * (prev_psar - prev_ep)

                # Step 2: SAR cannot be below prior two highs (resistance constraint)
                # This prevents SAR from entering the price range prematurely
                prior_high_1 = high[i - 1]
                prior_high_2 = high[i - 2] if i >= 2 else high[i - 1]
                sar_constraint = max(prior_high_1, prior_high_2)
                new_psar = max(new_psar, sar_constraint)

                # Step 3: Check for reversal
                if high[i] > new_psar:
                    # REVERSAL: Switch to uptrend
                    trend[i] = self.UPTREND
                    reversal[i] = 1  # Bullish reversal signal

                    # New SAR is the previous EP (lowest low of downtrend)
                    psar[i] = prev_ep

                    # New EP is current high
                    ep[i] = high[i]

                    # Reset AF
                    af[i] = self.af_start
                else:
                    # Continue downtrend
                    trend[i] = self.DOWNTREND
                    psar[i] = new_psar

                    # Update EP if new low made
                    if low[i] < prev_ep:
                        ep[i] = low[i]
                        # Increment AF (capped at max)
                        af[i] = min(prev_af + self.af_increment, self.af_max)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af

        return {
            "psar": psar,
            "psar_trend": trend,
            "psar_reversal": reversal,  # JPMORGAN FIX: Added reversal signals
            "psar_af": af,  # JPMORGAN FIX: Added AF for debugging/analysis
        }


class Ichimoku(TechnicalIndicator):
    """
    Ichimoku Kinko Hyo (Ichimoku Cloud) indicator.

    A comprehensive trend-following system that provides support/resistance,
    momentum, and trend direction signals in a single view.

    Components:
        1. Tenkan-sen (Conversion Line): (Highest High + Lowest Low) / 2 over 9 periods
           - Fast-moving signal line for short-term momentum
        2. Kijun-sen (Base Line): Same calculation over 26 periods
           - Slower line indicating medium-term trend
        3. Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
           - First cloud boundary
        4. Senkou Span B (Leading Span B): Midpoint over 52 periods, plotted 26 ahead
           - Second cloud boundary
        5. Chikou Span (Lagging Span): Close price plotted 26 periods back
           - Confirms trend by comparing to past price

    Trading Interpretation:
        - Price above cloud: Bullish trend
        - Price below cloud: Bearish trend
        - Price in cloud: Consolidation/transition
        - Tenkan crosses above Kijun: Bullish signal
        - Cloud color (A vs B): Future trend sentiment
        - Thick cloud: Strong support/resistance
        - Thin cloud: Weak support/resistance

    Complexity: O(n * max(tenkan, kijun, senkou_b))
    """

    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
        super().__init__("Ichimoku")
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b = senkou_b

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        n = len(close)

        def midpoint(h: np.ndarray, l: np.ndarray, period: int) -> np.ndarray:
            """VECTORIZED midpoint calculation using sliding_window_view."""
            result = np.full(len(h), np.nan)
            if len(h) >= period:
                from numpy.lib.stride_tricks import sliding_window_view
                h_windows = sliding_window_view(h, period)
                l_windows = sliding_window_view(l, period)
                max_h = np.max(h_windows, axis=1)
                min_l = np.min(l_windows, axis=1)
                result[period - 1:] = (max_h + min_l) / 2
            return result

        tenkan_sen = midpoint(high, low, self.tenkan)
        kijun_sen = midpoint(high, low, self.kijun)
        senkou_a = (tenkan_sen + kijun_sen) / 2
        senkou_b = midpoint(high, low, self.senkou_b)
        chikou_span = np.roll(close, -self.kijun)
        chikou_span[-self.kijun :] = np.nan

        return {
            "ichimoku_tenkan": tenkan_sen,
            "ichimoku_kijun": kijun_sen,
            "ichimoku_senkou_a": senkou_a,
            "ichimoku_senkou_b": senkou_b,
            "ichimoku_chikou": chikou_span,
        }


class SuperTrend(TechnicalIndicator):
    """
    SuperTrend indicator.

    A trend-following indicator that uses ATR-based bands to identify
    trend direction and potential reversal points.

    Algorithm:
        1. Calculate ATR over the specified period
        2. Upper Band = (High + Low) / 2 + (multiplier * ATR)
        3. Lower Band = (High + Low) / 2 - (multiplier * ATR)
        4. SuperTrend follows:
           - Lower band during uptrend (until price closes below)
           - Upper band during downtrend (until price closes above)
        5. Band values are "locked" to prevent whipsaws:
           - In uptrend: Lower band can only move up, never down
           - In downtrend: Upper band can only move down, never up

    Trading Interpretation:
        - Price above SuperTrend line: Bullish trend (go long)
        - Price below SuperTrend line: Bearish trend (go short)
        - SuperTrend flip: Trend reversal signal
        - Works best in trending markets
        - Larger multiplier = fewer signals but more reliable

    Parameters:
        - period: ATR period (default 10)
        - multiplier: ATR multiplier for band width (default 3.0)

    Complexity: O(n)
    """

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        super().__init__("SuperTrend")
        self.period = period
        self.multiplier = multiplier

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        atr = ATR(period=self.period).compute(df)[f"atr_{self.period}"]

        n = len(close)
        hl2 = (high + low) / 2
        upper_band = hl2 + self.multiplier * atr
        lower_band = hl2 - self.multiplier * atr

        supertrend = np.zeros(n)
        direction = np.zeros(n)  # 1 for up, -1 for down

        supertrend[0] = upper_band[0]
        direction[0] = 1

        for i in range(1, n):
            if np.isnan(upper_band[i]):
                supertrend[i] = np.nan
                direction[i] = np.nan
                continue

            if close[i - 1] <= supertrend[i - 1]:  # In downtrend
                supertrend[i] = upper_band[i]
                if close[i] > supertrend[i]:
                    direction[i] = 1
                    supertrend[i] = lower_band[i]
                else:
                    direction[i] = -1
                    if upper_band[i] < supertrend[i - 1]:
                        supertrend[i] = upper_band[i]
                    else:
                        supertrend[i] = supertrend[i - 1]
            else:  # In uptrend
                supertrend[i] = lower_band[i]
                if close[i] < supertrend[i]:
                    direction[i] = -1
                    supertrend[i] = upper_band[i]
                else:
                    direction[i] = 1
                    if lower_band[i] > supertrend[i - 1]:
                        supertrend[i] = lower_band[i]
                    else:
                        supertrend[i] = supertrend[i - 1]

        return {
            f"supertrend_{self.period}": supertrend,
            f"supertrend_dir_{self.period}": direction,
        }


# =============================================================================
# MOMENTUM INDICATORS
# =============================================================================


class RSI(TechnicalIndicator):
    """Relative Strength Index."""

    def __init__(self, periods: list[int] | None = None):
        super().__init__("RSI")
        self.periods = periods or [7, 14, 21]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        delta = np.diff(close, prepend=close[0])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        for period in self.periods:
            avg_gain = ADX._wilder_smooth(gains, period)
            avg_loss = ADX._wilder_smooth(losses, period)

            rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
            rsi = 100 - (100 / (1 + rs))
            results[f"rsi_{period}"] = rsi

        return results


class StochasticOscillator(TechnicalIndicator):
    """Stochastic Oscillator (%K and %D)."""

    def __init__(self, k_period: int = 14, d_period: int = 3):
        super().__init__("Stochastic")
        self.k_period = k_period
        self.d_period = d_period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        n = len(close)
        stoch_k = np.full(n, np.nan)

        # VECTORIZED using sliding_window_view
        if n >= self.k_period:
            from numpy.lib.stride_tricks import sliding_window_view
            high_windows = sliding_window_view(high, self.k_period)
            low_windows = sliding_window_view(low, self.k_period)

            highest = np.max(high_windows, axis=1)
            lowest = np.min(low_windows, axis=1)
            close_aligned = close[self.k_period - 1:]

            range_hl = highest - lowest
            # Avoid division by zero
            range_hl = np.where(range_hl == 0, 1, range_hl)
            stoch_k[self.k_period - 1:] = np.where(
                highest != lowest,
                100 * (close_aligned - lowest) / range_hl,
                50
            )

        stoch_d = SMA._rolling_mean(stoch_k, self.d_period)

        return {
            f"stoch_k_{self.k_period}": stoch_k,
            f"stoch_d_{self.d_period}": stoch_d,
        }


class StochasticRSI(TechnicalIndicator):
    """Stochastic RSI."""

    def __init__(self, rsi_period: int = 14, stoch_period: int = 14, k_period: int = 3, d_period: int = 3):
        super().__init__("StochRSI")
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.k_period = k_period
        self.d_period = d_period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        rsi = RSI(periods=[self.rsi_period]).compute(df)[f"rsi_{self.rsi_period}"]

        n = len(rsi)
        stoch_rsi = np.full(n, np.nan)

        for i in range(self.stoch_period - 1, n):
            if np.any(np.isnan(rsi[i - self.stoch_period + 1 : i + 1])):
                continue
            highest = np.max(rsi[i - self.stoch_period + 1 : i + 1])
            lowest = np.min(rsi[i - self.stoch_period + 1 : i + 1])
            if highest != lowest:
                stoch_rsi[i] = (rsi[i] - lowest) / (highest - lowest)
            else:
                stoch_rsi[i] = 0.5

        stoch_rsi_k = SMA._rolling_mean(stoch_rsi, self.k_period)
        stoch_rsi_d = SMA._rolling_mean(stoch_rsi_k, self.d_period)

        return {
            "stoch_rsi": stoch_rsi,
            "stoch_rsi_k": stoch_rsi_k,
            "stoch_rsi_d": stoch_rsi_d,
        }


class WilliamsR(TechnicalIndicator):
    """Williams %R indicator."""

    def __init__(self, period: int = 14):
        super().__init__("WilliamsR")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        n = len(close)
        williams_r = np.full(n, np.nan)

        # VECTORIZED using sliding_window_view
        if n >= self.period:
            from numpy.lib.stride_tricks import sliding_window_view
            high_windows = sliding_window_view(high, self.period)
            low_windows = sliding_window_view(low, self.period)

            highest = np.max(high_windows, axis=1)
            lowest = np.min(low_windows, axis=1)
            close_aligned = close[self.period - 1:]

            range_hl = highest - lowest
            range_hl_safe = np.where(range_hl == 0, 1, range_hl)
            williams_r[self.period - 1:] = np.where(
                highest != lowest,
                -100 * (highest - close_aligned) / range_hl_safe,
                -50
            )

        return {f"williams_r_{self.period}": williams_r}


class CCI(TechnicalIndicator):
    """Commodity Channel Index."""

    def __init__(self, period: int = 20):
        super().__init__("CCI")
        self.period = period
        self.constant = 0.015

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        tp = (high + low + close) / 3
        sma_tp = SMA._rolling_mean(tp, self.period)

        n = len(tp)
        mean_dev = np.full(n, np.nan)

        # VECTORIZED mean deviation calculation
        if n >= self.period:
            from numpy.lib.stride_tricks import sliding_window_view
            tp_windows = sliding_window_view(tp, self.period)
            # For each window, compute mean absolute deviation from sma_tp at that position
            # sma_tp[self.period - 1:] contains the SMA values aligned with windows
            sma_aligned = sma_tp[self.period - 1:]
            # Expand sma_aligned to match window shape for broadcasting
            deviations = np.abs(tp_windows - sma_aligned[:, np.newaxis])
            mean_dev[self.period - 1:] = np.mean(deviations, axis=1)

        cci = (tp - sma_tp) / (self.constant * np.where(mean_dev == 0, 1, mean_dev))

        return {f"cci_{self.period}": cci}


class UltimateOscillator(TechnicalIndicator):
    """Ultimate Oscillator."""

    def __init__(self, period1: int = 7, period2: int = 14, period3: int = 28):
        super().__init__("UltOsc")
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        n = len(close)
        bp = np.zeros(n)  # Buying pressure
        tr = np.zeros(n)  # True range

        for i in range(1, n):
            bp[i] = close[i] - min(low[i], close[i - 1])
            tr[i] = max(high[i], close[i - 1]) - min(low[i], close[i - 1])

        def rolling_sum(arr: np.ndarray, period: int) -> np.ndarray:
            result = np.full(len(arr), np.nan)
            for i in range(period - 1, len(arr)):
                result[i] = np.sum(arr[i - period + 1 : i + 1])
            return result

        bp1 = rolling_sum(bp, self.period1)
        tr1 = rolling_sum(tr, self.period1)
        bp2 = rolling_sum(bp, self.period2)
        tr2 = rolling_sum(tr, self.period2)
        bp3 = rolling_sum(bp, self.period3)
        tr3 = rolling_sum(tr, self.period3)

        avg1 = bp1 / np.where(tr1 == 0, 1, tr1)
        avg2 = bp2 / np.where(tr2 == 0, 1, tr2)
        avg3 = bp3 / np.where(tr3 == 0, 1, tr3)

        uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

        return {"ultimate_osc": uo}


class TSI(TechnicalIndicator):
    """True Strength Index."""

    def __init__(self, long_period: int = 25, short_period: int = 13, signal_period: int = 7):
        super().__init__("TSI")
        self.long_period = long_period
        self.short_period = short_period
        self.signal_period = signal_period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)

        pc = np.diff(close, prepend=close[0])
        abs_pc = np.abs(pc)

        double_smooth_pc = EMA._ema(EMA._ema(pc, self.long_period), self.short_period)
        double_smooth_abs_pc = EMA._ema(EMA._ema(abs_pc, self.long_period), self.short_period)

        tsi = 100 * double_smooth_pc / np.where(double_smooth_abs_pc == 0, 1, double_smooth_abs_pc)
        signal = EMA._ema(tsi, self.signal_period)

        return {"tsi": tsi, "tsi_signal": signal}


class ROC(TechnicalIndicator):
    """Rate of Change."""

    def __init__(self, periods: list[int] | None = None):
        super().__init__("ROC")
        self.periods = periods or [1, 5, 10, 20]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        for period in self.periods:
            roc = np.full(len(close), np.nan)
            roc[period:] = (close[period:] - close[:-period]) / close[:-period] * 100
            results[f"roc_{period}"] = roc

        return results


class MACD(TechnicalIndicator):
    """Moving Average Convergence Divergence."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("MACD")
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)

        ema_fast = EMA._ema(close, self.fast)
        ema_slow = EMA._ema(close, self.slow)
        macd_line = ema_fast - ema_slow
        signal_line = EMA._ema(macd_line, self.signal)
        histogram = macd_line - signal_line

        return {
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": histogram,
        }


class PPO(TechnicalIndicator):
    """Percentage Price Oscillator."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("PPO")
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)

        ema_fast = EMA._ema(close, self.fast)
        ema_slow = EMA._ema(close, self.slow)
        ppo_line = 100 * (ema_fast - ema_slow) / np.where(ema_slow == 0, 1, ema_slow)
        signal_line = EMA._ema(ppo_line, self.signal)
        histogram = ppo_line - signal_line

        return {
            "ppo_line": ppo_line,
            "ppo_signal": signal_line,
            "ppo_histogram": histogram,
        }


class Momentum(TechnicalIndicator):
    """Momentum indicator."""

    def __init__(self, periods: list[int] | None = None):
        super().__init__("Momentum")
        self.periods = periods or [10, 20]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        for period in self.periods:
            mom = np.full(len(close), np.nan)
            mom[period:] = close[period:] - close[:-period]
            results[f"momentum_{period}"] = mom

        return results


class CoppockCurve(TechnicalIndicator):
    """Coppock Curve indicator."""

    def __init__(self, wma_period: int = 10, roc1: int = 14, roc2: int = 11):
        super().__init__("Coppock")
        self.wma_period = wma_period
        self.roc1 = roc1
        self.roc2 = roc2

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)

        roc_obj = ROC(periods=[self.roc1, self.roc2])
        rocs = roc_obj.compute(df)

        combined = rocs[f"roc_{self.roc1}"] + rocs[f"roc_{self.roc2}"]
        coppock = WMA._wma(combined, self.wma_period)

        return {"coppock": coppock}


# =============================================================================
# VOLATILITY INDICATORS
# =============================================================================


class ATR(TechnicalIndicator):
    """Average True Range."""

    def __init__(self, period: int = 14):
        super().__init__("ATR")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        n = len(close)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]

        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])
            )

        atr = ADX._wilder_smooth(tr, self.period)

        return {f"atr_{self.period}": atr, "true_range": tr}


class NormalizedATR(TechnicalIndicator):
    """Normalized ATR (NATR)."""

    def __init__(self, period: int = 14):
        super().__init__("NATR")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        close = df["close"].to_numpy().astype(np.float64)

        atr = ATR(period=self.period).compute(df)[f"atr_{self.period}"]
        natr = 100 * atr / np.where(close == 0, 1, close)

        return {f"natr_{self.period}": natr}


class BollingerBands(TechnicalIndicator):
    """Bollinger Bands."""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__("BBands")
        self.period = period
        self.std_dev = std_dev

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)

        sma = SMA._rolling_mean(close, self.period)

        n = len(close)
        std = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            std[i] = np.std(close[i - self.period + 1 : i + 1], ddof=0)

        upper = sma + self.std_dev * std
        lower = sma - self.std_dev * std
        percent_b = (close - lower) / np.where(upper - lower == 0, 1, upper - lower)
        bandwidth = (upper - lower) / np.where(sma == 0, 1, sma)

        return {
            f"bb_upper_{self.period}": upper,
            f"bb_middle_{self.period}": sma,
            f"bb_lower_{self.period}": lower,
            f"bb_percent_b_{self.period}": percent_b,
            f"bb_bandwidth_{self.period}": bandwidth,
        }


class KeltnerChannel(TechnicalIndicator):
    """Keltner Channel."""

    def __init__(self, ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0):
        super().__init__("Keltner")
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        close = df["close"].to_numpy().astype(np.float64)

        ema_middle = EMA._ema(close, self.ema_period)
        atr = ATR(period=self.atr_period).compute(df)[f"atr_{self.atr_period}"]

        upper = ema_middle + self.multiplier * atr
        lower = ema_middle - self.multiplier * atr

        return {
            f"keltner_upper_{self.ema_period}": upper,
            f"keltner_middle_{self.ema_period}": ema_middle,
            f"keltner_lower_{self.ema_period}": lower,
        }


class DonchianChannel(TechnicalIndicator):
    """Donchian Channel."""

    def __init__(self, period: int = 20):
        super().__init__("Donchian")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)

        n = len(high)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)

        # VECTORIZED using sliding_window_view
        if n >= self.period:
            from numpy.lib.stride_tricks import sliding_window_view
            high_windows = sliding_window_view(high, self.period)
            low_windows = sliding_window_view(low, self.period)
            upper[self.period - 1:] = np.max(high_windows, axis=1)
            lower[self.period - 1:] = np.min(low_windows, axis=1)

        middle = (upper + lower) / 2

        return {
            f"donchian_upper_{self.period}": upper,
            f"donchian_middle_{self.period}": middle,
            f"donchian_lower_{self.period}": lower,
        }


class HistoricalVolatility(TechnicalIndicator):
    """Historical Volatility (standard deviation of returns)."""

    def __init__(self, periods: list[int] | None = None, annualize: bool = True):
        super().__init__("HV")
        self.periods = periods or [10, 20, 60]
        self.annualize = annualize
        self.bars_per_year = 252 * 26  # 15-min bars

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)

        log_returns = np.diff(np.log(close), prepend=np.nan)
        results = {}

        for period in self.periods:
            n = len(close)
            hv = np.full(n, np.nan)
            for i in range(period, n):
                hv[i] = np.std(log_returns[i - period + 1 : i + 1], ddof=1)

            if self.annualize:
                hv = hv * np.sqrt(self.bars_per_year)

            results[f"hv_{period}"] = hv

        return results


class ChaikinVolatility(TechnicalIndicator):
    """Chaikin Volatility."""

    def __init__(self, ema_period: int = 10, roc_period: int = 10):
        super().__init__("ChaikinVol")
        self.ema_period = ema_period
        self.roc_period = roc_period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)

        hl_diff = high - low
        ema_hl = EMA._ema(hl_diff, self.ema_period)

        chaikin_vol = np.full(len(ema_hl), np.nan)
        chaikin_vol[self.roc_period :] = (
            (ema_hl[self.roc_period :] - ema_hl[: -self.roc_period])
            / np.where(ema_hl[: -self.roc_period] == 0, 1, ema_hl[: -self.roc_period])
            * 100
        )

        return {"chaikin_volatility": chaikin_vol}


class GarmanKlassVolatility(TechnicalIndicator):
    """Garman-Klass Volatility estimator."""

    def __init__(self, period: int = 20, annualize: bool = True):
        super().__init__("GKVol")
        self.period = period
        self.annualize = annualize
        self.bars_per_year = 252 * 26

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["open", "high", "low", "close"])
        open_p = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_p) ** 2

        gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co

        n = len(close)
        gk_vol = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            gk_vol[i] = np.sqrt(np.mean(gk_var[i - self.period + 1 : i + 1]))

        if self.annualize:
            gk_vol = gk_vol * np.sqrt(self.bars_per_year)

        return {f"gk_volatility_{self.period}": gk_vol}


class ParkinsonVolatility(TechnicalIndicator):
    """Parkinson Volatility estimator."""

    def __init__(self, period: int = 20, annualize: bool = True):
        super().__init__("ParkinsonVol")
        self.period = period
        self.annualize = annualize
        self.bars_per_year = 252 * 26

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)

        log_hl = np.log(high / low) ** 2
        factor = 1 / (4 * np.log(2))

        n = len(high)
        parkinson = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            parkinson[i] = np.sqrt(factor * np.mean(log_hl[i - self.period + 1 : i + 1]))

        if self.annualize:
            parkinson = parkinson * np.sqrt(self.bars_per_year)

        return {f"parkinson_volatility_{self.period}": parkinson}


# =============================================================================
# VOLUME INDICATORS
# =============================================================================


class OBV(TechnicalIndicator):
    """On-Balance Volume."""

    def __init__(self):
        super().__init__("OBV")

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close", "volume"])
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)

        # VECTORIZED OBV calculation
        # Compute sign of price changes: +1 for up, -1 for down, 0 for flat
        price_diff = np.diff(close, prepend=close[0])
        direction = np.sign(price_diff)
        direction[0] = 0  # First bar has no prior comparison

        # Volume with direction
        directed_volume = direction * volume

        # OBV is cumulative sum of directed volume
        obv = np.cumsum(directed_volume)

        return {"obv": obv}


class AccumulationDistribution(TechnicalIndicator):
    """Accumulation/Distribution Line."""

    def __init__(self):
        super().__init__("AD")

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close", "volume"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)

        clv = np.where(
            high != low, ((close - low) - (high - close)) / (high - low), 0
        )
        mf_volume = clv * volume
        ad = np.cumsum(mf_volume)

        return {"ad_line": ad}


class ChaikinMoneyFlow(TechnicalIndicator):
    """Chaikin Money Flow."""

    def __init__(self, period: int = 20):
        super().__init__("CMF")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close", "volume"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)

        clv = np.where(
            high != low, ((close - low) - (high - close)) / (high - low), 0
        )
        mf_volume = clv * volume

        n = len(close)
        cmf = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            vol_sum = np.sum(volume[i - self.period + 1 : i + 1])
            if vol_sum != 0:
                cmf[i] = np.sum(mf_volume[i - self.period + 1 : i + 1]) / vol_sum

        return {f"cmf_{self.period}": cmf}


class MFI(TechnicalIndicator):
    """Money Flow Index."""

    def __init__(self, period: int = 14):
        super().__init__("MFI")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close", "volume"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)

        tp = (high + low + close) / 3
        raw_mf = tp * volume

        # VECTORIZED: Compute positive and negative money flow
        tp_diff = np.diff(tp, prepend=tp[0])
        pos_mf = np.where(tp_diff > 0, raw_mf, 0)
        neg_mf = np.where(tp_diff < 0, raw_mf, 0)
        pos_mf[0] = 0  # First bar has no comparison
        neg_mf[0] = 0

        n = len(close)
        mfi = np.full(n, np.nan)

        # VECTORIZED: Rolling sum using cumsum
        if n >= self.period:
            cumsum_pos = np.cumsum(pos_mf)
            cumsum_neg = np.cumsum(neg_mf)

            # Rolling sum = cumsum[i] - cumsum[i-period]
            pos_sum = np.zeros(n)
            neg_sum = np.zeros(n)

            pos_sum[self.period:] = cumsum_pos[self.period:] - cumsum_pos[:-self.period]
            neg_sum[self.period:] = cumsum_neg[self.period:] - cumsum_neg[:-self.period]

            # Handle first valid window
            pos_sum[self.period - 1] = cumsum_pos[self.period - 1]
            neg_sum[self.period - 1] = cumsum_neg[self.period - 1]

            # Calculate MFI
            neg_sum_safe = np.where(neg_sum == 0, 1e-10, neg_sum)
            mfi[self.period - 1:] = np.where(
                neg_sum[self.period - 1:] != 0,
                100 - (100 / (1 + pos_sum[self.period - 1:] / neg_sum_safe[self.period - 1:])),
                100
            )

        return {f"mfi_{self.period}": mfi}


class ForceIndex(TechnicalIndicator):
    """Force Index."""

    def __init__(self, period: int = 13):
        super().__init__("ForceIndex")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close", "volume"])
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)

        force = np.zeros(len(close))
        force[1:] = (close[1:] - close[:-1]) * volume[1:]

        force_ema = EMA._ema(force, self.period)

        return {f"force_index_{self.period}": force_ema}


class VolumeRatio(TechnicalIndicator):
    """Volume ratio vs moving average."""

    def __init__(self, period: int = 20):
        super().__init__("VolumeRatio")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["volume"])
        volume = df["volume"].to_numpy().astype(np.float64)

        vol_sma = SMA._rolling_mean(volume, self.period)
        vol_ratio = volume / np.where(vol_sma == 0, 1, vol_sma)

        return {f"volume_ratio_{self.period}": vol_ratio}


class VWAP(TechnicalIndicator):
    """Volume Weighted Average Price (session-based or rolling)."""

    def __init__(self, period: int | None = None):
        super().__init__("VWAP")
        self.period = period  # None for session-based

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close", "volume"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)

        tp = (high + low + close) / 3
        cum_tp_vol = np.cumsum(tp * volume)
        cum_vol = np.cumsum(volume)

        if self.period is None:
            vwap = cum_tp_vol / np.where(cum_vol == 0, 1, cum_vol)
            return {"vwap_cumulative": vwap}
        else:
            n = len(close)
            vwap = np.full(n, np.nan)
            for i in range(self.period - 1, n):
                pv_sum = np.sum(tp[i - self.period + 1 : i + 1] * volume[i - self.period + 1 : i + 1])
                vol_sum = np.sum(volume[i - self.period + 1 : i + 1])
                if vol_sum > 0:
                    vwap[i] = pv_sum / vol_sum
            return {f"vwap_{self.period}": vwap}


class PVT(TechnicalIndicator):
    """Price Volume Trend."""

    def __init__(self):
        super().__init__("PVT")

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close", "volume"])
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)

        pvt = np.zeros(len(close))
        for i in range(1, len(close)):
            if close[i - 1] != 0:
                pvt[i] = pvt[i - 1] + volume[i] * (close[i] - close[i - 1]) / close[i - 1]
            else:
                pvt[i] = pvt[i - 1]

        return {"pvt": pvt}


class EaseOfMovement(TechnicalIndicator):
    """Ease of Movement indicator."""

    def __init__(self, period: int = 14):
        super().__init__("EMV")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "volume"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)

        dm = ((high + low) / 2) - np.roll((high + low) / 2, 1)
        dm[0] = 0

        box_ratio = volume / 1e6 / np.where(high - low == 0, 1, high - low)
        emv = dm / np.where(box_ratio == 0, 1, box_ratio)
        emv_sma = SMA._rolling_mean(emv, self.period)

        return {"emv": emv, f"emv_sma_{self.period}": emv_sma}


# =============================================================================
# PRICE ACTION
# =============================================================================


class PivotPoints(TechnicalIndicator):
    """Pivot Points (Standard)."""

    def __init__(self):
        super().__init__("Pivot")

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        # Use previous bar's values for pivot calculation
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_close = np.roll(close, 1)
        prev_high[0] = high[0]
        prev_low[0] = low[0]
        prev_close[0] = close[0]

        pivot = (prev_high + prev_low + prev_close) / 3
        r1 = 2 * pivot - prev_low
        s1 = 2 * pivot - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)

        return {
            "pivot": pivot,
            "pivot_r1": r1,
            "pivot_r2": r2,
            "pivot_r3": r3,
            "pivot_s1": s1,
            "pivot_s2": s2,
            "pivot_s3": s3,
        }


class RollingHighLow(TechnicalIndicator):
    """Rolling High and Low."""

    def __init__(self, periods: list[int] | None = None):
        super().__init__("RollingHL")
        self.periods = periods or [20, 50, 100]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        results = {}

        # VECTORIZED using sliding_window_view
        from numpy.lib.stride_tricks import sliding_window_view
        n = len(high)

        for period in self.periods:
            roll_high = np.full(n, np.nan)
            roll_low = np.full(n, np.nan)

            if n >= period:
                high_windows = sliding_window_view(high, period)
                low_windows = sliding_window_view(low, period)
                roll_high[period - 1:] = np.max(high_windows, axis=1)
                roll_low[period - 1:] = np.min(low_windows, axis=1)

            results[f"rolling_high_{period}"] = roll_high
            results[f"rolling_low_{period}"] = roll_low

        return results


class PriceDistance(TechnicalIndicator):
    """Distance from various price levels."""

    def __init__(self, sma_periods: list[int] | None = None):
        super().__init__("PriceDistance")
        self.sma_periods = sma_periods or [20, 50, 200]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        for period in self.sma_periods:
            sma = SMA._rolling_mean(close, period)
            distance = (close - sma) / np.where(sma == 0, 1, sma) * 100
            results[f"distance_sma_{period}"] = distance

        return results


# =============================================================================
# COMPOSITE CALCULATOR
# =============================================================================


class TechnicalIndicatorCalculator:
    """
    Main calculator class that computes all technical indicators.

    Usage:
        # Core indicators only (~50)
        calculator = TechnicalIndicatorCalculator(include_extended=False)

        # Full 200+ indicator set (default)
        calculator = TechnicalIndicatorCalculator(include_extended=True)

        features = calculator.compute_all(df)  # Returns dict of all features
    """

    def __init__(self, include_all: bool = True, include_extended: bool = True):
        self.indicators: list[TechnicalIndicator] = []
        self.include_extended = include_extended

        if include_all:
            self._add_default_indicators()

        if include_extended:
            self._add_extended_indicators()

    def _add_default_indicators(self) -> None:
        """Add all default indicators."""
        # Trend
        self.indicators.extend([
            SMA(),
            EMA(),
            WMA(),
            DEMA(),
            TEMA(),
            KAMA(),
            HMA(),
            VWMA(),
            ADX(),
            Aroon(),
            ParabolicSAR(),
            Ichimoku(),
            SuperTrend(),
        ])

        # Momentum
        self.indicators.extend([
            RSI(),
            StochasticOscillator(),
            StochasticRSI(),
            WilliamsR(),
            CCI(),
            UltimateOscillator(),
            TSI(),
            ROC(),
            MACD(),
            PPO(),
            Momentum(),
            CoppockCurve(),
        ])

        # Volatility
        self.indicators.extend([
            ATR(),
            NormalizedATR(),
            BollingerBands(),
            KeltnerChannel(),
            DonchianChannel(),
            HistoricalVolatility(),
            ChaikinVolatility(),
            GarmanKlassVolatility(),
            ParkinsonVolatility(),
        ])

        # Volume
        self.indicators.extend([
            OBV(),
            AccumulationDistribution(),
            ChaikinMoneyFlow(),
            MFI(),
            ForceIndex(),
            VolumeRatio(),
            VWAP(),
            PVT(),
            EaseOfMovement(),
        ])

        # Price Action
        self.indicators.extend([
            PivotPoints(),
            RollingHighLow(),
            PriceDistance(),
        ])

    def _add_extended_indicators(self) -> None:
        """Add extended indicators from technical_extended module."""
        try:
            from .technical_extended import ExtendedTechnicalCalculator

            extended_calc = ExtendedTechnicalCalculator()
            self.indicators.extend(extended_calc.indicators)
        except ImportError:
            # Extended module not available, continue with core indicators
            pass

    def add_indicator(self, indicator: TechnicalIndicator) -> None:
        """Add a custom indicator."""
        self.indicators.append(indicator)

    def compute_all(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        """Compute all indicators and return combined results."""
        results = {}
        for indicator in self.indicators:
            try:
                indicator_results = indicator.compute(df)
                results.update(indicator_results)
            except ValueError as e:
                # Skip indicators that can't be computed due to missing columns
                continue
        return results

    def get_indicator_names(self) -> list[str]:
        """Get list of indicator names."""
        return [ind.name for ind in self.indicators]

    def get_feature_count(self, df: pl.DataFrame) -> int:
        """Get total number of features that will be computed."""
        return len(self.compute_all(df))
