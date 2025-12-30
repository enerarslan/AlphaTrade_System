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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


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

    def validate_input(self, df: pl.DataFrame, required_columns: list[str]) -> None:
        """Validate that required columns exist."""
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


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
        """Compute exponential moving average."""
        alpha = 2.0 / (period + 1)
        result = np.full(len(arr), np.nan)
        if len(arr) >= period:
            # Initialize with SMA
            result[period - 1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
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
        """Compute weighted moving average."""
        result = np.full(len(arr), np.nan)
        weights = np.arange(1, period + 1, dtype=np.float64)
        weight_sum = weights.sum()
        for i in range(period - 1, len(arr)):
            result[i] = np.sum(arr[i - period + 1 : i + 1] * weights) / weight_sum
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
        volatility = np.zeros(n - self.period)
        for i in range(n - self.period):
            volatility[i] = np.sum(np.abs(np.diff(close[i : i + self.period + 1])))

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
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        for i in range(1, n):
            h_diff = high[i] - high[i - 1]
            l_diff = low[i - 1] - low[i]

            plus_dm[i] = h_diff if h_diff > l_diff and h_diff > 0 else 0
            minus_dm[i] = l_diff if l_diff > h_diff and l_diff > 0 else 0

            tr[i] = max(
                high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])
            )

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

        # LOOK-AHEAD BIAS FIX: Window should be exactly 'period' elements
        # Previously used [i - period : i + 1] which gave period+1 elements
        # Fixed to use [i - period + 1 : i + 1] for exactly 'period' elements
        for i in range(self.period - 1, n):
            # Correct window: last 'period' bars ending at current bar i
            window_high = high[i - self.period + 1 : i + 1]  # Exactly period elements
            window_low = low[i - self.period + 1 : i + 1]    # Exactly period elements

            # Days since highest high / lowest low within the window
            days_since_high = (self.period - 1) - np.argmax(window_high)
            days_since_low = (self.period - 1) - np.argmin(window_low)

            aroon_up[i] = ((self.period - days_since_high) / self.period) * 100
            aroon_down[i] = ((self.period - days_since_low) / self.period) * 100

        aroon_osc = aroon_up - aroon_down

        return {
            f"aroon_up_{self.period}": aroon_up,
            f"aroon_down_{self.period}": aroon_down,
            f"aroon_osc_{self.period}": aroon_osc,
        }


class ParabolicSAR(TechnicalIndicator):
    """Parabolic SAR indicator."""

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
        psar = np.zeros(n)
        trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
        ep = np.zeros(n)  # Extreme point
        af = np.zeros(n)

        # Initialize
        trend[0] = 1
        psar[0] = low[0]
        ep[0] = high[0]
        af[0] = self.af_start

        for i in range(1, n):
            if trend[i - 1] == 1:  # Uptrend
                psar[i] = psar[i - 1] + af[i - 1] * (ep[i - 1] - psar[i - 1])
                psar[i] = min(psar[i], low[i - 1], low[i - 2] if i > 1 else low[i - 1])

                if low[i] < psar[i]:  # Reversal
                    trend[i] = -1
                    psar[i] = ep[i - 1]
                    ep[i] = low[i]
                    af[i] = self.af_start
                else:
                    trend[i] = 1
                    if high[i] > ep[i - 1]:
                        ep[i] = high[i]
                        af[i] = min(af[i - 1] + self.af_increment, self.af_max)
                    else:
                        ep[i] = ep[i - 1]
                        af[i] = af[i - 1]
            else:  # Downtrend
                psar[i] = psar[i - 1] - af[i - 1] * (psar[i - 1] - ep[i - 1])
                psar[i] = max(psar[i], high[i - 1], high[i - 2] if i > 1 else high[i - 1])

                if high[i] > psar[i]:  # Reversal
                    trend[i] = 1
                    psar[i] = ep[i - 1]
                    ep[i] = high[i]
                    af[i] = self.af_start
                else:
                    trend[i] = -1
                    if low[i] < ep[i - 1]:
                        ep[i] = low[i]
                        af[i] = min(af[i - 1] + self.af_increment, self.af_max)
                    else:
                        ep[i] = ep[i - 1]
                        af[i] = af[i - 1]

        return {"psar": psar, "psar_trend": trend}


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
            result = np.full(len(h), np.nan)
            for i in range(period - 1, len(h)):
                result[i] = (np.max(h[i - period + 1 : i + 1]) + np.min(l[i - period + 1 : i + 1])) / 2
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

        for i in range(self.k_period - 1, n):
            highest = np.max(high[i - self.k_period + 1 : i + 1])
            lowest = np.min(low[i - self.k_period + 1 : i + 1])
            if highest != lowest:
                stoch_k[i] = 100 * (close[i] - lowest) / (highest - lowest)
            else:
                stoch_k[i] = 50

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

        for i in range(self.period - 1, n):
            highest = np.max(high[i - self.period + 1 : i + 1])
            lowest = np.min(low[i - self.period + 1 : i + 1])
            if highest != lowest:
                williams_r[i] = -100 * (highest - close[i]) / (highest - lowest)
            else:
                williams_r[i] = -50

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

        for i in range(self.period - 1, n):
            mean_dev[i] = np.mean(np.abs(tp[i - self.period + 1 : i + 1] - sma_tp[i]))

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

        for i in range(self.period - 1, n):
            upper[i] = np.max(high[i - self.period + 1 : i + 1])
            lower[i] = np.min(low[i - self.period + 1 : i + 1])

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

        obv = np.zeros(len(close))
        obv[0] = volume[0]

        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

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

        pos_mf = np.zeros(len(tp))
        neg_mf = np.zeros(len(tp))

        for i in range(1, len(tp)):
            if tp[i] > tp[i - 1]:
                pos_mf[i] = raw_mf[i]
            elif tp[i] < tp[i - 1]:
                neg_mf[i] = raw_mf[i]

        n = len(close)
        mfi = np.full(n, np.nan)
        for i in range(self.period, n):
            pos_sum = np.sum(pos_mf[i - self.period + 1 : i + 1])
            neg_sum = np.sum(neg_mf[i - self.period + 1 : i + 1])
            if neg_sum != 0:
                mfi[i] = 100 - (100 / (1 + pos_sum / neg_sum))
            else:
                mfi[i] = 100

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

        for period in self.periods:
            n = len(high)
            roll_high = np.full(n, np.nan)
            roll_low = np.full(n, np.nan)

            for i in range(period - 1, n):
                roll_high[i] = np.max(high[i - period + 1 : i + 1])
                roll_low[i] = np.min(low[i - period + 1 : i + 1])

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
