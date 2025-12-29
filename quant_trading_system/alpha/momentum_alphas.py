"""
Momentum-based alpha factors module.

Implements momentum alpha factors including:
- Price momentum (various horizons)
- Relative strength
- Trend following
- Breakout signals
- Momentum acceleration/deceleration
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .alpha_base import AlphaFactor, AlphaHorizon, AlphaType


class PriceMomentum(AlphaFactor):
    """
    Price momentum alpha.

    Computes momentum as the rate of change in price over a specified period.
    Positive values indicate upward momentum, negative values indicate downward.
    """

    def __init__(
        self,
        lookback: int = 20,
        horizon: AlphaHorizon = AlphaHorizon.MEDIUM,
        use_log_returns: bool = True,
    ):
        super().__init__(
            name=f"price_momentum_{lookback}",
            alpha_type=AlphaType.MOMENTUM,
            horizon=horizon,
            lookback=lookback,
        )
        self.use_log_returns = use_log_returns

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute price momentum."""
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)
        alpha = np.full(n, np.nan)

        if self.use_log_returns:
            # Log returns over lookback period
            alpha[self.lookback :] = np.log(close[self.lookback :]) - np.log(
                close[: -self.lookback]
            )
        else:
            # Simple returns
            alpha[self.lookback :] = (
                close[self.lookback :] - close[: -self.lookback]
            ) / close[: -self.lookback]

        return alpha

    def get_params(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "use_log_returns": self.use_log_returns,
        }


class RsiMomentum(AlphaFactor):
    """
    RSI-based momentum alpha.

    Uses RSI to identify overbought/oversold conditions.
    Returns positive alpha when RSI indicates momentum, negative when exhaustion.
    """

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
        reversal_mode: bool = False,
    ):
        super().__init__(
            name=f"rsi_momentum_{period}",
            alpha_type=AlphaType.MOMENTUM,
            horizon=AlphaHorizon.SHORT,
            lookback=period,
        )
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.reversal_mode = reversal_mode

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute RSI momentum alpha."""
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)

        # Compute RSI
        delta = np.diff(close, prepend=close[0])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        # Wilder's smoothing
        avg_gain = self._wilder_smooth(gains, self.period)
        avg_loss = self._wilder_smooth(losses, self.period)

        rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
        rsi = 100 - (100 / (1 + rs))

        # Convert RSI to alpha signal
        if self.reversal_mode:
            # Mean reversion: short overbought, long oversold
            alpha = np.where(
                rsi > self.overbought,
                -(rsi - self.overbought) / (100 - self.overbought),
                np.where(
                    rsi < self.oversold,
                    (self.oversold - rsi) / self.oversold,
                    0,
                ),
            )
        else:
            # Momentum: follow RSI direction
            alpha = (rsi - 50) / 50

        return alpha

    @staticmethod
    def _wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
        """Wilder's smoothing method."""
        result = np.full(len(arr), np.nan)
        if len(arr) >= period:
            result[period - 1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = (result[i - 1] * (period - 1) + arr[i]) / period
        return result

    def get_params(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "overbought": self.overbought,
            "oversold": self.oversold,
            "reversal_mode": self.reversal_mode,
        }


class MacdMomentum(AlphaFactor):
    """
    MACD-based momentum alpha.

    Uses MACD histogram for momentum signals.
    Positive when MACD is above signal line, negative when below.
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        use_histogram: bool = True,
    ):
        super().__init__(
            name=f"macd_momentum_{fast}_{slow}",
            alpha_type=AlphaType.MOMENTUM,
            horizon=AlphaHorizon.MEDIUM,
            lookback=slow + signal,
        )
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.use_histogram = use_histogram

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute MACD momentum alpha."""
        close = df["close"].to_numpy().astype(np.float64)

        # Compute EMAs
        ema_fast = self._ema(close, self.fast)
        ema_slow = self._ema(close, self.slow)

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line
        signal_line = self._ema(macd_line, self.signal)

        if self.use_histogram:
            # Use histogram (MACD - Signal)
            alpha = macd_line - signal_line
        else:
            # Use MACD line directly
            alpha = macd_line

        # Normalize by price to make comparable across stocks
        alpha = alpha / np.where(close == 0, 1, close)

        return alpha

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        """Compute exponential moving average."""
        alpha = 2.0 / (period + 1)
        result = np.full(len(arr), np.nan)
        if len(arr) >= period:
            result[period - 1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
        return result

    def get_params(self) -> dict[str, Any]:
        return {
            "fast": self.fast,
            "slow": self.slow,
            "signal": self.signal,
            "use_histogram": self.use_histogram,
        }


class TrendStrength(AlphaFactor):
    """
    ADX-based trend strength alpha.

    Uses ADX to identify trend strength and DI+/DI- for direction.
    Strong positive alpha in strong uptrends, strong negative in downtrends.
    """

    def __init__(self, period: int = 14, adx_threshold: float = 25):
        super().__init__(
            name=f"trend_strength_{period}",
            alpha_type=AlphaType.MOMENTUM,
            horizon=AlphaHorizon.MEDIUM,
            lookback=period * 2,
        )
        self.period = period
        self.adx_threshold = adx_threshold

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute trend strength alpha."""
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
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # Smooth with Wilder's method
        atr = self._wilder_smooth(tr, self.period)
        plus_di = 100 * self._wilder_smooth(plus_dm, self.period) / np.where(atr == 0, 1, atr)
        minus_di = 100 * self._wilder_smooth(minus_dm, self.period) / np.where(atr == 0, 1, atr)

        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / np.where(
            plus_di + minus_di == 0, 1, plus_di + minus_di
        )
        adx = self._wilder_smooth(dx, self.period)

        # Alpha: direction (DI+ - DI-) weighted by trend strength (ADX)
        direction = (plus_di - minus_di) / 100  # Normalize to [-1, 1] range
        strength = adx / 100  # Normalize ADX

        # Only give signal when ADX indicates trend
        alpha = np.where(adx >= self.adx_threshold, direction * strength, 0)

        return alpha

    @staticmethod
    def _wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
        """Wilder's smoothing method."""
        result = np.full(len(arr), np.nan)
        if len(arr) >= period:
            result[period - 1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = (result[i - 1] * (period - 1) + arr[i]) / period
        return result

    def get_params(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "adx_threshold": self.adx_threshold,
        }


class BreakoutMomentum(AlphaFactor):
    """
    Breakout-based momentum alpha.

    Signals breakouts above/below recent highs/lows.
    Positive for upside breakouts, negative for downside.
    """

    def __init__(self, lookback: int = 20, atr_multiplier: float = 1.0):
        super().__init__(
            name=f"breakout_{lookback}",
            alpha_type=AlphaType.MOMENTUM,
            horizon=AlphaHorizon.SHORT,
            lookback=lookback,
        )
        self.atr_multiplier = atr_multiplier

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute breakout momentum alpha."""
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        n = len(close)
        alpha = np.full(n, np.nan)

        # Compute ATR for normalization
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        atr = np.full(n, np.nan)
        for i in range(self.lookback - 1, n):
            atr[i] = np.mean(tr[i - self.lookback + 1 : i + 1])

        for i in range(self.lookback, n):
            # Rolling high/low (excluding current bar)
            rolling_high = np.max(high[i - self.lookback : i])
            rolling_low = np.min(low[i - self.lookback : i])

            # Distance from breakout levels
            up_break = close[i] - rolling_high
            down_break = rolling_low - close[i]

            # Normalize by ATR
            if atr[i] > 0:
                up_signal = up_break / (atr[i] * self.atr_multiplier)
                down_signal = down_break / (atr[i] * self.atr_multiplier)

                # Combined alpha
                if up_break > 0:
                    alpha[i] = min(up_signal, 1.0)  # Upside breakout
                elif down_break > 0:
                    alpha[i] = max(-down_signal, -1.0)  # Downside breakout
                else:
                    # Within range - position relative to range
                    range_size = rolling_high - rolling_low
                    if range_size > 0:
                        position = (close[i] - rolling_low) / range_size - 0.5
                        alpha[i] = position * 0.5  # Weaker signal within range

        return alpha

    def get_params(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "atr_multiplier": self.atr_multiplier,
        }


class MomentumAcceleration(AlphaFactor):
    """
    Momentum acceleration alpha.

    Measures the rate of change of momentum (momentum of momentum).
    Positive when momentum is accelerating, negative when decelerating.
    """

    def __init__(self, momentum_period: int = 20, acceleration_period: int = 5):
        super().__init__(
            name=f"momentum_accel_{momentum_period}_{acceleration_period}",
            alpha_type=AlphaType.MOMENTUM,
            horizon=AlphaHorizon.SHORT,
            lookback=momentum_period + acceleration_period,
        )
        self.momentum_period = momentum_period
        self.acceleration_period = acceleration_period

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute momentum acceleration alpha."""
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)

        # First compute momentum
        momentum = np.full(n, np.nan)
        momentum[self.momentum_period :] = (
            close[self.momentum_period :] - close[: -self.momentum_period]
        ) / close[: -self.momentum_period]

        # Then compute acceleration (change in momentum)
        alpha = np.full(n, np.nan)
        valid_start = self.momentum_period + self.acceleration_period
        alpha[valid_start:] = (
            momentum[valid_start:] - momentum[valid_start - self.acceleration_period : -self.acceleration_period]
        )

        return alpha

    def get_params(self) -> dict[str, Any]:
        return {
            "momentum_period": self.momentum_period,
            "acceleration_period": self.acceleration_period,
        }


class RelativeMomentum(AlphaFactor):
    """
    Relative momentum alpha.

    Compares asset momentum to a benchmark or average.
    Positive when asset has stronger momentum than benchmark.
    """

    def __init__(
        self,
        lookback: int = 20,
        benchmark_col: str = "benchmark",
    ):
        super().__init__(
            name=f"relative_momentum_{lookback}",
            alpha_type=AlphaType.MOMENTUM,
            horizon=AlphaHorizon.MEDIUM,
            lookback=lookback,
        )
        self.benchmark_col = benchmark_col

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute relative momentum alpha."""
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)

        # Asset momentum
        asset_mom = np.full(n, np.nan)
        asset_mom[self.lookback :] = (
            close[self.lookback :] - close[: -self.lookback]
        ) / close[: -self.lookback]

        # Benchmark momentum
        if self.benchmark_col in df.columns:
            benchmark = df[self.benchmark_col].to_numpy().astype(np.float64)
            bench_mom = np.full(n, np.nan)
            bench_mom[self.lookback :] = (
                benchmark[self.lookback :] - benchmark[: -self.lookback]
            ) / benchmark[: -self.lookback]
        else:
            # If no benchmark, use zeros (making relative momentum = absolute momentum)
            bench_mom = np.zeros(n)

        # Relative momentum
        alpha = asset_mom - bench_mom

        return alpha

    def get_params(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "benchmark_col": self.benchmark_col,
        }


class CrossoverMomentum(AlphaFactor):
    """
    Moving average crossover momentum alpha.

    Uses crossover of fast and slow MAs as momentum signal.
    Positive when fast > slow, negative when fast < slow.
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        ma_type: str = "ema",
    ):
        super().__init__(
            name=f"crossover_{fast_period}_{slow_period}",
            alpha_type=AlphaType.MOMENTUM,
            horizon=AlphaHorizon.MEDIUM,
            lookback=slow_period,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute crossover momentum alpha."""
        close = df["close"].to_numpy().astype(np.float64)

        if self.ma_type == "ema":
            fast_ma = self._ema(close, self.fast_period)
            slow_ma = self._ema(close, self.slow_period)
        else:  # SMA
            fast_ma = self._sma(close, self.fast_period)
            slow_ma = self._sma(close, self.slow_period)

        # Alpha: normalized difference between MAs
        alpha = (fast_ma - slow_ma) / np.where(slow_ma == 0, 1, slow_ma)

        return alpha

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        """Compute EMA."""
        alpha = 2.0 / (period + 1)
        result = np.full(len(arr), np.nan)
        if len(arr) >= period:
            result[period - 1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def _sma(arr: np.ndarray, period: int) -> np.ndarray:
        """Compute SMA."""
        result = np.full(len(arr), np.nan)
        for i in range(period - 1, len(arr)):
            result[i] = np.mean(arr[i - period + 1 : i + 1])
        return result

    def get_params(self) -> dict[str, Any]:
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "ma_type": self.ma_type,
        }


class VolumeWeightedMomentum(AlphaFactor):
    """
    Volume-weighted momentum alpha.

    Weights momentum by volume to emphasize price moves with conviction.
    """

    def __init__(self, lookback: int = 20):
        super().__init__(
            name=f"vw_momentum_{lookback}",
            alpha_type=AlphaType.MOMENTUM,
            horizon=AlphaHorizon.MEDIUM,
            lookback=lookback,
        )

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute volume-weighted momentum alpha."""
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        n = len(close)

        alpha = np.full(n, np.nan)

        # Returns
        returns = np.zeros(n)
        returns[1:] = (close[1:] - close[:-1]) / close[:-1]

        for i in range(self.lookback, n):
            window_returns = returns[i - self.lookback + 1 : i + 1]
            window_volume = volume[i - self.lookback + 1 : i + 1]

            total_vol = np.sum(window_volume)
            if total_vol > 0:
                # Volume-weighted average return
                alpha[i] = np.sum(window_returns * window_volume) / total_vol

        return alpha

    def get_params(self) -> dict[str, Any]:
        return {"lookback": self.lookback}


class MomentumQuality(AlphaFactor):
    """
    Momentum quality alpha.

    Combines momentum with quality measures (volatility, consistency).
    Favors smooth, consistent momentum over volatile moves.
    """

    def __init__(self, lookback: int = 20, volatility_penalty: float = 0.5):
        super().__init__(
            name=f"momentum_quality_{lookback}",
            alpha_type=AlphaType.MOMENTUM,
            horizon=AlphaHorizon.MEDIUM,
            lookback=lookback,
        )
        self.volatility_penalty = volatility_penalty

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute momentum quality alpha."""
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)

        # Returns
        returns = np.zeros(n)
        returns[1:] = np.log(close[1:]) - np.log(close[:-1])

        alpha = np.full(n, np.nan)

        for i in range(self.lookback, n):
            window_returns = returns[i - self.lookback + 1 : i + 1]

            # Mean return (momentum)
            mean_ret = np.mean(window_returns)

            # Volatility
            vol = np.std(window_returns, ddof=1)

            # Quality-adjusted momentum
            if vol > 0:
                sharpe_like = mean_ret / vol
                alpha[i] = sharpe_like * np.sqrt(self.lookback)
            else:
                alpha[i] = mean_ret * self.lookback

        return alpha

    def get_params(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "volatility_penalty": self.volatility_penalty,
        }


def create_momentum_alphas() -> list[AlphaFactor]:
    """Create a standard set of momentum alphas."""
    return [
        # Price momentum at various horizons
        PriceMomentum(lookback=5, horizon=AlphaHorizon.SHORT),
        PriceMomentum(lookback=10, horizon=AlphaHorizon.SHORT),
        PriceMomentum(lookback=20, horizon=AlphaHorizon.MEDIUM),
        PriceMomentum(lookback=60, horizon=AlphaHorizon.LONG),
        # RSI momentum
        RsiMomentum(period=14, reversal_mode=False),
        RsiMomentum(period=7, reversal_mode=False),
        # MACD momentum
        MacdMomentum(fast=12, slow=26, signal=9),
        MacdMomentum(fast=8, slow=21, signal=5),
        # Trend strength
        TrendStrength(period=14),
        TrendStrength(period=7),
        # Breakout
        BreakoutMomentum(lookback=20),
        BreakoutMomentum(lookback=50),
        # Acceleration
        MomentumAcceleration(momentum_period=20, acceleration_period=5),
        # Crossover
        CrossoverMomentum(fast_period=10, slow_period=30),
        CrossoverMomentum(fast_period=5, slow_period=20),
        # Volume weighted
        VolumeWeightedMomentum(lookback=20),
        # Quality adjusted
        MomentumQuality(lookback=20),
    ]
