"""
Mean reversion alpha factors module.

Implements mean reversion alpha factors including:
- Bollinger Band reversion
- Z-score reversion
- Statistical mean reversion (half-life)
- RSI extremes reversion
- Pairs/spread reversion
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .alpha_base import AlphaFactor, AlphaHorizon, AlphaType


class BollingerReversion(AlphaFactor):
    """
    Bollinger Band mean reversion alpha.

    Generates signals based on price position relative to Bollinger Bands.
    Positive (long) when price is near lower band, negative (short) near upper band.
    """

    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        use_percent_b: bool = True,
    ):
        super().__init__(
            name=f"bb_reversion_{period}",
            alpha_type=AlphaType.MEAN_REVERSION,
            horizon=AlphaHorizon.SHORT,
            lookback=period,
        )
        self.period = period
        self.num_std = num_std
        self.use_percent_b = use_percent_b

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute Bollinger Band reversion alpha."""
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)

        # Compute Bollinger Bands
        sma = np.full(n, np.nan)
        std = np.full(n, np.nan)

        for i in range(self.period - 1, n):
            window = close[i - self.period + 1 : i + 1]
            sma[i] = np.mean(window)
            std[i] = np.std(window, ddof=0)

        upper = sma + self.num_std * std
        lower = sma - self.num_std * std

        if self.use_percent_b:
            # %B: 0 at lower band, 1 at upper band
            band_width = upper - lower
            percent_b = np.where(
                band_width > 0, (close - lower) / band_width, 0.5
            )
            # Convert to alpha: -1 near upper, +1 near lower
            alpha = 1 - 2 * percent_b
        else:
            # Z-score within bands
            alpha = np.where(std > 0, -(close - sma) / std, 0)

        return alpha

    def get_params(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "num_std": self.num_std,
            "use_percent_b": self.use_percent_b,
        }


class ZScoreReversion(AlphaFactor):
    """
    Z-score based mean reversion alpha.

    Computes z-score of price relative to rolling mean/std.
    High negative z-score = long signal (expect reversion up)
    High positive z-score = short signal (expect reversion down)
    """

    def __init__(
        self,
        lookback: int = 60,
        entry_threshold: float = 2.0,
        use_returns: bool = False,
    ):
        super().__init__(
            name=f"zscore_reversion_{lookback}",
            alpha_type=AlphaType.MEAN_REVERSION,
            horizon=AlphaHorizon.MEDIUM,
            lookback=lookback,
        )
        self.entry_threshold = entry_threshold
        self.use_returns = use_returns

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute Z-score reversion alpha."""
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)

        if self.use_returns:
            # Z-score of returns
            data = np.zeros(n)
            data[1:] = np.log(close[1:]) - np.log(close[:-1])
        else:
            data = close

        zscore = np.full(n, np.nan)

        for i in range(self.lookback - 1, n):
            window = data[i - self.lookback + 1 : i + 1]
            mean = np.mean(window)
            std = np.std(window, ddof=1)
            if std > 0:
                zscore[i] = (data[i] - mean) / std

        # Convert to alpha: negative z-score -> positive alpha (expect price to rise)
        # Clip to reasonable range
        alpha = -np.clip(zscore, -3, 3) / 3

        return alpha

    def get_params(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "entry_threshold": self.entry_threshold,
            "use_returns": self.use_returns,
        }


class HalfLifeReversion(AlphaFactor):
    """
    Half-life based mean reversion alpha.

    Uses estimated half-life of mean reversion to weight signals.
    Stronger signals when half-life is short (fast reversion expected).
    """

    def __init__(
        self,
        lookback: int = 60,
        min_half_life: int = 1,
        max_half_life: int = 100,
    ):
        super().__init__(
            name=f"halflife_reversion_{lookback}",
            alpha_type=AlphaType.MEAN_REVERSION,
            horizon=AlphaHorizon.MEDIUM,
            lookback=lookback,
        )
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute half-life reversion alpha."""
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)

        alpha = np.full(n, np.nan)

        for i in range(self.lookback - 1, n):
            window = close[i - self.lookback + 1 : i + 1]

            # Estimate half-life
            half_life = self._estimate_half_life(window)

            if half_life is None or half_life < self.min_half_life or half_life > self.max_half_life:
                alpha[i] = 0
                continue

            # Z-score for direction
            mean = np.mean(window)
            std = np.std(window, ddof=1)
            if std > 0:
                zscore = (close[i] - mean) / std
            else:
                zscore = 0

            # Weight by inverse of half-life (shorter = stronger signal)
            weight = 1 - (half_life - self.min_half_life) / (
                self.max_half_life - self.min_half_life
            )

            # Alpha: mean reversion direction weighted by half-life
            alpha[i] = -zscore * weight / 3  # Normalize

        return alpha

    def _estimate_half_life(self, prices: np.ndarray) -> float | None:
        """Estimate half-life of mean reversion."""
        if len(prices) < 10:
            return None

        # Regress price change on lagged price deviation from mean
        y = np.diff(prices)
        x = prices[:-1] - np.mean(prices[:-1])

        if len(y) < 5 or np.var(x) == 0:
            return None

        # OLS: y = theta * x
        theta = np.sum(x * y) / np.sum(x**2)

        if theta >= 0:
            return None  # No mean reversion

        half_life = -np.log(2) / theta

        return max(1, min(half_life, 1000))

    def get_params(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "min_half_life": self.min_half_life,
            "max_half_life": self.max_half_life,
        }


class RsiReversion(AlphaFactor):
    """
    RSI extreme reversion alpha.

    Generates mean reversion signals at RSI extremes.
    Long when RSI is oversold, short when overbought.
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        smooth: int = 3,
    ):
        super().__init__(
            name=f"rsi_reversion_{period}",
            alpha_type=AlphaType.MEAN_REVERSION,
            horizon=AlphaHorizon.SHORT,
            lookback=period + smooth,
        )
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.smooth = smooth

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute RSI reversion alpha."""
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)

        # Compute RSI
        delta = np.diff(close, prepend=close[0])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        avg_gain = self._wilder_smooth(gains, self.period)
        avg_loss = self._wilder_smooth(losses, self.period)

        rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
        rsi = 100 - (100 / (1 + rs))

        # Smooth RSI
        if self.smooth > 1:
            smoothed_rsi = np.full(n, np.nan)
            for i in range(self.smooth - 1, n):
                smoothed_rsi[i] = np.mean(rsi[i - self.smooth + 1 : i + 1])
            rsi = smoothed_rsi

        # Generate reversion signals
        alpha = np.zeros(n)

        # Oversold (long signal)
        oversold_mask = rsi < self.oversold
        alpha[oversold_mask] = (self.oversold - rsi[oversold_mask]) / self.oversold

        # Overbought (short signal)
        overbought_mask = rsi > self.overbought
        alpha[overbought_mask] = (self.overbought - rsi[overbought_mask]) / (100 - self.overbought)

        return alpha

    @staticmethod
    def _wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
        """Wilder's smoothing."""
        result = np.full(len(arr), np.nan)
        if len(arr) >= period:
            result[period - 1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = (result[i - 1] * (period - 1) + arr[i]) / period
        return result

    def get_params(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "oversold": self.oversold,
            "overbought": self.overbought,
            "smooth": self.smooth,
        }


class KeltnerReversion(AlphaFactor):
    """
    Keltner Channel reversion alpha.

    Similar to Bollinger but uses ATR for band width.
    Signals reversion when price touches channel boundaries.
    """

    def __init__(
        self,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
    ):
        super().__init__(
            name=f"keltner_reversion_{ema_period}",
            alpha_type=AlphaType.MEAN_REVERSION,
            horizon=AlphaHorizon.SHORT,
            lookback=max(ema_period, atr_period),
        )
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute Keltner Channel reversion alpha."""
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)

        # Compute EMA
        ema = self._ema(close, self.ema_period)

        # Compute ATR
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        atr = self._wilder_smooth(tr, self.atr_period)

        # Keltner Channels
        upper = ema + self.multiplier * atr
        lower = ema - self.multiplier * atr

        # Position within channel
        channel_width = upper - lower
        position = np.where(
            channel_width > 0, (close - lower) / channel_width, 0.5
        )

        # Reversion alpha: -1 at upper, +1 at lower
        alpha = 1 - 2 * position

        return alpha

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        """EMA calculation."""
        alpha = 2.0 / (period + 1)
        result = np.full(len(arr), np.nan)
        if len(arr) >= period:
            result[period - 1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def _wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
        """Wilder's smoothing."""
        result = np.full(len(arr), np.nan)
        if len(arr) >= period:
            result[period - 1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = (result[i - 1] * (period - 1) + arr[i]) / period
        return result

    def get_params(self) -> dict[str, Any]:
        return {
            "ema_period": self.ema_period,
            "atr_period": self.atr_period,
            "multiplier": self.multiplier,
        }


class OUReversion(AlphaFactor):
    """
    Ornstein-Uhlenbeck process reversion alpha.

    Uses OU process parameters to generate mean reversion signals.
    """

    def __init__(self, lookback: int = 60, min_theta: float = 0.01):
        super().__init__(
            name=f"ou_reversion_{lookback}",
            alpha_type=AlphaType.MEAN_REVERSION,
            horizon=AlphaHorizon.MEDIUM,
            lookback=lookback,
        )
        self.min_theta = min_theta

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute OU reversion alpha."""
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)

        alpha = np.full(n, np.nan)

        for i in range(self.lookback - 1, n):
            window = close[i - self.lookback + 1 : i + 1]

            # Fit OU parameters
            params = self._fit_ou(window)
            if params is None:
                alpha[i] = 0
                continue

            theta, mu, _ = params

            # Only trade if mean reversion is strong enough
            if theta < self.min_theta:
                alpha[i] = 0
                continue

            # Signal: deviation from equilibrium level
            deviation = close[i] - mu

            # Normalize by typical deviation
            typical_dev = np.std(window, ddof=1)
            if typical_dev > 0:
                normalized_dev = deviation / typical_dev
            else:
                normalized_dev = 0

            # Weight by speed of mean reversion
            weight = min(theta, 1.0)

            # Alpha: expect price to revert to mu
            alpha[i] = -normalized_dev * weight / 2

        return alpha

    def _fit_ou(self, prices: np.ndarray) -> tuple[float, float, float] | None:
        """Fit OU process parameters."""
        if len(prices) < 10:
            return None

        y = np.diff(prices)
        x = prices[:-1]

        if len(y) < 5:
            return None

        # Add intercept for regression
        X = np.column_stack([np.ones(len(x)), x])

        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None

        a, b = coeffs[0], coeffs[1]

        if b >= 0:  # No mean reversion
            return None

        theta = -b
        mu = a / theta if theta > 0 else np.mean(prices)

        residuals = y - X @ coeffs
        sigma = np.std(residuals, ddof=2)

        return theta, mu, sigma

    def get_params(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "min_theta": self.min_theta,
        }


class SpreadReversion(AlphaFactor):
    """
    Spread/ratio reversion alpha.

    Computes reversion signals based on spread between price and a reference
    (e.g., moving average, VWAP, or benchmark).
    """

    def __init__(
        self,
        lookback: int = 20,
        reference: str = "sma",
        threshold: float = 2.0,
    ):
        super().__init__(
            name=f"spread_reversion_{lookback}",
            alpha_type=AlphaType.MEAN_REVERSION,
            horizon=AlphaHorizon.SHORT,
            lookback=lookback,
        )
        self.reference = reference
        self.threshold = threshold

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute spread reversion alpha."""
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)

        # Compute reference price
        if self.reference == "sma":
            ref = np.full(n, np.nan)
            for i in range(self.lookback - 1, n):
                ref[i] = np.mean(close[i - self.lookback + 1 : i + 1])

        elif self.reference == "ema":
            ref = self._ema(close, self.lookback)

        elif self.reference == "vwap" and "volume" in df.columns:
            volume = df["volume"].to_numpy().astype(np.float64)
            high = df["high"].to_numpy().astype(np.float64)
            low = df["low"].to_numpy().astype(np.float64)
            tp = (high + low + close) / 3
            ref = np.full(n, np.nan)
            for i in range(self.lookback - 1, n):
                pv = tp[i - self.lookback + 1 : i + 1] * volume[i - self.lookback + 1 : i + 1]
                vol = volume[i - self.lookback + 1 : i + 1]
                if np.sum(vol) > 0:
                    ref[i] = np.sum(pv) / np.sum(vol)
        else:
            # Default to SMA
            ref = np.full(n, np.nan)
            for i in range(self.lookback - 1, n):
                ref[i] = np.mean(close[i - self.lookback + 1 : i + 1])

        # Compute spread
        spread = (close - ref) / np.where(ref == 0, 1, ref)

        # Rolling std of spread for normalization
        spread_zscore = np.full(n, np.nan)
        for i in range(self.lookback * 2 - 1, n):
            window = spread[i - self.lookback + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 5:
                mean_spread = np.mean(valid)
                std_spread = np.std(valid, ddof=1)
                if std_spread > 0:
                    spread_zscore[i] = (spread[i] - mean_spread) / std_spread

        # Reversion alpha
        alpha = -np.clip(spread_zscore, -3, 3) / 3

        return alpha

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        """EMA calculation."""
        alpha = 2.0 / (period + 1)
        result = np.full(len(arr), np.nan)
        if len(arr) >= period:
            result[period - 1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
        return result

    def get_params(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "reference": self.reference,
            "threshold": self.threshold,
        }


class StochasticReversion(AlphaFactor):
    """
    Stochastic oscillator reversion alpha.

    Generates mean reversion signals at stochastic extremes.
    """

    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        oversold: float = 20,
        overbought: float = 80,
    ):
        super().__init__(
            name=f"stoch_reversion_{k_period}",
            alpha_type=AlphaType.MEAN_REVERSION,
            horizon=AlphaHorizon.SHORT,
            lookback=k_period + d_period,
        )
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute stochastic reversion alpha."""
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)

        # Compute %K
        stoch_k = np.full(n, np.nan)
        for i in range(self.k_period - 1, n):
            highest = np.max(high[i - self.k_period + 1 : i + 1])
            lowest = np.min(low[i - self.k_period + 1 : i + 1])
            if highest != lowest:
                stoch_k[i] = 100 * (close[i] - lowest) / (highest - lowest)
            else:
                stoch_k[i] = 50

        # Compute %D (SMA of %K)
        stoch_d = np.full(n, np.nan)
        for i in range(self.k_period + self.d_period - 2, n):
            window = stoch_k[i - self.d_period + 1 : i + 1]
            stoch_d[i] = np.mean(window[~np.isnan(window)])

        # Generate reversion signals
        alpha = np.zeros(n)

        # Oversold (long signal)
        oversold_mask = stoch_d < self.oversold
        alpha[oversold_mask] = (self.oversold - stoch_d[oversold_mask]) / self.oversold

        # Overbought (short signal)
        overbought_mask = stoch_d > self.overbought
        alpha[overbought_mask] = (self.overbought - stoch_d[overbought_mask]) / (
            100 - self.overbought
        )

        return alpha

    def get_params(self) -> dict[str, Any]:
        return {
            "k_period": self.k_period,
            "d_period": self.d_period,
            "oversold": self.oversold,
            "overbought": self.overbought,
        }


class VolumeWeightedReversion(AlphaFactor):
    """
    Volume-weighted mean reversion alpha.

    Weights reversion signals by volume (expecting larger reversions
    after high-volume moves).
    """

    def __init__(self, lookback: int = 20, volume_threshold: float = 1.5):
        super().__init__(
            name=f"vw_reversion_{lookback}",
            alpha_type=AlphaType.MEAN_REVERSION,
            horizon=AlphaHorizon.SHORT,
            lookback=lookback,
        )
        self.volume_threshold = volume_threshold

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute volume-weighted reversion alpha."""
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        n = len(close)

        alpha = np.full(n, np.nan)

        for i in range(self.lookback - 1, n):
            # Price z-score
            price_window = close[i - self.lookback + 1 : i + 1]
            mean_price = np.mean(price_window)
            std_price = np.std(price_window, ddof=1)

            if std_price > 0:
                zscore = (close[i] - mean_price) / std_price
            else:
                zscore = 0

            # Volume relative to average
            vol_window = volume[i - self.lookback + 1 : i + 1]
            mean_vol = np.mean(vol_window)
            rel_vol = volume[i] / mean_vol if mean_vol > 0 else 1

            # Stronger signal on high volume
            if rel_vol > self.volume_threshold:
                vol_weight = min(rel_vol / self.volume_threshold, 2.0)
            else:
                vol_weight = 1.0

            # Reversion alpha weighted by volume
            alpha[i] = -zscore * vol_weight / 3

        return alpha

    def get_params(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "volume_threshold": self.volume_threshold,
        }


def create_mean_reversion_alphas() -> list[AlphaFactor]:
    """Create a standard set of mean reversion alphas."""
    return [
        # Bollinger Band reversion
        BollingerReversion(period=20, num_std=2.0),
        BollingerReversion(period=10, num_std=1.5),
        # Z-score reversion
        ZScoreReversion(lookback=20),
        ZScoreReversion(lookback=60),
        ZScoreReversion(lookback=20, use_returns=True),
        # Half-life reversion
        HalfLifeReversion(lookback=60),
        HalfLifeReversion(lookback=120),
        # RSI reversion
        RsiReversion(period=14),
        RsiReversion(period=7),
        # Keltner reversion
        KeltnerReversion(ema_period=20),
        # OU reversion
        OUReversion(lookback=60),
        # Spread reversion
        SpreadReversion(lookback=20, reference="sma"),
        SpreadReversion(lookback=20, reference="ema"),
        # Stochastic reversion
        StochasticReversion(k_period=14),
        # Volume-weighted reversion
        VolumeWeightedReversion(lookback=20),
    ]
