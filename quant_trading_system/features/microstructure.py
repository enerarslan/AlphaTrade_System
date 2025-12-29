"""
Market microstructure features module.

Implements microstructure features organized by category:
- Order flow (volume analysis, imbalance, acceleration)
- Price impact (Kyle's Lambda, Amihud illiquidity)
- Flow toxicity (VPIN)
- Intraday patterns (time-of-day effects, session metrics)
- Periodicity (day-of-week, month-end effects)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl


@dataclass
class MicrostructureResult:
    """Result container for microstructure feature calculations."""

    name: str
    values: np.ndarray
    params: dict[str, Any]
    metadata: dict[str, Any] | None = None


class MicrostructureFeature(ABC):
    """Abstract base class for microstructure features."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        """Compute the feature and return named arrays."""
        pass

    def validate_input(self, df: pl.DataFrame, required_columns: list[str]) -> None:
        """Validate that required columns exist."""
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


# =============================================================================
# ORDER FLOW FEATURES
# =============================================================================


class BuySellVolumeImbalance(MicrostructureFeature):
    """
    Buy/Sell volume imbalance estimation.
    Uses close position relative to high-low range to estimate buy/sell pressure.
    """

    def __init__(self, windows: list[int] | None = None):
        super().__init__("BuySellImbalance")
        self.windows = windows or [5, 10, 20]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close", "volume"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)

        # Estimate buy ratio using close location value (CLV)
        hl_range = high - low
        buy_ratio = np.where(
            hl_range > 0, (close - low) / hl_range, 0.5
        )

        buy_volume = volume * buy_ratio
        sell_volume = volume * (1 - buy_ratio)

        results = {}

        # Raw imbalance
        raw_imbalance = (buy_volume - sell_volume) / np.where(volume == 0, 1, volume)
        results["volume_imbalance_raw"] = raw_imbalance

        # Rolling imbalance
        for window in self.windows:
            imbalance = np.full(len(close), np.nan)
            for i in range(window - 1, len(close)):
                buy_sum = np.sum(buy_volume[i - window + 1 : i + 1])
                sell_sum = np.sum(sell_volume[i - window + 1 : i + 1])
                total = buy_sum + sell_sum
                if total > 0:
                    imbalance[i] = (buy_sum - sell_sum) / total
            results[f"volume_imbalance_{window}"] = imbalance

        return results


class VolumeAcceleration(MicrostructureFeature):
    """Volume acceleration (rate of change of volume)."""

    def __init__(self, windows: list[int] | None = None):
        super().__init__("VolumeAcceleration")
        self.windows = windows or [5, 10, 20]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["volume"])
        volume = df["volume"].to_numpy().astype(np.float64)
        results = {}

        for window in self.windows:
            # Rolling mean volume
            vol_ma = np.full(len(volume), np.nan)
            for i in range(window - 1, len(volume)):
                vol_ma[i] = np.mean(volume[i - window + 1 : i + 1])

            # Acceleration (change in MA)
            accel = np.full(len(volume), np.nan)
            accel[1:] = np.diff(vol_ma)

            # Normalize by average volume
            norm_accel = accel / np.where(vol_ma == 0, 1, vol_ma)

            results[f"volume_accel_{window}"] = norm_accel

        return results


class UnusualVolume(MicrostructureFeature):
    """Unusual volume detection (volume relative to rolling stats)."""

    def __init__(self, window: int = 20, threshold: float = 2.0):
        super().__init__("UnusualVolume")
        self.window = window
        self.threshold = threshold

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["volume"])
        volume = df["volume"].to_numpy().astype(np.float64)
        results = {}

        n = len(volume)
        vol_zscore = np.full(n, np.nan)
        unusual_flag = np.full(n, np.nan)

        for i in range(self.window - 1, n):
            window_vol = volume[i - self.window + 1 : i]  # Exclude current
            mean_vol = np.mean(window_vol)
            std_vol = np.std(window_vol, ddof=1)

            if std_vol > 0:
                vol_zscore[i] = (volume[i] - mean_vol) / std_vol
                unusual_flag[i] = 1 if vol_zscore[i] > self.threshold else 0
            else:
                vol_zscore[i] = 0
                unusual_flag[i] = 0

        results[f"volume_zscore_{self.window}"] = vol_zscore
        results[f"unusual_volume_flag_{self.window}"] = unusual_flag

        return results


class VolumeMomentum(MicrostructureFeature):
    """Volume momentum (trend in volume)."""

    def __init__(self, fast: int = 5, slow: int = 20):
        super().__init__("VolumeMomentum")
        self.fast = fast
        self.slow = slow

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["volume"])
        volume = df["volume"].to_numpy().astype(np.float64)
        results = {}

        n = len(volume)

        # Fast and slow EMAs of volume
        fast_ema = self._ema(volume, self.fast)
        slow_ema = self._ema(volume, self.slow)

        # Volume momentum
        vol_mom = fast_ema / np.where(slow_ema == 0, 1, slow_ema)

        results["volume_momentum"] = vol_mom
        results["volume_ema_fast"] = fast_ema
        results["volume_ema_slow"] = slow_ema

        return results

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


# =============================================================================
# PRICE IMPACT FEATURES
# =============================================================================


class KylesLambda(MicrostructureFeature):
    """
    Kyle's Lambda - price impact coefficient.
    Estimates how much price moves per unit of order flow.
    """

    def __init__(self, window: int = 20):
        super().__init__("KylesLambda")
        self.window = window

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close", "volume"])
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        results = {}

        n = len(close)
        kyles_lambda = np.full(n, np.nan)

        # Price changes
        price_change = np.diff(close, prepend=close[0])

        # Signed volume (using price direction)
        signed_volume = np.where(price_change >= 0, volume, -volume)

        for i in range(self.window - 1, n):
            dP = price_change[i - self.window + 1 : i + 1]
            dV = signed_volume[i - self.window + 1 : i + 1]

            # Regression: dP = lambda * dV
            var_dV = np.var(dV)
            if var_dV > 0:
                kyles_lambda[i] = np.cov(dP, dV)[0, 1] / var_dV

        results[f"kyles_lambda_{self.window}"] = kyles_lambda

        return results


class AmihudIlliquidity(MicrostructureFeature):
    """
    Amihud Illiquidity measure.
    |return| / dollar volume
    """

    def __init__(self, window: int = 20):
        super().__init__("AmihudIlliquidity")
        self.window = window

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close", "volume"])
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        results = {}

        n = len(close)

        # Returns
        returns = np.zeros(n)
        returns[1:] = (close[1:] - close[:-1]) / close[:-1]

        # Dollar volume
        dollar_volume = close * volume

        # Daily Amihud
        daily_amihud = np.abs(returns) / np.where(dollar_volume == 0, 1, dollar_volume)

        # Rolling average
        amihud = np.full(n, np.nan)
        for i in range(self.window - 1, n):
            amihud[i] = np.mean(daily_amihud[i - self.window + 1 : i + 1])

        results[f"amihud_{self.window}"] = amihud
        results["amihud_daily"] = daily_amihud

        # Log transform for scaling
        results[f"log_amihud_{self.window}"] = np.log1p(amihud * 1e6)

        return results


class RollSpread(MicrostructureFeature):
    """
    Roll's implied bid-ask spread estimate.
    Based on serial covariance of returns.
    """

    def __init__(self, window: int = 20):
        super().__init__("RollSpread")
        self.window = window

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        n = len(close)
        roll_spread = np.full(n, np.nan)

        # Price changes
        delta_p = np.diff(close, prepend=close[0])

        for i in range(self.window, n):
            dp = delta_p[i - self.window + 1 : i + 1]
            dp_lag = delta_p[i - self.window : i]

            cov = np.cov(dp, dp_lag)[0, 1]

            # Roll spread = 2 * sqrt(-cov) if cov < 0
            if cov < 0:
                roll_spread[i] = 2 * np.sqrt(-cov)
            else:
                roll_spread[i] = 0

        results[f"roll_spread_{self.window}"] = roll_spread

        return results


class EffectiveSpread(MicrostructureFeature):
    """
    Effective spread estimate using high-low range.
    Based on Corwin-Schultz estimator.
    """

    def __init__(self, window: int = 5):
        super().__init__("EffectiveSpread")
        self.window = window

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        results = {}

        n = len(high)
        spread = np.full(n, np.nan)

        for i in range(1, n):
            # Corwin-Schultz spread estimator
            beta = np.log(high[i - 1] / low[i - 1]) ** 2 + np.log(high[i] / low[i]) ** 2

            # Two-day high-low
            high_2d = max(high[i - 1], high[i])
            low_2d = min(low[i - 1], low[i])
            gamma = np.log(high_2d / low_2d) ** 2

            alpha_sq = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(
                gamma / (3 - 2 * np.sqrt(2))
            )

            if alpha_sq > 0:
                spread[i] = 2 * (np.exp(np.sqrt(alpha_sq)) - 1) / (1 + np.exp(np.sqrt(alpha_sq)))
            else:
                spread[i] = 0

        # Rolling average
        spread_ma = np.full(n, np.nan)
        for i in range(self.window - 1, n):
            spread_ma[i] = np.nanmean(spread[i - self.window + 1 : i + 1])

        results["effective_spread"] = spread
        results[f"effective_spread_ma_{self.window}"] = spread_ma

        return results


# =============================================================================
# FLOW TOXICITY FEATURES
# =============================================================================


class VPIN(MicrostructureFeature):
    """
    Volume-Synchronized Probability of Informed Trading (VPIN).
    Measures flow toxicity.
    """

    def __init__(self, bucket_size: int = 50, num_buckets: int = 20):
        super().__init__("VPIN")
        self.bucket_size = bucket_size
        self.num_buckets = num_buckets

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close", "volume"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        results = {}

        n = len(close)

        # Estimate buy volume using bulk volume classification
        price_change = np.diff(close, prepend=close[0])
        std_price = np.std(price_change[price_change != 0]) if np.sum(price_change != 0) > 0 else 1

        # CDF of standardized price change
        z = price_change / std_price
        buy_prob = 0.5 * (1 + np.clip(z / np.sqrt(2), -1, 1))  # Simplified CDF approx
        buy_volume = volume * buy_prob
        sell_volume = volume * (1 - buy_prob)

        # VPIN calculation with volume buckets
        vpin = np.full(n, np.nan)
        total_volume = np.sum(volume)
        avg_bar_volume = total_volume / n if n > 0 else 1

        bucket_volume = self.bucket_size * avg_bar_volume

        # Create buckets
        bucket_buy = []
        bucket_sell = []
        current_buy = 0
        current_sell = 0
        current_bucket_vol = 0

        for i in range(n):
            current_buy += buy_volume[i]
            current_sell += sell_volume[i]
            current_bucket_vol += volume[i]

            if current_bucket_vol >= bucket_volume:
                bucket_buy.append(current_buy)
                bucket_sell.append(current_sell)
                current_buy = 0
                current_sell = 0
                current_bucket_vol = 0

                # Calculate VPIN once we have enough buckets
                if len(bucket_buy) >= self.num_buckets:
                    recent_buy = np.array(bucket_buy[-self.num_buckets :])
                    recent_sell = np.array(bucket_sell[-self.num_buckets :])
                    total_bucket_vol = np.sum(recent_buy + recent_sell)
                    if total_bucket_vol > 0:
                        vpin[i] = np.sum(np.abs(recent_buy - recent_sell)) / total_bucket_vol

        # Forward fill VPIN values
        last_valid = np.nan
        for i in range(n):
            if not np.isnan(vpin[i]):
                last_valid = vpin[i]
            else:
                vpin[i] = last_valid

        results["vpin"] = vpin

        return results


class OrderFlowToxicity(MicrostructureFeature):
    """Order flow toxicity based on volume imbalance persistence."""

    def __init__(self, window: int = 20, persistence_window: int = 5):
        super().__init__("FlowToxicity")
        self.window = window
        self.persistence_window = persistence_window

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close", "volume"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        results = {}

        n = len(close)

        # Compute buy/sell volume using CLV
        hl_range = high - low
        buy_ratio = np.where(hl_range > 0, (close - low) / hl_range, 0.5)

        buy_volume = volume * buy_ratio
        sell_volume = volume * (1 - buy_ratio)

        # Volume imbalance
        total_vol = buy_volume + sell_volume
        imbalance = (buy_volume - sell_volume) / np.where(total_vol == 0, 1, total_vol)

        # Toxicity = rolling autocorrelation of imbalance (persistence)
        toxicity = np.full(n, np.nan)

        for i in range(self.window + self.persistence_window - 1, n):
            imb_window = imbalance[i - self.window + 1 : i + 1]

            # Autocorrelation
            mean_imb = np.mean(imb_window)
            var_imb = np.var(imb_window)

            if var_imb > 0:
                acf = np.corrcoef(
                    imb_window[: -self.persistence_window],
                    imb_window[self.persistence_window :],
                )[0, 1]
                toxicity[i] = acf if not np.isnan(acf) else 0
            else:
                toxicity[i] = 0

        results[f"flow_toxicity_{self.window}"] = toxicity

        return results


# =============================================================================
# INTRADAY PATTERNS
# =============================================================================


class TimeOfDayFeatures(MicrostructureFeature):
    """
    Time-of-day effects.
    Requires timestamp column.
    """

    def __init__(self):
        super().__init__("TimeOfDay")

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["timestamp"])
        results = {}

        timestamps = df["timestamp"].to_list()
        n = len(timestamps)

        hour_sin = np.zeros(n)
        hour_cos = np.zeros(n)
        minute_sin = np.zeros(n)
        minute_cos = np.zeros(n)
        is_first_hour = np.zeros(n)
        is_last_hour = np.zeros(n)
        is_lunch = np.zeros(n)

        for i, ts in enumerate(timestamps):
            if isinstance(ts, datetime):
                hour = ts.hour
                minute = ts.minute
            else:
                # Try to parse if string
                try:
                    dt = datetime.fromisoformat(str(ts))
                    hour = dt.hour
                    minute = dt.minute
                except (ValueError, AttributeError):
                    continue

            # Cyclical encoding
            hour_sin[i] = np.sin(2 * np.pi * hour / 24)
            hour_cos[i] = np.cos(2 * np.pi * hour / 24)
            minute_sin[i] = np.sin(2 * np.pi * minute / 60)
            minute_cos[i] = np.cos(2 * np.pi * minute / 60)

            # Market session flags (assuming US market 9:30-16:00)
            total_minutes = hour * 60 + minute
            is_first_hour[i] = 1 if 570 <= total_minutes < 630 else 0  # 9:30-10:30
            is_last_hour[i] = 1 if 900 <= total_minutes < 960 else 0  # 15:00-16:00
            is_lunch[i] = 1 if 720 <= total_minutes < 780 else 0  # 12:00-13:00

        results["hour_sin"] = hour_sin
        results["hour_cos"] = hour_cos
        results["minute_sin"] = minute_sin
        results["minute_cos"] = minute_cos
        results["is_first_hour"] = is_first_hour
        results["is_last_hour"] = is_last_hour
        results["is_lunch"] = is_lunch

        return results


class OpeningRangeMetrics(MicrostructureFeature):
    """Opening range metrics (first N bars of session)."""

    def __init__(self, opening_bars: int = 4):  # 4 bars = 1 hour for 15-min bars
        super().__init__("OpeningRange")
        self.opening_bars = opening_bars

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close", "timestamp"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        timestamps = df["timestamp"].to_list()
        results = {}

        n = len(close)

        # Track opening range per day
        or_high = np.full(n, np.nan)
        or_low = np.full(n, np.nan)
        or_breakout = np.full(n, np.nan)

        current_date = None
        day_bars = []
        day_indices = []

        for i, ts in enumerate(timestamps):
            if isinstance(ts, datetime):
                date = ts.date()
            else:
                try:
                    date = datetime.fromisoformat(str(ts)).date()
                except (ValueError, AttributeError):
                    continue

            if date != current_date:
                # New day - calculate OR for previous day's remaining bars
                if day_bars and len(day_bars) >= self.opening_bars:
                    or_h = max(h for h, _, _ in day_bars[: self.opening_bars])
                    or_l = min(l for _, l, _ in day_bars[: self.opening_bars])

                    for idx, (_, _, c) in zip(day_indices, day_bars):
                        or_high[idx] = or_h
                        or_low[idx] = or_l
                        if c > or_h:
                            or_breakout[idx] = 1
                        elif c < or_l:
                            or_breakout[idx] = -1
                        else:
                            or_breakout[idx] = 0

                current_date = date
                day_bars = []
                day_indices = []

            day_bars.append((high[i], low[i], close[i]))
            day_indices.append(i)

        # Handle last day
        if day_bars and len(day_bars) >= self.opening_bars:
            or_h = max(h for h, _, _ in day_bars[: self.opening_bars])
            or_l = min(l for _, l, _ in day_bars[: self.opening_bars])

            for idx, (_, _, c) in zip(day_indices, day_bars):
                or_high[idx] = or_h
                or_low[idx] = or_l
                if c > or_h:
                    or_breakout[idx] = 1
                elif c < or_l:
                    or_breakout[idx] = -1
                else:
                    or_breakout[idx] = 0

        # Distance from opening range
        or_dist_high = (close - or_high) / np.where(or_high == 0, 1, or_high)
        or_dist_low = (close - or_low) / np.where(or_low == 0, 1, or_low)

        results["or_high"] = or_high
        results["or_low"] = or_low
        results["or_breakout"] = or_breakout
        results["or_dist_high"] = or_dist_high
        results["or_dist_low"] = or_dist_low

        return results


class SessionMetrics(MicrostructureFeature):
    """Session-level metrics (gap, range usage, close location)."""

    def __init__(self):
        super().__init__("SessionMetrics")

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["open", "high", "low", "close", "timestamp"])
        open_p = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        timestamps = df["timestamp"].to_list()
        results = {}

        n = len(close)

        gap_pct = np.full(n, np.nan)
        range_usage = np.full(n, np.nan)
        close_location = np.full(n, np.nan)

        prev_close = None
        current_date = None

        for i, ts in enumerate(timestamps):
            if isinstance(ts, datetime):
                date = ts.date()
            else:
                try:
                    date = datetime.fromisoformat(str(ts)).date()
                except (ValueError, AttributeError):
                    continue

            if date != current_date:
                # New day - calculate gap
                if prev_close is not None and prev_close != 0:
                    gap_pct[i] = (open_p[i] - prev_close) / prev_close
                current_date = date

            # Close location value (0 = at low, 1 = at high)
            hl_range = high[i] - low[i]
            if hl_range > 0:
                close_location[i] = (close[i] - low[i]) / hl_range
                # Range usage: how much of the range was "used" by close
                range_usage[i] = abs(close[i] - open_p[i]) / hl_range
            else:
                close_location[i] = 0.5
                range_usage[i] = 0

            prev_close = close[i]

        results["gap_pct"] = gap_pct
        results["range_usage"] = range_usage
        results["close_location_value"] = close_location

        return results


class IntradayRangeUsage(MicrostructureFeature):
    """Intraday range usage (how much of daily range has been used)."""

    def __init__(self):
        super().__init__("IntradayRange")

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close", "timestamp"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        timestamps = df["timestamp"].to_list()
        results = {}

        n = len(close)

        intraday_high = np.full(n, np.nan)
        intraday_low = np.full(n, np.nan)
        range_position = np.full(n, np.nan)

        current_date = None
        day_high = None
        day_low = None

        for i, ts in enumerate(timestamps):
            if isinstance(ts, datetime):
                date = ts.date()
            else:
                try:
                    date = datetime.fromisoformat(str(ts)).date()
                except (ValueError, AttributeError):
                    continue

            if date != current_date:
                current_date = date
                day_high = high[i]
                day_low = low[i]
            else:
                day_high = max(day_high, high[i])
                day_low = min(day_low, low[i])

            intraday_high[i] = day_high
            intraday_low[i] = day_low

            hl_range = day_high - day_low
            if hl_range > 0:
                range_position[i] = (close[i] - day_low) / hl_range
            else:
                range_position[i] = 0.5

        results["intraday_high"] = intraday_high
        results["intraday_low"] = intraday_low
        results["intraday_range_position"] = range_position

        return results


# =============================================================================
# PERIODICITY FEATURES
# =============================================================================


class DayOfWeekFeatures(MicrostructureFeature):
    """Day of week effects."""

    def __init__(self):
        super().__init__("DayOfWeek")

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["timestamp"])
        timestamps = df["timestamp"].to_list()
        results = {}

        n = len(timestamps)

        # One-hot encoding for days
        is_monday = np.zeros(n)
        is_tuesday = np.zeros(n)
        is_wednesday = np.zeros(n)
        is_thursday = np.zeros(n)
        is_friday = np.zeros(n)

        # Cyclical encoding
        day_sin = np.zeros(n)
        day_cos = np.zeros(n)

        for i, ts in enumerate(timestamps):
            if isinstance(ts, datetime):
                dow = ts.weekday()  # 0=Monday, 4=Friday
            else:
                try:
                    dow = datetime.fromisoformat(str(ts)).weekday()
                except (ValueError, AttributeError):
                    continue

            # One-hot
            if dow == 0:
                is_monday[i] = 1
            elif dow == 1:
                is_tuesday[i] = 1
            elif dow == 2:
                is_wednesday[i] = 1
            elif dow == 3:
                is_thursday[i] = 1
            elif dow == 4:
                is_friday[i] = 1

            # Cyclical
            day_sin[i] = np.sin(2 * np.pi * dow / 5)
            day_cos[i] = np.cos(2 * np.pi * dow / 5)

        results["is_monday"] = is_monday
        results["is_tuesday"] = is_tuesday
        results["is_wednesday"] = is_wednesday
        results["is_thursday"] = is_thursday
        results["is_friday"] = is_friday
        results["day_sin"] = day_sin
        results["day_cos"] = day_cos

        return results


class MonthEndFeatures(MicrostructureFeature):
    """Month-end effects."""

    def __init__(self, days_threshold: int = 3):
        super().__init__("MonthEnd")
        self.days_threshold = days_threshold

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["timestamp"])
        timestamps = df["timestamp"].to_list()
        results = {}

        n = len(timestamps)

        is_month_end = np.zeros(n)
        is_month_start = np.zeros(n)
        is_quarter_end = np.zeros(n)
        day_of_month = np.zeros(n)
        month_sin = np.zeros(n)
        month_cos = np.zeros(n)

        for i, ts in enumerate(timestamps):
            if isinstance(ts, datetime):
                date = ts
            else:
                try:
                    date = datetime.fromisoformat(str(ts))
                except (ValueError, AttributeError):
                    continue

            day = date.day
            month = date.month

            # Days from month end (approximate)
            # Get last day of month
            if month in [1, 3, 5, 7, 8, 10, 12]:
                last_day = 31
            elif month in [4, 6, 9, 11]:
                last_day = 30
            else:
                last_day = 28 if date.year % 4 != 0 else 29

            days_to_end = last_day - day

            is_month_end[i] = 1 if days_to_end <= self.days_threshold else 0
            is_month_start[i] = 1 if day <= self.days_threshold else 0
            is_quarter_end[i] = 1 if month in [3, 6, 9, 12] and days_to_end <= self.days_threshold else 0

            day_of_month[i] = day

            # Cyclical month encoding
            month_sin[i] = np.sin(2 * np.pi * month / 12)
            month_cos[i] = np.cos(2 * np.pi * month / 12)

        results["is_month_end"] = is_month_end
        results["is_month_start"] = is_month_start
        results["is_quarter_end"] = is_quarter_end
        results["day_of_month"] = day_of_month
        results["month_sin"] = month_sin
        results["month_cos"] = month_cos

        return results


class OptionsExpirationFeatures(MicrostructureFeature):
    """Options expiration effects (third Friday of each month)."""

    def __init__(self, days_before: int = 2):
        super().__init__("OptionsExpiration")
        self.days_before = days_before

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["timestamp"])
        timestamps = df["timestamp"].to_list()
        results = {}

        n = len(timestamps)
        is_opex = np.zeros(n)
        days_to_opex = np.full(n, np.nan)

        for i, ts in enumerate(timestamps):
            if isinstance(ts, datetime):
                date = ts
            else:
                try:
                    date = datetime.fromisoformat(str(ts))
                except (ValueError, AttributeError):
                    continue

            # Find third Friday of the month
            third_friday = self._get_third_friday(date.year, date.month)

            if third_friday:
                delta = (third_friday - date.date()).days
                days_to_opex[i] = delta

                if 0 <= delta <= self.days_before:
                    is_opex[i] = 1

        results["is_opex_week"] = is_opex
        results["days_to_opex"] = days_to_opex

        return results

    @staticmethod
    def _get_third_friday(year: int, month: int) -> datetime | None:
        """Get the third Friday of a given month."""
        import calendar

        cal = calendar.Calendar()
        fridays = [
            day
            for day in cal.itermonthdays2(year, month)
            if day[0] != 0 and day[1] == 4  # Friday = 4
        ]
        if len(fridays) >= 3:
            return datetime(year, month, fridays[2][0]).date()
        return None


# =============================================================================
# COMPOSITE CALCULATOR
# =============================================================================


class MicrostructureFeatureCalculator:
    """
    Main calculator class that computes all microstructure features.

    Usage:
        calculator = MicrostructureFeatureCalculator()
        features = calculator.compute_all(df)  # Returns dict of all features
    """

    def __init__(self, include_all: bool = True):
        self.features: list[MicrostructureFeature] = []

        if include_all:
            self._add_default_features()

    def _add_default_features(self) -> None:
        """Add all default microstructure features."""
        # Order flow
        self.features.extend([
            BuySellVolumeImbalance(),
            VolumeAcceleration(),
            UnusualVolume(),
            VolumeMomentum(),
        ])

        # Price impact
        self.features.extend([
            KylesLambda(),
            AmihudIlliquidity(),
            RollSpread(),
            EffectiveSpread(),
        ])

        # Flow toxicity
        self.features.extend([
            VPIN(),
            OrderFlowToxicity(),
        ])

        # Intraday patterns
        self.features.extend([
            TimeOfDayFeatures(),
            OpeningRangeMetrics(),
            SessionMetrics(),
            IntradayRangeUsage(),
        ])

        # Periodicity
        self.features.extend([
            DayOfWeekFeatures(),
            MonthEndFeatures(),
            OptionsExpirationFeatures(),
        ])

    def add_feature(self, feature: MicrostructureFeature) -> None:
        """Add a custom feature."""
        self.features.append(feature)

    def compute_all(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        """Compute all features and return combined results."""
        results = {}
        for feature in self.features:
            try:
                feature_results = feature.compute(df)
                results.update(feature_results)
            except ValueError:
                # Skip features that can't be computed
                continue
        return results

    def get_feature_names(self) -> list[str]:
        """Get list of feature names."""
        return [f.name for f in self.features]

    def get_feature_count(self, df: pl.DataFrame) -> int:
        """Get total number of features that will be computed."""
        return len(self.compute_all(df))
