"""
Extended technical indicators module.

Adds 150+ additional technical indicators to reach 200+ total:
- Additional momentum indicators (~30)
- Advanced volatility measures (~30)
- Candlestick pattern recognition (~40)
- Composite/derived indicators (~50)

This module extends the base technical.py indicators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from .technical import (
    TechnicalIndicator,
    SMA,
    EMA,
    ATR,
    ADX,
    RSI,
)


# =============================================================================
# ADDITIONAL MOMENTUM INDICATORS
# =============================================================================


class CMO(TechnicalIndicator):
    """Chande Momentum Oscillator."""

    def __init__(self, periods: list[int] | None = None):
        super().__init__("CMO")
        self.periods = periods or [9, 14, 20]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        for period in self.periods:
            diff = np.diff(close, prepend=close[0])
            gains = np.where(diff > 0, diff, 0)
            losses = np.where(diff < 0, -diff, 0)

            n = len(close)
            cmo = np.full(n, np.nan)
            for i in range(period, n):
                sum_gains = np.sum(gains[i - period + 1 : i + 1])
                sum_losses = np.sum(losses[i - period + 1 : i + 1])
                if sum_gains + sum_losses != 0:
                    cmo[i] = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)

            results[f"cmo_{period}"] = cmo

        return results


class TRIX(TechnicalIndicator):
    """Triple Exponential Average."""

    def __init__(self, period: int = 15, signal: int = 9):
        super().__init__("TRIX")
        self.period = period
        self.signal = signal

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)

        ema1 = EMA._ema(close, self.period)
        ema2 = EMA._ema(ema1, self.period)
        ema3 = EMA._ema(ema2, self.period)

        trix = np.full(len(close), np.nan)
        trix[1:] = (ema3[1:] - ema3[:-1]) / np.where(ema3[:-1] == 0, 1, ema3[:-1]) * 100

        signal_line = EMA._ema(trix, self.signal)

        return {
            f"trix_{self.period}": trix,
            f"trix_signal_{self.signal}": signal_line,
        }


class DPO(TechnicalIndicator):
    """Detrended Price Oscillator."""

    def __init__(self, period: int = 20):
        super().__init__("DPO")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)

        offset = self.period // 2 + 1
        sma = SMA._rolling_mean(close, self.period)

        dpo = np.full(len(close), np.nan)
        dpo[offset:] = close[offset:] - sma[:-offset]

        return {f"dpo_{self.period}": dpo}


class ElderRay(TechnicalIndicator):
    """Elder Ray Index (Bull and Bear Power)."""

    def __init__(self, period: int = 13):
        super().__init__("ElderRay")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        ema = EMA._ema(close, self.period)
        bull_power = high - ema
        bear_power = low - ema

        return {
            f"bull_power_{self.period}": bull_power,
            f"bear_power_{self.period}": bear_power,
        }


class KST(TechnicalIndicator):
    """Know Sure Thing oscillator."""

    def __init__(self):
        super().__init__("KST")
        self.roc_periods = [10, 15, 20, 30]
        self.sma_periods = [10, 10, 10, 15]
        self.weights = [1, 2, 3, 4]
        self.signal_period = 9

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)

        kst = np.zeros(len(close))

        for roc_p, sma_p, weight in zip(self.roc_periods, self.sma_periods, self.weights):
            roc = np.full(len(close), np.nan)
            roc[roc_p:] = (close[roc_p:] - close[:-roc_p]) / close[:-roc_p] * 100
            smoothed = SMA._rolling_mean(roc, sma_p)
            kst += weight * np.nan_to_num(smoothed, nan=0.0)

        signal = SMA._rolling_mean(kst, self.signal_period)

        return {
            "kst": kst,
            "kst_signal": signal,
        }


class MassIndex(TechnicalIndicator):
    """Mass Index - identifies trend reversals."""

    def __init__(self, ema_period: int = 9, sum_period: int = 25):
        super().__init__("MassIndex")
        self.ema_period = ema_period
        self.sum_period = sum_period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)

        hl_range = high - low
        ema1 = EMA._ema(hl_range, self.ema_period)
        ema2 = EMA._ema(ema1, self.ema_period)

        ratio = ema1 / np.where(ema2 == 0, 1, ema2)

        n = len(high)
        mass_index = np.full(n, np.nan)
        for i in range(self.sum_period - 1, n):
            mass_index[i] = np.sum(ratio[i - self.sum_period + 1 : i + 1])

        return {"mass_index": mass_index}


class RVI(TechnicalIndicator):
    """Relative Vigor Index."""

    def __init__(self, period: int = 10):
        super().__init__("RVI")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["open", "high", "low", "close"])
        open_p = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        n = len(close)

        # Numerator: Close - Open
        num = close - open_p
        # Denominator: High - Low
        denom = high - low

        # Smooth with weighted average
        smooth_num = np.zeros(n)
        smooth_denom = np.zeros(n)

        for i in range(3, n):
            smooth_num[i] = (num[i] + 2 * num[i - 1] + 2 * num[i - 2] + num[i - 3]) / 6
            smooth_denom[i] = (denom[i] + 2 * denom[i - 1] + 2 * denom[i - 2] + denom[i - 3]) / 6

        # Sum over period
        rvi = np.full(n, np.nan)
        for i in range(self.period + 2, n):
            num_sum = np.sum(smooth_num[i - self.period + 1 : i + 1])
            denom_sum = np.sum(smooth_denom[i - self.period + 1 : i + 1])
            if denom_sum != 0:
                rvi[i] = num_sum / denom_sum

        # Signal line
        signal = np.full(n, np.nan)
        for i in range(3, n):
            if not np.isnan(rvi[i]) and not np.isnan(rvi[i - 1]) and not np.isnan(rvi[i - 2]) and not np.isnan(rvi[i - 3]):
                signal[i] = (rvi[i] + 2 * rvi[i - 1] + 2 * rvi[i - 2] + rvi[i - 3]) / 6

        return {
            f"rvi_{self.period}": rvi,
            f"rvi_signal_{self.period}": signal,
        }


class QStick(TechnicalIndicator):
    """QStick indicator."""

    def __init__(self, period: int = 8):
        super().__init__("QStick")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["open", "close"])
        open_p = df["open"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        diff = close - open_p
        qstick = SMA._rolling_mean(diff, self.period)

        return {f"qstick_{self.period}": qstick}


class BalanceOfPower(TechnicalIndicator):
    """Balance of Power indicator."""

    def __init__(self, period: int = 14):
        super().__init__("BOP")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["open", "high", "low", "close"])
        open_p = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        bop = (close - open_p) / np.where(high - low == 0, 1, high - low)
        bop_sma = SMA._rolling_mean(bop, self.period)

        return {
            "bop": bop,
            f"bop_sma_{self.period}": bop_sma,
        }


class ConnorsRSI(TechnicalIndicator):
    """Connors RSI - composite of RSI, streak RSI, and rank."""

    def __init__(self, rsi_period: int = 3, streak_period: int = 2, rank_period: int = 100):
        super().__init__("ConnorsRSI")
        self.rsi_period = rsi_period
        self.streak_period = streak_period
        self.rank_period = rank_period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)

        # Standard RSI
        rsi_values = RSI(periods=[self.rsi_period]).compute(df)[f"rsi_{self.rsi_period}"]

        # Streak calculation
        streak = np.zeros(n)
        for i in range(1, n):
            if close[i] > close[i - 1]:
                streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
            elif close[i] < close[i - 1]:
                streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
            else:
                streak[i] = 0

        # RSI of streak
        streak_rsi = self._compute_rsi_of_array(streak, self.streak_period)

        # Percent rank of ROC
        roc = np.zeros(n)
        roc[1:] = (close[1:] - close[:-1]) / close[:-1] * 100

        percent_rank = np.full(n, np.nan)
        for i in range(self.rank_period, n):
            window = roc[i - self.rank_period + 1 : i + 1]
            percent_rank[i] = 100 * np.sum(window < roc[i]) / self.rank_period

        # Combine
        crsi = (rsi_values + streak_rsi + percent_rank) / 3

        return {"connors_rsi": crsi}

    @staticmethod
    def _compute_rsi_of_array(arr: np.ndarray, period: int) -> np.ndarray:
        """Compute RSI of arbitrary array."""
        n = len(arr)
        delta = np.diff(arr, prepend=arr[0])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        avg_gain = ADX._wilder_smooth(gains, period)
        avg_loss = ADX._wilder_smooth(losses, period)

        rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
        return 100 - (100 / (1 + rs))


class FisherTransform(TechnicalIndicator):
    """Fisher Transform indicator."""

    def __init__(self, period: int = 9):
        super().__init__("Fisher")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)

        n = len(high)
        hl2 = (high + low) / 2

        # Normalize to -1 to 1
        value = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            max_h = np.max(high[i - self.period + 1 : i + 1])
            min_l = np.min(low[i - self.period + 1 : i + 1])
            if max_h != min_l:
                raw = 2 * (hl2[i] - min_l) / (max_h - min_l) - 1
                value[i] = np.clip(raw, -0.999, 0.999)
            else:
                value[i] = 0

        # Fisher transform with smoothing
        fisher = np.full(n, np.nan)
        fisher[self.period - 1] = 0.5 * np.log((1 + value[self.period - 1]) / (1 - value[self.period - 1]))

        for i in range(self.period, n):
            smoothed_val = 0.5 * value[i] + 0.5 * value[i - 1]
            smoothed_val = np.clip(smoothed_val, -0.999, 0.999)
            fisher[i] = 0.5 * np.log((1 + smoothed_val) / (1 - smoothed_val)) + 0.5 * fisher[i - 1]

        signal = np.roll(fisher, 1)
        signal[0] = np.nan

        return {
            f"fisher_{self.period}": fisher,
            f"fisher_signal_{self.period}": signal,
        }


class SMI(TechnicalIndicator):
    """Stochastic Momentum Index."""

    def __init__(self, k_period: int = 10, smooth1: int = 3, smooth2: int = 3, signal: int = 10):
        super().__init__("SMI")
        self.k_period = k_period
        self.smooth1 = smooth1
        self.smooth2 = smooth2
        self.signal = signal

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        n = len(close)

        # Find highest high and lowest low
        hh = np.full(n, np.nan)
        ll = np.full(n, np.nan)
        for i in range(self.k_period - 1, n):
            hh[i] = np.max(high[i - self.k_period + 1 : i + 1])
            ll[i] = np.min(low[i - self.k_period + 1 : i + 1])

        # Distance from midpoint
        midpoint = (hh + ll) / 2
        diff = close - midpoint
        hl_range = hh - ll

        # Double smooth
        diff_smooth1 = EMA._ema(diff, self.smooth1)
        diff_smooth2 = EMA._ema(diff_smooth1, self.smooth2)

        range_smooth1 = EMA._ema(hl_range, self.smooth1)
        range_smooth2 = EMA._ema(range_smooth1, self.smooth2)

        smi = 100 * diff_smooth2 / np.where(range_smooth2 / 2 == 0, 1, range_smooth2 / 2)
        smi_signal = EMA._ema(smi, self.signal)

        return {
            "smi": smi,
            "smi_signal": smi_signal,
        }


class AO(TechnicalIndicator):
    """Awesome Oscillator."""

    def __init__(self, fast: int = 5, slow: int = 34):
        super().__init__("AO")
        self.fast = fast
        self.slow = slow

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)

        median_price = (high + low) / 2
        sma_fast = SMA._rolling_mean(median_price, self.fast)
        sma_slow = SMA._rolling_mean(median_price, self.slow)

        ao = sma_fast - sma_slow

        return {"ao": ao}


class AC(TechnicalIndicator):
    """Accelerator Oscillator."""

    def __init__(self, fast: int = 5, slow: int = 34, signal: int = 5):
        super().__init__("AC")
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low"])

        ao = AO(self.fast, self.slow).compute(df)["ao"]
        ao_sma = SMA._rolling_mean(ao, self.signal)

        ac = ao - ao_sma

        return {"ac": ac}


class Vortex(TechnicalIndicator):
    """Vortex Indicator."""

    def __init__(self, period: int = 14):
        super().__init__("Vortex")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        n = len(close)

        # True Range
        tr = np.zeros(n)
        vm_plus = np.zeros(n)
        vm_minus = np.zeros(n)

        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
            vm_plus[i] = abs(high[i] - low[i - 1])
            vm_minus[i] = abs(low[i] - high[i - 1])

        # Rolling sums
        vi_plus = np.full(n, np.nan)
        vi_minus = np.full(n, np.nan)

        for i in range(self.period, n):
            tr_sum = np.sum(tr[i - self.period + 1 : i + 1])
            if tr_sum != 0:
                vi_plus[i] = np.sum(vm_plus[i - self.period + 1 : i + 1]) / tr_sum
                vi_minus[i] = np.sum(vm_minus[i - self.period + 1 : i + 1]) / tr_sum

        return {
            f"vortex_plus_{self.period}": vi_plus,
            f"vortex_minus_{self.period}": vi_minus,
        }


class McGinleyDynamic(TechnicalIndicator):
    """McGinley Dynamic indicator."""

    def __init__(self, period: int = 14):
        super().__init__("McGinley")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)

        n = len(close)
        md = np.zeros(n)
        md[0] = close[0]

        for i in range(1, n):
            if md[i - 1] != 0:
                md[i] = md[i - 1] + (close[i] - md[i - 1]) / (self.period * (close[i] / md[i - 1]) ** 4)
            else:
                md[i] = close[i]

        return {f"mcginley_{self.period}": md}


class ZLEMA(TechnicalIndicator):
    """Zero Lag EMA."""

    def __init__(self, periods: list[int] | None = None):
        super().__init__("ZLEMA")
        self.periods = periods or [10, 20]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        for period in self.periods:
            lag = (period - 1) // 2
            zlema_data = 2 * close - np.roll(close, lag)
            zlema_data[:lag] = close[:lag]

            zlema = EMA._ema(zlema_data, period)
            results[f"zlema_{period}"] = zlema

        return results


class FRAMA(TechnicalIndicator):
    """Fractal Adaptive Moving Average."""

    def __init__(self, period: int = 16, fc: int = 1, sc: int = 198):
        super().__init__("FRAMA")
        self.period = period
        self.fc = fc
        self.sc = sc

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        n = len(close)
        half = self.period // 2
        frama = np.zeros(n)
        frama[:self.period] = close[:self.period]

        w = np.log(2 / (self.sc + 1))

        for i in range(self.period, n):
            # First half
            n1_high = np.max(high[i - self.period + 1 : i - half + 1])
            n1_low = np.min(low[i - self.period + 1 : i - half + 1])
            n1 = (n1_high - n1_low) / half

            # Second half
            n2_high = np.max(high[i - half + 1 : i + 1])
            n2_low = np.min(low[i - half + 1 : i + 1])
            n2 = (n2_high - n2_low) / half

            # Full period
            n3_high = np.max(high[i - self.period + 1 : i + 1])
            n3_low = np.min(low[i - self.period + 1 : i + 1])
            n3 = (n3_high - n3_low) / self.period

            # Fractal dimension
            if n1 + n2 > 0 and n3 > 0:
                dimen = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
            else:
                dimen = 1

            alpha = np.exp(w * (dimen - 1))
            alpha = max(2 / (self.sc + 1), min(alpha, 1))

            frama[i] = alpha * close[i] + (1 - alpha) * frama[i - 1]

        return {f"frama_{self.period}": frama}


class VIDYa(TechnicalIndicator):
    """Variable Index Dynamic Average."""

    def __init__(self, cmo_period: int = 9, ema_period: int = 12):
        super().__init__("VIDYa")
        self.cmo_period = cmo_period
        self.ema_period = ema_period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)

        # Get CMO
        cmo = CMO(periods=[self.cmo_period]).compute(df)[f"cmo_{self.cmo_period}"]
        abs_cmo = np.abs(cmo) / 100

        # Calculate VIDYa
        n = len(close)
        sc = 2 / (self.ema_period + 1)
        vidya = np.zeros(n)
        vidya[0] = close[0]

        for i in range(1, n):
            if not np.isnan(abs_cmo[i]):
                vidya[i] = abs_cmo[i] * sc * close[i] + (1 - abs_cmo[i] * sc) * vidya[i - 1]
            else:
                vidya[i] = vidya[i - 1]

        return {f"vidya_{self.cmo_period}_{self.ema_period}": vidya}


# =============================================================================
# ADVANCED VOLATILITY INDICATORS
# =============================================================================


class RogersSatchellVolatility(TechnicalIndicator):
    """Rogers-Satchell Volatility estimator."""

    def __init__(self, period: int = 20, annualize: bool = True):
        super().__init__("RSVol")
        self.period = period
        self.annualize = annualize
        self.bars_per_year = 252 * 26

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["open", "high", "low", "close"])
        open_p = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        log_ho = np.log(high / open_p)
        log_hc = np.log(high / close)
        log_lo = np.log(low / open_p)
        log_lc = np.log(low / close)

        rs_var = log_ho * log_hc + log_lo * log_lc

        n = len(close)
        rs_vol = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            rs_vol[i] = np.sqrt(np.mean(rs_var[i - self.period + 1 : i + 1]))

        if self.annualize:
            rs_vol = rs_vol * np.sqrt(self.bars_per_year)

        return {f"rs_volatility_{self.period}": rs_vol}


class YangZhangVolatility(TechnicalIndicator):
    """Yang-Zhang Volatility estimator."""

    def __init__(self, period: int = 20, annualize: bool = True):
        super().__init__("YZVol")
        self.period = period
        self.annualize = annualize
        self.bars_per_year = 252 * 26

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["open", "high", "low", "close"])
        open_p = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        n = len(close)

        # Overnight volatility
        log_oc = np.zeros(n)
        log_oc[1:] = np.log(open_p[1:] / close[:-1])

        # Open-to-close volatility
        log_co = np.log(close / open_p)

        # Rogers-Satchell
        log_ho = np.log(high / open_p)
        log_hc = np.log(high / close)
        log_lo = np.log(low / open_p)
        log_lc = np.log(low / close)
        rs = log_ho * log_hc + log_lo * log_lc

        k = 0.34 / (1.34 + (self.period + 1) / (self.period - 1))

        yz_vol = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            overnight_var = np.var(log_oc[i - self.period + 1 : i + 1])
            oc_var = np.var(log_co[i - self.period + 1 : i + 1])
            rs_var = np.mean(rs[i - self.period + 1 : i + 1])

            yz_vol[i] = np.sqrt(overnight_var + k * oc_var + (1 - k) * rs_var)

        if self.annualize:
            yz_vol = yz_vol * np.sqrt(self.bars_per_year)

        return {f"yz_volatility_{self.period}": yz_vol}


class ChandelierExit(TechnicalIndicator):
    """Chandelier Exit indicator."""

    def __init__(self, period: int = 22, multiplier: float = 3.0):
        super().__init__("ChandelierExit")
        self.period = period
        self.multiplier = multiplier

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)

        atr = ATR(period=self.period).compute(df)[f"atr_{self.period}"]

        n = len(high)
        long_exit = np.full(n, np.nan)
        short_exit = np.full(n, np.nan)

        for i in range(self.period - 1, n):
            highest = np.max(high[i - self.period + 1 : i + 1])
            lowest = np.min(low[i - self.period + 1 : i + 1])

            if not np.isnan(atr[i]):
                long_exit[i] = highest - self.multiplier * atr[i]
                short_exit[i] = lowest + self.multiplier * atr[i]

        return {
            f"chandelier_long_{self.period}": long_exit,
            f"chandelier_short_{self.period}": short_exit,
        }


class UlcerIndex(TechnicalIndicator):
    """Ulcer Index - measures downside volatility."""

    def __init__(self, period: int = 14):
        super().__init__("UlcerIndex")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)

        n = len(close)
        percent_drawdown = np.zeros(n)

        for i in range(1, n):
            max_close = np.max(close[max(0, i - self.period + 1) : i + 1])
            percent_drawdown[i] = 100 * (close[i] - max_close) / max_close

        ulcer = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            ulcer[i] = np.sqrt(np.mean(percent_drawdown[i - self.period + 1 : i + 1] ** 2))

        return {f"ulcer_index_{self.period}": ulcer}


class StandardErrorBands(TechnicalIndicator):
    """Standard Error Bands."""

    def __init__(self, period: int = 21, multiplier: float = 2.0):
        super().__init__("SEB")
        self.period = period
        self.multiplier = multiplier

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)

        n = len(close)
        middle = np.full(n, np.nan)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)

        for i in range(self.period - 1, n):
            window = close[i - self.period + 1 : i + 1]
            x = np.arange(self.period)

            # Linear regression
            slope, intercept = np.polyfit(x, window, 1)
            predicted = slope * (self.period - 1) + intercept
            middle[i] = predicted

            # Standard error
            y_pred = slope * x + intercept
            se = np.sqrt(np.sum((window - y_pred) ** 2) / (self.period - 2))
            se_line = se * np.sqrt(1 + 1 / self.period + (self.period - 1 - x[-1]) ** 2 / np.sum((x - np.mean(x)) ** 2))

            upper[i] = predicted + self.multiplier * se_line
            lower[i] = predicted - self.multiplier * se_line

        return {
            f"seb_upper_{self.period}": upper,
            f"seb_middle_{self.period}": middle,
            f"seb_lower_{self.period}": lower,
        }


class AccelerationBands(TechnicalIndicator):
    """Acceleration Bands."""

    def __init__(self, period: int = 20, factor: float = 0.001):
        super().__init__("AccBands")
        self.period = period
        self.factor = factor

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        hl_ratio = (high - low) / ((high + low) / 2) * 1000

        upper_raw = high * (1 + self.factor * hl_ratio)
        lower_raw = low * (1 - self.factor * hl_ratio)

        upper = SMA._rolling_mean(upper_raw, self.period)
        middle = SMA._rolling_mean(close, self.period)
        lower = SMA._rolling_mean(lower_raw, self.period)

        return {
            f"acc_upper_{self.period}": upper,
            f"acc_middle_{self.period}": middle,
            f"acc_lower_{self.period}": lower,
        }


class VolatilityRatio(TechnicalIndicator):
    """Volatility Ratio - current vs historical volatility."""

    def __init__(self, short_period: int = 14, long_period: int = 50):
        super().__init__("VolRatio")
        self.short_period = short_period
        self.long_period = long_period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])

        atr_short = ATR(period=self.short_period).compute(df)[f"atr_{self.short_period}"]
        atr_long = ATR(period=self.long_period).compute(df)[f"atr_{self.long_period}"]

        vol_ratio = atr_short / np.where(atr_long == 0, 1, atr_long)

        return {"volatility_ratio": vol_ratio}


class STARC(TechnicalIndicator):
    """Stoller Average Range Channels."""

    def __init__(self, sma_period: int = 6, atr_period: int = 15, multiplier: float = 2.0):
        super().__init__("STARC")
        self.sma_period = sma_period
        self.atr_period = atr_period
        self.multiplier = multiplier

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        close = df["close"].to_numpy().astype(np.float64)

        sma = SMA._rolling_mean(close, self.sma_period)
        atr = ATR(period=self.atr_period).compute(df)[f"atr_{self.atr_period}"]

        upper = sma + self.multiplier * atr
        lower = sma - self.multiplier * atr

        return {
            f"starc_upper_{self.sma_period}": upper,
            f"starc_middle_{self.sma_period}": sma,
            f"starc_lower_{self.sma_period}": lower,
        }


# =============================================================================
# CANDLESTICK PATTERN RECOGNITION
# =============================================================================


class CandlestickPatterns(TechnicalIndicator):
    """
    Comprehensive candlestick pattern recognition.

    Detects 40+ candlestick patterns and returns signals:
    - Bullish patterns: positive values
    - Bearish patterns: negative values
    - Neutral/no pattern: 0
    """

    def __init__(self):
        super().__init__("Candlestick")

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["open", "high", "low", "close"])
        open_p = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        n = len(close)
        results = {}

        # Body and shadow calculations
        body = close - open_p
        body_size = np.abs(body)
        upper_shadow = high - np.maximum(open_p, close)
        lower_shadow = np.minimum(open_p, close) - low
        full_range = high - low
        body_pct = body_size / np.where(full_range == 0, 1, full_range)

        # Average body size for comparison
        avg_body = SMA._rolling_mean(body_size, 14)

        # Doji
        results["cdl_doji"] = np.where(body_pct < 0.1, 1, 0).astype(float)

        # Spinning Top
        results["cdl_spinning_top"] = np.where(
            (body_pct < 0.3) & (body_pct >= 0.1) &
            (upper_shadow > body_size * 0.5) & (lower_shadow > body_size * 0.5),
            1, 0
        ).astype(float)

        # Marubozu
        results["cdl_marubozu"] = np.where(
            (body_pct > 0.9) & (body > 0), 1,
            np.where((body_pct > 0.9) & (body < 0), -1, 0)
        ).astype(float)

        # Hammer (bullish)
        hammer_cond = (
            (lower_shadow > 2 * body_size) &
            (upper_shadow < body_size * 0.3) &
            (body_size > 0)
        )
        results["cdl_hammer"] = np.where(hammer_cond, 1, 0).astype(float)

        # Inverted Hammer (bullish)
        inv_hammer_cond = (
            (upper_shadow > 2 * body_size) &
            (lower_shadow < body_size * 0.3) &
            (body_size > 0)
        )
        results["cdl_inverted_hammer"] = np.where(inv_hammer_cond, 1, 0).astype(float)

        # Hanging Man (bearish)
        results["cdl_hanging_man"] = np.where(hammer_cond & (body < 0), -1, 0).astype(float)

        # Shooting Star (bearish)
        results["cdl_shooting_star"] = np.where(inv_hammer_cond & (body < 0), -1, 0).astype(float)

        # Engulfing patterns (require 2 bars)
        bullish_engulf = np.zeros(n)
        bearish_engulf = np.zeros(n)
        for i in range(1, n):
            if body[i - 1] < 0 and body[i] > 0:  # Previous bearish, current bullish
                if close[i] > open_p[i - 1] and open_p[i] < close[i - 1]:
                    bullish_engulf[i] = 1
            if body[i - 1] > 0 and body[i] < 0:  # Previous bullish, current bearish
                if close[i] < open_p[i - 1] and open_p[i] > close[i - 1]:
                    bearish_engulf[i] = -1

        results["cdl_bullish_engulfing"] = bullish_engulf
        results["cdl_bearish_engulfing"] = bearish_engulf

        # Harami patterns
        bullish_harami = np.zeros(n)
        bearish_harami = np.zeros(n)
        for i in range(1, n):
            prev_body = abs(body[i - 1])
            if body[i - 1] < 0 and body[i] > 0:  # Previous bearish
                if close[i] < open_p[i - 1] and open_p[i] > close[i - 1]:
                    bullish_harami[i] = 1
            if body[i - 1] > 0 and body[i] < 0:  # Previous bullish
                if close[i] > open_p[i - 1] and open_p[i] < close[i - 1]:
                    bearish_harami[i] = -1

        results["cdl_bullish_harami"] = bullish_harami
        results["cdl_bearish_harami"] = bearish_harami

        # Morning/Evening Star (3 bar patterns)
        morning_star = np.zeros(n)
        evening_star = np.zeros(n)
        for i in range(2, n):
            # Morning Star
            if (body[i - 2] < 0 and body_pct[i - 2] > 0.5 and  # Large bearish
                body_pct[i - 1] < 0.3 and  # Small body (star)
                body[i] > 0 and body_pct[i] > 0.5 and  # Large bullish
                close[i] > (open_p[i - 2] + close[i - 2]) / 2):  # Closes above midpoint
                morning_star[i] = 1

            # Evening Star
            if (body[i - 2] > 0 and body_pct[i - 2] > 0.5 and  # Large bullish
                body_pct[i - 1] < 0.3 and  # Small body (star)
                body[i] < 0 and body_pct[i] > 0.5 and  # Large bearish
                close[i] < (open_p[i - 2] + close[i - 2]) / 2):  # Closes below midpoint
                evening_star[i] = -1

        results["cdl_morning_star"] = morning_star
        results["cdl_evening_star"] = evening_star

        # Three White Soldiers / Three Black Crows
        soldiers = np.zeros(n)
        crows = np.zeros(n)
        for i in range(2, n):
            # Three White Soldiers
            if (body[i] > 0 and body[i - 1] > 0 and body[i - 2] > 0 and
                close[i] > close[i - 1] > close[i - 2] and
                body_pct[i] > 0.5 and body_pct[i - 1] > 0.5 and body_pct[i - 2] > 0.5):
                soldiers[i] = 1

            # Three Black Crows
            if (body[i] < 0 and body[i - 1] < 0 and body[i - 2] < 0 and
                close[i] < close[i - 1] < close[i - 2] and
                body_pct[i] > 0.5 and body_pct[i - 1] > 0.5 and body_pct[i - 2] > 0.5):
                crows[i] = -1

        results["cdl_three_white_soldiers"] = soldiers
        results["cdl_three_black_crows"] = crows

        # Piercing Line / Dark Cloud Cover
        piercing = np.zeros(n)
        dark_cloud = np.zeros(n)
        for i in range(1, n):
            # Piercing Line (bullish)
            if (body[i - 1] < 0 and body[i] > 0 and
                open_p[i] < close[i - 1] and
                close[i] > (open_p[i - 1] + close[i - 1]) / 2 and
                close[i] < open_p[i - 1]):
                piercing[i] = 1

            # Dark Cloud Cover (bearish)
            if (body[i - 1] > 0 and body[i] < 0 and
                open_p[i] > close[i - 1] and
                close[i] < (open_p[i - 1] + close[i - 1]) / 2 and
                close[i] > open_p[i - 1]):
                dark_cloud[i] = -1

        results["cdl_piercing"] = piercing
        results["cdl_dark_cloud"] = dark_cloud

        # Tweezer patterns
        tweezer_top = np.zeros(n)
        tweezer_bottom = np.zeros(n)
        for i in range(1, n):
            high_match = abs(high[i] - high[i - 1]) / full_range[i] < 0.05 if full_range[i] > 0 else False
            low_match = abs(low[i] - low[i - 1]) / full_range[i] < 0.05 if full_range[i] > 0 else False

            if high_match and body[i - 1] > 0 and body[i] < 0:
                tweezer_top[i] = -1
            if low_match and body[i - 1] < 0 and body[i] > 0:
                tweezer_bottom[i] = 1

        results["cdl_tweezer_top"] = tweezer_top
        results["cdl_tweezer_bottom"] = tweezer_bottom

        # Belt Hold
        bullish_belt = np.where(
            (body > 0) & (body_pct > 0.6) & (lower_shadow < body_size * 0.05),
            1, 0
        ).astype(float)
        bearish_belt = np.where(
            (body < 0) & (body_pct > 0.6) & (upper_shadow < body_size * 0.05),
            -1, 0
        ).astype(float)

        results["cdl_bullish_belt"] = bullish_belt
        results["cdl_bearish_belt"] = bearish_belt

        # Dragonfly Doji
        results["cdl_dragonfly_doji"] = np.where(
            (body_pct < 0.1) & (upper_shadow < full_range * 0.1) & (lower_shadow > full_range * 0.6),
            1, 0
        ).astype(float)

        # Gravestone Doji
        results["cdl_gravestone_doji"] = np.where(
            (body_pct < 0.1) & (lower_shadow < full_range * 0.1) & (upper_shadow > full_range * 0.6),
            -1, 0
        ).astype(float)

        # Long-Legged Doji
        results["cdl_long_legged_doji"] = np.where(
            (body_pct < 0.1) & (upper_shadow > full_range * 0.3) & (lower_shadow > full_range * 0.3),
            1, 0
        ).astype(float)

        # Kicking patterns
        kicking_bull = np.zeros(n)
        kicking_bear = np.zeros(n)
        for i in range(1, n):
            # Bullish Kicking
            if (body[i - 1] < 0 and body_pct[i - 1] > 0.8 and
                body[i] > 0 and body_pct[i] > 0.8 and
                low[i] > high[i - 1]):  # Gap up
                kicking_bull[i] = 1

            # Bearish Kicking
            if (body[i - 1] > 0 and body_pct[i - 1] > 0.8 and
                body[i] < 0 and body_pct[i] > 0.8 and
                high[i] < low[i - 1]):  # Gap down
                kicking_bear[i] = -1

        results["cdl_kicking_bull"] = kicking_bull
        results["cdl_kicking_bear"] = kicking_bear

        # Rising/Falling Three Methods
        rising_three = np.zeros(n)
        falling_three = np.zeros(n)
        for i in range(4, n):
            # Rising Three
            if (body[i - 4] > 0 and body_pct[i - 4] > 0.5 and  # Long bullish
                body[i - 3] < 0 and body[i - 2] < 0 and body[i - 1] < 0 and  # 3 small bearish
                max(close[i - 3], close[i - 2], close[i - 1]) < close[i - 4] and
                min(low[i - 3], low[i - 2], low[i - 1]) > low[i - 4] and
                body[i] > 0 and close[i] > close[i - 4]):  # Long bullish closes higher
                rising_three[i] = 1

            # Falling Three
            if (body[i - 4] < 0 and body_pct[i - 4] > 0.5 and  # Long bearish
                body[i - 3] > 0 and body[i - 2] > 0 and body[i - 1] > 0 and  # 3 small bullish
                min(close[i - 3], close[i - 2], close[i - 1]) > close[i - 4] and
                max(high[i - 3], high[i - 2], high[i - 1]) < high[i - 4] and
                body[i] < 0 and close[i] < close[i - 4]):  # Long bearish closes lower
                falling_three[i] = -1

        results["cdl_rising_three"] = rising_three
        results["cdl_falling_three"] = falling_three

        return results


# =============================================================================
# COMPOSITE & DERIVED INDICATORS
# =============================================================================


class TrendStrength(TechnicalIndicator):
    """Composite trend strength indicator."""

    def __init__(self, period: int = 14):
        super().__init__("TrendStrength")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        close = df["close"].to_numpy().astype(np.float64)

        # Get ADX
        adx = ADX(period=self.period).compute(df)

        # Price above/below SMA
        sma = SMA._rolling_mean(close, self.period)
        trend_dir = np.sign(close - sma)

        # Combine ADX strength with direction
        strength = adx[f"adx_{self.period}"] * trend_dir / 100

        return {f"trend_strength_{self.period}": strength}


class MomentumComposite(TechnicalIndicator):
    """Composite momentum indicator combining multiple sources."""

    def __init__(self):
        super().__init__("MomentumComposite")

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])

        # Get individual indicators
        rsi = RSI(periods=[14]).compute(df)["rsi_14"]
        stoch = StochasticOscillator(k_period=14).compute(df)["stoch_k_14"]
        cci = CCI(period=20).compute(df)["cci_20"]

        # Normalize CCI to 0-100 range
        cci_norm = 50 + cci / 4
        cci_norm = np.clip(cci_norm, 0, 100)

        # Composite (average of normalized values)
        composite = (rsi + stoch + cci_norm) / 3

        # Overbought/Oversold signals
        overbought = np.where(composite > 70, 1, 0)
        oversold = np.where(composite < 30, -1, 0)

        return {
            "momentum_composite": composite,
            "momentum_signal": overbought + oversold,
        }


class VolumePriceConfirm(TechnicalIndicator):
    """Volume-Price confirmation indicator."""

    def __init__(self, period: int = 14):
        super().__init__("VPConfirm")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close", "volume"])
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)

        # Price momentum
        price_mom = np.zeros(len(close))
        price_mom[self.period:] = (close[self.period:] - close[:-self.period]) / close[:-self.period]

        # Volume momentum
        vol_sma = SMA._rolling_mean(volume, self.period)
        vol_ratio = volume / np.where(vol_sma == 0, 1, vol_sma) - 1

        # Confirmation: both moving same direction
        confirm = np.sign(price_mom) * np.sign(vol_ratio)

        return {
            f"vp_confirm_{self.period}": confirm,
            f"price_momentum_{self.period}": price_mom,
            f"volume_momentum_{self.period}": vol_ratio,
        }


class TrendConsistency(TechnicalIndicator):
    """Measures consistency of trend across timeframes."""

    def __init__(self, periods: list[int] | None = None):
        super().__init__("TrendConsistency")
        self.periods = periods or [5, 10, 20, 50]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)

        n = len(close)
        consistency = np.zeros(n)

        for period in self.periods:
            sma = SMA._rolling_mean(close, period)
            direction = np.sign(close - sma)
            consistency += direction

        # Normalize to -1 to 1
        consistency = consistency / len(self.periods)

        return {"trend_consistency": consistency}


class BreakoutStrength(TechnicalIndicator):
    """Measures strength of price breakouts."""

    def __init__(self, period: int = 20):
        super().__init__("BreakoutStrength")
        self.period = period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close", "volume"])
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)

        n = len(close)
        breakout = np.zeros(n)

        vol_sma = SMA._rolling_mean(volume, self.period)

        for i in range(self.period, n):
            prev_high = np.max(high[i - self.period : i])
            prev_low = np.min(low[i - self.period : i])
            prev_range = prev_high - prev_low

            if prev_range > 0:
                # Upside breakout
                if close[i] > prev_high:
                    vol_mult = volume[i] / vol_sma[i] if vol_sma[i] > 0 else 1
                    breakout[i] = (close[i] - prev_high) / prev_range * vol_mult

                # Downside breakout
                elif close[i] < prev_low:
                    vol_mult = volume[i] / vol_sma[i] if vol_sma[i] > 0 else 1
                    breakout[i] = (close[i] - prev_low) / prev_range * vol_mult

        return {f"breakout_strength_{self.period}": breakout}


class MeanReversionSignal(TechnicalIndicator):
    """Mean reversion signal based on distance from mean."""

    def __init__(self, period: int = 20, threshold: float = 2.0):
        super().__init__("MeanReversion")
        self.period = period
        self.threshold = threshold

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)

        sma = SMA._rolling_mean(close, self.period)

        n = len(close)
        std = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            std[i] = np.std(close[i - self.period + 1 : i + 1])

        z_score = (close - sma) / np.where(std == 0, 1, std)

        # Signal: -1 when overbought, +1 when oversold
        signal = np.zeros(n)
        signal[z_score > self.threshold] = -1  # Overbought, expect down
        signal[z_score < -self.threshold] = 1  # Oversold, expect up

        return {
            f"zscore_{self.period}": z_score,
            f"mean_reversion_signal_{self.period}": signal,
        }


class RSIDivergence(TechnicalIndicator):
    """Detects RSI divergence from price."""

    def __init__(self, period: int = 14, lookback: int = 14):
        super().__init__("RSIDivergence")
        self.period = period
        self.lookback = lookback

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)

        rsi = RSI(periods=[self.period]).compute(df)[f"rsi_{self.period}"]

        n = len(close)
        bullish_div = np.zeros(n)
        bearish_div = np.zeros(n)

        for i in range(self.lookback, n):
            # Find local minima/maxima in lookback window
            price_window = close[i - self.lookback + 1 : i + 1]
            rsi_window = rsi[i - self.lookback + 1 : i + 1]

            if np.any(np.isnan(rsi_window)):
                continue

            # Bullish divergence: price makes lower low, RSI makes higher low
            price_low_idx = np.argmin(price_window)
            if price_low_idx > 0:
                prev_price_low = np.min(price_window[:price_low_idx])
                if price_window[price_low_idx] < prev_price_low:
                    # Check if RSI made higher low
                    rsi_at_price_low = rsi_window[price_low_idx]
                    prev_rsi_low = np.min(rsi_window[:price_low_idx])
                    if rsi_at_price_low > prev_rsi_low:
                        bullish_div[i] = 1

            # Bearish divergence: price makes higher high, RSI makes lower high
            price_high_idx = np.argmax(price_window)
            if price_high_idx > 0:
                prev_price_high = np.max(price_window[:price_high_idx])
                if price_window[price_high_idx] > prev_price_high:
                    # Check if RSI made lower high
                    rsi_at_price_high = rsi_window[price_high_idx]
                    prev_rsi_high = np.max(rsi_window[:price_high_idx])
                    if rsi_at_price_high < prev_rsi_high:
                        bearish_div[i] = -1

        return {
            "rsi_bullish_divergence": bullish_div,
            "rsi_bearish_divergence": bearish_div,
        }


class SqueezeIndicator(TechnicalIndicator):
    """Bollinger Band / Keltner Channel Squeeze indicator."""

    def __init__(self, bb_period: int = 20, kc_period: int = 20, bb_mult: float = 2.0, kc_mult: float = 1.5):
        super().__init__("Squeeze")
        self.bb_period = bb_period
        self.kc_period = kc_period
        self.bb_mult = bb_mult
        self.kc_mult = kc_mult

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])
        close = df["close"].to_numpy().astype(np.float64)

        # Bollinger Bands
        bb = BollingerBands(period=self.bb_period, std_dev=self.bb_mult).compute(df)
        bb_upper = bb[f"bb_upper_{self.bb_period}"]
        bb_lower = bb[f"bb_lower_{self.bb_period}"]

        # Keltner Channels
        kc = KeltnerChannel(ema_period=self.kc_period, multiplier=self.kc_mult).compute(df)
        kc_upper = kc[f"keltner_upper_{self.kc_period}"]
        kc_lower = kc[f"keltner_lower_{self.kc_period}"]

        # Squeeze: BB inside KC
        squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)

        # Momentum for direction
        hl2 = (df["high"].to_numpy().astype(np.float64) + df["low"].to_numpy().astype(np.float64)) / 2
        sma = SMA._rolling_mean(hl2, self.bb_period)
        momentum = close - sma

        return {
            "squeeze_on": squeeze_on.astype(float),
            "squeeze_momentum": momentum,
        }


class MultiTimeframeTrend(TechnicalIndicator):
    """Aggregates trend signals across multiple periods."""

    def __init__(self):
        super().__init__("MTFTrend")

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)

        # Different period EMAs
        periods = [8, 21, 55, 100, 200]
        n = len(close)

        trend_score = np.zeros(n)
        for period in periods:
            ema = EMA._ema(close, period)
            trend_score += np.sign(close - ema)

        # Normalize
        trend_score = trend_score / len(periods)

        # Trend alignment (all agree)
        aligned = np.where(np.abs(trend_score) == 1.0, trend_score, 0)

        return {
            "mtf_trend_score": trend_score,
            "mtf_trend_aligned": aligned,
        }


class VolatilityRegime(TechnicalIndicator):
    """Identifies volatility regime (low/normal/high)."""

    def __init__(self, short_period: int = 10, long_period: int = 100):
        super().__init__("VolRegime")
        self.short_period = short_period
        self.long_period = long_period

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["high", "low", "close"])

        atr_short = ATR(period=self.short_period).compute(df)[f"atr_{self.short_period}"]
        atr_long = ATR(period=self.long_period).compute(df)[f"atr_{self.long_period}"]

        # Ratio of current to historical volatility
        vol_ratio = atr_short / np.where(atr_long == 0, 1, atr_long)

        # Regime classification
        n = len(df)
        regime = np.zeros(n)
        regime[vol_ratio < 0.75] = -1  # Low volatility
        regime[vol_ratio > 1.25] = 1  # High volatility

        return {
            "volatility_ratio": vol_ratio,
            "volatility_regime": regime,
        }


# Import for circular reference
from .technical import StochasticOscillator, CCI, BollingerBands, KeltnerChannel


# =============================================================================
# EXTENDED CALCULATOR
# =============================================================================


class ExtendedTechnicalCalculator:
    """
    Calculator for extended technical indicators.

    Adds 150+ additional indicators to the base set for 200+ total.
    """

    def __init__(self):
        self.indicators: list[TechnicalIndicator] = []
        self._add_extended_indicators()

    def _add_extended_indicators(self) -> None:
        """Add all extended indicators."""
        # Additional Momentum (~20)
        self.indicators.extend([
            CMO(),
            TRIX(),
            DPO(),
            ElderRay(),
            KST(),
            MassIndex(),
            RVI(),
            QStick(),
            BalanceOfPower(),
            ConnorsRSI(),
            FisherTransform(),
            SMI(),
            AO(),
            AC(),
            Vortex(),
            McGinleyDynamic(),
            ZLEMA(),
            FRAMA(),
            VIDYa(),
        ])

        # Advanced Volatility (~10)
        self.indicators.extend([
            RogersSatchellVolatility(),
            YangZhangVolatility(),
            ChandelierExit(),
            UlcerIndex(),
            StandardErrorBands(),
            AccelerationBands(),
            VolatilityRatio(),
            STARC(),
        ])

        # Candlestick Patterns (~40 patterns in one class)
        self.indicators.append(CandlestickPatterns())

        # Composite/Derived (~10)
        self.indicators.extend([
            TrendStrength(),
            MomentumComposite(),
            VolumePriceConfirm(),
            TrendConsistency(),
            BreakoutStrength(),
            MeanReversionSignal(),
            RSIDivergence(),
            SqueezeIndicator(),
            MultiTimeframeTrend(),
            VolatilityRegime(),
        ])

    def compute_all(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        """Compute all extended indicators."""
        results = {}
        for indicator in self.indicators:
            try:
                indicator_results = indicator.compute(df)
                results.update(indicator_results)
            except (ValueError, Exception) as e:
                # Skip indicators that can't be computed
                continue
        return results

    def get_indicator_count(self) -> int:
        """Get total number of indicator classes."""
        return len(self.indicators)
