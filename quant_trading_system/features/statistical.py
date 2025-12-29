"""
Statistical features module.

Implements statistical features organized by category:
- Returns & distributions (simple, log, excess, risk-adjusted)
- Distribution metrics (rolling mean, std, skewness, kurtosis)
- Time series features (autocorrelation, stationarity tests)
- Mean reversion metrics (half-life, OU parameters)
- Regression features (slope, R², residuals)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy import stats


@dataclass
class StatisticalResult:
    """Result container for statistical feature calculations."""

    name: str
    values: np.ndarray
    params: dict[str, Any]
    metadata: dict[str, Any] | None = None


class StatisticalFeature(ABC):
    """Abstract base class for statistical features."""

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
# RETURNS & DISTRIBUTIONS
# =============================================================================


class SimpleReturns(StatisticalFeature):
    """Simple returns over various periods."""

    def __init__(self, periods: list[int] | None = None):
        super().__init__("SimpleReturns")
        self.periods = periods or [1, 5, 10, 20, 60]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        for period in self.periods:
            returns = np.full(len(close), np.nan)
            returns[period:] = (close[period:] - close[:-period]) / close[:-period]
            results[f"return_{period}"] = returns

        return results


class LogReturns(StatisticalFeature):
    """Log returns over various periods."""

    def __init__(self, periods: list[int] | None = None):
        super().__init__("LogReturns")
        self.periods = periods or [1, 5, 10, 20, 60]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        log_close = np.log(close)
        for period in self.periods:
            log_returns = np.full(len(close), np.nan)
            log_returns[period:] = log_close[period:] - log_close[:-period]
            results[f"log_return_{period}"] = log_returns

        return results


class ExcessReturns(StatisticalFeature):
    """Excess returns over a benchmark (requires benchmark column)."""

    def __init__(self, periods: list[int] | None = None, benchmark_col: str = "benchmark"):
        super().__init__("ExcessReturns")
        self.periods = periods or [1, 5, 20]
        self.benchmark_col = benchmark_col

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        # If benchmark column exists, compute excess returns
        if self.benchmark_col in df.columns:
            benchmark = df[self.benchmark_col].to_numpy().astype(np.float64)

            for period in self.periods:
                asset_ret = np.full(len(close), np.nan)
                bench_ret = np.full(len(close), np.nan)

                asset_ret[period:] = (close[period:] - close[:-period]) / close[:-period]
                bench_ret[period:] = (benchmark[period:] - benchmark[:-period]) / benchmark[:-period]

                excess = asset_ret - bench_ret
                results[f"excess_return_{period}"] = excess
        else:
            # Just compute regular returns if no benchmark
            for period in self.periods:
                returns = np.full(len(close), np.nan)
                returns[period:] = (close[period:] - close[:-period]) / close[:-period]
                results[f"return_{period}"] = returns

        return results


class RiskAdjustedReturns(StatisticalFeature):
    """Risk-adjusted returns (return / volatility)."""

    def __init__(self, periods: list[int] | None = None, vol_window: int = 20):
        super().__init__("RiskAdjustedReturns")
        self.periods = periods or [1, 5, 20]
        self.vol_window = vol_window

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        # Compute rolling volatility
        log_returns = np.diff(np.log(close), prepend=np.nan)
        rolling_vol = self._rolling_std(log_returns, self.vol_window)

        for period in self.periods:
            returns = np.full(len(close), np.nan)
            returns[period:] = (close[period:] - close[:-period]) / close[:-period]
            risk_adj = returns / np.where(rolling_vol == 0, 1e-10, rolling_vol)
            results[f"risk_adj_return_{period}"] = risk_adj

        return results

    @staticmethod
    def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling standard deviation."""
        result = np.full(len(arr), np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.nanstd(arr[i - window + 1 : i + 1], ddof=1)
        return result


# =============================================================================
# DISTRIBUTION METRICS
# =============================================================================


class RollingMean(StatisticalFeature):
    """Rolling mean of returns."""

    def __init__(self, windows: list[int] | None = None):
        super().__init__("RollingMean")
        self.windows = windows or [5, 10, 20, 60]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        returns = np.diff(np.log(close), prepend=np.nan)
        results = {}

        for window in self.windows:
            roll_mean = np.full(len(close), np.nan)
            for i in range(window - 1, len(close)):
                roll_mean[i] = np.nanmean(returns[i - window + 1 : i + 1])
            results[f"rolling_mean_{window}"] = roll_mean

        return results


class RollingStd(StatisticalFeature):
    """Rolling standard deviation of returns."""

    def __init__(self, windows: list[int] | None = None):
        super().__init__("RollingStd")
        self.windows = windows or [5, 10, 20, 60]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        returns = np.diff(np.log(close), prepend=np.nan)
        results = {}

        for window in self.windows:
            roll_std = np.full(len(close), np.nan)
            for i in range(window - 1, len(close)):
                roll_std[i] = np.nanstd(returns[i - window + 1 : i + 1], ddof=1)
            results[f"rolling_std_{window}"] = roll_std

        return results


class RollingVar(StatisticalFeature):
    """Rolling variance of returns."""

    def __init__(self, windows: list[int] | None = None):
        super().__init__("RollingVar")
        self.windows = windows or [5, 10, 20, 60]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        returns = np.diff(np.log(close), prepend=np.nan)
        results = {}

        for window in self.windows:
            roll_var = np.full(len(close), np.nan)
            for i in range(window - 1, len(close)):
                roll_var[i] = np.nanvar(returns[i - window + 1 : i + 1], ddof=1)
            results[f"rolling_var_{window}"] = roll_var

        return results


class RollingSkewness(StatisticalFeature):
    """Rolling skewness of returns."""

    def __init__(self, windows: list[int] | None = None):
        super().__init__("RollingSkewness")
        self.windows = windows or [20, 60, 120]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        returns = np.diff(np.log(close), prepend=np.nan)
        results = {}

        for window in self.windows:
            roll_skew = np.full(len(close), np.nan)
            for i in range(window - 1, len(close)):
                data = returns[i - window + 1 : i + 1]
                valid_data = data[~np.isnan(data)]
                if len(valid_data) >= 3:
                    roll_skew[i] = stats.skew(valid_data, bias=False)
            results[f"rolling_skew_{window}"] = roll_skew

        return results


class RollingKurtosis(StatisticalFeature):
    """Rolling kurtosis of returns."""

    def __init__(self, windows: list[int] | None = None):
        super().__init__("RollingKurtosis")
        self.windows = windows or [20, 60, 120]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        returns = np.diff(np.log(close), prepend=np.nan)
        results = {}

        for window in self.windows:
            roll_kurt = np.full(len(close), np.nan)
            for i in range(window - 1, len(close)):
                data = returns[i - window + 1 : i + 1]
                valid_data = data[~np.isnan(data)]
                if len(valid_data) >= 4:
                    roll_kurt[i] = stats.kurtosis(valid_data, bias=False)
            results[f"rolling_kurt_{window}"] = roll_kurt

        return results


class RollingQuantiles(StatisticalFeature):
    """Rolling quantiles of returns."""

    def __init__(self, window: int = 60, quantiles: list[float] | None = None):
        super().__init__("RollingQuantiles")
        self.window = window
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        returns = np.diff(np.log(close), prepend=np.nan)
        results = {}

        for q in self.quantiles:
            roll_q = np.full(len(close), np.nan)
            for i in range(self.window - 1, len(close)):
                data = returns[i - self.window + 1 : i + 1]
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    roll_q[i] = np.quantile(valid_data, q)
            q_name = int(q * 100)
            results[f"rolling_q{q_name}_{self.window}"] = roll_q

        return results


class ZScore(StatisticalFeature):
    """Z-score of current price relative to rolling window."""

    def __init__(self, windows: list[int] | None = None):
        super().__init__("ZScore")
        self.windows = windows or [20, 60, 120]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        for window in self.windows:
            zscore = np.full(len(close), np.nan)
            for i in range(window - 1, len(close)):
                data = close[i - window + 1 : i + 1]
                mean = np.mean(data)
                std = np.std(data, ddof=1)
                if std > 0:
                    zscore[i] = (close[i] - mean) / std
            results[f"zscore_{window}"] = zscore

        return results


# =============================================================================
# TAIL METRICS
# =============================================================================


class RollingVaR(StatisticalFeature):
    """Rolling Value at Risk."""

    def __init__(self, window: int = 60, confidence_levels: list[float] | None = None):
        super().__init__("RollingVaR")
        self.window = window
        self.confidence_levels = confidence_levels or [0.95, 0.99]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        returns = np.diff(np.log(close), prepend=np.nan)
        results = {}

        for conf in self.confidence_levels:
            var_arr = np.full(len(close), np.nan)
            for i in range(self.window - 1, len(close)):
                data = returns[i - self.window + 1 : i + 1]
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    var_arr[i] = np.quantile(valid_data, 1 - conf)
            conf_name = int(conf * 100)
            results[f"var_{conf_name}_{self.window}"] = var_arr

        return results


class RollingCVaR(StatisticalFeature):
    """Rolling Conditional Value at Risk (Expected Shortfall)."""

    def __init__(self, window: int = 60, confidence_level: float = 0.95):
        super().__init__("RollingCVaR")
        self.window = window
        self.confidence_level = confidence_level

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        returns = np.diff(np.log(close), prepend=np.nan)
        results = {}

        cvar_arr = np.full(len(close), np.nan)
        for i in range(self.window - 1, len(close)):
            data = returns[i - self.window + 1 : i + 1]
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                var_threshold = np.quantile(valid_data, 1 - self.confidence_level)
                tail_losses = valid_data[valid_data <= var_threshold]
                if len(tail_losses) > 0:
                    cvar_arr[i] = np.mean(tail_losses)

        conf_name = int(self.confidence_level * 100)
        results[f"cvar_{conf_name}_{self.window}"] = cvar_arr

        return results


class TailRatio(StatisticalFeature):
    """Tail ratio (ratio of positive to negative tail)."""

    def __init__(self, window: int = 60, percentile: float = 0.05):
        super().__init__("TailRatio")
        self.window = window
        self.percentile = percentile

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        returns = np.diff(np.log(close), prepend=np.nan)
        results = {}

        tail_ratio = np.full(len(close), np.nan)
        for i in range(self.window - 1, len(close)):
            data = returns[i - self.window + 1 : i + 1]
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                upper_tail = np.quantile(valid_data, 1 - self.percentile)
                lower_tail = np.quantile(valid_data, self.percentile)
                if lower_tail != 0:
                    tail_ratio[i] = abs(upper_tail / lower_tail)

        results[f"tail_ratio_{self.window}"] = tail_ratio

        return results


# =============================================================================
# TIME SERIES FEATURES
# =============================================================================


class Autocorrelation(StatisticalFeature):
    """Autocorrelation at various lags."""

    def __init__(self, window: int = 60, lags: list[int] | None = None):
        super().__init__("Autocorrelation")
        self.window = window
        self.lags = lags or [1, 2, 3, 5, 10, 20]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        returns = np.diff(np.log(close), prepend=np.nan)
        results = {}

        for lag in self.lags:
            acf = np.full(len(close), np.nan)
            for i in range(self.window + lag - 1, len(close)):
                data = returns[i - self.window - lag + 1 : i + 1]
                valid_data = data[~np.isnan(data)]
                if len(valid_data) >= self.window:
                    acf[i] = self._autocorr(valid_data, lag)
            results[f"acf_lag{lag}_{self.window}"] = acf

        return results

    @staticmethod
    def _autocorr(x: np.ndarray, lag: int) -> float:
        """Compute autocorrelation at a specific lag."""
        if len(x) <= lag:
            return np.nan
        mean = np.mean(x)
        var = np.var(x)
        if var == 0:
            return 0
        n = len(x)
        cov = np.sum((x[: n - lag] - mean) * (x[lag:] - mean)) / n
        return cov / var


class PartialAutocorrelation(StatisticalFeature):
    """Partial autocorrelation (PACF) at various lags."""

    def __init__(self, window: int = 60, lags: list[int] | None = None):
        super().__init__("PACF")
        self.window = window
        self.lags = lags or [1, 2, 3, 5]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        returns = np.diff(np.log(close), prepend=np.nan)
        results = {}

        max_lag = max(self.lags)
        for lag in self.lags:
            pacf = np.full(len(close), np.nan)
            for i in range(self.window + max_lag - 1, len(close)):
                data = returns[i - self.window - max_lag + 1 : i + 1]
                valid_data = data[~np.isnan(data)]
                if len(valid_data) >= self.window:
                    pacf[i] = self._pacf(valid_data, lag)
            results[f"pacf_lag{lag}_{self.window}"] = pacf

        return results

    def _pacf(self, x: np.ndarray, lag: int) -> float:
        """Compute partial autocorrelation using Durbin-Levinson."""
        if lag == 0:
            return 1.0
        if lag == 1:
            return Autocorrelation._autocorr(x, 1)

        # Yule-Walker approach for PACF
        acf_values = [Autocorrelation._autocorr(x, i) for i in range(lag + 1)]
        r = np.array(acf_values[1:])
        R = np.zeros((lag, lag))
        for i in range(lag):
            for j in range(lag):
                R[i, j] = acf_values[abs(i - j)]

        try:
            phi = np.linalg.solve(R, r)
            return phi[-1]
        except np.linalg.LinAlgError:
            return np.nan


class HurstExponent(StatisticalFeature):
    """Hurst exponent for long-range dependence detection."""

    def __init__(self, window: int = 100, min_lag: int = 2, max_lag: int = 20):
        super().__init__("HurstExponent")
        self.window = window
        self.min_lag = min_lag
        self.max_lag = max_lag

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        hurst = np.full(len(close), np.nan)
        for i in range(self.window - 1, len(close)):
            data = close[i - self.window + 1 : i + 1]
            hurst[i] = self._compute_hurst(data)

        results[f"hurst_{self.window}"] = hurst

        return results

    def _compute_hurst(self, ts: np.ndarray) -> float:
        """Compute Hurst exponent using R/S analysis."""
        if len(ts) < self.max_lag * 2:
            return np.nan

        lags = range(self.min_lag, self.max_lag)
        rs = []

        for lag in lags:
            rs_values = []
            for start in range(0, len(ts) - lag, lag):
                subset = ts[start : start + lag]
                if len(subset) < lag:
                    continue
                mean = np.mean(subset)
                deviations = np.cumsum(subset - mean)
                R = np.max(deviations) - np.min(deviations)
                S = np.std(subset, ddof=1)
                if S > 0:
                    rs_values.append(R / S)
            if rs_values:
                rs.append(np.mean(rs_values))
            else:
                rs.append(np.nan)

        valid_idx = ~np.isnan(rs)
        if np.sum(valid_idx) < 2:
            return np.nan

        log_lags = np.log(list(lags))[valid_idx]
        log_rs = np.log(np.array(rs)[valid_idx])

        try:
            slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
            return slope
        except ValueError:
            return np.nan


class ADFStatistic(StatisticalFeature):
    """Augmented Dickey-Fuller test statistic (rolling)."""

    def __init__(self, window: int = 60):
        super().__init__("ADF")
        self.window = window

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        adf_stat = np.full(len(close), np.nan)
        for i in range(self.window - 1, len(close)):
            data = close[i - self.window + 1 : i + 1]
            adf_stat[i] = self._adf_statistic(data)

        results[f"adf_stat_{self.window}"] = adf_stat

        return results

    def _adf_statistic(self, ts: np.ndarray) -> float:
        """Compute simplified ADF test statistic."""
        if len(ts) < 10:
            return np.nan

        # Simple ADF: regress diff(y) on y_lag1
        y_diff = np.diff(ts)
        y_lag = ts[:-1]

        if len(y_diff) < 2:
            return np.nan

        # Add intercept
        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        y = y_diff

        try:
            # OLS regression
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            gamma = coeffs[1]

            # Calculate standard error
            residuals = y - X @ coeffs
            n = len(y)
            k = 2
            sigma2 = np.sum(residuals**2) / (n - k)
            var_beta = sigma2 * np.linalg.inv(X.T @ X)
            se_gamma = np.sqrt(var_beta[1, 1])

            if se_gamma > 0:
                return gamma / se_gamma
            return np.nan
        except (np.linalg.LinAlgError, ValueError):
            return np.nan


# =============================================================================
# MEAN REVERSION METRICS
# =============================================================================


class HalfLife(StatisticalFeature):
    """Half-life of mean reversion."""

    def __init__(self, window: int = 60):
        super().__init__("HalfLife")
        self.window = window

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        half_life = np.full(len(close), np.nan)
        for i in range(self.window - 1, len(close)):
            data = close[i - self.window + 1 : i + 1]
            half_life[i] = self._compute_half_life(data)

        results[f"half_life_{self.window}"] = half_life

        return results

    def _compute_half_life(self, ts: np.ndarray) -> float:
        """Compute half-life from mean reversion speed."""
        if len(ts) < 10:
            return np.nan

        # Regress diff(y) on y_lag
        y_diff = np.diff(ts)
        y_lag = ts[:-1]

        # Subtract mean
        y_lag_dm = y_lag - np.mean(y_lag)

        if len(y_diff) < 2:
            return np.nan

        try:
            # OLS: y_diff = theta * y_lag_dm + epsilon
            theta = np.sum(y_lag_dm * y_diff) / np.sum(y_lag_dm**2)

            if theta >= 0:  # No mean reversion
                return np.nan

            half_life = -np.log(2) / theta
            return min(half_life, 1000)  # Cap at reasonable value
        except (ValueError, ZeroDivisionError):
            return np.nan


class OUParameters(StatisticalFeature):
    """Ornstein-Uhlenbeck process parameters (theta, mu, sigma)."""

    def __init__(self, window: int = 60):
        super().__init__("OUParams")
        self.window = window

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        theta = np.full(len(close), np.nan)
        mu = np.full(len(close), np.nan)
        sigma = np.full(len(close), np.nan)

        for i in range(self.window - 1, len(close)):
            data = close[i - self.window + 1 : i + 1]
            params = self._fit_ou(data)
            if params is not None:
                theta[i], mu[i], sigma[i] = params

        results[f"ou_theta_{self.window}"] = theta
        results[f"ou_mu_{self.window}"] = mu
        results[f"ou_sigma_{self.window}"] = sigma

        return results

    def _fit_ou(self, ts: np.ndarray) -> tuple[float, float, float] | None:
        """Fit Ornstein-Uhlenbeck process parameters."""
        if len(ts) < 10:
            return None

        # dt = 1 (one bar)
        dt = 1.0
        y_diff = np.diff(ts)
        y_lag = ts[:-1]

        if len(y_diff) < 2:
            return None

        try:
            # Add intercept for regression
            X = np.column_stack([np.ones(len(y_lag)), y_lag])
            coeffs = np.linalg.lstsq(X, y_diff, rcond=None)[0]

            a = coeffs[0]
            b = coeffs[1]

            if b >= 0:  # No mean reversion
                return None

            theta = -b / dt
            mu = a / (theta * dt) if theta > 0 else np.nan

            residuals = y_diff - X @ coeffs
            sigma = np.std(residuals, ddof=2) * np.sqrt(2 * theta / (1 - np.exp(-2 * theta * dt)))

            return theta, mu, sigma
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            return None


class MeanReversionStrength(StatisticalFeature):
    """Mean reversion strength indicator."""

    def __init__(self, window: int = 60):
        super().__init__("MRStrength")
        self.window = window

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        strength = np.full(len(close), np.nan)
        for i in range(self.window - 1, len(close)):
            data = close[i - self.window + 1 : i + 1]
            strength[i] = self._mr_strength(data)

        results[f"mr_strength_{self.window}"] = strength

        return results

    def _mr_strength(self, ts: np.ndarray) -> float:
        """Calculate mean reversion strength score."""
        if len(ts) < 10:
            return np.nan

        # Detrend
        x = np.arange(len(ts))
        try:
            slope, intercept = np.polyfit(x, ts, 1)
            detrended = ts - (slope * x + intercept)
        except (np.linalg.LinAlgError, ValueError):
            detrended = ts - np.mean(ts)

        # Mean reversion strength based on sign changes
        detrended_diff = np.diff(np.sign(detrended))
        sign_changes = np.sum(np.abs(detrended_diff) > 0)

        # Expected for random walk
        expected_changes = len(ts) / 2

        if expected_changes > 0:
            return sign_changes / expected_changes
        return np.nan


# =============================================================================
# REGRESSION FEATURES
# =============================================================================


class LinearRegressionSlope(StatisticalFeature):
    """Rolling linear regression slope."""

    def __init__(self, windows: list[int] | None = None):
        super().__init__("LinRegSlope")
        self.windows = windows or [10, 20, 50]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        for window in self.windows:
            slope = np.full(len(close), np.nan)
            for i in range(window - 1, len(close)):
                data = close[i - window + 1 : i + 1]
                x = np.arange(window)
                try:
                    slope[i], _ = np.polyfit(x, data, 1)
                except (np.linalg.LinAlgError, ValueError):
                    pass
            results[f"linreg_slope_{window}"] = slope

        return results


class RSquared(StatisticalFeature):
    """Rolling R-squared of linear regression."""

    def __init__(self, windows: list[int] | None = None):
        super().__init__("RSquared")
        self.windows = windows or [10, 20, 50]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        for window in self.windows:
            r2 = np.full(len(close), np.nan)
            for i in range(window - 1, len(close)):
                data = close[i - window + 1 : i + 1]
                x = np.arange(window)
                try:
                    _, _, r_value, _, _ = stats.linregress(x, data)
                    r2[i] = r_value**2
                except (ValueError, RuntimeWarning):
                    pass
            results[f"r_squared_{window}"] = r2

        return results


class LinearRegressionResiduals(StatisticalFeature):
    """Residuals from linear regression (deviation from trend)."""

    def __init__(self, windows: list[int] | None = None):
        super().__init__("LinRegResid")
        self.windows = windows or [20, 50]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        for window in self.windows:
            residual = np.full(len(close), np.nan)
            for i in range(window - 1, len(close)):
                data = close[i - window + 1 : i + 1]
                x = np.arange(window)
                try:
                    slope, intercept = np.polyfit(x, data, 1)
                    predicted = slope * (window - 1) + intercept
                    residual[i] = close[i] - predicted
                except (np.linalg.LinAlgError, ValueError):
                    pass
            results[f"linreg_residual_{window}"] = residual

        return results


class StandardError(StatisticalFeature):
    """Rolling standard error of linear regression."""

    def __init__(self, windows: list[int] | None = None):
        super().__init__("StdError")
        self.windows = windows or [20, 50]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        for window in self.windows:
            stderr = np.full(len(close), np.nan)
            for i in range(window - 1, len(close)):
                data = close[i - window + 1 : i + 1]
                x = np.arange(window)
                try:
                    slope, intercept = np.polyfit(x, data, 1)
                    predicted = slope * x + intercept
                    residuals = data - predicted
                    stderr[i] = np.std(residuals, ddof=2)
                except (np.linalg.LinAlgError, ValueError):
                    pass
            results[f"std_error_{window}"] = stderr

        return results


class Curvature(StatisticalFeature):
    """Curvature measure from quadratic fit."""

    def __init__(self, windows: list[int] | None = None):
        super().__init__("Curvature")
        self.windows = windows or [20, 50]

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        for window in self.windows:
            curvature = np.full(len(close), np.nan)
            for i in range(window - 1, len(close)):
                data = close[i - window + 1 : i + 1]
                x = np.arange(window)
                try:
                    coeffs = np.polyfit(x, data, 2)
                    curvature[i] = 2 * coeffs[0]  # Second derivative
                except (np.linalg.LinAlgError, ValueError):
                    pass
            results[f"curvature_{window}"] = curvature

        return results


# =============================================================================
# COMPOSITE CALCULATOR
# =============================================================================


class StatisticalFeatureCalculator:
    """
    Main calculator class that computes all statistical features.

    Usage:
        calculator = StatisticalFeatureCalculator()
        features = calculator.compute_all(df)  # Returns dict of all features
    """

    def __init__(self, include_all: bool = True):
        self.features: list[StatisticalFeature] = []

        if include_all:
            self._add_default_features()

    def _add_default_features(self) -> None:
        """Add all default statistical features."""
        # Returns
        self.features.extend([
            SimpleReturns(),
            LogReturns(),
            RiskAdjustedReturns(),
        ])

        # Distribution metrics
        self.features.extend([
            RollingMean(),
            RollingStd(),
            RollingVar(),
            RollingSkewness(),
            RollingKurtosis(),
            RollingQuantiles(),
            ZScore(),
        ])

        # Tail metrics
        self.features.extend([
            RollingVaR(),
            RollingCVaR(),
            TailRatio(),
        ])

        # Time series
        self.features.extend([
            Autocorrelation(),
            PartialAutocorrelation(),
            HurstExponent(),
            ADFStatistic(),
        ])

        # Mean reversion
        self.features.extend([
            HalfLife(),
            OUParameters(),
            MeanReversionStrength(),
        ])

        # Regression
        self.features.extend([
            LinearRegressionSlope(),
            RSquared(),
            LinearRegressionResiduals(),
            StandardError(),
            Curvature(),
        ])

    def add_feature(self, feature: StatisticalFeature) -> None:
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
