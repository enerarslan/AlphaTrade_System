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


def _adaptive_sampling_step(n_rows: int, base_step: int = 1) -> int:
    """Scale expensive rolling computations for large datasets."""
    step = max(1, int(base_step))
    n_rows = max(0, int(n_rows))
    if n_rows >= 100_000:
        return max(step, 40)
    if n_rows >= 50_000:
        return max(step, 24)
    if n_rows >= 20_000:
        return max(step, 12)
    if n_rows >= 10_000:
        return max(step, 6)
    return step


def _sample_indices(start_idx: int, end_exclusive: int, step: int) -> list[int]:
    """Build sampled rolling indices and always include the terminal row."""
    if end_exclusive <= start_idx:
        return []
    step = max(1, int(step))
    indices = list(range(int(start_idx), int(end_exclusive), step))
    last_idx = int(end_exclusive) - 1
    if not indices:
        return [last_idx]
    if indices[-1] != last_idx:
        indices.append(last_idx)
    return indices


def _forward_fill_nan(arr: np.ndarray) -> np.ndarray:
    """Fast forward fill for 1D float arrays."""
    out = np.asarray(arr, dtype=float).reshape(-1).copy()
    if out.size == 0:
        return out
    mask = np.isnan(out)
    if bool(np.all(mask)):
        return out
    idx = np.where(~mask, np.arange(out.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out[mask] = out[idx[mask]]
    return out


def _rolling_sum_count(arr: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return rolling finite-value sums and counts aligned to the trailing window."""
    values = np.asarray(arr, dtype=np.float64).reshape(-1)
    if window <= 0 or window > values.size:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)

    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    sums = np.cumsum(safe, dtype=np.float64)
    counts = np.cumsum(finite.astype(np.int64), dtype=np.int64)

    window_sums = sums[window - 1 :].copy()
    window_counts = counts[window - 1 :].copy()
    if window < values.size:
        window_sums[1:] -= sums[: -window]
        window_counts[1:] -= counts[: -window]
    return window_sums, window_counts


def _rolling_nanmean(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling NaN-aware means."""
    values = np.asarray(arr, dtype=np.float64).reshape(-1)
    result = np.full(values.size, np.nan, dtype=np.float64)
    window_sums, window_counts = _rolling_sum_count(values, int(window))
    if window_sums.size == 0:
        return result
    out = np.full(window_sums.size, np.nan, dtype=np.float64)
    valid = window_counts > 0
    out[valid] = window_sums[valid] / window_counts[valid]
    result[int(window) - 1 :] = out
    return result


def _rolling_nanvar(arr: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    """Compute rolling NaN-aware variance."""
    values = np.asarray(arr, dtype=np.float64).reshape(-1)
    result = np.full(values.size, np.nan, dtype=np.float64)
    window_sums, window_counts = _rolling_sum_count(values, int(window))
    if window_sums.size == 0:
        return result

    safe_squared = np.where(np.isfinite(values), values * values, 0.0)
    squared_sums = np.cumsum(safe_squared, dtype=np.float64)
    window_squared_sums = squared_sums[int(window) - 1 :].copy()
    if int(window) < values.size:
        window_squared_sums[1:] -= squared_sums[: -int(window)]

    out = np.full(window_sums.size, np.nan, dtype=np.float64)
    valid = window_counts > int(ddof)
    if np.any(valid):
        counts = window_counts[valid].astype(np.float64)
        numerator = window_squared_sums[valid] - (window_sums[valid] ** 2) / counts
        numerator = np.maximum(numerator, 0.0)
        out[valid] = numerator / np.maximum(counts - float(ddof), 1.0)
    result[int(window) - 1 :] = out
    return result


def _rolling_nanstd(arr: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    """Compute rolling NaN-aware standard deviation."""
    variance = _rolling_nanvar(arr, window, ddof=ddof)
    return np.sqrt(variance)


def _sampled_rolling_moment_windows(
    arr: np.ndarray,
    window: int,
    sample_positions: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return sampled sliding windows plus aligned output positions."""
    values = np.asarray(arr, dtype=np.float64).reshape(-1)
    if window <= 0 or window > values.size or not sample_positions:
        return np.empty((0, 0), dtype=np.float64), np.array([], dtype=np.int64)

    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=int(window))
    row_positions = np.asarray(sample_positions, dtype=np.int64) - (int(window) - 1)
    valid = (row_positions >= 0) & (row_positions < windows.shape[0])
    if not np.any(valid):
        return np.empty((0, int(window)), dtype=np.float64), np.array([], dtype=np.int64)
    row_positions = row_positions[valid]
    return windows[row_positions], row_positions + (int(window) - 1)


def _rolling_sampled_skewness(
    arr: np.ndarray,
    window: int,
    sample_positions: list[int],
) -> np.ndarray:
    """Compute sampled rolling skewness using vectorized central moments."""
    values = np.asarray(arr, dtype=np.float64).reshape(-1)
    result = np.full(values.size, np.nan, dtype=np.float64)
    windows, target_positions = _sampled_rolling_moment_windows(values, window, sample_positions)
    if windows.size == 0:
        return result

    finite = np.isfinite(windows)
    counts = finite.sum(axis=1).astype(np.float64)
    safe = np.where(finite, windows, 0.0)
    means = safe.sum(axis=1) / np.where(counts > 0.0, counts, 1.0)
    centered = np.where(finite, safe - means[:, None], 0.0)
    second = (centered**2).sum(axis=1) / np.where(counts > 0.0, counts, 1.0)
    third = (centered**3).sum(axis=1) / np.where(counts > 0.0, counts, 1.0)

    skew_values = np.full(target_positions.size, np.nan, dtype=np.float64)
    valid = (counts >= 3.0) & (second > 1e-18)
    if np.any(valid):
        base = third[valid] / np.power(second[valid], 1.5)
        correction = np.sqrt(counts[valid] * (counts[valid] - 1.0)) / (counts[valid] - 2.0)
        skew_values[valid] = correction * base

    result[target_positions] = skew_values
    return result


def _rolling_sampled_kurtosis(
    arr: np.ndarray,
    window: int,
    sample_positions: list[int],
) -> np.ndarray:
    """Compute sampled rolling excess kurtosis using vectorized central moments."""
    values = np.asarray(arr, dtype=np.float64).reshape(-1)
    result = np.full(values.size, np.nan, dtype=np.float64)
    windows, target_positions = _sampled_rolling_moment_windows(values, window, sample_positions)
    if windows.size == 0:
        return result

    finite = np.isfinite(windows)
    counts = finite.sum(axis=1).astype(np.float64)
    safe = np.where(finite, windows, 0.0)
    means = safe.sum(axis=1) / np.where(counts > 0.0, counts, 1.0)
    centered = np.where(finite, safe - means[:, None], 0.0)
    second = (centered**2).sum(axis=1) / np.where(counts > 0.0, counts, 1.0)
    fourth = (centered**4).sum(axis=1) / np.where(counts > 0.0, counts, 1.0)

    kurt_values = np.full(target_positions.size, np.nan, dtype=np.float64)
    valid = (counts >= 4.0) & (second > 1e-18)
    if np.any(valid):
        excess = (fourth[valid] / np.power(second[valid], 2.0)) - 3.0
        correction = ((counts[valid] - 1.0) / ((counts[valid] - 2.0) * (counts[valid] - 3.0))) * (
            ((counts[valid] + 1.0) * excess) + 6.0
        )
        kurt_values[valid] = correction

    result[target_positions] = kurt_values
    return result


def _rolling_nanquantile(arr: np.ndarray, window: int, quantile: float) -> np.ndarray:
    """Compute rolling NaN-aware quantiles."""
    values = np.asarray(arr, dtype=np.float64).reshape(-1)
    result = np.full(values.size, np.nan, dtype=np.float64)
    if window <= 0 or window > values.size:
        return result

    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=int(window))
    valid = np.isfinite(windows).sum(axis=1) > 0
    out = np.full(windows.shape[0], np.nan, dtype=np.float64)
    if np.any(valid):
        out[valid] = np.nanquantile(windows[valid], float(quantile), axis=1)
    result[int(window) - 1 :] = out
    return result


def _rolling_nanquantiles(
    arr: np.ndarray,
    window: int,
    quantiles: list[float] | np.ndarray,
) -> dict[float, np.ndarray]:
    """Compute multiple rolling NaN-aware quantiles from a single window view."""
    values = np.asarray(arr, dtype=np.float64).reshape(-1)
    quantile_array = np.asarray(list(quantiles), dtype=np.float64)
    outputs = {
        float(quantile): np.full(values.size, np.nan, dtype=np.float64)
        for quantile in quantile_array
    }
    if window <= 0 or window > values.size or quantile_array.size == 0:
        return outputs

    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=int(window))
    valid_rows = np.isfinite(windows).sum(axis=1) > 0
    if not np.any(valid_rows):
        return outputs

    computed = np.nanquantile(windows[valid_rows], quantile_array, axis=1)
    if computed.ndim == 1:
        computed = computed[np.newaxis, :]

    for index, quantile in enumerate(quantile_array):
        out = np.full(windows.shape[0], np.nan, dtype=np.float64)
        out[valid_rows] = computed[index]
        outputs[float(quantile)][int(window) - 1 :] = out
    return outputs


def _rolling_tail_mean(arr: np.ndarray, window: int, quantile: float) -> np.ndarray:
    """Compute rolling lower-tail mean using a quantile threshold per window."""
    values = np.asarray(arr, dtype=np.float64).reshape(-1)
    result = np.full(values.size, np.nan, dtype=np.float64)
    if window <= 0 or window > values.size:
        return result

    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=int(window))
    finite = np.isfinite(windows)
    valid_rows = finite.sum(axis=1) > 0
    if not np.any(valid_rows):
        return result

    thresholds = np.nanquantile(windows[valid_rows], float(quantile), axis=1)
    valid_windows = windows[valid_rows]
    valid_finite = finite[valid_rows]
    tail_mask = valid_finite & (valid_windows <= thresholds[:, None])
    tail_counts = tail_mask.sum(axis=1)
    tail_sums = np.where(tail_mask, valid_windows, 0.0).sum(axis=1)

    out = np.full(windows.shape[0], np.nan, dtype=np.float64)
    valid_tail = tail_counts > 0
    if np.any(valid_tail):
        out_positions = np.flatnonzero(valid_rows)
        out[out_positions[valid_tail]] = tail_sums[valid_tail] / tail_counts[valid_tail]
    result[int(window) - 1 :] = out
    return result


def _rolling_pairwise_correlation(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
) -> np.ndarray:
    """Compute rolling correlation for two aligned 1D arrays with NaN handling."""
    x_values = np.asarray(x, dtype=np.float64).reshape(-1)
    y_values = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(x_values.size, y_values.size)
    result = np.full(n, np.nan, dtype=np.float64)
    if window <= 0 or window > n:
        return result

    x_values = x_values[:n]
    y_values = y_values[:n]
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_safe = np.where(valid, x_values, 0.0)
    y_safe = np.where(valid, y_values, 0.0)
    xy_safe = x_safe * y_safe
    x2_safe = x_safe * x_safe
    y2_safe = y_safe * y_safe

    counts = np.cumsum(valid.astype(np.int64), dtype=np.int64)
    sum_x = np.cumsum(x_safe, dtype=np.float64)
    sum_y = np.cumsum(y_safe, dtype=np.float64)
    sum_xy = np.cumsum(xy_safe, dtype=np.float64)
    sum_x2 = np.cumsum(x2_safe, dtype=np.float64)
    sum_y2 = np.cumsum(y2_safe, dtype=np.float64)

    counts_w = counts[int(window) - 1 :].copy()
    sum_x_w = sum_x[int(window) - 1 :].copy()
    sum_y_w = sum_y[int(window) - 1 :].copy()
    sum_xy_w = sum_xy[int(window) - 1 :].copy()
    sum_x2_w = sum_x2[int(window) - 1 :].copy()
    sum_y2_w = sum_y2[int(window) - 1 :].copy()

    if int(window) < n:
        counts_w[1:] -= counts[: -int(window)]
        sum_x_w[1:] -= sum_x[: -int(window)]
        sum_y_w[1:] -= sum_y[: -int(window)]
        sum_xy_w[1:] -= sum_xy[: -int(window)]
        sum_x2_w[1:] -= sum_x2[: -int(window)]
        sum_y2_w[1:] -= sum_y2[: -int(window)]

    out = np.full(sum_x_w.size, np.nan, dtype=np.float64)
    valid_rows = counts_w > 2
    if np.any(valid_rows):
        row_counts = counts_w[valid_rows].astype(np.float64)
        cov_num = sum_xy_w[valid_rows] - ((sum_x_w[valid_rows] * sum_y_w[valid_rows]) / row_counts)
        var_x_num = sum_x2_w[valid_rows] - ((sum_x_w[valid_rows] ** 2) / row_counts)
        var_y_num = sum_y2_w[valid_rows] - ((sum_y_w[valid_rows] ** 2) / row_counts)
        denom = np.sqrt(np.maximum(var_x_num, 0.0) * np.maximum(var_y_num, 0.0))
        corr = np.full(row_counts.size, np.nan, dtype=np.float64)
        valid_denom = denom > 1e-18
        if np.any(valid_denom):
            corr[valid_denom] = cov_num[valid_denom] / denom[valid_denom]
            corr[valid_denom] = np.clip(corr[valid_denom], -1.0, 1.0)
        out[valid_rows] = corr
    result[int(window) - 1 :] = out
    return result


def _unbiased_excess_kurtosis(values: np.ndarray) -> float:
    """Return Fisher excess kurtosis with bias correction.

    SciPy's kurtosis path has been observed to abort the Python process on long
    rolling loops in the WSL training environment. Keep this implementation
    NumPy-only so large live-training datasets stay deterministic and stable.
    """
    sample = np.asarray(values, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    n = int(sample.size)
    if n < 4:
        return float("nan")

    centered = sample - float(np.mean(sample))
    second_moment = float(np.mean(centered**2))
    if not np.isfinite(second_moment) or second_moment <= 1e-18:
        return 0.0

    fourth_moment = float(np.mean(centered**4))
    excess = fourth_moment / (second_moment**2) - 3.0
    correction = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * excess + 6.0)
    return float(correction)


def _unbiased_skewness(values: np.ndarray) -> float:
    """Return adjusted Fisher-Pearson sample skewness."""
    sample = np.asarray(values, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    n = int(sample.size)
    if n < 3:
        return float("nan")

    centered = sample - float(np.mean(sample))
    second_moment = float(np.mean(centered**2))
    if not np.isfinite(second_moment) or second_moment <= 1e-18:
        return 0.0

    third_moment = float(np.mean(centered**3))
    skew = third_moment / (second_moment ** 1.5)
    correction = np.sqrt(n * (n - 1)) / (n - 2)
    return float(correction * skew)


def _linear_regression_diagnostics(
    values: np.ndarray,
) -> tuple[float, float, float, np.ndarray, float] | None:
    """Return slope, intercept, r-squared, residuals, and standard error."""
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or y.size < 2 or not np.all(np.isfinite(y)):
        return None

    x = np.arange(y.size, dtype=np.float64)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    y_centered = y - y_mean

    denom = float(np.sum(x_centered**2))
    if denom <= 1e-18:
        return None

    slope = float(np.sum(x_centered * y_centered) / denom)
    intercept = float(y_mean - slope * x_mean)
    predicted = slope * x + intercept
    residuals = y - predicted

    ss_tot = float(np.sum(y_centered**2))
    ss_res = float(np.sum(residuals**2))
    if ss_tot <= 1e-18:
        r_squared = 0.0
    else:
        r_squared = float(np.clip(1.0 - (ss_res / ss_tot), 0.0, 1.0))

    dof = max(1, y.size - 2)
    stderr = float(np.sqrt(ss_res / dof))
    return slope, intercept, r_squared, residuals, stderr


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
        return _rolling_nanstd(arr, window, ddof=1)


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
            results[f"rolling_mean_{window}"] = _rolling_nanmean(returns, window)

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
            results[f"rolling_std_{window}"] = _rolling_nanstd(returns, window, ddof=1)

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
            results[f"rolling_var_{window}"] = _rolling_nanvar(returns, window, ddof=1)

        return results


class RollingSkewness(StatisticalFeature):
    """Rolling skewness of returns."""

    def __init__(self, windows: list[int] | None = None, compute_step: int = 1):
        super().__init__("RollingSkewness")
        self.windows = windows or [20, 60, 120]
        self.compute_step = compute_step

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        returns = np.diff(np.log(close), prepend=np.nan)
        results = {}
        step = _adaptive_sampling_step(len(close), self.compute_step)

        for window in self.windows:
            roll_skew = _rolling_sampled_skewness(
                returns,
                window,
                _sample_indices(window - 1, len(close), step),
            )
            if step > 1:
                roll_skew = _forward_fill_nan(roll_skew)
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
        step = _adaptive_sampling_step(len(close), 1)

        for window in self.windows:
            roll_kurt = _rolling_sampled_kurtosis(
                returns,
                window,
                _sample_indices(window - 1, len(close), step),
            )
            if step > 1:
                roll_kurt = _forward_fill_nan(roll_kurt)
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

        quantile_results = _rolling_nanquantiles(returns, self.window, self.quantiles)
        for q in self.quantiles:
            q_name = int(q * 100)
            results[f"rolling_q{q_name}_{self.window}"] = quantile_results[float(q)]

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
            mean = _rolling_nanmean(close, window)
            std = _rolling_nanstd(close, window, ddof=1)
            valid = np.isfinite(close) & np.isfinite(mean) & (std > 1e-12)
            zscore[valid] = (close[valid] - mean[valid]) / std[valid]
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

        quantile_levels = [1.0 - float(conf) for conf in self.confidence_levels]
        quantile_results = _rolling_nanquantiles(returns, self.window, quantile_levels)
        for conf in self.confidence_levels:
            conf_name = int(conf * 100)
            results[f"var_{conf_name}_{self.window}"] = quantile_results[1.0 - float(conf)]

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

        cvar_arr = _rolling_tail_mean(returns, self.window, 1.0 - float(self.confidence_level))

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
        quantiles = _rolling_nanquantiles(
            returns,
            self.window,
            [float(self.percentile), 1.0 - float(self.percentile)],
        )
        lower_tail = quantiles[float(self.percentile)]
        upper_tail = quantiles[1.0 - float(self.percentile)]
        valid = np.isfinite(lower_tail) & np.isfinite(upper_tail) & (np.abs(lower_tail) > 1e-18)
        if np.any(valid):
            tail_ratio[valid] = np.abs(upper_tail[valid] / lower_tail[valid])

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
            if lag < len(returns):
                acf[lag:] = _rolling_pairwise_correlation(returns[lag:], returns[:-lag], self.window)
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

    def __init__(
        self,
        window: int = 60,
        lags: list[int] | None = None,
        compute_step: int = 1,
    ):
        super().__init__("PACF")
        self.window = window
        self.lags = lags or [1, 2, 3, 5]
        self.compute_step = compute_step

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        returns = np.diff(np.log(close), prepend=np.nan)
        results = {}

        max_lag = max(self.lags)
        step = _adaptive_sampling_step(len(close), self.compute_step)
        for lag in self.lags:
            pacf = np.full(len(close), np.nan)
            for i in _sample_indices(self.window + max_lag - 1, len(close), step):
                data = returns[i - self.window - max_lag + 1 : i + 1]
                valid_data = data[~np.isnan(data)]
                if len(valid_data) >= self.window:
                    pacf[i] = self._pacf(valid_data, lag)
            if step > 1:
                pacf = _forward_fill_nan(pacf)
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
        if not np.all(np.isfinite(acf_values)):
            return np.nan

        r = np.asarray(acf_values[1:], dtype=np.float64)
        R = np.zeros((lag, lag), dtype=np.float64)
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

    def __init__(
        self,
        window: int = 100,
        min_lag: int = 2,
        max_lag: int = 20,
        compute_step: int = 5,
    ):
        super().__init__("HurstExponent")
        self.window = window
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.compute_step = compute_step

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        hurst = np.full(len(close), np.nan)
        step = _adaptive_sampling_step(len(close), self.compute_step)
        for i in _sample_indices(self.window - 1, len(close), step):
            data = close[i - self.window + 1 : i + 1]
            hurst[i] = self._compute_hurst(data)
        if step > 1:
            hurst = _forward_fill_nan(hurst)

        results[f"hurst_{self.window}"] = hurst

        return results

    def _compute_hurst(self, ts: np.ndarray) -> float:
        """Compute Hurst exponent using a variance-of-increments approximation."""
        if len(ts) < self.max_lag * 2:
            return np.nan

        sample = np.asarray(ts, dtype=np.float64)
        lags = np.arange(self.min_lag, self.max_lag, dtype=np.int64)
        tau = np.full(lags.shape[0], np.nan, dtype=np.float64)

        for index, lag in enumerate(lags):
            diffs = sample[int(lag) :] - sample[: -int(lag)]
            diffs = diffs[np.isfinite(diffs)]
            if diffs.size < 2:
                continue
            sigma = float(np.std(diffs, ddof=1))
            if sigma > 1e-18:
                tau[index] = sigma

        valid_idx = np.isfinite(tau)
        if np.sum(valid_idx) < 2:
            return np.nan

        log_lags = np.log(lags[valid_idx].astype(np.float64))
        log_tau = np.log(tau[valid_idx])
        x_centered = log_lags - np.mean(log_lags)
        denom = float(np.sum(x_centered**2))
        if denom <= 1e-18:
            return np.nan
        slope = float(np.sum(x_centered * (log_tau - np.mean(log_tau))) / denom)
        return float(np.clip(slope * 2.0, 0.0, 1.5))


class ADFStatistic(StatisticalFeature):
    """
    JPMORGAN FIX: Complete Augmented Dickey-Fuller test implementation.

    Implements the full ADF test as per MacKinnon (1994, 2010) with:
    1. Automatic lag selection using AIC/BIC
    2. Different regression specifications (constant, trend, none)
    3. P-value calculation using MacKinnon critical values
    4. Critical values for 1%, 5%, 10% significance levels

    The ADF test is essential for:
    - Testing stationarity of price series
    - Mean reversion strategy identification
    - Cointegration analysis in pairs trading
    - Feature engineering for ML models

    Reference:
        MacKinnon, J.G. (2010). "Critical Values for Cointegration Tests"
    """

    # MacKinnon (2010) critical values for 'c' (constant) case
    # Format: [1%, 5%, 10%] for different sample sizes
    MACKINNON_CRITICAL_VALUES = {
        "c": {
            25: [-3.75, -3.00, -2.62],
            50: [-3.58, -2.93, -2.60],
            100: [-3.51, -2.89, -2.58],
            250: [-3.46, -2.87, -2.57],
            500: [-3.44, -2.86, -2.57],
            1000: [-3.43, -2.86, -2.57],
        },
        "ct": {  # constant + trend
            25: [-4.38, -3.60, -3.24],
            50: [-4.15, -3.50, -3.18],
            100: [-4.04, -3.45, -3.15],
            250: [-3.99, -3.43, -3.13],
            500: [-3.98, -3.42, -3.13],
            1000: [-3.96, -3.41, -3.12],
        },
        "n": {  # no constant
            25: [-2.66, -1.95, -1.60],
            50: [-2.62, -1.95, -1.61],
            100: [-2.60, -1.95, -1.61],
            250: [-2.58, -1.95, -1.62],
            500: [-2.58, -1.95, -1.62],
            1000: [-2.58, -1.95, -1.62],
        },
    }

    def __init__(
        self,
        window: int = 60,
        max_lag: int | None = None,
        regression: str = "c",
        autolag: str = "AIC",
        compute_step: int = 4,
    ):
        """
        Initialize ADF test calculator.

        Args:
            window: Rolling window size for calculation
            max_lag: Maximum number of lags to consider (None = auto)
            regression: Type of regression - 'c' (constant), 'ct' (constant+trend), 'n' (none)
            autolag: Method for automatic lag selection - 'AIC', 'BIC', or None for fixed
        """
        super().__init__("ADF")
        self.window = window
        self.max_lag = max_lag
        self.regression = regression
        self.autolag = autolag
        self.compute_step = compute_step

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        n = len(close)
        adf_stat = np.full(n, np.nan)
        adf_pvalue = np.full(n, np.nan)
        adf_lags = np.full(n, np.nan)
        adf_is_stationary = np.full(n, np.nan)  # 1 if stationary at 5% level

        step = _adaptive_sampling_step(n, self.compute_step)
        for i in _sample_indices(self.window - 1, n, step):
            data = close[i - self.window + 1 : i + 1]
            stat, pvalue, used_lag, is_stat = self._adf_test(data)
            adf_stat[i] = stat
            adf_pvalue[i] = pvalue
            adf_lags[i] = used_lag
            adf_is_stationary[i] = is_stat
        if step > 1:
            adf_stat = _forward_fill_nan(adf_stat)
            adf_pvalue = _forward_fill_nan(adf_pvalue)
            adf_lags = _forward_fill_nan(adf_lags)
            adf_is_stationary = _forward_fill_nan(adf_is_stationary)
        adf_is_stationary = np.nan_to_num(adf_is_stationary, nan=0.0)

        results[f"adf_stat_{self.window}"] = adf_stat
        results[f"adf_pvalue_{self.window}"] = adf_pvalue
        results[f"adf_lags_{self.window}"] = adf_lags
        results[f"adf_is_stationary_{self.window}"] = adf_is_stationary

        return results

    def _adf_test(self, ts: np.ndarray) -> tuple[float, float, int, int]:
        """
        Compute complete ADF test.

        Returns:
            Tuple of (test_statistic, p_value, used_lags, is_stationary)
        """
        n = len(ts)
        if n < 12:
            return np.nan, np.nan, 0, 0

        # Calculate max lag if not specified
        # Schwert (1989) formula: floor(12 * (n/100)^(1/4))
        if self.max_lag is None:
            max_lag = int(np.floor(12 * (n / 100) ** 0.25))
        else:
            max_lag = min(self.max_lag, n // 4)

        max_lag = max(1, min(max_lag, n // 4 - 1))

        # Select optimal lag using information criterion
        if self.autolag:
            best_lag, best_ic = 0, np.inf
            for lag in range(max_lag + 1):
                try:
                    stat, _, ic = self._adf_regression(ts, lag)
                    if not np.isnan(ic) and ic < best_ic:
                        best_ic = ic
                        best_lag = lag
                except Exception:
                    continue
            used_lag = best_lag
        else:
            used_lag = max_lag

        # Compute final ADF statistic
        try:
            adf_stat, gamma, _ = self._adf_regression(ts, used_lag)
        except Exception:
            return np.nan, np.nan, used_lag, 0

        if np.isnan(adf_stat):
            return np.nan, np.nan, used_lag, 0

        # Get p-value and check stationarity
        pvalue = self._get_pvalue(adf_stat, n)
        is_stationary = 1 if pvalue < 0.05 else 0

        return adf_stat, pvalue, used_lag, is_stationary

    def _adf_regression(self, ts: np.ndarray, nlags: int) -> tuple[float, float, float]:
        """
        Perform ADF regression with specified lags.

        Δy_t = α + β*t + γ*y_{t-1} + Σ(δ_i * Δy_{t-i}) + ε_t

        Returns:
            Tuple of (t_statistic_for_gamma, gamma_coefficient, information_criterion)
        """
        n = len(ts)

        # First difference
        y_diff = np.diff(ts)

        # Lagged level
        y_lag = ts[:-1]

        # Create lagged differences matrix
        if nlags > 0:
            diff_lags = np.column_stack([
                np.roll(y_diff, i)[nlags:] for i in range(1, nlags + 1)
            ])
            y_diff = y_diff[nlags:]
            y_lag = y_lag[nlags:]
        else:
            diff_lags = None

        n_obs = len(y_diff)
        if n_obs < 10:
            return np.nan, np.nan, np.nan

        # Build design matrix based on regression type
        if self.regression == "c":
            # Constant only
            if diff_lags is not None:
                X = np.column_stack([np.ones(n_obs), y_lag, diff_lags])
            else:
                X = np.column_stack([np.ones(n_obs), y_lag])
            gamma_idx = 1
        elif self.regression == "ct":
            # Constant + trend
            trend = np.arange(n_obs)
            if diff_lags is not None:
                X = np.column_stack([np.ones(n_obs), trend, y_lag, diff_lags])
            else:
                X = np.column_stack([np.ones(n_obs), trend, y_lag])
            gamma_idx = 2
        else:  # 'n' - no constant
            if diff_lags is not None:
                X = np.column_stack([y_lag, diff_lags])
            else:
                X = y_lag.reshape(-1, 1)
            gamma_idx = 0

        y = y_diff

        try:
            # OLS regression using QR decomposition for numerical stability
            Q, R = np.linalg.qr(X)
            coeffs = np.linalg.solve(R, Q.T @ y)

            gamma = coeffs[gamma_idx]

            # Calculate residuals and standard errors
            residuals = y - X @ coeffs
            n_params = X.shape[1]
            sigma2 = np.sum(residuals**2) / (n_obs - n_params)

            # Variance-covariance matrix
            try:
                R_inv = np.linalg.inv(R)
                var_beta = sigma2 * (R_inv @ R_inv.T)
                se_gamma = np.sqrt(var_beta[gamma_idx, gamma_idx])
            except np.linalg.LinAlgError:
                return np.nan, np.nan, np.nan

            if se_gamma <= 0:
                return np.nan, np.nan, np.nan

            t_stat = gamma / se_gamma

            # Information criterion for lag selection
            if self.autolag == "AIC":
                ic = n_obs * np.log(sigma2) + 2 * n_params
            elif self.autolag == "BIC":
                ic = n_obs * np.log(sigma2) + n_params * np.log(n_obs)
            else:
                ic = np.nan

            return t_stat, gamma, ic

        except (np.linalg.LinAlgError, ValueError):
            return np.nan, np.nan, np.nan

    def _get_pvalue(self, stat: float, n: int) -> float:
        """
        Get approximate p-value using MacKinnon critical values.

        Uses linear interpolation between critical values.
        """
        if np.isnan(stat):
            return np.nan

        # Get critical values for sample size
        cv_dict = self.MACKINNON_CRITICAL_VALUES.get(self.regression, self.MACKINNON_CRITICAL_VALUES["c"])

        # Find appropriate sample size bucket
        sizes = sorted(cv_dict.keys())
        if n <= sizes[0]:
            cv = cv_dict[sizes[0]]
        elif n >= sizes[-1]:
            cv = cv_dict[sizes[-1]]
        else:
            # Interpolate between two nearest sizes
            for i in range(len(sizes) - 1):
                if sizes[i] <= n < sizes[i + 1]:
                    cv_low = cv_dict[sizes[i]]
                    cv_high = cv_dict[sizes[i + 1]]
                    weight = (n - sizes[i]) / (sizes[i + 1] - sizes[i])
                    cv = [
                        cv_low[j] + weight * (cv_high[j] - cv_low[j])
                        for j in range(3)
                    ]
                    break
            else:
                cv = cv_dict[sizes[-1]]

        # Approximate p-value from critical values
        # cv = [1%, 5%, 10%] => p = [0.01, 0.05, 0.10]
        if stat < cv[0]:  # More negative than 1% critical value
            # Extrapolate below 1%
            return 0.001
        elif stat < cv[1]:  # Between 1% and 5%
            return 0.01 + (stat - cv[0]) / (cv[1] - cv[0]) * 0.04
        elif stat < cv[2]:  # Between 5% and 10%
            return 0.05 + (stat - cv[1]) / (cv[2] - cv[1]) * 0.05
        else:  # Above 10% critical value (not stationary)
            # Simple extrapolation for p > 0.10
            return min(0.10 + (stat - cv[2]) * 0.10, 0.99)


# =============================================================================
# MEAN REVERSION METRICS
# =============================================================================


class HalfLife(StatisticalFeature):
    """Half-life of mean reversion."""

    def __init__(self, window: int = 60, compute_step: int = 3):
        super().__init__("HalfLife")
        self.window = window
        self.compute_step = compute_step

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        half_life = np.full(len(close), np.nan)
        step = _adaptive_sampling_step(len(close), self.compute_step)
        for i in _sample_indices(self.window - 1, len(close), step):
            data = close[i - self.window + 1 : i + 1]
            half_life[i] = self._compute_half_life(data)
        if step > 1:
            half_life = _forward_fill_nan(half_life)

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

    def __init__(self, window: int = 60, compute_step: int = 3):
        super().__init__("OUParams")
        self.window = window
        self.compute_step = compute_step

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        theta = np.full(len(close), np.nan)
        mu = np.full(len(close), np.nan)
        sigma = np.full(len(close), np.nan)

        step = _adaptive_sampling_step(len(close), self.compute_step)
        for i in _sample_indices(self.window - 1, len(close), step):
            data = close[i - self.window + 1 : i + 1]
            params = self._fit_ou(data)
            if params is not None:
                theta[i], mu[i], sigma[i] = params
        if step > 1:
            theta = _forward_fill_nan(theta)
            mu = _forward_fill_nan(mu)
            sigma = _forward_fill_nan(sigma)

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

    def __init__(self, window: int = 60, compute_step: int = 3):
        super().__init__("MRStrength")
        self.window = window
        self.compute_step = compute_step

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        strength = np.full(len(close), np.nan)
        step = _adaptive_sampling_step(len(close), self.compute_step)
        for i in _sample_indices(self.window - 1, len(close), step):
            data = close[i - self.window + 1 : i + 1]
            strength[i] = self._mr_strength(data)
        if step > 1:
            strength = _forward_fill_nan(strength)

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

    def __init__(self, windows: list[int] | None = None, compute_step: int = 1):
        super().__init__("LinRegSlope")
        self.windows = windows or [10, 20, 50]
        self.compute_step = compute_step

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}
        step = _adaptive_sampling_step(len(close), self.compute_step)

        for window in self.windows:
            slope = np.full(len(close), np.nan)
            for i in _sample_indices(window - 1, len(close), step):
                data = close[i - window + 1 : i + 1]
                diagnostics = _linear_regression_diagnostics(data)
                if diagnostics is not None:
                    slope[i] = diagnostics[0]
            if step > 1:
                slope = _forward_fill_nan(slope)
            results[f"linreg_slope_{window}"] = slope

        return results


class RSquared(StatisticalFeature):
    """Rolling R-squared of linear regression."""

    def __init__(self, windows: list[int] | None = None, compute_step: int = 1):
        super().__init__("RSquared")
        self.windows = windows or [10, 20, 50]
        self.compute_step = compute_step

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}
        step = _adaptive_sampling_step(len(close), self.compute_step)

        for window in self.windows:
            r2 = np.full(len(close), np.nan)
            for i in _sample_indices(window - 1, len(close), step):
                data = close[i - window + 1 : i + 1]
                diagnostics = _linear_regression_diagnostics(data)
                if diagnostics is not None:
                    r2[i] = diagnostics[2]
            if step > 1:
                r2 = _forward_fill_nan(r2)
            results[f"r_squared_{window}"] = r2

        return results


class LinearRegressionResiduals(StatisticalFeature):
    """Residuals from linear regression (deviation from trend)."""

    def __init__(self, windows: list[int] | None = None, compute_step: int = 1):
        super().__init__("LinRegResid")
        self.windows = windows or [20, 50]
        self.compute_step = compute_step

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}
        step = _adaptive_sampling_step(len(close), self.compute_step)

        for window in self.windows:
            residual = np.full(len(close), np.nan)
            for i in _sample_indices(window - 1, len(close), step):
                data = close[i - window + 1 : i + 1]
                diagnostics = _linear_regression_diagnostics(data)
                if diagnostics is not None:
                    slope, intercept = diagnostics[0], diagnostics[1]
                    predicted = slope * (window - 1) + intercept
                    residual[i] = close[i] - predicted
            if step > 1:
                residual = _forward_fill_nan(residual)
            results[f"linreg_residual_{window}"] = residual

        return results


class StandardError(StatisticalFeature):
    """Rolling standard error of linear regression."""

    def __init__(self, windows: list[int] | None = None, compute_step: int = 1):
        super().__init__("StdError")
        self.windows = windows or [20, 50]
        self.compute_step = compute_step

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}
        step = _adaptive_sampling_step(len(close), self.compute_step)

        for window in self.windows:
            stderr = np.full(len(close), np.nan)
            for i in _sample_indices(window - 1, len(close), step):
                data = close[i - window + 1 : i + 1]
                diagnostics = _linear_regression_diagnostics(data)
                if diagnostics is not None:
                    stderr[i] = diagnostics[4]
            if step > 1:
                stderr = _forward_fill_nan(stderr)
            results[f"std_error_{window}"] = stderr

        return results


class Curvature(StatisticalFeature):
    """Curvature measure from quadratic fit."""

    def __init__(self, windows: list[int] | None = None, compute_step: int = 1):
        super().__init__("Curvature")
        self.windows = windows or [20, 50]
        self.compute_step = compute_step

    def compute(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}
        step = _adaptive_sampling_step(len(close), self.compute_step)

        for window in self.windows:
            curvature = np.full(len(close), np.nan)
            for i in _sample_indices(window - 1, len(close), step):
                data = close[i - window + 1 : i + 1]
                x = np.arange(window, dtype=np.float64)
                try:
                    coeffs = np.polyfit(x, data, 2)
                    curvature[i] = 2 * coeffs[0]  # Second derivative
                except (np.linalg.LinAlgError, ValueError):
                    pass
            if step > 1:
                curvature = _forward_fill_nan(curvature)
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
