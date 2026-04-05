"""
Cross-sectional features module.

Implements cross-sectional features organized by category:
- Relative metrics (vs benchmark, sector, peers)
- Peer comparison (percentile rank, z-score vs universe)
- Cross-asset correlations
- PCA-based features (factor loadings, idiosyncratic vol)
- Network features (correlation centrality, cluster membership)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


def _adaptive_sampling_step(n_rows: int, base_step: int = 1) -> int:
    """Scale expensive rolling cross-sectional computations for large datasets."""
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


def _rolling_pairwise_window_summaries(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return rolling pairwise counts and sums for aligned 1D arrays."""
    x_values = np.asarray(x, dtype=np.float64).reshape(-1)
    y_values = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(x_values.size, y_values.size)
    if window <= 0 or window > n:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty, empty, empty

    x_values = x_values[:n]
    y_values = y_values[:n]
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_safe = np.where(valid, x_values, 0.0)
    y_safe = np.where(valid, y_values, 0.0)

    counts = np.cumsum(valid.astype(np.int64), dtype=np.int64)
    sum_x = np.cumsum(x_safe, dtype=np.float64)
    sum_y = np.cumsum(y_safe, dtype=np.float64)
    sum_xy = np.cumsum(x_safe * y_safe, dtype=np.float64)
    sum_x2 = np.cumsum(x_safe * x_safe, dtype=np.float64)
    sum_y2 = np.cumsum(y_safe * y_safe, dtype=np.float64)

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

    return (
        counts_w.astype(np.float64),
        sum_x_w,
        sum_y_w,
        sum_xy_w,
        sum_x2_w,
        sum_y2_w,
    )


def _rolling_pairwise_beta(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling beta = cov(x, y) / var(y) for aligned 1D arrays."""
    x_values = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x_values.size
    result = np.full(n, np.nan, dtype=np.float64)
    summaries = _rolling_pairwise_window_summaries(x, y, window)
    if summaries[0].size == 0:
        return result

    counts_w, sum_x_w, sum_y_w, sum_xy_w, _, sum_y2_w = summaries
    out = np.full(counts_w.size, np.nan, dtype=np.float64)
    valid = counts_w > 2.0
    if np.any(valid):
        row_counts = counts_w[valid]
        cov_num = sum_xy_w[valid] - ((sum_x_w[valid] * sum_y_w[valid]) / row_counts)
        var_y_num = sum_y2_w[valid] - ((sum_y_w[valid] ** 2) / row_counts)
        valid_var = var_y_num > 1e-18
        beta = np.full(row_counts.size, np.nan, dtype=np.float64)
        if np.any(valid_var):
            beta[valid_var] = cov_num[valid_var] / var_y_num[valid_var]
        out[valid] = beta
    result[int(window) - 1 :] = out
    return result


def _rolling_pairwise_correlation(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling correlation for aligned 1D arrays."""
    x_values = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x_values.size
    result = np.full(n, np.nan, dtype=np.float64)
    summaries = _rolling_pairwise_window_summaries(x, y, window)
    if summaries[0].size == 0:
        return result

    counts_w, sum_x_w, sum_y_w, sum_xy_w, sum_x2_w, sum_y2_w = summaries
    out = np.full(counts_w.size, np.nan, dtype=np.float64)
    valid = counts_w > 2.0
    if np.any(valid):
        row_counts = counts_w[valid]
        cov_num = sum_xy_w[valid] - ((sum_x_w[valid] * sum_y_w[valid]) / row_counts)
        var_x_num = sum_x2_w[valid] - ((sum_x_w[valid] ** 2) / row_counts)
        var_y_num = sum_y2_w[valid] - ((sum_y_w[valid] ** 2) / row_counts)
        denom = np.sqrt(np.maximum(var_x_num, 0.0) * np.maximum(var_y_num, 0.0))
        corr = np.full(row_counts.size, np.nan, dtype=np.float64)
        valid_denom = denom > 1e-18
        if np.any(valid_denom):
            corr[valid_denom] = cov_num[valid_denom] / denom[valid_denom]
            corr[valid_denom] = np.clip(corr[valid_denom], -1.0, 1.0)
        out[valid] = corr
    result[int(window) - 1 :] = out
    return result


def _rolling_nanstd(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling standard deviation with NaN-aware cumulative sums."""
    values = np.asarray(arr, dtype=np.float64).reshape(-1)
    result = np.full(values.size, np.nan, dtype=np.float64)
    if window <= 0 or window > values.size:
        return result

    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    safe_sq = safe * safe
    counts = np.cumsum(finite.astype(np.int64), dtype=np.int64)
    sums = np.cumsum(safe, dtype=np.float64)
    sums_sq = np.cumsum(safe_sq, dtype=np.float64)

    counts_w = counts[int(window) - 1 :].copy()
    sums_w = sums[int(window) - 1 :].copy()
    sums_sq_w = sums_sq[int(window) - 1 :].copy()
    if int(window) < values.size:
        counts_w[1:] -= counts[: -int(window)]
        sums_w[1:] -= sums[: -int(window)]
        sums_sq_w[1:] -= sums_sq[: -int(window)]

    out = np.full(counts_w.size, np.nan, dtype=np.float64)
    valid = counts_w > 1
    if np.any(valid):
        row_counts = counts_w[valid].astype(np.float64)
        numerator = sums_sq_w[valid] - ((sums_w[valid] ** 2) / row_counts)
        numerator = np.maximum(numerator, 0.0)
        out[valid] = np.sqrt(numerator / np.maximum(row_counts - 1.0, 1.0))
    result[int(window) - 1 :] = out
    return result


@dataclass
class CrossSectionalResult:
    """Result container for cross-sectional feature calculations."""

    name: str
    values: np.ndarray
    params: dict[str, Any]
    metadata: dict[str, Any] | None = None


class CrossSectionalFeature(ABC):
    """Abstract base class for cross-sectional features."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(
        self,
        df: pl.DataFrame,
        universe_data: dict[str, pl.DataFrame] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Compute the feature and return named arrays.

        Args:
            df: DataFrame for the main symbol
            universe_data: Optional dict mapping symbol -> DataFrame for cross-sectional analysis

        Returns:
            Dictionary mapping feature names to numpy arrays
        """
        pass

    def validate_input(self, df: pl.DataFrame, required_columns: list[str]) -> None:
        """Validate that required columns exist."""
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _context_cache(self) -> dict[str, Any] | None:
        """Return the shared per-computation cache when available."""
        context = getattr(self, "_context", None)
        if context is None:
            return None
        return context.setdefault("cache", {})

    def align_series(
        self,
        base_df: pl.DataFrame,
        base_col: str,
        other_df: pl.DataFrame,
        other_col: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align two series by timestamp when available, else by common suffix length."""
        base = base_df[base_col].to_numpy().astype(np.float64)
        other = other_df[other_col].to_numpy().astype(np.float64)

        if "timestamp" in base_df.columns and "timestamp" in other_df.columns:
            try:
                left = base_df.select([
                    pl.col("timestamp"),
                    pl.col(base_col).alias("_base"),
                ])
                right = other_df.select([
                    pl.col("timestamp"),
                    pl.col(other_col).alias("_other"),
                ])
                joined = left.join(right, on="timestamp", how="inner").sort("timestamp")
                if len(joined) > 0:
                    return (
                        joined["_base"].to_numpy().astype(np.float64),
                        joined["_other"].to_numpy().astype(np.float64),
                    )
            except Exception:
                pass

        min_len = min(len(base), len(other))
        return base[-min_len:], other[-min_len:]

    @staticmethod
    def _nan_array(length: int) -> np.ndarray:
        """Create an NaN-filled feature array of target length."""
        return np.full(int(length), np.nan, dtype=np.float64)

    @staticmethod
    def _window_returns(close: np.ndarray, window: int) -> np.ndarray:
        """Compute trailing window returns aligned to input length."""
        arr = np.asarray(close, dtype=np.float64).reshape(-1)
        n = int(arr.size)
        ret = np.full(n, np.nan, dtype=np.float64)
        if window <= 0 or window >= n:
            return ret

        base = arr[:-window]
        nxt = arr[window:]
        valid = np.abs(base) > 1e-12
        out = np.full(n - window, np.nan, dtype=np.float64)
        out[valid] = (nxt[valid] - base[valid]) / base[valid]
        ret[window:] = out
        return ret

    @staticmethod
    def _log_returns(close: np.ndarray) -> np.ndarray:
        """Compute log returns aligned to input length."""
        arr = np.asarray(close, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return np.array([], dtype=np.float64)
        safe = np.where(arr > 1e-12, arr, np.nan)
        return np.diff(np.log(safe), prepend=np.nan)

    def _align_to_base(
        self,
        base_df: pl.DataFrame,
        other_df: pl.DataFrame,
        values: np.ndarray,
    ) -> np.ndarray:
        """Align an array from other_df onto base_df timestamps when available."""
        base_len = len(base_df)
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if base_len <= 0:
            return np.array([], dtype=np.float64)

        if "timestamp" in base_df.columns and "timestamp" in other_df.columns:
            try:
                base_ts = pd.to_datetime(
                    base_df["timestamp"].to_numpy(),
                    utc=True,
                    errors="coerce",
                )
                other_ts = pd.to_datetime(
                    other_df["timestamp"].to_numpy(),
                    utc=True,
                    errors="coerce",
                )
                series = pd.Series(arr, index=pd.DatetimeIndex(other_ts))
                series = series[~series.index.isna()]
                series = series[~series.index.duplicated(keep="last")]
                series = series.sort_index()
                aligned = series.reindex(
                    pd.DatetimeIndex(base_ts),
                    method="ffill",
                ).to_numpy(dtype=np.float64)
                if aligned.shape[0] == base_len:
                    return aligned
            except Exception:
                pass

        if arr.size >= base_len:
            return arr[-base_len:]
        if arr.size == 0:
            return np.full(base_len, np.nan, dtype=np.float64)
        padded = np.full(base_len, np.nan, dtype=np.float64)
        padded[-arr.size:] = arr
        return padded

    def _aligned_column(
        self,
        base_df: pl.DataFrame,
        other_df: pl.DataFrame,
        column: str,
    ) -> np.ndarray | None:
        """Return other_df[column] aligned to base_df timestamps/length."""
        cache = self._context_cache()
        cache_key = ("aligned_column", id(other_df), str(column))
        if cache is not None and cache_key in cache:
            return cache[cache_key]
        if column not in other_df.columns:
            return None
        values = other_df[column].to_numpy().astype(np.float64)
        aligned = self._align_to_base(base_df, other_df, values)
        if cache is not None:
            cache[cache_key] = aligned
        return aligned

    @staticmethod
    def _resolve_current_symbol(df: pl.DataFrame) -> str | None:
        """Resolve the current symbol from the base frame when available."""
        if "symbol" not in df.columns or len(df) == 0:
            return None
        try:
            values = df["symbol"].to_list()
        except Exception:
            return None
        if not values:
            return None
        symbol = str(values[0]).strip().upper()
        return symbol or None

    def _aligned_return_matrix(
        self,
        df: pl.DataFrame,
        universe_data: dict[str, pl.DataFrame] | None,
        window: int,
        *,
        exclude_current: bool = False,
    ) -> np.ndarray:
        """Build a dense aligned return matrix for the requested universe window."""
        cache = self._context_cache()
        cache_key = ("aligned_return_matrix", int(window), bool(exclude_current))
        if cache is not None and cache_key in cache:
            return cache[cache_key]
        if universe_data is None or not isinstance(universe_data, dict):
            return np.empty((len(df), 0), dtype=np.float64)

        current_symbol = self._resolve_current_symbol(df)
        aligned_returns: list[np.ndarray] = []
        for symbol_name, symbol_df in universe_data.items():
            if "close" not in symbol_df.columns:
                continue
            if exclude_current and current_symbol is not None:
                candidate = str(symbol_name).strip().upper()
                if candidate == current_symbol:
                    continue
            sym_close = symbol_df["close"].to_numpy().astype(np.float64)
            sym_ret = self._window_returns(sym_close, window)
            aligned = self._align_to_base(df, symbol_df, sym_ret)
            if aligned.shape[0] == len(df):
                aligned_returns.append(aligned)

        if not aligned_returns:
            matrix = np.empty((len(df), 0), dtype=np.float64)
        else:
            matrix = np.column_stack(aligned_returns)
        if cache is not None:
            cache[cache_key] = matrix
        return matrix

    def _aligned_log_return_universe(
        self,
        df: pl.DataFrame,
        universe_data: dict[str, pl.DataFrame] | None,
        *,
        exclude_current: bool = False,
    ) -> list[np.ndarray]:
        """Return universe log-return series aligned to the base timeline."""
        cache = self._context_cache()
        cache_key = ("aligned_log_return_universe", bool(exclude_current))
        if cache is not None and cache_key in cache:
            return cache[cache_key]
        if universe_data is None or not isinstance(universe_data, dict):
            return []

        current_symbol = self._resolve_current_symbol(df)
        aligned_returns: list[np.ndarray] = []
        for symbol_name, symbol_df in universe_data.items():
            if "close" not in symbol_df.columns:
                continue
            if exclude_current and current_symbol is not None:
                candidate = str(symbol_name).strip().upper()
                if candidate == current_symbol:
                    continue
            sym_close = symbol_df["close"].to_numpy().astype(np.float64)
            sym_ret = self._log_returns(sym_close)
            aligned = self._align_to_base(df, symbol_df, sym_ret)
            if aligned.shape[0] == len(df):
                aligned_returns.append(aligned)

        if cache is not None:
            cache[cache_key] = aligned_returns
        return aligned_returns


# =============================================================================
# RELATIVE METRICS
# =============================================================================


class RelativeStrength(CrossSectionalFeature):
    """Relative strength vs a benchmark (e.g., SPY)."""

    def __init__(self, windows: list[int] | None = None, benchmark_col: str = "benchmark"):
        super().__init__("RelativeStrength")
        self.windows = windows or [5, 10, 20, 60]
        self.benchmark_col = benchmark_col

    def compute(
        self,
        df: pl.DataFrame,
        universe_data: dict[str, pl.DataFrame] | None = None,
    ) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        # Get benchmark data if available
        if self.benchmark_col in df.columns:
            benchmark = df[self.benchmark_col].to_numpy().astype(np.float64)
        elif universe_data and "SPY" in universe_data:
            spy_df = universe_data["SPY"]
            benchmark = self._aligned_column(df, spy_df, "close")
        else:
            benchmark = None

        if benchmark is None:
            for window in self.windows:
                results[f"rs_vs_benchmark_{window}"] = self._nan_array(len(close))
            return results

        # Compute relative strength for each window (vectorized for large universes)
        for window in self.windows:
            n = len(close)
            rs = np.full(n, np.nan)
            if window <= 0 or window >= n:
                results[f"rs_vs_benchmark_{window}"] = rs
                continue

            close_base = close[:-window]
            close_next = close[window:]
            bench_base = benchmark[:-window]
            bench_next = benchmark[window:]

            asset_ret = np.full(n - window, np.nan, dtype=np.float64)
            bench_ret = np.full(n - window, np.nan, dtype=np.float64)

            valid_asset = np.isfinite(close_base) & np.isfinite(close_next) & (np.abs(close_base) > 1e-12)
            valid_bench = np.isfinite(bench_base) & np.isfinite(bench_next) & (np.abs(bench_base) > 1e-12)

            asset_ret[valid_asset] = (close_next[valid_asset] - close_base[valid_asset]) / close_base[valid_asset]
            bench_ret[valid_bench] = (bench_next[valid_bench] - bench_base[valid_bench]) / bench_base[valid_bench]

            ratio = np.full(n - window, np.nan, dtype=np.float64)
            valid_ratio = np.isfinite(asset_ret) & np.isfinite(bench_ret) & (np.abs(bench_ret) > 1e-12)
            ratio[valid_ratio] = asset_ret[valid_ratio] / bench_ret[valid_ratio]
            rs[window:] = ratio

            results[f"rs_vs_benchmark_{window}"] = rs

        return results


class BetaToMarket(CrossSectionalFeature):
    """Rolling beta to market (benchmark)."""

    def __init__(self, windows: list[int] | None = None, benchmark_col: str = "benchmark"):
        super().__init__("Beta")
        self.windows = windows or [20, 60, 120]
        self.benchmark_col = benchmark_col

    def compute(
        self,
        df: pl.DataFrame,
        universe_data: dict[str, pl.DataFrame] | None = None,
    ) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        # Get benchmark data
        if self.benchmark_col in df.columns:
            benchmark = df[self.benchmark_col].to_numpy().astype(np.float64)
        elif universe_data and "SPY" in universe_data:
            spy_df = universe_data["SPY"]
            benchmark = self._aligned_column(df, spy_df, "close")
        else:
            benchmark = None

        if benchmark is None:
            for window in self.windows:
                results[f"beta_{window}"] = self._nan_array(len(close))
            return results

        # Compute returns
        asset_ret = self._log_returns(close)
        bench_ret = self._log_returns(benchmark)

        for window in self.windows:
            results[f"beta_{window}"] = _rolling_pairwise_beta(asset_ret, bench_ret, window)

        return results


class RelativeVolume(CrossSectionalFeature):
    """Relative volume vs benchmark."""

    def __init__(self, window: int = 20, benchmark_col: str = "benchmark_volume"):
        super().__init__("RelativeVolume")
        self.window = window
        self.benchmark_col = benchmark_col

    def compute(
        self,
        df: pl.DataFrame,
        universe_data: dict[str, pl.DataFrame] | None = None,
    ) -> dict[str, np.ndarray]:
        self.validate_input(df, ["volume"])
        volume = df["volume"].to_numpy().astype(np.float64)
        results = {}

        # Get benchmark volume
        if self.benchmark_col in df.columns:
            bench_vol = df[self.benchmark_col].to_numpy().astype(np.float64)
        elif universe_data and "SPY" in universe_data:
            spy_df = universe_data["SPY"]
            bench_vol = self._aligned_column(df, spy_df, "volume")
        else:
            bench_vol = None

        if bench_vol is None:
            results[f"rel_volume_{self.window}"] = self._nan_array(len(volume))
            return results

        # Compute rolling relative volume (vectorized)
        rel_vol = np.full(len(volume), np.nan)
        window = int(self.window)
        if 0 < window <= len(volume):
            min_obs = max(3, window // 2)
            for i in range(window - 1, len(volume)):
                vol_window = volume[i - window + 1 : i + 1]
                bench_window = bench_vol[i - window + 1 : i + 1]
                valid_mask = np.isfinite(vol_window) & np.isfinite(bench_window)
                if np.sum(valid_mask) < min_obs:
                    continue

                vol_avg = np.mean(vol_window[valid_mask])
                bench_avg = np.mean(bench_window[valid_mask])
                if bench_avg > 1e-12:
                    rel_vol[i] = vol_avg / bench_avg

        results[f"rel_volume_{self.window}"] = rel_vol

        return results


# =============================================================================
# PEER COMPARISON
# =============================================================================


class PercentileRank(CrossSectionalFeature):
    """Percentile rank in universe for various metrics."""

    def __init__(self, windows: list[int] | None = None):
        super().__init__("PercentileRank")
        self.windows = windows or [5, 20, 60]

    def compute(
        self,
        df: pl.DataFrame,
        universe_data: dict[str, pl.DataFrame] | None = None,
    ) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        if universe_data is None or len(universe_data) < 2:
            for window in self.windows:
                results[f"return_percentile_{window}"] = self._nan_array(len(close))
            return results

        # Compute percentile rank
        for window in self.windows:
            percentile = np.full(len(close), np.nan)
            main_ret = self._window_returns(close, window)
            peer_returns = self._aligned_return_matrix(
                df,
                universe_data,
                window,
                exclude_current=True,
            )

            if peer_returns.shape[1] == 0:
                results[f"return_percentile_{window}"] = self._nan_array(len(close))
                continue

            combined = np.column_stack([main_ret, peer_returns])
            valid = np.isfinite(combined)
            finite_count = valid.sum(axis=1)
            main_values = main_ret.reshape(-1, 1)
            equals = np.isclose(combined, main_values, rtol=1e-9, atol=1e-12) & valid
            less_than = (combined < main_values) & valid
            percentile_valid = np.isfinite(main_ret) & (finite_count > 1)
            if np.any(percentile_valid):
                average_rank = (
                    (2.0 * less_than.sum(axis=1)) + equals.sum(axis=1) + 1.0
                ) / (2.0 * np.maximum(finite_count, 1))
                percentile[percentile_valid] = average_rank[percentile_valid]

            results[f"return_percentile_{window}"] = percentile

        return results


class ZScoreVsUniverse(CrossSectionalFeature):
    """Z-score vs universe mean/std."""

    def __init__(self, windows: list[int] | None = None):
        super().__init__("ZScoreUniverse")
        self.windows = windows or [5, 20, 60]

    def compute(
        self,
        df: pl.DataFrame,
        universe_data: dict[str, pl.DataFrame] | None = None,
    ) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        if universe_data is None or len(universe_data) < 2:
            for window in self.windows:
                results[f"zscore_universe_{window}"] = self._nan_array(len(close))
            return results

        # Compute returns for main symbol
        main_returns = {}
        for window in self.windows:
            ret = np.full(len(close), np.nan)
            ret[window:] = (close[window:] - close[:-window]) / close[:-window]
            main_returns[window] = ret

        # Compute returns for all symbols
        for window in self.windows:
            zscore = np.full(len(close), np.nan)
            main_ret = main_returns[window]
            peer_returns = self._aligned_return_matrix(
                df,
                universe_data,
                window,
                exclude_current=True,
            )

            if peer_returns.shape[1] == 0:
                results[f"zscore_universe_{window}"] = self._nan_array(len(close))
                continue

            finite = np.isfinite(peer_returns)
            counts = finite.sum(axis=1)
            safe = np.where(finite, peer_returns, 0.0)
            mean_ret = np.full(len(close), np.nan, dtype=np.float64)
            std_ret = np.full(len(close), np.nan, dtype=np.float64)
            valid_mean = counts > 0
            if np.any(valid_mean):
                mean_ret[valid_mean] = safe[valid_mean].sum(axis=1) / counts[valid_mean]
            valid_std = counts > 1
            if np.any(valid_std):
                centered = np.where(
                    finite[valid_std],
                    peer_returns[valid_std] - mean_ret[valid_std][:, None],
                    0.0,
                )
                numerator = (centered**2).sum(axis=1)
                denom = np.maximum(counts[valid_std] - 1, 1)
                std_ret[valid_std] = np.sqrt(np.maximum(numerator / denom, 0.0))
            valid = np.isfinite(main_ret) & valid_std & np.isfinite(std_ret) & (std_ret > 1e-12)
            if np.any(valid):
                zscore[valid] = (main_ret[valid] - mean_ret[valid]) / std_ret[valid]

            results[f"zscore_universe_{window}"] = zscore

        return results


class DistanceFromMedian(CrossSectionalFeature):
    """Distance from universe median."""

    def __init__(self, windows: list[int] | None = None):
        super().__init__("DistanceMedian")
        self.windows = windows or [5, 20, 60]

    def compute(
        self,
        df: pl.DataFrame,
        universe_data: dict[str, pl.DataFrame] | None = None,
    ) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        if universe_data is None or len(universe_data) < 2:
            for window in self.windows:
                results[f"dist_median_{window}"] = self._nan_array(len(close))
            return results

        # Compute returns
        main_returns = {}
        for window in self.windows:
            ret = np.full(len(close), np.nan)
            ret[window:] = (close[window:] - close[:-window]) / close[:-window]
            main_returns[window] = ret

        for window in self.windows:
            dist = np.full(len(close), np.nan)
            main_ret = main_returns[window]
            peer_returns = self._aligned_return_matrix(
                df,
                universe_data,
                window,
                exclude_current=True,
            )

            if peer_returns.shape[1] == 0:
                results[f"dist_median_{window}"] = self._nan_array(len(close))
                continue

            median_ret = np.full(len(close), np.nan, dtype=np.float64)
            valid_rows = np.isfinite(peer_returns).sum(axis=1) > 0
            if np.any(valid_rows):
                median_ret[valid_rows] = np.nanmedian(peer_returns[valid_rows], axis=1)
            valid = np.isfinite(main_ret) & np.isfinite(median_ret)
            if np.any(valid):
                dist[valid] = main_ret[valid] - median_ret[valid]

            results[f"dist_median_{window}"] = dist

        return results


class OutlierScore(CrossSectionalFeature):
    """Outlier score based on distance from universe distribution."""

    def __init__(self, window: int = 20, threshold: float = 2.0):
        super().__init__("OutlierScore")
        self.window = window
        self.threshold = threshold

    def compute(
        self,
        df: pl.DataFrame,
        universe_data: dict[str, pl.DataFrame] | None = None,
    ) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        if universe_data is None or len(universe_data) < 2:
            results["outlier_score"] = self._nan_array(len(close))
            results["is_outlier"] = self._nan_array(len(close))
            return results

        # Use ZScoreVsUniverse internally
        zscore_feature = ZScoreVsUniverse(windows=[self.window])
        zscore_results = zscore_feature.compute(df, universe_data)

        zscore = zscore_results[f"zscore_universe_{self.window}"]
        outlier_score = np.abs(zscore)
        is_outlier = self._nan_array(len(close))
        valid = np.isfinite(outlier_score)
        is_outlier[valid] = (outlier_score[valid] > self.threshold).astype(float)

        results["outlier_score"] = outlier_score
        results["is_outlier"] = is_outlier

        return results


# =============================================================================
# CORRELATION & COVARIANCE
# =============================================================================


class RollingCorrelation(CrossSectionalFeature):
    """Rolling correlation with benchmark and other assets."""

    def __init__(self, window: int = 60, benchmark_col: str = "benchmark"):
        super().__init__("RollingCorrelation")
        self.window = window
        self.benchmark_col = benchmark_col

    def compute(
        self,
        df: pl.DataFrame,
        universe_data: dict[str, pl.DataFrame] | None = None,
    ) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        # Get benchmark
        if self.benchmark_col in df.columns:
            benchmark = df[self.benchmark_col].to_numpy().astype(np.float64)
        elif universe_data and "SPY" in universe_data:
            spy_df = universe_data["SPY"]
            benchmark = self._aligned_column(df, spy_df, "close")
        else:
            benchmark = None

        # Compute returns
        asset_ret = self._log_returns(close)

        if benchmark is not None:
            bench_ret = self._log_returns(benchmark)
            results[f"corr_benchmark_{self.window}"] = _rolling_pairwise_correlation(
                asset_ret,
                bench_ret,
                self.window,
            )
        else:
            results[f"corr_benchmark_{self.window}"] = self._nan_array(len(close))

        # Correlation with VIX if available
        if universe_data and "VIX" in universe_data:
            vix_df = universe_data["VIX"]
            vix_aligned = self._aligned_column(df, vix_df, "close")
            if vix_aligned is not None:
                vix_ret = self._log_returns(vix_aligned)
                results[f"corr_vix_{self.window}"] = _rolling_pairwise_correlation(
                    asset_ret,
                    vix_ret,
                    self.window,
                )

        return results


class CorrelationChange(CrossSectionalFeature):
    """Detect changes in correlation regime."""

    def __init__(self, window: int = 60, lookback: int = 20):
        super().__init__("CorrelationChange")
        self.window = window
        self.lookback = lookback

    def compute(
        self,
        df: pl.DataFrame,
        universe_data: dict[str, pl.DataFrame] | None = None,
    ) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        # First compute rolling correlation with benchmark
        corr_feature = RollingCorrelation(window=self.window)
        corr_results = corr_feature.compute(df, universe_data)

        corr = corr_results.get(
            f"corr_benchmark_{self.window}",
            self._nan_array(len(close)),
        )

        # Compute change in correlation
        corr_change = np.full(len(close), np.nan)
        for i in range(self.window + self.lookback, len(close)):
            recent_corr = np.nanmean(corr[i - self.lookback + 1 : i + 1])
            prev_corr = np.nanmean(corr[i - self.window + 1 : i - self.lookback + 1])
            corr_change[i] = recent_corr - prev_corr

        results["corr_change"] = corr_change

        return results


# =============================================================================
# PCA-BASED FEATURES
# =============================================================================


class FactorLoadings(CrossSectionalFeature):
    """Factor loadings from PCA on universe returns."""

    def __init__(self, window: int = 60, n_factors: int = 5):
        super().__init__("FactorLoadings")
        self.window = window
        self.n_factors = n_factors

    def compute(
        self,
        df: pl.DataFrame,
        universe_data: dict[str, pl.DataFrame] | None = None,
    ) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        # Initialize outputs
        for i in range(self.n_factors):
            results[f"factor_loading_{i + 1}"] = np.full(len(close), np.nan)
        results["idiosyncratic_vol"] = np.full(len(close), np.nan)

        if universe_data is None or len(universe_data) < self.n_factors:
            for i in range(self.n_factors):
                results[f"factor_loading_{i + 1}"] = self._nan_array(len(close))
            results["idiosyncratic_vol"] = self._nan_array(len(close))
            return results

        # Compute returns for universe and align to main symbol timeline
        all_returns = self._aligned_log_return_universe(df, universe_data, exclude_current=False)

        if len(all_returns) < self.n_factors:
            for i in range(self.n_factors):
                results[f"factor_loading_{i + 1}"] = self._nan_array(len(close))
            results["idiosyncratic_vol"] = self._nan_array(len(close))
            return results

        main_ret = self._log_returns(close)
        step = _adaptive_sampling_step(len(close), 1)

        # Rolling PCA + regression for factor exposures
        for t in _sample_indices(self.window, len(close), step):
            window_slice = slice(t - self.window + 1, t + 1)
            return_matrix = np.column_stack([r[window_slice] for r in all_returns])
            main_window = main_ret[window_slice]

            valid_rows = np.isfinite(main_window) & ~np.any(~np.isfinite(return_matrix), axis=1)
            if np.sum(valid_rows) < max(self.n_factors + 2, 8):
                continue

            return_matrix = return_matrix[valid_rows]
            main_window = main_window[valid_rows]

            # Drop zero-variance assets for numerical stability
            asset_std = np.std(return_matrix, axis=0, ddof=1)
            non_constant_assets = asset_std > 1e-12
            if np.sum(non_constant_assets) < self.n_factors:
                continue

            return_matrix = return_matrix[:, non_constant_assets]
            matrix_mean = np.mean(return_matrix, axis=0)
            matrix_std = np.std(return_matrix, axis=0, ddof=1)
            matrix_std = np.where(matrix_std <= 1e-12, 1.0, matrix_std)
            standardized = (return_matrix - matrix_mean) / matrix_std

            main_std = np.std(main_window, ddof=1)
            if main_std <= 1e-12:
                continue
            main_standardized = (main_window - np.mean(main_window)) / main_std

            try:
                # Principal component directions for cross-sectional returns
                _, _, vt = np.linalg.svd(standardized, full_matrices=False)
                n_components = min(self.n_factors, vt.shape[0], vt.shape[1])
                if n_components <= 0:
                    continue

                factor_scores = standardized @ vt[:n_components].T
                if factor_scores.shape[0] <= n_components:
                    continue

                # Main asset loading to PCA factor time series
                loadings, _, _, _ = np.linalg.lstsq(factor_scores, main_standardized, rcond=None)

                for i in range(n_components):
                    results[f"factor_loading_{i + 1}"][t] = loadings[i]

                residuals = main_standardized - factor_scores @ loadings
                if residuals.size > 1:
                    results["idiosyncratic_vol"][t] = np.std(residuals, ddof=1)

            except (np.linalg.LinAlgError, ValueError):
                continue

        if step > 1:
            for i in range(self.n_factors):
                key = f"factor_loading_{i + 1}"
                results[key] = _forward_fill_nan(results[key])
            results["idiosyncratic_vol"] = _forward_fill_nan(results["idiosyncratic_vol"])

        return results


class IdiosyncraticVolatility(CrossSectionalFeature):
    """Idiosyncratic volatility (volatility not explained by market)."""

    def __init__(self, window: int = 60):
        super().__init__("IdioVol")
        self.window = window

    def compute(
        self,
        df: pl.DataFrame,
        universe_data: dict[str, pl.DataFrame] | None = None,
    ) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        # Compute beta first
        beta_feature = BetaToMarket(windows=[self.window])
        beta_results = beta_feature.compute(df, universe_data)
        beta = beta_results[f"beta_{self.window}"]

        # Get benchmark returns
        if "benchmark" in df.columns:
            benchmark = df["benchmark"].to_numpy().astype(np.float64)
        elif universe_data and "SPY" in universe_data:
            spy_df = universe_data["SPY"]
            benchmark = self._aligned_column(df, spy_df, "close")
        else:
            benchmark = None

        if benchmark is None:
            # Return total volatility if no benchmark
            asset_ret = np.diff(np.log(close), prepend=np.nan)
            idio_vol = _rolling_nanstd(asset_ret, self.window)
            results[f"idio_vol_{self.window}"] = idio_vol
            return results

        # Compute returns
        asset_ret = np.diff(np.log(close), prepend=np.nan)
        bench_ret = np.diff(np.log(benchmark), prepend=np.nan)

        # Compute idiosyncratic volatility as residual volatility
        residuals = asset_ret - (beta * bench_ret)
        idio_vol = _rolling_nanstd(residuals, self.window)

        results[f"idio_vol_{self.window}"] = idio_vol

        return results


# =============================================================================
# NETWORK FEATURES
# =============================================================================


class CorrelationCentrality(CrossSectionalFeature):
    """Centrality in the correlation network."""

    def __init__(self, window: int = 60):
        super().__init__("CorrCentrality")
        self.window = window

    def compute(
        self,
        df: pl.DataFrame,
        universe_data: dict[str, pl.DataFrame] | None = None,
    ) -> dict[str, np.ndarray]:
        self.validate_input(df, ["close"])
        close = df["close"].to_numpy().astype(np.float64)
        results = {}

        if universe_data is None or len(universe_data) < 3:
            results["corr_centrality"] = self._nan_array(len(close))
            return results

        # Compute returns for all symbols
        all_returns = self._aligned_log_return_universe(df, universe_data, exclude_current=False)

        main_ret = self._log_returns(close)

        if len(all_returns) < 2:
            results["corr_centrality"] = self._nan_array(len(close))
            return results

        correlation_columns: list[np.ndarray] = []
        for other_ret in all_returns:
            corr = _rolling_pairwise_correlation(main_ret, other_ret, self.window)
            correlation_columns.append(np.abs(corr))

        if correlation_columns:
            stacked = np.column_stack(correlation_columns)
            finite = np.isfinite(stacked)
            counts = finite.sum(axis=1)
            centrality = np.full(len(close), np.nan, dtype=np.float64)
            valid = counts > 0
            if np.any(valid):
                safe = np.where(finite, stacked, 0.0)
                centrality[valid] = safe[valid].sum(axis=1) / counts[valid]
        else:
            centrality = np.full(len(close), np.nan)

        results["corr_centrality"] = centrality

        return results


class AverageCorrelation(CrossSectionalFeature):
    """Average correlation with universe (measures market regime)."""

    def __init__(self, window: int = 60):
        super().__init__("AvgCorrelation")
        self.window = window

    def compute(
        self,
        df: pl.DataFrame,
        universe_data: dict[str, pl.DataFrame] | None = None,
    ) -> dict[str, np.ndarray]:
        # Similar to centrality but signed
        centrality_feature = CorrelationCentrality(window=self.window)
        results = centrality_feature.compute(df, universe_data)

        # Rename for clarity
        if "corr_centrality" in results:
            results["avg_correlation"] = results.pop("corr_centrality")

        return results


# =============================================================================
# COMPOSITE CALCULATOR
# =============================================================================


class CrossSectionalFeatureCalculator:
    """
    Main calculator class that computes all cross-sectional features.

    Usage:
        calculator = CrossSectionalFeatureCalculator()
        features = calculator.compute_all(df, universe_data)
    """

    def __init__(
        self,
        include_all: bool = True,
        robust_normalize: bool = True,
    ):
        self.features: list[CrossSectionalFeature] = []
        self.robust_normalize = bool(robust_normalize)

        if include_all:
            self._add_default_features()

    @staticmethod
    def _sanitize_output(values: np.ndarray, expected_length: int) -> np.ndarray:
        """Ensure a valid float64 feature array aligned to base length."""
        if expected_length <= 0:
            return np.array([], dtype=np.float64)

        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size >= expected_length:
            arr = arr[-expected_length:]
        elif arr.size == 0:
            arr = np.full(expected_length, np.nan, dtype=np.float64)
        else:
            padded = np.full(expected_length, np.nan, dtype=np.float64)
            padded[-arr.size:] = arr
            arr = padded

        arr[~np.isfinite(arr)] = np.nan
        return arr

    @staticmethod
    def _is_binary_feature(values: np.ndarray) -> bool:
        """Detect binary outputs (e.g., indicator flags)."""
        valid = values[np.isfinite(values)]
        if valid.size == 0:
            return False
        unique = np.unique(valid)
        return bool(unique.size <= 2 and np.all(np.isin(unique, [0.0, 1.0])))

    def _normalize_output(self, values: np.ndarray) -> np.ndarray:
        """Robustly normalize heavy-tailed continuous feature outputs."""
        arr = values.copy()
        valid = arr[np.isfinite(arr)]
        if valid.size < 20 or self._is_binary_feature(arr):
            return arr

        # Preserve naturally bounded ratio/probability style features.
        min_valid = float(np.min(valid))
        max_valid = float(np.max(valid))
        if 0.0 <= min_valid and max_valid <= 1.0:
            return arr

        lower, upper = np.percentile(valid, [1, 99])
        clipped = np.clip(arr, lower, upper)
        clipped_valid = clipped[np.isfinite(clipped)]
        if clipped_valid.size < 20:
            return clipped

        median = np.median(clipped_valid)
        q75, q25 = np.percentile(clipped_valid, [75, 25])
        iqr = q75 - q25
        if iqr <= 1e-12:
            return clipped

        normalized = (clipped - median) / iqr
        return np.clip(normalized, -10.0, 10.0)

    def _add_default_features(self) -> None:
        """Add all default cross-sectional features."""
        # Relative metrics
        self.features.extend([
            RelativeStrength(),
            BetaToMarket(),
            RelativeVolume(),
        ])

        # Peer comparison
        self.features.extend([
            PercentileRank(),
            ZScoreVsUniverse(),
            DistanceFromMedian(),
            OutlierScore(),
        ])

        # Correlation
        self.features.extend([
            RollingCorrelation(),
            CorrelationChange(),
        ])

        # PCA-based
        self.features.extend([
            FactorLoadings(),
            IdiosyncraticVolatility(),
        ])

        # Network
        self.features.extend([
            CorrelationCentrality(),
            AverageCorrelation(),
        ])

    def add_feature(self, feature: CrossSectionalFeature) -> None:
        """Add a custom feature."""
        self.features.append(feature)

    def compute_all(
        self,
        df: pl.DataFrame,
        universe_data: dict[str, pl.DataFrame] | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute all features and return combined results."""
        results: dict[str, np.ndarray] = {}
        expected_length = len(df)
        context = {"cache": {}}
        for feature in self.features:
            try:
                setattr(feature, "_context", context)
                feature_results = feature.compute(df, universe_data)
                for name, values in feature_results.items():
                    sanitized = self._sanitize_output(values, expected_length)
                    if self.robust_normalize:
                        sanitized = self._normalize_output(sanitized)
                    results[name] = sanitized
            except ValueError:
                # Skip features that can't be computed
                continue
            finally:
                if hasattr(feature, "_context"):
                    delattr(feature, "_context")
        return results

    def get_feature_names(self) -> list[str]:
        """Get list of feature names."""
        return [f.name for f in self.features]

    def get_feature_count(
        self,
        df: pl.DataFrame,
        universe_data: dict[str, pl.DataFrame] | None = None,
    ) -> int:
        """Get total number of features that will be computed."""
        return len(self.compute_all(df, universe_data))
