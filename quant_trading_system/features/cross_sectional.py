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
import polars as pl
from scipy import stats


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
            if "close" in spy_df.columns:
                benchmark = spy_df["close"].to_numpy().astype(np.float64)
                # Align lengths
                min_len = min(len(close), len(benchmark))
                close = close[:min_len]
                benchmark = benchmark[:min_len]
            else:
                benchmark = None
        else:
            benchmark = None

        if benchmark is None:
            # Return zeros if no benchmark
            for window in self.windows:
                results[f"rs_vs_benchmark_{window}"] = np.zeros(len(close))
            return results

        # Compute relative strength for each window
        for window in self.windows:
            rs = np.full(len(close), np.nan)
            for i in range(window, len(close)):
                asset_ret = (close[i] - close[i - window]) / close[i - window]
                bench_ret = (benchmark[i] - benchmark[i - window]) / benchmark[i - window]

                if bench_ret != 0:
                    rs[i] = asset_ret / bench_ret
                else:
                    rs[i] = 0

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
            if "close" in spy_df.columns:
                benchmark = spy_df["close"].to_numpy().astype(np.float64)
                min_len = min(len(close), len(benchmark))
                close = close[:min_len]
                benchmark = benchmark[:min_len]
            else:
                benchmark = None
        else:
            benchmark = None

        if benchmark is None:
            for window in self.windows:
                results[f"beta_{window}"] = np.ones(len(close))
            return results

        # Compute returns
        asset_ret = np.diff(np.log(close), prepend=np.nan)
        bench_ret = np.diff(np.log(benchmark), prepend=np.nan)

        for window in self.windows:
            beta = np.full(len(close), np.nan)
            for i in range(window, len(close)):
                asset_window = asset_ret[i - window + 1 : i + 1]
                bench_window = bench_ret[i - window + 1 : i + 1]

                # Remove NaNs
                valid_mask = ~(np.isnan(asset_window) | np.isnan(bench_window))
                if np.sum(valid_mask) > 2:
                    cov = np.cov(asset_window[valid_mask], bench_window[valid_mask])
                    if cov[1, 1] > 0:
                        beta[i] = cov[0, 1] / cov[1, 1]

            results[f"beta_{window}"] = beta

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
            if "volume" in spy_df.columns:
                bench_vol = spy_df["volume"].to_numpy().astype(np.float64)
                min_len = min(len(volume), len(bench_vol))
                volume = volume[:min_len]
                bench_vol = bench_vol[:min_len]
            else:
                bench_vol = None
        else:
            bench_vol = None

        if bench_vol is None:
            results[f"rel_volume_{self.window}"] = np.ones(len(volume))
            return results

        # Compute rolling relative volume
        rel_vol = np.full(len(volume), np.nan)
        for i in range(self.window - 1, len(volume)):
            vol_avg = np.mean(volume[i - self.window + 1 : i + 1])
            bench_avg = np.mean(bench_vol[i - self.window + 1 : i + 1])
            if bench_avg > 0:
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
            # Return 50th percentile if no universe data
            for window in self.windows:
                results[f"return_percentile_{window}"] = np.full(len(close), 0.5)
            return results

        # Compute returns for main symbol
        main_returns = {}
        for window in self.windows:
            ret = np.full(len(close), np.nan)
            ret[window:] = (close[window:] - close[:-window]) / close[:-window]
            main_returns[window] = ret

        # Compute returns for all symbols in universe
        universe_returns = {window: [] for window in self.windows}
        for symbol, symbol_df in universe_data.items():
            if "close" not in symbol_df.columns:
                continue
            sym_close = symbol_df["close"].to_numpy().astype(np.float64)
            for window in self.windows:
                ret = np.full(len(sym_close), np.nan)
                ret[window:] = (sym_close[window:] - sym_close[:-window]) / sym_close[:-window]
                universe_returns[window].append(ret)

        # Compute percentile rank
        for window in self.windows:
            percentile = np.full(len(close), np.nan)
            main_ret = main_returns[window]
            all_returns = universe_returns[window]

            if not all_returns:
                results[f"return_percentile_{window}"] = np.full(len(close), 0.5)
                continue

            # Align lengths
            min_len = min(len(main_ret), min(len(r) for r in all_returns))

            for i in range(window, min_len):
                if np.isnan(main_ret[i]):
                    continue

                # Get cross-sectional returns at this point
                cs_returns = [main_ret[i]]
                for ret in all_returns:
                    if i < len(ret) and not np.isnan(ret[i]):
                        cs_returns.append(ret[i])

                if len(cs_returns) > 1:
                    percentile[i] = stats.percentileofscore(cs_returns, main_ret[i]) / 100

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
                results[f"zscore_universe_{window}"] = np.zeros(len(close))
            return results

        # Compute returns for main symbol
        main_returns = {}
        for window in self.windows:
            ret = np.full(len(close), np.nan)
            ret[window:] = (close[window:] - close[:-window]) / close[:-window]
            main_returns[window] = ret

        # Compute returns for all symbols
        universe_returns = {window: [] for window in self.windows}
        for symbol, symbol_df in universe_data.items():
            if "close" not in symbol_df.columns:
                continue
            sym_close = symbol_df["close"].to_numpy().astype(np.float64)
            for window in self.windows:
                ret = np.full(len(sym_close), np.nan)
                ret[window:] = (sym_close[window:] - sym_close[:-window]) / sym_close[:-window]
                universe_returns[window].append(ret)

        for window in self.windows:
            zscore = np.full(len(close), np.nan)
            main_ret = main_returns[window]
            all_returns = universe_returns[window]

            if not all_returns:
                results[f"zscore_universe_{window}"] = np.zeros(len(close))
                continue

            min_len = min(len(main_ret), min(len(r) for r in all_returns))

            for i in range(window, min_len):
                if np.isnan(main_ret[i]):
                    continue

                cs_returns = []
                for ret in all_returns:
                    if i < len(ret) and not np.isnan(ret[i]):
                        cs_returns.append(ret[i])

                if len(cs_returns) > 1:
                    mean_ret = np.mean(cs_returns)
                    std_ret = np.std(cs_returns, ddof=1)
                    if std_ret > 0:
                        zscore[i] = (main_ret[i] - mean_ret) / std_ret

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
                results[f"dist_median_{window}"] = np.zeros(len(close))
            return results

        # Compute returns
        main_returns = {}
        for window in self.windows:
            ret = np.full(len(close), np.nan)
            ret[window:] = (close[window:] - close[:-window]) / close[:-window]
            main_returns[window] = ret

        universe_returns = {window: [] for window in self.windows}
        for symbol, symbol_df in universe_data.items():
            if "close" not in symbol_df.columns:
                continue
            sym_close = symbol_df["close"].to_numpy().astype(np.float64)
            for window in self.windows:
                ret = np.full(len(sym_close), np.nan)
                ret[window:] = (sym_close[window:] - sym_close[:-window]) / sym_close[:-window]
                universe_returns[window].append(ret)

        for window in self.windows:
            dist = np.full(len(close), np.nan)
            main_ret = main_returns[window]
            all_returns = universe_returns[window]

            if not all_returns:
                results[f"dist_median_{window}"] = np.zeros(len(close))
                continue

            min_len = min(len(main_ret), min(len(r) for r in all_returns))

            for i in range(window, min_len):
                if np.isnan(main_ret[i]):
                    continue

                cs_returns = []
                for ret in all_returns:
                    if i < len(ret) and not np.isnan(ret[i]):
                        cs_returns.append(ret[i])

                if len(cs_returns) > 0:
                    median_ret = np.median(cs_returns)
                    dist[i] = main_ret[i] - median_ret

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
            results["outlier_score"] = np.zeros(len(close))
            results["is_outlier"] = np.zeros(len(close))
            return results

        # Use ZScoreVsUniverse internally
        zscore_feature = ZScoreVsUniverse(windows=[self.window])
        zscore_results = zscore_feature.compute(df, universe_data)

        zscore = zscore_results[f"zscore_universe_{self.window}"]
        outlier_score = np.abs(zscore)
        is_outlier = np.where(outlier_score > self.threshold, 1, 0)

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
            if "close" in spy_df.columns:
                benchmark = spy_df["close"].to_numpy().astype(np.float64)
                min_len = min(len(close), len(benchmark))
                close = close[:min_len]
                benchmark = benchmark[:min_len]
            else:
                benchmark = None
        else:
            benchmark = None

        # Compute returns
        asset_ret = np.diff(np.log(close), prepend=np.nan)

        if benchmark is not None:
            bench_ret = np.diff(np.log(benchmark), prepend=np.nan)

            corr_benchmark = np.full(len(close), np.nan)
            for i in range(self.window, len(close)):
                asset_window = asset_ret[i - self.window + 1 : i + 1]
                bench_window = bench_ret[i - self.window + 1 : i + 1]

                valid_mask = ~(np.isnan(asset_window) | np.isnan(bench_window))
                if np.sum(valid_mask) > 2:
                    corr = np.corrcoef(asset_window[valid_mask], bench_window[valid_mask])
                    corr_benchmark[i] = corr[0, 1]

            results[f"corr_benchmark_{self.window}"] = corr_benchmark
        else:
            results[f"corr_benchmark_{self.window}"] = np.zeros(len(close))

        # Correlation with VIX if available
        if universe_data and "VIX" in universe_data:
            vix_df = universe_data["VIX"]
            if "close" in vix_df.columns:
                vix = vix_df["close"].to_numpy().astype(np.float64)
                min_len = min(len(close), len(vix))
                vix_ret = np.diff(np.log(vix[:min_len]), prepend=np.nan)

                corr_vix = np.full(min_len, np.nan)
                for i in range(self.window, min_len):
                    asset_window = asset_ret[i - self.window + 1 : i + 1]
                    vix_window = vix_ret[i - self.window + 1 : i + 1]

                    valid_mask = ~(np.isnan(asset_window) | np.isnan(vix_window))
                    if np.sum(valid_mask) > 2:
                        corr = np.corrcoef(asset_window[valid_mask], vix_window[valid_mask])
                        corr_vix[i] = corr[0, 1]

                # Pad to original length
                if min_len < len(close):
                    corr_vix = np.pad(corr_vix, (0, len(close) - min_len), constant_values=np.nan)

                results[f"corr_vix_{self.window}"] = corr_vix

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

        corr = corr_results.get(f"corr_benchmark_{self.window}", np.zeros(len(close)))

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

        if universe_data is None or len(universe_data) < self.n_factors:
            for i in range(self.n_factors):
                results[f"factor_loading_{i + 1}"] = np.zeros(len(close))
            results["idiosyncratic_vol"] = np.zeros(len(close))
            return results

        # Compute returns for all symbols
        all_returns = []
        symbol_names = []
        main_symbol_idx = None

        for idx, (symbol, symbol_df) in enumerate(universe_data.items()):
            if "close" not in symbol_df.columns:
                continue
            sym_close = symbol_df["close"].to_numpy().astype(np.float64)
            ret = np.diff(np.log(sym_close), prepend=np.nan)
            all_returns.append(ret)
            symbol_names.append(symbol)

        # Add main symbol if not in universe
        main_ret = np.diff(np.log(close), prepend=np.nan)
        if len(all_returns) > 0:
            # Align lengths
            min_len = min(len(main_ret), min(len(r) for r in all_returns))
            main_ret = main_ret[:min_len]
            all_returns = [r[:min_len] for r in all_returns]

        # Initialize outputs
        for i in range(self.n_factors):
            results[f"factor_loading_{i + 1}"] = np.full(len(close), np.nan)
        results["idiosyncratic_vol"] = np.full(len(close), np.nan)

        if len(all_returns) < self.n_factors:
            return results

        # Rolling PCA
        for t in range(self.window, min_len):
            # Build return matrix for window
            return_matrix = np.column_stack(
                [r[t - self.window + 1 : t + 1] for r in all_returns]
            )

            # Handle NaNs
            valid_rows = ~np.any(np.isnan(return_matrix), axis=1)
            if np.sum(valid_rows) < self.n_factors:
                continue

            return_matrix = return_matrix[valid_rows]

            # Standardize
            mean = np.mean(return_matrix, axis=0)
            std = np.std(return_matrix, axis=0, ddof=1)
            std = np.where(std == 0, 1, std)
            standardized = (return_matrix - mean) / std

            try:
                # SVD for PCA
                _, s, Vt = np.linalg.svd(standardized, full_matrices=False)

                # Get loadings for main symbol (project onto factors)
                main_window = main_ret[t - self.window + 1 : t + 1]
                main_window = main_window[valid_rows]
                main_standardized = (main_window - np.mean(main_window)) / (
                    np.std(main_window, ddof=1) or 1
                )

                loadings = main_standardized @ Vt.T[: self.n_factors].T

                for i in range(min(self.n_factors, len(loadings))):
                    results[f"factor_loading_{i + 1}"][t] = loadings[i]

                # Idiosyncratic volatility (residual after removing factor exposure)
                factor_returns = standardized @ Vt.T[: self.n_factors].T @ Vt[: self.n_factors]
                residuals = main_standardized - (
                    main_standardized @ Vt.T[: self.n_factors].T @ Vt[: self.n_factors]
                ).mean()
                results["idiosyncratic_vol"][t] = np.std(residuals, ddof=1)

            except np.linalg.LinAlgError:
                continue

        # Pad back to original length
        if min_len < len(close):
            for key in results:
                results[key] = np.pad(
                    results[key][:min_len],
                    (0, len(close) - min_len),
                    constant_values=np.nan,
                )

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
            if "close" in spy_df.columns:
                benchmark = spy_df["close"].to_numpy().astype(np.float64)
                min_len = min(len(close), len(benchmark))
                close = close[:min_len]
                benchmark = benchmark[:min_len]
            else:
                benchmark = None
        else:
            benchmark = None

        if benchmark is None:
            # Return total volatility if no benchmark
            asset_ret = np.diff(np.log(close), prepend=np.nan)
            idio_vol = np.full(len(close), np.nan)
            for i in range(self.window - 1, len(close)):
                idio_vol[i] = np.nanstd(asset_ret[i - self.window + 1 : i + 1], ddof=1)
            results[f"idio_vol_{self.window}"] = idio_vol
            return results

        # Compute returns
        asset_ret = np.diff(np.log(close), prepend=np.nan)
        bench_ret = np.diff(np.log(benchmark), prepend=np.nan)

        # Compute idiosyncratic volatility as residual volatility
        idio_vol = np.full(len(close), np.nan)
        for i in range(self.window, len(close)):
            if np.isnan(beta[i]):
                continue

            asset_window = asset_ret[i - self.window + 1 : i + 1]
            bench_window = bench_ret[i - self.window + 1 : i + 1]

            # Residuals from market model
            residuals = asset_window - beta[i] * bench_window

            valid = ~np.isnan(residuals)
            if np.sum(valid) > 2:
                idio_vol[i] = np.std(residuals[valid], ddof=1)

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
            results["corr_centrality"] = np.zeros(len(close))
            return results

        # Compute returns for all symbols
        all_returns = []
        for symbol, symbol_df in universe_data.items():
            if "close" not in symbol_df.columns:
                continue
            sym_close = symbol_df["close"].to_numpy().astype(np.float64)
            ret = np.diff(np.log(sym_close), prepend=np.nan)
            all_returns.append(ret)

        main_ret = np.diff(np.log(close), prepend=np.nan)

        if len(all_returns) < 2:
            results["corr_centrality"] = np.zeros(len(close))
            return results

        # Align lengths
        min_len = min(len(main_ret), min(len(r) for r in all_returns))
        main_ret = main_ret[:min_len]
        all_returns = [r[:min_len] for r in all_returns]

        centrality = np.full(len(close), np.nan)

        for t in range(self.window, min_len):
            # Compute correlation of main asset with all others
            correlations = []
            for other_ret in all_returns:
                main_window = main_ret[t - self.window + 1 : t + 1]
                other_window = other_ret[t - self.window + 1 : t + 1]

                valid_mask = ~(np.isnan(main_window) | np.isnan(other_window))
                if np.sum(valid_mask) > 2:
                    corr = np.corrcoef(main_window[valid_mask], other_window[valid_mask])
                    correlations.append(abs(corr[0, 1]))

            if correlations:
                # Centrality = average absolute correlation
                centrality[t] = np.mean(correlations)

        # Pad to original length
        if min_len < len(close):
            centrality = np.pad(
                centrality[:min_len],
                (0, len(close) - min_len),
                constant_values=np.nan,
            )

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

    def __init__(self, include_all: bool = True):
        self.features: list[CrossSectionalFeature] = []

        if include_all:
            self._add_default_features()

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
        results = {}
        for feature in self.features:
            try:
                feature_results = feature.compute(df, universe_data)
                results.update(feature_results)
            except ValueError:
                # Skip features that can't be computed
                continue
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
