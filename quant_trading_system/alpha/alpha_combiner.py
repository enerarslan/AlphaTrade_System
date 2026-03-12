"""
Alpha combination and blending module.

Provides methods for combining multiple alpha factors into composite signals,
including weighting schemes, orthogonalization, and neutralization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import polars as pl
from scipy import optimize
from scipy import stats

from .alpha_base import AlphaFactor, AlphaSignal, AlphaType, AlphaHorizon


def _align_alpha_with_forward_returns(
    alpha_values: np.ndarray,
    returns: np.ndarray,
    return_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align alpha and return arrays with an explicit lag convention."""
    alpha_arr = np.asarray(alpha_values, dtype=float).reshape(-1)
    returns_arr = np.asarray(returns, dtype=float).reshape(-1)
    lag = max(0, int(return_lag))

    if lag > 0 and alpha_arr.size > lag and returns_arr.size > lag:
        aligned_alpha = alpha_arr[:-lag]
        aligned_returns = returns_arr[lag:]
    else:
        aligned_alpha = alpha_arr
        aligned_returns = returns_arr

    min_len = min(int(aligned_alpha.size), int(aligned_returns.size))
    if min_len <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    return aligned_alpha[-min_len:], aligned_returns[-min_len:]


class WeightingMethod(str, Enum):
    """Alpha weighting method enumeration."""

    EQUAL = "equal"
    IC_WEIGHTED = "ic_weighted"
    SHARPE_WEIGHTED = "sharpe_weighted"
    INVERSE_VOLATILITY = "inverse_volatility"
    OPTIMIZED = "optimized"
    RANK_WEIGHTED = "rank_weighted"
    DECAY_WEIGHTED = "decay_weighted"


class NeutralizationMethod(str, Enum):
    """Alpha neutralization method enumeration."""

    MARKET = "market"
    SECTOR = "sector"
    FACTOR = "factor"
    BETA = "beta"
    NONE = "none"


class OrthogonalizationMethod(str, Enum):
    """Alpha orthogonalization method enumeration."""

    PCA = "pca"
    GRAM_SCHMIDT = "gram_schmidt"
    DECORRELATE = "decorrelate"
    NONE = "none"


class WeightUpdateMode(str, Enum):
    """OOS weight update mode."""

    EXPANDING = "expanding"
    ROLLING = "rolling"


@dataclass
class AlphaWeight:
    """Container for alpha weight information."""

    alpha_name: str
    weight: float
    ic: float = 0.0
    sharpe: float = 0.0
    volatility: float = 0.0
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alpha_name": self.alpha_name,
            "weight": self.weight,
            "ic": self.ic,
            "sharpe": self.sharpe,
            "volatility": self.volatility,
            "rank": self.rank,
        }


@dataclass
class CombinerConfig:
    """Configuration for alpha combiner."""

    weighting_method: WeightingMethod = WeightingMethod.EQUAL
    neutralization_method: NeutralizationMethod = NeutralizationMethod.NONE
    orthogonalization_method: OrthogonalizationMethod = OrthogonalizationMethod.NONE

    # Weight constraints
    max_weight: float = 0.5
    min_weight: float = 0.0
    sum_to_one: bool = True

    # Rolling window for metrics
    lookback_window: int = 60
    ic_lookback: int = 20
    vol_lookback: int = 20
    return_lag: int = 1
    oos_update_mode: WeightUpdateMode = WeightUpdateMode.EXPANDING
    oos_update_window: int = 252
    oos_update_blend: float = 0.5
    oos_min_observations: int = 30

    # Optimization settings
    turnover_penalty: float = 0.1
    diversification_weight: float = 0.1

    # Decay settings
    decay_halflife: int = 20

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "weighting_method": self.weighting_method.value,
            "neutralization_method": self.neutralization_method.value,
            "orthogonalization_method": self.orthogonalization_method.value,
            "max_weight": self.max_weight,
            "min_weight": self.min_weight,
            "sum_to_one": self.sum_to_one,
            "lookback_window": self.lookback_window,
            "return_lag": self.return_lag,
            "oos_update_mode": self.oos_update_mode.value,
            "oos_update_window": self.oos_update_window,
            "oos_update_blend": self.oos_update_blend,
            "oos_min_observations": self.oos_min_observations,
        }

    def __post_init__(self) -> None:
        if self.return_lag < 0:
            raise ValueError("return_lag must be non-negative")
        if self.oos_update_window <= 0:
            raise ValueError("oos_update_window must be positive")
        if self.oos_min_observations <= 0:
            raise ValueError("oos_min_observations must be positive")
        if not (0.0 <= self.oos_update_blend <= 1.0):
            raise ValueError("oos_update_blend must be in [0, 1]")


class AlphaWeighter(ABC):
    """Abstract base class for alpha weighting strategies."""

    @abstractmethod
    def compute_weights(
        self,
        alpha_values: dict[str, np.ndarray],
        returns: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Compute weights for each alpha.

        Args:
            alpha_values: Dictionary mapping alpha names to their value arrays
            returns: Optional forward returns for IC-based weighting
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping alpha names to weights
        """
        pass


class EqualWeighter(AlphaWeighter):
    """Equal weighting for all alphas."""

    def compute_weights(
        self,
        alpha_values: dict[str, np.ndarray],
        returns: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Assign equal weights to all alphas."""
        n_alphas = len(alpha_values)
        weight = 1.0 / n_alphas if n_alphas > 0 else 0.0
        return {name: weight for name in alpha_values}


class ICWeighter(AlphaWeighter):
    """Weight alphas by their Information Coefficient.

    CRITICAL: IC is calculated using LAGGED alpha values vs FORWARD returns
    to avoid look-ahead bias. Alpha at time t is correlated with returns
    at time t+1 (not contemporaneous returns).
    """

    def __init__(self, lookback: int = 20, min_ic: float = 0.0, return_lag: int = 1):
        """
        Initialize IC weighter.

        Args:
            lookback: Lookback period for IC calculation
            min_ic: Minimum IC to include alpha (default 0 = no threshold)
            return_lag: Number of periods to lag returns (default 1 = forward returns)
        """
        self.lookback = lookback
        self.min_ic = min_ic
        self.return_lag = return_lag

    def compute_weights(
        self,
        alpha_values: dict[str, np.ndarray],
        returns: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Weight by rolling IC.

        CRITICAL FIX: Uses lagged alpha values with forward returns to prevent
        look-ahead bias. IC(alpha[t], returns[t+lag]) instead of IC(alpha[t], returns[t]).
        """
        if returns is None:
            return EqualWeighter().compute_weights(alpha_values)

        ics = {}
        for name, values in alpha_values.items():
            lagged_alpha, forward_returns = _align_alpha_with_forward_returns(
                values,
                returns,
                self.return_lag,
            )

            # Compute rank IC
            valid_mask = ~(np.isnan(lagged_alpha) | np.isnan(forward_returns))
            if np.sum(valid_mask) < self.lookback:
                ics[name] = 0.0
                continue

            valid_alpha = lagged_alpha[valid_mask][-self.lookback:]
            valid_returns = forward_returns[valid_mask][-self.lookback:]

            if len(valid_alpha) >= 10:
                ic, _ = stats.spearmanr(valid_alpha, valid_returns)
                ics[name] = ic if not np.isnan(ic) else 0.0
            else:
                ics[name] = 0.0

        # Filter by minimum IC
        filtered_ics = {k: max(v, 0) for k, v in ics.items() if v >= self.min_ic}

        # Normalize to sum to 1
        total_ic = sum(filtered_ics.values())
        if total_ic > 0:
            return {k: v / total_ic for k, v in filtered_ics.items()}

        # Fall back to equal weight if no positive ICs
        return EqualWeighter().compute_weights(alpha_values)


class SharpeWeighter(AlphaWeighter):
    """Weight alphas by their Sharpe ratio.

    MAJOR FIX: Uses lagged alpha values with forward returns to prevent
    look-ahead bias. Alpha at time t is used to predict returns from t to t+1.
    """

    def __init__(
        self,
        lookback: int = 60,
        annualization: float = 252.0,
        return_lag: int = 1,
    ):
        """
        Initialize Sharpe weighter.

        Args:
            lookback: Lookback period for Sharpe calculation
            annualization: Annualization factor
            return_lag: Number of periods to lag returns (default 1 = forward returns)
        """
        self.lookback = lookback
        self.annualization = annualization
        self.return_lag = return_lag

    def compute_weights(
        self,
        alpha_values: dict[str, np.ndarray],
        returns: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Weight by Sharpe ratio of alpha times returns.

        MAJOR FIX: Now properly aligns alpha values with forward returns
        to prevent look-ahead bias.
        """
        if returns is None:
            return EqualWeighter().compute_weights(alpha_values)

        sharpes = {}
        for name, values in alpha_values.items():
            lagged_alpha, forward_returns = _align_alpha_with_forward_returns(
                values,
                returns,
                self.return_lag,
            )

            # Compute alpha returns (signal * forward returns)
            min_len = min(len(lagged_alpha), len(forward_returns))
            lagged_alpha = lagged_alpha[-min_len:]
            forward_returns = forward_returns[-min_len:]

            valid_mask = ~(np.isnan(lagged_alpha) | np.isnan(forward_returns))
            if np.sum(valid_mask) < self.lookback:
                sharpes[name] = 0.0
                continue

            alpha_returns = lagged_alpha[valid_mask] * forward_returns[valid_mask]
            alpha_returns = alpha_returns[-self.lookback:]

            if len(alpha_returns) >= 10:
                mean_ret = np.mean(alpha_returns)
                std_ret = np.std(alpha_returns, ddof=1)
                if std_ret > 0:
                    sharpe = mean_ret / std_ret * np.sqrt(self.annualization)
                    sharpes[name] = max(sharpe, 0)
                else:
                    sharpes[name] = 0.0
            else:
                sharpes[name] = 0.0

        # Normalize to sum to 1
        total_sharpe = sum(sharpes.values())
        if total_sharpe > 0:
            return {k: v / total_sharpe for k, v in sharpes.items()}

        return EqualWeighter().compute_weights(alpha_values)


class InverseVolatilityWeighter(AlphaWeighter):
    """Weight alphas inversely to their volatility."""

    def __init__(self, lookback: int = 20):
        """
        Initialize inverse volatility weighter.

        Args:
            lookback: Lookback period for volatility calculation
        """
        self.lookback = lookback

    def compute_weights(
        self,
        alpha_values: dict[str, np.ndarray],
        returns: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Weight inversely to alpha volatility."""
        inv_vols = {}

        for name, values in alpha_values.items():
            valid_values = values[~np.isnan(values)][-self.lookback :]

            if len(valid_values) >= 10:
                vol = np.std(valid_values, ddof=1)
                inv_vols[name] = 1.0 / vol if vol > 0 else 0.0
            else:
                inv_vols[name] = 0.0

        # Normalize to sum to 1
        total_inv_vol = sum(inv_vols.values())
        if total_inv_vol > 0:
            return {k: v / total_inv_vol for k, v in inv_vols.items()}

        return EqualWeighter().compute_weights(alpha_values)


class OptimizedWeighter(AlphaWeighter):
    """Optimize weights to maximize Sharpe ratio."""

    def __init__(
        self,
        lookback: int = 60,
        max_weight: float = 0.5,
        min_weight: float = 0.0,
        turnover_penalty: float = 0.1,
        return_lag: int = 1,
    ):
        """
        Initialize optimized weighter.

        Args:
            lookback: Lookback period for optimization
            max_weight: Maximum weight per alpha
            min_weight: Minimum weight per alpha
            turnover_penalty: Penalty for weight turnover
            return_lag: Number of periods to lag returns for alignment.
        """
        self.lookback = lookback
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.turnover_penalty = turnover_penalty
        self.return_lag = max(0, int(return_lag))
        self._previous_weights: dict[str, float] | None = None

    def compute_weights(
        self,
        alpha_values: dict[str, np.ndarray],
        returns: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Optimize weights to maximize Sharpe."""
        if returns is None:
            return EqualWeighter().compute_weights(alpha_values)

        alpha_names = list(alpha_values.keys())
        n_alphas = len(alpha_names)

        if n_alphas == 0:
            return {}

        # Build aligned alpha matrix with lag-safe return alignment.
        aligned_alpha: dict[str, np.ndarray] = {}
        aligned_returns: dict[str, np.ndarray] = {}
        for name in alpha_names:
            a, r = _align_alpha_with_forward_returns(
                alpha_values[name],
                returns,
                self.return_lag,
            )
            if a.size == 0 or r.size == 0:
                continue
            aligned_alpha[name] = a
            aligned_returns[name] = r

        if len(aligned_alpha) < 1:
            return EqualWeighter().compute_weights(alpha_values)

        min_len = min(
            min(len(v) for v in aligned_alpha.values()),
            self.lookback,
        )
        if min_len <= 0:
            return EqualWeighter().compute_weights(alpha_values)

        alpha_names = list(aligned_alpha.keys())
        n_alphas = len(alpha_names)
        alpha_matrix = np.column_stack([aligned_alpha[name][-min_len:] for name in alpha_names])
        reference_name = alpha_names[0]
        returns_slice = aligned_returns[reference_name][-min_len:]

        # Handle NaNs
        valid_mask = ~np.any(np.isnan(alpha_matrix), axis=1) & ~np.isnan(returns_slice)
        alpha_matrix = alpha_matrix[valid_mask]
        returns_slice = returns_slice[valid_mask]

        if len(alpha_matrix) < 10:
            return EqualWeighter().compute_weights(alpha_values)

        def neg_sharpe(weights: np.ndarray) -> float:
            """Negative Sharpe ratio for minimization."""
            combined_signal = alpha_matrix @ weights
            combined_returns = combined_signal * returns_slice

            mean_ret = np.mean(combined_returns)
            std_ret = np.std(combined_returns, ddof=1)

            if std_ret < 1e-8:
                return 0.0

            sharpe = mean_ret / std_ret

            # Add turnover penalty
            if self._previous_weights is not None:
                prev_w = np.array([self._previous_weights.get(name, 1.0 / n_alphas)
                                   for name in alpha_names])
                turnover = np.sum(np.abs(weights - prev_w))
                sharpe -= self.turnover_penalty * turnover

            return -sharpe

        # Initial guess
        x0 = np.ones(n_alphas) / n_alphas

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # Sum to 1
        ]

        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_alphas)]

        # Optimize
        result = optimize.minimize(
            neg_sharpe,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100},
        )

        if result.success:
            weights = result.x
        else:
            weights = x0

        # Store for turnover calculation
        weight_dict = {name: w for name, w in zip(alpha_names, weights)}
        self._previous_weights = weight_dict

        return weight_dict


class RankWeighter(AlphaWeighter):
    """Weight alphas by their IC rank.

    MAJOR FIX: Uses lagged alpha values with forward returns to prevent
    look-ahead bias.
    """

    def __init__(self, lookback: int = 20, return_lag: int = 1):
        """
        Initialize rank weighter.

        Args:
            lookback: Lookback period for IC calculation
            return_lag: Number of periods to lag returns (default 1 = forward returns)
        """
        self.lookback = lookback
        self.return_lag = return_lag

    def compute_weights(
        self,
        alpha_values: dict[str, np.ndarray],
        returns: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Weight by IC rank (higher IC = more weight).

        MAJOR FIX: Now properly aligns alpha values with forward returns
        to prevent look-ahead bias.
        """
        if returns is None:
            return EqualWeighter().compute_weights(alpha_values)

        # Compute ICs
        ics = {}
        for name, values in alpha_values.items():
            lagged_alpha, forward_returns = _align_alpha_with_forward_returns(
                values,
                returns,
                self.return_lag,
            )

            valid_mask = ~(np.isnan(lagged_alpha) | np.isnan(forward_returns))
            if np.sum(valid_mask) < self.lookback:
                ics[name] = 0.0
                continue

            valid_alpha = lagged_alpha[valid_mask][-self.lookback:]
            valid_returns = forward_returns[valid_mask][-self.lookback:]

            if len(valid_alpha) >= 10:
                ic, _ = stats.spearmanr(valid_alpha, valid_returns)
                ics[name] = ic if not np.isnan(ic) else 0.0
            else:
                ics[name] = 0.0

        # Rank alphas
        sorted_names = sorted(ics.keys(), key=lambda x: ics[x], reverse=True)
        n_alphas = len(sorted_names)

        # Assign weights by rank (higher rank = more weight)
        rank_weights = {}
        for rank, name in enumerate(sorted_names):
            # Linear rank weighting: top alpha gets n, bottom gets 1
            rank_weights[name] = n_alphas - rank

        # Normalize to sum to 1
        total = sum(rank_weights.values())
        if total > 0:
            return {k: v / total for k, v in rank_weights.items()}

        return EqualWeighter().compute_weights(alpha_values)


class DecayWeighter(AlphaWeighter):
    """Weight alphas with exponential decay based on recency of high IC.

    MAJOR FIX: Uses lagged alpha values with forward returns to prevent
    look-ahead bias.
    """

    def __init__(self, lookback: int = 60, halflife: int = 20, return_lag: int = 1):
        """
        Initialize decay weighter.

        Args:
            lookback: Lookback period
            halflife: Decay half-life in periods
            return_lag: Number of periods to lag returns (default 1 = forward returns)
        """
        self.lookback = lookback
        self.halflife = halflife
        self.return_lag = return_lag

    def compute_weights(
        self,
        alpha_values: dict[str, np.ndarray],
        returns: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Weight with exponential decay.

        MAJOR FIX: Now properly aligns alpha values with forward returns
        to prevent look-ahead bias.
        """
        if returns is None:
            return EqualWeighter().compute_weights(alpha_values)

        decay_factor = np.exp(-np.log(2) / self.halflife)
        weighted_ics = {}

        for name, values in alpha_values.items():
            lagged_alpha, forward_returns = _align_alpha_with_forward_returns(
                values,
                returns,
                self.return_lag,
            )

            valid_mask = ~(np.isnan(lagged_alpha) | np.isnan(forward_returns))
            n_valid = np.sum(valid_mask)

            if n_valid < 10:
                weighted_ics[name] = 0.0
                continue

            # Compute rolling ICs with decay weights
            window = min(self.lookback, n_valid)
            valid_alpha = lagged_alpha[valid_mask][-window:]
            valid_returns = forward_returns[valid_mask][-window:]

            # Compute IC for each sub-window
            sub_window = 10
            total_weighted_ic = 0.0
            total_weight = 0.0

            for i in range(0, len(valid_alpha) - sub_window + 1, 5):
                sub_alpha = valid_alpha[i : i + sub_window]
                sub_returns = valid_returns[i : i + sub_window]

                ic, _ = stats.spearmanr(sub_alpha, sub_returns)
                if np.isnan(ic):
                    ic = 0.0

                # Decay weight (more recent = higher weight)
                recency = (i + sub_window) / len(valid_alpha)
                weight = decay_factor ** ((1 - recency) * self.lookback)

                total_weighted_ic += max(ic, 0) * weight
                total_weight += weight

            weighted_ics[name] = total_weighted_ic / total_weight if total_weight > 0 else 0.0

        # Normalize to sum to 1
        total = sum(weighted_ics.values())
        if total > 0:
            return {k: v / total for k, v in weighted_ics.items()}

        return EqualWeighter().compute_weights(alpha_values)


class AlphaOrthogonalizer:
    """Orthogonalize alphas to remove redundancy."""

    def __init__(self, method: OrthogonalizationMethod = OrthogonalizationMethod.PCA):
        """
        Initialize orthogonalizer.

        Args:
            method: Orthogonalization method to use
        """
        self.method = method
        self._components: np.ndarray | None = None
        self._mean: np.ndarray | None = None

    def fit(self, alpha_matrix: np.ndarray) -> "AlphaOrthogonalizer":
        """
        Fit orthogonalizer on alpha matrix.

        Args:
            alpha_matrix: Matrix of alpha values (n_samples, n_alphas)

        Returns:
            Self
        """
        # Remove NaN rows
        valid_mask = ~np.any(np.isnan(alpha_matrix), axis=1)
        valid_matrix = alpha_matrix[valid_mask]

        if len(valid_matrix) < 10:
            return self

        self._mean = np.mean(valid_matrix, axis=0)
        centered = valid_matrix - self._mean

        if self.method == OrthogonalizationMethod.PCA:
            # PCA decomposition
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            self._components = vh.T

        elif self.method == OrthogonalizationMethod.GRAM_SCHMIDT:
            # Gram-Schmidt orthogonalization
            n_alphas = centered.shape[1]
            orthogonal = np.zeros_like(centered)

            for i in range(n_alphas):
                orthogonal[:, i] = centered[:, i]
                for j in range(i):
                    proj = (
                        np.dot(centered[:, i], orthogonal[:, j])
                        / np.dot(orthogonal[:, j], orthogonal[:, j])
                    ) * orthogonal[:, j]
                    orthogonal[:, i] -= proj

            # Normalize
            norms = np.linalg.norm(orthogonal, axis=0)
            norms[norms == 0] = 1.0
            self._components = orthogonal / norms

        elif self.method == OrthogonalizationMethod.DECORRELATE:
            # Decorrelation via correlation matrix eigendecomposition
            corr = np.corrcoef(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(corr)
            # Whitening transformation
            whitening = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-8))
            self._components = whitening

        return self

    def transform(self, alpha_matrix: np.ndarray) -> np.ndarray:
        """
        Transform alphas to orthogonal space.

        Args:
            alpha_matrix: Matrix of alpha values

        Returns:
            Orthogonalized alpha matrix
        """
        if self._components is None or self._mean is None:
            return alpha_matrix

        if self.method == OrthogonalizationMethod.NONE:
            return alpha_matrix

        # Handle NaNs
        result = np.full_like(alpha_matrix, np.nan)
        valid_mask = ~np.any(np.isnan(alpha_matrix), axis=1)

        centered = alpha_matrix[valid_mask] - self._mean
        result[valid_mask] = centered @ self._components

        return result

    def fit_transform(self, alpha_matrix: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(alpha_matrix)
        return self.transform(alpha_matrix)


class AlphaNeutralizer:
    """Neutralize alphas against factors."""

    def __init__(
        self,
        method: NeutralizationMethod = NeutralizationMethod.MARKET,
    ):
        """
        Initialize neutralizer.

        Args:
            method: Neutralization method to use
        """
        self.method = method
        self._factor_betas: dict[str, np.ndarray] | None = None

    def neutralize(
        self,
        alpha_values: np.ndarray,
        market_returns: np.ndarray | None = None,
        sector_labels: np.ndarray | None = None,
        factor_exposures: np.ndarray | None = None,
        betas: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Neutralize alpha values.

        Args:
            alpha_values: Alpha signal values
            market_returns: Market returns for market neutralization
            sector_labels: Sector labels for sector neutralization
            factor_exposures: Factor exposure matrix for factor neutralization
            betas: Asset betas for beta neutralization

        Returns:
            Neutralized alpha values
        """
        if self.method == NeutralizationMethod.NONE:
            return alpha_values

        if self.method == NeutralizationMethod.MARKET:
            return self._market_neutralize(alpha_values)

        elif self.method == NeutralizationMethod.SECTOR:
            if sector_labels is None:
                return self._market_neutralize(alpha_values)
            return self._sector_neutralize(alpha_values, sector_labels)

        elif self.method == NeutralizationMethod.FACTOR:
            if factor_exposures is None:
                return self._market_neutralize(alpha_values)
            return self._factor_neutralize(alpha_values, factor_exposures)

        elif self.method == NeutralizationMethod.BETA:
            if betas is None:
                return self._market_neutralize(alpha_values)
            return self._beta_neutralize(alpha_values, betas)

        return alpha_values

    def _market_neutralize(self, alpha_values: np.ndarray) -> np.ndarray:
        """Market neutralize by demeaning."""
        valid_mask = ~np.isnan(alpha_values)
        result = alpha_values.copy()

        if np.sum(valid_mask) > 0:
            result[valid_mask] -= np.mean(alpha_values[valid_mask])

        return result

    def _sector_neutralize(
        self,
        alpha_values: np.ndarray,
        sector_labels: np.ndarray,
    ) -> np.ndarray:
        """Sector neutralize by demeaning within each sector."""
        result = alpha_values.copy()
        unique_sectors = np.unique(sector_labels[~np.isnan(alpha_values)])

        for sector in unique_sectors:
            sector_mask = (sector_labels == sector) & ~np.isnan(alpha_values)
            if np.sum(sector_mask) > 0:
                result[sector_mask] -= np.mean(alpha_values[sector_mask])

        return result

    def _factor_neutralize(
        self,
        alpha_values: np.ndarray,
        factor_exposures: np.ndarray,
    ) -> np.ndarray:
        """Factor neutralize by regressing out factor exposures."""
        valid_mask = ~np.isnan(alpha_values)

        if np.sum(valid_mask) < factor_exposures.shape[1] + 1:
            return self._market_neutralize(alpha_values)

        # Regress alpha on factors
        valid_alpha = alpha_values[valid_mask]
        valid_factors = factor_exposures[valid_mask]

        # Add constant for intercept
        X = np.column_stack([np.ones(len(valid_factors)), valid_factors])

        # OLS regression
        betas, _, _, _ = np.linalg.lstsq(X, valid_alpha, rcond=None)

        # Compute residuals (neutralized alpha)
        result = alpha_values.copy()
        result[valid_mask] = valid_alpha - X @ betas

        return result

    def _beta_neutralize(
        self,
        alpha_values: np.ndarray,
        betas: np.ndarray,
    ) -> np.ndarray:
        """Beta neutralize by adjusting for market beta."""
        valid_mask = ~(np.isnan(alpha_values) | np.isnan(betas))

        if np.sum(valid_mask) < 10:
            return self._market_neutralize(alpha_values)

        result = alpha_values.copy()

        # Compute beta-weighted mean
        valid_alpha = alpha_values[valid_mask]
        valid_betas = betas[valid_mask]

        beta_weighted_mean = np.sum(valid_alpha * valid_betas) / np.sum(valid_betas)
        result[valid_mask] = valid_alpha - valid_betas * beta_weighted_mean

        return result


class AlphaCombiner:
    """
    Combine multiple alpha factors into a single composite signal.

    Supports various weighting schemes, orthogonalization, and neutralization.
    """

    def __init__(
        self,
        alphas: list[AlphaFactor] | None = None,
        config: CombinerConfig | None = None,
    ):
        """
        Initialize alpha combiner.

        Args:
            alphas: List of alpha factors to combine
            config: Configuration for combination
        """
        self.alphas = alphas or []
        self.config = config or CombinerConfig()

        # Initialize components
        self._weighter = self._create_weighter()
        self._orthogonalizer = AlphaOrthogonalizer(
            self.config.orthogonalization_method
        )
        self._neutralizer = AlphaNeutralizer(
            self.config.neutralization_method
        )

        # State
        self._weights: dict[str, float] = {}
        self._alpha_values: dict[str, np.ndarray] = {}
        self._oos_alpha_history: dict[str, np.ndarray] = {}
        self._oos_returns_history: np.ndarray = np.array([], dtype=float)
        self._weight_history: list[dict[str, float]] = []
        self._is_fitted = False

    def _create_weighter(self) -> AlphaWeighter:
        """Create weighter based on config."""
        method = self.config.weighting_method

        if method == WeightingMethod.EQUAL:
            return EqualWeighter()
        elif method == WeightingMethod.IC_WEIGHTED:
            return ICWeighter(self.config.ic_lookback, return_lag=self.config.return_lag)
        elif method == WeightingMethod.SHARPE_WEIGHTED:
            return SharpeWeighter(self.config.lookback_window, return_lag=self.config.return_lag)
        elif method == WeightingMethod.INVERSE_VOLATILITY:
            return InverseVolatilityWeighter(self.config.vol_lookback)
        elif method == WeightingMethod.OPTIMIZED:
            return OptimizedWeighter(
                self.config.lookback_window,
                self.config.max_weight,
                self.config.min_weight,
                self.config.turnover_penalty,
                self.config.return_lag,
            )
        elif method == WeightingMethod.RANK_WEIGHTED:
            return RankWeighter(self.config.ic_lookback, return_lag=self.config.return_lag)
        elif method == WeightingMethod.DECAY_WEIGHTED:
            return DecayWeighter(
                self.config.lookback_window,
                self.config.decay_halflife,
                self.config.return_lag,
            )
        else:
            return EqualWeighter()

    def add_alpha(self, alpha: AlphaFactor) -> "AlphaCombiner":
        """
        Add an alpha factor to the combiner.

        Args:
            alpha: Alpha factor to add

        Returns:
            Self for method chaining
        """
        self.alphas.append(alpha)
        self._is_fitted = False
        return self

    def remove_alpha(self, name: str) -> "AlphaCombiner":
        """
        Remove an alpha factor by name.

        Args:
            name: Name of alpha to remove

        Returns:
            Self for method chaining
        """
        self.alphas = [a for a in self.alphas if a.name != name]
        if name in self._weights:
            del self._weights[name]
        if name in self._alpha_values:
            del self._alpha_values[name]
        if name in self._oos_alpha_history:
            del self._oos_alpha_history[name]
        return self

    def fit(
        self,
        df: pl.DataFrame,
        returns: np.ndarray | None = None,
        features: dict[str, np.ndarray] | None = None,
    ) -> "AlphaCombiner":
        """
        Fit the combiner on historical data.

        Args:
            df: DataFrame with OHLCV data
            returns: Forward returns for weight optimization
            features: Optional precomputed features

        Returns:
            Self
        """
        if not self.alphas:
            self._is_fitted = True
            return self

        # Compute alpha values
        self._alpha_values = {}
        for alpha in self.alphas:
            values = alpha.compute(df, features)
            self._alpha_values[alpha.name] = values

        # Compute weights
        self._weights = self._weighter.compute_weights(
            self._alpha_values,
            returns,
        )

        # Apply constraints
        self._weights = self._apply_weight_constraints(self._weights)

        # Fit orthogonalizer if needed
        if self.config.orthogonalization_method != OrthogonalizationMethod.NONE:
            alpha_matrix = np.column_stack(list(self._alpha_values.values()))
            self._orthogonalizer.fit(alpha_matrix)

        self._oos_alpha_history = {}
        self._oos_returns_history = np.array([], dtype=float)
        self._weight_history = [self._weights.copy()]
        self._is_fitted = True
        return self

    def _append_oos_history(
        self,
        alpha_values: dict[str, np.ndarray],
        returns: np.ndarray,
        update_mode: WeightUpdateMode,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Append incoming OOS observations to update history."""
        if not alpha_values:
            return {}, np.array([], dtype=float)

        min_len = min(min(len(v) for v in alpha_values.values()), len(returns))
        if min_len <= 0:
            return self._oos_alpha_history, self._oos_returns_history

        batch_alpha = {name: np.asarray(values[-min_len:], dtype=float) for name, values in alpha_values.items()}
        batch_returns = np.asarray(returns[-min_len:], dtype=float)

        if not self._oos_alpha_history:
            self._oos_alpha_history = {name: values.copy() for name, values in batch_alpha.items()}
            self._oos_returns_history = batch_returns.copy()
        else:
            for name, values in batch_alpha.items():
                previous = self._oos_alpha_history.get(name, np.array([], dtype=float))
                self._oos_alpha_history[name] = np.concatenate([previous, values]).astype(float)
            self._oos_returns_history = np.concatenate([self._oos_returns_history, batch_returns]).astype(float)

        if update_mode == WeightUpdateMode.ROLLING:
            window = int(self.config.oos_update_window)
            self._oos_returns_history = self._oos_returns_history[-window:]
            for name in list(self._oos_alpha_history.keys()):
                self._oos_alpha_history[name] = self._oos_alpha_history[name][-window:]

        return self._oos_alpha_history, self._oos_returns_history

    def update_weights_oos(
        self,
        df: pl.DataFrame,
        returns: np.ndarray,
        features: dict[str, np.ndarray] | None = None,
        update_mode: WeightUpdateMode | None = None,
    ) -> dict[str, float]:
        """
        Update weights from out-of-sample observations.

        Supports expanding or rolling windows and blends updated weights with
        previous production weights to reduce regime-switch instability.
        """
        if not self.alphas:
            return {}

        mode = update_mode or self.config.oos_update_mode
        if isinstance(mode, str):
            mode = WeightUpdateMode(mode)

        alpha_values: dict[str, np.ndarray] = {}
        for alpha in self.alphas:
            alpha_values[alpha.name] = alpha.compute(df, features)

        history_alpha, history_returns = self._append_oos_history(
            alpha_values=alpha_values,
            returns=np.asarray(returns, dtype=float),
            update_mode=mode,
        )
        if len(history_returns) < int(self.config.oos_min_observations):
            return self._weights.copy()

        updated_weights = self._weighter.compute_weights(history_alpha, history_returns)
        updated_weights = self._apply_weight_constraints(updated_weights)

        if self._weights:
            blend = float(np.clip(self.config.oos_update_blend, 0.0, 1.0))
            all_names = {alpha.name for alpha in self.alphas}
            blended = {}
            for name in all_names:
                prev_w = float(self._weights.get(name, 0.0))
                new_w = float(updated_weights.get(name, 0.0))
                blended[name] = blend * prev_w + (1.0 - blend) * new_w
            updated_weights = self._apply_weight_constraints(blended)

        self._weights = updated_weights
        self._alpha_values = alpha_values
        self._weight_history.append(self._weights.copy())
        self._is_fitted = True
        return self._weights.copy()

    def _apply_weight_constraints(
        self,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Apply weight constraints."""
        # Apply min/max bounds
        constrained = {
            k: np.clip(v, self.config.min_weight, self.config.max_weight)
            for k, v in weights.items()
        }

        # Normalize to sum to 1 if required
        if self.config.sum_to_one:
            total = sum(constrained.values())
            if total > 0:
                constrained = {k: v / total for k, v in constrained.items()}

        return constrained

    def combine(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
        market_returns: np.ndarray | None = None,
        sector_labels: np.ndarray | None = None,
        factor_exposures: np.ndarray | None = None,
        betas: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Combine alphas into a single signal.

        Args:
            df: DataFrame with OHLCV data
            features: Optional precomputed features
            market_returns: Market returns for neutralization
            sector_labels: Sector labels for neutralization
            factor_exposures: Factor exposures for neutralization
            betas: Asset betas for neutralization

        Returns:
            Combined alpha signal array
        """
        if not self.alphas:
            return np.array([])

        # Compute current alpha values
        alpha_values = {}
        for alpha in self.alphas:
            values = alpha.compute(df, features)
            alpha_values[alpha.name] = values

        # Use fitted weights or compute new ones
        if self._is_fitted and self._weights:
            weights = self._weights
        else:
            weights = self._weighter.compute_weights(alpha_values)
            weights = self._apply_weight_constraints(weights)

        # Build alpha matrix
        alpha_names = list(alpha_values.keys())
        n = len(next(iter(alpha_values.values())))
        alpha_matrix = np.column_stack([alpha_values[name] for name in alpha_names])

        # Apply orthogonalization
        if self.config.orthogonalization_method != OrthogonalizationMethod.NONE:
            alpha_matrix = self._orthogonalizer.transform(alpha_matrix)

        # Weighted combination
        weight_array = np.array([weights.get(name, 0) for name in alpha_names])
        combined = np.nansum(alpha_matrix * weight_array, axis=1)

        # Apply neutralization
        combined = self._neutralizer.neutralize(
            combined,
            market_returns,
            sector_labels,
            factor_exposures,
            betas,
        )

        # Clip to [-1, 1]
        combined = np.clip(combined, -1, 1)

        return combined

    def get_weights(self) -> dict[str, float]:
        """Get current alpha weights."""
        return self._weights.copy()

    def get_weight_history(self) -> list[dict[str, float]]:
        """Get chronological history of fitted and OOS-updated weights."""
        return [snapshot.copy() for snapshot in self._weight_history]

    def get_oos_observation_count(self) -> int:
        """Get number of OOS observations currently retained for updates."""
        return int(len(self._oos_returns_history))

    def get_weight_info(self) -> list[AlphaWeight]:
        """Get detailed weight information for each alpha."""
        info = []
        sorted_weights = sorted(
            self._weights.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for rank, (name, weight) in enumerate(sorted_weights, 1):
            info.append(
                AlphaWeight(
                    alpha_name=name,
                    weight=weight,
                    rank=rank,
                )
            )

        return info

    def get_correlation_matrix(self) -> np.ndarray | None:
        """Get correlation matrix of alpha values."""
        if not self._alpha_values:
            return None

        alpha_matrix = np.column_stack(list(self._alpha_values.values()))
        valid_mask = ~np.any(np.isnan(alpha_matrix), axis=1)

        if np.sum(valid_mask) < 10:
            return None

        return np.corrcoef(alpha_matrix[valid_mask].T)

    def get_alpha_stats(
        self,
        returns: np.ndarray | None = None,
        return_lag: int | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Get statistics for each alpha.

        Args:
            returns: Optional forward returns for IC calculation

        Returns:
            Dictionary of alpha name to stats dictionary
        """
        stats_dict = {}

        for name, values in self._alpha_values.items():
            valid_values = values[~np.isnan(values)]

            alpha_stats = {
                "mean": float(np.mean(valid_values)) if len(valid_values) > 0 else 0.0,
                "std": float(np.std(valid_values, ddof=1)) if len(valid_values) > 1 else 0.0,
                "skew": float(stats.skew(valid_values)) if len(valid_values) > 2 else 0.0,
                "kurtosis": float(stats.kurtosis(valid_values)) if len(valid_values) > 3 else 0.0,
                "weight": self._weights.get(name, 0.0),
            }

            # Add lag-safe IC if returns provided
            if returns is not None:
                lag = self.config.return_lag if return_lag is None else max(0, int(return_lag))
                aligned_alpha, aligned_returns = _align_alpha_with_forward_returns(
                    values,
                    returns,
                    lag,
                )
                valid_mask = ~(np.isnan(aligned_alpha) | np.isnan(aligned_returns))
                if np.sum(valid_mask) >= 10:
                    ic, p_value = stats.spearmanr(
                        aligned_alpha[valid_mask],
                        aligned_returns[valid_mask],
                    )
                    alpha_stats["ic"] = float(ic) if not np.isnan(ic) else 0.0
                    alpha_stats["ic_pvalue"] = float(p_value) if not np.isnan(p_value) else 1.0
                else:
                    alpha_stats["ic"] = 0.0
                    alpha_stats["ic_pvalue"] = 1.0

            stats_dict[name] = alpha_stats

        return stats_dict


class CompositeAlphaFactor(AlphaFactor):
    """
    An AlphaFactor that combines multiple child alphas.

    This wraps AlphaCombiner as an AlphaFactor for use in nested combinations.
    """

    def __init__(
        self,
        name: str,
        alphas: list[AlphaFactor],
        config: CombinerConfig | None = None,
        horizon: AlphaHorizon = AlphaHorizon.MEDIUM,
    ):
        """
        Initialize composite alpha.

        Args:
            name: Name for the composite
            alphas: List of child alpha factors
            config: Combiner configuration
            horizon: Forecast horizon
        """
        super().__init__(
            name=name,
            alpha_type=AlphaType.COMPOSITE,
            horizon=horizon,
            lookback=max((a.lookback for a in alphas), default=20),
        )
        self._combiner = AlphaCombiner(alphas, config)

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute combined alpha values."""
        return self._combiner.combine(df, features)

    def get_params(self) -> dict[str, Any]:
        """Get parameters."""
        return {
            "name": self.name,
            "n_alphas": len(self._combiner.alphas),
            "config": self._combiner.config.to_dict(),
            "weights": self._combiner.get_weights(),
        }

    def fit(
        self,
        df: pl.DataFrame,
        target: np.ndarray | None = None,
    ) -> "CompositeAlphaFactor":
        """Fit the combiner."""
        self._combiner.fit(df, target)
        self._is_fitted = True
        return self


def create_combiner(
    alphas: list[AlphaFactor],
    method: WeightingMethod = WeightingMethod.IC_WEIGHTED,
    neutralize: NeutralizationMethod = NeutralizationMethod.MARKET,
    orthogonalize: OrthogonalizationMethod = OrthogonalizationMethod.NONE,
    max_weight: float = 0.5,
    return_lag: int = 1,
    oos_update_mode: WeightUpdateMode = WeightUpdateMode.EXPANDING,
    oos_update_window: int = 252,
    oos_update_blend: float = 0.5,
    oos_min_observations: int = 30,
) -> AlphaCombiner:
    """
    Factory function to create a configured AlphaCombiner.

    Args:
        alphas: List of alpha factors to combine
        method: Weighting method to use
        neutralize: Neutralization method
        orthogonalize: Orthogonalization method
        max_weight: Maximum weight per alpha
        return_lag: Return lag used for alpha/return alignment
        oos_update_mode: OOS weight update mode (expanding/rolling)
        oos_update_window: OOS rolling window size when mode=rolling
        oos_update_blend: Blend ratio between previous and new OOS weights
        oos_min_observations: Minimum OOS rows required for update

    Returns:
        Configured AlphaCombiner instance
    """
    config = CombinerConfig(
        weighting_method=method,
        neutralization_method=neutralize,
        orthogonalization_method=orthogonalize,
        max_weight=max_weight,
        return_lag=return_lag,
        oos_update_mode=oos_update_mode,
        oos_update_window=oos_update_window,
        oos_update_blend=oos_update_blend,
        oos_min_observations=oos_min_observations,
    )

    return AlphaCombiner(alphas, config)
