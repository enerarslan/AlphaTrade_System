"""
Portfolio optimization module.

Implements multiple portfolio optimization methods:
- Mean-Variance Optimization (Markowitz)
- Risk Parity
- Black-Litterman
- Hierarchical Risk Parity
- Robust Optimization

Includes rebalancing logic and transaction cost consideration.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np
from scipy import optimize
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from quant_trading_system.core.data_types import Portfolio, Position
from quant_trading_system.core.exceptions import RiskError

logger = logging.getLogger(__name__)


class OptimizationMethod(str, Enum):
    """Portfolio optimization method enumeration."""

    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    HIERARCHICAL_RISK_PARITY = "hrp"
    ROBUST = "robust"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"


class RebalanceTrigger(str, Enum):
    """Rebalancing trigger types."""

    TIME_BASED = "time_based"
    THRESHOLD_BASED = "threshold_based"
    SIGNAL_BASED = "signal_based"
    HYBRID = "hybrid"


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""

    target_weights: dict[str, float]
    method: OptimizationMethod
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    expected_sharpe: float = 0.0
    optimization_success: bool = True
    constraints_active: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_weights": self.target_weights,
            "method": self.method.value,
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "expected_sharpe": self.expected_sharpe,
            "optimization_success": self.optimization_success,
            "constraints_active": self.constraints_active,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RebalanceOrder:
    """Order generated from rebalancing."""

    symbol: str
    current_weight: float
    target_weight: float
    weight_change: float
    shares_to_trade: Decimal
    estimated_cost: Decimal = Decimal("0")
    is_buy: bool = True

    @property
    def turnover(self) -> float:
        """Get absolute turnover for this order."""
        return abs(self.weight_change)


@dataclass
class RebalanceResult:
    """Result of rebalancing calculation."""

    orders: list[RebalanceOrder]
    total_turnover: float
    estimated_total_cost: Decimal
    should_rebalance: bool
    trigger_reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class OptimizationConstraints:
    """Constraints for portfolio optimization."""

    def __init__(
        self,
        long_only: bool = True,
        max_weight: float = 0.20,
        min_weight: float = 0.0,
        max_turnover: float | None = None,
        sector_limits: dict[str, float] | None = None,
        target_return: float | None = None,
        target_risk: float | None = None,
    ) -> None:
        """Initialize optimization constraints.

        Args:
            long_only: Only allow long positions.
            max_weight: Maximum weight per asset.
            min_weight: Minimum weight per asset.
            max_turnover: Maximum allowed turnover.
            sector_limits: Maximum exposure per sector.
            target_return: Target portfolio return.
            target_risk: Target portfolio risk.
        """
        self.long_only = long_only
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.max_turnover = max_turnover
        self.sector_limits = sector_limits or {}
        self.target_return = target_return
        self.target_risk = target_risk


class BasePortfolioOptimizer(ABC):
    """Base class for portfolio optimizers."""

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        constraints: OptimizationConstraints | None = None,
    ) -> None:
        """Initialize optimizer.

        Args:
            risk_free_rate: Annual risk-free rate.
            constraints: Optimization constraints.
        """
        self.risk_free_rate = risk_free_rate
        self.constraints = constraints or OptimizationConstraints()

    @abstractmethod
    def optimize(
        self,
        symbols: list[str],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: dict[str, float] | None = None,
    ) -> OptimizationResult:
        """Optimize portfolio weights.

        Args:
            symbols: List of asset symbols.
            expected_returns: Expected returns array.
            covariance_matrix: Covariance matrix.
            current_weights: Current portfolio weights.

        Returns:
            Optimization result with target weights.
        """
        pass

    def _validate_inputs(
        self,
        symbols: list[str],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> None:
        """Validate optimization inputs."""
        n = len(symbols)
        if expected_returns.shape[0] != n:
            raise RiskError(
                "Expected returns dimension mismatch",
                details={"symbols": n, "returns": expected_returns.shape[0]},
            )
        if covariance_matrix.shape != (n, n):
            raise RiskError(
                "Covariance matrix dimension mismatch",
                details={"symbols": n, "cov_shape": covariance_matrix.shape},
            )


class MeanVarianceOptimizer(BasePortfolioOptimizer):
    """Mean-Variance (Markowitz) portfolio optimization."""

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        constraints: OptimizationConstraints | None = None,
        objective: str = "max_sharpe",
        regularization: float = 0.0,
    ) -> None:
        """Initialize mean-variance optimizer.

        Args:
            risk_free_rate: Annual risk-free rate.
            constraints: Optimization constraints.
            objective: Optimization objective ('max_sharpe', 'min_variance', 'target_return').
            regularization: L2 regularization strength for weights.
        """
        super().__init__(risk_free_rate, constraints)
        self.objective = objective
        self.regularization = regularization

    def optimize(
        self,
        symbols: list[str],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: dict[str, float] | None = None,
    ) -> OptimizationResult:
        """Optimize using mean-variance framework."""
        self._validate_inputs(symbols, expected_returns, covariance_matrix)
        n = len(symbols)

        # Initial weights
        x0 = np.ones(n) / n

        # Bounds
        if self.constraints.long_only:
            bounds = [(self.constraints.min_weight, self.constraints.max_weight) for _ in range(n)]
        else:
            bounds = [(-self.constraints.max_weight, self.constraints.max_weight) for _ in range(n)]

        # Constraints
        constraints_list = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}  # Weights sum to 1
        ]

        # Add turnover constraint if specified
        if self.constraints.max_turnover is not None and current_weights is not None:
            current_w = np.array([current_weights.get(s, 0.0) for s in symbols])
            constraints_list.append({
                "type": "ineq",
                "fun": lambda x: self.constraints.max_turnover - np.sum(np.abs(x - current_w)),
            })

        # Define objective function
        if self.objective == "max_sharpe":
            def objective(w: np.ndarray) -> float:
                ret = np.dot(w, expected_returns)
                vol = np.sqrt(np.dot(w.T, np.dot(covariance_matrix, w)))
                sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0
                reg = self.regularization * np.sum(w ** 2)
                return -(sharpe - reg)  # Negative for minimization
        elif self.objective == "min_variance":
            def objective(w: np.ndarray) -> float:
                vol = np.dot(w.T, np.dot(covariance_matrix, w))
                reg = self.regularization * np.sum(w ** 2)
                return vol + reg
        else:  # target_return
            target = self.constraints.target_return or np.mean(expected_returns)
            constraints_list.append({
                "type": "eq",
                "fun": lambda x: np.dot(x, expected_returns) - target,
            })
            def objective(w: np.ndarray) -> float:
                vol = np.dot(w.T, np.dot(covariance_matrix, w))
                reg = self.regularization * np.sum(w ** 2)
                return vol + reg

        # Run optimization
        result = optimize.minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_list,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        weights = result.x
        target_weights = {s: float(w) for s, w in zip(symbols, weights)}

        # Calculate expected metrics
        exp_ret = float(np.dot(weights, expected_returns))
        exp_vol = float(np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights))))
        exp_sharpe = (exp_ret - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0

        return OptimizationResult(
            target_weights=target_weights,
            method=OptimizationMethod.MEAN_VARIANCE,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            expected_sharpe=exp_sharpe,
            optimization_success=result.success,
            metadata={
                "objective": self.objective,
                "iterations": result.nit,
                "regularization": self.regularization,
            },
        )


class RiskParityOptimizer(BasePortfolioOptimizer):
    """Risk Parity portfolio optimization.

    Each position contributes equally to total portfolio risk.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        constraints: OptimizationConstraints | None = None,
        risk_budget: dict[str, float] | None = None,
    ) -> None:
        """Initialize risk parity optimizer.

        Args:
            risk_free_rate: Annual risk-free rate.
            constraints: Optimization constraints.
            risk_budget: Custom risk budget per asset (default: equal).
        """
        super().__init__(risk_free_rate, constraints)
        self.risk_budget = risk_budget

    def optimize(
        self,
        symbols: list[str],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: dict[str, float] | None = None,
    ) -> OptimizationResult:
        """Optimize for risk parity."""
        self._validate_inputs(symbols, expected_returns, covariance_matrix)
        n = len(symbols)

        # Risk budget (default: equal)
        if self.risk_budget:
            budget = np.array([self.risk_budget.get(s, 1.0 / n) for s in symbols])
        else:
            budget = np.ones(n) / n
        budget = budget / np.sum(budget)  # Normalize

        # Initial weights
        x0 = np.ones(n) / n

        # Objective: minimize squared difference between actual and target risk contributions
        def objective(w: np.ndarray) -> float:
            w = np.maximum(w, 1e-10)  # Avoid division by zero
            sigma = np.sqrt(np.dot(w.T, np.dot(covariance_matrix, w)))
            mrc = np.dot(covariance_matrix, w) / sigma  # Marginal risk contribution
            rc = w * mrc  # Risk contribution
            rc_pct = rc / sigma  # Risk contribution percentage
            # Squared error from target budget
            return np.sum((rc_pct - budget) ** 2)

        # Constraints
        bounds = [(1e-6, self.constraints.max_weight) for _ in range(n)]
        constraints_list = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

        result = optimize.minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_list,
            options={"maxiter": 1000},
        )

        weights = result.x
        target_weights = {s: float(w) for s, w in zip(symbols, weights)}

        # Calculate expected metrics
        exp_ret = float(np.dot(weights, expected_returns))
        exp_vol = float(np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights))))
        exp_sharpe = (exp_ret - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0

        return OptimizationResult(
            target_weights=target_weights,
            method=OptimizationMethod.RISK_PARITY,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            expected_sharpe=exp_sharpe,
            optimization_success=result.success,
            metadata={"risk_budget": dict(zip(symbols, budget.tolist()))},
        )


class BlackLittermanOptimizer(BasePortfolioOptimizer):
    """Black-Litterman portfolio optimization.

    Combines market equilibrium with investor views.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        constraints: OptimizationConstraints | None = None,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
    ) -> None:
        """Initialize Black-Litterman optimizer.

        Args:
            risk_free_rate: Annual risk-free rate.
            constraints: Optimization constraints.
            tau: Uncertainty in prior (typically 0.01-0.05).
            risk_aversion: Market risk aversion coefficient.
        """
        super().__init__(risk_free_rate, constraints)
        self.tau = tau
        self.risk_aversion = risk_aversion

    def optimize(
        self,
        symbols: list[str],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: dict[str, float] | None = None,
        market_weights: np.ndarray | None = None,
        views: list[tuple[np.ndarray, float, float]] | None = None,
    ) -> OptimizationResult:
        """Optimize using Black-Litterman model.

        Args:
            symbols: List of asset symbols.
            expected_returns: Expected returns (used as views).
            covariance_matrix: Covariance matrix.
            current_weights: Current portfolio weights.
            market_weights: Market capitalization weights (prior).
            views: List of (P, Q, confidence) tuples for investor views.

        Returns:
            Optimization result.
        """
        self._validate_inputs(symbols, expected_returns, covariance_matrix)
        n = len(symbols)

        # Market weights (default: equal)
        if market_weights is None:
            market_weights = np.ones(n) / n

        # Implied equilibrium returns (from CAPM)
        pi = self.risk_aversion * np.dot(covariance_matrix, market_weights)

        # If views are provided, incorporate them
        if views:
            P = np.vstack([v[0] for v in views])  # View matrix
            Q = np.array([v[1] for v in views])  # View returns
            omega_diag = np.array([1.0 / v[2] for v in views])  # Uncertainty
            Omega = np.diag(omega_diag)

            # Black-Litterman formula
            tau_sigma = self.tau * covariance_matrix
            tau_sigma_inv = np.linalg.inv(tau_sigma)
            omega_inv = np.linalg.inv(Omega)

            # Posterior expected returns
            M1 = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
            M2 = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
            bl_returns = M1 @ M2
        else:
            # Use model predictions as absolute views
            bl_returns = pi + self.tau * (expected_returns - pi)

        # Optimize using mean-variance with BL returns
        mv_optimizer = MeanVarianceOptimizer(
            risk_free_rate=self.risk_free_rate,
            constraints=self.constraints,
            objective="max_sharpe",
        )
        result = mv_optimizer.optimize(symbols, bl_returns, covariance_matrix, current_weights)

        return OptimizationResult(
            target_weights=result.target_weights,
            method=OptimizationMethod.BLACK_LITTERMAN,
            expected_return=result.expected_return,
            expected_volatility=result.expected_volatility,
            expected_sharpe=result.expected_sharpe,
            optimization_success=result.optimization_success,
            metadata={
                "tau": self.tau,
                "risk_aversion": self.risk_aversion,
                "implied_returns": pi.tolist(),
                "bl_returns": bl_returns.tolist(),
            },
        )


class HierarchicalRiskParityOptimizer(BasePortfolioOptimizer):
    """Hierarchical Risk Parity (HRP) optimization.

    Uses hierarchical clustering to allocate capital top-down,
    making it more robust to estimation error.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        constraints: OptimizationConstraints | None = None,
        linkage_method: str = "single",
    ) -> None:
        """Initialize HRP optimizer.

        Args:
            risk_free_rate: Annual risk-free rate.
            constraints: Optimization constraints.
            linkage_method: Hierarchical clustering method.
        """
        super().__init__(risk_free_rate, constraints)
        self.linkage_method = linkage_method

    def optimize(
        self,
        symbols: list[str],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: dict[str, float] | None = None,
    ) -> OptimizationResult:
        """Optimize using Hierarchical Risk Parity."""
        self._validate_inputs(symbols, expected_returns, covariance_matrix)
        n = len(symbols)

        # Step 1: Tree clustering
        corr_matrix = self._cov_to_corr(covariance_matrix)
        dist_matrix = np.sqrt((1 - corr_matrix) / 2)  # Distance matrix

        # Handle potential NaN/inf values
        dist_matrix = np.nan_to_num(dist_matrix, nan=0.5, posinf=1.0, neginf=0.0)
        np.fill_diagonal(dist_matrix, 0)

        # Hierarchical clustering
        condensed_dist = squareform(dist_matrix)
        link = linkage(condensed_dist, method=self.linkage_method)

        # Step 2: Quasi-diagonalization
        sorted_indices = self._get_quasi_diag(link)

        # Step 3: Recursive bisection
        weights = self._recursive_bisection(covariance_matrix, sorted_indices)

        # Apply weight constraints
        weights = np.clip(weights, self.constraints.min_weight, self.constraints.max_weight)
        weights = weights / np.sum(weights)  # Renormalize

        target_weights = {s: float(weights[i]) for i, s in enumerate(symbols)}

        # Calculate expected metrics
        exp_ret = float(np.dot(weights, expected_returns))
        exp_vol = float(np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights))))
        exp_sharpe = (exp_ret - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0

        return OptimizationResult(
            target_weights=target_weights,
            method=OptimizationMethod.HIERARCHICAL_RISK_PARITY,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            expected_sharpe=exp_sharpe,
            optimization_success=True,
            metadata={
                "linkage_method": self.linkage_method,
                "sorted_order": sorted_indices,
            },
        )

    def _cov_to_corr(self, cov: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        std = np.sqrt(np.diag(cov))
        std[std == 0] = 1e-10
        outer_std = np.outer(std, std)
        corr = cov / outer_std
        corr[np.isnan(corr)] = 0
        np.fill_diagonal(corr, 1)
        return corr

    def _get_quasi_diag(self, link: np.ndarray) -> list[int]:
        """Get quasi-diagonal ordering from linkage matrix."""
        n = link.shape[0] + 1
        sorted_indices = [n - 1]

        for i in range(n - 1):
            cluster_id = int(link[i, 0]) if link[i, 0] < n else int(link[i, 1])
            if cluster_id < n:
                sorted_indices.append(cluster_id)

        # Ensure we have all indices
        all_indices = set(range(n))
        remaining = list(all_indices - set(sorted_indices))
        sorted_indices.extend(remaining)

        return sorted_indices[:n]

    def _recursive_bisection(
        self,
        cov: np.ndarray,
        sorted_indices: list[int],
    ) -> np.ndarray:
        """Perform recursive bisection for weight allocation."""
        n = cov.shape[0]
        weights = np.ones(n)

        clusters = [sorted_indices]

        while clusters:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                # Split cluster in half
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Calculate cluster variances
                left_var = self._cluster_variance(cov, left)
                right_var = self._cluster_variance(cov, right)

                # Allocate inversely proportional to variance
                total_var = left_var + right_var
                if total_var > 0:
                    left_weight = 1 - left_var / total_var
                    right_weight = 1 - right_var / total_var
                else:
                    left_weight = right_weight = 0.5

                # Apply weights
                for i in left:
                    weights[i] *= left_weight
                for i in right:
                    weights[i] *= right_weight

                new_clusters.extend([left, right])

            clusters = [c for c in new_clusters if len(c) > 1]

        return weights

    def _cluster_variance(self, cov: np.ndarray, indices: list[int]) -> float:
        """Calculate inverse-variance portfolio variance for cluster."""
        sub_cov = cov[np.ix_(indices, indices)]
        ivp_weights = 1 / np.diag(sub_cov)
        ivp_weights = ivp_weights / np.sum(ivp_weights)
        return float(np.dot(ivp_weights.T, np.dot(sub_cov, ivp_weights)))


class PortfolioRebalancer:
    """Handles portfolio rebalancing decisions and trade generation."""

    def __init__(
        self,
        trigger: RebalanceTrigger = RebalanceTrigger.THRESHOLD_BASED,
        drift_threshold: float = 0.05,
        min_rebalance_interval_days: int = 1,
        transaction_cost_bps: float = 10.0,
    ) -> None:
        """Initialize rebalancer.

        Args:
            trigger: Type of rebalancing trigger.
            drift_threshold: Maximum allowed weight drift before rebalancing.
            min_rebalance_interval_days: Minimum days between rebalances.
            transaction_cost_bps: Transaction cost in basis points.
        """
        self.trigger = trigger
        self.drift_threshold = drift_threshold
        self.min_rebalance_interval_days = min_rebalance_interval_days
        self.transaction_cost_bps = transaction_cost_bps
        self._last_rebalance: datetime | None = None

    def should_rebalance(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> tuple[bool, str]:
        """Check if rebalancing is needed.

        Returns:
            Tuple of (should_rebalance, reason).
        """
        # Check minimum interval
        if self._last_rebalance is not None:
            days_since = (datetime.utcnow() - self._last_rebalance).days
            if days_since < self.min_rebalance_interval_days:
                return False, f"Too soon since last rebalance ({days_since} days)"

        # Calculate drift
        max_drift = 0.0
        for symbol in set(current_weights.keys()) | set(target_weights.keys()):
            current = current_weights.get(symbol, 0.0)
            target = target_weights.get(symbol, 0.0)
            drift = abs(current - target)
            max_drift = max(max_drift, drift)

        if max_drift > self.drift_threshold:
            return True, f"Drift exceeded threshold ({max_drift:.2%} > {self.drift_threshold:.2%})"

        return False, f"No rebalance needed (max drift: {max_drift:.2%})"

    def calculate_rebalance_orders(
        self,
        portfolio: Portfolio,
        target_weights: dict[str, float],
        prices: dict[str, Decimal],
    ) -> RebalanceResult:
        """Calculate orders needed to rebalance portfolio.

        Args:
            portfolio: Current portfolio state.
            target_weights: Target portfolio weights.
            prices: Current prices for each symbol.

        Returns:
            Rebalance result with orders.
        """
        orders: list[RebalanceOrder] = []
        total_value = portfolio.equity

        # Calculate current weights
        current_weights = {}
        for symbol, position in portfolio.positions.items():
            current_weights[symbol] = float(position.market_value / total_value)

        # Check if we should rebalance
        should_rebalance, reason = self.should_rebalance(current_weights, target_weights)

        if not should_rebalance:
            return RebalanceResult(
                orders=[],
                total_turnover=0.0,
                estimated_total_cost=Decimal("0"),
                should_rebalance=False,
                trigger_reason=reason,
            )

        # Calculate orders for each symbol
        total_turnover = 0.0
        estimated_total_cost = Decimal("0")

        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        for symbol in all_symbols:
            current_w = current_weights.get(symbol, 0.0)
            target_w = target_weights.get(symbol, 0.0)
            weight_change = target_w - current_w

            if abs(weight_change) < 0.001:  # Ignore tiny changes
                continue

            # Calculate shares to trade
            target_value = total_value * Decimal(str(target_w))
            current_value = total_value * Decimal(str(current_w))
            value_change = target_value - current_value

            price = prices.get(symbol, Decimal("0"))
            if price <= 0:
                continue

            shares_to_trade = abs(value_change / price)

            # Estimate transaction cost
            trade_cost = abs(value_change) * Decimal(str(self.transaction_cost_bps / 10000))

            order = RebalanceOrder(
                symbol=symbol,
                current_weight=current_w,
                target_weight=target_w,
                weight_change=weight_change,
                shares_to_trade=shares_to_trade,
                estimated_cost=trade_cost,
                is_buy=weight_change > 0,
            )
            orders.append(order)

            total_turnover += abs(weight_change)
            estimated_total_cost += trade_cost

        # Update last rebalance time
        if orders:
            self._last_rebalance = datetime.utcnow()

        return RebalanceResult(
            orders=orders,
            total_turnover=total_turnover / 2,  # One-way turnover
            estimated_total_cost=estimated_total_cost,
            should_rebalance=True,
            trigger_reason=reason,
        )


class PortfolioOptimizerFactory:
    """Factory for creating portfolio optimizers."""

    _optimizers: dict[OptimizationMethod, type[BasePortfolioOptimizer]] = {
        OptimizationMethod.MEAN_VARIANCE: MeanVarianceOptimizer,
        OptimizationMethod.MAX_SHARPE: MeanVarianceOptimizer,
        OptimizationMethod.MIN_VARIANCE: MeanVarianceOptimizer,
        OptimizationMethod.RISK_PARITY: RiskParityOptimizer,
        OptimizationMethod.BLACK_LITTERMAN: BlackLittermanOptimizer,
        OptimizationMethod.HIERARCHICAL_RISK_PARITY: HierarchicalRiskParityOptimizer,
    }

    @classmethod
    def create(
        cls,
        method: OptimizationMethod,
        risk_free_rate: float = 0.02,
        constraints: OptimizationConstraints | None = None,
        **kwargs: Any,
    ) -> BasePortfolioOptimizer:
        """Create a portfolio optimizer.

        Args:
            method: Optimization method to use.
            risk_free_rate: Annual risk-free rate.
            constraints: Optimization constraints.
            **kwargs: Method-specific parameters.

        Returns:
            Configured portfolio optimizer.

        Raises:
            ValueError: If method is not recognized.
        """
        optimizer_class = cls._optimizers.get(method)
        if optimizer_class is None:
            raise ValueError(f"Unknown optimization method: {method}")

        # Handle MeanVariance variants
        if method == OptimizationMethod.MAX_SHARPE:
            kwargs["objective"] = "max_sharpe"
        elif method == OptimizationMethod.MIN_VARIANCE:
            kwargs["objective"] = "min_variance"

        return optimizer_class(
            risk_free_rate=risk_free_rate,
            constraints=constraints,
            **kwargs,
        )
