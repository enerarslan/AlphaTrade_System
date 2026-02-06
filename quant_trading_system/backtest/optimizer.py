"""
Walk-forward optimization module for backtesting.

Provides robust parameter optimization with overfitting prevention:
- Walk-forward validation framework
- Multiple optimization methods (grid, random, Bayesian, genetic)
- Out-of-sample validation
- Overfitting detection and prevention
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Iterator

import numpy as np
from scipy.stats import uniform

from quant_trading_system.backtest.analyzer import PerformanceAnalyzer
from quant_trading_system.backtest.engine import BacktestConfig, BacktestEngine, BacktestState, Strategy

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpace:
    """Definition of parameter search space."""

    name: str
    param_type: str  # 'continuous', 'integer', 'categorical'
    low: float | None = None
    high: float | None = None
    categories: list[Any] | None = None
    log_scale: bool = False

    def sample(self) -> Any:
        """Sample a random value from the parameter space."""
        if self.param_type == "continuous":
            if self.log_scale:
                return np.exp(np.random.uniform(np.log(self.low), np.log(self.high)))
            return np.random.uniform(self.low, self.high)
        elif self.param_type == "integer":
            return np.random.randint(int(self.low), int(self.high) + 1)
        elif self.param_type == "categorical":
            return np.random.choice(self.categories)
        return None

    def grid_values(self, n_points: int = 10) -> list[Any]:
        """Get grid values for this parameter."""
        if self.param_type == "continuous":
            if self.log_scale:
                return np.exp(np.linspace(np.log(self.low), np.log(self.high), n_points)).tolist()
            return np.linspace(self.low, self.high, n_points).tolist()
        elif self.param_type == "integer":
            return list(range(int(self.low), int(self.high) + 1))
        elif self.param_type == "categorical":
            return self.categories or []
        return []


@dataclass
class OptimizationWindow:
    """A single optimization window in walk-forward."""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: dict[str, Any] | None = None
    train_metric: float = 0.0
    test_metric: float = 0.0

    @property
    def train_days(self) -> int:
        """Get number of training days."""
        return (self.train_end - self.train_start).days

    @property
    def test_days(self) -> int:
        """Get number of test days."""
        return (self.test_end - self.test_start).days


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""

    best_params: dict[str, Any]
    best_metric: float
    all_results: list[dict[str, Any]]
    windows: list[OptimizationWindow]
    oos_metric: float  # Aggregate out-of-sample metric
    is_ratio: float  # In-sample / Out-of-sample ratio
    optimization_method: str
    total_evaluations: int
    elapsed_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_overfit(self) -> bool:
        """Check if results suggest overfitting."""
        # If IS >> OOS, likely overfit
        return self.is_ratio > 2.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "best_params": self.best_params,
            "best_metric": self.best_metric,
            "oos_metric": self.oos_metric,
            "is_ratio": self.is_ratio,
            "optimization_method": self.optimization_method,
            "total_evaluations": self.total_evaluations,
            "elapsed_seconds": self.elapsed_seconds,
            "is_overfit": self.is_overfit,
            "num_windows": len(self.windows),
        }


class WalkForwardOptimizer:
    """Walk-forward optimization framework.

    Splits data into rolling in-sample (training) and out-of-sample (testing)
    windows to prevent overfitting and assess strategy robustness.
    """

    def __init__(
        self,
        train_period_days: int = 365,
        test_period_days: int = 90,
        step_days: int = 90,
        purge_days: int = 5,
        anchored: bool = False,
    ) -> None:
        """Initialize walk-forward optimizer.

        Args:
            train_period_days: Days in training window.
            test_period_days: Days in test window.
            step_days: Days to step forward between windows.
            purge_days: Gap between train and test to prevent leakage.
            anchored: If True, training always starts from beginning.
        """
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.step_days = step_days
        self.purge_days = purge_days
        self.anchored = anchored

    def generate_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[OptimizationWindow]:
        """Generate walk-forward windows.

        Args:
            start_date: Start of full period.
            end_date: End of full period.

        Returns:
            List of optimization windows.
        """
        windows = []
        window_id = 0

        current_train_start = start_date
        if self.anchored:
            # Anchored: training always starts from beginning
            train_start = start_date
        else:
            train_start = current_train_start

        while True:
            if self.anchored:
                train_end = train_start + timedelta(days=self.train_period_days + window_id * self.step_days)
            else:
                train_end = train_start + timedelta(days=self.train_period_days)

            test_start = train_end + timedelta(days=self.purge_days)
            test_end = test_start + timedelta(days=self.test_period_days)

            if test_end > end_date:
                break

            windows.append(OptimizationWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ))

            window_id += 1

            if not self.anchored:
                train_start = train_start + timedelta(days=self.step_days)

        return windows


class BaseOptimizer(ABC):
    """Base class for parameter optimizers."""

    def __init__(
        self,
        param_spaces: list[ParameterSpace],
        objective_function: Callable[[dict[str, Any]], float],
        maximize: bool = True,
    ) -> None:
        """Initialize optimizer.

        Args:
            param_spaces: List of parameter spaces.
            objective_function: Function to optimize (takes params, returns metric).
            maximize: If True, maximize objective; if False, minimize.
        """
        self.param_spaces = param_spaces
        self.objective_function = objective_function
        self.maximize = maximize
        self._results: list[dict[str, Any]] = []

    @abstractmethod
    def optimize(self, n_iterations: int = 100) -> dict[str, Any]:
        """Run optimization.

        Args:
            n_iterations: Maximum number of iterations.

        Returns:
            Best parameters found.
        """
        pass

    def _evaluate(self, params: dict[str, Any]) -> float:
        """Evaluate objective function and record result."""
        metric = self.objective_function(params)
        self._results.append({"params": params.copy(), "metric": metric})
        return metric

    def get_results(self) -> list[dict[str, Any]]:
        """Get all evaluation results."""
        return self._results.copy()


class GridSearchOptimizer(BaseOptimizer):
    """Exhaustive grid search optimizer."""

    def __init__(
        self,
        param_spaces: list[ParameterSpace],
        objective_function: Callable[[dict[str, Any]], float],
        maximize: bool = True,
        n_points_per_dim: int = 10,
    ) -> None:
        """Initialize grid search optimizer."""
        super().__init__(param_spaces, objective_function, maximize)
        self.n_points_per_dim = n_points_per_dim

    def optimize(self, n_iterations: int = 100) -> dict[str, Any]:
        """Run exhaustive grid search."""
        import itertools

        # Generate grid for each parameter
        param_grids = {}
        for space in self.param_spaces:
            param_grids[space.name] = space.grid_values(self.n_points_per_dim)

        # Generate all combinations
        keys = list(param_grids.keys())
        values = list(param_grids.values())
        combinations = list(itertools.product(*values))

        best_params = None
        best_metric = float("-inf") if self.maximize else float("inf")

        for combo in combinations:
            params = dict(zip(keys, combo))
            metric = self._evaluate(params)

            if (self.maximize and metric > best_metric) or (not self.maximize and metric < best_metric):
                best_metric = metric
                best_params = params.copy()

        return best_params or {}


class RandomSearchOptimizer(BaseOptimizer):
    """Random search optimizer."""

    def optimize(self, n_iterations: int = 100) -> dict[str, Any]:
        """Run random search."""
        best_params = None
        best_metric = float("-inf") if self.maximize else float("inf")

        for _ in range(n_iterations):
            # Sample random parameters
            params = {space.name: space.sample() for space in self.param_spaces}
            metric = self._evaluate(params)

            if (self.maximize and metric > best_metric) or (not self.maximize and metric < best_metric):
                best_metric = metric
                best_params = params.copy()

        return best_params or {}


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization using Gaussian Processes.

    Uses surrogate model to guide exploration efficiently.
    """

    def __init__(
        self,
        param_spaces: list[ParameterSpace],
        objective_function: Callable[[dict[str, Any]], float],
        maximize: bool = True,
        n_initial: int = 10,
        acquisition: str = "ei",  # 'ei' (Expected Improvement) or 'ucb'
    ) -> None:
        """Initialize Bayesian optimizer."""
        super().__init__(param_spaces, objective_function, maximize)
        self.n_initial = n_initial
        self.acquisition = acquisition

        # Only handle continuous parameters for now
        self._continuous_params = [s for s in param_spaces if s.param_type == "continuous"]

    def optimize(self, n_iterations: int = 100) -> dict[str, Any]:
        """Run Bayesian optimization."""
        # Initial random sampling
        X = []
        y = []

        for _ in range(min(self.n_initial, n_iterations)):
            params = {space.name: space.sample() for space in self.param_spaces}
            metric = self._evaluate(params)

            x_vec = [params[s.name] for s in self._continuous_params]
            X.append(x_vec)
            y.append(metric if self.maximize else -metric)

        X = np.array(X)
        y = np.array(y)

        best_idx = np.argmax(y)
        best_params = {s.name: X[best_idx, i] for i, s in enumerate(self._continuous_params)}
        best_metric = y[best_idx] if self.maximize else -y[best_idx]

        # Bayesian optimization loop
        for iteration in range(self.n_initial, n_iterations):
            # Simple acquisition: sample near best points with some exploration
            explore_prob = 0.2 * (1 - iteration / n_iterations)

            if np.random.random() < explore_prob:
                # Explore
                params = {space.name: space.sample() for space in self.param_spaces}
            else:
                # Exploit: sample near best
                best_x = X[np.argmax(y)]
                noise = np.random.normal(0, 0.1, len(best_x))
                new_x = best_x + noise * (np.array([s.high - s.low for s in self._continuous_params]))
                new_x = np.clip(new_x, [s.low for s in self._continuous_params], [s.high for s in self._continuous_params])
                params = {s.name: new_x[i] for i, s in enumerate(self._continuous_params)}

                # Keep non-continuous parameters present and explicit.
                for space in self.param_spaces:
                    if space.param_type == "continuous":
                        continue

                    if best_params and space.name in best_params:
                        params[space.name] = best_params[space.name]
                    else:
                        params[space.name] = space.sample()

            metric = self._evaluate(params)

            x_vec = [params[s.name] for s in self._continuous_params]
            X = np.vstack([X, x_vec])
            y = np.append(y, metric if self.maximize else -metric)

            if (self.maximize and metric > best_metric) or (not self.maximize and metric < best_metric):
                best_metric = metric
                best_params = params.copy()

        return best_params


class GeneticOptimizer(BaseOptimizer):
    """Genetic algorithm optimizer."""

    def __init__(
        self,
        param_spaces: list[ParameterSpace],
        objective_function: Callable[[dict[str, Any]], float],
        maximize: bool = True,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_ratio: float = 0.1,
    ) -> None:
        """Initialize genetic optimizer."""
        super().__init__(param_spaces, objective_function, maximize)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio

    def _create_individual(self) -> dict[str, Any]:
        """Create a random individual."""
        return {space.name: space.sample() for space in self.param_spaces}

    def _crossover(
        self,
        parent1: dict[str, Any],
        parent2: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform crossover between two parents."""
        child = {}
        for space in self.param_spaces:
            if np.random.random() < 0.5:
                child[space.name] = parent1[space.name]
            else:
                child[space.name] = parent2[space.name]
        return child

    def _mutate(self, individual: dict[str, Any]) -> dict[str, Any]:
        """Mutate an individual."""
        mutated = individual.copy()
        for space in self.param_spaces:
            if np.random.random() < self.mutation_rate:
                mutated[space.name] = space.sample()
        return mutated

    def optimize(self, n_iterations: int = 100) -> dict[str, Any]:
        """Run genetic algorithm optimization."""
        # Initialize population
        population = [self._create_individual() for _ in range(self.population_size)]
        fitness = np.array([self._evaluate(ind) for ind in population])

        n_elite = int(self.elite_ratio * self.population_size)
        n_generations = n_iterations // self.population_size

        for generation in range(n_generations):
            # Selection (tournament)
            new_population = []

            # Keep elites
            if self.maximize:
                elite_indices = np.argsort(fitness)[-n_elite:]
            else:
                elite_indices = np.argsort(fitness)[:n_elite]

            for idx in elite_indices:
                new_population.append(population[idx])

            # Generate offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                idx1, idx2 = np.random.choice(len(population), 2, replace=False)
                if (self.maximize and fitness[idx1] > fitness[idx2]) or (not self.maximize and fitness[idx1] < fitness[idx2]):
                    parent1 = population[idx1]
                else:
                    parent1 = population[idx2]

                idx1, idx2 = np.random.choice(len(population), 2, replace=False)
                if (self.maximize and fitness[idx1] > fitness[idx2]) or (not self.maximize and fitness[idx1] < fitness[idx2]):
                    parent2 = population[idx1]
                else:
                    parent2 = population[idx2]

                # Crossover
                if np.random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                # Mutation
                child = self._mutate(child)
                new_population.append(child)

            population = new_population
            fitness = np.array([self._evaluate(ind) for ind in population])

        # Return best
        if self.maximize:
            best_idx = np.argmax(fitness)
        else:
            best_idx = np.argmin(fitness)

        return population[best_idx]


class StrategyOptimizer:
    """High-level optimizer for trading strategies."""

    def __init__(
        self,
        strategy_factory: Callable[[dict[str, Any]], Strategy],
        param_spaces: list[ParameterSpace],
        walk_forward: WalkForwardOptimizer | None = None,
        backtest_config: BacktestConfig | None = None,
        objective: str = "sharpe",  # 'sharpe', 'sortino', 'return', 'calmar'
    ) -> None:
        """Initialize strategy optimizer.

        Args:
            strategy_factory: Function that creates Strategy from params.
            param_spaces: Parameter search spaces.
            walk_forward: Walk-forward optimizer settings.
            backtest_config: Backtest configuration.
            objective: Optimization objective metric.
        """
        self.strategy_factory = strategy_factory
        self.param_spaces = param_spaces
        self.walk_forward = walk_forward or WalkForwardOptimizer()
        self.backtest_config = backtest_config or BacktestConfig()
        self.objective = objective
        self._analyzer = PerformanceAnalyzer()

    def _get_objective_value(self, backtest_state: BacktestState) -> float:
        """Calculate objective value from backtest results."""
        try:
            report = self._analyzer.analyze(backtest_state)

            if self.objective == "sharpe":
                return report.risk_adjusted_metrics.sharpe_ratio
            elif self.objective == "sortino":
                return report.risk_adjusted_metrics.sortino_ratio
            elif self.objective == "return":
                return report.return_metrics.annualized_return
            elif self.objective == "calmar":
                return report.risk_adjusted_metrics.calmar_ratio
            else:
                return report.risk_adjusted_metrics.sharpe_ratio
        except Exception as e:
            logger.warning(f"Error calculating objective: {e}")
            return float("-inf")

    def optimize(
        self,
        data_handler_factory: Callable[[datetime, datetime], Any],
        start_date: datetime,
        end_date: datetime,
        method: str = "random",
        n_iterations: int = 100,
    ) -> OptimizationResult:
        """Run walk-forward optimization.

        Args:
            data_handler_factory: Factory function for data handlers.
            start_date: Start date of full period.
            end_date: End date of full period.
            method: Optimization method ('grid', 'random', 'bayesian', 'genetic').
            n_iterations: Number of iterations per window.

        Returns:
            Optimization result.
        """
        import time
        start_time = time.time()

        # Generate walk-forward windows
        windows = self.walk_forward.generate_windows(start_date, end_date)

        if not windows:
            raise ValueError("No valid walk-forward windows generated")

        all_results = []
        oos_metrics = []

        for window in windows:
            logger.info(f"Optimizing window {window.window_id + 1}/{len(windows)}")

            # Create objective function for this window
            def objective_fn(params: dict[str, Any]) -> float:
                try:
                    strategy = self.strategy_factory(params)
                    config = BacktestConfig(
                        initial_capital=self.backtest_config.initial_capital,
                        start_date=window.train_start,
                        end_date=window.train_end,
                    )
                    data_handler = data_handler_factory(window.train_start, window.train_end)
                    engine = BacktestEngine(data_handler, strategy, config)
                    state = engine.run()
                    return self._get_objective_value(state)
                except Exception as e:
                    logger.warning(f"Error in objective function: {e}")
                    return float("-inf")

            # Create optimizer
            if method == "grid":
                optimizer = GridSearchOptimizer(self.param_spaces, objective_fn)
            elif method == "bayesian":
                optimizer = BayesianOptimizer(self.param_spaces, objective_fn)
            elif method == "genetic":
                optimizer = GeneticOptimizer(self.param_spaces, objective_fn)
            else:  # random
                optimizer = RandomSearchOptimizer(self.param_spaces, objective_fn)

            # Run optimization on training period
            best_params = optimizer.optimize(n_iterations)
            window.best_params = best_params

            # Get training metric
            window.train_metric = max(r["metric"] for r in optimizer.get_results())

            # Evaluate on test period
            try:
                strategy = self.strategy_factory(best_params)
                config = BacktestConfig(
                    initial_capital=self.backtest_config.initial_capital,
                    start_date=window.test_start,
                    end_date=window.test_end,
                )
                data_handler = data_handler_factory(window.test_start, window.test_end)
                engine = BacktestEngine(data_handler, strategy, config)
                state = engine.run()
                window.test_metric = self._get_objective_value(state)
            except Exception as e:
                logger.warning(f"Error in test evaluation: {e}")
                window.test_metric = 0.0

            oos_metrics.append(window.test_metric)
            all_results.extend(optimizer.get_results())

        # Aggregate results
        elapsed = time.time() - start_time

        # Find overall best params (from most recent window or average)
        best_params = windows[-1].best_params or {}

        # Calculate metrics
        is_metrics = [w.train_metric for w in windows]
        avg_is = np.mean(is_metrics) if is_metrics else 0
        avg_oos = np.mean(oos_metrics) if oos_metrics else 0
        is_ratio = avg_is / avg_oos if avg_oos > 0 else float("inf")

        return OptimizationResult(
            best_params=best_params,
            best_metric=windows[-1].train_metric,
            all_results=all_results,
            windows=windows,
            oos_metric=avg_oos,
            is_ratio=is_ratio,
            optimization_method=method,
            total_evaluations=len(all_results),
            elapsed_seconds=elapsed,
            metadata={
                "objective": self.objective,
                "n_windows": len(windows),
                "is_metrics": is_metrics,
                "oos_metrics": oos_metrics,
            },
        )


class OverfitDetector:
    """Detects signs of overfitting in optimization results."""

    def __init__(
        self,
        is_oos_ratio_threshold: float = 2.0,
        min_oos_trades: int = 30,
        parameter_sensitivity_threshold: float = 0.3,
    ) -> None:
        """Initialize overfit detector.

        Args:
            is_oos_ratio_threshold: Max acceptable IS/OOS ratio.
            min_oos_trades: Minimum OOS trades for valid result.
            parameter_sensitivity_threshold: Max acceptable parameter sensitivity.
        """
        self.is_oos_ratio_threshold = is_oos_ratio_threshold
        self.min_oos_trades = min_oos_trades
        self.parameter_sensitivity_threshold = parameter_sensitivity_threshold

    def analyze(
        self,
        result: OptimizationResult,
    ) -> dict[str, Any]:
        """Analyze optimization result for overfitting signs.

        Args:
            result: Optimization result to analyze.

        Returns:
            Dictionary with overfitting analysis.
        """
        warnings = []
        score = 0.0  # 0 = definitely overfit, 1 = robust

        # Check IS/OOS ratio
        if result.is_ratio > self.is_oos_ratio_threshold:
            warnings.append(f"IS/OOS ratio ({result.is_ratio:.2f}) exceeds threshold")
            score -= 0.3
        else:
            score += 0.3

        # Check OOS performance degradation across windows
        if result.windows:
            oos_metrics = [w.test_metric for w in result.windows]
            if len(oos_metrics) > 1:
                # Check if OOS performance is declining
                slope = np.polyfit(range(len(oos_metrics)), oos_metrics, 1)[0]
                if slope < -0.1:
                    warnings.append("OOS performance declining across windows")
                    score -= 0.2

            # Check variance in OOS metrics
            if np.std(oos_metrics) > np.abs(np.mean(oos_metrics)) * 0.5:
                warnings.append("High variance in OOS metrics")
                score -= 0.2

        # Check parameter sensitivity
        if result.all_results:
            metrics = [r["metric"] for r in result.all_results]
            if np.std(metrics) < np.abs(np.mean(metrics)) * 0.1:
                warnings.append("Low sensitivity to parameters (possible data snooping)")
                score -= 0.1

        # Normalize score
        score = max(0.0, min(1.0, 0.5 + score))

        return {
            "is_likely_overfit": score < 0.4,
            "robustness_score": score,
            "warnings": warnings,
            "is_ratio": result.is_ratio,
            "oos_metric": result.oos_metric,
        }
