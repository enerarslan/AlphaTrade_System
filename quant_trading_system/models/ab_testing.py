"""
Model A/B Testing Framework

Production-grade framework for comparing model performance with statistical
rigor. Supports multiple allocation strategies, sequential testing, and
automatic model promotion.

Key Features:
- Traffic splitting with multiple allocation strategies
- Sequential testing with early stopping
- Bayesian inference for conversion rates
- Automated model promotion based on statistical significance
- Comprehensive metrics tracking and reporting

Author: AlphaTrade System
Version: 1.0.0
"""

from __future__ import annotations

import logging
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Optional, List, Dict, Any, Tuple, TypeVar, Generic, Callable
)
from collections import defaultdict
import threading

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class ExperimentStatus(str, Enum):
    """Status of an A/B test experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"


class AllocationStrategy(str, Enum):
    """Strategy for allocating traffic between variants."""
    FIXED = "fixed"  # Fixed percentage split
    EPSILON_GREEDY = "epsilon_greedy"  # Exploit best, explore with epsilon
    THOMPSON_SAMPLING = "thompson_sampling"  # Bayesian bandit
    UCB = "ucb"  # Upper Confidence Bound


class TestType(str, Enum):
    """Type of statistical test."""
    FREQUENTIST = "frequentist"  # Traditional hypothesis testing
    BAYESIAN = "bayesian"  # Bayesian inference
    SEQUENTIAL = "sequential"  # Sequential analysis with early stopping


class MetricType(str, Enum):
    """Type of metric being measured."""
    CONTINUOUS = "continuous"  # E.g., returns, sharpe
    BINARY = "binary"  # E.g., win/loss
    COUNT = "count"  # E.g., number of trades


@dataclass
class VariantConfig:
    """Configuration for an experiment variant."""
    name: str
    model_id: str
    weight: float = 0.5  # Traffic allocation weight
    is_control: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.weight < 0 or self.weight > 1:
            raise ValueError(f"Weight must be between 0 and 1: {self.weight}")


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test experiment."""
    experiment_id: str
    name: str
    description: str
    variants: List[VariantConfig]

    # Statistical settings
    test_type: TestType = TestType.SEQUENTIAL
    significance_level: float = 0.05
    power: float = 0.80
    minimum_detectable_effect: float = 0.01

    # Sample size
    min_samples_per_variant: int = 100
    max_samples_per_variant: int = 10000

    # Duration
    max_duration_days: int = 30
    min_duration_days: int = 7

    # Allocation
    allocation_strategy: AllocationStrategy = AllocationStrategy.FIXED

    # Auto-promotion
    auto_promote: bool = False
    promotion_threshold: float = 0.95  # Probability of being best

    # Guardrails
    enable_guardrails: bool = True
    max_loss_threshold: float = 0.10  # Stop if variant loses > 10%

    def __post_init__(self):
        if len(self.variants) < 2:
            raise ValueError("Experiment requires at least 2 variants")

        weights = sum(v.weight for v in self.variants)
        if abs(weights - 1.0) > 0.01:
            raise ValueError(f"Variant weights must sum to 1.0: {weights}")

        controls = [v for v in self.variants if v.is_control]
        if len(controls) != 1:
            raise ValueError("Exactly one variant must be marked as control")


# =============================================================================
# METRIC TRACKING
# =============================================================================

@dataclass
class MetricObservation:
    """Single observation of a metric."""
    timestamp: datetime
    value: float
    variant_name: str
    unit_id: str  # Symbol, trade_id, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VariantMetrics:
    """Aggregated metrics for a variant."""
    variant_name: str
    sample_size: int
    mean: float
    std: float
    min_value: float
    max_value: float
    sum_value: float
    sum_squared: float

    # For binary metrics
    successes: int = 0
    failures: int = 0

    @property
    def variance(self) -> float:
        if self.sample_size < 2:
            return 0.0
        return self.std ** 2

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.0

    @property
    def standard_error(self) -> float:
        if self.sample_size < 2:
            return float('inf')
        return self.std / np.sqrt(self.sample_size)


class MetricTracker:
    """Thread-safe metric tracking for experiments."""

    def __init__(self):
        self._lock = threading.RLock()
        self._observations: Dict[str, List[MetricObservation]] = defaultdict(list)
        self._running_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'n': 0, 'mean': 0, 'M2': 0, 'sum': 0, 'min': float('inf'), 'max': float('-inf')}
        )

    def record(self, observation: MetricObservation):
        """Record a metric observation with Welford's online algorithm."""
        with self._lock:
            self._observations[observation.variant_name].append(observation)

            # Update running statistics (Welford's algorithm)
            stats = self._running_stats[observation.variant_name]
            stats['n'] += 1
            stats['sum'] += observation.value
            stats['min'] = min(stats['min'], observation.value)
            stats['max'] = max(stats['max'], observation.value)

            delta = observation.value - stats['mean']
            stats['mean'] += delta / stats['n']
            delta2 = observation.value - stats['mean']
            stats['M2'] += delta * delta2

    def get_metrics(self, variant_name: str) -> VariantMetrics:
        """Get aggregated metrics for a variant."""
        with self._lock:
            stats = self._running_stats[variant_name]
            n = int(stats['n'])

            if n == 0:
                return VariantMetrics(
                    variant_name=variant_name,
                    sample_size=0,
                    mean=0.0,
                    std=0.0,
                    min_value=0.0,
                    max_value=0.0,
                    sum_value=0.0,
                    sum_squared=0.0,
                )

            variance = stats['M2'] / n if n > 0 else 0
            std = np.sqrt(variance) if variance > 0 else 0

            # Count successes/failures for binary interpretation
            observations = self._observations[variant_name]
            successes = sum(1 for o in observations if o.value > 0)
            failures = len(observations) - successes

            return VariantMetrics(
                variant_name=variant_name,
                sample_size=n,
                mean=stats['mean'],
                std=std,
                min_value=stats['min'] if n > 0 else 0,
                max_value=stats['max'] if n > 0 else 0,
                sum_value=stats['sum'],
                sum_squared=stats['M2'] + n * stats['mean'] ** 2,
                successes=successes,
                failures=failures,
            )

    def get_all_observations(self, variant_name: str) -> List[MetricObservation]:
        """Get all observations for a variant."""
        with self._lock:
            return list(self._observations[variant_name])

    def clear(self):
        """Clear all tracked data."""
        with self._lock:
            self._observations.clear()
            self._running_stats.clear()


# =============================================================================
# STATISTICAL TESTING
# =============================================================================

@dataclass
class TestResult:
    """Result of a statistical test."""
    is_significant: bool
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    relative_effect: float  # Percentage improvement
    power_achieved: float

    # For Bayesian tests
    probability_better: float = 0.0  # P(treatment > control)
    expected_loss: float = 0.0  # Expected loss if we choose treatment

    # Metadata
    test_type: TestType = TestType.FREQUENTIST
    sample_size_control: int = 0
    sample_size_treatment: int = 0


class StatisticalTester(ABC):
    """Abstract base class for statistical tests."""

    @abstractmethod
    def test(
        self,
        control: VariantMetrics,
        treatment: VariantMetrics,
        alpha: float = 0.05,
    ) -> TestResult:
        """Perform statistical test comparing treatment to control."""
        pass


class FrequentistTester(StatisticalTester):
    """
    Frequentist hypothesis testing using Welch's t-test.

    Handles unequal variances and sample sizes.
    """

    def test(
        self,
        control: VariantMetrics,
        treatment: VariantMetrics,
        alpha: float = 0.05,
    ) -> TestResult:
        """Perform Welch's t-test."""
        n1, n2 = control.sample_size, treatment.sample_size

        if n1 < 2 or n2 < 2:
            return TestResult(
                is_significant=False,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                effect_size=0.0,
                relative_effect=0.0,
                power_achieved=0.0,
                test_type=TestType.FREQUENTIST,
                sample_size_control=n1,
                sample_size_treatment=n2,
            )

        mean1, mean2 = control.mean, treatment.mean
        var1, var2 = control.variance, treatment.variance

        # Welch's t-test
        se = np.sqrt(var1/n1 + var2/n2)
        if se == 0:
            se = 1e-10

        t_stat = (mean2 - mean1) / se

        # Welch-Satterthwaite degrees of freedom
        df = (var1/n1 + var2/n2)**2 / (
            (var1/n1)**2 / (n1-1) + (var2/n2)**2 / (n2-1)
        )
        df = max(1, df)

        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        # Confidence interval for effect
        t_crit = stats.t.ppf(1 - alpha/2, df)
        effect = mean2 - mean1
        ci_low = effect - t_crit * se
        ci_high = effect + t_crit * se

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        cohens_d = effect / pooled_std if pooled_std > 0 else 0

        # Relative effect
        relative_effect = effect / abs(mean1) if abs(mean1) > 0 else 0

        # Power calculation
        power = self._calculate_power(n1, n2, effect, pooled_std, alpha)

        return TestResult(
            is_significant=p_value < alpha,
            p_value=p_value,
            confidence_interval=(ci_low, ci_high),
            effect_size=cohens_d,
            relative_effect=relative_effect,
            power_achieved=power,
            test_type=TestType.FREQUENTIST,
            sample_size_control=n1,
            sample_size_treatment=n2,
        )

    def _calculate_power(
        self,
        n1: int,
        n2: int,
        effect: float,
        std: float,
        alpha: float,
    ) -> float:
        """Calculate achieved power."""
        if std == 0 or n1 < 2 or n2 < 2:
            return 0.0

        se = std * np.sqrt(1/n1 + 1/n2)
        ncp = effect / se  # Non-centrality parameter
        df = n1 + n2 - 2

        t_crit = stats.t.ppf(1 - alpha/2, df)
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)

        return max(0, min(1, power))


class BayesianTester(StatisticalTester):
    """
    Bayesian inference for A/B testing.

    Uses conjugate priors for efficient computation:
    - Beta-Binomial for binary metrics
    - Normal-Normal for continuous metrics
    """

    N_SIMULATIONS: int = 10000

    def test(
        self,
        control: VariantMetrics,
        treatment: VariantMetrics,
        alpha: float = 0.05,
    ) -> TestResult:
        """Perform Bayesian comparison."""
        # Monte Carlo sampling
        control_samples = self._sample_posterior(control)
        treatment_samples = self._sample_posterior(treatment)

        # Probability treatment is better
        prob_better = np.mean(treatment_samples > control_samples)

        # Expected loss if we choose treatment when control is better
        loss_if_wrong = np.maximum(0, control_samples - treatment_samples)
        expected_loss = np.mean(loss_if_wrong)

        # Effect statistics
        effect_samples = treatment_samples - control_samples
        effect = np.mean(effect_samples)
        ci_low, ci_high = np.percentile(effect_samples, [100*alpha/2, 100*(1-alpha/2)])

        # Relative effect
        relative_effect = effect / abs(control.mean) if abs(control.mean) > 0 else 0

        # Effect size
        pooled_std = np.std(np.concatenate([control_samples, treatment_samples]))
        effect_size = effect / pooled_std if pooled_std > 0 else 0

        return TestResult(
            is_significant=prob_better > (1 - alpha),
            p_value=1 - prob_better,  # For compatibility
            confidence_interval=(ci_low, ci_high),
            effect_size=effect_size,
            relative_effect=relative_effect,
            power_achieved=prob_better,
            probability_better=prob_better,
            expected_loss=expected_loss,
            test_type=TestType.BAYESIAN,
            sample_size_control=control.sample_size,
            sample_size_treatment=treatment.sample_size,
        )

    def _sample_posterior(self, metrics: VariantMetrics) -> np.ndarray:
        """Sample from posterior distribution."""
        n = metrics.sample_size

        if n == 0:
            # Prior: Normal(0, 1)
            return np.random.normal(0, 1, self.N_SIMULATIONS)

        # Posterior with Normal-Normal conjugate
        prior_mean, prior_var = 0, 1
        data_var = metrics.variance if metrics.variance > 0 else 1

        # Posterior parameters
        posterior_var = 1 / (1/prior_var + n/data_var)
        posterior_mean = posterior_var * (prior_mean/prior_var + n*metrics.mean/data_var)

        return np.random.normal(
            posterior_mean,
            np.sqrt(posterior_var),
            self.N_SIMULATIONS
        )


class SequentialTester(StatisticalTester):
    """
    Sequential analysis with early stopping.

    Uses group sequential design with O'Brien-Fleming spending function.
    Allows for interim analyses while controlling overall Type I error.
    """

    def __init__(self, n_analyses: int = 5):
        self.n_analyses = n_analyses
        self._current_analysis = 0

    def test(
        self,
        control: VariantMetrics,
        treatment: VariantMetrics,
        alpha: float = 0.05,
    ) -> TestResult:
        """Perform sequential test with adjusted boundaries."""
        # Advance to the next look before spending alpha.
        self._current_analysis = min(self._current_analysis + 1, self.n_analyses)
        information_fraction = min(1.0, self._current_analysis / self.n_analyses)
        adjusted_alpha = self._obrien_fleming_boundary(alpha, information_fraction)

        # Perform frequentist test with adjusted alpha
        freq_tester = FrequentistTester()
        result = freq_tester.test(control, treatment, adjusted_alpha)

        # Override test type
        return TestResult(
            is_significant=result.is_significant,
            p_value=result.p_value,
            confidence_interval=result.confidence_interval,
            effect_size=result.effect_size,
            relative_effect=result.relative_effect,
            power_achieved=result.power_achieved,
            test_type=TestType.SEQUENTIAL,
            sample_size_control=result.sample_size_control,
            sample_size_treatment=result.sample_size_treatment,
        )

    def advance_analysis(self):
        """Move to next interim analysis."""
        self._current_analysis = min(
            self._current_analysis + 1,
            self.n_analyses
        )

    def _obrien_fleming_boundary(
        self,
        alpha: float,
        information_fraction: float,
    ) -> float:
        """Calculate O'Brien-Fleming spending function."""
        if information_fraction <= 0:
            return 0.0
        if information_fraction >= 1:
            return alpha

        # O'Brien-Fleming approximation
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_t = z_alpha / np.sqrt(information_fraction)

        return 2 * (1 - stats.norm.cdf(z_t))


# =============================================================================
# TRAFFIC ALLOCATION
# =============================================================================

class TrafficAllocator(ABC):
    """Abstract base class for traffic allocation."""

    @abstractmethod
    def allocate(self, unit_id: str, variants: List[VariantConfig]) -> str:
        """Allocate a unit to a variant."""
        pass


class FixedAllocator(TrafficAllocator):
    """Fixed weight allocation using deterministic hashing."""

    def allocate(self, unit_id: str, variants: List[VariantConfig]) -> str:
        """Allocate based on hash of unit_id."""
        hash_value = int(hashlib.md5(unit_id.encode()).hexdigest(), 16)
        random_value = (hash_value % 10000) / 10000

        cumulative = 0.0
        for variant in variants:
            cumulative += variant.weight
            if random_value < cumulative:
                return variant.name

        return variants[-1].name


class EpsilonGreedyAllocator(TrafficAllocator):
    """Epsilon-greedy allocation for exploration/exploitation."""

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        self._metrics: Dict[str, VariantMetrics] = {}

    def update_metrics(self, metrics: Dict[str, VariantMetrics]):
        """Update current metrics for each variant."""
        self._metrics = metrics

    def allocate(self, unit_id: str, variants: List[VariantConfig]) -> str:
        """Allocate with epsilon-greedy strategy."""
        hash_value = int(hashlib.md5(unit_id.encode()).hexdigest(), 16)
        random_value = (hash_value % 10000) / 10000

        if random_value < self.epsilon:
            # Explore: random allocation
            idx = hash_value % len(variants)
            return variants[idx].name
        else:
            # Exploit: choose best performing
            if not self._metrics:
                return variants[0].name

            best_variant = max(
                variants,
                key=lambda v: self._metrics.get(v.name, VariantMetrics(v.name, 0, 0, 0, 0, 0, 0, 0)).mean
            )
            return best_variant.name


class ThompsonSamplingAllocator(TrafficAllocator):
    """Thompson Sampling for multi-armed bandit allocation."""

    def __init__(self):
        self._successes: Dict[str, int] = defaultdict(lambda: 1)  # Prior
        self._failures: Dict[str, int] = defaultdict(lambda: 1)

    def record_outcome(self, variant_name: str, success: bool):
        """Record outcome for Bayesian update."""
        if success:
            self._successes[variant_name] += 1
        else:
            self._failures[variant_name] += 1

    def allocate(self, unit_id: str, variants: List[VariantConfig]) -> str:
        """Allocate using Thompson Sampling."""
        # Sample from Beta posterior for each variant
        samples = {}
        for variant in variants:
            alpha = self._successes[variant.name]
            beta = self._failures[variant.name]
            samples[variant.name] = np.random.beta(alpha, beta)

        # Choose variant with highest sample
        return max(samples, key=samples.get)


class UCBAllocator(TrafficAllocator):
    """Upper Confidence Bound allocation."""

    def __init__(self, exploration_factor: float = 2.0):
        self.exploration_factor = exploration_factor
        self._counts: Dict[str, int] = defaultdict(int)
        self._rewards: Dict[str, float] = defaultdict(float)
        self._total_count: int = 0

    def record_reward(self, variant_name: str, reward: float):
        """Record reward for UCB calculation."""
        self._counts[variant_name] += 1
        self._rewards[variant_name] += reward
        self._total_count += 1

    def allocate(self, unit_id: str, variants: List[VariantConfig]) -> str:
        """Allocate using UCB1 algorithm."""
        # Ensure each variant is tried at least once
        for variant in variants:
            if self._counts[variant.name] == 0:
                return variant.name

        # Calculate UCB for each variant
        ucb_values = {}
        for variant in variants:
            n = self._counts[variant.name]
            avg_reward = self._rewards[variant.name] / n if n > 0 else 0
            exploration_bonus = np.sqrt(
                self.exploration_factor * np.log(self._total_count) / n
            )
            ucb_values[variant.name] = avg_reward + exploration_bonus

        return max(ucb_values, key=ucb_values.get)


# =============================================================================
# EXPERIMENT MANAGER
# =============================================================================

@dataclass
class ExperimentResult:
    """Final result of an experiment."""
    experiment_id: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime]
    winner: Optional[str]
    test_results: Dict[str, TestResult]  # Treatment name -> result vs control
    variant_metrics: Dict[str, VariantMetrics]
    recommendation: str
    confidence: float


class Experiment:
    """
    A/B test experiment instance.

    Manages the full lifecycle of an experiment including traffic allocation,
    metric tracking, statistical testing, and result reporting.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.status = ExperimentStatus.DRAFT
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Components
        self._metric_tracker = MetricTracker()
        self._allocator = self._create_allocator()
        self._tester = self._create_tester()

        # State
        self._lock = threading.RLock()

    def _create_allocator(self) -> TrafficAllocator:
        """Create appropriate traffic allocator."""
        allocators = {
            AllocationStrategy.FIXED: FixedAllocator,
            AllocationStrategy.EPSILON_GREEDY: EpsilonGreedyAllocator,
            AllocationStrategy.THOMPSON_SAMPLING: ThompsonSamplingAllocator,
            AllocationStrategy.UCB: UCBAllocator,
        }
        return allocators[self.config.allocation_strategy]()

    def _create_tester(self) -> StatisticalTester:
        """Create appropriate statistical tester."""
        testers = {
            TestType.FREQUENTIST: FrequentistTester,
            TestType.BAYESIAN: BayesianTester,
            TestType.SEQUENTIAL: SequentialTester,
        }
        return testers[self.config.test_type]()

    def start(self):
        """Start the experiment."""
        with self._lock:
            if self.status != ExperimentStatus.DRAFT:
                raise ValueError(f"Cannot start experiment in status {self.status}")

            self.status = ExperimentStatus.RUNNING
            self.start_time = datetime.now()

            logger.info(
                f"Started experiment {self.config.experiment_id}: "
                f"{self.config.name}"
            )

    def pause(self):
        """Pause the experiment."""
        with self._lock:
            if self.status != ExperimentStatus.RUNNING:
                raise ValueError(f"Cannot pause experiment in status {self.status}")

            self.status = ExperimentStatus.PAUSED
            logger.info(f"Paused experiment {self.config.experiment_id}")

    def resume(self):
        """Resume a paused experiment."""
        with self._lock:
            if self.status != ExperimentStatus.PAUSED:
                raise ValueError(f"Cannot resume experiment in status {self.status}")

            self.status = ExperimentStatus.RUNNING
            logger.info(f"Resumed experiment {self.config.experiment_id}")

    def complete(self):
        """Mark experiment as complete."""
        with self._lock:
            self.status = ExperimentStatus.COMPLETED
            self.end_time = datetime.now()
            logger.info(f"Completed experiment {self.config.experiment_id}")

    def rollback(self):
        """Rollback experiment (e.g., due to guardrail violation)."""
        with self._lock:
            self.status = ExperimentStatus.ROLLED_BACK
            self.end_time = datetime.now()
            logger.warning(f"Rolled back experiment {self.config.experiment_id}")

    def allocate(self, unit_id: str) -> str:
        """Allocate a unit to a variant."""
        if self.status != ExperimentStatus.RUNNING:
            # Default to control when not running
            control = next(v for v in self.config.variants if v.is_control)
            return control.name

        return self._allocator.allocate(unit_id, self.config.variants)

    def record_metric(
        self,
        variant_name: str,
        value: float,
        unit_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a metric observation."""
        observation = MetricObservation(
            timestamp=datetime.now(),
            value=value,
            variant_name=variant_name,
            unit_id=unit_id,
            metadata=metadata or {},
        )
        self._metric_tracker.record(observation)

        # Update adaptive allocators
        if isinstance(self._allocator, ThompsonSamplingAllocator):
            self._allocator.record_outcome(variant_name, value > 0)
        elif isinstance(self._allocator, UCBAllocator):
            self._allocator.record_reward(variant_name, value)
        elif isinstance(self._allocator, EpsilonGreedyAllocator):
            metrics = {
                v.name: self._metric_tracker.get_metrics(v.name)
                for v in self.config.variants
            }
            self._allocator.update_metrics(metrics)

        # Check guardrails
        if self.config.enable_guardrails:
            self._check_guardrails()

    def _check_guardrails(self):
        """Check if any guardrails are violated."""
        control = next(v for v in self.config.variants if v.is_control)
        control_metrics = self._metric_tracker.get_metrics(control.name)

        for variant in self.config.variants:
            if variant.is_control:
                continue

            treatment_metrics = self._metric_tracker.get_metrics(variant.name)

            if treatment_metrics.sample_size < 30:  # Minimum for comparison
                continue

            # Check if treatment is significantly worse
            if control_metrics.mean > 0:
                relative_loss = (
                    (control_metrics.mean - treatment_metrics.mean)
                    / control_metrics.mean
                )
                if relative_loss > self.config.max_loss_threshold:
                    logger.warning(
                        f"Guardrail violated: {variant.name} is "
                        f"{relative_loss:.1%} worse than control"
                    )
                    self.rollback()
                    return

    def analyze(self) -> Dict[str, TestResult]:
        """Run statistical analysis on all variants vs control."""
        control = next(v for v in self.config.variants if v.is_control)
        control_metrics = self._metric_tracker.get_metrics(control.name)

        results = {}
        for variant in self.config.variants:
            if variant.is_control:
                continue

            treatment_metrics = self._metric_tracker.get_metrics(variant.name)
            result = self._tester.test(
                control_metrics,
                treatment_metrics,
                self.config.significance_level,
            )
            results[variant.name] = result

        # Check for auto-promotion
        if self.config.auto_promote and self.status == ExperimentStatus.RUNNING:
            self._check_auto_promotion(results)

        return results

    def _check_auto_promotion(self, results: Dict[str, TestResult]):
        """Check if any variant should be auto-promoted."""
        for variant_name, result in results.items():
            if result.probability_better >= self.config.promotion_threshold:
                logger.info(
                    f"Auto-promoting {variant_name} with "
                    f"probability {result.probability_better:.2%}"
                )
                self.complete()
                return

    def get_result(self) -> ExperimentResult:
        """Get current experiment result."""
        test_results = self.analyze()

        variant_metrics = {
            v.name: self._metric_tracker.get_metrics(v.name)
            for v in self.config.variants
        }

        # Determine winner
        winner = None
        best_prob = 0.0
        for variant_name, result in test_results.items():
            if result.is_significant and result.effect_size > 0:
                if result.probability_better > best_prob:
                    best_prob = result.probability_better
                    winner = variant_name

        # Generate recommendation
        if winner:
            recommendation = f"Promote {winner} (confidence: {best_prob:.1%})"
            confidence = best_prob
        elif any(r.sample_size_treatment < self.config.min_samples_per_variant
                 for r in test_results.values()):
            recommendation = "Continue experiment - insufficient data"
            confidence = 0.0
        else:
            recommendation = "No significant difference - keep control"
            confidence = 1 - max(r.probability_better for r in test_results.values())

        return ExperimentResult(
            experiment_id=self.config.experiment_id,
            status=self.status,
            start_time=self.start_time,
            end_time=self.end_time,
            winner=winner,
            test_results=test_results,
            variant_metrics=variant_metrics,
            recommendation=recommendation,
            confidence=confidence,
        )

    def should_continue(self) -> bool:
        """Check if experiment should continue running."""
        if self.status != ExperimentStatus.RUNNING:
            return False

        # Check max duration
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            if elapsed > timedelta(days=self.config.max_duration_days):
                logger.info(f"Experiment {self.config.experiment_id} reached max duration")
                return False

        # Check sample size
        for variant in self.config.variants:
            metrics = self._metric_tracker.get_metrics(variant.name)
            if metrics.sample_size >= self.config.max_samples_per_variant:
                logger.info(f"Experiment {self.config.experiment_id} reached max samples")
                return False

        # Check early stopping for sequential tests
        if self.config.test_type == TestType.SEQUENTIAL:
            results = self.analyze()
            if any(r.is_significant for r in results.values()):
                logger.info(f"Experiment {self.config.experiment_id} achieved significance")
                return False

        return True


class ExperimentManager:
    """
    Central manager for all A/B test experiments.

    Provides experiment lifecycle management, model routing, and reporting.
    """

    def __init__(self):
        self._experiments: Dict[str, Experiment] = {}
        self._active_experiments: Dict[str, str] = {}  # model_id -> experiment_id
        self._lock = threading.RLock()

    def create_experiment(self, config: ExperimentConfig) -> Experiment:
        """Create a new experiment."""
        with self._lock:
            if config.experiment_id in self._experiments:
                raise ValueError(f"Experiment {config.experiment_id} already exists")

            experiment = Experiment(config)
            self._experiments[config.experiment_id] = experiment

            logger.info(f"Created experiment: {config.experiment_id}")
            return experiment

    def start_experiment(self, experiment_id: str):
        """Start an experiment and register for model routing."""
        with self._lock:
            experiment = self._experiments.get(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")

            experiment.start()

            # Register model IDs for routing
            for variant in experiment.config.variants:
                self._active_experiments[variant.model_id] = experiment_id

    def get_model_for_unit(
        self,
        default_model_id: str,
        unit_id: str,
    ) -> str:
        """
        Get model ID to use for a given unit.

        Routes to experiment variant if unit is in an active experiment.
        """
        with self._lock:
            experiment_id = self._active_experiments.get(default_model_id)
            if not experiment_id:
                return default_model_id

            experiment = self._experiments.get(experiment_id)
            if not experiment or experiment.status != ExperimentStatus.RUNNING:
                return default_model_id

            # Get allocated variant
            variant_name = experiment.allocate(unit_id)
            variant = next(
                v for v in experiment.config.variants
                if v.name == variant_name
            )
            return variant.model_id

    def record_outcome(
        self,
        model_id: str,
        value: float,
        unit_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record an outcome for a model."""
        with self._lock:
            experiment_id = self._active_experiments.get(model_id)
            if not experiment_id:
                return

            experiment = self._experiments.get(experiment_id)
            if not experiment:
                return

            # Find variant name for this model
            variant = next(
                (v for v in experiment.config.variants if v.model_id == model_id),
                None
            )
            if variant:
                experiment.record_metric(variant.name, value, unit_id, metadata)

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self._experiments.get(experiment_id)

    def get_all_results(self) -> Dict[str, ExperimentResult]:
        """Get results for all experiments."""
        with self._lock:
            return {
                exp_id: exp.get_result()
                for exp_id, exp in self._experiments.items()
            }

    def cleanup_completed(self, days_old: int = 30):
        """Remove completed experiments older than threshold."""
        with self._lock:
            cutoff = datetime.now() - timedelta(days=days_old)
            to_remove = []

            for exp_id, exp in self._experiments.items():
                if exp.status in (ExperimentStatus.COMPLETED, ExperimentStatus.ROLLED_BACK):
                    if exp.end_time and exp.end_time < cutoff:
                        to_remove.append(exp_id)

            for exp_id in to_remove:
                del self._experiments[exp_id]
                # Clean up active experiments
                self._active_experiments = {
                    k: v for k, v in self._active_experiments.items()
                    if v != exp_id
                }

            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old experiments")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_experiment_manager: Optional[ExperimentManager] = None
_manager_lock = threading.Lock()


def get_experiment_manager() -> ExperimentManager:
    """Get singleton experiment manager."""
    global _experiment_manager
    with _manager_lock:
        if _experiment_manager is None:
            _experiment_manager = ExperimentManager()
        return _experiment_manager


def create_ab_test(
    name: str,
    control_model_id: str,
    treatment_model_id: str,
    control_weight: float = 0.5,
    test_type: TestType = TestType.SEQUENTIAL,
    **kwargs,
) -> Experiment:
    """
    Convenience function to create a simple A/B test.

    Args:
        name: Human-readable experiment name
        control_model_id: Model ID for control variant
        treatment_model_id: Model ID for treatment variant
        control_weight: Traffic fraction for control (default 0.5)
        test_type: Type of statistical test
        **kwargs: Additional ExperimentConfig parameters

    Returns:
        Configured experiment
    """
    experiment_id = f"ab_{hashlib.md5(name.encode()).hexdigest()[:8]}"

    config = ExperimentConfig(
        experiment_id=experiment_id,
        name=name,
        description=f"A/B test: {control_model_id} vs {treatment_model_id}",
        variants=[
            VariantConfig(
                name="control",
                model_id=control_model_id,
                weight=control_weight,
                is_control=True,
            ),
            VariantConfig(
                name="treatment",
                model_id=treatment_model_id,
                weight=1 - control_weight,
                is_control=False,
            ),
        ],
        test_type=test_type,
        **kwargs,
    )

    manager = get_experiment_manager()
    return manager.create_experiment(config)


def create_multi_variant_test(
    name: str,
    control_model_id: str,
    treatment_model_ids: List[str],
    test_type: TestType = TestType.BAYESIAN,
    **kwargs,
) -> Experiment:
    """
    Create a multi-variant test (A/B/n).

    Args:
        name: Human-readable experiment name
        control_model_id: Model ID for control
        treatment_model_ids: List of treatment model IDs
        test_type: Type of statistical test
        **kwargs: Additional ExperimentConfig parameters

    Returns:
        Configured experiment
    """
    n_variants = 1 + len(treatment_model_ids)
    weight = 1.0 / n_variants

    experiment_id = f"abn_{hashlib.md5(name.encode()).hexdigest()[:8]}"

    variants = [
        VariantConfig(
            name="control",
            model_id=control_model_id,
            weight=weight,
            is_control=True,
        )
    ]

    for i, model_id in enumerate(treatment_model_ids):
        variants.append(VariantConfig(
            name=f"treatment_{i+1}",
            model_id=model_id,
            weight=weight,
            is_control=False,
        ))

    config = ExperimentConfig(
        experiment_id=experiment_id,
        name=name,
        description=f"Multi-variant test with {n_variants} variants",
        variants=variants,
        test_type=test_type,
        allocation_strategy=AllocationStrategy.THOMPSON_SAMPLING,
        **kwargs,
    )

    manager = get_experiment_manager()
    return manager.create_experiment(config)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'ExperimentStatus',
    'AllocationStrategy',
    'TestType',
    'MetricType',

    # Config
    'VariantConfig',
    'ExperimentConfig',

    # Metrics
    'MetricObservation',
    'VariantMetrics',
    'MetricTracker',

    # Testing
    'TestResult',
    'StatisticalTester',
    'FrequentistTester',
    'BayesianTester',
    'SequentialTester',

    # Allocation
    'TrafficAllocator',
    'FixedAllocator',
    'EpsilonGreedyAllocator',
    'ThompsonSamplingAllocator',
    'UCBAllocator',

    # Experiment
    'ExperimentResult',
    'Experiment',
    'ExperimentManager',

    # Factory Functions
    'get_experiment_manager',
    'create_ab_test',
    'create_multi_variant_test',
]
