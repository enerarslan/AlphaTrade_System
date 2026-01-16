"""
JPMORGAN FIX: Alpha Factor Metrics Module.

Implements institutional-grade alpha factor evaluation metrics:
1. Information Coefficient (IC) - Correlation between alpha and forward returns
2. Rank IC - Spearman correlation (robust to outliers)
3. Information Ratio (IR) - IC / std(IC), risk-adjusted alpha quality
4. IC Decay - How predictive power decays over time
5. IC Stability - Consistency of alpha over different periods
6. Turnover Analysis - Alpha's churn rate
7. Breadth - Number of bets per period

These metrics are essential for:
- Alpha factor research and development
- Portfolio construction and optimization
- Model monitoring and performance attribution
- Regulatory reporting and risk management

Reference:
    "Active Portfolio Management" by Grinold & Kahn
    "Advances in Active Portfolio Management" by Fabozzi & Grant
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ICResult:
    """Result container for IC calculation."""

    ic: float  # Pearson correlation
    rank_ic: float  # Spearman correlation
    p_value: float  # Statistical significance
    t_stat: float  # T-statistic
    n_observations: int  # Number of observations used
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if IC is statistically significant."""
        return self.p_value < alpha

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ic": self.ic,
            "rank_ic": self.rank_ic,
            "p_value": self.p_value,
            "t_stat": self.t_stat,
            "n_observations": self.n_observations,
            "is_significant_5pct": self.is_significant(0.05),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class IRResult:
    """Result container for Information Ratio calculation."""

    ir: float  # Information Ratio
    mean_ic: float  # Mean IC over period
    std_ic: float  # Std of IC over period
    n_periods: int  # Number of IC observations
    hit_rate: float  # Percentage of positive ICs
    positive_count: int  # Number of positive ICs
    total_count: int  # Total IC observations
    annualized_ir: float = 0.0  # Annualized IR

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ir": self.ir,
            "mean_ic": self.mean_ic,
            "std_ic": self.std_ic,
            "n_periods": self.n_periods,
            "hit_rate": self.hit_rate,
            "positive_count": self.positive_count,
            "total_count": self.total_count,
            "annualized_ir": self.annualized_ir,
        }


@dataclass
class ICDecayResult:
    """Result container for IC decay analysis."""

    horizons: list[int]  # Forecast horizons
    ic_values: list[float]  # IC at each horizon
    rank_ic_values: list[float]  # Rank IC at each horizon
    half_life: float  # Estimated half-life of IC decay
    decay_rate: float  # Exponential decay rate

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "horizons": self.horizons,
            "ic_values": self.ic_values,
            "rank_ic_values": self.rank_ic_values,
            "half_life": self.half_life,
            "decay_rate": self.decay_rate,
        }


@dataclass
class AlphaMetricsReport:
    """Comprehensive alpha metrics report."""

    alpha_name: str
    ic_result: ICResult
    ir_result: IRResult
    ic_decay: ICDecayResult | None
    turnover: float
    breadth: float
    factor_return: float
    factor_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    stability_score: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alpha_name": self.alpha_name,
            "ic": self.ic_result.to_dict(),
            "ir": self.ir_result.to_dict(),
            "ic_decay": self.ic_decay.to_dict() if self.ic_decay else None,
            "turnover": self.turnover,
            "breadth": self.breadth,
            "factor_return": self.factor_return,
            "factor_volatility": self.factor_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "stability_score": self.stability_score,
            "created_at": self.created_at.isoformat(),
        }


class AlphaMetricsCalculator:
    """
    JPMORGAN FIX: Comprehensive alpha factor metrics calculator.

    Implements institutional-grade alpha evaluation following the
    fundamental law of active management:

    IR = IC * sqrt(BR)

    Where:
    - IR = Information Ratio (risk-adjusted excess return)
    - IC = Information Coefficient (forecasting skill)
    - BR = Breadth (number of independent bets)

    Usage:
        calculator = AlphaMetricsCalculator()

        # Compute point-in-time IC
        ic_result = calculator.compute_ic(alpha_signal, forward_returns)

        # Compute rolling IR
        ir_result = calculator.compute_ir(alpha_signal, forward_returns, window=60)

        # Full metrics report
        report = calculator.compute_full_report("momentum", alpha_signal, forward_returns)
    """

    def __init__(
        self,
        annualization_factor: float = 252.0,
        min_observations: int = 20,
    ):
        """
        Initialize metrics calculator.

        Args:
            annualization_factor: Factor for annualizing metrics (252 for daily)
            min_observations: Minimum observations required for calculation
        """
        self.annualization_factor = annualization_factor
        self.min_observations = min_observations

    def compute_ic(
        self,
        alpha_signal: np.ndarray,
        forward_returns: np.ndarray,
        return_lag: int = 1,
    ) -> ICResult:
        """
        Compute Information Coefficient between alpha and forward returns.

        CRITICAL: Uses lagged alpha values with forward returns to prevent
        look-ahead bias. Alpha at time t should predict returns at time t+lag.

        Args:
            alpha_signal: Alpha signal values
            forward_returns: Forward returns
            return_lag: Number of periods for forward returns

        Returns:
            ICResult with IC, Rank IC, p-value, and t-stat
        """
        # Align alpha with forward returns
        if return_lag > 0 and len(alpha_signal) > return_lag and len(forward_returns) > return_lag:
            lagged_alpha = alpha_signal[:-return_lag]
            fwd_returns = forward_returns[return_lag:]
        else:
            lagged_alpha = alpha_signal
            fwd_returns = forward_returns

        # Filter valid observations
        valid_mask = ~(np.isnan(lagged_alpha) | np.isnan(fwd_returns) |
                      np.isinf(lagged_alpha) | np.isinf(fwd_returns))
        n_valid = np.sum(valid_mask)

        if n_valid < self.min_observations:
            return ICResult(
                ic=0.0,
                rank_ic=0.0,
                p_value=1.0,
                t_stat=0.0,
                n_observations=n_valid,
            )

        valid_alpha = lagged_alpha[valid_mask]
        valid_returns = fwd_returns[valid_mask]

        # Pearson IC
        pearson_ic, pearson_pvalue = stats.pearsonr(valid_alpha, valid_returns)

        # Rank IC (Spearman) - more robust to outliers
        rank_ic, rank_pvalue = stats.spearmanr(valid_alpha, valid_returns)

        # T-statistic for IC
        # t = IC * sqrt(n-2) / sqrt(1 - IC^2)
        if abs(rank_ic) < 1.0:
            t_stat = rank_ic * np.sqrt(n_valid - 2) / np.sqrt(1 - rank_ic**2)
        else:
            t_stat = np.inf * np.sign(rank_ic)

        # Handle NaN values
        pearson_ic = 0.0 if np.isnan(pearson_ic) else pearson_ic
        rank_ic = 0.0 if np.isnan(rank_ic) else rank_ic

        return ICResult(
            ic=float(pearson_ic),
            rank_ic=float(rank_ic),
            p_value=float(rank_pvalue),
            t_stat=float(t_stat),
            n_observations=int(n_valid),
        )

    def compute_rolling_ic(
        self,
        alpha_signal: np.ndarray,
        forward_returns: np.ndarray,
        window: int = 20,
        return_lag: int = 1,
    ) -> np.ndarray:
        """
        Compute rolling IC over time.

        Args:
            alpha_signal: Alpha signal values
            forward_returns: Forward returns
            window: Rolling window size
            return_lag: Number of periods for forward returns

        Returns:
            Array of rolling IC values
        """
        n = len(alpha_signal)
        rolling_ic = np.full(n, np.nan)

        # Align alpha with forward returns
        if return_lag > 0 and n > return_lag:
            lagged_alpha = alpha_signal[:-return_lag]
            fwd_returns = forward_returns[return_lag:]
            effective_n = len(lagged_alpha)
        else:
            lagged_alpha = alpha_signal
            fwd_returns = forward_returns
            effective_n = n

        for i in range(window, effective_n):
            window_alpha = lagged_alpha[i - window:i]
            window_returns = fwd_returns[i - window:i]

            valid_mask = ~(np.isnan(window_alpha) | np.isnan(window_returns))
            if np.sum(valid_mask) >= self.min_observations // 2:
                ic, _ = stats.spearmanr(
                    window_alpha[valid_mask],
                    window_returns[valid_mask]
                )
                # Map back to original index
                original_idx = i + return_lag if return_lag > 0 else i
                if original_idx < n:
                    rolling_ic[original_idx] = ic if not np.isnan(ic) else 0.0

        return rolling_ic

    def compute_ir(
        self,
        alpha_signal: np.ndarray,
        forward_returns: np.ndarray,
        window: int = 60,
        return_lag: int = 1,
    ) -> IRResult:
        """
        Compute Information Ratio.

        IR = Mean(IC) / Std(IC)

        This measures the consistency and reliability of the alpha's
        forecasting ability.

        Args:
            alpha_signal: Alpha signal values
            forward_returns: Forward returns
            window: Window for rolling IC calculation
            return_lag: Number of periods for forward returns

        Returns:
            IRResult with IR, mean IC, std IC, and hit rate
        """
        # Compute rolling ICs
        rolling_ic = self.compute_rolling_ic(
            alpha_signal, forward_returns, window, return_lag
        )

        # Filter valid ICs
        valid_ics = rolling_ic[~np.isnan(rolling_ic)]

        if len(valid_ics) < 3:
            return IRResult(
                ir=0.0,
                mean_ic=0.0,
                std_ic=0.0,
                n_periods=0,
                hit_rate=0.0,
                positive_count=0,
                total_count=0,
                annualized_ir=0.0,
            )

        mean_ic = np.mean(valid_ics)
        std_ic = np.std(valid_ics, ddof=1)

        # Information Ratio
        ir = mean_ic / std_ic if std_ic > 1e-10 else 0.0

        # Hit rate (percentage of positive ICs)
        positive_count = np.sum(valid_ics > 0)
        hit_rate = positive_count / len(valid_ics)

        # Annualized IR
        # Assuming window is the IC calculation frequency
        annualized_ir = ir * np.sqrt(self.annualization_factor / window)

        return IRResult(
            ir=float(ir),
            mean_ic=float(mean_ic),
            std_ic=float(std_ic),
            n_periods=len(valid_ics),
            hit_rate=float(hit_rate),
            positive_count=int(positive_count),
            total_count=len(valid_ics),
            annualized_ir=float(annualized_ir),
        )

    def compute_ic_decay(
        self,
        alpha_signal: np.ndarray,
        returns: np.ndarray,
        max_horizon: int = 20,
    ) -> ICDecayResult:
        """
        Compute IC decay across different forecast horizons.

        This measures how quickly the alpha's predictive power decays
        as we look further into the future.

        Args:
            alpha_signal: Alpha signal values
            returns: Return series (will be shifted for different horizons)
            max_horizon: Maximum horizon to analyze

        Returns:
            ICDecayResult with IC values at each horizon
        """
        horizons = list(range(1, max_horizon + 1))
        ic_values = []
        rank_ic_values = []

        for horizon in horizons:
            # Compute forward returns for this horizon
            if horizon < len(returns):
                forward_returns = np.zeros(len(returns))
                for i in range(len(returns) - horizon):
                    # Cumulative return over horizon
                    forward_returns[i] = np.prod(1 + returns[i+1:i+horizon+1]) - 1

                ic_result = self.compute_ic(alpha_signal, forward_returns, return_lag=0)
                ic_values.append(ic_result.ic)
                rank_ic_values.append(ic_result.rank_ic)
            else:
                ic_values.append(0.0)
                rank_ic_values.append(0.0)

        # Estimate decay parameters using exponential fit
        # IC(h) = IC(1) * exp(-lambda * h)
        half_life = 0.0
        decay_rate = 0.0

        valid_ics = np.array(ic_values)
        if len(valid_ics) > 2 and valid_ics[0] > 0:
            # Fit exponential decay
            log_ics = np.log(np.maximum(valid_ics / valid_ics[0], 1e-10))
            try:
                slope, _, _, _, _ = stats.linregress(horizons, log_ics)
                decay_rate = -slope
                half_life = np.log(2) / decay_rate if decay_rate > 0 else np.inf
            except Exception:
                pass

        return ICDecayResult(
            horizons=horizons,
            ic_values=ic_values,
            rank_ic_values=rank_ic_values,
            half_life=float(half_life),
            decay_rate=float(decay_rate),
        )

    def compute_turnover(
        self,
        alpha_signal: np.ndarray,
        normalize: bool = True,
    ) -> float:
        """
        Compute alpha signal turnover.

        Measures how much the alpha signal changes over time,
        which impacts trading costs.

        Args:
            alpha_signal: Alpha signal values
            normalize: Whether to normalize by mean absolute signal

        Returns:
            Turnover value (lower is better for cost efficiency)
        """
        valid_signal = alpha_signal[~np.isnan(alpha_signal)]

        if len(valid_signal) < 2:
            return 0.0

        # Absolute changes
        changes = np.abs(np.diff(valid_signal))
        mean_change = np.mean(changes)

        if normalize:
            mean_abs_signal = np.mean(np.abs(valid_signal))
            if mean_abs_signal > 1e-10:
                return float(mean_change / mean_abs_signal)

        return float(mean_change)

    def compute_breadth(
        self,
        alpha_signals: np.ndarray,
        threshold: float = 0.0,
    ) -> float:
        """
        Compute alpha breadth (number of independent bets).

        For cross-sectional alpha, this is the number of assets
        with non-zero signal. For time-series, this relates to
        the frequency of signals.

        Args:
            alpha_signals: Alpha signals (can be 2D for cross-sectional)
            threshold: Minimum absolute signal to count as a bet

        Returns:
            Average breadth
        """
        if alpha_signals.ndim == 1:
            # Time-series: count non-zero signals
            valid_signal = alpha_signals[~np.isnan(alpha_signals)]
            return float(np.mean(np.abs(valid_signal) > threshold))
        else:
            # Cross-sectional: count per period
            breadth_per_period = []
            for row in alpha_signals:
                valid = row[~np.isnan(row)]
                if len(valid) > 0:
                    breadth_per_period.append(np.sum(np.abs(valid) > threshold))
            return float(np.mean(breadth_per_period)) if breadth_per_period else 0.0

    def compute_factor_return(
        self,
        alpha_signal: np.ndarray,
        forward_returns: np.ndarray,
        return_lag: int = 1,
    ) -> tuple[float, float, float]:
        """
        Compute factor return and Sharpe ratio.

        Factor return = sum(alpha * returns) / sum(|alpha|)

        Args:
            alpha_signal: Alpha signal values
            forward_returns: Forward returns
            return_lag: Number of periods for forward returns

        Returns:
            Tuple of (annualized return, volatility, Sharpe ratio)
        """
        # Align alpha with forward returns
        if return_lag > 0 and len(alpha_signal) > return_lag:
            lagged_alpha = alpha_signal[:-return_lag]
            fwd_returns = forward_returns[return_lag:]
        else:
            lagged_alpha = alpha_signal
            fwd_returns = forward_returns

        valid_mask = ~(np.isnan(lagged_alpha) | np.isnan(fwd_returns))

        if np.sum(valid_mask) < self.min_observations:
            return 0.0, 0.0, 0.0

        valid_alpha = lagged_alpha[valid_mask]
        valid_returns = fwd_returns[valid_mask]

        # Normalize alpha to unit leverage
        abs_alpha = np.abs(valid_alpha)
        sum_abs_alpha = np.sum(abs_alpha)
        if sum_abs_alpha < 1e-10:
            return 0.0, 0.0, 0.0

        normalized_alpha = valid_alpha / sum_abs_alpha * len(valid_alpha)

        # Factor returns
        factor_returns = normalized_alpha * valid_returns

        # Annualized metrics
        mean_return = np.mean(factor_returns) * self.annualization_factor
        volatility = np.std(factor_returns, ddof=1) * np.sqrt(self.annualization_factor)
        sharpe = mean_return / volatility if volatility > 1e-10 else 0.0

        return float(mean_return), float(volatility), float(sharpe)

    def compute_max_drawdown(
        self,
        alpha_signal: np.ndarray,
        forward_returns: np.ndarray,
        return_lag: int = 1,
    ) -> float:
        """
        Compute maximum drawdown of factor returns.

        Args:
            alpha_signal: Alpha signal values
            forward_returns: Forward returns
            return_lag: Number of periods for forward returns

        Returns:
            Maximum drawdown (as positive percentage)
        """
        # Align alpha with forward returns
        if return_lag > 0 and len(alpha_signal) > return_lag:
            lagged_alpha = alpha_signal[:-return_lag]
            fwd_returns = forward_returns[return_lag:]
        else:
            lagged_alpha = alpha_signal
            fwd_returns = forward_returns

        valid_mask = ~(np.isnan(lagged_alpha) | np.isnan(fwd_returns))

        if np.sum(valid_mask) < self.min_observations:
            return 0.0

        valid_alpha = lagged_alpha[valid_mask]
        valid_returns = fwd_returns[valid_mask]

        # Factor returns
        factor_returns = np.sign(valid_alpha) * valid_returns

        # Cumulative returns
        cumulative = np.cumprod(1 + factor_returns)

        # Running maximum
        running_max = np.maximum.accumulate(cumulative)

        # Drawdowns
        drawdowns = (running_max - cumulative) / running_max

        return float(np.max(drawdowns))

    def compute_stability_score(
        self,
        alpha_signal: np.ndarray,
        forward_returns: np.ndarray,
        n_splits: int = 5,
    ) -> float:
        """
        Compute alpha stability score across time periods.

        Measures how consistent the IC is across different periods.
        Higher stability = more reliable alpha.

        Args:
            alpha_signal: Alpha signal values
            forward_returns: Forward returns
            n_splits: Number of splits for stability analysis

        Returns:
            Stability score (0 to 1, higher is better)
        """
        n = len(alpha_signal)
        split_size = n // n_splits

        if split_size < self.min_observations:
            return 0.0

        split_ics = []
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size

            split_alpha = alpha_signal[start_idx:end_idx]
            split_returns = forward_returns[start_idx:end_idx]

            ic_result = self.compute_ic(split_alpha, split_returns, return_lag=1)
            if ic_result.n_observations >= self.min_observations // 2:
                split_ics.append(ic_result.rank_ic)

        if len(split_ics) < 2:
            return 0.0

        # Stability = 1 - normalized_std
        # High stability when all splits have similar IC
        mean_ic = np.mean(split_ics)
        std_ic = np.std(split_ics, ddof=1)

        if abs(mean_ic) < 1e-10:
            return 0.0

        # Coefficient of variation (normalized std)
        cv = std_ic / abs(mean_ic)

        # Convert to stability score (0 to 1)
        stability = 1.0 / (1.0 + cv)

        return float(stability)

    def compute_full_report(
        self,
        alpha_name: str,
        alpha_signal: np.ndarray,
        forward_returns: np.ndarray,
        compute_decay: bool = True,
        max_horizon: int = 20,
    ) -> AlphaMetricsReport:
        """
        Compute comprehensive alpha metrics report.

        Args:
            alpha_name: Name of the alpha
            alpha_signal: Alpha signal values
            forward_returns: Forward returns
            compute_decay: Whether to compute IC decay
            max_horizon: Maximum horizon for IC decay

        Returns:
            AlphaMetricsReport with all metrics
        """
        logger.info(f"Computing full metrics report for alpha: {alpha_name}")

        # IC
        ic_result = self.compute_ic(alpha_signal, forward_returns)

        # IR
        ir_result = self.compute_ir(alpha_signal, forward_returns)

        # IC Decay
        ic_decay = None
        if compute_decay:
            ic_decay = self.compute_ic_decay(alpha_signal, forward_returns, max_horizon)

        # Turnover
        turnover = self.compute_turnover(alpha_signal)

        # Breadth
        breadth = self.compute_breadth(alpha_signal)

        # Factor return metrics
        factor_return, factor_volatility, sharpe_ratio = self.compute_factor_return(
            alpha_signal, forward_returns
        )

        # Max drawdown
        max_drawdown = self.compute_max_drawdown(alpha_signal, forward_returns)

        # Stability
        stability_score = self.compute_stability_score(alpha_signal, forward_returns)

        return AlphaMetricsReport(
            alpha_name=alpha_name,
            ic_result=ic_result,
            ir_result=ir_result,
            ic_decay=ic_decay,
            turnover=turnover,
            breadth=breadth,
            factor_return=factor_return,
            factor_volatility=factor_volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            stability_score=stability_score,
        )


def compute_rank_ic(
    alpha_signal: np.ndarray,
    forward_returns: np.ndarray,
    return_lag: int = 1,
) -> float:
    """
    Convenience function to compute Rank IC.

    Args:
        alpha_signal: Alpha signal values
        forward_returns: Forward returns
        return_lag: Number of periods for forward returns

    Returns:
        Rank IC value
    """
    calculator = AlphaMetricsCalculator()
    result = calculator.compute_ic(alpha_signal, forward_returns, return_lag)
    return result.rank_ic


def compute_information_ratio(
    alpha_signal: np.ndarray,
    forward_returns: np.ndarray,
    window: int = 60,
) -> float:
    """
    Convenience function to compute Information Ratio.

    Args:
        alpha_signal: Alpha signal values
        forward_returns: Forward returns
        window: Rolling window for IC calculation

    Returns:
        Information Ratio value
    """
    calculator = AlphaMetricsCalculator()
    result = calculator.compute_ir(alpha_signal, forward_returns, window)
    return result.ir


# =============================================================================
# P0: Signal Decay Monitoring (Renaissance-Style Auto-Disable)
# =============================================================================


@dataclass
class SignalDecayAlert:
    """
    Alert when signal quality deteriorates.

    Used by SignalDecayMonitor to communicate signal health status
    and provide actionable recommendations.
    """

    status: str  # "healthy", "warning", "decay_detected"
    baseline_ic: float  # Reference IC established during calibration
    current_ic: float  # Recent rolling IC
    decay_pct: float  # Percentage decline from baseline
    ic_zscore: float  # How many std deviations from baseline
    recommendation: str | None = None  # Action recommendation
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "baseline_ic": self.baseline_ic,
            "current_ic": self.current_ic,
            "decay_pct": self.decay_pct,
            "ic_zscore": self.ic_zscore,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp.isoformat(),
        }


class SignalDecayMonitor:
    """
    Monitors signal quality over time and detects decay.

    Implements Renaissance Technologies-style signal quality monitoring:
    - Establishes baseline IC during calibration period
    - Tracks rolling IC and compares to baseline
    - Detects statistically significant decay
    - Provides auto-disable recommendations for deteriorating signals

    This is critical for:
    - Preventing losses from stale/broken signals
    - Early warning of regime changes affecting signal
    - Automated model lifecycle management

    Usage:
        monitor = SignalDecayMonitor(window_size=60, decay_threshold=-0.5)

        # During live trading
        for predictions, returns in trading_loop:
            alert = monitor.update(predictions, returns)

            if alert.status == "decay_detected":
                logger.warning(f"Signal decay: {alert.recommendation}")
                disable_signal()

    Reference:
        - "Advances in Financial Machine Learning" by López de Prado
        - Renaissance Technologies signal monitoring practices (public disclosures)
    """

    # Status constants
    STATUS_HEALTHY = "healthy"
    STATUS_WARNING = "warning"
    STATUS_DECAY = "decay_detected"

    def __init__(
        self,
        window_size: int = 60,
        decay_threshold: float = -0.5,
        warning_threshold: float = -0.3,
        min_samples: int = 20,
        zscore_threshold: float = -2.0,
    ):
        """
        Initialize Signal Decay Monitor.

        Args:
            window_size: Rolling window for IC calculation (bars)
            decay_threshold: Percentage decline to trigger decay alert (-0.5 = 50% decline)
            warning_threshold: Percentage decline to trigger warning (-0.3 = 30% decline)
            min_samples: Minimum samples before monitoring starts
            zscore_threshold: Z-score below which decay is detected
        """
        self.window_size = window_size
        self.decay_threshold = decay_threshold
        self.warning_threshold = warning_threshold
        self.min_samples = min_samples
        self.zscore_threshold = zscore_threshold

        # Internal state
        self._ic_history: list[float] = []
        self._baseline_ic: float | None = None
        self._baseline_std: float | None = None
        self._is_calibrated: bool = False
        self._predictions_buffer: list[np.ndarray] = []
        self._returns_buffer: list[np.ndarray] = []
        self._calculator = AlphaMetricsCalculator(min_observations=min(10, min_samples // 2))

        logger.debug(
            f"SignalDecayMonitor initialized: window={window_size}, "
            f"decay_threshold={decay_threshold}, warning={warning_threshold}"
        )

    def reset(self) -> None:
        """Reset monitor to initial state."""
        self._ic_history.clear()
        self._baseline_ic = None
        self._baseline_std = None
        self._is_calibrated = False
        self._predictions_buffer.clear()
        self._returns_buffer.clear()
        logger.info("SignalDecayMonitor reset")

    def calibrate(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
    ) -> float:
        """
        Calibrate baseline IC from historical data.

        Should be called once during initialization with historical
        in-sample or validation data.

        Args:
            predictions: Historical predictions
            actual_returns: Historical actual returns

        Returns:
            Baseline IC value
        """
        if len(predictions) < self.min_samples:
            logger.warning(
                f"Insufficient data for calibration: {len(predictions)} < {self.min_samples}"
            )
            return 0.0

        # Compute rolling ICs
        rolling_ics = self._calculator.compute_rolling_ic(
            predictions, actual_returns, window=self.window_size
        )
        valid_ics = rolling_ics[~np.isnan(rolling_ics)]

        if len(valid_ics) < 3:
            logger.warning("Insufficient valid ICs for calibration")
            return 0.0

        self._baseline_ic = float(np.mean(valid_ics))
        self._baseline_std = float(np.std(valid_ics, ddof=1))
        self._is_calibrated = True

        # Initialize IC history with calibration data
        self._ic_history = list(valid_ics[-self.window_size:])

        logger.info(
            f"SignalDecayMonitor calibrated: baseline_ic={self._baseline_ic:.4f}, "
            f"baseline_std={self._baseline_std:.4f}"
        )

        return self._baseline_ic

    def update(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
    ) -> SignalDecayAlert:
        """
        Update monitor with new predictions and check for decay.

        Call this periodically (e.g., end of each day) with recent
        predictions and their realized returns.

        Args:
            predictions: Recent predictions
            actual_returns: Realized returns corresponding to predictions

        Returns:
            SignalDecayAlert with current status and recommendations
        """
        # Compute current IC
        ic_result = self._calculator.compute_ic(predictions, actual_returns, return_lag=1)
        current_ic = ic_result.rank_ic

        # Store in history
        self._ic_history.append(current_ic)

        # Keep only recent history
        if len(self._ic_history) > self.window_size * 2:
            self._ic_history = self._ic_history[-self.window_size * 2:]

        # Auto-calibrate if not calibrated
        if not self._is_calibrated:
            if len(self._ic_history) >= self.min_samples:
                baseline_ics = self._ic_history[:self.min_samples]
                self._baseline_ic = float(np.mean(baseline_ics))
                self._baseline_std = float(np.std(baseline_ics, ddof=1))
                self._is_calibrated = True
                logger.info(
                    f"Auto-calibrated: baseline_ic={self._baseline_ic:.4f}"
                )
            else:
                # Not enough data yet
                return SignalDecayAlert(
                    status=self.STATUS_HEALTHY,
                    baseline_ic=0.0,
                    current_ic=current_ic,
                    decay_pct=0.0,
                    ic_zscore=0.0,
                    recommendation="Calibrating - insufficient data",
                )

        # Compute recent rolling IC
        recent_ics = self._ic_history[-self.window_size:]
        recent_ic = float(np.mean(recent_ics)) if recent_ics else current_ic

        # Calculate decay metrics
        baseline_ic = self._baseline_ic or 0.0
        baseline_std = self._baseline_std or 1.0

        # Decay percentage (negative means decline)
        if abs(baseline_ic) > 1e-10:
            decay_pct = (recent_ic - baseline_ic) / abs(baseline_ic)
        else:
            decay_pct = 0.0

        # Z-score of current IC relative to baseline
        if baseline_std > 1e-10:
            ic_zscore = (recent_ic - baseline_ic) / baseline_std
        else:
            ic_zscore = 0.0

        # Determine status and recommendation
        status = self.STATUS_HEALTHY
        recommendation = None

        if decay_pct <= self.decay_threshold or ic_zscore <= self.zscore_threshold:
            status = self.STATUS_DECAY
            recommendation = (
                f"CRITICAL: Signal decay detected ({decay_pct:.1%} decline). "
                f"Recommend disabling signal and investigating root cause. "
                f"Z-score: {ic_zscore:.2f}"
            )
            logger.warning(
                f"Signal decay detected: decay={decay_pct:.1%}, zscore={ic_zscore:.2f}"
            )

        elif decay_pct <= self.warning_threshold:
            status = self.STATUS_WARNING
            recommendation = (
                f"WARNING: Signal quality declining ({decay_pct:.1%}). "
                f"Monitor closely and prepare for potential disable. "
                f"Z-score: {ic_zscore:.2f}"
            )
            logger.info(
                f"Signal warning: decay={decay_pct:.1%}, zscore={ic_zscore:.2f}"
            )

        return SignalDecayAlert(
            status=status,
            baseline_ic=baseline_ic,
            current_ic=recent_ic,
            decay_pct=decay_pct,
            ic_zscore=ic_zscore,
            recommendation=recommendation,
        )

    def get_ic_history(self) -> list[float]:
        """Get IC history for analysis."""
        return self._ic_history.copy()

    def get_baseline(self) -> tuple[float, float]:
        """Get baseline IC and std."""
        return (self._baseline_ic or 0.0, self._baseline_std or 0.0)

    def is_calibrated(self) -> bool:
        """Check if monitor is calibrated."""
        return self._is_calibrated

    def get_health_summary(self) -> dict[str, Any]:
        """
        Get summary of signal health.

        Returns:
            Dictionary with health metrics
        """
        recent_ics = self._ic_history[-self.window_size:] if self._ic_history else []

        return {
            "is_calibrated": self._is_calibrated,
            "baseline_ic": self._baseline_ic,
            "baseline_std": self._baseline_std,
            "current_ic": float(np.mean(recent_ics)) if recent_ics else None,
            "ic_trend": self._compute_ic_trend(),
            "n_observations": len(self._ic_history),
        }

    def _compute_ic_trend(self) -> str:
        """Compute recent IC trend."""
        if len(self._ic_history) < 10:
            return "insufficient_data"

        recent = self._ic_history[-10:]
        first_half = np.mean(recent[:5])
        second_half = np.mean(recent[5:])

        diff = second_half - first_half
        if diff > 0.02:
            return "improving"
        elif diff < -0.02:
            return "declining"
        else:
            return "stable"
