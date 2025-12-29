"""
Alpha factor base class module.

Provides the abstract interface and common functionality for all alpha factors.
Alpha factors transform features into trading signals with values in [-1, 1].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import polars as pl


class AlphaType(str, Enum):
    """Alpha factor type enumeration."""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VALUE = "value"
    QUALITY = "quality"
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"
    ML_BASED = "ml_based"
    COMPOSITE = "composite"


class AlphaHorizon(str, Enum):
    """Alpha forecast horizon enumeration."""

    SHORT = "short"  # 1-4 bars
    MEDIUM = "medium"  # 4-20 bars
    LONG = "long"  # 20+ bars


@dataclass
class AlphaSignal:
    """Container for an alpha signal."""

    alpha_id: UUID
    alpha_name: str
    symbol: str
    timestamp: datetime
    value: float  # Signal value in [-1, 1]
    confidence: float  # Signal confidence in [0, 1]
    horizon: int  # Forecast horizon in bars
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def direction(self) -> str:
        """Get signal direction."""
        if self.value > 0.1:
            return "LONG"
        elif self.value < -0.1:
            return "SHORT"
        return "FLAT"

    @property
    def strength(self) -> str:
        """Get signal strength category."""
        abs_val = abs(self.value)
        if abs_val > 0.7:
            return "STRONG"
        elif abs_val > 0.3:
            return "MODERATE"
        return "WEAK"

    def is_actionable(self, min_value: float = 0.3, min_confidence: float = 0.5) -> bool:
        """Check if signal meets thresholds for action."""
        return abs(self.value) >= min_value and self.confidence >= min_confidence


@dataclass
class AlphaMetrics:
    """Performance metrics for an alpha factor."""

    alpha_name: str
    period_start: datetime
    period_end: datetime

    # Return metrics
    information_coefficient: float = 0.0  # IC: rank correlation with returns
    information_ratio: float = 0.0  # IR: IC / std(IC)
    mean_return: float = 0.0
    sharpe_ratio: float = 0.0

    # Signal metrics
    hit_rate: float = 0.0  # % of correct direction predictions
    avg_signal_value: float = 0.0
    signal_autocorrelation: float = 0.0
    turnover: float = 0.0  # Average daily signal change

    # Decay metrics
    decay_half_life: int = 0  # Bars until IC drops by half
    optimal_horizon: int = 0  # Horizon with highest IC

    # Risk metrics
    max_drawdown: float = 0.0
    tail_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alpha_name": self.alpha_name,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "ic": self.information_coefficient,
            "ir": self.information_ratio,
            "mean_return": self.mean_return,
            "sharpe": self.sharpe_ratio,
            "hit_rate": self.hit_rate,
            "turnover": self.turnover,
            "decay_half_life": self.decay_half_life,
            "optimal_horizon": self.optimal_horizon,
            "max_drawdown": self.max_drawdown,
        }


class AlphaFactor(ABC):
    """
    Abstract base class for alpha factors.

    Alpha factors transform market data and features into trading signals
    with values in the range [-1, 1], where:
    - Positive values indicate long bias
    - Negative values indicate short bias
    - Values near zero indicate no signal

    Subclasses must implement:
    - compute(): Generate raw alpha values
    - get_params(): Return factor parameters
    """

    def __init__(
        self,
        name: str,
        alpha_type: AlphaType,
        horizon: AlphaHorizon = AlphaHorizon.MEDIUM,
        lookback: int = 20,
        decay: float = 1.0,
    ):
        """
        Initialize alpha factor.

        Args:
            name: Unique identifier for the alpha
            alpha_type: Type of alpha factor
            horizon: Forecast horizon category
            lookback: Number of bars to look back
            decay: Exponential decay factor for signal weighting
        """
        self.name = name
        self.alpha_type = alpha_type
        self.horizon = horizon
        self.lookback = lookback
        self.decay = decay
        self._id = uuid4()
        self._is_fitted = False
        self._metadata: dict[str, Any] = {}

    @abstractmethod
    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """
        Compute raw alpha values.

        Args:
            df: DataFrame with OHLCV data
            features: Optional precomputed features

        Returns:
            Array of alpha values (not necessarily in [-1, 1] yet)
        """
        pass

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """
        Get alpha factor parameters.

        Returns:
            Dictionary of parameter names to values
        """
        pass

    def generate_signals(
        self,
        df: pl.DataFrame,
        symbol: str,
        features: dict[str, np.ndarray] | None = None,
        normalize: bool = True,
        clip: bool = True,
    ) -> list[AlphaSignal]:
        """
        Generate trading signals from alpha values.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol identifier
            features: Optional precomputed features
            normalize: Whether to normalize signals
            clip: Whether to clip to [-1, 1]

        Returns:
            List of AlphaSignal objects
        """
        # Compute raw alpha values
        raw_alpha = self.compute(df, features)

        # Normalize if requested
        if normalize:
            raw_alpha = self._normalize(raw_alpha)

        # Clip to [-1, 1] if requested
        if clip:
            raw_alpha = np.clip(raw_alpha, -1, 1)

        # Compute confidence
        confidence = self._compute_confidence(raw_alpha)

        # Get timestamps
        timestamps = df["timestamp"].to_list() if "timestamp" in df.columns else [
            datetime.utcnow() for _ in range(len(raw_alpha))
        ]

        # Create signals
        signals = []
        horizon_bars = self._horizon_to_bars()

        for i, (val, conf, ts) in enumerate(zip(raw_alpha, confidence, timestamps)):
            if np.isnan(val) or np.isnan(conf):
                continue

            signal = AlphaSignal(
                alpha_id=self._id,
                alpha_name=self.name,
                symbol=symbol,
                timestamp=ts if isinstance(ts, datetime) else datetime.utcnow(),
                value=float(val),
                confidence=float(conf),
                horizon=horizon_bars,
                metadata={"index": i},
            )
            signals.append(signal)

        return signals

    def _normalize(self, values: np.ndarray, window: int = 60) -> np.ndarray:
        """Normalize values using rolling z-score."""
        result = np.full_like(values, np.nan)
        n = len(values)

        for i in range(window - 1, n):
            window_data = values[i - window + 1 : i + 1]
            valid_data = window_data[~np.isnan(window_data)]

            if len(valid_data) >= 10:
                mean = np.mean(valid_data)
                std = np.std(valid_data, ddof=1)
                if std > 0:
                    result[i] = (values[i] - mean) / std

        return result

    def _compute_confidence(self, values: np.ndarray, window: int = 20) -> np.ndarray:
        """
        Compute confidence based on signal consistency.

        Higher confidence when signal has been consistent over recent history.
        """
        confidence = np.full_like(values, np.nan)
        n = len(values)

        for i in range(window - 1, n):
            window_data = values[i - window + 1 : i + 1]
            valid_data = window_data[~np.isnan(window_data)]

            if len(valid_data) >= 5:
                # Confidence based on sign consistency
                current_sign = np.sign(values[i]) if not np.isnan(values[i]) else 0
                same_sign = np.sum(np.sign(valid_data) == current_sign)
                sign_consistency = same_sign / len(valid_data)

                # Confidence based on magnitude relative to history
                abs_val = abs(values[i]) if not np.isnan(values[i]) else 0
                max_abs = np.max(np.abs(valid_data))
                magnitude_conf = abs_val / max_abs if max_abs > 0 else 0

                # Combined confidence
                confidence[i] = 0.5 * sign_consistency + 0.5 * min(magnitude_conf, 1.0)

        return confidence

    def _horizon_to_bars(self) -> int:
        """Convert horizon enum to number of bars."""
        if self.horizon == AlphaHorizon.SHORT:
            return 4
        elif self.horizon == AlphaHorizon.MEDIUM:
            return 12
        else:  # LONG
            return 40

    def fit(
        self,
        df: pl.DataFrame,
        target: np.ndarray | None = None,
    ) -> "AlphaFactor":
        """
        Fit the alpha factor (for ML-based alphas).

        Default implementation does nothing.
        Override in subclasses that need fitting.
        """
        self._is_fitted = True
        return self

    def evaluate(
        self,
        df: pl.DataFrame,
        forward_returns: np.ndarray,
        horizons: list[int] | None = None,
    ) -> AlphaMetrics:
        """
        Evaluate alpha performance.

        Args:
            df: DataFrame with OHLCV data
            forward_returns: Array of forward returns
            horizons: List of horizons to evaluate

        Returns:
            AlphaMetrics with performance statistics
        """
        from scipy import stats

        alpha_values = self.compute(df)
        alpha_values = self._normalize(alpha_values)

        # Remove NaNs for evaluation
        valid_mask = ~(np.isnan(alpha_values) | np.isnan(forward_returns))
        valid_alpha = alpha_values[valid_mask]
        valid_returns = forward_returns[valid_mask]

        if len(valid_alpha) < 20:
            return AlphaMetrics(
                alpha_name=self.name,
                period_start=datetime.utcnow(),
                period_end=datetime.utcnow(),
            )

        # Information Coefficient (Spearman rank correlation)
        ic, _ = stats.spearmanr(valid_alpha, valid_returns)
        ic = ic if not np.isnan(ic) else 0.0

        # Rolling IC for IR calculation
        rolling_ics = []
        window = 20
        for i in range(window, len(valid_alpha)):
            window_alpha = valid_alpha[i - window : i]
            window_ret = valid_returns[i - window : i]
            roll_ic, _ = stats.spearmanr(window_alpha, window_ret)
            if not np.isnan(roll_ic):
                rolling_ics.append(roll_ic)

        # Information Ratio
        ir = np.mean(rolling_ics) / np.std(rolling_ics) if rolling_ics and np.std(rolling_ics) > 0 else 0.0

        # Hit rate
        correct_direction = np.sum(np.sign(valid_alpha) == np.sign(valid_returns))
        hit_rate = correct_direction / len(valid_alpha)

        # Signal turnover
        signal_changes = np.abs(np.diff(valid_alpha))
        turnover = np.mean(signal_changes) if len(signal_changes) > 0 else 0.0

        # Mean return (assuming alpha is used as position sizing)
        alpha_returns = valid_alpha[:-1] * valid_returns[1:]
        mean_ret = np.mean(alpha_returns)
        std_ret = np.std(alpha_returns, ddof=1)
        sharpe = mean_ret / std_ret * np.sqrt(252 * 26) if std_ret > 0 else 0.0  # 15-min bars

        # Get timestamps
        timestamps = df["timestamp"].to_list() if "timestamp" in df.columns else []
        period_start = timestamps[0] if timestamps else datetime.utcnow()
        period_end = timestamps[-1] if timestamps else datetime.utcnow()

        return AlphaMetrics(
            alpha_name=self.name,
            period_start=period_start if isinstance(period_start, datetime) else datetime.utcnow(),
            period_end=period_end if isinstance(period_end, datetime) else datetime.utcnow(),
            information_coefficient=ic,
            information_ratio=ir,
            mean_return=mean_ret,
            sharpe_ratio=sharpe,
            hit_rate=hit_rate,
            turnover=turnover,
        )

    @property
    def is_fitted(self) -> bool:
        """Check if alpha is fitted."""
        return self._is_fitted

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.alpha_type.value})"


class CompositeAlpha(AlphaFactor):
    """
    Composite alpha that combines multiple alpha factors.

    Supports various combination methods:
    - Equal weight
    - IC-weighted
    - Custom weights
    """

    def __init__(
        self,
        name: str,
        alphas: list[AlphaFactor],
        weights: list[float] | None = None,
        combination_method: str = "equal",
    ):
        """
        Initialize composite alpha.

        Args:
            name: Unique identifier
            alphas: List of alpha factors to combine
            weights: Optional custom weights (must sum to 1)
            combination_method: 'equal', 'ic_weighted', or 'custom'
        """
        super().__init__(
            name=name,
            alpha_type=AlphaType.COMPOSITE,
            horizon=AlphaHorizon.MEDIUM,
        )
        self.alphas = alphas
        self.combination_method = combination_method

        if weights is not None:
            self.weights = np.array(weights)
        else:
            self.weights = np.ones(len(alphas)) / len(alphas)

        # Will be updated by IC weighting
        self._dynamic_weights = self.weights.copy()

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute combined alpha values."""
        n = len(df)
        combined = np.zeros(n)

        for i, alpha in enumerate(self.alphas):
            alpha_values = alpha.compute(df, features)

            # Normalize individual alpha
            alpha_values = self._normalize(alpha_values)

            # Handle NaNs
            alpha_values = np.nan_to_num(alpha_values, nan=0.0)

            # Weight and add
            weight = self._dynamic_weights[i]
            combined += weight * alpha_values

        return combined

    def get_params(self) -> dict[str, Any]:
        """Get composite alpha parameters."""
        return {
            "combination_method": self.combination_method,
            "num_alphas": len(self.alphas),
            "alpha_names": [a.name for a in self.alphas],
            "weights": self.weights.tolist(),
        }

    def update_weights_by_ic(
        self,
        df: pl.DataFrame,
        forward_returns: np.ndarray,
        window: int = 60,
        min_weight: float = 0.05,
    ) -> None:
        """
        Update weights based on recent IC.

        Args:
            df: DataFrame with OHLCV data
            forward_returns: Forward returns for evaluation
            window: Lookback window for IC calculation
            min_weight: Minimum weight per alpha
        """
        from scipy import stats

        ics = []
        for alpha in self.alphas:
            alpha_values = alpha.compute(df)
            alpha_values = self._normalize(alpha_values)

            # Use recent data for IC
            recent_alpha = alpha_values[-window:]
            recent_returns = forward_returns[-window:]

            valid_mask = ~(np.isnan(recent_alpha) | np.isnan(recent_returns))
            if np.sum(valid_mask) > 10:
                ic, _ = stats.spearmanr(recent_alpha[valid_mask], recent_returns[valid_mask])
                ics.append(abs(ic) if not np.isnan(ic) else 0.0)
            else:
                ics.append(0.0)

        # Convert IC to weights
        total_ic = sum(ics)
        if total_ic > 0:
            weights = [ic / total_ic for ic in ics]
        else:
            weights = [1 / len(self.alphas)] * len(self.alphas)

        # Apply minimum weight
        weights = np.maximum(weights, min_weight)
        weights = weights / np.sum(weights)

        self._dynamic_weights = weights


class AlphaRegistry:
    """
    Registry for managing alpha factors.

    Provides:
    - Registration and retrieval of alphas
    - Batch computation
    - Performance tracking
    """

    def __init__(self):
        self._alphas: dict[str, AlphaFactor] = {}
        self._metrics: dict[str, list[AlphaMetrics]] = {}

    def register(self, alpha: AlphaFactor) -> None:
        """Register an alpha factor."""
        if alpha.name in self._alphas:
            raise ValueError(f"Alpha '{alpha.name}' already registered")
        self._alphas[alpha.name] = alpha
        self._metrics[alpha.name] = []

    def unregister(self, name: str) -> None:
        """Unregister an alpha factor."""
        if name in self._alphas:
            del self._alphas[name]
            del self._metrics[name]

    def get(self, name: str) -> AlphaFactor | None:
        """Get alpha by name."""
        return self._alphas.get(name)

    def get_all(self, alpha_type: AlphaType | None = None) -> list[AlphaFactor]:
        """Get all registered alphas, optionally filtered by type."""
        if alpha_type is None:
            return list(self._alphas.values())
        return [a for a in self._alphas.values() if a.alpha_type == alpha_type]

    def compute_all(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute all registered alphas."""
        return {
            name: alpha.compute(df, features) for name, alpha in self._alphas.items()
        }

    def evaluate_all(
        self,
        df: pl.DataFrame,
        forward_returns: np.ndarray,
    ) -> dict[str, AlphaMetrics]:
        """Evaluate all registered alphas."""
        metrics = {}
        for name, alpha in self._alphas.items():
            alpha_metrics = alpha.evaluate(df, forward_returns)
            metrics[name] = alpha_metrics
            self._metrics[name].append(alpha_metrics)
        return metrics

    def get_top_alphas(
        self,
        n: int = 10,
        metric: str = "information_coefficient",
    ) -> list[tuple[str, float]]:
        """Get top N alphas by specified metric."""
        scores = []
        for name, metrics_list in self._metrics.items():
            if metrics_list:
                latest = metrics_list[-1]
                score = getattr(latest, metric, 0.0)
                scores.append((name, score))

        scores.sort(key=lambda x: abs(x[1]), reverse=True)
        return scores[:n]

    @property
    def names(self) -> list[str]:
        """Get list of registered alpha names."""
        return list(self._alphas.keys())

    def __len__(self) -> int:
        return len(self._alphas)

    def __contains__(self, name: str) -> bool:
        return name in self._alphas
