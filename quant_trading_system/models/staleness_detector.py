"""
Model Staleness Detection.

P2-M2 Enhancement: Implements rolling IC monitoring with automatic model
quarantine when predictions degrade below acceptable thresholds.

Key concepts:
- Monitor rolling Information Coefficient (IC) of model predictions
- Track prediction drift and accuracy decay over time
- Automatically quarantine models that become stale
- Alert when models need retraining

Benefits:
- Prevents stale models from trading during regime shifts
- Automatic detection of model degradation
- Rolling performance tracking
- Graceful degradation to fallback strategies

Author: AlphaTrade System
Version: 1.0.0
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelHealth(str, Enum):
    """Model health states."""

    HEALTHY = "healthy"  # Model performing well
    WARNING = "warning"  # Performance degrading, monitor closely
    STALE = "stale"  # Model quarantined, needs retraining
    UNKNOWN = "unknown"  # Not enough data to assess


class StalenessReason(str, Enum):
    """Reasons for model staleness."""

    LOW_IC = "low_ic"  # Information coefficient too low
    NEGATIVE_IC = "negative_ic"  # Predictions negatively correlated
    HIGH_IC_VARIANCE = "high_ic_variance"  # Unstable performance
    PREDICTION_DRIFT = "prediction_drift"  # Prediction distribution changed
    RETURN_DRIFT = "return_drift"  # Return distribution changed
    NO_RECENT_DATA = "no_recent_data"  # Missing recent observations
    MANUAL = "manual"  # Manually quarantined


@dataclass
class StalenessConfig:
    """Configuration for staleness detection."""

    # IC thresholds
    min_ic_healthy: float = 0.03  # Minimum IC for healthy status
    min_ic_warning: float = 0.01  # Minimum IC before warning
    min_ic_stale: float = -0.01  # IC below this triggers quarantine

    # Rolling window settings
    ic_lookback: int = 60  # Bars for IC calculation
    ic_smoothing_window: int = 20  # EMA smoothing for IC

    # Variance thresholds
    max_ic_std: float = 0.05  # Max IC standard deviation

    # Drift detection
    drift_lookback: int = 100  # Bars for drift detection
    drift_threshold: float = 0.5  # KS statistic threshold

    # Timing
    min_observations: int = 30  # Min observations before assessment
    stale_cooldown_minutes: int = 60  # Cooldown before un-quarantine

    # Actions
    auto_quarantine: bool = True  # Automatically quarantine stale models
    alert_on_warning: bool = True
    alert_on_stale: bool = True


@dataclass
class ModelHealthReport:
    """Health report for a model."""

    model_id: str
    health: ModelHealth
    reason: StalenessReason | None
    current_ic: float
    rolling_ic_mean: float
    rolling_ic_std: float
    observation_count: int
    last_observation_time: datetime | None
    quarantine_time: datetime | None
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "health": self.health.value,
            "reason": self.reason.value if self.reason else None,
            "current_ic": self.current_ic,
            "rolling_ic_mean": self.rolling_ic_mean,
            "rolling_ic_std": self.rolling_ic_std,
            "observation_count": self.observation_count,
            "last_observation_time": self.last_observation_time.isoformat() if self.last_observation_time else None,
            "quarantine_time": self.quarantine_time.isoformat() if self.quarantine_time else None,
            "recommendation": self.recommendation,
        }


class ICTracker:
    """
    Tracks rolling Information Coefficient for a model.

    IC measures the correlation between predictions and subsequent returns.
    A healthy model should have consistently positive IC.
    """

    def __init__(
        self,
        lookback: int = 60,
        smoothing_window: int = 20,
    ):
        """Initialize IC Tracker.

        Args:
            lookback: Number of observations for IC calculation.
            smoothing_window: EMA window for smoothing IC.
        """
        self.lookback = lookback
        self.smoothing_window = smoothing_window

        # Storage for predictions and returns
        self._predictions: deque[float] = deque(maxlen=lookback)
        self._returns: deque[float] = deque(maxlen=lookback)
        self._timestamps: deque[datetime] = deque(maxlen=lookback)

        # IC history
        self._ic_history: deque[float] = deque(maxlen=lookback)
        self._ic_ema: float | None = None

        self._lock = threading.RLock()

    def add_observation(
        self,
        prediction: float,
        actual_return: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Add a prediction/return observation.

        Args:
            prediction: Model prediction (typically direction or return forecast).
            actual_return: Realized return.
            timestamp: Observation timestamp.
        """
        with self._lock:
            self._predictions.append(prediction)
            self._returns.append(actual_return)
            self._timestamps.append(timestamp or datetime.now(timezone.utc))

            # Update IC if we have enough data
            if len(self._predictions) >= 10:
                ic = self._calculate_ic()
                self._ic_history.append(ic)
                self._update_ema(ic)

    def _calculate_ic(self) -> float:
        """Calculate Information Coefficient."""
        if len(self._predictions) < 10:
            return 0.0

        preds = np.array(self._predictions)
        rets = np.array(self._returns)

        # Handle constant predictions
        if np.std(preds) < 1e-10 or np.std(rets) < 1e-10:
            return 0.0

        # Spearman rank correlation (more robust)
        from scipy.stats import spearmanr
        ic, _ = spearmanr(preds, rets)

        return float(ic) if not np.isnan(ic) else 0.0

    def _update_ema(self, ic: float) -> None:
        """Update exponential moving average of IC."""
        alpha = 2 / (self.smoothing_window + 1)
        if self._ic_ema is None:
            self._ic_ema = ic
        else:
            self._ic_ema = alpha * ic + (1 - alpha) * self._ic_ema

    def get_current_ic(self) -> float:
        """Get most recent IC value."""
        with self._lock:
            if len(self._ic_history) == 0:
                return 0.0
            return self._ic_history[-1]

    def get_rolling_ic(self) -> float:
        """Get EMA-smoothed IC."""
        with self._lock:
            return self._ic_ema or 0.0

    def get_ic_stats(self) -> tuple[float, float]:
        """Get IC mean and standard deviation.

        Returns:
            Tuple of (mean, std).
        """
        with self._lock:
            if len(self._ic_history) < 5:
                return 0.0, 0.0
            return np.mean(self._ic_history), np.std(self._ic_history)

    def get_observation_count(self) -> int:
        """Get number of observations."""
        with self._lock:
            return len(self._predictions)

    def get_last_observation_time(self) -> datetime | None:
        """Get timestamp of last observation."""
        with self._lock:
            return self._timestamps[-1] if self._timestamps else None


class ModelStalenessDetector:
    """
    P2-M2 Enhancement: Model Staleness Detector with Auto-Quarantine.

    Monitors model performance using rolling Information Coefficient (IC)
    and other metrics. Automatically quarantines models that become stale
    to prevent degraded predictions from affecting trading.

    Example:
        >>> detector = ModelStalenessDetector(config)
        >>> detector.register_model("xgboost_v1")
        >>> detector.add_observation("xgboost_v1", prediction, actual_return)
        >>> health = detector.get_health("xgboost_v1")
        >>> if health.health == ModelHealth.STALE:
        ...     use_fallback_strategy()
    """

    def __init__(
        self,
        config: StalenessConfig | None = None,
        on_health_change: Callable[[str, ModelHealth, ModelHealth], None] | None = None,
        on_quarantine: Callable[[str, StalenessReason], None] | None = None,
    ):
        """Initialize Model Staleness Detector.

        Args:
            config: Staleness detection configuration.
            on_health_change: Callback when health status changes.
            on_quarantine: Callback when model is quarantined.
        """
        self.config = config or StalenessConfig()
        self._on_health_change = on_health_change
        self._on_quarantine = on_quarantine

        # Model tracking
        self._trackers: dict[str, ICTracker] = {}
        self._health_states: dict[str, ModelHealth] = {}
        self._quarantine_times: dict[str, datetime] = {}
        self._quarantine_reasons: dict[str, StalenessReason] = {}

        self._lock = threading.RLock()

        logger.info(
            f"ModelStalenessDetector initialized: "
            f"min_ic_healthy={self.config.min_ic_healthy}, "
            f"auto_quarantine={self.config.auto_quarantine}"
        )

    def register_model(self, model_id: str) -> None:
        """Register a model for tracking.

        Args:
            model_id: Unique model identifier.
        """
        with self._lock:
            if model_id not in self._trackers:
                self._trackers[model_id] = ICTracker(
                    lookback=self.config.ic_lookback,
                    smoothing_window=self.config.ic_smoothing_window,
                )
                self._health_states[model_id] = ModelHealth.UNKNOWN
                logger.info(f"Registered model '{model_id}' for staleness tracking")

    def add_observation(
        self,
        model_id: str,
        prediction: float,
        actual_return: float,
        timestamp: datetime | None = None,
    ) -> ModelHealth:
        """Add prediction/return observation and update health.

        Args:
            model_id: Model identifier.
            prediction: Model prediction.
            actual_return: Realized return.
            timestamp: Observation timestamp.

        Returns:
            Current model health status.
        """
        with self._lock:
            if model_id not in self._trackers:
                self.register_model(model_id)

            tracker = self._trackers[model_id]
            tracker.add_observation(prediction, actual_return, timestamp)

            # Update health assessment
            new_health = self._assess_health(model_id)
            old_health = self._health_states[model_id]

            if new_health != old_health:
                self._health_states[model_id] = new_health
                self._handle_health_change(model_id, old_health, new_health)

            return new_health

    def _assess_health(self, model_id: str) -> ModelHealth:
        """Assess model health based on IC and other metrics.

        Args:
            model_id: Model identifier.

        Returns:
            ModelHealth status.
        """
        tracker = self._trackers[model_id]

        # Check if we have enough data
        if tracker.get_observation_count() < self.config.min_observations:
            return ModelHealth.UNKNOWN

        # Check if model is in quarantine cooldown
        if model_id in self._quarantine_times:
            quarantine_time = self._quarantine_times[model_id]
            elapsed = (datetime.now(timezone.utc) - quarantine_time).total_seconds()
            if elapsed < self.config.stale_cooldown_minutes * 60:
                return ModelHealth.STALE

        # Get IC metrics
        current_ic = tracker.get_current_ic()
        rolling_ic = tracker.get_rolling_ic()
        ic_mean, ic_std = tracker.get_ic_stats()

        # Check for stale conditions
        if rolling_ic < self.config.min_ic_stale:
            if self.config.auto_quarantine:
                reason = StalenessReason.NEGATIVE_IC if rolling_ic < 0 else StalenessReason.LOW_IC
                self._quarantine_model(model_id, reason)
            return ModelHealth.STALE

        # Check for warning conditions
        if rolling_ic < self.config.min_ic_warning:
            return ModelHealth.WARNING

        if ic_std > self.config.max_ic_std:
            return ModelHealth.WARNING

        # Check for healthy
        if rolling_ic >= self.config.min_ic_healthy:
            # Clear any previous quarantine
            if model_id in self._quarantine_times:
                del self._quarantine_times[model_id]
                del self._quarantine_reasons[model_id]
                logger.info(f"Model '{model_id}' released from quarantine")
            return ModelHealth.HEALTHY

        return ModelHealth.WARNING

    def _quarantine_model(self, model_id: str, reason: StalenessReason) -> None:
        """Quarantine a model.

        Args:
            model_id: Model identifier.
            reason: Reason for quarantine.
        """
        self._quarantine_times[model_id] = datetime.now(timezone.utc)
        self._quarantine_reasons[model_id] = reason

        logger.warning(f"Model '{model_id}' quarantined: {reason.value}")

        if self._on_quarantine:
            try:
                self._on_quarantine(model_id, reason)
            except Exception as e:
                logger.error(f"Quarantine callback error: {e}")

    def _handle_health_change(
        self,
        model_id: str,
        old_health: ModelHealth,
        new_health: ModelHealth,
    ) -> None:
        """Handle health status change.

        Args:
            model_id: Model identifier.
            old_health: Previous health status.
            new_health: New health status.
        """
        logger.info(f"Model '{model_id}' health: {old_health.value} -> {new_health.value}")

        if self._on_health_change:
            try:
                self._on_health_change(model_id, old_health, new_health)
            except Exception as e:
                logger.error(f"Health change callback error: {e}")

    def get_health(self, model_id: str) -> ModelHealthReport:
        """Get health report for a model.

        Args:
            model_id: Model identifier.

        Returns:
            ModelHealthReport with current status.
        """
        with self._lock:
            if model_id not in self._trackers:
                return ModelHealthReport(
                    model_id=model_id,
                    health=ModelHealth.UNKNOWN,
                    reason=None,
                    current_ic=0.0,
                    rolling_ic_mean=0.0,
                    rolling_ic_std=0.0,
                    observation_count=0,
                    last_observation_time=None,
                    quarantine_time=None,
                    recommendation="Register model for tracking",
                )

            tracker = self._trackers[model_id]
            health = self._health_states[model_id]
            ic_mean, ic_std = tracker.get_ic_stats()

            # Generate recommendation
            recommendation = self._get_recommendation(model_id, health)

            return ModelHealthReport(
                model_id=model_id,
                health=health,
                reason=self._quarantine_reasons.get(model_id),
                current_ic=tracker.get_current_ic(),
                rolling_ic_mean=ic_mean,
                rolling_ic_std=ic_std,
                observation_count=tracker.get_observation_count(),
                last_observation_time=tracker.get_last_observation_time(),
                quarantine_time=self._quarantine_times.get(model_id),
                recommendation=recommendation,
            )

    def _get_recommendation(self, model_id: str, health: ModelHealth) -> str:
        """Get recommendation based on health status."""
        if health == ModelHealth.HEALTHY:
            return "Model performing well. Continue monitoring."
        elif health == ModelHealth.WARNING:
            return "Performance degrading. Consider retraining or investigating regime change."
        elif health == ModelHealth.STALE:
            cooldown_remaining = 0
            if model_id in self._quarantine_times:
                elapsed = (datetime.now(timezone.utc) - self._quarantine_times[model_id]).total_seconds()
                cooldown_remaining = max(0, self.config.stale_cooldown_minutes * 60 - elapsed)
            return f"Model quarantined. Retrain required. Cooldown: {cooldown_remaining/60:.1f} min"
        else:
            return f"Need at least {self.config.min_observations} observations to assess health."

    def is_healthy(self, model_id: str) -> bool:
        """Check if model is healthy (can be used for trading).

        Args:
            model_id: Model identifier.

        Returns:
            True if model is HEALTHY, False otherwise.
        """
        with self._lock:
            return self._health_states.get(model_id) == ModelHealth.HEALTHY

    def is_usable(self, model_id: str) -> bool:
        """Check if model is usable (HEALTHY or WARNING).

        Args:
            model_id: Model identifier.

        Returns:
            True if model can be used, False if STALE or UNKNOWN.
        """
        with self._lock:
            health = self._health_states.get(model_id, ModelHealth.UNKNOWN)
            return health in (ModelHealth.HEALTHY, ModelHealth.WARNING)

    def force_quarantine(self, model_id: str, reason: str = "manual") -> None:
        """Force a model into quarantine.

        Args:
            model_id: Model identifier.
            reason: Reason for manual quarantine.
        """
        with self._lock:
            self._quarantine_model(model_id, StalenessReason.MANUAL)
            self._health_states[model_id] = ModelHealth.STALE

    def release_quarantine(self, model_id: str) -> None:
        """Release a model from quarantine.

        Args:
            model_id: Model identifier.
        """
        with self._lock:
            if model_id in self._quarantine_times:
                del self._quarantine_times[model_id]
                del self._quarantine_reasons[model_id]
                self._health_states[model_id] = ModelHealth.WARNING
                logger.info(f"Model '{model_id}' manually released from quarantine")

    def get_all_health(self) -> dict[str, ModelHealthReport]:
        """Get health reports for all registered models.

        Returns:
            Dictionary of model_id -> ModelHealthReport.
        """
        with self._lock:
            return {
                model_id: self.get_health(model_id)
                for model_id in self._trackers.keys()
            }

    def get_usable_models(self) -> list[str]:
        """Get list of models that are usable.

        Returns:
            List of model IDs that are HEALTHY or WARNING.
        """
        with self._lock:
            return [
                model_id for model_id, health in self._health_states.items()
                if health in (ModelHealth.HEALTHY, ModelHealth.WARNING)
            ]

    def get_stale_models(self) -> list[str]:
        """Get list of stale models.

        Returns:
            List of model IDs that are STALE.
        """
        with self._lock:
            return [
                model_id for model_id, health in self._health_states.items()
                if health == ModelHealth.STALE
            ]


def create_staleness_detector(
    min_ic_healthy: float = 0.03,
    min_ic_warning: float = 0.01,
    auto_quarantine: bool = True,
    cooldown_minutes: int = 60,
    **kwargs: Any,
) -> ModelStalenessDetector:
    """Factory function to create staleness detector.

    Args:
        min_ic_healthy: Minimum IC for healthy status.
        min_ic_warning: Minimum IC before warning.
        auto_quarantine: Enable automatic quarantine.
        cooldown_minutes: Quarantine cooldown period.
        **kwargs: Additional config parameters.

    Returns:
        Configured ModelStalenessDetector instance.
    """
    config = StalenessConfig(
        min_ic_healthy=min_ic_healthy,
        min_ic_warning=min_ic_warning,
        auto_quarantine=auto_quarantine,
        stale_cooldown_minutes=cooldown_minutes,
        **kwargs,
    )
    return ModelStalenessDetector(config)
