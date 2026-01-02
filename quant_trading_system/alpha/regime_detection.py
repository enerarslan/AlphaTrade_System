"""
JPMORGAN FIX: Market Regime Detection Module.

Implements multiple regime detection methodologies for adaptive trading:
- Hidden Markov Model (HMM) based regime detection
- Volatility regime classification
- Trend/Mean-reversion regime identification
- Cross-sectional regime analysis

Regimes help adapt trading strategies to current market conditions.

Author: AlphaTrade System
Version: 1.0.0
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime classifications."""

    # Volatility regimes
    LOW_VOLATILITY = "low_volatility"
    NORMAL_VOLATILITY = "normal_volatility"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"

    # Trend regimes
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    SIDEWAYS = "sideways"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

    # Combined regimes
    BULL_LOW_VOL = "bull_low_vol"  # Ideal for momentum
    BULL_HIGH_VOL = "bull_high_vol"  # Risky momentum
    BEAR_LOW_VOL = "bear_low_vol"  # Grinding bear
    BEAR_HIGH_VOL = "bear_high_vol"  # Crisis/capitulation
    RANGE_BOUND = "range_bound"  # Mean reversion friendly


class RegimeCharacteristic(str, Enum):
    """Characteristics of market regimes."""

    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    RANDOM_WALK = "random_walk"
    MOMENTUM = "momentum"
    VOLATILE = "volatile"
    CALM = "calm"


@dataclass
class RegimeState:
    """Current regime state with probabilities."""

    regime: MarketRegime
    probability: float
    characteristics: list[RegimeCharacteristic]
    timestamp: datetime
    duration_bars: int = 0
    transition_probability: dict[MarketRegime, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regime": self.regime.value,
            "probability": self.probability,
            "characteristics": [c.value for c in self.characteristics],
            "timestamp": self.timestamp.isoformat(),
            "duration_bars": self.duration_bars,
            "transition_probability": {k.value: v for k, v in self.transition_probability.items()},
            "metadata": self.metadata,
        }


@dataclass
class RegimeTransition:
    """Record of a regime transition."""

    from_regime: MarketRegime
    to_regime: MarketRegime
    timestamp: datetime
    transition_probability: float
    duration_in_previous: int
    trigger_features: dict[str, float] = field(default_factory=dict)


class RegimeDetector(ABC):
    """Abstract base class for regime detection."""

    @abstractmethod
    def detect(self, data: pd.DataFrame) -> RegimeState:
        """Detect current market regime.

        Args:
            data: DataFrame with price/volume data.

        Returns:
            Current regime state.
        """
        pass

    @abstractmethod
    def get_regime_history(self) -> list[RegimeState]:
        """Get history of detected regimes."""
        pass


class VolatilityRegimeDetector(RegimeDetector):
    """
    Volatility-based regime detection.

    Classifies market into volatility regimes based on realized
    volatility relative to historical distribution.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        long_term_period: int = 252,
        low_vol_percentile: float = 25.0,
        high_vol_percentile: float = 75.0,
        crisis_percentile: float = 95.0,
    ):
        """Initialize volatility regime detector.

        Args:
            lookback_period: Short-term volatility window.
            long_term_period: Long-term volatility baseline.
            low_vol_percentile: Percentile for low volatility.
            high_vol_percentile: Percentile for high volatility.
            crisis_percentile: Percentile for crisis volatility.
        """
        self.lookback_period = lookback_period
        self.long_term_period = long_term_period
        self.low_vol_percentile = low_vol_percentile
        self.high_vol_percentile = high_vol_percentile
        self.crisis_percentile = crisis_percentile

        self._regime_history: list[RegimeState] = []
        self._volatility_history: list[float] = []
        self._current_regime: MarketRegime | None = None
        self._regime_start_bar: int = 0
        self._bar_count: int = 0

    def detect(self, data: pd.DataFrame) -> RegimeState:
        """Detect volatility regime."""
        if len(data) < self.lookback_period:
            return self._default_regime()

        # Calculate returns
        if "close" in data.columns:
            prices = data["close"]
        elif "Close" in data.columns:
            prices = data["Close"]
        else:
            return self._default_regime()

        returns = prices.pct_change().dropna()

        if len(returns) < self.lookback_period:
            return self._default_regime()

        # Calculate short-term realized volatility (annualized)
        short_term_vol = returns.tail(self.lookback_period).std() * np.sqrt(252)

        # Track volatility history
        self._volatility_history.append(short_term_vol)

        # Need enough history for percentile calculation
        if len(self._volatility_history) < self.long_term_period:
            # Use available history
            vol_history = self._volatility_history
        else:
            vol_history = self._volatility_history[-self.long_term_period:]

        # Calculate percentiles
        low_threshold = np.percentile(vol_history, self.low_vol_percentile)
        high_threshold = np.percentile(vol_history, self.high_vol_percentile)
        crisis_threshold = np.percentile(vol_history, self.crisis_percentile)

        # Classify regime
        if short_term_vol >= crisis_threshold:
            regime = MarketRegime.CRISIS
            probability = min(1.0, short_term_vol / crisis_threshold)
            characteristics = [
                RegimeCharacteristic.VOLATILE,
                RegimeCharacteristic.RANDOM_WALK,
            ]
        elif short_term_vol >= high_threshold:
            regime = MarketRegime.HIGH_VOLATILITY
            probability = (short_term_vol - high_threshold) / (crisis_threshold - high_threshold)
            characteristics = [RegimeCharacteristic.VOLATILE]
        elif short_term_vol <= low_threshold:
            regime = MarketRegime.LOW_VOLATILITY
            probability = 1 - (short_term_vol / low_threshold)
            characteristics = [RegimeCharacteristic.CALM]
        else:
            regime = MarketRegime.NORMAL_VOLATILITY
            probability = 0.5
            characteristics = []

        # Track regime duration
        self._bar_count += 1
        if regime != self._current_regime:
            if self._current_regime is not None:
                # Record transition
                pass
            self._current_regime = regime
            self._regime_start_bar = self._bar_count

        duration = self._bar_count - self._regime_start_bar

        # Create regime state
        state = RegimeState(
            regime=regime,
            probability=probability,
            characteristics=characteristics,
            timestamp=data.index[-1] if isinstance(data.index[-1], datetime) else datetime.now(timezone.utc),
            duration_bars=duration,
            metadata={
                "current_volatility": short_term_vol,
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "crisis_threshold": crisis_threshold,
                "vol_percentile": stats.percentileofscore(vol_history, short_term_vol),
            },
        )

        self._regime_history.append(state)
        return state

    def _default_regime(self) -> RegimeState:
        """Return default regime when not enough data."""
        return RegimeState(
            regime=MarketRegime.NORMAL_VOLATILITY,
            probability=0.5,
            characteristics=[],
            timestamp=datetime.now(timezone.utc),
            metadata={"reason": "insufficient_data"},
        )

    def get_regime_history(self) -> list[RegimeState]:
        """Get regime history."""
        return self._regime_history.copy()


class TrendRegimeDetector(RegimeDetector):
    """
    Trend-based regime detection.

    Classifies market into trend regimes based on moving average
    relationships and momentum indicators.
    """

    def __init__(
        self,
        short_ma_period: int = 20,
        medium_ma_period: int = 50,
        long_ma_period: int = 200,
        atr_period: int = 14,
        trend_strength_threshold: float = 0.02,
    ):
        """Initialize trend regime detector.

        Args:
            short_ma_period: Short moving average period.
            medium_ma_period: Medium moving average period.
            long_ma_period: Long moving average period.
            atr_period: ATR period for normalization.
            trend_strength_threshold: Minimum trend strength.
        """
        self.short_ma_period = short_ma_period
        self.medium_ma_period = medium_ma_period
        self.long_ma_period = long_ma_period
        self.atr_period = atr_period
        self.trend_strength_threshold = trend_strength_threshold

        self._regime_history: list[RegimeState] = []
        self._current_regime: MarketRegime | None = None
        self._regime_start_bar: int = 0
        self._bar_count: int = 0

    def detect(self, data: pd.DataFrame) -> RegimeState:
        """Detect trend regime."""
        required_len = max(self.long_ma_period, self.atr_period) + 10

        if len(data) < required_len:
            return self._default_regime()

        # Get price column
        if "close" in data.columns:
            close = data["close"]
        elif "Close" in data.columns:
            close = data["Close"]
        else:
            return self._default_regime()

        # Calculate moving averages
        short_ma = close.rolling(self.short_ma_period).mean()
        medium_ma = close.rolling(self.medium_ma_period).mean()
        long_ma = close.rolling(self.long_ma_period).mean()

        current_price = close.iloc[-1]
        short_ma_val = short_ma.iloc[-1]
        medium_ma_val = medium_ma.iloc[-1]
        long_ma_val = long_ma.iloc[-1]

        # Calculate trend strength (distance from long MA as % of price)
        trend_strength = (current_price - long_ma_val) / long_ma_val

        # MA alignment score (-1 to 1)
        ma_alignment = 0
        if short_ma_val > medium_ma_val > long_ma_val:
            ma_alignment = 1  # Bullish alignment
        elif short_ma_val < medium_ma_val < long_ma_val:
            ma_alignment = -1  # Bearish alignment
        elif short_ma_val > long_ma_val:
            ma_alignment = 0.5  # Weak bullish
        elif short_ma_val < long_ma_val:
            ma_alignment = -0.5  # Weak bearish

        # Price momentum (rate of change)
        momentum = (close.iloc[-1] - close.iloc[-self.short_ma_period]) / close.iloc[-self.short_ma_period]

        # Classify regime
        if trend_strength > self.trend_strength_threshold * 2 and ma_alignment == 1:
            regime = MarketRegime.STRONG_UPTREND
            probability = min(1.0, abs(trend_strength) / (self.trend_strength_threshold * 4))
            characteristics = [
                RegimeCharacteristic.TRENDING,
                RegimeCharacteristic.MOMENTUM,
            ]
        elif trend_strength > self.trend_strength_threshold and ma_alignment >= 0.5:
            regime = MarketRegime.WEAK_UPTREND
            probability = abs(trend_strength) / (self.trend_strength_threshold * 2)
            characteristics = [RegimeCharacteristic.TRENDING]
        elif trend_strength < -self.trend_strength_threshold * 2 and ma_alignment == -1:
            regime = MarketRegime.STRONG_DOWNTREND
            probability = min(1.0, abs(trend_strength) / (self.trend_strength_threshold * 4))
            characteristics = [
                RegimeCharacteristic.TRENDING,
                RegimeCharacteristic.MOMENTUM,
            ]
        elif trend_strength < -self.trend_strength_threshold and ma_alignment <= -0.5:
            regime = MarketRegime.WEAK_DOWNTREND
            probability = abs(trend_strength) / (self.trend_strength_threshold * 2)
            characteristics = [RegimeCharacteristic.TRENDING]
        else:
            regime = MarketRegime.SIDEWAYS
            probability = 1 - abs(trend_strength) / self.trend_strength_threshold
            characteristics = [RegimeCharacteristic.MEAN_REVERTING]

        # Track regime duration
        self._bar_count += 1
        if regime != self._current_regime:
            self._current_regime = regime
            self._regime_start_bar = self._bar_count

        duration = self._bar_count - self._regime_start_bar

        state = RegimeState(
            regime=regime,
            probability=probability,
            characteristics=characteristics,
            timestamp=data.index[-1] if isinstance(data.index[-1], datetime) else datetime.now(timezone.utc),
            duration_bars=duration,
            metadata={
                "trend_strength": trend_strength,
                "ma_alignment": ma_alignment,
                "momentum": momentum,
                "short_ma": short_ma_val,
                "medium_ma": medium_ma_val,
                "long_ma": long_ma_val,
            },
        )

        self._regime_history.append(state)
        return state

    def _default_regime(self) -> RegimeState:
        """Return default regime."""
        return RegimeState(
            regime=MarketRegime.SIDEWAYS,
            probability=0.5,
            characteristics=[],
            timestamp=datetime.now(timezone.utc),
            metadata={"reason": "insufficient_data"},
        )

    def get_regime_history(self) -> list[RegimeState]:
        """Get regime history."""
        return self._regime_history.copy()


class HMMRegimeDetector(RegimeDetector):
    """
    Hidden Markov Model based regime detection.

    Uses HMM to identify latent market states from observable
    return and volatility characteristics.
    """

    def __init__(
        self,
        n_states: int = 3,
        lookback_period: int = 252,
        features: list[str] | None = None,
        retrain_frequency: int = 20,
    ):
        """Initialize HMM regime detector.

        Args:
            n_states: Number of hidden states.
            lookback_period: Training window.
            features: Features to use for HMM.
            retrain_frequency: Bars between model retraining.
        """
        self.n_states = n_states
        self.lookback_period = lookback_period
        self.features = features or ["returns", "volatility"]
        self.retrain_frequency = retrain_frequency

        self._model = None
        self._regime_history: list[RegimeState] = []
        self._bar_count: int = 0
        self._last_train_bar: int = 0
        self._state_to_regime: dict[int, MarketRegime] = {}

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for HMM."""
        if "close" in data.columns:
            close = data["close"]
        elif "Close" in data.columns:
            close = data["Close"]
        else:
            raise ValueError("No close price column found")

        features = []

        if "returns" in self.features:
            returns = close.pct_change().fillna(0)
            features.append(returns.values)

        if "volatility" in self.features:
            vol = close.pct_change().rolling(20).std().fillna(0)
            features.append(vol.values)

        if "momentum" in self.features:
            momentum = (close - close.shift(20)) / close.shift(20)
            features.append(momentum.fillna(0).values)

        return np.column_stack(features)

    def _fit_hmm(self, X: np.ndarray) -> None:
        """Fit HMM to data."""
        try:
            from hmmlearn import hmm
        except ImportError:
            logger.warning("hmmlearn not installed. Using fallback detection.")
            self._model = None
            return

        # Fit Gaussian HMM
        model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )

        try:
            model.fit(X)
            self._model = model

            # Map states to regimes based on mean characteristics
            state_means = model.means_

            # Assume first feature is returns - sort by mean return
            return_means = state_means[:, 0]
            sorted_states = np.argsort(return_means)

            # Map states: lowest return = bear, middle = neutral, highest = bull
            if self.n_states == 3:
                self._state_to_regime = {
                    sorted_states[0]: MarketRegime.BEAR_HIGH_VOL,
                    sorted_states[1]: MarketRegime.RANGE_BOUND,
                    sorted_states[2]: MarketRegime.BULL_LOW_VOL,
                }
            elif self.n_states == 2:
                self._state_to_regime = {
                    sorted_states[0]: MarketRegime.BEAR_HIGH_VOL,
                    sorted_states[1]: MarketRegime.BULL_LOW_VOL,
                }
            else:
                # Default mapping
                self._state_to_regime = {
                    i: MarketRegime.RANGE_BOUND for i in range(self.n_states)
                }

        except Exception as e:
            logger.warning(f"HMM fitting failed: {e}")
            self._model = None

    def detect(self, data: pd.DataFrame) -> RegimeState:
        """Detect regime using HMM."""
        if len(data) < self.lookback_period:
            return self._default_regime()

        self._bar_count += 1

        # Prepare features
        try:
            X = self._prepare_features(data.tail(self.lookback_period))
        except Exception as e:
            logger.warning(f"Feature preparation failed: {e}")
            return self._default_regime()

        # Retrain model periodically
        if (
            self._model is None
            or self._bar_count - self._last_train_bar >= self.retrain_frequency
        ):
            self._fit_hmm(X)
            self._last_train_bar = self._bar_count

        if self._model is None:
            return self._fallback_detect(data)

        # Predict current state
        try:
            state_probs = self._model.predict_proba(X[-1:])
            current_state = np.argmax(state_probs)
            probability = float(state_probs[0, current_state])

            regime = self._state_to_regime.get(current_state, MarketRegime.RANGE_BOUND)

            # Determine characteristics based on state
            characteristics = []
            if regime in [MarketRegime.BULL_LOW_VOL, MarketRegime.BULL_HIGH_VOL]:
                characteristics.append(RegimeCharacteristic.TRENDING)
                characteristics.append(RegimeCharacteristic.MOMENTUM)
            elif regime in [MarketRegime.BEAR_LOW_VOL, MarketRegime.BEAR_HIGH_VOL]:
                characteristics.append(RegimeCharacteristic.TRENDING)
            else:
                characteristics.append(RegimeCharacteristic.MEAN_REVERTING)

            # Get transition probabilities
            trans_probs = {}
            for i, r in self._state_to_regime.items():
                trans_probs[r] = float(self._model.transmat_[current_state, i])

            state_result = RegimeState(
                regime=regime,
                probability=probability,
                characteristics=characteristics,
                timestamp=data.index[-1] if isinstance(data.index[-1], datetime) else datetime.now(timezone.utc),
                duration_bars=1,  # HMM doesn't track duration
                transition_probability=trans_probs,
                metadata={
                    "hmm_state": int(current_state),
                    "state_probabilities": state_probs[0].tolist(),
                },
            )

            self._regime_history.append(state_result)
            return state_result

        except Exception as e:
            logger.warning(f"HMM prediction failed: {e}")
            return self._fallback_detect(data)

    def _fallback_detect(self, data: pd.DataFrame) -> RegimeState:
        """Fallback detection when HMM fails."""
        # Use simple volatility-based detection
        vol_detector = VolatilityRegimeDetector()
        return vol_detector.detect(data)

    def _default_regime(self) -> RegimeState:
        """Return default regime."""
        return RegimeState(
            regime=MarketRegime.RANGE_BOUND,
            probability=0.5,
            characteristics=[],
            timestamp=datetime.now(timezone.utc),
            metadata={"reason": "insufficient_data"},
        )

    def get_regime_history(self) -> list[RegimeState]:
        """Get regime history."""
        return self._regime_history.copy()


class CompositeRegimeDetector(RegimeDetector):
    """
    Composite regime detector combining multiple methods.

    Aggregates signals from volatility, trend, and HMM detectors
    to produce a robust regime classification.
    """

    def __init__(
        self,
        use_volatility: bool = True,
        use_trend: bool = True,
        use_hmm: bool = False,  # Disabled by default (requires hmmlearn)
        volatility_weight: float = 0.4,
        trend_weight: float = 0.4,
        hmm_weight: float = 0.2,
    ):
        """Initialize composite detector.

        Args:
            use_volatility: Include volatility regime.
            use_trend: Include trend regime.
            use_hmm: Include HMM regime.
            volatility_weight: Weight for volatility signal.
            trend_weight: Weight for trend signal.
            hmm_weight: Weight for HMM signal.
        """
        self.use_volatility = use_volatility
        self.use_trend = use_trend
        self.use_hmm = use_hmm

        # Normalize weights
        total_weight = 0
        if use_volatility:
            total_weight += volatility_weight
        if use_trend:
            total_weight += trend_weight
        if use_hmm:
            total_weight += hmm_weight

        self.volatility_weight = volatility_weight / total_weight if use_volatility else 0
        self.trend_weight = trend_weight / total_weight if use_trend else 0
        self.hmm_weight = hmm_weight / total_weight if use_hmm else 0

        # Initialize detectors
        self.vol_detector = VolatilityRegimeDetector() if use_volatility else None
        self.trend_detector = TrendRegimeDetector() if use_trend else None
        self.hmm_detector = HMMRegimeDetector() if use_hmm else None

        self._regime_history: list[RegimeState] = []
        self._current_regime: MarketRegime | None = None
        self._regime_start_bar: int = 0
        self._bar_count: int = 0

    def _combine_regimes(
        self,
        vol_state: RegimeState | None,
        trend_state: RegimeState | None,
        hmm_state: RegimeState | None,
    ) -> MarketRegime:
        """Combine regime signals into unified classification."""
        # Map individual regimes to combined regime
        is_high_vol = vol_state and vol_state.regime in [
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.CRISIS,
        ]
        is_low_vol = vol_state and vol_state.regime == MarketRegime.LOW_VOLATILITY

        is_bullish = trend_state and trend_state.regime in [
            MarketRegime.STRONG_UPTREND,
            MarketRegime.WEAK_UPTREND,
        ]
        is_bearish = trend_state and trend_state.regime in [
            MarketRegime.STRONG_DOWNTREND,
            MarketRegime.WEAK_DOWNTREND,
        ]
        is_sideways = trend_state and trend_state.regime == MarketRegime.SIDEWAYS

        # Combine signals
        if is_bullish and is_low_vol:
            return MarketRegime.BULL_LOW_VOL
        elif is_bullish and is_high_vol:
            return MarketRegime.BULL_HIGH_VOL
        elif is_bearish and is_low_vol:
            return MarketRegime.BEAR_LOW_VOL
        elif is_bearish and is_high_vol:
            return MarketRegime.BEAR_HIGH_VOL
        elif is_sideways or (not is_bullish and not is_bearish):
            return MarketRegime.RANGE_BOUND
        elif is_bullish:
            return MarketRegime.WEAK_UPTREND
        elif is_bearish:
            return MarketRegime.WEAK_DOWNTREND
        else:
            return MarketRegime.RANGE_BOUND

    def detect(self, data: pd.DataFrame) -> RegimeState:
        """Detect composite regime."""
        vol_state = None
        trend_state = None
        hmm_state = None

        if self.vol_detector:
            vol_state = self.vol_detector.detect(data)
        if self.trend_detector:
            trend_state = self.trend_detector.detect(data)
        if self.hmm_detector:
            hmm_state = self.hmm_detector.detect(data)

        # Combine regimes
        combined_regime = self._combine_regimes(vol_state, trend_state, hmm_state)

        # Calculate weighted probability
        probability = 0.0
        if vol_state:
            probability += vol_state.probability * self.volatility_weight
        if trend_state:
            probability += trend_state.probability * self.trend_weight
        if hmm_state:
            probability += hmm_state.probability * self.hmm_weight

        # Combine characteristics
        characteristics = set()
        if vol_state:
            characteristics.update(vol_state.characteristics)
        if trend_state:
            characteristics.update(trend_state.characteristics)
        if hmm_state:
            characteristics.update(hmm_state.characteristics)

        # Track regime duration
        self._bar_count += 1
        if combined_regime != self._current_regime:
            self._current_regime = combined_regime
            self._regime_start_bar = self._bar_count

        duration = self._bar_count - self._regime_start_bar

        state = RegimeState(
            regime=combined_regime,
            probability=probability,
            characteristics=list(characteristics),
            timestamp=data.index[-1] if isinstance(data.index[-1], datetime) else datetime.now(timezone.utc),
            duration_bars=duration,
            metadata={
                "volatility_regime": vol_state.regime.value if vol_state else None,
                "trend_regime": trend_state.regime.value if trend_state else None,
                "hmm_regime": hmm_state.regime.value if hmm_state else None,
                "component_probabilities": {
                    "volatility": vol_state.probability if vol_state else None,
                    "trend": trend_state.probability if trend_state else None,
                    "hmm": hmm_state.probability if hmm_state else None,
                },
            },
        )

        self._regime_history.append(state)
        return state

    def get_regime_history(self) -> list[RegimeState]:
        """Get regime history."""
        return self._regime_history.copy()


class RegimeAdaptiveController:
    """
    Controller that adapts trading parameters based on regime.

    Provides regime-specific recommendations for:
    - Position sizing
    - Risk limits
    - Strategy selection
    - Execution parameters
    """

    def __init__(self, detector: RegimeDetector | None = None):
        """Initialize controller.

        Args:
            detector: Regime detector to use.
        """
        self.detector = detector or CompositeRegimeDetector()

        # Default regime-specific parameters
        self.regime_params: dict[MarketRegime, dict[str, Any]] = {
            MarketRegime.BULL_LOW_VOL: {
                "position_size_multiplier": 1.2,
                "stop_loss_multiplier": 1.0,
                "take_profit_multiplier": 1.5,
                "preferred_strategies": ["momentum", "trend_following"],
                "risk_budget_pct": 0.02,
            },
            MarketRegime.BULL_HIGH_VOL: {
                "position_size_multiplier": 0.8,
                "stop_loss_multiplier": 1.5,
                "take_profit_multiplier": 2.0,
                "preferred_strategies": ["momentum"],
                "risk_budget_pct": 0.015,
            },
            MarketRegime.BEAR_LOW_VOL: {
                "position_size_multiplier": 0.6,
                "stop_loss_multiplier": 1.2,
                "take_profit_multiplier": 1.0,
                "preferred_strategies": ["mean_reversion", "defensive"],
                "risk_budget_pct": 0.01,
            },
            MarketRegime.BEAR_HIGH_VOL: {
                "position_size_multiplier": 0.4,
                "stop_loss_multiplier": 2.0,
                "take_profit_multiplier": 1.5,
                "preferred_strategies": ["defensive", "hedging"],
                "risk_budget_pct": 0.005,
            },
            MarketRegime.RANGE_BOUND: {
                "position_size_multiplier": 1.0,
                "stop_loss_multiplier": 1.0,
                "take_profit_multiplier": 1.0,
                "preferred_strategies": ["mean_reversion", "pairs"],
                "risk_budget_pct": 0.015,
            },
        }

    def get_current_regime(self, data: pd.DataFrame) -> RegimeState:
        """Get current market regime."""
        return self.detector.detect(data)

    def get_regime_parameters(self, regime: MarketRegime) -> dict[str, Any]:
        """Get parameters for a specific regime."""
        return self.regime_params.get(
            regime,
            self.regime_params[MarketRegime.RANGE_BOUND],
        )

    def adapt_position_size(
        self,
        base_size: float,
        regime: MarketRegime,
    ) -> float:
        """Adapt position size based on regime."""
        params = self.get_regime_parameters(regime)
        return base_size * params["position_size_multiplier"]

    def adapt_stop_loss(
        self,
        base_stop: float,
        regime: MarketRegime,
    ) -> float:
        """Adapt stop loss based on regime."""
        params = self.get_regime_parameters(regime)
        return base_stop * params["stop_loss_multiplier"]

    def get_recommended_strategies(self, regime: MarketRegime) -> list[str]:
        """Get recommended strategies for regime."""
        params = self.get_regime_parameters(regime)
        return params["preferred_strategies"]

    def should_reduce_exposure(self, regime: MarketRegime) -> bool:
        """Check if exposure should be reduced."""
        return regime in [
            MarketRegime.BEAR_HIGH_VOL,
            MarketRegime.CRISIS,
        ]

    def should_increase_hedging(self, regime: MarketRegime) -> bool:
        """Check if hedging should be increased."""
        return regime in [
            MarketRegime.BEAR_HIGH_VOL,
            MarketRegime.BEAR_LOW_VOL,
            MarketRegime.CRISIS,
            MarketRegime.HIGH_VOLATILITY,
        ]


def create_regime_detector(
    method: str = "composite",
    **kwargs: Any,
) -> RegimeDetector:
    """Factory function to create regime detector.

    Args:
        method: Detection method ("volatility", "trend", "hmm", "composite").
        **kwargs: Additional parameters for detector.

    Returns:
        Configured regime detector.
    """
    if method == "volatility":
        return VolatilityRegimeDetector(**kwargs)
    elif method == "trend":
        return TrendRegimeDetector(**kwargs)
    elif method == "hmm":
        return HMMRegimeDetector(**kwargs)
    elif method == "composite":
        return CompositeRegimeDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detection method: {method}")
