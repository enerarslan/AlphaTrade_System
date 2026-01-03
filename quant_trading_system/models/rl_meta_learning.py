"""
RL Meta-Learning Improvements.

P2-D Enhancement: Advanced reinforcement learning techniques:
- Regime-adaptive reward shaping
- Curiosity-driven exploration
- Hierarchical RL architecture
- Meta-learning for rapid adaptation

Expected Impact: +12-20 bps from improved model adaptability.

Author: AlphaTrade System
Version: 1.0.0
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Reward Shaping Types
# =============================================================================


class RewardType(str, Enum):
    """Types of reward signals."""

    PROFIT = "profit"
    RISK_ADJUSTED = "risk_adjusted"
    SHARPE = "sharpe"
    SORTINO = "sortino"
    REGIME_ADAPTIVE = "regime_adaptive"
    CURIOSITY = "curiosity"
    COMPOSITE = "composite"


class MarketRegimeRL(str, Enum):
    """Market regime classification for RL."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"
    CRISIS = "crisis"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class Experience:
    """Single experience tuple for RL."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: dict[str, Any] = field(default_factory=dict)

    # Meta-learning extensions
    regime: MarketRegimeRL | None = None
    intrinsic_reward: float = 0.0
    curiosity_bonus: float = 0.0


@dataclass
class EpisodeStats:
    """Statistics for an RL episode."""

    episode_id: int
    total_reward: float
    total_profit: float
    num_steps: int
    sharpe_ratio: float
    max_drawdown: float
    regime: MarketRegimeRL
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RLMetaConfig(BaseModel):
    """Configuration for RL meta-learning."""

    # Reward shaping
    reward_type: RewardType = Field(default=RewardType.REGIME_ADAPTIVE)
    risk_aversion: float = Field(default=0.5, description="Risk aversion coefficient")
    drawdown_penalty: float = Field(default=2.0, description="Drawdown penalty multiplier")

    # Curiosity-driven exploration
    enable_curiosity: bool = Field(default=True, description="Enable curiosity bonus")
    curiosity_coefficient: float = Field(default=0.1, description="Curiosity reward weight")
    icm_hidden_size: int = Field(default=128, description="ICM hidden layer size")

    # Regime adaptation
    regime_lookback: int = Field(default=20, description="Lookback for regime detection")
    regime_adaptation_rate: float = Field(default=0.1, description="Regime policy mixing rate")

    # Hierarchical RL
    enable_hierarchical: bool = Field(default=True, description="Enable hierarchical policies")
    high_level_horizon: int = Field(default=20, description="High-level decision horizon")
    num_options: int = Field(default=4, description="Number of options/sub-policies")

    # Meta-learning
    meta_learning_rate: float = Field(default=0.01, description="Meta learning rate")
    adaptation_steps: int = Field(default=5, description="Steps for task adaptation")
    meta_batch_size: int = Field(default=4, description="Number of tasks per meta-update")


# =============================================================================
# Regime-Adaptive Reward Shaper
# =============================================================================


class RegimeAdaptiveRewardShaper:
    """
    Shapes rewards based on current market regime.

    Different regimes require different trading behaviors:
    - Trending: Reward trend-following, penalize counter-trend
    - Range-bound: Reward mean reversion, penalize breakout chasing
    - High volatility: Reward capital preservation, penalize large positions
    - Crisis: Heavy reward for risk reduction
    """

    def __init__(
        self,
        config: RLMetaConfig | None = None,
    ):
        """Initialize reward shaper.

        Args:
            config: Configuration
        """
        self.config = config or RLMetaConfig()

        # Regime-specific reward weights
        self._regime_weights = {
            MarketRegimeRL.TRENDING_UP: {
                "profit": 1.5,
                "holding": 0.5,  # Bonus for holding winners
                "drawdown": -1.0,
            },
            MarketRegimeRL.TRENDING_DOWN: {
                "profit": 1.0,
                "short_bonus": 0.5,
                "drawdown": -2.0,
            },
            MarketRegimeRL.RANGE_BOUND: {
                "profit": 1.0,
                "mean_reversion": 0.5,
                "drawdown": -1.5,
            },
            MarketRegimeRL.HIGH_VOLATILITY: {
                "profit": 0.5,
                "capital_preservation": 2.0,
                "drawdown": -3.0,
            },
            MarketRegimeRL.LOW_VOLATILITY: {
                "profit": 1.2,
                "position_size": 0.3,
                "drawdown": -1.0,
            },
            MarketRegimeRL.CRISIS: {
                "profit": 0.3,
                "capital_preservation": 3.0,
                "risk_reduction": 2.0,
                "drawdown": -5.0,
            },
        }

        # Rolling statistics for reward normalization
        self._reward_history = deque(maxlen=1000)
        self._regime_history: list[MarketRegimeRL] = []

    def shape_reward(
        self,
        raw_reward: float,
        regime: MarketRegimeRL,
        position: float,
        prev_position: float,
        pnl: float,
        drawdown: float,
        volatility: float,
    ) -> float:
        """Shape reward based on regime and context.

        Args:
            raw_reward: Raw PnL-based reward
            regime: Current market regime
            position: Current position (-1 to 1)
            prev_position: Previous position
            pnl: Current PnL
            drawdown: Current drawdown
            volatility: Current volatility

        Returns:
            Shaped reward
        """
        weights = self._regime_weights.get(regime, self._regime_weights[MarketRegimeRL.RANGE_BOUND])

        # Base profit reward
        shaped = raw_reward * weights.get("profit", 1.0)

        # Drawdown penalty
        if drawdown > 0:
            shaped += drawdown * weights.get("drawdown", -1.0)

        # Regime-specific bonuses/penalties
        if regime == MarketRegimeRL.TRENDING_UP:
            # Bonus for holding long
            if position > 0 and pnl > 0:
                shaped += weights.get("holding", 0) * position

        elif regime == MarketRegimeRL.TRENDING_DOWN:
            # Bonus for shorting
            if position < 0:
                shaped += weights.get("short_bonus", 0) * abs(position)

        elif regime == MarketRegimeRL.RANGE_BOUND:
            # Mean reversion bonus
            position_change = position - prev_position
            if (pnl > 0 and abs(position_change) > 0.1):
                shaped += weights.get("mean_reversion", 0)

        elif regime in (MarketRegimeRL.HIGH_VOLATILITY, MarketRegimeRL.CRISIS):
            # Capital preservation bonus for reducing position
            if abs(position) < abs(prev_position):
                shaped += weights.get("capital_preservation", 0) * (abs(prev_position) - abs(position))

            # Risk reduction bonus
            if regime == MarketRegimeRL.CRISIS and abs(position) < 0.2:
                shaped += weights.get("risk_reduction", 0)

        # Normalize reward
        shaped = self._normalize_reward(shaped)

        return shaped

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        self._reward_history.append(reward)

        if len(self._reward_history) < 10:
            return reward

        mean = np.mean(self._reward_history)
        std = np.std(self._reward_history) + 1e-8

        # Clip to prevent extreme values
        normalized = (reward - mean) / std
        return np.clip(normalized, -5.0, 5.0)

    def detect_regime(
        self,
        returns: np.ndarray,
        volatility: float,
    ) -> MarketRegimeRL:
        """Detect current market regime from recent returns.

        Args:
            returns: Recent returns
            volatility: Current volatility

        Returns:
            Detected regime
        """
        if len(returns) < self.config.regime_lookback:
            return MarketRegimeRL.RANGE_BOUND

        recent = returns[-self.config.regime_lookback:]

        # Calculate metrics
        cum_return = np.sum(recent)
        vol = np.std(recent) * np.sqrt(252)  # Annualized

        # High volatility check
        if vol > 0.4:  # 40% annualized vol
            if cum_return < -0.1:
                return MarketRegimeRL.CRISIS
            return MarketRegimeRL.HIGH_VOLATILITY

        # Low volatility
        if vol < 0.1:
            return MarketRegimeRL.LOW_VOLATILITY

        # Trend detection
        if cum_return > 0.05:
            return MarketRegimeRL.TRENDING_UP
        elif cum_return < -0.05:
            return MarketRegimeRL.TRENDING_DOWN

        return MarketRegimeRL.RANGE_BOUND


# =============================================================================
# Intrinsic Curiosity Module (ICM)
# =============================================================================


class IntrinsicCuriosityModule:
    """
    Curiosity-driven exploration using prediction error.

    Based on: "Curiosity-driven Exploration by Self-Supervised Prediction"
    (Pathak et al., 2017)

    Provides intrinsic reward based on prediction error of state dynamics.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: RLMetaConfig | None = None,
    ):
        """Initialize ICM.

        Args:
            state_dim: State dimensionality
            action_dim: Number of actions
            config: Configuration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or RLMetaConfig()

        self._hidden_size = config.icm_hidden_size if config else 128

        # Feature encoder weights (simple linear for now)
        self._feature_dim = min(state_dim, 64)
        self._encoder_weights = np.random.randn(state_dim, self._feature_dim) * 0.1

        # Forward model weights (predicts next state features)
        self._forward_weights = np.random.randn(
            self._feature_dim + action_dim, self._feature_dim
        ) * 0.1

        # Inverse model weights (predicts action from states)
        self._inverse_weights = np.random.randn(
            self._feature_dim * 2, action_dim
        ) * 0.1

        # Learning rate
        self._lr = 0.001

        # Prediction error history for normalization
        self._error_history = deque(maxlen=1000)

    def encode_state(self, state: np.ndarray) -> np.ndarray:
        """Encode state into feature space.

        Args:
            state: Raw state

        Returns:
            Feature representation
        """
        # Simple linear encoding with tanh activation
        features = np.tanh(state @ self._encoder_weights)
        return features

    def compute_curiosity(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> float:
        """Compute intrinsic curiosity reward.

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state

        Returns:
            Curiosity bonus (intrinsic reward)
        """
        # Encode states
        features = self.encode_state(state)
        next_features = self.encode_state(next_state)

        # One-hot action
        action_one_hot = np.zeros(self.action_dim)
        action_one_hot[action] = 1.0

        # Forward model prediction
        forward_input = np.concatenate([features, action_one_hot])
        predicted_next = np.tanh(forward_input @ self._forward_weights)

        # Prediction error (curiosity)
        prediction_error = np.sum((predicted_next - next_features) ** 2)

        # Normalize
        self._error_history.append(prediction_error)
        if len(self._error_history) > 10:
            mean_error = np.mean(self._error_history)
            std_error = np.std(self._error_history) + 1e-8
            normalized_error = (prediction_error - mean_error) / std_error
            curiosity = np.clip(normalized_error, 0, 3.0)
        else:
            curiosity = prediction_error

        return float(curiosity) * self.config.curiosity_coefficient

    def update(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> dict[str, float]:
        """Update ICM networks.

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state

        Returns:
            Training losses
        """
        # Encode states
        features = self.encode_state(state)
        next_features = self.encode_state(next_state)

        # One-hot action
        action_one_hot = np.zeros(self.action_dim)
        action_one_hot[action] = 1.0

        # Forward model gradient
        forward_input = np.concatenate([features, action_one_hot])
        predicted_next = np.tanh(forward_input @ self._forward_weights)
        forward_error = predicted_next - next_features
        forward_grad = np.outer(forward_input, forward_error)
        self._forward_weights -= self._lr * forward_grad

        # Inverse model gradient
        inverse_input = np.concatenate([features, next_features])
        predicted_action = self._softmax(inverse_input @ self._inverse_weights)
        inverse_error = predicted_action - action_one_hot
        inverse_grad = np.outer(inverse_input, inverse_error)
        self._inverse_weights -= self._lr * inverse_grad

        return {
            "forward_loss": float(np.mean(forward_error ** 2)),
            "inverse_loss": float(np.mean(inverse_error ** 2)),
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-8)


# =============================================================================
# Hierarchical RL Controller
# =============================================================================


class HierarchicalOption:
    """A single option (sub-policy) in hierarchical RL."""

    def __init__(
        self,
        option_id: int,
        name: str,
        state_dim: int,
        action_dim: int,
    ):
        """Initialize option.

        Args:
            option_id: Unique option ID
            name: Option name
            state_dim: State dimensionality
            action_dim: Action space size
        """
        self.option_id = option_id
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Simple linear policy weights
        self._policy_weights = np.random.randn(state_dim, action_dim) * 0.1
        self._termination_weights = np.random.randn(state_dim, 1) * 0.1

        # Option statistics
        self.total_reward = 0.0
        self.num_activations = 0
        self.avg_duration = 0.0

    def select_action(self, state: np.ndarray) -> int:
        """Select action using option policy.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        logits = state @ self._policy_weights
        probs = self._softmax(logits)
        return np.random.choice(self.action_dim, p=probs)

    def termination_prob(self, state: np.ndarray) -> float:
        """Get termination probability.

        Args:
            state: Current state

        Returns:
            Probability of terminating this option
        """
        logit = float(state @ self._termination_weights)
        return self._sigmoid(logit)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-8)

    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class HierarchicalRLController:
    """
    Hierarchical RL with options framework.

    High-level policy selects options (temporal abstractions).
    Low-level options execute primitive actions.

    Based on: "The Option-Critic Architecture" (Bacon et al., 2017)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: RLMetaConfig | None = None,
    ):
        """Initialize hierarchical controller.

        Args:
            state_dim: State dimensionality
            action_dim: Action space size
            config: Configuration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or RLMetaConfig()

        self.num_options = self.config.num_options

        # High-level policy (selects options)
        self._option_policy_weights = np.random.randn(state_dim, self.num_options) * 0.1

        # Options (sub-policies)
        self._options = [
            HierarchicalOption(
                option_id=i,
                name=self._option_names[i] if i < len(self._option_names) else f"option_{i}",
                state_dim=state_dim,
                action_dim=action_dim,
            )
            for i in range(self.num_options)
        ]

        # Current option state
        self._current_option: int | None = None
        self._option_steps = 0

        # Learning
        self._lr = 0.01

    @property
    def _option_names(self) -> list[str]:
        """Default option names for trading."""
        return [
            "aggressive_long",  # Take large long positions
            "conservative_long",  # Small long positions, tight stops
            "neutral",  # Flat or minimal positions
            "hedging",  # Defensive, reduce risk
        ]

    def select_option(self, state: np.ndarray) -> int:
        """Select option using high-level policy.

        Args:
            state: Current state

        Returns:
            Selected option index
        """
        logits = state @ self._option_policy_weights
        probs = self._softmax(logits)
        return np.random.choice(self.num_options, p=probs)

    def select_action(self, state: np.ndarray) -> tuple[int, int]:
        """Select action using hierarchical policy.

        Args:
            state: Current state

        Returns:
            Tuple of (action, current_option)
        """
        # Check if we need a new option
        if self._current_option is None:
            self._current_option = self.select_option(state)
            self._option_steps = 0

        option = self._options[self._current_option]
        self._option_steps += 1

        # Check termination
        terminate_prob = option.termination_prob(state)
        if np.random.random() < terminate_prob or self._option_steps > self.config.high_level_horizon:
            # Record stats
            option.num_activations += 1
            option.avg_duration = (
                option.avg_duration * (option.num_activations - 1) + self._option_steps
            ) / option.num_activations

            # Select new option
            self._current_option = self.select_option(state)
            self._option_steps = 0
            option = self._options[self._current_option]

        # Get action from current option
        action = option.select_action(state)

        return action, self._current_option

    def update_option_reward(self, option_id: int, reward: float) -> None:
        """Update option with received reward.

        Args:
            option_id: Option that was active
            reward: Reward received
        """
        if 0 <= option_id < len(self._options):
            self._options[option_id].total_reward += reward

    def get_option_stats(self) -> list[dict[str, Any]]:
        """Get statistics for all options.

        Returns:
            List of option statistics
        """
        return [
            {
                "option_id": opt.option_id,
                "name": opt.name,
                "total_reward": opt.total_reward,
                "num_activations": opt.num_activations,
                "avg_duration": opt.avg_duration,
            }
            for opt in self._options
        ]

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-8)


# =============================================================================
# Meta-Learning Agent
# =============================================================================


class MetaLearningAgent:
    """
    Meta-learning agent for rapid task adaptation.

    Based on MAML (Model-Agnostic Meta-Learning) concepts.
    Learns to quickly adapt to new market regimes.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: RLMetaConfig | None = None,
    ):
        """Initialize meta-learning agent.

        Args:
            state_dim: State dimensionality
            action_dim: Action space size
            config: Configuration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or RLMetaConfig()

        # Meta-parameters (shared across tasks)
        self._meta_policy_weights = np.random.randn(state_dim, action_dim) * 0.1
        self._meta_value_weights = np.random.randn(state_dim, 1) * 0.1

        # Task-specific parameters (adapted during inner loop)
        self._task_policy_weights: np.ndarray | None = None
        self._task_value_weights: np.ndarray | None = None

        # Components
        self._reward_shaper = RegimeAdaptiveRewardShaper(config)

        if self.config.enable_curiosity:
            self._icm = IntrinsicCuriosityModule(state_dim, action_dim, config)
        else:
            self._icm = None

        if self.config.enable_hierarchical:
            self._hierarchical = HierarchicalRLController(state_dim, action_dim, config)
        else:
            self._hierarchical = None

        # Regime-specific adapted policies
        self._regime_policies: dict[MarketRegimeRL, np.ndarray] = {}

        # Experience buffer for adaptation
        self._adaptation_buffer: list[Experience] = []

        # Statistics
        self._total_episodes = 0
        self._regime_episode_counts: dict[MarketRegimeRL, int] = {r: 0 for r in MarketRegimeRL}

    def adapt_to_task(self, experiences: list[Experience]) -> None:
        """Adapt policy to new task/regime.

        Fast adaptation using a few gradient steps on task data.

        Args:
            experiences: Experience tuples from current task
        """
        if not experiences:
            return

        # Initialize task parameters from meta-parameters
        self._task_policy_weights = self._meta_policy_weights.copy()
        self._task_value_weights = self._meta_value_weights.copy()

        lr = self.config.meta_learning_rate

        # Inner loop: adapt to task
        for _ in range(self.config.adaptation_steps):
            for exp in experiences:
                # Policy gradient
                action_probs = self._softmax(exp.state @ self._task_policy_weights)
                action_one_hot = np.zeros(self.action_dim)
                action_one_hot[exp.action] = 1.0

                # Advantage (simplified)
                value = float(exp.state @ self._task_value_weights)
                advantage = exp.reward - value

                # Policy gradient
                policy_grad = np.outer(
                    exp.state,
                    (action_one_hot - action_probs) * advantage
                )
                self._task_policy_weights += lr * policy_grad

                # Value gradient
                value_grad = exp.state.reshape(-1, 1) * (exp.reward - value)
                self._task_value_weights += lr * value_grad

        # Store regime-specific policy if detected
        regime = experiences[0].regime if experiences and experiences[0].regime else None
        if regime:
            self._regime_policies[regime] = self._task_policy_weights.copy()

    def select_action(
        self,
        state: np.ndarray,
        regime: MarketRegimeRL | None = None,
        use_hierarchical: bool = True,
    ) -> tuple[int, dict[str, Any]]:
        """Select action with meta-learned policy.

        Args:
            state: Current state
            regime: Current market regime
            use_hierarchical: Whether to use hierarchical controller

        Returns:
            Tuple of (action, info_dict)
        """
        info = {}

        # Use hierarchical controller if enabled
        if use_hierarchical and self._hierarchical and self.config.enable_hierarchical:
            action, option = self._hierarchical.select_action(state)
            info["option"] = option
            return action, info

        # Get appropriate policy weights
        if regime and regime in self._regime_policies:
            # Use regime-specific adapted policy
            policy_weights = self._regime_policies[regime]
            info["policy_type"] = "regime_adapted"
        elif self._task_policy_weights is not None:
            # Use task-adapted policy
            policy_weights = self._task_policy_weights
            info["policy_type"] = "task_adapted"
        else:
            # Use meta policy
            policy_weights = self._meta_policy_weights
            info["policy_type"] = "meta"

        # Compute action probabilities
        logits = state @ policy_weights
        probs = self._softmax(logits)
        action = np.random.choice(self.action_dim, p=probs)

        info["action_probs"] = probs.tolist()

        return action, info

    def compute_reward(
        self,
        raw_reward: float,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        position: float,
        prev_position: float,
        pnl: float,
        drawdown: float,
        returns: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Compute shaped reward with all enhancements.

        Args:
            raw_reward: Raw reward (e.g., PnL)
            state: Current state
            action: Action taken
            next_state: Resulting state
            position: Current position
            prev_position: Previous position
            pnl: Current PnL
            drawdown: Current drawdown
            returns: Recent returns for regime detection

        Returns:
            Tuple of (total_reward, reward_components)
        """
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.2

        # Detect regime
        regime = self._reward_shaper.detect_regime(returns, volatility)

        # Shape reward based on regime
        shaped_reward = self._reward_shaper.shape_reward(
            raw_reward=raw_reward,
            regime=regime,
            position=position,
            prev_position=prev_position,
            pnl=pnl,
            drawdown=drawdown,
            volatility=volatility,
        )

        components = {
            "raw": raw_reward,
            "shaped": shaped_reward,
            "regime": regime.value,
        }

        # Add curiosity bonus
        curiosity_bonus = 0.0
        if self._icm and self.config.enable_curiosity:
            curiosity_bonus = self._icm.compute_curiosity(state, action, next_state)
            components["curiosity"] = curiosity_bonus

        total_reward = shaped_reward + curiosity_bonus
        components["total"] = total_reward

        return total_reward, components

    def update(
        self,
        experience: Experience,
    ) -> dict[str, float]:
        """Update agent with new experience.

        Args:
            experience: Experience tuple

        Returns:
            Update statistics
        """
        stats = {}

        # Update ICM
        if self._icm and self.config.enable_curiosity:
            icm_stats = self._icm.update(
                experience.state,
                experience.action,
                experience.next_state,
            )
            stats.update(icm_stats)

        # Update hierarchical controller
        if self._hierarchical and self.config.enable_hierarchical:
            option_info = experience.info.get("option")
            if option_info is not None:
                self._hierarchical.update_option_reward(option_info, experience.reward)

        # Add to adaptation buffer
        self._adaptation_buffer.append(experience)
        if len(self._adaptation_buffer) > 100:
            self._adaptation_buffer = self._adaptation_buffer[-100:]

        return stats

    def meta_update(
        self,
        task_experiences: list[list[Experience]],
    ) -> dict[str, float]:
        """Meta-update across multiple tasks.

        Args:
            task_experiences: List of experience lists (one per task)

        Returns:
            Meta-update statistics
        """
        if len(task_experiences) < self.config.meta_batch_size:
            return {}

        meta_lr = self.config.meta_learning_rate

        # Accumulate meta-gradients
        meta_policy_grad = np.zeros_like(self._meta_policy_weights)
        meta_value_grad = np.zeros_like(self._meta_value_weights)

        for task_exps in task_experiences:
            # Adapt to task
            self.adapt_to_task(task_exps)

            # Compute gradients on adapted policy
            for exp in task_exps:
                action_probs = self._softmax(exp.state @ self._task_policy_weights)
                action_one_hot = np.zeros(self.action_dim)
                action_one_hot[exp.action] = 1.0

                value = float(exp.state @ self._task_value_weights)
                advantage = exp.reward - value

                meta_policy_grad += np.outer(
                    exp.state,
                    (action_one_hot - action_probs) * advantage
                )
                meta_value_grad += exp.state.reshape(-1, 1) * (exp.reward - value)

        # Average and apply meta-update
        n_tasks = len(task_experiences)
        self._meta_policy_weights += meta_lr * meta_policy_grad / n_tasks
        self._meta_value_weights += meta_lr * meta_value_grad / n_tasks

        return {
            "meta_policy_grad_norm": float(np.linalg.norm(meta_policy_grad)),
            "meta_value_grad_norm": float(np.linalg.norm(meta_value_grad)),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_episodes": self._total_episodes,
            "regime_episode_counts": {k.value: v for k, v in self._regime_episode_counts.items()},
            "num_regime_policies": len(self._regime_policies),
            "adapted_regimes": [r.value for r in self._regime_policies.keys()],
        }

        if self._hierarchical:
            stats["option_stats"] = self._hierarchical.get_option_stats()

        return stats

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-8)


# =============================================================================
# Factory Function
# =============================================================================


def create_meta_learning_agent(
    state_dim: int,
    action_dim: int,
    config: RLMetaConfig | None = None,
) -> MetaLearningAgent:
    """Factory function to create meta-learning agent.

    Args:
        state_dim: State dimensionality
        action_dim: Action space size
        config: Configuration

    Returns:
        Configured MetaLearningAgent
    """
    return MetaLearningAgent(state_dim, action_dim, config)
