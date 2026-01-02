"""
Reinforcement learning models for trading.

Implements PPO (Proximal Policy Optimization) for position sizing
and portfolio management decisions.

GPU Acceleration Features:
- Automatic device selection (CUDA, MPS, CPU)
- Mixed precision training (AMP) for faster training
- Memory optimization utilities
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from quant_trading_system.models.base import ModelType, TradingModel

logger = logging.getLogger(__name__)


# =============================================================================
# GPU ACCELERATION UTILITIES FOR RL
# =============================================================================


def select_device_rl(device: str | None = None) -> torch.device:
    """
    Select the optimal device for RL computation.

    Args:
        device: Optional device string ("cuda", "mps", "cpu", or None for auto)

    Returns:
        Selected torch.device
    """
    if device is None:
        if torch.cuda.is_available():
            selected_device = torch.device("cuda")
            logger.info(f"RL using CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            selected_device = torch.device("mps")
            logger.info("RL using Apple MPS (Metal Performance Shaders)")
        else:
            selected_device = torch.device("cpu")
            logger.info("RL using CPU (no GPU acceleration available)")
        return selected_device

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")

    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        logger.warning("MPS requested but not available, falling back to CPU")
        return torch.device("cpu")

    return torch.device(device)


class TradingEnvironment:
    """
    OpenAI Gym-style trading environment for RL training.

    The agent learns to determine position sizes based on
    market features and portfolio state.
    """

    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        max_position: float = 1.0,
        reward_scaling: float = 1.0,
    ):
        """
        Initialize trading environment.

        Args:
            prices: Price array of shape (n_steps,)
            features: Feature matrix of shape (n_steps, n_features)
            initial_balance: Starting portfolio value
            transaction_cost: Transaction cost as fraction
            max_position: Maximum position size (-1 to 1)
            reward_scaling: Scale factor for rewards
        """
        self.prices = prices
        self.features = features
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.reward_scaling = reward_scaling

        self.n_steps = len(prices)
        self.n_features = features.shape[1]

        # State: features + position + unrealized PnL + portfolio value
        self.state_dim = self.n_features + 3
        self.action_dim = 1  # Continuous position target

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.step_idx = 0
        self.position = 0.0
        self.cash = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.total_pnl = 0.0
        self.max_portfolio_value = self.initial_balance
        self.trades = 0

        return self._get_state()

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: Target position (-1 to 1)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Clip action to valid range
        action = np.clip(action, -self.max_position, self.max_position)

        current_price = self.prices[self.step_idx]
        prev_portfolio_value = self.portfolio_value

        # Calculate position change
        position_change = action - self.position
        trade_value = abs(position_change) * current_price * self.initial_balance

        # Apply transaction costs
        costs = trade_value * self.transaction_cost
        self.cash -= costs

        # Update position and entry price
        if abs(position_change) > 0.01:  # Minimum trade size
            self.trades += 1
            if self.position == 0:
                self.entry_price = current_price
            elif np.sign(position_change) != np.sign(self.position):
                # Position reversal
                self.entry_price = current_price

        self.position = action

        # Move to next step
        self.step_idx += 1
        done = self.step_idx >= self.n_steps - 1

        if not done:
            next_price = self.prices[self.step_idx]
        else:
            next_price = current_price

        # Calculate PnL
        price_change = (next_price - current_price) / current_price
        position_pnl = self.position * price_change * self.initial_balance
        self.unrealized_pnl = (next_price - self.entry_price) / self.entry_price * self.position * self.initial_balance if self.position != 0 else 0

        # Update portfolio value
        self.portfolio_value = prev_portfolio_value + position_pnl - costs
        self.total_pnl = self.portfolio_value - self.initial_balance

        # Track max for drawdown
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value

        # Calculate reward (risk-adjusted return)
        returns = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value

        # Sharpe-like reward with drawdown penalty
        reward = returns * self.reward_scaling
        reward -= 0.1 * abs(position_change)  # Penalize excessive trading
        reward -= 0.5 * max(0, drawdown - 0.1)  # Penalize large drawdowns

        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "total_pnl": self.total_pnl,
            "drawdown": drawdown,
            "trades": self.trades,
        }

        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        """Construct state vector."""
        features = self.features[self.step_idx]

        # Normalize portfolio metrics
        normalized_position = self.position / self.max_position
        normalized_pnl = self.unrealized_pnl / self.initial_balance
        normalized_value = (self.portfolio_value / self.initial_balance) - 1.0

        state = np.concatenate([
            features,
            [normalized_position, normalized_pnl, normalized_value],
        ]).astype(np.float32)

        return state


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden_dim: int = 128,
        continuous: bool = True,
    ):
        """
        Initialize Actor-Critic network.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            continuous: Whether action space is continuous
        """
        super().__init__()

        self.continuous = continuous

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action distribution and value."""
        features = self.shared(state)
        value = self.critic(features)

        if self.continuous:
            mean = torch.tanh(self.actor_mean(features))  # Bound to [-1, 1]
            std = torch.exp(self.actor_log_std).expand_as(mean)
            return mean, std, value
        else:
            logits = self.actor(features)
            return logits, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Returns:
            Tuple of (action, log_prob, value)
        """
        if self.continuous:
            mean, std, value = self.forward(state)
            dist = Normal(mean, std)

            if deterministic:
                action = mean
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob, value.squeeze(-1)
        else:
            logits, value = self.forward(state)
            dist = Categorical(logits=logits)

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            return action, log_prob, value.squeeze(-1)

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Returns:
            Tuple of (log_prob, entropy, value)
        """
        if self.continuous:
            mean, std, value = self.forward(state)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits, value = self.forward(state)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return log_prob, entropy, value.squeeze(-1)


class PPOAgent(TradingModel):
    """
    Proximal Policy Optimization agent for position sizing.

    Uses clipped surrogate objective for stable policy updates.
    Learns to maximize risk-adjusted returns while minimizing drawdowns.
    GPU Acceleration: Automatic device selection with optional AMP.
    """

    def __init__(
        self,
        name: str = "ppo",
        version: str = "1.0.0",
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        continuous: bool = True,
        device: str | None = None,
        use_amp: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize PPO agent.

        Args:
            name: Model identifier
            version: Version string
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy bonus coefficient
            value_loss_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm
            n_steps: Steps per rollout
            n_epochs: Epochs per update
            batch_size: Mini-batch size
            continuous: Whether to use continuous actions
            device: Device (cuda/mps/cpu/None for auto-detect)
            use_amp: Whether to use Automatic Mixed Precision (GPU only)
            **kwargs: Additional parameters
        """
        super().__init__(name, version, ModelType.REGRESSOR, **kwargs)

        self._params.update({
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_epsilon": clip_epsilon,
            "entropy_coef": entropy_coef,
            "value_loss_coef": value_loss_coef,
            "max_grad_norm": max_grad_norm,
            "n_steps": n_steps,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "continuous": continuous,
            "use_amp": use_amp,
        })

        # GPU Acceleration: Use auto-detection for optimal device
        self.device = select_device_rl(device)
        self._use_amp = use_amp and self.device.type == "cuda"
        self._scaler: torch.cuda.amp.GradScaler | None = None
        self._network: ActorCritic | None = None
        self._optimizer: torch.optim.Optimizer | None = None

        logger.info(f"PPO agent initialized on device: {self.device}")
        if self._use_amp:
            logger.info("AMP (Automatic Mixed Precision) enabled for PPO training")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weights: np.ndarray | None = None,
        prices: np.ndarray | None = None,
        n_episodes: int = 100,
        validation_split: float = 0.2,
        **kwargs: Any,
    ) -> "PPOAgent":
        """
        Train PPO agent.

        CRITICAL FIX: Uses time-series split to prevent data leakage.
        The agent trains only on the first (1 - validation_split) portion
        of the data to avoid look-ahead bias.

        Args:
            X: Feature matrix (used to construct environment)
            y: Price returns or prices for reward calculation
            validation_data: Optional separate validation data (X_val, prices_val)
            sample_weights: Not used for RL
            prices: Price array for environment (if y is not prices)
            n_episodes: Number of training episodes
            validation_split: Fraction of data to reserve for validation (default 0.2)
            **kwargs: Additional parameters

        Returns:
            self for method chaining
        """
        # Use y as prices if prices not provided
        if prices is None:
            prices = y

        # CRITICAL FIX: Apply time-series split to prevent data leakage
        # RL agents must train only on past data, never on future data
        n_samples = len(prices)
        train_size = int(n_samples * (1 - validation_split))

        if train_size < 100:
            raise ValueError(
                f"Insufficient training samples: {train_size}. "
                f"Need at least 100 samples for RL training."
            )

        # Split data chronologically (time-series aware)
        X_train = X[:train_size]
        prices_train = prices[:train_size]

        # Keep validation data for metrics (but don't use during training)
        X_val = X[train_size:]
        prices_val = prices[train_size:]

        # Create environment with TRAINING DATA ONLY
        env = TradingEnvironment(prices=prices_train, features=X_train)

        # Initialize network
        self._network = ActorCritic(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=self._params["hidden_dim"],
            continuous=self._params["continuous"],
        )
        self._network.to(self.device)

        self._optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=self._params["learning_rate"],
        )

        # Initialize AMP scaler for mixed precision training
        if self._use_amp:
            self._scaler = torch.cuda.amp.GradScaler()
            logger.info("AMP GradScaler initialized for PPO training")

        # Log GPU memory if using CUDA
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory allocated: {allocated:.2f} GB")

        # Training metrics
        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            # Collect rollout
            states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []

            state = env.reset()
            episode_reward = 0
            episode_length = 0

            while True:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device, non_blocking=True)

                with torch.no_grad():
                    # Use AMP autocast for faster inference
                    if self._use_amp:
                        with torch.cuda.amp.autocast():
                            action, log_prob, value = self._network.get_action(state_tensor)
                    else:
                        action, log_prob, value = self._network.get_action(state_tensor)

                action_np = action.cpu().numpy().flatten()[0]
                next_state, reward, done, info = env.step(action_np)

                states.append(state)
                actions.append(action_np)
                rewards.append(reward)
                values.append(value.item())
                log_probs.append(log_prob.item())
                dones.append(done)

                state = next_state
                episode_reward += reward
                episode_length += 1

                if done or episode_length >= self._params["n_steps"]:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Compute advantages
            advantages, returns = self._compute_gae(
                rewards, values, dones,
                self._params["gamma"],
                self._params["gae_lambda"],
            )

            # PPO update
            self._ppo_update(
                states, actions, log_probs, returns, advantages
            )

        # Store model
        self._model = self._network.state_dict()

        # Evaluate on validation set (out-of-sample)
        val_metrics = {}
        if len(X_val) >= 50:
            val_env = TradingEnvironment(prices=prices_val, features=X_val)
            val_reward, val_trades, val_final_value = self._evaluate_on_env(val_env)
            val_metrics = {
                "val_total_reward": val_reward,
                "val_trades": val_trades,
                "val_final_portfolio_value": val_final_value,
                "val_return_pct": (val_final_value / val_env.initial_balance - 1) * 100,
            }

        # Record training metrics
        metrics = {
            "mean_episode_reward": float(np.mean(episode_rewards[-10:])),
            "mean_episode_length": float(np.mean(episode_lengths[-10:])),
            "total_episodes": n_episodes,
            "train_samples": train_size,
            "val_samples": len(X_val),
            **val_metrics,
        }

        feature_names = [f"f{i}" for i in range(X_train.shape[1])] + ["position", "pnl", "value"]
        self._record_training(metrics, feature_names)

        return self

    def _evaluate_on_env(self, env: TradingEnvironment) -> tuple[float, int, float]:
        """Evaluate agent on an environment without training.

        Args:
            env: Trading environment to evaluate on.

        Returns:
            Tuple of (total_reward, num_trades, final_portfolio_value)
        """
        self._network.eval()
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, _, _ = self._network.get_action(state_tensor, deterministic=True)

            action_np = action.cpu().numpy().flatten()[0]
            state, reward, done, info = env.step(action_np)
            total_reward += reward

        return total_reward, env.trades, env.portfolio_value

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate position size predictions.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Position sizes from -1 to 1
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")

        # Add dummy portfolio state features if not present
        expected_features = len(self._feature_names)
        if X.shape[1] < expected_features:
            padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
            X = np.hstack([X, padding])

        self._network.eval()
        X_tensor = torch.FloatTensor(X).to(self.device, non_blocking=True)

        with torch.no_grad():
            # Use AMP autocast for faster inference on GPU
            if self._use_amp:
                with torch.cuda.amp.autocast():
                    actions, _, _ = self._network.get_action(X_tensor, deterministic=True)
            else:
                actions, _, _ = self._network.get_action(X_tensor, deterministic=True)
            predictions = actions.cpu().numpy().flatten()

        return predictions

    def get_feature_importance(self) -> dict[str, float]:
        """RL agents don't have direct feature importance."""
        return {name: 1.0 / len(self._feature_names) for name in self._feature_names}

    def _compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        dones: list[bool],
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.

        Returns:
            Tuple of (advantages, returns)
        """
        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)

        last_gae = 0
        last_return = 0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            returns[t] = advantages[t] + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def _ppo_update(
        self,
        states: list[np.ndarray],
        actions: list[float],
        old_log_probs: list[float],
        returns: np.ndarray,
        advantages: np.ndarray,
    ) -> None:
        """Perform PPO policy update with GPU acceleration and optional AMP."""
        states = torch.FloatTensor(np.array(states)).to(self.device, non_blocking=True)
        actions = torch.FloatTensor(np.array(actions)).unsqueeze(-1).to(self.device, non_blocking=True)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device, non_blocking=True)
        returns = torch.FloatTensor(returns).to(self.device, non_blocking=True)
        advantages = torch.FloatTensor(advantages).to(self.device, non_blocking=True)

        dataset_size = len(states)
        batch_size = min(self._params["batch_size"], dataset_size)

        for _ in range(self._params["n_epochs"]):
            # Generate random indices for mini-batches
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                self._optimizer.zero_grad(set_to_none=True)

                # GPU Acceleration: Use AMP autocast for mixed precision training
                if self._use_amp:
                    with torch.cuda.amp.autocast():
                        # Evaluate actions
                        new_log_probs, entropy, values = self._network.evaluate_actions(
                            batch_states, batch_actions
                        )

                        # Policy loss (clipped surrogate)
                        ratio = torch.exp(new_log_probs - batch_old_log_probs)
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(
                            ratio,
                            1 - self._params["clip_epsilon"],
                            1 + self._params["clip_epsilon"],
                        ) * batch_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # Value loss
                        value_loss = F.mse_loss(values, batch_returns)

                        # Entropy bonus
                        entropy_loss = -entropy.mean()

                        # Total loss
                        loss = (
                            policy_loss
                            + self._params["value_loss_coef"] * value_loss
                            + self._params["entropy_coef"] * entropy_loss
                        )

                    # Scale loss and backward pass
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self._optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._network.parameters(),
                        self._params["max_grad_norm"],
                    )
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                else:
                    # Evaluate actions
                    new_log_probs, entropy, values = self._network.evaluate_actions(
                        batch_states, batch_actions
                    )

                    # Policy loss (clipped surrogate)
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(
                        ratio,
                        1 - self._params["clip_epsilon"],
                        1 + self._params["clip_epsilon"],
                    ) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = F.mse_loss(values, batch_returns)

                    # Entropy bonus
                    entropy_loss = -entropy.mean()

                    # Total loss
                    loss = (
                        policy_loss
                        + self._params["value_loss_coef"] * value_loss
                        + self._params["entropy_coef"] * entropy_loss
                    )

                    # Standard backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self._network.parameters(),
                        self._params["max_grad_norm"],
                    )
                    self._optimizer.step()

    def save(self, path: str | Path) -> None:
        """Save the PPO agent to disk.

        CRITICAL FIX: Properly saves PyTorch model state.

        Args:
            path: Path to save model (without extension).
        """
        import json
        from pathlib import Path as PathLib

        path = PathLib(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save PyTorch model
        model_path = path.with_suffix(".pt")
        torch.save({
            "network_state_dict": self._network.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "params": self._params,
        }, model_path)

        # Save metadata
        metadata = {
            "name": self._name,
            "version": self._version,
            "model_type": self._model_type.value,
            "is_fitted": self._is_fitted,
            "feature_names": self._feature_names,
            "training_timestamp": self._training_timestamp.isoformat() if self._training_timestamp else None,
            "training_metrics": self._training_metrics,
            "params": {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                      for k, v in self._params.items()},
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def load(self, path: str | Path) -> "PPOAgent":
        """Load a PPO agent from disk with GPU-portable device mapping.

        CRITICAL FIX: Properly loads PyTorch model state with map_location
        for device portability - models saved on GPU can be loaded on CPU and vice versa.

        Args:
            path: Path to saved model (without extension).

        Returns:
            Self for method chaining.
        """
        import json
        from pathlib import Path as PathLib

        path = PathLib(path)
        model_path = path.with_suffix(".pt")
        metadata_path = path.with_suffix(".json")

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Restore params
        self._params = metadata.get("params", self._params)
        self._feature_names = metadata.get("feature_names", [])
        self._is_fitted = metadata.get("is_fitted", True)

        # GPU Portability: Auto-detect device if not initialized
        if not hasattr(self, 'device') or self.device is None:
            self.device = select_device_rl(None)
            self._use_amp = self._params.get("use_amp", True) and self.device.type == "cuda"

        # GPU Portability: Load PyTorch model with map_location for device portability
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Initialize network if needed
        # CRITICAL FIX: Use correct class name (ActorCritic, not ActorCriticNetwork)
        # and correct state_dict key (shared.0.weight, not actor.0.weight)
        if self._network is None:
            # The shared layer is the first layer in the network
            state_dim = checkpoint["network_state_dict"]["shared.0.weight"].shape[1]
            self._network = ActorCritic(
                state_dim=state_dim,
                action_dim=1,
                hidden_dim=int(self._params.get("hidden_dim", 128)),
                continuous=self._params.get("continuous", True),
            ).to(self.device)
            self._optimizer = torch.optim.Adam(
                self._network.parameters(),
                lr=float(self._params.get("learning_rate", 3e-4)),
            )

        self._network.load_state_dict(checkpoint["network_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._model = self._network.state_dict()

        logger.info(f"PPO agent loaded successfully on device: {self.device}, amp={self._use_amp}")

        return self


class A2CAgent(TradingModel):
    """
    Advantage Actor-Critic agent for position sizing.

    Simpler synchronous version of PPO with similar capabilities.
    GPU Acceleration: Automatic device selection with optional AMP.
    """

    def __init__(
        self,
        name: str = "a2c",
        version: str = "1.0.0",
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 5,
        continuous: bool = True,
        device: str | None = None,
        use_amp: bool = True,
        **kwargs: Any,
    ):
        """Initialize A2C agent with GPU acceleration."""
        super().__init__(name, version, ModelType.REGRESSOR, **kwargs)

        self._params.update({
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "entropy_coef": entropy_coef,
            "value_loss_coef": value_loss_coef,
            "max_grad_norm": max_grad_norm,
            "n_steps": n_steps,
            "continuous": continuous,
            "use_amp": use_amp,
        })

        # GPU Acceleration: Use auto-detection for optimal device
        self.device = select_device_rl(device)
        self._use_amp = use_amp and self.device.type == "cuda"
        self._scaler: torch.cuda.amp.GradScaler | None = None
        self._network: ActorCritic | None = None
        self._optimizer: torch.optim.Optimizer | None = None

        logger.info(f"A2C agent initialized on device: {self.device}")
        if self._use_amp:
            logger.info("AMP (Automatic Mixed Precision) enabled for A2C training")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weights: np.ndarray | None = None,
        prices: np.ndarray | None = None,
        n_episodes: int = 100,
        validation_split: float = 0.2,
        **kwargs: Any,
    ) -> "A2CAgent":
        """Train A2C agent.

        CRITICAL FIX: Uses time-series split to prevent data leakage.
        The agent trains only on the first (1 - validation_split) portion
        of the data to avoid look-ahead bias.
        """
        if prices is None:
            prices = y

        # CRITICAL FIX: Apply time-series split to prevent data leakage
        n_samples = len(prices)
        train_size = int(n_samples * (1 - validation_split))

        if train_size < 100:
            raise ValueError(
                f"Insufficient training samples: {train_size}. "
                f"Need at least 100 samples for RL training."
            )

        # Split data chronologically (time-series aware)
        X_train = X[:train_size]
        prices_train = prices[:train_size]
        X_val = X[train_size:]
        prices_val = prices[train_size:]

        # Create environment with TRAINING DATA ONLY
        env = TradingEnvironment(prices=prices_train, features=X_train)

        self._network = ActorCritic(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=self._params["hidden_dim"],
            continuous=self._params["continuous"],
        )
        self._network.to(self.device)

        self._optimizer = torch.optim.RMSprop(
            self._network.parameters(),
            lr=self._params["learning_rate"],
        )

        # Initialize AMP scaler for mixed precision training
        if self._use_amp:
            self._scaler = torch.cuda.amp.GradScaler()
            logger.info("AMP GradScaler initialized for A2C training")

        # Log GPU memory if using CUDA
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory allocated: {allocated:.2f} GB")

        episode_rewards = []

        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Collect n-step transitions
                states, actions, rewards, dones = [], [], [], []

                for _ in range(self._params["n_steps"]):
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device, non_blocking=True)

                    with torch.no_grad():
                        # Use AMP autocast for faster inference
                        if self._use_amp:
                            with torch.cuda.amp.autocast():
                                action, log_prob, value = self._network.get_action(state_tensor)
                        else:
                            action, log_prob, value = self._network.get_action(state_tensor)

                    action_np = action.cpu().numpy().flatten()[0]
                    next_state, reward, done, info = env.step(action_np)

                    states.append(state)
                    actions.append(action_np)
                    rewards.append(reward)
                    dones.append(done)

                    episode_reward += reward
                    state = next_state

                    if done:
                        break

                # Compute returns
                returns = self._compute_returns(rewards, dones, state)

                # A2C update
                self._a2c_update(states, actions, returns)

            episode_rewards.append(episode_reward)

        self._model = self._network.state_dict()

        # Evaluate on validation set (out-of-sample)
        val_metrics = {}
        if len(X_val) >= 50:
            val_env = TradingEnvironment(prices=prices_val, features=X_val)
            val_reward, val_trades, val_final_value = self._evaluate_on_env(val_env)
            val_metrics = {
                "val_total_reward": val_reward,
                "val_trades": val_trades,
                "val_final_portfolio_value": val_final_value,
                "val_return_pct": (val_final_value / val_env.initial_balance - 1) * 100,
            }

        metrics = {
            "mean_episode_reward": float(np.mean(episode_rewards[-10:])),
            "total_episodes": n_episodes,
            "train_samples": train_size,
            "val_samples": len(X_val),
            **val_metrics,
        }

        feature_names = [f"f{i}" for i in range(X_train.shape[1])] + ["position", "pnl", "value"]
        self._record_training(metrics, feature_names)

        return self

    def _evaluate_on_env(self, env: TradingEnvironment) -> tuple[float, int, float]:
        """Evaluate agent on an environment without training.

        Args:
            env: Trading environment to evaluate on.

        Returns:
            Tuple of (total_reward, num_trades, final_portfolio_value)
        """
        self._network.eval()
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, _, _ = self._network.get_action(state_tensor, deterministic=True)

            action_np = action.cpu().numpy().flatten()[0]
            state, reward, done, info = env.step(action_np)
            total_reward += reward

        return total_reward, env.trades, env.portfolio_value

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate position predictions with GPU acceleration."""
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")

        expected_features = len(self._feature_names)
        if X.shape[1] < expected_features:
            padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
            X = np.hstack([X, padding])

        self._network.eval()
        X_tensor = torch.FloatTensor(X).to(self.device, non_blocking=True)

        with torch.no_grad():
            # Use AMP autocast for faster inference on GPU
            if self._use_amp:
                with torch.cuda.amp.autocast():
                    actions, _, _ = self._network.get_action(X_tensor, deterministic=True)
            else:
                actions, _, _ = self._network.get_action(X_tensor, deterministic=True)
            predictions = actions.cpu().numpy().flatten()

        return predictions

    def get_feature_importance(self) -> dict[str, float]:
        """A2C agents don't have direct feature importance."""
        return {name: 1.0 / len(self._feature_names) for name in self._feature_names}

    def _compute_returns(
        self,
        rewards: list[float],
        dones: list[bool],
        final_state: np.ndarray,
    ) -> np.ndarray:
        """Compute discounted returns."""
        n = len(rewards)
        returns = np.zeros(n)

        # Bootstrap from final state if not done
        if not dones[-1]:
            state_tensor = torch.FloatTensor(final_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, _, final_value = self._network.get_action(state_tensor)
            R = final_value.item()
        else:
            R = 0

        for t in reversed(range(n)):
            R = rewards[t] + self._params["gamma"] * R * (1 - dones[t])
            returns[t] = R

        return returns

    def _a2c_update(
        self,
        states: list[np.ndarray],
        actions: list[float],
        returns: np.ndarray,
    ) -> None:
        """Perform A2C policy update with GPU acceleration and optional AMP."""
        states = torch.FloatTensor(np.array(states)).to(self.device, non_blocking=True)
        actions = torch.FloatTensor(np.array(actions)).unsqueeze(-1).to(self.device, non_blocking=True)
        returns = torch.FloatTensor(returns).to(self.device, non_blocking=True)

        self._optimizer.zero_grad(set_to_none=True)

        # GPU Acceleration: Use AMP autocast for mixed precision training
        if self._use_amp:
            with torch.cuda.amp.autocast():
                log_probs, entropy, values = self._network.evaluate_actions(states, actions)

                advantages = returns - values.detach()

                # Policy loss
                policy_loss = -(log_probs * advantages).mean()

                # Value loss
                value_loss = F.mse_loss(values, returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self._params["value_loss_coef"] * value_loss
                    + self._params["entropy_coef"] * entropy_loss
                )

            # Scale loss and backward pass
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(
                self._network.parameters(),
                self._params["max_grad_norm"],
            )
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            log_probs, entropy, values = self._network.evaluate_actions(states, actions)

            advantages = returns - values.detach()

            # Policy loss
            policy_loss = -(log_probs * advantages).mean()

            # Value loss
            value_loss = F.mse_loss(values, returns)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = (
                policy_loss
                + self._params["value_loss_coef"] * value_loss
                + self._params["entropy_coef"] * entropy_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._network.parameters(),
                self._params["max_grad_norm"],
            )
            self._optimizer.step()

    def save(self, path: str | Path) -> None:
        """Save the A2C agent to disk.

        CRITICAL FIX: Properly saves PyTorch model state.

        Args:
            path: Path to save model (without extension).
        """
        import json
        from pathlib import Path as PathLib

        path = PathLib(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save PyTorch model
        model_path = path.with_suffix(".pt")
        torch.save({
            "network_state_dict": self._network.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "params": self._params,
        }, model_path)

        # Save metadata
        metadata = {
            "name": self._name,
            "version": self._version,
            "model_type": self._model_type.value,
            "is_fitted": self._is_fitted,
            "feature_names": self._feature_names,
            "training_timestamp": self._training_timestamp.isoformat() if self._training_timestamp else None,
            "training_metrics": self._training_metrics,
            "params": {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                      for k, v in self._params.items()},
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def load(self, path: str | Path) -> "A2CAgent":
        """Load an A2C agent from disk.

        CRITICAL FIX: Properly loads PyTorch model state.

        Args:
            path: Path to saved model (without extension).

        Returns:
            Self for method chaining.
        """
        import json
        from pathlib import Path as PathLib

        path = PathLib(path)
        model_path = path.with_suffix(".pt")
        metadata_path = path.with_suffix(".json")

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Restore params
        self._params = metadata.get("params", self._params)
        self._feature_names = metadata.get("feature_names", [])
        self._is_fitted = metadata.get("is_fitted", True)

        # Load PyTorch model
        checkpoint = torch.load(model_path, map_location=self.device)

        # Initialize network if needed
        # CRITICAL FIX: Use correct class name (ActorCritic, not ActorCriticNetwork)
        # and correct state_dict key (shared.0.weight, not actor.0.weight)
        if self._network is None:
            # The shared layer is the first layer in the network
            state_dim = checkpoint["network_state_dict"]["shared.0.weight"].shape[1]
            self._network = ActorCritic(
                state_dim=state_dim,
                action_dim=1,
                hidden_dim=int(self._params.get("hidden_dim", 128)),
                continuous=self._params.get("continuous", True),
            ).to(self.device)
            self._optimizer = torch.optim.RMSprop(
                self._network.parameters(),
                lr=float(self._params.get("learning_rate", 3e-4)),
            )

        self._network.load_state_dict(checkpoint["network_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._model = self._network.state_dict()

        return self
