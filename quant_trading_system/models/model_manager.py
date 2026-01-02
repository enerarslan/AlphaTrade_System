"""
Model management for training, validation, and persistence.

Handles the full model lifecycle including data splitting, hyperparameter
optimization, model selection, versioning, and registry management.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np

from quant_trading_system.models.base import ModelType, TradingModel


class SplitMethod(str, Enum):
    """Data splitting methods."""

    WALK_FORWARD = "walk_forward"
    PURGED_KFOLD = "purged_kfold"
    EXPANDING_WINDOW = "expanding_window"


class TimeSeriesSplitter:
    """
    Time series cross-validation splitter.

    Implements walk-forward, purged K-fold, and expanding window
    validation strategies to prevent look-ahead bias.
    """

    def __init__(
        self,
        method: SplitMethod = SplitMethod.WALK_FORWARD,
        n_splits: int = 5,
        train_size: int | float | None = None,
        test_size: int | float | None = None,
        gap: int = 0,
    ):
        """
        Initialize splitter.

        Args:
            method: Splitting method
            n_splits: Number of splits
            train_size: Training window size (samples or fraction)
            test_size: Test window size (samples or fraction)
            gap: Gap between train and test (to prevent leakage)
        """
        self.method = method
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices.

        Args:
            X: Feature matrix
            y: Target array (unused, for API compatibility)

        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)

        if self.method == SplitMethod.WALK_FORWARD:
            return self._walk_forward_split(n_samples)
        elif self.method == SplitMethod.PURGED_KFOLD:
            return self._purged_kfold_split(n_samples)
        elif self.method == SplitMethod.EXPANDING_WINDOW:
            return self._expanding_window_split(n_samples)
        else:
            raise ValueError(f"Unknown split method: {self.method}")

    def _walk_forward_split(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Walk-forward validation with rolling window."""
        # Calculate window sizes
        if isinstance(self.train_size, float):
            train_size = int(n_samples * self.train_size / (self.n_splits + 1))
        else:
            train_size = self.train_size or n_samples // (self.n_splits + 1)

        if isinstance(self.test_size, float):
            test_size = int(n_samples * self.test_size / self.n_splits)
        else:
            test_size = self.test_size or train_size // 4

        splits = []
        for i in range(self.n_splits):
            test_start = train_size + i * test_size + self.gap
            test_end = min(test_start + test_size, n_samples)

            if test_end > n_samples:
                break

            # BUG FIX: Walk-forward should include ALL historical data from the start
            # Was: np.arange(i * test_size, train_size + i * test_size) - excluded early data
            train_indices = np.arange(0, train_size + i * test_size)
            test_indices = np.arange(test_start, test_end)

            splits.append((train_indices, test_indices))

        return splits

    def _purged_kfold_split(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Purged K-fold with gap between train and test."""
        fold_size = n_samples // self.n_splits
        splits = []

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples

            test_indices = np.arange(test_start, test_end)

            # Train on everything except test fold and gap
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[max(0, test_start - self.gap):min(n_samples, test_end + self.gap)] = False
            train_indices = np.where(train_mask)[0]

            splits.append((train_indices, test_indices))

        return splits

    def _expanding_window_split(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Expanding window with growing training set."""
        if isinstance(self.test_size, float):
            test_size = int(n_samples * self.test_size / self.n_splits)
        else:
            test_size = self.test_size or n_samples // (self.n_splits + 1)

        min_train = n_samples // (self.n_splits + 1)
        splits = []

        for i in range(self.n_splits):
            train_end = min_train + i * test_size
            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n_samples)

            if test_end > n_samples:
                break

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            splits.append((train_indices, test_indices))

        return splits


class ModelRegistry:
    """
    Registry for model versioning and management.

    Tracks model versions, metadata, and provides rollback capabilities.
    """

    def __init__(self, base_path: str | Path):
        """
        Initialize registry.

        Args:
            base_path: Base directory for model storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self._registry_file = self.base_path / "registry.json"
        self._registry: dict[str, list[dict]] = self._load_registry()

    def _load_registry(self) -> dict[str, list[dict]]:
        """Load registry from disk."""
        if self._registry_file.exists():
            with open(self._registry_file, "r") as f:
                return json.load(f)
        return {}

    def _save_registry(self) -> None:
        """Save registry to disk."""
        with open(self._registry_file, "w") as f:
            json.dump(self._registry, f, indent=2, default=str)

    def register(
        self,
        model: TradingModel,
        metrics: dict[str, float] | None = None,
        tags: list[str] | None = None,
        description: str = "",
    ) -> str:
        """
        Register a model version.

        Args:
            model: Model to register
            metrics: Performance metrics
            tags: Tags for categorization
            description: Model description

        Returns:
            Unique version ID
        """
        model_name = model.name
        version_id = f"{model.version}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Create model directory
        model_dir = self.base_path / model_name / version_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / "model"
        model.save(model_path)

        # Create metadata
        metadata = {
            "version_id": version_id,
            "model_name": model_name,
            "model_version": model.version,
            "model_type": model.model_type.value,
            "registered_at": datetime.utcnow().isoformat(),
            "metrics": metrics or model.training_metrics,
            "tags": tags or [],
            "description": description,
            "is_active": True,
            "path": str(model_dir),
        }

        # Update registry
        if model_name not in self._registry:
            self._registry[model_name] = []
        self._registry[model_name].append(metadata)
        self._save_registry()

        return version_id

    def get_model(
        self,
        model_name: str,
        version_id: str | None = None,
        model_class: type[TradingModel] | None = None,
    ) -> TradingModel:
        """
        Load a registered model.

        Args:
            model_name: Model name
            version_id: Specific version (None for latest)
            model_class: Model class for instantiation

        Returns:
            Loaded model

        Raises:
            ValueError: If model not found
        """
        if model_name not in self._registry:
            raise ValueError(f"Model {model_name} not found in registry")

        versions = self._registry[model_name]
        if not versions:
            raise ValueError(f"No versions found for model {model_name}")

        if version_id is None:
            # Get latest active version
            active_versions = [v for v in versions if v.get("is_active", True)]
            if not active_versions:
                raise ValueError(f"No active versions for model {model_name}")
            metadata = active_versions[-1]
        else:
            # Find specific version
            metadata = next((v for v in versions if v["version_id"] == version_id), None)
            if metadata is None:
                raise ValueError(f"Version {version_id} not found for model {model_name}")

        # Load model
        model_path = Path(metadata["path"]) / "model"

        if model_class is None:
            # Try to infer model class
            from quant_trading_system.models.classical_ml import (
                CatBoostModel,
                ElasticNetModel,
                LightGBMModel,
                RandomForestModel,
                XGBoostModel,
            )
            from quant_trading_system.models.deep_learning import LSTMModel, TCNModel, TransformerModel
            from quant_trading_system.models.ensemble import (
                AdaptiveEnsemble,
                AveragingEnsemble,
                StackingEnsemble,
                VotingEnsemble,
            )
            from quant_trading_system.models.reinforcement import A2CAgent, PPOAgent

            class_map = {
                "xgboost": XGBoostModel,
                "lightgbm": LightGBMModel,
                "catboost": CatBoostModel,
                "random_forest": RandomForestModel,
                "elastic_net": ElasticNetModel,
                "lstm": LSTMModel,
                "transformer": TransformerModel,
                "tcn": TCNModel,
                "ppo": PPOAgent,
                "a2c": A2CAgent,
                "voting_ensemble": VotingEnsemble,
                "averaging_ensemble": AveragingEnsemble,
                "stacking_ensemble": StackingEnsemble,
                "adaptive_ensemble": AdaptiveEnsemble,
            }
            model_class = class_map.get(model_name)
            if model_class is None:
                raise ValueError(f"Cannot infer model class for {model_name}")

        model = model_class(name=model_name)
        model.load(model_path)
        return model

    def list_models(self, model_name: str | None = None) -> list[dict]:
        """
        List registered models.

        Args:
            model_name: Filter by model name (None for all)

        Returns:
            List of model metadata
        """
        if model_name is not None:
            return self._registry.get(model_name, [])

        all_models = []
        for versions in self._registry.values():
            all_models.extend(versions)
        return all_models

    def deactivate(self, model_name: str, version_id: str) -> None:
        """Deactivate a model version."""
        if model_name not in self._registry:
            return

        for version in self._registry[model_name]:
            if version["version_id"] == version_id:
                version["is_active"] = False
                break

        self._save_registry()

    def delete(self, model_name: str, version_id: str) -> None:
        """Delete a model version."""
        if model_name not in self._registry:
            return

        versions = self._registry[model_name]
        for i, version in enumerate(versions):
            if version["version_id"] == version_id:
                # Remove files
                model_dir = Path(version["path"])
                if model_dir.exists():
                    shutil.rmtree(model_dir)

                # Remove from registry
                del versions[i]
                break

        self._save_registry()


class OverfittingDetector:
    """
    JPMORGAN FIX: Detect overfitting by comparing in-sample and out-of-sample metrics.

    Critical for institutional trading where overfitted models can cause significant losses.
    """

    def __init__(
        self,
        max_is_oos_ratio: float = 2.0,
        min_oos_sharpe: float = 0.3,
        max_oos_drawdown: float = 0.30,
        min_oos_win_rate: float = 0.40,
    ):
        """
        Initialize overfitting detector.

        Args:
            max_is_oos_ratio: Maximum acceptable ratio of IS to OOS performance.
                              Values > 2.0 strongly indicate overfitting.
            min_oos_sharpe: Minimum acceptable Sharpe ratio on OOS data
            max_oos_drawdown: Maximum acceptable drawdown on OOS data
            min_oos_win_rate: Minimum acceptable win rate on OOS data
        """
        self.max_is_oos_ratio = max_is_oos_ratio
        self.min_oos_sharpe = min_oos_sharpe
        self.max_oos_drawdown = max_oos_drawdown
        self.min_oos_win_rate = min_oos_win_rate

    def check_overfitting(
        self,
        is_score: float,
        oos_metrics: dict[str, float],
    ) -> tuple[bool, list[str]]:
        """
        Check if model shows signs of overfitting.

        Args:
            is_score: In-sample (CV) optimization score
            oos_metrics: Out-of-sample holdout metrics

        Returns:
            Tuple of (is_acceptable, list of warning messages)
        """
        warnings = []
        is_acceptable = True

        # Check IS/OOS ratio (score degradation)
        oos_sharpe = oos_metrics.get("sharpe_ratio", 0.0)
        if is_score > 0 and oos_sharpe > 0:
            is_oos_ratio = is_score / oos_sharpe
            if is_oos_ratio > self.max_is_oos_ratio:
                warnings.append(
                    f"OVERFITTING WARNING: IS/OOS ratio {is_oos_ratio:.2f} > {self.max_is_oos_ratio}. "
                    f"In-sample: {is_score:.2f}, Out-of-sample: {oos_sharpe:.2f}"
                )
                is_acceptable = False

        # Check minimum OOS Sharpe
        if oos_sharpe < self.min_oos_sharpe:
            warnings.append(
                f"LOW OOS PERFORMANCE: Sharpe {oos_sharpe:.2f} < {self.min_oos_sharpe}"
            )
            is_acceptable = False

        # Check maximum OOS drawdown
        oos_drawdown = oos_metrics.get("max_drawdown", 0.0)
        if oos_drawdown > self.max_oos_drawdown:
            warnings.append(
                f"HIGH OOS DRAWDOWN: {oos_drawdown:.1%} > {self.max_oos_drawdown:.1%}"
            )
            is_acceptable = False

        # Check minimum OOS win rate
        oos_win_rate = oos_metrics.get("win_rate", 0.0)
        if oos_win_rate < self.min_oos_win_rate:
            warnings.append(
                f"LOW OOS WIN RATE: {oos_win_rate:.1%} < {self.min_oos_win_rate:.1%}"
            )
            is_acceptable = False

        return is_acceptable, warnings

    def compute_is_oos_ratio(
        self,
        is_score: float,
        oos_score: float,
    ) -> float:
        """Compute IS/OOS performance ratio."""
        if oos_score <= 0:
            return float("inf") if is_score > 0 else 1.0
        return is_score / oos_score


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using various methods.

    Supports grid search, random search, and Bayesian optimization.
    """

    def __init__(
        self,
        model_class: type[TradingModel],
        param_space: dict[str, Any],
        metric: str = "sharpe_ratio",
        n_trials: int = 50,
        method: str = "random",
    ):
        """
        Initialize optimizer.

        Args:
            model_class: Model class to optimize
            param_space: Parameter search space
            metric: Optimization metric
            n_trials: Number of trials
            method: "grid", "random", or "bayesian"
        """
        self.model_class = model_class
        self.param_space = param_space
        self.metric = metric
        self.n_trials = n_trials
        self.method = method

        self.best_params: dict[str, Any] = {}
        self.best_score: float = float("-inf")
        self.trials: list[dict] = []

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_splitter: TimeSeriesSplitter,
        scoring_func: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ) -> dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            X: Feature matrix
            y: Target array
            cv_splitter: Cross-validation splitter
            scoring_func: Custom scoring function (y_true, y_pred) -> score

        Returns:
            Best parameters found
        """
        if self.method == "grid":
            return self._grid_search(X, y, cv_splitter, scoring_func)
        elif self.method == "random":
            return self._random_search(X, y, cv_splitter, scoring_func)
        elif self.method == "bayesian":
            return self._bayesian_search(X, y, cv_splitter, scoring_func)
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")

    def _evaluate_params(
        self,
        params: dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        cv_splitter: TimeSeriesSplitter,
        scoring_func: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ) -> float:
        """Evaluate a parameter combination using cross-validation."""
        scores = []

        for train_idx, test_idx in cv_splitter.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Create and train model
            model = self.model_class(**params)
            model.fit(X_train, y_train)

            # Evaluate
            predictions = model.predict(X_test)

            if scoring_func is not None:
                score = scoring_func(y_test, predictions)
            else:
                # Default: negative MSE (higher is better)
                score = -np.mean((y_test - predictions) ** 2)

            scores.append(score)

        return np.mean(scores)

    def _random_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_splitter: TimeSeriesSplitter,
        scoring_func: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ) -> dict[str, Any]:
        """Random search optimization."""
        for _ in range(self.n_trials):
            # Sample random parameters
            params = {}
            for param_name, param_config in self.param_space.items():
                if isinstance(param_config, list):
                    params[param_name] = np.random.choice(param_config)
                elif isinstance(param_config, tuple) and len(param_config) == 2:
                    low, high = param_config
                    if isinstance(low, int) and isinstance(high, int):
                        params[param_name] = np.random.randint(low, high + 1)
                    else:
                        params[param_name] = np.random.uniform(low, high)
                elif isinstance(param_config, dict):
                    if param_config.get("type") == "log":
                        low, high = param_config["range"]
                        params[param_name] = np.exp(np.random.uniform(np.log(low), np.log(high)))
                    else:
                        params[param_name] = param_config.get("default")
                else:
                    params[param_name] = param_config

            # Evaluate
            try:
                score = self._evaluate_params(params, X, y, cv_splitter, scoring_func)
            except Exception:
                score = float("-inf")

            self.trials.append({"params": params, "score": score})

            if score > self.best_score:
                self.best_score = score
                self.best_params = params

        return self.best_params

    def _grid_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_splitter: TimeSeriesSplitter,
        scoring_func: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ) -> dict[str, Any]:
        """Grid search optimization."""
        from itertools import product

        # Generate all combinations
        param_names = list(self.param_space.keys())
        param_values = []

        for config in self.param_space.values():
            if isinstance(config, list):
                param_values.append(config)
            elif isinstance(config, tuple):
                # Sample a few points from range
                low, high = config
                n_points = min(5, self.n_trials // len(self.param_space))
                if isinstance(low, int) and isinstance(high, int):
                    param_values.append(np.linspace(low, high, n_points, dtype=int).tolist())
                else:
                    param_values.append(np.linspace(low, high, n_points).tolist())
            else:
                param_values.append([config])

        for values in product(*param_values):
            params = dict(zip(param_names, values))

            try:
                score = self._evaluate_params(params, X, y, cv_splitter, scoring_func)
            except Exception:
                score = float("-inf")

            self.trials.append({"params": params, "score": score})

            if score > self.best_score:
                self.best_score = score
                self.best_params = params

        return self.best_params

    def _bayesian_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_splitter: TimeSeriesSplitter,
        scoring_func: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ) -> dict[str, Any]:
        """Bayesian optimization using optuna."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            # Fall back to random search
            return self._random_search(X, y, cv_splitter, scoring_func)

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for param_name, param_config in self.param_space.items():
                if isinstance(param_config, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_config)
                elif isinstance(param_config, tuple):
                    low, high = param_config
                    if isinstance(low, int) and isinstance(high, int):
                        params[param_name] = trial.suggest_int(param_name, low, high)
                    else:
                        params[param_name] = trial.suggest_float(param_name, low, high)
                elif isinstance(param_config, dict):
                    if param_config.get("type") == "log":
                        low, high = param_config["range"]
                        params[param_name] = trial.suggest_float(param_name, low, high, log=True)
                    else:
                        params[param_name] = param_config.get("default")

            try:
                score = self._evaluate_params(params, X, y, cv_splitter, scoring_func)
            except Exception:
                return float("-inf")

            self.trials.append({"params": params, "score": score})
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params = study.best_params
        self.best_score = study.best_value

        return self.best_params


class ModelManager:
    """
    High-level model management interface.

    Coordinates training, validation, optimization, and deployment.
    """

    def __init__(
        self,
        registry_path: str | Path = "models/registry",
    ):
        """
        Initialize model manager.

        Args:
            registry_path: Path for model registry
        """
        self.registry = ModelRegistry(registry_path)

    def train_model(
        self,
        model: TradingModel,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        gap: int = 5,
        **kwargs: Any,
    ) -> TradingModel:
        """
        Train a model with automatic validation split.

        CRITICAL FIX: Added gap parameter to prevent look-ahead bias.
        The gap creates a buffer zone between training and validation data
        to prevent information leakage from features computed near the boundary.

        Args:
            model: Model to train
            X: Feature matrix
            y: Target array
            validation_split: Fraction for validation
            gap: Number of samples to skip between train and validation
                 to prevent look-ahead bias (default: 5 bars)
            **kwargs: Additional training parameters

        Returns:
            Trained model
        """
        n_samples = len(X)

        # Calculate split point with gap
        # Gap ensures no feature leakage from validation period
        val_size = int(n_samples * validation_split)
        split_idx = n_samples - val_size - gap

        if split_idx <= 0:
            raise ValueError(
                f"Not enough samples ({n_samples}) for validation_split={validation_split} "
                f"and gap={gap}. Reduce validation_split or gap."
            )

        # Training data: [0, split_idx)
        # Gap zone: [split_idx, split_idx + gap) - EXCLUDED from both
        # Validation data: [split_idx + gap, n_samples)
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx + gap:]
        y_val = y[split_idx + gap:]

        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"Training split: train={len(X_train)}, gap={gap}, val={len(X_val)}, "
            f"total={n_samples}"
        )

        model.fit(X_train, y_train, validation_data=(X_val, y_val), **kwargs)

        return model

    def cross_validate(
        self,
        model: TradingModel,
        X: np.ndarray,
        y: np.ndarray,
        cv_method: SplitMethod = SplitMethod.WALK_FORWARD,
        n_splits: int = 5,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            model: Model to validate
            X: Feature matrix
            y: Target array
            cv_method: Cross-validation method
            n_splits: Number of folds
            metrics: Metrics to compute

        Returns:
            Dictionary of metric results
        """
        splitter = TimeSeriesSplitter(method=cv_method, n_splits=n_splits)
        splits = splitter.split(X, y)

        if metrics is None:
            metrics = ["mse", "mae", "r2"] if model.model_type == ModelType.REGRESSOR else ["accuracy", "f1"]

        results: dict[str, list[float]] = {metric: [] for metric in metrics}

        for train_idx, test_idx in splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train on fold
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Calculate metrics
            fold_metrics = self._calculate_metrics(y_test, predictions, metrics, model.model_type)
            for metric, value in fold_metrics.items():
                results[metric].append(value)

        # Aggregate results
        summary = {}
        for metric, values in results.items():
            summary[f"{metric}_mean"] = np.mean(values)
            summary[f"{metric}_std"] = np.std(values)
            summary[f"{metric}_values"] = values

        return summary

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: list[str],
        model_type: ModelType,
    ) -> dict[str, float]:
        """Calculate specified metrics."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        results = {}
        for metric in metrics:
            if metric == "mse":
                results[metric] = mean_squared_error(y_true, y_pred)
            elif metric == "mae":
                results[metric] = mean_absolute_error(y_true, y_pred)
            elif metric == "r2":
                results[metric] = r2_score(y_true, y_pred)
            elif metric == "accuracy":
                results[metric] = accuracy_score(y_true, y_pred)
            elif metric == "f1":
                results[metric] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            elif metric == "sharpe":
                # Simple Sharpe approximation
                returns = y_pred - y_true  # Assuming returns
                results[metric] = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)

        return results

    def optimize_and_train(
        self,
        model_class: type[TradingModel],
        X: np.ndarray,
        y: np.ndarray,
        param_space: dict[str, Any],
        n_trials: int = 50,
        optimization_method: str = "random",
        cv_method: SplitMethod = SplitMethod.WALK_FORWARD,
        register: bool = True,
        holdout_fraction: float = 0.15,
        gap: int = 5,
        reject_overfitted: bool = True,
        overfitting_config: dict[str, float] | None = None,
    ) -> tuple[TradingModel, dict[str, float]]:
        """
        Optimize hyperparameters and train final model.

        JPMORGAN FIX: Now properly holds out test data BEFORE optimization
        to prevent optimistic bias from training on data used for HP selection.
        Also includes overfitting detection to reject models that show
        significant performance degradation on holdout data.

        The data is split as follows:
        1. [0, train_val_end) - Used for HP optimization via cross-validation
        2. [train_val_end, train_val_end + gap) - Gap zone (excluded)
        3. [train_val_end + gap, end) - Holdout test set for final evaluation

        Final model is trained on train+val data only, NOT on holdout.

        Args:
            model_class: Model class to optimize
            X: Feature matrix
            y: Target array
            param_space: Parameter search space
            n_trials: Number of optimization trials
            optimization_method: "grid", "random", or "bayesian"
            cv_method: Cross-validation method
            register: Whether to register the model
            holdout_fraction: Fraction of data to hold out for final testing
            gap: Gap between train/val and holdout to prevent leakage
            reject_overfitted: If True, raise error for overfitted models
            overfitting_config: Configuration for OverfittingDetector

        Returns:
            Tuple of (trained model, holdout test metrics)

        Raises:
            OverfittingError: If reject_overfitted=True and model shows overfitting
        """
        import logging
        logger = logging.getLogger(__name__)

        n_samples = len(X)

        # CRITICAL: Create holdout test set BEFORE optimization
        holdout_size = int(n_samples * holdout_fraction)
        train_val_end = n_samples - holdout_size - gap

        if train_val_end < n_samples * 0.5:
            raise ValueError(
                f"holdout_fraction={holdout_fraction} and gap={gap} leave too little "
                f"data for optimization. Reduce holdout_fraction or gap."
            )

        # Split data
        X_train_val = X[:train_val_end]
        y_train_val = y[:train_val_end]
        X_holdout = X[train_val_end + gap:]
        y_holdout = y[train_val_end + gap:]

        logger.info(
            f"JPMORGAN DATA SPLIT: train_val={len(X_train_val)}, gap={gap}, "
            f"holdout={len(X_holdout)}, total={n_samples}"
        )

        # Optimize on train+val data only
        optimizer = HyperparameterOptimizer(
            model_class=model_class,
            param_space=param_space,
            n_trials=n_trials,
            method=optimization_method,
        )

        cv_splitter = TimeSeriesSplitter(method=cv_method, n_splits=5, gap=gap)
        best_params = optimizer.optimize(X_train_val, y_train_val, cv_splitter)

        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"CV optimization score (in-sample): {optimizer.best_score:.4f}")

        # Train final model on train+val data ONLY (not holdout)
        model = model_class(**best_params)
        model.fit(X_train_val, y_train_val)

        # Evaluate on holdout test set for unbiased performance estimate
        holdout_predictions = model.predict(X_holdout)
        holdout_metrics = self._calculate_holdout_metrics(
            y_holdout, holdout_predictions, model.model_type
        )

        logger.info(f"Holdout test metrics (out-of-sample): {holdout_metrics}")

        # JPMORGAN FIX: Overfitting detection
        detector_config = overfitting_config or {}
        detector = OverfittingDetector(**detector_config)
        is_acceptable, warnings = detector.check_overfitting(
            optimizer.best_score, holdout_metrics
        )

        # Compute and log IS/OOS ratio
        oos_sharpe = holdout_metrics.get("sharpe_ratio", 0.0)
        is_oos_ratio = detector.compute_is_oos_ratio(optimizer.best_score, oos_sharpe)
        holdout_metrics["is_oos_ratio"] = is_oos_ratio

        for warning in warnings:
            logger.warning(warning)

        if not is_acceptable:
            if reject_overfitted:
                from quant_trading_system.core.exceptions import ModelValidationError
                raise ModelValidationError(
                    f"Model rejected due to overfitting. Warnings: {warnings}"
                )
            else:
                logger.error(
                    "MODEL SHOWS OVERFITTING but reject_overfitted=False. "
                    "Proceeding with registration but exercise caution!"
                )
                holdout_metrics["overfitting_detected"] = True
        else:
            logger.info("Model passed overfitting validation checks")
            holdout_metrics["overfitting_detected"] = False

        # Register if requested
        if register:
            tags = ["optimized", "holdout_validated"]
            if not is_acceptable:
                tags.append("overfitting_warning")

            self.registry.register(
                model,
                metrics={
                    "optimization_score": optimizer.best_score,
                    "is_oos_ratio": is_oos_ratio,
                    **{f"holdout_{k}": v for k, v in holdout_metrics.items()},
                },
                tags=tags,
                description=f"Optimized with {optimization_method} search, holdout validated, IS/OOS ratio: {is_oos_ratio:.2f}",
            )

        return model, holdout_metrics

    def _calculate_holdout_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_type: "ModelType",
    ) -> dict[str, float]:
        """
        JPMORGAN FIX: Calculate comprehensive metrics on holdout test set.

        Includes trading-specific metrics for institutional-grade evaluation.
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        metrics = {}

        if model_type == ModelType.REGRESSOR:
            metrics["mse"] = float(mean_squared_error(y_true, y_pred))
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics["r2"] = float(r2_score(y_true, y_pred))

            # Directional accuracy for trading
            if len(y_true) > 1:
                direction_true = np.sign(np.diff(y_true))
                direction_pred = np.sign(np.diff(y_pred))
                metrics["directional_accuracy"] = float(
                    np.mean(direction_true == direction_pred)
                )

            # JPMORGAN FIX: Trading-specific holdout metrics
            # Simulated returns based on predictions
            if len(y_true) > 1:
                # Assume y_pred is return prediction, trade in direction of prediction
                simulated_returns = np.sign(y_pred[:-1]) * y_true[1:]

                if len(simulated_returns) > 0 and np.std(simulated_returns) > 1e-10:
                    # Sharpe ratio (annualized assuming 15-min bars, ~26 bars/day, 252 days/year)
                    annualization_factor = np.sqrt(26 * 252)
                    metrics["sharpe_ratio"] = float(
                        np.mean(simulated_returns) / np.std(simulated_returns) * annualization_factor
                    )

                    # Maximum drawdown
                    cumulative_returns = np.cumprod(1 + simulated_returns)
                    running_max = np.maximum.accumulate(cumulative_returns)
                    drawdown = (running_max - cumulative_returns) / running_max
                    metrics["max_drawdown"] = float(np.max(drawdown))

                    # Win rate
                    winning_trades = np.sum(simulated_returns > 0)
                    total_trades = len(simulated_returns)
                    metrics["win_rate"] = float(winning_trades / total_trades) if total_trades > 0 else 0.0

                    # Profit factor
                    gross_profit = np.sum(simulated_returns[simulated_returns > 0])
                    gross_loss = abs(np.sum(simulated_returns[simulated_returns < 0]))
                    metrics["profit_factor"] = float(
                        gross_profit / gross_loss if gross_loss > 1e-10 else float("inf")
                    )
                else:
                    metrics["sharpe_ratio"] = 0.0
                    metrics["max_drawdown"] = 0.0
                    metrics["win_rate"] = 0.0
                    metrics["profit_factor"] = 0.0
        else:
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        return metrics

    def get_best_model(
        self,
        model_name: str,
        metric: str = "sharpe_ratio",
    ) -> TradingModel:
        """
        Get the best performing model version.

        Args:
            model_name: Model name
            metric: Metric to compare

        Returns:
            Best model
        """
        versions = self.registry.list_models(model_name)
        if not versions:
            raise ValueError(f"No models found for {model_name}")

        # Sort by metric (descending)
        sorted_versions = sorted(
            [v for v in versions if v.get("is_active", True)],
            key=lambda v: v.get("metrics", {}).get(metric, float("-inf")),
            reverse=True,
        )

        if not sorted_versions:
            raise ValueError(f"No active models found for {model_name}")

        return self.registry.get_model(model_name, sorted_versions[0]["version_id"])
