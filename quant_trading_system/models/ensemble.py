"""
Ensemble methods for combining multiple trading models.

Implements voting, averaging, stacking, and dynamic ensembles
for improved prediction accuracy and robustness.
"""

from __future__ import annotations

import copy
import logging
from enum import Enum
from typing import Any

import numpy as np

from quant_trading_system.models.base import EnsembleModel, ModelType, TradingModel

logger = logging.getLogger(__name__)


def _clone_model(model: TradingModel) -> TradingModel:
    """Clone a model safely, handling GPU models correctly.

    CRITICAL FIX: copy.deepcopy() fails for PyTorch models on GPU.
    This function handles cloning by:
    1. For sklearn-based models: Use __class__ with same params
    2. For PyTorch models: Move to CPU, deepcopy, move back to GPU

    Args:
        model: Model to clone.

    Returns:
        A new model instance with the same parameters but unfitted.
    """
    try:
        # Check if model has a clone method (sklearn-style)
        if hasattr(model, 'clone'):
            return model.clone()

        # Check if it's a PyTorch-based model (has .device attribute)
        if hasattr(model, 'device') and hasattr(model, '_network'):
            import torch

            # Get the device
            device = model.device

            # For PyTorch models, create new instance with same params
            # Don't copy the fitted state - we want an unfitted clone
            model_class = model.__class__
            model_params = model._params.copy() if hasattr(model, '_params') else {}

            # Create new instance
            new_model = model_class(
                name=f"{model.name}_clone",
                **model_params
            )

            # Set the device
            if hasattr(new_model, 'device'):
                new_model.device = device

            return new_model

        # For other models, try to use __class__ with parameters
        if hasattr(model, '_params'):
            model_class = model.__class__
            model_params = model._params.copy()
            return model_class(
                name=f"{model.name}_clone",
                **model_params
            )

        # Fallback to deepcopy with proper handling
        # First try regular deepcopy
        try:
            return copy.deepcopy(model)
        except Exception as deepcopy_error:
            logger.warning(
                f"deepcopy failed for model {model.name}: {deepcopy_error}. "
                f"Attempting to create new instance from class."
            )
            # Last resort: create new instance
            model_class = model.__class__
            return model_class(name=f"{model.name}_clone")

    except Exception as e:
        logger.error(f"Failed to clone model {model.name}: {e}")
        raise RuntimeError(
            f"Cannot clone model {model.name}. GPU models require special handling. "
            f"Error: {e}"
        )


class AggregationMethod(str, Enum):
    """Prediction aggregation methods."""

    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    VOTING_HARD = "voting_hard"
    VOTING_SOFT = "voting_soft"


class VotingEnsemble(EnsembleModel):
    """
    Voting ensemble for classification.

    Combines predictions from multiple classifiers using
    hard voting (majority) or soft voting (probability average).
    """

    def __init__(
        self,
        name: str = "voting_ensemble",
        version: str = "1.0.0",
        voting: str = "soft",
        **kwargs: Any,
    ):
        """
        Initialize voting ensemble.

        Args:
            name: Model identifier
            version: Version string
            voting: "hard" for majority vote, "soft" for probability average
            **kwargs: Additional parameters
        """
        super().__init__(name, version, ModelType.CLASSIFIER, **kwargs)
        self._params["voting"] = voting

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "VotingEnsemble":
        """
        Train all base models.

        Args:
            X: Feature matrix
            y: Target array
            validation_data: Optional validation data
            sample_weights: Optional sample weights
            **kwargs: Additional parameters

        Returns:
            self for method chaining
        """
        for model in self._base_models:
            model.fit(X, y, validation_data, sample_weights, **kwargs)

        # Record training
        metrics = {"n_base_models": len(self._base_models)}
        feature_names = kwargs.get("feature_names", [f"f{i}" for i in range(X.shape[1])])
        self._record_training(metrics, feature_names)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using voting."""
        X = self._validate_input(X)

        if self._params["voting"] == "hard":
            # Hard voting - majority
            predictions = np.array([model.predict(X) for model in self._base_models])
            return np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(),
                axis=0,
                arr=predictions,
            )
        else:
            # Soft voting - average probabilities
            probas = self.predict_proba(X)
            return probas.argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability estimates by averaging."""
        X = self._validate_input(X)

        probas = []
        weights = self._weights if self._weights is not None else np.ones(len(self._base_models))

        for model, weight in zip(self._base_models, weights):
            try:
                proba = model.predict_proba(X)
                probas.append(proba * weight)
            except NotImplementedError:
                # Convert predictions to one-hot
                pred = model.predict(X)
                n_classes = len(np.unique(pred))
                one_hot = np.zeros((len(pred), n_classes))
                one_hot[np.arange(len(pred)), pred.astype(int)] = 1
                probas.append(one_hot * weight)

        # Weighted average
        avg_proba = np.sum(probas, axis=0) / np.sum(weights)
        return avg_proba


class AveragingEnsemble(EnsembleModel):
    """
    Averaging ensemble for regression.

    Combines predictions using various averaging methods
    (mean, weighted mean, median, trimmed mean).
    """

    def __init__(
        self,
        name: str = "averaging_ensemble",
        version: str = "1.0.0",
        method: AggregationMethod = AggregationMethod.WEIGHTED_MEAN,
        trim_ratio: float = 0.1,
        **kwargs: Any,
    ):
        """
        Initialize averaging ensemble.

        Args:
            name: Model identifier
            version: Version string
            method: Aggregation method
            trim_ratio: Fraction to trim for trimmed mean
            **kwargs: Additional parameters
        """
        super().__init__(name, version, ModelType.REGRESSOR, **kwargs)
        self._params["method"] = method.value
        self._params["trim_ratio"] = trim_ratio

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "AveragingEnsemble":
        """Train all base models."""
        for model in self._base_models:
            model.fit(X, y, validation_data, sample_weights, **kwargs)

        metrics = {"n_base_models": len(self._base_models)}
        feature_names = kwargs.get("feature_names", [f"f{i}" for i in range(X.shape[1])])
        self._record_training(metrics, feature_names)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using averaging."""
        X = self._validate_input(X)

        predictions = np.array([model.predict(X) for model in self._base_models])
        method = AggregationMethod(self._params["method"])

        if method == AggregationMethod.MEAN:
            return predictions.mean(axis=0)

        elif method == AggregationMethod.WEIGHTED_MEAN:
            weights = self._weights if self._weights is not None else np.ones(len(self._base_models))
            weights = weights / weights.sum()
            return np.average(predictions, axis=0, weights=weights)

        elif method == AggregationMethod.MEDIAN:
            return np.median(predictions, axis=0)

        elif method == AggregationMethod.TRIMMED_MEAN:
            from scipy import stats
            return stats.trim_mean(predictions, self._params["trim_ratio"], axis=0)

        else:
            raise ValueError(f"Unknown aggregation method: {method}")


class StackingEnsemble(EnsembleModel):
    """
    Stacking ensemble with meta-learner.

    Uses out-of-fold predictions from base models as features
    for a meta-learner that combines them optimally.
    """

    def __init__(
        self,
        name: str = "stacking_ensemble",
        version: str = "1.0.0",
        model_type: ModelType = ModelType.REGRESSOR,
        meta_learner: TradingModel | None = None,
        use_features: bool = True,
        n_folds: int = 5,
        **kwargs: Any,
    ):
        """
        Initialize stacking ensemble.

        Args:
            name: Model identifier
            version: Version string
            model_type: Classification or regression
            meta_learner: Model to combine base predictions (default: Ridge)
            use_features: Whether to include original features in meta-learner
            n_folds: Number of folds for OOF predictions
            **kwargs: Additional parameters
        """
        super().__init__(name, version, model_type, **kwargs)
        self._meta_learner = meta_learner
        self._params["use_features"] = use_features
        self._params["n_folds"] = n_folds

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "StackingEnsemble":
        """
        Train stacking ensemble.

        1. Generate out-of-fold predictions from base models
        2. Train meta-learner on OOF predictions
        3. Retrain base models on full data

        Args:
            X: Feature matrix
            y: Target array
            validation_data: Optional validation data
            sample_weights: Optional sample weights
            **kwargs: Additional parameters

        Returns:
            self for method chaining
        """
        # CRITICAL FIX: Use TimeSeriesSplit instead of KFold to prevent data leakage
        # KFold can shuffle data and use future data to predict past values
        # TimeSeriesSplit ensures chronological ordering is preserved
        from sklearn.model_selection import TimeSeriesSplit

        n_samples = X.shape[0]
        n_models = len(self._base_models)

        # Initialize meta-learner if not provided
        if self._meta_learner is None:
            from quant_trading_system.models.classical_ml import ElasticNetModel
            self._meta_learner = ElasticNetModel(
                name="meta_learner",
                alpha=0.1,
                l1_ratio=0.5,
            )

        # Generate OOF predictions using TIME-SERIES SPLIT
        # This ensures we never use future data to make predictions
        oof_predictions = np.zeros((n_samples, n_models))
        tscv = TimeSeriesSplit(n_splits=self._params["n_folds"])

        for model_idx, model in enumerate(self._base_models):
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]

                # CRITICAL FIX: Use proper model cloning that handles GPU models
                # copy.deepcopy() fails for PyTorch models on GPU (CUDA tensors
                # cannot be pickled, and model state may reference GPU memory)
                fold_model = _clone_model(model)
                fold_model.fit(X_train, y_train, **kwargs)
                oof_predictions[val_idx, model_idx] = fold_model.predict(X_val)

        # Prepare meta-features
        if self._params["use_features"]:
            meta_features = np.hstack([X, oof_predictions])
        else:
            meta_features = oof_predictions

        # Train meta-learner
        self._meta_learner.fit(meta_features, y, **kwargs)

        # Retrain base models on full data
        for model in self._base_models:
            model.fit(X, y, validation_data, sample_weights, **kwargs)

        # Record training
        metrics = {
            "n_base_models": n_models,
            "n_folds": self._params["n_folds"],
            "use_features": self._params["use_features"],
        }
        feature_names = kwargs.get("feature_names", [f"f{i}" for i in range(X.shape[1])])
        self._record_training(metrics, feature_names)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using stacking."""
        X = self._validate_input(X)

        # Get base model predictions
        base_predictions = np.column_stack([
            model.predict(X) for model in self._base_models
        ])

        # Prepare meta-features
        if self._params["use_features"]:
            meta_features = np.hstack([X, base_predictions])
        else:
            meta_features = base_predictions

        return self._meta_learner.predict(meta_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability estimates (classification only)."""
        if self._model_type != ModelType.CLASSIFIER:
            raise NotImplementedError("predict_proba only available for classifiers")

        X = self._validate_input(X)

        base_predictions = np.column_stack([
            model.predict(X) for model in self._base_models
        ])

        if self._params["use_features"]:
            meta_features = np.hstack([X, base_predictions])
        else:
            meta_features = base_predictions

        return self._meta_learner.predict_proba(meta_features)


class AdaptiveEnsemble(EnsembleModel):
    """
    Adaptive ensemble that adjusts weights based on recent performance.

    Tracks rolling performance of each model and increases weights
    for models performing well in recent periods.
    """

    def __init__(
        self,
        name: str = "adaptive_ensemble",
        version: str = "1.0.0",
        model_type: ModelType = ModelType.REGRESSOR,
        window_size: int = 50,
        min_weight: float = 0.1,
        adaptation_rate: float = 0.1,
        **kwargs: Any,
    ):
        """
        Initialize adaptive ensemble.

        Args:
            name: Model identifier
            version: Version string
            model_type: Classification or regression
            window_size: Rolling window for performance tracking
            min_weight: Minimum weight floor
            adaptation_rate: How quickly weights adapt
            **kwargs: Additional parameters
        """
        super().__init__(name, version, model_type, **kwargs)
        self._params["window_size"] = window_size
        self._params["min_weight"] = min_weight
        self._params["adaptation_rate"] = adaptation_rate

        self._performance_history: list[dict[int, float]] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "AdaptiveEnsemble":
        """Train all base models."""
        for model in self._base_models:
            model.fit(X, y, validation_data, sample_weights, **kwargs)

        # Initialize equal weights
        n_models = len(self._base_models)
        self._weights = np.ones(n_models) / n_models

        metrics = {"n_base_models": n_models}
        feature_names = kwargs.get("feature_names", [f"f{i}" for i in range(X.shape[1])])
        self._record_training(metrics, feature_names)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate weighted predictions."""
        X = self._validate_input(X)

        predictions = np.array([model.predict(X) for model in self._base_models])
        weights = self._weights / self._weights.sum()

        return np.average(predictions, axis=0, weights=weights)

    def update_weights(self, y_true: np.ndarray, predictions: dict[int, np.ndarray]) -> None:
        """
        Update model weights based on performance.

        Args:
            y_true: True values
            predictions: Dictionary mapping model index to predictions
        """
        # Calculate errors for each model
        errors = {}
        for model_idx, pred in predictions.items():
            if self._model_type == ModelType.CLASSIFIER:
                errors[model_idx] = 1 - (pred == y_true).mean()
            else:
                errors[model_idx] = np.mean((pred - y_true) ** 2)

        self._performance_history.append(errors)

        # Keep only recent history
        if len(self._performance_history) > self._params["window_size"]:
            self._performance_history = self._performance_history[-self._params["window_size"]:]

        # Calculate average errors over window
        avg_errors = np.zeros(len(self._base_models))
        for record in self._performance_history:
            for model_idx, error in record.items():
                avg_errors[model_idx] += error
        avg_errors /= len(self._performance_history)

        # Convert errors to weights (lower error = higher weight)
        # Using inverse error with softmax
        inv_errors = 1 / (avg_errors + 1e-8)
        new_weights = inv_errors / inv_errors.sum()

        # Apply minimum weight floor
        new_weights = np.maximum(new_weights, self._params["min_weight"])
        new_weights /= new_weights.sum()

        # Smooth weight updates
        alpha = self._params["adaptation_rate"]
        self._weights = (1 - alpha) * self._weights + alpha * new_weights


class RegimeAwareEnsemble(EnsembleModel):
    """
    Regime-aware ensemble that uses different weights for different market regimes.

    Learns to detect market regimes and applies regime-specific
    model weights for optimal performance.
    """

    def __init__(
        self,
        name: str = "regime_ensemble",
        version: str = "1.0.0",
        model_type: ModelType = ModelType.REGRESSOR,
        n_regimes: int = 3,
        **kwargs: Any,
    ):
        """
        Initialize regime-aware ensemble.

        Args:
            name: Model identifier
            version: Version string
            model_type: Classification or regression
            n_regimes: Number of market regimes to detect
            **kwargs: Additional parameters
        """
        super().__init__(name, version, model_type, **kwargs)
        self._params["n_regimes"] = n_regimes
        self._regime_weights: dict[int, np.ndarray] = {}
        self._regime_model: Any = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weights: np.ndarray | None = None,
        regime_features: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "RegimeAwareEnsemble":
        """
        Train regime-aware ensemble.

        Args:
            X: Feature matrix
            y: Target array
            validation_data: Optional validation data
            sample_weights: Optional sample weights
            regime_features: Features for regime detection (default: use X)
            **kwargs: Additional parameters

        Returns:
            self for method chaining
        """
        from sklearn.mixture import GaussianMixture

        # Train base models
        for model in self._base_models:
            model.fit(X, y, validation_data, sample_weights, **kwargs)

        # Detect regimes using GMM
        regime_X = regime_features if regime_features is not None else X
        self._regime_model = GaussianMixture(
            n_components=self._params["n_regimes"],
            random_state=42,
        )
        regimes = self._regime_model.fit_predict(regime_X)

        # Learn regime-specific weights
        n_models = len(self._base_models)

        for regime in range(self._params["n_regimes"]):
            regime_mask = regimes == regime
            if regime_mask.sum() < 10:  # Not enough samples
                self._regime_weights[regime] = np.ones(n_models) / n_models
                continue

            X_regime = X[regime_mask]
            y_regime = y[regime_mask]

            # Evaluate each model on this regime
            errors = []
            for model in self._base_models:
                pred = model.predict(X_regime)
                if self._model_type == ModelType.CLASSIFIER:
                    error = 1 - (pred == y_regime).mean()
                else:
                    error = np.mean((pred - y_regime) ** 2)
                errors.append(error)

            errors = np.array(errors)
            # Convert to weights (lower error = higher weight)
            inv_errors = 1 / (errors + 1e-8)
            weights = inv_errors / inv_errors.sum()
            self._regime_weights[regime] = weights

        metrics = {
            "n_base_models": n_models,
            "n_regimes": self._params["n_regimes"],
        }
        feature_names = kwargs.get("feature_names", [f"f{i}" for i in range(X.shape[1])])
        self._record_training(metrics, feature_names)

        return self

    def predict(self, X: np.ndarray, regime_features: np.ndarray | None = None) -> np.ndarray:
        """
        Generate regime-aware predictions.

        Args:
            X: Feature matrix
            regime_features: Features for regime detection

        Returns:
            Predictions array
        """
        X = self._validate_input(X)

        # Detect current regime
        regime_X = regime_features if regime_features is not None else X
        regimes = self._regime_model.predict(regime_X)

        # Get base predictions
        predictions = np.array([model.predict(X) for model in self._base_models])

        # Apply regime-specific weights
        result = np.zeros(X.shape[0])
        for regime in range(self._params["n_regimes"]):
            regime_mask = regimes == regime
            if regime_mask.sum() == 0:
                continue

            weights = self._regime_weights.get(
                regime,
                np.ones(len(self._base_models)) / len(self._base_models),
            )
            weights = weights / weights.sum()

            regime_pred = np.average(predictions[:, regime_mask], axis=0, weights=weights)
            result[regime_mask] = regime_pred

        return result


def create_model_ensemble(
    models: list[TradingModel],
    method: str = "stacking",
    model_type: ModelType = ModelType.REGRESSOR,
    **kwargs: Any,
) -> EnsembleModel:
    """
    Factory function to create ensemble models.

    Args:
        models: List of base models
        method: Ensemble method ("voting", "averaging", "stacking", "adaptive", "regime")
        model_type: Classification or regression
        **kwargs: Additional parameters for the ensemble

    Returns:
        Configured ensemble model
    """
    if method == "voting":
        ensemble = VotingEnsemble(**kwargs)
    elif method == "averaging":
        ensemble = AveragingEnsemble(**kwargs)
    elif method == "stacking":
        ensemble = StackingEnsemble(model_type=model_type, **kwargs)
    elif method == "adaptive":
        ensemble = AdaptiveEnsemble(model_type=model_type, **kwargs)
    elif method == "regime":
        ensemble = RegimeAwareEnsemble(model_type=model_type, **kwargs)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    for model in models:
        ensemble.add_model(model)

    return ensemble
