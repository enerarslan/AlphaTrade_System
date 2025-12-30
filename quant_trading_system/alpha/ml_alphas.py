"""
ML-based alpha factors module.

Implements machine learning alpha factors that use trained models
to generate trading signals from features.
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .alpha_base import AlphaFactor, AlphaHorizon, AlphaType


class MLAlpha(AlphaFactor):
    """
    Base class for ML-based alpha factors.

    Provides common functionality for ML alphas including:
    - Feature preparation
    - Model persistence
    - Prediction generation
    """

    def __init__(
        self,
        name: str,
        feature_names: list[str],
        horizon: AlphaHorizon = AlphaHorizon.MEDIUM,
        prediction_type: str = "regression",  # 'regression' or 'classification'
    ):
        super().__init__(
            name=name,
            alpha_type=AlphaType.ML_BASED,
            horizon=horizon,
        )
        self.feature_names = feature_names
        self.prediction_type = prediction_type
        self._model: Any = None
        self._scaler: Any = None

    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying ML model."""
        pass

    @abstractmethod
    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to training data."""
        pass

    @abstractmethod
    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from the model."""
        pass

    def fit(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray],
        target: np.ndarray,
        validation_split: float = 0.2,
    ) -> "MLAlpha":
        """
        Fit the ML model.

        Args:
            df: DataFrame with OHLCV data
            features: Dictionary of feature arrays
            target: Target variable array
            validation_split: Fraction of data for validation

        Returns:
            self
        """
        # Prepare feature matrix
        X = self._prepare_features(features)

        # Remove NaN rows
        valid_mask = ~(np.any(np.isnan(X), axis=1) | np.isnan(target))
        X_valid = X[valid_mask]
        y_valid = target[valid_mask]

        if len(X_valid) < 100:
            raise ValueError("Insufficient valid samples for training")

        # Split for validation (time-series aware)
        split_idx = int(len(X_valid) * (1 - validation_split))
        X_train = X_valid[:split_idx]
        y_train = y_valid[:split_idx]

        # Fit scaler
        self._fit_scaler(X_train)

        # Scale features
        X_train_scaled = self._scale_features(X_train)

        # Create and fit model
        self._model = self._create_model()
        self._fit_model(X_train_scaled, y_train)

        self._is_fitted = True
        return self

    def compute(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute ML alpha from model predictions."""
        if not self._is_fitted or self._model is None:
            return np.full(len(df), np.nan)

        if features is None:
            raise ValueError("Features required for ML alpha computation")

        # Prepare features
        X = self._prepare_features(features)
        n = len(X)

        # Handle NaN values
        valid_mask = ~np.any(np.isnan(X), axis=1)
        alpha = np.full(n, np.nan)

        if np.sum(valid_mask) == 0:
            return alpha

        # Scale and predict
        X_valid = X[valid_mask]
        X_scaled = self._scale_features(X_valid)
        predictions = self._predict_model(X_scaled)

        # Convert predictions to alpha
        if self.prediction_type == "classification":
            # Probability of positive class minus 0.5, scaled to [-1, 1]
            alpha[valid_mask] = (predictions - 0.5) * 2
        else:
            # Normalize regression predictions
            alpha[valid_mask] = self._normalize(predictions)

        return alpha

    def _prepare_features(self, features: dict[str, np.ndarray]) -> np.ndarray:
        """Prepare feature matrix from feature dictionary."""
        feature_arrays = []
        for name in self.feature_names:
            if name in features:
                feature_arrays.append(features[name])
            else:
                raise ValueError(f"Required feature '{name}' not found")
        return np.column_stack(feature_arrays)

    def _fit_scaler(self, X: np.ndarray) -> None:
        """Fit the feature scaler."""
        self._scaler = {
            "mean": np.nanmean(X, axis=0),
            "std": np.nanstd(X, axis=0) + 1e-10,
        }

    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features using fitted scaler."""
        if self._scaler is None:
            return X
        return (X - self._scaler["mean"]) / self._scaler["std"]

    def save(self, path: Path | str) -> None:
        """Save model to disk."""
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": self._model,
                "scaler": self._scaler,
                "feature_names": self.feature_names,
                "metadata": {
                    "name": self.name,
                    "prediction_type": self.prediction_type,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                },
            },
            path / f"{self.name}.joblib",
        )

    def load(self, path: Path | str) -> "MLAlpha":
        """Load model from disk."""
        import joblib

        path = Path(path)
        data = joblib.load(path / f"{self.name}.joblib")

        self._model = data["model"]
        self._scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self._is_fitted = True

        return self

    def get_params(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "prediction_type": self.prediction_type,
            "is_fitted": self._is_fitted,
        }


class XGBoostAlpha(MLAlpha):
    """
    XGBoost-based alpha factor.

    Uses XGBoost for gradient boosting predictions.
    """

    def __init__(
        self,
        feature_names: list[str],
        name: str = "xgb_alpha",
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        prediction_type: str = "regression",
    ):
        super().__init__(
            name=name,
            feature_names=feature_names,
            prediction_type=prediction_type,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def _create_model(self) -> Any:
        """Create XGBoost model."""
        try:
            import xgboost as xgb

            if self.prediction_type == "classification":
                return xgb.XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=42,
                    n_jobs=-1,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )
            else:
                return xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=42,
                    n_jobs=-1,
                )
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit XGBoost model."""
        self._model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if self.prediction_type == "classification":
            return self._model.predict_proba(X)[:, 1]
        else:
            return self._model.predict(X)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if self._model is None:
            return {}
        importances = self._model.feature_importances_
        return dict(zip(self.feature_names, importances))


class LightGBMAlpha(MLAlpha):
    """
    LightGBM-based alpha factor.

    Uses LightGBM for gradient boosting predictions.
    """

    def __init__(
        self,
        feature_names: list[str],
        name: str = "lgbm_alpha",
        n_estimators: int = 100,
        num_leaves: int = 31,
        learning_rate: float = 0.1,
        prediction_type: str = "regression",
    ):
        super().__init__(
            name=name,
            feature_names=feature_names,
            prediction_type=prediction_type,
        )
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate

    def _create_model(self) -> Any:
        """Create LightGBM model."""
        try:
            import lightgbm as lgb

            if self.prediction_type == "classification":
                return lgb.LGBMClassifier(
                    n_estimators=self.n_estimators,
                    num_leaves=self.num_leaves,
                    learning_rate=self.learning_rate,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
            else:
                return lgb.LGBMRegressor(
                    n_estimators=self.n_estimators,
                    num_leaves=self.num_leaves,
                    learning_rate=self.learning_rate,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit LightGBM model."""
        self._model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if self.prediction_type == "classification":
            return self._model.predict_proba(X)[:, 1]
        else:
            return self._model.predict(X)


class RandomForestAlpha(MLAlpha):
    """
    Random Forest-based alpha factor.
    """

    def __init__(
        self,
        feature_names: list[str],
        name: str = "rf_alpha",
        n_estimators: int = 100,
        max_depth: int | None = 10,
        prediction_type: str = "regression",
    ):
        super().__init__(
            name=name,
            feature_names=feature_names,
            prediction_type=prediction_type,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def _create_model(self) -> Any:
        """Create Random Forest model."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if self.prediction_type == "classification":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1,
            )
        else:
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1,
            )

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Random Forest model."""
        self._model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if self.prediction_type == "classification":
            return self._model.predict_proba(X)[:, 1]
        else:
            return self._model.predict(X)


class LinearAlpha(MLAlpha):
    """
    Linear model-based alpha factor (Ridge/Lasso/ElasticNet).
    """

    def __init__(
        self,
        feature_names: list[str],
        name: str = "linear_alpha",
        model_type: str = "ridge",  # 'ridge', 'lasso', 'elasticnet'
        alpha: float = 1.0,
        l1_ratio: float = 0.5,  # For elasticnet
    ):
        super().__init__(
            name=name,
            feature_names=feature_names,
            prediction_type="regression",
        )
        self.model_type = model_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def _create_model(self) -> Any:
        """Create linear model."""
        from sklearn.linear_model import ElasticNet, Lasso, Ridge

        if self.model_type == "ridge":
            return Ridge(alpha=self.alpha)
        elif self.model_type == "lasso":
            return Lasso(alpha=self.alpha)
        elif self.model_type == "elasticnet":
            return ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)
        else:
            return Ridge(alpha=self.alpha)

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit linear model."""
        self._model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self._model.predict(X)

    def get_coefficients(self) -> dict[str, float]:
        """Get model coefficients."""
        if self._model is None:
            return {}
        return dict(zip(self.feature_names, self._model.coef_))


class NeuralNetAlpha(MLAlpha):
    """
    Neural network-based alpha factor.

    Uses a simple feedforward neural network for predictions.
    """

    def __init__(
        self,
        feature_names: list[str],
        name: str = "nn_alpha",
        hidden_layers: list[int] | None = None,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        prediction_type: str = "regression",
    ):
        super().__init__(
            name=name,
            feature_names=feature_names,
            prediction_type=prediction_type,
        )
        self.hidden_layers = hidden_layers or [64, 32]
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size

    def _create_model(self) -> Any:
        """Create neural network model."""
        try:
            import torch
            import torch.nn as nn

            class FeedForwardNet(nn.Module):
                def __init__(
                    self,
                    input_dim: int,
                    hidden_layers: list[int],
                    dropout: float,
                    output_dim: int,
                ):
                    super().__init__()
                    layers = []
                    prev_dim = input_dim

                    for hidden_dim in hidden_layers:
                        layers.append(nn.Linear(prev_dim, hidden_dim))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(dropout))
                        prev_dim = hidden_dim

                    layers.append(nn.Linear(prev_dim, output_dim))
                    self.network = nn.Sequential(*layers)

                def forward(self, x):
                    return self.network(x)

            input_dim = len(self.feature_names)
            output_dim = 1

            return FeedForwardNet(
                input_dim, self.hidden_layers, self.dropout, output_dim
            )
        except ImportError:
            raise ImportError("PyTorch not installed. Run: pip install torch")

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit neural network model."""
        import torch
        import torch.nn as nn

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)

        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(device)

        if self.prediction_type == "classification":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)

        self._model.train()
        for epoch in range(self.epochs):
            # Mini-batch training
            permutation = torch.randperm(X_tensor.size(0))

            for i in range(0, X_tensor.size(0), self.batch_size):
                indices = permutation[i : i + self.batch_size]
                batch_x = X_tensor[indices]
                batch_y = y_tensor[indices]

                optimizer.zero_grad()
                outputs = self._model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            outputs = self._model(X_tensor).cpu().numpy().flatten()

            if self.prediction_type == "classification":
                return torch.sigmoid(torch.tensor(outputs)).numpy()
            return outputs


class EnsembleMLAlpha(MLAlpha):
    """
    Ensemble of ML models for alpha generation.

    Combines predictions from multiple models.
    """

    def __init__(
        self,
        feature_names: list[str],
        name: str = "ensemble_ml_alpha",
        models: list[str] | None = None,
        weights: list[float] | None = None,
        prediction_type: str = "regression",
    ):
        super().__init__(
            name=name,
            feature_names=feature_names,
            prediction_type=prediction_type,
        )
        self.model_types = models or ["xgb", "lgbm", "rf"]
        self.weights = weights or [1.0 / len(self.model_types)] * len(self.model_types)
        self._models: list[Any] = []

    def _create_model(self) -> Any:
        """Create ensemble of models."""
        return None  # Models created in fit

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit all models in ensemble."""
        self._models = []

        for model_type in self.model_types:
            if model_type == "xgb":
                model = XGBoostAlpha(
                    feature_names=self.feature_names,
                    prediction_type=self.prediction_type,
                )
            elif model_type == "lgbm":
                model = LightGBMAlpha(
                    feature_names=self.feature_names,
                    prediction_type=self.prediction_type,
                )
            elif model_type == "rf":
                model = RandomForestAlpha(
                    feature_names=self.feature_names,
                    prediction_type=self.prediction_type,
                )
            else:
                continue

            model._model = model._create_model()
            model._fit_model(X, y)
            self._models.append(model)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions."""
        predictions = []
        for model in self._models:
            pred = model._predict_model(X)
            predictions.append(pred)

        # Weighted average
        weighted_pred = np.zeros(len(X))
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += weight * pred

        return weighted_pred


class FeatureImportanceAlpha(MLAlpha):
    """
    Alpha that focuses on top-k important features.

    Uses feature importance to select most predictive features.
    """

    def __init__(
        self,
        all_features: list[str],
        name: str = "fi_alpha",
        top_k: int = 20,
        base_model: str = "xgb",
        prediction_type: str = "regression",
    ):
        super().__init__(
            name=name,
            feature_names=all_features,  # Will be reduced after importance calculation
            prediction_type=prediction_type,
        )
        self.all_features = all_features
        self.top_k = top_k
        self.base_model = base_model
        self._selected_features: list[str] = []

    def _create_model(self) -> Any:
        """Create base model for feature selection and prediction."""
        if self.base_model == "xgb":
            return XGBoostAlpha(
                feature_names=self._selected_features or self.all_features,
                prediction_type=self.prediction_type,
            )
        elif self.base_model == "lgbm":
            return LightGBMAlpha(
                feature_names=self._selected_features or self.all_features,
                prediction_type=self.prediction_type,
            )
        else:
            return RandomForestAlpha(
                feature_names=self._selected_features or self.all_features,
                prediction_type=self.prediction_type,
            )

    def fit(
        self,
        df: pl.DataFrame,
        features: dict[str, np.ndarray],
        target: np.ndarray,
        validation_split: float = 0.2,
    ) -> "FeatureImportanceAlpha":
        """Fit with feature selection."""
        # First pass: fit with all features
        temp_model = self._create_model()
        temp_model._model = temp_model._create_model()

        X = self._prepare_features(features)
        valid_mask = ~(np.any(np.isnan(X), axis=1) | np.isnan(target))
        X_valid = X[valid_mask]
        y_valid = target[valid_mask]

        split_idx = int(len(X_valid) * (1 - validation_split))
        X_train = X_valid[:split_idx]
        y_train = y_valid[:split_idx]

        # Scale and fit
        self._fit_scaler(X_train)
        X_train_scaled = self._scale_features(X_train)
        temp_model._fit_model(X_train_scaled, y_train)

        # Get feature importance
        importances = temp_model.get_feature_importance()

        # Select top-k features
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        self._selected_features = [f[0] for f in sorted_features[: self.top_k]]
        self.feature_names = self._selected_features

        # Second pass: fit with selected features
        super().fit(df, features, target, validation_split)

        return self

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit model with selected features."""
        self._model = self._create_model()
        self._model._model = self._model._create_model()
        self._model._fit_model(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self._model._predict_model(X)


def create_ml_alphas(feature_names: list[str]) -> list[MLAlpha]:
    """Create a set of ML alphas with given features."""
    return [
        XGBoostAlpha(
            feature_names=feature_names,
            name="xgb_alpha_reg",
            prediction_type="regression",
        ),
        XGBoostAlpha(
            feature_names=feature_names,
            name="xgb_alpha_cls",
            prediction_type="classification",
        ),
        LightGBMAlpha(
            feature_names=feature_names,
            name="lgbm_alpha_reg",
            prediction_type="regression",
        ),
        RandomForestAlpha(
            feature_names=feature_names,
            name="rf_alpha_reg",
            prediction_type="regression",
        ),
        LinearAlpha(
            feature_names=feature_names,
            name="ridge_alpha",
            model_type="ridge",
        ),
        EnsembleMLAlpha(
            feature_names=feature_names,
            name="ensemble_alpha",
            models=["xgb", "lgbm", "rf"],
        ),
    ]
