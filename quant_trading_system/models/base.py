"""
Abstract base model class for trading models.

Defines the interface that all models (classical ML, deep learning, RL) must implement.
Provides common functionality for training, prediction, persistence, and feature importance.

GPU ACCELERATION:
- Device property for consistent GPU handling across all model types
- Helper function for device selection (CUDA > MPS > CPU)
- Support for mixed precision training (AMP)
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# MAJOR FIX: Model versioning enforcement
# System version for model compatibility checking
# Format: MAJOR.MINOR.PATCH following semantic versioning
SYSTEM_MODEL_VERSION = "1.0.0"


@dataclass
class SemanticVersion:
    """Semantic version representation for comparison."""
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """
        Parse version string into SemanticVersion.

        Args:
            version_str: Version string like "1.2.3" or "1.2"

        Returns:
            SemanticVersion instance

        Raises:
            ValueError: If version string is invalid
        """
        if not version_str:
            raise ValueError("Version string cannot be empty")

        # Match MAJOR.MINOR.PATCH or MAJOR.MINOR
        match = re.match(r'^(\d+)\.(\d+)(?:\.(\d+))?$', version_str.strip())
        if not match:
            raise ValueError(f"Invalid version format: {version_str}. Expected MAJOR.MINOR.PATCH")

        major = int(match.group(1))
        minor = int(match.group(2))
        patch = int(match.group(3)) if match.group(3) else 0

        return cls(major=major, minor=minor, patch=patch)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __lt__(self, other: "SemanticVersion") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other: "SemanticVersion") -> bool:
        return self == other or self < other

    def __gt__(self, other: "SemanticVersion") -> bool:
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)

    def __ge__(self, other: "SemanticVersion") -> bool:
        return self == other or self > other


class VersionCompatibility(str, Enum):
    """Model version compatibility levels."""
    COMPATIBLE = "compatible"          # Fully compatible
    MINOR_MISMATCH = "minor_mismatch"  # Minor version differs (warning)
    MAJOR_MISMATCH = "major_mismatch"  # Major version differs (error)
    FUTURE_VERSION = "future_version"  # Model from newer system (error)


def check_version_compatibility(
    model_version: str,
    system_version: str = SYSTEM_MODEL_VERSION,
    strict: bool = False,
) -> tuple[VersionCompatibility, str]:
    """
    Check if model version is compatible with system version.

    Compatibility rules:
    - Same MAJOR.MINOR.PATCH: COMPATIBLE
    - Same MAJOR, different MINOR/PATCH (model older): MINOR_MISMATCH (warning)
    - Different MAJOR: MAJOR_MISMATCH (error in strict mode)
    - Model version newer than system: FUTURE_VERSION (error)

    Args:
        model_version: Model's version string
        system_version: Current system version
        strict: If True, treat any mismatch as error

    Returns:
        Tuple of (VersionCompatibility, message)
    """
    try:
        model_ver = SemanticVersion.parse(model_version)
        system_ver = SemanticVersion.parse(system_version)
    except ValueError as e:
        return VersionCompatibility.MAJOR_MISMATCH, f"Version parse error: {e}"

    # Exact match
    if model_ver == system_ver:
        return VersionCompatibility.COMPATIBLE, f"Model version {model_version} matches system version"

    # Model from future system version
    if model_ver > system_ver:
        return VersionCompatibility.FUTURE_VERSION, (
            f"Model version {model_version} is newer than system version {system_version}. "
            f"Model may use features not available in this system."
        )

    # Major version mismatch
    if model_ver.major != system_ver.major:
        return VersionCompatibility.MAJOR_MISMATCH, (
            f"Major version mismatch: model={model_version}, system={system_version}. "
            f"Model may be incompatible with current system architecture."
        )

    # Minor/patch mismatch (model is older)
    if strict:
        return VersionCompatibility.MINOR_MISMATCH, (
            f"Version mismatch (strict mode): model={model_version}, system={system_version}. "
        )

    return VersionCompatibility.MINOR_MISMATCH, (
        f"Minor version mismatch: model={model_version}, system={system_version}. "
        f"Model should be compatible but consider retraining."
    )


class ModelType(str, Enum):
    """Model type enumeration."""

    CLASSIFIER = "CLASSIFIER"
    REGRESSOR = "REGRESSOR"


class DeviceType(str, Enum):
    """Compute device type enumeration."""

    CUDA = "cuda"  # NVIDIA GPU
    MPS = "mps"    # Apple Metal (M1/M2/M3)
    CPU = "cpu"    # CPU fallback


def get_optimal_device() -> str:
    """
    Get the optimal compute device for PyTorch models.

    Priority: CUDA (NVIDIA) > MPS (Apple Metal) > CPU

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA GPU detected: {device_name}")
            return DeviceType.CUDA.value

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS (Metal Performance Shaders) detected")
            return DeviceType.MPS.value

        logger.info("No GPU detected, using CPU")
        return DeviceType.CPU.value

    except ImportError:
        # PyTorch not installed (classical ML models)
        return DeviceType.CPU.value


class TradingModel(ABC):
    """
    Abstract base class for all trading models.

    All models (XGBoost, LSTM, Transformer, PPO, etc.) must inherit from this class
    and implement the abstract methods. This ensures a consistent interface for
    training, prediction, and model management.

    Attributes:
        name: Model identifier name
        version: Semantic version string (MAJOR.MINOR.PATCH)
        model_type: Whether model is classifier or regressor
        is_fitted: Whether model has been trained
        feature_names: List of expected feature names
        training_timestamp: When model was last trained
        training_metrics: Metrics from training
    """

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        model_type: ModelType = ModelType.REGRESSOR,
        **kwargs: Any,
    ):
        """
        Initialize base model.

        Args:
            name: Model identifier name
            version: Semantic version string
            model_type: Classification or regression
            **kwargs: Additional model-specific parameters
        """
        self._name = name
        self._version = version
        self._model_type = model_type
        self._is_fitted = False
        self._feature_names: list[str] = []
        self._training_timestamp: datetime | None = None
        self._training_metrics: dict[str, float] = {}
        self._params: dict[str, Any] = kwargs
        self._model: Any = None  # Underlying model object

    @property
    def name(self) -> str:
        """Get model name."""
        return self._name

    @property
    def version(self) -> str:
        """Get model version."""
        return self._version

    @property
    def model_type(self) -> ModelType:
        """Get model type."""
        return self._model_type

    @property
    def is_fitted(self) -> bool:
        """Check if model has been trained."""
        return self._is_fitted

    @property
    def feature_names(self) -> list[str]:
        """Get expected feature names."""
        return self._feature_names.copy()

    @feature_names.setter
    def feature_names(self, names: list[str]) -> None:
        """Set expected feature names."""
        self._feature_names = list(names)

    @property
    def training_timestamp(self) -> datetime | None:
        """Get training timestamp."""
        return self._training_timestamp

    @property
    def training_metrics(self) -> dict[str, float]:
        """Get training metrics."""
        return self._training_metrics.copy()

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "TradingModel":
        """
        Train model on feature matrix and targets.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
            validation_data: Optional tuple of (X_val, y_val) for early stopping
            sample_weights: Optional sample weights
            **kwargs: Additional training parameters

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples,)
        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate probability estimates (classification only).

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Probability estimates of shape (n_samples, n_classes)

        Raises:
            NotImplementedError: If model doesn't support probability estimates
        """
        raise NotImplementedError(f"{self._name} does not support probability estimates")

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass

    def get_params(self) -> dict[str, Any]:
        """
        Get model hyperparameters.

        Returns:
            Dictionary of hyperparameters
        """
        return self._params.copy()

    def set_params(self, **params: Any) -> "TradingModel":
        """
        Update hyperparameters.

        Args:
            **params: Parameters to update

        Returns:
            self for method chaining
        """
        self._params.update(params)
        return self

    def save(self, path: str | Path) -> None:
        """
        Serialize model to disk.

        Saves the model along with metadata including feature names,
        training timestamp, metrics, and version info.

        Args:
            path: Path to save model (without extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model artifact
        model_path = path.with_suffix(".pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self._model, f)

        # Save metadata
        # MAJOR FIX: Include system version for compatibility checking
        metadata = {
            "name": self._name,
            "version": self._version,
            "system_version": SYSTEM_MODEL_VERSION,  # Track system version at save time
            "model_type": self._model_type.value,
            "is_fitted": self._is_fitted,
            "feature_names": self._feature_names,
            "training_timestamp": self._training_timestamp.isoformat() if self._training_timestamp else None,
            "training_metrics": self._training_metrics,
            "params": self._serialize_params(self._params),
            "checksum": self._compute_checksum(model_path),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def load(
        self,
        path: str | Path,
        strict_version: bool = False,
        allow_minor_mismatch: bool = True,
    ) -> "TradingModel":
        """
        Deserialize model from disk with version compatibility checking.

        Loads the model, verifies checksum integrity, and checks version compatibility.

        MAJOR FIX: Added model versioning enforcement to prevent loading
        incompatible models that could produce incorrect predictions.

        Args:
            path: Path to saved model (without extension)
            strict_version: If True, require exact version match
            allow_minor_mismatch: If True, allow minor version differences with warning

        Returns:
            self for method chaining

        Raises:
            FileNotFoundError: If model files don't exist
            ValueError: If checksum verification fails or version incompatible
        """
        path = Path(path)
        model_path = path.with_suffix(".pkl")
        metadata_path = path.with_suffix(".json")

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Load and verify metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # MAJOR FIX: Version compatibility check
        saved_system_version = metadata.get("system_version", "0.0.0")
        compatibility, message = check_version_compatibility(
            model_version=saved_system_version,
            system_version=SYSTEM_MODEL_VERSION,
            strict=strict_version,
        )

        if compatibility == VersionCompatibility.COMPATIBLE:
            logger.debug(f"Model version check passed: {message}")
        elif compatibility == VersionCompatibility.MINOR_MISMATCH:
            if allow_minor_mismatch:
                logger.warning(f"Model version warning: {message}")
            else:
                raise ValueError(f"Model version incompatible: {message}")
        elif compatibility in (VersionCompatibility.MAJOR_MISMATCH, VersionCompatibility.FUTURE_VERSION):
            logger.error(f"Model version error: {message}")
            raise ValueError(f"Model version incompatible: {message}")

        # Verify checksum
        current_checksum = self._compute_checksum(model_path)
        if metadata.get("checksum") and metadata["checksum"] != current_checksum:
            raise ValueError("Model checksum verification failed - file may be corrupted")

        # SECURITY WARNING: pickle.load() can execute arbitrary code.
        # Only load model files from trusted sources.
        # The checksum verification above provides some integrity check,
        # but does NOT protect against malicious files created with valid checksums.
        # For production use, consider using safer serialization formats like
        # safetensors for neural networks or joblib with mmap_mode for sklearn models.
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)

        # Restore metadata
        self._name = metadata["name"]
        self._version = metadata["version"]
        self._model_type = ModelType(metadata["model_type"])
        self._is_fitted = metadata["is_fitted"]
        self._feature_names = metadata["feature_names"]
        self._training_timestamp = (
            datetime.fromisoformat(metadata["training_timestamp"])
            if metadata["training_timestamp"]
            else None
        )
        self._training_metrics = metadata["training_metrics"]
        self._params = metadata.get("params", {})

        # Store loaded system version for reference
        self._loaded_system_version = saved_system_version

        logger.info(
            f"Loaded model {self._name} v{self._version} "
            f"(trained on system v{saved_system_version}, current system v{SYSTEM_MODEL_VERSION})"
        )

        return self

    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _serialize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Serialize parameters to JSON-compatible format."""
        serialized = {}
        for key, value in params.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                serialized[key] = value
            elif isinstance(value, (list, tuple)):
                serialized[key] = list(value)
            elif isinstance(value, dict):
                serialized[key] = self._serialize_params(value)
            else:
                serialized[key] = str(value)
        return serialized

    def _validate_input(self, X: np.ndarray, check_fitted: bool = True) -> np.ndarray:
        """
        Validate input data.

        Args:
            X: Input feature matrix
            check_fitted: Whether to check if model is fitted

        Returns:
            Validated array

        Raises:
            ValueError: If validation fails
        """
        if check_fitted and not self._is_fitted:
            raise ValueError(f"Model {self._name} has not been fitted yet")

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self._feature_names and X.shape[1] != len(self._feature_names):
            raise ValueError(
                f"Expected {len(self._feature_names)} features, got {X.shape[1]}"
            )

        # Check for NaN/Inf
        if np.any(~np.isfinite(X)):
            raise ValueError("Input contains NaN or Inf values")

        return X

    def _record_training(
        self,
        metrics: dict[str, float],
        feature_names: list[str] | None = None,
    ) -> None:
        """Record training completion."""
        self._is_fitted = True
        self._training_timestamp = datetime.now(timezone.utc)
        self._training_metrics = metrics
        if feature_names:
            self._feature_names = feature_names

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute training/evaluation metrics.

        DRY FIX: Centralized metrics computation to avoid code duplication
        across model implementations (XGBoost, LightGBM, CatBoost, RandomForest, etc.)

        Args:
            y_true: Ground truth labels/values
            y_pred: Predicted labels/values

        Returns:
            Dictionary of metric names to values
        """
        # Ensure predictions are 1D
        y_pred = np.asarray(y_pred).flatten()
        y_true = np.asarray(y_true).flatten()

        if self._model_type == ModelType.CLASSIFIER:
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
            )

            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
                "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            }
        else:
            from sklearn.metrics import (
                mean_absolute_error,
                mean_squared_error,
                r2_score,
            )

            return {
                "mse": float(mean_squared_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
            }

    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name={self._name}, version={self._version}, {status})"


class TimeSeriesModel(TradingModel):
    """
    Base class for time series models (LSTM, Transformer, TCN).

    Extends TradingModel with sequence-specific functionality like
    lookback window handling and sequence padding.
    """

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        model_type: ModelType = ModelType.REGRESSOR,
        lookback_window: int = 20,
        **kwargs: Any,
    ):
        """
        Initialize time series model.

        Args:
            name: Model identifier
            version: Version string
            model_type: Classification or regression
            lookback_window: Number of historical bars to use
            **kwargs: Additional parameters
        """
        super().__init__(name, version, model_type, **kwargs)
        self.lookback_window = lookback_window
        # Store in _params for serialization
        self._params["lookback_window"] = lookback_window

    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Create sequences from feature matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Optional target array

        Returns:
            Tuple of (X_seq, y_seq) where X_seq has shape
            (n_sequences, lookback_window, n_features)
        """
        n_samples, n_features = X.shape
        n_sequences = n_samples - self.lookback_window + 1

        if n_sequences <= 0:
            raise ValueError(
                f"Not enough samples ({n_samples}) for lookback window ({self.lookback_window})"
            )

        X_seq = np.zeros((n_sequences, self.lookback_window, n_features), dtype=np.float32)
        for i in range(n_sequences):
            X_seq[i] = X[i : i + self.lookback_window]

        y_seq = None
        if y is not None:
            y_seq = y[self.lookback_window - 1:]

        return X_seq, y_seq


class EnsembleModel(TradingModel):
    """
    Base class for ensemble models.

    Provides common functionality for managing multiple base models
    and aggregating their predictions.
    """

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        model_type: ModelType = ModelType.REGRESSOR,
        **kwargs: Any,
    ):
        """Initialize ensemble model."""
        super().__init__(name, version, model_type, **kwargs)
        self._base_models: list[TradingModel] = []
        self._weights: np.ndarray | None = None

    @property
    def base_models(self) -> list[TradingModel]:
        """Get base models."""
        return self._base_models.copy()

    @property
    def weights(self) -> np.ndarray | None:
        """Get model weights."""
        return self._weights.copy() if self._weights is not None else None

    def add_model(self, model: TradingModel, weight: float = 1.0) -> "EnsembleModel":
        """
        Add a base model to the ensemble.

        Args:
            model: Model to add
            weight: Initial weight for the model

        Returns:
            self for method chaining
        """
        self._base_models.append(model)
        if self._weights is None:
            self._weights = np.array([weight])
        else:
            self._weights = np.append(self._weights, weight)
        return self

    def normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        if self._weights is not None:
            self._weights = self._weights / np.sum(self._weights)

    def get_feature_importance(self) -> dict[str, float]:
        """Get averaged feature importance across base models."""
        if not self._base_models:
            return {}

        all_importance: dict[str, list[float]] = {}
        weights = self._weights if self._weights is not None else np.ones(len(self._base_models))

        for model, weight in zip(self._base_models, weights):
            try:
                importance = model.get_feature_importance()
                for feature, score in importance.items():
                    if feature not in all_importance:
                        all_importance[feature] = []
                    all_importance[feature].append(score * weight)
            except (NotImplementedError, ValueError):
                continue

        # Average across models
        return {
            feature: np.mean(scores) for feature, scores in all_importance.items()
        }
