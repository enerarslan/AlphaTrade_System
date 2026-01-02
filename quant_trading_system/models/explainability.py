"""
JPMORGAN FIX: Model Explainability Module with SHAP and LIME.

This module provides comprehensive model interpretability tools for
institutional-grade trading systems, including:
- SHAP (SHapley Additive exPlanations) for global and local explanations
- LIME (Local Interpretable Model-agnostic Explanations) for local predictions
- Feature importance analysis and visualization data
- Regulatory compliance documentation generation

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

logger = logging.getLogger(__name__)


class ExplanationType(str, Enum):
    """Types of model explanations."""

    SHAP_GLOBAL = "shap_global"
    SHAP_LOCAL = "shap_local"
    LIME_LOCAL = "lime_local"
    FEATURE_IMPORTANCE = "feature_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    INTERACTION = "interaction"


class ImportanceMethod(str, Enum):
    """Methods for calculating feature importance."""

    SHAP = "shap"
    PERMUTATION = "permutation"
    GAIN = "gain"  # For tree-based models
    WEIGHT = "weight"  # For tree-based models
    LIME = "lime"


@dataclass
class FeatureExplanation:
    """Explanation for a single feature."""

    feature_name: str
    importance_value: float
    shap_value: float | None = None
    contribution: float | None = None  # Contribution to prediction
    baseline_value: float | None = None
    direction: str | None = None  # "positive" or "negative"
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "importance_value": self.importance_value,
            "shap_value": self.shap_value,
            "contribution": self.contribution,
            "baseline_value": self.baseline_value,
            "direction": self.direction,
            "rank": self.rank,
        }


@dataclass
class LocalExplanation:
    """Local explanation for a single prediction."""

    prediction: float
    base_value: float  # Expected value / baseline
    feature_explanations: list[FeatureExplanation]
    explanation_type: ExplanationType
    instance_index: int | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def top_features(self) -> list[FeatureExplanation]:
        """Get top contributing features sorted by absolute importance."""
        return sorted(
            self.feature_explanations,
            key=lambda x: abs(x.importance_value),
            reverse=True,
        )

    @property
    def positive_contributors(self) -> list[FeatureExplanation]:
        """Get features that positively contributed to prediction."""
        return [f for f in self.feature_explanations if f.importance_value > 0]

    @property
    def negative_contributors(self) -> list[FeatureExplanation]:
        """Get features that negatively contributed to prediction."""
        return [f for f in self.feature_explanations if f.importance_value < 0]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction": self.prediction,
            "base_value": self.base_value,
            "explanation_type": self.explanation_type.value,
            "instance_index": self.instance_index,
            "timestamp": self.timestamp.isoformat(),
            "feature_explanations": [f.to_dict() for f in self.feature_explanations],
            "metadata": self.metadata,
        }


@dataclass
class GlobalExplanation:
    """Global explanation for the model."""

    model_name: str
    feature_importance: dict[str, float]
    shap_values: np.ndarray | None = None
    expected_value: float | None = None
    feature_names: list[str] = field(default_factory=list)
    num_samples: int = 0
    explanation_type: ExplanationType = ExplanationType.SHAP_GLOBAL
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def top_features(self) -> list[tuple[str, float]]:
        """Get top features sorted by importance."""
        return sorted(
            self.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

    def get_feature_rank(self, feature_name: str) -> int | None:
        """Get rank of a specific feature."""
        for i, (name, _) in enumerate(self.top_features):
            if name == feature_name:
                return i + 1
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "feature_importance": self.feature_importance,
            "expected_value": self.expected_value,
            "feature_names": self.feature_names,
            "num_samples": self.num_samples,
            "explanation_type": self.explanation_type.value,
            "timestamp": self.timestamp.isoformat(),
            "top_features": self.top_features[:20],
            "metadata": self.metadata,
        }


@dataclass
class InteractionEffect:
    """Interaction effect between two features."""

    feature_1: str
    feature_2: str
    interaction_strength: float
    shap_interaction_values: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_1": self.feature_1,
            "feature_2": self.feature_2,
            "interaction_strength": self.interaction_strength,
        }


class ModelExplainer(ABC):
    """Abstract base class for model explainers."""

    @abstractmethod
    def explain_local(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
        index: int | None = None,
    ) -> LocalExplanation:
        """Generate local explanation for a single prediction."""
        pass

    @abstractmethod
    def explain_global(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
    ) -> GlobalExplanation:
        """Generate global explanation for the model."""
        pass


class SHAPExplainer(ModelExplainer):
    """
    SHAP (SHapley Additive exPlanations) based model explainer.

    SHAP values provide a unified measure of feature importance that
    satisfies several desirable properties including local accuracy,
    missingness, and consistency.
    """

    def __init__(
        self,
        explainer_type: str = "auto",
        num_samples: int = 100,
        check_additivity: bool = True,
    ):
        """
        Initialize SHAP explainer.

        Args:
            explainer_type: Type of SHAP explainer ("auto", "tree", "kernel", "deep")
            num_samples: Number of samples for kernel SHAP
            check_additivity: Whether to check SHAP additivity
        """
        self.explainer_type = explainer_type
        self.num_samples = num_samples
        self.check_additivity = check_additivity
        self._explainer = None
        self._expected_value = None

    def _create_explainer(self, model: Any, X: np.ndarray) -> Any:
        """Create appropriate SHAP explainer for the model type."""
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not installed. Using fallback importance method.")
            return None

        if self.explainer_type == "auto":
            # Auto-detect model type
            model_type = type(model).__name__.lower()

            if any(t in model_type for t in ["xgb", "lightgbm", "lgbm", "catboost", "randomforest", "gradientboosting"]):
                try:
                    self._explainer = shap.TreeExplainer(model)
                    return self._explainer
                except Exception:
                    pass

            # Fall back to KernelExplainer
            try:
                # Use a sample of data as background
                background = shap.sample(X, min(100, len(X)))
                self._explainer = shap.KernelExplainer(
                    model.predict if hasattr(model, "predict") else model,
                    background,
                )
                return self._explainer
            except Exception as e:
                logger.warning(f"Could not create SHAP explainer: {e}")
                return None

        elif self.explainer_type == "tree":
            self._explainer = shap.TreeExplainer(model)
        elif self.explainer_type == "kernel":
            background = shap.sample(X, min(100, len(X)))
            self._explainer = shap.KernelExplainer(model.predict, background)
        elif self.explainer_type == "deep":
            self._explainer = shap.DeepExplainer(model, X[:100])

        return self._explainer

    def _compute_shap_values(
        self,
        model: Any,
        X: np.ndarray,
    ) -> tuple[np.ndarray | None, float | None]:
        """Compute SHAP values for the data."""
        try:
            import shap
        except ImportError:
            return None, None

        if self._explainer is None:
            self._create_explainer(model, X)

        if self._explainer is None:
            return None, None

        try:
            shap_values = self._explainer.shap_values(X, check_additivity=self.check_additivity)

            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            expected_value = self._explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

            return shap_values, float(expected_value)
        except Exception as e:
            logger.warning(f"Error computing SHAP values: {e}")
            return None, None

    def _fallback_importance(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Fallback feature importance when SHAP is unavailable."""
        # Try model's built-in feature importance
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            return dict(zip(feature_names, importance))

        # Try coefficients for linear models
        if hasattr(model, "coef_"):
            coef = np.abs(model.coef_).flatten()
            if len(coef) == len(feature_names):
                return dict(zip(feature_names, coef))

        # Permutation importance as last resort
        return self._permutation_importance(model, X, feature_names)

    def _permutation_importance(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
        n_repeats: int = 10,
    ) -> dict[str, float]:
        """Calculate permutation importance."""
        baseline_pred = model.predict(X)
        importance = {}

        for i, name in enumerate(feature_names):
            scores = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])
                permuted_pred = model.predict(X_permuted)
                score = np.mean(np.abs(baseline_pred - permuted_pred))
                scores.append(score)
            importance[name] = np.mean(scores)

        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance

    def explain_local(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
        index: int | None = None,
    ) -> LocalExplanation:
        """
        Generate local SHAP explanation for a single prediction.

        Args:
            model: Trained model
            X: Feature matrix (single instance or batch)
            feature_names: List of feature names
            index: Index of instance to explain (if X is batch)

        Returns:
            LocalExplanation object
        """
        # Get single instance
        if index is not None:
            X_instance = X[index:index+1]
        elif X.ndim == 1:
            X_instance = X.reshape(1, -1)
        else:
            X_instance = X[:1]

        # Get prediction
        prediction = float(model.predict(X_instance)[0])

        # Compute SHAP values
        shap_values, expected_value = self._compute_shap_values(model, X_instance)

        if shap_values is not None:
            # Create feature explanations from SHAP values
            shap_values_flat = shap_values.flatten()
            feature_explanations = []

            for i, name in enumerate(feature_names):
                shap_val = float(shap_values_flat[i])
                feature_explanations.append(
                    FeatureExplanation(
                        feature_name=name,
                        importance_value=shap_val,
                        shap_value=shap_val,
                        contribution=shap_val,
                        direction="positive" if shap_val > 0 else "negative",
                    )
                )

            # Sort and assign ranks
            feature_explanations.sort(key=lambda x: abs(x.importance_value), reverse=True)
            for i, fe in enumerate(feature_explanations):
                fe.rank = i + 1

            base_value = expected_value if expected_value is not None else 0.0
        else:
            # Fallback: use feature importance
            importance = self._fallback_importance(model, X_instance, feature_names)
            feature_explanations = [
                FeatureExplanation(
                    feature_name=name,
                    importance_value=imp,
                    direction="unknown",
                    rank=i + 1,
                )
                for i, (name, imp) in enumerate(
                    sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
                )
            ]
            base_value = prediction  # No baseline available

        return LocalExplanation(
            prediction=prediction,
            base_value=base_value,
            feature_explanations=feature_explanations,
            explanation_type=ExplanationType.SHAP_LOCAL,
            instance_index=index,
            metadata={"explainer_type": self.explainer_type},
        )

    def explain_global(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
    ) -> GlobalExplanation:
        """
        Generate global SHAP explanation for the model.

        Args:
            model: Trained model
            X: Feature matrix (sample of training data)
            feature_names: List of feature names

        Returns:
            GlobalExplanation object
        """
        model_name = type(model).__name__

        # Compute SHAP values for all samples
        shap_values, expected_value = self._compute_shap_values(model, X)

        if shap_values is not None:
            # Calculate mean absolute SHAP value per feature
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feature_importance = dict(zip(feature_names, mean_abs_shap))
        else:
            # Fallback
            feature_importance = self._fallback_importance(model, X, feature_names)
            shap_values = None
            expected_value = None

        return GlobalExplanation(
            model_name=model_name,
            feature_importance=feature_importance,
            shap_values=shap_values,
            expected_value=expected_value,
            feature_names=feature_names,
            num_samples=len(X),
            explanation_type=ExplanationType.SHAP_GLOBAL,
            metadata={"explainer_type": self.explainer_type},
        )

    def compute_interaction_effects(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
        top_k: int = 10,
    ) -> list[InteractionEffect]:
        """
        Compute feature interaction effects using SHAP interaction values.

        Args:
            model: Trained model
            X: Feature matrix
            feature_names: List of feature names
            top_k: Number of top interactions to return

        Returns:
            List of InteractionEffect objects
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not installed. Cannot compute interactions.")
            return []

        if self._explainer is None:
            self._create_explainer(model, X)

        if self._explainer is None or not hasattr(self._explainer, "shap_interaction_values"):
            logger.warning("Interaction values not supported for this explainer type.")
            return []

        try:
            interaction_values = self._explainer.shap_interaction_values(X)

            # Handle multi-class
            if isinstance(interaction_values, list):
                interaction_values = interaction_values[1] if len(interaction_values) > 1 else interaction_values[0]

            # Calculate mean absolute interaction strength
            mean_interactions = np.abs(interaction_values).mean(axis=0)

            # Extract top interactions (excluding diagonal)
            interactions = []
            n_features = len(feature_names)

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    interactions.append(
                        InteractionEffect(
                            feature_1=feature_names[i],
                            feature_2=feature_names[j],
                            interaction_strength=float(mean_interactions[i, j]),
                            shap_interaction_values=interaction_values[:, i, j],
                        )
                    )

            # Sort by strength and return top k
            interactions.sort(key=lambda x: abs(x.interaction_strength), reverse=True)
            return interactions[:top_k]

        except Exception as e:
            logger.warning(f"Error computing interaction effects: {e}")
            return []


class LIMEExplainer(ModelExplainer):
    """
    LIME (Local Interpretable Model-agnostic Explanations) based explainer.

    LIME explains individual predictions by learning a local linear model
    around the instance being explained.
    """

    def __init__(
        self,
        num_features: int = 10,
        num_samples: int = 5000,
        kernel_width: float | None = None,
        discretize_continuous: bool = True,
    ):
        """
        Initialize LIME explainer.

        Args:
            num_features: Number of features to include in explanation
            num_samples: Number of samples for the local model
            kernel_width: Width of the kernel (None for default)
            discretize_continuous: Whether to discretize continuous features
        """
        self.num_features = num_features
        self.num_samples = num_samples
        self.kernel_width = kernel_width
        self.discretize_continuous = discretize_continuous
        self._explainer = None

    def _create_explainer(self, X: np.ndarray, feature_names: list[str]) -> Any:
        """Create LIME explainer."""
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError:
            logger.warning("LIME not installed. Using fallback explanation.")
            return None

        self._explainer = LimeTabularExplainer(
            X,
            feature_names=feature_names,
            mode="regression",
            discretize_continuous=self.discretize_continuous,
            kernel_width=self.kernel_width,
        )
        return self._explainer

    def explain_local(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
        index: int | None = None,
    ) -> LocalExplanation:
        """
        Generate local LIME explanation for a single prediction.

        Args:
            model: Trained model
            X: Feature matrix
            feature_names: List of feature names
            index: Index of instance to explain

        Returns:
            LocalExplanation object
        """
        # Get single instance
        if index is not None:
            X_instance = X[index]
        elif X.ndim == 1:
            X_instance = X
        else:
            X_instance = X[0]

        # Get prediction
        prediction = float(model.predict(X_instance.reshape(1, -1))[0])

        # Create explainer if needed
        if self._explainer is None:
            self._create_explainer(X, feature_names)

        if self._explainer is not None:
            try:
                # Generate explanation
                exp = self._explainer.explain_instance(
                    X_instance,
                    model.predict,
                    num_features=min(self.num_features, len(feature_names)),
                    num_samples=self.num_samples,
                )

                # Extract feature weights
                weights = dict(exp.as_list())
                intercept = exp.intercept[0] if hasattr(exp, "intercept") else 0.0

                feature_explanations = []
                for i, name in enumerate(feature_names):
                    # Find matching weight (LIME may rename features)
                    weight = 0.0
                    for lime_name, w in weights.items():
                        if name in lime_name:
                            weight = w
                            break

                    feature_explanations.append(
                        FeatureExplanation(
                            feature_name=name,
                            importance_value=weight,
                            contribution=weight,
                            direction="positive" if weight > 0 else "negative",
                        )
                    )

                # Sort and assign ranks
                feature_explanations.sort(key=lambda x: abs(x.importance_value), reverse=True)
                for i, fe in enumerate(feature_explanations):
                    fe.rank = i + 1

                return LocalExplanation(
                    prediction=prediction,
                    base_value=intercept,
                    feature_explanations=feature_explanations,
                    explanation_type=ExplanationType.LIME_LOCAL,
                    instance_index=index,
                    metadata={"num_samples": self.num_samples},
                )

            except Exception as e:
                logger.warning(f"LIME explanation failed: {e}")

        # Fallback: uniform importance
        feature_explanations = [
            FeatureExplanation(
                feature_name=name,
                importance_value=1.0 / len(feature_names),
                direction="unknown",
                rank=i + 1,
            )
            for i, name in enumerate(feature_names)
        ]

        return LocalExplanation(
            prediction=prediction,
            base_value=prediction,
            feature_explanations=feature_explanations,
            explanation_type=ExplanationType.LIME_LOCAL,
            instance_index=index,
            metadata={"fallback": True},
        )

    def explain_global(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
    ) -> GlobalExplanation:
        """
        Generate global explanation by aggregating local LIME explanations.

        Note: This is computationally expensive as it explains each instance.
        Consider using a sample of the data.

        Args:
            model: Trained model
            X: Feature matrix (use a small sample)
            feature_names: List of feature names

        Returns:
            GlobalExplanation object
        """
        model_name = type(model).__name__

        # Limit to reasonable sample size
        sample_size = min(len(X), 100)
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        # Aggregate local explanations
        importance_sum = {name: 0.0 for name in feature_names}

        for i in range(len(X_sample)):
            local_exp = self.explain_local(model, X_sample, feature_names, index=i)
            for fe in local_exp.feature_explanations:
                importance_sum[fe.feature_name] += abs(fe.importance_value)

        # Normalize
        total = sum(importance_sum.values())
        if total > 0:
            feature_importance = {k: v / total for k, v in importance_sum.items()}
        else:
            feature_importance = {k: 1.0 / len(feature_names) for k in feature_names}

        return GlobalExplanation(
            model_name=model_name,
            feature_importance=feature_importance,
            feature_names=feature_names,
            num_samples=len(X_sample),
            explanation_type=ExplanationType.LIME_LOCAL,
            metadata={"aggregated_from_local": True},
        )


class ModelExplainabilityService:
    """
    Unified service for model explainability.

    Provides a high-level interface to generate explanations using
    multiple methods and create regulatory compliance reports.
    """

    def __init__(
        self,
        shap_explainer: SHAPExplainer | None = None,
        lime_explainer: LIMEExplainer | None = None,
    ):
        """
        Initialize explainability service.

        Args:
            shap_explainer: SHAP explainer instance
            lime_explainer: LIME explainer instance
        """
        self.shap_explainer = shap_explainer or SHAPExplainer()
        self.lime_explainer = lime_explainer or LIMEExplainer()
        self._explanation_cache: dict[str, Any] = {}

    def explain_prediction(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
        index: int = 0,
        methods: list[str] | None = None,
    ) -> dict[str, LocalExplanation]:
        """
        Generate local explanations using multiple methods.

        Args:
            model: Trained model
            X: Feature matrix
            feature_names: List of feature names
            index: Index of instance to explain
            methods: List of methods to use ("shap", "lime")

        Returns:
            Dictionary of explanations by method
        """
        methods = methods or ["shap", "lime"]
        explanations = {}

        if "shap" in methods:
            try:
                explanations["shap"] = self.shap_explainer.explain_local(
                    model, X, feature_names, index
                )
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")

        if "lime" in methods:
            try:
                explanations["lime"] = self.lime_explainer.explain_local(
                    model, X, feature_names, index
                )
            except Exception as e:
                logger.warning(f"LIME explanation failed: {e}")

        return explanations

    def explain_model(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
        methods: list[str] | None = None,
    ) -> dict[str, GlobalExplanation]:
        """
        Generate global explanations using multiple methods.

        Args:
            model: Trained model
            X: Feature matrix (sample of training data)
            feature_names: List of feature names
            methods: List of methods to use

        Returns:
            Dictionary of global explanations by method
        """
        methods = methods or ["shap"]
        explanations = {}

        if "shap" in methods:
            try:
                explanations["shap"] = self.shap_explainer.explain_global(
                    model, X, feature_names
                )
            except Exception as e:
                logger.warning(f"SHAP global explanation failed: {e}")

        if "lime" in methods:
            try:
                explanations["lime"] = self.lime_explainer.explain_global(
                    model, X, feature_names
                )
            except Exception as e:
                logger.warning(f"LIME global explanation failed: {e}")

        return explanations

    def generate_compliance_report(
        self,
        model: Any,
        model_name: str,
        X: np.ndarray,
        feature_names: list[str],
        predictions: np.ndarray | None = None,
        sample_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Generate regulatory compliance report with model explanations.

        This report includes:
        - Global feature importance
        - Sample local explanations
        - Model documentation

        Args:
            model: Trained model
            model_name: Name of the model
            X: Feature matrix
            feature_names: List of feature names
            predictions: Model predictions (optional)
            sample_indices: Indices of samples to explain locally

        Returns:
            Compliance report dictionary
        """
        report = {
            "report_type": "model_explainability",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "model_class": type(model).__name__,
            "num_features": len(feature_names),
            "feature_names": feature_names,
            "num_samples_analyzed": len(X),
        }

        # Global explanation
        try:
            global_exp = self.shap_explainer.explain_global(model, X, feature_names)
            report["global_explanation"] = global_exp.to_dict()
            report["top_10_features"] = global_exp.top_features[:10]
        except Exception as e:
            report["global_explanation_error"] = str(e)

        # Local explanations for samples
        sample_indices = sample_indices or list(range(min(5, len(X))))
        local_explanations = []

        for idx in sample_indices:
            try:
                local_exp = self.shap_explainer.explain_local(model, X, feature_names, idx)
                local_dict = local_exp.to_dict()
                local_dict["sample_index"] = idx
                if predictions is not None:
                    local_dict["actual_prediction"] = float(predictions[idx])
                local_explanations.append(local_dict)
            except Exception as e:
                local_explanations.append({
                    "sample_index": idx,
                    "error": str(e),
                })

        report["local_explanations"] = local_explanations

        # Interaction effects (if available)
        try:
            interactions = self.shap_explainer.compute_interaction_effects(
                model, X[:min(100, len(X))], feature_names, top_k=5
            )
            report["top_interactions"] = [i.to_dict() for i in interactions]
        except Exception as e:
            report["interaction_error"] = str(e)

        # Model metadata
        report["model_metadata"] = {
            "has_feature_importances": hasattr(model, "feature_importances_"),
            "has_coefficients": hasattr(model, "coef_"),
            "is_fitted": hasattr(model, "is_fitted_") or hasattr(model, "_is_fitted"),
        }

        return report

    def compare_explanations(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
        index: int = 0,
    ) -> dict[str, Any]:
        """
        Compare SHAP and LIME explanations for the same prediction.

        Useful for validating explanation consistency and identifying
        features where methods disagree.

        Args:
            model: Trained model
            X: Feature matrix
            feature_names: List of feature names
            index: Index of instance to explain

        Returns:
            Comparison report
        """
        explanations = self.explain_prediction(
            model, X, feature_names, index, methods=["shap", "lime"]
        )

        comparison = {
            "instance_index": index,
            "prediction": None,
            "shap_top_5": [],
            "lime_top_5": [],
            "agreement_score": 0.0,
            "disagreement_features": [],
        }

        if "shap" in explanations:
            comparison["prediction"] = explanations["shap"].prediction
            comparison["shap_top_5"] = [
                (fe.feature_name, fe.importance_value)
                for fe in explanations["shap"].top_features[:5]
            ]

        if "lime" in explanations:
            if comparison["prediction"] is None:
                comparison["prediction"] = explanations["lime"].prediction
            comparison["lime_top_5"] = [
                (fe.feature_name, fe.importance_value)
                for fe in explanations["lime"].top_features[:5]
            ]

        # Calculate agreement (overlap of top features)
        if comparison["shap_top_5"] and comparison["lime_top_5"]:
            shap_features = set(f[0] for f in comparison["shap_top_5"])
            lime_features = set(f[0] for f in comparison["lime_top_5"])
            overlap = len(shap_features & lime_features)
            comparison["agreement_score"] = overlap / 5.0
            comparison["disagreement_features"] = list(
                shap_features.symmetric_difference(lime_features)
            )

        return comparison


def create_explainability_service(
    explainer_type: str = "auto",
    num_shap_samples: int = 100,
    num_lime_features: int = 10,
) -> ModelExplainabilityService:
    """
    Factory function to create explainability service with default configuration.

    Args:
        explainer_type: Type of SHAP explainer
        num_shap_samples: Number of samples for SHAP
        num_lime_features: Number of features for LIME

    Returns:
        Configured ModelExplainabilityService
    """
    shap_explainer = SHAPExplainer(
        explainer_type=explainer_type,
        num_samples=num_shap_samples,
    )

    lime_explainer = LIMEExplainer(
        num_features=num_lime_features,
    )

    return ModelExplainabilityService(
        shap_explainer=shap_explainer,
        lime_explainer=lime_explainer,
    )
