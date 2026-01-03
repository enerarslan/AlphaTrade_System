"""
Classical machine learning models for trading.

Implements XGBoost, LightGBM, CatBoost, and Random Forest models
with trading-specific optimizations and hyperparameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_trading_system.models.base import ModelType, TradingModel


class XGBoostModel(TradingModel):
    """
    XGBoost gradient boosting model for trading signals.

    Optimized for financial time series with custom evaluation metrics
    (Sharpe ratio), sample weighting (recent data more important),
    and GPU acceleration support.
    """

    def __init__(
        self,
        name: str = "xgboost",
        version: str = "1.0.0",
        model_type: ModelType = ModelType.REGRESSOR,
        n_estimators: int = 1000,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        min_child_weight: int = 5,
        early_stopping_rounds: int = 50,
        use_gpu: bool = False,
        random_state: int = 42,
        **kwargs: Any,
    ):
        """
        Initialize XGBoost model.

        Args:
            name: Model identifier
            version: Version string
            model_type: Classification or regression
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate / shrinkage
            subsample: Row subsampling ratio
            colsample_bytree: Column subsampling ratio
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            min_child_weight: Minimum sum of instance weights in a leaf
            early_stopping_rounds: Stop if no improvement after N rounds
            use_gpu: Whether to use GPU acceleration
            random_state: Random seed for reproducibility
            **kwargs: Additional XGBoost parameters
        """
        super().__init__(name, version, model_type, **kwargs)

        self._params.update({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_child_weight": min_child_weight,
            "early_stopping_rounds": early_stopping_rounds,
            "use_gpu": use_gpu,
            "random_state": random_state,
        })

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "XGBoostModel":
        """
        Train XGBoost model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target array (n_samples,)
            validation_data: Optional (X_val, y_val) for early stopping
            sample_weights: Optional sample weights (recent > older)
            **kwargs: Additional fit parameters

        Returns:
            self for method chaining
        """
        import xgboost as xgb

        X = self._validate_input(X, check_fitted=False)

        # Build XGBoost parameters
        params = {
            "n_estimators": self._params["n_estimators"],
            "max_depth": self._params["max_depth"],
            "learning_rate": self._params["learning_rate"],
            "subsample": self._params["subsample"],
            "colsample_bytree": self._params["colsample_bytree"],
            "reg_alpha": self._params["reg_alpha"],
            "reg_lambda": self._params["reg_lambda"],
            "min_child_weight": self._params["min_child_weight"],
            "random_state": self._params["random_state"],
            "n_jobs": -1,
            "verbosity": 0,
        }

        # GPU support
        if self._params.get("use_gpu"):
            params["tree_method"] = "hist"
            params["device"] = "cuda"

        # Create model based on type
        if self._model_type == ModelType.CLASSIFIER:
            self._model = xgb.XGBClassifier(**params)
        else:
            self._model = xgb.XGBRegressor(**params)

        # Fit with early stopping if validation data provided
        fit_params: dict[str, Any] = {"sample_weight": sample_weights}

        if validation_data is not None:
            X_val, y_val = validation_data
            self._model.fit(
                X,
                y,
                eval_set=[(X_val, y_val)],
                verbose=False,
                **fit_params,
            )
        else:
            self._model.fit(X, y, **fit_params)

        # Record training
        feature_names = kwargs.get("feature_names", [f"f{i}" for i in range(X.shape[1])])
        metrics = self._compute_training_metrics(X, y)
        self._record_training(metrics, feature_names)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        X = self._validate_input(X)
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability estimates (classification only)."""
        if self._model_type != ModelType.CLASSIFIER:
            raise NotImplementedError("predict_proba only available for classifiers")
        X = self._validate_input(X)
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")

        importance = self._model.feature_importances_
        names = self._feature_names or [f"f{i}" for i in range(len(importance))]
        return dict(zip(names, importance.tolist()))

    def _compute_training_metrics(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute training metrics using base class method.

        DRY FIX: Delegates to base class _compute_metrics() for unified metrics calculation.
        """
        predictions = self._model.predict(X)
        return super()._compute_metrics(y, predictions)


class LightGBMModel(TradingModel):
    """
    LightGBM gradient boosting model.

    Faster training than XGBoost with lower memory usage.
    Excellent for large datasets with native categorical support.
    """

    def __init__(
        self,
        name: str = "lightgbm",
        version: str = "1.0.0",
        model_type: ModelType = ModelType.REGRESSOR,
        num_leaves: int = 63,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        n_estimators: int = 1000,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        lambda_l1: float = 0.1,
        lambda_l2: float = 0.1,
        min_data_in_leaf: int = 20,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        **kwargs: Any,
    ):
        """
        Initialize LightGBM model.

        Args:
            name: Model identifier
            version: Version string
            model_type: Classification or regression
            num_leaves: Maximum number of leaves per tree
            max_depth: Maximum tree depth (-1 for unlimited)
            learning_rate: Learning rate
            n_estimators: Number of boosting iterations
            feature_fraction: Feature subsampling ratio
            bagging_fraction: Row subsampling ratio
            bagging_freq: Bagging frequency
            lambda_l1: L1 regularization
            lambda_l2: L2 regularization
            min_data_in_leaf: Minimum samples in a leaf
            early_stopping_rounds: Early stopping patience
            random_state: Random seed
            **kwargs: Additional parameters
        """
        super().__init__(name, version, model_type, **kwargs)

        self._params.update({
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "min_data_in_leaf": min_data_in_leaf,
            "early_stopping_rounds": early_stopping_rounds,
            "random_state": random_state,
        })

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weights: np.ndarray | None = None,
        categorical_features: list[int] | None = None,
        **kwargs: Any,
    ) -> "LightGBMModel":
        """
        Train LightGBM model.

        Args:
            X: Feature matrix
            y: Target array
            validation_data: Optional validation data for early stopping
            sample_weights: Optional sample weights
            categorical_features: Indices of categorical features
            **kwargs: Additional parameters

        Returns:
            self for method chaining
        """
        import lightgbm as lgb

        X = self._validate_input(X, check_fitted=False)

        params = {
            "num_leaves": self._params["num_leaves"],
            "max_depth": self._params["max_depth"],
            "learning_rate": self._params["learning_rate"],
            "n_estimators": self._params["n_estimators"],
            "subsample": self._params["bagging_fraction"],
            "subsample_freq": self._params["bagging_freq"],
            "colsample_bytree": self._params["feature_fraction"],
            "reg_alpha": self._params["lambda_l1"],
            "reg_lambda": self._params["lambda_l2"],
            "min_child_samples": self._params["min_data_in_leaf"],
            "random_state": self._params["random_state"],
            "n_jobs": -1,
            "verbose": -1,
        }

        if self._model_type == ModelType.CLASSIFIER:
            self._model = lgb.LGBMClassifier(**params)
        else:
            self._model = lgb.LGBMRegressor(**params)

        fit_params: dict[str, Any] = {"sample_weight": sample_weights}
        if categorical_features:
            fit_params["categorical_feature"] = categorical_features

        if validation_data is not None:
            X_val, y_val = validation_data
            self._model.fit(
                X,
                y,
                eval_set=[(X_val, y_val)],
                **fit_params,
            )
        else:
            self._model.fit(X, y, **fit_params)

        feature_names = kwargs.get("feature_names", [f"f{i}" for i in range(X.shape[1])])
        metrics = self._compute_training_metrics(X, y)
        self._record_training(metrics, feature_names)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        X = self._validate_input(X)
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability estimates (classification only)."""
        if self._model_type != ModelType.CLASSIFIER:
            raise NotImplementedError("predict_proba only available for classifiers")
        X = self._validate_input(X)
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")

        importance = self._model.feature_importances_
        names = self._feature_names or [f"f{i}" for i in range(len(importance))]
        return dict(zip(names, importance.tolist()))

    def _compute_training_metrics(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute training metrics using base class method.

        DRY FIX: Delegates to base class _compute_metrics() for unified metrics calculation.
        """
        predictions = self._model.predict(X)
        return super()._compute_metrics(y, predictions)


class CatBoostModel(TradingModel):
    """
    CatBoost gradient boosting model.

    Best-in-class categorical feature handling with ordered boosting
    to reduce overfitting. GPU training support.
    """

    def __init__(
        self,
        name: str = "catboost",
        version: str = "1.0.0",
        model_type: ModelType = ModelType.REGRESSOR,
        iterations: int = 1000,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        bagging_temperature: float = 0.5,
        early_stopping_rounds: int = 50,
        use_gpu: bool = False,
        random_state: int = 42,
        **kwargs: Any,
    ):
        """
        Initialize CatBoost model.

        Args:
            name: Model identifier
            version: Version string
            model_type: Classification or regression
            iterations: Number of boosting iterations
            depth: Tree depth
            learning_rate: Learning rate
            l2_leaf_reg: L2 regularization coefficient
            random_strength: Random forest regularization
            bagging_temperature: Bayesian bootstrap temperature
            early_stopping_rounds: Early stopping patience
            use_gpu: Whether to use GPU
            random_state: Random seed
            **kwargs: Additional parameters
        """
        super().__init__(name, version, model_type, **kwargs)

        self._params.update({
            "iterations": iterations,
            "depth": depth,
            "learning_rate": learning_rate,
            "l2_leaf_reg": l2_leaf_reg,
            "random_strength": random_strength,
            "bagging_temperature": bagging_temperature,
            "early_stopping_rounds": early_stopping_rounds,
            "use_gpu": use_gpu,
            "random_state": random_state,
        })

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weights: np.ndarray | None = None,
        cat_features: list[int] | None = None,
        **kwargs: Any,
    ) -> "CatBoostModel":
        """
        Train CatBoost model.

        Args:
            X: Feature matrix
            y: Target array
            validation_data: Optional validation data
            sample_weights: Optional sample weights
            cat_features: Indices of categorical features
            **kwargs: Additional parameters

        Returns:
            self for method chaining
        """
        from catboost import CatBoostClassifier, CatBoostRegressor

        X = self._validate_input(X, check_fitted=False)

        params = {
            "iterations": self._params["iterations"],
            "depth": self._params["depth"],
            "learning_rate": self._params["learning_rate"],
            "l2_leaf_reg": self._params["l2_leaf_reg"],
            "random_strength": self._params["random_strength"],
            "bagging_temperature": self._params["bagging_temperature"],
            "random_seed": self._params["random_state"],
            "verbose": False,
            "allow_writing_files": False,
        }

        if self._params.get("use_gpu"):
            params["task_type"] = "GPU"

        if self._model_type == ModelType.CLASSIFIER:
            self._model = CatBoostClassifier(**params)
        else:
            self._model = CatBoostRegressor(**params)

        fit_params: dict[str, Any] = {"sample_weight": sample_weights}
        if cat_features:
            fit_params["cat_features"] = cat_features

        if validation_data is not None:
            X_val, y_val = validation_data
            fit_params["eval_set"] = (X_val, y_val)
            fit_params["early_stopping_rounds"] = self._params["early_stopping_rounds"]

        self._model.fit(X, y, **fit_params)

        feature_names = kwargs.get("feature_names", [f"f{i}" for i in range(X.shape[1])])
        metrics = self._compute_training_metrics(X, y)
        self._record_training(metrics, feature_names)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        X = self._validate_input(X)
        return self._model.predict(X).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability estimates (classification only)."""
        if self._model_type != ModelType.CLASSIFIER:
            raise NotImplementedError("predict_proba only available for classifiers")
        X = self._validate_input(X)
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")

        importance = self._model.get_feature_importance()
        names = self._feature_names or [f"f{i}" for i in range(len(importance))]
        return dict(zip(names, importance.tolist()))

    def _compute_training_metrics(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute training metrics using base class method.

        DRY FIX: Delegates to base class _compute_metrics() for unified metrics calculation.
        """
        predictions = self._model.predict(X).flatten()
        return super()._compute_metrics(y, predictions)


class RandomForestModel(TradingModel):
    """
    Random Forest ensemble model.

    Robust baseline model with natural feature importance.
    Less prone to overfitting than gradient boosting.
    """

    def __init__(
        self,
        name: str = "random_forest",
        version: str = "1.0.0",
        model_type: ModelType = ModelType.REGRESSOR,
        n_estimators: int = 500,
        max_depth: int | None = 20,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str | float = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ):
        """
        Initialize Random Forest model.

        Args:
            name: Model identifier
            version: Version string
            model_type: Classification or regression
            n_estimators: Number of trees
            max_depth: Maximum tree depth (None for unlimited)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf
            max_features: Features to consider for splits ("sqrt", "log2", or float)
            bootstrap: Whether to bootstrap samples
            oob_score: Whether to compute out-of-bag score
            random_state: Random seed
            n_jobs: Number of parallel jobs
            **kwargs: Additional parameters
        """
        super().__init__(name, version, model_type, **kwargs)

        self._params.update({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "bootstrap": bootstrap,
            "oob_score": oob_score,
            "random_state": random_state,
            "n_jobs": n_jobs,
        })

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "RandomForestModel":
        """
        Train Random Forest model.

        Args:
            X: Feature matrix
            y: Target array
            validation_data: Ignored (RF doesn't use early stopping)
            sample_weights: Optional sample weights
            **kwargs: Additional parameters

        Returns:
            self for method chaining
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        X = self._validate_input(X, check_fitted=False)

        params = {
            "n_estimators": self._params["n_estimators"],
            "max_depth": self._params["max_depth"],
            "min_samples_split": self._params["min_samples_split"],
            "min_samples_leaf": self._params["min_samples_leaf"],
            "max_features": self._params["max_features"],
            "bootstrap": self._params["bootstrap"],
            "oob_score": self._params["oob_score"],
            "random_state": self._params["random_state"],
            "n_jobs": self._params["n_jobs"],
        }

        if self._model_type == ModelType.CLASSIFIER:
            self._model = RandomForestClassifier(**params)
        else:
            self._model = RandomForestRegressor(**params)

        self._model.fit(X, y, sample_weight=sample_weights)

        feature_names = kwargs.get("feature_names", [f"f{i}" for i in range(X.shape[1])])
        metrics = self._compute_training_metrics(X, y)

        # Add OOB score if available
        if self._params["oob_score"] and hasattr(self._model, "oob_score_"):
            metrics["oob_score"] = float(self._model.oob_score_)

        self._record_training(metrics, feature_names)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        X = self._validate_input(X)
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability estimates (classification only)."""
        if self._model_type != ModelType.CLASSIFIER:
            raise NotImplementedError("predict_proba only available for classifiers")
        X = self._validate_input(X)
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")

        importance = self._model.feature_importances_
        names = self._feature_names or [f"f{i}" for i in range(len(importance))]
        return dict(zip(names, importance.tolist()))

    def _compute_training_metrics(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute training metrics using base class method.

        DRY FIX: Delegates to base class _compute_metrics() for unified metrics calculation.
        """
        predictions = self._model.predict(X)
        return super()._compute_metrics(y, predictions)


class ElasticNetModel(TradingModel):
    """
    Elastic Net linear model with L1 and L2 regularization.

    Useful for feature selection and as an interpretable baseline.
    Fast training and inference.
    """

    def __init__(
        self,
        name: str = "elastic_net",
        version: str = "1.0.0",
        model_type: ModelType = ModelType.REGRESSOR,
        alpha: float = 0.1,
        l1_ratio: float = 0.5,
        max_iter: int = 2000,
        tol: float = 1e-4,
        random_state: int = 42,
        **kwargs: Any,
    ):
        """
        Initialize Elastic Net model.

        Args:
            name: Model identifier
            version: Version string
            model_type: Regression only (use LogisticRegression for classification)
            alpha: Regularization strength
            l1_ratio: L1 vs L2 ratio (0=Ridge, 1=Lasso)
            max_iter: Maximum iterations
            tol: Tolerance for optimization
            random_state: Random seed
            **kwargs: Additional parameters
        """
        super().__init__(name, version, model_type, **kwargs)

        self._params.update({
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "max_iter": max_iter,
            "tol": tol,
            "random_state": random_state,
        })

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "ElasticNetModel":
        """Train Elastic Net model."""
        from sklearn.linear_model import ElasticNet
        from sklearn.preprocessing import StandardScaler

        X = self._validate_input(X, check_fitted=False)

        # Scale features for linear models
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = ElasticNet(
            alpha=self._params["alpha"],
            l1_ratio=self._params["l1_ratio"],
            max_iter=self._params["max_iter"],
            tol=self._params["tol"],
            random_state=self._params["random_state"],
        )

        self._model.fit(X_scaled, y, sample_weight=sample_weights)

        # Mark as fitted before computing metrics (needed for predict)
        self._is_fitted = True

        feature_names = kwargs.get("feature_names", [f"f{i}" for i in range(X.shape[1])])
        metrics = self._compute_training_metrics(X, y)
        self._record_training(metrics, feature_names)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        X = self._validate_input(X)
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature coefficients as importance."""
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")

        importance = np.abs(self._model.coef_)
        names = self._feature_names or [f"f{i}" for i in range(len(importance))]
        return dict(zip(names, importance.tolist()))

    def _compute_training_metrics(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute training metrics using base class method.

        DRY FIX: Delegates to base class _compute_metrics() for unified metrics calculation.
        """
        predictions = self.predict(X)
        return super()._compute_metrics(y, predictions)

    def predict_proba(
        self,
        X: np.ndarray,
        calibration_method: str = "platt",
    ) -> np.ndarray:
        """Generate pseudo-probability estimates for regression outputs.

        JPMorgan-level enhancement: Provides probability-like outputs from
        regression predictions using calibration techniques.

        For trading applications, this is useful when:
        - Combining with classifiers in ensemble models
        - Converting signals to confidence scores
        - Risk-adjusted position sizing based on model confidence

        Args:
            X: Feature matrix (n_samples, n_features)
            calibration_method: Method for converting regression to probabilities.
                - "platt": Platt scaling using sigmoid (default)
                - "minmax": Min-max scaling to [0, 1]
                - "quantile": Quantile-based normalization

        Returns:
            Array of shape (n_samples, 2) with probabilities for [down, up]
            where column 0 is P(return < 0) and column 1 is P(return > 0)

        Example:
            >>> model = ElasticNetModel()
            >>> model.fit(X_train, y_train)
            >>> proba = model.predict_proba(X_test)
            >>> confidence = proba[:, 1]  # Probability of positive return
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")

        X = self._validate_input(X)
        predictions = self.predict(X)

        if calibration_method == "platt":
            # Platt scaling: apply sigmoid to convert regression output to probability
            # Scale predictions to reasonable range for sigmoid
            if not hasattr(self, "_calibration_scale"):
                # Use training prediction statistics for scaling
                train_std = max(np.std(predictions), 1e-6)
                self._calibration_scale = train_std

            # Sigmoid transformation: P(positive) = 1 / (1 + exp(-prediction/scale))
            scaled_preds = predictions / self._calibration_scale
            prob_positive = 1.0 / (1.0 + np.exp(-scaled_preds))

        elif calibration_method == "minmax":
            # Min-max scaling to [0, 1]
            pred_min = np.min(predictions)
            pred_max = np.max(predictions)
            pred_range = max(pred_max - pred_min, 1e-6)
            prob_positive = (predictions - pred_min) / pred_range

        elif calibration_method == "quantile":
            # Quantile-based: rank / n gives uniform distribution
            from scipy import stats
            prob_positive = stats.rankdata(predictions) / len(predictions)

        else:
            raise ValueError(
                f"Unknown calibration method: {calibration_method}. "
                "Choose from: 'platt', 'minmax', 'quantile'"
            )

        # Ensure probabilities are in [0, 1]
        prob_positive = np.clip(prob_positive, 0.0, 1.0)
        prob_negative = 1.0 - prob_positive

        return np.column_stack([prob_negative, prob_positive])

    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        method: str = "isotonic",
    ) -> "ElasticNetModel":
        """Calibrate the model using holdout calibration data.

        JPMorgan-level enhancement: Proper probability calibration using
        scikit-learn's calibration framework for more accurate confidence scores.

        Args:
            X_cal: Calibration features
            y_cal: Calibration targets (binary: 0/1 or continuous)
            method: Calibration method ("isotonic" or "sigmoid")

        Returns:
            self for method chaining
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before calibration")

        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        # Get predictions on calibration set
        cal_predictions = self.predict(X_cal)

        # Binarize targets if continuous
        if not np.all(np.isin(y_cal, [0, 1])):
            y_cal_binary = (y_cal > 0).astype(int)
        else:
            y_cal_binary = y_cal

        if method == "isotonic":
            # Isotonic regression for non-parametric calibration
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(cal_predictions, y_cal_binary)

        elif method == "sigmoid":
            # Platt scaling with logistic regression
            self._calibrator = LogisticRegression()
            self._calibrator.fit(cal_predictions.reshape(-1, 1), y_cal_binary)

        else:
            raise ValueError(f"Unknown calibration method: {method}")

        self._calibration_method = method
        self._is_calibrated = True

        return self

    def predict_proba_calibrated(self, X: np.ndarray) -> np.ndarray:
        """Generate calibrated probability estimates.

        Requires calling calibrate() first.

        Args:
            X: Feature matrix

        Returns:
            Array of shape (n_samples, 2) with calibrated probabilities
        """
        if not getattr(self, "_is_calibrated", False):
            raise ValueError("Model not calibrated. Call calibrate() first.")

        predictions = self.predict(X)

        if self._calibration_method == "isotonic":
            prob_positive = self._calibrator.predict(predictions)
        else:  # sigmoid
            prob_positive = self._calibrator.predict_proba(
                predictions.reshape(-1, 1)
            )[:, 1]

        prob_positive = np.clip(prob_positive, 0.0, 1.0)
        prob_negative = 1.0 - prob_positive

        return np.column_stack([prob_negative, prob_positive])


def create_sample_weights(
    n_samples: int,
    decay: float = 0.99,
    method: str = "exponential",
) -> np.ndarray:
    """
    Create sample weights giving more importance to recent data.

    Args:
        n_samples: Number of samples
        decay: Decay factor (closer to 1 = slower decay)
        method: "exponential" or "linear"

    Returns:
        Array of sample weights
    """
    if method == "exponential":
        weights = decay ** np.arange(n_samples - 1, -1, -1)
    elif method == "linear":
        weights = np.linspace(1 - decay, 1.0, n_samples)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize to sum to n_samples
    return weights * (n_samples / weights.sum())
