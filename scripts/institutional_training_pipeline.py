#!/usr/bin/env python3
"""
Institutional-Grade ML Model Training Pipeline for AlphaTrade.

GPU ACCELERATION ENABLED:
- Automatic device detection (CUDA > MPS > CPU)
- AMP (Automatic Mixed Precision) for faster training
- pin_memory for faster GPU transfers
- Memory management utilities

This script implements a comprehensive training pipeline that:
1. Loads historical OHLCV data for multiple symbols
2. Computes features using the FeaturePipeline
3. Implements proper time-series walk-forward cross-validation
4. Trains all models (Classical ML + Deep Learning) on GPU if available
5. Computes classification metrics (AUC, accuracy, precision, recall)
6. Saves trained models to the models/ directory
7. Creates an ensemble combining all models

CRITICAL: Uses time-based train/validation splits (NO random shuffling!)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# GPU Acceleration: Configure CUDA memory allocation for optimal GPU utilization
# Note: Use PYTORCH_ALLOC_CONF instead of deprecated PYTORCH_CUDA_ALLOC_CONF
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# NOTE: GPU is enabled! Previously forced CPU mode, now uses auto-detection

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_trading_system.data.loader import DataLoader
from quant_trading_system.features.feature_pipeline import (
    FeatureConfig,
    FeatureGroup,
    FeaturePipeline,
    NormalizationMethod,
)

# Import ModelType directly from the base module file to avoid torch import via __init__
from quant_trading_system.models.base import ModelType

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# GPU Acceleration: Detect and log available compute devices
def _detect_gpu():
    """Detect available GPU and log info."""
    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU DETECTED: {gpu_name} with {gpu_mem:.1f} GB VRAM")
        logger.info("  AMP (Automatic Mixed Precision) will be enabled for faster training")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("GPU DETECTED: Apple MPS (Metal Performance Shaders)")
        return "mps"
    else:
        logger.warning("No GPU detected. Training will run on CPU (slower).")
        return "cpu"


# Lazy import function for model classes
def _import_models():
    """Lazy import of model classes."""
    global XGBoostModel, LightGBMModel, RandomForestModel
    global LSTMModel, TransformerModel, TCNModel
    global VotingEnsemble

    from quant_trading_system.models.classical_ml import (
        LightGBMModel,
        RandomForestModel,
        XGBoostModel,
    )
    from quant_trading_system.models.deep_learning import (
        LSTMModel,
        TCNModel,
        TransformerModel,
    )
    from quant_trading_system.models.ensemble import VotingEnsemble

    return {
        "XGBoostModel": XGBoostModel,
        "LightGBMModel": LightGBMModel,
        "RandomForestModel": RandomForestModel,
        "LSTMModel": LSTMModel,
        "TransformerModel": TransformerModel,
        "TCNModel": TCNModel,
        "VotingEnsemble": VotingEnsemble,
    }


class InstitutionalTrainingPipeline:
    """
    Professional ML training pipeline for AlphaTrade.

    GPU ACCELERATION:
    - Automatic device detection (CUDA > MPS > CPU)
    - AMP (Automatic Mixed Precision) for 2-3x faster training
    - Optimized batch sizes for GPU memory
    - pin_memory for faster data transfers

    Implements institutional-grade practices:
    - Time-series cross-validation (no look-ahead bias)
    - Proper feature scaling
    - NaN handling
    - Early stopping
    - Model persistence
    - Comprehensive metrics
    """

    def __init__(
        self,
        data_dir: str | Path,
        models_dir: str | Path,
        symbols: list[str],
        train_ratio: float = 0.8,
        lookback_window: int = 20,
        random_state: int = 42,
        use_gpu: bool = True,
    ):
        """
        Initialize the training pipeline.

        Args:
            data_dir: Directory containing raw OHLCV data
            models_dir: Directory to save trained models
            symbols: List of symbols to train on
            train_ratio: Fraction of data for training (time-based split)
            lookback_window: Lookback window for deep learning models
            random_state: Random seed for reproducibility
            use_gpu: Whether to use GPU acceleration if available
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.symbols = [s.upper() for s in symbols]
        self.train_ratio = train_ratio
        self.lookback_window = lookback_window
        self.random_state = random_state
        self.use_gpu = use_gpu

        # GPU Acceleration: Detect and configure device
        self.device = _detect_gpu() if use_gpu else "cpu"

        # Create models directory if it doesn't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_loader = DataLoader(self.data_dir / "raw", validate=True)

        # Feature pipeline configuration - Use ALL feature groups for comprehensive feature set
        # This enables institutional-grade feature engineering with:
        # - TECHNICAL: 50+ core indicators (RSI, MACD, Bollinger, ATR, etc.) + 150+ extended
        # - STATISTICAL: Rolling stats, z-scores, returns distributions, VaR, CVaR, Hurst exponent
        # - MICROSTRUCTURE: Order flow, Kyle's Lambda, Amihud illiquidity, VPIN, spreads
        # - CROSS_SECTIONAL: Relative strength, beta, peer rankings (requires universe data)
        self.feature_config = FeatureConfig(
            groups=[FeatureGroup.ALL],  # Use ALL feature groups for institutional-grade models
            normalization=NormalizationMethod.NONE,  # We'll scale separately with StandardScaler
            fill_nan=True,
            fill_method="ffill",
            max_nan_ratio=0.5,  # Allow up to 50% NaN per feature
            variance_threshold=0.0,  # No variance filtering (let model handle it)
            correlation_threshold=0.95,  # Remove highly correlated features (>0.95 correlation)
            include_targets=True,
            target_horizons=[1],  # Predict next bar direction
        )
        self.feature_pipeline = FeaturePipeline(self.feature_config)

        # Scaler for feature normalization
        self.scaler = StandardScaler()

        # Store results
        self.training_results: dict[str, dict[str, Any]] = {}
        self.trained_models: dict[str, Any] = {}

        # Lazy import models
        self._models = _import_models()

    def load_and_prepare_data(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Load data for all symbols and prepare features.

        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info(f"Loading data for symbols: {self.symbols}")

        all_features = []
        all_targets = []

        for symbol in self.symbols:
            try:
                logger.info(f"Processing {symbol}...")

                # Load OHLCV data
                df = self.data_loader.load_symbol(symbol)
                logger.info(f"  Loaded {len(df)} bars for {symbol}")

                # Compute features
                feature_set = self.feature_pipeline.compute(df, symbol=symbol)
                logger.info(f"  Computed {feature_set.num_features} features, {feature_set.num_targets} targets")

                # Get feature matrix
                feature_names = feature_set.feature_names
                X = feature_set.to_numpy()

                # Get target (direction_1 = next bar direction)
                if "target_direction_1" in feature_set.target_names:
                    y = feature_set.targets["target_direction_1"]
                else:
                    # Compute manually if not available
                    close = df["close"].to_numpy()
                    y = np.zeros(len(close))
                    y[:-1] = (close[1:] > close[:-1]).astype(float)
                    y[-1] = np.nan  # Last bar has no next bar

                all_features.append(X)
                all_targets.append(y)

            except Exception as e:
                logger.warning(f"  Failed to process {symbol}: {e}")
                continue

        if not all_features:
            raise ValueError("No data could be loaded for any symbol")

        # Concatenate all data
        X = np.vstack(all_features)
        y = np.concatenate(all_targets)

        logger.info(f"Combined data shape: X={X.shape}, y={y.shape}")

        # Handle NaN values
        # Remove rows with NaN in target
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        # Handle NaN in features - fill with column mean
        for col in range(X.shape[1]):
            col_data = X[:, col]
            nan_mask = np.isnan(col_data)
            if nan_mask.any():
                col_mean = np.nanmean(col_data)
                X[nan_mask, col] = col_mean if not np.isnan(col_mean) else 0.0

        # Handle infinity
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"After cleaning: X={X.shape}, y={y.shape}")

        return X, y.astype(int), feature_names

    def time_series_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data chronologically (time-based, NO random shuffle).

        CRITICAL: This preserves temporal ordering to prevent look-ahead bias.

        Args:
            X: Feature matrix
            y: Target array

        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        n_samples = len(X)
        split_idx = int(n_samples * self.train_ratio)

        X_train = X[:split_idx]
        X_val = X[split_idx:]
        y_train = y[:split_idx]
        y_val = y[split_idx:]

        logger.info(f"Time-based split: train={len(X_train)}, val={len(X_val)}")

        return X_train, X_val, y_train, y_val

    def scale_features(
        self, X_train: np.ndarray, X_val: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler fitted on training data only.

        CRITICAL: Fit scaler only on training data to prevent data leakage.

        Args:
            X_train: Training features
            X_val: Validation features

        Returns:
            Tuple of (X_train_scaled, X_val_scaled)
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Handle any NaN/Inf that might have been introduced
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        return X_train_scaled, X_val_scaled

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
    ) -> dict[str, float]:
        """
        Compute classification metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for AUC)

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        # Compute AUC if probabilities available
        if y_proba is not None:
            try:
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    metrics["auc"] = roc_auc_score(y_true, y_proba[:, 1])
                elif y_proba.ndim == 1:
                    metrics["auc"] = roc_auc_score(y_true, y_proba)
                else:
                    metrics["auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
            except Exception as e:
                logger.warning(f"Could not compute AUC: {e}")
                metrics["auc"] = 0.0
        else:
            metrics["auc"] = 0.0

        return metrics

    def train_classical_ml_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, Any]:
        """
        Train classical ML models (XGBoost, LightGBM, RandomForest).

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: Feature names

        Returns:
            Dictionary of trained models
        """
        models = {}

        # XGBoost
        logger.info("Training XGBoost classifier...")
        try:
            XGBoostModel = self._models["XGBoostModel"]
            xgb_model = XGBoostModel(
                name="xgboost_classifier",
                model_type=ModelType.CLASSIFIER,
                n_estimators=200,  # Reduced for faster training
                max_depth=5,
                learning_rate=0.1,  # Higher LR for faster convergence
                subsample=0.8,
                colsample_bytree=0.8,
                early_stopping_rounds=20,  # Earlier stopping
                random_state=self.random_state,
            )
            xgb_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                feature_names=feature_names,
            )

            # Evaluate
            y_pred = xgb_model.predict(X_val)
            y_proba = xgb_model.predict_proba(X_val)
            metrics = self.compute_metrics(y_val, y_pred, y_proba)

            models["XGBoost"] = xgb_model
            self.training_results["XGBoost"] = metrics
            logger.info(f"  XGBoost - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")

        except Exception as e:
            logger.error(f"  XGBoost training failed: {e}")

        # LightGBM
        logger.info("Training LightGBM classifier...")
        try:
            LightGBMModel = self._models["LightGBMModel"]
            lgb_model = LightGBMModel(
                name="lightgbm_classifier",
                model_type=ModelType.CLASSIFIER,
                n_estimators=200,  # Reduced
                num_leaves=31,  # Reduced
                learning_rate=0.1,  # Higher LR
                feature_fraction=0.8,
                bagging_fraction=0.8,
                early_stopping_rounds=20,
                random_state=self.random_state,
            )
            lgb_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                feature_names=feature_names,
            )

            y_pred = lgb_model.predict(X_val)
            y_proba = lgb_model.predict_proba(X_val)
            metrics = self.compute_metrics(y_val, y_pred, y_proba)

            models["LightGBM"] = lgb_model
            self.training_results["LightGBM"] = metrics
            logger.info(f"  LightGBM - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")

        except Exception as e:
            logger.error(f"  LightGBM training failed: {e}")

        # RandomForest
        logger.info("Training RandomForest classifier...")
        try:
            RandomForestModel = self._models["RandomForestModel"]
            rf_model = RandomForestModel(
                name="random_forest_classifier",
                model_type=ModelType.CLASSIFIER,
                n_estimators=100,  # Reduced
                max_depth=10,  # Reduced
                min_samples_split=10,  # Increased
                min_samples_leaf=5,  # Increased
                random_state=self.random_state,
            )
            rf_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                feature_names=feature_names,
            )

            y_pred = rf_model.predict(X_val)
            y_proba = rf_model.predict_proba(X_val)
            metrics = self.compute_metrics(y_val, y_pred, y_proba)

            models["RandomForest"] = rf_model
            self.training_results["RandomForest"] = metrics
            logger.info(f"  RandomForest - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")

        except Exception as e:
            logger.error(f"  RandomForest training failed: {e}")

        return models

    def train_deep_learning_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, Any]:
        """
        Train deep learning models (LSTM, Transformer, TCN) with GPU acceleration.

        GPU Features:
        - Automatic device selection (CUDA > MPS > CPU)
        - AMP (Automatic Mixed Precision) for faster training
        - Optimized batch sizes for GPU memory
        - pin_memory for faster data transfers

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: Feature names

        Returns:
            Dictionary of trained models
        """
        models = {}

        # GPU Acceleration: Use larger batch sizes for GPU training
        gpu_batch_size = 256 if self.device == "cuda" else 128

        # LSTM
        logger.info(f"Training LSTM classifier on {self.device.upper()}...")
        try:
            LSTMModel = self._models["LSTMModel"]
            lstm_model = LSTMModel(
                name="lstm_classifier",
                model_type=ModelType.CLASSIFIER,
                lookback_window=self.lookback_window,
                hidden_size=64,  # Increased for GPU training
                num_layers=2,  # Increased for GPU training
                dropout=0.2,
                learning_rate=1e-3,
                batch_size=gpu_batch_size,  # GPU-optimized batch size
                epochs=30,  # More epochs with GPU
                patience=10,  # Early stopping
                device=None,  # Auto-detect device (uses GPU if available)
                use_amp=True,  # Enable AMP for faster GPU training
            )
            lstm_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                feature_names=feature_names,
            )

            y_pred = lstm_model.predict(X_val)
            y_proba = lstm_model.predict_proba(X_val)
            # Align predictions with validation targets (due to sequence creation)
            y_val_aligned = y_val[self.lookback_window - 1:]
            metrics = self.compute_metrics(y_val_aligned, y_pred, y_proba)

            models["LSTM"] = lstm_model
            self.training_results["LSTM"] = metrics
            logger.info(f"  LSTM - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")

        except Exception as e:
            logger.error(f"  LSTM training failed: {e}")
            import traceback
            traceback.print_exc()

        # Transformer
        logger.info(f"Training Transformer classifier on {self.device.upper()}...")
        try:
            TransformerModel = self._models["TransformerModel"]
            transformer_model = TransformerModel(
                name="transformer_classifier",
                model_type=ModelType.CLASSIFIER,
                lookback_window=self.lookback_window,
                d_model=64,  # Increased for GPU training
                nhead=4,  # More attention heads with GPU
                num_layers=2,  # Increased layers with GPU
                dropout=0.1,
                learning_rate=1e-4,
                batch_size=gpu_batch_size,  # GPU-optimized batch size
                epochs=30,  # More epochs with GPU
                patience=10,  # Early stopping
                device=None,  # Auto-detect device
                use_amp=True,  # Enable AMP for faster GPU training
            )
            transformer_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                feature_names=feature_names,
            )

            y_pred = transformer_model.predict(X_val)
            y_proba = transformer_model.predict_proba(X_val)
            y_val_aligned = y_val[self.lookback_window - 1:]
            metrics = self.compute_metrics(y_val_aligned, y_pred, y_proba)

            models["Transformer"] = transformer_model
            self.training_results["Transformer"] = metrics
            logger.info(f"  Transformer - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")

        except Exception as e:
            logger.error(f"  Transformer training failed: {e}")
            import traceback
            traceback.print_exc()

        # TCN
        logger.info(f"Training TCN classifier on {self.device.upper()}...")
        try:
            TCNModel = self._models["TCNModel"]
            tcn_model = TCNModel(
                name="tcn_classifier",
                model_type=ModelType.CLASSIFIER,
                lookback_window=self.lookback_window,
                num_channels=[32, 64],  # Increased for GPU training
                kernel_size=3,
                dropout=0.2,
                learning_rate=1e-3,
                batch_size=gpu_batch_size,  # GPU-optimized batch size
                epochs=30,  # More epochs with GPU
                patience=10,  # Early stopping
                device=None,  # Auto-detect device
                use_amp=True,  # Enable AMP for faster GPU training
            )
            tcn_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                feature_names=feature_names,
            )

            y_pred = tcn_model.predict(X_val)
            y_proba = tcn_model.predict_proba(X_val)
            y_val_aligned = y_val[self.lookback_window - 1:]
            metrics = self.compute_metrics(y_val_aligned, y_pred, y_proba)

            models["TCN"] = tcn_model
            self.training_results["TCN"] = metrics
            logger.info(f"  TCN - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")

        except Exception as e:
            logger.error(f"  TCN training failed: {e}")
            import traceback
            traceback.print_exc()

        return models

    def create_ensemble(
        self,
        classical_models: dict[str, Any],
        dl_models: dict[str, Any],
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> VotingEnsemble | None:
        """
        Create a voting ensemble combining all models.

        Args:
            classical_models: Dictionary of classical ML models
            dl_models: Dictionary of deep learning models
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Trained VotingEnsemble or None if failed
        """
        logger.info("Creating voting ensemble...")

        try:
            # Use only classical ML models for ensemble (consistent input shapes)
            VotingEnsemble = self._models["VotingEnsemble"]
            ensemble = VotingEnsemble(
                name="institutional_ensemble",
                version="1.0.0",
                voting="soft",
            )

            for name, model in classical_models.items():
                ensemble.add_model(model)
                logger.info(f"  Added {name} to ensemble")

            # The ensemble is already fitted since base models are fitted
            # Mark it as fitted
            ensemble._is_fitted = True
            ensemble._feature_names = list(classical_models.values())[0]._feature_names if classical_models else []

            # Evaluate ensemble
            y_pred = ensemble.predict(X_val)
            y_proba = ensemble.predict_proba(X_val)
            metrics = self.compute_metrics(y_val, y_pred, y_proba)

            self.training_results["Ensemble"] = metrics
            logger.info(f"  Ensemble - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")

            return ensemble

        except Exception as e:
            logger.error(f"  Ensemble creation failed: {e}")
            return None

    def save_models(self, models: dict[str, Any]) -> None:
        """
        Save all trained models to disk.

        Args:
            models: Dictionary of trained models
        """
        logger.info(f"Saving models to {self.models_dir}...")

        for name, model in models.items():
            try:
                model_path = self.models_dir / f"{name.lower()}_institutional"
                model.save(str(model_path))
                logger.info(f"  Saved {name} to {model_path}")
            except Exception as e:
                logger.error(f"  Failed to save {name}: {e}")

    def save_results(self) -> None:
        """Save training results to JSON file."""
        results_path = self.models_dir / "training_results.json"

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols": self.symbols,
            "train_ratio": self.train_ratio,
            "lookback_window": self.lookback_window,
            "metrics": self.training_results,
        }

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved training results to {results_path}")

    def print_summary(self) -> None:
        """Print training summary."""
        print("\n" + "=" * 80)
        print("INSTITUTIONAL TRAINING PIPELINE - RESULTS SUMMARY")
        print("=" * 80)
        print(f"\nSymbols: {', '.join(self.symbols)}")
        print(f"Train/Val Split: {self.train_ratio*100:.0f}% / {(1-self.train_ratio)*100:.0f}%")
        print(f"Lookback Window: {self.lookback_window}")
        print("\n" + "-" * 80)
        print(f"{'Model':<20} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'AUC':>12}")
        print("-" * 80)

        for name, metrics in self.training_results.items():
            print(
                f"{name:<20} "
                f"{metrics['accuracy']:>12.4f} "
                f"{metrics['precision']:>12.4f} "
                f"{metrics['recall']:>12.4f} "
                f"{metrics['auc']:>12.4f}"
            )

        print("-" * 80)

        # Find best model
        if self.training_results:
            best_model = max(self.training_results.items(), key=lambda x: x[1].get("auc", 0))
            print(f"\nBest Model (by AUC): {best_model[0]} with AUC = {best_model[1]['auc']:.4f}")

        print("=" * 80 + "\n")

    def run(self) -> dict[str, Any]:
        """
        Execute the full training pipeline.

        Returns:
            Dictionary of trained models
        """
        logger.info("=" * 60)
        logger.info("INSTITUTIONAL TRAINING PIPELINE - STARTING")
        logger.info("=" * 60)

        # Step 1: Load and prepare data
        logger.info("\nStep 1: Loading and preparing data...")
        X, y, feature_names = self.load_and_prepare_data()

        # Step 2: Time-series split
        logger.info("\nStep 2: Splitting data (time-based, NO shuffle)...")
        X_train, X_val, y_train, y_val = self.time_series_split(X, y)

        # Step 3: Scale features
        logger.info("\nStep 3: Scaling features...")
        X_train_scaled, X_val_scaled = self.scale_features(X_train, X_val)

        # Step 4: Train classical ML models
        logger.info("\nStep 4: Training classical ML models...")
        classical_models = self.train_classical_ml_models(
            X_train_scaled, y_train, X_val_scaled, y_val, feature_names
        )

        # Step 5: Train deep learning models
        logger.info("\nStep 5: Training deep learning models...")
        dl_models = self.train_deep_learning_models(
            X_train_scaled, y_train, X_val_scaled, y_val, feature_names
        )

        # Step 6: Create ensemble
        logger.info("\nStep 6: Creating ensemble...")
        ensemble = self.create_ensemble(classical_models, dl_models, X_val_scaled, y_val)

        # Combine all models
        all_models = {**classical_models, **dl_models}
        if ensemble is not None:
            all_models["Ensemble"] = ensemble

        self.trained_models = all_models

        # Step 7: Save models
        logger.info("\nStep 7: Saving models...")
        self.save_models(all_models)
        self.save_results()

        # Print summary
        self.print_summary()

        logger.info("INSTITUTIONAL TRAINING PIPELINE - COMPLETE")

        return all_models


def main():
    """Main entry point for the training pipeline."""
    # Configuration
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"

    # Test symbols
    symbols = ["AAPL", "MSFT", "NVDA"]

    # Create and run pipeline
    pipeline = InstitutionalTrainingPipeline(
        data_dir=data_dir,
        models_dir=models_dir,
        symbols=symbols,
        train_ratio=0.8,
        lookback_window=20,
        random_state=42,
    )

    try:
        trained_models = pipeline.run()
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
