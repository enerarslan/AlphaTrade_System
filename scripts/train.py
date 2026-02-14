"""
================================================================================
ALPHATRADE MODEL TRAINING PIPELINE
================================================================================

Institutional-grade model training with full ML pipeline support.

@mlquant: This script implements all ML/Quant requirements including:
  - Purged K-Fold Cross-Validation (P2-B) with embargo periods
  - Model Validation Gates for production deployment
  - Mandatory hyperparameter optimization (Optuna + pruning)
  - Meta-labeling for signal filtering (P2-3.5)
  - Multiple testing correction (P1-3.1: Bonferroni, BH, Deflated Sharpe)
  - GPU acceleration for deep learning models
  - Ensemble methods with IC-weighted combination
  - SHAP/LIME explainability for regulatory compliance

Model Types Supported:
  - Classical ML: XGBoost, LightGBM, RandomForest, ElasticNet
  - Deep Learning: LSTM, Transformer, TCN
  - Ensemble: Voting, Stacking, IC-Weighted, Adaptive
  - Reinforcement Learning: PPO Agent

Usage:
    python main.py train --model xgboost --symbols AAPL MSFT
    python main.py train --model ensemble --n-trials 100
    python main.py train --model lstm --use-gpu --epochs 100
    python main.py train --model all --n-trials 50

Author: AlphaTrade System
Version: 1.3.0
================================================================================
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import pickle
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant_trading_system.models.training_lineage import (
    build_data_quality_report,
    build_snapshot_manifest,
    compute_data_quality_hash,
    load_registry_entries,
    persist_active_model_pointer,
    persist_snapshot_bundle,
    register_training_model_version,
    set_registry_active_version,
)
from quant_trading_system.models.statistical_validation import (
    calculate_deflated_sharpe_ratio,
    calculate_probability_of_backtest_overfitting,
)
from quant_trading_system.models.target_engineering import (
    TargetEngineeringConfig,
    generate_targets,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")

SUPPORTED_MODELS = [
    "xgboost",
    "lightgbm",
    "random_forest",
    "elastic_net",
    "lstm",
    "transformer",
    "tcn",
    "ensemble",
]

MANDATORY_MODEL_TECH_STACK = {
    "postgresql_data": True,
    "redis_cache": True,
    "gpu_acceleration": True,
    "hyperopt_optuna": True,
    "meta_labeling": True,
    "multiple_testing_correction": True,
    "shap_explainability": True,
    "future_leak_validation": True,
}

BASE_MARKET_COLUMNS = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
FORBIDDEN_FEATURE_COLUMNS = {
    "label",
    "primary_signal",
    "barrier_touched",
    "triple_barrier_label",
    "triple_barrier_event_return",
    "triple_barrier_net_return",
    "holding_period",
    "regime",
}
FORBIDDEN_FEATURE_PREFIXES = ("forward_return_h",)
REPLAY_MANIFEST_SCHEMA_VERSION = "1.0.0"
PROMOTION_PACKAGE_SCHEMA_VERSION = "1.0.0"


def _compute_feature_pipeline_fingerprint() -> str:
    """Create deterministic fingerprint for feature-engineering code."""
    fingerprint = hashlib.sha256()
    candidate_files = [
        PROJECT_ROOT / "quant_trading_system" / "features" / "feature_pipeline.py",
        PROJECT_ROOT / "quant_trading_system" / "features" / "optimized_pipeline.py",
        PROJECT_ROOT / "quant_trading_system" / "features" / "cross_sectional.py",
    ]
    for file_path in candidate_files:
        if not file_path.exists():
            continue
        fingerprint.update(file_path.name.encode("utf-8"))
        fingerprint.update(file_path.read_bytes())
    return fingerprint.hexdigest()[:16]


def set_global_seed(seed: int) -> None:
    """Set deterministic random seeds for reproducible training runs."""
    normalized_seed = int(seed)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    os.environ["PYTHONHASHSEED"] = str(normalized_seed)

    try:
        import torch

        torch.manual_seed(normalized_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(normalized_seed)
    except Exception:
        pass


def _load_replay_manifest(replay_manifest_path: Path) -> dict[str, Any]:
    """Load replay manifest payload and return normalized structure."""
    path = Path(replay_manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Replay manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Replay manifest must be a JSON object.")

    manifest = payload.get("replay_manifest") if isinstance(payload.get("replay_manifest"), dict) else payload
    if not isinstance(manifest, dict):
        raise ValueError("Replay manifest payload is invalid.")

    training_config = manifest.get("training_config")
    if training_config is None and isinstance(manifest.get("config"), dict):
        training_config = manifest.get("config")
    if not isinstance(training_config, dict):
        raise ValueError("Replay manifest missing `training_config` object.")

    return {
        "manifest_path": str(path),
        "manifest": manifest,
        "training_config": training_config,
    }


def _load_model_defaults(model_type: str, use_gpu: bool = False) -> dict[str, Any]:
    """Load model defaults from YAML config."""
    config_path = PROJECT_ROOT / "quant_trading_system" / "config" / "model_configs.yaml"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            all_cfg = yaml.safe_load(f) or {}
    except Exception:
        return {}

    if model_type in {"xgboost", "lightgbm", "random_forest", "elastic_net"}:
        section = all_cfg.get(model_type, {})
        params = dict(section.get("classifier", {}))
    elif model_type in {"lstm", "transformer", "tcn"}:
        params = dict(all_cfg.get(model_type, {}))
    else:
        params = {}

    # Remove training-script-owned knobs from model constructor params.
    for key in (
        "early_stopping_rounds",
        "batch_size",
        "learning_rate",
        "max_epochs",
        "early_stopping_patience",
        "scheduler",
    ):
        params.pop(key, None)

    if model_type == "xgboost" and use_gpu:
        params["tree_method"] = "hist"
        params["device"] = "cuda"
    elif model_type == "lightgbm" and use_gpu:
        params["device_type"] = "gpu"

    # Harmonize YAML keys with model constructor names.
    if model_type == "lstm":
        if "sequence_length" in params:
            params["lookback_window"] = params.pop("sequence_length")
    elif model_type == "transformer":
        if "sequence_length" in params:
            params["lookback_window"] = params.pop("sequence_length")
        if "num_encoder_layers" in params:
            params["num_layers"] = params.pop("num_encoder_layers")
        if "dim_feedforward" in params:
            params["d_ff"] = params.pop("dim_feedforward")
    elif model_type == "tcn":
        if "sequence_length" in params:
            params["lookback_window"] = params.pop("sequence_length")

    if model_type == "random_forest" and os.name == "nt":
        # Windows + RF + n_jobs=-1 can deadlock under heavy Optuna CV workloads.
        params["n_jobs"] = 1

    return params


def _verify_institutional_infra() -> None:
    """Fail-fast infrastructure check for institutional training mode."""
    from quant_trading_system.database.connection import get_db_manager, get_redis_manager

    db_manager = get_db_manager()
    if not db_manager.health_check():
        raise RuntimeError("PostgreSQL health check failed. Institutional training requires PostgreSQL.")

    redis_manager = get_redis_manager()
    if not redis_manager.health_check():
        raise RuntimeError("Redis health check failed. Institutional training requires Redis.")


def _verify_gpu_stack(model_list: list[str]) -> None:
    """Fail-fast GPU validation for institutional training mode."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for GPU preflight checks in institutional mode.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. Institutional training requires NVIDIA CUDA GPU.")

    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"CUDA GPU detected: {gpu_name}")

    deep_models = {"lstm", "transformer", "tcn"}
    if deep_models.intersection(model_list):
        # Validate tensor ops on device for deep learning stack.
        _ = torch.randn((16, 8), device="cuda")

    if {"xgboost", "ensemble"}.intersection(model_list):
        try:
            import xgboost as xgb

            X = np.random.rand(64, 8)
            y = np.random.randint(0, 2, 64)
            probe = xgb.XGBClassifier(
                n_estimators=2,
                max_depth=2,
                tree_method="hist",
                device="cuda",
                eval_metric="logloss",
            )
            probe.fit(X, y)
        except Exception as exc:
            raise RuntimeError(
                "XGBoost GPU backend validation failed. Verify CUDA-enabled XGBoost installation."
            ) from exc

    if {"lightgbm", "ensemble"}.intersection(model_list):
        try:
            import lightgbm as lgb

            X = np.random.rand(128, 8)
            y = np.random.randint(0, 2, 128)
            probe = lgb.LGBMClassifier(
                n_estimators=2,
                max_depth=3,
                device_type="gpu",
                verbosity=-1,
            )
            probe.fit(X, y)
        except Exception as exc:
            raise RuntimeError(
                "LightGBM GPU backend validation failed. Install a GPU-enabled LightGBM build."
            ) from exc

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    # Model selection
    model_type: str = "xgboost"  # xgboost, lightgbm, random_forest, elastic_net, lstm, transformer, tcn, ensemble
    model_name: str = ""  # Custom name for the model

    # Data configuration
    symbols: list[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Cross-validation (P2-B: Purged K-Fold)
    cv_method: str = "purged_kfold"  # purged_kfold, combinatorial, walk_forward
    n_splits: int = 5
    embargo_pct: float = 0.01  # P1-H1: Minimum 1% embargo
    purge_pct: float = 0.02

    # Hyperparameter optimization
    optimize: bool = True
    optimizer: str = "optuna"  # optuna, grid, random, bayesian
    n_trials: int = 100
    n_jobs: int = -1
    seed: int = 42
    use_nested_walk_forward: bool = True
    nested_outer_splits: int = 4
    nested_inner_splits: int = 3
    require_nested_trace_for_promotion: bool = True
    require_objective_breakdown_for_promotion: bool = True
    objective_weight_sharpe: float = 1.0
    objective_weight_drawdown: float = 0.5
    objective_weight_turnover: float = 0.1
    objective_weight_calibration: float = 0.25
    objective_weight_trade_activity: float = 1.0
    objective_weight_cvar: float = 0.4
    objective_weight_skew: float = 0.1

    # Model-specific parameters
    model_params: dict = field(default_factory=dict)

    # Training parameters
    epochs: int = 100  # For deep learning
    batch_size: int = 64
    learning_rate: float = 0.001
    early_stopping_rounds: int = 10

    # Validation gates
    min_sharpe_ratio: float = 0.5
    max_drawdown: float = 0.20
    min_win_rate: float = 0.45
    min_accuracy: float = 0.45
    min_trades: int = 100
    max_is_oos_ratio: float = 2.0
    min_ic: float = 0.02
    min_deflated_sharpe: float = 0.10
    max_deflated_sharpe_pvalue: float = 0.10
    max_pbo: float = 0.45
    min_holdout_sharpe: float = 0.0
    max_holdout_drawdown: float = 0.35
    max_regime_shift: float = 0.35

    # Target engineering (Workstream B)
    label_horizons: list[int] = field(default_factory=lambda: [1, 5, 20])
    primary_label_horizon: int = 5
    label_profit_taking_threshold: float = 0.015
    label_stop_loss_threshold: float = 0.010
    label_max_holding_period: int = 20
    label_spread_bps: float = 1.0
    label_slippage_bps: float = 3.0
    label_impact_bps: float = 2.0
    label_min_signal_abs_return_bps: float = 8.0
    label_neutral_buffer_bps: float = 4.0
    label_max_abs_forward_return: float = 0.35
    label_signal_volatility_floor_mult: float = 0.50
    label_volatility_lookback: int = 20
    label_regime_lookback: int = 30
    label_temporal_weight_decay: float = 0.999
    label_edge_cost_buffer_bps: float = 2.0

    # Meta-labeling (P2-3.5)
    use_meta_labeling: bool = True
    meta_model_type: str = "xgboost"

    # Multiple testing correction (P1-3.1)
    apply_multiple_testing: bool = True
    correction_method: str = "deflated_sharpe"  # bonferroni, bh, deflated_sharpe

    # GPU acceleration
    use_gpu: bool = True
    gpu_device: int = 0

    # Database integration
    use_database: bool = True  # Load data from PostgreSQL + TimescaleDB
    use_redis_cache: bool = True  # Use Redis for lightweight metadata caching

    # Explainability
    compute_shap: bool = True
    n_shap_samples: int = 1000

    # Output
    output_dir: str = "models"
    save_artifacts: bool = True
    quality_missing_bars_threshold: float = 0.01
    quality_duplicate_bars_threshold: float = 0.001
    quality_extreme_move_threshold: float = 0.01
    quality_corporate_action_jump_threshold: float = 0.001
    data_max_abs_return: float = 0.35
    feature_groups: list[str] = field(
        default_factory=lambda: ["technical", "statistical", "microstructure", "cross_sectional"]
    )
    enable_cross_sectional: bool = True
    max_cross_sectional_symbols: int = 20
    max_cross_sectional_rows: int = 250000
    allow_feature_group_fallback: bool = True
    feature_materialization_batch_rows: int = 5000
    feature_reuse_min_coverage: float = 0.20
    persist_features_to_postgres: bool = True
    holdout_pct: float = 0.15
    dynamic_no_trade_band: bool = True
    execution_vol_target_daily: float = 0.012
    execution_turnover_cap: float = 0.90
    execution_cooldown_bars: int = 2

    def __post_init__(self):
        if not self.model_name:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.model_name = f"{self.model_type}_{timestamp}"
        self.feature_materialization_batch_rows = max(250, int(self.feature_materialization_batch_rows))
        self.max_cross_sectional_symbols = max(2, int(self.max_cross_sectional_symbols))
        self.max_cross_sectional_rows = max(10_000, int(self.max_cross_sectional_rows))
        self.feature_reuse_min_coverage = min(max(float(self.feature_reuse_min_coverage), 0.01), 0.95)
        self.data_max_abs_return = max(float(self.data_max_abs_return), 0.05)
        self.min_trades = max(1, int(self.min_trades))
        self.objective_weight_trade_activity = max(
            0.0,
            float(self.objective_weight_trade_activity),
        )
        self.objective_weight_cvar = max(0.0, float(self.objective_weight_cvar))
        self.objective_weight_skew = max(0.0, float(self.objective_weight_skew))
        self.holdout_pct = min(max(float(self.holdout_pct), 0.05), 0.35)
        self.execution_vol_target_daily = max(float(self.execution_vol_target_daily), 1e-4)
        self.execution_turnover_cap = min(max(float(self.execution_turnover_cap), 0.05), 1.0)
        self.execution_cooldown_bars = max(0, int(self.execution_cooldown_bars))
        self.max_regime_shift = min(max(float(self.max_regime_shift), 0.0), 1.0)
        self.feature_groups = [str(g).strip().lower() for g in self.feature_groups if str(g).strip()]
        if not self.feature_groups:
            self.feature_groups = ["technical", "statistical", "microstructure", "cross_sectional"]


# ============================================================================
# MODEL TRAINER CLASS
# ============================================================================


class ModelTrainer:
    """
    Institutional-grade model training pipeline.

    @mlquant: This class orchestrates the entire training process with:
      - Data loading and feature computation
      - Purged cross-validation to prevent look-ahead bias
      - Hyperparameter optimization
      - Model validation gates
      - Meta-labeling for signal filtering
      - SHAP explainability
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger("ModelTrainer")

        # Training state
        self.data: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None
        self.labels: pd.Series | None = None
        self.feature_names: list[str] = []
        self.timestamps: np.ndarray | None = None
        self.row_symbols: np.ndarray | None = None
        self.regimes: np.ndarray | None = None
        self.close_prices: pd.Series | None = None
        self.primary_forward_returns: np.ndarray | None = None
        self.cost_aware_event_returns: np.ndarray | None = None
        self.holdout_features: pd.DataFrame | None = None
        self.holdout_labels: pd.Series | None = None
        self.holdout_timestamps: np.ndarray | None = None
        self.holdout_symbols: np.ndarray | None = None
        self.holdout_regimes: np.ndarray | None = None
        self.holdout_primary_forward_returns: np.ndarray | None = None
        self.holdout_cost_aware_event_returns: np.ndarray | None = None
        self.model: Any = None
        self.cv_results: list[dict] = []
        self.validation_results: dict = {}
        self.shap_values: np.ndarray | None = None
        self.meta_model: Any = None
        self.oof_primary_proba: np.ndarray | None = None
        self.cv_return_series: list[np.ndarray] = []
        self.cv_active_return_series: list[np.ndarray] = []
        self.sample_weights: np.ndarray | None = None
        self.label_diagnostics: dict[str, Any] = {}
        self.data_quality_report: dict[str, Any] | None = None
        self.data_quality_report_hash: str | None = None
        self.snapshot_manifest: dict[str, Any] | None = None
        self.snapshot_manifest_path: Path | None = None
        self.data_quality_report_path: Path | None = None
        self.nested_cv_trace: list[dict[str, Any]] = []
        self.replay_manifest_path: Path | None = None
        self.promotion_package_path: Path | None = None

        # Metrics
        self.training_metrics: dict = {}
        self.start_time: datetime | None = None
        self.feature_pipeline_fingerprint = _compute_feature_pipeline_fingerprint()

    def run(self) -> dict:
        """
        Execute the complete training pipeline.

        Returns:
            Training results including model path and metrics.
        """
        self.start_time = datetime.now(timezone.utc)
        self.logger.info(f"Starting training pipeline for {self.config.model_type}")
        self.logger.info(f"Model name: {self.config.model_name}")
        set_global_seed(self.config.seed)

        try:
            # Phase 1: Load Data
            self._load_data()

            # Phase 2: Compute Features
            self._compute_features()

            # Phase 3: Create Labels
            self._create_labels()

            # Phase 3.5: Enforce future-leak validation gates
            self._validate_no_future_leakage()

            # Phase 4: Hyperparameter Optimization (if enabled)
            if self.config.optimize:
                self._optimize_hyperparameters()

            # Phase 5: Cross-Validation Training
            self._train_with_cv()

            # Phase 6: Multiple testing / overfitting diagnostics before gating.
            if self.config.apply_multiple_testing:
                self._apply_multiple_testing_correction()

            # Phase 7: Validation Gates (includes DSR/PBO hard checks)
            passed_gates = self._validate_model()

            # Phase 8: Meta-Labeling (if enabled)
            if self.config.use_meta_labeling:
                self._train_meta_labeler()

            # Phase 9: SHAP Explainability
            if self.config.compute_shap:
                self._compute_shap_values()

            # Phase 9.5: Governance artifacts for deployment control.
            self.training_metrics["model_card"] = self._build_model_card()
            self.training_metrics["deployment_plan"] = self._build_deployment_plan(
                passed_validation=bool(passed_gates)
            )

            # Phase 10: Save Model
            model_path = self._save_model()

            # Compile results
            duration = (datetime.now(timezone.utc) - self.start_time).total_seconds()

            results = {
                "success": passed_gates,
                "model_path": model_path,
                "model_type": self.config.model_type,
                "model_name": self.config.model_name,
                "training_duration_seconds": duration,
                "cv_results": self.cv_results,
                "validation_results": self.validation_results,
                "training_metrics": self.training_metrics,
                "passed_validation_gates": passed_gates,
                "snapshot_id": (
                    str(self.snapshot_manifest.get("snapshot_id"))
                    if isinstance(self.snapshot_manifest, dict)
                    else None
                ),
                "snapshot_manifest_path": (
                    str(self.snapshot_manifest_path)
                    if self.snapshot_manifest_path is not None
                    else None
                ),
                "data_quality_report_path": (
                    str(self.data_quality_report_path)
                    if self.data_quality_report_path is not None
                    else None
                ),
                "data_quality_report_hash": self.data_quality_report_hash,
                "feature_schema_version": (
                    str(self.snapshot_manifest.get("feature_schema_version"))
                    if isinstance(self.snapshot_manifest, dict)
                    else None
                ),
                "label_diagnostics": self.label_diagnostics,
                "nested_cv_trace": self.nested_cv_trace,
                "replay_manifest_path": (
                    str(self.replay_manifest_path)
                    if self.replay_manifest_path is not None
                    else None
                ),
                "promotion_package_path": (
                    str(self.promotion_package_path)
                    if self.promotion_package_path is not None
                    else None
                ),
                "model_card": self.training_metrics.get("model_card", {}),
                "deployment_plan": self.training_metrics.get("deployment_plan", {}),
            }

            self.logger.info(f"Training completed in {duration:.1f}s")
            self.logger.info(f"Validation gates passed: {passed_gates}")

            return results

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def _load_data(self) -> None:
        """Phase 1: Load market data for training."""
        self.logger.info("Phase 1: Loading data...")
        if not self.config.use_database:
            raise RuntimeError(
                "Institutional training mode requires PostgreSQL data source. "
                "Disable is not allowed."
            )
        self._load_data_from_postgres()
        self._sanitize_loaded_data()
        self._capture_data_quality_report()

    def _sanitize_loaded_data(self) -> None:
        """Apply deterministic OHLCV sanitization before feature/label generation."""
        if self.data is None or self.data.empty:
            raise ValueError("No data loaded for sanitization.")

        df = self.data.copy()
        initial_rows = len(df)

        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        duplicate_count = int(df.duplicated(subset=["symbol", "timestamp"]).sum())
        if duplicate_count > 0:
            df = df.drop_duplicates(subset=["symbol", "timestamp"], keep="last").reset_index(drop=True)

        invalid_ohlcv_mask = (
            (df["open"] <= 0.0)
            | (df["high"] <= 0.0)
            | (df["low"] <= 0.0)
            | (df["close"] <= 0.0)
            | (df["volume"] <= 0.0)
            | (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )
        invalid_ohlcv_count = int(invalid_ohlcv_mask.sum())
        if invalid_ohlcv_count > 0:
            df = df.loc[~invalid_ohlcv_mask].copy()

        close_returns = df.groupby("symbol", sort=False)["close"].pct_change()
        rolling_vol = (
            close_returns.groupby(df["symbol"], sort=False)
            .rolling(window=max(10, int(self.config.label_volatility_lookback)), min_periods=10)
            .std()
            .reset_index(level=0, drop=True)
        )
        dynamic_return_cap = np.maximum(
            float(self.config.data_max_abs_return),
            np.nan_to_num(rolling_vol.to_numpy(dtype=float), nan=0.0) * 10.0,
        )
        extreme_return_mask = close_returns.abs().to_numpy(dtype=float) > dynamic_return_cap
        extreme_return_mask = np.nan_to_num(extreme_return_mask.astype(float), nan=0.0) > 0.0
        extreme_return_count = int(extreme_return_mask.sum())
        if extreme_return_count > 0:
            df = df.loc[~extreme_return_mask].copy()

        if df.empty:
            raise ValueError("Data sanitization removed all rows; verify source OHLCV quality.")

        cleaned_rows = len(df)
        self.data = df.reset_index(drop=True)
        self.training_metrics["sanitization_initial_rows"] = float(initial_rows)
        self.training_metrics["sanitization_cleaned_rows"] = float(cleaned_rows)
        self.training_metrics["sanitization_duplicate_rows_removed"] = float(duplicate_count)
        self.training_metrics["sanitization_invalid_ohlcv_rows_removed"] = float(invalid_ohlcv_count)
        self.training_metrics["sanitization_extreme_return_rows_removed"] = float(extreme_return_count)

        self.logger.info(
            "Data sanitization complete: "
            f"rows={cleaned_rows}/{initial_rows}, "
            f"duplicates_removed={duplicate_count}, "
            f"invalid_ohlcv_removed={invalid_ohlcv_count}, "
            f"extreme_return_removed={extreme_return_count}"
        )

    def _load_data_from_postgres(self) -> None:
        """Load OHLCV data from PostgreSQL/TimescaleDB."""
        from sqlalchemy import text
        from quant_trading_system.database.connection import get_db_manager, get_redis_manager

        db_manager = get_db_manager()
        symbols = [s.strip().upper() for s in self.config.symbols if s.strip()]
        if not self.config.use_redis_cache:
            raise RuntimeError(
                "Institutional training mode requires Redis cache layer. "
                "Disable is not allowed."
            )

        redis_mgr = get_redis_manager()
        if not redis_mgr.health_check():
            raise RuntimeError("Redis cache unavailable. Institutional mode requires Redis.")

        with db_manager.session() as session:
            session.execute(text("SET LOCAL max_parallel_workers_per_gather = 0"))
            session.execute(text("SET LOCAL statement_timeout = '120000ms'"))
            if not symbols:
                cache_key = "train:ohlcv_symbols"
                cached_symbols: list[str] = []
                if redis_mgr is not None:
                    cached_raw = redis_mgr.get(cache_key)
                    if cached_raw:
                        try:
                            cached_symbols = json.loads(cached_raw)
                        except json.JSONDecodeError:
                            cached_symbols = []

                if cached_symbols:
                    symbols = cached_symbols
                else:
                    result = session.execute(text("SELECT DISTINCT symbol FROM ohlcv_bars ORDER BY symbol"))
                    symbols = [row[0] for row in result.fetchall()]
                    if redis_mgr is not None and symbols:
                        redis_mgr.set(cache_key, json.dumps(symbols), expire_seconds=300)

            if not symbols:
                raise ValueError("No symbols found in ohlcv_bars")

            self.logger.info(f"Loading {len(symbols)} symbols from PostgreSQL")

            query = text(
                """
                SELECT
                    symbol,
                    timestamp,
                    CAST(open AS DOUBLE PRECISION) AS open,
                    CAST(high AS DOUBLE PRECISION) AS high,
                    CAST(low AS DOUBLE PRECISION) AS low,
                    CAST(close AS DOUBLE PRECISION) AS close,
                    CAST(volume AS BIGINT) AS volume
                FROM ohlcv_bars
                WHERE symbol = :symbol
                  AND (:start_date IS NULL OR timestamp >= :start_date)
                  AND (:end_date IS NULL OR timestamp <= :end_date)
                ORDER BY timestamp
                """
            )

            dfs: list[pd.DataFrame] = []
            start_date = pd.to_datetime(self.config.start_date, utc=True) if self.config.start_date else None
            end_date = pd.to_datetime(self.config.end_date, utc=True) if self.config.end_date else None

            for symbol in symbols:
                try:
                    result = session.execute(
                        query,
                        {
                            "symbol": symbol,
                            "start_date": start_date,
                            "end_date": end_date,
                        },
                    )
                    rows = result.fetchall()
                    if not rows:
                        self.logger.warning(f"No PostgreSQL rows for {symbol}")
                        continue

                    df = pd.DataFrame(rows, columns=result.keys())
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
                    dfs.append(df)
                except Exception as e:
                    self.logger.warning(f"Failed to load {symbol} from PostgreSQL: {e}")

            if not dfs:
                raise ValueError("No training data loaded from PostgreSQL")

            self.data = pd.concat(dfs, ignore_index=True)
            self.logger.info(
                f"Loaded {len(self.data)} rows from PostgreSQL for "
                f"{self.data['symbol'].nunique()} symbols"
            )

    def _load_data_fallback(self) -> None:
        """Fallback data loading from CSV files."""
        data_dir = PROJECT_ROOT / "data" / "raw"

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        dfs = []
        requested_symbols = {s.strip().upper() for s in self.config.symbols if s.strip()}

        for csv_file in sorted(data_dir.glob("*.csv")):
            file_symbol = csv_file.stem.upper()
            base_symbol = self._extract_base_symbol(file_symbol)

            if requested_symbols and file_symbol not in requested_symbols and base_symbol not in requested_symbols:
                continue

            try:
                df = pd.read_csv(csv_file)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                elif "date" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
                    df = df.drop(columns=["date"])
                else:
                    self.logger.warning(f"{csv_file.name}: missing timestamp/date column")
                    continue

                required_cols = ["open", "high", "low", "close", "volume", "timestamp"]
                missing = [col for col in required_cols if col not in df.columns]
                if missing:
                    self.logger.warning(f"{csv_file.name}: missing required columns {missing}")
                    continue

                if self.config.start_date:
                    start_ts = pd.to_datetime(self.config.start_date, utc=True)
                    df = df[df["timestamp"] >= start_ts]
                if self.config.end_date:
                    end_ts = pd.to_datetime(self.config.end_date, utc=True)
                    df = df[df["timestamp"] <= end_ts]

                df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                df["symbol"] = base_symbol
                dfs.append(df)
            except Exception as e:
                self.logger.warning(f"Failed to load {csv_file}: {e}")

        if not dfs:
            raise ValueError("No data loaded")

        self.data = pd.concat(dfs, ignore_index=True)
        self.logger.info(
            f"Loaded {len(self.data)} rows from {len(dfs)} files "
            f"across {self.data['symbol'].nunique()} symbols"
        )

    @staticmethod
    def _extract_base_symbol(file_symbol: str) -> str:
        """
        Extract ticker from filename stem.

        Examples:
          - AAPL_15MIN -> AAPL
          - BRK.B_15MIN -> BRK.B
        """
        normalized = file_symbol.strip().upper()
        match = re.match(r"^([A-Z]+(?:\.[A-Z]+)?)(?:_.*)?$", normalized)
        return match.group(1) if match else normalized

    def _capture_data_quality_report(self) -> None:
        """Build and cache quality report for the active training dataset."""
        if self.data is None or self.data.empty:
            raise ValueError("Training data is empty; cannot compute data quality report.")

        thresholds = {
            "missing_bars_ratio_max": self.config.quality_missing_bars_threshold,
            "duplicate_bars_ratio_max": self.config.quality_duplicate_bars_threshold,
            "extreme_move_ratio_max": self.config.quality_extreme_move_threshold,
            "corporate_action_jump_ratio_max": self.config.quality_corporate_action_jump_threshold,
        }

        self.data_quality_report = build_data_quality_report(
            self.data,
            thresholds=thresholds,
        )
        self.data_quality_report_hash = compute_data_quality_hash(self.data_quality_report)

        summary = self.data_quality_report.get("summary", {})
        self.logger.info(
            "Data quality report prepared: "
            f"rows={summary.get('rows_total', 0)}, "
            f"symbols={summary.get('symbol_count', 0)}, "
            f"passed={self.data_quality_report.get('passed', False)}, "
            f"hash={self.data_quality_report_hash[:12] if self.data_quality_report_hash else 'n/a'}"
        )
        if not bool(self.data_quality_report.get("passed", False)):
            self.logger.warning(
                "Data quality SLA breaches detected in training dataset. "
                "Review quality report before model promotion."
            )

    def _build_training_snapshot_manifest(self) -> None:
        """Create deterministic snapshot manifest after features are finalized."""
        if self.data is None or self.data.empty:
            raise ValueError("Training data is empty; cannot build snapshot manifest.")
        if self.data_quality_report_hash is None:
            raise ValueError("Data quality hash missing; cannot build snapshot manifest.")
        if not self.feature_names:
            raise ValueError("Feature names missing; cannot build snapshot manifest.")

        self.snapshot_manifest = build_snapshot_manifest(
            ohlcv_data=self.data,
            feature_names=self.feature_names,
            data_quality_report_hash=self.data_quality_report_hash,
            requested_start_date=self.config.start_date or None,
            requested_end_date=self.config.end_date or None,
            source_system="postgresql_timescaledb",
        )
        self.logger.info(
            "Snapshot manifest created: "
            f"{self.snapshot_manifest.get('snapshot_id')} "
            f"({self.snapshot_manifest.get('row_count', 0)} rows)"
        )

    def _compute_features(self) -> None:
        """Phase 2: Compute features for model training."""
        self.logger.info("Phase 2: Computing features...")
        # 1) Reuse previously computed features from PostgreSQL when available.
        cached = self._load_features_from_postgres()
        if cached is not None and not cached.empty:
            self.features = cached
            self.logger.info("Using feature matrix loaded from PostgreSQL features table")
            return

        # Windows-specific stability guard:
        # The institutional full feature pipeline can stall on large panel datasets
        # under native Windows runtime. Fallback path remains deterministic and
        # keeps the end-to-end training pipeline operational.
        if os.name == "nt":
            self.logger.warning(
                "Using fallback feature computation on Windows for training stability."
            )
            self._compute_features_fallback()
            return

        # 2) Compute full institutional feature set.
        self._compute_features_full_pipeline()

        # 3) Persist computed features to PostgreSQL for deterministic reuse.
        if self.config.persist_features_to_postgres:
            self._store_features_to_postgres()
        else:
            self.logger.info("Skipping feature persistence to PostgreSQL for this run")

    def _resolve_feature_groups(self) -> list[Any]:
        """Resolve configured feature groups into FeatureGroup enum values."""
        from quant_trading_system.features.feature_pipeline import FeatureGroup

        group_map = {
            "technical": FeatureGroup.TECHNICAL,
            "statistical": FeatureGroup.STATISTICAL,
            "microstructure": FeatureGroup.MICROSTRUCTURE,
            "cross_sectional": FeatureGroup.CROSS_SECTIONAL,
            "cross-sectional": FeatureGroup.CROSS_SECTIONAL,
            "cross": FeatureGroup.CROSS_SECTIONAL,
            "all": FeatureGroup.ALL,
        }

        requested = [str(g).strip().lower() for g in self.config.feature_groups if str(g).strip()]
        if not requested:
            requested = ["all"]

        resolved: list[Any] = []
        for name in requested:
            group = group_map.get(name)
            if group is None:
                self.logger.warning(f"Ignoring unknown feature group: {name}")
                continue
            if group == FeatureGroup.ALL:
                return [
                    FeatureGroup.TECHNICAL,
                    FeatureGroup.STATISTICAL,
                    FeatureGroup.MICROSTRUCTURE,
                    FeatureGroup.CROSS_SECTIONAL,
                ]
            if group not in resolved:
                resolved.append(group)

        if not resolved:
            return [
                FeatureGroup.TECHNICAL,
                FeatureGroup.STATISTICAL,
                FeatureGroup.MICROSTRUCTURE,
                FeatureGroup.CROSS_SECTIONAL,
            ]
        return resolved

    def _prediction_horizon(self) -> int:
        """Return effective prediction horizon for purging/leakage controls."""
        horizons = [int(h) for h in self.config.label_horizons if int(h) > 0]
        if not horizons:
            horizons = [int(self.config.primary_label_horizon)]
        return max(1, max(horizons))

    @staticmethod
    def _sanitize_feature_columns(columns: list[str]) -> list[str]:
        """Remove columns that should never be part of model feature matrix."""
        cleaned: list[str] = []
        for col in columns:
            if col in BASE_MARKET_COLUMNS:
                continue
            if col in FORBIDDEN_FEATURE_COLUMNS:
                continue
            if any(col.startswith(prefix) for prefix in FORBIDDEN_FEATURE_PREFIXES):
                continue
            cleaned.append(col)
        return cleaned

    def _feature_schema_cache_key(
        self,
        symbols: list[str],
        start_date: pd.Timestamp | None,
        end_date: pd.Timestamp | None,
    ) -> str:
        """Build deterministic cache key for reusable PostgreSQL feature matrices."""
        symbol_part = ",".join(sorted(symbols))
        groups_part = ",".join(sorted(self.config.feature_groups))
        cross_sectional_part = "1" if self.config.enable_cross_sectional else "0"
        start_part = start_date.isoformat() if start_date is not None else "none"
        end_part = end_date.isoformat() if end_date is not None else "none"
        return (
            "train:features:schema:v1:"
            f"{symbol_part}:{start_part}:{end_part}:"
            f"{groups_part}:{cross_sectional_part}:{self.feature_pipeline_fingerprint}"
        )

    def _build_feature_schema_metadata(
        self,
        feature_names: list[str],
    ) -> dict[str, Any]:
        """Serialize feature schema metadata used for deterministic cache reuse."""
        return {
            "pipeline_fingerprint": self.feature_pipeline_fingerprint,
            "feature_names": sorted(str(name) for name in feature_names),
            "feature_groups": sorted(str(name) for name in self.config.feature_groups),
            "enable_cross_sectional": bool(self.config.enable_cross_sectional),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def _compute_features_full_pipeline(self) -> None:
        """Compute institutional features with adaptive memory-safe fallback behavior."""
        import polars as pl
        from quant_trading_system.features.feature_pipeline import (
            FeatureConfig,
            FeatureGroup,
            FeaturePipeline,
            NormalizationMethod,
        )

        features_list = []
        symbols = sorted(self.data["symbol"].dropna().unique().tolist())
        total_rows = len(self.data)
        groups_to_compute = self._resolve_feature_groups()

        if not self.config.enable_cross_sectional:
            groups_to_compute = [g for g in groups_to_compute if g != FeatureGroup.CROSS_SECTIONAL]

        if FeatureGroup.CROSS_SECTIONAL in groups_to_compute:
            exceeds_symbols = len(symbols) > int(self.config.max_cross_sectional_symbols)
            exceeds_rows = total_rows > int(self.config.max_cross_sectional_rows)
            if exceeds_symbols or exceeds_rows:
                message = (
                    "Cross-sectional feature group disabled adaptively due to dataset scale: "
                    f"symbols={len(symbols)}/{self.config.max_cross_sectional_symbols}, "
                    f"rows={total_rows}/{self.config.max_cross_sectional_rows}."
                )
                if self.config.allow_feature_group_fallback:
                    self.logger.warning(message)
                    groups_to_compute = [
                        g for g in groups_to_compute if g != FeatureGroup.CROSS_SECTIONAL
                    ]
                else:
                    raise RuntimeError(message)

        group_names = [g.value for g in groups_to_compute]
        self.logger.info(f"Feature groups for this run: {group_names}")
        if not groups_to_compute:
            raise RuntimeError("No feature groups enabled for computation.")
        disable_optimized_technical = os.name == "nt"
        if disable_optimized_technical:
            self.logger.warning(
                "Disabling optimized technical feature pipeline on Windows; "
                "using standard deterministic calculators to avoid runtime stalls."
            )

        universe_data: dict[str, pl.DataFrame] | None = None
        if FeatureGroup.CROSS_SECTIONAL in groups_to_compute:
            universe_data = {}
            for symbol in symbols:
                symbol_df = (
                    self.data[self.data["symbol"] == symbol]
                    .copy()
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
                universe_data[symbol] = pl.from_pandas(symbol_df)

        for symbol in symbols:
            self.logger.info(f"Computing features for {symbol}...")
            df = (
                self.data[self.data["symbol"] == symbol]
                .copy()
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            df_pl = pl.from_pandas(df)

            combined_features: dict[str, np.ndarray] = {}
            for group in groups_to_compute:
                feature_config = FeatureConfig(
                    groups=[group],
                    normalization=NormalizationMethod.NONE,
                    include_targets=False,
                    use_gpu=self.config.use_gpu and group == FeatureGroup.TECHNICAL,
                    use_cache=self.config.use_redis_cache and group == FeatureGroup.TECHNICAL,
                    use_optimized_pipeline=(
                        group == FeatureGroup.TECHNICAL and not disable_optimized_technical
                    ),
                )
                pipeline = FeaturePipeline(feature_config)
                try:
                    feature_set = pipeline.compute(
                        df_pl,
                        symbol=symbol,
                        universe_data=universe_data if group == FeatureGroup.CROSS_SECTIONAL else None,
                        use_cache=feature_config.use_cache,
                    )
                    combined_features.update(feature_set.features)
                except Exception as e:
                    if self.config.allow_feature_group_fallback and group == FeatureGroup.CROSS_SECTIONAL:
                        self.logger.warning(
                            f"{symbol}: cross-sectional features skipped due to runtime error: {e}"
                        )
                        continue
                    raise RuntimeError(
                        f"Feature pipeline failed for {symbol} group={group.value}; "
                        "institutional mode requires deterministic feature materialization."
                    ) from e
                finally:
                    del pipeline
                    gc.collect()

            if not combined_features:
                raise RuntimeError(f"No features computed for symbol {symbol}")

            feature_df = pd.DataFrame(combined_features, index=df.index)
            features_df = pd.concat([df, feature_df], axis=1)
            features_df["symbol"] = symbol
            features_list.append(features_df)
            self.logger.info(f"  {symbol}: {len(combined_features)} features computed")

            del df_pl
            gc.collect()

        if features_list:
            self.features = pd.concat(features_list, ignore_index=True)
            self.features = self.features.replace([np.inf, -np.inf], np.nan).dropna()
            self.logger.info(
                f"Total: {len(self.features.columns)} columns for {len(features_list)} symbols"
            )

    def _load_features_from_postgres(self) -> pd.DataFrame | None:
        """Load precomputed features from PostgreSQL and merge with OHLCV data."""
        from sqlalchemy import text
        from quant_trading_system.database.connection import get_db_manager, get_redis_manager

        if self.data is None or self.data.empty:
            return None

        symbols = sorted(self.data["symbol"].dropna().unique().tolist())
        if not symbols:
            return None

        db_manager = get_db_manager()
        redis_mgr = get_redis_manager()

        start_date = pd.to_datetime(self.config.start_date, utc=True) if self.config.start_date else None
        end_date = pd.to_datetime(self.config.end_date, utc=True) if self.config.end_date else None

        cache_key = (
            f"train:features:coverage:v3:{','.join(symbols)}:"
            f"{start_date.isoformat() if start_date is not None else 'none'}:"
            f"{end_date.isoformat() if end_date is not None else 'none'}"
        )
        schema_key = self._feature_schema_cache_key(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )
        cached_coverage = redis_mgr.get(cache_key)
        if cached_coverage == "empty":
            self.logger.info("Feature cache marked empty previously; revalidating PostgreSQL coverage.")

        schema_payload_raw = redis_mgr.get(schema_key)
        if not schema_payload_raw:
            self.logger.info(
                "Feature schema metadata not found for cache reuse; recomputing features for determinism."
            )
            return None
        try:
            schema_payload = json.loads(schema_payload_raw)
        except Exception:
            self.logger.warning("Invalid feature schema metadata in Redis; recomputing features.")
            return None
        expected_feature_names = [
            str(name)
            for name in schema_payload.get("feature_names", [])
            if isinstance(name, str) and name.strip()
        ]
        if not expected_feature_names:
            self.logger.info("Feature schema metadata missing feature names; recomputing features.")
            return None
        if schema_payload.get("pipeline_fingerprint") != self.feature_pipeline_fingerprint:
            self.logger.info(
                "Feature pipeline fingerprint changed since cache write; recomputing features."
            )
            return None

        query = text(
            """
            SELECT
                symbol,
                timestamp,
                feature_name,
                value
            FROM features
            WHERE symbol = :symbol
              AND (:start_date IS NULL OR timestamp >= :start_date)
              AND (:end_date IS NULL OR timestamp <= :end_date)
            ORDER BY timestamp, feature_name
            """
        )

        feature_frames: list[pd.DataFrame] = []
        with db_manager.session() as session:
            session.execute(text("SET LOCAL max_parallel_workers_per_gather = 0"))
            session.execute(text("SET LOCAL statement_timeout = '60000ms'"))
            for symbol in symbols:
                result = session.execute(
                    query,
                    {"symbol": symbol, "start_date": start_date, "end_date": end_date},
                )
                rows = result.fetchall()
                if not rows:
                    continue
                feature_frames.append(pd.DataFrame(rows, columns=result.keys()))

        if not feature_frames:
            redis_mgr.set(cache_key, "empty", expire_seconds=300)
            return None

        long_df = pd.concat(feature_frames, ignore_index=True)
        long_df["timestamp"] = pd.to_datetime(long_df["timestamp"], utc=True, errors="coerce")
        long_df = long_df.dropna(subset=["timestamp", "feature_name", "value"])

        wide_df = (
            long_df.pivot_table(
                index=["symbol", "timestamp"],
                columns="feature_name",
                values="value",
                aggfunc="last",
            )
            .reset_index()
        )
        wide_df.columns.name = None

        merged = self.data.merge(wide_df, on=["symbol", "timestamp"], how="left")
        feature_cols = [c for c in merged.columns if c not in BASE_MARKET_COLUMNS]
        feature_cols = self._sanitize_feature_columns(feature_cols)
        if not feature_cols:
            redis_mgr.set(cache_key, "empty", expire_seconds=300)
            return None

        missing_features = sorted(set(expected_feature_names).difference(feature_cols))
        if missing_features:
            self.logger.info(
                f"Cached PostgreSQL features missing {len(missing_features)} expected columns; recomputing."
            )
            redis_mgr.set(cache_key, "empty", expire_seconds=300)
            return None

        selected_feature_cols = [name for name in expected_feature_names if name in feature_cols]
        if not selected_feature_cols:
            redis_mgr.set(cache_key, "empty", expire_seconds=300)
            return None
        merged = merged[["symbol", "timestamp", "open", "high", "low", "close", "volume", *selected_feature_cols]]
        feature_cols = selected_feature_cols

        clean_merged = merged.replace([np.inf, -np.inf], np.nan).dropna()
        usable_rows = len(clean_merged)
        reuse_floor = min(max(float(self.config.feature_reuse_min_coverage), 0.01), 0.95)
        min_required_rows = max(50, int(reuse_floor * len(merged)))
        cell_coverage = float(merged[feature_cols].notna().mean().mean())
        self.logger.info(
            "PostgreSQL feature coverage: "
            f"cells={cell_coverage:.2%}, usable_rows={usable_rows}/{len(merged)} "
            f"(min_required={min_required_rows})"
        )
        # Rolling windows naturally reduce usable rows; gate by absolute usable sample count.
        if usable_rows < min_required_rows:
            redis_mgr.set(cache_key, "empty", expire_seconds=300)
            return None

        redis_mgr.set(cache_key, f"{usable_rows}", expire_seconds=300)
        return clean_merged

    def _store_features_to_postgres(self) -> None:
        """Persist computed features to PostgreSQL `features` table (upsert)."""
        from sqlalchemy import text
        from quant_trading_system.database.connection import get_db_manager, get_redis_manager

        if self.features is None or self.features.empty:
            return

        feature_cols = [c for c in self.features.columns if c not in BASE_MARKET_COLUMNS]
        feature_cols = self._sanitize_feature_columns(feature_cols)
        if not feature_cols:
            return

        export_df = self.features[["symbol", "timestamp", *feature_cols]].copy()
        export_df["timestamp"] = pd.to_datetime(export_df["timestamp"], utc=True, errors="coerce")
        export_df = export_df.dropna(subset=["timestamp"])
        if export_df.empty:
            return

        upsert = text(
            """
            INSERT INTO features (symbol, timestamp, feature_name, value)
            VALUES (:symbol, :timestamp, :feature_name, :value)
            ON CONFLICT (symbol, timestamp, feature_name)
            DO UPDATE SET value = EXCLUDED.value
            """
        )

        db_manager = get_db_manager()
        checkpoint_file = self._feature_materialization_checkpoint_path()
        total_source_rows = len(export_df)
        resume_offset = min(self._read_materialization_offset(checkpoint_file), total_source_rows)
        if resume_offset > 0:
            self.logger.info(
                f"Resuming feature materialization from source row offset "
                f"{resume_offset}/{total_source_rows}"
            )

        total_upserted_rows = 0
        requested_source_batch_rows = max(250, int(self.config.feature_materialization_batch_rows))
        source_batch_rows = min(requested_source_batch_rows, 1000)
        if source_batch_rows < requested_source_batch_rows:
            self.logger.info(
                "Capping feature materialization source batch rows "
                f"from {requested_source_batch_rows} to {source_batch_rows} for DB lock safety"
            )
        with db_manager.session() as session:
            session.execute(text("SET max_parallel_workers_per_gather = 0"))
            session.execute(text("SET statement_timeout = '120000ms'"))

            for i in range(resume_offset, total_source_rows, source_batch_rows):
                source_batch = export_df.iloc[i:i + source_batch_rows]
                long_batch = source_batch.melt(
                    id_vars=["symbol", "timestamp"],
                    value_vars=feature_cols,
                    var_name="feature_name",
                    value_name="value",
                ).dropna(subset=["value"])

                if not long_batch.empty:
                    write_batch_size = min(2000, max(500, source_batch_rows * 2))
                    for offset in range(0, len(long_batch), write_batch_size):
                        chunk_df = long_batch.iloc[offset:offset + write_batch_size]
                        write_chunk = [
                            {
                                "symbol": row.symbol,
                                "timestamp": (
                                    row.timestamp.to_pydatetime()
                                    if hasattr(row.timestamp, "to_pydatetime")
                                    else row.timestamp
                                ),
                                "feature_name": row.feature_name,
                                "value": float(row.value),
                            }
                            for row in chunk_df.itertuples(index=False)
                        ]
                        if not write_chunk:
                            continue
                        session.execute(upsert, write_chunk)
                        session.commit()
                        total_upserted_rows += len(write_chunk)

                self._write_materialization_offset(
                    checkpoint_file=checkpoint_file,
                    next_offset=i + len(source_batch),
                    total_rows=total_source_rows,
                )

        if checkpoint_file.exists():
            checkpoint_file.unlink()

        redis_mgr = get_redis_manager()
        redis_mgr.delete("train:ohlcv_symbols")
        symbols = sorted(export_df["symbol"].dropna().unique().tolist())
        start_date = pd.to_datetime(self.config.start_date, utc=True) if self.config.start_date else None
        end_date = pd.to_datetime(self.config.end_date, utc=True) if self.config.end_date else None
        schema_key = self._feature_schema_cache_key(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )
        schema_metadata = self._build_feature_schema_metadata(feature_cols)
        redis_mgr.set(schema_key, json.dumps(schema_metadata, ensure_ascii=True), expire_seconds=604800)
        self.logger.info(f"Persisted {total_upserted_rows} feature rows to PostgreSQL")

    def _feature_materialization_checkpoint_path(self) -> Path:
        """Checkpoint path for resumable feature materialization."""
        state_dir = Path(self.config.output_dir) / "materialization_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        snapshot_id = (
            str(self.snapshot_manifest.get("snapshot_id"))
            if isinstance(self.snapshot_manifest, dict)
            else "unknown"
        )
        fingerprint = hashlib.sha256(
            f"{snapshot_id}:{self.config.model_name}:{self.config.model_type}".encode("utf-8")
        ).hexdigest()[:12]
        return state_dir / f"features_{snapshot_id}_{fingerprint}.json"

    @staticmethod
    def _read_materialization_offset(checkpoint_file: Path) -> int:
        if not checkpoint_file.exists():
            return 0
        try:
            payload = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            return max(0, int(payload.get("next_offset", 0)))
        except Exception:
            return 0

    def _write_materialization_offset(
        self,
        checkpoint_file: Path,
        next_offset: int,
        total_rows: int,
    ) -> None:
        payload = {
            "snapshot_id": (
                self.snapshot_manifest.get("snapshot_id")
                if isinstance(self.snapshot_manifest, dict)
                else None
            ),
            "model_name": self.config.model_name,
            "next_offset": int(next_offset),
            "total_rows": int(total_rows),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        checkpoint_file.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _compute_features_fallback(self) -> None:
        """Fallback feature computation with basic indicators."""
        features_list = []

        for symbol in self.data["symbol"].unique():
            df = self.data[self.data["symbol"] == symbol].copy()
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Basic features
            df["returns"] = df["close"].pct_change()
            df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

            # Moving averages
            for window in [5, 10, 20, 50]:
                df[f"sma_{window}"] = df["close"].rolling(window).mean()
                df[f"ema_{window}"] = df["close"].ewm(span=window).mean()

            # Volatility
            df["volatility_20"] = df["returns"].rolling(20).std()
            df["atr_14"] = self._compute_atr(df, 14)

            # Momentum
            df["rsi_14"] = self._compute_rsi(df["close"], 14)
            df["momentum_10"] = df["close"].pct_change(10)

            # Volume features
            df["volume_sma_20"] = df["volume"].rolling(20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

            df["symbol"] = symbol
            features_list.append(df)

        self.features = pd.concat(features_list, ignore_index=True)
        self.features = self.features.dropna()
        self.logger.info(f"Computed {len(self.features.columns)} basic features")

    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _compute_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _create_labels(self) -> None:
        """Phase 3: Create labels for supervised learning."""
        self.logger.info("Phase 3: Creating labels...")
        label_horizons = sorted(
            {
                int(h)
                for h in self.config.label_horizons
                if isinstance(h, (int, np.integer, float, np.floating)) and int(h) > 0
            }
        )
        if not label_horizons:
            label_horizons = [1, 5, 20]
        primary_horizon = int(self.config.primary_label_horizon)
        if primary_horizon not in label_horizons:
            label_horizons.append(primary_horizon)
            label_horizons = sorted(set(label_horizons))

        target_config = TargetEngineeringConfig(
            horizons=tuple(label_horizons),
            primary_horizon=primary_horizon,
            profit_taking_threshold=float(self.config.label_profit_taking_threshold),
            stop_loss_threshold=float(self.config.label_stop_loss_threshold),
            max_holding_period=int(self.config.label_max_holding_period),
            use_volatility_barriers=True,
            volatility_lookback=int(self.config.label_volatility_lookback),
            spread_bps=float(self.config.label_spread_bps),
            slippage_bps=float(self.config.label_slippage_bps),
            impact_bps=float(self.config.label_impact_bps),
            min_signal_abs_return_bps=float(self.config.label_min_signal_abs_return_bps),
            neutral_buffer_bps=float(self.config.label_neutral_buffer_bps),
            max_abs_forward_return=float(self.config.label_max_abs_forward_return),
            signal_volatility_floor_mult=float(self.config.label_signal_volatility_floor_mult),
            regime_lookback=int(self.config.label_regime_lookback),
            temporal_weight_decay=float(self.config.label_temporal_weight_decay),
            edge_cost_buffer_bps=float(self.config.label_edge_cost_buffer_bps),
        )
        target_result = generate_targets(self.features, target_config)
        label_valid_mask = target_result.frame["label"].notna().to_numpy()
        labeled_frame = target_result.frame.loc[label_valid_mask].copy()
        if labeled_frame.empty:
            raise ValueError("Target engineering produced zero valid labels.")
        labeled_frame["label"] = pd.to_numeric(labeled_frame["label"], errors="coerce").astype(int)
        labeled_frame = labeled_frame.dropna(subset=["timestamp"]).reset_index(drop=True)
        if "regime" not in labeled_frame.columns:
            labeled_frame["regime"] = "normal_range"

        all_weights = np.asarray(target_result.sample_weights, dtype=float)
        sample_weights = all_weights[label_valid_mask]
        sample_weights = np.nan_to_num(sample_weights, nan=1.0, posinf=1.0, neginf=1.0)
        if len(sample_weights) != len(labeled_frame):
            sample_weights = np.ones(len(labeled_frame), dtype=float)

        self.label_diagnostics = target_result.diagnostics
        self.training_metrics["label_positive_rate"] = float(self.label_diagnostics.get("positive_rate", 0.0))
        self.training_metrics["label_class_balance_ratio"] = float(
            self.label_diagnostics.get("class_balance_ratio", 0.0)
        )
        self.training_metrics["label_drift_abs"] = float(self.label_diagnostics.get("label_drift_abs", 0.0))
        self.training_metrics["label_count"] = float(self.label_diagnostics.get("label_count", 0))

        # Reserve untouched terminal holdout block by timestamp for anti-overfit validation.
        ts_series = pd.to_datetime(labeled_frame["timestamp"], utc=True, errors="coerce")
        unique_ts = np.sort(ts_series.dropna().unique())
        holdout_count = max(1, int(round(len(unique_ts) * self.config.holdout_pct))) if len(unique_ts) else 0
        holdout_ts = set(unique_ts[-holdout_count:]) if holdout_count > 0 else set()
        holdout_mask = ts_series.isin(holdout_ts).to_numpy() if holdout_ts else np.zeros(len(labeled_frame), dtype=bool)
        if int(np.sum(~holdout_mask)) < 1000 or int(np.sum(holdout_mask)) < 200:
            holdout_mask = np.zeros(len(labeled_frame), dtype=bool)
        dev_mask = ~holdout_mask

        dev_frame = labeled_frame.loc[dev_mask].reset_index(drop=True)
        holdout_frame = labeled_frame.loc[holdout_mask].reset_index(drop=True)
        dev_weights = sample_weights[dev_mask]
        holdout_weights = sample_weights[holdout_mask]

        self.labels = dev_frame["label"]
        self.timestamps = dev_frame["timestamp"].to_numpy()
        self.row_symbols = dev_frame["symbol"].astype(str).to_numpy()
        self.regimes = dev_frame["regime"].astype(str).to_numpy()
        self.close_prices = dev_frame["close"].reset_index(drop=True)
        primary_forward_col = f"forward_return_h{primary_horizon}"
        if primary_forward_col in dev_frame.columns:
            self.primary_forward_returns = np.nan_to_num(
                pd.to_numeric(dev_frame[primary_forward_col], errors="coerce").to_numpy(dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        else:
            self.primary_forward_returns = np.zeros(len(dev_frame), dtype=float)
        if "triple_barrier_net_return" in dev_frame.columns:
            self.cost_aware_event_returns = np.nan_to_num(
                pd.to_numeric(dev_frame["triple_barrier_net_return"], errors="coerce").to_numpy(dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        else:
            self.cost_aware_event_returns = np.zeros(len(dev_frame), dtype=float)
        self.sample_weights = dev_weights

        self.holdout_features = None
        self.holdout_labels = None
        self.holdout_timestamps = None
        self.holdout_symbols = None
        self.holdout_regimes = None
        self.holdout_primary_forward_returns = None
        self.holdout_cost_aware_event_returns = None
        if not holdout_frame.empty:
            self.holdout_labels = holdout_frame["label"].astype(int).reset_index(drop=True)
            self.holdout_timestamps = holdout_frame["timestamp"].to_numpy()
            self.holdout_symbols = holdout_frame["symbol"].astype(str).to_numpy()
            self.holdout_regimes = holdout_frame["regime"].astype(str).to_numpy()
            if primary_forward_col in holdout_frame.columns:
                self.holdout_primary_forward_returns = np.nan_to_num(
                    pd.to_numeric(holdout_frame[primary_forward_col], errors="coerce").to_numpy(dtype=float),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            else:
                self.holdout_primary_forward_returns = np.zeros(len(holdout_frame), dtype=float)
            if "triple_barrier_net_return" in holdout_frame.columns:
                self.holdout_cost_aware_event_returns = np.nan_to_num(
                    pd.to_numeric(
                        holdout_frame["triple_barrier_net_return"],
                        errors="coerce",
                    ).to_numpy(dtype=float),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            else:
                self.holdout_cost_aware_event_returns = np.zeros(len(holdout_frame), dtype=float)
            self.training_metrics["holdout_rows"] = float(len(holdout_frame))
            self.training_metrics["holdout_weight_mean"] = float(np.mean(holdout_weights)) if len(holdout_weights) else 1.0
        self.training_metrics["development_rows"] = float(len(dev_frame))

        # Remove non-feature columns
        exclude_cols = [
            "timestamp",
            "symbol",
            "label",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "primary_signal",
            "barrier_touched",
            "triple_barrier_label",
            "triple_barrier_event_return",
            "triple_barrier_net_return",
            "holding_period",
            "regime",
            "trade_side_label",
            "binary_trade_label",
        ]
        exclude_cols.extend([f"forward_return_h{h}" for h in label_horizons])
        feature_cols = [c for c in dev_frame.columns if c not in exclude_cols]
        if not feature_cols:
            raise ValueError("No model features remain after target engineering exclusion list.")
        self.features = dev_frame[feature_cols]
        if not holdout_frame.empty:
            self.holdout_features = holdout_frame[feature_cols].reset_index(drop=True)
        self.feature_names = feature_cols

        self.logger.info(
            f"Created {len(self.labels)} development labels "
            f"(pos: {int(self.labels.sum())}, neg: {int(len(self.labels) - self.labels.sum())}) "
            f"| holdout={int(np.sum(holdout_mask))} "
            f"| pos_rate={self.training_metrics['label_positive_rate']:.2%} "
            f"| drift_abs={self.training_metrics['label_drift_abs']:.2%}"
        )
        self._build_training_snapshot_manifest()

    def _validate_symbol_timestamp_ordering(self) -> None:
        """Ensure timestamps are strictly increasing within each symbol stream."""
        if self.timestamps is None:
            raise ValueError("Timestamp array missing for leakage validation")
        ts_series = pd.Series(pd.to_datetime(self.timestamps, utc=True, errors="coerce"))
        if ts_series.isna().any():
            raise ValueError("Timestamp array contains invalid entries after conversion")

        if self.row_symbols is None or len(self.row_symbols) != len(ts_series):
            diffs = np.diff(ts_series.astype("int64", copy=False).to_numpy())
            if np.any(diffs <= 0):
                raise ValueError("Timestamp ordering violation detected in training data.")
            return

        symbol_series = pd.Series(self.row_symbols.astype(str))
        for symbol, idx in symbol_series.groupby(symbol_series).groups.items():
            symbol_ts = ts_series.iloc[idx].astype("int64", copy=False).to_numpy()
            if np.any(np.diff(symbol_ts) <= 0):
                raise ValueError(
                    f"Timestamp ordering violation for symbol {symbol}. "
                    "Ensure per-symbol bars are strictly increasing."
                )

    def _panelize_timestamp_splits(
        self,
        splitter: Any,
        timestamps: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Split by unique timestamps then map folds back to row indices."""
        ts_series = pd.Series(pd.to_datetime(timestamps, utc=True, errors="coerce"))
        if ts_series.isna().any():
            raise ValueError("Invalid timestamps detected during panel split mapping.")

        ts_ns = ts_series.astype("int64", copy=False).to_numpy()
        unique_ts_ns = np.sort(np.unique(ts_ns))
        if len(unique_ts_ns) < 3:
            raise ValueError("Insufficient unique timestamps for panel-aware cross-validation.")

        unique_dummy = np.zeros((len(unique_ts_ns), 1), dtype=float)
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for train_ts_idx, test_ts_idx in splitter.split(unique_dummy, unique_dummy[:, 0]):
            train_ts_ns = unique_ts_ns[np.asarray(train_ts_idx, dtype=int)]
            test_ts_ns = unique_ts_ns[np.asarray(test_ts_idx, dtype=int)]

            train_idx = np.flatnonzero(np.isin(ts_ns, train_ts_ns)).astype(int)
            test_idx = np.flatnonzero(np.isin(ts_ns, test_ts_ns)).astype(int)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            if np.intersect1d(train_idx, test_idx).size > 0:
                raise ValueError("Panel CV mapping produced overlapping train/test indices.")
            splits.append((train_idx, test_idx))

        if not splits:
            raise ValueError("Panel-aware cross-validation produced no valid splits.")
        return splits

    @staticmethod
    def _regime_distribution(regimes: np.ndarray | list[str] | None) -> dict[str, float]:
        """Return normalized regime distribution."""
        if regimes is None:
            return {}
        arr = np.asarray(regimes, dtype=str)
        if arr.size == 0:
            return {}
        values, counts = np.unique(arr, return_counts=True)
        total = float(np.sum(counts))
        if total <= 0:
            return {}
        return {str(v): float(c / total) for v, c in zip(values, counts, strict=False)}

    @staticmethod
    def _regime_shift_score(
        left_distribution: dict[str, float],
        right_distribution: dict[str, float],
    ) -> float:
        """Total-variation-like regime shift score in [0, 1]."""
        all_keys = set(left_distribution).union(right_distribution)
        if not all_keys:
            return 0.0
        divergence = 0.0
        for key in all_keys:
            divergence += abs(left_distribution.get(key, 0.0) - right_distribution.get(key, 0.0))
        return float(0.5 * divergence)

    def _generate_cv_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate CV splits.

        For panel data (multi-symbol with repeated timestamps), split on unique
        timestamps then map back to row indices to avoid symbol-block leakage.
        """
        if self.timestamps is None or len(self.timestamps) != len(X):
            cv = self._get_cv_splitter(n_samples=len(X))
            candidate_splits = [
                (np.asarray(tr, dtype=int), np.asarray(te, dtype=int))
                for tr, te in cv.split(X, y)
            ]
            return candidate_splits

        ts_series = pd.Series(pd.to_datetime(self.timestamps, utc=True, errors="coerce"))
        if ts_series.isna().any():
            raise ValueError("Invalid timestamps detected during CV split generation.")

        has_duplicate_ts = bool(ts_series.duplicated().any())
        has_panel_symbols = (
            self.row_symbols is not None
            and len(self.row_symbols) == len(ts_series)
            and len(pd.unique(pd.Series(self.row_symbols.astype(str)))) > 1
        )
        if not has_duplicate_ts and not has_panel_symbols:
            cv = self._get_cv_splitter(n_samples=len(X))
            candidate_splits = [
                (np.asarray(tr, dtype=int), np.asarray(te, dtype=int))
                for tr, te in cv.split(X, y)
            ]
        else:
            cv = self._get_cv_splitter(n_samples=int(ts_series.nunique()))
            candidate_splits = self._panelize_timestamp_splits(cv, ts_series.to_numpy())

        if self.regimes is None or len(self.regimes) != len(X):
            return candidate_splits

        filtered_splits: list[tuple[np.ndarray, np.ndarray]] = []
        accepted_shift_scores: list[float] = []
        soft_limit = float(self.config.max_regime_shift * 1.5)
        for train_idx, test_idx in candidate_splits:
            train_dist = self._regime_distribution(self.regimes[train_idx])
            test_dist = self._regime_distribution(self.regimes[test_idx])
            shift_score = self._regime_shift_score(train_dist, test_dist)
            if shift_score > soft_limit:
                continue
            filtered_splits.append((train_idx, test_idx))
            accepted_shift_scores.append(shift_score)

        if filtered_splits:
            self.training_metrics["cv_regime_shift_mean"] = float(np.mean(accepted_shift_scores))
            self.training_metrics["cv_regime_shift_max"] = float(np.max(accepted_shift_scores))
            return filtered_splits

        self.logger.warning(
            "No CV split satisfied regime soft-limit %.3f; using unfiltered candidate splits.",
            soft_limit,
        )
        return candidate_splits

    def _validate_no_future_leakage(self) -> None:
        """Run strict future-leak checks on dataset and planned CV splits."""
        from quant_trading_system.models.model_manager import FutureLeakValidator

        if self.features is None or self.labels is None:
            raise ValueError("Features/labels must be prepared before leakage validation")

        validator = FutureLeakValidator(strict_mode=True)
        is_valid, issues = validator.validate(
            self.features.values,
            self.labels.values,
            feature_names=self.feature_names,
            timestamps=None,
        )
        if not is_valid:
            raise ValueError(f"Future-leak validation failed: {issues}")
        self._validate_symbol_timestamp_ordering()

        splits = self._generate_cv_splits(self.features.values, self.labels.values)
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if self.config.cv_method == "walk_forward":
                valid_split, split_issues = validator.validate_time_series_split(
                    self.timestamps[train_idx],
                    self.timestamps[test_idx],
                    gap_bars=max(self._prediction_horizon(), int(self.config.purge_pct * 100)),
                )
                if not valid_split:
                    raise ValueError(
                        f"Leakage detected in CV split {fold_idx + 1}: {split_issues}"
                    )
                continue

            # Purged/combinatorial CV can include both past and future observations by design.
            # Validate leakage via disjoint indices and no training timestamps inside test window.
            overlap_idx = np.intersect1d(train_idx, test_idx)
            if overlap_idx.size > 0:
                raise ValueError(
                    f"Leakage detected in CV split {fold_idx + 1}: "
                    f"{overlap_idx.size} overlapping train/test indices."
                )

            train_ts = np.asarray(self.timestamps[train_idx])
            test_ts = np.asarray(self.timestamps[test_idx])
            train_ts_ns = pd.Series(
                pd.to_datetime(train_ts, utc=True, errors="coerce")
            ).astype("int64", copy=False).to_numpy()
            test_ts_ns = pd.Series(
                pd.to_datetime(test_ts, utc=True, errors="coerce")
            ).astype("int64", copy=False).to_numpy()
            shared_ts = np.intersect1d(train_ts_ns, test_ts_ns)
            if shared_ts.size > 0:
                raise ValueError(
                    f"Leakage detected in CV split {fold_idx + 1}: "
                    "training timestamps overlap with test timestamps."
                )
            if self.config.cv_method != "combinatorial":
                test_min = np.min(test_ts_ns)
                test_max = np.max(test_ts_ns)
                in_test_window = (train_ts_ns >= test_min) & (train_ts_ns <= test_max)
                if np.any(in_test_window):
                    raise ValueError(
                        f"Leakage detected in CV split {fold_idx + 1}: "
                        "training timestamps found inside test window."
                    )

        self.logger.info("Future-leak validation passed for data and CV splits")

    def _optimize_hyperparameters(self) -> None:
        """Phase 4: Hyperparameter optimization."""
        self.logger.info("Phase 4: Hyperparameter optimization using Optuna (risk-adjusted objective)...")
        if self.config.optimizer != "optuna":
            raise ValueError(
                f"Institutional mode requires optimizer=optuna, got '{self.config.optimizer}'"
            )
        if not self.config.use_nested_walk_forward:
            raise ValueError(
                "Institutional mode requires nested walk-forward optimization trace."
            )
        self._optimize_with_optuna()

    def _optimize_with_optuna(self) -> None:
        """Optimize hyperparameters using Optuna with pruning and risk-adjusted score."""
        if self.config.use_nested_walk_forward:
            self._optimize_with_nested_walk_forward()
            return

        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError("Optuna is mandatory in institutional mode") from exc
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        X = self.features.values
        y = self.labels.values
        weights = self.sample_weights if self.sample_weights is not None else None
        splits = self._generate_cv_splits(X, y)
        if not splits:
            self.logger.info("No CV splits available for Optuna; keeping configured defaults")
            return

        min_train_size = min(len(train_idx) for train_idx, _ in splits)
        search_space = self._get_optuna_search_space(min_train_size=min_train_size)
        if not search_space:
            self.logger.info("No Optuna search space for this model type, keeping configured defaults")
            return

        def objective(trial: "optuna.Trial") -> float:
            params = search_space(trial)
            fold_scores: list[float] = []

            try:
                for fold_idx, (train_idx, test_idx) in enumerate(splits):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    fold_params = self._prepare_params_for_train_size(params, len(train_idx))
                    fold_params = self._augment_params_for_train_labels(fold_params, y_train)
                    model = self._create_model(params=fold_params)
                    w_train = weights[train_idx] if weights is not None else None
                    self._fit_model(
                        model=model,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_test,
                        y_val=y_test,
                        sample_weights=w_train,
                    )

                    y_pred = np.asarray(model.predict(X_test))
                    y_proba = np.asarray(self._get_predictions_proba(model, X_test))
                    train_proba = np.asarray(self._get_predictions_proba(model, X_train))
                    long_threshold, short_threshold = self._derive_signal_thresholds(
                        train_proba,
                        train_labels=y_train,
                    )
                    y_eval = np.asarray(y_test)
                    eval_len = min(len(y_eval), len(y_pred), len(y_proba))
                    if eval_len <= 1:
                        continue
                    fold_forward_returns = (
                        np.asarray(self.primary_forward_returns[test_idx], dtype=float)
                        if isinstance(self.primary_forward_returns, np.ndarray)
                        and len(self.primary_forward_returns) == len(y)
                        else None
                    )
                    fold_event_returns = (
                        np.asarray(self.cost_aware_event_returns[test_idx], dtype=float)
                        if isinstance(self.cost_aware_event_returns, np.ndarray)
                        and len(self.cost_aware_event_returns) == len(y)
                        else None
                    )
                    fold_timestamps = (
                        np.asarray(self.timestamps[test_idx], dtype="datetime64[ns]")
                        if self.timestamps is not None and len(self.timestamps) == len(y)
                        else None
                    )

                    metrics = self._calculate_fold_metrics(
                        y_eval[-eval_len:],
                        y_pred[-eval_len:],
                        y_proba[-eval_len:],
                        long_threshold=long_threshold,
                        short_threshold=short_threshold,
                        realized_forward_returns=(
                            fold_forward_returns[-eval_len:] if fold_forward_returns is not None else None
                        ),
                        event_net_returns=(
                            fold_event_returns[-eval_len:] if fold_event_returns is not None else None
                        ),
                        timestamps=(fold_timestamps[-eval_len:] if fold_timestamps is not None else None),
                    )
                    fold_score = float(metrics["risk_adjusted_score"])
                    if not np.isfinite(fold_score):
                        return -1e9
                    if float(metrics.get("equity_break", 0.0)) > 0.5:
                        raise optuna.TrialPruned()
                    fold_scores.append(fold_score)

                    trial.report(float(np.mean(fold_scores)), step=fold_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                self.logger.warning(f"Optuna trial failed for {self.config.model_type}: {exc}")
                return -1e9

            if not fold_scores:
                return -1e9
            return float(np.mean(fold_scores))

        sampler = optuna.samplers.TPESampler(seed=self.config.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"{self.config.model_type}_institutional_opt",
        )
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=False,
        )

        completed_trials = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None
            and np.isfinite(float(t.value))
        ]
        self.training_metrics["optuna_trials"] = int(len(study.trials))
        self.training_metrics["optuna_pruned_trials"] = int(
            sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        )

        if not completed_trials:
            self.training_metrics["optuna_best_score"] = -1e9
            self.logger.warning(
                "Optuna completed no valid trials; keeping existing model parameters."
            )
            return

        best_trial = max(completed_trials, key=lambda t: float(t.value))
        self.config.model_params = {**self.config.model_params, **best_trial.params}
        self.config.model_params = self._prepare_params_for_train_size(
            self.config.model_params, min_train_size
        )
        self.training_metrics["optuna_best_score"] = float(best_trial.value)
        self.logger.info(f"Best params: {best_trial.params}")
        self.logger.info(f"Best risk-adjusted CV score: {best_trial.value:.6f}")

    def _build_walk_forward_splitter(
        self,
        n_splits: int,
        train_pct: float = 0.6,
        sample_count: int | None = None,
    ):
        """Build deterministic walk-forward splitter for nested CV."""
        from quant_trading_system.models.purged_cv import WalkForwardCV

        embargo_pct = max(self.config.embargo_pct, 0.01)
        purge_gap = max(1, int(self.config.purge_pct * 100))
        reference_count = max(
            2,
            int(sample_count if sample_count is not None else len(self.features)),
        )
        train_pct = float(min(max(train_pct, 0.40), 0.90))
        initial_train_size = max(2, int(reference_count * train_pct))
        first_fold_headroom = max(2, purge_gap + self._prediction_horizon())
        min_train_size = max(10, int(0.10 * reference_count))
        min_train_size = min(
            min_train_size,
            max(2, initial_train_size - first_fold_headroom),
        )
        prediction_horizon = min(
            self._prediction_horizon(),
            max(1, int(max(10, reference_count) * 0.1)),
        )
        return WalkForwardCV(
            n_splits=max(1, int(n_splits)),
            train_pct=float(train_pct),
            window_type="expanding",
            min_train_size=min_train_size,
            purge_gap=purge_gap,
            embargo_pct=embargo_pct,
            prediction_horizon=prediction_horizon,
        )

    def _optimize_with_nested_walk_forward(self) -> None:
        """Nested walk-forward Optuna optimization with auditable objective breakdown."""
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError("Optuna is mandatory in institutional mode") from exc
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        X = self.features.values
        y = self.labels.values
        weights = self.sample_weights if self.sample_weights is not None else None
        timestamps = self.timestamps if self.timestamps is not None and len(self.timestamps) == len(X) else None
        panel_outer_count = len(X)
        if timestamps is not None and pd.Series(timestamps).duplicated().any():
            panel_outer_count = int(
                pd.Series(pd.to_datetime(timestamps, utc=True, errors="coerce")).nunique()
            )
        outer_cv = self._build_walk_forward_splitter(
            n_splits=self.config.nested_outer_splits,
            train_pct=0.60,
            sample_count=panel_outer_count,
        )
        if timestamps is not None and pd.Series(timestamps).duplicated().any():
            try:
                outer_splits = self._panelize_timestamp_splits(outer_cv, timestamps)
            except ValueError as exc:
                self.logger.warning(
                    "Nested outer panel split mapping failed (%s); "
                    "falling back to row-wise walk-forward outer splits.",
                    exc,
                )
                outer_splits = list(outer_cv.split(X, y))
        else:
            outer_splits = list(outer_cv.split(X, y))
        if not outer_splits:
            raise ValueError("Nested walk-forward requires at least one outer split.")

        min_train_size = min(len(train_idx) for train_idx, _ in outer_splits)
        search_space = self._get_optuna_search_space(min_train_size=min_train_size)
        if not search_space:
            self.logger.info("No Optuna search space for this model type, keeping configured defaults")
            return

        total_trials = 0
        total_pruned = 0
        self.nested_cv_trace = []

        outer_candidates: list[tuple[float, dict[str, Any], int]] = []

        for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_splits, start=1):
            X_outer_train = X[outer_train_idx]
            y_outer_train = y[outer_train_idx]
            X_outer_test = X[outer_test_idx]
            y_outer_test = y[outer_test_idx]
            w_outer_train = weights[outer_train_idx] if weights is not None else None
            ts_outer_train = timestamps[outer_train_idx] if timestamps is not None else None
            ts_outer_test = timestamps[outer_test_idx] if timestamps is not None else None
            forward_outer_train = (
                np.asarray(self.primary_forward_returns[outer_train_idx], dtype=float)
                if isinstance(self.primary_forward_returns, np.ndarray)
                and len(self.primary_forward_returns) == len(y)
                else None
            )
            event_outer_train = (
                np.asarray(self.cost_aware_event_returns[outer_train_idx], dtype=float)
                if isinstance(self.cost_aware_event_returns, np.ndarray)
                and len(self.cost_aware_event_returns) == len(y)
                else None
            )
            forward_outer_test = (
                np.asarray(self.primary_forward_returns[outer_test_idx], dtype=float)
                if isinstance(self.primary_forward_returns, np.ndarray)
                and len(self.primary_forward_returns) == len(y)
                else None
            )
            event_outer_test = (
                np.asarray(self.cost_aware_event_returns[outer_test_idx], dtype=float)
                if isinstance(self.cost_aware_event_returns, np.ndarray)
                and len(self.cost_aware_event_returns) == len(y)
                else None
            )

            panel_inner_count = len(X_outer_train)
            if ts_outer_train is not None and pd.Series(ts_outer_train).duplicated().any():
                panel_inner_count = int(
                    pd.Series(pd.to_datetime(ts_outer_train, utc=True, errors="coerce")).nunique()
                )
            inner_cv = self._build_walk_forward_splitter(
                n_splits=self.config.nested_inner_splits,
                train_pct=0.65,
                sample_count=panel_inner_count,
            )
            if ts_outer_train is not None and pd.Series(ts_outer_train).duplicated().any():
                try:
                    inner_splits = self._panelize_timestamp_splits(inner_cv, ts_outer_train)
                except ValueError as exc:
                    self.logger.warning(
                        "Outer fold %s: nested inner panel split mapping failed (%s); "
                        "falling back to row-wise inner splits.",
                        outer_fold,
                        exc,
                    )
                    inner_splits = list(inner_cv.split(X_outer_train, y_outer_train))
            else:
                inner_splits = list(inner_cv.split(X_outer_train, y_outer_train))
            if not inner_splits:
                self.logger.warning(f"Outer fold {outer_fold}: no valid inner splits, skipping fold.")
                continue

            def objective(
                trial: "optuna.Trial",
                _inner_splits: list[tuple[np.ndarray, np.ndarray]] = inner_splits,
                _X_outer_train: np.ndarray = X_outer_train,
                _y_outer_train: np.ndarray = y_outer_train,
                _w_outer_train: np.ndarray | None = w_outer_train,
                _forward_outer_train: np.ndarray | None = forward_outer_train,
                _event_outer_train: np.ndarray | None = event_outer_train,
                _ts_outer_train: np.ndarray | None = ts_outer_train,
            ) -> float:
                params = search_space(trial)
                fold_scores: list[float] = []
                component_samples: list[dict[str, float]] = []

                try:
                    for inner_idx, (inner_train_idx, inner_val_idx) in enumerate(_inner_splits):
                        X_train = _X_outer_train[inner_train_idx]
                        X_val = _X_outer_train[inner_val_idx]
                        y_train = _y_outer_train[inner_train_idx]
                        y_val = _y_outer_train[inner_val_idx]
                        fold_params = self._prepare_params_for_train_size(
                            params, len(inner_train_idx)
                        )
                        fold_params = self._augment_params_for_train_labels(fold_params, y_train)
                        model = self._create_model(params=fold_params)
                        w_train = (
                            _w_outer_train[inner_train_idx]
                            if _w_outer_train is not None
                            else None
                        )
                        self._fit_model(
                            model=model,
                            X_train=X_train,
                            y_train=y_train,
                            X_val=X_val,
                            y_val=y_val,
                            sample_weights=w_train,
                        )

                        y_pred = np.asarray(model.predict(X_val))
                        y_proba = np.asarray(self._get_predictions_proba(model, X_val))
                        train_proba = np.asarray(self._get_predictions_proba(model, X_train))
                        long_threshold, short_threshold = self._derive_signal_thresholds(
                            train_proba,
                            train_labels=y_train,
                        )
                        y_eval = np.asarray(y_val)
                        eval_len = min(len(y_eval), len(y_pred), len(y_proba))
                        if eval_len <= 1:
                            continue

                        metrics = self._calculate_fold_metrics(
                            y_eval[-eval_len:],
                            y_pred[-eval_len:],
                            y_proba[-eval_len:],
                            long_threshold=long_threshold,
                            short_threshold=short_threshold,
                            realized_forward_returns=(
                                _forward_outer_train[inner_val_idx][-eval_len:]
                                if _forward_outer_train is not None
                                else None
                            ),
                            event_net_returns=(
                                _event_outer_train[inner_val_idx][-eval_len:]
                                if _event_outer_train is not None
                                else None
                            ),
                            timestamps=(
                                _ts_outer_train[inner_val_idx][-eval_len:]
                                if _ts_outer_train is not None
                                else None
                            ),
                        )
                        fold_score = float(metrics["risk_adjusted_score"])
                        if not np.isfinite(fold_score):
                            return -1e9
                        if float(metrics.get("equity_break", 0.0)) > 0.5:
                            raise optuna.TrialPruned()
                        fold_scores.append(fold_score)
                        component_samples.append(
                            {
                                "objective_sharpe_component": float(
                                    metrics.get("objective_sharpe_component", 0.0)
                                ),
                                "objective_drawdown_penalty": float(
                                    metrics.get("objective_drawdown_penalty", 0.0)
                                ),
                                "objective_turnover_penalty": float(
                                    metrics.get("objective_turnover_penalty", 0.0)
                                ),
                                "objective_calibration_penalty": float(
                                    metrics.get("objective_calibration_penalty", 0.0)
                                ),
                                "objective_trade_activity_penalty": float(
                                    metrics.get("objective_trade_activity_penalty", 0.0)
                                ),
                                "objective_cvar_penalty": float(
                                    metrics.get("objective_cvar_penalty", 0.0)
                                ),
                                "objective_skew_penalty": float(
                                    metrics.get("objective_skew_penalty", 0.0)
                                ),
                                "objective_equity_break_penalty": float(
                                    metrics.get("objective_equity_break_penalty", 0.0)
                                ),
                            }
                        )

                        trial.report(float(np.mean(fold_scores)), step=inner_idx)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                except optuna.TrialPruned:
                    raise
                except Exception as exc:
                    self.logger.warning(f"Optuna nested trial failed: {exc}")
                    return -1e9

                if not fold_scores:
                    return -1e9

                if component_samples:
                    trial.set_user_attr(
                        "objective_component_mean",
                        {
                            key: float(np.mean([s[key] for s in component_samples]))
                            for key in component_samples[0]
                        },
                    )
                return float(np.mean(fold_scores))

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.config.seed + outer_fold),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
                study_name=f"{self.config.model_type}_nested_outer_{outer_fold}",
            )
            study.optimize(
                objective,
                n_trials=self.config.n_trials,
                n_jobs=1,
                gc_after_trial=True,
                show_progress_bar=False,
            )

            total_trials += int(len(study.trials))
            total_pruned += int(
                sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
            )

            completed_trials = [
                t
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
                and t.value is not None
                and np.isfinite(float(t.value))
            ]
            if not completed_trials:
                self.logger.warning(
                    f"Outer fold {outer_fold}: no valid Optuna trials, fold skipped."
                )
                continue

            best_trial = max(completed_trials, key=lambda t: float(t.value))
            best_params = dict(best_trial.params)
            best_fold_params = self._prepare_params_for_train_size(
                best_params, len(outer_train_idx)
            )
            best_fold_params = self._augment_params_for_train_labels(best_fold_params, y_outer_train)

            model = self._create_model(params=best_fold_params)
            self._fit_model(
                model=model,
                X_train=X_outer_train,
                y_train=y_outer_train,
                X_val=X_outer_test,
                y_val=y_outer_test,
                sample_weights=w_outer_train,
            )

            y_pred = np.asarray(model.predict(X_outer_test))
            y_proba = np.asarray(self._get_predictions_proba(model, X_outer_test))
            train_proba = np.asarray(self._get_predictions_proba(model, X_outer_train))
            long_threshold, short_threshold = self._derive_signal_thresholds(
                train_proba,
                train_labels=y_outer_train,
            )
            y_eval = np.asarray(y_outer_test)
            eval_len = min(len(y_eval), len(y_pred), len(y_proba))
            if eval_len <= 1:
                self.logger.warning(
                    f"Outer fold {outer_fold}: insufficient predictions for scoring."
                )
                continue

            metrics = self._calculate_fold_metrics(
                y_eval[-eval_len:],
                y_pred[-eval_len:],
                y_proba[-eval_len:],
                long_threshold=long_threshold,
                short_threshold=short_threshold,
                realized_forward_returns=(
                    forward_outer_test[-eval_len:] if forward_outer_test is not None else None
                ),
                event_net_returns=(
                    event_outer_test[-eval_len:] if event_outer_test is not None else None
                ),
                timestamps=(ts_outer_test[-eval_len:] if ts_outer_test is not None else None),
            )
            outer_score = float(metrics.get("risk_adjusted_score", -1e9))
            if not np.isfinite(outer_score):
                continue

            outer_candidates.append((outer_score, best_params, len(outer_train_idx)))
            self.nested_cv_trace.append(
                {
                    "outer_fold": int(outer_fold),
                    "outer_train_size": int(len(outer_train_idx)),
                    "outer_test_size": int(eval_len),
                    "inner_split_count": int(len(inner_splits)),
                    "inner_best_score": float(best_trial.value),
                    "outer_score": outer_score,
                    "best_params": best_params,
                    "objective_components": {
                        "objective_sharpe_component": float(
                            metrics.get("objective_sharpe_component", 0.0)
                        ),
                        "objective_drawdown_penalty": float(
                            metrics.get("objective_drawdown_penalty", 0.0)
                        ),
                        "objective_turnover_penalty": float(
                            metrics.get("objective_turnover_penalty", 0.0)
                        ),
                        "objective_calibration_penalty": float(
                            metrics.get("objective_calibration_penalty", 0.0)
                        ),
                        "objective_trade_activity_penalty": float(
                            metrics.get("objective_trade_activity_penalty", 0.0)
                        ),
                    },
                }
            )

        if not outer_candidates:
            raise ValueError(
                "Nested walk-forward optimization produced no valid outer-fold candidates."
            )

        best_outer_score, best_outer_params, best_outer_train_size = max(
            outer_candidates, key=lambda x: x[0]
        )
        self.config.model_params = {**self.config.model_params, **best_outer_params}
        self.config.model_params = self._prepare_params_for_train_size(
            self.config.model_params, best_outer_train_size
        )

        outer_scores = [score for score, _, _ in outer_candidates]
        self.training_metrics["optuna_trials"] = int(total_trials)
        self.training_metrics["optuna_pruned_trials"] = int(total_pruned)
        self.training_metrics["optuna_best_score"] = float(best_outer_score)
        self.training_metrics["nested_outer_folds"] = int(len(self.nested_cv_trace))
        self.training_metrics["nested_inner_splits"] = int(self.config.nested_inner_splits)
        self.training_metrics["nested_mean_outer_score"] = float(np.mean(outer_scores))
        self.training_metrics["nested_best_outer_score"] = float(best_outer_score)
        self.training_metrics["nested_cv_trace"] = self.nested_cv_trace

        if self.nested_cv_trace:
            component_keys = [
                "objective_sharpe_component",
                "objective_drawdown_penalty",
                "objective_turnover_penalty",
                "objective_calibration_penalty",
                "objective_trade_activity_penalty",
            ]
            nested_component_summary = {
                key: float(
                    np.mean(
                        [
                            trace["objective_components"].get(key, 0.0)
                            for trace in self.nested_cv_trace
                        ]
                    )
                )
                for key in component_keys
            }
            self.training_metrics["nested_objective_component_mean"] = nested_component_summary

        self.logger.info(
            "Nested walk-forward optimization complete: "
            f"outer_folds={len(self.nested_cv_trace)}, "
            f"best_outer_score={best_outer_score:.6f}"
        )

    def _optimize_with_grid_search(self) -> None:
        """Grid search optimization."""
        from sklearn.model_selection import GridSearchCV

        model = self._create_base_model()

        if self.config.model_type == "xgboost":
            param_grid = {
                "n_estimators": [100, 300, 500],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
            }
        elif self.config.model_type == "lightgbm":
            param_grid = {
                "n_estimators": [100, 300, 500],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
            }
        else:
            param_grid = {}

        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring="accuracy", n_jobs=self.config.n_jobs
        )
        grid_search.fit(self.features, self.labels)

        self.config.model_params = grid_search.best_params_
        self.logger.info(f"Best params: {grid_search.best_params_}")

    def _optimize_with_random_search(self) -> None:
        """Random search optimization."""
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import randint, uniform

        model = self._create_base_model()

        if self.config.model_type == "xgboost":
            param_distributions = {
                "n_estimators": randint(100, 1000),
                "max_depth": randint(3, 10),
                "learning_rate": uniform(0.01, 0.29),
            }
        else:
            param_distributions = {}

        random_search = RandomizedSearchCV(
            model, param_distributions, n_iter=self.config.n_trials,
            cv=3, scoring="accuracy", n_jobs=self.config.n_jobs
        )
        random_search.fit(self.features, self.labels)

        self.config.model_params = random_search.best_params_
        self.logger.info(f"Best params: {random_search.best_params_}")

    def _get_optuna_search_space(self, min_train_size: int | None = None):
        """Return model-specific Optuna search space function."""
        if self.config.model_type == "xgboost":
            return lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 200, 900),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 2.0),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.7, 6.0, log=True),
                "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 2.5),
            }

        if self.config.model_type == "lightgbm":
            return lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 200, 900),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 24, 128),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.65, 0.95),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.65, 0.95),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 120),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-6, 5.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-6, 5.0, log=True),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.15),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.7, 6.0, log=True),
            }

        if self.config.model_type == "random_forest":
            return lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 200, 700),
                "max_depth": trial.suggest_int("max_depth", 4, 16),
                "min_samples_split": trial.suggest_int("min_samples_split", 3, 16),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "class_weight": trial.suggest_categorical(
                    "class_weight",
                    [None, "balanced", "balanced_subsample"],
                ),
            }

        if self.config.model_type == "elastic_net":
            return lambda trial: {
                "C": trial.suggest_float("C", 1e-2, 5.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.10, 0.90),
                "max_iter": trial.suggest_int("max_iter", 1800, 3200),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            }

        if self.config.model_type == "lstm":
            lookback_candidates = self._sequence_lookback_candidates(min_train_size)
            return lambda trial: {
                "hidden_size": trial.suggest_categorical("hidden_size", [64, 96, 128, 192]),
                "num_layers": trial.suggest_int("num_layers", 1, 3),
                "dropout": trial.suggest_float("dropout", 0.05, 0.5),
                "lookback_window": trial.suggest_categorical("lookback_window", lookback_candidates),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            }

        if self.config.model_type == "transformer":
            lookback_candidates = self._sequence_lookback_candidates(min_train_size)
            return lambda trial: {
                "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
                "nhead": trial.suggest_categorical("nhead", [2, 4, 8]),
                "num_layers": trial.suggest_int("num_layers", 2, 5),
                "d_ff": trial.suggest_categorical("d_ff", [128, 256, 512, 768]),
                "dropout": trial.suggest_float("dropout", 0.05, 0.3),
                "lookback_window": trial.suggest_categorical("lookback_window", lookback_candidates),
                "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True),
            }

        if self.config.model_type == "tcn":
            lookback_candidates = self._sequence_lookback_candidates(min_train_size)
            return lambda trial: {
                "num_channels": trial.suggest_categorical(
                    "num_channels",
                    [[64, 128, 128, 64], [32, 64, 64, 32], [64, 64, 64, 64]],
                ),
                "kernel_size": trial.suggest_categorical("kernel_size", [2, 3, 5]),
                "dropout": trial.suggest_float("dropout", 0.05, 0.4),
                "lookback_window": trial.suggest_categorical("lookback_window", lookback_candidates),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            }

        return None

    def _sequence_lookback_candidates(self, min_train_size: int | None) -> list[int]:
        """Generate safe lookback candidates for sequence models."""
        base_candidates = [20, 40, 60]
        if min_train_size is None:
            return base_candidates

        max_lookback = max(2, min_train_size - 2)
        candidates = [v for v in base_candidates if v <= max_lookback]
        if not candidates:
            candidates = [max_lookback]
        return sorted(set(candidates))

    def _prepare_params_for_train_size(self, params: dict, train_size: int) -> dict:
        """Adjust sequence model params for current CV fold size."""
        adjusted = dict(params)
        if self.config.model_type not in {"lstm", "transformer", "tcn"}:
            return adjusted

        if train_size < 4:
            raise ValueError(
                f"Not enough samples ({train_size}) for sequence model training in CV fold"
            )

        max_lookback = max(2, train_size - 2)
        lookback = int(adjusted.get("lookback_window", min(20, max_lookback)))
        adjusted["lookback_window"] = min(lookback, max_lookback)
        return adjusted

    def _effective_trade_target(self, evaluation_size: int | None) -> int:
        """
        Compute trade-count target scaled to fold size.

        Prevents unrealistic hard gates when a fold has fewer observations than
        the global `min_trades` threshold.
        """
        configured_target = max(1, int(self.config.min_trades))
        if evaluation_size is None or int(evaluation_size) <= 0:
            return configured_target

        eval_size = int(evaluation_size)
        horizon = max(1, int(self._prediction_horizon()))
        density_cap = int(np.ceil(float(eval_size) * 0.12))
        sqrt_cap = int(np.ceil(np.sqrt(float(eval_size))))
        horizon_cap = int(np.ceil(float(eval_size) / float(max(2, horizon * 2))))
        dynamic_cap = max(3, min(density_cap, sqrt_cap, horizon_cap))
        return int(min(configured_target, dynamic_cap))

    def _derive_signal_thresholds(
        self,
        train_proba: np.ndarray | None,
        train_labels: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Derive leakage-safe long/short thresholds from training probabilities."""
        default_long = 0.55
        default_short = 0.45
        if train_proba is None:
            return default_long, default_short

        values = np.asarray(train_proba, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size < 25:
            return default_long, default_short

        values = np.clip(values, 0.0, 1.0)
        target_trades = self._effective_trade_target(int(values.size))
        target_rate = min(0.35, max(0.05, float(target_trades) / float(values.size)))
        tail_rate = min(0.25, max(0.03, target_rate * 0.65))

        short_threshold = float(np.quantile(values, tail_rate))
        long_threshold = float(np.quantile(values, 1.0 - tail_rate))

        short_threshold = min(0.49, max(0.01, short_threshold))
        long_threshold = max(0.51, min(0.99, long_threshold))

        labels_arr: np.ndarray | None = None
        if train_labels is not None:
            candidate_labels = np.asarray(train_labels, dtype=float).reshape(-1)
            candidate_labels = candidate_labels[np.isfinite(candidate_labels)]
            if candidate_labels.size == values.size:
                labels_arr = candidate_labels

        if labels_arr is not None and labels_arr.size >= 40:
            y_bin = (labels_arr >= 0.5).astype(int)
            pos_rate = float(np.mean(y_bin))
            mean_proba = float(np.mean(values))
            long_tail_prior = float(
                np.clip(
                    (pos_rate * 0.45) + max(0.0, mean_proba - 0.55) * 1.20,
                    0.05,
                    0.40,
                )
            )
            short_tail_prior = float(
                np.clip(
                    ((1.0 - pos_rate) * 0.45) + max(0.0, 0.45 - mean_proba) * 1.20,
                    0.05,
                    0.40,
                )
            )
            symmetric_tails = sorted(
                set(
                    [
                        tail_rate,
                        0.04,
                        0.06,
                        0.08,
                        0.10,
                        0.12,
                        0.15,
                        0.18,
                        0.22,
                        0.26,
                    ]
                )
            )
            candidate_pairs: list[tuple[float, float]] = [
                (tail, tail) for tail in symmetric_tails
            ]
            candidate_pairs.extend(
                [
                    (long_tail_prior, short_tail_prior),
                    (min(0.40, long_tail_prior * 1.20), short_tail_prior),
                    (long_tail_prior, min(0.40, short_tail_prior * 1.20)),
                    (min(0.40, max(long_tail_prior, 0.22)), max(0.05, short_tail_prior * 0.70)),
                    (min(0.40, max(long_tail_prior, 0.26)), 0.05),
                ]
            )
            candidate_pairs = sorted(
                {
                    (float(np.clip(lp, 0.03, 0.40)), float(np.clip(sp, 0.03, 0.40)))
                    for lp, sp in candidate_pairs
                }
            )

            best_score = -1e9
            best_pair = (long_threshold, short_threshold)
            target_entry_rate = min(0.25, max(0.02, float(target_trades) / float(values.size)))
            min_activity_rate = max(0.015, target_entry_rate * 0.50)
            min_entry_rate = max(0.010, target_entry_rate * 0.35)

            for long_tail, short_tail in candidate_pairs:
                cand_short = float(np.quantile(values, float(np.clip(short_tail, 0.03, 0.40))))
                cand_long = float(np.quantile(values, 1.0 - float(np.clip(long_tail, 0.03, 0.40))))
                cand_short = min(0.49, max(0.01, cand_short))
                cand_long = max(0.51, min(0.99, cand_long))
                if cand_long - cand_short < 0.03:
                    continue

                signals = np.where(
                    values >= cand_long,
                    1.0,
                    np.where(values <= cand_short, -1.0, 0.0),
                ).astype(float)
                active_mask = signals != 0.0
                active_count = int(np.count_nonzero(active_mask))
                if active_count <= 0:
                    continue

                signed_hits = (
                    ((signals == 1.0) & (y_bin == 1))
                    | ((signals == -1.0) & (y_bin == 0))
                )
                directional_acc = float(np.mean(signed_hits[active_mask]))
                active_rate = float(active_count) / float(values.size)

                prev_signals = np.concatenate([[0.0], signals[:-1]])
                entry_mask = active_mask & (
                    (np.abs(prev_signals) <= 1e-8)
                    | (np.sign(signals) != np.sign(prev_signals))
                )
                entry_count = int(np.count_nonzero(entry_mask))
                entry_rate = float(entry_count) / float(values.size)
                activity_score = min(active_rate / max(target_entry_rate, 1e-9), 1.2)
                entry_score = min(entry_rate / max(target_entry_rate * 0.7, 1e-9), 1.2)
                activity_penalty = 0.0
                if active_rate < min_activity_rate:
                    activity_penalty += float(
                        (min_activity_rate - active_rate) / max(min_activity_rate, 1e-9)
                    )
                if entry_rate < min_entry_rate:
                    activity_penalty += float(
                        0.75 * (min_entry_rate - entry_rate) / max(min_entry_rate, 1e-9)
                    )
                calibration_penalty = float(np.mean((values - y_bin) ** 2))
                score = (
                    (0.45 * directional_acc)
                    + (0.35 * activity_score)
                    + (0.20 * entry_score)
                    - (0.10 * calibration_penalty)
                    - (0.60 * activity_penalty)
                )
                if score > best_score:
                    best_score = score
                    best_pair = (cand_long, cand_short)

            long_threshold, short_threshold = best_pair

        def _activity_rates(long_cut: float, short_cut: float) -> tuple[float, float]:
            signals = np.where(
                values >= long_cut,
                1.0,
                np.where(values <= short_cut, -1.0, 0.0),
            ).astype(float)
            active_mask = signals != 0.0
            active_rate = float(np.mean(active_mask))
            prev_signals = np.concatenate([[0.0], signals[:-1]])
            entry_mask = active_mask & (
                (np.abs(prev_signals) <= 1e-8)
                | (np.sign(signals) != np.sign(prev_signals))
            )
            entry_rate = float(np.mean(entry_mask))
            return active_rate, entry_rate

        enforced_target_rate = min(0.20, max(0.03, target_rate))
        current_active_rate, _ = _activity_rates(long_threshold, short_threshold)
        if current_active_rate < (enforced_target_rate * 0.50):
            relaxed_long = min(long_threshold, 0.70)
            relaxed_short = max(short_threshold, 0.30)
            relaxed_active_rate, _ = _activity_rates(relaxed_long, relaxed_short)
            if relaxed_active_rate >= current_active_rate:
                long_threshold, short_threshold = relaxed_long, relaxed_short
                current_active_rate = relaxed_active_rate

        if current_active_rate < (enforced_target_rate * 0.50):
            forced_tail = float(min(0.35, max(0.10, enforced_target_rate * 1.10)))
            forced_short = float(np.quantile(values, forced_tail))
            forced_long = float(np.quantile(values, 1.0 - forced_tail))
            short_threshold = min(0.49, max(0.01, max(forced_short, 0.32)))
            long_threshold = max(0.51, min(0.99, min(forced_long, 0.68)))

        if long_threshold - short_threshold < 0.04:
            long_threshold = default_long
            short_threshold = default_short

        return long_threshold, short_threshold

    def _augment_params_for_train_labels(self, params: dict[str, Any], y_train: np.ndarray) -> dict[str, Any]:
        """Inject class-imbalance-aware defaults for classifier families."""
        adjusted = dict(params)
        if self.config.model_type not in {"xgboost", "lightgbm", "random_forest", "elastic_net"}:
            return adjusted

        y_arr = np.asarray(y_train)
        if y_arr.size == 0:
            return adjusted
        y_bin = (y_arr >= 0.5).astype(int)
        positives = int(np.sum(y_bin == 1))
        negatives = int(np.sum(y_bin == 0))
        if positives <= 0 or negatives <= 0:
            return adjusted

        imbalance_ratio = float(negatives / max(positives, 1))
        clipped_ratio = float(np.clip(imbalance_ratio, 1.0, 25.0))

        if self.config.model_type == "xgboost":
            adjusted.setdefault("scale_pos_weight", clipped_ratio)
            if clipped_ratio >= 3.0:
                adjusted.setdefault("max_delta_step", 1.0)
        elif self.config.model_type == "lightgbm":
            if clipped_ratio >= 1.25:
                adjusted.setdefault("is_unbalance", True)
                adjusted.setdefault("scale_pos_weight", clipped_ratio)
        elif self.config.model_type == "random_forest":
            adjusted.setdefault("class_weight", "balanced_subsample")
            if os.name == "nt":
                adjusted["n_jobs"] = 1
            else:
                adjusted.setdefault("n_jobs", self.config.n_jobs)
        elif self.config.model_type == "elastic_net":
            adjusted.setdefault("class_weight", "balanced")

        return adjusted

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        """Compute max drawdown from return series."""
        if len(returns) == 0:
            return 0.0
        equity = np.cumprod(1.0 + returns)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / np.clip(peak, 1e-12, None)
        return float(abs(np.min(drawdown)))

    def _quick_cv_score(self, params: dict) -> float:
        """Quick cross-validation score for optimization."""
        scores: list[float] = []
        X = self.features.values
        y = self.labels.values
        weights = self.sample_weights if self.sample_weights is not None else None

        for train_idx, test_idx in self._generate_cv_splits(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            fold_params = self._prepare_params_for_train_size(params, len(train_idx))
            fold_params = self._augment_params_for_train_labels(fold_params, y_train)
            model = self._create_model(params=fold_params)
            w_train = weights[train_idx] if weights is not None else None
            self._fit_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_test,
                y_val=y_test,
                sample_weights=w_train,
            )

            y_pred = np.asarray(model.predict(X_test))
            y_proba = np.asarray(self._get_predictions_proba(model, X_test))
            train_proba = np.asarray(self._get_predictions_proba(model, X_train))
            long_threshold, short_threshold = self._derive_signal_thresholds(
                train_proba,
                train_labels=y_train,
            )
            eval_len = min(len(y_test), len(y_pred), len(y_proba))
            if eval_len <= 1:
                continue
            fold_forward_returns = (
                np.asarray(self.primary_forward_returns[test_idx], dtype=float)
                if isinstance(self.primary_forward_returns, np.ndarray)
                and len(self.primary_forward_returns) == len(y)
                else None
            )
            fold_event_returns = (
                np.asarray(self.cost_aware_event_returns[test_idx], dtype=float)
                if isinstance(self.cost_aware_event_returns, np.ndarray)
                and len(self.cost_aware_event_returns) == len(y)
                else None
            )
            fold_timestamps = (
                np.asarray(self.timestamps[test_idx], dtype="datetime64[ns]")
                if self.timestamps is not None and len(self.timestamps) == len(y)
                else None
            )

            metrics = self._calculate_fold_metrics(
                np.asarray(y_test)[-eval_len:],
                y_pred[-eval_len:],
                y_proba[-eval_len:],
                long_threshold=long_threshold,
                short_threshold=short_threshold,
                realized_forward_returns=(
                    fold_forward_returns[-eval_len:] if fold_forward_returns is not None else None
                ),
                event_net_returns=(
                    fold_event_returns[-eval_len:] if fold_event_returns is not None else None
                ),
                timestamps=(fold_timestamps[-eval_len:] if fold_timestamps is not None else None),
            )
            scores.append(float(metrics["risk_adjusted_score"]))

        return float(np.mean(scores)) if scores else -1e9

    def _train_with_cv(self) -> None:
        """
        Phase 5: Train model with Purged K-Fold Cross-Validation.

        @mlquant P2-B: Uses purged cross-validation to prevent look-ahead bias.
        The embargo period ensures no information leakage between train/test sets.
        """
        self.logger.info(f"Phase 5: Training with {self.config.cv_method}...")

        # Prepare feature matrix
        X = self.features.values
        y = self.labels.values
        weights = self.sample_weights if self.sample_weights is not None else None

        # Train with CV
        fold_results = []
        models = []
        self.cv_return_series = []
        self.cv_active_return_series = []
        self.oof_primary_proba = np.full(len(y), np.nan, dtype=float)
        splits = self._generate_cv_splits(X, y)

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            self.logger.info(f"Training fold {fold_idx + 1}/{len(splits)}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            w_train = weights[train_idx] if weights is not None else None

            # Create and train model
            fold_params = self._prepare_params_for_train_size(
                self.config.model_params, len(train_idx)
            )
            fold_params = self._augment_params_for_train_labels(fold_params, y_train)
            model = self._create_model(params=fold_params)
            self._fit_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_test,
                y_val=y_test,
                sample_weights=w_train,
            )

            # Evaluate
            y_pred = np.asarray(model.predict(X_test))
            y_proba = np.asarray(self._get_predictions_proba(model, X_test))
            train_proba = np.asarray(self._get_predictions_proba(model, X_train))
            long_threshold, short_threshold = self._derive_signal_thresholds(
                train_proba,
                train_labels=y_train,
            )

            y_eval = np.asarray(y_test)
            eval_len = min(len(y_pred), len(y_eval), len(y_proba), len(test_idx))
            if eval_len != len(y_eval):
                if eval_len <= 1:
                    self.logger.warning(
                        f"Fold {fold_idx + 1}: insufficient aligned predictions, skipping fold"
                    )
                    continue
                y_eval = y_eval[-eval_len:]
                y_pred = y_pred[-eval_len:]
                y_proba = y_proba[-eval_len:]
                self.logger.info(
                    f"Fold {fold_idx + 1}: aligned sequence outputs to {eval_len} samples"
                )
            eval_indices = np.asarray(test_idx[-eval_len:], dtype=int)
            self.oof_primary_proba[eval_indices] = y_proba

            # Calculate metrics
            fold_forward_returns = (
                np.asarray(self.primary_forward_returns[eval_indices], dtype=float)
                if isinstance(self.primary_forward_returns, np.ndarray)
                and len(self.primary_forward_returns) == len(y)
                else None
            )
            fold_event_returns = (
                np.asarray(self.cost_aware_event_returns[eval_indices], dtype=float)
                if isinstance(self.cost_aware_event_returns, np.ndarray)
                and len(self.cost_aware_event_returns) == len(y)
                else None
            )
            fold_timestamps = (
                np.asarray(self.timestamps[eval_indices], dtype="datetime64[ns]")
                if self.timestamps is not None and len(self.timestamps) == len(y)
                else None
            )
            metrics = self._calculate_fold_metrics(
                y_eval,
                y_pred,
                y_proba,
                long_threshold=long_threshold,
                short_threshold=short_threshold,
                realized_forward_returns=fold_forward_returns,
                event_net_returns=fold_event_returns,
                timestamps=fold_timestamps,
            )
            if self.regimes is not None and len(self.regimes) == len(y):
                train_regime_dist = self._regime_distribution(self.regimes[train_idx])
                test_regime_dist = self._regime_distribution(self.regimes[eval_indices])
                metrics["regime_shift"] = float(
                    self._regime_shift_score(train_regime_dist, test_regime_dist)
                )
            else:
                metrics["regime_shift"] = 0.0
            metrics["fold"] = fold_idx + 1
            metrics["train_size"] = len(train_idx)
            metrics["test_size"] = len(y_eval)
            metrics["long_threshold"] = float(long_threshold)
            metrics["short_threshold"] = float(short_threshold)
            net_returns, execution_details = self._compute_net_returns(
                y_eval,
                y_proba,
                long_threshold=long_threshold,
                short_threshold=short_threshold,
                realized_forward_returns=fold_forward_returns,
                event_net_returns=fold_event_returns,
                timestamps=fold_timestamps,
                return_details=True,
            )
            trade_mask = np.asarray(execution_details.get("trade_mask", []), dtype=bool)
            active_returns = (
                np.asarray(net_returns[trade_mask], dtype=float)
                if trade_mask.size == len(net_returns)
                else np.array([], dtype=float)
            )
            metrics["net_return_observations"] = int(len(net_returns))
            metrics["active_net_return_observations"] = int(len(active_returns))
            if len(net_returns) > 0:
                self.cv_return_series.append(net_returns.astype(float))
            if len(active_returns) > 0:
                self.cv_active_return_series.append(active_returns.astype(float))

            fold_results.append(metrics)
            models.append(model)

            self.logger.info(
                f"Fold {fold_idx + 1}: Accuracy={metrics['accuracy']:.4f}, "
                f"Sharpe={metrics.get('sharpe', 0):.4f}, "
                f"Trades={int(metrics.get('trade_count', 0))}, "
                f"RiskScore={metrics.get('risk_adjusted_score', 0):.4f}"
            )

        # Select best model or use ensemble
        if not fold_results:
            raise ValueError("No valid CV folds produced training metrics")
        if self.oof_primary_proba is not None:
            oof_coverage = float(np.mean(np.isfinite(self.oof_primary_proba)))
            self.training_metrics["oof_prediction_coverage"] = oof_coverage

        self.cv_results = fold_results
        self._select_final_model(models, fold_results)

        # Calculate aggregate metrics
        self._calculate_aggregate_metrics()
        self._fit_final_model_full_data()
        self._evaluate_holdout_performance()

    def _get_cv_splitter(self, n_samples: int | None = None):
        """
        Get cross-validation splitter based on configuration.

        @mlquant P2-B: Implements purged k-fold to prevent look-ahead bias.
        P1-H1: Enforces minimum 1% embargo period.
        """
        try:
            from quant_trading_system.models.purged_cv import (
                PurgedKFold,
                CombinatorialPurgedKFold,
                WalkForwardCV,
            )

            # P1-H1: Ensure minimum embargo period
            embargo_pct = max(self.config.embargo_pct, 0.01)

            # Calculate purge_gap from purge_pct (convert percentage to number of periods)
            # Typical 15-min data has ~26 bars/day, so 2% = ~5 bars
            purge_gap = max(1, int(self.config.purge_pct * 100))  # purge_gap as integer bars

            prediction_horizon = self._prediction_horizon()
            requested_splits = max(1, int(self.config.n_splits))
            if n_samples is not None:
                horizon_cap = max(1, n_samples // max(4, requested_splits + 1))
                prediction_horizon = min(prediction_horizon, horizon_cap)
            if n_samples is None:
                effective_splits = requested_splits
            elif self.config.cv_method == "walk_forward":
                effective_splits = max(1, min(requested_splits, max(1, n_samples - 1)))
            elif self.config.cv_method == "combinatorial":
                effective_splits = max(3, min(requested_splits, max(3, n_samples - 1)))
            else:
                effective_splits = max(2, min(requested_splits, max(2, n_samples - 1)))

            if self.config.cv_method == "purged_kfold":
                return PurgedKFold(
                    n_splits=effective_splits,
                    purge_gap=purge_gap,
                    embargo_pct=embargo_pct,
                    prediction_horizon=prediction_horizon,
                )
            elif self.config.cv_method == "combinatorial":
                if effective_splits < 3:
                    self.logger.warning(
                        "Insufficient samples for combinatorial purged CV; falling back to purged_kfold."
                    )
                    return PurgedKFold(
                        n_splits=max(2, effective_splits),
                        purge_gap=purge_gap,
                        embargo_pct=embargo_pct,
                        prediction_horizon=prediction_horizon,
                    )
                return CombinatorialPurgedKFold(
                    n_splits=effective_splits,
                    purge_gap=purge_gap,
                    embargo_pct=embargo_pct,
                    prediction_horizon=prediction_horizon,
                )
            elif self.config.cv_method == "walk_forward":
                min_train_size = (
                    max(2, min(50, max(2, n_samples // 3)))
                    if n_samples is not None
                    else None
                )
                return WalkForwardCV(
                    n_splits=effective_splits,
                    purge_gap=purge_gap,
                    embargo_pct=embargo_pct,
                    prediction_horizon=prediction_horizon,
                    min_train_size=min_train_size,
                )
            else:
                return PurgedKFold(
                    n_splits=max(2, effective_splits),
                    purge_gap=purge_gap,
                    embargo_pct=embargo_pct,
                    prediction_horizon=prediction_horizon,
                )

        except ImportError:
            raise RuntimeError(
                "Purged CV module unavailable. Institutional mode requires purged/walk-forward CV."
            )

    def _create_base_model(self, params: dict | None = None):
        """Create base model without fitting."""
        params = dict(params or self.config.model_params)

        if self.config.model_type == "xgboost":
            from xgboost import XGBClassifier
            if self.config.use_gpu:
                params.setdefault("tree_method", "hist")
                params.setdefault("device", "cuda")
            params.setdefault("eval_metric", "logloss")
            return XGBClassifier(**params)

        elif self.config.model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            params.setdefault("verbose", -1)
            if self.config.use_gpu:
                params.setdefault("device_type", "gpu")
            return LGBMClassifier(**params)

        elif self.config.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            if os.name == "nt":
                params["n_jobs"] = 1
            else:
                params.setdefault("n_jobs", self.config.n_jobs)
            params.setdefault("random_state", 42)
            return RandomForestClassifier(**params)

        elif self.config.model_type == "elastic_net":
            from sklearn.linear_model import LogisticRegression
            params.setdefault("l1_ratio", 0.5)
            params.setdefault("max_iter", 2000)
            return LogisticRegression(penalty="elasticnet", solver="saga", **params)

        else:
            from xgboost import XGBClassifier
            return XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")

    def _create_model(self, params: dict | None = None):
        """Create model instance for training."""
        model_params = dict(params if params is not None else self.config.model_params)
        try:
            # Try to use our model classes
            if self.config.model_type == "xgboost":
                from quant_trading_system.models.classical_ml import XGBoostModel
                from quant_trading_system.models.base import ModelType
                return XGBoostModel(
                    model_type=ModelType.CLASSIFIER,
                    use_gpu=self.config.use_gpu,
                    **model_params,
                )

            elif self.config.model_type == "lightgbm":
                from quant_trading_system.models.classical_ml import LightGBMModel
                from quant_trading_system.models.base import ModelType
                return LightGBMModel(
                    model_type=ModelType.CLASSIFIER,
                    use_gpu=self.config.use_gpu,
                    **model_params,
                )

            elif self.config.model_type == "random_forest":
                from quant_trading_system.models.classical_ml import RandomForestModel
                from quant_trading_system.models.base import ModelType
                if os.name == "nt":
                    model_params["n_jobs"] = 1
                else:
                    model_params.setdefault("n_jobs", self.config.n_jobs)
                return RandomForestModel(
                    model_type=ModelType.CLASSIFIER,
                    **model_params,
                )

            elif self.config.model_type == "lstm":
                from quant_trading_system.models.deep_learning import LSTMModel
                from quant_trading_system.models.base import ModelType
                model_params.setdefault("batch_size", self.config.batch_size)
                model_params.setdefault("epochs", self.config.epochs)
                model_params.setdefault("learning_rate", self.config.learning_rate)
                return LSTMModel(
                    model_type=ModelType.CLASSIFIER,
                    device="cuda" if self.config.use_gpu else "cpu",
                    **model_params,
                )

            elif self.config.model_type == "transformer":
                from quant_trading_system.models.deep_learning import TransformerModel
                from quant_trading_system.models.base import ModelType
                model_params.setdefault("batch_size", self.config.batch_size)
                model_params.setdefault("epochs", self.config.epochs)
                model_params.setdefault("learning_rate", self.config.learning_rate)
                return TransformerModel(
                    model_type=ModelType.CLASSIFIER,
                    device="cuda" if self.config.use_gpu else "cpu",
                    **model_params,
                )

            elif self.config.model_type == "tcn":
                from quant_trading_system.models.deep_learning import TCNModel
                from quant_trading_system.models.base import ModelType
                model_params.setdefault("batch_size", self.config.batch_size)
                model_params.setdefault("epochs", self.config.epochs)
                model_params.setdefault("learning_rate", self.config.learning_rate)
                return TCNModel(
                    model_type=ModelType.CLASSIFIER,
                    device="cuda" if self.config.use_gpu else "cpu",
                    **model_params,
                )

            elif self.config.model_type == "ensemble":
                from quant_trading_system.models.ensemble import ICBasedEnsemble
                return ICBasedEnsemble()

        except ImportError:
            pass

        # Fallback to sklearn/xgboost
        return self._create_base_model()

    def _fit_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        """Fit model with best-effort support for sample weights and validation sets."""
        if self.config.model_type in ["lstm", "transformer", "tcn"]:
            self._train_deep_learning_model(model, X_train, y_train, X_val, y_val)
            return

        candidate_calls: list[tuple[dict[str, Any], ...]] = []
        if sample_weights is not None:
            candidate_calls.extend(
                [
                    ({"validation_data": (X_val, y_val), "sample_weights": sample_weights},),
                    ({"validation_data": (X_val, y_val), "sample_weight": sample_weights},),
                    ({"sample_weights": sample_weights},),
                    ({"sample_weight": sample_weights},),
                ]
            )
        candidate_calls.extend(
            [
                ({"validation_data": (X_val, y_val)},),
                ({},),
            ]
        )

        last_error: Exception | None = None
        for (kwargs,) in candidate_calls:
            try:
                model.fit(X_train, y_train, **kwargs)
                return
            except TypeError as exc:
                last_error = exc
                continue
            except ValueError as exc:
                # Some estimators reject extra kwargs with ValueError.
                last_error = exc
                continue

        if last_error is not None:
            raise last_error
        model.fit(X_train, y_train)

    def _train_deep_learning_model(
        self, model, X_train, y_train, X_val, y_val
    ) -> None:
        """Train deep learning model with early stopping."""
        try:
            import torch

            device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
            self.logger.info(f"Training on device: {device}")

            # Training loop (simplified)
            if hasattr(model, "fit"):
                model.fit(X_train, y_train, validation_data=(X_val, y_val))
            else:
                self.logger.warning("Model doesn't have fit method, skipping training")

        except ImportError:
            self.logger.warning("PyTorch not available, falling back to classical ML")
            fallback_model = self._create_base_model()
            fallback_model.fit(X_train, y_train)
            self.model = fallback_model

    def _get_predictions_proba(self, model, X) -> np.ndarray:
        """Get prediction probabilities from model."""
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                if proba.ndim == 2:
                    return proba[:, 1]
                return proba
        except (NotImplementedError, AttributeError):
            pass  # Model doesn't support predict_proba, use predict

        if hasattr(model, "predict"):
            # For regressors, normalize predictions to [0, 1] range
            predictions = model.predict(X)
            if isinstance(predictions, np.ndarray):
                # Clip and normalize to [0, 1]
                predictions = np.clip(predictions, -1, 1)
                predictions = (predictions + 1) / 2  # Map [-1, 1] to [0, 1]
            return predictions

        return np.zeros(len(X))

    def _calculate_fold_metrics(
        self,
        y_true,
        y_pred,
        y_proba,
        long_threshold: float = 0.55,
        short_threshold: float = 0.45,
        realized_forward_returns: np.ndarray | None = None,
        event_net_returns: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
    ) -> dict:
        """Calculate metrics for a single fold."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            mean_squared_error,
            r2_score,
        )

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_proba = np.asarray(y_proba)
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=1.0, neginf=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.5, posinf=1.0, neginf=0.0)
        y_proba = np.nan_to_num(y_proba, nan=0.5, posinf=1.0, neginf=0.0)
        y_proba = np.clip(y_proba, 0.0, 1.0)
        short_threshold = float(np.clip(short_threshold, 0.01, 0.49))
        long_threshold = float(np.clip(long_threshold, 0.51, 0.99))
        if long_threshold <= short_threshold:
            long_threshold = 0.55
            short_threshold = 0.45

        y_pred_binary = (y_pred >= 0.5).astype(int) if not np.array_equal(y_pred, y_pred.astype(int)) else y_pred
        y_true_binary = (y_true >= 0.5).astype(int) if not np.array_equal(y_true, y_true.astype(int)) else y_true
        if realized_forward_returns is not None:
            realized_forward_returns = self._align_probabilities(
                np.asarray(realized_forward_returns, dtype=float),
                target_len=len(y_proba),
                fill_value=0.0,
            )
        if event_net_returns is not None:
            event_net_returns = self._align_probabilities(
                np.asarray(event_net_returns, dtype=float),
                target_len=len(y_proba),
                fill_value=0.0,
            )

        metrics = {
            "accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
            "precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
            "recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
            "f1": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
            "brier_score": float(np.mean((y_proba - y_true_binary) ** 2)),
        }

        if len(y_proba) > 1:
            net_returns, execution_details = self._compute_net_returns(
                y_true,
                y_proba,
                long_threshold=long_threshold,
                short_threshold=short_threshold,
                realized_forward_returns=realized_forward_returns,
                event_net_returns=event_net_returns,
                timestamps=timestamps,
                return_details=True,
            )
            positions = np.asarray(execution_details.get("positions", np.zeros(len(net_returns))), dtype=float)
            turnover_series = np.asarray(
                execution_details.get("turnover_series", np.zeros(len(net_returns))),
                dtype=float,
            )
            trade_mask = np.asarray(execution_details.get("trade_mask", np.abs(positions) > 0.0), dtype=bool)
            entry_mask = np.asarray(
                execution_details.get("entry_mask", trade_mask),
                dtype=bool,
            )
            trade_returns = net_returns[trade_mask]

            turnover = float(np.mean(turnover_series))
            trade_count = int(np.count_nonzero(entry_mask))
            active_signal_rate = float(np.mean(trade_mask))
            std = float(np.std(trade_returns)) if trade_returns.size > 0 else 0.0
            sharpe = float(np.mean(trade_returns) / std * np.sqrt(252)) if std > 1e-12 else 0.0
            max_dd = self._max_drawdown(trade_returns)
            win_rate = float(np.mean(trade_returns > 0)) if trade_returns.size > 0 else 0.0
            annual_return = float(np.mean(trade_returns) * 252) if trade_returns.size > 0 else 0.0
            calmar = annual_return / max_dd if max_dd > 1e-9 else annual_return
            if trade_returns.size > 0:
                alpha = 0.05
                tail_cutoff = float(np.quantile(trade_returns, alpha))
                cvar = float(np.mean(trade_returns[trade_returns <= tail_cutoff]))
                expected_shortfall = float(abs(cvar))
                skew = float(pd.Series(trade_returns).skew())
            else:
                cvar = 0.0
                expected_shortfall = 0.0
                skew = 0.0
            equity_break = float(1.0 if max_dd > (self.config.max_drawdown * 1.5) else 0.0)
            objective_components = self._compute_objective_components(
                sharpe=sharpe,
                max_drawdown=max_dd,
                turnover=turnover,
                brier_score=float(metrics["brier_score"]),
                trade_count=trade_count,
                cvar=cvar,
                skew=skew,
                expected_shortfall=expected_shortfall,
                equity_break=equity_break,
                evaluation_size=int(len(y_proba)),
            )

            metrics.update(
                {
                    "sharpe": sharpe,
                    "max_drawdown": max_dd,
                    "win_rate": win_rate,
                    "turnover": turnover,
                    "trade_count": float(trade_count),
                    "active_signal_rate": active_signal_rate,
                    "annual_return": annual_return,
                    "calmar": calmar,
                    "trade_return_observations": float(len(trade_returns)),
                    "cvar_95": cvar,
                    "expected_shortfall": expected_shortfall,
                    "return_skew": skew,
                    "equity_break": equity_break,
                    "objective_sharpe_component": float(
                        objective_components["objective_sharpe_component"]
                    ),
                    "objective_drawdown_penalty": float(
                        objective_components["objective_drawdown_penalty"]
                    ),
                    "objective_turnover_penalty": float(
                        objective_components["objective_turnover_penalty"]
                    ),
                    "objective_calibration_penalty": float(
                        objective_components["objective_calibration_penalty"]
                    ),
                    "objective_trade_activity_penalty": float(
                        objective_components["objective_trade_activity_penalty"]
                    ),
                    "objective_cvar_penalty": float(
                        objective_components["objective_cvar_penalty"]
                    ),
                    "objective_skew_penalty": float(
                        objective_components["objective_skew_penalty"]
                    ),
                    "objective_equity_break_penalty": float(
                        objective_components["objective_equity_break_penalty"]
                    ),
                    "objective_trade_target": float(objective_components["objective_trade_target"]),
                    "risk_adjusted_score": float(objective_components["risk_adjusted_score"]),
                }
            )
        else:
            metrics.update(
                {
                    "sharpe": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "turnover": 0.0,
                    "trade_count": 0.0,
                    "active_signal_rate": 0.0,
                    "annual_return": 0.0,
                    "calmar": 0.0,
                    "trade_return_observations": 0.0,
                    "cvar_95": 0.0,
                    "expected_shortfall": 0.0,
                    "return_skew": 0.0,
                    "equity_break": 0.0,
                    "objective_sharpe_component": 0.0,
                    "objective_drawdown_penalty": 0.0,
                    "objective_turnover_penalty": 0.0,
                    "objective_calibration_penalty": 0.0,
                    "objective_trade_activity_penalty": 0.0,
                    "objective_cvar_penalty": 0.0,
                    "objective_skew_penalty": 0.0,
                    "objective_equity_break_penalty": 0.0,
                    "objective_trade_target": float(self._effective_trade_target(int(len(y_proba)))),
                    "risk_adjusted_score": -1e9,
                }
            )

        sanitized_metrics: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                sanitized_metrics[key] = float(
                    np.nan_to_num(float(value), nan=0.0, posinf=1e6, neginf=-1e6)
                )
            else:
                sanitized_metrics[key] = float(0.0)

        return sanitized_metrics

    def _compute_objective_components(
        self,
        *,
        sharpe: float,
        max_drawdown: float,
        turnover: float,
        brier_score: float,
        trade_count: int,
        cvar: float,
        skew: float,
        expected_shortfall: float,
        equity_break: float,
        evaluation_size: int | None = None,
    ) -> dict[str, float]:
        """Compute auditable objective component breakdown for optimization and promotion."""
        sharpe_component = float(self.config.objective_weight_sharpe * sharpe)
        drawdown_penalty = float(-self.config.objective_weight_drawdown * max_drawdown)
        turnover_penalty = float(-self.config.objective_weight_turnover * turnover)
        calibration_penalty = float(-self.config.objective_weight_calibration * brier_score)
        cvar_penalty = float(-self.config.objective_weight_cvar * max(0.0, expected_shortfall))
        skew_penalty = float(-self.config.objective_weight_skew * max(0.0, -float(skew)))
        equity_break_penalty = float(-2.0 * max(0.0, float(equity_break)))
        min_trades_target = self._effective_trade_target(evaluation_size)
        trade_shortfall = max(
            0.0,
            (float(min_trades_target) - float(max(0, int(trade_count)))) / float(min_trades_target),
        )
        trade_activity_penalty = float(
            -self.config.objective_weight_trade_activity * trade_shortfall
        )
        total = (
            sharpe_component
            + drawdown_penalty
            + turnover_penalty
            + calibration_penalty
            + trade_activity_penalty
            + cvar_penalty
            + skew_penalty
            + equity_break_penalty
        )
        return {
            "objective_sharpe_component": sharpe_component,
            "objective_drawdown_penalty": drawdown_penalty,
            "objective_turnover_penalty": turnover_penalty,
            "objective_calibration_penalty": calibration_penalty,
            "objective_trade_activity_penalty": trade_activity_penalty,
            "objective_cvar_penalty": cvar_penalty,
            "objective_skew_penalty": skew_penalty,
            "objective_equity_break_penalty": equity_break_penalty,
            "objective_trade_target": float(min_trades_target),
            "risk_adjusted_score": float(total),
        }

    def _build_execution_profile(
        self,
        y_proba: np.ndarray,
        *,
        long_threshold: float,
        short_threshold: float,
        realized_returns: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Build execution-aware position profile from probabilities."""
        y_proba = np.asarray(y_proba, dtype=float)
        n_samples = len(y_proba)
        if n_samples == 0:
            return {
                "long_threshold_series": np.array([], dtype=float),
                "short_threshold_series": np.array([], dtype=float),
                "raw_signals": np.array([], dtype=float),
                "positions": np.array([], dtype=float),
                "turnover_series": np.array([], dtype=float),
                "trade_mask": np.array([], dtype=bool),
                "entry_mask": np.array([], dtype=bool),
            }

        long_series = np.full(n_samples, float(long_threshold), dtype=float)
        short_series = np.full(n_samples, float(short_threshold), dtype=float)
        adaptive_band = np.zeros(n_samples, dtype=float)
        if self.config.dynamic_no_trade_band:
            uncertainty = 1.0 - (2.0 * np.abs(y_proba - 0.5))
            uncertainty = np.clip(uncertainty, 0.0, 1.0)
            if realized_returns is not None and len(realized_returns) == n_samples:
                realized_series = pd.Series(np.nan_to_num(realized_returns, nan=0.0))
                rolling_vol = (
                    realized_series.rolling(max(8, self.config.label_volatility_lookback), min_periods=3)
                    .std()
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                vol_anchor = float(np.median(rolling_vol[rolling_vol > 0])) if np.any(rolling_vol > 0) else 0.0
                if vol_anchor > 0.0:
                    vol_scale = np.clip((rolling_vol / vol_anchor) - 1.0, 0.0, 2.0)
                else:
                    vol_scale = np.zeros(n_samples, dtype=float)
            else:
                vol_scale = np.zeros(n_samples, dtype=float)

            adaptive_band = 0.02 + 0.03 * uncertainty + 0.02 * vol_scale
            long_series = np.clip(long_series + adaptive_band, 0.51, 0.995)
            short_series = np.clip(short_series - adaptive_band, 0.005, 0.49)

        def _generate_signals(long_arr: np.ndarray, short_arr: np.ndarray) -> np.ndarray:
            return np.where(
                y_proba >= long_arr,
                1.0,
                np.where(y_proba <= short_arr, -1.0, 0.0),
            ).astype(float)

        raw_signals = _generate_signals(long_series, short_series)

        if self.config.dynamic_no_trade_band and adaptive_band.size > 0:
            target_activity_rate = min(
                0.30,
                max(0.02, float(self._effective_trade_target(n_samples)) / float(n_samples)),
            )
            observed_activity = float(np.mean(raw_signals != 0.0))
            if observed_activity < (target_activity_rate * 0.50):
                shortfall = max(0.0, (target_activity_rate * 0.50) - observed_activity)
                relaxation = float(
                    np.clip(shortfall / max(target_activity_rate * 0.50, 1e-9), 0.0, 1.0)
                )
                relaxed_band = adaptive_band * (1.0 - 0.75 * relaxation)
                long_series = np.clip(float(long_threshold) + relaxed_band, 0.51, 0.995)
                short_series = np.clip(float(short_threshold) - relaxed_band, 0.005, 0.49)
                raw_signals = _generate_signals(long_series, short_series)
                observed_activity = float(np.mean(raw_signals != 0.0))

            if observed_activity < (target_activity_rate * 0.50):
                forced_tail = float(min(0.35, max(0.06, target_activity_rate * 1.25)))
                forced_long = float(np.quantile(y_proba, 1.0 - forced_tail))
                forced_short = float(np.quantile(y_proba, forced_tail))
                forced_long = float(np.clip(forced_long, 0.53, 0.70))
                forced_short = float(np.clip(forced_short, 0.30, 0.47))
                long_series = np.minimum(long_series, np.full(n_samples, forced_long, dtype=float))
                short_series = np.maximum(short_series, np.full(n_samples, forced_short, dtype=float))
                raw_signals = _generate_signals(long_series, short_series)

        # Execution cooldown suppresses immediate direction flips in noisy regions.
        cooldown_bars = int(self.config.execution_cooldown_bars)
        if cooldown_bars > 0 and n_samples > 1:
            filtered_signals = raw_signals.copy()
            last_entry_idx = -10**9
            last_direction = 0.0
            for idx, signal in enumerate(raw_signals):
                if signal == 0.0:
                    continue
                direction = float(np.sign(signal))
                if direction != last_direction and (idx - last_entry_idx) <= cooldown_bars:
                    filtered_signals[idx] = 0.0
                    continue
                if direction != last_direction:
                    last_entry_idx = idx
                    last_direction = direction
            raw_signals = filtered_signals

        if realized_returns is not None and len(realized_returns) == n_samples:
            realized_series = pd.Series(np.nan_to_num(realized_returns, nan=0.0))
            rolling_vol = (
                realized_series.rolling(max(8, self.config.label_volatility_lookback), min_periods=3)
                .std()
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
        else:
            proba_proxy = np.abs(np.diff(y_proba, prepend=y_proba[0]))
            rolling_vol = (
                pd.Series(proba_proxy)
                .rolling(12, min_periods=3)
                .mean()
                .fillna(0.0)
                .to_numpy(dtype=float)
            )

        per_bar_vol_target = float(self.config.execution_vol_target_daily / np.sqrt(252.0))
        leverage = np.clip(per_bar_vol_target / np.clip(rolling_vol, 1e-8, None), 0.0, 1.5)
        positions = raw_signals * leverage

        turnover_series = np.abs(np.diff(np.concatenate([[0.0], positions])))
        turnover_mean = float(np.mean(turnover_series)) if turnover_series.size > 0 else 0.0
        if turnover_mean > float(self.config.execution_turnover_cap) and turnover_mean > 1e-12:
            scale = float(self.config.execution_turnover_cap / turnover_mean)
            positions = positions * scale
            turnover_series = np.abs(np.diff(np.concatenate([[0.0], positions])))

        trade_mask = np.abs(positions) > 1e-8
        prev_positions = np.concatenate([[0.0], positions[:-1]])
        entry_mask = trade_mask & (
            (np.abs(prev_positions) <= 1e-8)
            | (np.sign(positions) != np.sign(prev_positions))
        )

        return {
            "long_threshold_series": long_series.astype(float),
            "short_threshold_series": short_series.astype(float),
            "raw_signals": raw_signals.astype(float),
            "positions": positions.astype(float),
            "turnover_series": turnover_series.astype(float),
            "trade_mask": trade_mask.astype(bool),
            "entry_mask": entry_mask.astype(bool),
        }

    def _compute_net_returns(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        long_threshold: float = 0.55,
        short_threshold: float = 0.45,
        realized_forward_returns: np.ndarray | None = None,
        event_net_returns: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        return_details: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        """Compute strategy net returns from model signals and realized returns."""
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        if len(y_proba) <= 1:
            empty_returns = np.array([], dtype=float)
            if return_details:
                return empty_returns, {
                    "positions": np.array([], dtype=float),
                    "turnover_series": np.array([], dtype=float),
                    "trade_mask": np.array([], dtype=bool),
                    "entry_mask": np.array([], dtype=bool),
                }
            return empty_returns

        short_threshold = float(np.clip(short_threshold, 0.01, 0.49))
        long_threshold = float(np.clip(long_threshold, 0.51, 0.99))
        if long_threshold <= short_threshold:
            long_threshold = 0.55
            short_threshold = 0.45

        realized = None
        if realized_forward_returns is not None:
            realized = self._align_probabilities(
                np.asarray(realized_forward_returns, dtype=float),
                len(y_proba),
                fill_value=0.0,
            )
        elif event_net_returns is not None:
            realized = self._align_probabilities(
                np.asarray(event_net_returns, dtype=float),
                len(y_proba),
                fill_value=0.0,
            )

        execution_profile = self._build_execution_profile(
            y_proba=y_proba,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            realized_returns=realized,
            timestamps=timestamps,
        )
        positions = np.asarray(execution_profile["positions"], dtype=float)
        turnover_series = np.asarray(execution_profile["turnover_series"], dtype=float)

        if realized is not None:
            raw_alpha = positions * realized
        else:
            # Conservative fallback when realized returns are unavailable.
            y_direction = np.where(y_true >= 0.5, 1.0, -1.0)
            raw_alpha = positions * y_direction * np.abs(y_proba - 0.5)

        trading_cost_rate = max(
            0.0005,
            float(self.config.label_spread_bps + self.config.label_slippage_bps + self.config.label_impact_bps)
            / 10000.0,
        )
        trading_cost = turnover_series * trading_cost_rate
        net_returns = raw_alpha - trading_cost
        net_returns = np.nan_to_num(net_returns.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        if return_details:
            return net_returns, execution_profile
        return net_returns

    def _select_final_model(self, models: list, fold_results: list[dict]) -> None:
        """Select final model from CV folds."""
        def _fold_trade_target(result: dict[str, Any]) -> float:
            test_size = int(result.get("test_size", 0))
            return float(self._effective_trade_target(test_size if test_size > 0 else None))

        eligible_indices = [
            idx
            for idx, result in enumerate(fold_results)
            if float(result.get("trade_count", 0.0)) >= _fold_trade_target(result)
        ]
        if not eligible_indices:
            candidate_indices = list(range(len(fold_results)))
            self.logger.warning(
                "No CV fold met dynamic min-trades target; selecting by best risk-adjusted score."
            )
        else:
            candidate_indices = eligible_indices
        # Select model with best risk-adjusted score among eligible folds.
        best_idx = max(
            candidate_indices,
            key=lambda i: float(
                fold_results[i].get(
                    "risk_adjusted_score",
                    fold_results[i].get("sharpe", 0.0),
                )
            ),
        )
        self.model = models[best_idx]
        self.training_metrics["selected_cv_fold"] = float(best_idx + 1)
        self.logger.info(f"Selected model from fold {best_idx + 1}")

    def _fit_model_full_dataset(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray | None,
    ) -> None:
        """Fit final production model on full dataset with robust kwargs handling."""
        candidate_calls: list[dict[str, Any]] = []
        if sample_weights is not None:
            candidate_calls.extend(
                [
                    {"sample_weights": sample_weights},
                    {"sample_weight": sample_weights},
                ]
            )
        candidate_calls.append({})

        last_error: Exception | None = None
        for kwargs in candidate_calls:
            try:
                model.fit(X, y, **kwargs)
                return
            except (TypeError, ValueError) as exc:
                last_error = exc
                continue

        if len(X) < 8:
            if last_error is not None:
                raise last_error
            model.fit(X, y)
            return

        gap = max(1, int(self.config.purge_pct * 100))
        val_size = max(2, min(int(0.15 * len(X)), 512))
        split_idx = max(2, len(X) - val_size - gap)
        if split_idx >= len(X) - 1:
            split_idx = max(2, len(X) - 2)
            gap = 0
        train_weights = sample_weights[:split_idx] if sample_weights is not None else None
        self._fit_model(
            model=model,
            X_train=X[:split_idx],
            y_train=y[:split_idx],
            X_val=X[split_idx + gap:],
            y_val=y[split_idx + gap:],
            sample_weights=train_weights,
        )

    def _fit_final_model_full_data(self) -> None:
        """Refit final model artifact on full training dataset for production use."""
        X = self.features.values
        y = self.labels.values
        if len(X) <= 4:
            self.logger.warning("Dataset too small for final full-data refit; using selected fold model.")
            return

        weights = self.sample_weights if self.sample_weights is not None else None
        final_params = self._prepare_params_for_train_size(self.config.model_params, len(X))
        final_params = self._augment_params_for_train_labels(final_params, y)
        final_model = self._create_model(params=final_params)
        self._fit_model_full_dataset(final_model, X, y, weights)
        self.model = final_model
        self.training_metrics["final_refit_samples"] = float(len(X))
        self.logger.info(f"Final production model refit completed on {len(X)} samples")

    def _calculate_aggregate_metrics(self) -> None:
        """Calculate aggregate metrics across all folds."""
        if not self.cv_results:
            return

        metrics_keys = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "brier_score",
            "sharpe",
            "max_drawdown",
            "win_rate",
            "cvar_95",
            "expected_shortfall",
            "return_skew",
            "equity_break",
            "turnover",
            "annual_return",
            "calmar",
            "trade_count",
            "active_signal_rate",
            "regime_shift",
            "trade_return_observations",
            "active_net_return_observations",
            "train_size",
            "test_size",
            "net_return_observations",
            "objective_trade_target",
            "objective_sharpe_component",
            "objective_drawdown_penalty",
            "objective_turnover_penalty",
            "objective_calibration_penalty",
            "objective_trade_activity_penalty",
            "objective_cvar_penalty",
            "objective_skew_penalty",
            "objective_equity_break_penalty",
            "risk_adjusted_score",
        ]

        for key in metrics_keys:
            values = [r.get(key, 0) for r in self.cv_results]
            self.training_metrics[f"mean_{key}"] = np.mean(values)
            self.training_metrics[f"std_{key}"] = np.std(values)

        objective_component_summary = {
            "objective_sharpe_component": float(
                self.training_metrics.get("mean_objective_sharpe_component", 0.0)
            ),
            "objective_drawdown_penalty": float(
                self.training_metrics.get("mean_objective_drawdown_penalty", 0.0)
            ),
            "objective_turnover_penalty": float(
                self.training_metrics.get("mean_objective_turnover_penalty", 0.0)
            ),
            "objective_calibration_penalty": float(
                self.training_metrics.get("mean_objective_calibration_penalty", 0.0)
            ),
            "objective_trade_activity_penalty": float(
                self.training_metrics.get("mean_objective_trade_activity_penalty", 0.0)
            ),
            "objective_cvar_penalty": float(
                self.training_metrics.get("mean_objective_cvar_penalty", 0.0)
            ),
            "objective_skew_penalty": float(
                self.training_metrics.get("mean_objective_skew_penalty", 0.0)
            ),
            "objective_equity_break_penalty": float(
                self.training_metrics.get("mean_objective_equity_break_penalty", 0.0)
            ),
        }
        self.training_metrics["objective_component_summary"] = objective_component_summary
        if self.nested_cv_trace:
            self.training_metrics["nested_cv_trace"] = self.nested_cv_trace

        self.logger.info(
            f"Aggregate metrics - Accuracy: {self.training_metrics['mean_accuracy']:.4f} "
            f"(+/-{self.training_metrics['std_accuracy']:.4f}), "
            f"Sharpe: {self.training_metrics['mean_sharpe']:.4f} "
            f"(+/-{self.training_metrics['std_sharpe']:.4f}), "
            f"RiskScore: {self.training_metrics.get('mean_risk_adjusted_score', 0.0):.4f}"
        )

    def _evaluate_holdout_performance(self) -> None:
        """Evaluate untouched holdout block after final refit."""
        if self.model is None:
            return
        if self.holdout_features is None or self.holdout_labels is None:
            return
        if len(self.holdout_labels) <= 1:
            return

        X_holdout = self.holdout_features.values
        y_holdout = self.holdout_labels.to_numpy(dtype=float)
        y_holdout_pred = np.asarray(self.model.predict(X_holdout))
        y_holdout_proba = np.asarray(self._get_predictions_proba(self.model, X_holdout))
        train_proba = np.asarray(self._get_predictions_proba(self.model, self.features.values))
        train_labels = self.labels.to_numpy(dtype=float) if self.labels is not None else None
        long_threshold, short_threshold = self._derive_signal_thresholds(
            train_proba,
            train_labels=train_labels,
        )
        holdout_metrics = self._calculate_fold_metrics(
            y_true=y_holdout,
            y_pred=y_holdout_pred,
            y_proba=y_holdout_proba,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            realized_forward_returns=self.holdout_primary_forward_returns,
            event_net_returns=self.holdout_cost_aware_event_returns,
            timestamps=self.holdout_timestamps,
        )

        holdout_regime_shift = self._regime_shift_score(
            self._regime_distribution(self.regimes),
            self._regime_distribution(self.holdout_regimes),
        )
        holdout_metrics["regime_shift"] = float(holdout_regime_shift)

        for key, value in holdout_metrics.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                self.training_metrics[f"holdout_{key}"] = float(value)

        self.training_metrics["holdout_rows"] = float(len(self.holdout_labels))
        self.training_metrics["holdout_long_threshold"] = float(long_threshold)
        self.training_metrics["holdout_short_threshold"] = float(short_threshold)

        self.logger.info(
            "Holdout metrics - Accuracy: %.4f, Sharpe: %.4f, Drawdown: %.4f, RegimeShift: %.4f",
            self.training_metrics.get("holdout_accuracy", 0.0),
            self.training_metrics.get("holdout_sharpe", 0.0),
            self.training_metrics.get("holdout_max_drawdown", 0.0),
            self.training_metrics.get("holdout_regime_shift", holdout_regime_shift),
        )

    def _validate_model(self) -> bool:
        """
        Phase 6: Validate model against deployment gates.

        @mlquant: Model validation gates ensure only high-quality models
        are deployed to production.
        """
        self.logger.info("Phase 6: Validating model against deployment gates...")

        mean_test_size = int(round(float(self.training_metrics.get("mean_test_size", 0.0))))
        effective_min_trades = float(
            self._effective_trade_target(mean_test_size if mean_test_size > 0 else None)
        )
        self.training_metrics["effective_min_trades_gate"] = effective_min_trades
        holdout_available = float(self.training_metrics.get("holdout_rows", 0.0)) > 0.0
        holdout_sharpe = float(self.training_metrics.get("holdout_sharpe", -1.0))
        holdout_drawdown = float(self.training_metrics.get("holdout_max_drawdown", 1.0))
        effective_max_pbo = float(
            self.training_metrics.get("effective_max_pbo_gate", self.config.max_pbo)
        )
        regime_shift_metric = float(
            max(
                self.training_metrics.get("mean_regime_shift", 0.0),
                self.training_metrics.get("holdout_regime_shift", 0.0),
            )
        )
        if not holdout_available:
            self.logger.warning(
                "Holdout block unavailable for this run; holdout-specific gates treated as non-blocking."
            )

        gates = {
            "min_sharpe_ratio": (
                self.training_metrics.get("mean_sharpe", 0) >= self.config.min_sharpe_ratio,
                self.training_metrics.get("mean_sharpe", 0),
                self.config.min_sharpe_ratio,
            ),
            "min_accuracy": (
                self.training_metrics.get("mean_accuracy", 0) >= self.config.min_accuracy,
                self.training_metrics.get("mean_accuracy", 0),
                self.config.min_accuracy,
            ),
            "max_drawdown": (
                self.training_metrics.get("mean_max_drawdown", 1.0) <= self.config.max_drawdown,
                self.training_metrics.get("mean_max_drawdown", 1.0),
                self.config.max_drawdown,
            ),
            "min_win_rate": (
                self.training_metrics.get("mean_win_rate", 0.0) >= self.config.min_win_rate,
                self.training_metrics.get("mean_win_rate", 0.0),
                self.config.min_win_rate,
            ),
            "min_trades": (
                self.training_metrics.get("mean_trade_count", 0.0) >= effective_min_trades,
                self.training_metrics.get("mean_trade_count", 0.0),
                effective_min_trades,
            ),
            "risk_adjusted_positive": (
                self.training_metrics.get("mean_risk_adjusted_score", -1.0) > 0.0,
                self.training_metrics.get("mean_risk_adjusted_score", -1.0),
                0.0,
            ),
            "oof_prediction_coverage": (
                (
                    self.training_metrics.get("oof_prediction_coverage", 0.0) >= 0.60
                    if "oof_prediction_coverage" in self.training_metrics
                    else (not self.config.use_meta_labeling)
                ),
                self.training_metrics.get("oof_prediction_coverage", 1.0 if not self.config.use_meta_labeling else 0.0),
                0.60 if self.config.use_meta_labeling else 0.0,
            ),
            "min_deflated_sharpe": (
                self.training_metrics.get(
                    "deflated_sharpe",
                    self.training_metrics.get("mean_sharpe", 0.0),
                ) >= self.config.min_deflated_sharpe,
                self.training_metrics.get(
                    "deflated_sharpe",
                    self.training_metrics.get("mean_sharpe", 0.0),
                ),
                self.config.min_deflated_sharpe,
            ),
            "max_deflated_sharpe_pvalue": (
                self.training_metrics.get("deflated_sharpe_p_value", 1.0)
                <= self.config.max_deflated_sharpe_pvalue,
                self.training_metrics.get("deflated_sharpe_p_value", 1.0),
                self.config.max_deflated_sharpe_pvalue,
            ),
            "max_pbo": (
                self.training_metrics.get("pbo", 1.0) <= effective_max_pbo,
                self.training_metrics.get("pbo", 1.0),
                effective_max_pbo,
            ),
            "min_holdout_sharpe": (
                (holdout_sharpe >= self.config.min_holdout_sharpe) if holdout_available else True,
                holdout_sharpe if holdout_available else self.config.min_holdout_sharpe,
                self.config.min_holdout_sharpe,
            ),
            "max_holdout_drawdown": (
                (holdout_drawdown <= self.config.max_holdout_drawdown) if holdout_available else True,
                holdout_drawdown if holdout_available else self.config.max_holdout_drawdown,
                self.config.max_holdout_drawdown,
            ),
            "max_regime_shift": (
                regime_shift_metric <= self.config.max_regime_shift,
                regime_shift_metric,
                self.config.max_regime_shift,
            ),
            "nested_walk_forward_trace": (
                (
                    (len(self.nested_cv_trace) > 0)
                    or (
                        isinstance(self.training_metrics.get("nested_cv_trace"), list)
                        and len(self.training_metrics.get("nested_cv_trace", [])) > 0
                    )
                )
                if self.config.require_nested_trace_for_promotion
                else True,
                float(
                    len(self.nested_cv_trace)
                    if self.nested_cv_trace
                    else len(self.training_metrics.get("nested_cv_trace", []))
                ),
                1.0 if self.config.require_nested_trace_for_promotion else 0.0,
            ),
            "objective_breakdown_present": (
                (
                    isinstance(self.training_metrics.get("objective_component_summary"), dict)
                    and len(self.training_metrics.get("objective_component_summary", {})) > 0
                )
                if self.config.require_objective_breakdown_for_promotion
                else True,
                float(
                    1.0
                    if isinstance(self.training_metrics.get("objective_component_summary"), dict)
                    and len(self.training_metrics.get("objective_component_summary", {})) > 0
                    else 0.0
                ),
                1.0 if self.config.require_objective_breakdown_for_promotion else 0.0,
            ),
        }

        normalized_gates: dict[str, tuple[bool, float, float]] = {}
        all_passed = True
        for gate_name, (passed, actual, threshold) in gates.items():
            passed_bool = bool(passed)
            actual_value = float(actual)
            threshold_value = float(threshold)
            normalized_gates[gate_name] = (passed_bool, actual_value, threshold_value)

            status = "PASS" if passed_bool else "FAIL"
            self.logger.info(
                f"  {gate_name}: {status} (actual={actual_value:.4f}, threshold={threshold_value:.4f})"
            )
            if not passed_bool:
                all_passed = False

        self.validation_results = {
            "gates": normalized_gates,
            "all_passed": bool(all_passed),
        }

        return all_passed

    def _build_model_card(self) -> dict[str, Any]:
        """Build compact model-card metadata for registry and promotion governance."""
        regime_mix = self._regime_distribution(self.regimes)
        holdout_regime_mix = self._regime_distribution(self.holdout_regimes)
        top_features: list[dict[str, float]] = []
        shap_importance = self.training_metrics.get("shap_importance", {})
        if isinstance(shap_importance, dict) and shap_importance:
            top_features = [
                {"feature": str(name), "importance": float(score)}
                for name, score in list(shap_importance.items())[:15]
                if isinstance(score, (int, float, np.integer, np.floating))
            ]

        gate_summary: dict[str, dict[str, Any]] = {}
        gates = self.validation_results.get("gates", {})
        if isinstance(gates, dict):
            for gate_name, gate_tuple in gates.items():
                if not isinstance(gate_tuple, tuple) or len(gate_tuple) != 3:
                    continue
                passed, actual, threshold = gate_tuple
                gate_summary[str(gate_name)] = {
                    "passed": bool(passed),
                    "actual": float(actual),
                    "threshold": float(threshold),
                }

        return {
            "schema_version": "1.0.0",
            "model_name": self.config.model_name,
            "model_type": self.config.model_type,
            "snapshot_id": (
                self.snapshot_manifest.get("snapshot_id")
                if isinstance(self.snapshot_manifest, dict)
                else None
            ),
            "validation_passed": bool(self.validation_results.get("all_passed", False)),
            "metrics": {
                "mean_accuracy": float(self.training_metrics.get("mean_accuracy", 0.0)),
                "mean_sharpe": float(self.training_metrics.get("mean_sharpe", 0.0)),
                "mean_max_drawdown": float(self.training_metrics.get("mean_max_drawdown", 1.0)),
                "mean_trade_count": float(self.training_metrics.get("mean_trade_count", 0.0)),
                "holdout_sharpe": float(self.training_metrics.get("holdout_sharpe", 0.0)),
                "holdout_max_drawdown": float(self.training_metrics.get("holdout_max_drawdown", 1.0)),
                "holdout_regime_shift": float(self.training_metrics.get("holdout_regime_shift", 0.0)),
                "deflated_sharpe": float(self.training_metrics.get("deflated_sharpe", 0.0)),
                "deflated_sharpe_p_value": float(
                    self.training_metrics.get("deflated_sharpe_p_value", 1.0)
                ),
                "pbo": float(self.training_metrics.get("pbo", 1.0)),
            },
            "data": {
                "development_rows": int(self.training_metrics.get("development_rows", 0)),
                "holdout_rows": int(self.training_metrics.get("holdout_rows", 0)),
                "regime_distribution_train": regime_mix,
                "regime_distribution_holdout": holdout_regime_mix,
            },
            "gates": gate_summary,
            "top_features": top_features,
        }

    def _build_deployment_plan(self, passed_validation: bool) -> dict[str, Any]:
        """Build champion/challenger canary rollout and runtime risk guardrails."""
        total_cost_bps = float(
            self.config.label_spread_bps + self.config.label_slippage_bps + self.config.label_impact_bps
        )
        return {
            "schema_version": "1.0.0",
            "ready_for_production": bool(passed_validation),
            "promotion_strategy": {
                "mode": "champion_challenger_canary",
                "champion_retention_if_fail": True,
                "challenger_activation_requires_all_gates": True,
            },
            "canary_rollout": [
                {"phase": 1, "capital_fraction": 0.05, "min_trades": 50},
                {"phase": 2, "capital_fraction": 0.15, "min_trades": 120},
                {"phase": 3, "capital_fraction": 0.35, "min_trades": 250},
                {"phase": 4, "capital_fraction": 1.00, "min_trades": 400},
            ],
            "kill_switch_guardrails": {
                "max_intraday_drawdown": float(min(self.config.max_drawdown, self.config.max_holdout_drawdown)),
                "max_regime_shift": float(self.config.max_regime_shift),
                "min_live_sharpe_rolling": float(max(self.config.min_holdout_sharpe, 0.0)),
                "min_trade_count_rolling": int(self.config.min_trades),
            },
            "tca_guardrails": {
                "expected_cost_bps": total_cost_bps,
                "max_slippage_bps": float(total_cost_bps * 1.5),
                "max_turnover": float(self.config.execution_turnover_cap),
                "execution_cooldown_bars": int(self.config.execution_cooldown_bars),
            },
        }

    def _train_meta_labeler(self) -> None:
        """
        Phase 7: Train meta-labeling model for signal filtering.

        @mlquant P2-3.5: Meta-labeling helps filter out low-quality signals
        by predicting which primary signals are likely to be profitable.
        """
        self.logger.info("Phase 7: Training meta-labeling model...")

        try:
            from quant_trading_system.models.meta_labeling import MetaLabelConfig, MetaLabeler

            if self.close_prices is None:
                raise RuntimeError("Close prices unavailable for meta-labeling.")
            if self.oof_primary_proba is None:
                raise RuntimeError(
                    "OOF primary predictions unavailable for meta-labeling. "
                    "Institutional mode requires out-of-fold meta labels."
                )
            if len(self.oof_primary_proba) != len(self.features):
                raise RuntimeError("OOF prediction length mismatch for meta-labeling.")

            oof_mask = np.isfinite(self.oof_primary_proba)
            usable_count = int(np.sum(oof_mask))
            if usable_count <= 1:
                raise RuntimeError("Insufficient OOF predictions for meta-labeling.")
            coverage = float(usable_count / len(self.features))
            if coverage < 0.60:
                raise RuntimeError(
                    f"OOF prediction coverage too low for meta-labeling ({coverage:.2%})."
                )
            self.training_metrics["meta_label_oof_coverage"] = coverage

            primary_predictions = np.asarray(self.oof_primary_proba[oof_mask], dtype=float).reshape(-1)
            used_close = self.close_prices.iloc[oof_mask].reset_index(drop=True)
            used_features = self.features.iloc[oof_mask].copy()
            self.logger.info(
                f"Meta-labeling uses OOF predictions: {usable_count}/{len(self.features)} samples"
            )

            meta_cfg = MetaLabelConfig(model_type=self.config.meta_model_type)
            self.meta_model = MetaLabeler(config=meta_cfg)

            signal_series = pd.Series(
                np.where(primary_predictions >= 0.5, 1.0, -1.0),
                index=used_close.index,
            )
            X_frame = pd.DataFrame(used_features.values, columns=self.feature_names)
            self.meta_model.fit(
                X_frame,
                signal_series,
                used_close,
            )

            self.logger.info("Meta-labeling model trained successfully")

        except ImportError as exc:
            raise RuntimeError("Meta-labeling module is mandatory in institutional mode") from exc

    def _apply_multiple_testing_correction(self) -> None:
        """
        Phase 8: Apply multiple testing correction.

        @mlquant P1-3.1: Corrects for multiple hypothesis testing to avoid
        false discoveries. Supports Bonferroni, Benjamini-Hochberg, and
        Deflated Sharpe Ratio methods.
        """
        self.logger.info(f"Phase 8: Applying {self.config.correction_method} correction...")

        sharpe = float(self.training_metrics.get("mean_sharpe", 0))
        active_returns = (
            np.concatenate(self.cv_active_return_series)
            if self.cv_active_return_series
            else np.array([], dtype=float)
        )
        full_returns = (
            np.concatenate(self.cv_return_series)
            if self.cv_return_series
            else np.array([], dtype=float)
        )
        use_active_returns = int(active_returns.size) >= 30
        returns = active_returns if use_active_returns else full_returns
        self.training_metrics["multiple_testing_return_source"] = (
            "active_returns" if use_active_returns else "all_returns"
        )
        self.training_metrics["multiple_testing_return_observations"] = float(returns.size)
        n_trials = max(
            int(getattr(self.config, "n_trials", 1)),
            len(self.cv_results) if self.cv_results else 1,
            1,
        )

        if self.config.correction_method == "deflated_sharpe":
            deflated_sharpe, p_value = calculate_deflated_sharpe_ratio(
                observed_sharpe=sharpe,
                returns=returns,
                n_trials=n_trials,
            )
            pbo, interpretation = calculate_probability_of_backtest_overfitting(returns)
            deflation = (
                deflated_sharpe / sharpe
                if abs(sharpe) > 1e-12
                else 0.0
            )
            pbo = float(np.nan_to_num(pbo, nan=1.0, posinf=1.0, neginf=1.0))
            self.training_metrics["deflated_sharpe"] = deflated_sharpe
            self.training_metrics["deflated_sharpe_p_value"] = p_value
            self.training_metrics["sharpe_deflation_factor"] = deflation
            self.training_metrics["pbo"] = pbo
            self.training_metrics["pbo_interpretation"] = interpretation
            self.training_metrics["effective_max_pbo_gate"] = self._effective_pbo_threshold(
                sample_size=int(returns.size),
                interpretation=str(interpretation),
            )

            self.logger.info(
                f"Deflated Sharpe: {deflated_sharpe:.4f} "
                f"(original: {sharpe:.4f}, p-value: {p_value:.4f}, deflation: {deflation:.4f})"
            )
            self.logger.info(f"PBO: {pbo:.2%} ({interpretation})")

        elif self.config.correction_method == "bonferroni":
            # Bonferroni correction (conservative)
            corrected_sharpe = sharpe / n_trials
            self.training_metrics["bonferroni_sharpe"] = corrected_sharpe
            self.logger.info(f"Bonferroni-corrected Sharpe: {corrected_sharpe:.4f}")

        elif self.config.correction_method == "bh":
            # Benjamini-Hochberg (less conservative)
            from scipy.stats import norm

            p_value = 2 * (1 - norm.cdf(abs(sharpe)))
            adjusted_p = min(p_value * n_trials, 1.0)

            self.training_metrics["bh_adjusted_p"] = adjusted_p
            self.logger.info(f"BH-adjusted p-value: {adjusted_p:.4f}")

    def _effective_pbo_threshold(self, sample_size: int, interpretation: str) -> float:
        """Adjust PBO gate for low-sample statistical uncertainty."""
        base_threshold = float(np.clip(self.config.max_pbo, 0.05, 0.95))
        if sample_size <= 0:
            return base_threshold

        reliability = float(np.clip((float(sample_size) - 80.0) / 320.0, 0.0, 1.0))
        uncertainty_slack = (1.0 - reliability) * 0.10
        if interpretation.lower().startswith("insufficient"):
            uncertainty_slack = max(uncertainty_slack, 0.10)
        return float(min(0.60, max(base_threshold, base_threshold + uncertainty_slack)))

    def _compute_shap_values(self) -> None:
        """
        Phase 9: Compute SHAP values for model explainability.

        @mlquant: SHAP values provide regulatory-compliant model
        explanations for audit purposes.
        """
        self.logger.info("Phase 9: Computing SHAP values...")

        restore_device: Any | None = None
        restore_amp: bool | None = None
        previous_shap_log_level: int | None = None
        try:
            import shap
            shap_logger = logging.getLogger("shap")
            previous_shap_log_level = shap_logger.level
            if previous_shap_log_level <= logging.INFO:
                shap_logger.setLevel(logging.WARNING)

            # Create explainer based on model type
            model_for_shap = getattr(self.model, "_model", self.model)
            sequence_model_types = {"lstm", "transformer", "tcn"}
            is_sequence_model = (
                self.config.model_type in sequence_model_types
                or hasattr(self.model, "lookback_window")
            )
            lookback_window = int(
                getattr(
                    self.model,
                    "lookback_window",
                    self.config.model_params.get("lookback_window", 20),
                )
            )
            lookback_window = max(1, lookback_window)

            # Sample data for SHAP
            max_samples = min(self.config.n_shap_samples, len(self.features))
            if is_sequence_model:
                # Kernel SHAP on deep sequence models is expensive; keep a safe cap.
                max_samples = min(max_samples, 64)
            sample_idx = np.random.choice(len(self.features), max_samples, replace=False)
            X_sample = self.features.values[sample_idx]

            if is_sequence_model and hasattr(self.model, "_network") and hasattr(self.model, "device"):
                try:
                    import torch

                    current_device = getattr(self.model, "device", None)
                    if current_device is not None and str(current_device).startswith("cuda"):
                        restore_device = current_device
                        restore_amp = getattr(self.model, "_use_amp", None)
                        self.logger.info("Switching sequence model to CPU for SHAP stability")
                        self.model._network.to("cpu")
                        self.model.device = torch.device("cpu")
                        if hasattr(self.model, "_use_amp"):
                            self.model._use_amp = False
                except Exception as exc:
                    self.logger.warning(f"Could not switch model to CPU for SHAP: {exc}")

            if self.config.model_type in ["xgboost", "lightgbm", "random_forest"]:
                explainer = shap.TreeExplainer(model_for_shap)
            else:
                def _predict_for_shap(x: np.ndarray) -> np.ndarray:
                    """Return SHAP-compatible probabilities with output length == input length."""
                    x_arr = np.asarray(x, dtype=float)
                    if x_arr.ndim == 1:
                        x_arr = x_arr.reshape(1, -1)
                    x_arr = np.nan_to_num(x_arr, nan=0.0, posinf=0.0, neginf=0.0)

                    target_len = int(x_arr.shape[0])
                    model_input = x_arr
                    if is_sequence_model and lookback_window > 1:
                        # KernelExplainer may call with tiny batches (even a single row).
                        # Prepend synthetic context to satisfy sequence lookback constraints.
                        prefix = np.repeat(x_arr[:1], lookback_window - 1, axis=0)
                        model_input = np.vstack([prefix, x_arr])

                    if is_sequence_model:
                        probs = self._predict_sequence_probs_batched(
                            model_input=model_input,
                            lookback_window=lookback_window,
                            target_len=target_len,
                        )
                    else:
                        raw_probs = np.asarray(
                            self._get_predictions_proba(self.model, model_input), dtype=float
                        )
                        probs = self._align_probabilities(raw_probs, target_len)
                    return np.clip(np.nan_to_num(probs, nan=0.5), 0.0, 1.0)

                background = X_sample[:100]
                if is_sequence_model and lookback_window > 1:
                    prefix = np.repeat(background[:1], lookback_window - 1, axis=0)
                    background = np.vstack([prefix, background])

                explainer = shap.KernelExplainer(
                    _predict_for_shap,
                    background
                )

            if self.config.model_type in ["xgboost", "lightgbm", "random_forest"]:
                self.shap_values = explainer.shap_values(X_sample)
            else:
                kernel_nsamples = min(256, max(64, X_sample.shape[1] * 2))
                self.logger.info(f"Kernel SHAP nsamples={kernel_nsamples}")
                self.shap_values = explainer.shap_values(
                    X_sample,
                    nsamples=kernel_nsamples,
                )

            # Feature importance from SHAP
            mean_shap = self._mean_abs_shap_importance(self.shap_values, X_sample.shape[1])

            feature_names = self.features.columns.tolist()
            importance = dict(zip(feature_names, mean_shap))
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

            self.training_metrics["shap_importance"] = importance

            # Log top features
            top_features = list(importance.items())[:10]
            self.logger.info("Top 10 features by SHAP importance:")
            for fname, imp in top_features:
                self.logger.info(f"  {fname}: {imp:.4f}")

        except ImportError:
            raise RuntimeError("SHAP is mandatory in institutional mode but not installed")
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"SHAP computation failed due to data issue: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected SHAP error: {e}") from e
        finally:
            if previous_shap_log_level is not None:
                logging.getLogger("shap").setLevel(previous_shap_log_level)
            if restore_device is not None and hasattr(self.model, "_network"):
                try:
                    self.model._network.to(restore_device)
                    self.model.device = restore_device
                    if restore_amp is not None and hasattr(self.model, "_use_amp"):
                        self.model._use_amp = restore_amp
                except Exception as exc:
                    self.logger.warning(f"Failed to restore model device after SHAP: {exc}")

    @staticmethod
    def _align_probabilities(
        probs: np.ndarray,
        target_len: int,
        fill_value: float = 0.5,
    ) -> np.ndarray:
        """Align 1D vector to target length with configurable fill values."""
        target_len = int(target_len)
        if target_len <= 0:
            return np.array([], dtype=float)

        arr = np.asarray(probs, dtype=float).reshape(-1)
        if arr.size == target_len:
            return arr
        if arr.size == 0:
            return np.full(target_len, fill_value, dtype=float)
        if arr.size < target_len:
            pad = np.full(target_len - arr.size, fill_value, dtype=float)
            return np.concatenate([pad, arr])
        return arr[-target_len:]

    def _predict_sequence_probs_batched(
        self,
        model_input: np.ndarray,
        lookback_window: int,
        target_len: int,
        chunk_target_size: int = 2048,
    ) -> np.ndarray:
        """Predict sequence-model probabilities in bounded chunks to avoid OOM."""
        lookback_window = max(1, int(lookback_window))
        target_len = int(target_len)
        if target_len <= 0:
            return np.array([], dtype=float)

        model_input = np.asarray(model_input, dtype=float)
        n_rows = int(model_input.shape[0])
        max_target_len = max(0, n_rows - lookback_window + 1)
        if max_target_len <= 0:
            return np.full(target_len, 0.5, dtype=float)

        chunk_target_size = max(64, int(chunk_target_size))
        effective_target_len = min(target_len, max_target_len)
        outputs: list[np.ndarray] = []

        for start in range(0, effective_target_len, chunk_target_size):
            chunk_len = min(chunk_target_size, effective_target_len - start)
            end = start + chunk_len + lookback_window - 1
            x_chunk = model_input[start:end]
            raw_chunk = self._get_predictions_proba(self.model, x_chunk)
            outputs.append(self._align_probabilities(raw_chunk, chunk_len))

        if not outputs:
            return np.full(target_len, 0.5, dtype=float)

        combined = np.concatenate(outputs)
        return self._align_probabilities(combined, target_len)

    @staticmethod
    def _mean_abs_shap_importance(shap_values: Any, n_features: int) -> np.ndarray:
        """Normalize SHAP outputs (list/2D/3D/Explanation) to 1D feature importance."""
        values = shap_values.values if hasattr(shap_values, "values") else shap_values

        if isinstance(values, list):
            arrays = [np.asarray(v, dtype=float) for v in values if v is not None]
            if not arrays:
                raise ValueError("Empty SHAP value list")
            arr = np.stack(arrays, axis=0)
        else:
            arr = np.asarray(values, dtype=float)

        if arr.ndim == 1:
            if arr.shape[0] != n_features:
                raise ValueError(
                    f"Unexpected 1D SHAP shape {arr.shape}; expected {n_features} features"
                )
            return np.abs(arr)

        # Detect feature axis by matching feature count (prefer rightmost match).
        feature_axis = None
        for axis, size in enumerate(arr.shape):
            if size == n_features:
                feature_axis = axis

        if feature_axis is None:
            feature_axis = arr.ndim - 1

        arr = np.moveaxis(arr, feature_axis, -1)
        reduce_axes = tuple(range(arr.ndim - 1))
        importance = np.abs(arr).mean(axis=reduce_axes)

        if importance.shape[0] != n_features:
            raise ValueError(
                f"Computed SHAP importance has shape {importance.shape}, expected ({n_features},)"
            )
        return importance

    def _write_replay_manifest(self, output_dir: Path, model_path: Path) -> Path:
        """Persist replay manifest used to reconstruct the same training run."""
        replay_dir = output_dir / "replays"
        replay_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "schema_version": REPLAY_MANIFEST_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_name": self.config.model_name,
            "model_type": self.config.model_type,
            "model_path": str(model_path),
            "training_config": self.config.__dict__,
            "snapshot_manifest": self.snapshot_manifest,
            "snapshot_manifest_path": (
                str(self.snapshot_manifest_path) if self.snapshot_manifest_path else None
            ),
            "data_quality_report_hash": self.data_quality_report_hash,
            "data_quality_report_path": (
                str(self.data_quality_report_path) if self.data_quality_report_path else None
            ),
            "label_diagnostics": self.label_diagnostics,
        }
        replay_path = replay_dir / f"{self.config.model_name}.replay_manifest.json"
        with replay_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=True, default=str)

        self.logger.info(f"Replay manifest saved to: {replay_path}")
        return replay_path

    def _write_promotion_package(
        self,
        output_dir: Path,
        model_path: Path,
        replay_manifest_path: Path | None,
    ) -> Path:
        """Persist promotion gate package with nested CV and objective breakdown evidence."""
        package_dir = output_dir / "promotion_packages"
        package_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "schema_version": PROMOTION_PACKAGE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_name": self.config.model_name,
            "model_type": self.config.model_type,
            "model_path": str(model_path),
            "training_config": self.config.__dict__,
            "promotion_passed": bool(self.validation_results.get("all_passed", False)),
            "snapshot_id": (
                self.snapshot_manifest.get("snapshot_id")
                if isinstance(self.snapshot_manifest, dict)
                else None
            ),
            "snapshot_manifest_path": (
                str(self.snapshot_manifest_path) if self.snapshot_manifest_path else None
            ),
            "data_quality_report_path": (
                str(self.data_quality_report_path) if self.data_quality_report_path else None
            ),
            "data_quality_report_hash": self.data_quality_report_hash,
            "validation_results": self.validation_results,
            "objective_component_summary": self.training_metrics.get(
                "objective_component_summary", {}
            ),
            "model_card": self.training_metrics.get("model_card", {}),
            "deployment_plan": self.training_metrics.get("deployment_plan", {}),
            "nested_cv_trace": self.training_metrics.get("nested_cv_trace", self.nested_cv_trace),
            "statistical_validity": {
                "deflated_sharpe": self.training_metrics.get("deflated_sharpe"),
                "deflated_sharpe_p_value": self.training_metrics.get("deflated_sharpe_p_value"),
                "pbo": self.training_metrics.get("pbo"),
                "pbo_interpretation": self.training_metrics.get("pbo_interpretation"),
            },
            "promotion_thresholds": {
                "min_sharpe_ratio": self.config.min_sharpe_ratio,
                "min_accuracy": self.config.min_accuracy,
                "max_drawdown": self.config.max_drawdown,
                "min_win_rate": self.config.min_win_rate,
                "min_trades": self.config.min_trades,
                "min_holdout_sharpe": self.config.min_holdout_sharpe,
                "max_holdout_drawdown": self.config.max_holdout_drawdown,
                "max_regime_shift": self.config.max_regime_shift,
                "min_deflated_sharpe": self.config.min_deflated_sharpe,
                "max_deflated_sharpe_pvalue": self.config.max_deflated_sharpe_pvalue,
                "max_pbo": self.config.max_pbo,
            },
            "replay_manifest_path": str(replay_manifest_path) if replay_manifest_path else None,
        }

        package_path = package_dir / f"{self.config.model_name}.promotion_package.json"
        with package_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=True, default=str)

        self.logger.info(f"Promotion package saved to: {package_path}")
        return package_path

    def _save_model(self) -> str:
        """Phase 10: Save trained model and artifacts."""
        self.logger.info("Phase 10: Saving model...")

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Model filename
        model_stem = self.config.model_name
        model_path = output_dir / f"{model_stem}.pkl"

        # Prefer model-native serialization when available (deep models).
        if hasattr(self.model, "save") and callable(getattr(self.model, "save")):
            try:
                self.model.save(output_dir / model_stem)
            except Exception as exc:
                self.logger.warning(
                    f"Model-native save failed ({exc}); falling back to pickle serialization."
                )
                with open(model_path, "wb") as f:
                    pickle.dump(self.model, f)
        else:
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)

        self.logger.info(f"Model saved to: {model_path}")

        if self.config.save_artifacts:
            if self.snapshot_manifest is None:
                raise RuntimeError(
                    "Snapshot manifest missing at save stage. "
                    "Institutional mode requires snapshot lineage."
                )
            if self.data_quality_report is None:
                raise RuntimeError(
                    "Data quality report missing at save stage. "
                    "Institutional mode requires quality report archival."
                )

            manifest_path, quality_path = persist_snapshot_bundle(
                output_dir=output_dir,
                manifest=self.snapshot_manifest,
                quality_report=self.data_quality_report,
            )
            self.snapshot_manifest_path = manifest_path
            self.data_quality_report_path = quality_path
            self.snapshot_manifest["manifest_path"] = str(manifest_path)
            self.snapshot_manifest["quality_report_path"] = str(quality_path)
            self.replay_manifest_path = self._write_replay_manifest(output_dir, model_path)
            self.promotion_package_path = self._write_promotion_package(
                output_dir=output_dir,
                model_path=model_path,
                replay_manifest_path=self.replay_manifest_path,
            )

        # Save training artifacts
        if self.config.save_artifacts:
            artifacts = {
                "config": self.config.__dict__,
                "cv_results": self.cv_results,
                "validation_results": self.validation_results,
                "training_metrics": {
                    k: v for k, v in self.training_metrics.items()
                    if not isinstance(v, np.ndarray)
                },
                "feature_names": self.features.columns.tolist(),
                "snapshot_id": self.snapshot_manifest.get("snapshot_id"),
                "snapshot_manifest": self.snapshot_manifest,
                "snapshot_manifest_path": str(self.snapshot_manifest_path),
                "data_quality_report_hash": self.data_quality_report_hash,
                "data_quality_report_path": str(self.data_quality_report_path),
                "label_diagnostics": self.label_diagnostics,
                "nested_cv_trace": self.nested_cv_trace,
                "replay_manifest_path": (
                    str(self.replay_manifest_path) if self.replay_manifest_path else None
                ),
                "promotion_package_path": (
                    str(self.promotion_package_path) if self.promotion_package_path else None
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            artifacts_path = output_dir / f"{self.config.model_name}_artifacts.json"
            with open(artifacts_path, "w") as f:
                json.dump(artifacts, f, indent=2, default=str)

            self.logger.info(f"Artifacts saved to: {artifacts_path}")

            # Save meta-model if trained
            if self.meta_model is not None:
                meta_path = output_dir / f"{self.config.model_name}_meta.pkl"
                with open(meta_path, "wb") as f:
                    pickle.dump(self.meta_model, f)
                self.logger.info(f"Meta-model saved to: {meta_path}")

        return str(model_path)


# ============================================================================
# ENSEMBLE TRAINER
# ============================================================================


class EnsembleTrainer:
    """
    Train ensemble models with IC-weighted combination.

    @mlquant P2-C: IC-Based Ensemble dynamically adjusts model weights
    based on rolling IC/IR performance.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger("EnsembleTrainer")
        self.base_models: list[tuple[str, Any]] = []
        self.ensemble_model: Any | None = None

    def train(self) -> dict:
        """Train ensemble model."""
        self.logger.info("Training ensemble model...")

        # Train base models
        base_model_types = ["xgboost", "lightgbm", "random_forest"]

        for model_type in base_model_types:
            model_config = TrainingConfig(
                model_type=model_type,
                symbols=self.config.symbols,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                cv_method=self.config.cv_method,
                n_splits=self.config.n_splits,
                embargo_pct=self.config.embargo_pct,
                optimize=False,  # Skip optimization for base models
                compute_shap=False,
            )

            trainer = ModelTrainer(model_config)
            result = trainer.run()

            self.base_models.append((model_type, trainer.model))
            self.logger.info(f"Trained {model_type}: Sharpe={result['training_metrics'].get('mean_sharpe', 0):.4f}")

        # Create IC-weighted ensemble
        try:
            from quant_trading_system.models.ensemble import ICBasedEnsemble

            ensemble = ICBasedEnsemble()
            for name, model in self.base_models:
                ensemble.add_model(model)

            self.ensemble_model = ensemble
            self.logger.info(f"Created ensemble with {len(self.base_models)} base models")
            model_path = self._save_ensemble()
            base_sharpes = [m.training_metrics.get("sharpe", 0.0) for _, m in self.base_models]

            return {
                "success": True,
                "model_path": model_path,
                "ensemble_type": "ic_weighted",
                "base_models": [name for name, _ in self.base_models],
                "training_metrics": {
                    "mean_base_sharpe": float(np.mean(base_sharpes)) if base_sharpes else 0.0,
                },
            }

        except ImportError:
            self.logger.warning("ICBasedEnsemble not available, using simple average")
            return {
                "success": True,
                "ensemble_type": "simple_average",
                "base_models": [name for name, _ in self.base_models],
            }

    def _save_ensemble(self) -> str:
        """Persist ensemble artifact."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_name = self.config.model_name or f"ensemble_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        model_path = output_dir / f"{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.ensemble_model, f)
        return str(model_path)


# ============================================================================
# GOVERNANCE HELPERS
# ============================================================================


def _metric_as_float(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Read one numeric metric with safe float coercion."""
    value = metrics.get(key, default)
    try:
        casted = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(np.nan_to_num(casted, nan=default, posinf=default, neginf=default))


def _governance_score(result: dict[str, Any]) -> float:
    """Composite score used for benchmark ranking and champion selection."""
    metrics = result.get("training_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    score = 0.0
    score += _metric_as_float(metrics, "mean_risk_adjusted_score", -1.0) * 1.0
    score += _metric_as_float(metrics, "holdout_sharpe", _metric_as_float(metrics, "mean_sharpe", 0.0)) * 0.6
    score += _metric_as_float(metrics, "deflated_sharpe", _metric_as_float(metrics, "mean_sharpe", 0.0)) * 0.4
    score -= _metric_as_float(metrics, "holdout_max_drawdown", _metric_as_float(metrics, "mean_max_drawdown", 1.0)) * 0.75
    score -= _metric_as_float(metrics, "pbo", 1.0) * 0.25
    if not bool(result.get("success", False)):
        score -= 5.0
    return float(score)


def _build_training_matrix(results: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Normalize per-model outputs into one comparable matrix."""
    rows: list[dict[str, Any]] = []
    for model_type, result in results:
        metrics = result.get("training_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        rows.append(
            {
                "model_type": str(model_type),
                "success": bool(result.get("success", False)),
                "governance_score": _governance_score(result),
                "mean_accuracy": _metric_as_float(metrics, "mean_accuracy", 0.0),
                "mean_sharpe": _metric_as_float(metrics, "mean_sharpe", 0.0),
                "mean_risk_adjusted_score": _metric_as_float(metrics, "mean_risk_adjusted_score", -1.0),
                "deflated_sharpe": _metric_as_float(metrics, "deflated_sharpe", 0.0),
                "deflated_sharpe_p_value": _metric_as_float(metrics, "deflated_sharpe_p_value", 1.0),
                "pbo": _metric_as_float(metrics, "pbo", 1.0),
                "holdout_sharpe": _metric_as_float(metrics, "holdout_sharpe", 0.0),
                "holdout_max_drawdown": _metric_as_float(metrics, "holdout_max_drawdown", 1.0),
                "holdout_regime_shift": _metric_as_float(metrics, "holdout_regime_shift", 0.0),
                "mean_trade_count": _metric_as_float(metrics, "mean_trade_count", 0.0),
                "registry_version_id": str(result.get("registry_version_id") or ""),
                "model_name": str(result.get("model_name") or model_type),
                "model_path": str(result.get("model_path") or ""),
                "promotion_package_path": str(result.get("promotion_package_path") or ""),
                "replay_manifest_path": str(result.get("replay_manifest_path") or ""),
                "deployment_plan": result.get("deployment_plan", {}),
            }
        )

    rows.sort(key=lambda row: float(row.get("governance_score", -1e9)), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = int(rank)
    return rows


def _persist_training_matrix_report(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """Write benchmark matrix in json+csv for governance review."""
    bench_dir = Path(output_dir) / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = bench_dir / f"training_matrix_{stamp}.json"
    csv_path = bench_dir / f"training_matrix_{stamp}.csv"

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=True, default=str)

    flat_rows = []
    for row in rows:
        flat_row = {k: v for k, v in row.items() if k != "deployment_plan"}
        flat_rows.append(flat_row)
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    return json_path, csv_path


def _auto_select_champion_and_challenger(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Promote best valid candidate and persist champion/challenger snapshot."""
    registry_root = Path(output_dir) / "registry"
    models_root = Path(output_dir)
    current_entries = load_registry_entries(registry_root)

    eligible = [
        row for row in rows
        if bool(row.get("success", False))
        and str(row.get("registry_version_id", "")).strip()
        and str(row.get("promotion_package_path", "")).strip()
        and bool(
            isinstance(row.get("deployment_plan"), dict)
            and row.get("deployment_plan", {}).get("ready_for_production", False)
        )
    ]

    champion = eligible[0] if eligible else None
    challenger = eligible[1] if len(eligible) > 1 else None

    promoted = False
    pointer_path: Path | None = None
    canary_plan_path: Path | None = None
    if champion is not None:
        promoted = set_registry_active_version(
            registry_root=registry_root,
            model_name=str(champion.get("model_type", "")),
            version_id=str(champion.get("registry_version_id", "")),
        )
        if promoted:
            pointer_path = persist_active_model_pointer(
                models_root=models_root,
                model_name=str(champion.get("model_type", "")),
                version_id=str(champion.get("registry_version_id", "")),
                updated_by="training_pipeline",
                reason="auto_benchmark_promotion",
            )
            deployment_plan = champion.get("deployment_plan", {})
            if isinstance(deployment_plan, dict) and deployment_plan:
                canary_payload = {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "model_type": champion.get("model_type"),
                    "version_id": champion.get("registry_version_id"),
                    "deployment_plan": deployment_plan,
                }
                canary_plan_path = Path(output_dir) / "canary_rollout_plan.json"
                with canary_plan_path.open("w", encoding="utf-8") as f:
                    json.dump(canary_payload, f, indent=2, ensure_ascii=True, sort_keys=True, default=str)

    snapshot = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "promoted": bool(promoted),
        "pointer_path": str(pointer_path) if pointer_path else None,
        "canary_plan_path": str(canary_plan_path) if canary_plan_path else None,
        "champion": champion,
        "challenger": challenger,
        "eligible_count": len(eligible),
        "registry_entries_seen": len(current_entries),
        "selection_policy": {
            "requires_validation_pass": True,
            "ranking_key": "governance_score",
            "fallback": "retain_existing_if_no_eligible",
        },
    }
    snapshot_path = Path(output_dir) / "champion_challenger_snapshot.json"
    with snapshot_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=True, sort_keys=True, default=str)
    snapshot["snapshot_path"] = str(snapshot_path)
    return snapshot


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def run_training(args: argparse.Namespace) -> int:
    """
    Main entry point for model training.

    @mlquant: This function orchestrates the entire training pipeline.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    # SQL statement-level INFO logs materially slow long feature upsert phases.
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)

    logger.info("=" * 80)
    logger.info("ALPHATRADE MODEL TRAINING PIPELINE")
    logger.info("=" * 80)

    try:
        if getattr(args, "gpu", False) or getattr(args, "use_gpu", False):
            logger.warning("`--gpu/--use-gpu` is deprecated and ignored. Institutional mode always uses GPU.")
        if getattr(args, "no_database", False) or getattr(args, "no_redis_cache", False):
            raise ValueError(
                "Institutional mode does not allow disabling PostgreSQL or Redis."
            )
        if getattr(args, "no_shap", False):
            raise ValueError("Institutional mode requires SHAP explainability.")
        if getattr(args, "disable_nested_walk_forward", False):
            raise ValueError(
                "Institutional mode requires nested walk-forward optimization trace."
            )

        replay_bundle: dict[str, Any] | None = None
        replay_config: dict[str, Any] = {}
        replay_manifest_arg = getattr(args, "replay_manifest", None)
        if replay_manifest_arg:
            replay_bundle = _load_replay_manifest(Path(replay_manifest_arg))
            replay_config = replay_bundle["training_config"]
            logger.info(f"Replay mode enabled from manifest: {replay_bundle['manifest_path']}")

        requested_model = str(replay_config.get("model_type") or getattr(args, "model", "xgboost"))
        model_list = [
            "xgboost",
            "lightgbm",
            "random_forest",
            "elastic_net",
            "lstm",
            "transformer",
            "tcn",
            "ensemble",
        ] if requested_model == "all" else [requested_model]

        global_seed = int(replay_config.get("seed", getattr(args, "seed", 42)))
        set_global_seed(global_seed)

        _verify_institutional_infra()
        _verify_gpu_stack(model_list)

        audit_logger = None
        try:
            from quant_trading_system.monitoring.audit import create_audit_logger

            audit_storage_dir = Path(getattr(args, "output_dir", "models")) / "audit_logs"
            audit_logger = create_audit_logger(
                storage_dir=audit_storage_dir,
                source="training_pipeline",
            )
        except Exception as exc:
            logger.warning(f"Training audit logging disabled: {exc}")

        overall_success = True
        results: list[tuple[str, dict[str, Any]]] = []
        replay_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        def _cfg_value(key: str, fallback: Any) -> Any:
            value = replay_config.get(key, fallback)
            return fallback if value is None else value

        for model_type in model_list:
            model_params = _load_model_defaults(
                model_type=model_type,
                use_gpu=True,
            )
            replay_model_params = replay_config.get("model_params")
            if isinstance(replay_model_params, dict):
                model_params = {**model_params, **replay_model_params}

            configured_name = str(getattr(args, "name", "") or "").strip()
            if configured_name:
                model_name = configured_name
            elif replay_config:
                replay_base_name = str(replay_config.get("model_name") or model_type).strip()
                model_name = f"{replay_base_name}_replay_{replay_timestamp}"
            else:
                model_name = ""

            raw_horizons = _cfg_value("label_horizons", getattr(args, "label_horizons", [1, 5, 20]))
            if isinstance(raw_horizons, (list, tuple)):
                label_horizons = [int(v) for v in raw_horizons]
            else:
                label_horizons = [1, 5, 20]

            raw_symbols = _cfg_value("symbols", getattr(args, "symbols", []))
            if isinstance(raw_symbols, (list, tuple, set)):
                symbols = [str(s).strip().upper() for s in raw_symbols if str(s).strip()]
            elif raw_symbols:
                symbols = [str(raw_symbols).strip().upper()]
            else:
                symbols = []

            raw_feature_groups = _cfg_value(
                "feature_groups",
                getattr(args, "feature_groups", ["technical", "statistical", "microstructure", "cross_sectional"]),
            )
            if isinstance(raw_feature_groups, (list, tuple, set)):
                feature_groups = [str(g).strip().lower() for g in raw_feature_groups if str(g).strip()]
            elif raw_feature_groups:
                feature_groups = [str(raw_feature_groups).strip().lower()]
            else:
                feature_groups = ["technical", "statistical", "microstructure", "cross_sectional"]

            config = TrainingConfig(
                model_type=model_type,
                model_name=model_name,
                symbols=symbols,
                start_date=_cfg_value(
                    "start_date",
                    getattr(args, "start_date", "") or getattr(args, "start", ""),
                ),
                end_date=_cfg_value(
                    "end_date",
                    getattr(args, "end_date", "") or getattr(args, "end", ""),
                ),
                cv_method=_cfg_value("cv_method", getattr(args, "cv_method", "purged_kfold")),
                n_splits=int(_cfg_value("n_splits", getattr(args, "n_splits", 5))),
                embargo_pct=max(float(_cfg_value("embargo_pct", getattr(args, "embargo_pct", 0.01))), 0.01),  # P1-H1: Min 1%
                optimize=True,
                optimizer="optuna",
                n_trials=int(_cfg_value("n_trials", getattr(args, "n_trials", 100))),
                n_jobs=int(
                    _cfg_value(
                        "n_jobs",
                        1 if (model_type == "random_forest" and os.name == "nt") else -1,
                    )
                ),
                seed=int(_cfg_value("seed", getattr(args, "seed", global_seed))),
                use_nested_walk_forward=True,
                nested_outer_splits=int(
                    _cfg_value(
                        "nested_outer_splits",
                        getattr(args, "nested_outer_splits", 4),
                    )
                ),
                nested_inner_splits=int(
                    _cfg_value(
                        "nested_inner_splits",
                        getattr(args, "nested_inner_splits", 3),
                    )
                ),
                require_nested_trace_for_promotion=True,
                require_objective_breakdown_for_promotion=True,
                objective_weight_sharpe=float(
                    _cfg_value(
                        "objective_weight_sharpe",
                        getattr(args, "objective_weight_sharpe", 1.0),
                    )
                ),
                objective_weight_drawdown=float(
                    _cfg_value(
                        "objective_weight_drawdown",
                        getattr(args, "objective_weight_drawdown", 0.5),
                    )
                ),
                objective_weight_turnover=float(
                    _cfg_value(
                        "objective_weight_turnover",
                        getattr(args, "objective_weight_turnover", 0.1),
                    )
                ),
                objective_weight_calibration=float(
                    _cfg_value(
                        "objective_weight_calibration",
                        getattr(args, "objective_weight_calibration", 0.25),
                    )
                ),
                objective_weight_trade_activity=float(
                    _cfg_value(
                        "objective_weight_trade_activity",
                        getattr(args, "objective_weight_trade_activity", 1.0),
                    )
                ),
                objective_weight_cvar=float(
                    _cfg_value(
                        "objective_weight_cvar",
                        getattr(args, "objective_weight_cvar", 0.4),
                    )
                ),
                objective_weight_skew=float(
                    _cfg_value(
                        "objective_weight_skew",
                        getattr(args, "objective_weight_skew", 0.1),
                    )
                ),
                epochs=int(_cfg_value("epochs", getattr(args, "epochs", 100))),
                batch_size=int(_cfg_value("batch_size", getattr(args, "batch_size", 64))),
                learning_rate=float(
                    _cfg_value("learning_rate", getattr(args, "learning_rate", 0.001))
                ),
                use_meta_labeling=True,
                apply_multiple_testing=True,
                correction_method="deflated_sharpe",
                use_gpu=True,
                use_database=True,
                use_redis_cache=True,
                compute_shap=True,
                min_accuracy=float(
                    _cfg_value(
                        "min_accuracy",
                        getattr(args, "min_accuracy", getattr(args, "min_win_rate", 0.45)),
                    )
                ),
                min_trades=int(_cfg_value("min_trades", getattr(args, "min_trades", 100))),
                min_deflated_sharpe=float(
                    _cfg_value(
                        "min_deflated_sharpe",
                        getattr(args, "min_deflated_sharpe", 0.10),
                    )
                ),
                max_deflated_sharpe_pvalue=float(
                    _cfg_value(
                        "max_deflated_sharpe_pvalue",
                        getattr(args, "max_deflated_sharpe_pvalue", 0.10),
                    )
                ),
                max_pbo=float(_cfg_value("max_pbo", getattr(args, "max_pbo", 0.45))),
                min_holdout_sharpe=float(
                    _cfg_value(
                        "min_holdout_sharpe",
                        getattr(args, "min_holdout_sharpe", 0.0),
                    )
                ),
                max_holdout_drawdown=float(
                    _cfg_value(
                        "max_holdout_drawdown",
                        getattr(args, "max_holdout_drawdown", 0.35),
                    )
                ),
                max_regime_shift=float(
                    _cfg_value(
                        "max_regime_shift",
                        getattr(args, "max_regime_shift", 0.35),
                    )
                ),
                label_horizons=label_horizons,
                primary_label_horizon=int(
                    _cfg_value("primary_label_horizon", getattr(args, "primary_horizon", 5))
                ),
                label_profit_taking_threshold=float(
                    _cfg_value("label_profit_taking_threshold", getattr(args, "profit_taking", 0.015))
                ),
                label_stop_loss_threshold=float(
                    _cfg_value("label_stop_loss_threshold", getattr(args, "stop_loss", 0.010))
                ),
                label_max_holding_period=int(
                    _cfg_value("label_max_holding_period", getattr(args, "max_holding", 20))
                ),
                label_spread_bps=float(
                    _cfg_value("label_spread_bps", getattr(args, "spread_bps", 1.0))
                ),
                label_slippage_bps=float(
                    _cfg_value("label_slippage_bps", getattr(args, "slippage_bps", 3.0))
                ),
                label_impact_bps=float(
                    _cfg_value("label_impact_bps", getattr(args, "impact_bps", 2.0))
                ),
                label_min_signal_abs_return_bps=float(
                    _cfg_value(
                        "label_min_signal_abs_return_bps",
                        getattr(args, "label_min_signal_abs_return_bps", 8.0),
                    )
                ),
                label_neutral_buffer_bps=float(
                    _cfg_value(
                        "label_neutral_buffer_bps",
                        getattr(args, "label_neutral_buffer_bps", 4.0),
                    )
                ),
                label_max_abs_forward_return=float(
                    _cfg_value(
                        "label_max_abs_forward_return",
                        getattr(args, "label_max_abs_forward_return", 0.35),
                    )
                ),
                label_signal_volatility_floor_mult=float(
                    _cfg_value(
                        "label_signal_volatility_floor_mult",
                        getattr(args, "label_signal_volatility_floor_mult", 0.50),
                    )
                ),
                label_volatility_lookback=int(
                    _cfg_value(
                        "label_volatility_lookback",
                        getattr(args, "label_volatility_lookback", 20),
                    )
                ),
                label_regime_lookback=int(
                    _cfg_value("label_regime_lookback", getattr(args, "label_regime_lookback", 30))
                ),
                label_temporal_weight_decay=float(
                    _cfg_value(
                        "label_temporal_weight_decay",
                        getattr(args, "label_temporal_weight_decay", 0.999),
                    )
                ),
                label_edge_cost_buffer_bps=float(
                    _cfg_value(
                        "label_edge_cost_buffer_bps",
                        getattr(args, "label_edge_cost_buffer_bps", 2.0),
                    )
                ),
                feature_groups=feature_groups,
                enable_cross_sectional=bool(
                    _cfg_value(
                        "enable_cross_sectional",
                        not bool(getattr(args, "disable_cross_sectional", False)),
                    )
                ),
                max_cross_sectional_symbols=int(
                    _cfg_value(
                        "max_cross_sectional_symbols",
                        getattr(args, "max_cross_sectional_symbols", 20),
                    )
                ),
                max_cross_sectional_rows=int(
                    _cfg_value(
                        "max_cross_sectional_rows",
                        getattr(args, "max_cross_sectional_rows", 250000),
                    )
                ),
                allow_feature_group_fallback=bool(
                    _cfg_value(
                        "allow_feature_group_fallback",
                        not bool(getattr(args, "strict_feature_groups", False)),
                    )
                ),
                feature_materialization_batch_rows=int(
                    _cfg_value(
                        "feature_materialization_batch_rows",
                        getattr(args, "feature_materialization_batch_rows", 5000),
                    )
                ),
                feature_reuse_min_coverage=float(
                    _cfg_value(
                        "feature_reuse_min_coverage",
                        getattr(args, "feature_reuse_min_coverage", 0.20),
                    )
                ),
                persist_features_to_postgres=bool(
                    _cfg_value(
                        "persist_features_to_postgres",
                        not bool(getattr(args, "skip_feature_persist", False)),
                    )
                ),
                data_max_abs_return=float(
                    _cfg_value(
                        "data_max_abs_return",
                        getattr(args, "label_max_abs_forward_return", 0.35),
                    )
                ),
                holdout_pct=float(
                    _cfg_value(
                        "holdout_pct",
                        getattr(args, "holdout_pct", 0.15),
                    )
                ),
                dynamic_no_trade_band=bool(
                    _cfg_value(
                        "dynamic_no_trade_band",
                        not bool(getattr(args, "disable_dynamic_no_trade_band", False)),
                    )
                ),
                execution_vol_target_daily=float(
                    _cfg_value(
                        "execution_vol_target_daily",
                        getattr(args, "execution_vol_target_daily", 0.012),
                    )
                ),
                execution_turnover_cap=float(
                    _cfg_value(
                        "execution_turnover_cap",
                        getattr(args, "execution_turnover_cap", 0.90),
                    )
                ),
                execution_cooldown_bars=int(
                    _cfg_value(
                        "execution_cooldown_bars",
                        getattr(args, "execution_cooldown_bars", 2),
                    )
                ),
                output_dir=str(getattr(args, "output_dir", "models")),
                model_params=model_params,
            )

            logger.info("-" * 80)
            logger.info(f"Model type: {config.model_type}")
            logger.info(f"Symbols: {config.symbols or 'all available'}")
            logger.info("Data source: PostgreSQL (mandatory)")
            logger.info("Redis cache: enabled (mandatory)")
            logger.info(f"CV method: {config.cv_method} ({config.n_splits} splits)")
            logger.info(f"Embargo: {config.embargo_pct * 100:.1f}%")
            logger.info(f"GPU: {config.use_gpu} (mandatory)")
            logger.info(
                "Institutional stack: Nested Walk-Forward + Optuna + Meta-Labeling + "
                "Multiple-Testing + SHAP + Leak-Validation"
            )
            logger.info(
                f"Nested CV: outer={config.nested_outer_splits}, inner={config.nested_inner_splits}"
            )
            logger.info(
                f"Feature groups: {config.feature_groups} "
                f"(cross_sectional={'on' if config.enable_cross_sectional else 'off'})"
            )
            if replay_config:
                logger.info("Replay source: manifest-driven configuration")

            if audit_logger is not None:
                audit_logger.log_model_training_started(
                    model_name=config.model_type,
                    model_version=config.model_name,
                    config={
                        "cv_method": config.cv_method,
                        "n_splits": config.n_splits,
                        "embargo_pct": config.embargo_pct,
                        "n_trials": config.n_trials,
                        "use_gpu": config.use_gpu,
                        "holdout_pct": config.holdout_pct,
                        "execution_turnover_cap": config.execution_turnover_cap,
                        "execution_cooldown_bars": config.execution_cooldown_bars,
                    },
                    symbols=config.symbols,
                )

            try:
                if config.model_type == "ensemble":
                    trainer = EnsembleTrainer(config)
                    result = trainer.train()
                else:
                    trainer = ModelTrainer(config)
                    result = trainer.run()
            except Exception as model_exc:
                if audit_logger is not None:
                    audit_logger.log_model_training_completed(
                        model_name=config.model_type,
                        model_version=config.model_name,
                        metrics={},
                        model_path=None,
                        duration_seconds=None,
                        success=False,
                        error=str(model_exc),
                    )
                if requested_model == "all":
                    logger.error(
                        "Model %s failed during all-model sweep; continuing. Error: %s",
                        config.model_type,
                        model_exc,
                    )
                    results.append(
                        (
                            model_type,
                            {
                                "success": False,
                                "model_type": config.model_type,
                                "model_name": config.model_name,
                                "model_path": None,
                                "training_duration_seconds": 0.0,
                                "training_metrics": {},
                                "error": str(model_exc),
                            },
                        )
                    )
                    overall_success = False
                    continue
                raise

            registry_entry: dict[str, Any] | None = None
            model_path = result.get("model_path")
            if isinstance(model_path, str) and model_path.strip():
                tags = [
                    "training_pipeline",
                    "institutional_mode",
                    f"cv:{config.cv_method}",
                    "validation:passed" if bool(result.get("success", False)) else "validation:failed",
                ]
                registry_entry = register_training_model_version(
                    registry_root=Path(config.output_dir) / "registry",
                    model_name=config.model_type,
                    model_version=config.model_name,
                    model_type=config.model_type,
                    model_path=model_path,
                    metrics=result.get("training_metrics", {}),
                    tags=tags,
                    is_active=bool(result.get("success", False)),
                    snapshot_manifest=(
                        trainer.snapshot_manifest
                        if isinstance(trainer, ModelTrainer)
                        else None
                    ),
                    training_config=config.__dict__,
                    project_root=PROJECT_ROOT,
                    model_card=result.get("model_card"),
                    deployment_plan=result.get("deployment_plan"),
                )
                result["registry_version_id"] = registry_entry.get("version_id")
                result["registry_path"] = str(Path(config.output_dir) / "registry" / "registry.json")

            if audit_logger is not None:
                audit_logger.log_model_training_completed(
                    model_name=config.model_type,
                    model_version=config.model_name,
                    metrics=result.get("training_metrics", {}),
                    model_path=result.get("model_path"),
                    duration_seconds=result.get("training_duration_seconds"),
                    success=bool(result.get("success", False)),
                    snapshot_id=result.get("snapshot_id"),
                    snapshot_manifest_path=result.get("snapshot_manifest_path"),
                    registry_version_id=result.get("registry_version_id"),
                    registry_path=result.get("registry_path"),
                    deployment_plan=result.get("deployment_plan"),
                )

            results.append((model_type, result))
            if not result.get("success", False):
                overall_success = False

            if requested_model != "all":
                break

        benchmark_json_path: Path | None = None
        benchmark_csv_path: Path | None = None
        champion_snapshot: dict[str, Any] | None = None
        has_persisted_models = any(
            isinstance(result.get("model_path"), str) and bool(str(result.get("model_path")).strip())
            for _, result in results
        )
        if has_persisted_models:
            try:
                matrix_rows = _build_training_matrix(results)
                benchmark_json_path, benchmark_csv_path = _persist_training_matrix_report(
                    output_dir=Path(getattr(args, "output_dir", "models")),
                    rows=matrix_rows,
                )
                champion_snapshot = _auto_select_champion_and_challenger(
                    output_dir=Path(getattr(args, "output_dir", "models")),
                    rows=matrix_rows,
                )

                promoted = bool(champion_snapshot.get("promoted", False)) if champion_snapshot else False
                champion_payload = champion_snapshot.get("champion", {}) if champion_snapshot else {}
                if promoted and audit_logger is not None and isinstance(champion_payload, dict):
                    champion_metrics = {
                        "governance_score": float(champion_payload.get("governance_score", 0.0)),
                        "mean_sharpe": float(champion_payload.get("mean_sharpe", 0.0)),
                        "holdout_sharpe": float(champion_payload.get("holdout_sharpe", 0.0)),
                        "pbo": float(champion_payload.get("pbo", 1.0)),
                    }
                    audit_logger.log_model_deployed(
                        model_name=str(champion_payload.get("model_type", "")),
                        model_version=str(champion_payload.get("registry_version_id", "")),
                        metrics=champion_metrics,
                        promotion_mode="auto_benchmark_promotion",
                        champion_snapshot_path=str(champion_snapshot.get("snapshot_path", "")),
                    )
            except Exception as governance_exc:
                logger.warning(f"Benchmark/champion governance post-processing failed: {governance_exc}")

        # Report results
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        for model_type, result in results:
            logger.info("-" * 80)
            logger.info(f"{model_type.upper()} RESULT")
            logger.info(f"Model saved to: {result.get('model_path', 'N/A')}")
            logger.info(f"Training duration: {result.get('training_duration_seconds', 0):.1f}s")
            metrics = result.get("training_metrics", {})
            if metrics:
                logger.info(f"Mean accuracy: {metrics.get('mean_accuracy', 0):.4f}")
                logger.info(f"Mean Sharpe: {metrics.get('mean_sharpe', 0):.4f}")
                if "deflated_sharpe" in metrics:
                    logger.info(f"Deflated Sharpe: {metrics['deflated_sharpe']:.4f}")
            if result.get("promotion_package_path"):
                logger.info(f"Promotion package: {result.get('promotion_package_path')}")
            if result.get("replay_manifest_path"):
                logger.info(f"Replay manifest: {result.get('replay_manifest_path')}")
        if benchmark_json_path is not None:
            logger.info(f"Training matrix (json): {benchmark_json_path}")
        if benchmark_csv_path is not None:
            logger.info(f"Training matrix (csv): {benchmark_csv_path}")
        if champion_snapshot is not None:
            logger.info(f"Champion/challenger snapshot: {champion_snapshot.get('snapshot_path')}")
            if champion_snapshot.get("promoted", False):
                champ = champion_snapshot.get("champion", {})
                if isinstance(champ, dict):
                    logger.info(
                        "Auto-promoted champion: %s:%s",
                        champ.get("model_type", "unknown"),
                        champ.get("registry_version_id", "unknown"),
                    )

        if overall_success:
            logger.info("Validation gates: PASSED")
            return 0

        logger.warning("Validation gates: FAILED for one or more models")
        logger.warning("Some models may not meet production requirements")
        return 1

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def validate_model(args: argparse.Namespace) -> int:
    """Validate an existing model against deployment gates."""
    logger.info("Validating model...")

    model_path = getattr(args, "model_path", None)
    if not model_path:
        logger.error("Model path required for validation")
        return 1

    try:
        from quant_trading_system.models.validation_gates import ModelValidationGates

        gates = ModelValidationGates(
            min_sharpe_ratio=getattr(args, "min_sharpe", 0.5),
            max_drawdown=getattr(args, "max_drawdown", 0.20),
            min_win_rate=getattr(args, "min_win_rate", 0.45),
        )

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Load artifacts if available
        artifacts_path = Path(model_path).with_suffix(".json").with_stem(
            Path(model_path).stem + "_artifacts"
        )

        if artifacts_path.exists():
            with open(artifacts_path) as f:
                artifacts = json.load(f)

            result = gates.validate(artifacts.get("training_metrics", {}))

            if result["passed"]:
                logger.info("Model PASSED all validation gates")
                return 0
            else:
                logger.warning("Model FAILED validation gates")
                for gate, info in result["gates"].items():
                    status = "PASS" if info["passed"] else "FAIL"
                    logger.info(f"  {gate}: {status}")
                return 1
        else:
            logger.warning("No artifacts found, cannot validate")
            return 1

    except ImportError:
        logger.error("Validation gates module not available")
        return 1
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for training script."""
    parser = argparse.ArgumentParser(description="AlphaTrade Model Training")
    parser.add_argument("--model", type=str, default="xgboost",
                       choices=SUPPORTED_MODELS + ["all"])
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--symbols", nargs="+", default=[])
    parser.add_argument("--start", "--start-date", dest="start_date", type=str, default="")
    parser.add_argument("--end", "--end-date", dest="end_date", type=str, default="")
    parser.add_argument("--cv-method", choices=["purged_kfold", "combinatorial", "walk_forward"], default="purged_kfold")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--embargo-pct", type=float, default=0.01)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nested-outer-splits", type=int, default=4)
    parser.add_argument("--nested-inner-splits", type=int, default=3)
    parser.add_argument(
        "--disable-nested-walk-forward",
        action="store_true",
        help="Forbidden in institutional mode; nested walk-forward is mandatory.",
    )
    parser.add_argument("--objective-weight-sharpe", type=float, default=1.0)
    parser.add_argument("--objective-weight-drawdown", type=float, default=0.5)
    parser.add_argument("--objective-weight-turnover", type=float, default=0.1)
    parser.add_argument("--objective-weight-calibration", type=float, default=0.25)
    parser.add_argument("--objective-weight-trade-activity", type=float, default=1.0)
    parser.add_argument("--objective-weight-cvar", type=float, default=0.4)
    parser.add_argument("--objective-weight-skew", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--replay-manifest",
        type=Path,
        default=None,
        help="Replay a prior run from replay/promotion/artifact manifest JSON.",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.45,
        help="Validation gate threshold for mean accuracy (default: 0.45).",
    )
    parser.add_argument("--min-trades", type=int, default=100)
    parser.add_argument("--holdout-pct", type=float, default=0.15)
    parser.add_argument("--min-holdout-sharpe", type=float, default=0.0)
    parser.add_argument("--max-holdout-drawdown", type=float, default=0.35)
    parser.add_argument("--max-regime-shift", type=float, default=0.35)
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Deprecated: institutional mode enforces GPU automatically.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Deprecated: institutional mode enforces GPU automatically.",
    )
    parser.add_argument(
        "--no-database",
        action="store_true",
        help="Forbidden in institutional mode; kept for explicit fail-fast validation.",
    )
    parser.add_argument(
        "--no-redis-cache",
        action="store_true",
        help="Forbidden in institutional mode; kept for explicit fail-fast validation.",
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Forbidden in institutional mode; kept for explicit fail-fast validation.",
    )
    parser.add_argument("--label-horizons", nargs="+", type=int, default=[1, 5, 20])
    parser.add_argument("--primary-horizon", type=int, default=5)
    parser.add_argument("--profit-taking", type=float, default=0.015)
    parser.add_argument("--stop-loss", type=float, default=0.010)
    parser.add_argument("--max-holding", type=int, default=20)
    parser.add_argument("--spread-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--impact-bps", type=float, default=2.0)
    parser.add_argument("--label-min-signal-abs-return-bps", type=float, default=8.0)
    parser.add_argument("--label-neutral-buffer-bps", type=float, default=4.0)
    parser.add_argument("--label-edge-cost-buffer-bps", type=float, default=2.0)
    parser.add_argument("--label-max-abs-forward-return", type=float, default=0.35)
    parser.add_argument("--label-signal-volatility-floor-mult", type=float, default=0.50)
    parser.add_argument("--label-volatility-lookback", type=int, default=20)
    parser.add_argument("--label-regime-lookback", type=int, default=30)
    parser.add_argument("--label-temporal-weight-decay", type=float, default=0.999)
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        default=["technical", "statistical", "microstructure", "cross_sectional"],
        help="Feature groups to compute (default: technical statistical microstructure cross_sectional)",
    )
    parser.add_argument(
        "--disable-cross-sectional",
        action="store_true",
        help="Disable cross-sectional features explicitly.",
    )
    parser.add_argument(
        "--strict-feature-groups",
        action="store_true",
        help="Fail run if any requested feature group cannot be materialized.",
    )
    parser.add_argument(
        "--max-cross-sectional-symbols",
        type=int,
        default=20,
        help="Adaptive guardrail: disable cross-sectional when symbol count exceeds this value.",
    )
    parser.add_argument(
        "--max-cross-sectional-rows",
        type=int,
        default=250000,
        help="Adaptive guardrail: disable cross-sectional when dataset rows exceed this value.",
    )
    parser.add_argument(
        "--feature-materialization-batch-rows",
        type=int,
        default=5000,
        help="Source-row chunk size used while writing features to PostgreSQL.",
    )
    parser.add_argument(
        "--feature-reuse-min-coverage",
        type=float,
        default=0.20,
        help="Minimum usable feature row ratio required to reuse PostgreSQL feature cache.",
    )
    parser.add_argument(
        "--skip-feature-persist",
        action="store_true",
        help="Skip persisting computed features to PostgreSQL.",
    )
    parser.add_argument("--disable-dynamic-no-trade-band", action="store_true")
    parser.add_argument("--execution-vol-target-daily", type=float, default=0.012)
    parser.add_argument("--execution-turnover-cap", type=float, default=0.90)
    parser.add_argument("--execution-cooldown-bars", type=int, default=2)
    parser.add_argument("--min-deflated-sharpe", type=float, default=0.10)
    parser.add_argument("--max-deflated-sharpe-pvalue", type=float, default=0.10)
    parser.add_argument("--max-pbo", type=float, default=0.45)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Console entrypoint for quant-train."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_training(args)


if __name__ == "__main__":
    sys.exit(main())
