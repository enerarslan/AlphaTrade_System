"""Unit tests for scripts/train.py."""

from __future__ import annotations

import json
import sys
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from quant_trading_system.models.training_lineage import register_training_model_version
from scripts import train as train_script


def _base_args(**overrides):
    args = {
        "model": "xgboost",
        "name": "",
        "symbols": [],
        "start": "",
        "end": "",
        "start_date": "",
        "end_date": "",
        "cv_method": "purged_kfold",
        "n_splits": 2,
        "embargo_pct": 0.01,
        "n_trials": 1,
        "epochs": 1,
        "batch_size": 8,
        "learning_rate": 0.001,
        "seed": 42,
        "nested_outer_splits": 2,
        "nested_inner_splits": 2,
        "disable_nested_walk_forward": False,
        "objective_weight_sharpe": 1.0,
        "objective_weight_drawdown": 0.5,
        "objective_weight_turnover": 0.1,
        "objective_weight_calibration": 0.25,
        "replay_manifest": None,
        "gpu": False,
        "use_gpu": False,
        "no_database": False,
        "no_redis_cache": False,
        "no_shap": False,
        "output_dir": "models",
        "feature_groups": ["technical", "statistical", "microstructure", "cross_sectional"],
        "disable_cross_sectional": False,
        "strict_feature_groups": False,
        "max_cross_sectional_symbols": 20,
        "max_cross_sectional_rows": 250000,
        "feature_materialization_batch_rows": 5000,
        "feature_reuse_min_coverage": 0.20,
    }
    args.update(overrides)
    return Namespace(**args)


def test_extract_base_symbol():
    assert train_script.ModelTrainer._extract_base_symbol("AAPL_15MIN") == "AAPL"
    assert train_script.ModelTrainer._extract_base_symbol("BRK.B_15MIN") == "BRK.B"
    assert train_script.ModelTrainer._extract_base_symbol("MSFT") == "MSFT"


def test_run_training_uses_start_end_alias(monkeypatch):
    captured = {}

    class DummyModelTrainer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: None)

    args = _base_args(start="2024-01-01", end="2024-12-31")
    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert captured["config"].start_date == "2024-01-01"
    assert captured["config"].end_date == "2024-12-31"
    assert captured["config"].use_gpu is True


def test_run_training_maps_feature_pipeline_flags(monkeypatch):
    captured = {}

    class DummyModelTrainer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: None)

    args = _base_args(
        feature_groups=["technical", "statistical"],
        disable_cross_sectional=True,
        strict_feature_groups=True,
        max_cross_sectional_symbols=12,
        max_cross_sectional_rows=120000,
        feature_materialization_batch_rows=3000,
        feature_reuse_min_coverage=0.35,
    )
    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert captured["config"].feature_groups == ["technical", "statistical"]
    assert captured["config"].enable_cross_sectional is False
    assert captured["config"].allow_feature_group_fallback is False
    assert captured["config"].max_cross_sectional_symbols == 12
    assert captured["config"].max_cross_sectional_rows == 120000
    assert captured["config"].feature_materialization_batch_rows == 3000
    assert captured["config"].feature_reuse_min_coverage == 0.35


def test_run_training_all_dispatch(monkeypatch):
    trained = []

    class DummyModelTrainer:
        def __init__(self, config):
            self.config = config
            trained.append(config.model_type)

        def run(self):
            return {"success": True, "training_metrics": {}}

    class DummyEnsembleTrainer:
        def __init__(self, config):
            self.config = config
            trained.append(config.model_type)

        def train(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "EnsembleTrainer", DummyEnsembleTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: None)

    args = _base_args(model="all")
    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert trained == [
        "xgboost",
        "lightgbm",
        "random_forest",
        "elastic_net",
        "lstm",
        "transformer",
        "tcn",
        "ensemble",
    ]


def test_run_training_all_continues_when_one_model_fails(monkeypatch):
    trained = []

    class DummyModelTrainer:
        def __init__(self, config):
            self.config = config
            trained.append(config.model_type)

        def run(self):
            if self.config.model_type == "lightgbm":
                raise RuntimeError("intentional failure")
            return {"success": True, "training_metrics": {}}

    class DummyEnsembleTrainer:
        def __init__(self, config):
            self.config = config
            trained.append(config.model_type)

        def train(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "EnsembleTrainer", DummyEnsembleTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: None)

    args = _base_args(model="all")
    exit_code = train_script.run_training(args)

    assert exit_code == 1
    assert trained[0] == "xgboost"
    assert "lightgbm" in trained
    assert "random_forest" in trained
    assert "ensemble" in trained
    assert trained[-1] == "ensemble"


def test_run_training_rejects_optional_disable_flags():
    args = _base_args(no_database=True)
    exit_code = train_script.run_training(args)
    assert exit_code == 1


def test_run_training_rejects_disable_nested_flag():
    args = _base_args(disable_nested_walk_forward=True)
    exit_code = train_script.run_training(args)
    assert exit_code == 1


def test_get_cv_splitter_walk_forward_applies_purge_gap():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            cv_method="walk_forward",
            n_splits=3,
            purge_pct=0.03,
        )
    )
    cv = trainer._get_cv_splitter()
    assert getattr(cv, "purge_gap", None) == 3


def test_validate_model_uses_min_accuracy_gate():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            min_accuracy=0.60,
            min_win_rate=0.40,
            min_sharpe_ratio=0.50,
            max_drawdown=0.20,
        )
    )
    trainer.training_metrics = {
        "mean_sharpe": 0.80,
        "mean_accuracy": 0.55,
        "mean_max_drawdown": 0.10,
        "mean_win_rate": 0.70,
        "mean_risk_adjusted_score": 0.25,
    }

    passed = trainer._validate_model()
    assert passed is False
    assert trainer.validation_results["gates"]["min_accuracy"][0] is False
    assert trainer.validation_results["gates"]["min_win_rate"][0] is True


def test_effective_trade_target_scales_with_fold_size():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(model_type="xgboost", min_trades=100)
    )

    assert trainer._effective_trade_target(None) == 100
    assert trainer._effective_trade_target(50) == 3
    assert trainer._effective_trade_target(400) == 10


def test_validate_model_uses_dynamic_min_trades_threshold():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            min_trades=100,
            use_meta_labeling=False,
            require_nested_trace_for_promotion=False,
            require_objective_breakdown_for_promotion=False,
        )
    )
    trainer.training_metrics = {
        "mean_sharpe": 1.0,
        "mean_accuracy": 0.65,
        "mean_max_drawdown": 0.10,
        "mean_win_rate": 0.60,
        "mean_trade_count": 20.0,
        "mean_test_size": 50.0,
        "mean_risk_adjusted_score": 0.25,
        "deflated_sharpe": 0.8,
        "deflated_sharpe_p_value": 0.02,
        "pbo": 0.20,
    }

    passed = trainer._validate_model()

    assert passed is True
    assert trainer.training_metrics["effective_min_trades_gate"] == pytest.approx(3.0)
    assert trainer.validation_results["gates"]["min_trades"][2] == pytest.approx(3.0)


def test_load_model_defaults_random_forest_windows_forces_single_job(monkeypatch):
    monkeypatch.setattr(train_script.os, "name", "nt", raising=False)
    defaults = train_script._load_model_defaults("random_forest", use_gpu=True)
    assert defaults.get("n_jobs") == 1


def test_augment_params_random_forest_windows_forces_single_job(monkeypatch):
    monkeypatch.setattr(train_script.os, "name", "nt", raising=False)
    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="random_forest"))
    params = trainer._augment_params_for_train_labels({}, np.array([0, 1, 1, 0], dtype=float))
    assert params.get("class_weight") == "balanced_subsample"
    assert params.get("n_jobs") == 1


def test_compute_shap_values_sequence_model_handles_single_row_kernel_calls(monkeypatch):
    class DummySequenceModel:
        def __init__(self, lookback_window: int = 5):
            self.lookback_window = lookback_window
            self.call_sizes: list[int] = []

        def predict_proba(self, X):
            self.call_sizes.append(len(X))
            if len(X) < self.lookback_window:
                raise ValueError(
                    f"Not enough samples ({len(X)}) for lookback window ({self.lookback_window})"
                )
            n_predictions = len(X) - self.lookback_window + 1
            return np.tile(np.array([[0.4, 0.6]]), (n_predictions, 1))

    class DummyKernelExplainer:
        def __init__(self, predict_fn, background):
            self.predict_fn = predict_fn
            self.background = background

        def shap_values(self, X, nsamples=None):
            single_row = X[:1]
            y_single = self.predict_fn(single_row)
            assert y_single.shape[0] == 1
            return np.zeros((X.shape[0], X.shape[1]))

    class DummyShapModule:
        KernelExplainer = DummyKernelExplainer

    monkeypatch.setitem(sys.modules, "shap", DummyShapModule)

    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="lstm",
            n_shap_samples=8,
            model_params={"lookback_window": 5},
            compute_shap=True,
        )
    )
    trainer.features = pd.DataFrame(np.random.rand(8, 3), columns=["f1", "f2", "f3"])
    trainer.model = DummySequenceModel(lookback_window=5)
    trainer.training_metrics = {}

    trainer._compute_shap_values()

    assert "shap_importance" in trainer.training_metrics
    assert trainer.model.call_sizes
    assert min(trainer.model.call_sizes) >= trainer.model.lookback_window


def test_calculate_fold_metrics_uses_active_trade_returns():
    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="xgboost"))
    metrics = trainer._calculate_fold_metrics(
        y_true=np.array([1, 1, 0, 0], dtype=float),
        y_pred=np.array([1, 0, 0, 0], dtype=float),
        y_proba=np.array([0.70, 0.50, 0.50, 0.50], dtype=float),
        long_threshold=0.55,
        short_threshold=0.45,
        realized_forward_returns=np.array([0.01, 0.01, 0.01, 0.01], dtype=float),
    )

    assert metrics["trade_count"] == pytest.approx(1.0)
    assert metrics["trade_return_observations"] == pytest.approx(1.0)
    assert metrics["win_rate"] == pytest.approx(1.0)


def test_build_execution_profile_relaxes_band_when_activity_too_low():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            min_trades=20,
            dynamic_no_trade_band=True,
        )
    )
    y_proba = np.full(120, 0.565, dtype=float)
    realized = np.full(120, 0.001, dtype=float)

    profile = trainer._build_execution_profile(
        y_proba=y_proba,
        long_threshold=0.55,
        short_threshold=0.45,
        realized_returns=realized,
    )

    assert np.count_nonzero(profile["raw_signals"]) > 0
    assert float(np.mean(profile["long_threshold_series"])) < 0.58


def test_derive_signal_thresholds_with_labels_boosts_activity():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(model_type="xgboost", min_trades=40)
    )
    rng = np.random.default_rng(17)
    train_proba = np.clip(rng.normal(loc=0.68, scale=0.08, size=240), 0.01, 0.99)
    train_labels = (rng.random(240) < 0.52).astype(float)

    long_unlabeled, short_unlabeled = trainer._derive_signal_thresholds(train_proba)
    long_labeled, short_labeled = trainer._derive_signal_thresholds(
        train_proba,
        train_labels=train_labels,
    )

    signals_unlabeled = np.where(
        train_proba >= long_unlabeled,
        1.0,
        np.where(train_proba <= short_unlabeled, -1.0, 0.0),
    )
    signals_labeled = np.where(
        train_proba >= long_labeled,
        1.0,
        np.where(train_proba <= short_labeled, -1.0, 0.0),
    )
    assert float(np.mean(signals_labeled != 0.0)) >= float(np.mean(signals_unlabeled != 0.0))


def test_validate_model_normalizes_gate_bool_types():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            use_meta_labeling=False,
            require_nested_trace_for_promotion=False,
            require_objective_breakdown_for_promotion=False,
        )
    )
    trainer.training_metrics = {
        "mean_sharpe": np.float64(0.6),
        "mean_accuracy": np.float64(0.7),
        "mean_max_drawdown": np.float64(0.1),
        "mean_win_rate": np.float64(0.6),
        "mean_trade_count": np.float64(150.0),
        "mean_risk_adjusted_score": np.float64(0.2),
        "mean_test_size": np.float64(500.0),
        "deflated_sharpe": np.float64(0.3),
        "deflated_sharpe_p_value": np.float64(0.05),
        "pbo": np.float64(0.2),
    }

    trainer._validate_model()
    gate_flag = trainer.validation_results["gates"]["min_accuracy"][0]
    assert isinstance(gate_flag, bool)


def test_save_model_uses_model_native_save_when_available(tmp_path):
    class DummyNativeSaveModel:
        def __init__(self):
            self.saved_path = None

        def save(self, path):
            self.saved_path = path
            with open(f"{path}.pkl", "wb") as f:
                f.write(b"model")
            with open(f"{path}.json", "w", encoding="utf-8") as f:
                f.write("{}")

    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            model_name="native_save_model",
            output_dir=str(tmp_path),
            save_artifacts=False,
        )
    )
    trainer.model = DummyNativeSaveModel()

    model_path = trainer._save_model()

    assert model_path == str(tmp_path / "native_save_model.pkl")
    assert (tmp_path / "native_save_model.pkl").exists()
    assert trainer.model.saved_path == (tmp_path / "native_save_model")


def test_build_parser_supports_institutional_failfast_flags_and_name():
    parser = train_script.build_parser()
    args = parser.parse_args(
        [
            "--model",
            "elastic_net",
            "--name",
            "elastic_v1",
            "--no-database",
            "--no-redis-cache",
            "--no-shap",
        ]
    )
    assert args.model == "elastic_net"
    assert args.name == "elastic_v1"
    assert args.no_database is True
    assert args.no_redis_cache is True
    assert args.no_shap is True


def test_build_parser_supports_feature_pipeline_flags():
    parser = train_script.build_parser()
    args = parser.parse_args(
        [
            "--feature-groups",
            "technical",
            "statistical",
            "--disable-cross-sectional",
            "--strict-feature-groups",
            "--max-cross-sectional-symbols",
            "14",
            "--max-cross-sectional-rows",
            "150000",
            "--feature-materialization-batch-rows",
            "4000",
            "--feature-reuse-min-coverage",
            "0.3",
        ]
    )
    assert args.feature_groups == ["technical", "statistical"]
    assert args.disable_cross_sectional is True
    assert args.strict_feature_groups is True
    assert args.max_cross_sectional_symbols == 14
    assert args.max_cross_sectional_rows == 150000
    assert args.feature_materialization_batch_rows == 4000
    assert args.feature_reuse_min_coverage == 0.3


def test_build_parser_supports_advanced_risk_and_execution_flags():
    parser = train_script.build_parser()
    args = parser.parse_args(
        [
            "--objective-weight-cvar",
            "0.55",
            "--objective-weight-skew",
            "0.15",
            "--holdout-pct",
            "0.2",
            "--min-holdout-sharpe",
            "0.1",
            "--max-holdout-drawdown",
            "0.3",
            "--max-regime-shift",
            "0.25",
            "--label-edge-cost-buffer-bps",
            "2.5",
            "--disable-dynamic-no-trade-band",
            "--execution-vol-target-daily",
            "0.01",
            "--execution-turnover-cap",
            "0.8",
            "--execution-cooldown-bars",
            "5",
        ]
    )
    assert args.objective_weight_cvar == pytest.approx(0.55)
    assert args.objective_weight_skew == pytest.approx(0.15)
    assert args.holdout_pct == pytest.approx(0.2)
    assert args.min_holdout_sharpe == pytest.approx(0.1)
    assert args.max_holdout_drawdown == pytest.approx(0.3)
    assert args.max_regime_shift == pytest.approx(0.25)
    assert args.label_edge_cost_buffer_bps == pytest.approx(2.5)
    assert args.disable_dynamic_no_trade_band is True
    assert args.execution_vol_target_daily == pytest.approx(0.01)
    assert args.execution_turnover_cap == pytest.approx(0.8)
    assert args.execution_cooldown_bars == 5


def test_build_training_matrix_ranks_by_governance_score():
    rows = train_script._build_training_matrix(
        [
            (
                "xgboost",
                {
                    "success": True,
                    "model_name": "xgb_a",
                    "training_metrics": {
                        "mean_risk_adjusted_score": 0.40,
                        "holdout_sharpe": 0.30,
                        "deflated_sharpe": 0.20,
                        "holdout_max_drawdown": 0.10,
                        "pbo": 0.15,
                    },
                },
            ),
            (
                "lightgbm",
                {
                    "success": True,
                    "model_name": "lgb_b",
                    "training_metrics": {
                        "mean_risk_adjusted_score": 0.10,
                        "holdout_sharpe": 0.05,
                        "deflated_sharpe": 0.00,
                        "holdout_max_drawdown": 0.20,
                        "pbo": 0.30,
                    },
                },
            ),
        ]
    )

    assert rows[0]["model_type"] == "xgboost"
    assert rows[0]["rank"] == 1
    assert rows[0]["governance_score"] > rows[1]["governance_score"]


def test_auto_select_champion_and_challenger_promotes_best(tmp_path):
    first = register_training_model_version(
        registry_root=tmp_path / "registry",
        model_name="xgboost",
        model_version="xgb_a",
        model_type="xgboost",
        model_path=str(tmp_path / "xgb_a.pkl"),
        metrics={"mean_risk_adjusted_score": 0.2},
        is_active=False,
    )
    second = register_training_model_version(
        registry_root=tmp_path / "registry",
        model_name="lightgbm",
        model_version="lgb_b",
        model_type="lightgbm",
        model_path=str(tmp_path / "lgb_b.pkl"),
        metrics={"mean_risk_adjusted_score": 0.4},
        is_active=False,
    )

    rows = [
        {
            "model_type": "lightgbm",
            "success": True,
            "governance_score": 0.8,
            "registry_version_id": second["version_id"],
            "promotion_package_path": "models/promotion_packages/lgb_b.json",
            "deployment_plan": {
                "ready_for_production": True,
                "canary_rollout": [{"phase": 1, "capital_fraction": 0.05}],
            },
        },
        {
            "model_type": "xgboost",
            "success": True,
            "governance_score": 0.5,
            "registry_version_id": first["version_id"],
            "promotion_package_path": "models/promotion_packages/xgb_a.json",
            "deployment_plan": {"ready_for_production": True},
        },
    ]

    snapshot = train_script._auto_select_champion_and_challenger(tmp_path, rows)

    assert snapshot["promoted"] is True
    assert snapshot["champion"]["model_type"] == "lightgbm"
    assert (tmp_path / "active_model.json").exists()
    assert (tmp_path / "champion_challenger_snapshot.json").exists()
    assert (tmp_path / "canary_rollout_plan.json").exists()


def test_auto_select_champion_requires_ready_plan_and_promotion_package(tmp_path):
    entry = register_training_model_version(
        registry_root=tmp_path / "registry",
        model_name="ensemble",
        model_version="ens_a",
        model_type="ensemble",
        model_path=str(tmp_path / "ens_a.pkl"),
        metrics={"mean_accuracy": 0.6},
        is_active=False,
    )
    rows = [
        {
            "model_type": "ensemble",
            "success": True,
            "governance_score": 0.9,
            "registry_version_id": entry["version_id"],
            "promotion_package_path": "",
            "deployment_plan": {},
        }
    ]

    snapshot = train_script._auto_select_champion_and_challenger(tmp_path, rows)
    assert snapshot["promoted"] is False
    assert snapshot["champion"] is None
    assert not (tmp_path / "active_model.json").exists()


def test_multiple_testing_correction_adds_deflated_sharpe_and_pbo_metrics():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            correction_method="deflated_sharpe",
            n_trials=20,
        )
    )
    trainer.training_metrics = {"mean_sharpe": 1.1}
    trainer.cv_results = [{"fold": 1}, {"fold": 2}]
    trainer.cv_return_series = [
        np.random.default_rng(1).normal(0.0005, 0.01, size=300),
        np.random.default_rng(2).normal(0.0003, 0.012, size=300),
    ]

    trainer._apply_multiple_testing_correction()

    assert "deflated_sharpe" in trainer.training_metrics
    assert "deflated_sharpe_p_value" in trainer.training_metrics
    assert "pbo" in trainer.training_metrics
    assert "pbo_interpretation" in trainer.training_metrics


def test_multiple_testing_correction_sets_effective_pbo_gate_and_source():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            correction_method="deflated_sharpe",
            n_trials=12,
            max_pbo=0.45,
        )
    )
    trainer.training_metrics = {"mean_sharpe": 0.4}
    trainer.cv_results = [{"fold": 1}, {"fold": 2}]
    trainer.cv_return_series = [np.array([0.0, 0.0, 0.0, 0.0], dtype=float)]
    trainer.cv_active_return_series = [np.array([0.001, -0.001], dtype=float)]

    trainer._apply_multiple_testing_correction()

    assert trainer.training_metrics["multiple_testing_return_source"] == "all_returns"
    assert trainer.training_metrics["multiple_testing_return_observations"] == pytest.approx(4.0)
    assert float(trainer.training_metrics["effective_max_pbo_gate"]) >= 0.45


def test_validate_model_fails_when_pbo_gate_breaches():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            max_pbo=0.30,
        )
    )
    trainer.training_metrics = {
        "mean_sharpe": 1.0,
        "mean_accuracy": 0.65,
        "mean_max_drawdown": 0.10,
        "mean_win_rate": 0.60,
        "mean_risk_adjusted_score": 0.25,
        "deflated_sharpe": 0.8,
        "deflated_sharpe_p_value": 0.02,
        "pbo": 0.55,
    }

    passed = trainer._validate_model()
    assert passed is False
    assert trainer.validation_results["gates"]["max_pbo"][0] is False


def test_validate_model_uses_effective_pbo_gate_override():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            max_pbo=0.45,
            use_meta_labeling=False,
        )
    )
    trainer.training_metrics = {
        "mean_sharpe": 1.0,
        "mean_accuracy": 0.65,
        "mean_max_drawdown": 0.10,
        "mean_win_rate": 0.60,
        "mean_trade_count": 20.0,
        "mean_test_size": 50.0,
        "mean_risk_adjusted_score": 0.25,
        "deflated_sharpe": 0.8,
        "deflated_sharpe_p_value": 0.02,
        "pbo": 0.50,
        "effective_max_pbo_gate": 0.55,
        "nested_cv_trace": [{"outer_fold": 1}],
        "objective_component_summary": {"objective_sharpe_component": 0.5},
    }
    trainer.nested_cv_trace = [{"outer_fold": 1}]

    passed = trainer._validate_model()

    assert passed is True
    assert trainer.validation_results["gates"]["max_pbo"][0] is True
    assert trainer.validation_results["gates"]["max_pbo"][2] == pytest.approx(0.55)


def test_fit_model_falls_back_to_sample_weight_keyword():
    captured = {}

    class DummyModel:
        def fit(self, X, y, sample_weight=None):
            captured["sample_weight"] = sample_weight

    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(model_type="xgboost")
    )
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, size=10)
    w = np.linspace(0.5, 1.5, 10)

    trainer._fit_model(
        model=DummyModel(),
        X_train=X,
        y_train=y,
        X_val=X,
        y_val=y,
        sample_weights=w,
    )

    assert np.allclose(captured["sample_weight"], w)


def test_load_replay_manifest_accepts_artifacts_shape(tmp_path):
    manifest_path = tmp_path / "artifacts.json"
    manifest_path.write_text(
        json.dumps(
            {
                "config": {
                    "model_type": "lightgbm",
                    "symbols": ["AAPL", "MSFT"],
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = train_script._load_replay_manifest(manifest_path)

    assert loaded["training_config"]["model_type"] == "lightgbm"
    assert loaded["training_config"]["symbols"] == ["AAPL", "MSFT"]


def test_run_training_replay_manifest_overrides_cli(monkeypatch, tmp_path):
    captured = {}

    class DummyModelTrainer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: None)

    replay_path = tmp_path / "replay_manifest.json"
    replay_path.write_text(
        json.dumps(
            {
                "training_config": {
                    "model_type": "lightgbm",
                    "model_name": "prod_candidate",
                    "symbols": ["NVDA", "AMD"],
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                }
            }
        ),
        encoding="utf-8",
    )

    args = _base_args(
        model="xgboost",
        symbols=["AAPL"],
        replay_manifest=str(replay_path),
    )
    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert captured["config"].model_type == "lightgbm"
    assert captured["config"].symbols == ["NVDA", "AMD"]
    assert captured["config"].start_date == "2023-01-01"
    assert captured["config"].end_date == "2023-12-31"
    assert captured["config"].model_name.startswith("prod_candidate_replay_")


def test_validate_model_requires_nested_trace_for_promotion():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            max_pbo=0.60,
        )
    )
    trainer.training_metrics = {
        "mean_sharpe": 1.1,
        "mean_accuracy": 0.62,
        "mean_max_drawdown": 0.10,
        "mean_win_rate": 0.60,
        "mean_risk_adjusted_score": 0.20,
        "deflated_sharpe": 0.7,
        "deflated_sharpe_p_value": 0.05,
        "pbo": 0.20,
        "objective_component_summary": {
            "objective_sharpe_component": 1.0,
            "objective_drawdown_penalty": -0.05,
            "objective_turnover_penalty": -0.01,
            "objective_calibration_penalty": -0.02,
        },
    }
    trainer.nested_cv_trace = []

    passed = trainer._validate_model()

    assert passed is False
    assert trainer.validation_results["gates"]["nested_walk_forward_trace"][0] is False


def test_generate_cv_splits_panel_data_uses_timestamp_blocks():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            cv_method="purged_kfold",
            n_splits=2,
            purge_pct=0.02,
        )
    )
    timestamps = pd.to_datetime(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
        ],
        utc=True,
    ).to_numpy()
    trainer.timestamps = timestamps
    trainer.row_symbols = np.array(["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"])

    X = np.random.default_rng(11).normal(size=(6, 3))
    y = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    splits = trainer._generate_cv_splits(X, y)

    assert splits
    for train_idx, test_idx in splits:
        train_ts = set(pd.to_datetime(timestamps[train_idx], utc=True))
        test_ts = set(pd.to_datetime(timestamps[test_idx], utc=True))
        assert train_ts.isdisjoint(test_ts)


def test_validate_no_future_leakage_handles_panel_timestamps_without_false_positive():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            cv_method="walk_forward",
            n_splits=2,
            purge_pct=0.02,
        )
    )
    rng = np.random.default_rng(7)
    timestamps = pd.to_datetime(
        [
            *pd.date_range("2024-01-01", periods=12, freq="D", tz="UTC"),
            *pd.date_range("2024-01-01", periods=12, freq="D", tz="UTC"),
        ]
    ).to_numpy()
    trainer.timestamps = timestamps
    trainer.row_symbols = np.array(["AAPL"] * 12 + ["MSFT"] * 12)
    trainer.features = pd.DataFrame(
        rng.normal(size=(24, 4)),
        columns=["f1", "f2", "f3", "f4"],
    )
    trainer.feature_names = trainer.features.columns.tolist()
    trainer.labels = pd.Series(rng.integers(0, 2, size=24))

    trainer._validate_no_future_leakage()


def test_train_meta_labeler_uses_oof_predictions(monkeypatch):
    class DummyMetaLabelConfig:
        def __init__(self, model_type):
            self.model_type = model_type

    class DummyMetaLabeler:
        def __init__(self, config):
            self.config = config
            self.fit_rows = 0

        def fit(self, X, signals, prices):
            self.fit_rows = len(X)

    import quant_trading_system.models.meta_labeling as meta_module

    monkeypatch.setattr(meta_module, "MetaLabelConfig", DummyMetaLabelConfig)
    monkeypatch.setattr(meta_module, "MetaLabeler", DummyMetaLabeler)

    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(model_type="xgboost", meta_model_type="xgboost")
    )
    trainer.features = pd.DataFrame(
        np.random.default_rng(3).normal(size=(10, 3)),
        columns=["a", "b", "c"],
    )
    trainer.feature_names = trainer.features.columns.tolist()
    trainer.close_prices = pd.Series(np.linspace(100, 110, 10))
    trainer.oof_primary_proba = np.array([0.6, 0.4, np.nan, 0.7, 0.2, 0.55, 0.45, 0.8, 0.3, 0.65])
    trainer.training_metrics = {}

    trainer._train_meta_labeler()

    assert trainer.meta_model is not None
    assert trainer.meta_model.fit_rows == 9
    assert trainer.training_metrics["meta_label_oof_coverage"] == pytest.approx(0.9)


def test_compute_net_returns_prefers_realized_forward_returns():
    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="xgboost"))
    y_true = np.array([1, 0, 1, 0], dtype=int)
    y_proba = np.array([0.8, 0.2, 0.6, 0.4], dtype=float)
    realized = np.array([0.01, -0.02, 0.03, -0.01], dtype=float)

    net = trainer._compute_net_returns(
        y_true=y_true,
        y_proba=y_proba,
        realized_forward_returns=realized,
    )

    assert net.shape[0] == 4
    assert np.isfinite(net).all()
    assert float(np.mean(net)) != 0.0
