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
        "nested_outer_stability_ratio_cap": 1.25,
        "nested_outer_stability_min_trials": 8,
        "disable_nested_walk_forward": False,
        "objective_weight_sharpe": 1.0,
        "objective_weight_drawdown": 0.5,
        "objective_weight_turnover": 0.1,
        "objective_weight_calibration": 0.25,
        "objective_weight_trade_activity": 1.0,
        "objective_weight_cvar": 0.4,
        "objective_weight_skew": 0.1,
        "objective_weight_tail_risk": 0.35,
        "objective_weight_symbol_concentration": 0.20,
        "objective_expected_shortfall_cap": 0.012,
        "replay_manifest": None,
        "gpu": False,
        "use_gpu": False,
        "no_database": False,
        "no_redis_cache": False,
        "no_shap": False,
        "output_dir": "models",
        "label_horizons": [1, 5, 20],
        "primary_horizon": 5,
        "primary_horizon_sweep": [],
        "meta_label_min_confidence": 0.55,
        "disable_meta_dynamic_threshold": False,
        "holdout_pct": 0.15,
        "min_holdout_sharpe": 0.0,
        "min_holdout_regime_sharpe": -0.10,
        "min_holdout_symbol_coverage": 0.60,
        "min_holdout_symbol_p25_sharpe": -0.10,
        "max_holdout_symbol_underwater_ratio": 0.55,
        "max_holdout_drawdown": 0.35,
        "max_regime_shift": 0.35,
        "max_symbol_concentration_hhi": 0.65,
        "disable_auto_live_profile": False,
        "auto_live_profile_symbol_threshold": 40,
        "auto_live_profile_min_years": 4.0,
        "feature_groups": ["technical", "statistical", "microstructure", "cross_sectional"],
        "disable_symbol_quality_filter": False,
        "symbol_quality_min_rows": 1200,
        "symbol_quality_min_symbols": 8,
        "symbol_quality_max_missing_ratio": 0.12,
        "symbol_quality_max_extreme_move_ratio": 0.08,
        "symbol_quality_max_corporate_action_ratio": 0.02,
        "symbol_quality_min_median_dollar_volume": 1_000_000.0,
        "disable_cross_sectional": False,
        "strict_feature_groups": False,
        "allow_partial_feature_fallback": False,
        "max_cross_sectional_symbols": 20,
        "max_cross_sectional_rows": 250000,
        "feature_materialization_batch_rows": 5000,
        "feature_reuse_min_coverage": 0.20,
        "skip_feature_persist": False,
        "windows_fallback_features": False,
        "disable_dynamic_no_trade_band": False,
        "execution_vol_target_daily": 0.012,
        "execution_turnover_cap": 0.90,
        "execution_cooldown_bars": 2,
        "execution_max_symbol_entry_share": 0.68,
        "min_accuracy": 0.45,
        "min_trades": 100,
        "min_deflated_sharpe": 0.10,
        "max_deflated_sharpe_pvalue": 0.10,
        "max_pbo": 0.45,
        "disable_lightgbm_monotonic_constraints": False,
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
        windows_fallback_features=True,
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
    assert captured["config"].windows_force_fallback_features is True


def test_run_training_maps_allow_partial_feature_fallback(monkeypatch):
    captured = {}

    class DummyModelTrainer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: None)

    args = _base_args(allow_partial_feature_fallback=True, strict_feature_groups=False)
    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert captured["config"].allow_feature_group_fallback is True


def test_run_training_maps_tail_risk_and_monotonic_flags(monkeypatch):
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
        model="lightgbm",
        objective_weight_tail_risk=1.1,
        objective_expected_shortfall_cap=0.0045,
        nested_outer_stability_ratio_cap=1.9,
        nested_outer_stability_min_trials=17,
        execution_max_symbol_entry_share=0.61,
        disable_lightgbm_monotonic_constraints=True,
    )
    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert captured["config"].objective_weight_tail_risk == pytest.approx(1.1)
    assert captured["config"].objective_expected_shortfall_cap == pytest.approx(0.0045)
    assert captured["config"].nested_outer_stability_ratio_cap == pytest.approx(1.9)
    assert captured["config"].nested_outer_stability_min_trials == 17
    assert captured["config"].execution_max_symbol_entry_share == pytest.approx(0.61)
    assert captured["config"].lightgbm_use_monotonic_constraints is False


def test_run_training_maps_symbol_concentration_and_quality_filters(monkeypatch):
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
        objective_weight_symbol_concentration=0.33,
        min_holdout_regime_sharpe=-0.05,
        max_symbol_concentration_hhi=0.58,
        disable_symbol_quality_filter=True,
        symbol_quality_min_rows=1500,
        symbol_quality_min_symbols=10,
        symbol_quality_max_missing_ratio=0.10,
        symbol_quality_max_extreme_move_ratio=0.07,
        symbol_quality_max_corporate_action_ratio=0.015,
        symbol_quality_min_median_dollar_volume=2_500_000.0,
    )
    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert captured["config"].objective_weight_symbol_concentration == pytest.approx(0.33)
    assert captured["config"].min_holdout_regime_sharpe == pytest.approx(-0.05)
    assert captured["config"].max_symbol_concentration_hhi == pytest.approx(0.58)
    assert captured["config"].enable_symbol_quality_filter is False
    assert captured["config"].symbol_quality_min_rows == 1500
    assert captured["config"].symbol_quality_min_symbols == 10
    assert captured["config"].symbol_quality_max_missing_ratio == pytest.approx(0.10)
    assert captured["config"].symbol_quality_max_extreme_move_ratio == pytest.approx(0.07)
    assert captured["config"].symbol_quality_max_corporate_action_ratio == pytest.approx(0.015)
    assert captured["config"].symbol_quality_min_median_dollar_volume == pytest.approx(2_500_000.0)


def test_run_training_maps_horizon_and_meta_threshold_flags(monkeypatch):
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
        primary_horizon=3,
        primary_horizon_sweep=[1, 3, 10],
        meta_label_min_confidence=0.62,
        disable_meta_dynamic_threshold=True,
    )
    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert captured["config"].primary_horizon_sweep == [1, 3, 10]
    assert captured["config"].primary_label_horizon == 10
    assert captured["config"].meta_label_min_confidence == pytest.approx(0.62)
    assert captured["config"].meta_label_dynamic_threshold is False


def test_run_training_horizon_sweep_dispatch(monkeypatch):
    trained = []

    class DummyModelTrainer:
        def __init__(self, config):
            self.config = config
            trained.append((config.model_type, config.primary_label_horizon, config.model_name))

        def run(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: None)

    args = _base_args(model="xgboost", name="sweep", primary_horizon_sweep=[1, 5, 10])
    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert [h for _, h, _ in trained] == [1, 5, 10]
    assert all(model == "xgboost" for model, _, _ in trained)
    assert all(name.startswith("sweep_xgboost_h") for _, _, name in trained)


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


def test_prepare_params_for_train_size_regularizes_lightgbm():
    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="lightgbm"))
    raw_params = {
        "num_leaves": 128,
        "max_depth": 11,
        "n_estimators": 900,
        "min_data_in_leaf": 4,
        "learning_rate": 0.005,
        "lambda_l1": 0.0,
        "lambda_l2": 0.1,
    }

    adjusted = trainer._prepare_params_for_train_size(raw_params, train_size=200)

    assert adjusted["num_leaves"] <= 20
    assert adjusted["max_depth"] <= 6
    assert adjusted["n_estimators"] <= 420
    assert adjusted["min_data_in_leaf"] >= 20
    assert adjusted["learning_rate"] >= 0.015
    assert adjusted["lambda_l1"] >= 0.10
    assert adjusted["lambda_l2"] >= 1.00


def test_optuna_search_space_regularizes_lightgbm_for_small_folds():
    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="lightgbm"))
    search_space = trainer._get_optuna_search_space(min_train_size=180)
    assert search_space is not None

    class DummyTrial:
        def __init__(self):
            self.int_bounds = {}
            self.float_bounds = {}

        def suggest_int(self, name, low, high, **kwargs):
            self.int_bounds[name] = (low, high)
            return high

        def suggest_float(self, name, low, high, **kwargs):
            self.float_bounds[name] = (low, high)
            return high

    trial = DummyTrial()
    params = search_space(trial)

    assert trial.int_bounds["max_depth"][1] == 6
    assert trial.int_bounds["bagging_freq"][1] == 4
    assert trial.float_bounds["scale_pos_weight"][1] == pytest.approx(1.15)
    assert params["min_data_in_leaf"] >= 30
    assert params["max_bin"] <= 191


def test_optuna_search_space_regularizes_xgboost_for_small_folds():
    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="xgboost"))
    search_space = trainer._get_optuna_search_space(min_train_size=200)
    assert search_space is not None

    class DummyTrial:
        def __init__(self):
            self.int_bounds = {}
            self.float_bounds = {}

        def suggest_int(self, name, low, high, **kwargs):
            self.int_bounds[name] = (low, high)
            return high

        def suggest_float(self, name, low, high, **kwargs):
            self.float_bounds[name] = (low, high)
            return high

    trial = DummyTrial()
    _ = search_space(trial)

    assert trial.int_bounds["n_estimators"][1] <= 760
    assert trial.int_bounds["max_depth"][1] <= 6
    assert trial.float_bounds["learning_rate"][1] == pytest.approx(0.08)
    assert trial.float_bounds["scale_pos_weight"][1] <= 3.0


def test_fold_reliability_weight_penalizes_low_support():
    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="xgboost", min_trades=40))
    strong_metrics = {
        "test_size": 220,
        "trade_count": 18.0,
        "trade_return_observations": 17.0,
        "active_signal_rate": 0.14,
        "sharpe": 0.8,
        "win_rate": 0.55,
    }
    weak_metrics = {
        "test_size": 220,
        "trade_count": 18.0,
        "trade_return_observations": 2.0,
        "active_signal_rate": 0.015,
        "sharpe": -0.4,
        "win_rate": 0.35,
    }

    strong_weight = trainer._fold_reliability_weight(strong_metrics, evaluation_size=220)
    weak_weight = trainer._fold_reliability_weight(weak_metrics, evaluation_size=220)

    assert 0.10 <= weak_weight <= 1.0
    assert 0.10 <= strong_weight <= 1.0
    assert strong_weight > weak_weight


def test_has_binary_class_support_detects_single_class_folds():
    assert train_script.ModelTrainer._has_binary_class_support(np.array([0, 1, 0, 1], dtype=float)) is True
    assert train_script.ModelTrainer._has_binary_class_support(np.array([1, 1, 1], dtype=float)) is False
    assert train_script.ModelTrainer._has_binary_class_support(np.array([0, 0, 0], dtype=float)) is False


def test_infer_regime_side_bias_shifts_long_short_by_regime_and_drift():
    bull_regimes = np.array(["bull_trend"] * 180, dtype=object)
    bear_regimes = np.array(["bear_trend"] * 180, dtype=object)
    bull_returns = np.full(180, 0.0012, dtype=float)
    bear_returns = np.full(180, -0.0012, dtype=float)

    bull_long_shift, bull_short_shift = train_script.ModelTrainer._infer_regime_side_bias(
        bull_regimes,
        bull_returns,
    )
    bear_long_shift, bear_short_shift = train_script.ModelTrainer._infer_regime_side_bias(
        bear_regimes,
        bear_returns,
    )

    assert bull_long_shift < 0.0
    assert bull_short_shift < 0.0
    assert bear_long_shift > 0.0
    assert bear_short_shift > 0.0


def test_outer_trial_stability_gate_rejects_high_dispersion_trials():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            nested_outer_stability_ratio_cap=0.30,
            nested_outer_stability_min_trials=5,
        )
    )
    stable_ok, stable_ratio = trainer._outer_trial_stability_gate(
        [0.62, 0.60, 0.58, 0.61, 0.59, 0.60],
        best_value=0.62,
    )
    unstable_ok, unstable_ratio = trainer._outer_trial_stability_gate(
        [0.85, -0.55, 0.90, -0.60, 0.80, -0.45, 0.75],
        best_value=0.90,
    )

    assert stable_ok is True
    assert stable_ratio <= 0.30
    assert unstable_ok is False
    assert unstable_ratio > 0.30


def test_outer_trial_stability_gate_bypasses_when_trials_below_minimum():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            nested_outer_stability_ratio_cap=0.01,
            nested_outer_stability_min_trials=10,
        )
    )
    is_stable, stability_ratio = trainer._outer_trial_stability_gate(
        [0.7, -0.7, 0.65],
        best_value=0.7,
    )

    assert is_stable is True
    assert stability_ratio == pytest.approx(0.0)


def test_apply_model_priors_infers_lightgbm_monotonic_constraints():
    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="lightgbm"))
    trainer.feature_names = ["momentum_10", "volatility_20", "atr_14", "rsi_14"]
    trainer.config.model_params = {}

    trainer._apply_model_priors()

    assert trainer.config.model_params["monotone_constraints"] == [1, -1, -1, 0]
    assert trainer.training_metrics["lightgbm_monotone_constraint_count"] == pytest.approx(3.0)


def test_aggregate_fold_objective_penalizes_instability_and_downside():
    stable_score = train_script.ModelTrainer._aggregate_fold_objective(
        [0.9, 1.0, 0.8],
        [1.0, 1.0, 1.0],
    )
    unstable_score = train_script.ModelTrainer._aggregate_fold_objective(
        [0.9, -0.4, 1.1],
        [1.0, 1.0, 1.0],
    )

    assert stable_score > unstable_score


def test_inner_trial_stability_penalty_increases_for_unstable_folds():
    stable_penalty = train_script.ModelTrainer._inner_trial_stability_penalty(
        [0.9, 0.95, 0.88],
        [1.0, 1.0, 1.0],
    )
    unstable_penalty = train_script.ModelTrainer._inner_trial_stability_penalty(
        [0.9, -0.5, 1.1],
        [1.0, 1.0, 1.0],
    )

    assert unstable_penalty > stable_penalty
    assert stable_penalty >= 0.0


def test_parameter_robustness_penalty_penalizes_extreme_xgboost_params():
    extreme = train_script.ModelTrainer._parameter_robustness_penalty(
        "xgboost",
        {
            "max_depth": 7,
            "n_estimators": 760,
            "learning_rate": 0.08,
            "subsample": 0.72,
            "colsample_bytree": 0.72,
            "min_child_weight": 3,
            "scale_pos_weight": 3.0,
        },
    )
    moderate = train_script.ModelTrainer._parameter_robustness_penalty(
        "xgboost",
        {
            "max_depth": 5,
            "n_estimators": 420,
            "learning_rate": 0.04,
            "subsample": 0.82,
            "colsample_bytree": 0.82,
            "min_child_weight": 6,
            "scale_pos_weight": 1.2,
        },
    )

    assert extreme > moderate
    assert moderate >= 0.0


def test_compute_objective_components_adds_tail_risk_penalty():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            objective_weight_tail_risk=1.0,
            objective_expected_shortfall_cap=0.01,
        )
    )

    components = trainer._compute_objective_components(
        sharpe=0.2,
        max_drawdown=0.05,
        turnover=0.1,
        brier_score=0.2,
        trade_count=15,
        cvar=-0.02,
        skew=0.1,
        expected_shortfall=0.04,
        symbol_concentration_hhi=0.50,
        equity_break=0.0,
        evaluation_size=120,
    )

    assert components["objective_tail_risk_penalty"] <= -1.9
    assert components["objective_expected_shortfall_cap"] == pytest.approx(0.01)


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
    assert 0.55 <= metrics["win_rate"] <= 0.75
    assert 0.10 <= metrics["sharpe_observation_confidence"] < 0.40


def test_calculate_fold_metrics_shrinks_sparse_sharpe_extremes():
    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="xgboost"))
    y_proba = np.array([0.95] + [0.50] * 59, dtype=float)
    y_true = np.concatenate([[1.0], np.zeros(59, dtype=float)])
    y_pred = y_true.copy()
    realized = np.array([0.08] + [0.0] * 59, dtype=float)

    metrics = trainer._calculate_fold_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        long_threshold=0.55,
        short_threshold=0.45,
        realized_forward_returns=realized,
    )

    assert metrics["trade_return_observations"] == pytest.approx(1.0)
    assert metrics["sharpe_observation_confidence"] < 0.5
    assert metrics["sharpe"] < 2.0


def test_calculate_fold_metrics_computes_symbol_concentration_penalty():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            objective_weight_symbol_concentration=0.50,
            execution_max_symbol_entry_share=0.95,
        )
    )
    y_proba = np.array([0.80, 0.80, 0.80, 0.80, 0.80, 0.80], dtype=float)
    y_true = np.array([1, 1, 1, 1, 1, 1], dtype=float)
    y_pred = np.array([1, 1, 1, 1, 1, 1], dtype=float)
    symbols = np.array(["AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "MSFT"], dtype=object)
    realized = np.array([0.01, 0.008, 0.011, 0.009, 0.007, 0.001], dtype=float)

    metrics = trainer._calculate_fold_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        long_threshold=0.55,
        short_threshold=0.45,
        realized_forward_returns=realized,
        symbols=symbols,
    )

    assert metrics["symbol_concentration_hhi"] > 0.35
    assert metrics["symbol_effective_count"] >= 1.0
    assert metrics["objective_symbol_concentration_penalty"] < 0.0


def test_calculate_fold_metrics_sorts_execution_inputs_by_timestamp(monkeypatch):
    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="xgboost"))
    captured: dict[str, np.ndarray] = {}

    def _fake_compute_net_returns(
        y_true,
        y_proba,
        long_threshold=0.55,
        short_threshold=0.45,
        realized_forward_returns=None,
        event_net_returns=None,
        timestamps=None,
        symbols=None,
        return_details=False,
    ):
        captured["y_proba"] = np.asarray(y_proba, dtype=float).copy()
        captured["timestamps"] = np.asarray(timestamps).copy() if timestamps is not None else np.array([])
        captured["symbols"] = np.asarray(symbols, dtype=object).copy() if symbols is not None else np.array([])
        n = len(y_proba)
        empty = np.zeros(n, dtype=float)
        details = {
            "positions": np.zeros(n, dtype=float),
            "turnover_series": np.zeros(n, dtype=float),
            "trade_mask": np.zeros(n, dtype=bool),
            "entry_mask": np.zeros(n, dtype=bool),
        }
        return (empty, details) if return_details else empty

    monkeypatch.setattr(trainer, "_compute_net_returns", _fake_compute_net_returns)

    trainer._calculate_fold_metrics(
        y_true=np.array([1.0, 0.0, 1.0], dtype=float),
        y_pred=np.array([1.0, 0.0, 1.0], dtype=float),
        y_proba=np.array([0.9, 0.1, 0.7], dtype=float),
        timestamps=np.array(
            ["2024-01-02T10:00:00Z", "2024-01-01T10:00:00Z", "2024-01-01T11:00:00Z"],
            dtype=object,
        ),
        symbols=np.array(["AAPL", "MSFT", "AAPL"], dtype=object),
    )

    np.testing.assert_allclose(captured["y_proba"], np.array([0.1, 0.7, 0.9], dtype=float))
    assert captured["symbols"].tolist() == ["MSFT", "AAPL", "AAPL"]


def test_calculate_fold_metrics_uses_portfolio_aggregated_series(monkeypatch):
    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="xgboost"))

    def _fake_compute_net_returns(
        y_true,
        y_proba,
        long_threshold=0.55,
        short_threshold=0.45,
        realized_forward_returns=None,
        event_net_returns=None,
        timestamps=None,
        symbols=None,
        return_details=False,
    ):
        n = len(y_proba)
        row_returns = np.array([0.20, -0.20, 0.20, -0.20], dtype=float)[:n]
        details = {
            "positions": np.ones(n, dtype=float),
            "turnover_series": np.zeros(n, dtype=float),
            "trade_mask": np.ones(n, dtype=bool),
            "entry_mask": np.ones(n, dtype=bool),
            "portfolio_returns": np.array([0.01, 0.03], dtype=float),
            "portfolio_turnover_series": np.zeros(2, dtype=float),
            "portfolio_trade_mask": np.ones(2, dtype=bool),
            "portfolio_entry_mask": np.ones(2, dtype=bool),
        }
        return (row_returns, details) if return_details else row_returns

    monkeypatch.setattr(trainer, "_compute_net_returns", _fake_compute_net_returns)

    metrics = trainer._calculate_fold_metrics(
        y_true=np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
        y_pred=np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
        y_proba=np.array([0.9, 0.9, 0.9, 0.9], dtype=float),
    )

    assert metrics["sharpe"] > 1.0


def test_build_execution_profile_applies_symbol_entry_share_cap():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            dynamic_no_trade_band=False,
            execution_cooldown_bars=0,
            execution_max_symbol_entry_share=0.55,
        )
    )
    n = 40
    y_proba = np.asarray([0.8 if i % 2 == 0 else 0.2 for i in range(n)], dtype=float)
    symbols = np.asarray((["AAPL"] * 30) + (["MSFT"] * 10), dtype=object)

    profile = trainer._build_execution_profile(
        y_proba=y_proba,
        long_threshold=0.55,
        short_threshold=0.45,
        symbols=symbols,
    )
    entry_mask = np.asarray(profile["entry_mask"], dtype=bool)
    entry_symbols = symbols[entry_mask]
    if entry_symbols.size > 0:
        max_share = float(pd.Series(entry_symbols).value_counts(normalize=True).max())
        assert max_share <= 0.70


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


def test_derive_signal_thresholds_uses_return_edge_to_suppress_losing_side():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(model_type="xgboost", min_trades=40)
    )
    rng = np.random.default_rng(29)
    train_proba = np.clip(rng.normal(loc=0.56, scale=0.12, size=420), 0.01, 0.99)
    train_labels = (train_proba >= 0.5).astype(float)
    # Positive-only return stream => short side should be penalized/suppressed.
    train_returns = np.abs(rng.normal(loc=0.0015, scale=0.0007, size=420))

    long_threshold, short_threshold = trainer._derive_signal_thresholds(
        train_proba,
        train_labels=train_labels,
        train_returns=train_returns,
    )

    assert long_threshold >= 0.51
    assert short_threshold <= 0.20


def test_derive_signal_thresholds_applies_worst_regime_safety_widening():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(model_type="xgboost", min_trades=40)
    )
    rng = np.random.default_rng(133)
    train_proba = np.clip(rng.normal(loc=0.55, scale=0.11, size=420), 0.01, 0.99)
    train_labels = (train_proba >= 0.5).astype(float)
    regimes = np.array(
        ["bull_trend"] * 210 + ["bear_trend"] * 210,
        dtype=object,
    )

    base_returns = np.where(
        train_proba >= 0.55,
        0.0012,
        np.where(train_proba <= 0.45, -0.0012, 0.0),
    ).astype(float)
    # Worst regime is adversarial to directional signals.
    train_returns = base_returns.copy()
    train_returns[210:] = -train_returns[210:]

    long_no_regime, short_no_regime = trainer._derive_signal_thresholds(
        train_proba,
        train_labels=train_labels,
        train_returns=train_returns,
    )
    long_with_regime, short_with_regime = trainer._derive_signal_thresholds(
        train_proba,
        train_labels=train_labels,
        train_returns=train_returns,
        train_regimes=regimes,
    )

    gap_no_regime = float(long_no_regime - short_no_regime)
    gap_with_regime = float(long_with_regime - short_with_regime)
    assert gap_with_regime >= gap_no_regime


def test_derive_signal_thresholds_becomes_highly_selective_when_both_sides_lose():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(model_type="xgboost", min_trades=40)
    )
    rng = np.random.default_rng(177)
    train_proba = np.clip(rng.normal(loc=0.55, scale=0.13, size=420), 0.01, 0.99)
    train_labels = (rng.random(420) < 0.5).astype(float)
    # Force both directional edges to be negative:
    # long zone gets negative returns, short zone gets positive returns.
    train_returns = np.where(
        train_proba >= 0.55,
        -0.0012,
        np.where(train_proba <= 0.45, 0.0012, 0.0),
    ).astype(float)

    long_threshold, short_threshold = trainer._derive_signal_thresholds(
        train_proba,
        train_labels=train_labels,
        train_returns=train_returns,
    )

    assert long_threshold >= 0.86
    assert short_threshold <= 0.18


def test_derive_signal_thresholds_preserves_activity_for_low_dispersion_scores():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(model_type="xgboost", min_trades=35)
    )
    rng = np.random.default_rng(91)
    train_proba = np.clip(rng.normal(loc=0.50, scale=0.012, size=280), 0.01, 0.99)
    train_labels = (rng.random(280) < 0.50).astype(float)

    long_threshold, short_threshold = trainer._derive_signal_thresholds(
        train_proba,
        train_labels=train_labels,
    )
    signals = np.where(
        train_proba >= long_threshold,
        1.0,
        np.where(train_proba <= short_threshold, -1.0, 0.0),
    )

    assert float(np.mean(signals != 0.0)) >= 0.02
    assert long_threshold <= 0.72
    assert short_threshold >= 0.18


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


def test_select_final_model_prefers_reliability_adjusted_score():
    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="xgboost", min_trades=40))
    models = [object(), object()]
    fold_results = [
        {
            "trade_count": 20.0,
            "test_size": 240.0,
            "risk_adjusted_score": 1.45,
            "trade_return_observations": 2.0,
            "active_signal_rate": 0.02,
            "sharpe": -0.2,
            "win_rate": 0.35,
        },
        {
            "trade_count": 20.0,
            "test_size": 240.0,
            "risk_adjusted_score": 1.20,
            "trade_return_observations": 18.0,
            "active_signal_rate": 0.12,
            "sharpe": 0.6,
            "win_rate": 0.58,
        },
    ]

    trainer._select_final_model(models, fold_results)

    assert trainer.model is models[1]
    assert trainer.training_metrics["selected_cv_fold"] == pytest.approx(2.0)


def test_build_execution_profile_emergency_relaxation_restores_activity():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            min_trades=35,
            dynamic_no_trade_band=True,
        )
    )
    rng = np.random.default_rng(1234)
    y_proba = np.clip(rng.normal(loc=0.50, scale=0.013, size=220), 0.01, 0.99)
    realized = np.full(220, 0.001, dtype=float)

    profile = trainer._build_execution_profile(
        y_proba=y_proba,
        long_threshold=0.62,
        short_threshold=0.38,
        realized_returns=realized,
    )

    assert np.count_nonzero(profile["raw_signals"]) > 0


def test_build_execution_profile_regime_bias_suppresses_counter_trend_side():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            dynamic_no_trade_band=True,
        )
    )
    rng = np.random.default_rng(202)
    trend = np.linspace(0.50, 0.68, 180)
    y_proba = np.clip(trend + rng.normal(loc=0.0, scale=0.01, size=180), 0.01, 0.99)

    profile = trainer._build_execution_profile(
        y_proba=y_proba,
        long_threshold=0.57,
        short_threshold=0.43,
        realized_returns=None,
    )

    assert float(np.mean(profile["short_threshold_series"])) <= 0.34


def test_build_execution_profile_is_realized_return_leakage_safe():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            dynamic_no_trade_band=True,
        )
    )
    rng = np.random.default_rng(99)
    y_proba = np.clip(rng.normal(loc=0.54, scale=0.04, size=160), 0.01, 0.99)
    realized_pos = np.full(160, 0.002, dtype=float)
    realized_neg = np.full(160, -0.002, dtype=float)

    profile_pos = trainer._build_execution_profile(
        y_proba=y_proba,
        long_threshold=0.57,
        short_threshold=0.43,
        realized_returns=realized_pos,
    )
    profile_neg = trainer._build_execution_profile(
        y_proba=y_proba,
        long_threshold=0.57,
        short_threshold=0.43,
        realized_returns=realized_neg,
    )

    np.testing.assert_allclose(profile_pos["long_threshold_series"], profile_neg["long_threshold_series"])
    np.testing.assert_allclose(profile_pos["short_threshold_series"], profile_neg["short_threshold_series"])
    np.testing.assert_allclose(profile_pos["raw_signals"], profile_neg["raw_signals"])
    np.testing.assert_allclose(profile_pos["positions"], profile_neg["positions"])


def test_compute_features_windows_full_pipeline_fails_fast_without_partial_fallback(monkeypatch):
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            allow_feature_group_fallback=False,
            persist_features_to_postgres=False,
        )
    )
    trainer.data = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 4,
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC"),
            "open": [1.0, 1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0, 1.0],
            "volume": [100, 110, 120, 130],
        }
    )

    monkeypatch.setattr(train_script.os, "name", "nt", raising=False)
    monkeypatch.setattr(trainer, "_load_features_from_postgres", lambda: None)
    monkeypatch.setattr(trainer, "_compute_features_full_pipeline", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    fallback_called = {"value": False}
    monkeypatch.setattr(trainer, "_compute_features_fallback", lambda: fallback_called.__setitem__("value", True))

    with pytest.raises(RuntimeError, match="Full feature pipeline failed on Windows"):
        trainer._compute_features()
    assert fallback_called["value"] is False


def test_compute_features_windows_allows_partial_fallback_when_enabled(monkeypatch):
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            allow_feature_group_fallback=True,
            persist_features_to_postgres=False,
        )
    )
    trainer.data = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 4,
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC"),
            "open": [1.0, 1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0, 1.0],
            "volume": [100, 110, 120, 130],
        }
    )

    monkeypatch.setattr(train_script.os, "name", "nt", raising=False)
    monkeypatch.setattr(trainer, "_load_features_from_postgres", lambda: None)
    monkeypatch.setattr(trainer, "_compute_features_full_pipeline", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    fallback_called = {"value": False}
    monkeypatch.setattr(trainer, "_compute_features_fallback", lambda: fallback_called.__setitem__("value", True))

    trainer._compute_features()
    assert fallback_called["value"] is True


def test_finalize_feature_matrix_preserves_rows_with_sparse_full_features():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            feature_reuse_min_coverage=0.4,
        )
    )
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 6,
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="15min", tz="UTC"),
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100, 101, 102, 103, 104, 105],
            "volume": [1000, 1010, 1020, 1030, 1040, 1050],
            "dense_feature": [0.1, np.nan, 0.3, 0.4, np.nan, 0.6],
            "sparse_feature": [np.nan, np.nan, np.nan, np.nan, 1.0, np.nan],
        }
    )

    finalized = trainer._finalize_feature_matrix(frame)

    assert not finalized.empty
    assert "dense_feature" in finalized.columns
    assert "sparse_feature" not in finalized.columns
    assert finalized["dense_feature"].isna().sum() == 0


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
            "--windows-fallback-features",
        ]
    )
    assert args.feature_groups == ["technical", "statistical"]
    assert args.disable_cross_sectional is True
    assert args.strict_feature_groups is True
    assert args.max_cross_sectional_symbols == 14
    assert args.max_cross_sectional_rows == 150000
    assert args.feature_materialization_batch_rows == 4000
    assert args.feature_reuse_min_coverage == 0.3
    assert args.windows_fallback_features is True


def test_build_parser_supports_allow_partial_feature_fallback_flag():
    parser = train_script.build_parser()
    args = parser.parse_args(["--allow-partial-feature-fallback"])
    assert args.allow_partial_feature_fallback is True


def test_build_parser_supports_advanced_risk_and_execution_flags():
    parser = train_script.build_parser()
    args = parser.parse_args(
        [
            "--objective-weight-cvar",
            "0.55",
            "--objective-weight-skew",
            "0.15",
            "--objective-weight-tail-risk",
            "0.9",
            "--objective-weight-symbol-concentration",
            "0.45",
            "--objective-expected-shortfall-cap",
            "0.004",
            "--nested-outer-stability-ratio-cap",
            "1.6",
            "--nested-outer-stability-min-trials",
            "11",
            "--holdout-pct",
            "0.2",
            "--min-holdout-sharpe",
            "0.1",
            "--min-holdout-regime-sharpe",
            "-0.05",
            "--min-holdout-symbol-coverage",
            "0.67",
            "--min-holdout-symbol-p25-sharpe",
            "-0.03",
            "--max-holdout-symbol-underwater-ratio",
            "0.4",
            "--max-holdout-drawdown",
            "0.3",
            "--max-regime-shift",
            "0.25",
            "--max-symbol-concentration-hhi",
            "0.6",
            "--disable-auto-live-profile",
            "--auto-live-profile-symbol-threshold",
            "44",
            "--auto-live-profile-min-years",
            "4.5",
            "--label-edge-cost-buffer-bps",
            "2.5",
            "--disable-symbol-quality-filter",
            "--symbol-quality-min-rows",
            "1400",
            "--symbol-quality-min-symbols",
            "9",
            "--symbol-quality-max-missing-ratio",
            "0.11",
            "--symbol-quality-max-extreme-move-ratio",
            "0.07",
            "--symbol-quality-max-corporate-action-ratio",
            "0.018",
            "--symbol-quality-min-median-dollar-volume",
            "1500000",
            "--disable-dynamic-no-trade-band",
            "--execution-vol-target-daily",
            "0.01",
            "--execution-turnover-cap",
            "0.8",
            "--execution-cooldown-bars",
            "5",
            "--execution-max-symbol-entry-share",
            "0.62",
            "--disable-lightgbm-monotonic-constraints",
        ]
    )
    assert args.objective_weight_cvar == pytest.approx(0.55)
    assert args.objective_weight_skew == pytest.approx(0.15)
    assert args.objective_weight_tail_risk == pytest.approx(0.9)
    assert args.objective_weight_symbol_concentration == pytest.approx(0.45)
    assert args.objective_expected_shortfall_cap == pytest.approx(0.004)
    assert args.nested_outer_stability_ratio_cap == pytest.approx(1.6)
    assert args.nested_outer_stability_min_trials == 11
    assert args.holdout_pct == pytest.approx(0.2)
    assert args.min_holdout_sharpe == pytest.approx(0.1)
    assert args.min_holdout_regime_sharpe == pytest.approx(-0.05)
    assert args.min_holdout_symbol_coverage == pytest.approx(0.67)
    assert args.min_holdout_symbol_p25_sharpe == pytest.approx(-0.03)
    assert args.max_holdout_symbol_underwater_ratio == pytest.approx(0.4)
    assert args.max_holdout_drawdown == pytest.approx(0.3)
    assert args.max_regime_shift == pytest.approx(0.25)
    assert args.max_symbol_concentration_hhi == pytest.approx(0.6)
    assert args.disable_auto_live_profile is True
    assert args.auto_live_profile_symbol_threshold == 44
    assert args.auto_live_profile_min_years == pytest.approx(4.5)
    assert args.disable_symbol_quality_filter is True
    assert args.symbol_quality_min_rows == 1400
    assert args.symbol_quality_min_symbols == 9
    assert args.symbol_quality_max_missing_ratio == pytest.approx(0.11)
    assert args.symbol_quality_max_extreme_move_ratio == pytest.approx(0.07)
    assert args.symbol_quality_max_corporate_action_ratio == pytest.approx(0.018)
    assert args.symbol_quality_min_median_dollar_volume == pytest.approx(1_500_000.0)
    assert args.label_edge_cost_buffer_bps == pytest.approx(2.5)
    assert args.disable_dynamic_no_trade_band is True
    assert args.execution_vol_target_daily == pytest.approx(0.01)
    assert args.execution_turnover_cap == pytest.approx(0.8)
    assert args.execution_cooldown_bars == 5
    assert args.execution_max_symbol_entry_share == pytest.approx(0.62)
    assert args.disable_lightgbm_monotonic_constraints is True


def test_build_parser_supports_horizon_sweep_and_meta_threshold_flags():
    parser = train_script.build_parser()
    args = parser.parse_args(
        [
            "--primary-horizon-sweep",
            "1",
            "5",
            "20",
            "--meta-label-min-confidence",
            "0.63",
            "--disable-meta-dynamic-threshold",
        ]
    )
    assert args.primary_horizon_sweep == [1, 5, 20]
    assert args.meta_label_min_confidence == pytest.approx(0.63)
    assert args.disable_meta_dynamic_threshold is True


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


def test_multiple_testing_correction_records_pbo_diagnostics_and_gate_metric():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            correction_method="deflated_sharpe",
            n_trials=25,
            max_pbo=0.45,
        )
    )
    rng = np.random.default_rng(123)
    trainer.training_metrics = {
        "mean_sharpe": 0.9,
        "holdout_rows": 320.0,
        "holdout_sharpe": 0.1,
    }
    trainer.cv_results = [{"fold": 1}, {"fold": 2}, {"fold": 3}]
    trainer.cv_return_series = [rng.normal(0.0006, 0.011, size=420)]

    trainer._apply_multiple_testing_correction()

    assert "pbo_ci_upper_95" in trainer.training_metrics
    assert "pbo_reliability" in trainer.training_metrics
    assert "effective_pbo_gate_metric" in trainer.training_metrics
    assert float(trainer.training_metrics["effective_pbo_gate_metric"]) >= float(
        trainer.training_metrics["pbo"]
    )


def test_effective_pbo_threshold_tightens_with_holdout_gap():
    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="xgboost", max_pbo=0.45))

    relaxed = trainer._effective_pbo_threshold(
        sample_size=300,
        interpretation="Moderate overfitting risk",
        holdout_gap_ratio=0.0,
        pbo_reliability=0.8,
    )
    tightened = trainer._effective_pbo_threshold(
        sample_size=300,
        interpretation="Moderate overfitting risk",
        holdout_gap_ratio=1.2,
        pbo_reliability=0.2,
    )

    assert tightened < relaxed


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


def test_validate_model_uses_effective_pbo_gate_metric_when_present():
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
        "pbo": 0.20,
        "effective_pbo_gate_metric": 0.55,
        "nested_cv_trace": [{"outer_fold": 1}],
        "objective_component_summary": {"objective_sharpe_component": 0.5},
    }
    trainer.nested_cv_trace = [{"outer_fold": 1}]

    passed = trainer._validate_model()

    assert passed is False
    assert trainer.validation_results["gates"]["max_pbo"][0] is False
    assert trainer.validation_results["gates"]["max_pbo"][1] == pytest.approx(0.55)


def test_validate_model_holdout_sharpe_gate_uses_confidence_adjustment():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            min_holdout_sharpe=0.0,
            use_meta_labeling=False,
        )
    )
    trainer.training_metrics = {
        "mean_sharpe": 1.0,
        "mean_accuracy": 0.65,
        "mean_max_drawdown": 0.10,
        "mean_win_rate": 0.60,
        "mean_trade_count": 12.0,
        "mean_test_size": 80.0,
        "mean_risk_adjusted_score": 0.25,
        "mean_symbol_concentration_hhi": 0.25,
        "holdout_rows": 300.0,
        "holdout_sharpe": 0.12,
        "holdout_trade_return_observations": 2.0,
        "holdout_sharpe_observation_confidence": 0.10,
        "holdout_regime_count_evaluated": 1.0,
        "holdout_worst_regime_sharpe": 0.12,
        "holdout_max_drawdown": 0.12,
        "deflated_sharpe": 0.8,
        "deflated_sharpe_p_value": 0.02,
        "pbo": 0.20,
        "nested_cv_trace": [{"outer_fold": 1}],
        "objective_component_summary": {"objective_sharpe_component": 0.5},
    }
    trainer.nested_cv_trace = [{"outer_fold": 1}]

    passed = trainer._validate_model()

    assert passed is False
    assert trainer.validation_results["gates"]["min_holdout_sharpe"][0] is False
    assert trainer.training_metrics["effective_holdout_sharpe_gate_metric"] < 0.0


def test_validate_model_fails_when_holdout_symbol_coverage_breaches():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            min_holdout_symbol_coverage=0.70,
            use_meta_labeling=False,
        )
    )
    trainer.training_metrics = {
        "mean_sharpe": 1.0,
        "mean_accuracy": 0.65,
        "mean_max_drawdown": 0.10,
        "mean_win_rate": 0.60,
        "mean_trade_count": 14.0,
        "mean_test_size": 80.0,
        "mean_risk_adjusted_score": 0.25,
        "mean_symbol_concentration_hhi": 0.25,
        "holdout_rows": 260.0,
        "holdout_sharpe": 0.25,
        "holdout_worst_regime_sharpe": 0.10,
        "holdout_max_drawdown": 0.12,
        "holdout_symbol_coverage_ratio": 0.45,
        "holdout_symbol_sharpe_p25": 0.05,
        "holdout_symbol_underwater_ratio": 0.20,
        "deflated_sharpe": 0.8,
        "deflated_sharpe_p_value": 0.02,
        "pbo": 0.20,
        "nested_cv_trace": [{"outer_fold": 1}],
        "objective_component_summary": {"objective_sharpe_component": 0.5},
    }
    trainer.nested_cv_trace = [{"outer_fold": 1}]

    passed = trainer._validate_model()

    assert passed is False
    assert trainer.validation_results["gates"]["min_holdout_symbol_coverage"][0] is False


def test_validate_model_fails_when_holdout_symbol_underwater_ratio_breaches():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            max_holdout_symbol_underwater_ratio=0.35,
            use_meta_labeling=False,
        )
    )
    trainer.training_metrics = {
        "mean_sharpe": 1.0,
        "mean_accuracy": 0.65,
        "mean_max_drawdown": 0.10,
        "mean_win_rate": 0.60,
        "mean_trade_count": 14.0,
        "mean_test_size": 80.0,
        "mean_risk_adjusted_score": 0.25,
        "mean_symbol_concentration_hhi": 0.25,
        "holdout_rows": 260.0,
        "holdout_sharpe": 0.25,
        "holdout_worst_regime_sharpe": 0.10,
        "holdout_max_drawdown": 0.12,
        "holdout_symbol_coverage_ratio": 0.85,
        "holdout_symbol_sharpe_p25": 0.05,
        "holdout_symbol_underwater_ratio": 0.60,
        "deflated_sharpe": 0.8,
        "deflated_sharpe_p_value": 0.02,
        "pbo": 0.20,
        "nested_cv_trace": [{"outer_fold": 1}],
        "objective_component_summary": {"objective_sharpe_component": 0.5},
    }
    trainer.nested_cv_trace = [{"outer_fold": 1}]

    passed = trainer._validate_model()

    assert passed is False
    assert trainer.validation_results["gates"]["max_holdout_symbol_underwater_ratio"][0] is False


def test_auto_live_profile_applies_for_5y_46_symbol_dataset():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            n_trials=40,
            n_splits=4,
            nested_outer_splits=3,
            nested_inner_splits=2,
            max_pbo=0.45,
            min_holdout_sharpe=0.0,
            min_holdout_regime_sharpe=-0.10,
        )
    )
    symbols = [f"S{i:02d}" for i in range(46)]
    dates = pd.date_range("2019-01-01", "2024-12-31", freq="10D", tz="UTC")
    rows = [
        {"symbol": symbol, "timestamp": ts}
        for symbol in symbols
        for ts in dates
    ]
    trainer.data = pd.DataFrame(rows)

    trainer._apply_live_multi_symbol_profile()

    assert trainer.training_metrics["auto_live_profile_applied"] is True
    assert trainer.training_metrics["training_profile_mode"] == "institutional_live_multi_symbol"
    assert trainer.config.n_trials >= 180
    assert trainer.config.n_splits >= 6
    assert trainer.config.nested_outer_splits >= 5
    assert trainer.config.max_pbo <= 0.35
    assert trainer.config.min_holdout_sharpe >= 0.10
    assert trainer.config.min_holdout_symbol_coverage >= 0.65


def test_auto_live_profile_skips_small_or_short_dataset():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            n_trials=80,
            auto_live_profile_symbol_threshold=40,
            auto_live_profile_min_years=4.0,
        )
    )
    symbols = [f"S{i:02d}" for i in range(12)]
    dates = pd.date_range("2023-01-01", "2024-01-01", freq="14D", tz="UTC")
    rows = [
        {"symbol": symbol, "timestamp": ts}
        for symbol in symbols
        for ts in dates
    ]
    trainer.data = pd.DataFrame(rows)

    trainer._apply_live_multi_symbol_profile()

    assert trainer.training_metrics["auto_live_profile_applied"] is False
    assert trainer.training_metrics["training_profile_mode"] == "default"
    assert trainer.config.n_trials == 80


def test_validate_model_fails_when_symbol_concentration_gate_breaches():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            max_symbol_concentration_hhi=0.55,
            use_meta_labeling=False,
        )
    )
    trainer.training_metrics = {
        "mean_sharpe": 1.0,
        "mean_accuracy": 0.65,
        "mean_max_drawdown": 0.10,
        "mean_win_rate": 0.60,
        "mean_trade_count": 12.0,
        "mean_test_size": 80.0,
        "mean_risk_adjusted_score": 0.25,
        "mean_symbol_concentration_hhi": 0.80,
        "deflated_sharpe": 0.8,
        "deflated_sharpe_p_value": 0.02,
        "pbo": 0.20,
        "nested_cv_trace": [{"outer_fold": 1}],
        "objective_component_summary": {"objective_sharpe_component": 0.5},
    }
    trainer.nested_cv_trace = [{"outer_fold": 1}]

    passed = trainer._validate_model()

    assert passed is False
    assert trainer.validation_results["gates"]["max_symbol_concentration_hhi"][0] is False


def test_validate_model_fails_when_worst_holdout_regime_sharpe_breaches():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            min_holdout_regime_sharpe=0.05,
            use_meta_labeling=False,
        )
    )
    trainer.training_metrics = {
        "mean_sharpe": 1.0,
        "mean_accuracy": 0.65,
        "mean_max_drawdown": 0.10,
        "mean_win_rate": 0.60,
        "mean_trade_count": 12.0,
        "mean_test_size": 80.0,
        "mean_risk_adjusted_score": 0.25,
        "mean_symbol_concentration_hhi": 0.25,
        "holdout_rows": 240.0,
        "holdout_sharpe": 0.30,
        "holdout_worst_regime_sharpe": -0.20,
        "holdout_max_drawdown": 0.12,
        "deflated_sharpe": 0.8,
        "deflated_sharpe_p_value": 0.02,
        "pbo": 0.20,
        "nested_cv_trace": [{"outer_fold": 1}],
        "objective_component_summary": {"objective_sharpe_component": 0.5},
    }
    trainer.nested_cv_trace = [{"outer_fold": 1}]

    passed = trainer._validate_model()

    assert passed is False
    assert trainer.validation_results["gates"]["min_holdout_regime_sharpe"][0] is False


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


def test_horizon_policy_tightens_thresholds_for_shorter_horizon():
    short_trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(model_type="xgboost", primary_label_horizon=1)
    )
    long_trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(model_type="xgboost", primary_label_horizon=20)
    )
    short_policy = short_trainer._horizon_threshold_policy()
    long_policy = long_trainer._horizon_threshold_policy()

    assert short_policy["target_rate_scale"] < long_policy["target_rate_scale"]
    assert short_policy["min_gap"] > long_policy["min_gap"]
    assert short_policy["forced_long_cap"] < long_policy["forced_long_cap"]
    assert short_policy["forced_short_floor"] > long_policy["forced_short_floor"]


def test_meta_confidence_for_horizon_is_higher_on_short_horizons():
    short_trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            primary_label_horizon=1,
            meta_label_min_confidence=0.55,
        )
    )
    long_trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            primary_label_horizon=25,
            meta_label_min_confidence=0.55,
        )
    )
    assert short_trainer._meta_confidence_for_horizon() == pytest.approx(0.63)
    assert long_trainer._meta_confidence_for_horizon() == pytest.approx(0.53)


def test_train_meta_labeler_uses_oof_predictions(monkeypatch):
    class DummyMetaLabelConfig:
        def __init__(self, model_type, min_confidence=0.55, dynamic_threshold=True):
            self.model_type = model_type
            self.min_confidence = min_confidence
            self.dynamic_threshold = dynamic_threshold

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
    assert trainer.training_metrics["meta_label_min_confidence"] == pytest.approx(0.59)
    assert trainer.training_metrics["meta_label_dynamic_threshold"] == pytest.approx(1.0)


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


def test_compute_net_returns_panel_mode_aggregates_portfolio_series():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            dynamic_no_trade_band=False,
            execution_cooldown_bars=0,
        )
    )
    y_true = np.array([1, 1, 1, 1], dtype=int)
    y_proba = np.array([0.8, 0.8, 0.8, 0.8], dtype=float)
    realized = np.array([0.01, 0.03, 0.02, 0.04], dtype=float)
    timestamps = np.array(
        [
            "2024-01-01T10:00:00Z",
            "2024-01-01T10:00:00Z",
            "2024-01-01T10:05:00Z",
            "2024-01-01T10:05:00Z",
        ],
        dtype=object,
    )
    symbols = np.array(["AAPL", "MSFT", "AAPL", "MSFT"], dtype=object)

    net, details = trainer._compute_net_returns(
        y_true=y_true,
        y_proba=y_proba,
        realized_forward_returns=realized,
        timestamps=timestamps,
        symbols=symbols,
        return_details=True,
    )

    expected = np.array(
        [
            float(np.mean(net[[0, 1]])),
            float(np.mean(net[[2, 3]])),
        ],
        dtype=float,
    )
    np.testing.assert_allclose(
        np.asarray(details["portfolio_returns"], dtype=float),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.asarray(details["portfolio_trade_mask"], dtype=bool).shape[0] == 2
