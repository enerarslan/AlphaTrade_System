"""Unit tests for scripts/train.py."""

from __future__ import annotations

import json
import logging
import sys
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from quant_trading_system.models.training_lineage import (
    build_data_quality_report,
    build_snapshot_manifest,
    compute_data_quality_hash,
    persist_dataset_snapshot_bundle,
    register_training_model_version,
)
from scripts import train as train_script


def _base_args(**overrides):
    args = {
        "model": "xgboost",
        "name": "",
        "training_profile": "promotion",
        "symbols": [],
        "symbols_file": None,
        "start": "",
        "end": "",
        "start_date": "",
        "end_date": "",
        "timeframe": "15Min",
        "include_premarket": False,
        "include_postmarket": False,
        "timeframes": [],
        "cv_method": "purged_kfold",
        "n_splits": 2,
        "embargo_pct": 0.01,
        "n_trials": 1,
        "optuna_storage_dir": None,
        "disable_optuna_resume": False,
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
        "objective_weight_turnover": 0.2,
        "objective_weight_calibration": 0.35,
        "objective_weight_trade_activity": 1.0,
        "objective_weight_cvar": 0.4,
        "objective_weight_skew": 0.1,
        "objective_weight_tail_risk": 0.35,
        "objective_weight_symbol_concentration": 0.20,
        "objective_expected_shortfall_cap": 0.012,
        "replay_manifest": None,
        "dataset_snapshot_bundle": None,
        "strict_snapshot_replay": False,
        "force_promotion_precheck_bypass": False,
        "disable_auto_snapshot_reuse": False,
        "gpu": False,
        "use_gpu": False,
        "require_gpu": False,
        "no_database": False,
        "no_redis_cache": False,
        "no_shap": False,
        "output_dir": "models",
        "label_horizons": [1, 5, 20],
        "primary_horizon": 5,
        "primary_horizon_sweep": [],
        "disable_uniqueness_weighting": False,
        "label_uniqueness_weight_floor": 0.25,
        "disable_volatility_inverse_weighting": False,
        "label_volatility_weight_cap": 2.5,
        "meta_label_min_confidence": 0.55,
        "disable_meta_dynamic_threshold": False,
        "disable_meta_labeling": False,
        "disable_probability_calibration": False,
        "probability_calibration_method": "isotonic",
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
        "training_bar_mode": "time",
        "intrinsic_bar_type": "volume",
        "intrinsic_bar_threshold": 0.0,
        "intrinsic_target_bars_per_day": 100,
        "disable_symbol_quality_filter": False,
        "target_universe_size": 0,
        "universe_selection_buffer_size": 24,
        "symbol_quality_min_rows": 1200,
        "symbol_quality_min_symbols": 8,
        "symbol_quality_max_missing_ratio": 0.12,
        "symbol_quality_max_extreme_move_ratio": 0.08,
        "symbol_quality_max_corporate_action_ratio": 0.02,
        "symbol_quality_min_median_dollar_volume": 1_000_000.0,
        "disable_cross_sectional": False,
        "disable_reference_features": False,
        "disable_tick_microstructure_features": False,
        "strict_feature_groups": False,
        "allow_partial_feature_fallback": False,
        "max_cross_sectional_symbols": 20,
        "max_cross_sectional_rows": 500000,
        "feature_materialization_batch_rows": 5000,
        "feature_reuse_min_coverage": 0.20,
        "skip_feature_persist": False,
        "feature_set_id": "default",
        "disable_feature_selection": False,
        "feature_selection_min_ic": 0.015,
        "feature_selection_max_corr": 0.90,
        "feature_selection_max_features": 180,
        "feature_selection_stability_iterations": 16,
        "feature_selection_min_stability_support": 0.60,
        "windows_fallback_features": False,
        "disable_dynamic_no_trade_band": False,
        "execution_vol_target_daily": 0.012,
        "execution_turnover_cap": 0.60,
        "execution_cooldown_bars": 2,
        "execution_max_symbol_entry_share": 0.68,
        "min_confidence_position_scale": 0.20,
        "warm_start_model": None,
        "min_accuracy": 0.45,
        "min_trades": 100,
        "min_deflated_sharpe": 0.10,
        "max_deflated_sharpe_pvalue": 0.10,
        "max_pbo": 0.45,
        "min_white_reality_stat": 0.0,
        "max_white_reality_pvalue": 0.10,
        "disable_lightgbm_monotonic_constraints": False,
    }
    args.update(overrides)
    return Namespace(**args)


def test_extract_base_symbol():
    assert train_script.ModelTrainer._extract_base_symbol("AAPL_15MIN") == "AAPL"
    assert train_script.ModelTrainer._extract_base_symbol("BRK.B_15MIN") == "BRK.B"
    assert train_script.ModelTrainer._extract_base_symbol("MSFT") == "MSFT"


def test_verify_gpu_stack_allows_tree_gpu_without_torch_cuda(monkeypatch):
    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

    class _FakeTorch:
        cuda = _FakeCuda()

    class _FakeLGBMClassifier:
        def __init__(self, **kwargs):
            assert kwargs["device"] == "cuda"

        def fit(self, X, y):
            assert len(X) == 128
            assert len(y) == 128

    class _FakeLightGBMModule:
        LGBMClassifier = _FakeLGBMClassifier

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setitem(sys.modules, "lightgbm", _FakeLightGBMModule())

    assert train_script._verify_gpu_stack(["lightgbm"]) is True


def test_verify_gpu_stack_requires_cuda_for_deep_models(monkeypatch):
    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

    class _FakeTorch:
        cuda = _FakeCuda()

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())

    with pytest.raises(RuntimeError, match="CUDA GPU not detected"):
        train_script._verify_gpu_stack(["tcn"])


def test_training_config_normalizes_timeframes_and_modes():
    config = train_script.TrainingConfig(
        model_type="xgboost",
        timeframes=["1d", "1h", "15m"],
        training_bar_mode="intrinsic",
        intrinsic_bar_type="volume",
    )

    assert config.timeframes == ["15Min", "1Hour", "1Day"]
    assert config.training_bar_mode == "intrinsic"


def test_training_config_normalizes_market_session_flags():
    config = train_script.TrainingConfig(
        model_type="xgboost",
        include_premarket=1,
        include_postmarket="",
    )

    assert config.include_premarket is True
    assert config.include_postmarket is False


def test_training_config_requires_enabled_gpu_when_gpu_is_mandatory():
    with pytest.raises(ValueError, match="require_gpu=True requires use_gpu=True"):
        train_script.TrainingConfig(
            model_type="lightgbm",
            use_gpu=False,
            require_gpu=True,
        )


def test_training_config_normalizes_universe_reference_symbols():
    config = train_script.TrainingConfig(
        model_type="xgboost",
        universe_reference_symbols=[" spy ", "qqq", "SPY", ""],
    )

    assert config.universe_reference_symbols == ["QQQ", "SPY"]


def test_select_candidate_symbols_from_postgres_prioritizes_timeframe_coverage():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            timeframe="15Min",
            timeframes=["1Hour", "1Day"],
            target_universe_size=2,
            universe_selection_buffer_size=1,
            enable_cross_sectional=False,
            enable_reference_features=False,
        )
    )

    class _FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return list(self._rows)

    class _FakeSession:
        def __init__(self):
            self.calls: list[tuple[str, dict[str, object]]] = []

        def execute(self, statement, params):
            sql = str(statement)
            self.calls.append((sql, dict(params)))
            assert "timeframe_coverage" in sql
            return _FakeResult(
                [
                    ("AAPL", 3200, 250, 5_000_000.0, 3),
                    ("MSFT", 3300, 255, 4_800_000.0, 2),
                    ("TSLA", 3100, 245, 4_600_000.0, 1),
                ]
            )

    class _FakeRedis:
        def __init__(self):
            self.saved_key = ""
            self.saved_payload = ""

        def get(self, key):
            return None

        def set(self, key, value, expire_seconds=None):
            self.saved_key = key
            self.saved_payload = value

    session = _FakeSession()
    redis_mgr = _FakeRedis()

    symbols = trainer._select_candidate_symbols_from_postgres(
        session,
        redis_mgr,
        timeframe="15Min",
        requested_timeframes=["15Min", "1Hour", "1Day"],
        start_date=pd.Timestamp("2024-01-01T00:00:00Z"),
        end_date=pd.Timestamp("2024-12-31T23:59:59Z"),
    )

    assert symbols == ["AAPL", "MSFT", "TSLA"]
    assert trainer.training_metrics["database_universe_requested_timeframes"] == [
        "15Min",
        "1Hour",
        "1Day",
    ]
    assert trainer.training_metrics["database_universe_full_timeframe_coverage_count"] == 1.0
    assert trainer.training_metrics["database_universe_selection_source"] == "liquidity_ranked"
    assert session.calls[0][1]["requested_timeframes"] == ["15Min", "1Hour", "1Day"]
    assert "15Min,1Hour,1Day" in redis_mgr.saved_key
    assert json.loads(redis_mgr.saved_payload) == ["AAPL", "MSFT", "TSLA"]


def test_store_features_to_postgres_uses_development_frame_identifiers(monkeypatch, tmp_path):
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
    )
    trainer.features = pd.DataFrame({"alpha": [0.1, 0.2], "beta": [1.0, 2.0]})
    trainer.development_frame = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "timestamp": pd.to_datetime(
                ["2024-01-02T14:30:00Z", "2024-01-02T14:45:00Z"],
                utc=True,
            ),
            "open": [100.0, 200.0],
            "high": [101.0, 201.0],
            "low": [99.5, 199.5],
            "close": [100.5, 200.5],
            "volume": [1000, 2000],
            "alpha": [0.1, 0.2],
            "beta": [1.0, 2.0],
        }
    )
    trainer.data = trainer.development_frame[
        ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    ].copy()

    class _FakeCursor:
        def __init__(self):
            self.executed: list[tuple[str, object]] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            self.executed.append((str(sql), params))

    class _FakeRawConnection:
        def __init__(self):
            self.cursor_obj = _FakeCursor()
            self.commits = 0
            self.closed = False

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            self.commits += 1

        def close(self):
            self.closed = True

    fake_raw_connection = _FakeRawConnection()
    fake_db_manager = SimpleNamespace(
        engine=SimpleNamespace(raw_connection=lambda: fake_raw_connection)
    )
    fake_redis = SimpleNamespace(
        delete=MagicMock(),
        set=MagicMock(),
    )

    import quant_trading_system.database.connection as db_connection

    monkeypatch.setattr(db_connection, "get_db_manager", lambda: fake_db_manager)
    monkeypatch.setattr(db_connection, "get_redis_manager", lambda: fake_redis)
    monkeypatch.setattr(
        trainer,
        "_feature_materialization_checkpoint_path",
        lambda: tmp_path / "feature_checkpoint.json",
    )
    monkeypatch.setattr(
        trainer,
        "_resolve_feature_materialization_source_batch_rows",
        lambda feature_count: 100,
    )
    monkeypatch.setattr(
        trainer,
        "_copy_feature_rows_to_stage",
        lambda cursor, stage_table, long_batch, timeframe, feature_set_id: len(long_batch),
    )

    trainer._store_features_to_postgres()

    delete_calls = [
        params for sql, params in fake_raw_connection.cursor_obj.executed if "DELETE FROM features" in sql
    ]
    assert delete_calls
    assert delete_calls[0][0] == ["AAPL", "MSFT"]
    fake_redis.set.assert_called()
    assert trainer.features_persisted_to_postgres is True


def test_symbol_missing_ratio_ignores_daily_weekends():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-03T21:00:00Z",
                    "2025-01-06T21:00:00Z",
                    "2025-01-07T21:00:00Z",
                ],
                utc=True,
            )
        }
    )

    ratio = train_script.ModelTrainer._symbol_missing_ratio(frame)

    assert ratio == pytest.approx(0.0)


def test_apply_corporate_action_adjustments_back_adjusts_splits(monkeypatch):
    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="xgboost"))
    trainer.data = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "timestamp": pd.to_datetime(["2025-01-09T21:00:00Z", "2025-01-10T21:00:00Z"], utc=True),
            "open": [396.0, 99.0],
            "high": [404.0, 101.0],
            "low": [392.0, 98.0],
            "close": [400.0, 100.0],
            "volume": [100.0, 400.0],
        }
    )
    monkeypatch.setattr(
        trainer,
        "_load_corporate_actions_for_adjustment",
        lambda **kwargs: pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "action_type": ["SPLIT"],
                "ex_date": [pd.Timestamp("2025-01-10").date()],
                "amount": [0.0],
                "split_ratio": [4.0],
            }
        ),
    )

    trainer._apply_corporate_action_adjustments()

    assert trainer.data.loc[0, "close"] == pytest.approx(100.0)
    assert trainer.data.loc[0, "open"] == pytest.approx(99.0)
    assert trainer.data.loc[0, "volume"] == pytest.approx(400.0)
    assert trainer.training_metrics["corporate_action_adjustment_split_events"] == pytest.approx(
        1.0
    )


def test_apply_corporate_action_adjustments_back_adjusts_dividends(monkeypatch):
    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="xgboost"))
    trainer.data = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "timestamp": pd.to_datetime(["2025-01-09T21:00:00Z", "2025-01-10T21:00:00Z"], utc=True),
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.0, 99.0],
            "volume": [1000.0, 1100.0],
        }
    )
    monkeypatch.setattr(
        trainer,
        "_load_corporate_actions_for_adjustment",
        lambda **kwargs: pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "action_type": ["DIVIDEND"],
                "ex_date": [pd.Timestamp("2025-01-10").date()],
                "amount": [1.0],
                "split_ratio": [0.0],
            }
        ),
    )

    trainer._apply_corporate_action_adjustments()

    assert trainer.data.loc[0, "close"] == pytest.approx(99.0)
    assert trainer.data.loc[0, "open"] == pytest.approx(99.0)
    assert trainer.data.loc[0, "volume"] == pytest.approx(1000.0)
    assert trainer.training_metrics["corporate_action_adjustment_dividend_events"] == pytest.approx(
        1.0
    )


def test_verify_institutional_infra_rejects_missing_ohlcv_table(monkeypatch):
    import quant_trading_system.database.connection as conn_module

    fake_db = MagicMock()
    fake_db.health_check.return_value = True
    fake_db.engine = object()

    fake_redis = MagicMock()
    fake_redis.health_check.return_value = True

    monkeypatch.setattr(conn_module, "get_db_manager", lambda: fake_db)
    monkeypatch.setattr(conn_module, "get_redis_manager", lambda: fake_redis)

    class FakeInspector:
        def get_table_names(self):
            return ["features"]

    monkeypatch.setattr(train_script, "sa_inspect", lambda engine: FakeInspector())

    config = train_script.TrainingConfig(
        model_type="xgboost",
        enable_reference_features=False,
        adjust_prices_for_corporate_actions=False,
    )

    with pytest.raises(RuntimeError, match="Missing required tables: ohlcv_bars"):
        train_script._verify_institutional_infra(config)


def test_verify_institutional_infra_rejects_empty_ohlcv_table(monkeypatch):
    import quant_trading_system.database.connection as conn_module

    fake_db = MagicMock()
    fake_db.health_check.return_value = True
    fake_db.engine = object()
    fake_session = MagicMock()
    fake_session.execute.return_value.scalar.return_value = None
    fake_db.session.return_value = nullcontext(fake_session)

    fake_redis = MagicMock()
    fake_redis.health_check.return_value = True

    monkeypatch.setattr(conn_module, "get_db_manager", lambda: fake_db)
    monkeypatch.setattr(conn_module, "get_redis_manager", lambda: fake_redis)

    class FakeInspector:
        def get_table_names(self):
            return ["ohlcv_bars", "features"]

        def get_columns(self, table_name):
            if table_name == "ohlcv_bars":
                return [
                    {"name": name}
                    for name in (
                        "symbol",
                        "timestamp",
                        "timeframe",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    )
                ]
            if table_name == "features":
                return [
                    {"name": name}
                    for name in (
                        "symbol",
                        "timestamp",
                        "timeframe",
                        "feature_name",
                        "feature_set_id",
                        "value",
                    )
                ]
            return []

    monkeypatch.setattr(train_script, "sa_inspect", lambda engine: FakeInspector())

    config = train_script.TrainingConfig(
        model_type="xgboost",
        enable_reference_features=False,
        adjust_prices_for_corporate_actions=False,
    )

    with pytest.raises(RuntimeError, match="ohlcv_bars table is empty"):
        train_script._verify_institutional_infra(config)


def test_verify_institutional_infra_rejects_missing_timeframe_schema_columns(monkeypatch):
    import quant_trading_system.database.connection as conn_module

    fake_db = MagicMock()
    fake_db.health_check.return_value = True
    fake_db.engine = object()
    fake_session = MagicMock()
    fake_session.execute.return_value.scalar.return_value = 1
    fake_db.session.return_value = nullcontext(fake_session)

    fake_redis = MagicMock()
    fake_redis.health_check.return_value = True

    monkeypatch.setattr(conn_module, "get_db_manager", lambda: fake_db)
    monkeypatch.setattr(conn_module, "get_redis_manager", lambda: fake_redis)

    class FakeInspector:
        def get_table_names(self):
            return ["ohlcv_bars", "features"]

        def get_columns(self, table_name):
            if table_name == "ohlcv_bars":
                return [
                    {"name": name}
                    for name in ("symbol", "timestamp", "open", "high", "low", "close", "volume")
                ]
            if table_name == "features":
                return [{"name": name} for name in ("symbol", "timestamp", "feature_name", "value")]
            return []

    monkeypatch.setattr(train_script, "sa_inspect", lambda engine: FakeInspector())

    config = train_script.TrainingConfig(
        model_type="xgboost",
        enable_reference_features=False,
        adjust_prices_for_corporate_actions=False,
    )

    with pytest.raises(RuntimeError, match="Missing required columns"):
        train_script._verify_institutional_infra(config)


def test_verify_institutional_infra_allows_reference_fallback_when_optional_tables_missing(
    monkeypatch,
    caplog,
):
    import quant_trading_system.database.connection as conn_module

    fake_db = MagicMock()
    fake_db.health_check.return_value = True
    fake_db.engine = object()
    fake_session = MagicMock()
    fake_session.execute.return_value.scalar.return_value = 1
    fake_db.session.return_value = nullcontext(fake_session)

    fake_redis = MagicMock()
    fake_redis.health_check.return_value = True

    monkeypatch.setattr(conn_module, "get_db_manager", lambda: fake_db)
    monkeypatch.setattr(conn_module, "get_redis_manager", lambda: fake_redis)

    class FakeInspector:
        def get_table_names(self):
            return ["ohlcv_bars", "features"]

        def get_columns(self, table_name):
            if table_name == "ohlcv_bars":
                return [
                    {"name": name}
                    for name in (
                        "symbol",
                        "timestamp",
                        "timeframe",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    )
                ]
            if table_name == "features":
                return [
                    {"name": name}
                    for name in (
                        "symbol",
                        "timestamp",
                        "timeframe",
                        "feature_name",
                        "feature_set_id",
                        "value",
                    )
                ]
            return []

    monkeypatch.setattr(train_script, "sa_inspect", lambda engine: FakeInspector())

    config = train_script.TrainingConfig(
        model_type="xgboost",
        enable_reference_features=True,
        allow_feature_group_fallback=True,
        adjust_prices_for_corporate_actions=False,
    )

    with caplog.at_level(logging.WARNING):
        train_script._verify_institutional_infra(config)

    assert "Optional reference tables unavailable" in caplog.text


def test_run_training_uses_start_end_alias(monkeypatch):
    captured = {}

    class DummyModelTrainer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

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
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

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


def test_run_training_loads_symbols_from_file(monkeypatch, tmp_path):
    captured = {}

    class DummyModelTrainer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

    symbols_path = tmp_path / "symbols.txt"
    symbols_path.write_text("aapl\nmsft, nvda\n", encoding="utf-8")

    args = _base_args(symbols_file=symbols_path)
    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert captured["config"].symbols == ["AAPL", "MSFT", "NVDA"]


def test_generate_cv_splits_passes_times_to_splitter(monkeypatch):
    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="xgboost"))
    trainer.timestamps = pd.to_datetime(
        [
            "2025-01-02T00:00:00Z",
            "2025-01-03T00:00:00Z",
            "2025-01-06T00:00:00Z",
            "2025-01-07T00:00:00Z",
        ],
        utc=True,
    ).to_numpy()
    captured = {}

    class _DummySplitter:
        def split(self, X, y, times=None):
            captured["times"] = times
            yield np.array([0, 1]), np.array([2, 3])

    monkeypatch.setattr(trainer, "_get_cv_splitter", lambda n_samples=None: _DummySplitter())

    splits = trainer._generate_cv_splits(np.zeros((4, 2)), np.zeros(4))

    assert len(splits) == 1
    assert captured["times"] is not None


def test_run_training_maps_allow_partial_feature_fallback(monkeypatch):
    captured = {}

    class DummyModelTrainer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

    args = _base_args(allow_partial_feature_fallback=True, strict_feature_groups=False)
    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert captured["config"].allow_feature_group_fallback is True


def test_auto_live_profile_does_not_override_explicit_cross_sectional_disable():
    config = train_script.TrainingConfig(
        model_type="xgboost",
        enable_cross_sectional=False,
        cross_sectional_user_locked=True,
        auto_live_profile_symbol_threshold=2,
        auto_live_profile_min_years=1.0,
    )
    trainer = train_script.ModelTrainer(config)
    trainer.data = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "timestamp": pd.to_datetime(
                [
                    "2021-01-01T00:00:00Z",
                    "2021-01-01T00:00:00Z",
                    "2023-01-01T00:00:00Z",
                    "2023-01-01T00:00:00Z",
                ],
                utc=True,
            ),
            "open": [1.0, 1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0, 1.0],
            "volume": [1_000.0, 1_000.0, 1_000.0, 1_000.0],
        }
    )

    trainer._apply_live_multi_symbol_profile()

    assert trainer.training_metrics["auto_live_profile_applied"] is True
    assert trainer.config.enable_cross_sectional is False


def test_run_training_maps_tail_risk_and_monotonic_flags(monkeypatch):
    captured = {}

    class DummyModelTrainer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

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
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

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
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

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
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

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
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

    args = _base_args(model="all")
    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert trained == [
        "xgboost",
        "lightgbm",
        "lightgbm_ranker",
        "xgboost_regressor",
        "lightgbm_regressor",
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
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

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


def test_run_training_promotion_profile_rejects_optional_governance_disables():
    assert train_script.run_training(_base_args(no_shap=True)) == 1
    assert train_script.run_training(_base_args(disable_meta_labeling=True)) == 1


def test_run_training_blocks_strict_snapshot_promotion_without_ready_research_candidate(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

    class DummyModelTrainer:
        def __init__(self, _config):
            raise AssertionError("promotion precheck should block before trainer construction")

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)

    benchmark_dir = tmp_path / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "training_matrix_20260402T000000Z.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "model_type": "lightgbm_ranker",
                        "model_name": "wave1_ranker_research_h12",
                        "training_profile": "research",
                        "run_id": "wave1_ranker_research_h12",
                        "primary_label_horizon": 12,
                        "snapshot_id": "snap_bad",
                        "dataset_bundle_hash": "bundle_bad",
                        "data_quality_report_hash": "dq_bad",
                        "success": False,
                        "mean_trade_count": 11.0,
                        "mean_risk_adjusted_score": -2.0,
                        "holdout_max_drawdown": 0.43,
                        "holdout_symbol_sharpe_p25": -1.18,
                        "pbo": 0.60,
                        "white_reality_pvalue": 0.12,
                        "expected_edge_policy_trained": 0.0,
                        "symbol_quality_dropped_symbols": 1.0,
                        "data_quality_report_passed": 0.0,
                        "validation_results": {
                            "layers": {
                                "model_utility": {"passed": False},
                                "execution_robustness": {"passed": False},
                                "cross_symbol_robustness": {"passed": False},
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    bundle_dir = tmp_path / "snapshots" / "snap_bad"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_dir / "dataset_bundle.manifest.json"
    bundle_path.write_text(
        json.dumps(
            {
                "snapshot_id": "snap_bad",
                "bundle_hash": "bundle_bad",
                "snapshot_manifest": {"data_quality_report_hash": "dq_bad"},
            }
        ),
        encoding="utf-8",
    )

    args = _base_args(
        model="lightgbm_ranker",
        output_dir=str(tmp_path),
        dataset_snapshot_bundle=str(bundle_path),
        strict_snapshot_replay=True,
        primary_horizon=12,
    )

    assert train_script.run_training(args) == 1


def test_run_training_allows_bypassed_strict_snapshot_promotion_precheck(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

    class DummyModelTrainer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    bundle_dir = tmp_path / "snapshots" / "snap_clean"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_dir / "dataset_bundle.manifest.json"
    bundle_path.write_text(
        json.dumps(
            {
                "snapshot_id": "snap_clean",
                "bundle_hash": "bundle_clean",
                "snapshot_manifest": {"data_quality_report_hash": "dq_clean"},
            }
        ),
        encoding="utf-8",
    )

    args = _base_args(
        model="lightgbm_ranker",
        output_dir=str(tmp_path),
        dataset_snapshot_bundle=str(bundle_path),
        strict_snapshot_replay=True,
        force_promotion_precheck_bypass=True,
    )

    assert train_script.run_training(args) == 0
    assert captured["config"].dataset_snapshot_bundle_path == str(bundle_path)


def test_run_training_rejects_disable_nested_flag():
    args = _base_args(disable_nested_walk_forward=True)
    exit_code = train_script.run_training(args)
    assert exit_code == 1


def test_run_training_research_profile_applies_fast_presets(monkeypatch):
    captured = {}

    class DummyModelTrainer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

    args = _base_args(
        training_profile="research",
        model="lightgbm",
        n_trials=100,
        n_splits=5,
        nested_outer_splits=4,
        nested_inner_splits=3,
        feature_selection_stability_iterations=16,
    )
    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert captured["config"].training_profile == "research"
    assert captured["config"].n_trials == 30
    assert captured["config"].n_splits == 3
    assert captured["config"].nested_outer_splits == 2
    assert captured["config"].nested_inner_splits == 2
    assert captured["config"].use_meta_labeling is False
    assert captured["config"].compute_shap is False
    assert captured["config"].auto_live_profile_enabled is False
    assert captured["config"].feature_selection_stability_iterations == 12


def test_run_training_research_profile_preserves_explicit_budget_overrides(monkeypatch):
    captured = {}

    class DummyModelTrainer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

    args = _base_args(
        training_profile="research",
        model="lightgbm",
        n_trials=9,
        feature_selection_stability_iterations=5,
    )
    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert captured["config"].n_trials == 9
    assert captured["config"].feature_selection_stability_iterations == 5


def test_run_training_research_profile_preserves_explicit_cli_overrides_equal_to_defaults(
    monkeypatch,
):
    captured = {}

    class DummyModelTrainer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

    parser = train_script.build_parser()
    argv = [
        "--training-profile",
        "research",
        "--model",
        "lightgbm",
        "--n-trials",
        "100",
        "--n-splits",
        "5",
        "--nested-outer-splits",
        "4",
        "--nested-inner-splits",
        "3",
        "--feature-selection-stability-iterations",
        "16",
    ]
    args = parser.parse_args(argv)
    train_script._attach_explicit_profile_overrides(parser, args, argv)

    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert captured["config"].n_trials == 100
    assert captured["config"].n_splits == 5
    assert captured["config"].nested_outer_splits == 4
    assert captured["config"].nested_inner_splits == 3
    assert captured["config"].feature_selection_stability_iterations == 16
    assert captured["config"].use_meta_labeling is False
    assert captured["config"].compute_shap is False


def test_run_training_primary_challenger_alias_dispatches_lightgbm_and_tcn(monkeypatch):
    trained: list[str] = []

    class DummyModelTrainer:
        def __init__(self, config):
            self.config = config
            trained.append(config.model_type)

        def run(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

    args = _base_args(model="primary_challenger", training_profile="research")
    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert trained == ["lightgbm", "tcn"]


def test_run_training_auto_reuses_matching_snapshot_bundle(monkeypatch, tmp_path):
    captured = {}

    class DummyModelTrainer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

    scope = train_script._build_snapshot_training_scope(
        train_script.TrainingConfig(model_type="lightgbm", output_dir=str(tmp_path))
    )
    bundle_path = tmp_path / "snapshots" / "snap_123" / "dataset_bundle.manifest.json"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(
        json.dumps(
            {
                "created_at": "2026-04-01T10:00:00+00:00",
                "snapshot_id": "snap_123",
                "training_scope": scope,
                "snapshot_manifest": {},
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )

    args = _base_args(
        model="lightgbm",
        output_dir=str(tmp_path),
        n_trials=100,
        n_splits=5,
        nested_outer_splits=4,
        nested_inner_splits=3,
        feature_selection_stability_iterations=16,
    )
    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert captured["config"].dataset_snapshot_bundle_path == str(bundle_path)


def test_run_training_can_disable_auto_snapshot_reuse(monkeypatch, tmp_path):
    captured = {}

    class DummyModelTrainer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

    scope = train_script._build_snapshot_training_scope(
        train_script.TrainingConfig(model_type="lightgbm", output_dir=str(tmp_path))
    )
    bundle_path = tmp_path / "snapshots" / "snap_456" / "dataset_bundle.manifest.json"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(
        json.dumps(
            {
                "created_at": "2026-04-01T10:00:00+00:00",
                "snapshot_id": "snap_456",
                "training_scope": scope,
                "snapshot_manifest": {},
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )

    args = _base_args(
        model="lightgbm",
        output_dir=str(tmp_path),
        disable_auto_snapshot_reuse=True,
        n_trials=100,
        n_splits=5,
        nested_outer_splits=4,
        nested_inner_splits=3,
        feature_selection_stability_iterations=16,
    )
    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert captured["config"].dataset_snapshot_bundle_path == ""


def test_run_training_passes_market_session_flags(monkeypatch):
    captured = {}

    class DummyModelTrainer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

    exit_code = train_script.run_training(
        _base_args(
            model="lightgbm",
            training_profile="research",
            include_premarket=True,
            include_postmarket=True,
        )
    )

    assert exit_code == 0
    assert captured["config"].include_premarket is True
    assert captured["config"].include_postmarket is True


def test_run_training_passes_optuna_resume_settings(monkeypatch, tmp_path):
    captured = {}

    class DummyModelTrainer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

    args = _base_args(
        model="lightgbm",
        output_dir=str(tmp_path),
        optuna_storage_dir=str(tmp_path / "optuna_state"),
        disable_optuna_resume=True,
        n_trials=100,
        n_splits=5,
        nested_outer_splits=4,
        nested_inner_splits=3,
        feature_selection_stability_iterations=16,
    )
    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert captured["config"].optuna_storage_dir == str(tmp_path / "optuna_state")
    assert captured["config"].optuna_resume_enabled is False


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


def test_prepare_optuna_study_uses_sqlite_storage_for_resume(tmp_path):
    captured: dict[str, object] = {}

    trial_state = SimpleNamespace(
        COMPLETE="complete",
        PRUNED="pruned",
        FAIL="fail",
        RUNNING="running",
        WAITING="waiting",
    )

    class DummyStudy:
        def __init__(self):
            self.trials = [
                SimpleNamespace(state=trial_state.COMPLETE),
                SimpleNamespace(state=trial_state.RUNNING),
            ]

    def _create_study(**kwargs):
        captured.update(kwargs)
        return DummyStudy()

    fake_optuna = SimpleNamespace(
        create_study=_create_study,
        trial=SimpleNamespace(TrialState=trial_state),
    )
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="lightgbm_ranker",
            model_name="resume_test",
            output_dir=str(tmp_path),
            n_trials=12,
        )
    )
    trainer.snapshot_manifest = {"snapshot_id": "snap_123"}

    _study, info = trainer._prepare_optuna_study(
        fake_optuna,
        namespace="nested_outer_01",
        direction="maximize",
        sampler=object(),
        pruner=object(),
    )

    assert captured["load_if_exists"] is True
    assert str(captured["storage"]).startswith("sqlite:///")
    assert info["resume_enabled"] is True
    assert info["resumed_from_storage"] is True
    assert info["existing_trials"] == 2
    assert info["finalized_trials"] == 1
    assert info["remaining_trials"] == 11
    assert Path(info["manifest_path"]).exists()
    assert str(info["storage_path"]).endswith(".sqlite3")


def test_fold_reliability_weight_penalizes_low_support():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(model_type="xgboost", min_trades=40)
    )
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
    assert (
        train_script.ModelTrainer._has_binary_class_support(np.array([0, 1, 0, 1], dtype=float))
        is True
    )
    assert (
        train_script.ModelTrainer._has_binary_class_support(np.array([1, 1, 1], dtype=float))
        is False
    )
    assert (
        train_script.ModelTrainer._has_binary_class_support(np.array([0, 0, 0], dtype=float))
        is False
    )


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


def test_feature_selection_reapplies_lightgbm_monotonic_constraints(monkeypatch):
    import quant_trading_system.models.feature_selection as feature_selection_module

    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="lightgbm"))
    trainer.features = pd.DataFrame(
        {
            "momentum_10": [0.1, 0.2, 0.3, 0.4],
            "volatility_20": [1.0, 0.9, 1.1, 1.0],
            "rsi_14": [40.0, 42.0, 39.0, 43.0],
        }
    )
    trainer.labels = np.array([0, 1, 0, 1], dtype=int)
    trainer.feature_names = trainer.features.columns.tolist()
    trainer.config.model_params = {}
    trainer._apply_model_priors()

    monkeypatch.setattr(
        feature_selection_module,
        "select_training_features",
        lambda features, labels, config: SimpleNamespace(
            selected_features=["momentum_10", "rsi_14"],
            diagnostics={
                "initial_feature_count": 3,
                "selected_feature_count": 2,
                "correlation_pruned_count": 1,
            },
            information_coefficients={"momentum_10": 0.12, "rsi_14": 0.01},
            stability_scores={"momentum_10": 1.0, "rsi_14": 0.6},
        ),
    )

    trainer._apply_feature_selection()

    assert trainer.feature_names == ["momentum_10", "rsi_14"]
    assert trainer.config.model_params["monotone_constraints"] == [1, 0]


def test_apply_feature_selection_records_detailed_audit(monkeypatch):
    import quant_trading_system.models.feature_selection as feature_selection_module

    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="xgboost"))
    trainer.features = pd.DataFrame(
        {
            "momentum_10": [0.1, 0.2, 0.3, 0.4],
            "volatility_20": [1.0, 0.9, 1.1, 1.0],
            "atr_14": [0.5, 0.7, 0.6, 0.4],
            "rsi_14": [40.0, 42.0, 39.0, 43.0],
        }
    )
    trainer.labels = np.array([0, 1, 0, 1], dtype=int)
    trainer.feature_names = trainer.features.columns.tolist()

    monkeypatch.setattr(
        feature_selection_module,
        "select_training_features",
        lambda features, labels, config: SimpleNamespace(
            selected_features=["momentum_10", "rsi_14"],
            diagnostics={
                "initial_feature_count": 4,
                "selected_feature_count": 2,
                "correlation_pruned_count": 1,
                "stability_selected_count": 2,
            },
            information_coefficients={
                "momentum_10": 0.12,
                "volatility_20": 0.11,
                "atr_14": 0.05,
                "rsi_14": 0.03,
            },
            correlation_pruned_features=["volatility_20"],
            stability_scores={
                "momentum_10": 1.0,
                "volatility_20": 0.8,
                "atr_14": 0.2,
                "rsi_14": 0.9,
            },
        ),
    )

    trainer._apply_feature_selection()

    audit = trainer.training_metrics["feature_selection_audit"]
    rejected = {item["feature"]: item["reason"] for item in audit["rejected_features"]}

    assert audit["selection_binding"] is True
    assert audit["selected_feature_count"] == 2
    assert rejected["volatility_20"] == "correlation_pruned"
    assert rejected["atr_14"] == "stability_support"
    assert audit["selected_family_counts"]["momentum"] == 1
    assert trainer.training_metrics["feature_selection_binding"] == pytest.approx(1.0)


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


def test_load_model_defaults_lightgbm_uses_cuda_when_gpu_enabled():
    defaults = train_script._load_model_defaults("lightgbm", use_gpu=True)
    assert defaults.get("device") == "cuda"


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


def test_evaluate_holdout_performance_aligns_sequence_model_outputs(monkeypatch):
    class DummySequenceModel:
        lookback_window = 3

        def predict(self, X):
            if len(X) == 6:
                return np.array([1.0, 0.0, 1.0, 0.0], dtype=float)
            return np.zeros(len(X), dtype=float)

    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(model_type="tcn", model_params={"lookback_window": 3})
    )
    trainer.model = DummySequenceModel()
    trainer.features = pd.DataFrame(np.random.rand(8, 2), columns=["f1", "f2"])
    trainer.labels = pd.Series(np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float))
    trainer.primary_forward_returns = np.zeros(8, dtype=float)
    trainer.regimes = np.array(["trend"] * 8, dtype=object)
    trainer.training_metrics = {}

    trainer.holdout_features = pd.DataFrame(np.random.rand(6, 2), columns=["f1", "f2"])
    trainer.holdout_labels = pd.Series(np.array([0, 1, 0, 1, 0, 1], dtype=float))
    trainer.holdout_symbols = np.array(["AAPL"] * 6, dtype=object)
    trainer.holdout_regimes = np.array(["trend"] * 6, dtype=object)
    trainer.holdout_primary_forward_returns = np.linspace(0.01, 0.06, 6, dtype=float)
    trainer.holdout_cost_aware_event_returns = np.linspace(0.01, 0.06, 6, dtype=float)
    trainer.holdout_primary_event_directions = np.array([1, -1, 1, -1, 1, -1], dtype=float)
    trainer.holdout_timestamps = np.array(
        pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC"),
        dtype="datetime64[ns]",
    )

    def _fake_get_predictions_proba(model, X):
        if len(X) == 6:
            return np.array([0.8, 0.2, 0.7, 0.3], dtype=float)
        return np.linspace(0.4, 0.6, len(X), dtype=float)

    captured_lengths: list[int] = []

    def _fake_calculate_fold_metrics(y_true, y_pred, y_proba, **kwargs):
        captured_lengths.append(len(np.asarray(y_true)))
        if kwargs.get("symbols") is not None:
            assert len(np.asarray(kwargs["symbols"])) == len(np.asarray(y_true))
        if kwargs.get("timestamps") is not None:
            assert len(np.asarray(kwargs["timestamps"])) == len(np.asarray(y_true))
        return {
            "accuracy": 0.5,
            "sharpe": 0.1,
            "max_drawdown": 0.05,
            "trade_count": 2.0,
            "risk_adjusted_score": 0.02,
        }

    monkeypatch.setattr(trainer, "_get_predictions_proba", _fake_get_predictions_proba)
    monkeypatch.setattr(trainer, "_derive_signal_thresholds", lambda *args, **kwargs: (0.55, 0.45))
    monkeypatch.setattr(trainer, "_calculate_fold_metrics", _fake_calculate_fold_metrics)

    trainer._evaluate_holdout_performance()

    assert captured_lengths
    assert all(length == 4 for length in captured_lengths)
    assert trainer.training_metrics["holdout_rows_raw"] == pytest.approx(6.0)
    assert trainer.training_metrics["holdout_rows_aligned"] == pytest.approx(4.0)
    assert trainer.training_metrics["holdout_rows"] == pytest.approx(4.0)


def test_evaluate_holdout_performance_records_tail_loss_contributors(monkeypatch):
    class DummyModel:
        def predict(self, X):
            if len(X) == 60:
                return np.concatenate(
                    [np.ones(30, dtype=float), np.zeros(30, dtype=float)]
                )
            return np.zeros(len(X), dtype=float)

    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="xgboost"))
    trainer.model = DummyModel()
    trainer.features = pd.DataFrame(np.random.rand(80, 2), columns=["f1", "f2"])
    trainer.labels = pd.Series(np.tile(np.array([0, 1], dtype=float), 40))
    trainer.primary_forward_returns = np.zeros(80, dtype=float)
    trainer.regimes = np.array(["trend"] * 80, dtype=object)
    trainer.training_metrics = {}

    rows = 60
    trainer.holdout_features = pd.DataFrame(np.random.rand(rows, 2), columns=["f1", "f2"])
    trainer.holdout_labels = pd.Series(
        np.concatenate([np.ones(30, dtype=float), np.zeros(30, dtype=float)])
    )
    trainer.holdout_symbols = np.array((["AAPL"] * 30) + (["MSFT"] * 30), dtype=object)
    trainer.holdout_regimes = np.array((["crash"] * 30) + (["trend"] * 30), dtype=object)
    trainer.holdout_primary_forward_returns = np.concatenate(
        [np.full(30, -0.01, dtype=float), np.full(30, 0.01, dtype=float)]
    )
    trainer.holdout_cost_aware_event_returns = trainer.holdout_primary_forward_returns.copy()
    trainer.holdout_primary_event_directions = np.concatenate(
        [np.ones(30, dtype=float), -np.ones(30, dtype=float)]
    )
    trainer.holdout_timestamps = np.array(
        pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
        dtype="datetime64[ns]",
    )

    monkeypatch.setattr(
        trainer,
        "_get_predictions_proba",
        lambda model, X: np.concatenate(
            [np.full(30, 0.80, dtype=float), np.full(30, 0.20, dtype=float)]
        )
        if len(X) == rows
        else np.linspace(0.4, 0.6, len(X), dtype=float),
    )
    monkeypatch.setattr(trainer, "_derive_signal_thresholds", lambda *args, **kwargs: (0.55, 0.45))

    def _fake_calculate_fold_metrics(y_true, y_pred, y_proba, **kwargs):
        symbols = np.asarray(kwargs.get("symbols"), dtype=object) if kwargs.get("symbols") is not None else None
        if symbols is not None and len(set(symbols.tolist())) == 1:
            group_name = str(symbols[0])
            if group_name == "AAPL":
                return {
                    "accuracy": 0.25,
                    "sharpe": -0.8,
                    "max_drawdown": 0.30,
                    "trade_count": 2.0,
                    "win_rate": 0.20,
                    "turnover": 0.10,
                    "active_signal_rate": 0.50,
                    "annual_return": -0.04,
                    "calmar": -0.13,
                    "pnl": -0.020,
                    "loss_pnl": -0.030,
                    "tail_loss_pnl": -0.018,
                    "underwater_ratio": 0.75,
                    "cvar_95": -0.025,
                    "risk_adjusted_score": -0.20,
                    "sharpe_observation_confidence": 0.80,
                    "symbol_concentration_hhi": 1.0,
                }
            return {
                "accuracy": 0.75,
                "sharpe": 0.6,
                "max_drawdown": 0.08,
                "trade_count": 2.0,
                "win_rate": 0.70,
                "turnover": 0.08,
                "active_signal_rate": 0.50,
                "annual_return": 0.03,
                "calmar": 0.37,
                "pnl": 0.018,
                "loss_pnl": -0.004,
                "tail_loss_pnl": -0.001,
                "underwater_ratio": 0.20,
                "cvar_95": -0.010,
                "risk_adjusted_score": 0.15,
                "sharpe_observation_confidence": 0.85,
                "symbol_concentration_hhi": 1.0,
            }
        return {
            "accuracy": 0.5,
            "sharpe": 0.1,
            "max_drawdown": 0.10,
            "trade_count": 4.0,
            "win_rate": 0.5,
            "turnover": 0.09,
            "active_signal_rate": 0.50,
            "annual_return": 0.0,
            "calmar": 0.0,
            "pnl": -0.002,
            "loss_pnl": -0.034,
            "tail_loss_pnl": -0.019,
            "underwater_ratio": 0.40,
            "cvar_95": -0.018,
            "risk_adjusted_score": 0.02,
            "sharpe_observation_confidence": 0.90,
            "symbol_concentration_hhi": 0.5,
        }

    monkeypatch.setattr(trainer, "_calculate_fold_metrics", _fake_calculate_fold_metrics)

    trainer._evaluate_holdout_performance()

    assert trainer.training_metrics["holdout_symbol_metrics"]["AAPL"]["tail_loss_pnl"] == pytest.approx(
        -0.018
    )
    assert trainer.training_metrics["holdout_symbol_metrics"]["AAPL"]["underwater_ratio"] == pytest.approx(
        0.75
    )
    assert trainer.training_metrics["holdout_tail_loss_contributors_by_symbol"][0]["name"] == "AAPL"
    assert trainer.training_metrics["holdout_tail_loss_contributors_by_regime"][0]["name"] == "crash"


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
        event_directions=None,
        timestamps=None,
        symbols=None,
        return_details=False,
    ):
        captured["y_proba"] = np.asarray(y_proba, dtype=float).copy()
        captured["timestamps"] = (
            np.asarray(timestamps).copy() if timestamps is not None else np.array([])
        )
        captured["symbols"] = (
            np.asarray(symbols, dtype=object).copy() if symbols is not None else np.array([])
        )
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
        event_directions=None,
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


def test_calculate_fold_metrics_uses_timeframe_aware_annualization(monkeypatch):
    def _fake_compute_net_returns(
        y_true,
        y_proba,
        long_threshold=0.55,
        short_threshold=0.45,
        realized_forward_returns=None,
        event_net_returns=None,
        event_directions=None,
        timestamps=None,
        symbols=None,
        return_details=False,
    ):
        row_returns = np.array([0.02, 0.01, 0.03, 0.01], dtype=float)
        details = {
            "positions": np.ones(4, dtype=float),
            "turnover_series": np.zeros(4, dtype=float),
            "trade_mask": np.ones(4, dtype=bool),
            "entry_mask": np.ones(4, dtype=bool),
            "portfolio_returns": np.array([0.02, 0.01, 0.03, 0.01], dtype=float),
            "portfolio_turnover_series": np.zeros(4, dtype=float),
            "portfolio_trade_mask": np.ones(4, dtype=bool),
            "portfolio_entry_mask": np.ones(4, dtype=bool),
        }
        return (row_returns, details) if return_details else row_returns

    daily_trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(model_type="xgboost", timeframe="1Day")
    )
    intraday_trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(model_type="xgboost", timeframe="1Hour")
    )
    monkeypatch.setattr(daily_trainer, "_compute_net_returns", _fake_compute_net_returns)
    monkeypatch.setattr(intraday_trainer, "_compute_net_returns", _fake_compute_net_returns)

    daily_metrics = daily_trainer._calculate_fold_metrics(
        y_true=np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
        y_pred=np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
        y_proba=np.array([0.9, 0.9, 0.9, 0.9], dtype=float),
    )
    intraday_metrics = intraday_trainer._calculate_fold_metrics(
        y_true=np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
        y_pred=np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
        y_proba=np.array([0.9, 0.9, 0.9, 0.9], dtype=float),
    )

    assert intraday_metrics["sharpe"] > daily_metrics["sharpe"]
    assert intraday_metrics["calmar"] > daily_metrics["calmar"]


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
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(model_type="xgboost", min_trades=40)
    )
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

    np.testing.assert_allclose(
        profile_pos["long_threshold_series"], profile_neg["long_threshold_series"]
    )
    np.testing.assert_allclose(
        profile_pos["short_threshold_series"], profile_neg["short_threshold_series"]
    )
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
    monkeypatch.setattr(
        trainer,
        "_compute_features_full_pipeline",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    fallback_called = {"value": False}
    monkeypatch.setattr(
        trainer, "_compute_features_fallback", lambda: fallback_called.__setitem__("value", True)
    )

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
    monkeypatch.setattr(
        trainer,
        "_compute_features_full_pipeline",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    fallback_called = {"value": False}
    monkeypatch.setattr(
        trainer, "_compute_features_fallback", lambda: fallback_called.__setitem__("value", True)
    )

    trainer._compute_features()
    assert fallback_called["value"] is True


def test_compute_features_full_pipeline_allows_wave1_cross_sectional_row_budget(monkeypatch):
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            feature_groups=["cross_sectional"],
            enable_reference_features=False,
            enable_tick_microstructure_features=False,
            persist_features_to_postgres=False,
            max_cross_sectional_rows=500000,
        )
    )
    row_count = 260000
    timestamps = pd.date_range("2024-01-01", periods=row_count // 2, freq="15min", tz="UTC")
    trainer.data = pd.DataFrame(
        {
            "symbol": np.repeat(["AAPL", "MSFT"], len(timestamps)),
            "timestamp": np.tile(timestamps.to_numpy(), 2),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 100.0,
        }
    )

    monkeypatch.setattr(train_script.os, "name", "posix", raising=False)

    def _fake_symbol_features(**kwargs):
        df = kwargs["df"]
        return df.loc[:, ["symbol", "timestamp", "open", "high", "low", "close", "volume"]].assign(
            alpha=1.0
        )

    monkeypatch.setattr(trainer, "_compute_symbol_feature_frame", _fake_symbol_features)
    monkeypatch.setattr(trainer, "_augment_reference_features", lambda frame: frame)
    monkeypatch.setattr(trainer, "_augment_tick_microstructure_features", lambda frame: frame)
    monkeypatch.setattr(trainer, "_finalize_feature_matrix", lambda frame: frame)

    trainer._compute_features_full_pipeline()

    assert trainer.features is not None
    assert "alpha" in trainer.features.columns
    assert len(trainer.features) == row_count


def test_training_config_defaults_cross_sectional_row_budget_to_wave1_scope():
    config = train_script.TrainingConfig(model_type="xgboost")

    assert config.max_cross_sectional_rows == 500000


def test_compute_features_defers_postgres_persistence_until_after_selection(monkeypatch):
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            persist_features_to_postgres=True,
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

    monkeypatch.setattr(train_script.os, "name", "posix", raising=False)
    monkeypatch.setattr(trainer, "_load_features_from_postgres", lambda: None)
    monkeypatch.setattr(
        trainer,
        "_compute_features_full_pipeline",
        lambda: setattr(
            trainer,
            "features",
            trainer.data.assign(feature_alpha=np.array([0.1, 0.2, 0.3, 0.4])),
        ),
    )
    persist_called = {"value": False}
    monkeypatch.setattr(
        trainer,
        "_store_features_to_postgres",
        lambda: persist_called.__setitem__("value", True),
    )

    trainer._compute_features()

    assert persist_called["value"] is False
    assert trainer.feature_cache_reused is False
    assert trainer.features_materialized_in_run is True


def test_persist_features_to_postgres_if_needed_skips_reused_cache(monkeypatch):
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            persist_features_to_postgres=True,
        )
    )
    trainer.features_materialized_in_run = True
    trainer.feature_cache_reused = True
    persist_calls = {"count": 0}
    monkeypatch.setattr(
        trainer,
        "_store_features_to_postgres",
        lambda: persist_calls.__setitem__("count", persist_calls["count"] + 1),
    )

    trainer._persist_features_to_postgres_if_needed()

    assert persist_calls["count"] == 0


def test_persist_features_to_postgres_if_needed_persists_fresh_selected_features(monkeypatch):
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            persist_features_to_postgres=True,
        )
    )
    trainer.features_materialized_in_run = True
    trainer.feature_cache_reused = False
    trainer.snapshot_replay_loaded = False
    persist_calls = {"count": 0}
    monkeypatch.setattr(
        trainer,
        "_store_features_to_postgres",
        lambda: persist_calls.__setitem__("count", persist_calls["count"] + 1),
    )

    trainer._persist_features_to_postgres_if_needed()

    assert persist_calls["count"] == 1


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


def test_write_promotion_package_includes_position_sizing_policy(tmp_path):
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            model_name="pkg_model",
            output_dir=str(tmp_path),
            save_artifacts=False,
        )
    )
    trainer.feature_names = ["feat_a", "feat_b"]
    trainer.expected_edge_model = object()
    trainer.training_metrics = {
        "holdout_long_threshold": 0.61,
        "holdout_short_threshold": 0.39,
        "holdout_side_policy": {
            "long": {"enabled": True, "signal_scale": 1.10},
            "short": {"enabled": False, "signal_scale": 0.0},
        },
        "expected_edge_training_selected_context_features": ["flow_imbalance"],
        "expected_edge_training_summary": {"selected_rate": 0.42},
        "expected_edge_holdout_summary": {"selected_edge_lift": 0.003},
        "expected_edge_regime_policy": {
            "enabled": True,
            "regimes": {
                "high_vol": {"signal_scale": 0.75},
            },
        },
        "symbol_quality_universe": ["AAPL", "MSFT"],
        "symbol_quality_dropped_list": ["TSLA"],
        "symbol_quality_report": {"AAPL": {"passes": True, "quality_score": 0.92}},
    }
    trainer.validation_results = {"all_passed": True}
    trainer.snapshot_manifest = {"feature_schema_version": "schema-1", "snapshot_id": "snap-1"}

    package_path = trainer._write_promotion_package(
        output_dir=tmp_path,
        model_path=tmp_path / "pkg_model.pkl",
        replay_manifest_path=None,
    )

    payload = json.loads(package_path.read_text(encoding="utf-8"))

    assert payload["position_sizing_policy"]["use_portfolio_target_sizing"] is True
    assert payload["position_sizing_policy"]["max_position_pct"] == pytest.approx(0.10)
    assert payload["position_sizing_policy"]["max_total_positions"] == 20
    assert payload["signal_policy"]["long_side_policy"]["signal_scale"] == pytest.approx(1.10)
    assert payload["signal_policy"]["short_side_policy"]["enabled"] is False
    assert payload["expected_edge_policy"]["enabled"] is True
    assert payload["expected_edge_policy"]["selected_context_features"] == ["flow_imbalance"]
    assert payload["expected_edge_policy"]["regime_conditioned_policy"]["regimes"]["high_vol"][
        "signal_scale"
    ] == pytest.approx(0.75)
    assert payload["universe_quality_policy"]["selected_symbols"] == ["AAPL", "MSFT"]
    assert payload["universe_quality_policy"]["dropped_symbols"] == ["TSLA"]


def test_train_expected_edge_policy_records_training_and_holdout_metrics():
    class DummyProbabilityModel:
        def predict_proba(self, X):
            X = np.asarray(X, dtype=float)
            probability = np.clip(X[:, 0], 0.0, 1.0)
            return np.column_stack([1.0 - probability, probability])

    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            expected_edge_min_samples=40,
            expected_edge_min_coverage=0.15,
            save_artifacts=False,
        )
    )
    rows = 160
    feat_a = np.where(np.arange(rows) % 2 == 0, 0.72, 0.28).astype(float)
    flow_feature = np.linspace(-1.0, 1.0, rows, dtype=float)
    trainer.features = pd.DataFrame(
        {
            "feat_a": feat_a,
            "flow_imbalance_signal": flow_feature,
            "macro_regime_score": np.sin(np.linspace(0.0, 3.14, rows)),
        }
    )
    trainer.row_symbols = np.where(np.arange(rows) % 2 == 0, "AAPL", "MSFT").astype(object)
    trainer.regimes = np.where(np.arange(rows) % 3 == 0, "high_vol", "trend_up").astype(object)
    trainer.oof_primary_proba = feat_a.copy()
    trainer.primary_forward_returns = np.where(feat_a >= 0.5, 0.015, -0.012).astype(float)
    trainer.sample_weights = np.ones(rows, dtype=float)
    trainer.model = DummyProbabilityModel()
    trainer.holdout_features = pd.DataFrame(
        {
            "feat_a": [0.70, 0.68, 0.31, 0.29],
            "flow_imbalance_signal": [0.6, 0.3, -0.4, -0.7],
            "macro_regime_score": [0.2, 0.1, -0.1, -0.2],
        }
    )
    trainer.holdout_symbols = np.array(["AAPL", "AAPL", "MSFT", "MSFT"], dtype=object)
    trainer.holdout_regimes = np.array(["trend_up", "trend_up", "high_vol", "high_vol"], dtype=object)
    trainer.holdout_primary_forward_returns = np.array([0.014, 0.010, -0.011, -0.013], dtype=float)
    trainer.training_metrics["holdout_long_threshold"] = 0.60
    trainer.training_metrics["holdout_short_threshold"] = 0.40

    trainer._train_expected_edge_policy()

    assert trainer.expected_edge_model is not None
    assert trainer.training_metrics["expected_edge_policy_enabled"] == pytest.approx(1.0)
    assert trainer.training_metrics["expected_edge_training_candidate_count"] >= 40
    assert trainer.training_metrics["expected_edge_holdout_selected_count"] >= 1
    assert "flow_imbalance_signal" in trainer.training_metrics[
        "expected_edge_training_selected_context_features"
    ]
    candidate_floor = trainer.training_metrics["expected_edge_candidate_floor"]
    assert candidate_floor["meets_min_samples"] is True
    assert candidate_floor["candidate_count"] >= 40
    training_funnel = trainer.training_metrics["expected_edge_training_signal_funnel"]
    assert training_funnel["candidate_count"] >= training_funnel["selected_count"]
    assert trainer.training_metrics["expected_edge_training_signal_funnel_by_symbol"]["AAPL"][
        "candidate_count"
    ] >= 1
    regime_policy = trainer.training_metrics["expected_edge_regime_policy"]
    assert regime_policy["enabled"] is True
    assert sorted(regime_policy["regimes"].keys()) == ["high_vol", "trend_up"]
    assert trainer.training_metrics["expected_edge_holdout_regime_policy_enabled"] == pytest.approx(1.0)
    assert trainer.training_metrics["expected_edge_holdout_signal_funnel_by_regime"]["high_vol"][
        "candidate_count"
    ] >= 1


def test_train_expected_edge_policy_records_candidate_floor_when_skipped():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            expected_edge_min_samples=20,
            expected_edge_min_coverage=0.30,
            save_artifacts=False,
        )
    )
    rows = 24
    trainer.features = pd.DataFrame(
        {
            "feat_a": np.full(rows, 0.52, dtype=float),
            "flow_imbalance_signal": np.linspace(-0.2, 0.2, rows, dtype=float),
        }
    )
    trainer.row_symbols = np.where(np.arange(rows) % 2 == 0, "AAPL", "MSFT").astype(object)
    trainer.regimes = np.where(np.arange(rows) % 2 == 0, "trend_up", "high_vol").astype(object)
    trainer.oof_primary_proba = np.array(
        [0.52] * 18 + [0.70, 0.72, 0.74, 0.26, 0.24, 0.22],
        dtype=float,
    )
    trainer.primary_forward_returns = np.where(trainer.oof_primary_proba >= 0.5, 0.01, -0.01).astype(
        float
    )

    trainer._train_expected_edge_policy()

    assert trainer.expected_edge_model is None
    assert "candidate trades" in trainer.training_metrics["expected_edge_policy_reason"]
    assert "received 6" in trainer.training_metrics["expected_edge_policy_reason"]
    candidate_floor = trainer.training_metrics["expected_edge_candidate_floor"]
    assert candidate_floor["candidate_count"] == 6
    assert candidate_floor["configured_min_samples"] >= 20
    assert candidate_floor["meets_min_samples"] is False
    precheck_funnel = trainer.training_metrics["expected_edge_training_precheck_signal_funnel"]
    assert precheck_funnel["candidate_count"] == 6
    assert trainer.training_metrics["expected_edge_training_precheck_signal_funnel_by_regime"][
        "high_vol"
    ]["candidate_count"] >= 1


def test_run_records_training_run_events(monkeypatch, tmp_path):
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            output_dir=str(tmp_path),
            save_artifacts=False,
            optimize=False,
            compute_shap=False,
            use_meta_labeling=False,
            enable_expected_edge_policy=False,
        )
    )
    events: list[dict[str, object]] = []

    def _capture_event(index_root, event):
        events.append({"index_root": index_root, "event": dict(event)})
        return tmp_path / "run_index" / "training_runs.jsonl"

    monkeypatch.setattr(train_script, "append_training_run_event", _capture_event)
    monkeypatch.setattr(trainer, "_load_data", lambda: None)
    monkeypatch.setattr(trainer, "_compute_features", lambda: None)
    monkeypatch.setattr(trainer, "_create_labels", lambda: None)
    monkeypatch.setattr(trainer, "_apply_feature_selection", lambda: None)
    monkeypatch.setattr(trainer, "_persist_features_to_postgres_if_needed", lambda: None)
    monkeypatch.setattr(trainer, "_validate_no_future_leakage", lambda: None)
    monkeypatch.setattr(trainer, "_train_with_cv", lambda: None)
    monkeypatch.setattr(trainer, "_build_model_card", lambda: {"summary": "ok"})
    monkeypatch.setattr(
        trainer,
        "_build_deployment_plan",
        lambda passed_validation: {"ready_for_production": bool(passed_validation)},
    )
    monkeypatch.setattr(trainer, "_validate_model", lambda: True)

    def _fake_save_model():
        trainer.artifacts_path = tmp_path / "model_artifacts.json"
        return str(tmp_path / "model.pkl")

    monkeypatch.setattr(trainer, "_save_model", _fake_save_model)
    trainer.training_metrics.update(
        {
            "mean_sharpe": 1.1,
            "mean_trade_count": 120.0,
            "mean_risk_adjusted_score": 0.8,
            "holdout_sharpe": 0.9,
            "holdout_max_drawdown": 0.15,
        }
    )

    result = trainer.run()

    statuses = [entry["event"]["status"] for entry in events]
    assert statuses == ["started", "completed"]
    assert result["success"] is True
    assert result["run_id"] == trainer.run_id
    assert str(result["run_event_index_path"]).endswith("training_runs.jsonl")
    assert events[-1]["event"]["metrics"]["mean_trade_count"] == 120.0
    assert str(events[-1]["index_root"]).endswith("run_index")


def test_run_snapshot_only_persists_review_and_skips_model_fitting(monkeypatch, tmp_path):
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            output_dir=str(tmp_path),
            save_artifacts=True,
            snapshot_only=True,
            optimize=True,
        )
    )
    events: list[dict[str, object]] = []

    def _capture_event(index_root, event):
        events.append({"index_root": index_root, "event": dict(event)})
        return tmp_path / "run_index" / "training_runs.jsonl"

    monkeypatch.setattr(train_script, "append_training_run_event", _capture_event)

    def _fake_load_data():
        trainer.data_quality_report = {
            "passed": True,
            "summary": {"symbol_count": 12},
            "threshold_breaches": {},
            "top_missing_bar_symbols": [],
            "top_missing_bar_windows": [],
        }
        trainer.data_quality_report_hash = "dq_task1"
        trainer.training_metrics["symbol_quality_input_symbols"] = 12.0
        trainer.training_metrics["symbol_quality_selected_symbols"] = 12.0
        trainer.training_metrics["symbol_quality_dropped_symbols"] = 0.0
        trainer.training_metrics["symbol_quality_universe"] = ["AAPL", "MSFT"]
        trainer.training_metrics["symbol_quality_dropped_list"] = []

    monkeypatch.setattr(trainer, "_load_data", _fake_load_data)
    monkeypatch.setattr(trainer, "_compute_features", lambda: None)

    def _fake_create_labels():
        trainer.snapshot_manifest = {"snapshot_id": "snap_task1"}

    monkeypatch.setattr(trainer, "_create_labels", _fake_create_labels)
    monkeypatch.setattr(trainer, "_apply_feature_selection", lambda: None)
    monkeypatch.setattr(trainer, "_persist_features_to_postgres_if_needed", lambda: None)
    monkeypatch.setattr(trainer, "_validate_no_future_leakage", lambda: None)

    def _fake_persist_snapshot_artifacts(output_dir):
        trainer.snapshot_manifest_path = output_dir / "snap_task1.manifest.json"
        trainer.snapshot_manifest_path.write_text("{}", encoding="utf-8")
        trainer.data_quality_report_path = output_dir / "snap_task1.quality.json"
        trainer.data_quality_report_path.write_text("{}", encoding="utf-8")
        bundle_dir = output_dir / "snapshots" / "snap_task1"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        trainer.dataset_snapshot_bundle_manifest = {"bundle_hash": "bundle_task1"}
        trainer.dataset_snapshot_bundle_manifest_path = bundle_dir / "dataset_bundle.manifest.json"
        trainer.dataset_snapshot_bundle_manifest_path.write_text("{}", encoding="utf-8")
        trainer.snapshot_review_path = output_dir / "task1.snapshot_review.json"
        trainer.snapshot_review_path.write_text("{}", encoding="utf-8")
        review_payload = trainer._build_snapshot_review()
        review_payload["snapshot_review_path"] = str(trainer.snapshot_review_path)
        trainer.training_metrics["snapshot_review"] = review_payload
        trainer.training_metrics["snapshot_review_ready"] = True
        trainer.training_metrics["snapshot_review_failed_checks"] = []
        return review_payload

    monkeypatch.setattr(trainer, "_persist_snapshot_artifacts", _fake_persist_snapshot_artifacts)
    monkeypatch.setattr(
        trainer,
        "_optimize_hyperparameters",
        lambda: (_ for _ in ()).throw(AssertionError("snapshot-only should skip optuna")),
    )
    monkeypatch.setattr(
        trainer,
        "_train_with_cv",
        lambda: (_ for _ in ()).throw(AssertionError("snapshot-only should skip CV")),
    )
    monkeypatch.setattr(
        trainer,
        "_save_model",
        lambda: (_ for _ in ()).throw(AssertionError("snapshot-only should skip save_model")),
    )

    result = trainer.run()

    statuses = [entry["event"]["status"] for entry in events]
    assert statuses == ["started", "completed"]
    assert result["snapshot_only"] is True
    assert result["success"] is True
    assert result["model_path"] is None
    assert result["snapshot_id"] == "snap_task1"
    assert result["dataset_bundle_hash"] == "bundle_task1"
    assert str(result["snapshot_review_path"]).endswith("task1.snapshot_review.json")
    assert result["training_metrics"]["snapshot_review_ready"] is True


def test_run_snapshot_only_fast_fails_after_phase1_quality_review(monkeypatch, tmp_path):
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            output_dir=str(tmp_path),
            save_artifacts=True,
            snapshot_only=True,
            optimize=True,
        )
    )
    events: list[dict[str, object]] = []

    def _capture_event(index_root, event):
        events.append({"index_root": index_root, "event": dict(event)})
        return tmp_path / "run_index" / "training_runs.jsonl"

    monkeypatch.setattr(train_script, "append_training_run_event", _capture_event)

    def _fake_load_data():
        trainer.data_quality_report = {
            "passed": False,
            "summary": {"symbol_count": 12},
            "threshold_breaches": {
                "missing_bars_ratio_max": {"actual": 0.18, "threshold": 0.05}
            },
            "top_missing_bar_symbols": [{"symbol": "GLD", "missing_bars_count": 42}],
            "top_missing_bar_windows": [],
        }
        trainer.data_quality_report_hash = "dq_bad_task1"
        trainer.training_metrics["symbol_quality_input_symbols"] = 12.0
        trainer.training_metrics["symbol_quality_selected_symbols"] = 11.0
        trainer.training_metrics["symbol_quality_dropped_symbols"] = 1.0
        trainer.training_metrics["symbol_quality_universe"] = ["AAPL", "MSFT"]
        trainer.training_metrics["symbol_quality_dropped_list"] = ["GLD"]

    monkeypatch.setattr(trainer, "_load_data", _fake_load_data)
    monkeypatch.setattr(
        trainer,
        "_compute_features",
        lambda: (_ for _ in ()).throw(
            AssertionError("preflight failure should skip feature materialization")
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_apply_feature_selection",
        lambda: (_ for _ in ()).throw(
            AssertionError("preflight failure should skip feature selection")
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_persist_features_to_postgres_if_needed",
        lambda: (_ for _ in ()).throw(
            AssertionError("preflight failure should skip feature cache persistence")
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_validate_no_future_leakage",
        lambda: (_ for _ in ()).throw(
            AssertionError("preflight failure should skip leakage validation")
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_optimize_hyperparameters",
        lambda: (_ for _ in ()).throw(AssertionError("preflight failure should skip optuna")),
    )
    monkeypatch.setattr(
        trainer,
        "_train_with_cv",
        lambda: (_ for _ in ()).throw(AssertionError("preflight failure should skip CV")),
    )
    monkeypatch.setattr(
        trainer,
        "_save_model",
        lambda: (_ for _ in ()).throw(AssertionError("preflight failure should skip save_model")),
    )

    def _fake_persist_snapshot_review_only(output_dir):
        trainer.data_quality_report_path = output_dir / "wave1.preflight_quality.json"
        trainer.data_quality_report_path.write_text("{}", encoding="utf-8")
        trainer.snapshot_review_path = output_dir / "wave1.snapshot_review.json"
        trainer.snapshot_review_path.write_text("{}", encoding="utf-8")
        review_payload = trainer._build_snapshot_review()
        review_payload["preflight_only"] = True
        review_payload["snapshot_review_path"] = str(trainer.snapshot_review_path)
        trainer.training_metrics["snapshot_review"] = review_payload
        trainer.training_metrics["snapshot_review_ready"] = False
        trainer.training_metrics["snapshot_review_failed_checks"] = list(
            review_payload["failed_checks"]
        )
        return review_payload

    monkeypatch.setattr(trainer, "_persist_snapshot_review_only", _fake_persist_snapshot_review_only)

    result = trainer.run()

    statuses = [entry["event"]["status"] for entry in events]
    assert statuses == ["started", "completed"]
    assert result["snapshot_only"] is True
    assert result["preflight_only"] is True
    assert result["success"] is False
    assert result["model_path"] is None
    assert result["snapshot_id"] is None
    assert result["data_quality_report_passed"] is False
    assert set(result["snapshot_review"]["failed_checks"]) == {
        "data_quality_passed",
        "no_silent_symbol_drop",
    }
    assert str(result["data_quality_report_path"]).endswith("wave1.preflight_quality.json")
    assert str(result["snapshot_review_path"]).endswith("wave1.snapshot_review.json")
    assert result["training_metrics"]["snapshot_review_ready"] is False


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


def test_build_parser_supports_training_profiles_and_snapshot_flags():
    parser = train_script.build_parser()
    args = parser.parse_args(
        [
            "--model",
            "primary_challenger",
            "--training-profile",
            "research",
            "--dataset-snapshot-bundle",
            "models/snapshots/snap_123/dataset_bundle.manifest.json",
            "--strict-snapshot-replay",
            "--force-promotion-precheck-bypass",
            "--disable-auto-snapshot-reuse",
            "--snapshot-only",
            "--optuna-storage-dir",
            "models/optuna_state",
            "--disable-optuna-resume",
            "--disable-meta-labeling",
            "--disable-feature-selection",
            "--feature-selection-stability-iterations",
            "6",
            "--warm-start-model",
            "models/lightgbm_prev.pkl",
        ]
    )
    assert args.model == "primary_challenger"
    assert args.training_profile == "research"
    assert str(args.dataset_snapshot_bundle).endswith("dataset_bundle.manifest.json")
    assert args.strict_snapshot_replay is True
    assert args.force_promotion_precheck_bypass is True
    assert args.disable_auto_snapshot_reuse is True
    assert args.snapshot_only is True
    assert str(args.optuna_storage_dir).endswith("optuna_state")
    assert args.disable_optuna_resume is True
    assert args.disable_meta_labeling is True
    assert args.disable_feature_selection is True
    assert args.feature_selection_stability_iterations == 6
    assert str(args.warm_start_model).endswith("lightgbm_prev.pkl")


def test_build_parser_supports_feature_pipeline_flags():
    parser = train_script.build_parser()
    args = parser.parse_args(
        [
            "--feature-groups",
            "technical",
            "statistical",
            "--disable-cross-sectional",
            "--target-universe-size",
            "48",
            "--universe-selection-buffer-size",
            "18",
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
    assert args.target_universe_size == 48
    assert args.universe_selection_buffer_size == 18
    assert args.strict_feature_groups is True
    assert args.max_cross_sectional_symbols == 14
    assert args.max_cross_sectional_rows == 150000
    assert args.feature_materialization_batch_rows == 4000
    assert args.feature_reuse_min_coverage == 0.3
    assert args.windows_fallback_features is True


def test_compute_symbol_multitimeframe_feature_frame_prefers_db_native_timeframes(monkeypatch):
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            timeframe="15Min",
            timeframes=["1Hour"],
        )
    )
    base_frame = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "timestamp": pd.to_datetime(
                ["2024-01-01T14:30:00Z", "2024-01-01T14:45:00Z"], utc=True
            ),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000, 1100],
        }
    )
    hourly_frame = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "timestamp": pd.to_datetime(["2024-01-01T15:00:00Z"], utc=True),
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.5],
            "volume": [2100],
        }
    )
    trainer.ohlcv_panels_by_timeframe = {
        "15Min": base_frame.copy(),
        "1Hour": hourly_frame.copy(),
    }

    captured_rows: dict[str, int] = {}

    def _fake_compute_symbol_feature_frame(
        *,
        df,
        symbol,
        groups_to_compute,
        universe_data,
        disable_optimized_technical,
    ):
        timeframe_value = pd.to_datetime(df["timestamp"], utc=True).diff().dropna()
        timeframe_key = "1Hour" if len(df) == 1 else "15Min"
        captured_rows[timeframe_key] = len(df)
        return df.copy()

    monkeypatch.setattr(
        trainer,
        "_compute_symbol_feature_frame",
        _fake_compute_symbol_feature_frame,
    )

    result = trainer._compute_symbol_multitimeframe_feature_frame(
        df=base_frame,
        symbol="AAPL",
        groups_to_compute=[object()],
        universe_data=None,
        disable_optimized_technical=False,
    )

    assert captured_rows["15Min"] == 2
    assert captured_rows["1Hour"] == 1
    assert not result.empty


def test_build_parser_supports_symbols_file():
    parser = train_script.build_parser()
    args = parser.parse_args(["--symbols-file", "config/symbols.txt"])
    assert str(args.symbols_file).endswith("config\\symbols.txt") or str(
        args.symbols_file
    ).endswith("config/symbols.txt")


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
            "--probability-calibration-method",
            "sigmoid",
            "--disable-probability-calibration",
            "--min-confidence-position-scale",
            "0.3",
        ]
    )
    assert args.primary_horizon_sweep == [1, 5, 20]
    assert args.meta_label_min_confidence == pytest.approx(0.63)
    assert args.disable_meta_dynamic_threshold is True
    assert args.probability_calibration_method == "sigmoid"
    assert args.disable_probability_calibration is True
    assert args.min_confidence_position_scale == pytest.approx(0.3)


def test_build_parser_supports_market_session_flags():
    parser = train_script.build_parser()
    args = parser.parse_args(["--include-premarket", "--include-postmarket"])
    assert args.include_premarket is True
    assert args.include_postmarket is True


def test_build_parser_supports_white_reality_gate_flags():
    parser = train_script.build_parser()
    args = parser.parse_args(
        ["--min-white-reality-stat", "0.05", "--max-white-reality-pvalue", "0.08"]
    )
    assert args.min_white_reality_stat == pytest.approx(0.05)
    assert args.max_white_reality_pvalue == pytest.approx(0.08)


def test_build_parser_accepts_lightgbm_ranker_model_choice():
    parser = train_script.build_parser()
    args = parser.parse_args(["--model", "lightgbm_ranker"])
    assert args.model == "lightgbm_ranker"


def test_build_parser_accepts_return_regressor_model_choices():
    parser = train_script.build_parser()
    args_xgb = parser.parse_args(["--model", "xgboost_regressor"])
    args_lgb = parser.parse_args(["--model", "lightgbm_regressor"])
    assert args_xgb.model == "xgboost_regressor"
    assert args_lgb.model == "lightgbm_regressor"


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


def test_build_pre_promotion_checklist_reports_failed_work_plan_checks():
    checklist = train_script._build_pre_promotion_checklist(
        {
            "success": False,
            "model_name": "ranker_h12",
            "model_type": "lightgbm_ranker",
            "snapshot_id": "snap_123",
            "dataset_bundle_hash": "bundle_123",
            "data_quality_report_hash": "dq_123",
            "training_metrics": {
                "training_profile": "research",
                "data_quality_report_passed": False,
                "mean_trade_count": 11.0,
                "mean_risk_adjusted_score": -2.0,
                "holdout_max_drawdown": 0.43,
                "holdout_symbol_sharpe_p25": -1.18,
                "pbo": 0.60,
                "white_reality_pvalue": 0.12,
                "expected_edge_policy_reason": "received 0 candidate trades",
                "symbol_quality_dropped_symbols": 1.0,
            },
            "validation_results": {
                "layers": {
                    "model_utility": {"passed": False},
                    "execution_robustness": {"passed": False},
                    "cross_symbol_robustness": {"passed": False},
                }
            },
        }
    )

    assert checklist["ready"] is False
    assert "data_quality_passed" in checklist["failed_checks"]
    assert "expected_edge_trained" in checklist["failed_checks"]
    assert checklist["checks"]["expected_edge_trained"]["reason"] == "received 0 candidate trades"


def test_build_training_matrix_captures_snapshot_identity_and_prepromotion_status():
    rows = train_script._build_training_matrix(
        [
            (
                "lightgbm_ranker",
                {
                    "success": True,
                    "model_name": "ranker_h12_clean",
                    "run_id": "wave1_ranker_h12_clean",
                    "snapshot_id": "snap_clean",
                    "dataset_bundle_hash": "bundle_clean",
                    "data_quality_report_hash": "dq_clean",
                    "data_quality_report_passed": True,
                    "training_metrics": {
                        "training_profile": "research",
                        "mean_risk_adjusted_score": 0.45,
                        "mean_trade_count": 155.0,
                        "holdout_sharpe": 0.62,
                        "holdout_max_drawdown": 0.22,
                        "holdout_symbol_sharpe_p25": 0.04,
                        "deflated_sharpe": 0.35,
                        "pbo": 0.18,
                        "white_reality_pvalue": 0.04,
                        "expected_edge_policy_reason": "trained",
                        "expected_edge_policy_enabled": 1.0,
                        "symbol_quality_dropped_symbols": 0.0,
                    },
                    "validation_results": {
                        "layers": {
                            "model_utility": {"passed": True},
                            "execution_robustness": {"passed": True},
                            "cross_symbol_robustness": {"passed": True},
                        }
                    },
                },
            )
        ]
    )

    assert rows[0]["snapshot_id"] == "snap_clean"
    assert rows[0]["dataset_bundle_hash"] == "bundle_clean"
    assert rows[0]["data_quality_report_hash"] == "dq_clean"
    assert rows[0]["pre_promotion_ready"] == pytest.approx(1.0)
    assert rows[0]["pre_promotion_failed_check_count"] == pytest.approx(0.0)


def test_training_config_normalizes_limits_and_exports_cost_model():
    cfg = train_script.TrainingConfig(
        model_type="xgboost",
        feature_reuse_min_coverage=2.5,
        execution_turnover_cap=2.0,
        min_confidence_position_scale=2.0,
        probability_calibration_method="SIGMOID",
        primary_horizon_sweep=[-1, 5, 0, 3],
        label_spread_bps=1.5,
        label_slippage_bps=2.0,
        label_impact_bps=3.5,
    )

    assert cfg.feature_reuse_min_coverage == pytest.approx(0.95)
    assert cfg.execution_turnover_cap == pytest.approx(1.0)
    assert cfg.min_confidence_position_scale == pytest.approx(1.0)
    assert cfg.probability_calibration_method == "sigmoid"
    assert cfg.primary_horizon_sweep == [3, 5]
    assert cfg.to_trading_cost_model().execution_cost_bps == pytest.approx(7.0)


def test_fit_probability_calibrator_uses_resolved_method_when_attached():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="lightgbm",
            probability_calibration_method="isotonic",
        )
    )
    trainer.oof_primary_proba = np.tile(np.array([0.15, 0.25, 0.35, 0.65, 0.75, 0.85]), 20)
    trainer.labels = pd.Series(np.tile(np.array([0, 0, 0, 1, 1, 1]), 20))

    trainer._fit_probability_calibrator_from_oof()

    assert trainer.probability_calibrator is not None
    assert trainer.training_metrics["probability_calibration_enabled"] == pytest.approx(1.0)
    assert trainer.training_metrics["probability_calibration_method"] == "sigmoid"

    fitted_model = SimpleNamespace()
    trainer._attach_probability_calibrator_to_model(fitted_model)

    payload = fitted_model._alphatrade_probability_calibration
    assert payload["method"] == "sigmoid"
    assert payload["calibrator"] is trainer.probability_calibrator


def test_get_predictions_proba_applies_attached_probability_calibration():
    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="lightgbm"))

    class DummyCalibrator:
        def predict(self, x):
            return np.clip(np.asarray(x, dtype=float) + 0.10, 0.0, 1.0)

    class DummyModel:
        def __init__(self):
            self._alphatrade_probability_calibration = {
                "method": "isotonic",
                "calibrator": DummyCalibrator(),
            }

        def predict_proba(self, X):
            raw = np.array([0.20, 0.55, 0.80], dtype=float)
            return np.column_stack([1.0 - raw, raw])

    calibrated = trainer._get_predictions_proba(DummyModel(), np.zeros((3, 2)))

    assert np.allclose(calibrated, np.array([0.30, 0.65, 0.90]))


def test_build_ranking_groups_segments_by_timestamp():
    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="lightgbm_ranker"))
    timestamps = np.array(
        [
            "2024-01-01T10:00:00",
            "2024-01-01T10:00:00",
            "2024-01-01T10:01:00",
            "2024-01-01T10:02:00",
            "2024-01-01T10:02:00",
            "2024-01-01T10:02:00",
        ],
        dtype="datetime64[ns]",
    )
    groups = trainer._build_ranking_groups(timestamps)
    assert groups.tolist() == [2, 1, 3]


def test_horizon_leaderboards_assign_per_horizon_candidates():
    rows = [
        {
            "model_type": "lightgbm_ranker",
            "primary_label_horizon": 5,
            "success": True,
            "governance_score": 0.91,
            "registry_version_id": "v5a",
            "promotion_package_path": "pkg5a.json",
            "deployment_plan": {"ready_for_production": True},
            "validation_layers": {
                "model_utility": True,
                "execution_robustness": True,
                "cross_symbol_robustness": True,
            },
            "run_id": "lightgbm_ranker_h5",
        },
        {
            "model_type": "xgboost",
            "primary_label_horizon": 20,
            "success": True,
            "governance_score": 0.88,
            "registry_version_id": "v20a",
            "promotion_package_path": "pkg20a.json",
            "deployment_plan": {"ready_for_production": True},
            "validation_layers": {
                "model_utility": True,
                "execution_robustness": True,
                "cross_symbol_robustness": True,
            },
            "run_id": "xgboost_h20",
        },
        {
            "model_type": "random_forest",
            "primary_label_horizon": 5,
            "success": True,
            "governance_score": 0.70,
            "registry_version_id": "v5b",
            "promotion_package_path": "pkg5b.json",
            "deployment_plan": {"ready_for_production": True},
            "validation_layers": {
                "model_utility": True,
                "execution_robustness": True,
                "cross_symbol_robustness": True,
            },
            "run_id": "rf_h5",
        },
    ]

    leaderboards = train_script._build_horizon_leaderboards(rows)
    assert leaderboards["5"]["champion"]["model_type"] == "lightgbm_ranker"
    assert leaderboards["20"]["champion"]["model_type"] == "xgboost"


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
            "primary_label_horizon": 5,
            "success": True,
            "governance_score": 0.8,
            "registry_version_id": second["version_id"],
            "promotion_package_path": "models/promotion_packages/lgb_b.json",
            "deployment_plan": {
                "ready_for_production": True,
                "canary_rollout": [{"phase": 1, "capital_fraction": 0.05}],
            },
            "validation_layers": {
                "model_utility": True,
                "execution_robustness": True,
                "cross_symbol_robustness": True,
            },
        },
        {
            "model_type": "xgboost",
            "primary_label_horizon": 20,
            "success": True,
            "governance_score": 0.5,
            "registry_version_id": first["version_id"],
            "promotion_package_path": "models/promotion_packages/xgb_a.json",
            "deployment_plan": {"ready_for_production": True},
            "validation_layers": {
                "model_utility": True,
                "execution_robustness": True,
                "cross_symbol_robustness": True,
            },
        },
    ]

    snapshot = train_script._auto_select_champion_and_challenger(tmp_path, rows)

    assert snapshot["promoted"] is True
    assert snapshot["champion"]["model_type"] == "lightgbm"
    assert "horizon_leaderboards" in snapshot
    assert snapshot["horizon_leaderboards"]["5"]["champion"]["model_type"] == "lightgbm"
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
    assert "white_reality_stat" in trainer.training_metrics
    assert "white_reality_pvalue" in trainer.training_metrics


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


def test_multiple_testing_correction_uses_timeframe_annualization(monkeypatch):
    captured: dict[str, float] = {}

    def _fake_white_reality_check(
        returns,
        n_bootstrap,
        block_size,
        random_seed,
        annualization_factor,
    ):
        captured["annualization_factor"] = float(annualization_factor)
        return 0.1, 0.05, "ok"

    monkeypatch.setattr(
        train_script,
        "calculate_white_reality_check",
        _fake_white_reality_check,
    )

    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            correction_method="deflated_sharpe",
            timeframe="1Hour",
            n_trials=10,
        )
    )
    trainer.training_metrics = {"mean_sharpe": 0.8}
    trainer.cv_results = [{"fold": 1}, {"fold": 2}]
    trainer.cv_return_series = [np.array([0.001, -0.001, 0.002, 0.0005] * 20, dtype=float)]

    trainer._apply_multiple_testing_correction()

    assert captured["annualization_factor"] == pytest.approx(7 * 252)
    assert trainer.training_metrics["white_reality_annualization_factor"] == pytest.approx(7 * 252)


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
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(model_type="xgboost", max_pbo=0.45)
    )

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


def test_validate_model_records_three_layer_gate_failures():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            use_meta_labeling=False,
        )
    )
    trainer.training_metrics = {
        "mean_sharpe": 1.0,
        "mean_accuracy": 0.70,
        "mean_max_drawdown": 0.10,
        "mean_win_rate": 0.60,
        "mean_trade_count": 40.0,
        "mean_test_size": 80.0,
        "mean_risk_adjusted_score": 0.30,
        "mean_regime_shift": 0.10,
        "mean_symbol_concentration_hhi": 0.20,
        "holdout_rows": 280.0,
        "holdout_sharpe": 0.20,
        "holdout_worst_regime_sharpe": 0.05,
        "holdout_max_drawdown": 0.12,
        "holdout_symbol_coverage_ratio": 0.80,
        "holdout_symbol_sharpe_p25": 0.00,
        "holdout_symbol_underwater_ratio": 0.20,
        "deflated_sharpe": 0.80,
        "deflated_sharpe_p_value": 0.20,
        "pbo": 0.20,
        "nested_cv_trace": [{"outer_fold": 1}],
        "objective_component_summary": {"objective_sharpe_component": 0.5},
    }
    trainer.nested_cv_trace = [{"outer_fold": 1}]

    passed = trainer._validate_model()

    assert passed is False
    assert trainer.validation_results["layers"]["model_utility"]["passed"] is False
    assert trainer.validation_results["layers"]["execution_robustness"]["passed"] is True
    assert trainer.validation_results["layers"]["cross_symbol_robustness"]["passed"] is True
    assert trainer.validation_results["all_layers_passed"] is False


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


def test_validate_model_fails_when_white_reality_gate_breaches():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            max_white_reality_pvalue=0.10,
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
        "white_reality_stat": 0.1,
        "white_reality_pvalue": 0.35,
        "nested_cv_trace": [{"outer_fold": 1}],
        "objective_component_summary": {"objective_sharpe_component": 0.5},
    }
    trainer.nested_cv_trace = [{"outer_fold": 1}]

    passed = trainer._validate_model()

    assert passed is False
    assert trainer.validation_results["gates"]["max_white_reality_pvalue"][0] is False


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
    rows = [{"symbol": symbol, "timestamp": ts} for symbol in symbols for ts in dates]
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
    rows = [{"symbol": symbol, "timestamp": ts} for symbol in symbols for ts in dates]
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

    trainer = train_script.ModelTrainer(train_script.TrainingConfig(model_type="xgboost"))
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
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

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


def test_run_training_require_gpu_fails_when_gpu_stack_unavailable(monkeypatch, caplog):
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: False)

    args = _base_args(model="lightgbm", require_gpu=True)
    with caplog.at_level(logging.ERROR):
        exit_code = train_script.run_training(args)

    assert exit_code == 1
    assert "GPU acceleration was required" in caplog.text


def test_run_training_replay_manifest_passes_dataset_snapshot_bundle(monkeypatch, tmp_path):
    captured = {}

    class DummyModelTrainer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return {"success": True, "training_metrics": {}}

    monkeypatch.setattr(train_script, "ModelTrainer", DummyModelTrainer)
    monkeypatch.setattr(train_script, "_verify_institutional_infra", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_verify_gpu_stack", lambda _model_list: True)

    bundle_path = tmp_path / "snapshots" / "snap_123" / "dataset_bundle.manifest.json"
    replay_path = tmp_path / "replay_manifest.json"
    replay_path.write_text(
        json.dumps(
            {
                "training_config": {
                    "model_type": "lightgbm",
                    "model_name": "prod_candidate",
                },
                "dataset_snapshot_bundle_path": str(bundle_path),
            }
        ),
        encoding="utf-8",
    )

    args = _base_args(replay_manifest=str(replay_path))
    exit_code = train_script.run_training(args)

    assert exit_code == 0
    assert captured["config"].dataset_snapshot_bundle_path == str(bundle_path)
    assert captured["config"].strict_snapshot_replay is True


def test_load_training_dataset_snapshot_restores_state(tmp_path):
    raw = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "timestamp": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-01T00:15:00Z", "2025-01-01T00:30:00Z"],
                utc=True,
            ),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000.0, 1100.0, 1200.0],
        }
    )
    quality_report = build_data_quality_report(raw.loc[:, ["symbol", "timestamp", "close"]])
    quality_hash = compute_data_quality_hash(quality_report)
    snapshot_manifest = build_snapshot_manifest(
        ohlcv_data=raw,
        feature_names=["f_alpha", "f_beta"],
        data_quality_report_hash=quality_hash,
    )
    dev_frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-01T00:15:00Z"],
                utc=True,
            ),
            "symbol": ["AAPL", "AAPL"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000.0, 1100.0],
            "label": [1, 0],
            "regime": ["trend_up", "normal_range"],
            "forward_return_h5": [0.01, -0.02],
            "triple_barrier_net_return": [0.008, -0.015],
            "f_alpha": [0.1, 0.2],
            "f_beta": [1.0, 0.5],
        }
    )
    holdout_frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01T00:30:00Z"], utc=True),
            "symbol": ["AAPL"],
            "open": [102.0],
            "high": [103.0],
            "low": [101.0],
            "close": [102.5],
            "volume": [1200.0],
            "label": [1],
            "regime": ["high_vol"],
            "forward_return_h5": [0.03],
            "triple_barrier_net_return": [0.02],
            "f_alpha": [0.3],
            "f_beta": [0.8],
        }
    )
    bundle_path, _ = persist_dataset_snapshot_bundle(
        output_dir=tmp_path,
        snapshot_manifest=snapshot_manifest,
        raw_ohlcv_data=raw,
        development_frame=dev_frame,
        holdout_frame=holdout_frame,
        feature_names=["f_alpha", "f_beta"],
        data_quality_report=quality_report,
        development_sample_weights=np.array([1.0, 0.8]),
        holdout_sample_weights=np.array([0.9]),
        cv_splits=[(np.array([0]), np.array([1]))],
        label_diagnostics={"positive_rate": 0.5},
    )

    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            dataset_snapshot_bundle_path=str(bundle_path),
            strict_snapshot_replay=True,
            primary_label_horizon=5,
        )
    )

    assert trainer._load_training_dataset_snapshot() is True
    assert trainer.snapshot_replay_loaded is True
    assert trainer.features.columns.tolist() == ["f_alpha", "f_beta"]
    assert trainer.labels.tolist() == [1, 0]
    assert trainer.holdout_labels.tolist() == [1]
    assert trainer.sample_weights.tolist() == [1.0, 0.8]
    assert trainer.holdout_sample_weights.tolist() == [0.9]
    assert len(trainer.cached_cv_splits) == 1


def test_load_features_from_postgres_recomputes_when_ohlcv_hash_changes(monkeypatch):
    import quant_trading_system.database.connection as conn_module

    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            symbols=["AAPL"],
            timeframe="15Min",
        )
    )
    trainer.data = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "timestamp": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-01T00:15:00Z"],
                utc=True,
            ),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000.0, 1100.0],
        }
    )
    feature_set_id = trainer._resolve_feature_set_id(["AAPL"])
    schema_payload = {
        "pipeline_fingerprint": trainer.feature_pipeline_fingerprint,
        "source_ohlcv_hash": "stale_hash",
        "feature_names": ["f_alpha"],
        "feature_groups": sorted(trainer.config.feature_groups),
        "enable_cross_sectional": bool(trainer.config.enable_cross_sectional),
        "enable_reference_features": bool(trainer.config.enable_reference_features),
        "timeframe": trainer.config.timeframe,
        "feature_set_id": feature_set_id,
    }

    class FakeRedis:
        def get(self, key):
            if "train:features:schema:" in key:
                return json.dumps(schema_payload)
            return None

        def set(self, *args, **kwargs):
            return None

    monkeypatch.setattr(conn_module, "get_db_manager", lambda: MagicMock())
    monkeypatch.setattr(conn_module, "get_redis_manager", lambda: FakeRedis())

    assert trainer._load_features_from_postgres() is None


def test_load_features_from_postgres_recomputes_when_feature_selection_signature_changes(
    monkeypatch,
):
    import quant_trading_system.database.connection as conn_module

    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            symbols=["AAPL"],
            timeframe="15Min",
        )
    )
    trainer.data = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "timestamp": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-01T00:15:00Z"],
                utc=True,
            ),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000.0, 1100.0],
        }
    )
    feature_set_id = trainer._resolve_feature_set_id(["AAPL"])
    schema_payload = {
        "pipeline_fingerprint": trainer.feature_pipeline_fingerprint,
        "source_ohlcv_hash": trainer._current_ohlcv_fingerprint(),
        "feature_names": ["f_alpha"],
        "feature_groups": sorted(trainer.config.feature_groups),
        "enable_cross_sectional": bool(trainer.config.enable_cross_sectional),
        "enable_reference_features": bool(trainer.config.enable_reference_features),
        "enable_tick_microstructure_features": bool(
            trainer.config.enable_tick_microstructure_features
        ),
        "timeframe": trainer.config.timeframe,
        "timeframes": list(trainer.config.timeframes),
        "training_bar_mode": trainer.config.training_bar_mode,
        "intrinsic_bar_type": trainer.config.intrinsic_bar_type,
        "feature_set_id": feature_set_id,
        "cache_stage": "post_selection",
        "feature_selection_signature": {
            **trainer._feature_selection_schema_signature(),
            "max_features": trainer.config.feature_selection_max_features - 1,
        },
    }

    class FakeRedis:
        def get(self, key):
            if "train:features:schema:" in key:
                return json.dumps(schema_payload)
            return None

        def set(self, *args, **kwargs):
            return None

    monkeypatch.setattr(conn_module, "get_db_manager", lambda: MagicMock())
    monkeypatch.setattr(conn_module, "get_redis_manager", lambda: FakeRedis())

    assert trainer._load_features_from_postgres() is None


def test_snapshot_training_scope_tracks_market_session_policy():
    regular_scope = train_script._build_snapshot_training_scope(
        train_script.TrainingConfig(model_type="lightgbm")
    )
    extended_scope = train_script._build_snapshot_training_scope(
        train_script.TrainingConfig(model_type="lightgbm", include_premarket=True)
    )

    assert regular_scope["include_premarket"] is False
    assert regular_scope["include_postmarket"] is False
    assert extended_scope["include_premarket"] is True
    assert regular_scope != extended_scope


def test_apply_market_session_filter_removes_out_of_session_rows():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(model_type="lightgbm", timeframe="15Min")
    )
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 5,
            "timestamp": pd.to_datetime(
                [
                    "2024-01-16T13:00:00Z",
                    "2024-01-16T14:30:00Z",
                    "2024-01-16T14:45:00Z",
                    "2024-01-16T20:45:00Z",
                    "2024-01-16T21:00:00Z",
                ],
                utc=True,
            ),
            "open": [99.0, 100.0, 101.0, 102.0, 103.0],
            "high": [100.0, 101.0, 102.0, 103.0, 104.0],
            "low": [98.0, 99.0, 100.0, 101.0, 102.0],
            "close": [99.5, 100.5, 101.5, 102.5, 103.5],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        }
    )
    trainer.data = frame.copy()
    trainer.ohlcv_panels_by_timeframe = {"15Min": frame.copy()}

    trainer._apply_market_session_filter()

    assert trainer.data["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist() == [
        "2024-01-16T14:30:00Z",
        "2024-01-16T14:45:00Z",
        "2024-01-16T20:45:00Z",
    ]
    assert trainer.training_metrics["market_session_filter_applied"] is True
    assert trainer.training_metrics["market_session_rows_removed"] == pytest.approx(2.0)
    assert len(trainer.ohlcv_panels_by_timeframe["15Min"]) == 3


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


def test_compute_net_returns_prefers_cost_aware_event_returns_without_double_costs():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            dynamic_no_trade_band=False,
            execution_cooldown_bars=0,
        )
    )
    y_true = np.array([1, 1, 1, 1], dtype=int)
    y_proba = np.array([0.8, 0.8, 0.8, 0.8], dtype=float)
    realized = np.array([0.05, 0.05, 0.05, 0.05], dtype=float)
    event_net = np.array([0.01, 0.01, 0.01, 0.01], dtype=float)
    event_directions = np.ones(4, dtype=float)

    net, details = trainer._compute_net_returns(
        y_true=y_true,
        y_proba=y_proba,
        realized_forward_returns=realized,
        event_net_returns=event_net,
        event_directions=event_directions,
        return_details=True,
    )
    event_only = trainer._compute_net_returns(
        y_true=y_true,
        y_proba=y_proba,
        event_net_returns=event_net,
        event_directions=event_directions,
    )
    raw_only = trainer._compute_net_returns(
        y_true=y_true,
        y_proba=y_proba,
        realized_forward_returns=realized,
    )

    np.testing.assert_allclose(net, np.asarray(event_only, dtype=float), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        net,
        np.asarray(details["positions"], dtype=float) * event_net,
        rtol=1e-12,
        atol=1e-12,
    )
    assert not np.allclose(net, np.asarray(raw_only, dtype=float))


def test_compute_net_returns_uses_event_directions_for_short_side_events():
    trainer = train_script.ModelTrainer(
        train_script.TrainingConfig(
            model_type="xgboost",
            dynamic_no_trade_band=False,
            execution_cooldown_bars=0,
        )
    )
    y_true = np.array([1, 1, 1], dtype=int)
    y_proba = np.array([0.1, 0.1, 0.1], dtype=float)
    event_net = np.array([0.01, 0.01, 0.01], dtype=float)
    event_directions = np.full(3, -1.0, dtype=float)

    net = trainer._compute_net_returns(
        y_true=y_true,
        y_proba=y_proba,
        event_net_returns=event_net,
        event_directions=event_directions,
    )

    assert np.all(np.asarray(net, dtype=float) > 0.0)


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
