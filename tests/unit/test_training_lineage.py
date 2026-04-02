"""Unit tests for training lineage snapshot and registry helpers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quant_trading_system.models.training_lineage import (
    append_training_run_event,
    build_data_quality_report,
    build_snapshot_manifest,
    compute_data_quality_hash,
    estimate_missing_bars_count,
    load_dataset_snapshot_bundle,
    load_registry_entries,
    persist_dataset_snapshot_bundle,
    persist_active_model_pointer,
    persist_snapshot_bundle,
    register_training_model_version,
    set_registry_active_version,
)


def test_build_data_quality_report_detects_missing_duplicate_and_outlier() -> None:
    data = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
            "timestamp": [
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:15:00Z",
                "2025-01-01T00:15:00Z",  # duplicate
                "2025-01-01T00:30:00Z",
                "2025-01-01T01:00:00Z",  # leaves one missing bar at 00:45
            ],
            "close": [100.0, 101.0, 101.0, 102.0, 150.0],  # extreme move
        }
    )

    report = build_data_quality_report(data)
    summary = report["summary"]

    assert summary["missing_bars_count"] >= 1
    assert summary["duplicate_bars_count"] >= 1
    assert summary["extreme_move_count"] >= 1
    assert "missing_bars" in report["threshold_breaches"]


def test_estimate_missing_bars_count_ignores_daily_weekends() -> None:
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2025-01-03T21:00:00Z",  # Friday
                "2025-01-06T21:00:00Z",  # Monday
                "2025-01-07T21:00:00Z",  # Tuesday
            ],
            utc=True,
        )
    )

    missing_count, inferred_bar_seconds = estimate_missing_bars_count(timestamps)

    assert missing_count == 0
    assert inferred_bar_seconds is not None
    assert inferred_bar_seconds >= 24 * 3600


def test_estimate_missing_bars_count_detects_daily_business_day_gaps() -> None:
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2025-01-06T21:00:00Z",  # Monday
                "2025-01-07T21:00:00Z",  # Tuesday
                "2025-01-10T21:00:00Z",  # Friday, missing Wed/Thu
            ],
            utc=True,
        )
    )

    missing_count, inferred_bar_seconds = estimate_missing_bars_count(timestamps)

    assert missing_count == 2
    assert inferred_bar_seconds is not None


def test_append_training_run_event_writes_jsonl_payload(tmp_path) -> None:
    index_path = append_training_run_event(
        tmp_path / "run_index",
        {
            "status": "completed",
            "run_id": "lightgbm_ranker_20260402T000000Z",
            "metrics": {"mean_sharpe": 1.25},
        },
    )

    payload = json.loads(index_path.read_text(encoding="utf-8").strip())
    assert index_path.name == "training_runs.jsonl"
    assert payload["status"] == "completed"
    assert payload["run_id"] == "lightgbm_ranker_20260402T000000Z"
    assert payload["metrics"]["mean_sharpe"] == 1.25
    assert payload["schema_version"] == "1.0.0"
    assert "recorded_at" in payload


def test_build_snapshot_manifest_is_deterministic_for_same_inputs() -> None:
    data = pd.DataFrame(
        {
            "symbol": ["MSFT", "AAPL", "AAPL", "MSFT"],
            "timestamp": [
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:15:00Z",
                "2025-01-01T00:15:00Z",
            ],
            "close": [10.0, 20.0, 21.0, 11.0],
        }
    )

    quality_hash = compute_data_quality_hash(build_data_quality_report(data))
    first = build_snapshot_manifest(
        ohlcv_data=data,
        feature_names=["f_2", "f_1", "f_1"],
        data_quality_report_hash=quality_hash,
        requested_start_date="2025-01-01",
        requested_end_date="2025-01-31",
    )
    second = build_snapshot_manifest(
        ohlcv_data=data,
        feature_names=["f_1", "f_2"],
        data_quality_report_hash=quality_hash,
        requested_start_date="2025-01-01",
        requested_end_date="2025-01-31",
    )

    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["feature_schema_version"] == second["feature_schema_version"]


def test_register_training_model_version_writes_dashboard_compatible_registry(tmp_path) -> None:
    data = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T00:15:00Z"],
            "close": [100.0, 101.0],
        }
    )
    report = build_data_quality_report(data)
    quality_hash = compute_data_quality_hash(report)
    manifest = build_snapshot_manifest(
        ohlcv_data=data,
        feature_names=["ret_1", "vol_20"],
        data_quality_report_hash=quality_hash,
    )

    manifest_path, quality_path = persist_snapshot_bundle(tmp_path, manifest, report)
    manifest["manifest_path"] = str(manifest_path)
    manifest["quality_report_path"] = str(quality_path)

    entry = register_training_model_version(
        registry_root=tmp_path / "registry",
        model_name="xgboost",
        model_version="xgb_20250101",
        model_type="xgboost",
        model_path=str(tmp_path / "xgb_20250101.pkl"),
        metrics={"mean_accuracy": 0.61, "ignored": "n/a"},
        tags=["unit_test"],
        is_active=True,
        snapshot_manifest=manifest,
        training_config={"cv_method": "purged_kfold", "n_splits": 5},
        project_root=tmp_path,
    )

    registry_file = tmp_path / "registry" / "registry.json"
    assert registry_file.exists()

    payload = json.loads(registry_file.read_text(encoding="utf-8"))
    assert "xgboost" in payload
    assert isinstance(payload["xgboost"], list)
    assert payload["xgboost"][0]["version_id"] == entry["version_id"]
    assert payload["xgboost"][0]["metrics"]["mean_accuracy"] == 0.61
    assert "ignored" not in payload["xgboost"][0]["metrics"]
    assert payload["xgboost"][0]["lineage"]["snapshot_id"] == manifest["snapshot_id"]


def test_set_registry_active_version_and_persist_active_model_pointer(tmp_path) -> None:
    first = register_training_model_version(
        registry_root=tmp_path / "registry",
        model_name="xgboost",
        model_version="xgb_a",
        model_type="xgboost",
        model_path=str(tmp_path / "xgb_a.pkl"),
        metrics={"mean_accuracy": 0.55},
        is_active=False,
    )
    second = register_training_model_version(
        registry_root=tmp_path / "registry",
        model_name="lightgbm",
        model_version="lgb_b",
        model_type="lightgbm",
        model_path=str(tmp_path / "lgb_b.pkl"),
        metrics={"mean_accuracy": 0.60},
        is_active=False,
    )

    promoted = set_registry_active_version(
        registry_root=tmp_path / "registry",
        model_name="lightgbm",
        version_id=second["version_id"],
    )
    assert promoted is True

    entries = load_registry_entries(tmp_path / "registry")
    active_entries = [row for row in entries if bool(row.get("is_active", False))]
    assert len(active_entries) == 1
    assert active_entries[0]["version_id"] == second["version_id"]
    assert active_entries[0]["version_id"] != first["version_id"]

    pointer_path = persist_active_model_pointer(
        models_root=tmp_path,
        model_name="lightgbm",
        version_id=second["version_id"],
        updated_by="unit_test",
        reason="auto_benchmark_promotion",
    )
    assert pointer_path.exists()
    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    assert payload["model_name"] == "lightgbm"
    assert payload["version_id"] == second["version_id"]
    assert payload["updated_by"] == "unit_test"
    assert payload["reason"] == "auto_benchmark_promotion"


def test_persist_and_load_dataset_snapshot_bundle_round_trip(tmp_path) -> None:
    raw = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "timestamp": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-01T00:15:00Z", "2025-01-01T00:00:00Z"],
                utc=True,
            ),
            "open": [100.0, 101.0, 200.0],
            "high": [101.0, 102.0, 201.0],
            "low": [99.0, 100.0, 199.0],
            "close": [100.5, 101.5, 200.5],
            "volume": [1000.0, 1100.0, 900.0],
        }
    )
    quality_report = build_data_quality_report(raw.loc[:, ["symbol", "timestamp", "close"]])
    quality_hash = compute_data_quality_hash(quality_report)
    snapshot_manifest = build_snapshot_manifest(
        ohlcv_data=raw,
        feature_names=["f_alpha", "f_beta"],
        data_quality_report_hash=quality_hash,
    )

    development_frame = pd.DataFrame(
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

    bundle_path, bundle_manifest = persist_dataset_snapshot_bundle(
        output_dir=tmp_path,
        snapshot_manifest=snapshot_manifest,
        raw_ohlcv_data=raw,
        development_frame=development_frame,
        holdout_frame=holdout_frame,
        feature_names=["f_beta", "f_alpha"],
        training_scope={"timeframe": "15Min", "feature_set_id": "default"},
        data_quality_report=quality_report,
        development_sample_weights=np.array([1.0, 0.8]),
        holdout_sample_weights=np.array([0.9]),
        cv_splits=[(np.array([0]), np.array([1]))],
        label_diagnostics={"positive_rate": 0.5},
    )

    assert bundle_path.exists()
    assert bundle_manifest["snapshot_id"] == snapshot_manifest["snapshot_id"]

    loaded = load_dataset_snapshot_bundle(bundle_path)

    pd.testing.assert_frame_equal(loaded["raw_ohlcv_data"], raw)
    pd.testing.assert_frame_equal(loaded["development_frame"], development_frame)
    pd.testing.assert_frame_equal(loaded["holdout_frame"], holdout_frame)
    assert loaded["feature_names"] == ["f_alpha", "f_beta"]
    assert loaded["bundle_manifest"]["training_scope"] == {
        "timeframe": "15Min",
        "feature_set_id": "default",
    }
    assert loaded["development_sample_weights"].tolist() == [1.0, 0.8]
    assert loaded["holdout_sample_weights"].tolist() == [0.9]
    assert loaded["cv_splits"][0][0].tolist() == [0]
    assert loaded["cv_splits"][0][1].tolist() == [1]
    assert loaded["data_quality_report"]["summary"]["rows_total"] == 3
