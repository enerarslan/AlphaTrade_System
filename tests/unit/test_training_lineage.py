"""Unit tests for training lineage snapshot and registry helpers."""

from __future__ import annotations

import json

import pandas as pd

from quant_trading_system.models.training_lineage import (
    build_data_quality_report,
    build_snapshot_manifest,
    compute_data_quality_hash,
    persist_snapshot_bundle,
    register_training_model_version,
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
