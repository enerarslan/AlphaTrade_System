"""Training lineage and governance helpers for institutional model training."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SNAPSHOT_MANIFEST_SCHEMA_VERSION = "1.0.0"
MODEL_REGISTRY_SCHEMA_VERSION = "1.0.0"

DEFAULT_DATA_QUALITY_THRESHOLDS: dict[str, float] = {
    "missing_bars_ratio_max": 0.01,
    "duplicate_bars_ratio_max": 0.001,
    "extreme_move_ratio_max": 0.01,
    "corporate_action_jump_ratio_max": 0.001,
}


def _canonical_hash(payload: Any) -> str:
    """Create deterministic SHA-256 hash for JSON-like payloads."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _to_utc_timestamp(value: Any) -> pd.Timestamp | None:
    """Coerce value to UTC timestamp."""
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _to_iso(value: Any) -> str | None:
    """Convert datetime-like value to ISO-8601 UTC string."""
    ts = _to_utc_timestamp(value)
    return ts.isoformat() if ts is not None else None


def _safe_float(value: Any) -> float:
    """Convert numeric value to finite float."""
    try:
        casted = float(value)
    except (TypeError, ValueError):
        return 0.0
    return float(np.nan_to_num(casted, nan=0.0, posinf=0.0, neginf=0.0))


def build_data_quality_report(
    ohlcv_data: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build deterministic data quality report for OHLCV dataset."""
    required_cols = {"symbol", "timestamp", "close"}
    missing_cols = sorted(required_cols.difference(ohlcv_data.columns))
    if missing_cols:
        raise ValueError(f"Data quality report requires columns: {missing_cols}")

    merged_thresholds = dict(DEFAULT_DATA_QUALITY_THRESHOLDS)
    if thresholds:
        for key, value in thresholds.items():
            if key in merged_thresholds:
                merged_thresholds[key] = _safe_float(value)

    df = ohlcv_data.loc[:, ["symbol", "timestamp", "close"]].copy()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["symbol", "timestamp", "close"])
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("Data quality report cannot be generated from empty dataset.")

    duplicate_count = int(df.duplicated(subset=["symbol", "timestamp"]).sum())
    unique_df = df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    unique_df = unique_df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    missing_bars_count = 0
    extreme_move_count = 0
    corporate_action_jump_count = 0
    total_return_observations = 0
    per_symbol: dict[str, dict[str, Any]] = {}

    for symbol, group in unique_df.groupby("symbol", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        timestamps = group["timestamp"]
        symbol_missing = 0
        inferred_bar_seconds = None

        if len(timestamps) >= 2:
            diffs = timestamps.diff().dropna().dt.total_seconds()
            positive_diffs = diffs[diffs > 0]
            if not positive_diffs.empty:
                inferred_bar_seconds = float(np.median(positive_diffs.to_numpy()))
                if inferred_bar_seconds > 0:
                    regular_gap_limit = inferred_bar_seconds * 1.5
                    # Treat long gaps (overnight/weekend/session breaks) as non-missing.
                    # Missing bars are counted only inside regular trading segments.
                    session_break_limit = max(inferred_bar_seconds * 24.0, 6.0 * 3600.0)
                    missing_from_gaps = np.where(
                        (positive_diffs > regular_gap_limit) & (positive_diffs <= session_break_limit),
                        np.floor(positive_diffs / inferred_bar_seconds) - 1.0,
                        0.0,
                    )
                    symbol_missing = int(np.maximum(missing_from_gaps, 0.0).sum())

        returns = group["close"].pct_change()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        symbol_extreme = int((returns.abs() > 0.20).sum())
        symbol_corporate_action_jump = int((returns.abs() > 0.35).sum())

        missing_bars_count += symbol_missing
        extreme_move_count += symbol_extreme
        corporate_action_jump_count += symbol_corporate_action_jump
        total_return_observations += int(len(returns))

        per_symbol[str(symbol)] = {
            "rows": int(len(group)),
            "start": _to_iso(timestamps.iloc[0]) if len(timestamps) else None,
            "end": _to_iso(timestamps.iloc[-1]) if len(timestamps) else None,
            "inferred_bar_seconds": _safe_float(inferred_bar_seconds) if inferred_bar_seconds else None,
            "missing_bars_count": int(symbol_missing),
            "extreme_move_count": int(symbol_extreme),
            "corporate_action_jump_count": int(symbol_corporate_action_jump),
        }

    observed_unique_rows = int(len(unique_df))
    expected_rows = observed_unique_rows + missing_bars_count
    missing_bars_ratio = _safe_float(
        missing_bars_count / expected_rows if expected_rows > 0 else 0.0
    )
    duplicate_bars_ratio = _safe_float(
        duplicate_count / len(df) if len(df) > 0 else 0.0
    )
    extreme_move_ratio = _safe_float(
        extreme_move_count / total_return_observations if total_return_observations > 0 else 0.0
    )
    corporate_action_jump_ratio = _safe_float(
        corporate_action_jump_count / total_return_observations
        if total_return_observations > 0
        else 0.0
    )

    breaches = {
        "missing_bars": missing_bars_ratio > merged_thresholds["missing_bars_ratio_max"],
        "duplicate_bars": duplicate_bars_ratio > merged_thresholds["duplicate_bars_ratio_max"],
        "extreme_move_outliers": extreme_move_ratio > merged_thresholds["extreme_move_ratio_max"],
        "corporate_action_consistency": (
            corporate_action_jump_ratio > merged_thresholds["corporate_action_jump_ratio_max"]
        ),
    }

    return {
        "schema_version": SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "rows_total": int(len(df)),
            "rows_unique_symbol_timestamp": observed_unique_rows,
            "symbol_count": int(unique_df["symbol"].nunique()),
            "missing_bars_count": int(missing_bars_count),
            "missing_bars_ratio": missing_bars_ratio,
            "duplicate_bars_count": int(duplicate_count),
            "duplicate_bars_ratio": duplicate_bars_ratio,
            "extreme_move_count": int(extreme_move_count),
            "extreme_move_ratio": extreme_move_ratio,
            "corporate_action_jump_count": int(corporate_action_jump_count),
            "corporate_action_jump_ratio": corporate_action_jump_ratio,
            "return_observations": int(total_return_observations),
        },
        "thresholds": merged_thresholds,
        "threshold_breaches": breaches,
        "passed": not any(breaches.values()),
        "per_symbol": per_symbol,
    }


def compute_data_quality_hash(quality_report: dict[str, Any]) -> str:
    """Compute stable hash for data quality report (excluding volatile timestamps)."""
    canonical_payload = {
        "schema_version": quality_report.get("schema_version"),
        "summary": quality_report.get("summary", {}),
        "thresholds": quality_report.get("thresholds", {}),
        "threshold_breaches": quality_report.get("threshold_breaches", {}),
        "passed": bool(quality_report.get("passed", False)),
        "per_symbol": quality_report.get("per_symbol", {}),
    }
    return _canonical_hash(canonical_payload)


def build_snapshot_manifest(
    ohlcv_data: pd.DataFrame,
    feature_names: list[str],
    data_quality_report_hash: str,
    requested_start_date: str | None = None,
    requested_end_date: str | None = None,
    source_system: str = "postgresql_timescaledb",
) -> dict[str, Any]:
    """Build deterministic dataset snapshot manifest for one training run."""
    if ohlcv_data.empty:
        raise ValueError("Snapshot manifest cannot be generated for empty OHLCV dataset.")
    if not feature_names:
        raise ValueError("Snapshot manifest requires at least one feature name.")

    df = ohlcv_data.loc[:, ["symbol", "timestamp"]].copy()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["symbol", "timestamp"])
    if df.empty:
        raise ValueError("Snapshot manifest cannot be generated from invalid timestamps.")

    symbols = sorted(df["symbol"].unique().tolist())
    observed_start = _to_iso(df["timestamp"].min())
    observed_end = _to_iso(df["timestamp"].max())
    sorted_features = sorted({str(name) for name in feature_names})
    feature_schema_version = _canonical_hash(sorted_features)

    snapshot_basis = {
        "symbols": symbols,
        "observed_start": observed_start,
        "observed_end": observed_end,
        "row_count": int(len(df)),
        "feature_schema_version": feature_schema_version,
        "data_quality_report_hash": data_quality_report_hash,
        "requested_start_date": requested_start_date or "",
        "requested_end_date": requested_end_date or "",
        "source_system": source_system,
    }
    snapshot_id = f"snap_{_canonical_hash(snapshot_basis)[:16]}"

    return {
        "schema_version": SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_system": source_system,
        "symbol_universe": symbols,
        "row_count": int(len(df)),
        "observed_date_bounds": {
            "start": observed_start,
            "end": observed_end,
        },
        "requested_date_bounds": {
            "start": requested_start_date,
            "end": requested_end_date,
        },
        "feature_count": len(sorted_features),
        "feature_list": sorted_features,
        "feature_schema_version": feature_schema_version,
        "data_quality_report_hash": data_quality_report_hash,
    }


def persist_snapshot_bundle(
    output_dir: Path,
    manifest: dict[str, Any],
    quality_report: dict[str, Any],
) -> tuple[Path, Path]:
    """Persist snapshot manifest and quality report for training lineage."""
    snapshot_id = str(manifest.get("snapshot_id") or "").strip()
    if not snapshot_id:
        raise ValueError("Snapshot manifest missing snapshot_id.")

    snapshots_dir = Path(output_dir) / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = snapshots_dir / f"{snapshot_id}.manifest.json"
    quality_path = snapshots_dir / f"{snapshot_id}.quality.json"

    manifest_payload = dict(manifest)
    manifest_payload["manifest_path"] = str(manifest_path)
    manifest_payload["quality_report_path"] = str(quality_path)

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2, ensure_ascii=True, sort_keys=True, default=str)

    with quality_path.open("w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=True, sort_keys=True, default=str)

    return manifest_path, quality_path


def _resolve_git_commit_hash(project_root: Path) -> str:
    """Resolve current git commit hash, or 'unknown' outside git environments."""
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return "unknown"

    candidate = (result.stdout or "").strip()
    if result.returncode != 0 or len(candidate) < 7:
        return "unknown"
    return candidate


def _dependency_lock_hash(project_root: Path) -> str:
    """Hash dependency lock inputs used for training reproducibility."""
    dependency_files = [
        project_root / "pyproject.toml",
        project_root / "requirements.txt",
    ]
    hasher = hashlib.sha256()
    found = False
    for file_path in dependency_files:
        if not file_path.exists():
            continue
        found = True
        hasher.update(file_path.name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(file_path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest() if found else "unknown"


def _sanitize_metrics(metrics: dict[str, Any] | None) -> dict[str, float]:
    """Keep numeric and finite metrics only."""
    if not isinstance(metrics, dict):
        return {}
    sanitized: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            sanitized[str(key)] = _safe_float(value)
    return sanitized


def register_training_model_version(
    registry_root: Path,
    model_name: str,
    model_version: str,
    model_type: str,
    model_path: str | None,
    metrics: dict[str, Any] | None,
    tags: list[str] | None = None,
    is_active: bool = False,
    snapshot_manifest: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    project_root: Path | None = None,
    model_card: dict[str, Any] | None = None,
    deployment_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one model version record to dashboard-compatible registry."""
    registry_root = Path(registry_root)
    registry_root.mkdir(parents=True, exist_ok=True)
    registry_file = registry_root / "registry.json"

    payload: dict[str, list[dict[str, Any]]]
    if registry_file.exists():
        try:
            with registry_file.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            payload = loaded if isinstance(loaded, dict) else {}
        except Exception:
            payload = {}
    else:
        payload = {}

    versions = payload.setdefault(model_name, [])
    registered_at = datetime.now(timezone.utc)
    version_stamp = registered_at.strftime("%Y%m%dT%H%M%SZ")
    version_id = f"{model_version}_{version_stamp}"

    root = project_root or Path.cwd()
    lineage: dict[str, Any] = {
        "code_commit_hash": _resolve_git_commit_hash(root),
        "dependency_lock_hash": _dependency_lock_hash(root),
        "python_version": sys.version.split()[0],
    }

    if isinstance(snapshot_manifest, dict):
        lineage.update(
            {
                "snapshot_id": snapshot_manifest.get("snapshot_id"),
                "snapshot_manifest_hash": _canonical_hash(snapshot_manifest),
                "feature_schema_version": snapshot_manifest.get("feature_schema_version"),
                "data_quality_report_hash": snapshot_manifest.get("data_quality_report_hash"),
                "snapshot_manifest_path": snapshot_manifest.get("manifest_path"),
                "snapshot_quality_report_path": snapshot_manifest.get("quality_report_path"),
            }
        )

    config_fields = [
        "model_type",
        "symbols",
        "start_date",
        "end_date",
        "cv_method",
        "n_splits",
        "embargo_pct",
        "n_trials",
        "use_gpu",
        "holdout_pct",
        "min_holdout_sharpe",
        "max_holdout_drawdown",
        "max_regime_shift",
        "execution_vol_target_daily",
        "execution_turnover_cap",
        "execution_cooldown_bars",
        "objective_weight_cvar",
        "objective_weight_skew",
    ]
    config_summary: dict[str, Any] = {}
    if isinstance(training_config, dict):
        for key in config_fields:
            if key in training_config:
                config_summary[key] = training_config.get(key)

    entry = {
        "version_id": version_id,
        "model_name": model_name,
        "model_version": model_version,
        "model_type": model_type,
        "registered_at": registered_at.isoformat(),
        "metrics": _sanitize_metrics(metrics),
        "tags": [str(tag) for tag in tags] if tags else [],
        "description": "Institutional training pipeline registration",
        "is_active": bool(is_active),
        "path": str(model_path or ""),
        "registry_schema_version": MODEL_REGISTRY_SCHEMA_VERSION,
        "lineage": lineage,
        "training_config": config_summary,
        "model_card": model_card if isinstance(model_card, dict) else {},
        "deployment_plan": deployment_plan if isinstance(deployment_plan, dict) else {},
    }

    versions.append(entry)

    with registry_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=True, default=str)

    return entry


def load_registry_entries(registry_root: Path) -> list[dict[str, Any]]:
    """Load flattened registry entries sorted by recency."""
    registry_file = Path(registry_root) / "registry.json"
    if not registry_file.exists():
        return []

    try:
        with registry_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    if not isinstance(payload, dict):
        return []

    rows: list[dict[str, Any]] = []
    for _, versions in payload.items():
        if not isinstance(versions, list):
            continue
        for entry in versions:
            if isinstance(entry, dict):
                rows.append(entry)

    def _sort_key(entry: dict[str, Any]) -> str:
        return str(entry.get("registered_at", ""))

    rows.sort(key=_sort_key, reverse=True)
    return rows


def set_registry_active_version(
    registry_root: Path,
    model_name: str,
    version_id: str,
) -> bool:
    """Set exactly one active version across registry entries."""
    registry_file = Path(registry_root) / "registry.json"
    if not registry_file.exists():
        return False

    try:
        with registry_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False

    matched = False
    for _, versions in payload.items():
        if not isinstance(versions, list):
            continue
        for entry in versions:
            if not isinstance(entry, dict):
                continue
            is_match = (
                str(entry.get("model_name", "")) == str(model_name)
                and str(entry.get("version_id", "")) == str(version_id)
            )
            entry["is_active"] = bool(is_match)
            if is_match:
                matched = True

    if not matched:
        return False

    with registry_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=True, default=str)
    return True


def persist_active_model_pointer(
    models_root: Path,
    model_name: str,
    version_id: str,
    updated_by: str = "training_pipeline",
    reason: str = "auto_promotion",
) -> Path:
    """Persist champion pointer compatible with dashboard expectations."""
    pointer_path = Path(models_root) / "active_model.json"
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": str(model_name),
        "version_id": str(version_id),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "updated_by": str(updated_by),
        "reason": str(reason),
    }
    with pointer_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True, default=str)
    return pointer_path
