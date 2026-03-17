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
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

SNAPSHOT_MANIFEST_SCHEMA_VERSION = "1.0.0"
MODEL_REGISTRY_SCHEMA_VERSION = "1.0.0"
DATASET_SNAPSHOT_BUNDLE_SCHEMA_VERSION = "1.0.0"

DEFAULT_DATA_QUALITY_THRESHOLDS: dict[str, float] = {
    "missing_bars_ratio_max": 0.01,
    "duplicate_bars_ratio_max": 0.001,
    "extreme_move_ratio_max": 0.01,
    "corporate_action_jump_ratio_max": 0.001,
}
_US_TRADING_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
_DAILY_BAR_SECONDS_FLOOR = 20.0 * 3600.0


def _canonical_hash(payload: Any) -> str:
    """Create deterministic SHA-256 hash for JSON-like payloads."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    """Hash a file deterministically."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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


def _count_missing_daily_bars(timestamps: pd.Series) -> int:
    """Estimate missing bars for daily data without penalizing weekends/holidays."""
    if timestamps.empty:
        return 0

    normalized = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().dt.normalize()
    if len(normalized) < 2:
        return 0

    missing = 0
    for prev, curr in zip(normalized.iloc[:-1], normalized.iloc[1:], strict=False):
        if curr <= prev:
            continue
        expected = len(pd.date_range(prev, curr, freq=_US_TRADING_DAY))
        if expected > 1:
            missing += max(0, expected - 2)
    return int(missing)


def estimate_missing_bars_count(timestamps: pd.Series) -> tuple[int, float | None]:
    """Estimate missing bars count and inferred bar spacing from timestamps."""
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return 0, None

    diffs = ts.diff().dropna().dt.total_seconds()
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.empty:
        return 0, None

    inferred_bar_seconds = float(np.median(positive_diffs.to_numpy()))
    if inferred_bar_seconds <= 0.0:
        return 0, None

    if inferred_bar_seconds >= _DAILY_BAR_SECONDS_FLOOR:
        return _count_missing_daily_bars(ts), inferred_bar_seconds

    regular_gap_limit = inferred_bar_seconds * 1.5
    session_break_limit = max(inferred_bar_seconds * 24.0, 6.0 * 3600.0)
    missing_from_gaps = np.where(
        (positive_diffs > regular_gap_limit) & (positive_diffs <= session_break_limit),
        np.floor(positive_diffs / inferred_bar_seconds) - 1.0,
        0.0,
    )
    return int(np.maximum(missing_from_gaps, 0.0).sum()), inferred_bar_seconds


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
            symbol_missing, inferred_bar_seconds = estimate_missing_bars_count(timestamps)

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


def compute_frame_content_hash(
    frame: pd.DataFrame | None,
    columns: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Compute deterministic content hash for a pandas frame."""
    requested_columns = [str(column) for column in (columns or [])]
    if frame is None:
        return _canonical_hash(
            {
                "row_count": 0,
                "columns": requested_columns,
                "dtypes": [],
                "content_hash": "none",
            }
        )

    if columns is None:
        selected_columns = [str(column) for column in frame.columns.tolist()]
    else:
        selected_columns = [column for column in requested_columns if column in frame.columns]

    working = frame.loc[:, selected_columns].copy() if selected_columns else pd.DataFrame()
    working = working.reset_index(drop=True)

    if working.empty and not len(working.columns):
        return _canonical_hash(
            {
                "row_count": int(len(frame)),
                "columns": selected_columns,
                "dtypes": [],
                "content_hash": "empty",
            }
        )

    for column in working.columns:
        series = working[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            normalized = pd.to_datetime(series, utc=True, errors="coerce")
            working[column] = normalized.astype("int64")
        elif pd.api.types.is_numeric_dtype(series):
            working[column] = pd.to_numeric(series, errors="coerce")
        elif pd.api.types.is_bool_dtype(series):
            working[column] = series.astype(bool)
        else:
            working[column] = series.astype(str)

    row_hashes = pd.util.hash_pandas_object(working, index=False).to_numpy(dtype="uint64")
    frame_basis = {
        "row_count": int(len(working)),
        "columns": [str(column) for column in working.columns.tolist()],
        "dtypes": [str(dtype) for dtype in working.dtypes.tolist()],
    }
    return _canonical_hash(
        {
            **frame_basis,
            "row_hashes": hashlib.sha256(row_hashes.tobytes()).hexdigest(),
        }
    )


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
    ohlcv_hash_columns = [
        column
        for column in ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        if column in ohlcv_data.columns
    ]
    raw_ohlcv_hash = compute_frame_content_hash(ohlcv_data, columns=ohlcv_hash_columns)

    snapshot_basis = {
        "symbols": symbols,
        "observed_start": observed_start,
        "observed_end": observed_end,
        "row_count": int(len(df)),
        "feature_schema_version": feature_schema_version,
        "raw_ohlcv_hash": raw_ohlcv_hash,
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
        "raw_ohlcv_hash": raw_ohlcv_hash,
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


def persist_dataset_snapshot_bundle(
    output_dir: Path,
    snapshot_manifest: dict[str, Any],
    raw_ohlcv_data: pd.DataFrame | None,
    development_frame: pd.DataFrame,
    feature_names: list[str],
    data_quality_report: dict[str, Any] | None = None,
    development_sample_weights: np.ndarray | None = None,
    holdout_frame: pd.DataFrame | None = None,
    holdout_sample_weights: np.ndarray | None = None,
    cv_splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    label_diagnostics: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Persist immutable training dataset artifacts for deterministic replay."""
    snapshot_id = str(snapshot_manifest.get("snapshot_id") or "").strip()
    if not snapshot_id:
        raise ValueError("Dataset snapshot bundle requires snapshot_manifest.snapshot_id.")
    if development_frame is None or development_frame.empty:
        raise ValueError("Dataset snapshot bundle requires a non-empty development frame.")

    bundle_dir = Path(output_dir) / "snapshots" / snapshot_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, dict[str, Any]] = {}

    def _persist_frame_artifact(name: str, frame: pd.DataFrame | None) -> None:
        if frame is None:
            return
        artifact_path = bundle_dir / f"{name}.pkl.gz"
        frame.to_pickle(artifact_path, compression="gzip")
        artifacts[name] = {
            "path": artifact_path.name,
            "sha256": _file_sha256(artifact_path),
            "row_count": int(len(frame)),
            "columns": [str(column) for column in frame.columns.tolist()],
            "frame_hash": compute_frame_content_hash(frame),
        }

    def _persist_array_artifact(name: str, values: np.ndarray | None) -> None:
        if values is None:
            return
        arr = np.asarray(values)
        artifact_path = bundle_dir / f"{name}.npy"
        np.save(artifact_path, arr)
        artifacts[name] = {
            "path": artifact_path.name,
            "sha256": _file_sha256(artifact_path),
            "length": int(arr.shape[0]) if arr.ndim > 0 else int(arr.size),
        }

    _persist_frame_artifact("raw_ohlcv", raw_ohlcv_data)
    _persist_frame_artifact("development_frame", development_frame)
    _persist_frame_artifact("holdout_frame", holdout_frame)
    _persist_array_artifact("development_sample_weights", development_sample_weights)
    _persist_array_artifact("holdout_sample_weights", holdout_sample_weights)

    if cv_splits:
        cv_payload = [
            {
                "train_indices": np.asarray(train_idx, dtype=int).tolist(),
                "test_indices": np.asarray(test_idx, dtype=int).tolist(),
            }
            for train_idx, test_idx in cv_splits
        ]
        cv_path = bundle_dir / "cv_splits.json"
        with cv_path.open("w", encoding="utf-8") as handle:
            json.dump(cv_payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        artifacts["cv_splits"] = {
            "path": cv_path.name,
            "sha256": _file_sha256(cv_path),
            "split_count": int(len(cv_payload)),
        }

    bundle_manifest = {
        "schema_version": DATASET_SNAPSHOT_BUNDLE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": snapshot_id,
        "snapshot_manifest": dict(snapshot_manifest),
        "feature_names": sorted({str(name) for name in feature_names}),
        "data_quality_report": dict(data_quality_report or {}),
        "label_diagnostics": dict(label_diagnostics or {}),
        "artifacts": artifacts,
        "development_rows": int(len(development_frame)),
        "holdout_rows": int(len(holdout_frame)) if holdout_frame is not None else 0,
    }
    bundle_manifest["bundle_hash"] = _canonical_hash(
        {
            "schema_version": bundle_manifest["schema_version"],
            "snapshot_id": bundle_manifest["snapshot_id"],
            "snapshot_manifest": bundle_manifest["snapshot_manifest"],
            "feature_names": bundle_manifest["feature_names"],
            "data_quality_report": bundle_manifest["data_quality_report"],
            "label_diagnostics": bundle_manifest["label_diagnostics"],
            "artifacts": bundle_manifest["artifacts"],
            "development_rows": bundle_manifest["development_rows"],
            "holdout_rows": bundle_manifest["holdout_rows"],
        }
    )

    bundle_manifest_path = bundle_dir / "dataset_bundle.manifest.json"
    with bundle_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(bundle_manifest, handle, indent=2, ensure_ascii=True, sort_keys=True, default=str)

    return bundle_manifest_path, bundle_manifest


def load_dataset_snapshot_bundle(
    bundle_manifest_path: Path,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Load immutable training dataset artifacts from bundle manifest."""
    manifest_path = Path(bundle_manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if not isinstance(manifest, dict):
        raise ValueError("Dataset snapshot bundle manifest must be a JSON object.")

    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("Dataset snapshot bundle manifest missing artifacts block.")

    bundle_dir = manifest_path.parent

    def _resolve_path(artifact_name: str) -> Path | None:
        payload = artifacts.get(artifact_name)
        if not isinstance(payload, dict):
            return None
        raw_path = payload.get("path")
        if not raw_path:
            return None
        candidate = Path(str(raw_path))
        if candidate.is_absolute():
            return candidate
        return bundle_dir / candidate

    def _validate_hash(artifact_name: str, artifact_path: Path) -> None:
        if not verify_hashes:
            return
        payload = artifacts.get(artifact_name, {})
        expected_hash = str(payload.get("sha256") or "").strip()
        if expected_hash and _file_sha256(artifact_path) != expected_hash:
            raise ValueError(f"Snapshot artifact hash mismatch for {artifact_name}: {artifact_path}")

    def _load_frame(artifact_name: str) -> pd.DataFrame | None:
        artifact_path = _resolve_path(artifact_name)
        if artifact_path is None or not artifact_path.exists():
            return None
        _validate_hash(artifact_name, artifact_path)
        return pd.read_pickle(artifact_path, compression="gzip")

    def _load_array(artifact_name: str) -> np.ndarray | None:
        artifact_path = _resolve_path(artifact_name)
        if artifact_path is None or not artifact_path.exists():
            return None
        _validate_hash(artifact_name, artifact_path)
        return np.load(artifact_path, allow_pickle=False)

    cv_splits_path = _resolve_path("cv_splits")
    cv_splits: list[tuple[np.ndarray, np.ndarray]] = []
    if cv_splits_path is not None and cv_splits_path.exists():
        _validate_hash("cv_splits", cv_splits_path)
        with cv_splits_path.open("r", encoding="utf-8") as handle:
            raw_cv = json.load(handle)
        if isinstance(raw_cv, list):
            for payload in raw_cv:
                if not isinstance(payload, dict):
                    continue
                train_idx = np.asarray(payload.get("train_indices", []), dtype=int)
                test_idx = np.asarray(payload.get("test_indices", []), dtype=int)
                cv_splits.append((train_idx, test_idx))

    return {
        "bundle_manifest": manifest,
        "bundle_manifest_path": manifest_path,
        "snapshot_manifest": (
            manifest.get("snapshot_manifest", {})
            if isinstance(manifest.get("snapshot_manifest"), dict)
            else {}
        ),
        "feature_names": [
            str(name)
            for name in manifest.get("feature_names", [])
            if isinstance(name, str) and name.strip()
        ],
        "data_quality_report": (
            manifest.get("data_quality_report", {})
            if isinstance(manifest.get("data_quality_report"), dict)
            else {}
        ),
        "label_diagnostics": (
            manifest.get("label_diagnostics", {})
            if isinstance(manifest.get("label_diagnostics"), dict)
            else {}
        ),
        "raw_ohlcv_data": _load_frame("raw_ohlcv"),
        "development_frame": _load_frame("development_frame"),
        "holdout_frame": _load_frame("holdout_frame"),
        "development_sample_weights": _load_array("development_sample_weights"),
        "holdout_sample_weights": _load_array("holdout_sample_weights"),
        "cv_splits": cv_splits,
    }


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
                "raw_ohlcv_hash": snapshot_manifest.get("raw_ohlcv_hash"),
                "data_quality_report_hash": snapshot_manifest.get("data_quality_report_hash"),
                "snapshot_manifest_path": snapshot_manifest.get("manifest_path"),
                "snapshot_quality_report_path": snapshot_manifest.get("quality_report_path"),
                "dataset_snapshot_bundle_path": snapshot_manifest.get(
                    "dataset_bundle_manifest_path"
                ),
                "dataset_snapshot_bundle_hash": snapshot_manifest.get("dataset_bundle_hash"),
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
