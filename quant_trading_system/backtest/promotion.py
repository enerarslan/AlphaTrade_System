"""
Promotion package contract helpers for artifact-driven backtests and replay.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quant_trading_system.data.timeframe import DEFAULT_TIMEFRAME, normalize_timeframe


def _coalesce(*values: Any) -> Any:
    """Return the first non-empty value."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _resolve_optional_path(base_dir: Path, raw_path: Any) -> Path | None:
    """Resolve package-relative or absolute artifact paths."""
    candidate = _coalesce(raw_path)
    if candidate is None:
        return None

    path = Path(str(candidate))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_json(path: Path | None) -> dict[str, Any]:
    """Load JSON payload when available."""
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _normalize_string_list(values: Any) -> list[str]:
    """Normalize arbitrary input into a non-empty list of strings."""
    if isinstance(values, (list, tuple, set)):
        return [str(value).strip() for value in values if str(value).strip()]
    if values is None:
        return []
    value = str(values).strip()
    return [value] if value else []


@dataclass(frozen=True, slots=True)
class PromotionPackageContract:
    """Resolved promotion package contract used by backtest and replay."""

    package_path: Path
    schema_version: str
    raw_payload: dict[str, Any]
    artifacts_payload: dict[str, Any]
    model_name: str
    model_type: str
    model_path: Path
    artifacts_path: Path | None
    meta_model_path: Path | None
    feature_names: tuple[str, ...]
    feature_groups: tuple[str, ...]
    symbols: tuple[str, ...]
    timeframe: str
    timeframes: tuple[str, ...]
    feature_schema_version: str | None
    long_threshold: float
    short_threshold: float
    horizon_bars: int
    meta_label_enabled: bool
    meta_label_threshold: float | None
    model_source: str
    cost_model: dict[str, float]


def load_promotion_package(package_path: str | Path) -> PromotionPackageContract:
    """Load and validate a promotion package contract."""
    path = Path(package_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Promotion package not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw_payload = json.load(handle)

    if not isinstance(raw_payload, dict):
        raise ValueError(f"Promotion package must contain a JSON object: {path}")

    package_dir = path.parent
    training_config = raw_payload.get("training_config", {})
    if not isinstance(training_config, dict):
        training_config = {}

    model_path = _resolve_optional_path(package_dir, raw_payload.get("model_path"))
    if model_path is None:
        raise ValueError(f"Promotion package missing required model_path: {path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Promotion package model not found: {model_path}")

    artifacts_path = _resolve_optional_path(package_dir, raw_payload.get("artifacts_path"))
    if artifacts_path is None:
        artifacts_path = model_path.with_name(f"{model_path.stem}_artifacts.json")
    if artifacts_path is not None and not artifacts_path.exists():
        artifacts_path = None

    artifacts_payload = _load_json(artifacts_path)
    artifacts_training_metrics = artifacts_payload.get("training_metrics", {})
    if not isinstance(artifacts_training_metrics, dict):
        artifacts_training_metrics = {}

    feature_contract = raw_payload.get("feature_contract", {})
    if not isinstance(feature_contract, dict):
        feature_contract = {}

    signal_policy = raw_payload.get("signal_policy", {})
    if not isinstance(signal_policy, dict):
        signal_policy = {}

    raw_feature_names = _coalesce(
        feature_contract.get("selected_features"),
        feature_contract.get("feature_names"),
        raw_payload.get("feature_names"),
        artifacts_payload.get("feature_names"),
    )
    feature_names = tuple(_normalize_string_list(raw_feature_names))

    raw_feature_groups = _coalesce(
        feature_contract.get("feature_groups"),
        training_config.get("feature_groups"),
    )
    feature_groups = tuple(_normalize_string_list(raw_feature_groups))

    symbols = tuple(_normalize_string_list(training_config.get("symbols")))
    timeframe = normalize_timeframe(training_config.get("timeframe"), default=DEFAULT_TIMEFRAME)
    raw_timeframes = _normalize_string_list(training_config.get("timeframes"))
    timeframes = tuple(
        normalize_timeframe(value, default=timeframe) for value in (raw_timeframes or [timeframe])
    )

    long_threshold = float(
        _coalesce(
            signal_policy.get("long_threshold"),
            artifacts_training_metrics.get("holdout_long_threshold"),
            0.55,
        )
    )
    short_threshold = float(
        _coalesce(
            signal_policy.get("short_threshold"),
            artifacts_training_metrics.get("holdout_short_threshold"),
            0.45,
        )
    )
    if long_threshold <= short_threshold:
        long_threshold = 0.55
        short_threshold = 0.45

    horizon_bars = max(
        1,
        int(
            float(
                _coalesce(
                    signal_policy.get("horizon_bars"),
                    training_config.get("primary_label_horizon"),
                    1,
                )
            )
        ),
    )

    explicit_meta_path = _resolve_optional_path(package_dir, raw_payload.get("meta_model_path"))
    default_meta_path = model_path.with_name(f"{model_path.stem}_meta.pkl")
    meta_model_path = explicit_meta_path or default_meta_path
    if meta_model_path is not None and not meta_model_path.exists():
        meta_model_path = None

    meta_label_enabled = bool(
        _coalesce(signal_policy.get("meta_label_enabled"), meta_model_path is not None)
    ) and meta_model_path is not None
    meta_label_threshold = _coalesce(
        signal_policy.get("meta_label_threshold"),
        artifacts_training_metrics.get("meta_label_min_confidence"),
        training_config.get("meta_label_min_confidence"),
    )
    meta_label_threshold = (
        None if meta_label_threshold is None else float(meta_label_threshold)
    )

    raw_cost_model = raw_payload.get("execution_cost_model", {})
    if not isinstance(raw_cost_model, dict):
        raw_cost_model = {}
    cost_model = {
        "spread_bps": float(
            _coalesce(raw_cost_model.get("spread_bps"), training_config.get("label_spread_bps"), 0.0)
        ),
        "slippage_bps": float(
            _coalesce(
                raw_cost_model.get("slippage_bps"),
                training_config.get("label_slippage_bps"),
                0.0,
            )
        ),
        "impact_bps": float(
            _coalesce(raw_cost_model.get("impact_bps"), training_config.get("label_impact_bps"), 0.0)
        ),
    }

    model_name = str(_coalesce(raw_payload.get("model_name"), model_path.stem)).strip()
    model_type = str(_coalesce(raw_payload.get("model_type"), training_config.get("model_type"), "")).strip()
    model_source = str(
        _coalesce(signal_policy.get("model_source"), f"promotion_package:{model_name}")
    ).strip()
    feature_schema_version = _coalesce(
        feature_contract.get("feature_schema_version"),
        raw_payload.get("feature_schema_version"),
        raw_payload.get("snapshot_manifest", {}).get("feature_schema_version")
        if isinstance(raw_payload.get("snapshot_manifest"), dict)
        else None,
        artifacts_payload.get("snapshot_manifest", {}).get("feature_schema_version")
        if isinstance(artifacts_payload.get("snapshot_manifest"), dict)
        else None,
    )
    feature_schema_version = (
        None if feature_schema_version is None else str(feature_schema_version).strip() or None
    )

    return PromotionPackageContract(
        package_path=path,
        schema_version=str(_coalesce(raw_payload.get("schema_version"), "1.0.0")),
        raw_payload=raw_payload,
        artifacts_payload=artifacts_payload,
        model_name=model_name,
        model_type=model_type,
        model_path=model_path,
        artifacts_path=artifacts_path,
        meta_model_path=meta_model_path,
        feature_names=feature_names,
        feature_groups=feature_groups,
        symbols=symbols,
        timeframe=timeframe,
        timeframes=timeframes,
        feature_schema_version=feature_schema_version,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        horizon_bars=horizon_bars,
        meta_label_enabled=meta_label_enabled,
        meta_label_threshold=meta_label_threshold,
        model_source=model_source,
        cost_model=cost_model,
    )
