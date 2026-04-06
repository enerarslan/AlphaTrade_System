"""
Promotion package contract helpers for artifact-driven backtests and replay.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_trading_system.data.data_access import filter_ohlcv_frame_to_market_session
from quant_trading_system.data.timeframe import DEFAULT_TIMEFRAME, normalize_timeframe
from quant_trading_system.features.feature_pipeline import (
    FeatureConfig,
    FeatureGroup,
    FeaturePipeline,
    NormalizationMethod,
)
from quant_trading_system.features.multi_timeframe import MultiTimeframeFeatureEngine
from quant_trading_system.features.optimized_pipeline import (
    CUDF_AVAILABLE,
    ComputeMode,
    OptimizedFeaturePipeline,
    OptimizedPipelineConfig,
)
from quant_trading_system.models.expected_edge_policy import resolve_regime_policy_frame
from quant_trading_system.models.feature_schema import prepare_model_inference_input
from quant_trading_system.models.symbol_quality import (
    SymbolQualityThresholds,
    assess_symbol_quality,
)
from quant_trading_system.models.target_engineering import infer_regime_series

logger = logging.getLogger(__name__)


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


def _align_vector(values: Any, length: int, fill_value: float = 0.0) -> np.ndarray:
    """Align arbitrary scalar/1D values to the requested output length."""
    if length <= 0:
        return np.array([], dtype=float)
    if values is None:
        dtype = object if isinstance(fill_value, str) else float
        return np.full(length, fill_value, dtype=dtype)
    arr = np.asarray(values).reshape(-1)
    if arr.size == length:
        return arr
    if arr.size == 0:
        dtype = object if isinstance(fill_value, str) else float
        return np.full(length, fill_value, dtype=dtype)
    if arr.size < length:
        aligned = np.full(length, fill_value, dtype=arr.dtype if arr.dtype != object else object)
        aligned[-arr.size :] = arr
        return aligned
    return arr[-length:]


def _rank_scores_to_unit_interval(values: np.ndarray | list[float]) -> np.ndarray:
    """Map arbitrary ranker scores into a unit interval using stable average ranks."""
    raw_values = np.asarray(values, dtype=float).reshape(-1)
    if raw_values.size == 0:
        return raw_values.astype(float)

    normalized = np.full(raw_values.shape, 0.5, dtype=float)
    finite_mask = np.isfinite(raw_values)
    finite_values = raw_values[finite_mask]
    if finite_values.size <= 1:
        return normalized
    if float(np.max(finite_values) - np.min(finite_values)) <= 1e-12:
        normalized[finite_mask] = 0.5
        return normalized

    order = np.argsort(finite_values, kind="mergesort")
    sorted_values = finite_values[order]
    ranks = np.empty(sorted_values.size, dtype=float)
    start = 0
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and np.isclose(
            sorted_values[end],
            sorted_values[start],
            rtol=1e-9,
            atol=1e-12,
        ):
            end += 1
        ranks[order[start:end]] = 0.5 * float(start + end - 1)
        start = end

    normalized[finite_mask] = np.clip((ranks + 0.5) / float(sorted_values.size), 0.0, 1.0)
    return normalized


def _normalize_ranker_scores(
    raw_scores: np.ndarray | list[float],
    *,
    timestamps: np.ndarray | list[Any] | None = None,
    normalization_mode: str = "query_percentile",
) -> np.ndarray:
    """Normalize ranker scores globally or within timestamp queries."""
    scores = np.asarray(raw_scores, dtype=float).reshape(-1)
    if scores.size == 0:
        return scores.astype(float)

    mode = str(normalization_mode or "query_percentile").strip().lower()
    if mode != "query_percentile" or timestamps is None:
        return _rank_scores_to_unit_interval(scores)

    timestamp_values = pd.to_datetime(np.asarray(timestamps), utc=True, errors="coerce")
    if timestamp_values.size != scores.size or np.asarray(pd.isna(timestamp_values)).any():
        return _rank_scores_to_unit_interval(scores)

    normalized = np.empty(scores.size, dtype=float)
    ts_arr = np.asarray(timestamp_values, dtype="datetime64[ns]")
    change_points = np.flatnonzero(ts_arr[1:] != ts_arr[:-1]) + 1
    boundaries = np.concatenate(([0], change_points, [scores.size]))
    start = 0
    for end in boundaries[1:]:
        normalized[start:end] = _rank_scores_to_unit_interval(scores[start:end])
        start = int(end)
    return normalized


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
    expected_edge_model_path: Path | None
    feature_names: tuple[str, ...]
    feature_groups: tuple[str, ...]
    symbols: tuple[str, ...]
    timeframe: str
    timeframes: tuple[str, ...]
    feature_schema_version: str | None
    enable_cross_sectional: bool
    enable_reference_features: bool
    reference_feature_sources: tuple[str, ...]
    enable_tick_microstructure_features: bool
    long_threshold: float
    short_threshold: float
    horizon_bars: int
    max_holding_bars: int
    take_profit_pct: float
    stop_loss_pct: float
    meta_label_enabled: bool
    meta_label_threshold: float | None
    model_source: str
    long_side_policy: dict[str, Any]
    short_side_policy: dict[str, Any]
    cost_model: dict[str, float]
    use_portfolio_target_sizing: bool
    max_position_pct: float
    max_total_positions: int
    confidence_position_sizing: bool
    min_confidence_position_scale: float
    expected_edge_policy_enabled: bool
    expected_edge_policy: dict[str, Any]
    score_semantics: str
    candidate_selection_policy: dict[str, Any]
    holding_horizon_bars: int
    meta_model_enabled: bool
    meta_threshold: float | None
    short_policy: dict[str, Any]
    borrow_check_required: bool
    adv_lookback_days: int
    exit_policy: dict[str, Any]
    ranker_score_normalization: str | None
    ranker_query_key: str | None
    ranker_requires_cross_sectional_panel: bool
    enable_universe_quality_gate: bool
    universe_quality_policy: dict[str, Any]


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

    expected_edge_policy = raw_payload.get("expected_edge_policy", {})
    if not isinstance(expected_edge_policy, dict):
        expected_edge_policy = {}
    ranker_scoring = raw_payload.get("ranker_scoring", {})
    if not isinstance(ranker_scoring, dict):
        ranker_scoring = {}

    position_sizing_policy = raw_payload.get("position_sizing_policy", {})
    if not isinstance(position_sizing_policy, dict):
        position_sizing_policy = {}

    universe_quality_policy = raw_payload.get("universe_quality_policy", {})
    if not isinstance(universe_quality_policy, dict):
        universe_quality_policy = {}
    candidate_selection_policy = raw_payload.get("candidate_selection_policy", {})
    if not isinstance(candidate_selection_policy, dict):
        candidate_selection_policy = {}
    short_policy = raw_payload.get("short_policy", {})
    if not isinstance(short_policy, dict):
        short_policy = {}
    exit_policy = raw_payload.get("exit_policy", {})
    if not isinstance(exit_policy, dict):
        exit_policy = {}

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
    enable_cross_sectional = bool(
        _coalesce(
            feature_contract.get("enable_cross_sectional"),
            training_config.get("enable_cross_sectional"),
            False,
        )
    )
    enable_reference_features = bool(
        _coalesce(
            feature_contract.get("enable_reference_features"),
            training_config.get("enable_reference_features"),
            False,
        )
    )
    from quant_trading_system.features.reference import normalize_reference_feature_sources

    reference_feature_sources = tuple(
        normalize_reference_feature_sources(
            _coalesce(
                feature_contract.get("reference_feature_sources"),
                training_config.get("reference_feature_sources"),
                ["all"] if enable_reference_features else [],
            )
        )
    )
    enable_tick_microstructure_features = bool(
        _coalesce(
            feature_contract.get("enable_tick_microstructure_features"),
            training_config.get("enable_tick_microstructure_features"),
            False,
        )
    )

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
    max_holding_bars = max(
        1,
        int(
            float(
                _coalesce(
                    signal_policy.get("max_holding_bars"),
                    training_config.get("label_max_holding_period"),
                    horizon_bars,
                )
            )
        ),
    )
    take_profit_pct = float(
        _coalesce(
            signal_policy.get("take_profit_pct"),
            training_config.get("label_profit_taking_threshold"),
            0.0,
        )
    )
    stop_loss_pct = float(
        abs(
            float(
                _coalesce(
                    signal_policy.get("stop_loss_pct"),
                    training_config.get("label_stop_loss_threshold"),
                    0.0,
                )
            )
        )
    )

    explicit_meta_path = _resolve_optional_path(package_dir, raw_payload.get("meta_model_path"))
    default_meta_path = model_path.with_name(f"{model_path.stem}_meta.pkl")
    meta_model_path = explicit_meta_path or default_meta_path
    if meta_model_path is not None and not meta_model_path.exists():
        meta_model_path = None

    explicit_expected_edge_path = _resolve_optional_path(
        package_dir,
        raw_payload.get("expected_edge_model_path"),
    )
    default_expected_edge_path = model_path.with_name(f"{model_path.stem}_expected_edge.pkl")
    expected_edge_model_path = explicit_expected_edge_path or default_expected_edge_path
    if expected_edge_model_path is not None and not expected_edge_model_path.exists():
        expected_edge_model_path = None

    meta_label_enabled = (
        bool(_coalesce(signal_policy.get("meta_label_enabled"), meta_model_path is not None))
        and meta_model_path is not None
    )
    meta_label_threshold = _coalesce(
        signal_policy.get("meta_label_threshold"),
        artifacts_training_metrics.get("meta_label_min_confidence"),
        training_config.get("meta_label_min_confidence"),
    )
    meta_label_threshold = None if meta_label_threshold is None else float(meta_label_threshold)

    raw_cost_model = raw_payload.get("execution_cost_model", {})
    if not isinstance(raw_cost_model, dict):
        raw_cost_model = {}
    cost_model = {
        "spread_bps": float(
            _coalesce(
                raw_cost_model.get("spread_bps"), training_config.get("label_spread_bps"), 0.0
            )
        ),
        "slippage_bps": float(
            _coalesce(
                raw_cost_model.get("slippage_bps"),
                training_config.get("label_slippage_bps"),
                0.0,
            )
        ),
        "impact_bps": float(
            _coalesce(
                raw_cost_model.get("impact_bps"), training_config.get("label_impact_bps"), 0.0
            )
        ),
    }

    max_position_pct = float(_coalesce(position_sizing_policy.get("max_position_pct"), 0.10))
    max_position_pct = min(1.0, max(0.0, max_position_pct))
    max_total_positions = max(
        1,
        int(float(_coalesce(position_sizing_policy.get("max_total_positions"), 20))),
    )
    use_portfolio_target_sizing = bool(
        _coalesce(position_sizing_policy.get("use_portfolio_target_sizing"), True)
    )
    confidence_position_sizing = bool(
        _coalesce(position_sizing_policy.get("confidence_position_sizing"), True)
    )
    min_confidence_position_scale = float(
        _coalesce(position_sizing_policy.get("min_confidence_position_scale"), 0.0)
    )
    min_confidence_position_scale = min(1.0, max(0.0, min_confidence_position_scale))
    expected_edge_policy_enabled = (
        bool(_coalesce(expected_edge_policy.get("enabled"), expected_edge_model_path is not None))
        and expected_edge_model_path is not None
    )
    holding_horizon_bars = max(
        1,
        int(
            float(
                _coalesce(
                    raw_payload.get("holding_horizon_bars"),
                    exit_policy.get("holding_horizon_bars"),
                    signal_policy.get("max_holding_bars"),
                    max_holding_bars,
                )
            )
        ),
    )
    meta_model_enabled = (
        bool(_coalesce(raw_payload.get("meta_model_enabled"), meta_label_enabled))
        and meta_model_path is not None
    )
    meta_threshold = _coalesce(
        raw_payload.get("meta_threshold"),
        meta_label_threshold,
    )
    meta_threshold = None if meta_threshold is None else float(meta_threshold)
    borrow_check_required = bool(
        _coalesce(
            raw_payload.get("borrow_check_required"),
            short_policy.get("borrow_check_required"),
            False,
        )
    )
    adv_lookback_days = max(
        1,
        int(
            float(
                _coalesce(
                    raw_payload.get("adv_lookback_days"),
                    raw_cost_model.get("adv_lookback_days"),
                    20,
                )
            )
        ),
    )
    if not candidate_selection_policy:
        candidate_selection_policy = {
            "mode": "cross_sectional_long_short",
            "selection": "top_bottom_by_score",
            "universe": "promoted_symbols",
            "max_total_positions": max_total_positions,
        }
    if not short_policy:
        short_policy = {
            "enabled": True,
            "mode": "long_short",
            "borrow_check_required": borrow_check_required,
        }
    if not exit_policy:
        exit_policy = {
            "style": "horizon_stop_take_profit",
            "holding_horizon_bars": holding_horizon_bars,
            "prediction_horizon_bars": horizon_bars,
            "take_profit_pct": take_profit_pct,
            "stop_loss_pct": stop_loss_pct,
        }
    model_name = str(_coalesce(raw_payload.get("model_name"), model_path.stem)).strip()
    model_type = str(
        _coalesce(raw_payload.get("model_type"), training_config.get("model_type"), "")
    ).strip()
    score_semantics = str(
        _coalesce(
            raw_payload.get("score_semantics"),
            ranker_scoring.get("score_semantics"),
            "cross_sectional_rank_percentile"
            if "rank" in model_type.lower()
            else "probability_up",
        )
    ).strip() or (
        "cross_sectional_rank_percentile" if "rank" in model_type.lower() else "probability_up"
    )
    ranker_score_normalization = _coalesce(
        ranker_scoring.get("normalization"),
        "query_percentile" if "rank" in model_type.lower() else None,
    )
    ranker_query_key = _coalesce(
        ranker_scoring.get("query_key"),
        "timestamp" if "rank" in model_type.lower() else None,
    )
    ranker_requires_cross_sectional_panel = bool(
        _coalesce(
            ranker_scoring.get("requires_cross_sectional_panel"),
            True if "rank" in model_type.lower() else False,
        )
    )
    model_source = str(
        _coalesce(signal_policy.get("model_source"), f"promotion_package:{model_name}")
    ).strip()
    long_side_policy = signal_policy.get("long_side_policy", {})
    if not isinstance(long_side_policy, dict):
        long_side_policy = {}
    short_side_policy = signal_policy.get("short_side_policy", {})
    if not isinstance(short_side_policy, dict):
        short_side_policy = {}
    enable_universe_quality_gate = bool(
        _coalesce(
            universe_quality_policy.get("enabled"),
            training_config.get("enable_symbol_quality_filter"),
            False,
        )
    )
    feature_schema_version = _coalesce(
        feature_contract.get("feature_schema_version"),
        raw_payload.get("feature_schema_version"),
        (
            raw_payload.get("snapshot_manifest", {}).get("feature_schema_version")
            if isinstance(raw_payload.get("snapshot_manifest"), dict)
            else None
        ),
        (
            artifacts_payload.get("snapshot_manifest", {}).get("feature_schema_version")
            if isinstance(artifacts_payload.get("snapshot_manifest"), dict)
            else None
        ),
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
        expected_edge_model_path=expected_edge_model_path,
        feature_names=feature_names,
        feature_groups=feature_groups,
        symbols=symbols,
        timeframe=timeframe,
        timeframes=timeframes,
        feature_schema_version=feature_schema_version,
        enable_cross_sectional=enable_cross_sectional,
        enable_reference_features=enable_reference_features,
        reference_feature_sources=reference_feature_sources,
        enable_tick_microstructure_features=enable_tick_microstructure_features,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        horizon_bars=horizon_bars,
        max_holding_bars=max_holding_bars,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        meta_label_enabled=meta_label_enabled,
        meta_label_threshold=meta_label_threshold,
        model_source=model_source,
        long_side_policy=long_side_policy,
        short_side_policy=short_side_policy,
        cost_model=cost_model,
        use_portfolio_target_sizing=use_portfolio_target_sizing,
        max_position_pct=max_position_pct,
        max_total_positions=max_total_positions,
        confidence_position_sizing=confidence_position_sizing,
        min_confidence_position_scale=min_confidence_position_scale,
        expected_edge_policy_enabled=expected_edge_policy_enabled,
        expected_edge_policy=expected_edge_policy,
        score_semantics=score_semantics,
        candidate_selection_policy=candidate_selection_policy,
        holding_horizon_bars=holding_horizon_bars,
        meta_model_enabled=meta_model_enabled,
        meta_threshold=meta_threshold,
        short_policy=short_policy,
        borrow_check_required=borrow_check_required,
        adv_lookback_days=adv_lookback_days,
        exit_policy=exit_policy,
        ranker_score_normalization=(
            None
            if ranker_score_normalization is None
            else str(ranker_score_normalization).strip() or None
        ),
        ranker_query_key=(
            None if ranker_query_key is None else str(ranker_query_key).strip() or None
        ),
        ranker_requires_cross_sectional_panel=ranker_requires_cross_sectional_panel,
        enable_universe_quality_gate=enable_universe_quality_gate,
        universe_quality_policy=universe_quality_policy,
    )


def _estimate_trailing_liquidity_frame(
    frame: pd.DataFrame | None,
    *,
    timeframe: str | None,
    lookback_days: int,
) -> pd.DataFrame:
    """Estimate trailing ADV/ADDV and per-bar spread telemetry aligned by timestamp."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["timestamp", "avg_daily_volume", "avg_daily_dollar_volume", "spread_bps"])

    normalized = _to_pandas_frame(frame)
    filtered, _ = filter_ohlcv_frame_to_market_session(
        normalized,
        timeframe=timeframe,
        include_premarket=False,
        include_postmarket=False,
    )
    if filtered.empty:
        filtered = normalized.copy()
    working = filtered.copy()
    if "timestamp" not in working.columns:
        return pd.DataFrame(columns=["timestamp", "avg_daily_volume", "avg_daily_dollar_volume", "spread_bps"])

    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    working["close"] = pd.to_numeric(working.get("close"), errors="coerce")
    working["high"] = pd.to_numeric(working.get("high"), errors="coerce")
    working["low"] = pd.to_numeric(working.get("low"), errors="coerce")
    working["volume"] = pd.to_numeric(working.get("volume"), errors="coerce")
    working = working.dropna(subset=["timestamp", "close", "volume"])
    if working.empty:
        return pd.DataFrame(columns=["timestamp", "avg_daily_volume", "avg_daily_dollar_volume", "spread_bps"])

    working = working.sort_values("timestamp").reset_index(drop=True)
    eastern_timestamps = working["timestamp"].dt.tz_convert("America/New_York")
    working["trade_date"] = eastern_timestamps.dt.normalize()
    working["dollar_volume"] = working["close"] * working["volume"]

    daily = (
        working.groupby("trade_date", sort=True)
        .agg(
            day_volume=("volume", "sum"),
            day_dollar_volume=("dollar_volume", "sum"),
        )
        .sort_index()
    )
    if daily.empty:
        return pd.DataFrame(columns=["timestamp", "avg_daily_volume", "avg_daily_dollar_volume", "spread_bps"])

    trailing = daily.shift(1).rolling(max(1, int(lookback_days)), min_periods=1).mean()
    fallback = daily.rolling(max(1, int(lookback_days)), min_periods=1).mean()
    trailing = trailing.where(trailing.notna(), fallback)
    trailing = trailing.rename(
        columns={
            "day_volume": "avg_daily_volume",
            "day_dollar_volume": "avg_daily_dollar_volume",
        }
    )

    metrics = working.loc[:, ["timestamp", "trade_date", "high", "low", "close"]].copy()
    metrics = metrics.merge(trailing, left_on="trade_date", right_index=True, how="left")
    spread_bps = np.full(len(metrics), np.nan, dtype=float)
    valid_spread = pd.to_numeric(metrics["close"], errors="coerce").to_numpy(dtype=float)
    valid_high = pd.to_numeric(metrics["high"], errors="coerce").to_numpy(dtype=float)
    valid_low = pd.to_numeric(metrics["low"], errors="coerce").to_numpy(dtype=float)
    positive_close_mask = np.isfinite(valid_spread) & (valid_spread > 0.0)
    spread_bps[positive_close_mask] = (
        (valid_high[positive_close_mask] - valid_low[positive_close_mask])
        / valid_spread[positive_close_mask]
    ) * 10000.0
    metrics["spread_bps"] = spread_bps
    return metrics.loc[:, ["timestamp", "avg_daily_volume", "avg_daily_dollar_volume", "spread_bps"]]


def resolve_feature_groups(contract: PromotionPackageContract) -> list[FeatureGroup]:
    """Resolve promotion-package feature groups into pipeline enums."""
    group_map = {
        "technical": FeatureGroup.TECHNICAL,
        "statistical": FeatureGroup.STATISTICAL,
        "microstructure": FeatureGroup.MICROSTRUCTURE,
        "cross_sectional": FeatureGroup.CROSS_SECTIONAL,
        "cross-sectional": FeatureGroup.CROSS_SECTIONAL,
        "cross": FeatureGroup.CROSS_SECTIONAL,
        "all": FeatureGroup.ALL,
    }
    resolved: list[FeatureGroup] = []
    for raw_name in contract.feature_groups:
        group = group_map.get(str(raw_name).strip().lower())
        if group is None:
            continue
        if group == FeatureGroup.ALL:
            return [
                FeatureGroup.TECHNICAL,
                FeatureGroup.STATISTICAL,
                FeatureGroup.MICROSTRUCTURE,
                FeatureGroup.CROSS_SECTIONAL,
            ]
        if group == FeatureGroup.CROSS_SECTIONAL and not contract.enable_cross_sectional:
            continue
        if group not in resolved:
            resolved.append(group)
    return resolved or [FeatureGroup.TECHNICAL, FeatureGroup.STATISTICAL]


def _extract_timestamps(frame: Any) -> pd.Series:
    """Extract UTC timestamps from pandas/polars-like inputs."""
    if hasattr(frame, "columns") and "timestamp" in frame.columns:
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        return pd.Series(timestamps).reset_index(drop=True)
    if hasattr(frame, "index"):
        timestamps = pd.to_datetime(frame.index, utc=True, errors="coerce")
        return pd.Series(timestamps).reset_index(drop=True)
    return pd.Series(dtype="datetime64[ns, UTC]")


def _to_pandas_frame(frame: Any) -> pd.DataFrame:
    """Normalize market data to a pandas DataFrame."""
    if isinstance(frame, pd.DataFrame):
        pdf = frame.copy()
    elif hasattr(frame, "to_pandas"):
        pdf = frame.to_pandas()
    else:
        pdf = pd.DataFrame(frame)
    if "timestamp" not in pdf.columns:
        pdf = pdf.reset_index()
        first_column = pdf.columns[0]
        if str(first_column) != "timestamp":
            pdf = pdf.rename(columns={first_column: "timestamp"})
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], utc=True, errors="coerce")
    return pdf


def _load_pickle(path: Path) -> Any:
    """Load a trusted pickle artifact."""
    with path.open("rb") as handle:
        return pickle.load(handle)


class PromotionSignalAdapter:
    """Artifact-driven promotion package adapter shared by backtest and replay."""

    def __init__(
        self,
        contract: PromotionPackageContract,
        use_gpu: bool = False,
        logger_: logging.Logger | None = None,
    ) -> None:
        self.contract = contract
        self.use_gpu = use_gpu
        self.logger = logger_ or logger
        self._model: Any | None = None
        self._meta_model: Any | None = None
        self._expected_edge_model: Any | None = None
        self.feature_pipeline: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            model_path = getattr(self.contract, "model_path", None)
            if model_path is None:
                raise ValueError("Promotion contract missing model_path for model load.")
            self._model = _load_pickle(model_path)
        return self._model

    def _load_meta_model(self) -> Any | None:
        meta_model_path = getattr(self.contract, "meta_model_path", None)
        if meta_model_path is None:
            return None
        if self._meta_model is None:
            self._meta_model = _load_pickle(meta_model_path)
        return self._meta_model

    def _load_expected_edge_model(self) -> Any | None:
        expected_edge_model_path = getattr(self.contract, "expected_edge_model_path", None)
        if expected_edge_model_path is None:
            return None
        if self._expected_edge_model is None:
            self._expected_edge_model = _load_pickle(expected_edge_model_path)
        return self._expected_edge_model

    def _is_ranker_contract(self, model: Any | None = None) -> bool:
        """Return True when the promotion contract requires ranker scoring semantics."""
        model_type = str(getattr(self.contract, "model_type", "") or "").strip().lower()
        model_class_name = type(model).__name__.lower() if model is not None else ""
        return "rank" in model_type or "ranker" in model_class_name

    def _choose_pipeline(self) -> Any:
        groups = resolve_feature_groups(self.contract)
        if self.use_gpu and CUDF_AVAILABLE:
            pipeline = OptimizedFeaturePipeline(
                config=OptimizedPipelineConfig(compute_mode=ComputeMode.GPU)
            )
        else:
            pipeline = FeaturePipeline(config=FeatureConfig(groups=groups))
        self.feature_pipeline = pipeline
        return pipeline

    def _resolve_side_policy(self, side: str) -> dict[str, Any]:
        """Return normalized long/short inference policy."""
        raw_policy = (
            getattr(self.contract, "long_side_policy", {})
            if str(side).strip().lower() == "long"
            else getattr(self.contract, "short_side_policy", {})
        )
        if not isinstance(raw_policy, dict):
            raw_policy = {}
        return {
            "enabled": bool(raw_policy.get("enabled", True)),
            "signal_scale": float(np.clip(float(raw_policy.get("signal_scale", 1.0)), 0.0, 1.25)),
            "confidence_scale": float(
                np.clip(float(raw_policy.get("confidence_scale", 1.0)), 0.0, 1.20)
            ),
            "threshold_adjustment": float(
                np.clip(float(raw_policy.get("threshold_adjustment", 0.0)), -0.08, 0.08)
            ),
            "trade_count": int(max(0, int(float(raw_policy.get("trade_count", 0))))),
            "score": float(raw_policy.get("score", 0.0) or 0.0),
            "mean_signed_return": float(raw_policy.get("mean_signed_return", 0.0) or 0.0),
            "sample_confidence": float(raw_policy.get("sample_confidence", 0.0) or 0.0),
        }

    def _effective_thresholds(self) -> tuple[float, float]:
        """Resolve threshold adjustments from side-specific inference policy."""
        long_policy = self._resolve_side_policy("long")
        short_policy = self._resolve_side_policy("short")
        long_threshold = float(
            np.clip(
                float(self.contract.long_threshold) + float(long_policy["threshold_adjustment"]),
                0.51,
                0.99,
            )
        )
        short_threshold = float(
            np.clip(
                float(self.contract.short_threshold) + float(short_policy["threshold_adjustment"]),
                0.01,
                0.49,
            )
        )
        if long_threshold <= short_threshold:
            return float(self.contract.long_threshold), float(self.contract.short_threshold)
        return long_threshold, short_threshold

    def _resolve_runtime_regimes(
        self,
        symbol_frame: pd.DataFrame,
        raw_frame: pd.DataFrame,
        *,
        length: int,
    ) -> np.ndarray:
        """Resolve canonical runtime regime labels for promotion inference."""
        if length <= 0:
            return np.array([], dtype=object)
        if "regime" in symbol_frame.columns:
            candidate = (
                pd.Series(symbol_frame["regime"], dtype="object")
                .fillna("normal_range")
                .astype(str)
                .str.strip()
            )
            if (candidate != "").any():
                values = candidate.where(candidate != "", "normal_range").to_numpy(dtype=object)
                return _align_vector(values, length, "normal_range").astype(object)

        training_config = (
            getattr(self.contract, "raw_payload", {}).get("training_config", {})
            if isinstance(getattr(self.contract, "raw_payload", {}), dict)
            else {}
        )
        regime_lookback = int(float(_coalesce(training_config.get("label_regime_lookback"), 30)))
        inferred = infer_regime_series(
            raw_frame.reset_index(drop=False) if "close" not in raw_frame.columns else raw_frame,
            lookback=max(2, regime_lookback),
            vol_lookback=max(2, min(20, regime_lookback)),
        )
        return _align_vector(inferred.to_numpy(dtype=object), length, "normal_range").astype(object)

    def _resolve_symbol_quality_thresholds(self) -> SymbolQualityThresholds | None:
        """Return runtime symbol quality thresholds when universe gating is enabled."""
        if not bool(getattr(self.contract, "enable_universe_quality_gate", False)):
            return None
        policy = (
            getattr(self.contract, "universe_quality_policy", {})
            if isinstance(getattr(self.contract, "universe_quality_policy", {}), dict)
            else {}
        )
        return SymbolQualityThresholds(
            min_rows=int(
                float(
                    _coalesce(
                        policy.get("min_rows"),
                        (
                            getattr(self.contract, "raw_payload", {})
                            .get("training_config", {})
                            .get("symbol_quality_min_rows")
                            if isinstance(getattr(self.contract, "raw_payload", {}), dict)
                            else None
                        ),
                        1200,
                    )
                )
            ),
            max_missing_ratio=float(
                _coalesce(
                    policy.get("max_missing_ratio"),
                    (
                        getattr(self.contract, "raw_payload", {})
                        .get("training_config", {})
                        .get("symbol_quality_max_missing_ratio")
                        if isinstance(getattr(self.contract, "raw_payload", {}), dict)
                        else None
                    ),
                    0.12,
                )
            ),
            max_extreme_move_ratio=float(
                _coalesce(
                    policy.get("max_extreme_move_ratio"),
                    (
                        getattr(self.contract, "raw_payload", {})
                        .get("training_config", {})
                        .get("symbol_quality_max_extreme_move_ratio")
                        if isinstance(getattr(self.contract, "raw_payload", {}), dict)
                        else None
                    ),
                    0.08,
                )
            ),
            max_corporate_action_ratio=float(
                _coalesce(
                    policy.get("max_corporate_action_ratio"),
                    (
                        getattr(self.contract, "raw_payload", {})
                        .get("training_config", {})
                        .get("symbol_quality_max_corporate_action_ratio")
                        if isinstance(getattr(self.contract, "raw_payload", {}), dict)
                        else None
                    ),
                    0.02,
                )
            ),
            min_median_dollar_volume=float(
                _coalesce(
                    policy.get("min_median_dollar_volume"),
                    (
                        getattr(self.contract, "raw_payload", {})
                        .get("training_config", {})
                        .get("symbol_quality_min_median_dollar_volume")
                        if isinstance(getattr(self.contract, "raw_payload", {}), dict)
                        else None
                    ),
                    1_000_000.0,
                )
            ),
        )

    def _assess_runtime_symbol_quality(
        self, data: dict[str, pd.DataFrame]
    ) -> dict[str, dict[str, Any]]:
        """Assess symbol quality for promotion-package backtest/paper inference."""
        thresholds = self._resolve_symbol_quality_thresholds()
        if thresholds is None:
            return {
                symbol: {"passes": True, "quality_score": 1.0, "reasons": []}
                for symbol in data.keys()
            }

        policy = (
            getattr(self.contract, "universe_quality_policy", {})
            if isinstance(getattr(self.contract, "universe_quality_policy", {}), dict)
            else {}
        )
        selected_symbols = {
            value.upper() for value in _normalize_string_list(policy.get("selected_symbols"))
        }
        contract_symbols = getattr(self.contract, "symbols", ())
        if not selected_symbols and contract_symbols:
            selected_symbols = {str(symbol).strip().upper() for symbol in contract_symbols}

        assessments: dict[str, dict[str, Any]] = {}
        for symbol, frame in data.items():
            assessment = assess_symbol_quality(frame, thresholds)
            reasons = list(assessment.get("reasons", []))
            in_training_universe = True
            if selected_symbols and str(symbol).strip().upper() not in selected_symbols:
                in_training_universe = False
                reasons.append("not_in_training_universe")
            assessment["in_training_universe"] = in_training_universe
            assessment["passes"] = bool(assessment.get("passes", False) and in_training_universe)
            assessment["reasons"] = reasons
            assessments[str(symbol).strip().upper()] = assessment
        return assessments

    @staticmethod
    def _resolve_probability_calibration_payload(model: Any) -> dict[str, Any] | None:
        """Read attached probability calibration metadata from wrapper or raw model."""
        for candidate in (model, getattr(model, "_model", None)):
            payload = getattr(candidate, "_alphatrade_probability_calibration", None)
            if isinstance(payload, dict) and payload.get("calibrator") is not None:
                return payload
        return None

    @staticmethod
    def _apply_probability_calibration(values: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
        """Apply attached post-hoc calibration to probability-like model outputs."""
        calibrator = payload.get("calibrator")
        method = str(payload.get("method") or "isotonic").strip().lower()
        raw_values = np.asarray(values, dtype=float).reshape(-1)
        if raw_values.size == 0 or calibrator is None:
            return raw_values
        if method == "isotonic":
            calibrated = calibrator.predict(raw_values)
        else:
            calibrated = calibrator.predict_proba(raw_values.reshape(-1, 1))[:, 1]
        return np.clip(np.asarray(calibrated, dtype=float).reshape(-1), 0.0, 1.0)

    def _compute_symbol_feature_frame(
        self,
        *,
        df: pd.DataFrame,
        symbol: str,
        groups: list[FeatureGroup],
        universe_data: dict[str, Any] | None,
    ) -> pd.DataFrame:
        """Compute a deterministic feature frame using the same group semantics as training."""
        import polars as pl

        ordered = df.copy().sort_values("timestamp").reset_index(drop=True)
        ordered["symbol"] = symbol
        if not groups:
            return ordered

        df_pl = pl.from_pandas(ordered)
        combined_features: dict[str, np.ndarray] = {}
        for group in groups:
            feature_config = FeatureConfig(
                groups=[group],
                normalization=NormalizationMethod.NONE,
                include_targets=False,
                use_gpu=self.use_gpu and group == FeatureGroup.TECHNICAL,
                use_cache=False,
                use_optimized_pipeline=(group == FeatureGroup.TECHNICAL),
            )
            pipeline = FeaturePipeline(feature_config)
            feature_set = pipeline.compute(
                df_pl,
                symbol=symbol,
                universe_data=(universe_data if group == FeatureGroup.CROSS_SECTIONAL else None),
                use_cache=False,
            )
            combined_features.update(feature_set.features)

        if not combined_features:
            return ordered

        feature_df = pd.DataFrame(combined_features, index=ordered.index)
        features = pd.concat([ordered, feature_df], axis=1)
        features["symbol"] = symbol
        return features

    def compute_features(self, data: dict[str, Any]) -> dict[str, pd.DataFrame]:
        """Compute package-compatible feature frames from raw market data."""
        import polars as pl

        groups = resolve_feature_groups(self.contract)
        normalized_data: dict[str, pd.DataFrame] = {}
        for raw_symbol, frame in data.items():
            symbol = str(raw_symbol).strip().upper()
            pdf = _to_pandas_frame(frame).copy()
            pdf["symbol"] = symbol
            normalized_data[symbol] = pdf.sort_values("timestamp").reset_index(drop=True)
        quality_assessments = self._assess_runtime_symbol_quality(normalized_data)

        universe_data: dict[str, pl.DataFrame] | None = None
        if FeatureGroup.CROSS_SECTIONAL in groups:
            universe_data = {
                symbol: pl.from_pandas(frame)
                for symbol, frame in normalized_data.items()
                if quality_assessments.get(symbol, {}).get("passes", True)
            }

        feature_frames: dict[str, pd.DataFrame] = {}
        timeframes = tuple(self.contract.timeframes or (self.contract.timeframe,))
        if len(timeframes) > 1:
            engine = MultiTimeframeFeatureEngine(
                base_timeframe=self.contract.timeframe,
                timeframes=list(timeframes),
            )
            base_timeframe = normalize_timeframe(self.contract.timeframe, default=DEFAULT_TIMEFRAME)
            for symbol, pdf in normalized_data.items():
                if not quality_assessments.get(symbol, {}).get("passes", True):
                    continue
                ohlcv_frames = engine.build_ohlcv_frames(pdf)
                layered_frames: dict[str, pd.DataFrame] = {}
                for timeframe in engine.normalized_timeframes():
                    timeframe_frame = ohlcv_frames.get(timeframe)
                    if timeframe_frame is None or timeframe_frame.empty:
                        continue
                    timeframe_groups = list(groups)
                    if timeframe != base_timeframe:
                        timeframe_groups = [
                            group
                            for group in timeframe_groups
                            if group != FeatureGroup.CROSS_SECTIONAL
                        ]
                    layered_frames[timeframe] = self._compute_symbol_feature_frame(
                        df=timeframe_frame,
                        symbol=symbol,
                        groups=timeframe_groups,
                        universe_data=(universe_data if timeframe == base_timeframe else None),
                    )

                if base_timeframe not in layered_frames:
                    raise ValueError(
                        f"Promotion package features missing base timeframe {base_timeframe} for {symbol}"
                    )
                aligned = engine.align_feature_frames(
                    base_frame=layered_frames[base_timeframe],
                    feature_frames=layered_frames,
                    include_resampled_ohlcv=True,
                )
                aligned["symbol"] = symbol
                feature_frames[symbol] = aligned.reset_index(drop=True)
        else:
            for symbol, pdf in normalized_data.items():
                if not quality_assessments.get(symbol, {}).get("passes", True):
                    continue
                feature_frames[symbol] = self._compute_symbol_feature_frame(
                    df=pdf,
                    symbol=symbol,
                    groups=groups,
                    universe_data=universe_data,
                ).reset_index(drop=True)

        if not feature_frames:
            return feature_frames

        merged = pd.concat(feature_frames.values(), ignore_index=True)
        if self.contract.enable_reference_features:
            try:
                from quant_trading_system.features.reference import (
                    ReferenceFeatureBuilder,
                    build_reference_feature_config,
                )

                merged = ReferenceFeatureBuilder(
                    config=build_reference_feature_config(
                        enabled=self.contract.enable_reference_features,
                        selected_sources=getattr(self.contract, "reference_feature_sources", ()),
                    ),
                    logger_=self.logger,
                ).augment(merged)
            except Exception as exc:
                self.logger.warning("Promotion reference feature augmentation skipped: %s", exc)
        if self.contract.enable_tick_microstructure_features:
            try:
                from quant_trading_system.features.tick_microstructure import (
                    TickMicrostructureFeatureBuilder,
                    TickMicrostructureFeatureConfig,
                )

                merged = TickMicrostructureFeatureBuilder(
                    TickMicrostructureFeatureConfig(timeframe=self.contract.timeframe),
                    logger_=self.logger,
                ).augment(merged)
            except Exception as exc:
                self.logger.warning("Promotion tick microstructure augmentation skipped: %s", exc)

        merged["symbol"] = merged["symbol"].astype(str).str.upper().str.strip()
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
        merged = merged.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        split: dict[str, pd.DataFrame] = {}
        for symbol in merged["symbol"].dropna().unique().tolist():
            split[str(symbol)] = merged[merged["symbol"] == symbol].copy().reset_index(drop=True)
        return split

    def _predict_model_probabilities(
        self,
        model: Any,
        X_valid: np.ndarray,
        *,
        timestamps: np.ndarray | list[Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return probability-like scores using the same transforms as training."""
        is_ranker_model = self._is_ranker_contract(model)
        X_inference = prepare_model_inference_input(
            model,
            X_valid,
            fallback_feature_names=getattr(self.contract, "feature_names", ()),
        )

        if is_ranker_model:
            raw_scores = np.asarray(model.predict(X_inference), dtype=float).reshape(-1)
            score = _normalize_ranker_scores(
                raw_scores,
                timestamps=timestamps,
                normalization_mode=str(
                    getattr(self.contract, "ranker_score_normalization", None) or "query_percentile"
                ),
            )
            return score.astype(float), raw_scores.astype(float)

        try:
            if hasattr(model, "predict_proba"):
                probabilities = np.asarray(model.predict_proba(X_inference), dtype=float)
                if probabilities.ndim == 2:
                    score = (
                        probabilities[:, -1] if probabilities.shape[1] >= 2 else probabilities[:, 0]
                    )
                else:
                    score = probabilities.reshape(-1)
                raw_score = score.astype(float)
                payload = self._resolve_probability_calibration_payload(model)
                if payload is not None:
                    score = self._apply_probability_calibration(score, payload)
                return score.astype(float), raw_score
        except (AttributeError, NotImplementedError):
            pass

        raw_prediction = np.asarray(model.predict(X_inference), dtype=float).reshape(-1)
        clipped = np.clip(raw_prediction, -1.0, 1.0)
        score = (clipped + 1.0) / 2.0
        payload = self._resolve_probability_calibration_payload(model)
        if payload is not None:
            score = self._apply_probability_calibration(score, payload)
        return score.astype(float), raw_prediction.astype(float)

    @staticmethod
    def _extract_close_series(frame: Any, length: int) -> pd.Series:
        """Extract close prices aligned to the requested length."""
        pdf = _to_pandas_frame(frame)
        if "close" in pdf.columns:
            close_series = pd.Series(pdf["close"])
        elif "Close" in pdf.columns:
            close_series = pd.Series(pdf["Close"])
        else:
            close_series = pd.Series(np.full(length, np.nan, dtype=float))
        return (
            pd.to_numeric(close_series, errors="coerce")
            .reset_index(drop=True)
            .reindex(range(length))
        )

    def _score_ranker_panel(
        self,
        *,
        model: Any,
        normalized_data: dict[str, pd.DataFrame],
        feature_frames: dict[str, pd.DataFrame],
        required_features: list[str],
        quality_assessments: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Score ranker inputs across the full timestamp panel to preserve training parity."""
        payloads: dict[str, dict[str, Any]] = {}
        panel_parts: list[pd.DataFrame] = []

        for symbol, raw_frame in normalized_data.items():
            feature_df = feature_frames.get(symbol)
            if feature_df is None:
                raise ValueError(f"Missing feature frame for promotion symbol {symbol}")
            symbol_frame = feature_df.copy().reset_index(drop=True)
            missing_features = [
                feature_name
                for feature_name in required_features
                if feature_name not in symbol_frame.columns
            ]
            if missing_features:
                preview = ", ".join(missing_features[:10])
                raise ValueError(
                    f"Missing {len(missing_features)} required features for {symbol}: {preview}"
                )

            X_df = (
                symbol_frame.loc[:, required_features]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )
            valid_mask = X_df.notna().all(axis=1)
            probability = np.full(len(X_df), np.nan, dtype=float)
            raw_prediction = np.full(len(X_df), np.nan, dtype=float)
            timestamps = _extract_timestamps(raw_frame).reindex(range(len(X_df)))
            payloads[symbol] = {
                "symbol_frame": symbol_frame,
                "X_df": X_df,
                "valid_mask": valid_mask,
                "probability": probability,
                "raw_prediction": raw_prediction,
                "timestamps": timestamps,
            }

            quality = quality_assessments.get(symbol, {"passes": True})
            if not bool(quality.get("passes", True)) or not bool(valid_mask.any()):
                continue

            candidate_rows = X_df.loc[valid_mask].copy()
            candidate_rows["timestamp"] = pd.to_datetime(
                timestamps.loc[valid_mask].to_numpy(),
                utc=True,
                errors="coerce",
            )
            candidate_rows["symbol"] = str(symbol)
            candidate_rows["row_index"] = np.flatnonzero(valid_mask.to_numpy(dtype=bool))
            candidate_rows = candidate_rows.dropna(subset=["timestamp"])
            if not candidate_rows.empty:
                panel_parts.append(candidate_rows)

        if not panel_parts:
            return payloads

        panel = (
            pd.concat(panel_parts, ignore_index=True)
            .sort_values(["timestamp", "symbol", "row_index"], kind="stable")
            .reset_index(drop=True)
        )
        panel_scores, panel_raw = self._predict_model_probabilities(
            model,
            panel.loc[:, required_features].to_numpy(dtype=float),
            timestamps=panel["timestamp"].to_numpy(),
        )
        panel["probability"] = np.clip(panel_scores, 0.0, 1.0)
        panel["raw_prediction"] = panel_raw

        for symbol, symbol_panel in panel.groupby("symbol", sort=False):
            payload = payloads.get(str(symbol))
            if payload is None:
                continue
            row_index = symbol_panel["row_index"].to_numpy(dtype=int)
            payload["probability"][row_index] = symbol_panel["probability"].to_numpy(dtype=float)
            payload["raw_prediction"][row_index] = symbol_panel["raw_prediction"].to_numpy(
                dtype=float
            )

        return payloads

    def generate_signal_frames(
        self,
        data: dict[str, Any],
        features: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Generate promotion-package signal frames aligned to raw market bars."""
        model = self._load_model()
        meta_model = self._load_meta_model()
        expected_edge_model = (
            self._load_expected_edge_model()
            if bool(getattr(self.contract, "expected_edge_policy_enabled", False))
            else None
        )
        required_features = list(self.contract.feature_names or getattr(model, "feature_names", []))
        if not required_features:
            raise ValueError(
                "Promotion package does not define feature_names and loaded model exposes none."
            )

        feature_frames = features or self.compute_features(data)
        normalized_data = {
            str(symbol).strip().upper(): _to_pandas_frame(frame) for symbol, frame in data.items()
        }
        quality_assessments = self._assess_runtime_symbol_quality(normalized_data)
        effective_long_threshold, effective_short_threshold = self._effective_thresholds()
        long_policy = self._resolve_side_policy("long")
        short_policy = self._resolve_side_policy("short")
        expected_edge_policy_payload = (
            getattr(self.contract, "expected_edge_policy", {})
            if isinstance(getattr(self.contract, "expected_edge_policy", {}), dict)
            else {}
        )
        regime_policy_payload = (
            expected_edge_policy_payload.get("regime_conditioned_policy", {})
            if isinstance(expected_edge_policy_payload.get("regime_conditioned_policy", {}), dict)
            else {}
        )
        ranker_payloads = (
            self._score_ranker_panel(
                model=model,
                normalized_data=normalized_data,
                feature_frames=feature_frames,
                required_features=required_features,
                quality_assessments=quality_assessments,
            )
            if self._is_ranker_contract(model)
            else {}
        )
        signal_frames: dict[str, pd.DataFrame] = {}
        for symbol, raw_frame in normalized_data.items():
            quality = quality_assessments.get(
                symbol,
                {"passes": True, "quality_score": 1.0, "reasons": []},
            )
            timestamps = _extract_timestamps(raw_frame)
            liquidity_frame = _estimate_trailing_liquidity_frame(
                raw_frame,
                timeframe=getattr(self.contract, "timeframe", DEFAULT_TIMEFRAME),
                lookback_days=max(1, int(getattr(self.contract, "adv_lookback_days", 20))),
            )
            aligned_liquidity = pd.DataFrame({"timestamp": timestamps}).merge(
                liquidity_frame,
                on="timestamp",
                how="left",
            )
            avg_daily_volume_series = aligned_liquidity.get("avg_daily_volume")
            if avg_daily_volume_series is None:
                avg_daily_volume_series = pd.Series(np.nan, index=aligned_liquidity.index)
            avg_daily_volume = pd.to_numeric(
                avg_daily_volume_series,
                errors="coerce",
            ).to_numpy(dtype=float)
            avg_daily_dollar_volume_series = aligned_liquidity.get("avg_daily_dollar_volume")
            if avg_daily_dollar_volume_series is None:
                avg_daily_dollar_volume_series = pd.Series(np.nan, index=aligned_liquidity.index)
            avg_daily_dollar_volume = pd.to_numeric(
                avg_daily_dollar_volume_series,
                errors="coerce",
            ).to_numpy(dtype=float)
            spread_bps_series = aligned_liquidity.get("spread_bps")
            if spread_bps_series is None:
                spread_bps_series = pd.Series(np.nan, index=aligned_liquidity.index)
            spread_bps = pd.to_numeric(
                spread_bps_series,
                errors="coerce",
            ).to_numpy(dtype=float)
            candidate_selection_values = np.full(
                len(timestamps),
                dict(getattr(self.contract, "candidate_selection_policy", {})),
                dtype=object,
            )
            short_policy_values = np.full(
                len(timestamps),
                dict(getattr(self.contract, "short_policy", {})),
                dtype=object,
            )
            exit_policy_values = np.full(
                len(timestamps),
                dict(getattr(self.contract, "exit_policy", {})),
                dtype=object,
            )
            raw_meta_threshold = getattr(self.contract, "meta_threshold", None)
            meta_threshold_value = (
                float(raw_meta_threshold) if raw_meta_threshold is not None else np.nan
            )
            if not bool(quality.get("passes", True)):
                signal_frames[symbol] = pd.DataFrame(
                    {
                        "timestamp": timestamps,
                        "signal": np.zeros(len(timestamps), dtype=float),
                        "confidence": np.zeros(len(timestamps), dtype=float),
                        "horizon": int(self.contract.horizon_bars),
                        "model_source": self.contract.model_source,
                        "probability": np.full(len(timestamps), np.nan, dtype=float),
                        "raw_prediction": np.full(len(timestamps), np.nan, dtype=float),
                        "long_threshold": float(effective_long_threshold),
                        "short_threshold": float(effective_short_threshold),
                        "base_long_threshold": float(self.contract.long_threshold),
                        "base_short_threshold": float(self.contract.short_threshold),
                        "meta_confidence": np.full(len(timestamps), np.nan, dtype=float),
                        "take_profit_pct": float(self.contract.take_profit_pct),
                        "stop_loss_pct": float(self.contract.stop_loss_pct),
                        "max_holding_bars": int(self.contract.max_holding_bars),
                        "holding_horizon_bars": int(
                            getattr(self.contract, "holding_horizon_bars", self.contract.max_holding_bars)
                        ),
                        "universe_eligible": False,
                        "universe_quality_score": float(quality.get("quality_score", 0.0)),
                        "universe_quality_reasons": "|".join(
                            str(reason) for reason in quality.get("reasons", [])
                        ),
                        "long_side_policy_scale": float(long_policy["signal_scale"]),
                        "short_side_policy_scale": float(short_policy["signal_scale"]),
                        "runtime_regime": np.full(len(timestamps), "normal_range", dtype=object),
                        "regime_policy_enabled": np.zeros(len(timestamps), dtype=bool),
                        "regime_policy_signal_scale": np.zeros(len(timestamps), dtype=float),
                        "regime_policy_confidence_scale": np.zeros(len(timestamps), dtype=float),
                        "regime_long_threshold": np.full(
                            len(timestamps), float(effective_long_threshold), dtype=float
                        ),
                        "regime_short_threshold": np.full(
                            len(timestamps), float(effective_short_threshold), dtype=float
                        ),
                        "expected_edge": np.zeros(len(timestamps), dtype=float),
                        "edge_pass_probability": np.zeros(len(timestamps), dtype=float),
                        "edge_loss_probability": np.zeros(len(timestamps), dtype=float),
                        "edge_policy_pass": np.zeros(len(timestamps), dtype=bool),
                        "edge_policy_scale": np.zeros(len(timestamps), dtype=float),
                        "edge_policy_confidence_scale": np.zeros(len(timestamps), dtype=float),
                        "edge_policy_enabled": bool(expected_edge_model is not None),
                        "score_semantics": np.full(
                            len(timestamps),
                            str(getattr(self.contract, "score_semantics", "probability_up")),
                            dtype=object,
                        ),
                        "candidate_selection_policy": candidate_selection_values,
                        "short_policy": short_policy_values,
                        "exit_policy": exit_policy_values,
                        "meta_model_enabled": bool(getattr(self.contract, "meta_model_enabled", False)),
                        "meta_threshold": np.full(
                            len(timestamps),
                            meta_threshold_value,
                            dtype=float,
                        ),
                        "borrow_check_required": np.full(
                            len(timestamps),
                            bool(getattr(self.contract, "borrow_check_required", False)),
                            dtype=bool,
                        ),
                        "adv_lookback_days": np.full(
                            len(timestamps),
                            int(getattr(self.contract, "adv_lookback_days", 20)),
                            dtype=int,
                        ),
                        "raw_candidate": np.zeros(len(timestamps), dtype=bool),
                        "meta_passed": np.zeros(len(timestamps), dtype=bool),
                        "edge_passed": np.zeros(len(timestamps), dtype=bool),
                        "regime_blocked": np.zeros(len(timestamps), dtype=bool),
                        "candidate_direction": np.full(len(timestamps), "", dtype=object),
                        "avg_daily_volume": avg_daily_volume,
                        "avg_daily_dollar_volume": avg_daily_dollar_volume,
                        "spread_bps": spread_bps,
                    }
                )
                continue

            scoring_payload = ranker_payloads.get(symbol)
            if scoring_payload is not None:
                symbol_frame = scoring_payload["symbol_frame"].copy()
                X_df = scoring_payload["X_df"].copy()
                valid_mask = scoring_payload["valid_mask"].copy()
                probability = np.asarray(scoring_payload["probability"], dtype=float)
                raw_prediction = np.asarray(scoring_payload["raw_prediction"], dtype=float)
            else:
                feature_df = feature_frames.get(symbol)
                if feature_df is None:
                    raise ValueError(f"Missing feature frame for promotion symbol {symbol}")
                symbol_frame = feature_df.copy().reset_index(drop=True)
                missing_features = [
                    feature_name
                    for feature_name in required_features
                    if feature_name not in symbol_frame.columns
                ]
                if missing_features:
                    preview = ", ".join(missing_features[:10])
                    raise ValueError(
                        f"Missing {len(missing_features)} required features for {symbol}: {preview}"
                    )

                X_df = (
                    symbol_frame.loc[:, required_features]
                    .apply(pd.to_numeric, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                )
                valid_mask = X_df.notna().all(axis=1)
                probability = np.full(len(X_df), np.nan, dtype=float)
                raw_prediction = np.full(len(X_df), np.nan, dtype=float)
                if bool(valid_mask.any()):
                    score, raw = self._predict_model_probabilities(
                        model,
                        X_df.loc[valid_mask].to_numpy(dtype=float),
                    )
                    probability[valid_mask.to_numpy()] = np.clip(score, 0.0, 1.0)
                    raw_prediction[valid_mask.to_numpy()] = raw

            runtime_regimes = self._resolve_runtime_regimes(
                symbol_frame, raw_frame, length=len(X_df)
            )
            regime_policy_frame = resolve_regime_policy_frame(
                runtime_regimes,
                regime_policy_payload,
                length=len(X_df),
            )
            regime_enabled = (
                regime_policy_frame["enabled"].to_numpy(dtype=bool)
                if "enabled" in regime_policy_frame.columns
                else np.ones(len(X_df), dtype=bool)
            )
            regime_long_threshold = np.clip(
                float(effective_long_threshold)
                + regime_policy_frame.get(
                    "long_threshold_adjustment",
                    pd.Series(np.zeros(len(X_df), dtype=float)),
                ).to_numpy(dtype=float),
                0.51,
                0.99,
            )
            regime_short_threshold = np.clip(
                float(effective_short_threshold)
                + regime_policy_frame.get(
                    "short_threshold_adjustment",
                    pd.Series(np.zeros(len(X_df), dtype=float)),
                ).to_numpy(dtype=float),
                0.01,
                0.49,
            )
            invalid_threshold_mask = regime_long_threshold <= regime_short_threshold
            regime_long_threshold = np.where(
                invalid_threshold_mask,
                float(effective_long_threshold),
                regime_long_threshold,
            )
            regime_short_threshold = np.where(
                invalid_threshold_mask,
                float(effective_short_threshold),
                regime_short_threshold,
            )

            candidate_long_mask = probability >= regime_long_threshold
            candidate_short_mask = probability <= regime_short_threshold
            regime_blocked = (~regime_enabled) & (candidate_long_mask | candidate_short_mask)
            raw_candidate = valid_mask.to_numpy(dtype=bool) & (
                candidate_long_mask | candidate_short_mask
            )
            candidate_direction = np.full(len(X_df), "", dtype=object)
            candidate_direction[candidate_long_mask] = "long"
            candidate_direction[candidate_short_mask] = "short"
            long_mask = regime_enabled & candidate_long_mask
            short_mask = regime_enabled & candidate_short_mask
            signal_values = np.zeros(len(X_df), dtype=float)
            long_scale = np.maximum(1e-6, 1.0 - regime_long_threshold)
            short_scale = np.maximum(1e-6, regime_short_threshold)
            signal_values[long_mask] = np.clip(
                (probability[long_mask] - regime_long_threshold[long_mask]) / long_scale[long_mask],
                0.0,
                1.0,
            )
            signal_values[short_mask] = -np.clip(
                (regime_short_threshold[short_mask] - probability[short_mask])
                / short_scale[short_mask],
                0.0,
                1.0,
            )

            confidence = np.clip(np.abs(probability - 0.5) * 2.0, 0.0, 1.0)
            if not bool(long_policy["enabled"]):
                signal_values[long_mask] = 0.0
                confidence[long_mask] = 0.0
            else:
                signal_values[long_mask] = np.clip(
                    signal_values[long_mask] * float(long_policy["signal_scale"]),
                    0.0,
                    1.0,
                )
                confidence[long_mask] = np.clip(
                    confidence[long_mask] * float(long_policy["confidence_scale"]),
                    0.0,
                    1.0,
                )
            if not bool(short_policy["enabled"]):
                signal_values[short_mask] = 0.0
                confidence[short_mask] = 0.0
            else:
                signal_values[short_mask] = -np.clip(
                    np.abs(signal_values[short_mask]) * float(short_policy["signal_scale"]),
                    0.0,
                    1.0,
                )
                confidence[short_mask] = np.clip(
                    confidence[short_mask] * float(short_policy["confidence_scale"]),
                    0.0,
                    1.0,
                )
            edge_signal_values = signal_values.copy()
            edge_confidence = confidence.copy()

            meta_confidence = np.full(len(X_df), np.nan, dtype=float)
            if (
                meta_model is not None
                and self.contract.meta_label_enabled
                and hasattr(meta_model, "filter_signals")
            ):
                close_series = self._extract_close_series(raw_frame, len(X_df))
                primary_direction = pd.Series(np.sign(signal_values), dtype=float)
                filtered = meta_model.filter_signals(
                    X_df.fillna(0.0),
                    primary_direction,
                    prices=close_series,
                    threshold=self.contract.meta_label_threshold,
                )
                filtered_signal = pd.to_numeric(
                    filtered.get("filtered_signal", primary_direction),
                    errors="coerce",
                ).fillna(0.0)
                meta_confidence = pd.to_numeric(
                    filtered.get("confidence", np.nan),
                    errors="coerce",
                ).to_numpy(dtype=float)
                signal_values = np.where(
                    filtered_signal.to_numpy(dtype=float) > 0.0,
                    np.maximum(signal_values, 0.0),
                    np.where(
                        filtered_signal.to_numpy(dtype=float) < 0.0,
                        np.minimum(signal_values, 0.0),
                        0.0,
                    ),
                )
                confidence = np.clip(
                    np.where(np.isfinite(meta_confidence), meta_confidence, confidence),
                    0.0,
                    1.0,
                )
            meta_passed = raw_candidate & (np.abs(signal_values) > 0.0)

            expected_edge = np.zeros(len(X_df), dtype=float)
            edge_pass_probability = np.zeros(len(X_df), dtype=float)
            edge_loss_probability = np.zeros(len(X_df), dtype=float)
            edge_policy_pass = (np.abs(edge_signal_values) > 0.0).astype(bool)
            edge_policy_scale = np.where(
                np.abs(edge_signal_values) > 0.0,
                1.0,
                0.0,
            ).astype(float)
            edge_policy_confidence_scale = np.where(
                np.abs(edge_signal_values) > 0.0,
                1.0,
                0.0,
            ).astype(float)
            if expected_edge_model is not None and hasattr(expected_edge_model, "predict_policy"):
                edge_policy = expected_edge_model.predict_policy(
                    symbol_frame,
                    probabilities=probability,
                    long_threshold=float(effective_long_threshold),
                    short_threshold=float(effective_short_threshold),
                    raw_predictions=raw_prediction,
                    signal_values=edge_signal_values,
                    confidence=edge_confidence,
                    symbols=np.full(len(X_df), str(symbol), dtype=object),
                    regimes=runtime_regimes,
                    regime_policy=regime_policy_payload,
                )
                expected_edge = (
                    pd.to_numeric(
                        pd.Series(_align_vector(edge_policy.get("expected_edge"), len(X_df), 0.0)),
                        errors="coerce",
                    )
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                edge_pass_probability = (
                    pd.to_numeric(
                        pd.Series(
                            _align_vector(edge_policy.get("edge_pass_probability"), len(X_df), 0.0)
                        ),
                        errors="coerce",
                    )
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                edge_loss_probability = (
                    pd.to_numeric(
                        pd.Series(
                            _align_vector(edge_policy.get("edge_loss_probability"), len(X_df), 0.0)
                        ),
                        errors="coerce",
                    )
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                edge_policy_pass = (
                    pd.Series(_align_vector(edge_policy.get("edge_policy_pass"), len(X_df), 0.0))
                    .fillna(False)
                    .astype(bool)
                    .to_numpy(dtype=bool)
                )
                edge_policy_scale = (
                    pd.to_numeric(
                        pd.Series(
                            _align_vector(edge_policy.get("edge_policy_scale"), len(X_df), 0.0)
                        ),
                        errors="coerce",
                    )
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                edge_policy_confidence_scale = (
                    pd.to_numeric(
                        pd.Series(
                            _align_vector(
                                edge_policy.get("edge_policy_confidence_scale"),
                                len(X_df),
                                0.0,
                            )
                        ),
                        errors="coerce",
                    )
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                runtime_regimes = _align_vector(
                    edge_policy.get("runtime_regime"),
                    len(X_df),
                    "normal_range",
                ).astype(object)
                regime_enabled = (
                    pd.Series(
                        _align_vector(edge_policy.get("regime_policy_enabled"), len(X_df), 0.0)
                    )
                    .fillna(False)
                    .astype(bool)
                    .to_numpy(dtype=bool)
                )
                regime_signal_scale = (
                    pd.to_numeric(
                        pd.Series(
                            _align_vector(
                                edge_policy.get("regime_policy_signal_scale"),
                                len(X_df),
                                1.0,
                            )
                        ),
                        errors="coerce",
                    )
                    .fillna(1.0)
                    .to_numpy(dtype=float)
                )
                regime_confidence_scale_payload = edge_policy.get("regime_policy_confidence_scale")
                if regime_confidence_scale_payload is None:
                    regime_confidence_scale_payload = edge_policy.get(
                        "edge_policy_confidence_scale"
                    )
                regime_confidence_scale = (
                    pd.to_numeric(
                        pd.Series(
                            _align_vector(
                                regime_confidence_scale_payload,
                                len(X_df),
                                1.0,
                            )
                        ),
                        errors="coerce",
                    )
                    .fillna(1.0)
                    .to_numpy(dtype=float)
                )
                signal_values = np.where(edge_policy_pass, signal_values * edge_policy_scale, 0.0)
                confidence = np.clip(
                    np.where(
                        edge_policy_pass,
                        ((0.5 * confidence) + (0.5 * edge_pass_probability))
                        * edge_policy_confidence_scale,
                        0.0,
                    ),
                    0.0,
                    1.0,
                )
            else:
                regime_signal_scale = regime_policy_frame.get(
                    "signal_scale",
                    pd.Series(np.ones(len(X_df), dtype=float)),
                ).to_numpy(dtype=float)
                regime_confidence_scale = regime_policy_frame.get(
                    "confidence_scale",
                    pd.Series(np.ones(len(X_df), dtype=float)),
                ).to_numpy(dtype=float)
            edge_passed = meta_passed & edge_policy_pass & (np.abs(signal_values) > 0.0)

            timestamps = pd.to_datetime(symbol_frame.get("timestamp"), utc=True, errors="coerce")
            signal_frames[symbol] = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "signal": np.clip(signal_values, -1.0, 1.0),
                    "confidence": confidence,
                    "horizon": int(self.contract.horizon_bars),
                    "model_source": self.contract.model_source,
                    "probability": probability,
                    "raw_prediction": raw_prediction,
                    "long_threshold": regime_long_threshold,
                    "short_threshold": regime_short_threshold,
                    "base_long_threshold": float(self.contract.long_threshold),
                    "base_short_threshold": float(self.contract.short_threshold),
                    "meta_confidence": meta_confidence,
                    "take_profit_pct": float(self.contract.take_profit_pct),
                    "stop_loss_pct": float(self.contract.stop_loss_pct),
                    "max_holding_bars": int(self.contract.max_holding_bars),
                    "holding_horizon_bars": int(
                        getattr(self.contract, "holding_horizon_bars", self.contract.max_holding_bars)
                    ),
                    "universe_eligible": True,
                    "universe_quality_score": float(quality.get("quality_score", 1.0)),
                    "universe_quality_reasons": "|".join(
                        str(reason) for reason in quality.get("reasons", [])
                    ),
                    "long_side_policy_scale": float(long_policy["signal_scale"]),
                    "short_side_policy_scale": float(short_policy["signal_scale"]),
                    "runtime_regime": runtime_regimes,
                    "regime_policy_enabled": regime_enabled,
                    "regime_policy_signal_scale": regime_signal_scale,
                    "regime_policy_confidence_scale": regime_confidence_scale,
                    "regime_long_threshold": regime_long_threshold,
                    "regime_short_threshold": regime_short_threshold,
                    "expected_edge": expected_edge,
                    "edge_pass_probability": edge_pass_probability,
                    "edge_loss_probability": edge_loss_probability,
                    "edge_policy_pass": edge_policy_pass,
                    "edge_policy_scale": edge_policy_scale,
                    "edge_policy_confidence_scale": edge_policy_confidence_scale,
                    "edge_policy_enabled": bool(expected_edge_model is not None),
                    "score_semantics": np.full(
                        len(X_df),
                        str(getattr(self.contract, "score_semantics", "probability_up")),
                        dtype=object,
                    ),
                    "candidate_selection_policy": candidate_selection_values[: len(X_df)],
                    "short_policy": short_policy_values[: len(X_df)],
                    "exit_policy": exit_policy_values[: len(X_df)],
                    "meta_model_enabled": bool(getattr(self.contract, "meta_model_enabled", False)),
                    "meta_threshold": np.full(
                        len(X_df),
                        meta_threshold_value,
                        dtype=float,
                    ),
                    "borrow_check_required": np.full(
                        len(X_df),
                        bool(getattr(self.contract, "borrow_check_required", False)),
                        dtype=bool,
                    ),
                    "adv_lookback_days": np.full(
                        len(X_df),
                        int(getattr(self.contract, "adv_lookback_days", 20)),
                        dtype=int,
                    ),
                    "raw_candidate": raw_candidate,
                    "meta_passed": meta_passed,
                    "edge_passed": edge_passed,
                    "regime_blocked": regime_blocked,
                    "candidate_direction": candidate_direction,
                    "avg_daily_volume": avg_daily_volume[: len(X_df)],
                    "avg_daily_dollar_volume": avg_daily_dollar_volume[: len(X_df)],
                    "spread_bps": spread_bps[: len(X_df)],
                }
            )
        return signal_frames
