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
    python main.py train --model lstm --require-model-gpu --epochs 100
    python main.py train --model all --n-trials 50

Author: AlphaTrade System
Version: 1.3.0
================================================================================
"""

import argparse
import gc
import hashlib
import io
import json
import logging
import os
import pickle
import random
import re
import socket
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import inspect as sa_inspect, text

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant_trading_system.models.training_lineage import (
    append_training_run_event,
    build_data_quality_report,
    build_snapshot_manifest,
    compute_data_quality_hash,
    compute_frame_content_hash,
    load_dataset_snapshot_bundle,
    load_registry_entries,
    persist_dataset_snapshot_bundle,
    persist_active_model_pointer,
    persist_snapshot_bundle,
    register_training_model_version,
    set_registry_active_version,
)
from quant_trading_system.models.symbol_quality import (
    SymbolQualityThresholds,
    assess_symbol_quality,
)
from quant_trading_system.models.signal_policy import derive_asymmetric_signal_policy
from quant_trading_system.models.expected_edge_policy import (
    ExpectedEdgePolicyConfig,
    ExpectedEdgePolicyModel,
    derive_base_signal,
    derive_regime_conditioned_policy,
)
from quant_trading_system.models.feature_schema import prepare_model_inference_input
from quant_trading_system.data.data_access import filter_ohlcv_frame_to_market_session
from quant_trading_system.models.statistical_validation import (
    calculate_deflated_sharpe_ratio,
    calculate_probability_of_backtest_overfitting,
    calculate_white_reality_check,
)
from quant_trading_system.models.target_engineering import (
    TargetEngineeringConfig,
    generate_targets,
)
from quant_trading_system.models.trading_costs import TradingCostModel
from quant_trading_system.data.timeframe import (
    DEFAULT_TIMEFRAME,
    estimate_periods_per_year,
    normalize_timeframe,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)
warnings.filterwarnings(
    "ignore",
    message="no explicit representation of timezones available for np.datetime64",
)

SUPPORTED_MODELS = [
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

DEFAULT_PRIMARY_MODEL = "lightgbm"
PRIMARY_CHALLENGER_MODEL_ALIAS = "primary_challenger"
SUPPORTED_TRAINING_PROFILES = ("promotion", "research")
EXPLICIT_PROFILE_OVERRIDE_ATTR = "_explicit_profile_override_dests"
PROFILE_TUNABLE_ARG_DEFAULTS: dict[str, Any] = {
    "n_splits": 5,
    "n_trials": 100,
    "nested_outer_splits": 4,
    "nested_inner_splits": 3,
    "no_shap": False,
    "disable_meta_labeling": False,
    "disable_auto_live_profile": False,
    "feature_selection_stability_iterations": 16,
}
TRAINING_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "promotion": {},
    "research": {
        "n_splits": 3,
        "n_trials": 30,
        "nested_outer_splits": 2,
        "nested_inner_splits": 2,
        "allow_unstable_outer_fold_fallback": True,
        "use_meta_labeling": False,
        "compute_shap": False,
        "auto_live_profile_enabled": False,
        "feature_selection_stability_iterations": 12,
        "require_nested_trace_for_promotion": False,
        "require_objective_breakdown_for_promotion": False,
    },
}
SNAPSHOT_TRAINING_SCOPE_SCHEMA_VERSION = "1.0.0"
PRE_PROMOTION_CHECKLIST_SCHEMA_VERSION = "1.0.0"
PRE_PROMOTION_WORK_PLAN_THRESHOLDS: dict[str, float] = {
    "mean_trade_count_min": 100.0,
    "mean_risk_adjusted_score_min": 0.0,
    "holdout_max_drawdown_max": 0.35,
    "holdout_symbol_sharpe_p25_min": -0.10,
    "pbo_max": 0.45,
    "white_reality_pvalue_max": 0.10,
}


def _normalize_training_profile(value: Any) -> str:
    """Normalize named training profile selection."""
    profile = str(value or "promotion").strip().lower()
    if profile not in SUPPORTED_TRAINING_PROFILES:
        supported = ", ".join(SUPPORTED_TRAINING_PROFILES)
        raise ValueError(f"training_profile must be one of: {supported}")
    return profile


def _resolve_profiled_value(
    *,
    replay_config: dict[str, Any],
    config_key: str,
    args: argparse.Namespace,
    profile_defaults: dict[str, Any],
    fallback: Any,
    arg_dest: str | None = None,
    arg_transform: Callable[[Any], Any] | None = None,
) -> Any:
    """Resolve config values with replay-config precedence, CLI overrides, then profile presets."""
    if config_key in replay_config:
        return replay_config[config_key]
    explicit_profile_overrides = getattr(args, EXPLICIT_PROFILE_OVERRIDE_ATTR, set())
    if arg_dest and hasattr(args, arg_dest):
        arg_value = getattr(args, arg_dest)
        if arg_dest in explicit_profile_overrides:
            return arg_transform(arg_value) if arg_transform else arg_value
        if arg_value != PROFILE_TUNABLE_ARG_DEFAULTS.get(arg_dest, object()):
            return arg_transform(arg_value) if arg_transform else arg_value
    if config_key in profile_defaults:
        return profile_defaults[config_key]
    if arg_dest and hasattr(args, arg_dest):
        arg_value = getattr(args, arg_dest)
        return arg_transform(arg_value) if arg_transform else arg_value
    return fallback


def _attach_explicit_profile_overrides(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str] | None,
) -> argparse.Namespace:
    """Track CLI options that were explicitly provided, even when equal to parser defaults."""

    def _resolve_action(
        candidate_parser: argparse.ArgumentParser,
        option: str,
    ) -> argparse.Action | None:
        action = candidate_parser._option_string_actions.get(option)
        if action is not None:
            return action
        for parser_action in candidate_parser._actions:
            if not isinstance(parser_action, argparse._SubParsersAction):
                continue
            for subparser in parser_action.choices.values():
                nested_action = _resolve_action(subparser, option)
                if nested_action is not None:
                    return nested_action
        return None

    explicit_dests: set[str] = set()
    for raw_arg in argv or []:
        if not raw_arg.startswith("-"):
            continue
        option = raw_arg.split("=", 1)[0]
        action = _resolve_action(parser, option)
        if action is None:
            continue
        explicit_dests.add(action.dest)
    setattr(args, EXPLICIT_PROFILE_OVERRIDE_ATTR, explicit_dests)
    return args


def _running_in_wsl() -> bool:
    """Return whether the current Python process is running inside WSL."""
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    proc_version = Path("/proc/version")
    if not proc_version.exists():
        return False
    try:
        return "microsoft" in proc_version.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False


def _run_git_command(project_root: Path, *args: str) -> tuple[int, str]:
    """Run a short git command and return exit code plus trimmed stdout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return 1, ""
    return int(result.returncode), str(result.stdout or "").strip()


def _resolve_dependency_lock_hash(project_root: Path) -> str:
    """Hash dependency lock inputs used for reproducible training runs."""
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


def _resolve_training_provenance(project_root: Path) -> dict[str, Any]:
    """Collect lightweight provenance for artifact and promotion audits."""
    commit_rc, commit_hash = _run_git_command(project_root, "rev-parse", "HEAD")
    branch_rc, branch_name = _run_git_command(project_root, "rev-parse", "--abbrev-ref", "HEAD")
    status_rc, status_output = _run_git_command(
        project_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    has_git_repo = bool(commit_rc == 0 and len(commit_hash) >= 7)
    dirty_state = None if status_rc != 0 else bool(status_output)
    host_name = (
        os.environ.get("COMPUTERNAME")
        or os.environ.get("HOSTNAME")
        or socket.gethostname()
        or "unknown"
    )
    wsl_distro = os.environ.get("WSL_DISTRO_NAME", "").strip() or None
    app_env = os.environ.get("APP_ENV", "").strip().lower() or "development"
    provenance = {
        "git_commit_hash": commit_hash if has_git_repo else "unknown",
        "git_branch": branch_name if branch_rc == 0 and branch_name else "unknown",
        "git_dirty": dirty_state,
        "git_repo_available": bool(has_git_repo),
        "dependency_lock_hash": _resolve_dependency_lock_hash(project_root),
        "python_version": sys.version.split()[0],
        "host": host_name,
        "cwd": os.getcwd(),
        "project_root": str(project_root),
        "platform": sys.platform,
        "running_in_wsl": bool(_running_in_wsl()),
        "wsl_distro": wsl_distro,
        "app_env": app_env,
    }
    provenance["git_commit"] = provenance["git_commit_hash"]
    provenance["environment_lock_hash"] = provenance["dependency_lock_hash"]
    provenance["training_host"] = provenance["host"]
    return provenance


def _promotion_repo_provenance_ready(project_root: Path) -> bool:
    """Return whether promotion training is running from a real Git checkout/worktree."""
    provenance = _resolve_training_provenance(project_root)
    return bool(provenance.get("git_repo_available", False))


def _log_training_environment_guidance(profile: str) -> None:
    """Emit filesystem/runtime guidance aligned with the training execution plan."""
    try:
        cwd = os.path.abspath(os.getcwd())
    except OSError:
        cwd = os.getcwd()

    if os.name == "nt":
        logger.warning(
            "Training is running on Windows. For feature-heavy %s runs, prefer WSL Ubuntu with the "
            "repo on the Linux filesystem (for example ~/AlphaTrade).",
            profile,
        )
        return

    if _running_in_wsl() and str(cwd).startswith("/mnt/"):
        logger.warning(
            "Training is running from %s inside WSL. Move the repo under ~/AlphaTrade or another "
            "Linux-native path to avoid Windows mount filesystem overhead.",
            cwd,
        )


def _build_snapshot_feature_selection_signature(config: "TrainingConfig") -> dict[str, Any]:
    """Serialize feature-selection settings that affect snapshot contents."""
    label_horizons = sorted(
        {
            int(h)
            for h in config.label_horizons
            if isinstance(h, (int, np.integer, float, np.floating)) and int(h) > 0
        }
    )
    return {
        "enabled": bool(config.enable_feature_selection),
        "min_information_coefficient": float(config.feature_selection_min_ic),
        "max_correlation": float(config.feature_selection_max_corr),
        "max_features": int(config.feature_selection_max_features),
        "stability_iterations": int(config.feature_selection_stability_iterations),
        "min_stability_support": float(config.feature_selection_min_stability_support),
        "primary_label_horizon": int(config.primary_label_horizon),
        "label_horizons": label_horizons,
    }


def _build_snapshot_training_scope(config: "TrainingConfig") -> dict[str, Any]:
    """Return deterministic data/feature/label scope for snapshot auto-reuse."""
    symbols = sorted(
        {str(symbol).strip().upper() for symbol in config.symbols if str(symbol).strip()}
    )
    feature_groups = sorted(
        {str(group).strip().lower() for group in config.feature_groups if str(group).strip()}
    )
    timeframes = sorted(
        {str(timeframe).strip() for timeframe in config.timeframes if str(timeframe).strip()}
    )
    label_horizons = sorted(
        {
            int(h)
            for h in config.label_horizons
            if isinstance(h, (int, np.integer, float, np.floating)) and int(h) > 0
        }
    )
    return {
        "schema_version": SNAPSHOT_TRAINING_SCOPE_SCHEMA_VERSION,
        "symbols": symbols,
        "start_date": str(config.start_date or ""),
        "end_date": str(config.end_date or ""),
        "timeframe": str(config.timeframe),
        "timeframes": timeframes,
        "include_premarket": bool(config.include_premarket),
        "include_postmarket": bool(config.include_postmarket),
        "feature_groups": feature_groups,
        "training_bar_mode": str(config.training_bar_mode),
        "intrinsic_bar_type": str(config.intrinsic_bar_type),
        "intrinsic_bar_threshold": float(config.intrinsic_bar_threshold),
        "intrinsic_target_bars_per_day": int(config.intrinsic_target_bars_per_day),
        "enable_reference_features": bool(config.enable_reference_features),
        "reference_feature_sources": list(config.reference_feature_sources),
        "enable_tick_microstructure_features": bool(config.enable_tick_microstructure_features),
        "enable_cross_sectional": bool(config.enable_cross_sectional),
        "enable_symbol_quality_filter": bool(config.enable_symbol_quality_filter),
        "target_universe_size": int(config.target_universe_size),
        "universe_selection_buffer_size": int(config.universe_selection_buffer_size),
        "universe_reference_symbols": sorted(config.universe_reference_symbols),
        "symbol_quality_min_rows": int(config.symbol_quality_min_rows),
        "symbol_quality_min_symbols": int(config.symbol_quality_min_symbols),
        "symbol_quality_max_missing_ratio": float(config.symbol_quality_max_missing_ratio),
        "symbol_quality_max_extreme_move_ratio": float(
            config.symbol_quality_max_extreme_move_ratio
        ),
        "symbol_quality_max_corporate_action_ratio": float(
            config.symbol_quality_max_corporate_action_ratio
        ),
        "symbol_quality_min_median_dollar_volume": float(
            config.symbol_quality_min_median_dollar_volume
        ),
        "max_cross_sectional_symbols": int(config.max_cross_sectional_symbols),
        "max_cross_sectional_rows": int(config.max_cross_sectional_rows),
        "allow_feature_group_fallback": bool(config.allow_feature_group_fallback),
        "windows_force_fallback_features": bool(config.windows_force_fallback_features),
        "feature_set_id": str(config.feature_set_id),
        "adjust_prices_for_corporate_actions": bool(config.adjust_prices_for_corporate_actions),
        "data_max_abs_return": float(config.data_max_abs_return),
        "label_horizons": label_horizons,
        "primary_label_horizon": int(config.primary_label_horizon),
        "label_profit_taking_threshold": float(config.label_profit_taking_threshold),
        "label_stop_loss_threshold": float(config.label_stop_loss_threshold),
        "label_max_holding_period": int(config.label_max_holding_period),
        "label_spread_bps": float(config.label_spread_bps),
        "label_slippage_bps": float(config.label_slippage_bps),
        "label_impact_bps": float(config.label_impact_bps),
        "label_min_signal_abs_return_bps": float(config.label_min_signal_abs_return_bps),
        "label_neutral_buffer_bps": float(config.label_neutral_buffer_bps),
        "label_max_abs_forward_return": float(config.label_max_abs_forward_return),
        "label_signal_volatility_floor_mult": float(config.label_signal_volatility_floor_mult),
        "label_volatility_lookback": int(config.label_volatility_lookback),
        "label_regime_lookback": int(config.label_regime_lookback),
        "label_temporal_weight_decay": float(config.label_temporal_weight_decay),
        "label_edge_cost_buffer_bps": float(config.label_edge_cost_buffer_bps),
        "label_apply_uniqueness_weighting": bool(config.label_apply_uniqueness_weighting),
        "label_uniqueness_weight_floor": float(config.label_uniqueness_weight_floor),
        "label_apply_volatility_inverse_weighting": bool(
            config.label_apply_volatility_inverse_weighting
        ),
        "label_volatility_weight_cap": float(config.label_volatility_weight_cap),
        "feature_selection_signature": _build_snapshot_feature_selection_signature(config),
    }


_SNAPSHOT_SCOPE_SEQUENCE_KEYS = {
    "symbols",
    "feature_groups",
    "timeframes",
    "universe_reference_symbols",
    "label_horizons",
    "reference_feature_sources",
}
_SNAPSHOT_SCOPE_UPPER_KEYS = {"symbols", "universe_reference_symbols"}
_SNAPSHOT_SCOPE_LOWER_KEYS = {"feature_groups"}
_REFERENCE_FEATURE_EXACT_SOURCE_MAP: dict[str, tuple[str, ...]] = {
    "ref_days_since_dividend": ("corporate_actions",),
    "ref_days_since_split": ("corporate_actions",),
    "ref_last_dividend_amount": ("corporate_actions",),
    "ref_last_split_ratio": ("corporate_actions",),
    "ref_pe_ratio": ("fundamentals",),
    "ref_price_to_book": ("fundamentals",),
    "ref_dividend_per_share": ("fundamentals",),
    "ref_dividend_yield": ("fundamentals",),
    "ref_operating_margin_ttm": ("fundamentals",),
    "ref_profit_margin": ("fundamentals",),
    "ref_beta": ("fundamentals",),
    "ref_analyst_target_upside": ("fundamentals",),
    "ref_price_to_52w_high": ("fundamentals",),
    "ref_price_to_52w_low": ("fundamentals",),
    "ref_fundamental_days_since_snapshot": ("fundamentals",),
    "ref_days_since_earnings": ("earnings",),
    "ref_news_filing_sentiment_pressure": ("news", "sec_filings"),
    "ref_earnings_short_pressure": ("earnings", "short_sale"),
    "ref_ftd_short_pressure": ("ftd", "short_sale"),
    "ref_news_analyst_alignment": ("news", "fundamentals"),
    "ref_macro_news_stress": ("macro", "news"),
}
_REFERENCE_FEATURE_PREFIX_SOURCE_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("macro_", ("macro",)),
    ("ref_news_", ("news",)),
    ("ref_filing_", ("sec_filings",)),
    ("ref_short_", ("short_sale",)),
    ("ref_market_cap_", ("fundamentals",)),
    ("ref_shares_outstanding_", ("fundamentals",)),
    ("ref_revenue_ttm_", ("fundamentals",)),
    ("ref_earnings_", ("earnings",)),
    ("ref_ftd_", ("ftd",)),
    ("ref_corporate_action_", ("corporate_actions",)),
    ("ref_dividend_", ("corporate_actions",)),
    ("ref_split_", ("corporate_actions",)),
)


def _normalize_snapshot_scope_value(key: str, value: Any) -> Any:
    """Normalize snapshot scope values for bundle compatibility checks."""
    if key == "reference_feature_sources":
        from quant_trading_system.features.reference import normalize_reference_feature_sources

        try:
            return normalize_reference_feature_sources(value)
        except Exception:
            pass

    if key in _SNAPSHOT_SCOPE_SEQUENCE_KEYS:
        if value is None:
            return []
        raw_values = value if isinstance(value, (list, tuple, set)) else [value]
        normalized_values: list[Any] = []
        for raw_value in raw_values:
            if raw_value is None:
                continue
            if key == "label_horizons":
                try:
                    normalized_values.append(int(raw_value))
                except (TypeError, ValueError):
                    normalized_values.append(str(raw_value).strip())
                continue
            token = str(raw_value).strip()
            if not token:
                continue
            if key in _SNAPSHOT_SCOPE_UPPER_KEYS:
                token = token.upper()
            elif key in _SNAPSHOT_SCOPE_LOWER_KEYS:
                token = token.lower()
            normalized_values.append(token)
        return sorted(normalized_values)
    return value


def _infer_reference_feature_sources_from_names(
    feature_names: list[str] | tuple[str, ...] | None,
) -> tuple[set[str], list[str]]:
    """Infer reference-data sources from a bundle feature list."""
    inferred_sources: set[str] = set()
    unknown_reference_features: list[str] = []
    for raw_name in feature_names or []:
        feature_name = str(raw_name).strip()
        if not feature_name:
            continue
        normalized_name = feature_name.lower()
        mapped_sources = _REFERENCE_FEATURE_EXACT_SOURCE_MAP.get(normalized_name)
        if mapped_sources is not None:
            inferred_sources.update(mapped_sources)
            continue

        matched = False
        for prefix, sources in _REFERENCE_FEATURE_PREFIX_SOURCE_MAP:
            if normalized_name.startswith(prefix):
                inferred_sources.update(sources)
                matched = True
                break
        if matched:
            continue

        if normalized_name.startswith("macro_") or normalized_name.startswith("ref_"):
            unknown_reference_features.append(feature_name)
    return inferred_sources, sorted(set(unknown_reference_features))


def _validate_snapshot_bundle_manifest_scope(
    bundle_manifest: dict[str, Any],
    config: "TrainingConfig",
) -> list[str]:
    """Return auditable scope-mismatch reasons for a snapshot bundle manifest."""
    expected_scope = _build_snapshot_training_scope(config)
    actual_scope = bundle_manifest.get("training_scope", {})
    issues: list[str] = []

    if not isinstance(actual_scope, dict) or not actual_scope:
        issues.append("bundle manifest is missing a compatible training_scope block")
    else:
        for key in sorted(set(expected_scope).union(actual_scope)):
            expected_value = _normalize_snapshot_scope_value(key, expected_scope.get(key))
            actual_value = _normalize_snapshot_scope_value(key, actual_scope.get(key))
            if expected_value != actual_value:
                issues.append(f"{key}: expected={expected_value!r} actual={actual_value!r}")

    bundle_feature_names = bundle_manifest.get("feature_names", [])
    if isinstance(bundle_feature_names, list):
        bundle_sources, unknown_reference_features = _infer_reference_feature_sources_from_names(
            bundle_feature_names
        )
        if not bool(config.enable_reference_features):
            if bundle_sources or unknown_reference_features:
                preview = sorted(bundle_sources)[:4] or unknown_reference_features[:4]
                issues.append(
                    "reference features present while enable_reference_features=False "
                    f"(examples={preview!r})"
                )
        elif config.reference_feature_sources:
            allowed_sources = set(config.reference_feature_sources)
            unexpected_sources = sorted(bundle_sources.difference(allowed_sources))
            if unexpected_sources:
                issues.append(
                    "bundle feature list contains disallowed reference sources "
                    f"(unexpected={unexpected_sources!r})"
                )
            if unknown_reference_features:
                issues.append(
                    "bundle feature list contains unmapped reference features under a "
                    f"source-selective policy (examples={unknown_reference_features[:4]!r})"
                )

    return issues


def _find_reusable_snapshot_bundle(config: "TrainingConfig") -> Path | None:
    """Discover the newest dataset snapshot bundle that matches the current training scope."""
    snapshots_root = Path(config.output_dir) / "snapshots"
    if not snapshots_root.exists():
        return None

    expected_scope = _build_snapshot_training_scope(config)
    candidates: list[tuple[str, Path]] = []
    for manifest_path in snapshots_root.rglob("dataset_bundle.manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("training_scope") != expected_scope:
            continue
        if _validate_snapshot_bundle_manifest_scope(payload, config):
            continue
        created_at = str(payload.get("created_at") or "")
        candidates.append((created_at, manifest_path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


LIGHTGBM_FAMILY_MODELS = {"lightgbm", "lightgbm_ranker", "lightgbm_regressor"}
RANKING_MODELS = {"lightgbm_ranker"}
REGRESSION_MODELS = {"xgboost_regressor", "lightgbm_regressor"}
LIGHTGBM_MIN_DATA_IN_LEAF_SEARCH_MIN = 16
LIGHTGBM_MIN_DATA_IN_LEAF_SEARCH_MAX = 128
LIGHTGBM_MIN_DATA_IN_LEAF_FIT_MAX = 160
LIGHTGBM_DEAD_SCORE_STD_EPS = 1e-6
LIGHTGBM_DEAD_SCORE_SPAN_EPS = 1e-5

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

CORE_TRAINING_SCHEMA_CONTRACT: dict[str, set[str]] = {
    "ohlcv_bars": {"symbol", "timestamp", "timeframe", "open", "high", "low", "close", "volume"},
    "features": {"symbol", "timestamp", "timeframe", "feature_name", "feature_set_id", "value"},
}
REFERENCE_TRAINING_SCHEMA_CONTRACT: dict[str, set[str]] = {
    "corporate_actions": {"symbol", "action_type", "ex_date", "amount", "split_from", "split_to"},
    "fundamental_snapshots": {
        "symbol",
        "as_of_date",
        "created_at",
        "market_cap",
        "shares_outstanding",
        "pe_ratio",
        "price_to_book",
        "dividend_per_share",
        "dividend_yield",
        "revenue_ttm",
        "operating_margin_ttm",
        "profit_margin",
        "beta",
        "week_52_high",
        "week_52_low",
        "analyst_target_price",
    },
    "earnings_events": {
        "symbol",
        "fiscal_date_ending",
        "reported_date",
        "announcement_timestamp",
        "availability_timestamp",
        "first_seen_at",
        "created_at",
        "reported_eps",
        "estimated_eps",
        "surprise",
        "surprise_pct",
    },
    "sec_filings": {"symbol", "form", "accepted_at", "filed_date"},
    "macro_observations": {"series_id", "observation_date", "value"},
    "macro_vintage_observations": {"series_id", "observation_date", "realtime_start", "value"},
    "news_articles": {"article_id", "created_at_source", "symbols", "sentiment"},
    "short_sale_volumes": {
        "symbol",
        "trade_date",
        "short_volume",
        "short_exempt_volume",
        "total_volume",
    },
    "fails_to_deliver": {"symbol", "settlement_date", "quantity", "price", "ftd_metadata"},
}
REFERENCE_SOURCE_SCHEMA_TABLES: dict[str, tuple[str, ...]] = {
    "macro": ("macro_observations", "macro_vintage_observations"),
    "short_sale": ("short_sale_volumes",),
    "news": ("news_articles",),
    "sec_filings": ("sec_filings",),
    "fundamentals": ("fundamental_snapshots",),
    "earnings": ("earnings_events",),
    "ftd": ("fails_to_deliver",),
    "corporate_actions": ("corporate_actions",),
}


def _required_training_schema_contract(
    config: "TrainingConfig | None" = None,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    required = {table: set(columns) for table, columns in CORE_TRAINING_SCHEMA_CONTRACT.items()}
    optional: dict[str, set[str]] = {}

    if config is None:
        return required, optional

    if bool(config.adjust_prices_for_corporate_actions):
        required["corporate_actions"] = set(REFERENCE_TRAINING_SCHEMA_CONTRACT["corporate_actions"])

    if bool(config.enable_reference_features):
        active_sources = list(
            config.reference_feature_sources or REFERENCE_SOURCE_SCHEMA_TABLES.keys()
        )
        reference_contract: dict[str, set[str]] = {}
        for source_name in active_sources:
            for table_name in REFERENCE_SOURCE_SCHEMA_TABLES.get(str(source_name), ()):
                reference_contract[table_name] = set(REFERENCE_TRAINING_SCHEMA_CONTRACT[table_name])
        if bool(config.allow_feature_group_fallback):
            optional.update(reference_contract)
        else:
            required.update(reference_contract)

    return required, optional


def _compute_feature_pipeline_fingerprint() -> str:
    """Create deterministic fingerprint for feature-engineering code."""
    fingerprint = hashlib.sha256()
    candidate_files = [
        PROJECT_ROOT / "quant_trading_system" / "features" / "feature_pipeline.py",
        PROJECT_ROOT / "quant_trading_system" / "features" / "optimized_pipeline.py",
        PROJECT_ROOT / "quant_trading_system" / "features" / "cross_sectional.py",
        PROJECT_ROOT / "quant_trading_system" / "features" / "reference.py",
        PROJECT_ROOT / "quant_trading_system" / "features" / "multi_timeframe.py",
        PROJECT_ROOT / "quant_trading_system" / "features" / "tick_microstructure.py",
        PROJECT_ROOT / "quant_trading_system" / "data" / "training_bars.py",
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

    manifest = (
        payload.get("replay_manifest")
        if isinstance(payload.get("replay_manifest"), dict)
        else payload
    )
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


def _load_symbols_file(symbols_file: Path) -> list[str]:
    """Load symbols from newline/comma separated text or JSON payloads."""
    path = Path(symbols_file)
    raw_text = path.read_text(encoding="utf-8")
    stripped = raw_text.strip()
    if not stripped:
        return []

    payload: Any | None = None
    if stripped[0] in "[{":
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None

    candidates: list[Any]
    if isinstance(payload, dict):
        candidates = payload.get("symbols", [])
    elif isinstance(payload, list):
        candidates = payload
    else:
        candidates = re.split(r"[\s,;]+", raw_text)

    symbols: list[str] = []
    for raw_symbol in candidates:
        normalized = str(raw_symbol).strip().upper()
        if normalized and normalized not in symbols:
            symbols.append(normalized)
    return symbols


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

    if model_type in {
        "xgboost",
        "lightgbm",
        "lightgbm_ranker",
        "xgboost_regressor",
        "lightgbm_regressor",
        "random_forest",
        "elastic_net",
    }:
        section = all_cfg.get(model_type, {})
        if not section and model_type == "lightgbm_ranker":
            legacy_section = all_cfg.get("lightgbm", {})
            legacy_ranker = (
                legacy_section.get("ranker", {}) if isinstance(legacy_section, dict) else {}
            )
            section = {"ranker": legacy_ranker} if legacy_ranker else {}
        if not section and model_type == "xgboost_regressor":
            section = all_cfg.get("xgboost", {})
        if not section and model_type == "lightgbm_regressor":
            section = all_cfg.get("lightgbm", {})
        subsection_name = "classifier"
        if model_type.endswith("_regressor"):
            subsection_name = "regressor"
        elif model_type == "lightgbm_ranker":
            subsection_name = "ranker"
        if model_type == "lightgbm_ranker":
            params = dict(section.get("ranker", {}))
        else:
            params = dict(section.get(subsection_name, section.get("classifier", {})))
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

    if model_type in {"xgboost", "xgboost_regressor"} and use_gpu:
        params["tree_method"] = "hist"
        params["device"] = "cuda"
    elif model_type in LIGHTGBM_FAMILY_MODELS and use_gpu:
        params["device"] = "cuda"

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


def _verify_institutional_infra(config: "TrainingConfig | None" = None) -> None:
    """Fail-fast infrastructure check for institutional training mode."""
    from quant_trading_system.database.connection import get_db_manager, get_redis_manager

    db_manager = get_db_manager()
    if not db_manager.health_check():
        raise RuntimeError(
            "PostgreSQL health check failed. Institutional training requires PostgreSQL."
        )

    redis_manager = get_redis_manager()
    if not redis_manager.health_check():
        raise RuntimeError("Redis health check failed. Institutional training requires Redis.")

    try:
        inspector = sa_inspect(db_manager.engine)
        available_tables = set(inspector.get_table_names())
    except Exception as exc:
        raise RuntimeError(f"Failed to inspect PostgreSQL schema: {exc}") from exc

    required_contract, optional_contract = _required_training_schema_contract(config)

    missing_tables = sorted(set(required_contract).difference(available_tables))
    if missing_tables:
        raise RuntimeError(
            "PostgreSQL schema validation failed. Missing required tables: "
            f"{', '.join(missing_tables)}."
        )

    missing_columns_by_table: dict[str, list[str]] = {}
    for table_name, required_columns in required_contract.items():
        try:
            available_columns = {
                str(column.get("name"))
                for column in inspector.get_columns(table_name)
                if column.get("name") is not None
            }
        except Exception as exc:
            raise RuntimeError(
                f"Failed to inspect PostgreSQL columns for {table_name}: {exc}"
            ) from exc
        missing_columns = sorted(required_columns.difference(available_columns))
        if missing_columns:
            missing_columns_by_table[table_name] = missing_columns

    if missing_columns_by_table:
        detail = "; ".join(
            f"{table_name} missing [{', '.join(columns)}]"
            for table_name, columns in sorted(missing_columns_by_table.items())
        )
        raise RuntimeError(
            "PostgreSQL schema validation failed. Missing required columns: " f"{detail}."
        )

    missing_optional_tables = sorted(set(optional_contract).difference(available_tables))
    if missing_optional_tables:
        logger.warning(
            "Optional reference tables unavailable; training will rely on feature-group fallback: %s",
            ", ".join(missing_optional_tables),
        )

    optional_column_gaps: dict[str, list[str]] = {}
    for table_name, optional_columns in optional_contract.items():
        if table_name not in available_tables:
            continue
        try:
            available_columns = {
                str(column.get("name"))
                for column in inspector.get_columns(table_name)
                if column.get("name") is not None
            }
        except Exception as exc:
            raise RuntimeError(
                f"Failed to inspect PostgreSQL columns for optional table {table_name}: {exc}"
            ) from exc
        missing_columns = sorted(optional_columns.difference(available_columns))
        if missing_columns:
            optional_column_gaps[table_name] = missing_columns

    if optional_column_gaps:
        logger.warning(
            "Optional reference columns unavailable; training will rely on feature-group fallback: %s",
            "; ".join(
                f"{table_name} missing [{', '.join(columns)}]"
                for table_name, columns in sorted(optional_column_gaps.items())
            ),
        )

    try:
        with db_manager.session() as session:
            has_ohlcv_data = (
                session.execute(text("SELECT 1 FROM ohlcv_bars LIMIT 1")).scalar() is not None
            )
    except Exception as exc:
        raise RuntimeError(
            f"PostgreSQL data validation failed while querying ohlcv_bars: {exc}"
        ) from exc

    if not has_ohlcv_data:
        raise RuntimeError(
            "PostgreSQL ohlcv_bars table is empty. Institutional training requires market data."
        )


def _verify_gpu_stack(model_list: list[str]) -> bool:
    """Return whether GPU acceleration is usable for the requested sweep."""
    deep_models = {"lstm", "transformer", "tcn"}
    requires_torch = bool(deep_models.intersection(model_list))
    torch_cuda_available = False

    try:
        import torch
    except ImportError as exc:
        if requires_torch:
            raise RuntimeError("PyTorch is required to train deep models.") from exc
        torch = None
        logger.info("PyTorch not installed; probing non-deep GPU backends directly.")

    if torch is not None:
        torch_cuda_available = bool(torch.cuda.is_available())
        if torch_cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA GPU detected: {gpu_name}")

            # Validate tensor ops on device for deep learning stack.
            if requires_torch:
                _ = torch.randn((16, 8), device="cuda")
        elif requires_torch:
            raise RuntimeError("CUDA GPU not detected. Deep learning models require PyTorch CUDA.")
        else:
            logger.info("PyTorch CUDA not available; probing non-deep GPU backends directly.")

    gpu_backend_available = False

    if {"xgboost", "xgboost_regressor", "ensemble"}.intersection(model_list):
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
            gpu_backend_available = True
        except Exception as exc:
            logger.warning(
                "XGBoost GPU backend unavailable; falling back to CPU parameters: %s",
                exc,
            )
            return False

    if {"lightgbm", "lightgbm_ranker", "lightgbm_regressor", "ensemble"}.intersection(model_list):
        try:
            import lightgbm as lgb

            X = np.random.rand(128, 8)
            y = np.random.randint(0, 2, 128)
            probe = lgb.LGBMClassifier(
                n_estimators=2,
                max_depth=3,
                device="cuda",
                verbosity=-1,
            )
            probe.fit(X, y)
            gpu_backend_available = True
        except Exception as exc:
            logger.warning(
                "LightGBM GPU backend unavailable; falling back to CPU parameters: %s",
                exc,
            )
            return False

    if requires_torch:
        return True

    return gpu_backend_available or torch_cuda_available


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    # Model selection
    model_type: str = (
        "xgboost"  # xgboost, lightgbm, lightgbm_ranker, xgboost_regressor, lightgbm_regressor, random_forest, elastic_net, lstm, transformer, tcn, ensemble
    )
    model_name: str = ""  # Custom name for the model
    training_profile: str = "promotion"

    # Data configuration
    symbols: list[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    timeframe: str = DEFAULT_TIMEFRAME
    timeframes: list[str] = field(default_factory=list)
    include_premarket: bool = False
    include_postmarket: bool = False
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
    optuna_storage_dir: str = ""
    optuna_resume_enabled: bool = True
    seed: int = 42
    use_nested_walk_forward: bool = True
    nested_outer_splits: int = 4
    nested_inner_splits: int = 3
    require_nested_trace_for_promotion: bool = True
    require_objective_breakdown_for_promotion: bool = True
    objective_weight_sharpe: float = 1.0
    objective_weight_drawdown: float = 0.5
    objective_weight_turnover: float = 0.2
    objective_weight_calibration: float = 0.35
    objective_weight_trade_activity: float = 1.0
    objective_weight_cvar: float = 0.4
    objective_weight_skew: float = 0.1
    objective_skew_penalty_cap: float = 1.5
    objective_weight_tail_risk: float = 0.35
    objective_weight_symbol_concentration: float = 0.20
    objective_expected_shortfall_cap: float = 0.012
    nested_outer_stability_ratio_cap: float = 1.25
    nested_outer_stability_min_trials: int = 8
    allow_unstable_outer_fold_fallback: bool = False

    # Model-specific parameters
    model_params: dict = field(default_factory=dict)

    # Training parameters
    epochs: int = 100  # For deep learning
    batch_size: int = 64
    learning_rate: float = 0.001
    early_stopping_rounds: int = 10
    primary_horizon_sweep: list[int] = field(default_factory=list)

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
    min_white_reality_stat: float = 0.0
    max_white_reality_pvalue: float = 0.10
    min_holdout_sharpe: float = 0.0
    min_holdout_regime_sharpe: float = -0.10
    max_holdout_drawdown: float = 0.35
    max_regime_shift: float = 0.35
    max_symbol_concentration_hhi: float = 0.65
    min_holdout_symbol_coverage: float = 0.60
    min_holdout_symbol_p25_sharpe: float = -0.10
    max_holdout_symbol_underwater_ratio: float = 0.55

    # Target engineering (Workstream B)
    label_horizons: list[int] = field(default_factory=lambda: [1, 5, 20])
    primary_label_horizon: int = 5
    label_profit_taking_threshold: float = 0.015
    label_stop_loss_threshold: float = 0.010
    label_max_holding_period: int = 20
    label_spread_bps: float = 1.0
    label_slippage_bps: float = 3.0
    label_impact_bps: float = 2.0
    label_min_signal_abs_return_bps: float = 10.0
    label_neutral_buffer_bps: float = 5.0
    label_max_abs_forward_return: float = 0.35
    label_signal_volatility_floor_mult: float = 0.50
    label_volatility_lookback: int = 20
    label_regime_lookback: int = 30
    label_temporal_weight_decay: float = 0.999
    label_edge_cost_buffer_bps: float = 3.0

    # Meta-labeling (P2-3.5)
    use_meta_labeling: bool = True
    meta_model_type: str = "xgboost"
    meta_label_min_confidence: float = 0.55
    meta_label_dynamic_threshold: bool = True
    enable_probability_calibration: bool = True
    enable_ranker_probability_calibration: bool = False
    probability_calibration_method: str = "isotonic"
    probability_calibration_guardrail_enabled: bool = True
    probability_calibration_min_dispersion_ratio: float = 0.05
    probability_calibration_min_oof_active_rate: float = 0.01
    probability_calibration_min_oof_active_rate_ratio: float = 0.20
    probability_calibration_max_near_mid_rate: float = 0.98
    enable_expected_edge_policy: bool = True
    expected_edge_min_samples: int = 120
    expected_edge_min_coverage: float = 0.55
    expected_edge_max_context_features: int = 24
    expected_edge_min_pass_probability: float = 0.53
    expected_edge_min_expected_edge: float = 0.0
    expected_edge_min_signal_scale: float = 0.25
    expected_edge_max_signal_scale: float = 1.10
    expected_edge_use_symbol_priors: bool = False

    # Multiple testing correction (P1-3.1)
    apply_multiple_testing: bool = True
    correction_method: str = "deflated_sharpe"  # bonferroni, bh, deflated_sharpe

    # GPU acceleration
    use_gpu: bool = True
    require_gpu: bool = False
    require_feature_gpu: bool = False
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
    snapshot_only: bool = False
    dataset_snapshot_bundle_path: str = ""
    strict_snapshot_replay: bool = False
    auto_snapshot_reuse_enabled: bool = True
    quality_missing_bars_threshold: float = 0.01
    quality_duplicate_bars_threshold: float = 0.001
    quality_extreme_move_threshold: float = 0.01
    quality_corporate_action_jump_threshold: float = 0.001
    enable_symbol_quality_filter: bool = True
    target_universe_size: int = 0
    universe_selection_buffer_size: int = 24
    universe_reference_symbols: list[str] = field(
        default_factory=lambda: ["SPY", "QQQ", "IWM", "VIX"]
    )
    symbol_quality_min_rows: int = 1200
    symbol_quality_min_symbols: int = 8
    symbol_quality_max_missing_ratio: float = 0.12
    symbol_quality_max_extreme_move_ratio: float = 0.08
    symbol_quality_max_corporate_action_ratio: float = 0.02
    symbol_quality_min_median_dollar_volume: float = 1_000_000.0
    data_max_abs_return: float = 0.35
    feature_groups: list[str] = field(
        default_factory=lambda: ["technical", "statistical", "microstructure", "cross_sectional"]
    )
    training_bar_mode: str = "time"
    intrinsic_bar_type: str = "volume"
    intrinsic_bar_threshold: float = 0.0
    intrinsic_target_bars_per_day: int = 100
    enable_reference_features: bool = True
    reference_feature_sources: list[str] = field(default_factory=list)
    adjust_prices_for_corporate_actions: bool = True
    enable_tick_microstructure_features: bool = True
    enable_cross_sectional: bool = True
    cross_sectional_user_locked: bool = False
    max_cross_sectional_symbols: int = 20
    max_cross_sectional_rows: int = 500000
    allow_feature_group_fallback: bool = False
    feature_materialization_batch_rows: int = 5000
    feature_reuse_min_coverage: float = 0.20
    persist_features_to_postgres: bool = True
    feature_set_id: str = "default"
    enable_feature_selection: bool = True
    feature_selection_min_ic: float = 0.015
    feature_selection_max_corr: float = 0.90
    feature_selection_max_features: int = 180
    feature_selection_stability_iterations: int = 16
    feature_selection_min_stability_support: float = 0.60
    windows_force_fallback_features: bool = False
    holdout_pct: float = 0.15
    dynamic_no_trade_band: bool = True
    execution_vol_target_daily: float = 0.012
    execution_turnover_cap: float = 0.60
    execution_cooldown_bars: int = 2
    execution_max_symbol_entry_share: float = 0.68
    min_confidence_position_scale: float = 0.20
    lightgbm_use_monotonic_constraints: bool = True
    auto_live_profile_enabled: bool = True
    auto_live_profile_symbol_threshold: int = 40
    auto_live_profile_min_years: float = 4.0
    warm_start_model_path: str = ""
    label_apply_uniqueness_weighting: bool = True
    label_uniqueness_weight_floor: float = 0.25
    label_apply_volatility_inverse_weighting: bool = True
    label_volatility_weight_cap: float = 2.5

    def __post_init__(self):
        from quant_trading_system.features.multi_timeframe import normalize_timeframes
        from quant_trading_system.features.reference import normalize_reference_feature_sources

        if not self.model_name:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.model_name = f"{self.model_type}_{timestamp}"
        self.training_profile = _normalize_training_profile(self.training_profile)
        self.feature_materialization_batch_rows = max(
            250, int(self.feature_materialization_batch_rows)
        )
        self.max_cross_sectional_symbols = max(2, int(self.max_cross_sectional_symbols))
        self.max_cross_sectional_rows = max(10_000, int(self.max_cross_sectional_rows))
        self.feature_reuse_min_coverage = min(
            max(float(self.feature_reuse_min_coverage), 0.01), 0.95
        )
        self.optuna_storage_dir = str(self.optuna_storage_dir or "").strip()
        self.optuna_resume_enabled = bool(self.optuna_resume_enabled)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.timeframes = normalize_timeframes(self.timeframe, self.timeframes)
        self.include_premarket = bool(self.include_premarket)
        self.include_postmarket = bool(self.include_postmarket)
        self.use_gpu = bool(self.use_gpu)
        self.require_gpu = bool(self.require_gpu)
        self.require_feature_gpu = bool(self.require_feature_gpu)
        if self.require_gpu and not self.use_gpu:
            raise ValueError("require_gpu=True requires use_gpu=True.")
        if self.require_feature_gpu and not self.use_gpu:
            raise ValueError("require_feature_gpu=True requires use_gpu=True.")
        self.cross_sectional_user_locked = bool(self.cross_sectional_user_locked)
        self.feature_set_id = str(self.feature_set_id or "default").strip() or "default"
        self.snapshot_only = bool(self.snapshot_only)
        self.dataset_snapshot_bundle_path = str(self.dataset_snapshot_bundle_path or "").strip()
        self.strict_snapshot_replay = bool(self.strict_snapshot_replay)
        self.auto_snapshot_reuse_enabled = bool(self.auto_snapshot_reuse_enabled)
        self.data_max_abs_return = max(float(self.data_max_abs_return), 0.05)
        self.min_trades = max(1, int(self.min_trades))
        self.objective_weight_trade_activity = max(
            0.0,
            float(self.objective_weight_trade_activity),
        )
        self.objective_weight_cvar = max(0.0, float(self.objective_weight_cvar))
        self.objective_weight_skew = max(0.0, float(self.objective_weight_skew))
        self.objective_skew_penalty_cap = max(0.0, float(self.objective_skew_penalty_cap))
        self.objective_weight_tail_risk = max(0.0, float(self.objective_weight_tail_risk))
        self.objective_weight_symbol_concentration = max(
            0.0,
            float(self.objective_weight_symbol_concentration),
        )
        self.objective_expected_shortfall_cap = max(
            1e-5,
            float(self.objective_expected_shortfall_cap),
        )
        self.nested_outer_stability_ratio_cap = max(
            0.1,
            float(self.nested_outer_stability_ratio_cap),
        )
        self.nested_outer_stability_min_trials = max(3, int(self.nested_outer_stability_min_trials))
        self.allow_unstable_outer_fold_fallback = bool(self.allow_unstable_outer_fold_fallback)
        self.primary_horizon_sweep = sorted(
            {
                int(h)
                for h in self.primary_horizon_sweep
                if isinstance(h, (int, float, np.integer, np.floating)) and int(h) > 0
            }
        )
        self.meta_label_min_confidence = float(
            np.clip(float(self.meta_label_min_confidence), 0.45, 0.95)
        )
        self.meta_label_dynamic_threshold = bool(self.meta_label_dynamic_threshold)
        self.enable_probability_calibration = bool(self.enable_probability_calibration)
        self.enable_ranker_probability_calibration = bool(
            self.enable_ranker_probability_calibration
        )
        calibration_method = str(self.probability_calibration_method or "isotonic").strip().lower()
        if calibration_method not in {"isotonic", "sigmoid"}:
            raise ValueError("probability_calibration_method must be 'isotonic' or 'sigmoid'")
        self.probability_calibration_method = calibration_method
        self.probability_calibration_guardrail_enabled = bool(
            self.probability_calibration_guardrail_enabled
        )
        self.probability_calibration_min_dispersion_ratio = float(
            np.clip(float(self.probability_calibration_min_dispersion_ratio), 0.0, 1.0)
        )
        self.probability_calibration_min_oof_active_rate = float(
            np.clip(float(self.probability_calibration_min_oof_active_rate), 0.0, 1.0)
        )
        self.probability_calibration_min_oof_active_rate_ratio = float(
            np.clip(float(self.probability_calibration_min_oof_active_rate_ratio), 0.0, 1.0)
        )
        self.probability_calibration_max_near_mid_rate = float(
            np.clip(float(self.probability_calibration_max_near_mid_rate), 0.50, 1.0)
        )
        self.enable_expected_edge_policy = bool(self.enable_expected_edge_policy)
        self.expected_edge_min_samples = max(32, int(self.expected_edge_min_samples))
        self.expected_edge_min_coverage = float(
            np.clip(float(self.expected_edge_min_coverage), 0.10, 1.0)
        )
        self.expected_edge_max_context_features = max(
            4, int(self.expected_edge_max_context_features)
        )
        self.expected_edge_min_pass_probability = float(
            np.clip(float(self.expected_edge_min_pass_probability), 0.45, 0.90)
        )
        self.expected_edge_min_expected_edge = float(self.expected_edge_min_expected_edge)
        self.expected_edge_min_signal_scale = float(
            np.clip(float(self.expected_edge_min_signal_scale), 0.0, 1.0)
        )
        self.expected_edge_max_signal_scale = float(
            np.clip(
                float(
                    max(
                        self.expected_edge_min_signal_scale,
                        float(self.expected_edge_max_signal_scale),
                    )
                ),
                self.expected_edge_min_signal_scale,
                1.25,
            )
        )
        self.expected_edge_use_symbol_priors = bool(self.expected_edge_use_symbol_priors)
        self.holdout_pct = min(max(float(self.holdout_pct), 0.05), 0.35)
        self.execution_vol_target_daily = max(float(self.execution_vol_target_daily), 1e-4)
        self.execution_turnover_cap = min(max(float(self.execution_turnover_cap), 0.05), 1.0)
        self.execution_cooldown_bars = max(0, int(self.execution_cooldown_bars))
        self.execution_max_symbol_entry_share = float(
            np.clip(float(self.execution_max_symbol_entry_share), 0.50, 0.95)
        )
        if self.training_profile == "promotion" and self.allow_unstable_outer_fold_fallback:
            raise ValueError(
                "Promotion profile cannot allow unstable outer-fold fallback candidates."
            )
        self.min_confidence_position_scale = float(
            np.clip(float(self.min_confidence_position_scale), 0.0, 1.0)
        )
        self.max_regime_shift = min(max(float(self.max_regime_shift), 0.0), 1.0)
        self.max_symbol_concentration_hhi = min(
            max(float(self.max_symbol_concentration_hhi), 0.10),
            1.0,
        )
        self.min_holdout_symbol_coverage = float(
            np.clip(float(self.min_holdout_symbol_coverage), 0.0, 1.0)
        )
        self.min_holdout_symbol_p25_sharpe = float(
            np.clip(float(self.min_holdout_symbol_p25_sharpe), -3.0, 3.0)
        )
        self.max_holdout_symbol_underwater_ratio = float(
            np.clip(float(self.max_holdout_symbol_underwater_ratio), 0.0, 1.0)
        )
        self.min_holdout_regime_sharpe = float(
            np.clip(float(self.min_holdout_regime_sharpe), -3.0, 3.0)
        )
        self.auto_live_profile_enabled = bool(self.auto_live_profile_enabled)
        self.auto_live_profile_symbol_threshold = max(
            1, int(self.auto_live_profile_symbol_threshold)
        )
        self.auto_live_profile_min_years = max(0.5, float(self.auto_live_profile_min_years))
        self.enable_symbol_quality_filter = bool(self.enable_symbol_quality_filter)
        self.target_universe_size = max(0, int(self.target_universe_size))
        self.universe_selection_buffer_size = max(0, int(self.universe_selection_buffer_size))
        self.universe_reference_symbols = sorted(
            {
                str(symbol).strip().upper()
                for symbol in self.universe_reference_symbols
                if str(symbol).strip()
            }
        )
        self.symbol_quality_min_rows = max(200, int(self.symbol_quality_min_rows))
        self.symbol_quality_min_symbols = max(1, int(self.symbol_quality_min_symbols))
        self.symbol_quality_max_missing_ratio = float(
            np.clip(float(self.symbol_quality_max_missing_ratio), 0.0, 0.95)
        )
        self.symbol_quality_max_extreme_move_ratio = float(
            np.clip(float(self.symbol_quality_max_extreme_move_ratio), 0.0, 0.95)
        )
        self.symbol_quality_max_corporate_action_ratio = float(
            np.clip(float(self.symbol_quality_max_corporate_action_ratio), 0.0, 0.95)
        )
        self.symbol_quality_min_median_dollar_volume = max(
            0.0,
            float(self.symbol_quality_min_median_dollar_volume),
        )
        self.training_bar_mode = str(self.training_bar_mode or "time").strip().lower()
        if self.training_bar_mode not in {"time", "intrinsic"}:
            raise ValueError("training_bar_mode must be either 'time' or 'intrinsic'")
        self.intrinsic_bar_type = str(self.intrinsic_bar_type or "volume").strip().lower()
        self.intrinsic_bar_threshold = max(0.0, float(self.intrinsic_bar_threshold))
        self.intrinsic_target_bars_per_day = max(10, int(self.intrinsic_target_bars_per_day))
        self.feature_groups = [
            str(g).strip().lower() for g in self.feature_groups if str(g).strip()
        ]
        if not self.feature_groups:
            self.feature_groups = ["technical", "statistical", "microstructure", "cross_sectional"]
        self.enable_reference_features = bool(self.enable_reference_features)
        normalized_reference_sources = normalize_reference_feature_sources(
            self.reference_feature_sources
            if self.reference_feature_sources
            else (["all"] if self.enable_reference_features else [])
        )
        self.reference_feature_sources = (
            list(normalized_reference_sources) if self.enable_reference_features else []
        )
        self.enable_tick_microstructure_features = bool(self.enable_tick_microstructure_features)
        self.adjust_prices_for_corporate_actions = bool(self.adjust_prices_for_corporate_actions)
        self.enable_feature_selection = bool(self.enable_feature_selection)
        self.feature_selection_min_ic = max(0.0, float(self.feature_selection_min_ic))
        self.feature_selection_max_corr = float(
            np.clip(float(self.feature_selection_max_corr), 0.50, 0.999)
        )
        self.feature_selection_max_features = max(10, int(self.feature_selection_max_features))
        self.feature_selection_stability_iterations = max(
            1, int(self.feature_selection_stability_iterations)
        )
        self.feature_selection_min_stability_support = float(
            np.clip(float(self.feature_selection_min_stability_support), 0.0, 1.0)
        )
        self.warm_start_model_path = str(self.warm_start_model_path or "").strip()
        self.label_apply_uniqueness_weighting = bool(self.label_apply_uniqueness_weighting)
        self.label_uniqueness_weight_floor = max(0.01, float(self.label_uniqueness_weight_floor))
        self.label_apply_volatility_inverse_weighting = bool(
            self.label_apply_volatility_inverse_weighting
        )
        self.label_volatility_weight_cap = max(1.0, float(self.label_volatility_weight_cap))

    def to_trading_cost_model(self) -> TradingCostModel:
        """Build canonical execution cost model from training configuration."""
        return TradingCostModel(
            spread_bps=float(self.label_spread_bps),
            slippage_bps=float(self.label_slippage_bps),
            impact_bps=float(self.label_impact_bps),
        )


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
        self.ohlcv_panels_by_timeframe: dict[str, pd.DataFrame] = {}
        self.features: pd.DataFrame | None = None
        self.labels: pd.Series | None = None
        self.feature_names: list[str] = []
        self.timestamps: np.ndarray | None = None
        self.row_symbols: np.ndarray | None = None
        self.regimes: np.ndarray | None = None
        self.primary_event_directions: np.ndarray | None = None
        self.close_prices: pd.Series | None = None
        self.primary_forward_returns: np.ndarray | None = None
        self.cost_aware_event_returns: np.ndarray | None = None
        self.holdout_features: pd.DataFrame | None = None
        self.holdout_labels: pd.Series | None = None
        self.holdout_close_prices: pd.Series | None = None
        self.holdout_timestamps: np.ndarray | None = None
        self.holdout_symbols: np.ndarray | None = None
        self.holdout_regimes: np.ndarray | None = None
        self.holdout_primary_event_directions: np.ndarray | None = None
        self.holdout_primary_forward_returns: np.ndarray | None = None
        self.holdout_cost_aware_event_returns: np.ndarray | None = None
        self.holdout_sample_weights: np.ndarray | None = None
        self.development_frame: pd.DataFrame | None = None
        self.holdout_frame: pd.DataFrame | None = None
        self.model: Any = None
        self.cv_results: list[dict] = []
        self.validation_results: dict = {}
        self.shap_values: np.ndarray | None = None
        self.meta_model: Any = None
        self.expected_edge_model: Any = None
        self.probability_calibrator: Any = None
        self.probability_calibration_method_resolved: str | None = None
        self.oof_primary_proba: np.ndarray | None = None
        self.oof_primary_proba_raw: np.ndarray | None = None
        self.cv_return_series: list[np.ndarray] = []
        self.cv_active_return_series: list[np.ndarray] = []
        self.expected_edge_cv_return_series: list[np.ndarray] = []
        self.expected_edge_cv_active_return_series: list[np.ndarray] = []
        self.sample_weights: np.ndarray | None = None
        self.label_diagnostics: dict[str, Any] = {}
        self.data_quality_report: dict[str, Any] | None = None
        self.data_quality_report_hash: str | None = None
        self.snapshot_manifest: dict[str, Any] | None = None
        self.snapshot_manifest_path: Path | None = None
        self.data_quality_report_path: Path | None = None
        self.dataset_snapshot_bundle_manifest: dict[str, Any] | None = None
        self.dataset_snapshot_bundle_manifest_path: Path | None = None
        self.cached_cv_splits: list[tuple[np.ndarray, np.ndarray]] = []
        self.snapshot_replay_loaded: bool = False
        self.nested_cv_trace: list[dict[str, Any]] = []
        self.snapshot_review_path: Path | None = None
        self.replay_manifest_path: Path | None = None
        self.promotion_package_path: Path | None = None
        self.artifacts_path: Path | None = None
        self.run_id: str = str(self.config.model_name)
        self.run_event_index_path: Path | None = None
        self._warm_start_model_cache: Any | None = None
        self._feature_pipeline_cache: dict[tuple[str, bool], Any] = {}
        self.feature_cache_reused: bool = False
        self.features_materialized_in_run: bool = False
        self.features_persisted_to_postgres: bool = False

        # Metrics
        provenance = _resolve_training_provenance(PROJECT_ROOT)
        self.training_metrics: dict = {
            "training_profile": self.config.training_profile,
            "run_id": self.run_id,
            "snapshot_only": bool(self.config.snapshot_only),
            "training_provenance": provenance,
            "git_commit": provenance.get("git_commit"),
            "git_commit_hash": provenance.get("git_commit_hash"),
            "git_branch": provenance.get("git_branch"),
            "git_dirty": provenance.get("git_dirty"),
            "environment_lock_hash": provenance.get("environment_lock_hash"),
            "dependency_lock_hash": provenance.get("dependency_lock_hash"),
            "host": provenance.get("host"),
            "training_host": provenance.get("host"),
            "wsl_distro": provenance.get("wsl_distro"),
            "training_wsl_distro": provenance.get("wsl_distro"),
            "training_repo_provenance_ready": float(
                bool(provenance.get("git_repo_available", False))
            ),
        }
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
        self._record_training_run_event("started")

        try:
            loaded_snapshot = False
            if self.config.dataset_snapshot_bundle_path:
                loaded_snapshot = self._load_training_dataset_snapshot()

            if not loaded_snapshot:
                # Phase 1: Load Data
                self._load_data()

                if self.config.snapshot_only:
                    preflight_review = self._build_snapshot_review()
                    if not bool(preflight_review.get("ready", False)):
                        failed_checks = list(preflight_review.get("failed_checks", []))
                        self.logger.warning(
                            "Snapshot-only preflight failed after Phase 1; "
                            "skipping feature materialization. Failed checks: %s",
                            ", ".join(failed_checks) if failed_checks else "unknown",
                        )
                        output_dir = Path(self.config.output_dir)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        review_payload = self._persist_snapshot_review_only(output_dir)
                        duration = (datetime.now(timezone.utc) - self.start_time).total_seconds()
                        results = self._build_snapshot_only_results(review_payload, duration)
                        self._record_training_run_event("completed", results=results)
                        self.logger.info("Snapshot-only preflight completed in %.1fs", duration)
                        self.logger.info("Snapshot review readiness: FAILED")
                        return results

                # Phase 2: Compute Features
                self._compute_features()

                # Phase 3: Create Labels
                self._create_labels()

            # Phase 3.25: Feature selection on development set only
            self._apply_feature_selection()

            # Phase 3.3: Persist the post-selection feature cache for deterministic reuse.
            self._persist_features_to_postgres_if_needed()

            # Phase 3.5: Enforce future-leak validation gates
            self._validate_no_future_leakage()

            if self.config.snapshot_only:
                self.logger.info(
                    "Snapshot-only mode: persisting dataset bundle and review artifacts "
                    "without optimization or model fitting."
                )
                output_dir = Path(self.config.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                review_payload = self._persist_snapshot_artifacts(output_dir)
                duration = (datetime.now(timezone.utc) - self.start_time).total_seconds()
                results = self._build_snapshot_only_results(review_payload, duration)
                snapshot_ready = bool(results.get("success", False))
                self._record_training_run_event("completed", results=results)
                self.logger.info("Snapshot-only pipeline completed in %.1fs", duration)
                self.logger.info(
                    "Snapshot review readiness: %s",
                    "PASSED" if snapshot_ready else "FAILED",
                )
                return results

            # Phase 4: Hyperparameter Optimization (if enabled)
            if self.config.optimize:
                self._optimize_hyperparameters()

            # Phase 5: Cross-Validation Training
            self._train_with_cv()

            # Phase 6: Meta-Labeling (if enabled)
            if self.config.use_meta_labeling:
                self._train_meta_labeler()

            # Phase 6.5: OOF expected-edge policy (if enabled)
            if self.config.enable_expected_edge_policy:
                self._train_expected_edge_policy()

            # Phase 7: Multiple testing / overfitting diagnostics before gating.
            if self.config.apply_multiple_testing:
                self._apply_multiple_testing_correction()

            # Phase 8: Validation Gates (includes DSR/PBO hard checks)
            passed_gates = self._validate_model()

            # Phase 9: SHAP Explainability
            if self.config.compute_shap:
                self._compute_shap_values()

            # Phase 9.5: Governance artifacts for deployment control.
            self.training_metrics["model_card"] = self._build_model_card()
            self.training_metrics["deployment_plan"] = self._build_deployment_plan(
                passed_validation=bool(passed_gates)
            )
            self._record_pre_promotion_checklist()

            # Phase 10: Save Model
            model_path = self._save_model()

            # Compile results
            duration = (datetime.now(timezone.utc) - self.start_time).total_seconds()

            results = {
                "success": passed_gates,
                "run_id": self.run_id,
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
                "data_quality_report_passed": (
                    bool(self.data_quality_report.get("passed", False))
                    if isinstance(self.data_quality_report, dict)
                    else None
                ),
                "feature_schema_version": (
                    str(self.snapshot_manifest.get("feature_schema_version"))
                    if isinstance(self.snapshot_manifest, dict)
                    else None
                ),
                "dataset_snapshot_bundle_path": (
                    str(self.dataset_snapshot_bundle_manifest_path)
                    if self.dataset_snapshot_bundle_manifest_path is not None
                    else None
                ),
                "dataset_bundle_hash": (
                    str(self.dataset_snapshot_bundle_manifest.get("bundle_hash"))
                    if isinstance(self.dataset_snapshot_bundle_manifest, dict)
                    else (
                        str(self.snapshot_manifest.get("dataset_bundle_hash"))
                        if isinstance(self.snapshot_manifest, dict)
                        else None
                    )
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
                "artifacts_path": (
                    str(self.artifacts_path) if self.artifacts_path is not None else None
                ),
                "snapshot_review_path": (
                    str(self.snapshot_review_path)
                    if self.snapshot_review_path is not None
                    else None
                ),
                "run_event_index_path": (
                    str(self.run_event_index_path)
                    if self.run_event_index_path is not None
                    else None
                ),
                "model_card": self.training_metrics.get("model_card", {}),
                "deployment_plan": self.training_metrics.get("deployment_plan", {}),
                "pre_promotion_checklist": self.training_metrics.get("pre_promotion_checklist", {}),
            }

            self._record_training_run_event("completed", results=results)
            self.logger.info(f"Training completed in {duration:.1f}s")
            self.logger.info(f"Validation gates passed: {passed_gates}")

            return results

        except KeyboardInterrupt:
            self._record_training_run_event("interrupted", error="keyboard_interrupt")
            self.logger.error("Training interrupted by operator.")
            raise
        except Exception as e:
            self._record_training_run_event("failed", error=e)
            self.logger.error(f"Training failed: {e}")
            raise

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert numpy/pandas-heavy payloads into JSON-compatible structures."""
        if isinstance(value, dict):
            return {str(k): ModelTrainer._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [ModelTrainer._json_safe(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, pd.Series):
            return [ModelTrainer._json_safe(v) for v in value.tolist()]
        if isinstance(value, np.ndarray):
            return [ModelTrainer._json_safe(v) for v in value.tolist()]
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
        if isinstance(value, (datetime, pd.Timestamp)):
            ts = pd.to_datetime(value, utc=True, errors="coerce")
            return ts.isoformat() if not pd.isna(ts) else None
        if value is None or isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def _feature_family_name(feature_name: str) -> str:
        """Map feature names into stable family buckets for selection diagnostics."""
        tokens = [token for token in re.split(r"[^a-z0-9]+", str(feature_name).lower()) if token]
        return tokens[0] if tokens else "other"

    def _count_feature_families(self, feature_names: list[str]) -> dict[str, int]:
        """Count selected or rejected features by inferred family."""
        counts: dict[str, int] = {}
        for feature_name in feature_names:
            family = self._feature_family_name(feature_name)
            counts[family] = counts.get(family, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: item[0]))

    @staticmethod
    def _feature_quality_clip_rate(values: np.ndarray) -> float:
        """Estimate tail-pressure clip rate using a symmetric winsorization probe."""
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = arr[np.isfinite(arr)]
        if finite.size < 20:
            return 0.0
        lower, upper = np.percentile(finite, [1, 99])
        if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
            return 0.0
        return float(np.mean((finite < lower) | (finite > upper)))

    def _build_feature_quality_audit(
        self,
        matrix: pd.DataFrame,
        feature_names: list[str],
    ) -> dict[str, Any]:
        """Summarize feature-level numerical stability signals for artifact governance."""
        if matrix is None or matrix.empty or not feature_names:
            return {
                "feature_count": 0,
                "probe_clip_percentiles": [1, 99],
                "by_feature": {},
                "family_summary": {},
                "worst_null_features": [],
                "worst_clip_features": [],
            }

        by_feature: dict[str, dict[str, float]] = {}
        family_rollup: dict[str, dict[str, float]] = {}

        for feature_name in feature_names:
            series = pd.to_numeric(matrix[feature_name], errors="coerce")
            values = series.to_numpy(dtype=np.float64, copy=False)
            total = max(1, int(values.size))
            finite_mask = np.isfinite(values)
            null_rate = float(np.mean(np.isnan(values)))
            finite_rate = float(np.mean(finite_mask))
            clipped_rate = self._feature_quality_clip_rate(values)
            family = self._feature_family_name(feature_name)

            by_feature[str(feature_name)] = {
                "null_rate": null_rate,
                "finite_rate": finite_rate,
                "clipped_rate": clipped_rate,
                "sample_count": float(total),
            }

            rollup = family_rollup.setdefault(
                family,
                {
                    "feature_count": 0.0,
                    "null_rate_sum": 0.0,
                    "finite_rate_sum": 0.0,
                    "clipped_rate_sum": 0.0,
                    "max_null_rate": 0.0,
                    "min_finite_rate": 1.0,
                    "max_clipped_rate": 0.0,
                },
            )
            rollup["feature_count"] += 1.0
            rollup["null_rate_sum"] += null_rate
            rollup["finite_rate_sum"] += finite_rate
            rollup["clipped_rate_sum"] += clipped_rate
            rollup["max_null_rate"] = max(float(rollup["max_null_rate"]), null_rate)
            rollup["min_finite_rate"] = min(float(rollup["min_finite_rate"]), finite_rate)
            rollup["max_clipped_rate"] = max(float(rollup["max_clipped_rate"]), clipped_rate)

        family_summary: dict[str, dict[str, float]] = {}
        for family, rollup in family_rollup.items():
            count = max(1.0, float(rollup["feature_count"]))
            family_summary[family] = {
                "feature_count": count,
                "mean_null_rate": float(rollup["null_rate_sum"] / count),
                "mean_finite_rate": float(rollup["finite_rate_sum"] / count),
                "mean_clipped_rate": float(rollup["clipped_rate_sum"] / count),
                "max_null_rate": float(rollup["max_null_rate"]),
                "min_finite_rate": float(rollup["min_finite_rate"]),
                "max_clipped_rate": float(rollup["max_clipped_rate"]),
            }

        ordered_null = sorted(
            by_feature.items(),
            key=lambda item: (-float(item[1]["null_rate"]), str(item[0])),
        )
        ordered_clip = sorted(
            by_feature.items(),
            key=lambda item: (-float(item[1]["clipped_rate"]), str(item[0])),
        )
        null_rates = np.asarray([payload["null_rate"] for payload in by_feature.values()], dtype=float)
        finite_rates = np.asarray(
            [payload["finite_rate"] for payload in by_feature.values()],
            dtype=float,
        )
        clipped_rates = np.asarray(
            [payload["clipped_rate"] for payload in by_feature.values()],
            dtype=float,
        )
        return {
            "feature_count": int(len(by_feature)),
            "probe_clip_percentiles": [1, 99],
            "mean_null_rate": float(np.mean(null_rates)) if null_rates.size else 0.0,
            "max_null_rate": float(np.max(null_rates)) if null_rates.size else 0.0,
            "mean_finite_rate": float(np.mean(finite_rates)) if finite_rates.size else 1.0,
            "min_finite_rate": float(np.min(finite_rates)) if finite_rates.size else 1.0,
            "mean_clipped_rate": float(np.mean(clipped_rates)) if clipped_rates.size else 0.0,
            "max_clipped_rate": float(np.max(clipped_rates)) if clipped_rates.size else 0.0,
            "by_feature": by_feature,
            "family_summary": dict(sorted(family_summary.items())),
            "worst_null_features": [
                {"feature": name, **payload}
                for name, payload in ordered_null[:10]
                if float(payload["null_rate"]) > 0.0
            ],
            "worst_clip_features": [
                {"feature": name, **payload}
                for name, payload in ordered_clip[:10]
                if float(payload["clipped_rate"]) > 0.0
            ],
        }

    def _build_feature_selection_audit(
        self,
        *,
        old_feature_names: list[str],
        selection_result: Any,
        selected_features: list[str],
    ) -> dict[str, Any]:
        """Build a detailed feature-selection audit for training artifacts."""
        information_coefficients = {
            str(name): float(value)
            for name, value in getattr(selection_result, "information_coefficients", {}).items()
        }
        stability_scores = {
            str(name): float(score)
            for name, score in getattr(selection_result, "stability_scores", {}).items()
        }
        screened_features = list(information_coefficients.keys())
        screened_set = set(screened_features)
        selected_set = set(selected_features)
        correlation_pruned = {
            str(name) for name in getattr(selection_result, "correlation_pruned_features", [])
        }
        low_stability = {
            name
            for name, score in stability_scores.items()
            if float(score) < float(self.config.feature_selection_min_stability_support)
        }
        rejected_features: list[dict[str, Any]] = []
        for feature_name in old_feature_names:
            if feature_name in selected_set:
                continue
            rejection_reason = "information_coefficient"
            if feature_name in correlation_pruned:
                rejection_reason = "correlation_pruned"
            elif feature_name in low_stability:
                rejection_reason = "stability_support"
            elif feature_name in screened_set:
                rejection_reason = "max_feature_cap"
            rejected_features.append(
                {
                    "feature": str(feature_name),
                    "family": self._feature_family_name(feature_name),
                    "reason": rejection_reason,
                    "information_coefficient": float(
                        information_coefficients.get(feature_name, 0.0)
                    ),
                    "stability_score": float(stability_scores.get(feature_name, 0.0)),
                }
            )

        diagnostics = dict(getattr(selection_result, "diagnostics", {}) or {})
        return {
            "selection_binding": bool(len(selected_features) < len(old_feature_names)),
            "selected_feature_count": int(len(selected_features)),
            "initial_feature_count": int(len(old_feature_names)),
            "screened_feature_count": int(len(screened_features)),
            "correlation_pruned_feature_count": int(len(correlation_pruned)),
            "low_stability_feature_count": int(len(low_stability.difference(selected_set))),
            "rejected_feature_count": int(len(rejected_features)),
            "min_information_coefficient": float(self.config.feature_selection_min_ic),
            "max_correlation": float(self.config.feature_selection_max_corr),
            "max_features": int(self.config.feature_selection_max_features),
            "min_stability_support": float(self.config.feature_selection_min_stability_support),
            "selected_features": list(selected_features),
            "screened_features": screened_features,
            "correlation_pruned_features": sorted(correlation_pruned),
            "rejected_features": rejected_features,
            "selected_family_counts": self._count_feature_families(selected_features),
            "rejected_family_counts": self._count_feature_families(
                [str(item["feature"]) for item in rejected_features]
            ),
            "diagnostics": self._json_safe(diagnostics),
        }

    def _hydrate_snapshot_feature_selection_summary(self) -> None:
        """Restore upstream feature-selection diagnostics from a snapshot bundle when present."""
        manifest = self.dataset_snapshot_bundle_manifest
        if not isinstance(manifest, dict):
            return
        summary = manifest.get("feature_selection_summary", {})
        if not isinstance(summary, dict) or not summary:
            return

        upstream_binding = float(
            summary.get("development_binding", summary.get("binding", 0.0))
        )
        upstream_initial = float(
            summary.get(
                "development_initial_feature_count",
                summary.get("initial_feature_count", 0.0),
            )
        )
        upstream_selected = float(
            summary.get(
                "development_selected_feature_count",
                summary.get("selected_feature_count", 0.0),
            )
        )
        self.training_metrics["feature_selection_upstream_binding"] = upstream_binding
        self.training_metrics["feature_selection_upstream_initial_feature_count"] = upstream_initial
        self.training_metrics["feature_selection_upstream_selected_feature_count"] = (
            upstream_selected
        )
        self.training_metrics.setdefault("feature_selection_development_binding", upstream_binding)
        self.training_metrics.setdefault(
            "feature_selection_development_initial_feature_count",
            upstream_initial,
        )
        self.training_metrics.setdefault(
            "feature_selection_development_selected_feature_count",
            upstream_selected,
        )
        upstream_audit = summary.get("audit")
        if isinstance(upstream_audit, dict) and upstream_audit:
            self.training_metrics.setdefault(
                "feature_selection_upstream_audit",
                self._json_safe(upstream_audit),
            )

    def _training_run_index_root(self) -> Path:
        """Return the directory that stores durable training-run events."""
        return Path(self.config.output_dir) / "run_index"

    def _optuna_state_dir(self) -> Path:
        """Return the directory that stores resumable Optuna state."""
        configured = str(getattr(self.config, "optuna_storage_dir", "") or "").strip()
        if configured:
            return Path(configured)
        return Path(self.config.output_dir) / "optuna_state"

    def _build_optuna_study_artifacts(self, namespace: str) -> dict[str, Any]:
        """Resolve deterministic names and file paths for one Optuna study."""
        snapshot_id = (
            str(self.snapshot_manifest.get("snapshot_id"))
            if isinstance(self.snapshot_manifest, dict)
            else "live_data"
        )
        raw_name = "__".join(
            [
                str(self.config.model_name),
                str(self.config.model_type),
                str(self.config.training_profile),
                f"h{int(self.config.primary_label_horizon)}",
                str(self.config.timeframe),
                snapshot_id,
                str(namespace),
            ]
        )
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._")
        if not safe_name:
            safe_name = "optuna_study"
        safe_name = safe_name[:120]
        digest = hashlib.sha256(raw_name.encode("utf-8")).hexdigest()[:12]
        study_name = f"{safe_name}_{digest}"
        state_dir = self._optuna_state_dir()
        return {
            "namespace": str(namespace),
            "study_name": study_name,
            "state_dir": state_dir,
            "storage_path": state_dir / f"{study_name}.sqlite3",
            "manifest_path": state_dir / f"{study_name}.study.json",
        }

    @staticmethod
    def _optuna_trial_state_summary(study: Any, optuna: Any) -> dict[str, int]:
        """Summarize current Optuna trial states for resume bookkeeping."""
        states = getattr(optuna, "trial", None)
        trial_state = getattr(states, "TrialState", None)
        counts = {
            "existing_trials": 0,
            "complete_trials": 0,
            "pruned_trials": 0,
            "failed_trials": 0,
            "running_trials": 0,
            "waiting_trials": 0,
            "finalized_trials": 0,
        }
        if trial_state is None:
            return counts

        finalized_states = {
            getattr(trial_state, "COMPLETE", object()),
            getattr(trial_state, "PRUNED", object()),
            getattr(trial_state, "FAIL", object()),
        }
        for trial in list(getattr(study, "trials", []) or []):
            counts["existing_trials"] += 1
            state = getattr(trial, "state", None)
            if state == getattr(trial_state, "COMPLETE", None):
                counts["complete_trials"] += 1
            elif state == getattr(trial_state, "PRUNED", None):
                counts["pruned_trials"] += 1
            elif state == getattr(trial_state, "FAIL", None):
                counts["failed_trials"] += 1
            elif state == getattr(trial_state, "RUNNING", None):
                counts["running_trials"] += 1
            elif state == getattr(trial_state, "WAITING", None):
                counts["waiting_trials"] += 1
            if state in finalized_states:
                counts["finalized_trials"] += 1
        return counts

    def _write_optuna_study_manifest(self, payload: dict[str, Any]) -> Path | None:
        """Persist one Optuna study manifest for resume/debugging."""
        manifest_path = payload.get("manifest_path")
        if not manifest_path:
            return None
        path = Path(str(manifest_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_payload = self._json_safe(payload)
        path.write_text(
            json.dumps(safe_payload, indent=2, ensure_ascii=True, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return path

    def _prepare_optuna_study(
        self,
        optuna: Any,
        *,
        namespace: str,
        direction: str,
        sampler: Any,
        pruner: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Create or resume an Optuna study backed by durable SQLite storage."""
        artifacts = self._build_optuna_study_artifacts(namespace)
        study_kwargs: dict[str, Any] = {
            "direction": direction,
            "sampler": sampler,
            "pruner": pruner,
            "study_name": artifacts["study_name"],
        }
        if self.config.optuna_resume_enabled:
            state_dir = Path(artifacts["state_dir"])
            state_dir.mkdir(parents=True, exist_ok=True)
            storage_path = Path(artifacts["storage_path"]).resolve()
            study_kwargs["storage"] = f"sqlite:///{storage_path.as_posix()}"
            study_kwargs["load_if_exists"] = True

        study = optuna.create_study(**study_kwargs)
        summary = self._optuna_trial_state_summary(study, optuna)
        study_info = {
            **artifacts,
            **summary,
            "resume_enabled": bool(self.config.optuna_resume_enabled),
            "resumed_from_storage": bool(
                self.config.optuna_resume_enabled and summary["existing_trials"] > 0
            ),
            "remaining_trials": int(
                max(int(self.config.n_trials) - int(summary["finalized_trials"]), 0)
            ),
            "direction": str(direction),
            "run_id": self.run_id,
            "model_name": self.config.model_name,
            "model_type": self.config.model_type,
            "training_profile": self.config.training_profile,
            "primary_label_horizon": int(self.config.primary_label_horizon),
            "snapshot_id": (
                str(self.snapshot_manifest.get("snapshot_id"))
                if isinstance(self.snapshot_manifest, dict)
                else None
            ),
            "status": "prepared",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = self._write_optuna_study_manifest(study_info)
        if manifest_path is not None:
            study_info["manifest_path"] = manifest_path
        return study, study_info

    def _record_optuna_study_metrics(
        self,
        prefix: str,
        study_info: dict[str, Any],
    ) -> None:
        """Flatten study persistence metadata into training metrics."""
        self.training_metrics[f"{prefix}_study"] = self._json_safe(study_info)
        for key in (
            "study_name",
            "remaining_trials",
            "existing_trials",
            "complete_trials",
            "pruned_trials",
            "failed_trials",
            "running_trials",
            "finalized_trials",
        ):
            value = study_info.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)):
                self.training_metrics[f"{prefix}_{key}"] = float(value)
            elif value is not None:
                self.training_metrics[f"{prefix}_{key}"] = value
        for key in ("resume_enabled", "resumed_from_storage"):
            self.training_metrics[f"{prefix}_{key}"] = float(bool(study_info.get(key, False)))
        storage_path = study_info.get("storage_path")
        if storage_path is not None:
            self.training_metrics[f"{prefix}_storage_path"] = str(storage_path)
        manifest_path = study_info.get("manifest_path")
        if manifest_path is not None:
            self.training_metrics[f"{prefix}_manifest_path"] = str(manifest_path)

    def _record_training_run_event(
        self,
        status: str,
        *,
        results: dict[str, Any] | None = None,
        error: Exception | str | None = None,
    ) -> None:
        """Append a durable lifecycle event for the current training run."""
        duration_seconds = 0.0
        if self.start_time is not None:
            duration_seconds = float(
                max((datetime.now(timezone.utc) - self.start_time).total_seconds(), 0.0)
            )

        validation_passed = None
        if isinstance(results, dict) and "passed_validation_gates" in results:
            validation_passed = bool(results.get("passed_validation_gates"))

        payload = {
            "event_type": "training_run",
            "status": str(status),
            "run_id": self.run_id,
            "model_name": self.config.model_name,
            "model_type": self.config.model_type,
            "training_profile": self.config.training_profile,
            "timeframe": self.config.timeframe,
            "primary_label_horizon": int(self.config.primary_label_horizon),
            "output_dir": str(Path(self.config.output_dir)),
            "duration_seconds": duration_seconds,
            "validation_passed": validation_passed,
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
            "dataset_snapshot_bundle_path": (
                str(self.dataset_snapshot_bundle_manifest_path)
                if self.dataset_snapshot_bundle_manifest_path is not None
                else None
            ),
            "replay_manifest_path": (
                str(self.replay_manifest_path) if self.replay_manifest_path is not None else None
            ),
            "promotion_package_path": (
                str(self.promotion_package_path)
                if self.promotion_package_path is not None
                else None
            ),
            "artifacts_path": str(self.artifacts_path) if self.artifacts_path is not None else None,
            "provenance": self._json_safe(self.training_metrics.get("training_provenance", {})),
            "error": str(error) if error is not None else None,
            "metrics": {
                "mean_sharpe": float(self.training_metrics.get("mean_sharpe", 0.0)),
                "mean_trade_count": float(self.training_metrics.get("mean_trade_count", 0.0)),
                "mean_risk_adjusted_score": float(
                    self.training_metrics.get("mean_risk_adjusted_score", 0.0)
                ),
                "holdout_sharpe": float(self.training_metrics.get("holdout_sharpe", 0.0)),
                "holdout_max_drawdown": float(
                    self.training_metrics.get("holdout_max_drawdown", 0.0)
                ),
                "holdout_symbol_sharpe_p25": float(
                    self.training_metrics.get("holdout_symbol_sharpe_p25", 0.0)
                ),
                "pbo": float(self.training_metrics.get("pbo", 0.0)),
                "white_reality_pvalue": float(
                    self.training_metrics.get("white_reality_pvalue", 1.0)
                ),
            },
        }

        if self.data_quality_report is not None:
            payload["data_quality"] = {
                "passed": bool(self.data_quality_report.get("passed", False)),
                "summary": self._json_safe(self.data_quality_report.get("summary", {})),
                "threshold_breaches": self._json_safe(
                    self.data_quality_report.get("threshold_breaches", {})
                ),
            }
        if isinstance(results, dict):
            payload["result"] = {
                "success": bool(results.get("success", False)),
                "model_path": results.get("model_path"),
            }

        try:
            self.run_event_index_path = append_training_run_event(
                self._training_run_index_root(),
                self._json_safe(payload),
            )
            self.training_metrics["run_event_index_path"] = str(self.run_event_index_path)
        except Exception as exc:  # pragma: no cover - logging fallback
            self.logger.warning("Failed to append training-run event: %s", exc)

    def _load_data(self) -> None:
        """Phase 1: Load market data for training."""
        self.logger.info("Phase 1: Loading data...")
        if not self.config.use_database:
            raise RuntimeError(
                "Institutional training mode requires PostgreSQL data source. "
                "Disable is not allowed."
            )
        self._load_data_from_postgres()
        self._apply_corporate_action_adjustments()
        self._sanitize_loaded_data()
        self._apply_market_session_filter()
        self._apply_symbol_universe_filters()
        self._apply_training_bar_mode()
        self._capture_data_quality_report()
        self._apply_live_multi_symbol_profile()

    def _apply_live_multi_symbol_profile(self) -> None:
        """Auto-tune training config for large multi-symbol live deployment datasets."""
        if self.data is None or self.data.empty:
            return

        symbol_count = int(self.data["symbol"].nunique())
        ts = pd.to_datetime(self.data["timestamp"], utc=True, errors="coerce").dropna()
        if ts.empty:
            return

        span_days = float((ts.max() - ts.min()).total_seconds() / 86400.0)
        span_years = span_days / 365.25
        row_count = int(len(self.data))
        qualifies = bool(
            self.config.auto_live_profile_enabled
            and symbol_count >= int(self.config.auto_live_profile_symbol_threshold)
            and span_years >= float(self.config.auto_live_profile_min_years)
        )

        self.training_metrics["dataset_symbol_count"] = float(symbol_count)
        self.training_metrics["dataset_timespan_years"] = float(span_years)
        self.training_metrics["dataset_rows"] = float(row_count)
        self.training_metrics["auto_live_profile_applied"] = bool(qualifies)
        self.training_metrics["training_profile_mode"] = (
            "institutional_live_multi_symbol" if qualifies else "default"
        )

        if not qualifies:
            return

        updates: dict[str, tuple[float, float]] = {}

        def _raise_floor(attr: str, target: float, as_int: bool = False) -> None:
            current_raw = getattr(self.config, attr)
            current = float(current_raw)
            new_float = float(max(current, target))
            new_raw = int(round(new_float)) if as_int else float(new_float)
            setattr(self.config, attr, new_raw)
            if abs(new_float - current) > 1e-12:
                updates[attr] = (current, float(new_raw))

        def _tighten_ceiling(attr: str, target: float, as_int: bool = False) -> None:
            current_raw = getattr(self.config, attr)
            current = float(current_raw)
            new_float = float(min(current, target))
            new_raw = int(round(new_float)) if as_int else float(new_float)
            setattr(self.config, attr, new_raw)
            if abs(new_float - current) > 1e-12:
                updates[attr] = (current, float(new_raw))

        _raise_floor("n_trials", 180.0, as_int=True)
        _raise_floor("n_splits", 6.0, as_int=True)
        _raise_floor("nested_outer_splits", 5.0, as_int=True)
        _raise_floor("nested_inner_splits", 4.0, as_int=True)
        _raise_floor("min_trades", 180.0, as_int=True)
        _raise_floor("holdout_pct", 0.20)
        _raise_floor("objective_weight_tail_risk", 0.45)
        _raise_floor("objective_weight_symbol_concentration", 0.30)
        _raise_floor("objective_weight_cvar", 0.55)
        _raise_floor("min_holdout_sharpe", 0.10)
        _raise_floor("min_holdout_regime_sharpe", 0.00)
        _raise_floor("min_holdout_symbol_coverage", 0.65)
        _raise_floor("min_holdout_symbol_p25_sharpe", -0.05)
        _tighten_ceiling("max_pbo", 0.35)
        _tighten_ceiling("max_white_reality_pvalue", 0.08)
        _tighten_ceiling("max_holdout_drawdown", 0.30)
        _tighten_ceiling("max_symbol_concentration_hhi", 0.55)
        _tighten_ceiling("max_holdout_symbol_underwater_ratio", 0.45)

        current_max_symbols = int(getattr(self.config, "max_cross_sectional_symbols"))
        target_max_symbols = int(max(current_max_symbols, symbol_count + 4))
        if target_max_symbols != current_max_symbols:
            self.config.max_cross_sectional_symbols = int(target_max_symbols)
            updates["max_cross_sectional_symbols"] = (
                float(current_max_symbols),
                float(target_max_symbols),
            )

        current_max_rows = int(getattr(self.config, "max_cross_sectional_rows"))
        target_max_rows = int(max(current_max_rows, int(round(row_count * 1.2))))
        if target_max_rows != current_max_rows:
            self.config.max_cross_sectional_rows = int(target_max_rows)
            updates["max_cross_sectional_rows"] = (
                float(current_max_rows),
                float(target_max_rows),
            )

        if not bool(self.config.enable_cross_sectional) and not bool(
            self.config.cross_sectional_user_locked
        ):
            self.config.enable_cross_sectional = True
            updates["enable_cross_sectional"] = (0.0, 1.0)

        self.training_metrics["auto_live_profile_updates"] = {
            key: {"from": float(old), "to": float(new)} for key, (old, new) in updates.items()
        }
        self.logger.info(
            "Applied institutional live multi-symbol profile: symbols=%d years=%.2f updates=%d",
            symbol_count,
            span_years,
            len(updates),
        )

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
            df = df.drop_duplicates(subset=["symbol", "timestamp"], keep="last").reset_index(
                drop=True
            )

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
        base_timeframe = normalize_timeframe(self.config.timeframe)
        if base_timeframe in self.ohlcv_panels_by_timeframe:
            self.ohlcv_panels_by_timeframe[base_timeframe] = self.data.copy()
        self.training_metrics["sanitization_initial_rows"] = float(initial_rows)
        self.training_metrics["sanitization_cleaned_rows"] = float(cleaned_rows)
        self.training_metrics["sanitization_duplicate_rows_removed"] = float(duplicate_count)
        self.training_metrics["sanitization_invalid_ohlcv_rows_removed"] = float(
            invalid_ohlcv_count
        )
        self.training_metrics["sanitization_extreme_return_rows_removed"] = float(
            extreme_return_count
        )

        self.logger.info(
            "Data sanitization complete: "
            f"rows={cleaned_rows}/{initial_rows}, "
            f"duplicates_removed={duplicate_count}, "
            f"invalid_ohlcv_removed={invalid_ohlcv_count}, "
            f"extreme_return_removed={extreme_return_count}"
        )

    def _apply_market_session_filter(self) -> None:
        """Remove sparse out-of-session bars before quality and feature diagnostics."""
        if self.data is None or self.data.empty:
            raise ValueError("No data loaded for market-session filtering.")
        requested_premarket = bool(self.config.include_premarket)
        requested_postmarket = bool(self.config.include_postmarket)
        panels = dict(getattr(self, "ohlcv_panels_by_timeframe", {}) or {})
        base_timeframe = normalize_timeframe(self.config.timeframe)
        if base_timeframe not in panels:
            panels[base_timeframe] = self.data.copy()

        filtered_panels: dict[str, pd.DataFrame] = {}
        filter_summary: dict[str, dict[str, Any]] = {}
        rows_removed = 0
        any_filter_applied = False

        for timeframe_name, panel in panels.items():
            filtered_panel, metadata = filter_ohlcv_frame_to_market_session(
                panel,
                timeframe=timeframe_name,
                include_premarket=requested_premarket,
                include_postmarket=requested_postmarket,
            )
            if not filtered_panel.empty:
                filtered_panel["timestamp"] = pd.to_datetime(
                    filtered_panel["timestamp"], utc=True, errors="coerce"
                )
                filtered_panel = (
                    filtered_panel.dropna(subset=["timestamp"])
                    .sort_values(["symbol", "timestamp"])
                    .reset_index(drop=True)
                )
            filtered_panels[normalize_timeframe(timeframe_name)] = filtered_panel
            filter_summary[normalize_timeframe(timeframe_name)] = {
                "applied": bool(metadata.get("applied", False)),
                "reason": str(metadata.get("reason", "")),
                "input_rows": int(metadata.get("input_rows", len(panel))),
                "output_rows": int(metadata.get("output_rows", len(filtered_panel))),
                "removed_rows": int(metadata.get("removed_rows", 0)),
            }
            rows_removed += int(metadata.get("removed_rows", 0))
            any_filter_applied = any_filter_applied or bool(metadata.get("applied", False))

        base_panel = filtered_panels.get(base_timeframe)
        if base_panel is None or base_panel.empty:
            raise ValueError(
                "Market-session filter removed all base-timeframe rows; "
                "verify the OHLCV session coverage."
            )

        self.ohlcv_panels_by_timeframe = filtered_panels
        self.data = base_panel.copy().reset_index(drop=True)
        self.training_metrics["market_session_filter_applied"] = bool(any_filter_applied)
        self.training_metrics["market_session_include_premarket"] = bool(requested_premarket)
        self.training_metrics["market_session_include_postmarket"] = bool(requested_postmarket)
        self.training_metrics["market_session_rows_removed"] = float(rows_removed)
        self.training_metrics["market_session_row_counts_by_timeframe"] = {
            tf_name: {
                "input_rows": int(summary["input_rows"]),
                "output_rows": int(summary["output_rows"]),
                "removed_rows": int(summary["removed_rows"]),
            }
            for tf_name, summary in filter_summary.items()
        }
        self.training_metrics["market_session_filter_summary"] = self._json_safe(filter_summary)
        self.logger.info(
            "Market-session filter applied: premarket=%s postmarket=%s base_rows=%d removed=%d",
            requested_premarket,
            requested_postmarket,
            len(self.data),
            rows_removed,
        )

    def _load_corporate_actions_for_adjustment(
        self,
        *,
        symbols: list[str],
        min_trade_date: date,
        max_trade_date: date,
    ) -> pd.DataFrame:
        """Load dividend and split actions needed for OHLCV back-adjustment."""
        from sqlalchemy import select
        from quant_trading_system.database.connection import get_db_manager
        from quant_trading_system.database.models import CorporateAction

        if not symbols:
            return pd.DataFrame(
                columns=["symbol", "action_type", "ex_date", "amount", "split_ratio"]
            )

        db_manager = get_db_manager()
        with db_manager.session() as session:
            rows = list(
                session.execute(
                    select(
                        CorporateAction.symbol,
                        CorporateAction.action_type,
                        CorporateAction.ex_date,
                        CorporateAction.amount,
                        CorporateAction.split_from,
                        CorporateAction.split_to,
                    ).where(
                        CorporateAction.symbol.in_(symbols),
                        CorporateAction.ex_date >= min_trade_date,
                        CorporateAction.ex_date <= max_trade_date,
                    )
                ).all()
            )

        if not rows:
            return pd.DataFrame(
                columns=["symbol", "action_type", "ex_date", "amount", "split_ratio"]
            )

        actions = pd.DataFrame(
            rows,
            columns=["symbol", "action_type", "ex_date", "amount", "split_from", "split_to"],
        )
        actions["symbol"] = actions["symbol"].astype(str).str.upper().str.strip()
        actions["action_type"] = actions["action_type"].astype(str).str.upper().str.strip()
        actions["amount"] = pd.to_numeric(actions["amount"], errors="coerce").fillna(0.0)
        split_from = pd.to_numeric(actions["split_from"], errors="coerce")
        split_to = pd.to_numeric(actions["split_to"], errors="coerce")
        denominator = split_from.replace(0.0, np.nan)
        split_ratio = (split_to / denominator).replace([np.inf, -np.inf], np.nan)
        missing_ratio = split_ratio.isna()
        split_ratio.loc[missing_ratio] = split_to.loc[missing_ratio]
        actions["split_ratio"] = split_ratio.fillna(0.0)
        return actions[["symbol", "action_type", "ex_date", "amount", "split_ratio"]].copy()

    def _apply_corporate_action_adjustments(self) -> None:
        """Back-adjust OHLCV history for splits and dividends before training."""
        if (
            not bool(self.config.adjust_prices_for_corporate_actions)
            or self.data is None
            or self.data.empty
        ):
            return

        df = self.data.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["symbol", "timestamp"]).copy()
        if df.empty:
            return

        symbols = sorted(df["symbol"].astype(str).str.upper().str.strip().unique().tolist())
        min_trade_date = pd.Timestamp(df["timestamp"].min()).date()
        max_trade_date = pd.Timestamp(df["timestamp"].max()).date()
        actions = self._load_corporate_actions_for_adjustment(
            symbols=symbols,
            min_trade_date=min_trade_date,
            max_trade_date=max_trade_date,
        )
        if actions.empty:
            self.training_metrics["corporate_action_adjustment_split_events"] = 0.0
            self.training_metrics["corporate_action_adjustment_dividend_events"] = 0.0
            self.training_metrics["corporate_action_adjustment_row_operations"] = 0.0
            return

        price_cols = ["open", "high", "low", "close"]
        for column in [*price_cols, "volume"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        split_events = 0
        dividend_events = 0
        adjusted_row_operations = 0

        for symbol, row_index in df.groupby("symbol", sort=False).groups.items():
            symbol_actions = actions[actions["symbol"] == symbol].sort_values(
                "ex_date", ascending=False
            )
            if symbol_actions.empty:
                continue

            symbol_frame = df.loc[row_index].sort_values("timestamp").copy()
            symbol_timestamps = pd.to_datetime(symbol_frame["timestamp"], utc=True, errors="coerce")

            for action in symbol_actions.itertuples(index=False):
                cutoff = pd.Timestamp(action.ex_date, tz="UTC")
                prior_mask = symbol_timestamps < cutoff
                affected_rows = int(prior_mask.sum())
                if affected_rows <= 0:
                    continue

                if action.action_type == "SPLIT":
                    ratio = float(action.split_ratio)
                    if ratio <= 0.0 or np.isclose(ratio, 1.0):
                        continue
                    symbol_frame.loc[prior_mask, price_cols] = (
                        symbol_frame.loc[prior_mask, price_cols].astype(float) / ratio
                    )
                    symbol_frame.loc[prior_mask, "volume"] = np.round(
                        symbol_frame.loc[prior_mask, "volume"].astype(float) * ratio
                    )
                    split_events += 1
                    adjusted_row_operations += affected_rows
                    continue

                if action.action_type != "DIVIDEND":
                    continue
                amount = float(action.amount)
                if amount <= 0.0:
                    continue
                previous_rows = symbol_frame.loc[prior_mask]
                if previous_rows.empty:
                    continue
                previous_close = float(previous_rows["close"].iloc[-1])
                if previous_close <= amount or previous_close <= 0.0:
                    continue
                factor = (previous_close - amount) / previous_close
                if factor <= 0.0 or np.isclose(factor, 1.0):
                    continue
                symbol_frame.loc[prior_mask, price_cols] = (
                    symbol_frame.loc[prior_mask, price_cols].astype(float) * factor
                )
                dividend_events += 1
                adjusted_row_operations += affected_rows

            df.loc[symbol_frame.index, [*price_cols, "volume"]] = symbol_frame[
                [*price_cols, "volume"]
            ]

        self.data = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        self.training_metrics["corporate_action_adjustment_split_events"] = float(split_events)
        self.training_metrics["corporate_action_adjustment_dividend_events"] = float(
            dividend_events
        )
        self.training_metrics["corporate_action_adjustment_row_operations"] = float(
            adjusted_row_operations
        )
        self.logger.info(
            "Applied corporate action back-adjustments: splits=%d dividends=%d row_ops=%d",
            split_events,
            dividend_events,
            adjusted_row_operations,
        )

    def _apply_symbol_universe_filters(self) -> None:
        """Apply symbol-level quality filter before feature generation."""
        if self.data is None or self.data.empty:
            raise ValueError("No data loaded for symbol quality filtering.")
        if not self.config.enable_symbol_quality_filter:
            return

        thresholds = SymbolQualityThresholds(
            min_rows=int(self.config.symbol_quality_min_rows),
            max_missing_ratio=float(self.config.symbol_quality_max_missing_ratio),
            max_extreme_move_ratio=float(self.config.symbol_quality_max_extreme_move_ratio),
            max_corporate_action_ratio=float(self.config.symbol_quality_max_corporate_action_ratio),
            min_median_dollar_volume=float(self.config.symbol_quality_min_median_dollar_volume),
        )
        records: list[dict[str, Any]] = []
        for symbol, group in self.data.groupby("symbol", sort=True):
            g = group.sort_values("timestamp").reset_index(drop=True)
            assessment = assess_symbol_quality(g, thresholds)
            records.append(
                {
                    "symbol": str(symbol),
                    "rows": int(assessment["rows"]),
                    "missing_ratio": float(assessment["missing_ratio"]),
                    "extreme_move_ratio": float(assessment["extreme_move_ratio"]),
                    "corporate_action_ratio": float(assessment["corporate_action_ratio"]),
                    "median_dollar_volume": float(assessment["median_dollar_volume"]),
                    "passes": bool(assessment["passes"]),
                    "quality_score": float(assessment["quality_score"]),
                    "reasons": list(assessment.get("reasons", [])),
                }
            )

        if not records:
            raise ValueError("Symbol quality filter found no symbol records.")

        min_symbols = int(self.config.symbol_quality_min_symbols)
        passing_symbols = [r["symbol"] for r in records if bool(r["passes"])]
        if len(passing_symbols) < min_symbols:
            ranked = sorted(
                records,
                key=lambda r: (float(r["quality_score"]), float(r["rows"])),
                reverse=True,
            )
            passing_symbols = [str(r["symbol"]) for r in ranked[: min(min_symbols, len(ranked))]]
            self.logger.warning(
                "Symbol quality filter kept fewer than min symbols (%d). "
                "Using top-quality fallback universe of %d symbols.",
                min_symbols,
                len(passing_symbols),
            )

        keep_set = set(passing_symbols)
        filtered = self.data[self.data["symbol"].isin(keep_set)].copy().reset_index(drop=True)
        if filtered.empty:
            raise ValueError("Symbol quality filter removed entire universe.")

        dropped_symbols = sorted(set(self.data["symbol"].astype(str)) - keep_set)
        self.data = filtered
        self.training_metrics["symbol_quality_input_symbols"] = float(len(records))
        self.training_metrics["symbol_quality_selected_symbols"] = float(len(keep_set))
        self.training_metrics["symbol_quality_dropped_symbols"] = float(len(dropped_symbols))
        self.training_metrics["symbol_quality_universe"] = sorted(list(keep_set))
        self.training_metrics["symbol_quality_dropped_list"] = dropped_symbols
        self.training_metrics["symbol_quality_report"] = {
            str(r["symbol"]): {
                "rows": float(r["rows"]),
                "missing_ratio": float(r["missing_ratio"]),
                "extreme_move_ratio": float(r["extreme_move_ratio"]),
                "corporate_action_ratio": float(r["corporate_action_ratio"]),
                "median_dollar_volume": float(r["median_dollar_volume"]),
                "passes": bool(r["passes"]),
                "quality_score": float(r["quality_score"]),
                "reasons": list(r.get("reasons", [])),
            }
            for r in records
        }
        self.logger.info(
            "Symbol quality filter complete: selected=%d/%d, dropped=%d",
            len(keep_set),
            len(records),
            len(dropped_symbols),
        )

    @staticmethod
    def _symbol_missing_ratio(group: pd.DataFrame) -> float:
        """Backwards-compatible helper for tests and diagnostics."""
        from quant_trading_system.models.training_lineage import estimate_missing_bars_count

        if group is None or group.empty:
            return 1.0
        ts = pd.to_datetime(group["timestamp"], utc=True, errors="coerce").dropna().sort_values()
        if len(ts) < 2:
            return 0.0
        missing_count, inferred_bar_seconds = estimate_missing_bars_count(ts)
        if inferred_bar_seconds is None or inferred_bar_seconds <= 0.0:
            return 0.0
        expected_rows = float(len(ts) + missing_count)
        if expected_rows <= 0.0:
            return 0.0
        return float(np.clip(missing_count / expected_rows, 0.0, 1.0))

    def _apply_training_bar_mode(self) -> None:
        """Convert sanitized OHLCV panel into the requested training bar representation."""
        if self.data is None or self.data.empty:
            raise ValueError("No data loaded for training bar transformation.")
        if self.config.training_bar_mode == "time":
            self.training_metrics["training_bar_mode"] = "time"
            self.training_metrics["training_bar_input_rows"] = float(len(self.data))
            self.training_metrics["training_bar_output_rows"] = float(len(self.data))
            return

        from quant_trading_system.data.training_bars import TrainingBarBuilder, TrainingBarConfig

        builder = TrainingBarBuilder(
            TrainingBarConfig(
                mode=self.config.training_bar_mode,
                intrinsic_bar_type=self.config.intrinsic_bar_type,
                intrinsic_threshold=self.config.intrinsic_bar_threshold,
                target_bars_per_day=self.config.intrinsic_target_bars_per_day,
                use_trade_prints_if_available=True,
            ),
            logger_=self.logger,
        )
        input_rows = len(self.data)
        transformed = builder.build(self.data)
        if transformed is None or transformed.empty:
            raise ValueError("Intrinsic training bar mode produced no rows.")
        self.data = transformed.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        self.training_metrics["training_bar_mode"] = self.config.training_bar_mode
        self.training_metrics["training_bar_input_rows"] = float(input_rows)
        self.training_metrics["training_bar_output_rows"] = float(len(self.data))
        self.training_metrics["training_bar_intrinsic_type"] = self.config.intrinsic_bar_type
        self.logger.info(
            "Training bar mode applied: mode=%s rows=%d->%d",
            self.config.training_bar_mode,
            input_rows,
            len(self.data),
        )

    def _select_candidate_symbols_from_postgres(
        self,
        session: Any,
        redis_mgr: Any,
        *,
        timeframe: str,
        requested_timeframes: list[str] | None,
        start_date: pd.Timestamp | None,
        end_date: pd.Timestamp | None,
    ) -> list[str]:
        """Select a database-first training universe before loading full OHLCV panels."""
        from sqlalchemy import bindparam

        requested_timeframe_scope = [
            normalize_timeframe(value, default=timeframe)
            for value in (requested_timeframes or [timeframe])
            if str(value).strip()
        ]
        if not requested_timeframe_scope:
            requested_timeframe_scope = [normalize_timeframe(timeframe)]
        requested_timeframe_scope = list(dict.fromkeys(requested_timeframe_scope))

        cache_key = (
            f"train:universe:{timeframe}:"
            f"{','.join(requested_timeframe_scope)}:"
            f"{self.config.target_universe_size}:{self.config.universe_selection_buffer_size}:"
            f"{start_date.isoformat() if start_date is not None else 'none'}:"
            f"{end_date.isoformat() if end_date is not None else 'none'}"
        )
        if redis_mgr is not None:
            cached_raw = redis_mgr.get(cache_key)
            if cached_raw:
                try:
                    cached_symbols = [
                        str(symbol).strip().upper()
                        for symbol in json.loads(cached_raw)
                        if str(symbol).strip()
                    ]
                    if cached_symbols:
                        self.training_metrics["database_universe_requested_timeframes"] = list(
                            requested_timeframe_scope
                        )
                        self.training_metrics["database_universe_selection_source"] = "redis_cache"
                        return cached_symbols
                except json.JSONDecodeError:
                    pass

        target_size = int(self.config.target_universe_size)
        if target_size <= 0:
            result = session.execute(
                text("""
                    SELECT DISTINCT symbol
                    FROM ohlcv_bars
                    WHERE timeframe = :timeframe
                      AND (:start_date IS NULL OR timestamp >= :start_date)
                      AND (:end_date IS NULL OR timestamp <= :end_date)
                    ORDER BY symbol
                    """),
                {
                    "timeframe": timeframe,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
            symbols = [
                str(row[0]).strip().upper() for row in result.fetchall() if str(row[0]).strip()
            ]
            self.training_metrics["database_universe_requested_timeframes"] = list(
                requested_timeframe_scope
            )
            self.training_metrics["database_universe_selection_source"] = "full_timeframe_scan"
            return symbols

        candidate_limit = max(
            target_size,
            target_size + int(self.config.universe_selection_buffer_size),
        )
        result = session.execute(
            text("""
                WITH base_metrics AS (
                    SELECT
                        symbol,
                        COUNT(*) AS row_count,
                        COUNT(DISTINCT DATE(timestamp)) AS active_days,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (
                            ORDER BY (CAST(close AS DOUBLE PRECISION) * CAST(volume AS DOUBLE PRECISION))
                        ) AS median_dollar_volume
                    FROM ohlcv_bars
                    WHERE timeframe = :timeframe
                      AND (:start_date IS NULL OR timestamp >= :start_date)
                      AND (:end_date IS NULL OR timestamp <= :end_date)
                    GROUP BY symbol
                    HAVING COUNT(*) >= :min_rows
                ),
                timeframe_coverage AS (
                    SELECT
                        symbol,
                        COUNT(DISTINCT timeframe) AS timeframe_coverage_count
                    FROM ohlcv_bars
                    WHERE timeframe IN :requested_timeframes
                      AND (:start_date IS NULL OR timestamp >= :start_date)
                      AND (:end_date IS NULL OR timestamp <= :end_date)
                    GROUP BY symbol
                )
                SELECT
                    base_metrics.symbol,
                    base_metrics.row_count,
                    base_metrics.active_days,
                    base_metrics.median_dollar_volume,
                    COALESCE(timeframe_coverage.timeframe_coverage_count, 0) AS timeframe_coverage_count
                FROM base_metrics
                LEFT JOIN timeframe_coverage
                    ON timeframe_coverage.symbol = base_metrics.symbol
                ORDER BY
                    timeframe_coverage_count DESC,
                    median_dollar_volume DESC NULLS LAST,
                    row_count DESC,
                    active_days DESC,
                    symbol ASC
                LIMIT :candidate_limit
                """).bindparams(bindparam("requested_timeframes", expanding=True)),
            {
                "timeframe": timeframe,
                "requested_timeframes": requested_timeframe_scope,
                "start_date": start_date,
                "end_date": end_date,
                "min_rows": int(self.config.symbol_quality_min_rows),
                "candidate_limit": candidate_limit,
            },
        )
        candidate_rows = result.fetchall()
        symbols = [
            str(row[0]).strip().upper() for row in candidate_rows if row and str(row[0]).strip()
        ]
        full_timeframe_coverage_count = sum(
            1
            for row in candidate_rows
            if row and int(row[4] or 0) >= len(requested_timeframe_scope)
        )

        preferred_reference_symbols = (
            list(self.config.universe_reference_symbols)
            if (self.config.enable_cross_sectional or self.config.enable_reference_features)
            else []
        )
        if preferred_reference_symbols:
            reference_result = session.execute(
                text("""
                    SELECT DISTINCT symbol
                    FROM ohlcv_bars
                    WHERE timeframe = :timeframe
                      AND symbol IN :symbols
                      AND (:start_date IS NULL OR timestamp >= :start_date)
                      AND (:end_date IS NULL OR timestamp <= :end_date)
                    ORDER BY symbol
                    """).bindparams(bindparam("symbols", expanding=True)),
                {
                    "timeframe": timeframe,
                    "symbols": preferred_reference_symbols,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
            for row in reference_result.fetchall():
                symbol = str(row[0]).strip().upper()
                if symbol and symbol not in symbols:
                    symbols.append(symbol)

        if redis_mgr is not None and symbols:
            redis_mgr.set(cache_key, json.dumps(symbols), expire_seconds=300)

        self.training_metrics["database_universe_requested_timeframes"] = list(
            requested_timeframe_scope
        )
        self.training_metrics["database_universe_target_size"] = float(target_size)
        self.training_metrics["database_universe_candidate_count"] = float(len(candidate_rows))
        self.training_metrics["database_universe_full_timeframe_coverage_count"] = float(
            full_timeframe_coverage_count
        )
        self.training_metrics["database_universe_selected_symbols"] = list(symbols)
        self.training_metrics["database_universe_selection_source"] = "liquidity_ranked"
        return symbols

    def _load_ohlcv_panel_from_postgres(
        self,
        session: Any,
        *,
        symbols: list[str],
        timeframe: str,
        start_date: pd.Timestamp | None,
        end_date: pd.Timestamp | None,
    ) -> pd.DataFrame:
        """Bulk-load one timeframe panel from PostgreSQL for the selected universe."""
        from sqlalchemy import bindparam

        if not symbols:
            return pd.DataFrame(
                columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"]
            )

        query = text("""
            SELECT
                symbol,
                timestamp,
                CAST(open AS DOUBLE PRECISION) AS open,
                CAST(high AS DOUBLE PRECISION) AS high,
                CAST(low AS DOUBLE PRECISION) AS low,
                CAST(close AS DOUBLE PRECISION) AS close,
                CAST(volume AS BIGINT) AS volume
            FROM ohlcv_bars
            WHERE symbol IN :symbols
              AND timeframe = :timeframe
              AND (:start_date IS NULL OR timestamp >= :start_date)
              AND (:end_date IS NULL OR timestamp <= :end_date)
            ORDER BY symbol, timestamp
            """).bindparams(bindparam("symbols", expanding=True))

        result = session.execute(
            query,
            {
                "symbols": [
                    str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()
                ],
                "timeframe": normalize_timeframe(timeframe),
                "start_date": start_date,
                "end_date": end_date,
            },
        )
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame(columns=result.keys())

        frame = pd.DataFrame(rows, columns=result.keys())
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        return frame.dropna(subset=["timestamp"]).reset_index(drop=True)

    def _load_data_from_postgres(self) -> None:
        """Load OHLCV data from PostgreSQL/TimescaleDB."""
        from sqlalchemy import text
        from quant_trading_system.database.connection import get_db_manager, get_redis_manager

        db_manager = get_db_manager()
        symbols = [s.strip().upper() for s in self.config.symbols if s.strip()]
        timeframe = normalize_timeframe(self.config.timeframe)
        if not self.config.use_redis_cache:
            raise RuntimeError(
                "Institutional training mode requires Redis cache layer. " "Disable is not allowed."
            )

        redis_mgr = get_redis_manager()
        if not redis_mgr.health_check():
            raise RuntimeError("Redis cache unavailable. Institutional mode requires Redis.")

        with db_manager.session() as session:
            session.execute(text("SET LOCAL max_parallel_workers_per_gather = 0"))
            session.execute(text("SET LOCAL statement_timeout = '120000ms'"))
            start_date = (
                pd.to_datetime(self.config.start_date, utc=True) if self.config.start_date else None
            )
            end_date = (
                pd.to_datetime(self.config.end_date, utc=True) if self.config.end_date else None
            )
            requested_timeframes = list(self.config.timeframes) or [timeframe]
            if not symbols:
                symbols = self._select_candidate_symbols_from_postgres(
                    session,
                    redis_mgr,
                    timeframe=timeframe,
                    requested_timeframes=requested_timeframes,
                    start_date=start_date,
                    end_date=end_date,
                )

            if not symbols:
                raise ValueError("No symbols found in ohlcv_bars")

            self.logger.info(
                "Loading %d symbols from PostgreSQL (%s)",
                len(symbols),
                ", ".join(requested_timeframes),
            )

            timeframe_panels: dict[str, pd.DataFrame] = {}
            for requested_timeframe in requested_timeframes:
                try:
                    panel = self._load_ohlcv_panel_from_postgres(
                        session,
                        symbols=symbols,
                        timeframe=requested_timeframe,
                        start_date=start_date,
                        end_date=end_date,
                    )
                except Exception as e:
                    self.logger.warning(
                        "Failed to load timeframe %s from PostgreSQL: %s",
                        requested_timeframe,
                        e,
                    )
                    continue
                if panel.empty:
                    self.logger.warning(
                        "No PostgreSQL rows for requested timeframe %s", requested_timeframe
                    )
                    continue
                timeframe_panels[normalize_timeframe(requested_timeframe)] = panel

            if timeframe not in timeframe_panels:
                raise ValueError("No training data loaded from PostgreSQL for base timeframe")

            self.ohlcv_panels_by_timeframe = timeframe_panels
            self.data = timeframe_panels[timeframe].copy()
            self.logger.info(
                f"Loaded {len(self.data)} rows from PostgreSQL for "
                f"{self.data['symbol'].nunique()} symbols ({timeframe})"
            )
            if len(timeframe_panels) > 1:
                self.training_metrics["database_timeframes_loaded"] = sorted(
                    list(timeframe_panels.keys())
                )
                self.training_metrics["database_timeframe_row_counts"] = {
                    tf_name: float(len(panel)) for tf_name, panel in timeframe_panels.items()
                }

    def _load_data_fallback(self) -> None:
        """CSV fallback is forbidden in institutional mode."""
        raise RuntimeError(
            "Institutional training mode requires PostgreSQL + Redis. CSV fallback is disabled."
        )

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

    def _record_data_quality_metrics(self) -> None:
        """Flatten the active data-quality report into stable training metrics."""
        if not isinstance(self.data_quality_report, dict) or not self.data_quality_report:
            return

        summary = self.data_quality_report.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}

        self.training_metrics["data_quality_report_passed"] = bool(
            self.data_quality_report.get("passed", False)
        )
        self.training_metrics["data_quality_symbol_count"] = float(summary.get("symbol_count", 0.0))
        self.training_metrics["data_quality_missing_bars_count"] = float(
            summary.get("missing_bars_count", 0.0)
        )
        self.training_metrics["data_quality_missing_bars_ratio"] = float(
            summary.get("missing_bars_ratio", 0.0)
        )
        self.training_metrics["data_quality_duplicate_bars_count"] = float(
            summary.get("duplicate_bars_count", 0.0)
        )
        self.training_metrics["data_quality_duplicate_bars_ratio"] = float(
            summary.get("duplicate_bars_ratio", 0.0)
        )
        self.training_metrics["data_quality_symbols_with_missing_bars"] = float(
            summary.get("symbols_with_missing_bars", 0.0)
        )
        self.training_metrics["data_quality_top_missing_bar_symbols"] = self._json_safe(
            self.data_quality_report.get("top_missing_bar_symbols", [])
        )
        self.training_metrics["data_quality_top_missing_bar_windows"] = self._json_safe(
            self.data_quality_report.get("top_missing_bar_windows", [])
        )
        if self.data_quality_report_hash:
            self.training_metrics["data_quality_report_hash"] = str(self.data_quality_report_hash)

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
        self._record_data_quality_metrics()

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
            top_symbols = self.data_quality_report.get("top_missing_bar_symbols", [])
            if isinstance(top_symbols, list) and top_symbols:
                symbol_preview = ", ".join(
                    f"{entry.get('symbol')}:{int(entry.get('missing_bars_count', 0))}"
                    for entry in top_symbols[:5]
                    if isinstance(entry, dict)
                )
                if symbol_preview:
                    self.logger.warning("Top missing-bar symbols: %s", symbol_preview)
            top_windows = self.data_quality_report.get("top_missing_bar_windows", [])
            if isinstance(top_windows, list) and top_windows:
                first_window = top_windows[0]
                if isinstance(first_window, dict):
                    self.logger.warning(
                        "Largest missing-bar window: %s %s -> %s (missing=%s)",
                        first_window.get("symbol", "unknown"),
                        first_window.get("gap_start", "n/a"),
                        first_window.get("gap_end", "n/a"),
                        first_window.get("estimated_missing_bars", 0),
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

    def _restore_training_state_from_split_frames(
        self,
        dev_frame: pd.DataFrame,
        holdout_frame: pd.DataFrame | None,
        label_horizons: list[int],
        dev_weights: np.ndarray | None,
        holdout_weights: np.ndarray | None,
    ) -> None:
        """Restore trainer matrices from split frames for live runs and snapshot replay."""
        dev_frame = dev_frame.reset_index(drop=True).copy()
        holdout_frame = (
            holdout_frame.reset_index(drop=True).copy()
            if holdout_frame is not None and not holdout_frame.empty
            else None
        )
        if dev_frame.empty:
            raise ValueError("Development frame is empty; cannot restore training state.")

        primary_horizon = int(self.config.primary_label_horizon)
        primary_forward_col = f"forward_return_h{primary_horizon}"
        target_mode = "classification_label"

        if self._is_regression_model():
            target_mode = "return_regression"
            if "triple_barrier_net_return" in dev_frame.columns:
                dev_target = pd.to_numeric(dev_frame["triple_barrier_net_return"], errors="coerce")
                holdout_target = (
                    pd.to_numeric(holdout_frame["triple_barrier_net_return"], errors="coerce")
                    if holdout_frame is not None
                    and "triple_barrier_net_return" in holdout_frame.columns
                    else pd.Series(dtype=float)
                )
                target_mode = "triple_barrier_net_return"
            elif primary_forward_col in dev_frame.columns:
                dev_target = pd.to_numeric(dev_frame[primary_forward_col], errors="coerce")
                holdout_target = (
                    pd.to_numeric(holdout_frame[primary_forward_col], errors="coerce")
                    if holdout_frame is not None and primary_forward_col in holdout_frame.columns
                    else pd.Series(dtype=float)
                )
                target_mode = primary_forward_col
            else:
                raise ValueError(
                    "Regression challenger requires return target column "
                    "(triple_barrier_net_return or primary forward return)."
                )

            dev_target = dev_target.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            dev_target = dev_target.clip(lower=-0.50, upper=0.50).astype(float)
            self.labels = dev_target.reset_index(drop=True)
            if len(self.labels) < 50:
                raise ValueError("Regression target produced insufficient development samples.")
            self.training_metrics["target_mode"] = target_mode
            self.training_metrics["target_mean"] = float(np.mean(self.labels))
            self.training_metrics["target_std"] = float(np.std(self.labels))
        else:
            self.labels = dev_frame["label"].astype(int)
            if int(self.labels.nunique(dropna=True)) < 2:
                pos_rate = float(np.mean(pd.to_numeric(self.labels, errors="coerce").fillna(0.0)))
                raise ValueError(
                    "Label generation produced a single class in development set "
                    f"(pos_rate={pos_rate:.2%}). Adjust label horizons/thresholds before training."
                )
            self.training_metrics["target_mode"] = target_mode

        self.timestamps = dev_frame["timestamp"].to_numpy()
        self.row_symbols = dev_frame["symbol"].astype(str).to_numpy()
        self.regimes = (
            dev_frame["regime"].astype(str).to_numpy()
            if "regime" in dev_frame.columns
            else np.full(len(dev_frame), "normal_range", dtype=object)
        )
        self.primary_event_directions = (
            np.nan_to_num(
                pd.to_numeric(dev_frame["primary_signal"], errors="coerce").to_numpy(dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if "primary_signal" in dev_frame.columns
            else None
        )
        self.close_prices = dev_frame["close"].reset_index(drop=True)
        if primary_forward_col in dev_frame.columns:
            self.primary_forward_returns = np.nan_to_num(
                pd.to_numeric(dev_frame[primary_forward_col], errors="coerce").to_numpy(
                    dtype=float
                ),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        else:
            self.primary_forward_returns = np.zeros(len(dev_frame), dtype=float)
        if "triple_barrier_net_return" in dev_frame.columns:
            self.cost_aware_event_returns = np.nan_to_num(
                pd.to_numeric(dev_frame["triple_barrier_net_return"], errors="coerce").to_numpy(
                    dtype=float
                ),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        else:
            self.cost_aware_event_returns = None

        self.development_frame = dev_frame.copy()
        if dev_weights is None or len(dev_weights) != len(dev_frame):
            dev_weights = np.ones(len(dev_frame), dtype=float)
        self.sample_weights = np.asarray(dev_weights, dtype=float)

        self.holdout_features = None
        self.holdout_labels = None
        self.holdout_close_prices = None
        self.holdout_timestamps = None
        self.holdout_symbols = None
        self.holdout_regimes = None
        self.holdout_primary_event_directions = None
        self.holdout_primary_forward_returns = None
        self.holdout_cost_aware_event_returns = None
        self.holdout_sample_weights = None
        self.holdout_frame = holdout_frame.copy() if holdout_frame is not None else None

        if holdout_frame is not None and not holdout_frame.empty:
            if self._is_regression_model():
                holdout_target = holdout_target.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                self.holdout_labels = (
                    holdout_target.clip(lower=-0.50, upper=0.50)
                    .astype(float)
                    .reset_index(drop=True)
                )
            else:
                self.holdout_labels = holdout_frame["label"].astype(int).reset_index(drop=True)
            self.holdout_close_prices = (
                holdout_frame["close"].reset_index(drop=True)
                if "close" in holdout_frame.columns
                else None
            )
            self.holdout_timestamps = holdout_frame["timestamp"].to_numpy()
            self.holdout_symbols = holdout_frame["symbol"].astype(str).to_numpy()
            self.holdout_regimes = (
                holdout_frame["regime"].astype(str).to_numpy()
                if "regime" in holdout_frame.columns
                else np.full(len(holdout_frame), "normal_range", dtype=object)
            )
            self.holdout_primary_event_directions = (
                np.nan_to_num(
                    pd.to_numeric(holdout_frame["primary_signal"], errors="coerce").to_numpy(
                        dtype=float
                    ),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                if "primary_signal" in holdout_frame.columns
                else None
            )
            if primary_forward_col in holdout_frame.columns:
                self.holdout_primary_forward_returns = np.nan_to_num(
                    pd.to_numeric(holdout_frame[primary_forward_col], errors="coerce").to_numpy(
                        dtype=float
                    ),
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
                self.holdout_cost_aware_event_returns = None

            if holdout_weights is None or len(holdout_weights) != len(holdout_frame):
                holdout_weights = np.ones(len(holdout_frame), dtype=float)
            self.holdout_sample_weights = np.asarray(holdout_weights, dtype=float)
            self.training_metrics["holdout_rows"] = float(len(holdout_frame))
            self.training_metrics["holdout_weight_mean"] = (
                float(np.mean(self.holdout_sample_weights))
                if len(self.holdout_sample_weights)
                else 1.0
            )

        self.training_metrics["development_rows"] = float(len(dev_frame))

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
            "signal_volatility",
        ]
        exclude_cols.extend([f"forward_return_h{h}" for h in label_horizons])
        feature_cols = [c for c in dev_frame.columns if c not in exclude_cols]
        if not feature_cols:
            raise ValueError("No model features remain after target engineering exclusion list.")
        self.features = dev_frame[feature_cols]
        if holdout_frame is not None and not holdout_frame.empty:
            self.holdout_features = holdout_frame[feature_cols].reset_index(drop=True)
        self.feature_names = feature_cols
        self._apply_model_priors()

    def _hydrate_snapshot_symbol_quality_metrics(self) -> None:
        """Backfill symbol-quality audit fields when replaying an approved snapshot bundle."""
        symbol_universe: list[str] = []
        if isinstance(self.snapshot_manifest, dict):
            symbol_universe = [
                str(symbol).strip().upper()
                for symbol in self.snapshot_manifest.get("symbol_universe", [])
                if str(symbol).strip()
            ]
        if not symbol_universe and isinstance(self.dataset_snapshot_bundle_manifest, dict):
            training_scope = self.dataset_snapshot_bundle_manifest.get("training_scope", {})
            if isinstance(training_scope, dict):
                symbol_universe = [
                    str(symbol).strip().upper()
                    for symbol in training_scope.get("symbols", [])
                    if str(symbol).strip()
                ]
        if not symbol_universe:
            return

        normalized_universe = sorted(set(symbol_universe))
        if float(self.training_metrics.get("symbol_quality_input_symbols", 0.0) or 0.0) <= 0.0:
            self.training_metrics["symbol_quality_input_symbols"] = float(len(normalized_universe))
        if float(self.training_metrics.get("symbol_quality_selected_symbols", 0.0) or 0.0) <= 0.0:
            self.training_metrics["symbol_quality_selected_symbols"] = float(
                len(normalized_universe)
            )
        self.training_metrics.setdefault("symbol_quality_dropped_symbols", 0.0)
        self.training_metrics.setdefault("symbol_quality_dropped_list", [])
        if not self.training_metrics.get("symbol_quality_universe"):
            self.training_metrics["symbol_quality_universe"] = normalized_universe
        if not self.training_metrics.get("symbol_quality_report"):
            self.training_metrics["symbol_quality_report"] = {
                symbol: {"passes": True, "source": "snapshot_bundle_replay"}
                for symbol in normalized_universe
            }

    def _load_training_dataset_snapshot(self) -> bool:
        """Load immutable dataset snapshot bundle for deterministic replay."""
        bundle_path = Path(self.config.dataset_snapshot_bundle_path)
        if not bundle_path.exists():
            if self.config.strict_snapshot_replay:
                raise FileNotFoundError(f"Dataset snapshot bundle not found: {bundle_path}")
            self.logger.warning(
                "Dataset snapshot bundle missing (%s); falling back to live data load.",
                bundle_path,
            )
            return False

        try:
            bundle_manifest = json.loads(bundle_path.read_text(encoding="utf-8"))
        except Exception as exc:
            if self.config.strict_snapshot_replay:
                raise ValueError(
                    f"Failed to parse dataset snapshot bundle manifest {bundle_path}: {exc}"
                ) from exc
            self.logger.warning(
                "Failed to parse dataset snapshot bundle manifest (%s); falling back to live data load.",
                bundle_path,
            )
            self.training_metrics["snapshot_bundle_scope_validated"] = False
            self.training_metrics["snapshot_bundle_scope_issues"] = [
                f"manifest_parse_error: {exc}"
            ]
            return False

        bundle_scope_issues = (
            _validate_snapshot_bundle_manifest_scope(bundle_manifest, self.config)
            if isinstance(bundle_manifest, dict)
            else ["bundle manifest is not a JSON object"]
        )
        if bundle_scope_issues:
            summary = "; ".join(bundle_scope_issues[:6])
            message = (
                "Dataset snapshot bundle does not match the requested training scope "
                f"({bundle_path}): {summary}"
            )
            self.training_metrics["snapshot_bundle_scope_validated"] = False
            self.training_metrics["snapshot_bundle_scope_issues"] = list(bundle_scope_issues)
            if self.config.strict_snapshot_replay:
                raise ValueError(
                    message
                    + ". Rebuild a scope-matched bundle with --snapshot-only and rerun."
                )
            self.logger.warning("%s; rebuilding from live data instead.", message)
            return False

        self.training_metrics["snapshot_bundle_scope_validated"] = True
        self.training_metrics["snapshot_bundle_scope_issues"] = []
        bundle = load_dataset_snapshot_bundle(bundle_path)
        development_frame = bundle.get("development_frame")
        if development_frame is None or development_frame.empty:
            raise ValueError(
                "Dataset snapshot bundle is missing development_frame artifact for replay."
            )

        self.logger.info("Phase 1-3: Loading immutable training snapshot bundle...")
        self.snapshot_replay_loaded = True
        self.dataset_snapshot_bundle_manifest = bundle.get("bundle_manifest", {})
        self.dataset_snapshot_bundle_manifest_path = Path(
            bundle.get("bundle_manifest_path", bundle_path)
        )
        self.snapshot_manifest = bundle.get("snapshot_manifest", {}) or None
        if isinstance(self.snapshot_manifest, dict):
            manifest_path = self.snapshot_manifest.get("manifest_path")
            quality_path = self.snapshot_manifest.get("quality_report_path")
            self.snapshot_manifest_path = Path(manifest_path) if manifest_path else None
            self.data_quality_report_path = Path(quality_path) if quality_path else None
            self.data_quality_report_hash = self.snapshot_manifest.get("data_quality_report_hash")

        self.data = bundle.get("raw_ohlcv_data")
        self.data_quality_report = bundle.get("data_quality_report") or None
        if (
            self.data_quality_report is None
            and self.data_quality_report_path is not None
            and self.data_quality_report_path.exists()
        ):
            try:
                self.data_quality_report = json.loads(
                    self.data_quality_report_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                if self.config.strict_snapshot_replay:
                    raise ValueError(
                        f"Failed to load data quality report from snapshot bundle: {exc}"
                    ) from exc
        if self.data_quality_report_hash is None and self.data_quality_report is not None:
            self.data_quality_report_hash = compute_data_quality_hash(self.data_quality_report)
        self._record_data_quality_metrics()
        self.label_diagnostics = bundle.get("label_diagnostics", {}) or {}
        if self.label_diagnostics:
            self.training_metrics["label_positive_rate"] = float(
                self.label_diagnostics.get("positive_rate", 0.0)
            )
            self.training_metrics["label_class_balance_ratio"] = float(
                self.label_diagnostics.get("class_balance_ratio", 0.0)
            )
            self.training_metrics["label_drift_abs"] = float(
                self.label_diagnostics.get("label_drift_abs", 0.0)
            )
            self.training_metrics["label_count"] = float(
                self.label_diagnostics.get("label_count", 0.0)
            )
            self.training_metrics["label_forward_outlier_filtered_count"] = float(
                self.label_diagnostics.get("forward_outlier_filtered_count", 0.0)
            )
            self.training_metrics["label_forward_outlier_filtered_rate"] = float(
                self.label_diagnostics.get("forward_outlier_filtered_rate", 0.0)
            )
            self.training_metrics["label_neutral_filtered_count"] = float(
                self.label_diagnostics.get("neutral_filtered_count", 0.0)
            )

        label_horizons = sorted(
            {
                int(h)
                for h in self.config.label_horizons
                if isinstance(h, (int, np.integer, float, np.floating)) and int(h) > 0
            }
        )
        if not label_horizons:
            label_horizons = [1, 5, 20]
        if int(self.config.primary_label_horizon) not in label_horizons:
            label_horizons.append(int(self.config.primary_label_horizon))
            label_horizons = sorted(set(label_horizons))

        self.cached_cv_splits = bundle.get("cv_splits", []) or []
        self.training_metrics["snapshot_bundle_reused"] = True
        self.training_metrics["snapshot_bundle_cv_split_count"] = float(len(self.cached_cv_splits))
        self.training_metrics["snapshot_bundle_path"] = str(
            self.dataset_snapshot_bundle_manifest_path
        )
        self._hydrate_snapshot_feature_selection_summary()
        self._hydrate_snapshot_symbol_quality_metrics()

        self._restore_training_state_from_split_frames(
            dev_frame=development_frame,
            holdout_frame=bundle.get("holdout_frame"),
            label_horizons=label_horizons,
            dev_weights=bundle.get("development_sample_weights"),
            holdout_weights=bundle.get("holdout_sample_weights"),
        )
        self.logger.info(
            "Loaded immutable dataset snapshot %s: dev_rows=%d holdout_rows=%d",
            (
                self.snapshot_manifest.get("snapshot_id")
                if isinstance(self.snapshot_manifest, dict)
                else "unknown"
            ),
            len(self.features) if self.features is not None else 0,
            len(self.holdout_features) if self.holdout_features is not None else 0,
        )
        return True

    def _compute_features(self) -> None:
        """Phase 2: Compute features for model training."""
        self.logger.info("Phase 2: Computing features...")
        self.feature_cache_reused = False
        self.features_materialized_in_run = False
        self.features_persisted_to_postgres = False
        # 1) Reuse previously computed features from PostgreSQL when available.
        cached = self._load_features_from_postgres()
        if cached is not None and not cached.empty:
            self.features = cached
            self.feature_cache_reused = True
            self.training_metrics["feature_gpu_contract"] = {
                "feature_compute_skipped": True,
                "source": "postgres_feature_cache",
                "gpu_requested": bool(self.config.require_gpu),
                "model_gpu_enabled": bool(self.config.use_gpu),
                "model_gpu_required": bool(self.config.require_gpu),
                "feature_gpu_required": bool(self.config.require_feature_gpu),
                "cudf_available": None,
                "fully_gpu_ready_groups": [],
                "partially_gpu_accelerated_groups": [],
                "gpu_accelerated_groups": [],
                "cpu_only_groups": [],
                "cpu_materialized_groups": [],
                "cpu_optimized_groups": [],
                "partial_gpu_feature_names": {},
                "fully_gpu_ready": None,
                "acceleration_mode": "unknown_cached",
            }
            self.training_metrics["feature_gpu_fully_ready"] = float("nan")
            self.logger.info("Using feature matrix loaded from PostgreSQL features table")
            return

        # 2) Compute full institutional feature set.
        if os.name == "nt" and self.config.windows_force_fallback_features:
            self.logger.warning(
                "Using fallback feature computation on Windows because "
                "--windows-fallback-features is enabled."
            )
            self._compute_features_fallback()
            self.features_materialized_in_run = bool(
                self.features is not None and not self.features.empty
            )
            return
        if os.name == "nt":
            self.logger.info("Windows runtime detected: enforcing full feature pipeline.")
            try:
                self._compute_features_full_pipeline()
            except Exception as exc:
                if self.config.allow_feature_group_fallback:
                    self.logger.warning(
                        "Full feature pipeline failed on Windows (%s); "
                        "allowing deterministic basic fallback because "
                        "--allow-partial-feature-fallback is enabled.",
                        exc,
                    )
                    self._compute_features_fallback()
                    self.features_materialized_in_run = bool(
                        self.features is not None and not self.features.empty
                    )
                    return
                raise RuntimeError(
                    "Full feature pipeline failed on Windows and partial fallback is disabled. "
                    "Use --allow-partial-feature-fallback only for emergency continuity, "
                    "or fix the feature pipeline failure."
                ) from exc
        else:
            self._compute_features_full_pipeline()

        self.features_materialized_in_run = bool(
            self.features is not None and not self.features.empty
        )

    def _record_feature_gpu_contract(
        self,
        *,
        groups_to_compute: list[Any],
        disable_optimized_technical: bool,
    ) -> None:
        """Record auditable feature GPU readiness and fail fast when strict GPU mode is impossible."""
        from quant_trading_system.features.feature_pipeline import describe_feature_gpu_readiness

        readiness = describe_feature_gpu_readiness(
            groups_to_compute,
            use_gpu=bool(self.config.use_gpu),
            use_optimized_pipeline=not disable_optimized_technical,
            disable_optimized_technical=disable_optimized_technical,
        )
        gpu_accelerated_groups = list(
            dict.fromkeys(
                readiness["fully_gpu_ready_groups"]
                + readiness["partially_gpu_accelerated_groups"]
            )
        )
        cpu_only_groups = [
            group
            for group in readiness["requested_groups"]
            if group not in gpu_accelerated_groups
        ]
        fully_gpu_ready = bool(readiness["fully_gpu_ready"])
        payload = {
            "feature_compute_skipped": False,
            "source": "materialized_in_run",
            "gpu_requested": bool(self.config.require_gpu),
            "model_gpu_enabled": bool(self.config.use_gpu),
            "model_gpu_required": bool(self.config.require_gpu),
            "feature_gpu_required": bool(self.config.require_feature_gpu),
            "cudf_available": bool(readiness["cudf_available"]),
            "groups_to_compute": list(readiness["requested_groups"]),
            "fully_gpu_ready_groups": list(readiness["fully_gpu_ready_groups"]),
            "partially_gpu_accelerated_groups": list(
                readiness["partially_gpu_accelerated_groups"]
            ),
            "partial_gpu_feature_names": dict(readiness["partial_gpu_feature_names"]),
            "gpu_accelerated_groups": list(gpu_accelerated_groups),
            "cpu_only_groups": list(cpu_only_groups),
            "cpu_materialized_groups": list(readiness["cpu_materialized_groups"]),
            "cpu_optimized_groups": list(readiness["cpu_optimized_groups"]),
            "fully_gpu_ready": bool(fully_gpu_ready),
            "acceleration_mode": str(readiness["acceleration_mode"]),
        }
        self.training_metrics["feature_gpu_contract"] = payload
        self.training_metrics["feature_gpu_fully_ready"] = 1.0 if fully_gpu_ready else 0.0
        self.training_metrics["feature_gpu_cudf_available"] = (
            1.0 if readiness["cudf_available"] else 0.0
        )
        if self.config.require_feature_gpu and not fully_gpu_ready:
            reason = "requested feature groups still include families without complete CUDA ports."
            if not readiness["cudf_available"]:
                reason = (
                    "cuDF is unavailable, so even the partial optimized technical path cannot run."
                )
            raise RuntimeError(
                "Feature GPU contract failed while --require-feature-gpu was set: "
                f"{reason} Requested groups={', '.join(readiness['requested_groups'])}; "
                f"Fully GPU-ready groups={', '.join(readiness['fully_gpu_ready_groups']) or 'none'}; "
                f"Partially GPU-accelerated groups="
                f"{', '.join(readiness['partially_gpu_accelerated_groups']) or 'none'}; "
                f"CPU-materialized groups="
                f"{', '.join(readiness['cpu_materialized_groups']) or 'none'}."
            )

    def _persist_features_to_postgres_if_needed(self) -> None:
        """Persist only freshly materialized post-selection features."""
        if self.config.snapshot_only:
            self.logger.info(
                "Skipping feature persistence because snapshot-only mode only needs bundle artifacts"
            )
            return
        if not self.config.persist_features_to_postgres:
            self.logger.info("Skipping feature persistence to PostgreSQL for this run")
            return
        if self.snapshot_replay_loaded:
            self.logger.info(
                "Skipping feature persistence because snapshot replay supplied features"
            )
            return
        if self.feature_cache_reused:
            self.logger.info(
                "Skipping feature persistence because PostgreSQL feature cache was reused"
            )
            return
        if not self.features_materialized_in_run:
            self.logger.info(
                "Skipping feature persistence because no new feature matrix was materialized"
            )
            return
        self._store_features_to_postgres()

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

    def _annualization_periods(self) -> int:
        """Return timeframe-aware annualization periods for intraday metrics."""
        return int(estimate_periods_per_year(self.config.timeframe))

    def _is_ranker_model(self) -> bool:
        """Return True when current model type uses ranking objective/training."""
        return str(self.config.model_type).strip().lower() in RANKING_MODELS

    def _is_regression_model(self) -> bool:
        """Return True when current model type predicts continuous returns."""
        return str(self.config.model_type).strip().lower() in REGRESSION_MODELS

    def _requires_binary_class_support(self) -> bool:
        """Return True when fold training requires both binary classes present."""
        return not self._is_regression_model()

    def _is_lightgbm_family_model(self) -> bool:
        """Return True for lightgbm classifier/ranker variants."""
        return str(self.config.model_type).strip().lower() in LIGHTGBM_FAMILY_MODELS

    @staticmethod
    def _build_ranking_groups(timestamps: np.ndarray | list[Any] | None) -> np.ndarray:
        """Build contiguous query-group sizes for ranker training."""
        if timestamps is None:
            return np.array([], dtype=int)
        ts = pd.to_datetime(np.asarray(timestamps), utc=True, errors="coerce")
        if len(ts) == 0:
            return np.array([], dtype=int)
        if np.asarray(pd.isna(ts)).any():
            return np.array([len(ts)], dtype=int)
        ts_arr = np.asarray(ts, dtype="datetime64[ns]")
        change_points = np.flatnonzero(ts_arr[1:] != ts_arr[:-1]) + 1
        boundaries = np.concatenate(([0], change_points, [len(ts_arr)]))
        groups = np.diff(boundaries).astype(int)
        groups = groups[groups > 0]
        if int(np.sum(groups)) != len(ts_arr):
            return np.array([len(ts_arr)], dtype=int)
        return groups

    @staticmethod
    def _rank_scores_to_unit_interval(values: np.ndarray | list[float]) -> np.ndarray:
        """Map arbitrary scores into a probability-like unit interval using average ranks."""
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

        normalized[finite_mask] = np.clip(
            (ranks + 0.5) / float(sorted_values.size),
            0.0,
            1.0,
        )
        return normalized

    def _normalize_ranker_scores(
        self,
        raw_scores: np.ndarray | list[float],
        timestamps: np.ndarray | list[Any] | None = None,
    ) -> np.ndarray:
        """Convert ranker scores into comparable unit-interval ranks."""
        scores = np.asarray(raw_scores, dtype=float).reshape(-1)
        if scores.size == 0:
            return scores.astype(float)

        normalization_mode = "global_percentile"
        normalized: np.ndarray | None = None
        if timestamps is not None:
            timestamp_values = np.asarray(timestamps)
            if timestamp_values.size == scores.size:
                groups = self._build_ranking_groups(timestamp_values)
                if (
                    groups.size > 0
                    and int(np.sum(groups)) == scores.size
                    and int(np.max(groups)) > 1
                ):
                    normalized = np.empty(scores.size, dtype=float)
                    start = 0
                    for group_size in groups.tolist():
                        end = start + int(group_size)
                        normalized[start:end] = self._rank_scores_to_unit_interval(
                            scores[start:end]
                        )
                        start = end
                    if start == scores.size:
                        normalization_mode = "query_percentile"
                    else:
                        normalized = None

        if normalized is None:
            normalized = self._rank_scores_to_unit_interval(scores)

        self.training_metrics["ranker_score_normalization"] = normalization_mode
        return normalized

    def _ranker_scoring_contract(self) -> dict[str, Any]:
        """Return the persisted runtime contract required for ranker parity."""
        return {
            "normalization": "query_percentile",
            "query_key": "timestamp",
            "requires_cross_sectional_panel": True,
        }

    def _attach_ranker_runtime_metadata(self, model: Any) -> None:
        """Persist ranker runtime metadata on the fitted estimator for downstream consumers."""
        if not self._is_ranker_model():
            return

        payload = self._ranker_scoring_contract()
        fit_feature_names = [str(name) for name in self.feature_names] if self.feature_names else []
        for candidate in (model, getattr(model, "_model", None)):
            if candidate is None:
                continue
            try:
                candidate._alphatrade_ranker_scoring = payload
            except Exception:
                pass
            if fit_feature_names:
                try:
                    candidate.feature_names = fit_feature_names
                except Exception:
                    pass

    def _prepare_model_input_for_prediction(self, model: Any, X: Any) -> Any:
        """Align prediction inputs to the fitted feature schema when available."""
        fallback_feature_names = (
            [str(name) for name in self.feature_names] if self._is_lightgbm_family_model() else None
        )
        return prepare_model_inference_input(
            model,
            X,
            fallback_feature_names=fallback_feature_names,
        )

    def _predict_with_model(self, model: Any, X: Any) -> Any:
        """Call estimator.predict with schema-aligned inference inputs."""
        return model.predict(self._prepare_model_input_for_prediction(model, X))

    def _predict_proba_with_model(self, model: Any, X: Any) -> Any:
        """Call estimator.predict_proba with schema-aligned inference inputs."""
        return model.predict_proba(self._prepare_model_input_for_prediction(model, X))

    @staticmethod
    def _ranker_eval_at(model: Any) -> list[int]:
        """Resolve eval_at from model params when explicitly configured."""
        cached_eval_at = getattr(model, "_alphatrade_eval_at", None)
        if cached_eval_at is not None:
            if isinstance(cached_eval_at, (list, tuple, set, np.ndarray)):
                resolved = [int(value) for value in cached_eval_at]
            else:
                resolved = [int(cached_eval_at)]
            return [value for value in resolved if value > 0]
        get_params = getattr(model, "get_params", None)
        if not callable(get_params):
            return []
        try:
            params = get_params(deep=False)
        except TypeError:
            params = get_params()
        except Exception:
            return []
        raw_eval_at = params.get("eval_at")
        if raw_eval_at is None:
            return []
        if isinstance(raw_eval_at, (list, tuple, set, np.ndarray)):
            resolved = [int(value) for value in raw_eval_at]
        else:
            resolved = [int(raw_eval_at)]
        return [value for value in resolved if value > 0]

    @staticmethod
    def _summarize_lightgbm_model_structure(model: Any) -> dict[str, Any]:
        """Summarize LightGBM booster structure for dead-model detection."""
        estimator = getattr(model, "_model", model)
        booster = getattr(estimator, "booster_", None)
        summary: dict[str, Any] = {
            "available": False,
            "num_trees": 0.0,
            "split_count": 0.0,
            "nonzero_feature_count": 0.0,
            "has_splits": False,
        }
        if booster is None:
            return summary

        summary["available"] = True
        try:
            split_importance = np.asarray(
                booster.feature_importance(importance_type="split"),
                dtype=float,
            ).reshape(-1)
        except Exception:
            split_importance = np.array([], dtype=float)

        try:
            num_trees = float(booster.num_trees())
        except Exception:
            num_trees = float(len(getattr(estimator, "estimators_", [])))

        split_count = float(np.sum(split_importance)) if split_importance.size > 0 else 0.0
        nonzero_feature_count = (
            float(np.count_nonzero(split_importance > 0.0)) if split_importance.size > 0 else 0.0
        )
        summary.update(
            {
                "num_trees": num_trees,
                "split_count": split_count,
                "nonzero_feature_count": nonzero_feature_count,
                "has_splits": bool(split_count > 0.0),
            }
        )
        return summary

    @staticmethod
    def _is_dead_probability_summary(summary: dict[str, Any] | None) -> bool:
        """Return True when score dispersion has effectively collapsed."""
        if not isinstance(summary, dict):
            return False
        finite_count = int(summary.get("finite_count", summary.get("rows", 0)) or 0)
        std = float(summary.get("std", 0.0) or 0.0)
        span = float(summary.get("span", 0.0) or 0.0)
        return bool(
            finite_count <= 1
            or span <= LIGHTGBM_DEAD_SCORE_SPAN_EPS
            or std <= LIGHTGBM_DEAD_SCORE_STD_EPS
        )

    def _is_dead_lightgbm_model(
        self,
        model: Any,
        *,
        probability_summary: dict[str, Any] | None = None,
    ) -> bool:
        """Return True when LightGBM failed to learn usable structure or score dispersion."""
        if not self._is_lightgbm_family_model():
            return False
        structure_summary = self._summarize_lightgbm_model_structure(model)
        if structure_summary.get("available") and not bool(
            structure_summary.get("has_splits", False)
        ):
            return True
        return self._is_dead_probability_summary(probability_summary)

    def _primary_horizon(self) -> int:
        """Return configured primary horizon used for horizon-specific policies."""
        return max(1, int(getattr(self.config, "primary_label_horizon", 1)))

    def _horizon_threshold_policy(self) -> dict[str, float]:
        """Return thresholding policy calibrated to primary prediction horizon."""
        horizon = self._primary_horizon()
        if horizon <= 1:
            return {
                "target_rate_scale": 0.75,
                "min_gap": 0.06,
                "forced_long_cap": 0.66,
                "forced_short_floor": 0.34,
            }
        if horizon <= 3:
            return {
                "target_rate_scale": 0.85,
                "min_gap": 0.05,
                "forced_long_cap": 0.67,
                "forced_short_floor": 0.33,
            }
        if horizon <= 5:
            return {
                "target_rate_scale": 1.00,
                "min_gap": 0.04,
                "forced_long_cap": 0.68,
                "forced_short_floor": 0.32,
            }
        if horizon <= 10:
            return {
                "target_rate_scale": 1.05,
                "min_gap": 0.035,
                "forced_long_cap": 0.70,
                "forced_short_floor": 0.30,
            }
        return {
            "target_rate_scale": 1.12,
            "min_gap": 0.03,
            "forced_long_cap": 0.72,
            "forced_short_floor": 0.28,
        }

    def _meta_confidence_for_horizon(self) -> float:
        """Derive meta-label confidence floor from primary horizon."""
        base = float(np.clip(self.config.meta_label_min_confidence, 0.45, 0.95))
        horizon = self._primary_horizon()
        if horizon <= 1:
            adj = 0.08
        elif horizon <= 3:
            adj = 0.06
        elif horizon <= 5:
            adj = 0.04
        elif horizon <= 10:
            adj = 0.02
        elif horizon <= 20:
            adj = 0.00
        else:
            adj = -0.02
        return float(np.clip(base + adj, 0.45, 0.95))

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

    @staticmethod
    def _downcast_feature_payload(
        matrix: pd.DataFrame,
        *,
        preserve_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        """Reduce feature-matrix memory footprint before wide-frame operations.

        Training feature matrices are numerically tolerant to float32 precision.
        Keeping feature payloads in float64 causes WSL OOM kills once symbol-level
        frames, augmented matrices, and finalize-time intermediates coexist.
        """
        if matrix is None or matrix.empty:
            return matrix

        preserve = set(preserve_columns or set())
        numeric_cols = [
            column
            for column in matrix.select_dtypes(include=[np.number]).columns
            if column not in preserve
        ]
        for column in numeric_cols:
            series = pd.to_numeric(matrix[column], errors="coerce")
            if pd.api.types.is_float_dtype(series) or series.isna().any():
                matrix[column] = series.astype(np.float32)
            else:
                matrix[column] = pd.to_numeric(series, downcast="integer")
        return matrix

    def _finalize_feature_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Finalize computed feature matrix with coverage-aware cleaning."""
        if matrix is None or matrix.empty:
            raise RuntimeError("Feature matrix is empty after feature computation.")

        working = matrix
        working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
        working = working.dropna(
            subset=["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        )
        if working.empty:
            raise RuntimeError("Feature matrix has no valid OHLCV rows after initial sanitization.")

        feature_cols = [c for c in working.columns if c not in BASE_MARKET_COLUMNS]
        feature_cols = self._sanitize_feature_columns(feature_cols)
        if not feature_cols:
            raise RuntimeError("No usable feature columns were produced by feature pipeline.")

        working.loc[:, feature_cols] = working[feature_cols].replace([np.inf, -np.inf], np.nan)
        working = self._downcast_feature_payload(
            working,
            preserve_columns=set(BASE_MARKET_COLUMNS),
        )

        min_col_coverage = float(np.clip(self.config.feature_reuse_min_coverage * 0.50, 0.02, 0.35))
        col_coverage = working[feature_cols].notna().mean(axis=0)
        kept_feature_cols = [
            col for col in feature_cols if float(col_coverage.get(col, 0.0)) >= min_col_coverage
        ]
        if not kept_feature_cols:
            sorted_cols = col_coverage.sort_values(ascending=False).index.tolist()
            kept_feature_cols = sorted_cols[: min(40, len(sorted_cols))]
        dropped_feature_cols = len(feature_cols) - len(kept_feature_cols)
        if dropped_feature_cols > 0:
            self.logger.warning(
                "Dropping %d sparse feature columns (min_coverage=%.2f%%) to preserve "
                "full-feature training matrix viability.",
                dropped_feature_cols,
                min_col_coverage * 100.0,
            )

        min_row_coverage = float(np.clip(self.config.feature_reuse_min_coverage * 0.75, 0.05, 0.50))
        row_coverage = working[kept_feature_cols].notna().mean(axis=1)
        input_rows = len(working)
        working = working.loc[row_coverage >= min_row_coverage]
        if working.empty:
            raise RuntimeError(
                "Feature matrix lost all rows after coverage filtering. "
                "Adjust feature groups or coverage thresholds."
            )

        working = working.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        feature_quality_audit = self._build_feature_quality_audit(working, kept_feature_cols)
        self.training_metrics["feature_quality_audit"] = feature_quality_audit
        self.training_metrics["feature_quality_summary"] = {
            key: value
            for key, value in feature_quality_audit.items()
            if key not in {"by_feature", "family_summary", "worst_null_features", "worst_clip_features"}
        }
        self.training_metrics["feature_quality_family_summary"] = feature_quality_audit.get(
            "family_summary",
            {},
        )
        self.training_metrics["feature_quality_worst_null_features"] = feature_quality_audit.get(
            "worst_null_features",
            [],
        )
        self.training_metrics["feature_quality_worst_clip_features"] = feature_quality_audit.get(
            "worst_clip_features",
            [],
        )
        working[kept_feature_cols] = working.groupby("symbol", sort=False)[
            kept_feature_cols
        ].ffill()
        working[kept_feature_cols] = working[kept_feature_cols].fillna(0.0)
        working[kept_feature_cols] = working[kept_feature_cols].replace([np.inf, -np.inf], 0.0)

        final_matrix = working[
            ["symbol", "timestamp", "open", "high", "low", "close", "volume", *kept_feature_cols]
        ]
        self.logger.info(
            "Feature matrix finalized: rows=%d->%d, features=%d->%d, row_coverage_floor=%.2f%%",
            input_rows,
            len(final_matrix),
            len(feature_cols),
            len(kept_feature_cols),
            min_row_coverage * 100.0,
        )
        return final_matrix

    def _feature_schema_cache_key(
        self,
        symbols: list[str],
        start_date: pd.Timestamp | None,
        end_date: pd.Timestamp | None,
    ) -> str:
        """Build deterministic cache key for reusable PostgreSQL feature matrices."""
        feature_set_id = self._resolve_feature_set_id(symbols)
        symbol_part = ",".join(sorted(symbols))
        groups_part = ",".join(sorted(self.config.feature_groups))
        cross_sectional_part = "1" if self.config.enable_cross_sectional else "0"
        reference_part = "1" if self.config.enable_reference_features else "0"
        reference_sources_part = ",".join(self.config.reference_feature_sources or ["all"])
        tick_part = "1" if self.config.enable_tick_microstructure_features else "0"
        timeframes_part = ",".join(self.config.timeframes)
        start_part = start_date.isoformat() if start_date is not None else "none"
        end_part = end_date.isoformat() if end_date is not None else "none"
        return (
            "train:features:schema:v3:"
            f"{symbol_part}:{start_part}:{end_part}:"
            f"{self.config.timeframe}:{feature_set_id}:{groups_part}:"
            f"{cross_sectional_part}:{reference_part}:{reference_sources_part}:"
            f"{tick_part}:{timeframes_part}:"
            f"{self.config.training_bar_mode}:{self.config.intrinsic_bar_type}:"
            f"{self.feature_pipeline_fingerprint}"
        )

    def _feature_selection_schema_signature(self) -> dict[str, Any]:
        """Serialize feature-selection knobs that change the persisted cache contract."""
        return _build_snapshot_feature_selection_signature(self.config)

    def _current_ohlcv_fingerprint(self) -> str:
        """Fingerprint current OHLCV panel for safe feature-cache reuse."""
        if self.data is None or self.data.empty:
            return ""
        hash_columns = [
            column
            for column in ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
            if column in self.data.columns
        ]
        return compute_frame_content_hash(self.data, columns=hash_columns)

    def _build_feature_schema_metadata(
        self,
        feature_names: list[str],
        feature_set_id: str,
    ) -> dict[str, Any]:
        """Serialize feature schema metadata used for deterministic cache reuse."""
        return {
            "pipeline_fingerprint": self.feature_pipeline_fingerprint,
            "source_ohlcv_hash": self._current_ohlcv_fingerprint(),
            "source_ohlcv_rows": int(len(self.data)) if self.data is not None else 0,
            "feature_names": sorted(str(name) for name in feature_names),
            "feature_groups": sorted(str(name) for name in self.config.feature_groups),
            "enable_cross_sectional": bool(self.config.enable_cross_sectional),
            "enable_reference_features": bool(self.config.enable_reference_features),
            "reference_feature_sources": list(self.config.reference_feature_sources),
            "enable_tick_microstructure_features": bool(
                self.config.enable_tick_microstructure_features
            ),
            "timeframe": self.config.timeframe,
            "timeframes": list(self.config.timeframes),
            "training_bar_mode": self.config.training_bar_mode,
            "intrinsic_bar_type": self.config.intrinsic_bar_type,
            "feature_set_id": feature_set_id,
            "cache_stage": "post_selection",
            "feature_selection_signature": self._feature_selection_schema_signature(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def _resolve_feature_set_id(self, symbols: list[str] | None = None) -> str:
        """Build a deterministic feature scope id for the active symbol universe."""
        if symbols is None:
            if self.data is not None and not self.data.empty and "symbol" in self.data.columns:
                symbols = sorted(
                    {
                        str(symbol).strip().upper()
                        for symbol in self.data["symbol"].dropna().tolist()
                        if str(symbol).strip()
                    }
                )
            else:
                symbols = sorted(
                    {
                        str(symbol).strip().upper()
                        for symbol in self.config.symbols
                        if str(symbol).strip()
                    }
                )

        namespace = (
            re.sub(r"[^A-Za-z0-9_.-]+", "_", self.config.feature_set_id).strip("_") or "default"
        )
        scope_payload = json.dumps(
            {
                "namespace": namespace,
                "symbols": sorted(symbols),
                "timeframe": self.config.timeframe,
                "timeframes": list(self.config.timeframes),
                "groups": sorted(self.config.feature_groups),
                "cross_sectional": bool(self.config.enable_cross_sectional),
                "reference_features": bool(self.config.enable_reference_features),
                "reference_feature_sources": list(self.config.reference_feature_sources),
                "tick_microstructure_features": bool(
                    self.config.enable_tick_microstructure_features
                ),
                "training_bar_mode": self.config.training_bar_mode,
                "intrinsic_bar_type": self.config.intrinsic_bar_type,
                "pipeline_fingerprint": self.feature_pipeline_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha1(scope_payload.encode("utf-8")).hexdigest()[:12]
        return f"{namespace[:32]}_{digest}"

    def _augment_reference_features(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Augment computed market features with point-in-time reference/event layers."""
        from quant_trading_system.features.reference import (
            ReferenceFeatureBuilder,
            build_reference_feature_config,
        )

        if matrix is None or matrix.empty:
            return matrix

        try:
            builder = ReferenceFeatureBuilder(
                config=build_reference_feature_config(
                    enabled=self.config.enable_reference_features,
                    selected_sources=self.config.reference_feature_sources,
                ),
                logger_=self.logger,
            )
            enriched = builder.augment(matrix)
            added_cols = [column for column in enriched.columns if column not in matrix.columns]
            self.logger.info(
                "Reference feature augmentation added %d columns from sources=%s",
                len(added_cols),
                ",".join(self.config.reference_feature_sources),
            )
            return enriched
        except Exception as exc:
            if self.config.allow_feature_group_fallback:
                self.logger.warning(
                    "Reference feature augmentation skipped due to runtime error: %s",
                    exc,
                )
                return matrix
            raise RuntimeError(
                "Reference feature augmentation failed; institutional training requires a "
                "deterministic point-in-time reference layer."
            ) from exc

    def _augment_tick_microstructure_features(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Augment bar-aligned training matrix with quote/trade microstructure features."""
        from quant_trading_system.features.tick_microstructure import (
            TickMicrostructureFeatureBuilder,
            TickMicrostructureFeatureConfig,
        )

        if matrix is None or matrix.empty:
            return matrix

        try:
            builder = TickMicrostructureFeatureBuilder(
                TickMicrostructureFeatureConfig(timeframe=self.config.timeframe),
                logger_=self.logger,
            )
            enriched = builder.augment(matrix)
            added_cols = [column for column in enriched.columns if column not in matrix.columns]
            self.logger.info(
                "Tick microstructure augmentation added %d columns",
                len(added_cols),
            )
            return enriched
        except Exception as exc:
            if self.config.allow_feature_group_fallback:
                self.logger.warning(
                    "Tick microstructure augmentation skipped due to runtime error: %s",
                    exc,
                )
                return matrix
            raise RuntimeError(
                "Tick microstructure augmentation failed; institutional training requires "
                "deterministic tick/quote enrichment when enabled."
            ) from exc

    def _compute_symbol_feature_frame(
        self,
        *,
        df: pd.DataFrame,
        symbol: str,
        groups_to_compute: list[Any],
        universe_data: Any,
        disable_optimized_technical: bool,
    ) -> pd.DataFrame:
        import polars as pl
        from quant_trading_system.features.feature_pipeline import (
            FeatureConfig,
            FeatureGroup,
            FeaturePipeline,
            NormalizationMethod,
        )

        df = df.copy().sort_values("timestamp").reset_index(drop=True)
        df_pl = pl.from_pandas(df)
        combined_features: dict[str, np.ndarray] = {}

        for group in groups_to_compute:
            group_start = time.perf_counter()
            self.logger.info("  %s: computing %s feature group...", symbol, group.value)
            use_optimized_pipeline = (
                group == FeatureGroup.TECHNICAL and not disable_optimized_technical
            )
            pipeline_cache_key = (group.value, bool(use_optimized_pipeline))
            pipeline = self._feature_pipeline_cache.get(pipeline_cache_key)
            if pipeline is None:
                feature_config = FeatureConfig(
                    groups=[group],
                    normalization=NormalizationMethod.NONE,
                    include_targets=False,
                    use_gpu=self.config.use_gpu and group == FeatureGroup.TECHNICAL,
                    # Training materializes each symbol/timeframe only once, so per-pipeline
                    # LRU caching adds key-hash overhead without real cache hits.
                    use_cache=False,
                    use_optimized_pipeline=use_optimized_pipeline,
                )
                pipeline = FeaturePipeline(feature_config)
                self._feature_pipeline_cache[pipeline_cache_key] = pipeline
            try:
                feature_set = pipeline.compute(
                    df_pl,
                    symbol=symbol,
                    universe_data=(
                        universe_data if group == FeatureGroup.CROSS_SECTIONAL else None
                    ),
                    use_cache=False,
                )
                group_feature_count = len(feature_set.features)
                combined_features.update(feature_set.features)
                self.logger.info(
                    "  %s: %s group produced %d features in %.2fs",
                    symbol,
                    group.value,
                    group_feature_count,
                    time.perf_counter() - group_start,
                )
            except Exception as e:
                if (
                    self.config.allow_feature_group_fallback
                    and group == FeatureGroup.CROSS_SECTIONAL
                ):
                    self.logger.warning(
                        "%s: cross-sectional features skipped due to runtime error: %s",
                        symbol,
                        e,
                    )
                    continue
                raise RuntimeError(
                    f"Feature pipeline failed for {symbol} group={group.value}; "
                    "institutional mode requires deterministic feature materialization."
                ) from e

        if not combined_features:
            raise RuntimeError(f"No features computed for symbol {symbol}")

        feature_df = pd.DataFrame(
            {
                name: np.asarray(values, dtype=np.float32)
                for name, values in combined_features.items()
            },
            index=df.index,
        )
        features_df = pd.concat([df, feature_df], axis=1)
        features_df["symbol"] = symbol
        return features_df

    def _compute_symbol_multitimeframe_feature_frame(
        self,
        *,
        df: pd.DataFrame,
        symbol: str,
        groups_to_compute: list[Any],
        universe_data: Any,
        disable_optimized_technical: bool,
    ) -> pd.DataFrame:
        from quant_trading_system.features.feature_pipeline import FeatureGroup
        from quant_trading_system.features.multi_timeframe import MultiTimeframeFeatureEngine

        engine = MultiTimeframeFeatureEngine(
            base_timeframe=self.config.timeframe,
            timeframes=self.config.timeframes,
        )
        ohlcv_frames: dict[str, pd.DataFrame] = {}
        db_native_panels = getattr(self, "ohlcv_panels_by_timeframe", {}) or {}
        for timeframe_name in engine.normalized_timeframes():
            panel = db_native_panels.get(timeframe_name)
            if panel is None or panel.empty:
                continue
            symbol_frame = (
                panel[panel["symbol"] == symbol]
                .copy()
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            if not symbol_frame.empty:
                ohlcv_frames[timeframe_name] = symbol_frame

        if len(ohlcv_frames) < len(engine.normalized_timeframes()):
            fallback_frames = engine.build_ohlcv_frames(df)
            for timeframe_name, timeframe_frame in fallback_frames.items():
                if timeframe_name not in ohlcv_frames and timeframe_frame is not None:
                    ohlcv_frames[timeframe_name] = timeframe_frame

        feature_frames: dict[str, pd.DataFrame] = {}
        base_timeframe = normalize_timeframe(self.config.timeframe)

        for timeframe in engine.normalized_timeframes():
            timeframe_frame = ohlcv_frames.get(timeframe)
            if timeframe_frame is None or timeframe_frame.empty:
                self.logger.warning("%s: timeframe %s produced no rows", symbol, timeframe)
                continue

            timeframe_groups = list(groups_to_compute)
            if timeframe != base_timeframe:
                timeframe_groups = [
                    group for group in timeframe_groups if group != FeatureGroup.CROSS_SECTIONAL
                ]
            if not timeframe_groups:
                feature_frames[timeframe] = timeframe_frame.copy()
                continue

            self.logger.info(
                "  %s: computing multi-timeframe layer %s (%d rows)",
                symbol,
                timeframe,
                len(timeframe_frame),
            )
            feature_frames[timeframe] = self._compute_symbol_feature_frame(
                df=timeframe_frame,
                symbol=symbol,
                groups_to_compute=timeframe_groups,
                universe_data=(universe_data if timeframe == base_timeframe else None),
                disable_optimized_technical=disable_optimized_technical,
            )

        if base_timeframe not in feature_frames:
            raise RuntimeError(
                f"Base timeframe {base_timeframe} features missing for symbol {symbol}"
            )

        aligned = engine.align_feature_frames(
            base_frame=feature_frames[base_timeframe],
            feature_frames=feature_frames,
            include_resampled_ohlcv=True,
        )
        aligned["symbol"] = symbol
        return aligned

    def _apply_feature_selection(self) -> None:
        """Run development-set feature screening without touching holdout labels."""
        from quant_trading_system.models.feature_selection import (
            FeatureSelectionConfig,
            select_training_features,
        )

        if not self.config.enable_feature_selection:
            return
        if (
            self.features is None
            or self.features.empty
            or self.labels is None
            or len(self.labels) == 0
        ):
            return

        old_feature_names = [str(name) for name in self.feature_names]
        if len(old_feature_names) <= 1:
            return
        self._hydrate_snapshot_feature_selection_summary()

        selection_target: pd.Series | np.ndarray = self.labels
        selection_groups: pd.Series | np.ndarray | None = None
        feature_selection_target_mode = "classification_label"
        group_aware_ic = False
        effective_max_features = int(self.config.feature_selection_max_features)
        if (
            self.config.model_type == "lightgbm_ranker"
            and self.development_frame is not None
            and not self.development_frame.empty
        ):
            selection_groups = self.development_frame["timestamp"].to_numpy()
            group_aware_ic = True
            if "triple_barrier_net_return" in self.development_frame.columns:
                rank_target = pd.to_numeric(
                    self.development_frame["triple_barrier_net_return"],
                    errors="coerce",
                )
                rank_target = rank_target.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                rank_target = rank_target.clip(lower=-0.50, upper=0.50)
                if int(rank_target.nunique(dropna=True)) > 1:
                    selection_target = rank_target
                    feature_selection_target_mode = "triple_barrier_net_return"
            adaptive_ranker_feature_cap = max(
                24,
                int(round(np.sqrt(len(old_feature_names)) * 6.0)),
            )
            effective_max_features = min(
                effective_max_features,
                adaptive_ranker_feature_cap,
            )

        selection_kwargs: dict[str, Any] = {}
        if selection_groups is not None:
            selection_kwargs["groups"] = selection_groups
        selection_result = select_training_features(
            self.features,
            selection_target,
            FeatureSelectionConfig(
                min_information_coefficient=self.config.feature_selection_min_ic,
                max_correlation=self.config.feature_selection_max_corr,
                max_features=effective_max_features,
                stability_iterations=self.config.feature_selection_stability_iterations,
                min_stability_support=self.config.feature_selection_min_stability_support,
                group_aware_ic=group_aware_ic,
                min_group_size=3,
                random_state=self.config.seed,
            ),
            **selection_kwargs,
        )
        selected_features = [
            feature
            for feature in selection_result.selected_features
            if feature in self.features.columns
        ]
        if not selected_features:
            self.logger.warning(
                "Feature selection returned no candidates; keeping original matrix."
            )
            return

        self.features = self.features[selected_features].copy()
        if self.holdout_features is not None and not self.holdout_features.empty:
            self.holdout_features = self.holdout_features[selected_features].copy()
        self.feature_names = selected_features
        self._apply_model_priors()

        keep_selected = set(selected_features)
        for attr in ("development_frame", "holdout_frame"):
            frame = getattr(self, attr, None)
            if frame is None or frame.empty:
                continue
            retain_cols = [column for column in frame.columns if column not in old_feature_names]
            retain_cols.extend([column for column in old_feature_names if column in keep_selected])
            setattr(self, attr, frame[retain_cols].copy())

        diagnostics = selection_result.diagnostics
        self.training_metrics["feature_selection_applied"] = True
        self.training_metrics["feature_selection_initial_feature_count"] = float(
            diagnostics.get("initial_feature_count", len(old_feature_names))
        )
        self.training_metrics["feature_selection_selected_feature_count"] = float(
            diagnostics.get("selected_feature_count", len(selected_features))
        )
        self.training_metrics["feature_selection_correlation_pruned_count"] = float(
            diagnostics.get("correlation_pruned_count", 0.0)
        )
        self.training_metrics["feature_selection_stability_selected_count"] = float(
            diagnostics.get("stability_selected_count", 0.0)
        )
        self.training_metrics["feature_selection_min_ic"] = float(
            self.config.feature_selection_min_ic
        )
        self.training_metrics["feature_selection_target_mode"] = feature_selection_target_mode
        self.training_metrics["feature_selection_group_aware_ic"] = float(group_aware_ic)
        self.training_metrics["feature_selection_effective_max_features"] = float(
            effective_max_features
        )
        self.training_metrics["feature_selection_selected_features"] = selected_features
        self.training_metrics["feature_selection_top_information_coefficients"] = {
            str(name): float(value)
            for name, value in list(selection_result.information_coefficients.items())[:20]
        }
        self.training_metrics["feature_selection_stability_scores"] = {
            str(name): float(score)
            for name, score in selection_result.stability_scores.items()
            if name in keep_selected
        }
        feature_selection_audit = self._build_feature_selection_audit(
            old_feature_names=old_feature_names,
            selection_result=selection_result,
            selected_features=selected_features,
        )
        current_stage_binding = bool(feature_selection_audit.get("selection_binding", False))
        upstream_binding = bool(self.training_metrics.get("feature_selection_upstream_binding", 0.0))
        effective_binding = bool(current_stage_binding or upstream_binding)
        self.training_metrics["feature_selection_current_stage_binding"] = float(
            current_stage_binding
        )
        self.training_metrics["feature_selection_current_stage_initial_feature_count"] = float(
            len(old_feature_names)
        )
        self.training_metrics["feature_selection_current_stage_selected_feature_count"] = float(
            len(selected_features)
        )
        self.training_metrics.setdefault(
            "feature_selection_development_binding",
            float(current_stage_binding),
        )
        self.training_metrics.setdefault(
            "feature_selection_development_initial_feature_count",
            float(len(old_feature_names)),
        )
        self.training_metrics.setdefault(
            "feature_selection_development_selected_feature_count",
            float(len(selected_features)),
        )
        self.training_metrics["feature_selection_binding"] = float(effective_binding)
        self.training_metrics["feature_selection_audit"] = feature_selection_audit
        self.logger.info(
            "Feature selection reduced development matrix from %d to %d columns",
            len(old_feature_names),
            len(selected_features),
        )
        if not feature_selection_audit.get("selection_binding", False):
            self.logger.warning(
                "Feature selection remained non-binding; development matrix stayed at %d columns.",
                len(selected_features),
            )
        if self.data_quality_report_hash is not None:
            self._build_training_snapshot_manifest()

    def _compute_features_full_pipeline(self) -> None:
        """Compute institutional features with adaptive memory-safe fallback behavior."""
        import polars as pl
        from quant_trading_system.features.feature_pipeline import FeatureGroup

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
        self._record_feature_gpu_contract(
            groups_to_compute=groups_to_compute,
            disable_optimized_technical=disable_optimized_technical,
        )
        multi_timeframe_enabled = len(self.config.timeframes) > 1
        if multi_timeframe_enabled:
            self.logger.info("Multi-timeframe feature fusion enabled: %s", self.config.timeframes)

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
            if multi_timeframe_enabled:
                features_df = self._compute_symbol_multitimeframe_feature_frame(
                    df=df,
                    symbol=symbol,
                    groups_to_compute=groups_to_compute,
                    universe_data=universe_data,
                    disable_optimized_technical=disable_optimized_technical,
                )
            else:
                features_df = self._compute_symbol_feature_frame(
                    df=df,
                    symbol=symbol,
                    groups_to_compute=groups_to_compute,
                    universe_data=universe_data,
                    disable_optimized_technical=disable_optimized_technical,
                )
            features_list.append(features_df)
            feature_count = max(
                0,
                len(
                    [
                        column
                        for column in features_df.columns
                        if column
                        not in {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
                    ]
                ),
            )
            self.logger.info("  %s: %d features computed", symbol, feature_count)

        if features_list:
            symbol_frame_count = len(features_list)
            raw_features = pd.concat(features_list, ignore_index=True)
            features_list.clear()
            del features_list
            gc.collect()
            raw_features = self._downcast_feature_payload(
                raw_features,
                preserve_columns=set(BASE_MARKET_COLUMNS),
            )
            if self.config.enable_reference_features:
                raw_features = self._augment_reference_features(raw_features)
                raw_features = self._downcast_feature_payload(
                    raw_features,
                    preserve_columns=set(BASE_MARKET_COLUMNS),
                )
                gc.collect()
            if self.config.enable_tick_microstructure_features:
                raw_features = self._augment_tick_microstructure_features(raw_features)
                raw_features = self._downcast_feature_payload(
                    raw_features,
                    preserve_columns=set(BASE_MARKET_COLUMNS),
                )
                gc.collect()
            self.features = self._finalize_feature_matrix(raw_features)
            del raw_features
            gc.collect()
            self.training_metrics["multi_timeframe_enabled"] = bool(multi_timeframe_enabled)
            self.training_metrics["multi_timeframe_scopes"] = list(self.config.timeframes)
            self.logger.info(
                f"Total: {len(self.features.columns)} columns for {symbol_frame_count} symbols"
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
        timeframe = normalize_timeframe(self.config.timeframe)
        feature_set_id = self._resolve_feature_set_id(symbols)

        start_date = (
            pd.to_datetime(self.config.start_date, utc=True) if self.config.start_date else None
        )
        end_date = pd.to_datetime(self.config.end_date, utc=True) if self.config.end_date else None

        cache_key = (
            f"train:features:coverage:v3:{','.join(symbols)}:"
            f"{timeframe}:{feature_set_id}:"
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
            self.logger.info(
                "Feature cache marked empty previously; revalidating PostgreSQL coverage."
            )

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
        cached_source_hash = str(schema_payload.get("source_ohlcv_hash") or "").strip()
        current_source_hash = self._current_ohlcv_fingerprint()
        if not cached_source_hash:
            self.logger.info(
                "Feature cache metadata missing source OHLCV hash; recomputing features."
            )
            return None
        if current_source_hash and cached_source_hash != current_source_hash:
            self.logger.info("Feature cache source OHLCV hash changed; recomputing features.")
            return None
        if schema_payload.get("timeframe") != timeframe:
            self.logger.info("Feature cache timeframe changed; recomputing features.")
            return None
        if list(schema_payload.get("timeframes", [])) != list(self.config.timeframes):
            self.logger.info("Feature cache multi-timeframe scope changed; recomputing features.")
            return None
        if schema_payload.get("training_bar_mode") != self.config.training_bar_mode:
            self.logger.info("Feature cache training bar mode changed; recomputing features.")
            return None
        if schema_payload.get("intrinsic_bar_type") != self.config.intrinsic_bar_type:
            self.logger.info("Feature cache intrinsic bar type changed; recomputing features.")
            return None
        if list(schema_payload.get("reference_feature_sources", [])) != list(
            self.config.reference_feature_sources
        ):
            self.logger.info("Feature cache reference source scope changed; recomputing features.")
            return None
        if bool(schema_payload.get("enable_tick_microstructure_features", False)) != bool(
            self.config.enable_tick_microstructure_features
        ):
            self.logger.info(
                "Feature cache tick microstructure scope changed; recomputing features."
            )
            return None
        if schema_payload.get("feature_set_id") != feature_set_id:
            self.logger.info("Feature cache scope changed; recomputing features.")
            return None
        if schema_payload.get("cache_stage") != "post_selection":
            self.logger.info("Feature cache stage changed; recomputing features.")
            return None
        if (
            schema_payload.get("feature_selection_signature")
            != self._feature_selection_schema_signature()
        ):
            self.logger.info("Feature selection cache signature changed; recomputing features.")
            return None

        query = text("""
            SELECT
                symbol,
                timestamp,
                feature_name,
                value
            FROM features
            WHERE symbol = :symbol
              AND timeframe = :timeframe
              AND feature_set_id = :feature_set_id
              AND (:start_date IS NULL OR timestamp >= :start_date)
              AND (:end_date IS NULL OR timestamp <= :end_date)
            ORDER BY timestamp, feature_name
            """)

        feature_frames: list[pd.DataFrame] = []
        with db_manager.session() as session:
            session.execute(text("SET LOCAL max_parallel_workers_per_gather = 0"))
            session.execute(text("SET LOCAL statement_timeout = '60000ms'"))
            for symbol in symbols:
                result = session.execute(
                    query,
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "feature_set_id": feature_set_id,
                        "start_date": start_date,
                        "end_date": end_date,
                    },
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

        wide_df = long_df.pivot_table(
            index=["symbol", "timestamp"],
            columns="feature_name",
            values="value",
            aggfunc="last",
        ).reset_index()
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
        merged = merged[
            [
                "symbol",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                *selected_feature_cols,
            ]
        ]
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
        from quant_trading_system.database.connection import get_db_manager, get_redis_manager

        if self.features is None or self.features.empty:
            return

        feature_cols = [c for c in self.features.columns if c not in BASE_MARKET_COLUMNS]
        feature_cols = self._sanitize_feature_columns(feature_cols)
        if not feature_cols:
            return

        export_source = self.features
        if self.development_frame is not None and not self.development_frame.empty:
            export_source = self.development_frame
        required_export_cols = ["symbol", "timestamp", *feature_cols]
        missing_export_cols = [
            col for col in required_export_cols if col not in export_source.columns
        ]
        if missing_export_cols:
            raise KeyError(
                f"Persisted feature source is missing required columns: {missing_export_cols}"
            )

        export_df = export_source[required_export_cols].copy()
        export_df["timestamp"] = pd.to_datetime(export_df["timestamp"], utc=True, errors="coerce")
        export_df = export_df.dropna(subset=["timestamp"])
        if export_df.empty:
            return
        timeframe = normalize_timeframe(self.config.timeframe)
        feature_set_id = self._resolve_feature_set_id(
            sorted(export_df["symbol"].dropna().unique().tolist())
        )
        symbols = sorted(export_df["symbol"].dropna().unique().tolist())
        start_date = (
            pd.to_datetime(self.config.start_date, utc=True) if self.config.start_date else None
        )
        end_date = pd.to_datetime(self.config.end_date, utc=True) if self.config.end_date else None

        db_manager = get_db_manager()
        checkpoint_file = self._feature_materialization_checkpoint_path()
        total_source_rows = len(export_df)
        feature_count = len(feature_cols)
        estimated_feature_rows = int(total_source_rows * max(feature_count, 1))
        checkpoint_payload = self._read_materialization_checkpoint(checkpoint_file)
        resume_offset = min(int(checkpoint_payload.get("next_offset", 0)), total_source_rows)
        if checkpoint_payload:
            checkpoint_feature_count = int(checkpoint_payload.get("feature_count", -1))
            checkpoint_stage = str(checkpoint_payload.get("cache_stage") or "")
            if checkpoint_feature_count != feature_count or checkpoint_stage != "post_selection":
                self.logger.info(
                    "Discarding stale feature materialization checkpoint because cache semantics changed"
                )
                resume_offset = 0
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
        if resume_offset > 0:
            self.logger.info(
                "Resuming feature materialization from source row offset "
                f"{resume_offset}/{total_source_rows}"
            )

        source_batch_rows = self._resolve_feature_materialization_source_batch_rows(
            feature_count=len(feature_cols)
        )
        self.training_metrics["feature_cache_stage"] = "post_selection"
        self.training_metrics["feature_cache_selected_feature_count"] = float(len(feature_cols))
        self.training_metrics["feature_cache_estimated_row_count"] = float(estimated_feature_rows)

        total_upserted_rows = 0
        raw_conn = db_manager.engine.raw_connection()
        stage_table = "tmp_training_features_stage"
        try:
            with raw_conn.cursor() as cursor:
                cursor.execute("SET statement_timeout = '600000ms'")
                cursor.execute("SET synchronous_commit = OFF")
                cursor.execute(f"""
                    CREATE TEMP TABLE IF NOT EXISTS {stage_table} (
                        symbol TEXT NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        timeframe TEXT NOT NULL,
                        feature_name TEXT NOT NULL,
                        feature_set_id TEXT NOT NULL,
                        value DOUBLE PRECISION NOT NULL
                    )
                    """)
                raw_conn.commit()

                if resume_offset == 0:
                    cursor.execute(
                        """
                        DELETE FROM features
                        WHERE symbol = ANY(%s)
                          AND timeframe = %s
                          AND feature_set_id = %s
                          AND (%s IS NULL OR timestamp >= %s)
                          AND (%s IS NULL OR timestamp <= %s)
                        """,
                        (
                            symbols,
                            timeframe,
                            feature_set_id,
                            start_date,
                            start_date,
                            end_date,
                            end_date,
                        ),
                    )
                    raw_conn.commit()

                for i in range(resume_offset, total_source_rows, source_batch_rows):
                    source_batch = export_df.iloc[i : i + source_batch_rows]
                    long_batch = source_batch.melt(
                        id_vars=["symbol", "timestamp"],
                        value_vars=feature_cols,
                        var_name="feature_name",
                        value_name="value",
                    ).dropna(subset=["value"])

                    if not long_batch.empty:
                        staged_rows = self._copy_feature_rows_to_stage(
                            cursor=cursor,
                            stage_table=stage_table,
                            long_batch=long_batch,
                            timeframe=timeframe,
                            feature_set_id=feature_set_id,
                        )
                        if staged_rows > 0:
                            cursor.execute(f"""
                                INSERT INTO features (
                                    symbol,
                                    timestamp,
                                    timeframe,
                                    feature_name,
                                    feature_set_id,
                                    value
                                )
                                SELECT
                                    symbol,
                                    timestamp,
                                    timeframe,
                                    feature_name,
                                    feature_set_id,
                                    value
                                FROM {stage_table}
                                ON CONFLICT (symbol, timestamp, timeframe, feature_name, feature_set_id)
                                DO UPDATE SET value = EXCLUDED.value
                                """)
                            cursor.execute(f"TRUNCATE TABLE {stage_table}")
                            total_upserted_rows += staged_rows

                    self._write_materialization_offset(
                        checkpoint_file=checkpoint_file,
                        next_offset=i + len(source_batch),
                        total_rows=total_source_rows,
                        feature_count=feature_count,
                    )
                    raw_conn.commit()

                cursor.execute(f"DROP TABLE IF EXISTS {stage_table}")
                raw_conn.commit()
        finally:
            raw_conn.close()

        if checkpoint_file.exists():
            checkpoint_file.unlink()

        redis_mgr = get_redis_manager()
        redis_mgr.delete(f"train:ohlcv_symbols:{timeframe}")
        schema_key = self._feature_schema_cache_key(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )
        schema_metadata = self._build_feature_schema_metadata(feature_cols, feature_set_id)
        redis_mgr.set(
            schema_key, json.dumps(schema_metadata, ensure_ascii=True), expire_seconds=604800
        )
        self.features_persisted_to_postgres = True
        self.logger.info(f"Persisted {total_upserted_rows} feature rows to PostgreSQL")

    def _resolve_feature_materialization_source_batch_rows(self, feature_count: int) -> int:
        """Cap source batches by estimated staged long rows instead of a fixed tiny batch."""
        requested = max(250, int(self.config.feature_materialization_batch_rows))
        if feature_count <= 0:
            return requested
        max_long_rows = 1_000_000
        adaptive_limit = max(250, int(max_long_rows / max(feature_count, 1)))
        source_batch_rows = min(requested, adaptive_limit)
        if source_batch_rows < requested:
            self.logger.info(
                "Capping feature materialization source batch rows "
                f"from {requested} to {source_batch_rows} to keep staged writes bounded"
            )
        return source_batch_rows

    def _copy_feature_rows_to_stage(
        self,
        cursor: Any,
        stage_table: str,
        long_batch: pd.DataFrame,
        timeframe: str,
        feature_set_id: str,
    ) -> int:
        """COPY a melted feature batch into a temporary staging table."""
        stage_df = long_batch.copy()
        stage_df["timestamp"] = pd.to_datetime(stage_df["timestamp"], utc=True, errors="coerce")
        stage_df["value"] = pd.to_numeric(stage_df["value"], errors="coerce")
        stage_df = stage_df.dropna(subset=["timestamp", "feature_name", "value"])
        if stage_df.empty:
            return 0

        stage_df = stage_df.assign(
            timeframe=timeframe,
            feature_set_id=feature_set_id,
            symbol=stage_df["symbol"].astype(str),
            feature_name=stage_df["feature_name"].astype(str),
        )[["symbol", "timestamp", "timeframe", "feature_name", "feature_set_id", "value"]]

        buffer = io.StringIO()
        stage_df.to_csv(
            buffer,
            sep="\t",
            index=False,
            header=False,
            na_rep="\\N",
            date_format="%Y-%m-%d %H:%M:%S%z",
        )
        buffer.seek(0)
        cursor.copy_from(
            buffer,
            stage_table,
            sep="\t",
            null="\\N",
            columns=(
                "symbol",
                "timestamp",
                "timeframe",
                "feature_name",
                "feature_set_id",
                "value",
            ),
        )
        return int(len(stage_df))

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
    def _read_materialization_checkpoint(checkpoint_file: Path) -> dict[str, Any]:
        if not checkpoint_file.exists():
            return {}
        try:
            payload = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
            return {}
        except Exception:
            return {}

    def _write_materialization_offset(
        self,
        checkpoint_file: Path,
        next_offset: int,
        total_rows: int,
        feature_count: int,
    ) -> None:
        payload = {
            "snapshot_id": (
                self.snapshot_manifest.get("snapshot_id")
                if isinstance(self.snapshot_manifest, dict)
                else None
            ),
            "model_name": self.config.model_name,
            "cache_stage": "post_selection",
            "feature_count": int(feature_count),
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
            apply_uniqueness_weighting=bool(self.config.label_apply_uniqueness_weighting),
            uniqueness_weight_floor=float(self.config.label_uniqueness_weight_floor),
            apply_volatility_inverse_weighting=bool(
                self.config.label_apply_volatility_inverse_weighting
            ),
            volatility_weight_cap=float(self.config.label_volatility_weight_cap),
        )
        target_result = generate_targets(self.features, target_config)
        label_valid_mask = target_result.frame["label"].notna().to_numpy()
        labeled_frame = target_result.frame.loc[label_valid_mask].copy()
        if labeled_frame.empty:
            raise ValueError("Target engineering produced zero valid labels.")
        labeled_frame["label"] = pd.to_numeric(labeled_frame["label"], errors="coerce").astype(int)
        labeled_frame = labeled_frame.dropna(subset=["timestamp"])
        labeled_frame = labeled_frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        if "regime" not in labeled_frame.columns:
            labeled_frame["regime"] = "normal_range"

        all_weights = np.asarray(target_result.sample_weights, dtype=float)
        sample_weights = all_weights[label_valid_mask]
        sample_weights = np.nan_to_num(sample_weights, nan=1.0, posinf=1.0, neginf=1.0)
        if len(sample_weights) != len(labeled_frame):
            sample_weights = np.ones(len(labeled_frame), dtype=float)

        self.label_diagnostics = target_result.diagnostics
        self.training_metrics["label_positive_rate"] = float(
            self.label_diagnostics.get("positive_rate", 0.0)
        )
        self.training_metrics["label_class_balance_ratio"] = float(
            self.label_diagnostics.get("class_balance_ratio", 0.0)
        )
        self.training_metrics["label_drift_abs"] = float(
            self.label_diagnostics.get("label_drift_abs", 0.0)
        )
        self.training_metrics["label_count"] = float(self.label_diagnostics.get("label_count", 0))
        self.training_metrics["label_forward_outlier_filtered_count"] = float(
            self.label_diagnostics.get("forward_outlier_filtered_count", 0)
        )
        self.training_metrics["label_forward_outlier_filtered_rate"] = float(
            self.label_diagnostics.get("forward_outlier_filtered_rate", 0.0)
        )
        self.training_metrics["label_neutral_filtered_count"] = float(
            self.label_diagnostics.get("neutral_filtered_count", 0)
        )

        # Reserve untouched terminal holdout block by timestamp for anti-overfit validation.
        ts_series = pd.to_datetime(labeled_frame["timestamp"], utc=True, errors="coerce")
        unique_ts = np.sort(ts_series.dropna().unique())
        holdout_count = (
            max(1, int(round(len(unique_ts) * self.config.holdout_pct))) if len(unique_ts) else 0
        )
        holdout_ts = set(unique_ts[-holdout_count:]) if holdout_count > 0 else set()
        holdout_mask = (
            ts_series.isin(holdout_ts).to_numpy()
            if holdout_ts
            else np.zeros(len(labeled_frame), dtype=bool)
        )
        dev_rows = int(np.sum(~holdout_mask))
        holdout_rows = int(np.sum(holdout_mask))
        if dev_rows < 1000 or holdout_rows < 200:
            raise ValueError(
                "Institutional training requires a non-trivial untouched holdout block. "
                f"Computed dev_rows={dev_rows}, holdout_rows={holdout_rows}, "
                f"unique_timestamps={len(unique_ts)}, holdout_pct={self.config.holdout_pct:.2f}."
            )
        dev_mask = ~holdout_mask

        dev_frame = labeled_frame.loc[dev_mask].reset_index(drop=True)
        holdout_frame = labeled_frame.loc[holdout_mask].reset_index(drop=True)
        dev_weights = sample_weights[dev_mask]
        holdout_weights = sample_weights[holdout_mask]
        self._restore_training_state_from_split_frames(
            dev_frame=dev_frame,
            holdout_frame=holdout_frame,
            label_horizons=label_horizons,
            dev_weights=dev_weights,
            holdout_weights=holdout_weights,
        )

        if self._is_regression_model():
            self.logger.info(
                "Created %d development regression targets "
                "(mean=%.6f, std=%.6f) | holdout=%d | target_mode=%s",
                len(self.labels),
                float(self.training_metrics.get("target_mean", 0.0)),
                float(self.training_metrics.get("target_std", 0.0)),
                int(np.sum(holdout_mask)),
                str(self.training_metrics.get("target_mode", "return_regression")),
            )
        else:
            self.logger.info(
                f"Created {len(self.labels)} development labels "
                f"(pos: {int(self.labels.sum())}, neg: {int(len(self.labels) - self.labels.sum())}) "
                f"| holdout={int(np.sum(holdout_mask))} "
                f"| pos_rate={self.training_metrics['label_positive_rate']:.2%} "
                f"| drift_abs={self.training_metrics['label_drift_abs']:.2%} "
                f"| outlier_filtered={int(self.training_metrics['label_forward_outlier_filtered_count'])}"
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
            diffs = np.diff(pd.DatetimeIndex(ts_series).asi8)
            if np.any(diffs <= 0):
                raise ValueError("Timestamp ordering violation detected in training data.")
            return

        symbol_series = pd.Series(self.row_symbols.astype(str))
        for symbol, idx in symbol_series.groupby(symbol_series).groups.items():
            symbol_ts = pd.DatetimeIndex(ts_series.iloc[idx]).asi8
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

        ts_ns = pd.DatetimeIndex(ts_series).asi8
        unique_ts_ns = np.sort(np.unique(ts_ns))
        if len(unique_ts_ns) < 3:
            raise ValueError("Insufficient unique timestamps for panel-aware cross-validation.")

        unique_dummy = np.zeros((len(unique_ts_ns), 1), dtype=float)
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        unique_times = pd.to_datetime(unique_ts_ns, utc=True)
        try:
            raw_splits = splitter.split(unique_dummy, unique_dummy[:, 0], times=unique_times)
        except TypeError:
            raw_splits = splitter.split(unique_dummy, unique_dummy[:, 0])

        for train_ts_idx, test_ts_idx in raw_splits:
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
    def _split_with_optional_times(
        splitter: Any,
        X: np.ndarray,
        y: np.ndarray,
        times: np.ndarray | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Call CV splitters with timestamps when supported."""
        if times is None:
            iterator = splitter.split(X, y)
        else:
            time_values = pd.to_datetime(times, utc=True, errors="coerce")
            if pd.isna(time_values).any():
                raise ValueError("Invalid timestamps detected during CV split generation.")
            try:
                iterator = splitter.split(X, y, times=time_values)
            except TypeError:
                iterator = splitter.split(X, y)

        return [
            (np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int))
            for train_idx, test_idx in iterator
        ]

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

    @staticmethod
    def _infer_lightgbm_monotonic_constraints(feature_names: list[str]) -> list[int]:
        """Infer conservative monotonic constraints from feature names."""
        constraints: list[int] = []
        for raw_name in feature_names:
            name = str(raw_name).strip().lower()
            direction = 0

            if "momentum" in name:
                direction = 1
            elif "volatility" in name or "atr" in name:
                direction = -1

            constraints.append(int(direction))
        return constraints

    def _apply_model_priors(self) -> None:
        """Apply model-family priors that depend on prepared feature schema."""
        if not self._is_lightgbm_family_model():
            return
        if not self.config.lightgbm_use_monotonic_constraints:
            self.config.model_params.pop("monotone_constraints", None)
            return
        if not self.feature_names:
            return

        constraints = self._infer_lightgbm_monotonic_constraints(self.feature_names)
        constrained_count = int(np.count_nonzero(constraints))
        if constrained_count <= 0:
            self.config.model_params.pop("monotone_constraints", None)
            self.training_metrics["lightgbm_monotone_constraint_count"] = 0.0
            return

        # Keep constraints reasonably sparse to avoid forcing brittle directional assumptions.
        if constrained_count > max(3, int(round(0.80 * len(constraints)))):
            self.logger.info(
                "Skipping monotonic constraints because inferred density is too high: %d/%d",
                constrained_count,
                len(constraints),
            )
            self.config.model_params.pop("monotone_constraints", None)
            self.training_metrics["lightgbm_monotone_constraint_count"] = 0.0
            return

        self.config.model_params["monotone_constraints"] = constraints
        self.training_metrics["lightgbm_monotone_constraint_count"] = float(constrained_count)

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
        if self.cached_cv_splits:
            replay_splits: list[tuple[np.ndarray, np.ndarray]] = []
            for train_idx, test_idx in self.cached_cv_splits:
                train_arr = np.asarray(train_idx, dtype=int)
                test_arr = np.asarray(test_idx, dtype=int)
                if train_arr.size and int(np.max(train_arr)) >= len(X):
                    self.logger.warning(
                        "Cached snapshot CV split exceeds current sample count; regenerating splits."
                    )
                    replay_splits = []
                    break
                if test_arr.size and int(np.max(test_arr)) >= len(X):
                    self.logger.warning(
                        "Cached snapshot CV split exceeds current sample count; regenerating splits."
                    )
                    replay_splits = []
                    break
                replay_splits.append((train_arr.copy(), test_arr.copy()))
            if replay_splits:
                self.training_metrics["snapshot_bundle_cv_splits_reused"] = True
                return replay_splits

        if self.timestamps is None or len(self.timestamps) != len(X):
            cv = self._get_cv_splitter(n_samples=len(X))
            candidate_splits = self._split_with_optional_times(cv, X, y, times=None)
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
            candidate_splits = self._split_with_optional_times(cv, X, y, times=ts_series.to_numpy())
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

    def _leakage_validation_order_vector(self) -> np.ndarray:
        """Build a strictly increasing order vector for panel-aware leak validation."""
        if self.timestamps is None:
            raise ValueError("Timestamp array missing for leakage validation")

        ts_series = pd.Series(pd.to_datetime(self.timestamps, utc=True, errors="coerce"))
        if ts_series.isna().any():
            raise ValueError("Timestamp array contains invalid entries after conversion")

        if self.row_symbols is None or len(self.row_symbols) != len(ts_series):
            return ts_series.astype("int64", copy=False).to_numpy()
        # Panel datasets legitimately contain duplicate timestamps across symbols.
        # Strict temporal integrity is enforced separately by _validate_symbol_timestamp_ordering().
        return np.arange(len(ts_series), dtype=np.int64)

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
            timestamps=self._leakage_validation_order_vector(),
        )
        if not is_valid:
            raise ValueError(f"Future-leak validation failed: {issues}")
        self._validate_symbol_timestamp_ordering()

        splits = self._generate_cv_splits(self.features.values, self.labels.values)
        reference_sample_count = len(self.features)
        if self.timestamps is not None and len(self.timestamps) == len(self.features):
            ts_reference = pd.to_datetime(self.timestamps, utc=True, errors="coerce")
            if pd.notna(ts_reference).all():
                reference_sample_count = int(pd.Series(ts_reference).nunique())
        expected_gap_bars = max(
            self._effective_cv_prediction_horizon(reference_sample_count),
            int(self.config.purge_pct * 100),
        )
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if self.config.cv_method == "walk_forward":
                valid_split, split_issues = validator.validate_time_series_split(
                    self.timestamps[train_idx],
                    self.timestamps[test_idx],
                    gap_bars=expected_gap_bars,
                    reference_timestamps=self.timestamps,
                )
                if not valid_split:
                    raise ValueError(f"Leakage detected in CV split {fold_idx + 1}: {split_issues}")
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
            train_ts_ns = (
                pd.Series(pd.to_datetime(train_ts, utc=True, errors="coerce"))
                .astype("int64", copy=False)
                .to_numpy()
            )
            test_ts_ns = (
                pd.Series(pd.to_datetime(test_ts, utc=True, errors="coerce"))
                .astype("int64", copy=False)
                .to_numpy()
            )
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

    def _select_nested_outer_candidates(
        self,
        stable_candidates: list[tuple[float, float, float, dict[str, Any], int, float]],
        unstable_candidates: list[tuple[float, float, float, dict[str, Any], int, float]],
    ) -> tuple[list[tuple[float, float, float, dict[str, Any], int, float]], bool]:
        """Select stable nested-CV candidates, allowing research-only unstable fallback."""
        if stable_candidates:
            return stable_candidates, False
        if not unstable_candidates:
            raise ValueError(
                "Nested walk-forward optimization produced no valid outer-fold candidates."
            )
        if not bool(self.config.allow_unstable_outer_fold_fallback):
            raise ValueError(
                "No stable outer-fold candidates passed stability gate; unstable fallback is "
                "disabled for this training profile."
            )
        self.logger.warning(
            "No stable outer-fold candidates passed stability gate; "
            "falling back to best unstable candidate."
        )
        return unstable_candidates, True

    def _optimize_hyperparameters(self) -> None:
        """Phase 4: Hyperparameter optimization."""
        self.logger.info(
            "Phase 4: Hyperparameter optimization using Optuna (risk-adjusted objective)..."
        )
        if self.config.optimizer != "optuna":
            raise ValueError(
                f"Institutional mode requires optimizer=optuna, got '{self.config.optimizer}'"
            )
        if not self.config.use_nested_walk_forward:
            raise ValueError("Institutional mode requires nested walk-forward optimization trace.")
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
        regimes = self.regimes if self.regimes is not None and len(self.regimes) == len(y) else None
        row_symbols = (
            self.row_symbols
            if self.row_symbols is not None and len(self.row_symbols) == len(y)
            else None
        )
        splits = self._generate_cv_splits(X, y)
        if not splits:
            self.logger.info("No CV splits available for Optuna; keeping configured defaults")
            return

        min_train_size = min(len(train_idx) for train_idx, _ in splits)
        search_space = self._get_optuna_search_space(min_train_size=min_train_size)
        if not search_space:
            self.logger.info(
                "No Optuna search space for this model type, keeping configured defaults"
            )
            return

        def objective(trial: "optuna.Trial") -> float:
            params = search_space(trial)
            param_robustness_penalty = self._parameter_robustness_penalty(
                self.config.model_type,
                params,
            )
            fold_scores: list[float] = []
            fold_weights: list[float] = []

            try:
                for fold_idx, (train_idx, test_idx) in enumerate(splits):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    if self._requires_binary_class_support() and not self._has_binary_class_support(
                        y_train
                    ):
                        continue
                    fold_params = self._prepare_params_for_train_size(params, len(train_idx))
                    fold_params = self._augment_params_for_train_labels(fold_params, y_train)
                    model = self._create_model(params=fold_params)
                    w_train = weights[train_idx] if weights is not None else None
                    train_ts = (
                        np.asarray(self.timestamps[train_idx], dtype="datetime64[ns]")
                        if self.timestamps is not None and len(self.timestamps) == len(y)
                        else None
                    )
                    test_ts = (
                        np.asarray(self.timestamps[test_idx], dtype="datetime64[ns]")
                        if self.timestamps is not None and len(self.timestamps) == len(y)
                        else None
                    )
                    self._fit_model(
                        model=model,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_test,
                        y_val=y_test,
                        sample_weights=w_train,
                        train_timestamps=train_ts,
                        val_timestamps=test_ts,
                    )

                    y_pred = np.asarray(self._predict_with_model(model, X_test))
                    y_proba = np.asarray(
                        self._get_predictions_proba(model, X_test, timestamps=test_ts)
                    )
                    train_proba = np.asarray(
                        self._get_predictions_proba(model, X_train, timestamps=train_ts)
                    )
                    train_forward_returns = (
                        np.asarray(self.primary_forward_returns[train_idx], dtype=float)
                        if isinstance(self.primary_forward_returns, np.ndarray)
                        and len(self.primary_forward_returns) == len(y)
                        else None
                    )
                    train_regimes = (
                        np.asarray(regimes[train_idx], dtype=object)
                        if regimes is not None
                        else None
                    )
                    long_threshold, short_threshold = self._derive_signal_thresholds(
                        train_proba,
                        train_labels=y_train,
                        train_returns=train_forward_returns,
                        train_regimes=train_regimes,
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
                    fold_event_directions = (
                        np.asarray(self.primary_event_directions[test_idx], dtype=float)
                        if isinstance(self.primary_event_directions, np.ndarray)
                        and len(self.primary_event_directions) == len(y)
                        else None
                    )
                    fold_timestamps = (
                        np.asarray(self.timestamps[test_idx], dtype="datetime64[ns]")
                        if self.timestamps is not None and len(self.timestamps) == len(y)
                        else None
                    )
                    fold_symbols = (
                        np.asarray(row_symbols[test_idx], dtype=object)
                        if row_symbols is not None
                        else None
                    )

                    metrics = self._calculate_fold_metrics(
                        y_eval[-eval_len:],
                        y_pred[-eval_len:],
                        y_proba[-eval_len:],
                        long_threshold=long_threshold,
                        short_threshold=short_threshold,
                        realized_forward_returns=(
                            fold_forward_returns[-eval_len:]
                            if fold_forward_returns is not None
                            else None
                        ),
                        event_net_returns=(
                            fold_event_returns[-eval_len:]
                            if fold_event_returns is not None
                            else None
                        ),
                        event_directions=(
                            fold_event_directions[-eval_len:]
                            if fold_event_directions is not None
                            else None
                        ),
                        timestamps=(
                            fold_timestamps[-eval_len:] if fold_timestamps is not None else None
                        ),
                        symbols=(fold_symbols[-eval_len:] if fold_symbols is not None else None),
                    )
                    fold_score = float(metrics["risk_adjusted_score"])
                    if not np.isfinite(fold_score):
                        return -1e9
                    equity_break = float(metrics.get("equity_break", 0.0)) > 0.5
                    if equity_break:
                        fold_score -= 0.75
                    fold_scores.append(fold_score)
                    fold_weight = self._fold_reliability_weight(
                        metrics,
                        evaluation_size=eval_len,
                    )
                    if equity_break:
                        fold_weight = min(fold_weight, 0.35)
                    fold_weights.append(fold_weight)

                    trial.report(
                        float(
                            self._aggregate_fold_objective(fold_scores, fold_weights)
                            - self._inner_trial_stability_penalty(fold_scores, fold_weights)
                            - param_robustness_penalty
                        ),
                        step=fold_idx,
                    )
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                self.logger.warning(f"Optuna trial failed for {self.config.model_type}: {exc}")
                return -1e9

            if not fold_scores:
                return -1e9
            return float(
                self._aggregate_fold_objective(fold_scores, fold_weights)
                - self._inner_trial_stability_penalty(fold_scores, fold_weights)
                - param_robustness_penalty
            )

        sampler = optuna.samplers.TPESampler(seed=self.config.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        study, study_info = self._prepare_optuna_study(
            optuna,
            namespace="primary_search",
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
        self._record_optuna_study_metrics("optuna", study_info)
        if int(study_info.get("running_trials", 0)) > 0:
            self.logger.warning(
                "Optuna study %s contains %d stale running trial(s) from a prior interruption; "
                "new trials will continue from durable storage.",
                study_info["study_name"],
                int(study_info["running_trials"]),
            )
        remaining_trials = int(study_info.get("remaining_trials", self.config.n_trials))
        if remaining_trials > 0:
            study.optimize(
                objective,
                n_trials=remaining_trials,
                n_jobs=1,
                gc_after_trial=True,
                show_progress_bar=False,
            )
        else:
            self.logger.info(
                "Optuna study %s already has %d finalized trials; reusing saved search state.",
                study_info["study_name"],
                int(study_info.get("finalized_trials", 0)),
            )
        study_info.update(self._optuna_trial_state_summary(study, optuna))
        study_info["remaining_trials"] = int(
            max(int(self.config.n_trials) - int(study_info.get("finalized_trials", 0)), 0)
        )
        study_info["status"] = "completed"
        study_info["recorded_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path = self._write_optuna_study_manifest(study_info)
        if manifest_path is not None:
            study_info["manifest_path"] = manifest_path
        self._record_optuna_study_metrics("optuna", study_info)

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
        regimes = self.regimes if self.regimes is not None and len(self.regimes) == len(y) else None
        timestamps = (
            self.timestamps
            if self.timestamps is not None and len(self.timestamps) == len(X)
            else None
        )
        row_symbols = (
            self.row_symbols
            if self.row_symbols is not None and len(self.row_symbols) == len(y)
            else None
        )
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
                outer_splits = self._split_with_optional_times(outer_cv, X, y, times=timestamps)
        else:
            outer_splits = self._split_with_optional_times(outer_cv, X, y, times=timestamps)
        if not outer_splits:
            raise ValueError("Nested walk-forward requires at least one outer split.")

        min_train_size = min(len(train_idx) for train_idx, _ in outer_splits)
        search_space = self._get_optuna_search_space(min_train_size=min_train_size)
        if not search_space:
            self.logger.info(
                "No Optuna search space for this model type, keeping configured defaults"
            )
            return

        total_trials = 0
        total_pruned = 0
        self.nested_cv_trace = []
        outer_candidates: list[tuple[float, float, float, dict[str, Any], int, float]] = []
        outer_unstable_candidates: list[tuple[float, float, float, dict[str, Any], int, float]] = []
        outer_stability_ratios: list[float] = []

        for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_splits, start=1):
            X_outer_train = X[outer_train_idx]
            y_outer_train = y[outer_train_idx]
            X_outer_test = X[outer_test_idx]
            y_outer_test = y[outer_test_idx]
            if self._requires_binary_class_support() and not self._has_binary_class_support(
                y_outer_train
            ):
                self.logger.warning(
                    "Outer fold %s: training labels contain a single class; fold skipped.",
                    outer_fold,
                )
                continue
            w_outer_train = weights[outer_train_idx] if weights is not None else None
            regimes_outer_train = (
                np.asarray(regimes[outer_train_idx], dtype=object) if regimes is not None else None
            )
            symbols_outer_train = (
                np.asarray(row_symbols[outer_train_idx], dtype=object)
                if row_symbols is not None
                else None
            )
            symbols_outer_test = (
                np.asarray(row_symbols[outer_test_idx], dtype=object)
                if row_symbols is not None
                else None
            )
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
            signal_outer_train = (
                np.asarray(self.primary_event_directions[outer_train_idx], dtype=float)
                if isinstance(self.primary_event_directions, np.ndarray)
                and len(self.primary_event_directions) == len(y)
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
            signal_outer_test = (
                np.asarray(self.primary_event_directions[outer_test_idx], dtype=float)
                if isinstance(self.primary_event_directions, np.ndarray)
                and len(self.primary_event_directions) == len(y)
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
                    inner_splits = self._split_with_optional_times(
                        inner_cv,
                        X_outer_train,
                        y_outer_train,
                        times=ts_outer_train,
                    )
            else:
                inner_splits = self._split_with_optional_times(
                    inner_cv,
                    X_outer_train,
                    y_outer_train,
                    times=ts_outer_train,
                )
            if not inner_splits:
                self.logger.warning(
                    f"Outer fold {outer_fold}: no valid inner splits, skipping fold."
                )
                continue

            def objective(
                trial: "optuna.Trial",
                _inner_splits: list[tuple[np.ndarray, np.ndarray]] = inner_splits,
                _X_outer_train: np.ndarray = X_outer_train,
                _y_outer_train: np.ndarray = y_outer_train,
                _w_outer_train: np.ndarray | None = w_outer_train,
                _forward_outer_train: np.ndarray | None = forward_outer_train,
                _event_outer_train: np.ndarray | None = event_outer_train,
                _regimes_outer_train: np.ndarray | None = regimes_outer_train,
                _symbols_outer_train: np.ndarray | None = symbols_outer_train,
                _ts_outer_train: np.ndarray | None = ts_outer_train,
            ) -> float:
                params = search_space(trial)
                param_robustness_penalty = self._parameter_robustness_penalty(
                    self.config.model_type,
                    params,
                )
                fold_scores: list[float] = []
                fold_weights: list[float] = []
                component_samples: list[dict[str, float]] = []

                try:
                    for inner_idx, (inner_train_idx, inner_val_idx) in enumerate(_inner_splits):
                        X_train = _X_outer_train[inner_train_idx]
                        X_val = _X_outer_train[inner_val_idx]
                        y_train = _y_outer_train[inner_train_idx]
                        y_val = _y_outer_train[inner_val_idx]
                        if (
                            self._requires_binary_class_support()
                            and not self._has_binary_class_support(y_train)
                        ):
                            continue
                        fold_params = self._prepare_params_for_train_size(
                            params, len(inner_train_idx)
                        )
                        fold_params = self._augment_params_for_train_labels(fold_params, y_train)
                        model = self._create_model(params=fold_params)
                        w_train = (
                            _w_outer_train[inner_train_idx] if _w_outer_train is not None else None
                        )
                        self._fit_model(
                            model=model,
                            X_train=X_train,
                            y_train=y_train,
                            X_val=X_val,
                            y_val=y_val,
                            sample_weights=w_train,
                            train_timestamps=(
                                np.asarray(_ts_outer_train[inner_train_idx], dtype="datetime64[ns]")
                                if _ts_outer_train is not None
                                else None
                            ),
                            val_timestamps=(
                                np.asarray(_ts_outer_train[inner_val_idx], dtype="datetime64[ns]")
                                if _ts_outer_train is not None
                                else None
                            ),
                        )

                        y_pred = np.asarray(self._predict_with_model(model, X_val))
                        y_proba = np.asarray(
                            self._get_predictions_proba(
                                model,
                                X_val,
                                timestamps=(
                                    np.asarray(
                                        _ts_outer_train[inner_val_idx], dtype="datetime64[ns]"
                                    )
                                    if _ts_outer_train is not None
                                    else None
                                ),
                            )
                        )
                        train_proba = np.asarray(
                            self._get_predictions_proba(
                                model,
                                X_train,
                                timestamps=(
                                    np.asarray(
                                        _ts_outer_train[inner_train_idx], dtype="datetime64[ns]"
                                    )
                                    if _ts_outer_train is not None
                                    else None
                                ),
                            )
                        )
                        train_forward_returns = (
                            np.asarray(_forward_outer_train[inner_train_idx], dtype=float)
                            if _forward_outer_train is not None
                            else None
                        )
                        train_regimes = (
                            np.asarray(_regimes_outer_train[inner_train_idx], dtype=object)
                            if _regimes_outer_train is not None
                            else None
                        )
                        long_threshold, short_threshold = self._derive_signal_thresholds(
                            train_proba,
                            train_labels=y_train,
                            train_returns=train_forward_returns,
                            train_regimes=train_regimes,
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
                            event_directions=(
                                signal_outer_train[inner_val_idx][-eval_len:]
                                if signal_outer_train is not None
                                else None
                            ),
                            timestamps=(
                                _ts_outer_train[inner_val_idx][-eval_len:]
                                if _ts_outer_train is not None
                                else None
                            ),
                            symbols=(
                                _symbols_outer_train[inner_val_idx][-eval_len:]
                                if _symbols_outer_train is not None
                                else None
                            ),
                        )
                        fold_score = float(metrics["risk_adjusted_score"])
                        if not np.isfinite(fold_score):
                            return -1e9
                        equity_break = float(metrics.get("equity_break", 0.0)) > 0.5
                        if equity_break:
                            fold_score -= 0.75
                        fold_scores.append(fold_score)
                        fold_weight = self._fold_reliability_weight(
                            metrics,
                            evaluation_size=eval_len,
                        )
                        if equity_break:
                            fold_weight = min(fold_weight, 0.35)
                        fold_weights.append(fold_weight)
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
                                "objective_tail_risk_penalty": float(
                                    metrics.get("objective_tail_risk_penalty", 0.0)
                                ),
                                "objective_symbol_concentration_penalty": float(
                                    metrics.get("objective_symbol_concentration_penalty", 0.0)
                                ),
                                "objective_symbol_tail_penalty": float(
                                    metrics.get("objective_symbol_tail_penalty", 0.0)
                                ),
                                "objective_skew_penalty": float(
                                    metrics.get("objective_skew_penalty", 0.0)
                                ),
                                "objective_equity_break_penalty": float(
                                    metrics.get("objective_equity_break_penalty", 0.0)
                                ),
                            }
                        )

                        aggregate_score = self._aggregate_fold_objective(fold_scores, fold_weights)
                        stability_penalty = self._inner_trial_stability_penalty(
                            fold_scores,
                            fold_weights,
                        )
                        trial.report(
                            float(aggregate_score - stability_penalty - param_robustness_penalty),
                            step=inner_idx,
                        )
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
                aggregate_score = self._aggregate_fold_objective(fold_scores, fold_weights)
                stability_penalty = self._inner_trial_stability_penalty(
                    fold_scores,
                    fold_weights,
                )
                trial.set_user_attr("param_robustness_penalty", float(param_robustness_penalty))
                trial.set_user_attr("inner_stability_penalty", float(stability_penalty))
                return float(aggregate_score - stability_penalty - param_robustness_penalty)

            study, study_info = self._prepare_optuna_study(
                optuna,
                namespace=f"nested_outer_{int(outer_fold):02d}",
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.config.seed + outer_fold),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
            )
            nested_studies = dict(self.training_metrics.get("nested_optuna_studies", {}) or {})
            nested_studies[str(outer_fold)] = self._json_safe(study_info)
            self.training_metrics["nested_optuna_studies"] = nested_studies
            if int(study_info.get("running_trials", 0)) > 0:
                self.logger.warning(
                    "Outer fold %s: Optuna study %s contains %d stale running trial(s); "
                    "resuming from durable storage.",
                    outer_fold,
                    study_info["study_name"],
                    int(study_info["running_trials"]),
                )
            remaining_trials = int(study_info.get("remaining_trials", self.config.n_trials))
            if remaining_trials > 0:
                study.optimize(
                    objective,
                    n_trials=remaining_trials,
                    n_jobs=1,
                    gc_after_trial=True,
                    show_progress_bar=False,
                )
            else:
                self.logger.info(
                    "Outer fold %s: reusing Optuna study %s with %d finalized trials.",
                    outer_fold,
                    study_info["study_name"],
                    int(study_info.get("finalized_trials", 0)),
                )
            study_info.update(self._optuna_trial_state_summary(study, optuna))
            study_info["remaining_trials"] = int(
                max(int(self.config.n_trials) - int(study_info.get("finalized_trials", 0)), 0)
            )
            study_info["status"] = "completed"
            study_info["recorded_at"] = datetime.now(timezone.utc).isoformat()
            manifest_path = self._write_optuna_study_manifest(study_info)
            if manifest_path is not None:
                study_info["manifest_path"] = manifest_path
            nested_studies[str(outer_fold)] = self._json_safe(study_info)
            self.training_metrics["nested_optuna_studies"] = nested_studies

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
            trial_values = [float(t.value) for t in completed_trials]
            outer_surface_stable, outer_stability_ratio = self._outer_trial_stability_gate(
                trial_values,
                float(best_trial.value),
            )
            outer_stability_ratios.append(float(outer_stability_ratio))
            best_params = dict(best_trial.params)
            best_fold_params = self._prepare_params_for_train_size(
                best_params, len(outer_train_idx)
            )
            best_fold_params = self._augment_params_for_train_labels(
                best_fold_params, y_outer_train
            )

            model = self._create_model(params=best_fold_params)
            try:
                self._fit_model(
                    model=model,
                    X_train=X_outer_train,
                    y_train=y_outer_train,
                    X_val=X_outer_test,
                    y_val=y_outer_test,
                    sample_weights=w_outer_train,
                    train_timestamps=ts_outer_train,
                    val_timestamps=ts_outer_test,
                )
            except Exception as exc:
                self.logger.warning(
                    "Outer fold %s: final fit failed (%s), fold skipped.",
                    outer_fold,
                    exc,
                )
                continue

            y_pred = np.asarray(self._predict_with_model(model, X_outer_test))
            y_proba = np.asarray(
                self._get_predictions_proba(model, X_outer_test, timestamps=ts_outer_test)
            )
            train_proba = np.asarray(
                self._get_predictions_proba(model, X_outer_train, timestamps=ts_outer_train)
            )
            long_threshold, short_threshold = self._derive_signal_thresholds(
                train_proba,
                train_labels=y_outer_train,
                train_returns=forward_outer_train,
                train_regimes=regimes_outer_train,
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
                event_directions=(
                    signal_outer_test[-eval_len:] if signal_outer_test is not None else None
                ),
                timestamps=(ts_outer_test[-eval_len:] if ts_outer_test is not None else None),
                symbols=(
                    symbols_outer_test[-eval_len:] if symbols_outer_test is not None else None
                ),
            )
            outer_score = float(metrics.get("risk_adjusted_score", -1e9))
            if not np.isfinite(outer_score):
                continue
            outer_reliability = self._fold_reliability_weight(
                metrics,
                evaluation_size=eval_len,
            )
            outer_adjusted_score = float(outer_score * (0.20 + (0.80 * outer_reliability)))
            candidate_row = (
                outer_adjusted_score,
                outer_score,
                outer_reliability,
                best_params,
                len(outer_train_idx),
                float(outer_stability_ratio),
            )
            if outer_surface_stable:
                outer_candidates.append(candidate_row)
            else:
                outer_unstable_candidates.append(candidate_row)
                self.logger.warning(
                    "Outer fold %s: rejected due to unstable Optuna surface "
                    "(ratio=%.4f > cap=%.4f).",
                    outer_fold,
                    outer_stability_ratio,
                    float(self.config.nested_outer_stability_ratio_cap),
                )
            self.nested_cv_trace.append(
                {
                    "outer_fold": int(outer_fold),
                    "outer_train_size": int(len(outer_train_idx)),
                    "outer_test_size": int(eval_len),
                    "inner_split_count": int(len(inner_splits)),
                    "inner_best_score": float(best_trial.value),
                    "outer_score": outer_score,
                    "outer_adjusted_score": outer_adjusted_score,
                    "outer_reliability": outer_reliability,
                    "outer_stability_ratio": float(outer_stability_ratio),
                    "outer_candidate_status": (
                        "accepted" if outer_surface_stable else "rejected_unstable"
                    ),
                    "optuna_study_name": str(study_info.get("study_name", "")),
                    "optuna_storage_path": (
                        str(study_info["storage_path"])
                        if study_info.get("storage_path") is not None
                        else None
                    ),
                    "optuna_manifest_path": (
                        str(study_info["manifest_path"])
                        if study_info.get("manifest_path") is not None
                        else None
                    ),
                    "optuna_resumed_from_storage": bool(
                        study_info.get("resumed_from_storage", False)
                    ),
                    "optuna_existing_trials": int(study_info.get("existing_trials", 0)),
                    "optuna_finalized_trials": int(study_info.get("finalized_trials", 0)),
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
                        "objective_cvar_penalty": float(metrics.get("objective_cvar_penalty", 0.0)),
                        "objective_tail_risk_penalty": float(
                            metrics.get("objective_tail_risk_penalty", 0.0)
                        ),
                        "objective_symbol_concentration_penalty": float(
                            metrics.get("objective_symbol_concentration_penalty", 0.0)
                        ),
                        "objective_symbol_tail_penalty": float(
                            metrics.get("objective_symbol_tail_penalty", 0.0)
                        ),
                        "objective_skew_penalty": float(metrics.get("objective_skew_penalty", 0.0)),
                        "objective_equity_break_penalty": float(
                            metrics.get("objective_equity_break_penalty", 0.0)
                        ),
                    },
                }
            )

        selected_outer_candidates, stability_fallback_used = self._select_nested_outer_candidates(
            outer_candidates,
            outer_unstable_candidates,
        )

        (
            best_outer_adjusted_score,
            best_outer_score,
            best_outer_reliability,
            best_outer_params,
            best_outer_train_size,
            best_outer_stability_ratio,
        ) = max(selected_outer_candidates, key=lambda x: x[0])
        self.config.model_params = {**self.config.model_params, **best_outer_params}
        self.config.model_params = self._prepare_params_for_train_size(
            self.config.model_params, best_outer_train_size
        )

        outer_scores = [raw_score for _, raw_score, _, _, _, _ in selected_outer_candidates]
        outer_adjusted_scores = [
            adjusted_score for adjusted_score, _, _, _, _, _ in selected_outer_candidates
        ]
        self.training_metrics["optuna_trials"] = int(total_trials)
        self.training_metrics["optuna_pruned_trials"] = int(total_pruned)
        self.training_metrics["optuna_best_score"] = float(best_outer_adjusted_score)
        self.training_metrics["nested_outer_folds"] = int(len(self.nested_cv_trace))
        self.training_metrics["nested_outer_candidate_folds"] = int(len(outer_candidates))
        self.training_metrics["nested_outer_unstable_fold_count"] = int(
            len(outer_unstable_candidates)
        )
        self.training_metrics["nested_outer_stability_fallback_used"] = bool(
            stability_fallback_used
        )
        self.training_metrics["nested_inner_splits"] = int(self.config.nested_inner_splits)
        self.training_metrics["nested_mean_outer_score"] = float(np.mean(outer_scores))
        self.training_metrics["nested_mean_outer_adjusted_score"] = float(
            np.mean(outer_adjusted_scores)
        )
        self.training_metrics["nested_best_outer_score"] = float(best_outer_score)
        self.training_metrics["nested_best_outer_adjusted_score"] = float(best_outer_adjusted_score)
        self.training_metrics["nested_best_outer_reliability"] = float(best_outer_reliability)
        self.training_metrics["nested_best_outer_stability_ratio"] = float(
            best_outer_stability_ratio
        )
        if outer_stability_ratios:
            self.training_metrics["nested_outer_stability_ratio_mean"] = float(
                np.mean(outer_stability_ratios)
            )
            self.training_metrics["nested_outer_stability_ratio_max"] = float(
                np.max(outer_stability_ratios)
            )
        self.training_metrics["nested_cv_trace"] = self.nested_cv_trace

        if self.nested_cv_trace:
            component_keys = [
                "objective_sharpe_component",
                "objective_drawdown_penalty",
                "objective_turnover_penalty",
                "objective_calibration_penalty",
                "objective_trade_activity_penalty",
                "objective_cvar_penalty",
                "objective_tail_risk_penalty",
                "objective_symbol_concentration_penalty",
                "objective_symbol_tail_penalty",
                "objective_skew_penalty",
                "objective_equity_break_penalty",
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
            f"best_outer_adjusted_score={best_outer_adjusted_score:.6f}, "
            f"best_outer_score={best_outer_score:.6f}"
        )

    def _optimize_with_grid_search(self) -> None:
        """Grid search optimization."""
        if self._is_ranker_model():
            raise ValueError(
                "Grid search is disabled for ranker models. Use Optuna with timestamp-query scoring."
            )
        from sklearn.model_selection import GridSearchCV

        model = self._create_base_model()
        scoring: str | None
        if self._is_regression_model():
            scoring = "neg_mean_squared_error"
        else:
            scoring = "roc_auc"

        if self.config.model_type == "xgboost":
            param_grid = {
                "n_estimators": [100, 300, 500],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
            }
        elif self._is_lightgbm_family_model():
            param_grid = {
                "n_estimators": [100, 300, 500],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
            }
        else:
            param_grid = {}

        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring=scoring, n_jobs=self.config.n_jobs
        )
        grid_search.fit(self.features, self.labels)

        self.config.model_params = grid_search.best_params_
        self.logger.info(f"Best params: {grid_search.best_params_}")

    def _optimize_with_random_search(self) -> None:
        """Random search optimization."""
        if self._is_ranker_model():
            raise ValueError(
                "Random search is disabled for ranker models. Use Optuna with timestamp-query scoring."
            )
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import randint, uniform

        model = self._create_base_model()
        scoring: str | None
        if self._is_regression_model():
            scoring = "neg_mean_squared_error"
        else:
            scoring = "roc_auc"

        if self.config.model_type == "xgboost":
            param_distributions = {
                "n_estimators": randint(100, 1000),
                "max_depth": randint(3, 10),
                "learning_rate": uniform(0.01, 0.29),
            }
        else:
            param_distributions = {}

        random_search = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=self.config.n_trials,
            cv=3,
            scoring=scoring,
            n_jobs=self.config.n_jobs,
        )
        random_search.fit(self.features, self.labels)

        self.config.model_params = random_search.best_params_
        self.logger.info(f"Best params: {random_search.best_params_}")

    def _get_optuna_search_space(self, min_train_size: int | None = None):
        """Return model-specific Optuna search space function."""
        if self.config.model_type in {"xgboost", "xgboost_regressor"}:
            effective_train = max(128, int(min_train_size if min_train_size is not None else 1024))
            estimators_low = max(180, min(320, int(np.ceil(float(effective_train) * 0.28))))
            estimators_high = max(360, min(760, int(np.ceil(float(effective_train) * 0.95))))
            if estimators_high <= estimators_low:
                estimators_high = estimators_low + 40
            max_depth_high = 6 if effective_train <= 1200 else 7
            min_child_high = max(6, min(10, int(np.ceil(np.sqrt(float(effective_train)) * 0.24))))
            base_space = lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", estimators_low, estimators_high),
                "max_depth": trial.suggest_int("max_depth", 3, max_depth_high),
                "learning_rate": trial.suggest_float("learning_rate", 0.012, 0.08, log=True),
                "subsample": trial.suggest_float("subsample", 0.72, 0.92),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.72, 0.92),
                "min_child_weight": trial.suggest_int("min_child_weight", 3, min_child_high),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.5, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.3, 4.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 1.25),
            }
            if self.config.model_type == "xgboost":
                return lambda trial: {
                    **base_space(trial),
                    "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.9, 3.0, log=True),
                    "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 1.5),
                }
            return base_space

        if self._is_lightgbm_family_model():
            effective_train = max(128, int(min_train_size if min_train_size is not None else 1024))
            leaf_floor, leaf_ceiling = self._resolve_lightgbm_leaf_bounds(effective_train)
            num_leaves_high = max(24, min(96, int(np.ceil(np.sqrt(float(effective_train)) * 2.0))))
            estimators_high = max(260, min(520, int(np.ceil(float(effective_train) * 0.60))))

            base_space = lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 180, estimators_high),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, num_leaves_high),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.75, 0.90),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.76, 0.92),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 4),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", leaf_floor, leaf_ceiling),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.05, 8.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.5, 10.0, log=True),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.03, 0.35),
                "max_bin": trial.suggest_int("max_bin", 127, 191),
            }
            if self.config.model_type == "lightgbm":
                return lambda trial: {
                    **base_space(trial),
                    "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.95, 1.15),
                }
            return base_space

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
                "lookback_window": trial.suggest_categorical(
                    "lookback_window", lookback_candidates
                ),
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
                "lookback_window": trial.suggest_categorical(
                    "lookback_window", lookback_candidates
                ),
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
                "lookback_window": trial.suggest_categorical(
                    "lookback_window", lookback_candidates
                ),
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

    @staticmethod
    def _resolve_lightgbm_leaf_bounds(train_size: int | None) -> tuple[int, int]:
        """Return a realistic min-data-in-leaf search range for LightGBM folds."""
        effective_train = max(128, int(train_size if train_size is not None else 1024))
        if effective_train <= 512:
            leaf_floor = LIGHTGBM_MIN_DATA_IN_LEAF_SEARCH_MIN
        elif effective_train <= 5_000:
            leaf_floor = 20
        elif effective_train <= 50_000:
            leaf_floor = 24
        else:
            leaf_floor = 32

        leaf_ceiling = min(
            LIGHTGBM_MIN_DATA_IN_LEAF_SEARCH_MAX,
            max(
                leaf_floor + 12,
                int(np.ceil(np.sqrt(float(effective_train)) * 0.35)),
            ),
        )
        return int(leaf_floor), int(max(leaf_ceiling, leaf_floor))

    @classmethod
    def _resolve_lightgbm_leaf_cap(cls, train_size: int) -> int:
        """Return the maximum leaf size allowed after fold-size adjustment."""
        leaf_floor, _ = cls._resolve_lightgbm_leaf_bounds(train_size)
        effective_train = max(128, int(train_size))
        leaf_cap = min(
            LIGHTGBM_MIN_DATA_IN_LEAF_FIT_MAX,
            max(
                leaf_floor + 12,
                int(np.ceil(np.sqrt(float(effective_train)) * 0.45)),
            ),
        )
        return int(max(leaf_cap, leaf_floor))

    def _prepare_params_for_train_size(self, params: dict, train_size: int) -> dict:
        """Adjust sequence model params for current CV fold size."""
        adjusted = dict(params)

        if self._is_lightgbm_family_model():
            leaf_floor, _ = self._resolve_lightgbm_leaf_bounds(train_size)
            leaf_cap = self._resolve_lightgbm_leaf_cap(train_size)
            existing_leaf = int(
                adjusted.get("min_data_in_leaf", adjusted.get("min_child_samples", 20))
            )
            effective_leaf = int(np.clip(existing_leaf, leaf_floor, leaf_cap))
            adjusted["min_data_in_leaf"] = effective_leaf
            adjusted["min_child_samples"] = effective_leaf

            max_leaves = max(16, min(96, int(max(16, train_size // max(effective_leaf, 1)))))
            existing_leaves = int(adjusted.get("num_leaves", 31))
            adjusted["num_leaves"] = int(np.clip(existing_leaves, 16, max_leaves))

            existing_depth = int(adjusted.get("max_depth", 6))
            adjusted["max_depth"] = int(min(existing_depth, 6))

            existing_estimators = int(adjusted.get("n_estimators", 500))
            adjusted["n_estimators"] = int(min(existing_estimators, 420))

            existing_lr = float(adjusted.get("learning_rate", 0.03))
            adjusted["learning_rate"] = float(np.clip(existing_lr, 0.015, 0.06))

            adjusted["feature_fraction"] = float(
                np.clip(float(adjusted.get("feature_fraction", 0.78)), 0.70, 0.90)
            )
            adjusted["bagging_fraction"] = float(
                np.clip(float(adjusted.get("bagging_fraction", 0.82)), 0.70, 0.94)
            )
            adjusted["bagging_freq"] = max(1, int(adjusted.get("bagging_freq", 1)))
            adjusted["lambda_l1"] = max(0.10, float(adjusted.get("lambda_l1", 0.10)))
            adjusted["lambda_l2"] = max(1.00, float(adjusted.get("lambda_l2", 1.00)))
            if self.config.model_type == "lightgbm":
                adjusted["scale_pos_weight"] = float(
                    np.clip(float(adjusted.get("scale_pos_weight", 1.0)), 0.90, 1.30)
                )
            else:
                adjusted.pop("scale_pos_weight", None)

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

    def _fold_reliability_weight(
        self,
        metrics: dict[str, float],
        evaluation_size: int | None = None,
    ) -> float:
        """Estimate fold reliability from trade participation and sample support."""
        eval_size = int(
            evaluation_size if evaluation_size is not None else metrics.get("test_size", 0)
        )
        eval_size = max(1, eval_size)
        target_trades = float(self._effective_trade_target(eval_size))
        trade_count = float(max(0.0, metrics.get("trade_count", 0.0)))
        trade_obs = float(
            max(
                0.0,
                metrics.get(
                    "trade_return_observations",
                    metrics.get("active_net_return_observations", trade_count),
                ),
            )
        )
        active_rate = float(np.clip(metrics.get("active_signal_rate", 0.0), 0.0, 1.0))
        sharpe_confidence = float(
            np.clip(metrics.get("sharpe_observation_confidence", 1.0), 0.0, 1.0)
        )

        trade_coverage = float(np.clip(trade_count / max(target_trades, 1.0), 0.0, 1.0))
        obs_target = max(6.0, target_trades * 1.20)
        observation_coverage = float(np.clip(trade_obs / obs_target, 0.0, 1.0))
        active_floor = float(np.clip(target_trades / float(eval_size), 0.02, 0.20))
        activity_coverage = float(np.clip(active_rate / max(active_floor, 1e-9), 0.0, 1.0))

        weight = 0.20 + (
            0.80
            * ((0.45 * trade_coverage) + (0.35 * observation_coverage) + (0.20 * activity_coverage))
        )
        weight *= float(np.clip(0.70 + (0.30 * sharpe_confidence), 0.65, 1.0))

        sharpe = float(metrics.get("sharpe", 0.0))
        if sharpe < 0.0:
            weight *= float(np.clip(1.0 + (0.12 * sharpe), 0.65, 1.0))
        win_rate = float(np.clip(metrics.get("win_rate", 0.5), 0.0, 1.0))
        if trade_obs >= max(4.0, target_trades * 0.35) and win_rate < 0.45:
            weight *= float(np.clip(0.85 + ((win_rate - 0.30) * 0.60), 0.72, 1.0))

        return float(np.clip(weight, 0.10, 1.0))

    @staticmethod
    def _has_binary_class_support(y: np.ndarray | list[float]) -> bool:
        """Return True when label vector contains both classes."""
        arr = np.asarray(y, dtype=float).reshape(-1)
        if arr.size <= 1:
            return False
        y_bin = (arr >= 0.5).astype(int)
        return bool(np.unique(y_bin).size >= 2)

    @staticmethod
    def _aggregate_fold_objective(fold_scores: list[float], fold_weights: list[float]) -> float:
        """Aggregate fold objective with stability and downside penalties."""
        if not fold_scores:
            return -1e9

        n = min(len(fold_scores), len(fold_weights))
        if n <= 0:
            return -1e9

        raw_scores = np.asarray(fold_scores[:n], dtype=float)
        raw_weights = np.asarray(fold_weights[:n], dtype=float)
        finite_mask = np.isfinite(raw_scores)
        if not np.any(finite_mask):
            return -1e9

        scores = raw_scores[finite_mask]
        weights = raw_weights[finite_mask]
        weights = np.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)
        weights = np.clip(weights, 0.05, 1.0)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            weights = np.ones_like(scores, dtype=float)
            weight_sum = float(np.sum(weights))
        norm_weights = weights / weight_sum

        weighted_mean = float(np.sum(scores * norm_weights))
        dispersion = float(np.sqrt(np.sum(norm_weights * ((scores - weighted_mean) ** 2))))
        downside = float(np.sum(norm_weights * np.maximum(0.0, -scores)))
        lower_tail = float(np.quantile(scores, 0.25)) if scores.size > 0 else 0.0
        worst_score = float(np.min(scores)) if scores.size > 0 else 0.0
        tail_penalty = float(max(0.0, -lower_tail) + (0.40 * max(0.0, -worst_score)))
        negative_fold_ratio = float(np.mean(scores < 0.0)) if scores.size > 0 else 0.0
        sign_imbalance_penalty = float(np.clip(negative_fold_ratio - 0.35, 0.0, 1.0))

        return float(
            weighted_mean
            - (0.12 * dispersion)
            - (0.18 * downside)
            - (0.12 * tail_penalty)
            - (0.08 * sign_imbalance_penalty)
        )

    @staticmethod
    def _inner_trial_stability_penalty(
        fold_scores: list[float],
        fold_weights: list[float],
    ) -> float:
        """Compute additional nested-inner stability penalty for trial ranking."""
        if not fold_scores:
            return 0.0

        n = min(len(fold_scores), len(fold_weights))
        if n <= 1:
            return 0.0

        raw_scores = np.asarray(fold_scores[:n], dtype=float)
        raw_weights = np.asarray(fold_weights[:n], dtype=float)
        finite_mask = np.isfinite(raw_scores)
        if int(np.count_nonzero(finite_mask)) <= 1:
            return 0.0

        scores = raw_scores[finite_mask]
        weights = np.clip(
            np.nan_to_num(raw_weights[finite_mask], nan=0.0, posinf=1.0, neginf=0.0),
            0.05,
            1.0,
        )
        weights = weights / max(float(np.sum(weights)), 1e-9)

        weighted_mean = float(np.sum(scores * weights))
        dispersion = float(np.sqrt(np.sum(weights * ((scores - weighted_mean) ** 2))))
        lower_quartile = float(np.quantile(scores, 0.25))
        worst_fold = float(np.min(scores))
        negative_ratio = float(np.mean(scores < 0.0))
        iqr = float(np.quantile(scores, 0.75) - np.quantile(scores, 0.25))

        return float(
            (0.16 * dispersion)
            + (0.20 * max(0.0, -lower_quartile))
            + (0.18 * max(0.0, -worst_fold))
            + (0.10 * max(0.0, negative_ratio - 0.30))
            + (0.08 * max(0.0, iqr))
        )

    @staticmethod
    def _parameter_robustness_penalty(model_type: str, params: dict[str, Any]) -> float:
        """Penalize hyperparameter extremes that often correlate with overfitting."""
        mt = str(model_type).strip().lower()
        p = params if isinstance(params, dict) else {}
        penalty = 0.0

        if mt in {"xgboost", "xgboost_regressor"}:
            max_depth = float(p.get("max_depth", 5))
            n_estimators = float(p.get("n_estimators", 400))
            learning_rate = float(p.get("learning_rate", 0.05))
            subsample = float(p.get("subsample", 0.82))
            colsample = float(p.get("colsample_bytree", 0.82))
            min_child_weight = float(p.get("min_child_weight", 4))
            scale_pos_weight = float(p.get("scale_pos_weight", 1.0))

            penalty += 0.08 * max(0.0, max_depth - 5.0)
            penalty += 0.04 * max(0.0, (n_estimators - 620.0) / 180.0)
            penalty += 0.08 * max(0.0, (learning_rate - 0.055) / 0.025)
            penalty += 0.05 * max(0.0, abs(subsample - 0.82) / 0.10)
            penalty += 0.05 * max(0.0, abs(colsample - 0.82) / 0.10)
            penalty += 0.03 * max(0.0, (6.0 - min_child_weight) / 3.0)
            penalty += 0.05 * max(0.0, (scale_pos_weight - 2.2) / 0.8)
            return float(max(0.0, penalty))

        if mt in LIGHTGBM_FAMILY_MODELS:
            num_leaves = float(p.get("num_leaves", 24))
            max_depth = float(p.get("max_depth", 5))
            learning_rate = float(p.get("learning_rate", 0.03))
            min_leaf = float(p.get("min_data_in_leaf", 40))

            penalty += 0.06 * max(0.0, (num_leaves - 36.0) / 12.0)
            penalty += 0.05 * max(0.0, max_depth - 5.0)
            penalty += 0.06 * max(0.0, (learning_rate - 0.040) / 0.015)
            penalty += 0.04 * max(0.0, (35.0 - min_leaf) / 10.0)
            return float(max(0.0, penalty))

        return 0.0

    @staticmethod
    def _regime_edge_profile(
        signals: np.ndarray,
        returns_arr: np.ndarray | None,
        regimes_arr: np.ndarray | None,
    ) -> dict[str, float]:
        """Summarize regime-level active directional edge for robust thresholding."""
        if returns_arr is None or regimes_arr is None:
            return {
                "evaluated_regimes": 0.0,
                "mean_edge": 0.0,
                "worst_edge": 0.0,
                "edge_dispersion": 0.0,
                "negative_regime_share": 0.0,
            }

        signals_arr = np.asarray(signals, dtype=float).reshape(-1)
        returns = np.asarray(returns_arr, dtype=float).reshape(-1)
        regimes = np.asarray(regimes_arr, dtype=object).reshape(-1)
        n = min(len(signals_arr), len(returns), len(regimes))
        if n <= 0:
            return {
                "evaluated_regimes": 0.0,
                "mean_edge": 0.0,
                "worst_edge": 0.0,
                "edge_dispersion": 0.0,
                "negative_regime_share": 0.0,
            }

        signals_arr = signals_arr[:n]
        returns = np.nan_to_num(returns[:n], nan=0.0, posinf=0.0, neginf=0.0)
        regimes = regimes[:n].astype(str)
        active_mask = np.abs(signals_arr) > 1e-8
        if not np.any(active_mask):
            return {
                "evaluated_regimes": 0.0,
                "mean_edge": 0.0,
                "worst_edge": 0.0,
                "edge_dispersion": 0.0,
                "negative_regime_share": 0.0,
            }

        active_returns = signals_arr * returns
        regime_edges: list[float] = []
        regime_counts: list[int] = []
        min_regime_observations = max(20, int(round(n * 0.02)))

        for regime_name in sorted({str(x).strip() for x in regimes if str(x).strip()}):
            regime_mask = (regimes == regime_name) & active_mask
            obs = int(np.count_nonzero(regime_mask))
            if obs < min_regime_observations:
                continue
            regime_edges.append(float(np.mean(active_returns[regime_mask])))
            regime_counts.append(obs)

        if not regime_edges:
            return {
                "evaluated_regimes": 0.0,
                "mean_edge": 0.0,
                "worst_edge": 0.0,
                "edge_dispersion": 0.0,
                "negative_regime_share": 0.0,
            }

        edge_arr = np.asarray(regime_edges, dtype=float)
        count_weights = np.asarray(regime_counts, dtype=float)
        count_weights = count_weights / max(float(np.sum(count_weights)), 1e-9)
        return {
            "evaluated_regimes": float(edge_arr.size),
            "mean_edge": float(np.sum(edge_arr * count_weights)),
            "worst_edge": float(np.min(edge_arr)),
            "edge_dispersion": float(np.std(edge_arr)),
            "negative_regime_share": float(np.mean(edge_arr < 0.0)),
        }

    @staticmethod
    def _infer_regime_side_bias(
        train_regimes: np.ndarray | None,
        train_returns: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Infer long/short threshold shifts from regime mix and return drift."""
        if train_regimes is None:
            return 0.0, 0.0

        regimes = np.asarray(train_regimes, dtype=str).reshape(-1)
        if regimes.size == 0:
            return 0.0, 0.0

        regime_values = np.char.lower(np.char.strip(regimes))
        bull_tokens = ("bull", "uptrend", "risk_on", "breakout")
        bear_tokens = ("bear", "downtrend", "risk_off", "crash")
        bull_score = 0.0
        bear_score = 0.0
        for regime_name in regime_values:
            if any(token in regime_name for token in bull_tokens):
                bull_score += 1.0
            if any(token in regime_name for token in bear_tokens):
                bear_score += 1.0

        sample_size = float(len(regime_values))
        bull_share = float(bull_score / sample_size)
        bear_share = float(bear_score / sample_size)

        drift_bias = 0.0
        if train_returns is not None:
            returns = np.asarray(train_returns, dtype=float).reshape(-1)
            returns = returns[np.isfinite(returns)]
            if returns.size >= 25:
                mean_ret = float(np.mean(returns))
                pos_rate = float(np.mean(returns > 0.0))
                if mean_ret > 2e-5 and pos_rate >= 0.54:
                    drift_bias += 1.0
                elif mean_ret < -2e-5 and pos_rate <= 0.46:
                    drift_bias -= 1.0

        regime_bias = float(np.clip((bull_share - bear_share) + (0.45 * drift_bias), -1.0, 1.0))
        long_shift = float(np.clip(-0.06 * regime_bias, -0.08, 0.08))
        short_shift = float(np.clip(-0.10 * regime_bias, -0.12, 0.12))
        return long_shift, short_shift

    def _outer_trial_stability_gate(
        self,
        trial_values: list[float],
        best_value: float,
    ) -> tuple[bool, float]:
        """Check whether outer-fold Optuna surface is stable enough for promotion."""
        finite_values = [float(v) for v in trial_values if np.isfinite(float(v))]
        if len(finite_values) < int(self.config.nested_outer_stability_min_trials):
            return True, 0.0

        values = np.asarray(finite_values, dtype=float)
        dispersion = float(np.std(values))
        scale = float(max(0.35, abs(float(best_value))))
        stability_ratio = float(dispersion / scale)
        is_stable = bool(stability_ratio <= float(self.config.nested_outer_stability_ratio_cap))
        return is_stable, stability_ratio

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
        horizon_cap = int(np.ceil(float(eval_size) / float(max(2.0, horizon * 2.2))))
        dynamic_cap = max(3, min(density_cap, sqrt_cap, horizon_cap))
        return int(min(configured_target, dynamic_cap))

    def _derive_signal_thresholds(
        self,
        train_proba: np.ndarray | None,
        train_labels: np.ndarray | None = None,
        train_returns: np.ndarray | None = None,
        train_regimes: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Derive leakage-safe long/short thresholds from training probabilities."""
        default_long = 0.55
        default_short = 0.45
        ranker_mode = self._is_ranker_model()
        if train_proba is None:
            return default_long, default_short

        raw_values = np.asarray(train_proba, dtype=float).reshape(-1)
        finite_mask = np.isfinite(raw_values)
        values = raw_values[finite_mask]
        if values.size < 25:
            return default_long, default_short

        values = np.clip(values, 0.0, 1.0)
        distribution_summary = self._summarize_probability_distribution(values)
        if self._is_dead_probability_summary(distribution_summary):
            proba_center = float(np.clip(distribution_summary.get("mean", 0.5), 0.0, 1.0))
            long_threshold = float(np.clip(max(default_long, proba_center + 0.05), 0.55, 0.99))
            short_threshold = float(np.clip(min(default_short, proba_center - 0.05), 0.01, 0.45))
            if long_threshold <= short_threshold:
                long_threshold = 0.75
                short_threshold = 0.25
            return long_threshold, short_threshold

        target_trades = self._effective_trade_target(int(values.size))
        threshold_policy = self._horizon_threshold_policy()
        target_rate_base = min(0.35, max(0.05, float(target_trades) / float(values.size)))
        target_rate = float(
            np.clip(
                target_rate_base * float(threshold_policy["target_rate_scale"]),
                0.04,
                0.45,
            )
        )
        tail_rate = min(0.25, max(0.03, target_rate * 0.65))
        min_gap = float(np.clip(float(threshold_policy["min_gap"]), 0.025, 0.08))
        forced_long_cap = float(np.clip(float(threshold_policy["forced_long_cap"]), 0.55, 0.90))
        forced_short_floor = float(
            np.clip(float(threshold_policy["forced_short_floor"]), 0.10, 0.45)
        )

        short_threshold = float(np.quantile(values, tail_rate))
        long_threshold = float(np.quantile(values, 1.0 - tail_rate))

        short_threshold = min(0.49, max(0.01, short_threshold))
        long_threshold = max(0.51, min(0.99, long_threshold))

        labels_arr: np.ndarray | None = None
        label_positive_rate: float | None = None
        regression_mode = self._is_regression_model()
        if train_labels is not None:
            candidate_labels = np.asarray(train_labels, dtype=float).reshape(-1)
            if candidate_labels.size == raw_values.size:
                candidate_labels = candidate_labels[finite_mask]
            elif candidate_labels.size != values.size:
                candidate_labels = np.array([], dtype=float)
            if candidate_labels.size == values.size:
                if regression_mode:
                    labels_arr = np.nan_to_num(
                        candidate_labels,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    label_positive_rate = float(np.mean(labels_arr > 0.0))
                else:
                    labels_arr = np.clip(
                        np.nan_to_num(candidate_labels, nan=0.5, posinf=1.0, neginf=0.0),
                        0.0,
                        1.0,
                    )
                    label_positive_rate = float(np.mean(labels_arr >= 0.5))

        returns_arr: np.ndarray | None = None
        if train_returns is not None:
            candidate_returns = np.asarray(train_returns, dtype=float).reshape(-1)
            if candidate_returns.size == raw_values.size:
                candidate_returns = candidate_returns[finite_mask]
            elif candidate_returns.size != values.size:
                candidate_returns = np.array([], dtype=float)
            if candidate_returns.size == values.size:
                returns_arr = np.nan_to_num(candidate_returns, nan=0.0, posinf=0.0, neginf=0.0)

        regimes_arr: np.ndarray | None = None
        if train_regimes is not None:
            candidate_regimes = np.asarray(train_regimes, dtype=object).reshape(-1)
            if candidate_regimes.size == raw_values.size:
                candidate_regimes = candidate_regimes[finite_mask]
            elif candidate_regimes.size != values.size:
                candidate_regimes = np.array([], dtype=object)
            if candidate_regimes.size == values.size:
                regimes_arr = candidate_regimes

        if labels_arr is not None and labels_arr.size >= 40:
            y_bin = (
                (labels_arr > 0.0).astype(int)
                if regression_mode
                else (labels_arr >= 0.5).astype(int)
            )
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
                {
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
                }
            )
            candidate_pairs: list[tuple[float, float]] = [(tail, tail) for tail in symmetric_tails]
            if not ranker_mode:
                candidate_pairs.extend(
                    [
                        (long_tail_prior, short_tail_prior),
                        (min(0.40, long_tail_prior * 1.20), short_tail_prior),
                        (long_tail_prior, min(0.40, short_tail_prior * 1.20)),
                        (
                            min(0.40, max(long_tail_prior, 0.22)),
                            max(0.05, short_tail_prior * 0.70),
                        ),
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
                if cand_long - cand_short < max(0.025, min_gap * 0.75):
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

                signed_hits = ((signals == 1.0) & (y_bin == 1)) | ((signals == -1.0) & (y_bin == 0))
                directional_acc = float(np.mean(signed_hits[active_mask]))
                active_rate = float(active_count) / float(values.size)

                prev_signals = np.concatenate([[0.0], signals[:-1]])
                entry_mask = active_mask & (
                    (np.abs(prev_signals) <= 1e-8) | (np.sign(signals) != np.sign(prev_signals))
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
                calibration_penalty = 0.0 if ranker_mode else float(np.mean((values - y_bin) ** 2))
                score = (
                    (0.65 * directional_acc)
                    + (0.20 * activity_score)
                    + (0.15 * entry_score)
                    - (0.10 * calibration_penalty)
                    - (0.60 * activity_penalty)
                )
                if directional_acc < 0.50:
                    score -= float((0.50 - directional_acc) * 0.80)
                if returns_arr is not None and returns_arr.size == values.size:
                    active_returns = signals * returns_arr
                    edge_mean = float(np.mean(active_returns[active_mask]))
                    edge_score = float(np.tanh(edge_mean * 250.0))
                    long_mask = signals == 1.0
                    short_mask = signals == -1.0
                    long_frac = float(np.mean(long_mask[active_mask])) if active_count > 0 else 0.0
                    short_frac = (
                        float(np.mean(short_mask[active_mask])) if active_count > 0 else 0.0
                    )
                    long_edge = float(np.mean(returns_arr[long_mask])) if np.any(long_mask) else 0.0
                    short_edge = (
                        float(np.mean(-returns_arr[short_mask])) if np.any(short_mask) else 0.0
                    )
                    side_penalty = 0.0
                    if long_edge < 0.0:
                        side_penalty += long_frac * min(1.0, abs(long_edge) * 250.0)
                    if short_edge < 0.0:
                        side_penalty += short_frac * min(1.0, abs(short_edge) * 250.0)
                    score += (0.55 * edge_score) - (0.45 * side_penalty)
                    if regimes_arr is not None and regimes_arr.size == values.size:
                        regime_profile = self._regime_edge_profile(
                            signals=signals,
                            returns_arr=returns_arr,
                            regimes_arr=regimes_arr,
                        )
                        if regime_profile["evaluated_regimes"] >= 2.0:
                            regime_mean_edge = float(regime_profile["mean_edge"])
                            regime_worst_edge = float(regime_profile["worst_edge"])
                            regime_dispersion = float(regime_profile["edge_dispersion"])
                            negative_share = float(regime_profile["negative_regime_share"])
                            regime_reward = (0.35 * np.tanh(regime_mean_edge * 220.0)) + (
                                0.30 * np.tanh(regime_worst_edge * 260.0)
                            )
                            regime_penalty = (
                                (0.50 * max(0.0, -regime_worst_edge * 300.0))
                                + (0.20 * negative_share)
                                + (0.10 * regime_dispersion * 300.0)
                            )
                            score += float(regime_reward - regime_penalty)
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
                (np.abs(prev_signals) <= 1e-8) | (np.sign(signals) != np.sign(prev_signals))
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
            short_threshold = min(0.49, max(0.01, max(forced_short, forced_short_floor)))
            long_threshold = max(0.51, min(0.99, min(forced_long, forced_long_cap)))

        final_active_rate, _ = _activity_rates(long_threshold, short_threshold)
        min_final_activity = max(0.02, min(0.12, enforced_target_rate * 0.70))
        if final_active_rate < min_final_activity:
            proba_center = float(np.clip(np.mean(values), 0.45, 0.55))
            proba_dispersion = float(np.std(values))
            adaptive_band = float(np.clip(max(0.03, proba_dispersion * 0.90), 0.03, 0.12))
            long_threshold = float(np.clip(proba_center + adaptive_band, 0.52, 0.72))
            short_threshold = float(np.clip(proba_center - adaptive_band, 0.28, 0.48))

        if (not ranker_mode) and returns_arr is not None and returns_arr.size == values.size:
            side_signals = np.where(
                values >= long_threshold,
                1.0,
                np.where(values <= short_threshold, -1.0, 0.0),
            ).astype(float)
            long_mask = side_signals == 1.0
            short_mask = side_signals == -1.0
            min_side_samples = max(4, int(round(values.size * 0.01)))
            long_count = int(np.count_nonzero(long_mask))
            short_count = int(np.count_nonzero(short_mask))
            long_edge = (
                float(np.mean(returns_arr[long_mask])) if long_count >= min_side_samples else 0.0
            )
            short_edge = (
                float(np.mean(-returns_arr[short_mask])) if short_count >= min_side_samples else 0.0
            )
            long_edge_mag = abs(float(long_edge))
            short_edge_mag = abs(float(short_edge))
            severe_edge_cutoff = 4e-4
            if long_edge > 1e-5 and short_edge < -1e-5:
                short_cap = 0.12 if short_edge_mag >= severe_edge_cutoff else 0.18
                short_threshold = min(short_threshold, short_cap)
            elif short_edge > 1e-5 and long_edge < -1e-5:
                long_floor = 0.88 if long_edge_mag >= severe_edge_cutoff else 0.82
                long_threshold = max(long_threshold, long_floor)
            elif long_edge < -1e-5 and short_edge < -1e-5:
                edge_floor = (
                    0.92 if max(long_edge_mag, short_edge_mag) >= severe_edge_cutoff else 0.86
                )
                edge_cap = (
                    0.08 if max(long_edge_mag, short_edge_mag) >= severe_edge_cutoff else 0.14
                )
                long_threshold = max(long_threshold, edge_floor)
                short_threshold = min(short_threshold, edge_cap)

        if (not ranker_mode) and label_positive_rate is not None:
            if label_positive_rate >= 0.52:
                short_threshold = min(short_threshold, 0.20)
            elif label_positive_rate <= 0.48:
                long_threshold = max(long_threshold, 0.80)

        if not ranker_mode:
            regime_long_shift, regime_short_shift = self._infer_regime_side_bias(
                regimes_arr,
                returns_arr,
            )
            if abs(regime_long_shift) > 1e-9 or abs(regime_short_shift) > 1e-9:
                long_threshold = float(np.clip(long_threshold + regime_long_shift, 0.51, 0.95))
                short_threshold = float(np.clip(short_threshold + regime_short_shift, 0.05, 0.49))

        if (
            regimes_arr is not None
            and returns_arr is not None
            and regimes_arr.size == values.size
            and returns_arr.size == values.size
        ):
            final_signals = np.where(
                values >= long_threshold,
                1.0,
                np.where(values <= short_threshold, -1.0, 0.0),
            ).astype(float)
            final_regime_profile = self._regime_edge_profile(
                signals=final_signals,
                returns_arr=returns_arr,
                regimes_arr=regimes_arr,
            )
            if final_regime_profile["evaluated_regimes"] >= 2.0:
                worst_regime_edge = float(final_regime_profile["worst_edge"])
                negative_share = float(final_regime_profile["negative_regime_share"])
                safety_shift = 0.0
                if worst_regime_edge < -3e-4 or negative_share >= 0.70:
                    safety_shift = 0.03
                elif worst_regime_edge < -1e-4 or negative_share >= 0.50:
                    safety_shift = 0.015
                if safety_shift > 0.0:
                    long_threshold = float(np.clip(long_threshold + safety_shift, 0.52, 0.94))
                    short_threshold = float(np.clip(short_threshold - safety_shift, 0.06, 0.48))

        if ranker_mode:
            half_gap = float(
                np.clip(
                    ((max(0.0, long_threshold - 0.5) + max(0.0, 0.5 - short_threshold)) / 2.0),
                    max(0.03, min_gap / 2.0),
                    0.45,
                )
            )
            long_threshold = float(np.clip(0.5 + half_gap, 0.51, 0.95))
            short_threshold = float(np.clip(0.5 - half_gap, 0.05, 0.49))

        if long_threshold - short_threshold < min_gap:
            long_threshold = default_long
            short_threshold = default_short

        return long_threshold, short_threshold

    def _augment_params_for_train_labels(
        self, params: dict[str, Any], y_train: np.ndarray
    ) -> dict[str, Any]:
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
            if clipped_ratio >= 1.60:
                adjusted.setdefault("is_unbalance", True)
                adjusted.setdefault("scale_pos_weight", float(np.clip(clipped_ratio, 1.0, 4.0)))
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
        score_weights: list[float] = []
        X = self.features.values
        y = self.labels.values
        weights = self.sample_weights if self.sample_weights is not None else None
        regimes = self.regimes if self.regimes is not None and len(self.regimes) == len(y) else None
        row_symbols = (
            self.row_symbols
            if self.row_symbols is not None and len(self.row_symbols) == len(y)
            else None
        )

        for train_idx, test_idx in self._generate_cv_splits(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            if self._requires_binary_class_support() and not self._has_binary_class_support(
                y_train
            ):
                continue
            fold_params = self._prepare_params_for_train_size(params, len(train_idx))
            fold_params = self._augment_params_for_train_labels(fold_params, y_train)
            model = self._create_model(params=fold_params)
            w_train = weights[train_idx] if weights is not None else None
            train_ts = (
                np.asarray(self.timestamps[train_idx], dtype="datetime64[ns]")
                if self.timestamps is not None and len(self.timestamps) == len(y)
                else None
            )
            test_ts = (
                np.asarray(self.timestamps[test_idx], dtype="datetime64[ns]")
                if self.timestamps is not None and len(self.timestamps) == len(y)
                else None
            )
            self._fit_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_test,
                y_val=y_test,
                sample_weights=w_train,
                train_timestamps=train_ts,
                val_timestamps=test_ts,
            )

            y_pred = np.asarray(self._predict_with_model(model, X_test))
            y_proba_raw = np.asarray(
                self._get_raw_predictions_proba(model, X_test, timestamps=test_ts)
            )
            train_proba_raw = np.asarray(
                self._get_raw_predictions_proba(model, X_train, timestamps=train_ts)
            )
            if self._is_dead_lightgbm_model(
                model,
                probability_summary=self._summarize_probability_distribution(train_proba_raw),
            ) or self._is_dead_probability_summary(
                self._summarize_probability_distribution(y_proba_raw)
            ):
                continue

            y_proba = np.asarray(self._get_predictions_proba(model, X_test, timestamps=test_ts))
            train_proba = np.asarray(
                self._get_predictions_proba(model, X_train, timestamps=train_ts)
            )
            train_forward_returns = (
                np.asarray(self.primary_forward_returns[train_idx], dtype=float)
                if isinstance(self.primary_forward_returns, np.ndarray)
                and len(self.primary_forward_returns) == len(y)
                else None
            )
            long_threshold, short_threshold = self._derive_signal_thresholds(
                train_proba,
                train_labels=y_train,
                train_returns=train_forward_returns,
                train_regimes=(
                    np.asarray(regimes[train_idx], dtype=object) if regimes is not None else None
                ),
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
            fold_event_directions = (
                np.asarray(self.primary_event_directions[test_idx], dtype=float)
                if isinstance(self.primary_event_directions, np.ndarray)
                and len(self.primary_event_directions) == len(y)
                else None
            )
            fold_timestamps = (
                np.asarray(self.timestamps[test_idx], dtype="datetime64[ns]")
                if self.timestamps is not None and len(self.timestamps) == len(y)
                else None
            )
            fold_symbols = (
                np.asarray(row_symbols[test_idx], dtype=object) if row_symbols is not None else None
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
                event_directions=(
                    fold_event_directions[-eval_len:] if fold_event_directions is not None else None
                ),
                timestamps=(fold_timestamps[-eval_len:] if fold_timestamps is not None else None),
                symbols=(fold_symbols[-eval_len:] if fold_symbols is not None else None),
            )
            scores.append(float(metrics["risk_adjusted_score"]))
            score_weights.append(
                self._fold_reliability_weight(
                    metrics,
                    evaluation_size=eval_len,
                )
            )

        return self._aggregate_fold_objective(scores, score_weights) if scores else -1e9

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
        regimes = self.regimes if self.regimes is not None and len(self.regimes) == len(y) else None
        row_symbols = (
            self.row_symbols
            if self.row_symbols is not None and len(self.row_symbols) == len(y)
            else None
        )

        # Train with CV
        fold_results = []
        models = []
        self.cv_return_series = []
        self.cv_active_return_series = []
        self.oof_primary_proba = np.full(len(y), np.nan, dtype=float)
        self.oof_primary_proba_raw = np.full(len(y), np.nan, dtype=float)
        self.training_metrics["lightgbm_dead_cv_fold_count"] = 0.0
        splits = self._generate_cv_splits(X, y)

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            self.logger.info(f"Training fold {fold_idx + 1}/{len(splits)}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            if self._requires_binary_class_support() and not self._has_binary_class_support(
                y_train
            ):
                self.logger.warning(
                    f"Fold {fold_idx + 1}: training labels contain a single class, skipping fold"
                )
                continue
            w_train = weights[train_idx] if weights is not None else None

            # Create and train model
            fold_params = self._prepare_params_for_train_size(
                self.config.model_params, len(train_idx)
            )
            fold_params = self._augment_params_for_train_labels(fold_params, y_train)
            model = self._create_model(params=fold_params)
            train_ts = (
                np.asarray(self.timestamps[train_idx], dtype="datetime64[ns]")
                if self.timestamps is not None and len(self.timestamps) == len(y)
                else None
            )
            test_ts = (
                np.asarray(self.timestamps[test_idx], dtype="datetime64[ns]")
                if self.timestamps is not None and len(self.timestamps) == len(y)
                else None
            )

            self._fit_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_test,
                y_val=y_test,
                sample_weights=w_train,
                train_timestamps=train_ts,
                val_timestamps=test_ts,
            )

            # Evaluate
            y_pred = np.asarray(self._predict_with_model(model, X_test))
            y_proba_raw = np.asarray(
                self._get_raw_predictions_proba(model, X_test, timestamps=test_ts)
            )
            train_proba_raw = np.asarray(
                self._get_raw_predictions_proba(model, X_train, timestamps=train_ts)
            )
            y_proba = np.asarray(self._get_predictions_proba(model, X_test, timestamps=test_ts))
            dead_train_summary = self._summarize_probability_distribution(train_proba_raw)
            dead_eval_summary = self._summarize_probability_distribution(y_proba_raw)
            if self._is_dead_lightgbm_model(
                model,
                probability_summary=dead_train_summary,
            ) or self._is_dead_probability_summary(dead_eval_summary):
                self.training_metrics["lightgbm_dead_cv_fold_count"] = float(
                    self.training_metrics.get("lightgbm_dead_cv_fold_count", 0.0) + 1.0
                )
                self.logger.warning(
                    "Fold %d: rejecting dead LightGBM model (split_count=%s, raw_std=%.8f, raw_span=%.8f)",
                    fold_idx + 1,
                    int(self._summarize_lightgbm_model_structure(model).get("split_count", 0.0)),
                    float(dead_eval_summary.get("std", 0.0)),
                    float(dead_eval_summary.get("span", 0.0)),
                )
                continue
            train_proba = np.asarray(
                self._get_predictions_proba(model, X_train, timestamps=train_ts)
            )
            train_forward_returns = (
                np.asarray(self.primary_forward_returns[train_idx], dtype=float)
                if isinstance(self.primary_forward_returns, np.ndarray)
                and len(self.primary_forward_returns) == len(y)
                else None
            )
            long_threshold, short_threshold = self._derive_signal_thresholds(
                train_proba,
                train_labels=y_train,
                train_returns=train_forward_returns,
                train_regimes=(
                    np.asarray(regimes[train_idx], dtype=object) if regimes is not None else None
                ),
            )

            y_eval = np.asarray(y_test)
            eval_len = min(len(y_pred), len(y_eval), len(y_proba_raw), len(y_proba), len(test_idx))
            if eval_len != len(y_eval):
                if eval_len <= 1:
                    self.logger.warning(
                        f"Fold {fold_idx + 1}: insufficient aligned predictions, skipping fold"
                    )
                    continue
                y_eval = y_eval[-eval_len:]
                y_pred = y_pred[-eval_len:]
                y_proba_raw = y_proba_raw[-eval_len:]
                y_proba = y_proba[-eval_len:]
                self.logger.info(
                    f"Fold {fold_idx + 1}: aligned sequence outputs to {eval_len} samples"
                )
            eval_indices = np.asarray(test_idx[-eval_len:], dtype=int)
            self.oof_primary_proba_raw[eval_indices] = y_proba_raw
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
            fold_event_directions = (
                np.asarray(self.primary_event_directions[eval_indices], dtype=float)
                if isinstance(self.primary_event_directions, np.ndarray)
                and len(self.primary_event_directions) == len(y)
                else None
            )
            fold_timestamps = (
                np.asarray(self.timestamps[eval_indices], dtype="datetime64[ns]")
                if self.timestamps is not None and len(self.timestamps) == len(y)
                else None
            )
            fold_symbols = (
                np.asarray(row_symbols[eval_indices], dtype=object)
                if row_symbols is not None
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
                event_directions=fold_event_directions,
                timestamps=fold_timestamps,
                symbols=fold_symbols,
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
            fold_reliability = self._fold_reliability_weight(metrics, evaluation_size=eval_len)
            metrics["fold_reliability"] = float(fold_reliability)
            metrics["selection_score"] = float(
                float(metrics.get("risk_adjusted_score", -1e9)) * (0.20 + (0.80 * fold_reliability))
            )
            net_returns, execution_details = self._compute_net_returns(
                y_eval,
                y_proba,
                long_threshold=long_threshold,
                short_threshold=short_threshold,
                realized_forward_returns=fold_forward_returns,
                event_net_returns=fold_event_returns,
                event_directions=fold_event_directions,
                timestamps=fold_timestamps,
                symbols=fold_symbols,
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
                f"RiskScore={metrics.get('risk_adjusted_score', 0):.4f}, "
                f"Reliability={metrics.get('fold_reliability', 0):.3f}"
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
        self._fit_probability_calibrator_from_oof()
        self._fit_final_model_full_data()
        self._evaluate_holdout_performance()

    def _effective_cv_prediction_horizon(self, n_samples: int | None = None) -> int:
        """Return the capped prediction horizon actually used by CV splitters."""
        prediction_horizon = self._prediction_horizon()
        requested_splits = max(1, int(self.config.n_splits))
        if n_samples is not None:
            horizon_cap = max(1, n_samples // max(4, requested_splits + 1))
            prediction_horizon = min(prediction_horizon, horizon_cap)
        return prediction_horizon

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

            prediction_horizon = self._effective_cv_prediction_horizon(n_samples)
            requested_splits = max(1, int(self.config.n_splits))
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
                    max(2, min(50, max(2, n_samples // 3))) if n_samples is not None else None
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

        elif self.config.model_type == "xgboost_regressor":
            from xgboost import XGBRegressor

            if self.config.use_gpu:
                params.setdefault("tree_method", "hist")
                params.setdefault("device", "cuda")
            params.setdefault("eval_metric", "rmse")
            return XGBRegressor(**params)

        elif self.config.model_type == "lightgbm":
            from lightgbm import LGBMClassifier

            params.setdefault("verbose", -1)
            if self.config.use_gpu:
                params.setdefault("device", "cuda")
            return LGBMClassifier(**params)

        elif self.config.model_type == "lightgbm_ranker":
            from lightgbm import LGBMRanker

            raw_eval_at = params.pop("eval_at", None)
            params.setdefault("verbose", -1)
            params.setdefault("objective", "lambdarank")
            params.setdefault("metric", "ndcg")
            if self.config.use_gpu:
                params.setdefault("device", "cuda")
            model = LGBMRanker(**params)
            if raw_eval_at is not None:
                if isinstance(raw_eval_at, (list, tuple, set, np.ndarray)):
                    resolved_eval_at = [int(value) for value in raw_eval_at]
                else:
                    resolved_eval_at = [int(raw_eval_at)]
                setattr(
                    model,
                    "_alphatrade_eval_at",
                    [value for value in resolved_eval_at if value > 0],
                )
            return model

        elif self.config.model_type == "lightgbm_regressor":
            from lightgbm import LGBMRegressor

            params.setdefault("verbose", -1)
            if self.config.use_gpu:
                params.setdefault("device", "cuda")
            return LGBMRegressor(**params)

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

            elif self.config.model_type == "xgboost_regressor":
                from quant_trading_system.models.classical_ml import XGBoostModel
                from quant_trading_system.models.base import ModelType

                return XGBoostModel(
                    model_type=ModelType.REGRESSOR,
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

            elif self.config.model_type == "lightgbm_regressor":
                from quant_trading_system.models.classical_ml import LightGBMModel
                from quant_trading_system.models.base import ModelType

                return LightGBMModel(
                    model_type=ModelType.REGRESSOR,
                    use_gpu=self.config.use_gpu,
                    **model_params,
                )

            elif self.config.model_type == "lightgbm_ranker":
                from lightgbm import LGBMRanker

                model_params.setdefault("verbose", -1)
                model_params.setdefault("objective", "lambdarank")
                model_params.setdefault("metric", "ndcg")
                if self.config.use_gpu:
                    model_params.setdefault("device", "cuda")
                return LGBMRanker(**model_params)

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
        train_timestamps: np.ndarray | None = None,
        val_timestamps: np.ndarray | None = None,
        warm_start_model: Any | None = None,
    ) -> None:
        """Fit model with best-effort support for sample weights and validation sets."""
        if self.config.model_type in ["lstm", "transformer", "tcn"]:
            self._train_deep_learning_model(model, X_train, y_train, X_val, y_val)
            return

        if self._is_ranker_model():
            self._fit_ranker_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                sample_weights=sample_weights,
                train_timestamps=train_timestamps,
                val_timestamps=val_timestamps,
            )
            return

        model_module = str(getattr(model.__class__, "__module__", ""))
        fit_feature_names: list[str] = []
        if model_module.startswith("quant_trading_system.models"):
            if self.feature_names and len(self.feature_names) == int(X_train.shape[1]):
                fit_feature_names = [str(name) for name in self.feature_names]
            else:
                fit_feature_names = [f"f{i}" for i in range(int(X_train.shape[1]))]

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
        if warm_start_model is not None:
            warm_variants: list[tuple[dict[str, Any], ...]] = []
            for (kwargs,) in candidate_calls:
                warm_kwargs = dict(kwargs)
                warm_kwargs["warm_start_model"] = warm_start_model
                warm_variants.append((warm_kwargs,))
                if self.config.model_type in {"xgboost", "xgboost_regressor"}:
                    xgb_kwargs = dict(kwargs)
                    xgb_kwargs["xgb_model"] = warm_start_model
                    warm_variants.append((xgb_kwargs,))
                elif self._is_lightgbm_family_model():
                    init_kwargs = dict(kwargs)
                    init_kwargs["init_model"] = warm_start_model
                    warm_variants.append((init_kwargs,))
            candidate_calls = warm_variants + candidate_calls

        last_error: Exception | None = None
        for (kwargs,) in candidate_calls:
            call_kwargs = dict(kwargs)
            if fit_feature_names:
                call_kwargs.setdefault("feature_names", fit_feature_names)
            try:
                model.fit(X_train, y_train, **call_kwargs)
                if warm_start_model is not None and any(
                    key in call_kwargs for key in ("warm_start_model", "xgb_model", "init_model")
                ):
                    self.training_metrics["warm_start_applied"] = True
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

    def _fit_ranker_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray | None = None,
        train_timestamps: np.ndarray | None = None,
        val_timestamps: np.ndarray | None = None,
    ) -> None:
        """Fit learning-to-rank models with timestamp query grouping."""
        if train_timestamps is None:
            raise ValueError("Ranker training requires train_timestamps for query grouping.")

        fit_feature_names = (
            [str(name) for name in self.feature_names]
            if self.feature_names and len(self.feature_names) == int(X_train.shape[1])
            else []
        )
        X_train_fit: Any = (
            pd.DataFrame(X_train, columns=fit_feature_names) if fit_feature_names else X_train
        )
        X_val_fit: Any = (
            pd.DataFrame(X_val, columns=fit_feature_names)
            if fit_feature_names and X_val is not None
            else X_val
        )

        train_group = self._build_ranking_groups(train_timestamps)
        if train_group.size == 0:
            train_group = np.array([int(len(X_train_fit))], dtype=int)
        if int(np.sum(train_group)) != int(len(X_train_fit)):
            train_group = np.array([int(len(X_train_fit))], dtype=int)

        use_eval = (
            X_val is not None
            and y_val is not None
            and len(X_val) > 0
            and val_timestamps is not None
            and len(val_timestamps) == len(X_val)
        )
        eval_group = (
            self._build_ranking_groups(val_timestamps) if use_eval else np.array([], dtype=int)
        )
        if use_eval and eval_group.size == 0:
            use_eval = False

        eval_at = self._ranker_eval_at(model)
        fit_eval_at = eval_at if eval_at else [1, 3, 5, 10]
        include_eval_at_kwarg = not bool(eval_at)

        candidate_calls: list[dict[str, Any]] = []
        if use_eval and sample_weights is not None:
            call_kwargs = {
                "group": train_group,
                "sample_weight": sample_weights,
                "eval_set": [(X_val_fit, y_val)],
                "eval_group": [eval_group],
            }
            if include_eval_at_kwarg:
                call_kwargs["eval_at"] = fit_eval_at
            candidate_calls.append(call_kwargs)
        if sample_weights is not None:
            candidate_calls.append({"group": train_group, "sample_weight": sample_weights})
        if use_eval:
            call_kwargs = {
                "group": train_group,
                "eval_set": [(X_val_fit, y_val)],
                "eval_group": [eval_group],
            }
            if include_eval_at_kwarg:
                call_kwargs["eval_at"] = fit_eval_at
            candidate_calls.append(call_kwargs)
        candidate_calls.append({"group": train_group})

        last_error: Exception | None = None
        for kwargs in candidate_calls:
            try:
                model.fit(X_train_fit, y_train, **kwargs)
                self._attach_ranker_runtime_metadata(model)
                return
            except (TypeError, ValueError) as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise last_error
        model.fit(X_train_fit, y_train, group=train_group)
        self._attach_ranker_runtime_metadata(model)

    def _load_warm_start_model(self) -> Any | None:
        """Load optional prior model artifact for final production refit only."""
        warm_start_path = str(getattr(self.config, "warm_start_model_path", "") or "").strip()
        if not warm_start_path:
            return None
        if self._warm_start_model_cache is not None:
            return self._warm_start_model_cache

        path = Path(warm_start_path)
        if not path.exists():
            raise FileNotFoundError(f"Warm-start model not found: {path}")
        with path.open("rb") as handle:
            self._warm_start_model_cache = pickle.load(handle)
        self.training_metrics["warm_start_model_path"] = str(path)
        return self._warm_start_model_cache

    def _train_deep_learning_model(self, model, X_train, y_train, X_val, y_val) -> None:
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

    def _supports_probability_calibration(self) -> bool:
        """Return whether this model family supports probability calibration."""
        if not bool(self.config.enable_probability_calibration):
            return False
        if self._is_ranker_model():
            return bool(self.config.enable_ranker_probability_calibration)
        if self.config.model_type in {"xgboost_regressor", "lightgbm_regressor"}:
            return False
        return True

    @staticmethod
    def _resolve_probability_calibration_target(model: Any) -> Any:
        """Return the object on which calibration state should be persisted."""
        wrapped_model = getattr(model, "_model", None)
        return wrapped_model if wrapped_model is not None else model

    def _resolve_probability_calibration_payload(self, model: Any) -> dict[str, Any] | None:
        """Read attached probability calibration metadata from wrapper or raw model."""
        for candidate in (model, getattr(model, "_model", None)):
            payload = getattr(candidate, "_alphatrade_probability_calibration", None)
            if isinstance(payload, dict) and payload.get("calibrator") is not None:
                return payload
        return None

    def _apply_probability_calibration(
        self,
        values: np.ndarray,
        payload: dict[str, Any],
    ) -> np.ndarray:
        """Apply attached post-hoc probability calibration to raw prediction scores."""
        calibrator = payload.get("calibrator")
        method = str(payload.get("method") or "isotonic").strip().lower()
        x = np.asarray(values, dtype=float).reshape(-1)
        if x.size == 0 or calibrator is None:
            return x
        if method == "isotonic":
            calibrated = calibrator.predict(x)
        else:
            calibrated = calibrator.predict_proba(x.reshape(-1, 1))[:, 1]
        return np.clip(np.asarray(calibrated, dtype=float).reshape(-1), 0.0, 1.0)

    def _attach_probability_calibrator_to_model(self, model: Any) -> None:
        """Attach fitted probability calibration state to the saved model artifact."""
        if model is None:
            return
        calibration_target = self._resolve_probability_calibration_target(model)
        if self.probability_calibrator is None:
            if hasattr(calibration_target, "_alphatrade_probability_calibration"):
                delattr(calibration_target, "_alphatrade_probability_calibration")
            return
        calibration_target._alphatrade_probability_calibration = {
            "method": str(
                self.probability_calibration_method_resolved
                or self.config.probability_calibration_method
            ),
            "calibrator": self.probability_calibrator,
        }

    def _evaluate_probability_calibration_guardrails(
        self,
        *,
        raw_values: np.ndarray,
        calibrated_values: np.ndarray,
        labels: np.ndarray,
        train_returns: np.ndarray | None = None,
        train_regimes: np.ndarray | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Reject post-hoc calibration when it collapses OOF dispersion or activity."""
        raw_arr = np.asarray(raw_values, dtype=float).reshape(-1)
        calibrated_arr = np.asarray(calibrated_values, dtype=float).reshape(-1)
        label_arr = np.asarray(labels, dtype=float).reshape(-1)
        summary: dict[str, Any] = {
            "enabled": bool(self.config.probability_calibration_guardrail_enabled),
            "accepted": True,
            "reason": "guardrail_disabled",
            "activity_collapse": False,
            "dispersion_collapse": False,
            "rejection_reasons": [],
        }
        if not bool(self.config.probability_calibration_guardrail_enabled):
            return True, summary
        if (
            raw_arr.size <= 0
            or calibrated_arr.size != raw_arr.size
            or label_arr.size != raw_arr.size
        ):
            summary["reason"] = "guardrail_unavailable"
            return True, summary

        signed_returns = (
            np.asarray(train_returns, dtype=float).reshape(-1)
            if train_returns is not None
            and len(np.asarray(train_returns).reshape(-1)) == raw_arr.size
            else None
        )
        aligned_regimes = (
            np.asarray(train_regimes, dtype=object).reshape(-1)
            if train_regimes is not None
            and len(np.asarray(train_regimes, dtype=object).reshape(-1)) == raw_arr.size
            else None
        )
        raw_summary = self._summarize_probability_distribution(raw_arr)
        calibrated_summary = self._summarize_probability_distribution(calibrated_arr)
        shift_summary = self._summarize_probability_shift(raw_arr, calibrated_arr)
        raw_long, raw_short = self._derive_signal_thresholds(
            raw_arr,
            train_labels=label_arr,
            train_returns=signed_returns,
            train_regimes=aligned_regimes,
        )
        calibrated_long, calibrated_short = self._derive_signal_thresholds(
            calibrated_arr,
            train_labels=label_arr,
            train_returns=signed_returns,
            train_regimes=aligned_regimes,
        )
        raw_hits = self._summarize_threshold_hits(raw_arr, raw_long, raw_short)
        calibrated_hits = self._summarize_threshold_hits(
            calibrated_arr,
            calibrated_long,
            calibrated_short,
        )
        raw_active_rate = float(raw_hits.get("active_rate", 0.0))
        calibrated_active_rate = float(calibrated_hits.get("active_rate", 0.0))
        min_active_rate = float(self.config.probability_calibration_min_oof_active_rate)
        min_active_rate_ratio = float(self.config.probability_calibration_min_oof_active_rate_ratio)
        min_dispersion_ratio = float(self.config.probability_calibration_min_dispersion_ratio)
        near_mid_limit = float(self.config.probability_calibration_max_near_mid_rate)
        guardable_raw_active_rate = max(0.02, min_active_rate)
        min_allowed_active_rate = max(min_active_rate, raw_active_rate * min_active_rate_ratio)
        raw_std = float(raw_summary.get("std", 0.0))
        raw_span = float(raw_summary.get("span", 0.0))
        std_ratio = float(shift_summary.get("std_ratio", 1.0))
        span_ratio = float(shift_summary.get("span_ratio", 1.0))
        activity_collapse = bool(
            raw_active_rate >= guardable_raw_active_rate
            and calibrated_active_rate + 1e-9 < min_allowed_active_rate
        )
        dispersion_collapse = bool(
            (raw_std > 1e-6 or raw_span > 1e-5)
            and std_ratio + 1e-12 < min_dispersion_ratio
            and span_ratio + 1e-12 < min_dispersion_ratio
            and float(calibrated_summary.get("near_mid_rate", 0.0)) >= near_mid_limit
        )
        rejection_reasons: list[str] = []
        if activity_collapse:
            rejection_reasons.append("activity_collapse")
        if dispersion_collapse:
            rejection_reasons.append("dispersion_collapse")
        accepted = not rejection_reasons
        summary.update(
            {
                "accepted": bool(accepted),
                "reason": "passed" if accepted else ",".join(rejection_reasons),
                "activity_collapse": bool(activity_collapse),
                "dispersion_collapse": bool(dispersion_collapse),
                "rejection_reasons": list(rejection_reasons),
                "raw_active_rate": raw_active_rate,
                "calibrated_active_rate": calibrated_active_rate,
                "min_allowed_active_rate": float(min_allowed_active_rate),
                "active_rate_ratio": float(calibrated_active_rate / max(raw_active_rate, 1e-9)),
                "raw_std": raw_std,
                "raw_span": raw_span,
                "calibrated_std": float(calibrated_summary.get("std", 0.0)),
                "calibrated_span": float(calibrated_summary.get("span", 0.0)),
                "std_ratio": std_ratio,
                "span_ratio": span_ratio,
                "raw_near_mid_rate": float(raw_summary.get("near_mid_rate", 0.0)),
                "calibrated_near_mid_rate": float(calibrated_summary.get("near_mid_rate", 0.0)),
                "raw_tail_rate": float(raw_summary.get("tail_rate", 0.0)),
                "calibrated_tail_rate": float(calibrated_summary.get("tail_rate", 0.0)),
                "raw_long_threshold": float(raw_long),
                "raw_short_threshold": float(raw_short),
                "calibrated_long_threshold": float(calibrated_long),
                "calibrated_short_threshold": float(calibrated_short),
            }
        )
        self._record_summary_metrics("probability_calibration_guardrails", summary)
        self.training_metrics["probability_calibration_guardrail_raw_threshold_hits"] = (
            self._json_safe(raw_hits)
        )
        self.training_metrics["probability_calibration_guardrail_calibrated_threshold_hits"] = (
            self._json_safe(calibrated_hits)
        )
        return accepted, summary

    def _fit_probability_calibrator_from_oof(self) -> None:
        """Fit a post-hoc probability calibrator from out-of-fold predictions only."""
        self.probability_calibrator = None
        self.probability_calibration_method_resolved = None
        self.training_metrics["probability_calibration_enabled"] = 0.0
        if not self._supports_probability_calibration():
            self.training_metrics["probability_calibration_reason"] = "disabled_or_unsupported"
            self._refresh_oof_probability_diagnostics()
            return
        raw_source = self.oof_primary_proba_raw
        if raw_source is None:
            raw_source = self.oof_primary_proba
        self._refresh_oof_probability_diagnostics()
        if raw_source is None or self.labels is None:
            self.training_metrics["probability_calibration_reason"] = "missing_oof_predictions"
            return

        raw_values = np.asarray(raw_source, dtype=float).reshape(-1)
        labels = pd.to_numeric(self.labels, errors="coerce").to_numpy(dtype=float).reshape(-1)
        if raw_values.size != labels.size or raw_values.size == 0:
            self.training_metrics["probability_calibration_reason"] = "shape_mismatch"
            return

        mask = np.isfinite(raw_values) & np.isfinite(labels)
        if not np.any(mask):
            self.training_metrics["probability_calibration_reason"] = "no_finite_rows"
            return

        X_cal = np.clip(raw_values[mask], 0.0, 1.0)
        y_cal = labels[mask]
        unique_labels = sorted({int(v) for v in np.unique(y_cal) if v in {0.0, 1.0}})
        if len(unique_labels) < 2:
            self.training_metrics["probability_calibration_reason"] = "single_class_oof"
            return
        if len(X_cal) < 60:
            self.training_metrics["probability_calibration_reason"] = "insufficient_samples"
            return

        sample_weight = None
        if self.sample_weights is not None and len(self.sample_weights) == len(raw_values):
            sample_weight = np.asarray(self.sample_weights, dtype=float)[mask]
            sample_weight = np.clip(sample_weight, 1e-6, None)

        requested_method = str(self.config.probability_calibration_method)
        resolved_method = requested_method
        if requested_method == "isotonic" and len(X_cal) < 200:
            resolved_method = "sigmoid"

        if resolved_method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(X_cal, y_cal, sample_weight=sample_weight)
            calibrated = calibrator.predict(X_cal)
        else:
            from sklearn.linear_model import LogisticRegression

            calibrator = LogisticRegression()
            calibrator.fit(X_cal.reshape(-1, 1), y_cal.astype(int), sample_weight=sample_weight)
            calibrated = calibrator.predict_proba(X_cal.reshape(-1, 1))[:, 1]

        calibrated = np.clip(np.asarray(calibrated, dtype=float).reshape(-1), 0.0, 1.0)
        brier_before = float(np.mean((X_cal - y_cal) ** 2))
        brier_after = float(np.mean((calibrated - y_cal) ** 2))
        if brier_after > (brier_before + 1e-4):
            self.training_metrics["probability_calibration_reason"] = "brier_not_improved"
            self.training_metrics["probability_calibration_brier_before"] = brier_before
            self.training_metrics["probability_calibration_brier_after"] = brier_after
            return

        signed_returns = self._resolve_signed_realized_returns(
            forward_returns=self.primary_forward_returns,
            event_net_returns=self.cost_aware_event_returns,
            event_directions=self.primary_event_directions,
        )
        masked_signed_returns = None
        if signed_returns is not None and len(signed_returns) == len(raw_values):
            masked_signed_returns = np.asarray(signed_returns, dtype=float)[mask]
        masked_regimes = None
        if self.regimes is not None and len(self.regimes) == len(raw_values):
            masked_regimes = np.asarray(self.regimes, dtype=object)[mask]
        guardrail_passed, guardrail_summary = self._evaluate_probability_calibration_guardrails(
            raw_values=X_cal,
            calibrated_values=calibrated,
            labels=y_cal,
            train_returns=masked_signed_returns,
            train_regimes=masked_regimes,
        )
        self.training_metrics["probability_calibration_guardrails"] = self._json_safe(
            guardrail_summary
        )
        if not guardrail_passed:
            self.training_metrics["probability_calibration_reason"] = "guardrail_" + str(
                guardrail_summary.get("reason", "rejected")
            )
            self.training_metrics["probability_calibration_brier_before"] = brier_before
            self.training_metrics["probability_calibration_brier_after"] = brier_after
            self.training_metrics["probability_calibration_brier_improvement"] = float(
                brier_before - brier_after
            )
            self._refresh_oof_probability_diagnostics()
            self.logger.warning(
                "Probability calibration rejected by OOF guardrail (%s); keeping raw scores.",
                guardrail_summary.get("reason", "rejected"),
            )
            return

        self.probability_calibrator = calibrator
        self.probability_calibration_method_resolved = resolved_method
        self.oof_primary_proba[mask] = calibrated
        self.training_metrics["probability_calibration_enabled"] = 1.0
        self.training_metrics["probability_calibration_method"] = resolved_method
        self.training_metrics["probability_calibration_requested_method"] = requested_method
        self.training_metrics["probability_calibration_sample_count"] = float(len(X_cal))
        self.training_metrics["probability_calibration_brier_before"] = brier_before
        self.training_metrics["probability_calibration_brier_after"] = brier_after
        self.training_metrics["probability_calibration_brier_improvement"] = float(
            brier_before - brier_after
        )
        self.training_metrics["probability_calibration_reason"] = "applied"
        self._refresh_oof_probability_diagnostics()
        self.logger.info(
            "Probability calibration applied using %s on %d OOF rows " "(Brier %.6f -> %.6f)",
            resolved_method,
            len(X_cal),
            brier_before,
            brier_after,
        )

    @staticmethod
    def _resolve_signed_realized_returns(
        *,
        forward_returns: np.ndarray | pd.Series | None,
        event_net_returns: np.ndarray | pd.Series | None,
        event_directions: np.ndarray | pd.Series | None,
    ) -> np.ndarray | None:
        """Resolve realized returns signed relative to the long direction."""
        if event_net_returns is not None:
            signed = np.asarray(event_net_returns, dtype=float).reshape(-1)
            if event_directions is not None:
                directions = np.asarray(event_directions, dtype=float).reshape(-1)
                target_len = min(len(signed), len(directions))
                if target_len > 0:
                    signed = signed[-target_len:]
                    directions = directions[-target_len:]
                    signed = signed * np.sign(directions)
                    return np.nan_to_num(signed, nan=0.0, posinf=0.0, neginf=0.0)
            return np.nan_to_num(signed, nan=0.0, posinf=0.0, neginf=0.0)
        if forward_returns is None:
            return None
        return np.nan_to_num(
            np.asarray(forward_returns, dtype=float).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    @staticmethod
    def _resolve_trade_realized_returns(
        signed_realized_returns: np.ndarray | None,
        signal_values: np.ndarray | None,
    ) -> np.ndarray | None:
        """Project signed realized returns into the model's chosen trade direction."""
        if signed_realized_returns is None or signal_values is None:
            return None
        signed = np.asarray(signed_realized_returns, dtype=float).reshape(-1)
        signal = np.asarray(signal_values, dtype=float).reshape(-1)
        target_len = min(len(signed), len(signal))
        if target_len <= 0:
            return None
        trade_returns = signed[-target_len:] * np.sign(signal[-target_len:])
        return np.nan_to_num(trade_returns, nan=0.0, posinf=0.0, neginf=0.0)

    def _resolve_expected_edge_thresholds(self) -> tuple[float, float]:
        """Resolve the admission thresholds used by the edge policy layer."""
        long_threshold = float(self.training_metrics.get("holdout_long_threshold", np.nan))
        short_threshold = float(self.training_metrics.get("holdout_short_threshold", np.nan))
        if (
            np.isfinite(long_threshold)
            and np.isfinite(short_threshold)
            and long_threshold > short_threshold
        ):
            return long_threshold, short_threshold
        if self.oof_primary_proba is not None and self.labels is not None:
            derived_long, derived_short = self._derive_signal_thresholds(
                np.asarray(self.oof_primary_proba, dtype=float),
                train_labels=np.asarray(self.labels, dtype=float),
                train_returns=self.primary_forward_returns,
                train_regimes=self.regimes,
            )
            if derived_long > derived_short:
                return float(derived_long), float(derived_short)
        return 0.55, 0.45

    def _apply_meta_signal_filter(
        self,
        *,
        feature_frame: pd.DataFrame | None,
        signal_values: np.ndarray,
        close_prices: pd.Series | np.ndarray | None,
        confidence: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Apply the fitted meta-labeler to directional signals when runtime inputs exist."""
        signal = np.asarray(signal_values, dtype=float).reshape(-1)
        resolved_confidence = np.clip(
            np.asarray(
                confidence if confidence is not None else (np.abs(signal) > 1e-8).astype(float),
                dtype=float,
            ).reshape(-1),
            0.0,
            1.0,
        )
        candidate_mask = np.abs(signal) > 1e-8
        summary: dict[str, Any] = {
            "applied": False,
            "candidate_count": int(np.count_nonzero(candidate_mask)),
            "retained_count": int(np.count_nonzero(candidate_mask)),
            "candidate_rate": float(np.mean(candidate_mask)) if signal.size else 0.0,
            "retained_rate": float(np.mean(candidate_mask)) if signal.size else 0.0,
            "retention_ratio": 1.0 if np.any(candidate_mask) else 0.0,
            "confidence_mean": (
                float(np.mean(resolved_confidence[candidate_mask]))
                if np.any(candidate_mask)
                else 0.0
            ),
        }
        if (
            (not self.config.use_meta_labeling)
            or self.meta_model is None
            or feature_frame is None
            or close_prices is None
            or (not hasattr(self.meta_model, "filter_signals"))
        ):
            return signal, resolved_confidence, summary

        feature_df = feature_frame.reset_index(drop=True).copy()
        close_series = pd.Series(close_prices).reset_index(drop=True)
        target_len = min(len(feature_df), len(close_series), len(signal), len(resolved_confidence))
        if target_len <= 0:
            return signal, resolved_confidence, summary

        signal = signal[-target_len:]
        resolved_confidence = resolved_confidence[-target_len:]
        feature_df = feature_df.iloc[-target_len:].reset_index(drop=True)
        close_series = close_series.iloc[-target_len:].reset_index(drop=True)
        primary_direction = pd.Series(np.sign(signal), index=feature_df.index, dtype=float)
        meta_threshold = float(
            np.clip(
                self.training_metrics.get(
                    "meta_label_min_confidence",
                    self.config.meta_label_min_confidence,
                ),
                0.45,
                0.95,
            )
        )
        try:
            filtered = self.meta_model.filter_signals(
                feature_df.fillna(0.0),
                primary_direction,
                prices=close_series,
                threshold=meta_threshold,
            )
        except Exception as exc:
            summary["reason"] = f"meta_filter_failed:{exc}"
            self.logger.warning("Meta filter skipped during training evaluation: %s", exc)
            return signal, resolved_confidence, summary

        filtered_signal = (
            pd.to_numeric(
                filtered.get("filtered_signal", primary_direction),
                errors="coerce",
            )
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        meta_confidence = pd.to_numeric(
            filtered.get("confidence", np.nan),
            errors="coerce",
        ).to_numpy(dtype=float)
        filtered_values = np.where(
            filtered_signal > 0.0,
            np.maximum(signal, 0.0),
            np.where(filtered_signal < 0.0, np.minimum(signal, 0.0), 0.0),
        )
        resolved_confidence = np.clip(
            np.where(np.isfinite(meta_confidence), meta_confidence, resolved_confidence),
            0.0,
            1.0,
        )
        retained_mask = np.abs(filtered_values) > 1e-8
        retained_count = int(np.count_nonzero(retained_mask))
        candidate_count = int(np.count_nonzero(candidate_mask[-target_len:]))
        summary.update(
            {
                "applied": True,
                "candidate_count": candidate_count,
                "retained_count": retained_count,
                "candidate_rate": float(candidate_count / target_len) if target_len else 0.0,
                "retained_rate": float(retained_count / target_len) if target_len else 0.0,
                "retention_ratio": (
                    float(retained_count / candidate_count) if candidate_count > 0 else 0.0
                ),
                "confidence_mean": (
                    float(np.mean(resolved_confidence[retained_mask]))
                    if retained_count > 0
                    else 0.0
                ),
                "threshold": meta_threshold,
            }
        )
        return filtered_values, resolved_confidence, summary

    @staticmethod
    def _align_policy_stream_inputs(
        *,
        signal_values: np.ndarray,
        trade_returns: np.ndarray,
        timestamps: np.ndarray | None = None,
        policy_frame: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, pd.DataFrame | None, int]:
        """Align policy-stream arrays to a common trailing window."""
        signal = np.asarray(signal_values, dtype=float).reshape(-1)
        realized = np.asarray(trade_returns, dtype=float).reshape(-1)
        target_len = min(len(signal), len(realized))
        if policy_frame is not None:
            target_len = min(target_len, len(policy_frame))
        if timestamps is not None:
            target_len = min(target_len, len(np.asarray(timestamps).reshape(-1)))
        if target_len <= 0:
            return (
                np.array([], dtype=float),
                np.array([], dtype=float),
                None,
                None,
                0,
            )

        signal = signal[-target_len:]
        realized = np.nan_to_num(realized[-target_len:], nan=0.0, posinf=0.0, neginf=0.0)
        policy = (
            policy_frame.iloc[-target_len:].reset_index(drop=True)
            if policy_frame is not None
            else None
        )
        aligned_timestamps = None
        if timestamps is not None:
            aligned_timestamps = np.asarray(timestamps).reshape(-1)[-target_len:]
        return signal, realized, aligned_timestamps, policy, target_len

    @staticmethod
    def _policy_adjusted_signal_values(
        *,
        signal: np.ndarray,
        policy: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Apply expected-edge admission and sizing to a base signal vector."""
        adjusted_signal = signal.copy()
        if policy is not None and not policy.empty:
            policy_pass = (
                policy["edge_policy_pass"].to_numpy(dtype=bool)
                if "edge_policy_pass" in policy.columns
                else (np.abs(signal) > 1e-8)
            )
            policy_scale = (
                np.clip(policy["edge_policy_scale"].to_numpy(dtype=float), 0.0, 1.5)
                if "edge_policy_scale" in policy.columns
                else np.where(policy_pass, 1.0, 0.0).astype(float)
            )
            adjusted_signal = np.where(
                policy_pass,
                np.sign(signal) * np.abs(signal) * policy_scale,
                0.0,
            )
        return adjusted_signal

    def _build_policy_adjusted_stream(
        self,
        *,
        signal_values: np.ndarray,
        trade_returns: np.ndarray,
        timestamps: np.ndarray | None = None,
        policy_frame: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Build row-level and timestamp-aggregated execution series for a policy-adjusted stream."""
        signal, realized, aligned_timestamps, policy, target_len = self._align_policy_stream_inputs(
            signal_values=signal_values,
            trade_returns=trade_returns,
            timestamps=timestamps,
            policy_frame=policy_frame,
        )
        if target_len <= 0:
            return {
                "target_len": 0,
                "base_signal": np.array([], dtype=float),
                "adjusted_signal": np.array([], dtype=float),
                "row_returns": np.array([], dtype=float),
                "turnover_series": np.array([], dtype=float),
                "trade_mask": np.array([], dtype=bool),
                "entry_mask": np.array([], dtype=bool),
                "portfolio_returns": np.array([], dtype=float),
                "portfolio_turnover_series": np.array([], dtype=float),
                "portfolio_trade_mask": np.array([], dtype=bool),
                "portfolio_entry_mask": np.array([], dtype=bool),
                "timestamps": None,
            }

        adjusted_signal = self._policy_adjusted_signal_values(signal=signal, policy=policy)
        trade_mask = np.abs(adjusted_signal) > 1e-8
        prev_signal = np.concatenate([[0.0], adjusted_signal[:-1]])
        entry_mask = trade_mask & (
            (np.abs(prev_signal) <= 1e-8) | (np.sign(adjusted_signal) != np.sign(prev_signal))
        )
        turnover_series = np.abs(np.diff(np.concatenate([[0.0], adjusted_signal])))
        row_returns = np.where(trade_mask, np.abs(adjusted_signal) * realized, 0.0)
        portfolio = self._aggregate_panel_portfolio_series(
            timestamps=aligned_timestamps,
            net_returns=row_returns,
            turnover_series=turnover_series,
            trade_mask=trade_mask,
            entry_mask=entry_mask,
        )
        return {
            "target_len": int(target_len),
            "base_signal": np.asarray(signal, dtype=float),
            "adjusted_signal": np.asarray(adjusted_signal, dtype=float),
            "row_returns": np.asarray(row_returns, dtype=float),
            "turnover_series": np.asarray(turnover_series, dtype=float),
            "trade_mask": np.asarray(trade_mask, dtype=bool),
            "entry_mask": np.asarray(entry_mask, dtype=bool),
            "portfolio_returns": np.asarray(portfolio["portfolio_returns"], dtype=float),
            "portfolio_turnover_series": np.asarray(
                portfolio["portfolio_turnover_series"], dtype=float
            ),
            "portfolio_trade_mask": np.asarray(portfolio["portfolio_trade_mask"], dtype=bool),
            "portfolio_entry_mask": np.asarray(portfolio["portfolio_entry_mask"], dtype=bool),
            "timestamps": aligned_timestamps,
        }

    def _summarize_policy_stream_metrics(
        self,
        *,
        stream: dict[str, Any],
    ) -> dict[str, Any]:
        """Summarize portfolio-level diagnostics for a policy-adjusted stream."""
        target_len = int(stream.get("target_len", 0))
        if target_len <= 0:
            return {
                "policy_signal_count": 0,
                "policy_selected_count": 0,
                "policy_selected_rate": 0.0,
                "policy_trade_count": 0.0,
                "policy_active_signal_rate": 0.0,
                "policy_turnover": 0.0,
                "policy_sharpe": 0.0,
                "policy_max_drawdown": 0.0,
                "policy_trade_return_observations": 0.0,
                "policy_sharpe_observation_confidence": 0.0,
                "policy_annual_return": 0.0,
                "policy_calmar": 0.0,
                "policy_underwater_ratio": 0.0,
                "policy_selected_mean_trade_return": 0.0,
                "policy_selected_win_rate": 0.0,
                "policy_pnl": 0.0,
                "policy_loss_pnl": 0.0,
                "policy_tail_loss_pnl": 0.0,
                "policy_cvar_95": 0.0,
                "policy_expected_shortfall": 0.0,
                "policy_return_skew": 0.0,
                "policy_risk_adjusted_score": 0.0,
            }

        base_signal = np.asarray(stream.get("base_signal", []), dtype=float)
        adjusted_signal = np.asarray(stream.get("adjusted_signal", []), dtype=float)
        portfolio_returns = np.asarray(stream.get("portfolio_returns", []), dtype=float)
        portfolio_turnover = np.asarray(stream.get("portfolio_turnover_series", []), dtype=float)
        portfolio_trade_mask = np.asarray(stream.get("portfolio_trade_mask", []), dtype=bool)
        portfolio_entry_mask = np.asarray(stream.get("portfolio_entry_mask", []), dtype=bool)
        trade_return_series = portfolio_returns[portfolio_trade_mask]
        performance_returns = portfolio_returns if portfolio_returns.size > 0 else trade_return_series

        turnover = float(np.mean(portfolio_turnover)) if portfolio_turnover.size > 0 else 0.0
        trade_count = int(np.count_nonzero(portfolio_entry_mask))
        active_signal_rate = (
            float(np.mean(portfolio_trade_mask)) if portfolio_trade_mask.size > 0 else 0.0
        )
        std = float(np.std(performance_returns)) if performance_returns.size > 0 else 0.0
        annualization_periods = float(self._annualization_periods())
        sharpe = (
            float(np.mean(performance_returns) / std * np.sqrt(annualization_periods))
            if std > 1e-12
            else 0.0
        )
        max_dd = self._max_drawdown(performance_returns)
        trade_obs_count = int(len(trade_return_series))
        min_obs_for_confident_sharpe = max(
            6,
            int(round(float(self._effective_trade_target(int(target_len))) * 0.80)),
        )
        sharpe_observation_confidence = float(
            np.clip(
                float(trade_obs_count) / float(max(1, min_obs_for_confident_sharpe)),
                0.10,
                1.0,
            )
        )
        sharpe = float(sharpe * sharpe_observation_confidence)
        annual_return = (
            float(np.mean(performance_returns) * annualization_periods)
            if performance_returns.size > 0
            else 0.0
        )
        calmar = annual_return / max_dd if max_dd > 1e-9 else annual_return
        pnl = float(np.sum(performance_returns)) if performance_returns.size > 0 else 0.0
        loss_pnl = (
            float(np.sum(performance_returns[performance_returns < 0.0]))
            if performance_returns.size > 0
            else 0.0
        )
        if performance_returns.size > 0:
            cumulative_equity = np.cumsum(performance_returns)
            running_peak = np.maximum.accumulate(cumulative_equity)
            underwater_ratio = float(np.mean(cumulative_equity < (running_peak - 1e-12)))
            alpha = 0.05
            tail_cutoff = float(np.quantile(performance_returns, alpha))
            cvar = float(np.mean(performance_returns[performance_returns <= tail_cutoff]))
            tail_loss_cutoff = float(np.quantile(performance_returns, 0.10))
            tail_loss_pnl = float(
                np.sum(performance_returns[performance_returns <= tail_loss_cutoff])
            )
            expected_shortfall = float(abs(cvar))
            skew = float(pd.Series(performance_returns).skew())
        else:
            underwater_ratio = 0.0
            cvar = 0.0
            tail_loss_pnl = 0.0
            expected_shortfall = 0.0
            skew = 0.0

        objective_components = self._compute_objective_components(
            sharpe=sharpe,
            max_drawdown=max_dd,
            turnover=turnover,
            brier_score=0.0,
            trade_count=trade_count,
            cvar=cvar,
            skew=skew,
            expected_shortfall=expected_shortfall,
            symbol_concentration_hhi=0.0,
            equity_break=float(1.0 if max_dd > (self.config.max_drawdown * 1.5) else 0.0),
            evaluation_size=int(target_len),
            symbol_sharpe_p25=None,
        )

        return {
            "policy_signal_count": int(np.count_nonzero(np.abs(base_signal) > 1e-8)),
            "policy_selected_count": int(np.count_nonzero(np.abs(adjusted_signal) > 1e-8)),
            "policy_selected_rate": float(np.mean(np.abs(adjusted_signal) > 1e-8))
            if target_len
            else 0.0,
            "policy_trade_count": float(trade_count),
            "policy_active_signal_rate": float(active_signal_rate),
            "policy_turnover": float(turnover),
            "policy_sharpe": float(sharpe),
            "policy_max_drawdown": float(max_dd),
            "policy_trade_return_observations": float(trade_obs_count),
            "policy_sharpe_observation_confidence": float(sharpe_observation_confidence),
            "policy_annual_return": float(annual_return),
            "policy_calmar": float(calmar),
            "policy_underwater_ratio": float(underwater_ratio),
            "policy_selected_mean_trade_return": (
                float(np.mean(trade_return_series)) if trade_return_series.size else 0.0
            ),
            "policy_selected_win_rate": (
                float(np.mean(trade_return_series > 0.0)) if trade_return_series.size else 0.0
            ),
            "policy_pnl": float(pnl),
            "policy_loss_pnl": float(loss_pnl),
            "policy_tail_loss_pnl": float(tail_loss_pnl),
            "policy_cvar_95": float(cvar),
            "policy_expected_shortfall": float(expected_shortfall),
            "policy_return_skew": float(skew),
            "policy_risk_adjusted_score": float(objective_components["risk_adjusted_score"]),
        }

    def _summarize_policy_adjusted_signal_stream(
        self,
        *,
        signal_values: np.ndarray,
        trade_returns: np.ndarray,
        timestamps: np.ndarray | None = None,
        policy_frame: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Summarize the realized holdout stream after expected-edge admission and sizing."""
        stream = self._build_policy_adjusted_stream(
            signal_values=signal_values,
            trade_returns=trade_returns,
            timestamps=timestamps,
            policy_frame=policy_frame,
        )
        return self._summarize_policy_stream_metrics(stream=stream)

    def _summarize_policy_adjusted_metrics_by_group(
        self,
        *,
        signal_values: np.ndarray,
        trade_returns: np.ndarray,
        group_labels: np.ndarray | None,
        timestamps: np.ndarray | None = None,
        policy_frame: pd.DataFrame | None = None,
        min_rows: int = 1,
    ) -> dict[str, dict[str, float]]:
        """Summarize policy-adjusted execution diagnostics for each group."""
        signal, realized, aligned_timestamps, policy, target_len = self._align_policy_stream_inputs(
            signal_values=signal_values,
            trade_returns=trade_returns,
            timestamps=timestamps,
            policy_frame=policy_frame,
        )
        if target_len <= 0 or group_labels is None:
            return {}

        labels_arr = np.asarray(group_labels, dtype=object).reshape(-1)
        if labels_arr.size < target_len:
            return {}
        labels_arr = labels_arr[-target_len:]
        normalized = (
            pd.Series(labels_arr, dtype="object")
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .replace("", "unknown")
            .to_numpy(dtype=object)
        )

        results: dict[str, dict[str, float]] = {}
        for group_name in sorted({str(value) for value in normalized if str(value).strip()}):
            group_mask = normalized == group_name
            group_rows = int(np.count_nonzero(group_mask))
            if group_rows < int(max(1, min_rows)):
                continue
            group_summary = self._summarize_policy_adjusted_signal_stream(
                signal_values=signal[group_mask],
                trade_returns=realized[group_mask],
                timestamps=(
                    aligned_timestamps[group_mask]
                    if aligned_timestamps is not None and len(aligned_timestamps) == target_len
                    else None
                ),
                policy_frame=(
                    policy.loc[group_mask].reset_index(drop=True)
                    if policy is not None and len(policy) == target_len
                    else None
                ),
            )
            results[str(group_name)] = {
                "rows": float(group_rows),
                "accuracy": 0.0,
                "sharpe": float(group_summary.get("policy_sharpe", 0.0)),
                "max_drawdown": float(group_summary.get("policy_max_drawdown", 0.0)),
                "trade_count": float(group_summary.get("policy_trade_count", 0.0)),
                "win_rate": float(group_summary.get("policy_selected_win_rate", 0.0)),
                "turnover": float(group_summary.get("policy_turnover", 0.0)),
                "active_signal_rate": float(group_summary.get("policy_active_signal_rate", 0.0)),
                "annual_return": float(group_summary.get("policy_annual_return", 0.0)),
                "calmar": float(group_summary.get("policy_calmar", 0.0)),
                "pnl": float(group_summary.get("policy_pnl", 0.0)),
                "loss_pnl": float(group_summary.get("policy_loss_pnl", 0.0)),
                "tail_loss_pnl": float(group_summary.get("policy_tail_loss_pnl", 0.0)),
                "underwater_ratio": float(group_summary.get("policy_underwater_ratio", 0.0)),
                "cvar_95": float(group_summary.get("policy_cvar_95", 0.0)),
                "risk_adjusted_score": float(group_summary.get("policy_risk_adjusted_score", 0.0)),
                "sharpe_observation_confidence": float(
                    group_summary.get("policy_sharpe_observation_confidence", 0.0)
                ),
                "trade_return_observations": float(
                    group_summary.get("policy_trade_return_observations", 0.0)
                ),
            }
        return results

    def _resolve_effective_holdout_metric_payload(self) -> dict[str, Any]:
        """Resolve which holdout metrics should drive gates and overfitting diagnostics."""
        payload = {
            "source": "base",
            "sharpe": float(self.training_metrics.get("holdout_sharpe", 0.0)),
            "trade_count": float(self.training_metrics.get("holdout_trade_count", 0.0)),
            "active_signal_rate": float(
                self.training_metrics.get("holdout_active_signal_rate", 0.0)
            ),
            "max_drawdown": float(self.training_metrics.get("holdout_max_drawdown", 0.0)),
            "trade_return_observations": float(
                self.training_metrics.get("holdout_trade_return_observations", 0.0)
            ),
            "sharpe_observation_confidence": float(
                self.training_metrics.get("holdout_sharpe_observation_confidence", 0.0)
            ),
        }
        expected_edge_enabled = bool(self.training_metrics.get("expected_edge_policy_enabled", 0.0))
        policy_sharpe = float(
            self.training_metrics.get("expected_edge_holdout_policy_sharpe", np.nan)
        )
        if expected_edge_enabled and np.isfinite(policy_sharpe):
            payload.update(
                {
                    "source": "expected_edge_policy",
                    "sharpe": policy_sharpe,
                    "trade_count": float(
                        self.training_metrics.get("expected_edge_holdout_policy_trade_count", 0.0)
                    ),
                    "active_signal_rate": float(
                        self.training_metrics.get(
                            "expected_edge_holdout_policy_active_signal_rate",
                            self.training_metrics.get("expected_edge_holdout_selected_rate", 0.0),
                        )
                    ),
                    "max_drawdown": float(
                        self.training_metrics.get("expected_edge_holdout_policy_max_drawdown", 0.0)
                    ),
                    "trade_return_observations": float(
                        self.training_metrics.get(
                            "expected_edge_holdout_policy_trade_return_observations",
                            self.training_metrics.get("expected_edge_holdout_selected_count", 0.0),
                        )
                    ),
                    "sharpe_observation_confidence": float(
                        self.training_metrics.get(
                            "expected_edge_holdout_policy_sharpe_observation_confidence",
                            0.0,
                        )
                    ),
                }
            )
        self.training_metrics["effective_holdout_metric_source"] = payload["source"]
        self.training_metrics["effective_holdout_trade_count_metric"] = float(
            payload["trade_count"]
        )
        self.training_metrics["effective_holdout_active_signal_rate_metric"] = float(
            payload["active_signal_rate"]
        )
        self.training_metrics["effective_holdout_max_drawdown_metric"] = float(
            payload["max_drawdown"]
        )
        self.training_metrics["effective_holdout_trade_return_observations"] = float(
            payload["trade_return_observations"]
        )
        self.training_metrics["effective_holdout_sharpe_observation_confidence"] = float(
            payload["sharpe_observation_confidence"]
        )
        raw_threshold_active_rate = float(
            self.training_metrics.get(
                "holdout_raw_threshold_hits_active_rate",
                self.training_metrics.get("holdout_base_signal_funnel_selected_rate", 0.0),
            )
        )
        execution_signal_active_rate = float(
            self.training_metrics.get(
                "holdout_execution_signal_activity_active_signal_rate",
                self.training_metrics.get("holdout_execution_signal_funnel_selected_rate", 0.0),
            )
        )
        portfolio_trade_active_rate = float(
            self.training_metrics.get("holdout_active_signal_rate", 0.0)
        )
        expected_edge_selected_rate = float(
            self.training_metrics.get("expected_edge_holdout_selected_rate", 0.0)
        )
        activity_summary = {
            "source": payload["source"],
            "raw_threshold_active_rate": raw_threshold_active_rate,
            "execution_signal_active_rate": execution_signal_active_rate,
            "effective_gate_active_rate": float(payload["active_signal_rate"]),
            "portfolio_trade_active_rate": portfolio_trade_active_rate,
            "expected_edge_selected_rate": expected_edge_selected_rate,
        }
        self.training_metrics["holdout_signal_activity_summary"] = activity_summary
        self.training_metrics["holdout_raw_threshold_active_rate_metric"] = (
            raw_threshold_active_rate
        )
        self.training_metrics["holdout_execution_signal_active_rate_metric"] = (
            execution_signal_active_rate
        )
        self.training_metrics["holdout_portfolio_trade_active_rate_metric"] = (
            portfolio_trade_active_rate
        )
        self.training_metrics["holdout_expected_edge_selected_rate_metric"] = (
            expected_edge_selected_rate
        )
        return payload

    def _record_expected_edge_metrics(self, prefix: str, summary: dict[str, Any]) -> None:
        """Flatten expected-edge summaries into training metrics for artifacts."""
        self.training_metrics[f"{prefix}_summary"] = dict(summary)
        for key, value in summary.items():
            metric_key = f"{prefix}_{key}"
            if isinstance(value, (int, float, np.floating, np.integer)):
                self.training_metrics[metric_key] = float(value)
            elif isinstance(value, list):
                self.training_metrics[metric_key] = list(value)

    def _record_signal_funnel_metrics(
        self,
        prefix: str,
        summary: dict[str, Any],
        *,
        by_symbol: dict[str, Any] | None = None,
        by_regime: dict[str, Any] | None = None,
    ) -> None:
        """Persist signal-funnel diagnostics for artifact review and run analysis."""
        self.training_metrics[f"{prefix}_signal_funnel"] = self._json_safe(summary)
        if by_symbol:
            self.training_metrics[f"{prefix}_signal_funnel_by_symbol"] = self._json_safe(by_symbol)
        if by_regime:
            self.training_metrics[f"{prefix}_signal_funnel_by_regime"] = self._json_safe(by_regime)
        for key, value in summary.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                self.training_metrics[f"{prefix}_signal_funnel_{key}"] = float(value)

    def _record_summary_metrics(self, metric_name: str, summary: dict[str, Any]) -> None:
        """Persist generic diagnostics payloads and flatten scalar metrics for artifacts."""
        self.training_metrics[metric_name] = self._json_safe(summary)
        for key, value in summary.items():
            if isinstance(value, (np.bool_, bool)):
                self.training_metrics[f"{metric_name}_{key}"] = float(value)
            elif isinstance(value, (int, float, np.floating, np.integer)):
                self.training_metrics[f"{metric_name}_{key}"] = float(value)

    @staticmethod
    def _summarize_probability_distribution(probabilities: np.ndarray) -> dict[str, Any]:
        """Summarize score spread to diagnose compression toward the no-trade region."""
        values = np.asarray(probabilities, dtype=float).reshape(-1)
        rows = int(values.size)
        finite = np.clip(values[np.isfinite(values)], 0.0, 1.0)
        summary: dict[str, Any] = {
            "rows": rows,
            "finite_count": int(finite.size),
            "non_finite_count": int(rows - finite.size),
        }
        if finite.size <= 0:
            summary.update(
                {
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                    "std": 0.0,
                    "span": 0.0,
                    "p01": 0.0,
                    "p05": 0.0,
                    "p25": 0.0,
                    "p50": 0.0,
                    "p75": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                    "near_mid_count": 0,
                    "near_mid_rate": 0.0,
                    "low_confidence_count": 0,
                    "low_confidence_rate": 0.0,
                    "tail_count": 0,
                    "tail_rate": 0.0,
                }
            )
            return summary

        distance_from_mid = np.abs(finite - 0.5)
        summary.update(
            {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "span": float(np.max(finite) - np.min(finite)),
                "p01": float(np.quantile(finite, 0.01)),
                "p05": float(np.quantile(finite, 0.05)),
                "p25": float(np.quantile(finite, 0.25)),
                "p50": float(np.quantile(finite, 0.50)),
                "p75": float(np.quantile(finite, 0.75)),
                "p95": float(np.quantile(finite, 0.95)),
                "p99": float(np.quantile(finite, 0.99)),
                "near_mid_count": int(np.count_nonzero(distance_from_mid <= 0.02)),
                "near_mid_rate": float(np.mean(distance_from_mid <= 0.02)),
                "low_confidence_count": int(np.count_nonzero(distance_from_mid <= 0.05)),
                "low_confidence_rate": float(np.mean(distance_from_mid <= 0.05)),
                "tail_count": int(np.count_nonzero(distance_from_mid >= 0.10)),
                "tail_rate": float(np.mean(distance_from_mid >= 0.10)),
            }
        )
        return summary

    @staticmethod
    def _summarize_probability_shift(
        baseline_values: np.ndarray,
        comparison_values: np.ndarray,
    ) -> dict[str, Any]:
        """Summarize how much a second score stream compresses or widens probability spread."""
        baseline = np.asarray(baseline_values, dtype=float).reshape(-1)
        comparison = np.asarray(comparison_values, dtype=float).reshape(-1)
        target_len = min(len(baseline), len(comparison))
        summary: dict[str, Any] = {
            "rows": int(target_len),
            "finite_count": 0,
            "non_finite_count": int(target_len),
        }
        if target_len <= 0:
            summary.update(
                {
                    "mean_delta": 0.0,
                    "std_delta": 0.0,
                    "mean_abs_delta": 0.0,
                    "p50_abs_delta": 0.0,
                    "p95_abs_delta": 0.0,
                    "max_abs_delta": 0.0,
                    "toward_mid_count": 0,
                    "toward_mid_rate": 0.0,
                    "away_from_mid_count": 0,
                    "away_from_mid_rate": 0.0,
                    "mid_cross_count": 0,
                    "mid_cross_rate": 0.0,
                    "std_ratio": 1.0,
                    "span_ratio": 1.0,
                }
            )
            return summary

        baseline = np.clip(baseline[-target_len:], 0.0, 1.0)
        comparison = np.clip(comparison[-target_len:], 0.0, 1.0)
        finite_mask = np.isfinite(baseline) & np.isfinite(comparison)
        baseline = baseline[finite_mask]
        comparison = comparison[finite_mask]
        summary["finite_count"] = int(baseline.size)
        summary["non_finite_count"] = int(target_len - baseline.size)
        if baseline.size <= 0:
            summary.update(
                {
                    "mean_delta": 0.0,
                    "std_delta": 0.0,
                    "mean_abs_delta": 0.0,
                    "p50_abs_delta": 0.0,
                    "p95_abs_delta": 0.0,
                    "max_abs_delta": 0.0,
                    "toward_mid_count": 0,
                    "toward_mid_rate": 0.0,
                    "away_from_mid_count": 0,
                    "away_from_mid_rate": 0.0,
                    "mid_cross_count": 0,
                    "mid_cross_rate": 0.0,
                    "std_ratio": 1.0,
                    "span_ratio": 1.0,
                }
            )
            return summary

        delta = comparison - baseline
        baseline_mid_distance = np.abs(baseline - 0.5)
        comparison_mid_distance = np.abs(comparison - 0.5)
        toward_mid = comparison_mid_distance < (baseline_mid_distance - 1e-9)
        away_from_mid = comparison_mid_distance > (baseline_mid_distance + 1e-9)
        mid_cross = (
            ((baseline - 0.5) * (comparison - 0.5) < 0.0)
            & (baseline_mid_distance > 1e-8)
            & (comparison_mid_distance > 1e-8)
        )
        baseline_std = float(np.std(baseline))
        comparison_std = float(np.std(comparison))
        baseline_span = float(np.max(baseline) - np.min(baseline))
        comparison_span = float(np.max(comparison) - np.min(comparison))
        abs_delta = np.abs(delta)
        summary.update(
            {
                "mean_delta": float(np.mean(delta)),
                "std_delta": float(np.std(delta)),
                "mean_abs_delta": float(np.mean(abs_delta)),
                "p50_abs_delta": float(np.quantile(abs_delta, 0.50)),
                "p95_abs_delta": float(np.quantile(abs_delta, 0.95)),
                "max_abs_delta": float(np.max(abs_delta)),
                "toward_mid_count": int(np.count_nonzero(toward_mid)),
                "toward_mid_rate": float(np.mean(toward_mid)),
                "away_from_mid_count": int(np.count_nonzero(away_from_mid)),
                "away_from_mid_rate": float(np.mean(away_from_mid)),
                "mid_cross_count": int(np.count_nonzero(mid_cross)),
                "mid_cross_rate": float(np.mean(mid_cross)),
                "std_ratio": (
                    float(comparison_std / baseline_std) if baseline_std > 1e-12 else 1.0
                ),
                "span_ratio": (
                    float(comparison_span / baseline_span) if baseline_span > 1e-12 else 1.0
                ),
            }
        )
        return summary

    @staticmethod
    def _summarize_threshold_hits(
        probabilities: np.ndarray,
        long_thresholds: float | np.ndarray,
        short_thresholds: float | np.ndarray,
    ) -> dict[str, Any]:
        """Summarize score hits above/below thresholds to expose no-trade compression."""
        values = np.asarray(probabilities, dtype=float).reshape(-1)
        if np.isscalar(long_thresholds):
            long_values = np.full(len(values), float(long_thresholds), dtype=float)
        else:
            long_values = np.asarray(long_thresholds, dtype=float).reshape(-1)
        if np.isscalar(short_thresholds):
            short_values = np.full(len(values), float(short_thresholds), dtype=float)
        else:
            short_values = np.asarray(short_thresholds, dtype=float).reshape(-1)
        target_len = min(len(values), len(long_values), len(short_values))
        summary: dict[str, Any] = {
            "rows": int(target_len),
            "finite_count": 0,
            "non_finite_count": int(target_len),
        }
        if target_len <= 0:
            summary.update(
                {
                    "long_hit_count": 0,
                    "short_hit_count": 0,
                    "active_count": 0,
                    "within_band_count": 0,
                    "long_hit_rate": 0.0,
                    "short_hit_rate": 0.0,
                    "active_rate": 0.0,
                    "within_band_rate": 0.0,
                    "mean_long_threshold": 0.0,
                    "mean_short_threshold": 0.0,
                    "mean_threshold_gap": 0.0,
                    "min_threshold_gap": 0.0,
                    "max_threshold_gap": 0.0,
                    "threshold_overlap_count": 0,
                }
            )
            return summary

        values = values[-target_len:]
        long_values = long_values[-target_len:]
        short_values = short_values[-target_len:]
        finite_mask = np.isfinite(values) & np.isfinite(long_values) & np.isfinite(short_values)
        values = values[finite_mask]
        long_values = long_values[finite_mask]
        short_values = short_values[finite_mask]
        summary["finite_count"] = int(values.size)
        summary["non_finite_count"] = int(target_len - values.size)
        if values.size <= 0:
            summary.update(
                {
                    "long_hit_count": 0,
                    "short_hit_count": 0,
                    "active_count": 0,
                    "within_band_count": 0,
                    "long_hit_rate": 0.0,
                    "short_hit_rate": 0.0,
                    "active_rate": 0.0,
                    "within_band_rate": 0.0,
                    "mean_long_threshold": 0.0,
                    "mean_short_threshold": 0.0,
                    "mean_threshold_gap": 0.0,
                    "min_threshold_gap": 0.0,
                    "max_threshold_gap": 0.0,
                    "threshold_overlap_count": 0,
                }
            )
            return summary

        long_hit = values >= long_values
        short_hit = values <= short_values
        active = long_hit | short_hit
        gap = long_values - short_values
        summary.update(
            {
                "long_hit_count": int(np.count_nonzero(long_hit)),
                "short_hit_count": int(np.count_nonzero(short_hit)),
                "active_count": int(np.count_nonzero(active)),
                "within_band_count": int(np.count_nonzero(~active)),
                "long_hit_rate": float(np.mean(long_hit)),
                "short_hit_rate": float(np.mean(short_hit)),
                "active_rate": float(np.mean(active)),
                "within_band_rate": float(np.mean(~active)),
                "mean_long_threshold": float(np.mean(long_values)),
                "mean_short_threshold": float(np.mean(short_values)),
                "mean_threshold_gap": float(np.mean(gap)),
                "min_threshold_gap": float(np.min(gap)),
                "max_threshold_gap": float(np.max(gap)),
                "threshold_overlap_count": int(np.count_nonzero(gap <= 0.0)),
            }
        )
        return summary

    @staticmethod
    def _summarize_signal_activity(
        signal_values: np.ndarray,
        *,
        positions: np.ndarray | None = None,
        trade_mask: np.ndarray | None = None,
        entry_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Summarize final signal activity after execution-aware filtering."""
        signals = np.asarray(signal_values, dtype=float).reshape(-1)
        rows = int(signals.size)
        active_mask = np.isfinite(signals) & (np.abs(signals) > 1e-8)
        summary: dict[str, Any] = {
            "rows": rows,
            "active_signal_count": int(np.count_nonzero(active_mask)),
            "active_signal_rate": float(np.mean(active_mask)) if rows else 0.0,
            "long_signal_count": int(np.count_nonzero(signals > 1e-8)),
            "short_signal_count": int(np.count_nonzero(signals < -1e-8)),
        }
        if positions is not None:
            position_values = np.asarray(positions, dtype=float).reshape(-1)
            target_len = min(len(position_values), rows)
            if target_len > 0:
                position_values = position_values[-target_len:]
                summary["position_active_count"] = int(
                    np.count_nonzero(np.abs(position_values) > 1e-8)
                )
                summary["mean_abs_position"] = float(np.mean(np.abs(position_values)))
        if trade_mask is not None:
            trade_values = np.asarray(trade_mask, dtype=bool).reshape(-1)
            target_len = min(len(trade_values), rows)
            if target_len > 0:
                summary["trade_mask_count"] = int(np.count_nonzero(trade_values[-target_len:]))
        if entry_mask is not None:
            entry_values = np.asarray(entry_mask, dtype=bool).reshape(-1)
            target_len = min(len(entry_values), rows)
            if target_len > 0:
                summary["entry_count"] = int(np.count_nonzero(entry_values[-target_len:]))
        return summary

    def _refresh_oof_probability_diagnostics(
        self,
        *,
        long_threshold: float | None = None,
        short_threshold: float | None = None,
    ) -> None:
        """Refresh OOF diagnostics after calibration or threshold changes."""
        raw_values = (
            np.asarray(self.oof_primary_proba_raw, dtype=float).reshape(-1)
            if self.oof_primary_proba_raw is not None
            else None
        )
        calibrated_values = (
            np.asarray(self.oof_primary_proba, dtype=float).reshape(-1)
            if self.oof_primary_proba is not None
            else None
        )
        if raw_values is None and calibrated_values is None:
            return
        if raw_values is None and calibrated_values is not None:
            raw_values = calibrated_values.copy()
        if calibrated_values is None and raw_values is not None:
            calibrated_values = raw_values.copy()
        if raw_values is None or calibrated_values is None:
            return

        self._record_summary_metrics(
            "oof_raw_probability_distribution",
            self._summarize_probability_distribution(raw_values),
        )
        self._record_summary_metrics(
            "oof_calibrated_probability_distribution",
            self._summarize_probability_distribution(calibrated_values),
        )
        self._record_summary_metrics(
            "oof_raw_to_calibrated_probability_shift",
            self._summarize_probability_shift(raw_values, calibrated_values),
        )
        if long_threshold is None or short_threshold is None:
            return
        self._record_summary_metrics(
            "oof_raw_threshold_hits",
            self._summarize_threshold_hits(raw_values, long_threshold, short_threshold),
        )
        self._record_summary_metrics(
            "oof_calibrated_threshold_hits",
            self._summarize_threshold_hits(calibrated_values, long_threshold, short_threshold),
        )

    def _build_signal_funnel_summary(
        self,
        *,
        probabilities: np.ndarray,
        signal_values: np.ndarray,
        trade_returns: np.ndarray,
        policy_frame: pd.DataFrame | None = None,
        symbols: np.ndarray | None = None,
        regimes: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Build row, symbol, and regime funnel summaries for trade-admission diagnostics."""
        target_len = min(len(probabilities), len(signal_values), len(trade_returns))
        if policy_frame is not None:
            target_len = min(target_len, len(policy_frame))
        if target_len <= 0:
            empty_summary = {
                "rows": 0,
                "candidate_count": 0,
                "selected_count": 0,
                "candidate_rate": 0.0,
                "selected_rate": 0.0,
            }
            return empty_summary, {}, {}

        probabilities = np.asarray(probabilities, dtype=float)[-target_len:]
        signal_values = np.asarray(signal_values, dtype=float)[-target_len:]
        trade_returns = np.asarray(trade_returns, dtype=float)[-target_len:]
        policy = (
            policy_frame.iloc[-target_len:].reset_index(drop=True)
            if policy_frame is not None
            else None
        )
        symbol_values = None
        if symbols is not None:
            symbol_arr = np.asarray(symbols, dtype=object).reshape(-1)
            if symbol_arr.size >= target_len:
                symbol_values = symbol_arr[-target_len:]
        regime_values = None
        if regimes is not None:
            regime_arr = np.asarray(regimes, dtype=object).reshape(-1)
            if regime_arr.size >= target_len:
                regime_values = regime_arr[-target_len:]

        finite_probability_mask = np.isfinite(probabilities)
        finite_trade_return_mask = np.isfinite(trade_returns)
        candidate_mask = finite_probability_mask & (np.abs(signal_values) > 1e-8)
        candidate_with_return_mask = candidate_mask & finite_trade_return_mask
        long_candidate_mask = candidate_mask & (signal_values > 0.0)
        short_candidate_mask = candidate_mask & (signal_values < 0.0)

        regime_enabled_mask = np.ones(target_len, dtype=bool)
        pass_probability_mask = np.ones(target_len, dtype=bool)
        expected_edge_mask = np.ones(target_len, dtype=bool)
        selected_mask = candidate_with_return_mask.copy()
        blocked_by_regime_mask = np.zeros(target_len, dtype=bool)
        blocked_by_probability_mask = np.zeros(target_len, dtype=bool)
        blocked_by_expected_edge_mask = np.zeros(target_len, dtype=bool)

        if policy is not None and not policy.empty:
            if "regime_policy_enabled" in policy.columns:
                regime_enabled_mask = policy["regime_policy_enabled"].to_numpy(dtype=bool)
            if {
                "edge_pass_probability",
                "regime_policy_min_pass_probability",
            }.issubset(policy.columns):
                pass_probability_mask = np.isfinite(
                    policy["edge_pass_probability"].to_numpy(dtype=float)
                ) & (
                    policy["edge_pass_probability"].to_numpy(dtype=float)
                    >= policy["regime_policy_min_pass_probability"].to_numpy(dtype=float)
                )
            if {"expected_edge", "regime_policy_min_expected_edge"}.issubset(policy.columns):
                expected_edge_mask = np.isfinite(policy["expected_edge"].to_numpy(dtype=float)) & (
                    policy["expected_edge"].to_numpy(dtype=float)
                    >= policy["regime_policy_min_expected_edge"].to_numpy(dtype=float)
                )
            if "edge_policy_pass" in policy.columns:
                selected_mask = candidate_with_return_mask & policy["edge_policy_pass"].to_numpy(
                    dtype=bool
                )
            blocked_by_regime_mask = candidate_mask & ~regime_enabled_mask
            blocked_by_probability_mask = (
                candidate_mask & regime_enabled_mask & ~pass_probability_mask
            )
            blocked_by_expected_edge_mask = (
                candidate_mask & regime_enabled_mask & pass_probability_mask & ~expected_edge_mask
            )

        def _candidate_return_mean(mask: np.ndarray) -> float:
            values = trade_returns[mask & np.isfinite(trade_returns)]
            return float(np.mean(values)) if values.size else 0.0

        def _selected_win_rate(mask: np.ndarray) -> float:
            values = trade_returns[mask & np.isfinite(trade_returns)]
            return float(np.mean(values > 0.0)) if values.size else 0.0

        def _summary_for_mask(mask: np.ndarray) -> dict[str, Any]:
            group_rows = int(np.count_nonzero(mask))
            group_candidate_mask = candidate_mask & mask
            group_candidate_with_return_mask = candidate_with_return_mask & mask
            group_selected_mask = selected_mask & mask
            candidate_count = int(np.count_nonzero(group_candidate_mask))
            selected_count = int(np.count_nonzero(group_selected_mask))
            candidate_mean_trade_return = _candidate_return_mean(group_candidate_with_return_mask)
            selected_mean_trade_return = _candidate_return_mean(group_selected_mask)
            return {
                "rows": group_rows,
                "candidate_count": candidate_count,
                "candidate_with_return_count": int(
                    np.count_nonzero(group_candidate_with_return_mask)
                ),
                "selected_count": selected_count,
                "candidate_rate": float(candidate_count / group_rows) if group_rows else 0.0,
                "selected_rate": float(selected_count / group_rows) if group_rows else 0.0,
                "selected_vs_candidate_rate": (
                    float(selected_count / candidate_count) if candidate_count else 0.0
                ),
                "blocked_by_regime_count": int(np.count_nonzero(blocked_by_regime_mask & mask)),
                "blocked_by_probability_count": int(
                    np.count_nonzero(blocked_by_probability_mask & mask)
                ),
                "blocked_by_expected_edge_count": int(
                    np.count_nonzero(blocked_by_expected_edge_mask & mask)
                ),
                "candidate_mean_trade_return": candidate_mean_trade_return,
                "selected_mean_trade_return": selected_mean_trade_return,
                "selected_edge_lift": (
                    float(selected_mean_trade_return - candidate_mean_trade_return)
                    if candidate_count
                    else 0.0
                ),
                "selected_win_rate": _selected_win_rate(group_selected_mask),
            }

        full_mask = np.ones(target_len, dtype=bool)
        summary = _summary_for_mask(full_mask)
        summary.update(
            {
                "rows": int(target_len),
                "finite_probability_count": int(np.count_nonzero(finite_probability_mask)),
                "finite_trade_return_count": int(np.count_nonzero(finite_trade_return_mask)),
                "zero_signal_count": int(np.count_nonzero(np.abs(signal_values) <= 1e-8)),
                "long_candidate_count": int(np.count_nonzero(long_candidate_mask)),
                "short_candidate_count": int(np.count_nonzero(short_candidate_mask)),
                "regime_enabled_candidate_count": int(
                    np.count_nonzero(candidate_mask & regime_enabled_mask)
                ),
                "probability_gate_pass_count": int(
                    np.count_nonzero(candidate_mask & regime_enabled_mask & pass_probability_mask)
                ),
                "expected_edge_gate_pass_count": int(
                    np.count_nonzero(
                        candidate_mask
                        & regime_enabled_mask
                        & pass_probability_mask
                        & expected_edge_mask
                    )
                ),
            }
        )

        def _group_breakdown(labels: np.ndarray | None) -> dict[str, Any]:
            if labels is None:
                return {}
            normalized = (
                pd.Series(labels, dtype="object")
                .fillna("unknown")
                .astype(str)
                .str.strip()
                .replace("", "unknown")
                .to_numpy(dtype=object)
            )
            payload: dict[str, Any] = {}
            for group_name in sorted({str(value) for value in normalized if str(value).strip()}):
                group_mask = normalized == group_name
                payload[str(group_name)] = _summary_for_mask(group_mask)
            return payload

        return summary, _group_breakdown(symbol_values), _group_breakdown(regime_values)

    def _evaluate_expected_edge_policy_holdout(
        self,
        policy_model: ExpectedEdgePolicyModel,
        *,
        long_threshold: float,
        short_threshold: float,
        regime_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate the expected-edge policy on aligned holdout predictions."""
        if self.holdout_features is None or self.holdout_features.empty or self.model is None:
            return {"source": "unavailable"}

        X_holdout = self.holdout_features.values
        holdout_probabilities = self._align_probabilities(
            np.asarray(
                self._get_predictions_proba(
                    self.model,
                    X_holdout,
                    timestamps=self.holdout_timestamps,
                ),
                dtype=float,
            ),
            len(self.holdout_features),
        )
        holdout_signal = derive_base_signal(
            holdout_probabilities,
            long_threshold=float(long_threshold),
            short_threshold=float(short_threshold),
        )
        holdout_signed_returns = self._resolve_signed_realized_returns(
            forward_returns=self.holdout_primary_forward_returns,
            event_net_returns=self.holdout_cost_aware_event_returns,
            event_directions=self.holdout_primary_event_directions,
        )
        if holdout_signed_returns is None:
            return {"source": "missing_returns"}

        aligned_len = min(
            len(self.holdout_features), len(holdout_probabilities), len(holdout_signed_returns)
        )
        if aligned_len <= 1:
            return {"source": "insufficient_rows"}
        holdout_frame = self.holdout_features.iloc[-aligned_len:].reset_index(drop=True)
        holdout_probabilities = holdout_probabilities[-aligned_len:]
        holdout_signal = holdout_signal[-aligned_len:]
        holdout_signed_returns = holdout_signed_returns[-aligned_len:]
        holdout_confidence = np.clip(np.abs(holdout_probabilities - 0.5) * 2.0, 0.0, 1.0)
        holdout_signal, holdout_confidence, meta_summary = self._apply_meta_signal_filter(
            feature_frame=holdout_frame,
            signal_values=holdout_signal,
            close_prices=self.holdout_close_prices,
            confidence=holdout_confidence,
        )
        holdout_trade_returns = self._resolve_trade_realized_returns(
            holdout_signed_returns,
            holdout_signal,
        )
        if holdout_trade_returns is None:
            return {"source": "missing_returns"}
        holdout_regimes = None
        if self.holdout_regimes is not None and len(self.holdout_regimes) >= aligned_len:
            holdout_regimes = np.asarray(self.holdout_regimes, dtype=object)[-aligned_len:]
        holdout_symbols = None
        if self.holdout_symbols is not None and len(self.holdout_symbols) >= aligned_len:
            holdout_symbols = np.asarray(self.holdout_symbols, dtype=object)[-aligned_len:]
        holdout_timestamps = None
        if self.holdout_timestamps is not None and len(self.holdout_timestamps) >= aligned_len:
            holdout_timestamps = np.asarray(self.holdout_timestamps)[-aligned_len:]

        policy_frame = policy_model.predict_policy(
            holdout_frame,
            probabilities=holdout_probabilities,
            long_threshold=float(long_threshold),
            short_threshold=float(short_threshold),
            signal_values=holdout_signal,
            confidence=holdout_confidence,
            symbols=holdout_symbols,
            regimes=holdout_regimes,
            regime_policy=regime_policy,
        )
        candidate_mask = np.abs(holdout_signal) > 1e-8
        selected_mask = (
            candidate_mask
            & policy_frame["edge_policy_pass"].to_numpy(dtype=bool)
            & np.isfinite(holdout_trade_returns)
        )
        candidate_returns = holdout_trade_returns[
            candidate_mask & np.isfinite(holdout_trade_returns)
        ]
        selected_returns = holdout_trade_returns[selected_mask]
        expected_edge = policy_frame["expected_edge"].to_numpy(dtype=float)
        corr_mask = candidate_mask & np.isfinite(holdout_trade_returns) & np.isfinite(expected_edge)
        correlation = 0.0
        if int(np.count_nonzero(corr_mask)) >= 20:
            correlation = float(
                np.corrcoef(expected_edge[corr_mask], holdout_trade_returns[corr_mask])[0, 1]
            )
            if not np.isfinite(correlation):
                correlation = 0.0

        holdout_funnel_summary, holdout_funnel_by_symbol, holdout_funnel_by_regime = (
            self._build_signal_funnel_summary(
                probabilities=holdout_probabilities,
                signal_values=holdout_signal,
                trade_returns=holdout_trade_returns,
                policy_frame=policy_frame,
                symbols=holdout_symbols,
                regimes=holdout_regimes,
            )
        )
        self._record_signal_funnel_metrics(
            "expected_edge_holdout",
            holdout_funnel_summary,
            by_symbol=holdout_funnel_by_symbol,
            by_regime=holdout_funnel_by_regime,
        )
        policy_stream_summary = self._summarize_policy_adjusted_signal_stream(
            signal_values=holdout_signal,
            trade_returns=holdout_trade_returns,
            timestamps=holdout_timestamps,
            policy_frame=policy_frame,
        )
        holdout_regime_sample_floor = max(30, int(round(0.05 * aligned_len)))
        holdout_symbol_sample_floor = max(20, int(round(0.01 * aligned_len)))
        policy_regime_metrics = self._summarize_policy_adjusted_metrics_by_group(
            signal_values=holdout_signal,
            trade_returns=holdout_trade_returns,
            group_labels=holdout_regimes,
            timestamps=holdout_timestamps,
            policy_frame=policy_frame,
            min_rows=holdout_regime_sample_floor,
        )
        policy_symbol_metrics = self._summarize_policy_adjusted_metrics_by_group(
            signal_values=holdout_signal,
            trade_returns=holdout_trade_returns,
            group_labels=holdout_symbols,
            timestamps=holdout_timestamps,
            policy_frame=policy_frame,
            min_rows=holdout_symbol_sample_floor,
        )

        def _top_tail_contributors(
            metrics_by_group: dict[str, dict[str, float]],
            *,
            top_k: int = 8,
        ) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for group_name, group_metrics in metrics_by_group.items():
                tail_loss_pnl = float(group_metrics.get("tail_loss_pnl", 0.0))
                pnl = float(group_metrics.get("pnl", 0.0))
                if tail_loss_pnl >= 0.0 and pnl >= 0.0:
                    continue
                rows.append(
                    {
                        "name": str(group_name),
                        "tail_loss_pnl": tail_loss_pnl,
                        "pnl": pnl,
                        "loss_pnl": float(group_metrics.get("loss_pnl", 0.0)),
                        "underwater_ratio": float(group_metrics.get("underwater_ratio", 0.0)),
                        "trade_count": float(group_metrics.get("trade_count", 0.0)),
                        "sharpe": float(group_metrics.get("sharpe", 0.0)),
                        "max_drawdown": float(group_metrics.get("max_drawdown", 0.0)),
                    }
                )
            rows.sort(
                key=lambda item: (
                    float(item.get("tail_loss_pnl", 0.0)),
                    float(item.get("pnl", 0.0)),
                    -float(item.get("trade_count", 0.0)),
                )
            )
            return rows[:top_k]

        policy_symbol_count_total = 0.0
        if holdout_symbols is not None:
            policy_symbol_count_total = float(
                len(sorted({str(s).strip() for s in holdout_symbols if str(s).strip()}))
            )
        policy_symbol_count_evaluated = float(len(policy_symbol_metrics))
        policy_symbol_coverage_ratio = (
            float(policy_symbol_count_evaluated / policy_symbol_count_total)
            if policy_symbol_count_total > 0.0
            else 0.0
        )
        policy_symbol_sharpe_median = float(policy_stream_summary.get("policy_sharpe", 0.0))
        policy_symbol_sharpe_p25 = float(policy_symbol_sharpe_median)
        policy_symbol_sharpe_std = 0.0
        policy_symbol_underwater_ratio = 0.0
        policy_symbol_worst_sharpe = float(policy_symbol_sharpe_median)
        if policy_symbol_metrics:
            symbol_sharpes = np.asarray(
                [m.get("sharpe", 0.0) for m in policy_symbol_metrics.values()],
                dtype=float,
            )
            policy_symbol_sharpe_median = float(np.median(symbol_sharpes))
            policy_symbol_sharpe_p25 = float(np.quantile(symbol_sharpes, 0.25))
            policy_symbol_sharpe_std = float(np.std(symbol_sharpes))
            policy_symbol_underwater_ratio = float(np.mean(symbol_sharpes < 0.0))
            policy_symbol_worst_sharpe = float(np.min(symbol_sharpes))

        policy_worst_regime_sharpe = float(policy_stream_summary.get("policy_sharpe", 0.0))
        if policy_regime_metrics:
            policy_worst_regime_sharpe = float(
                min(v.get("sharpe", 0.0) for v in policy_regime_metrics.values())
            )
        policy_tail_loss_contributors_by_symbol = _top_tail_contributors(policy_symbol_metrics)
        policy_tail_loss_contributors_by_regime = _top_tail_contributors(policy_regime_metrics)

        return {
            "source": "holdout",
            "candidate_count": int(np.count_nonzero(candidate_mask)),
            "selected_count": int(np.count_nonzero(selected_mask)),
            "selected_rate": float(np.mean(selected_mask)) if aligned_len else 0.0,
            "candidate_mean_trade_return": (
                float(np.mean(candidate_returns)) if candidate_returns.size else 0.0
            ),
            "selected_mean_trade_return": (
                float(np.mean(selected_returns)) if selected_returns.size else 0.0
            ),
            "selected_win_rate": (
                float(np.mean(selected_returns > 0.0)) if selected_returns.size else 0.0
            ),
            "selected_edge_lift": (
                float(np.mean(selected_returns) - np.mean(candidate_returns))
                if selected_returns.size and candidate_returns.size
                else 0.0
            ),
            "expected_edge_correlation": correlation,
            "meta_filter_applied": float(bool(meta_summary.get("applied", False))),
            "meta_signal_retention_ratio": float(meta_summary.get("retention_ratio", 0.0)),
            "regime_policy_enabled": float(bool(regime_policy and regime_policy.get("enabled"))),
            "regime_count": float(
                len(regime_policy.get("regimes", {})) if isinstance(regime_policy, dict) else 0
            ),
            "worst_regime_sharpe": float(policy_worst_regime_sharpe),
            "regime_count_evaluated": float(len(policy_regime_metrics)),
            "symbol_count_total": float(policy_symbol_count_total),
            "symbol_count_evaluated": float(policy_symbol_count_evaluated),
            "symbol_coverage_ratio": float(policy_symbol_coverage_ratio),
            "symbol_sharpe_median": float(policy_symbol_sharpe_median),
            "symbol_sharpe_p25": float(policy_symbol_sharpe_p25),
            "symbol_sharpe_std": float(policy_symbol_sharpe_std),
            "symbol_underwater_ratio": float(policy_symbol_underwater_ratio),
            "symbol_worst_sharpe": float(policy_symbol_worst_sharpe),
            "regime_metrics": self._json_safe(policy_regime_metrics),
            "symbol_metrics": self._json_safe(policy_symbol_metrics),
            "tail_loss_contributors_by_symbol": self._json_safe(
                policy_tail_loss_contributors_by_symbol
            ),
            "tail_loss_contributors_by_regime": self._json_safe(
                policy_tail_loss_contributors_by_regime
            ),
            **policy_stream_summary,
        }

    def _train_expected_edge_policy(self) -> None:
        """Train the second-stage expected-edge policy from OOF-safe predictions."""
        self.logger.info("Phase 8.5: Training expected-edge policy...")
        self.training_metrics["expected_edge_policy_enabled"] = 0.0
        self.expected_edge_cv_return_series = []
        self.expected_edge_cv_active_return_series = []

        if self.features is None or self.features.empty:
            self.training_metrics["expected_edge_policy_reason"] = "missing_features"
            return
        if self.oof_primary_proba is None:
            self.training_metrics["expected_edge_policy_reason"] = "missing_oof_predictions"
            return

        signed_returns = self._resolve_signed_realized_returns(
            forward_returns=self.primary_forward_returns,
            event_net_returns=self.cost_aware_event_returns,
            event_directions=self.primary_event_directions,
        )
        if signed_returns is None:
            self.training_metrics["expected_edge_policy_reason"] = "missing_returns"
            return

        long_threshold, short_threshold = self._resolve_expected_edge_thresholds()
        probabilities = np.asarray(self.oof_primary_proba, dtype=float).reshape(-1)
        target_len = min(len(self.features), len(probabilities), len(signed_returns))
        if target_len <= 1:
            self.training_metrics["expected_edge_policy_reason"] = "insufficient_rows"
            return

        feature_frame = self.features.iloc[-target_len:].reset_index(drop=True)
        probabilities = probabilities[-target_len:]
        signed_returns = signed_returns[-target_len:]
        close_prices = (
            self.close_prices.iloc[-target_len:].reset_index(drop=True)
            if self.close_prices is not None and len(self.close_prices) >= target_len
            else None
        )
        signal_values = derive_base_signal(
            probabilities,
            long_threshold=float(long_threshold),
            short_threshold=float(short_threshold),
        )
        signal_confidence = np.clip(np.abs(probabilities - 0.5) * 2.0, 0.0, 1.0)
        signal_values, signal_confidence, meta_summary = self._apply_meta_signal_filter(
            feature_frame=feature_frame,
            signal_values=signal_values,
            close_prices=close_prices,
            confidence=signal_confidence,
        )
        self._record_summary_metrics(
            "expected_edge_meta_filter",
            meta_summary,
        )
        trade_returns = self._resolve_trade_realized_returns(signed_returns, signal_values)
        if trade_returns is None:
            self.training_metrics["expected_edge_policy_reason"] = "missing_trade_returns"
            return

        sample_weights = None
        if self.sample_weights is not None and len(self.sample_weights) >= target_len:
            sample_weights = np.asarray(self.sample_weights, dtype=float)[-target_len:]

        candidate_mask = np.isfinite(trade_returns) & (np.abs(signal_values) > 1e-8)
        candidate_rate = float(np.mean(candidate_mask)) if target_len else 0.0
        effective_min_coverage = ExpectedEdgePolicyModel.resolve_effective_min_coverage(
            min_coverage=float(self.config.expected_edge_min_coverage),
            min_samples=int(self.config.expected_edge_min_samples),
            row_count=int(target_len),
        )
        candidate_floor = {
            "rows": int(target_len),
            "candidate_count": int(np.count_nonzero(candidate_mask)),
            "candidate_rate": candidate_rate,
            "finite_trade_return_count": int(np.count_nonzero(np.isfinite(trade_returns))),
            "configured_min_samples": int(self.config.expected_edge_min_samples),
            "configured_min_coverage": float(self.config.expected_edge_min_coverage),
            "effective_min_coverage": float(effective_min_coverage),
            "meets_min_samples": bool(
                np.count_nonzero(candidate_mask) >= int(self.config.expected_edge_min_samples)
            ),
            "meets_min_coverage": bool(candidate_rate >= float(effective_min_coverage)),
            "long_candidate_count": int(np.count_nonzero(candidate_mask & (signal_values > 0.0))),
            "short_candidate_count": int(np.count_nonzero(candidate_mask & (signal_values < 0.0))),
        }
        self.training_metrics["expected_edge_candidate_floor"] = self._json_safe(candidate_floor)
        self._record_expected_edge_metrics("expected_edge_candidate_floor", candidate_floor)
        aligned_symbols = None
        if self.row_symbols is not None and len(self.row_symbols) >= target_len:
            aligned_symbols = np.asarray(self.row_symbols, dtype=object)[-target_len:]
        aligned_regimes = None
        if self.regimes is not None and len(self.regimes) >= target_len:
            aligned_regimes = np.asarray(self.regimes, dtype=object)[-target_len:]
        training_funnel_precheck, training_funnel_by_symbol, training_funnel_by_regime = (
            self._build_signal_funnel_summary(
                probabilities=probabilities,
                signal_values=signal_values,
                trade_returns=trade_returns,
                symbols=aligned_symbols,
                regimes=aligned_regimes,
            )
        )
        self._record_signal_funnel_metrics(
            "expected_edge_training_precheck",
            training_funnel_precheck,
            by_symbol=training_funnel_by_symbol,
            by_regime=training_funnel_by_regime,
        )

        policy_model = ExpectedEdgePolicyModel(
            config=ExpectedEdgePolicyConfig(
                min_samples=int(self.config.expected_edge_min_samples),
                min_coverage=float(self.config.expected_edge_min_coverage),
                max_context_features=int(self.config.expected_edge_max_context_features),
                min_pass_probability=float(self.config.expected_edge_min_pass_probability),
                min_expected_edge=float(self.config.expected_edge_min_expected_edge),
                min_signal_scale=float(self.config.expected_edge_min_signal_scale),
                max_signal_scale=float(self.config.expected_edge_max_signal_scale),
                use_symbol_priors=bool(self.config.expected_edge_use_symbol_priors),
                random_state=int(self.config.seed),
            )
        )

        try:
            training_summary = policy_model.fit(
                feature_frame,
                probabilities=probabilities,
                trade_returns=trade_returns,
                long_threshold=float(long_threshold),
                short_threshold=float(short_threshold),
                signal_values=signal_values,
                confidence=signal_confidence,
                sample_weights=sample_weights,
                symbols=aligned_symbols,
            )
        except ValueError as exc:
            self.training_metrics["expected_edge_policy_reason"] = str(exc)
            self.logger.warning("Expected-edge policy skipped: %s", exc)
            return

        self.expected_edge_model = policy_model
        self.training_metrics["expected_edge_policy_enabled"] = 1.0
        self.training_metrics["expected_edge_policy_reason"] = "trained"
        self.training_metrics["expected_edge_min_pass_probability"] = float(
            self.config.expected_edge_min_pass_probability
        )
        self.training_metrics["expected_edge_min_expected_edge"] = float(
            self.config.expected_edge_min_expected_edge
        )
        self.training_metrics["expected_edge_min_signal_scale"] = float(
            self.config.expected_edge_min_signal_scale
        )
        self.training_metrics["expected_edge_max_signal_scale"] = float(
            self.config.expected_edge_max_signal_scale
        )
        self.training_metrics["expected_edge_use_symbol_priors"] = float(
            bool(self.config.expected_edge_use_symbol_priors)
        )
        self._record_expected_edge_metrics("expected_edge_training", training_summary)
        training_policy_frame = policy_model.predict_policy(
            feature_frame,
            probabilities=probabilities,
            long_threshold=float(long_threshold),
            short_threshold=float(short_threshold),
            signal_values=signal_values,
            confidence=signal_confidence,
            symbols=aligned_symbols,
        )
        training_funnel_summary, training_funnel_by_symbol, training_funnel_by_regime = (
            self._build_signal_funnel_summary(
                probabilities=probabilities,
                signal_values=signal_values,
                trade_returns=trade_returns,
                policy_frame=training_policy_frame,
                symbols=aligned_symbols,
                regimes=aligned_regimes,
            )
        )
        self._record_signal_funnel_metrics(
            "expected_edge_training",
            training_funnel_summary,
            by_symbol=training_funnel_by_symbol,
            by_regime=training_funnel_by_regime,
        )
        regime_policy = derive_regime_conditioned_policy(
            regimes=aligned_regimes,
            trade_returns=trade_returns,
            signal_values=signal_values,
            selected_mask=training_policy_frame["edge_policy_pass"].to_numpy(dtype=bool),
            min_samples=int(self.config.expected_edge_min_samples),
            edge_reference=float(getattr(policy_model, "edge_reference_", 0.001)),
        )
        self.training_metrics["expected_edge_regime_policy"] = regime_policy
        if isinstance(regime_policy, dict):
            self.training_metrics["expected_edge_regime_policy_enabled"] = float(
                bool(regime_policy.get("enabled"))
            )
            self.training_metrics["expected_edge_regime_policy_reason"] = str(
                regime_policy.get("reason", "unavailable")
            )
            self.training_metrics["expected_edge_regime_policy_regimes"] = sorted(
                str(name)
                for name in (
                    regime_policy.get("regimes", {}).keys()
                    if isinstance(regime_policy.get("regimes", {}), dict)
                    else []
                )
            )
        training_runtime_policy_frame = policy_model.predict_policy(
            feature_frame,
            probabilities=probabilities,
            long_threshold=float(long_threshold),
            short_threshold=float(short_threshold),
            signal_values=signal_values,
            confidence=signal_confidence,
            symbols=aligned_symbols,
            regimes=aligned_regimes,
            regime_policy=regime_policy,
        )
        training_policy_stream = self._build_policy_adjusted_stream(
            signal_values=signal_values,
            trade_returns=trade_returns,
            timestamps=(
                np.asarray(self.timestamps)[-target_len:]
                if self.timestamps is not None and len(self.timestamps) >= target_len
                else None
            ),
            policy_frame=training_runtime_policy_frame,
        )
        training_policy_stream_summary = self._summarize_policy_stream_metrics(
            stream=training_policy_stream
        )
        self._record_expected_edge_metrics(
            "expected_edge_training_policy",
            training_policy_stream_summary,
        )
        training_policy_returns = np.asarray(
            training_policy_stream.get("portfolio_returns", []),
            dtype=float,
        )
        if training_policy_returns.size > 0:
            self.expected_edge_cv_return_series = [training_policy_returns.astype(float)]
        training_policy_trade_mask = np.asarray(
            training_policy_stream.get("portfolio_trade_mask", []),
            dtype=bool,
        )
        if training_policy_returns.size == training_policy_trade_mask.size:
            active_returns = training_policy_returns[training_policy_trade_mask]
            if active_returns.size > 0:
                self.expected_edge_cv_active_return_series = [active_returns.astype(float)]
        holdout_summary = self._evaluate_expected_edge_policy_holdout(
            policy_model,
            long_threshold=float(long_threshold),
            short_threshold=float(short_threshold),
            regime_policy=regime_policy,
        )
        self._record_expected_edge_metrics("expected_edge_holdout", holdout_summary)
        for summary_key in (
            "regime_metrics",
            "symbol_metrics",
            "tail_loss_contributors_by_symbol",
            "tail_loss_contributors_by_regime",
        ):
            summary_value = holdout_summary.get(summary_key)
            if summary_value:
                self.training_metrics[f"expected_edge_holdout_{summary_key}"] = summary_value
        self.logger.info(
            "Expected-edge policy trained: selected_rate=%.2f%% holdout_lift=%.6f regimes=%d",
            float(training_summary.get("selected_rate", 0.0)) * 100.0,
            float(holdout_summary.get("selected_edge_lift", 0.0)),
            len(regime_policy.get("regimes", {})) if isinstance(regime_policy, dict) else 0,
        )

    def _get_raw_predictions_proba(self, model, X, timestamps=None) -> np.ndarray:
        """Get raw prediction probabilities from model before post-hoc calibration."""
        if self._is_ranker_model():
            raw_scores = np.asarray(self._predict_with_model(model, X), dtype=float).reshape(-1)
            return self._normalize_ranker_scores(raw_scores, timestamps=timestamps).astype(float)

        try:
            if hasattr(model, "predict_proba"):
                proba = self._predict_proba_with_model(model, X)
                if proba.ndim == 2:
                    return proba[:, 1]
                return proba
        except (NotImplementedError, AttributeError):
            pass  # Model doesn't support predict_proba, use predict

        if hasattr(model, "predict"):
            # For regressors, normalize predictions to [0, 1] range
            predictions = self._predict_with_model(model, X)
            if isinstance(predictions, np.ndarray):
                # Clip and normalize to [0, 1]
                predictions = np.clip(predictions, -1, 1)
                predictions = (predictions + 1) / 2  # Map [-1, 1] to [0, 1]
            return predictions

        return np.zeros(len(X))

    def _get_predictions_proba(self, model, X, timestamps=None) -> np.ndarray:
        """Get prediction probabilities from model, applying attached calibration when present."""
        raw_values = np.asarray(
            self._get_raw_predictions_proba(model, X, timestamps=timestamps),
            dtype=float,
        ).reshape(-1)
        payload = self._resolve_probability_calibration_payload(model)
        if payload is None:
            return raw_values
        return self._apply_probability_calibration(raw_values, payload)

    def _calculate_fold_metrics(
        self,
        y_true,
        y_pred,
        y_proba,
        long_threshold: float = 0.55,
        short_threshold: float = 0.45,
        realized_forward_returns: np.ndarray | None = None,
        event_net_returns: np.ndarray | None = None,
        event_directions: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        symbols: np.ndarray | None = None,
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
        if event_directions is not None:
            event_directions = self._align_probabilities(
                np.asarray(event_directions, dtype=float),
                target_len=len(y_proba),
                fill_value=0.0,
            )
        aligned_symbols: np.ndarray | None = None
        if symbols is not None:
            raw_symbols = np.asarray(symbols, dtype=object).reshape(-1)
            if raw_symbols.size == len(y_proba):
                aligned_symbols = raw_symbols
            elif raw_symbols.size > len(y_proba):
                aligned_symbols = raw_symbols[-len(y_proba) :]
            else:
                aligned_symbols = None

        aligned_timestamps: np.ndarray | None = None
        if timestamps is not None:
            raw_timestamps = np.asarray(timestamps).reshape(-1)
            if raw_timestamps.size == len(y_proba):
                aligned_timestamps = raw_timestamps
            elif raw_timestamps.size > len(y_proba):
                aligned_timestamps = raw_timestamps[-len(y_proba) :]
            else:
                aligned_timestamps = None

        # Panel-safe execution ordering: ensure all execution/risk metrics are evaluated
        # in chronological order across symbols, not in symbol-block order.
        if aligned_timestamps is not None and len(aligned_timestamps) == len(y_proba):
            ts_dt = pd.to_datetime(aligned_timestamps, utc=True, errors="coerce")
            if not ts_dt.isna().all():
                ts_int = np.asarray(ts_dt.view("int64"), dtype=np.int64)
                sort_idx = np.argsort(ts_int, kind="stable")
                y_true = y_true[sort_idx]
                y_pred = y_pred[sort_idx]
                y_proba = y_proba[sort_idx]
                if realized_forward_returns is not None and len(realized_forward_returns) == len(
                    sort_idx
                ):
                    realized_forward_returns = realized_forward_returns[sort_idx]
                if event_net_returns is not None and len(event_net_returns) == len(sort_idx):
                    event_net_returns = event_net_returns[sort_idx]
                if event_directions is not None and len(event_directions) == len(sort_idx):
                    event_directions = event_directions[sort_idx]
                aligned_timestamps = aligned_timestamps[sort_idx]
                if aligned_symbols is not None and len(aligned_symbols) == len(sort_idx):
                    aligned_symbols = aligned_symbols[sort_idx]

        ranker_mode = self._is_ranker_model()
        if self._is_regression_model() or ranker_mode:
            y_pred_binary = (y_proba >= 0.5).astype(int)
            y_true_binary = (y_true > 0.0).astype(int)
        else:
            y_pred_binary = (
                (y_pred >= 0.5).astype(int)
                if not np.array_equal(y_pred, y_pred.astype(int))
                else y_pred
            )
            y_true_binary = (
                (y_true >= 0.5).astype(int)
                if not np.array_equal(y_true, y_true.astype(int))
                else y_true
            )

        regression_proxy = y_proba if (self._is_regression_model() or ranker_mode) else y_pred
        metrics = {
            "accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
            "precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
            "recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
            "f1": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
            "mse": float(mean_squared_error(y_true, regression_proxy)),
            "r2": float(r2_score(y_true, regression_proxy)) if len(set(y_true)) > 1 else 0.0,
            "brier_score": (0.0 if ranker_mode else float(np.mean((y_proba - y_true_binary) ** 2))),
        }

        if len(y_proba) > 1:
            net_returns, execution_details = self._compute_net_returns(
                y_true,
                y_proba,
                long_threshold=long_threshold,
                short_threshold=short_threshold,
                realized_forward_returns=realized_forward_returns,
                event_net_returns=event_net_returns,
                event_directions=event_directions,
                timestamps=aligned_timestamps,
                symbols=aligned_symbols,
                return_details=True,
            )
            positions = np.asarray(
                execution_details.get("positions", np.zeros(len(net_returns))), dtype=float
            )
            turnover_series = np.asarray(
                execution_details.get("turnover_series", np.zeros(len(net_returns))),
                dtype=float,
            )
            trade_mask = np.asarray(
                execution_details.get("trade_mask", np.abs(positions) > 0.0), dtype=bool
            )
            entry_mask = np.asarray(
                execution_details.get("entry_mask", trade_mask),
                dtype=bool,
            )
            portfolio_returns = np.asarray(
                execution_details.get("portfolio_returns", net_returns),
                dtype=float,
            )
            portfolio_turnover_series = np.asarray(
                execution_details.get("portfolio_turnover_series", turnover_series),
                dtype=float,
            )
            portfolio_trade_mask = np.asarray(
                execution_details.get(
                    "portfolio_trade_mask",
                    np.abs(portfolio_returns) > 0.0,
                ),
                dtype=bool,
            )
            trade_returns = portfolio_returns[portfolio_trade_mask]
            performance_returns = portfolio_returns if portfolio_returns.size > 0 else trade_returns

            turnover = float(np.mean(portfolio_turnover_series))
            trade_count = int(np.count_nonzero(entry_mask))
            active_signal_rate = float(np.mean(portfolio_trade_mask))
            std = float(np.std(performance_returns)) if performance_returns.size > 0 else 0.0
            annualization_periods = float(self._annualization_periods())
            sharpe = (
                float(np.mean(performance_returns) / std * np.sqrt(annualization_periods))
                if std > 1e-12
                else 0.0
            )
            max_dd = self._max_drawdown(performance_returns)
            trade_obs_count = int(len(trade_returns))
            min_obs_for_confident_sharpe = max(
                6,
                int(round(float(self._effective_trade_target(int(len(y_proba)))) * 0.80)),
            )
            sharpe_observation_confidence = float(
                np.clip(
                    float(trade_obs_count) / float(max(1, min_obs_for_confident_sharpe)),
                    0.10,
                    1.0,
                )
            )
            sharpe = float(sharpe * sharpe_observation_confidence)
            if trade_obs_count > 0:
                wins = int(np.count_nonzero(trade_returns > 0))
                prior_strength = float(np.clip(min_obs_for_confident_sharpe * 0.35, 2.0, 10.0))
                win_rate = float(
                    (float(wins) + (0.5 * prior_strength))
                    / (float(trade_obs_count) + prior_strength)
                )
            else:
                win_rate = 0.0
            annual_return = (
                float(np.mean(performance_returns) * annualization_periods)
                if performance_returns.size > 0
                else 0.0
            )
            calmar = annual_return / max_dd if max_dd > 1e-9 else annual_return
            cumulative_pnl = (
                float(np.sum(performance_returns)) if performance_returns.size > 0 else 0.0
            )
            loss_pnl = (
                float(np.sum(performance_returns[performance_returns < 0.0]))
                if performance_returns.size > 0
                else 0.0
            )
            if performance_returns.size > 0:
                cumulative_equity = np.cumsum(performance_returns)
                running_peak = np.maximum.accumulate(cumulative_equity)
                underwater_ratio = float(np.mean(cumulative_equity < (running_peak - 1e-12)))
            else:
                underwater_ratio = 0.0
            if performance_returns.size > 0:
                alpha = 0.05
                tail_cutoff = float(np.quantile(performance_returns, alpha))
                cvar = float(np.mean(performance_returns[performance_returns <= tail_cutoff]))
                tail_loss_cutoff = float(np.quantile(performance_returns, 0.10))
                tail_loss_pnl = float(
                    np.sum(performance_returns[performance_returns <= tail_loss_cutoff])
                )
                expected_shortfall = float(abs(cvar))
                skew = float(pd.Series(performance_returns).skew())
            else:
                cvar = 0.0
                tail_loss_pnl = 0.0
                expected_shortfall = 0.0
                skew = 0.0

            symbol_concentration_hhi = 0.0
            symbol_effective_count = 0.0
            entry_symbol_count = 0.0
            if aligned_symbols is not None and len(aligned_symbols) == len(y_proba):
                entry_symbols = np.asarray(aligned_symbols[entry_mask], dtype=object)
                if entry_symbols.size > 0:
                    symbol_counts = pd.Series(entry_symbols).value_counts(normalize=True)
                    if not symbol_counts.empty:
                        shares = symbol_counts.to_numpy(dtype=float)
                        symbol_concentration_hhi = float(np.sum(shares**2))
                        symbol_effective_count = float(1.0 / max(symbol_concentration_hhi, 1e-9))
                        entry_symbol_count = float(symbol_counts.shape[0])
            symbol_tail_summary = self._summarize_symbol_tail_metrics(
                net_returns=net_returns,
                trade_mask=trade_mask,
                symbols=aligned_symbols,
                evaluation_size=int(len(y_proba)),
            )

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
                symbol_concentration_hhi=symbol_concentration_hhi,
                equity_break=equity_break,
                evaluation_size=int(len(y_proba)),
                symbol_sharpe_p25=float(symbol_tail_summary.get("symbol_sharpe_p25", 0.0)),
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
                    "pnl": cumulative_pnl,
                    "loss_pnl": loss_pnl,
                    "tail_loss_pnl": tail_loss_pnl,
                    "underwater_ratio": underwater_ratio,
                    "trade_return_observations": float(len(trade_returns)),
                    "sharpe_observation_confidence": float(sharpe_observation_confidence),
                    "cvar_95": cvar,
                    "expected_shortfall": expected_shortfall,
                    "return_skew": skew,
                    "equity_break": equity_break,
                    "symbol_concentration_hhi": float(symbol_concentration_hhi),
                    "symbol_effective_count": float(symbol_effective_count),
                    "entry_symbol_count": float(entry_symbol_count),
                    "symbol_count_total": float(symbol_tail_summary["symbol_count_total"]),
                    "symbol_count_evaluated": float(symbol_tail_summary["symbol_count_evaluated"]),
                    "symbol_coverage_ratio": float(symbol_tail_summary["symbol_coverage_ratio"]),
                    "symbol_sharpe_p25": float(symbol_tail_summary["symbol_sharpe_p25"]),
                    "symbol_sharpe_median": float(symbol_tail_summary["symbol_sharpe_median"]),
                    "symbol_sharpe_worst": float(symbol_tail_summary["symbol_sharpe_worst"]),
                    "symbol_sharpe_negative_share": float(
                        symbol_tail_summary["symbol_sharpe_negative_share"]
                    ),
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
                    "objective_cvar_penalty": float(objective_components["objective_cvar_penalty"]),
                    "objective_tail_risk_penalty": float(
                        objective_components["objective_tail_risk_penalty"]
                    ),
                    "objective_symbol_concentration_penalty": float(
                        objective_components["objective_symbol_concentration_penalty"]
                    ),
                    "objective_symbol_tail_penalty": float(
                        objective_components["objective_symbol_tail_penalty"]
                    ),
                    "objective_skew_penalty": float(objective_components["objective_skew_penalty"]),
                    "objective_equity_break_penalty": float(
                        objective_components["objective_equity_break_penalty"]
                    ),
                    "objective_trade_target": float(objective_components["objective_trade_target"]),
                    "objective_expected_shortfall_cap": float(
                        objective_components["objective_expected_shortfall_cap"]
                    ),
                    "objective_symbol_tail_floor": float(
                        objective_components["objective_symbol_tail_floor"]
                    ),
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
                    "pnl": 0.0,
                    "loss_pnl": 0.0,
                    "tail_loss_pnl": 0.0,
                    "underwater_ratio": 0.0,
                    "trade_return_observations": 0.0,
                    "sharpe_observation_confidence": 0.0,
                    "cvar_95": 0.0,
                    "expected_shortfall": 0.0,
                    "return_skew": 0.0,
                    "equity_break": 0.0,
                    "symbol_concentration_hhi": 0.0,
                    "symbol_effective_count": 0.0,
                    "entry_symbol_count": 0.0,
                    "symbol_count_total": 0.0,
                    "symbol_count_evaluated": 0.0,
                    "symbol_coverage_ratio": 0.0,
                    "symbol_sharpe_p25": 0.0,
                    "symbol_sharpe_median": 0.0,
                    "symbol_sharpe_worst": 0.0,
                    "symbol_sharpe_negative_share": 0.0,
                    "objective_sharpe_component": 0.0,
                    "objective_drawdown_penalty": 0.0,
                    "objective_turnover_penalty": 0.0,
                    "objective_calibration_penalty": 0.0,
                    "objective_trade_activity_penalty": 0.0,
                    "objective_cvar_penalty": 0.0,
                    "objective_tail_risk_penalty": 0.0,
                    "objective_symbol_concentration_penalty": 0.0,
                    "objective_symbol_tail_penalty": 0.0,
                    "objective_symbol_tail_floor": 0.0,
                    "objective_skew_penalty": 0.0,
                    "objective_equity_break_penalty": 0.0,
                    "objective_trade_target": float(
                        self._effective_trade_target(int(len(y_proba)))
                    ),
                    "objective_expected_shortfall_cap": float(
                        self.config.objective_expected_shortfall_cap
                    ),
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

    def _summarize_symbol_tail_metrics(
        self,
        *,
        net_returns: np.ndarray,
        trade_mask: np.ndarray,
        symbols: np.ndarray | None,
        evaluation_size: int,
    ) -> dict[str, float]:
        """Summarize cross-symbol tail robustness from fold-level execution returns."""
        summary = {
            "symbol_count_total": 0.0,
            "symbol_count_evaluated": 0.0,
            "symbol_coverage_ratio": 0.0,
            "symbol_sharpe_p25": 0.0,
            "symbol_sharpe_median": 0.0,
            "symbol_sharpe_worst": 0.0,
            "symbol_sharpe_negative_share": 0.0,
        }
        if symbols is None:
            return summary

        aligned_symbols = np.asarray(symbols, dtype=object).reshape(-1)
        aligned_returns = np.asarray(net_returns, dtype=float).reshape(-1)
        aligned_trade_mask = np.asarray(trade_mask, dtype=bool).reshape(-1)
        if (
            aligned_symbols.size != aligned_returns.size
            or aligned_trade_mask.size != aligned_returns.size
            or aligned_returns.size == 0
        ):
            return summary

        unique_symbols = sorted(
            {str(value).strip() for value in aligned_symbols if str(value).strip()}
        )
        if not unique_symbols:
            return summary

        summary["symbol_count_total"] = float(len(unique_symbols))
        annualization_periods = float(self._annualization_periods())
        sample_floor = max(12, int(round(0.01 * float(max(1, evaluation_size)))))
        symbol_sharpes: list[float] = []
        for symbol_name in unique_symbols:
            symbol_mask = aligned_symbols == symbol_name
            symbol_rows = int(np.count_nonzero(symbol_mask))
            if symbol_rows < sample_floor:
                continue

            symbol_returns = aligned_returns[symbol_mask]
            performance_returns = symbol_returns
            std = float(np.std(performance_returns)) if performance_returns.size > 0 else 0.0
            sharpe = (
                float(np.mean(performance_returns) / std * np.sqrt(annualization_periods))
                if std > 1e-12
                else 0.0
            )
            trade_obs_count = int(np.count_nonzero(aligned_trade_mask[symbol_mask]))
            min_obs_for_confident_sharpe = max(
                6,
                int(round(float(self._effective_trade_target(symbol_rows)) * 0.80)),
            )
            confidence = float(
                np.clip(
                    float(trade_obs_count) / float(max(1, min_obs_for_confident_sharpe)),
                    0.10,
                    1.0,
                )
            )
            symbol_sharpes.append(float(sharpe * confidence))

        if not symbol_sharpes:
            return summary

        sharpe_arr = np.asarray(symbol_sharpes, dtype=float)
        summary["symbol_count_evaluated"] = float(sharpe_arr.size)
        summary["symbol_coverage_ratio"] = float(sharpe_arr.size / max(1, len(unique_symbols)))
        summary["symbol_sharpe_p25"] = float(np.quantile(sharpe_arr, 0.25))
        summary["symbol_sharpe_median"] = float(np.median(sharpe_arr))
        summary["symbol_sharpe_worst"] = float(np.min(sharpe_arr))
        summary["symbol_sharpe_negative_share"] = float(np.mean(sharpe_arr < 0.0))
        return summary

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
        symbol_concentration_hhi: float,
        equity_break: float,
        evaluation_size: int | None = None,
        symbol_sharpe_p25: float | None = None,
    ) -> dict[str, float]:
        """Compute auditable objective component breakdown for optimization and promotion."""
        sharpe_component = float(self.config.objective_weight_sharpe * sharpe)
        drawdown_penalty = float(-self.config.objective_weight_drawdown * max_drawdown)
        turnover_penalty = float(-self.config.objective_weight_turnover * turnover)
        calibration_weight = (
            0.0 if self._is_ranker_model() else self.config.objective_weight_calibration
        )
        calibration_penalty = float(-calibration_weight * brier_score)
        cvar_penalty = float(-self.config.objective_weight_cvar * max(0.0, expected_shortfall))
        expected_shortfall_cap = float(max(1e-6, self.config.objective_expected_shortfall_cap))
        tail_risk_excess = float(
            max(0.0, (float(expected_shortfall) - expected_shortfall_cap) / expected_shortfall_cap)
        )
        tail_risk_penalty = float(
            -self.config.objective_weight_tail_risk * min(2.0, tail_risk_excess)
        )
        concentration_excess = float(max(0.0, float(symbol_concentration_hhi) - 0.35))
        symbol_concentration_penalty = float(
            -self.config.objective_weight_symbol_concentration
            * min(1.5, concentration_excess / 0.35)
        )
        symbol_tail_penalty = 0.0
        symbol_tail_floor = float(min(0.0, self.config.min_holdout_symbol_p25_sharpe))
        if symbol_sharpe_p25 is not None:
            tail_shortfall = max(0.0, symbol_tail_floor - float(symbol_sharpe_p25))
            tail_scale = max(0.25, abs(symbol_tail_floor) + 0.10)
            symbol_tail_penalty = float(
                -self.config.objective_weight_tail_risk * min(2.0, tail_shortfall / tail_scale)
            )
        raw_skew_penalty = float(-self.config.objective_weight_skew * max(0.0, -float(skew)))
        skew_penalty_cap = float(self.config.objective_skew_penalty_cap)
        if skew_penalty_cap > 0.0:
            skew_penalty = float(max(raw_skew_penalty, -skew_penalty_cap))
        else:
            skew_penalty = raw_skew_penalty
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
            + tail_risk_penalty
            + symbol_concentration_penalty
            + symbol_tail_penalty
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
            "objective_tail_risk_penalty": tail_risk_penalty,
            "objective_symbol_concentration_penalty": symbol_concentration_penalty,
            "objective_symbol_tail_penalty": symbol_tail_penalty,
            "objective_symbol_tail_floor": float(symbol_tail_floor),
            "objective_skew_penalty": skew_penalty,
            "objective_skew_penalty_raw": raw_skew_penalty,
            "objective_skew_penalty_cap": float(skew_penalty_cap),
            "objective_equity_break_penalty": equity_break_penalty,
            "objective_trade_target": float(min_trades_target),
            "objective_expected_shortfall_cap": float(expected_shortfall_cap),
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
        symbols: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Build execution-aware position profile from probabilities."""
        y_proba = np.asarray(y_proba, dtype=float)
        n_samples = len(y_proba)
        ranker_mode = self._is_ranker_model()
        if n_samples == 0:
            return {
                "base_long_threshold_series": np.array([], dtype=float),
                "base_short_threshold_series": np.array([], dtype=float),
                "long_threshold_series": np.array([], dtype=float),
                "short_threshold_series": np.array([], dtype=float),
                "raw_signals": np.array([], dtype=float),
                "positions": np.array([], dtype=float),
                "turnover_series": np.array([], dtype=float),
                "trade_mask": np.array([], dtype=bool),
                "entry_mask": np.array([], dtype=bool),
                "diagnostics": {
                    "base_threshold_hits": self._summarize_threshold_hits(
                        np.array([], dtype=float),
                        np.array([], dtype=float),
                        np.array([], dtype=float),
                    ),
                    "band_threshold_hits": self._summarize_threshold_hits(
                        np.array([], dtype=float),
                        np.array([], dtype=float),
                        np.array([], dtype=float),
                    ),
                    "final_signal_summary": self._summarize_signal_activity(
                        np.array([], dtype=float),
                    ),
                    "band_adjustment_summary": {
                        "rows": 0,
                        "dynamic_band_enabled": bool(self.config.dynamic_no_trade_band),
                        "target_activity_rate": 0.0,
                        "dynamic_relaxation_applied": False,
                        "dynamic_relaxation_level": 0.0,
                        "forced_tail_applied": False,
                        "emergency_relaxation_applied": False,
                        "dead_score_guard_triggered": False,
                        "cooldown_suppressed_count": 0,
                        "symbol_cap_suppressed_count": 0,
                        "mean_long_threshold_shift": 0.0,
                        "mean_short_threshold_shift": 0.0,
                        "max_long_threshold_shift": 0.0,
                        "max_short_threshold_shift": 0.0,
                        "adaptive_band_mean": 0.0,
                        "adaptive_band_max": 0.0,
                        "adaptive_band_nonzero_count": 0,
                        "base_active_rate": 0.0,
                        "band_active_rate": 0.0,
                        "final_signal_rate": 0.0,
                        "suppressed_signal_count": 0,
                    },
                },
            }

        base_long_series = np.full(n_samples, float(long_threshold), dtype=float)
        base_short_series = np.full(n_samples, float(short_threshold), dtype=float)
        long_series = base_long_series.copy()
        short_series = base_short_series.copy()
        adaptive_band = np.zeros(n_samples, dtype=float)
        target_activity_rate = 0.0
        dynamic_relaxation_level = 0.0
        forced_tail_applied = False
        emergency_relaxation_applied = False
        probability_summary = self._summarize_probability_distribution(y_proba)
        dead_score_guard_triggered = self._is_dead_probability_summary(probability_summary)
        if self.config.dynamic_no_trade_band:
            uncertainty = 1.0 - (2.0 * np.abs(y_proba - 0.5))
            uncertainty = np.clip(uncertainty, 0.0, 1.0)
            proba_velocity = np.abs(np.diff(y_proba, prepend=y_proba[0]))
            proba_rolling_vol = (
                pd.Series(proba_velocity)
                .rolling(max(8, self.config.label_volatility_lookback), min_periods=3)
                .mean()
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            vol_anchor = (
                float(np.median(proba_rolling_vol[proba_rolling_vol > 0]))
                if np.any(proba_rolling_vol > 0)
                else 0.0
            )
            if vol_anchor > 0.0:
                vol_scale = np.clip((proba_rolling_vol / vol_anchor) - 1.0, 0.0, 2.0)
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

        if (
            self.config.dynamic_no_trade_band
            and adaptive_band.size > 0
            and not dead_score_guard_triggered
        ):
            target_activity_rate = min(
                0.18 if ranker_mode else 0.30,
                max(
                    0.01 if ranker_mode else 0.02,
                    float(self._effective_trade_target(n_samples)) / float(n_samples),
                ),
            )
            observed_activity = float(np.mean(raw_signals != 0.0))
            if observed_activity < (target_activity_rate * 0.50):
                shortfall = max(0.0, (target_activity_rate * 0.50) - observed_activity)
                relaxation = float(
                    np.clip(shortfall / max(target_activity_rate * 0.50, 1e-9), 0.0, 1.0)
                )
                dynamic_relaxation_level = float(max(dynamic_relaxation_level, float(relaxation)))
                relaxed_band = adaptive_band * (1.0 - 0.75 * relaxation)
                long_series = np.clip(float(long_threshold) + relaxed_band, 0.51, 0.995)
                short_series = np.clip(float(short_threshold) - relaxed_band, 0.005, 0.49)
                raw_signals = _generate_signals(long_series, short_series)
                observed_activity = float(np.mean(raw_signals != 0.0))

            if observed_activity < (target_activity_rate * 0.50):
                forced_tail_applied = True
                forced_tail = float(
                    min(
                        0.18 if ranker_mode else 0.35,
                        max(
                            0.04 if ranker_mode else 0.06,
                            target_activity_rate * (1.10 if ranker_mode else 1.25),
                        ),
                    )
                )
                forced_long = float(np.quantile(y_proba, 1.0 - forced_tail))
                forced_short = float(np.quantile(y_proba, forced_tail))
                forced_long = float(
                    np.clip(
                        forced_long,
                        0.56 if ranker_mode else 0.53,
                        0.68 if ranker_mode else 0.70,
                    )
                )
                forced_short = float(
                    np.clip(
                        forced_short,
                        0.32 if ranker_mode else 0.30,
                        0.44 if ranker_mode else 0.47,
                    )
                )
                long_series = np.minimum(long_series, np.full(n_samples, forced_long, dtype=float))
                short_series = np.maximum(
                    short_series, np.full(n_samples, forced_short, dtype=float)
                )
                raw_signals = _generate_signals(long_series, short_series)
                observed_activity = float(np.mean(raw_signals != 0.0))

            if observed_activity < (target_activity_rate * 0.50):
                emergency_relaxation_applied = True
                center = float(np.clip(np.median(y_proba), 0.45, 0.55))
                dispersion = float(np.std(y_proba))
                center_band = float(
                    np.clip(
                        max(
                            0.015 if ranker_mode else 0.02,
                            dispersion * (0.60 if ranker_mode else 0.75),
                        ),
                        0.015 if ranker_mode else 0.02,
                        0.06 if ranker_mode else 0.10,
                    )
                )
                emergency_long = float(
                    np.clip(
                        center + center_band,
                        0.53 if ranker_mode else 0.51,
                        0.63 if ranker_mode else 0.66,
                    )
                )
                emergency_short = float(
                    np.clip(
                        center - center_band,
                        0.37 if ranker_mode else 0.34,
                        0.47 if ranker_mode else 0.49,
                    )
                )
                long_series = np.minimum(
                    long_series, np.full(n_samples, emergency_long, dtype=float)
                )
                short_series = np.maximum(
                    short_series, np.full(n_samples, emergency_short, dtype=float)
                )
                raw_signals = _generate_signals(long_series, short_series)

            # Regime-conditioned biasing from model confidence trend (leakage-safe).
            if (not ranker_mode) and n_samples >= 12:
                centered_proba = y_proba - 0.5
                drift_window = max(8, int(self.config.label_volatility_lookback))
                rolling_mean = (
                    pd.Series(centered_proba)
                    .rolling(drift_window, min_periods=4)
                    .mean()
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                rolling_std = (
                    pd.Series(centered_proba)
                    .rolling(drift_window, min_periods=4)
                    .std()
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                normalized_drift = np.tanh(
                    rolling_mean / np.clip(rolling_std * 1.8 + 0.02, 0.02, None)
                )
                bull_bias = np.clip(normalized_drift, 0.0, 1.0)
                bear_bias = np.clip(-normalized_drift, 0.0, 1.0)

                long_series = np.clip(
                    long_series - (0.022 * bull_bias) + (0.018 * bear_bias), 0.51, 0.995
                )
                short_series = np.clip(
                    short_series - (0.018 * bull_bias) + (0.022 * bear_bias), 0.005, 0.49
                )
                strong_bull = bull_bias >= 0.65
                strong_bear = bear_bias >= 0.65
                if np.any(strong_bull):
                    short_series[strong_bull] = np.minimum(short_series[strong_bull], 0.24)
                    long_series[strong_bull] = np.minimum(long_series[strong_bull], 0.64)
                if np.any(strong_bear):
                    long_series[strong_bear] = np.maximum(long_series[strong_bear], 0.76)
                    short_series[strong_bear] = np.maximum(short_series[strong_bear], 0.40)
                min_gap = 0.02
                gap_mask = (long_series - short_series) < min_gap
                if np.any(gap_mask):
                    midpoint = (long_series + short_series) / 2.0
                    long_series[gap_mask] = np.clip(
                        midpoint[gap_mask] + (0.5 * min_gap), 0.51, 0.995
                    )
                    short_series[gap_mask] = np.clip(
                        midpoint[gap_mask] - (0.5 * min_gap), 0.005, 0.49
                    )

                raw_signals = _generate_signals(long_series, short_series)

        band_threshold_hits = self._summarize_threshold_hits(y_proba, long_series, short_series)

        # Execution cooldown suppresses immediate direction flips in noisy regions.
        cooldown_bars = int(self.config.execution_cooldown_bars)
        cooldown_suppressed_count = 0
        if cooldown_bars > 0 and n_samples > 1:
            pre_cooldown_signals = raw_signals.copy()
            filtered_signals = pre_cooldown_signals.copy()
            last_entry_idx = -(10**9)
            last_direction = 0.0
            for idx, signal in enumerate(pre_cooldown_signals):
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
            cooldown_suppressed_count = int(
                np.count_nonzero(
                    (np.abs(pre_cooldown_signals) > 1e-8) & (np.abs(raw_signals) <= 1e-8)
                )
            )

        # Symbol diversification guardrail: thin weakest entries from dominant symbols
        # to reduce concentration risk while preserving strongest edges.
        symbol_cap_suppressed_count = 0
        if symbols is not None:
            symbols_arr = np.asarray(symbols, dtype=object).reshape(-1)
            if symbols_arr.size == n_samples:
                unique_symbols = pd.unique(pd.Series(symbols_arr.astype(str)))
                unique_count = int(len(unique_symbols))
                if unique_count >= 2:
                    pre_symbol_cap_signals = raw_signals.copy()
                    min_balanced_share = float(1.0 / max(1, unique_count))
                    symbol_share_cap = float(
                        max(
                            self.config.execution_max_symbol_entry_share,
                            min(0.95, min_balanced_share + 0.12),
                        )
                    )
                    adjusted_signals = raw_signals.copy()
                    for _ in range(3):
                        provisional_trade_mask = np.abs(adjusted_signals) > 1e-8
                        provisional_prev = np.concatenate([[0.0], adjusted_signals[:-1]])
                        provisional_entry_mask = provisional_trade_mask & (
                            (np.abs(provisional_prev) <= 1e-8)
                            | (np.sign(adjusted_signals) != np.sign(provisional_prev))
                        )
                        entry_idx = np.flatnonzero(provisional_entry_mask)
                        if entry_idx.size <= 0:
                            break
                        entry_symbols = pd.Series(symbols_arr[entry_idx].astype(str))
                        share = entry_symbols.value_counts(normalize=True)
                        if share.empty or float(share.max()) <= (symbol_share_cap + 1e-9):
                            break
                        total_entries = int(entry_idx.size)
                        target_max_entries = max(
                            1,
                            int(np.floor(float(total_entries) * symbol_share_cap)),
                        )
                        changed = False
                        for dominant_symbol, dominant_count in share.mul(total_entries).items():
                            dominant_count_int = int(round(float(dominant_count)))
                            excess = dominant_count_int - target_max_entries
                            if excess <= 0:
                                continue
                            dominant_mask = symbols_arr[entry_idx].astype(str) == str(
                                dominant_symbol
                            )
                            dominant_entry_idx = entry_idx[dominant_mask]
                            if dominant_entry_idx.size <= 0:
                                continue
                            strengths = np.abs(y_proba[dominant_entry_idx] - 0.5)
                            drop_n = int(min(excess, dominant_entry_idx.size))
                            drop_order = np.argsort(strengths)[:drop_n]
                            adjusted_signals[dominant_entry_idx[drop_order]] = 0.0
                            changed = True
                        if not changed:
                            break
                    raw_signals = adjusted_signals
                    symbol_cap_suppressed_count = int(
                        np.count_nonzero(
                            (np.abs(pre_symbol_cap_signals) > 1e-8) & (np.abs(raw_signals) <= 1e-8)
                        )
                    )

        proba_proxy = np.abs(np.diff(y_proba, prepend=y_proba[0]))
        rolling_vol = (
            pd.Series(proba_proxy)
            .rolling(max(8, self.config.label_volatility_lookback), min_periods=3)
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
            (np.abs(prev_positions) <= 1e-8) | (np.sign(positions) != np.sign(prev_positions))
        )
        base_threshold_hits = self._summarize_threshold_hits(
            y_proba,
            base_long_series,
            base_short_series,
        )
        final_signal_summary = self._summarize_signal_activity(
            raw_signals,
            positions=positions,
            trade_mask=trade_mask,
            entry_mask=entry_mask,
        )
        band_adjustment_summary = {
            "rows": int(n_samples),
            "dynamic_band_enabled": bool(self.config.dynamic_no_trade_band),
            "target_activity_rate": float(target_activity_rate),
            "dynamic_relaxation_applied": bool(dynamic_relaxation_level > 0.0),
            "dynamic_relaxation_level": float(dynamic_relaxation_level),
            "forced_tail_applied": bool(forced_tail_applied),
            "emergency_relaxation_applied": bool(emergency_relaxation_applied),
            "dead_score_guard_triggered": bool(dead_score_guard_triggered),
            "cooldown_suppressed_count": int(cooldown_suppressed_count),
            "symbol_cap_suppressed_count": int(symbol_cap_suppressed_count),
            "mean_long_threshold_shift": float(np.mean(long_series - base_long_series)),
            "mean_short_threshold_shift": float(np.mean(base_short_series - short_series)),
            "max_long_threshold_shift": float(np.max(long_series - base_long_series)),
            "max_short_threshold_shift": float(np.max(base_short_series - short_series)),
            "adaptive_band_mean": float(np.mean(adaptive_band)) if adaptive_band.size > 0 else 0.0,
            "adaptive_band_max": float(np.max(adaptive_band)) if adaptive_band.size > 0 else 0.0,
            "adaptive_band_nonzero_count": int(np.count_nonzero(adaptive_band > 1e-12)),
            "base_active_rate": float(base_threshold_hits.get("active_rate", 0.0)),
            "band_active_rate": float(band_threshold_hits.get("active_rate", 0.0)),
            "final_signal_rate": float(final_signal_summary.get("active_signal_rate", 0.0)),
            "suppressed_signal_count": int(
                max(
                    0,
                    int(band_threshold_hits.get("active_count", 0))
                    - int(final_signal_summary.get("active_signal_count", 0)),
                )
            ),
        }

        return {
            "base_long_threshold_series": base_long_series.astype(float),
            "base_short_threshold_series": base_short_series.astype(float),
            "long_threshold_series": long_series.astype(float),
            "short_threshold_series": short_series.astype(float),
            "raw_signals": raw_signals.astype(float),
            "positions": positions.astype(float),
            "turnover_series": turnover_series.astype(float),
            "trade_mask": trade_mask.astype(bool),
            "entry_mask": entry_mask.astype(bool),
            "diagnostics": {
                "base_threshold_hits": base_threshold_hits,
                "band_threshold_hits": band_threshold_hits,
                "final_signal_summary": final_signal_summary,
                "band_adjustment_summary": band_adjustment_summary,
            },
        }

    def _compute_single_stream_net_returns(
        self,
        *,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        long_threshold: float,
        short_threshold: float,
        realized_returns: np.ndarray | None,
        returns_are_cost_aware: bool,
        timestamps: np.ndarray | None,
        symbols: np.ndarray | None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Compute execution profile and net returns for a single stream."""
        execution_profile = self._build_execution_profile(
            y_proba=y_proba,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            realized_returns=realized_returns,
            timestamps=timestamps,
            symbols=symbols,
        )
        positions = np.asarray(execution_profile["positions"], dtype=float)
        turnover_series = np.asarray(execution_profile["turnover_series"], dtype=float)

        if realized_returns is not None:
            raw_alpha = positions * realized_returns
        else:
            # Conservative fallback when realized returns are unavailable.
            if self._is_regression_model():
                y_direction = np.where(y_true > 0.0, 1.0, -1.0)
            else:
                y_direction = np.where(y_true >= 0.5, 1.0, -1.0)
            raw_alpha = positions * y_direction * np.abs(y_proba - 0.5)

        trading_cost_rate = max(
            0.0005, float(self.config.to_trading_cost_model().execution_cost_rate)
        )
        trading_cost = (
            np.zeros_like(turnover_series, dtype=float)
            if returns_are_cost_aware
            else turnover_series * trading_cost_rate
        )
        net_returns = np.nan_to_num(
            (raw_alpha - trading_cost).astype(float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return net_returns, execution_profile

    @staticmethod
    def _aggregate_panel_portfolio_series(
        *,
        timestamps: np.ndarray | None,
        net_returns: np.ndarray,
        turnover_series: np.ndarray,
        trade_mask: np.ndarray,
        entry_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Aggregate row-level panel execution to timestamp-level portfolio series."""
        n_rows = int(len(net_returns))
        row_fallback = {
            "portfolio_returns": np.asarray(net_returns, dtype=float),
            "portfolio_turnover_series": np.asarray(turnover_series, dtype=float),
            "portfolio_trade_mask": np.asarray(trade_mask, dtype=bool),
            "portfolio_entry_mask": np.asarray(entry_mask, dtype=bool),
        }
        if n_rows <= 0 or timestamps is None:
            return row_fallback

        timestamp_arr = np.asarray(timestamps).reshape(-1)
        if timestamp_arr.size != n_rows:
            return row_fallback

        ts_dt = pd.to_datetime(timestamp_arr, utc=True, errors="coerce")
        valid_mask = np.asarray(~pd.isna(ts_dt), dtype=bool)
        if (not np.any(valid_mask)) or (not np.all(valid_mask)):
            return row_fallback

        ts_int = np.asarray(ts_dt.view("int64"), dtype=np.int64)
        sort_idx = np.argsort(ts_int, kind="stable")
        sorted_ts = ts_int[sort_idx]
        sorted_net = np.asarray(net_returns[sort_idx], dtype=float)
        sorted_turnover = np.asarray(turnover_series[sort_idx], dtype=float)
        sorted_trade = np.asarray(trade_mask[sort_idx], dtype=bool).astype(np.int8)
        sorted_entry = np.asarray(entry_mask[sort_idx], dtype=bool).astype(np.int8)

        unique_ts, start_idx, counts = np.unique(
            sorted_ts,
            return_index=True,
            return_counts=True,
        )
        del unique_ts  # Only aggregation outputs are required downstream.

        portfolio_returns = np.add.reduceat(sorted_net, start_idx) / np.clip(counts, 1, None)
        portfolio_turnover_series = np.add.reduceat(sorted_turnover, start_idx) / np.clip(
            counts, 1, None
        )
        portfolio_trade_mask = np.maximum.reduceat(sorted_trade, start_idx).astype(bool)
        portfolio_entry_mask = np.maximum.reduceat(sorted_entry, start_idx).astype(bool)

        return {
            "portfolio_returns": np.asarray(portfolio_returns, dtype=float),
            "portfolio_turnover_series": np.asarray(portfolio_turnover_series, dtype=float),
            "portfolio_trade_mask": np.asarray(portfolio_trade_mask, dtype=bool),
            "portfolio_entry_mask": np.asarray(portfolio_entry_mask, dtype=bool),
        }

    def _compute_net_returns(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        long_threshold: float = 0.55,
        short_threshold: float = 0.45,
        realized_forward_returns: np.ndarray | None = None,
        event_net_returns: np.ndarray | None = None,
        event_directions: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        symbols: np.ndarray | None = None,
        return_details: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Compute strategy net returns from model signals and realized returns."""
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        n_rows = int(len(y_proba))
        if n_rows <= 1:
            empty_returns = np.array([], dtype=float)
            if return_details:
                return empty_returns, {
                    "base_long_threshold_series": np.array([], dtype=float),
                    "base_short_threshold_series": np.array([], dtype=float),
                    "long_threshold_series": np.array([], dtype=float),
                    "short_threshold_series": np.array([], dtype=float),
                    "raw_signals": np.array([], dtype=float),
                    "positions": np.array([], dtype=float),
                    "turnover_series": np.array([], dtype=float),
                    "trade_mask": np.array([], dtype=bool),
                    "entry_mask": np.array([], dtype=bool),
                    "portfolio_returns": np.array([], dtype=float),
                    "portfolio_turnover_series": np.array([], dtype=float),
                    "portfolio_trade_mask": np.array([], dtype=bool),
                    "portfolio_entry_mask": np.array([], dtype=bool),
                }
            return empty_returns

        short_threshold = float(np.clip(short_threshold, 0.01, 0.49))
        long_threshold = float(np.clip(long_threshold, 0.51, 0.99))
        if long_threshold <= short_threshold:
            long_threshold = 0.55
            short_threshold = 0.45

        realized_forward: np.ndarray | None = None
        if realized_forward_returns is not None:
            realized_forward = self._align_probabilities(
                np.asarray(realized_forward_returns, dtype=float),
                n_rows,
                fill_value=0.0,
            )
        cost_aware_returns: np.ndarray | None = None
        if event_net_returns is not None:
            cost_aware_returns = self._align_probabilities(
                np.asarray(event_net_returns, dtype=float),
                n_rows,
                fill_value=0.0,
            )
        aligned_event_directions: np.ndarray | None = None
        if event_directions is not None:
            aligned_event_directions = self._align_probabilities(
                np.asarray(event_directions, dtype=float),
                n_rows,
                fill_value=0.0,
            )

        realized: np.ndarray | None = realized_forward
        returns_are_cost_aware = False
        if cost_aware_returns is not None:
            realized = cost_aware_returns.copy()
            if aligned_event_directions is not None and len(aligned_event_directions) == n_rows:
                direction = np.sign(aligned_event_directions)
                non_flat = np.abs(direction) > 1e-8
                realized[non_flat] = cost_aware_returns[non_flat] * direction[non_flat]
            returns_are_cost_aware = True

        timestamp_arr: np.ndarray | None = None
        if timestamps is not None:
            candidate_timestamps = np.asarray(timestamps).reshape(-1)
            if candidate_timestamps.size == n_rows:
                timestamp_arr = candidate_timestamps

        symbols_arr: np.ndarray | None = None
        if symbols is not None:
            candidate_symbols = np.asarray(symbols, dtype=object).reshape(-1)
            if candidate_symbols.size == n_rows:
                symbols_arr = candidate_symbols

        unique_symbol_count = (
            int(pd.Series(symbols_arr.astype(str)).nunique())
            if symbols_arr is not None and symbols_arr.size == n_rows
            else 0
        )

        if unique_symbol_count >= 2:
            row_net_returns = np.zeros(n_rows, dtype=float)
            row_base_long_threshold_series = np.full(n_rows, float(long_threshold), dtype=float)
            row_base_short_threshold_series = np.full(n_rows, float(short_threshold), dtype=float)
            row_long_threshold_series = np.full(n_rows, float(long_threshold), dtype=float)
            row_short_threshold_series = np.full(n_rows, float(short_threshold), dtype=float)
            row_raw_signals = np.zeros(n_rows, dtype=float)
            row_positions = np.zeros(n_rows, dtype=float)
            row_turnover = np.zeros(n_rows, dtype=float)
            row_trade_mask = np.zeros(n_rows, dtype=bool)
            row_entry_mask = np.zeros(n_rows, dtype=bool)

            sort_key: np.ndarray | None = None
            if timestamp_arr is not None:
                ts_dt = pd.to_datetime(timestamp_arr, utc=True, errors="coerce")
                if not np.asarray(pd.isna(ts_dt), dtype=bool).any():
                    sort_key = np.asarray(ts_dt.view("int64"), dtype=np.int64)

            symbol_labels = symbols_arr.astype(str)
            for symbol_name in pd.unique(pd.Series(symbol_labels)):
                symbol_idx = np.flatnonzero(symbol_labels == str(symbol_name))
                if symbol_idx.size <= 0:
                    continue
                if sort_key is not None and symbol_idx.size > 1:
                    local_order = np.argsort(sort_key[symbol_idx], kind="stable")
                    symbol_idx = symbol_idx[local_order]

                symbol_realized = (
                    np.asarray(realized[symbol_idx], dtype=float)
                    if realized is not None and len(realized) == n_rows
                    else None
                )
                symbol_timestamps = (
                    np.asarray(timestamp_arr[symbol_idx], dtype=object)
                    if timestamp_arr is not None and len(timestamp_arr) == n_rows
                    else None
                )
                symbol_net, symbol_details = self._compute_single_stream_net_returns(
                    y_true=np.asarray(y_true[symbol_idx], dtype=float),
                    y_proba=np.asarray(y_proba[symbol_idx], dtype=float),
                    long_threshold=long_threshold,
                    short_threshold=short_threshold,
                    realized_returns=symbol_realized,
                    returns_are_cost_aware=returns_are_cost_aware,
                    timestamps=symbol_timestamps,
                    symbols=None,
                )
                row_net_returns[symbol_idx] = np.asarray(symbol_net, dtype=float)
                row_base_long_threshold_series[symbol_idx] = np.asarray(
                    symbol_details.get(
                        "base_long_threshold_series",
                        np.full(len(symbol_idx), float(long_threshold), dtype=float),
                    ),
                    dtype=float,
                )
                row_base_short_threshold_series[symbol_idx] = np.asarray(
                    symbol_details.get(
                        "base_short_threshold_series",
                        np.full(len(symbol_idx), float(short_threshold), dtype=float),
                    ),
                    dtype=float,
                )
                row_long_threshold_series[symbol_idx] = np.asarray(
                    symbol_details.get(
                        "long_threshold_series",
                        np.full(len(symbol_idx), float(long_threshold), dtype=float),
                    ),
                    dtype=float,
                )
                row_short_threshold_series[symbol_idx] = np.asarray(
                    symbol_details.get(
                        "short_threshold_series",
                        np.full(len(symbol_idx), float(short_threshold), dtype=float),
                    ),
                    dtype=float,
                )
                row_raw_signals[symbol_idx] = np.asarray(
                    symbol_details.get("raw_signals", np.zeros(len(symbol_idx), dtype=float)),
                    dtype=float,
                )
                row_positions[symbol_idx] = np.asarray(
                    symbol_details.get("positions", np.zeros(len(symbol_idx))),
                    dtype=float,
                )
                row_turnover[symbol_idx] = np.asarray(
                    symbol_details.get("turnover_series", np.zeros(len(symbol_idx))),
                    dtype=float,
                )
                row_trade_mask[symbol_idx] = np.asarray(
                    symbol_details.get("trade_mask", np.zeros(len(symbol_idx), dtype=bool)),
                    dtype=bool,
                )
                row_entry_mask[symbol_idx] = np.asarray(
                    symbol_details.get("entry_mask", np.zeros(len(symbol_idx), dtype=bool)),
                    dtype=bool,
                )

            execution_profile: dict[str, Any] = {
                "base_long_threshold_series": row_base_long_threshold_series,
                "base_short_threshold_series": row_base_short_threshold_series,
                "long_threshold_series": row_long_threshold_series,
                "short_threshold_series": row_short_threshold_series,
                "raw_signals": row_raw_signals,
                "positions": row_positions,
                "turnover_series": row_turnover,
                "trade_mask": row_trade_mask,
                "entry_mask": row_entry_mask,
            }
            execution_profile.update(
                self._aggregate_panel_portfolio_series(
                    timestamps=timestamp_arr,
                    net_returns=row_net_returns,
                    turnover_series=row_turnover,
                    trade_mask=row_trade_mask,
                    entry_mask=row_entry_mask,
                )
            )

            if return_details:
                return row_net_returns, execution_profile
            return row_net_returns

        net_returns, execution_profile = self._compute_single_stream_net_returns(
            y_true=np.asarray(y_true, dtype=float),
            y_proba=np.asarray(y_proba, dtype=float),
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            realized_returns=realized,
            returns_are_cost_aware=returns_are_cost_aware,
            timestamps=timestamp_arr,
            symbols=symbols_arr,
        )
        execution_profile.update(
            {
                "portfolio_returns": np.asarray(net_returns, dtype=float),
                "portfolio_turnover_series": np.asarray(
                    execution_profile.get("turnover_series", np.zeros_like(net_returns)),
                    dtype=float,
                ),
                "portfolio_trade_mask": np.asarray(
                    execution_profile.get(
                        "trade_mask",
                        np.zeros(len(net_returns), dtype=bool),
                    ),
                    dtype=bool,
                ),
                "portfolio_entry_mask": np.asarray(
                    execution_profile.get(
                        "entry_mask",
                        np.zeros(len(net_returns), dtype=bool),
                    ),
                    dtype=bool,
                ),
            }
        )
        if return_details:
            return np.asarray(net_returns, dtype=float), execution_profile
        return np.asarray(net_returns, dtype=float)

    def _select_final_model(self, models: list, fold_results: list[dict]) -> None:
        """Select final model from CV folds."""

        def _fold_trade_target(result: dict[str, Any]) -> float:
            test_size = int(result.get("test_size", 0))
            return float(self._effective_trade_target(test_size if test_size > 0 else None))

        def _fold_selection_score(result: dict[str, Any]) -> float:
            raw_score = float(result.get("risk_adjusted_score", result.get("sharpe", 0.0)))
            reliability = self._fold_reliability_weight(result, int(result.get("test_size", 0)))
            return float(raw_score * (0.20 + (0.80 * reliability)))

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
            key=lambda i: _fold_selection_score(fold_results[i]),
        )
        self.model = models[best_idx]
        self.training_metrics["selected_cv_fold"] = float(best_idx + 1)
        self.training_metrics["selected_cv_fold_score"] = _fold_selection_score(
            fold_results[best_idx]
        )
        self.training_metrics["selected_cv_fold_reliability"] = self._fold_reliability_weight(
            fold_results[best_idx],
            int(fold_results[best_idx].get("test_size", 0)),
        )
        self.logger.info(f"Selected model from fold {best_idx + 1}")

    def _fit_model_full_dataset(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray | None,
        warm_start_model: Any | None = None,
    ) -> None:
        """Fit final production model on full dataset with robust kwargs handling."""
        full_timestamps = (
            np.asarray(self.timestamps, dtype="datetime64[ns]")
            if self.timestamps is not None and len(self.timestamps) == len(X)
            else None
        )
        if self._is_ranker_model():
            if len(X) < 8:
                split_idx = max(2, len(X) - 2)
                gap = 0
            else:
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
                X_val=X[split_idx + gap :],
                y_val=y[split_idx + gap :],
                sample_weights=train_weights,
                train_timestamps=(
                    full_timestamps[:split_idx] if full_timestamps is not None else None
                ),
                val_timestamps=(
                    full_timestamps[split_idx + gap :] if full_timestamps is not None else None
                ),
                warm_start_model=warm_start_model,
            )
            return

        model_module = str(getattr(model.__class__, "__module__", ""))
        fit_feature_names: list[str] = []
        if model_module.startswith("quant_trading_system.models"):
            if self.feature_names and len(self.feature_names) == int(X.shape[1]):
                fit_feature_names = [str(name) for name in self.feature_names]
            else:
                fit_feature_names = [f"f{i}" for i in range(int(X.shape[1]))]

        candidate_calls: list[dict[str, Any]] = []
        if sample_weights is not None:
            candidate_calls.extend(
                [
                    {"sample_weights": sample_weights},
                    {"sample_weight": sample_weights},
                ]
            )
        candidate_calls.append({})
        if warm_start_model is not None:
            warm_calls: list[dict[str, Any]] = []
            for kwargs in candidate_calls:
                warm_kwargs = dict(kwargs)
                warm_kwargs["warm_start_model"] = warm_start_model
                warm_calls.append(warm_kwargs)
                if self.config.model_type in {"xgboost", "xgboost_regressor"}:
                    xgb_kwargs = dict(kwargs)
                    xgb_kwargs["xgb_model"] = warm_start_model
                    warm_calls.append(xgb_kwargs)
                elif self._is_lightgbm_family_model():
                    init_kwargs = dict(kwargs)
                    init_kwargs["init_model"] = warm_start_model
                    warm_calls.append(init_kwargs)
            candidate_calls = warm_calls + candidate_calls

        last_error: Exception | None = None
        for kwargs in candidate_calls:
            call_kwargs = dict(kwargs)
            if fit_feature_names:
                call_kwargs.setdefault("feature_names", fit_feature_names)
            try:
                model.fit(X, y, **call_kwargs)
                if warm_start_model is not None and any(
                    key in call_kwargs for key in ("warm_start_model", "xgb_model", "init_model")
                ):
                    self.training_metrics["warm_start_applied"] = True
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
            X_val=X[split_idx + gap :],
            y_val=y[split_idx + gap :],
            sample_weights=train_weights,
            train_timestamps=(full_timestamps[:split_idx] if full_timestamps is not None else None),
            val_timestamps=(
                full_timestamps[split_idx + gap :] if full_timestamps is not None else None
            ),
            warm_start_model=warm_start_model,
        )

    def _fit_final_model_full_data(self) -> None:
        """Refit final model artifact on full training dataset for production use."""
        X = self.features.values
        y = self.labels.values
        if len(X) <= 4:
            self.logger.warning(
                "Dataset too small for final full-data refit; using selected fold model."
            )
            self._attach_probability_calibrator_to_model(self.model)
            final_refit_feature_count = float(len(self.feature_names or list(self.features.columns)))
            self.training_metrics["feature_selection_final_refit_binding"] = 0.0
            self.training_metrics["feature_selection_final_refit_initial_feature_count"] = (
                final_refit_feature_count
            )
            self.training_metrics["feature_selection_final_refit_selected_feature_count"] = (
                final_refit_feature_count
            )
            return

        weights = self.sample_weights if self.sample_weights is not None else None
        final_params = self._prepare_params_for_train_size(self.config.model_params, len(X))
        final_params = self._augment_params_for_train_labels(final_params, y)
        final_model = self._create_model(params=final_params)
        warm_start_model = None
        self.training_metrics["warm_start_applied"] = False
        if self.config.warm_start_model_path and self.config.model_type in {
            "xgboost",
            "xgboost_regressor",
            "lightgbm",
            "lightgbm_regressor",
        }:
            warm_start_model = self._load_warm_start_model()
        self._fit_model_full_dataset(final_model, X, y, weights, warm_start_model=warm_start_model)
        self.model = final_model
        if self._is_lightgbm_family_model():
            self._record_summary_metrics(
                "final_model_structure",
                self._summarize_lightgbm_model_structure(final_model),
            )
        self._attach_probability_calibrator_to_model(self.model)
        final_refit_feature_count = float(len(self.feature_names or list(self.features.columns)))
        self.training_metrics["feature_selection_final_refit_binding"] = 0.0
        self.training_metrics["feature_selection_final_refit_initial_feature_count"] = (
            final_refit_feature_count
        )
        self.training_metrics["feature_selection_final_refit_selected_feature_count"] = (
            final_refit_feature_count
        )
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
            "symbol_concentration_hhi",
            "symbol_effective_count",
            "entry_symbol_count",
            "turnover",
            "annual_return",
            "calmar",
            "trade_count",
            "active_signal_rate",
            "regime_shift",
            "trade_return_observations",
            "active_net_return_observations",
            "fold_reliability",
            "selection_score",
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
            "objective_tail_risk_penalty",
            "objective_symbol_concentration_penalty",
            "objective_symbol_tail_penalty",
            "objective_skew_penalty",
            "objective_equity_break_penalty",
            "objective_expected_shortfall_cap",
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
            "objective_tail_risk_penalty": float(
                self.training_metrics.get("mean_objective_tail_risk_penalty", 0.0)
            ),
            "objective_symbol_concentration_penalty": float(
                self.training_metrics.get("mean_objective_symbol_concentration_penalty", 0.0)
            ),
            "objective_symbol_tail_penalty": float(
                self.training_metrics.get("mean_objective_symbol_tail_penalty", 0.0)
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
        y_holdout_pred = np.asarray(self._predict_with_model(self.model, X_holdout))
        y_holdout_raw_proba = np.asarray(
            self._get_raw_predictions_proba(
                self.model,
                X_holdout,
                timestamps=self.holdout_timestamps,
            )
        )
        y_holdout_proba = np.asarray(
            self._get_predictions_proba(
                self.model,
                X_holdout,
                timestamps=self.holdout_timestamps,
            )
        )
        train_proba = np.asarray(
            self._get_predictions_proba(
                self.model,
                self.features.values,
                timestamps=self.timestamps,
            )
        )
        train_labels = self.labels.to_numpy(dtype=float) if self.labels is not None else None
        long_threshold, short_threshold = self._derive_signal_thresholds(
            train_proba,
            train_labels=train_labels,
            train_returns=self.primary_forward_returns,
            train_regimes=self.regimes,
        )
        holdout_raw_rows = int(len(y_holdout))
        holdout_eval_len = min(
            len(y_holdout),
            len(y_holdout_pred),
            len(y_holdout_raw_proba),
            len(y_holdout_proba),
        )
        if holdout_eval_len <= 1:
            self.logger.warning(
                "Insufficient aligned holdout predictions, skipping holdout metrics"
            )
            return

        def _align_holdout_array(
            values: np.ndarray | pd.Series | None,
            *,
            dtype: Any | None = None,
        ) -> np.ndarray | None:
            if values is None:
                return None
            arr = np.asarray(values, dtype=dtype).reshape(-1)
            if arr.size < holdout_eval_len:
                return None
            return arr[-holdout_eval_len:]

        if holdout_eval_len != holdout_raw_rows:
            y_holdout = y_holdout[-holdout_eval_len:]
            y_holdout_pred = y_holdout_pred[-holdout_eval_len:]
            y_holdout_raw_proba = y_holdout_raw_proba[-holdout_eval_len:]
            y_holdout_proba = y_holdout_proba[-holdout_eval_len:]
            self.logger.info("Aligned holdout sequence outputs to %d samples", holdout_eval_len)

        holdout_timestamps = _align_holdout_array(self.holdout_timestamps, dtype="datetime64[ns]")
        holdout_symbols_arr = _align_holdout_array(self.holdout_symbols, dtype=object)
        holdout_regimes = _align_holdout_array(self.holdout_regimes, dtype=str)
        holdout_primary_forward_returns = _align_holdout_array(
            self.holdout_primary_forward_returns,
            dtype=float,
        )
        holdout_cost_aware_event_returns = _align_holdout_array(
            self.holdout_cost_aware_event_returns,
            dtype=float,
        )
        holdout_primary_event_directions = _align_holdout_array(
            self.holdout_primary_event_directions,
            dtype=float,
        )
        if holdout_timestamps is not None and len(holdout_timestamps) == len(y_holdout):
            ts_dt = pd.to_datetime(holdout_timestamps, utc=True, errors="coerce")
            if not ts_dt.isna().all():
                sort_idx = np.argsort(
                    np.asarray(ts_dt.view("int64"), dtype=np.int64), kind="stable"
                )
                y_holdout = y_holdout[sort_idx]
                y_holdout_pred = y_holdout_pred[sort_idx]
                y_holdout_raw_proba = y_holdout_raw_proba[sort_idx]
                y_holdout_proba = y_holdout_proba[sort_idx]
                holdout_timestamps = holdout_timestamps[sort_idx]
                if holdout_symbols_arr is not None and len(holdout_symbols_arr) == len(sort_idx):
                    holdout_symbols_arr = holdout_symbols_arr[sort_idx]
                if holdout_regimes is not None and len(holdout_regimes) == len(sort_idx):
                    holdout_regimes = holdout_regimes[sort_idx]
                if holdout_primary_forward_returns is not None and len(
                    holdout_primary_forward_returns
                ) == len(sort_idx):
                    holdout_primary_forward_returns = holdout_primary_forward_returns[sort_idx]
                if holdout_cost_aware_event_returns is not None and len(
                    holdout_cost_aware_event_returns
                ) == len(sort_idx):
                    holdout_cost_aware_event_returns = holdout_cost_aware_event_returns[sort_idx]
                if holdout_primary_event_directions is not None and len(
                    holdout_primary_event_directions
                ) == len(sort_idx):
                    holdout_primary_event_directions = holdout_primary_event_directions[sort_idx]

        self._refresh_oof_probability_diagnostics(
            long_threshold=float(long_threshold),
            short_threshold=float(short_threshold),
        )
        self._record_summary_metrics(
            "holdout_raw_probability_distribution",
            self._summarize_probability_distribution(y_holdout_raw_proba),
        )
        self._record_summary_metrics(
            "holdout_calibrated_probability_distribution",
            self._summarize_probability_distribution(y_holdout_proba),
        )
        self._record_summary_metrics(
            "holdout_raw_to_calibrated_probability_shift",
            self._summarize_probability_shift(y_holdout_raw_proba, y_holdout_proba),
        )
        self._record_summary_metrics(
            "holdout_raw_threshold_hits",
            self._summarize_threshold_hits(y_holdout_raw_proba, long_threshold, short_threshold),
        )
        self._record_summary_metrics(
            "holdout_calibrated_threshold_hits",
            self._summarize_threshold_hits(y_holdout_proba, long_threshold, short_threshold),
        )

        holdout_signed_returns = self._resolve_signed_realized_returns(
            forward_returns=holdout_primary_forward_returns,
            event_net_returns=holdout_cost_aware_event_returns,
            event_directions=holdout_primary_event_directions,
        )
        holdout_base_signal = derive_base_signal(
            y_holdout_proba,
            long_threshold=float(long_threshold),
            short_threshold=float(short_threshold),
        )
        holdout_trade_returns = self._resolve_trade_realized_returns(
            holdout_signed_returns,
            holdout_base_signal,
        )
        _, holdout_execution_details = self._compute_net_returns(
            y_holdout,
            y_holdout_proba,
            long_threshold=float(long_threshold),
            short_threshold=float(short_threshold),
            realized_forward_returns=holdout_primary_forward_returns,
            event_net_returns=holdout_cost_aware_event_returns,
            event_directions=holdout_primary_event_directions,
            timestamps=holdout_timestamps,
            symbols=holdout_symbols_arr,
            return_details=True,
        )
        holdout_execution_long_series = np.asarray(
            holdout_execution_details.get(
                "long_threshold_series",
                np.full(len(y_holdout_proba), float(long_threshold), dtype=float),
            ),
            dtype=float,
        )
        holdout_execution_short_series = np.asarray(
            holdout_execution_details.get(
                "short_threshold_series",
                np.full(len(y_holdout_proba), float(short_threshold), dtype=float),
            ),
            dtype=float,
        )
        holdout_execution_signal = np.asarray(
            holdout_execution_details.get("raw_signals", holdout_base_signal),
            dtype=float,
        )
        holdout_execution_threshold_hits = self._summarize_threshold_hits(
            y_holdout_proba,
            holdout_execution_long_series,
            holdout_execution_short_series,
        )
        self._record_summary_metrics(
            "holdout_execution_threshold_hits",
            holdout_execution_threshold_hits,
        )
        holdout_execution_signal_activity = self._summarize_signal_activity(
            holdout_execution_signal,
            positions=holdout_execution_details.get("positions"),
            trade_mask=holdout_execution_details.get(
                "portfolio_trade_mask",
                holdout_execution_details.get("trade_mask"),
            ),
            entry_mask=holdout_execution_details.get(
                "portfolio_entry_mask",
                holdout_execution_details.get("entry_mask"),
            ),
        )
        self._record_summary_metrics(
            "holdout_execution_signal_activity",
            holdout_execution_signal_activity,
        )
        holdout_execution_base_long_series = np.asarray(
            holdout_execution_details.get(
                "base_long_threshold_series",
                np.full(len(y_holdout_proba), float(long_threshold), dtype=float),
            ),
            dtype=float,
        )
        holdout_execution_base_short_series = np.asarray(
            holdout_execution_details.get(
                "base_short_threshold_series",
                np.full(len(y_holdout_proba), float(short_threshold), dtype=float),
            ),
            dtype=float,
        )
        self._record_summary_metrics(
            "holdout_execution_band_adjustment_summary",
            {
                "rows": int(len(y_holdout_proba)),
                "mean_long_threshold_shift": float(
                    np.mean(holdout_execution_long_series - holdout_execution_base_long_series)
                ),
                "mean_short_threshold_shift": float(
                    np.mean(holdout_execution_base_short_series - holdout_execution_short_series)
                ),
                "max_long_threshold_shift": float(
                    np.max(holdout_execution_long_series - holdout_execution_base_long_series)
                ),
                "max_short_threshold_shift": float(
                    np.max(holdout_execution_base_short_series - holdout_execution_short_series)
                ),
                "base_active_rate": float(
                    self.training_metrics.get("holdout_calibrated_threshold_hits_active_rate", 0.0)
                ),
                "band_active_rate": float(holdout_execution_threshold_hits.get("active_rate", 0.0)),
                "final_signal_rate": float(
                    holdout_execution_signal_activity.get("active_signal_rate", 0.0)
                ),
                "suppressed_signal_count": int(
                    max(
                        0,
                        int(holdout_execution_threshold_hits.get("active_count", 0))
                        - int(holdout_execution_signal_activity.get("active_signal_count", 0)),
                    )
                ),
            },
        )
        if holdout_trade_returns is not None:
            holdout_base_funnel, holdout_base_by_symbol, holdout_base_by_regime = (
                self._build_signal_funnel_summary(
                    probabilities=y_holdout_proba,
                    signal_values=holdout_base_signal,
                    trade_returns=holdout_trade_returns,
                    symbols=holdout_symbols_arr,
                    regimes=holdout_regimes,
                )
            )
            self._record_signal_funnel_metrics(
                "holdout_base",
                holdout_base_funnel,
                by_symbol=holdout_base_by_symbol,
                by_regime=holdout_base_by_regime,
            )
            holdout_execution_trade_returns = self._resolve_trade_realized_returns(
                holdout_signed_returns,
                holdout_execution_signal,
            )
            if holdout_execution_trade_returns is not None:
                (
                    holdout_execution_funnel,
                    holdout_execution_by_symbol,
                    holdout_execution_by_regime,
                ) = self._build_signal_funnel_summary(
                    probabilities=y_holdout_proba,
                    signal_values=holdout_execution_signal,
                    trade_returns=holdout_execution_trade_returns,
                    symbols=holdout_symbols_arr,
                    regimes=holdout_regimes,
                )
                self._record_signal_funnel_metrics(
                    "holdout_execution",
                    holdout_execution_funnel,
                    by_symbol=holdout_execution_by_symbol,
                    by_regime=holdout_execution_by_regime,
                )

        holdout_metrics = self._calculate_fold_metrics(
            y_true=y_holdout,
            y_pred=y_holdout_pred,
            y_proba=y_holdout_proba,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            realized_forward_returns=holdout_primary_forward_returns,
            event_net_returns=holdout_cost_aware_event_returns,
            event_directions=holdout_primary_event_directions,
            timestamps=holdout_timestamps,
            symbols=holdout_symbols_arr,
        )

        holdout_regime_shift = self._regime_shift_score(
            self._regime_distribution(self.regimes),
            self._regime_distribution(holdout_regimes),
        )
        holdout_metrics["regime_shift"] = float(holdout_regime_shift)

        holdout_regime_metrics: dict[str, dict[str, float]] = {}
        holdout_worst_regime_sharpe = float(holdout_metrics.get("sharpe", 0.0))
        holdout_regime_sample_floor = max(30, int(round(0.05 * len(y_holdout))))
        if holdout_regimes is not None and len(holdout_regimes) == len(y_holdout):
            regime_arr = np.asarray(holdout_regimes, dtype=str).reshape(-1)
            unique_regimes = sorted({str(r).strip() for r in regime_arr if str(r).strip()})
            for regime_name in unique_regimes:
                regime_mask = regime_arr == regime_name
                regime_rows = int(np.count_nonzero(regime_mask))
                if regime_rows < holdout_regime_sample_floor:
                    continue
                regime_metrics = self._calculate_fold_metrics(
                    y_true=y_holdout[regime_mask],
                    y_pred=y_holdout_pred[regime_mask],
                    y_proba=y_holdout_proba[regime_mask],
                    long_threshold=long_threshold,
                    short_threshold=short_threshold,
                    realized_forward_returns=(
                        np.asarray(holdout_primary_forward_returns, dtype=float)[regime_mask]
                        if holdout_primary_forward_returns is not None
                        and len(holdout_primary_forward_returns) == len(y_holdout)
                        else None
                    ),
                    event_net_returns=(
                        np.asarray(holdout_cost_aware_event_returns, dtype=float)[regime_mask]
                        if holdout_cost_aware_event_returns is not None
                        and len(holdout_cost_aware_event_returns) == len(y_holdout)
                        else None
                    ),
                    event_directions=(
                        np.asarray(holdout_primary_event_directions, dtype=float)[regime_mask]
                        if holdout_primary_event_directions is not None
                        and len(holdout_primary_event_directions) == len(y_holdout)
                        else None
                    ),
                    timestamps=(
                        np.asarray(holdout_timestamps, dtype="datetime64[ns]")[regime_mask]
                        if holdout_timestamps is not None
                        and len(holdout_timestamps) == len(y_holdout)
                        else None
                    ),
                    symbols=(
                        holdout_symbols_arr[regime_mask]
                        if holdout_symbols_arr is not None
                        else None
                    ),
                )
                holdout_regime_metrics[str(regime_name)] = {
                    "rows": float(regime_rows),
                    "accuracy": float(regime_metrics.get("accuracy", 0.0)),
                    "sharpe": float(regime_metrics.get("sharpe", 0.0)),
                    "max_drawdown": float(regime_metrics.get("max_drawdown", 0.0)),
                    "trade_count": float(regime_metrics.get("trade_count", 0.0)),
                    "win_rate": float(regime_metrics.get("win_rate", 0.0)),
                    "turnover": float(regime_metrics.get("turnover", 0.0)),
                    "active_signal_rate": float(regime_metrics.get("active_signal_rate", 0.0)),
                    "annual_return": float(regime_metrics.get("annual_return", 0.0)),
                    "calmar": float(regime_metrics.get("calmar", 0.0)),
                    "pnl": float(regime_metrics.get("pnl", 0.0)),
                    "loss_pnl": float(regime_metrics.get("loss_pnl", 0.0)),
                    "tail_loss_pnl": float(regime_metrics.get("tail_loss_pnl", 0.0)),
                    "underwater_ratio": float(regime_metrics.get("underwater_ratio", 0.0)),
                    "cvar_95": float(regime_metrics.get("cvar_95", 0.0)),
                    "risk_adjusted_score": float(regime_metrics.get("risk_adjusted_score", -1e9)),
                    "sharpe_observation_confidence": float(
                        regime_metrics.get("sharpe_observation_confidence", 0.0)
                    ),
                    "symbol_concentration_hhi": float(
                        regime_metrics.get("symbol_concentration_hhi", 0.0)
                    ),
                }
            if holdout_regime_metrics:
                holdout_worst_regime_sharpe = float(
                    min(v.get("sharpe", 0.0) for v in holdout_regime_metrics.values())
                )

        holdout_symbol_metrics: dict[str, dict[str, float]] = {}
        holdout_symbol_count_total = 0.0
        holdout_symbol_count_evaluated = 0.0
        holdout_symbol_coverage_ratio = 0.0
        holdout_symbol_sharpe_median = float(holdout_metrics.get("sharpe", 0.0))
        holdout_symbol_sharpe_p25 = float(holdout_symbol_sharpe_median)
        holdout_symbol_sharpe_std = 0.0
        holdout_symbol_underwater_ratio = 0.0
        holdout_symbol_worst_sharpe = float(holdout_symbol_sharpe_median)
        holdout_symbol_sample_floor = max(20, int(round(0.01 * len(y_holdout))))
        if holdout_symbols_arr is not None:
            unique_symbols = sorted({str(s).strip() for s in holdout_symbols_arr if str(s).strip()})
            holdout_symbol_count_total = float(len(unique_symbols))
            for symbol_name in unique_symbols:
                symbol_mask = holdout_symbols_arr == symbol_name
                symbol_rows = int(np.count_nonzero(symbol_mask))
                if symbol_rows < holdout_symbol_sample_floor:
                    continue
                symbol_metrics = self._calculate_fold_metrics(
                    y_true=y_holdout[symbol_mask],
                    y_pred=y_holdout_pred[symbol_mask],
                    y_proba=y_holdout_proba[symbol_mask],
                    long_threshold=long_threshold,
                    short_threshold=short_threshold,
                    realized_forward_returns=(
                        np.asarray(holdout_primary_forward_returns, dtype=float)[symbol_mask]
                        if holdout_primary_forward_returns is not None
                        and len(holdout_primary_forward_returns) == len(y_holdout)
                        else None
                    ),
                    event_net_returns=(
                        np.asarray(holdout_cost_aware_event_returns, dtype=float)[symbol_mask]
                        if holdout_cost_aware_event_returns is not None
                        and len(holdout_cost_aware_event_returns) == len(y_holdout)
                        else None
                    ),
                    event_directions=(
                        np.asarray(holdout_primary_event_directions, dtype=float)[symbol_mask]
                        if holdout_primary_event_directions is not None
                        and len(holdout_primary_event_directions) == len(y_holdout)
                        else None
                    ),
                    timestamps=(
                        np.asarray(holdout_timestamps, dtype="datetime64[ns]")[symbol_mask]
                        if holdout_timestamps is not None
                        and len(holdout_timestamps) == len(y_holdout)
                        else None
                    ),
                    symbols=holdout_symbols_arr[symbol_mask],
                )
                holdout_symbol_metrics[str(symbol_name)] = {
                    "rows": float(symbol_rows),
                    "accuracy": float(symbol_metrics.get("accuracy", 0.0)),
                    "sharpe": float(symbol_metrics.get("sharpe", 0.0)),
                    "max_drawdown": float(symbol_metrics.get("max_drawdown", 0.0)),
                    "trade_count": float(symbol_metrics.get("trade_count", 0.0)),
                    "win_rate": float(symbol_metrics.get("win_rate", 0.0)),
                    "turnover": float(symbol_metrics.get("turnover", 0.0)),
                    "active_signal_rate": float(symbol_metrics.get("active_signal_rate", 0.0)),
                    "annual_return": float(symbol_metrics.get("annual_return", 0.0)),
                    "calmar": float(symbol_metrics.get("calmar", 0.0)),
                    "pnl": float(symbol_metrics.get("pnl", 0.0)),
                    "loss_pnl": float(symbol_metrics.get("loss_pnl", 0.0)),
                    "tail_loss_pnl": float(symbol_metrics.get("tail_loss_pnl", 0.0)),
                    "underwater_ratio": float(symbol_metrics.get("underwater_ratio", 0.0)),
                    "cvar_95": float(symbol_metrics.get("cvar_95", 0.0)),
                    "risk_adjusted_score": float(symbol_metrics.get("risk_adjusted_score", -1e9)),
                    "sharpe_observation_confidence": float(
                        symbol_metrics.get("sharpe_observation_confidence", 0.0)
                    ),
                }
            holdout_symbol_count_evaluated = float(len(holdout_symbol_metrics))
            if holdout_symbol_count_total > 0.0:
                holdout_symbol_coverage_ratio = float(
                    holdout_symbol_count_evaluated / holdout_symbol_count_total
                )
            if holdout_symbol_metrics:
                symbol_sharpes = np.asarray(
                    [m.get("sharpe", 0.0) for m in holdout_symbol_metrics.values()],
                    dtype=float,
                )
                holdout_symbol_sharpe_median = float(np.median(symbol_sharpes))
                holdout_symbol_sharpe_p25 = float(np.quantile(symbol_sharpes, 0.25))
                holdout_symbol_sharpe_std = float(np.std(symbol_sharpes))
                holdout_symbol_worst_sharpe = float(np.min(symbol_sharpes))
                holdout_symbol_underwater_ratio = float(np.mean(symbol_sharpes < 0.0))

        def _top_tail_contributors(
            metrics_by_group: dict[str, dict[str, float]],
            *,
            top_k: int = 8,
        ) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for group_name, group_metrics in metrics_by_group.items():
                tail_loss_pnl = float(group_metrics.get("tail_loss_pnl", 0.0))
                pnl = float(group_metrics.get("pnl", 0.0))
                if tail_loss_pnl >= 0.0 and pnl >= 0.0:
                    continue
                rows.append(
                    {
                        "name": str(group_name),
                        "tail_loss_pnl": tail_loss_pnl,
                        "pnl": pnl,
                        "loss_pnl": float(group_metrics.get("loss_pnl", 0.0)),
                        "underwater_ratio": float(group_metrics.get("underwater_ratio", 0.0)),
                        "trade_count": float(group_metrics.get("trade_count", 0.0)),
                        "sharpe": float(group_metrics.get("sharpe", 0.0)),
                        "max_drawdown": float(group_metrics.get("max_drawdown", 0.0)),
                    }
                )
            rows.sort(
                key=lambda item: (
                    float(item.get("tail_loss_pnl", 0.0)),
                    float(item.get("pnl", 0.0)),
                    -float(item.get("trade_count", 0.0)),
                )
            )
            return rows[:top_k]

        base_holdout_metrics = self._json_safe(dict(holdout_metrics))
        base_holdout_regime_metrics = self._json_safe(holdout_regime_metrics)
        base_holdout_symbol_metrics = self._json_safe(holdout_symbol_metrics)
        effective_holdout_group_source = "base"
        expected_edge_holdout_available = bool(
            self.training_metrics.get("expected_edge_policy_enabled", 0.0)
        ) and np.isfinite(
            float(
                self.training_metrics.get(
                    "expected_edge_holdout_policy_sharpe",
                    np.nan,
                )
            )
        )
        if expected_edge_holdout_available:
            effective_holdout_group_source = "expected_edge_policy"
            holdout_metrics = dict(holdout_metrics)
            for metric_key, training_key in (
                ("sharpe", "expected_edge_holdout_policy_sharpe"),
                ("trade_count", "expected_edge_holdout_policy_trade_count"),
                ("active_signal_rate", "expected_edge_holdout_policy_active_signal_rate"),
                ("max_drawdown", "expected_edge_holdout_policy_max_drawdown"),
                ("trade_return_observations", "expected_edge_holdout_policy_trade_return_observations"),
                (
                    "sharpe_observation_confidence",
                    "expected_edge_holdout_policy_sharpe_observation_confidence",
                ),
                ("annual_return", "expected_edge_holdout_policy_annual_return"),
                ("calmar", "expected_edge_holdout_policy_calmar"),
                ("underwater_ratio", "expected_edge_holdout_policy_underwater_ratio"),
                ("win_rate", "expected_edge_holdout_policy_selected_win_rate"),
                ("pnl", "expected_edge_holdout_policy_pnl"),
                ("loss_pnl", "expected_edge_holdout_policy_loss_pnl"),
                ("tail_loss_pnl", "expected_edge_holdout_policy_tail_loss_pnl"),
                ("cvar_95", "expected_edge_holdout_policy_cvar_95"),
                ("expected_shortfall", "expected_edge_holdout_policy_expected_shortfall"),
                ("return_skew", "expected_edge_holdout_policy_return_skew"),
                ("risk_adjusted_score", "expected_edge_holdout_policy_risk_adjusted_score"),
            ):
                if training_key in self.training_metrics:
                    holdout_metrics[metric_key] = float(self.training_metrics.get(training_key, 0.0))

            expected_edge_regime_metrics = self.training_metrics.get(
                "expected_edge_holdout_regime_metrics",
                {},
            )
            if isinstance(expected_edge_regime_metrics, dict):
                holdout_regime_metrics = self._json_safe(expected_edge_regime_metrics)
            holdout_worst_regime_sharpe = float(
                self.training_metrics.get(
                    "expected_edge_holdout_worst_regime_sharpe",
                    holdout_worst_regime_sharpe,
                )
            )

            expected_edge_symbol_metrics = self.training_metrics.get(
                "expected_edge_holdout_symbol_metrics",
                {},
            )
            if isinstance(expected_edge_symbol_metrics, dict):
                holdout_symbol_metrics = self._json_safe(expected_edge_symbol_metrics)
            holdout_symbol_count_total = float(
                self.training_metrics.get("expected_edge_holdout_symbol_count_total", 0.0)
            )
            holdout_symbol_count_evaluated = float(
                self.training_metrics.get("expected_edge_holdout_symbol_count_evaluated", 0.0)
            )
            holdout_symbol_coverage_ratio = float(
                self.training_metrics.get("expected_edge_holdout_symbol_coverage_ratio", 0.0)
            )
            holdout_symbol_sharpe_median = float(
                self.training_metrics.get("expected_edge_holdout_symbol_sharpe_median", 0.0)
            )
            holdout_symbol_sharpe_p25 = float(
                self.training_metrics.get("expected_edge_holdout_symbol_sharpe_p25", 0.0)
            )
            holdout_symbol_sharpe_std = float(
                self.training_metrics.get("expected_edge_holdout_symbol_sharpe_std", 0.0)
            )
            holdout_symbol_underwater_ratio = float(
                self.training_metrics.get("expected_edge_holdout_symbol_underwater_ratio", 0.0)
            )
            holdout_symbol_worst_sharpe = float(
                self.training_metrics.get("expected_edge_holdout_symbol_worst_sharpe", 0.0)
            )

        holdout_tail_loss_contributors_by_symbol = _top_tail_contributors(holdout_symbol_metrics)
        holdout_tail_loss_contributors_by_regime = _top_tail_contributors(holdout_regime_metrics)
        self.training_metrics["holdout_base_metrics"] = base_holdout_metrics
        self.training_metrics["holdout_base_regime_metrics"] = base_holdout_regime_metrics
        self.training_metrics["holdout_base_symbol_metrics"] = base_holdout_symbol_metrics
        self.training_metrics["effective_holdout_group_metric_source"] = effective_holdout_group_source

        self.training_metrics["holdout_worst_regime_sharpe"] = float(holdout_worst_regime_sharpe)
        self.training_metrics["holdout_regime_count_evaluated"] = float(len(holdout_regime_metrics))
        self.training_metrics["holdout_regime_metrics"] = holdout_regime_metrics
        self.training_metrics["holdout_tail_loss_contributors_by_regime"] = (
            holdout_tail_loss_contributors_by_regime
        )
        self.training_metrics["holdout_symbol_count_total"] = float(holdout_symbol_count_total)
        self.training_metrics["holdout_symbol_count_evaluated"] = float(
            holdout_symbol_count_evaluated
        )
        self.training_metrics["holdout_symbol_coverage_ratio"] = float(
            holdout_symbol_coverage_ratio
        )
        self.training_metrics["holdout_symbol_sharpe_median"] = float(holdout_symbol_sharpe_median)
        self.training_metrics["holdout_symbol_sharpe_p25"] = float(holdout_symbol_sharpe_p25)
        self.training_metrics["holdout_symbol_sharpe_std"] = float(holdout_symbol_sharpe_std)
        self.training_metrics["holdout_symbol_worst_sharpe"] = float(holdout_symbol_worst_sharpe)
        self.training_metrics["holdout_symbol_underwater_ratio"] = float(
            holdout_symbol_underwater_ratio
        )
        self.training_metrics["holdout_symbol_metrics"] = holdout_symbol_metrics
        self.training_metrics["holdout_tail_loss_contributors_by_symbol"] = (
            holdout_tail_loss_contributors_by_symbol
        )

        for key, value in holdout_metrics.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                self.training_metrics[f"holdout_{key}"] = float(value)

        self.training_metrics["holdout_rows_raw"] = float(holdout_raw_rows)
        self.training_metrics["holdout_rows_aligned"] = float(len(y_holdout))
        self.training_metrics["holdout_rows"] = float(len(y_holdout))
        self.training_metrics["holdout_long_threshold"] = float(long_threshold)
        self.training_metrics["holdout_short_threshold"] = float(short_threshold)
        side_policy_returns = self._resolve_signed_realized_returns(
            forward_returns=holdout_primary_forward_returns,
            event_net_returns=holdout_cost_aware_event_returns,
            event_directions=holdout_primary_event_directions,
        )
        holdout_side_policy = derive_asymmetric_signal_policy(
            y_holdout_proba,
            side_policy_returns,
            long_threshold=float(long_threshold),
            short_threshold=float(short_threshold),
        )
        self.training_metrics["holdout_side_policy"] = holdout_side_policy
        for side_name in ("long", "short"):
            policy = (
                holdout_side_policy.get(side_name, {})
                if isinstance(holdout_side_policy, dict)
                else {}
            )
            for metric_name in (
                "enabled",
                "signal_scale",
                "confidence_scale",
                "threshold_adjustment",
                "trade_count",
                "active_rate",
                "mean_signed_return",
                "win_rate",
                "score",
                "sample_confidence",
            ):
                value = policy.get(metric_name)
                if isinstance(value, bool):
                    self.training_metrics[f"holdout_{side_name}_{metric_name}"] = float(value)
                elif isinstance(value, (int, float, np.floating, np.integer)):
                    self.training_metrics[f"holdout_{side_name}_{metric_name}"] = float(value)

        self.logger.info(
            "Holdout metrics - Accuracy: %.4f, Sharpe: %.4f, WorstRegimeSharpe: %.4f, "
            "Drawdown: %.4f, RegimeShift: %.4f, SymbolCoverage: %.2f%%, UnderwaterSymbols: %.2f%%",
            self.training_metrics.get("holdout_accuracy", 0.0),
            self.training_metrics.get("holdout_sharpe", 0.0),
            self.training_metrics.get("holdout_worst_regime_sharpe", holdout_worst_regime_sharpe),
            self.training_metrics.get("holdout_max_drawdown", 0.0),
            self.training_metrics.get("holdout_regime_shift", holdout_regime_shift),
            self.training_metrics.get("holdout_symbol_coverage_ratio", 0.0) * 100.0,
            self.training_metrics.get("holdout_symbol_underwater_ratio", 0.0) * 100.0,
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
        effective_holdout = self._resolve_effective_holdout_metric_payload()
        holdout_available = float(self.training_metrics.get("holdout_rows", 0.0)) > 0.0
        holdout_sharpe = float(
            effective_holdout.get("sharpe", self.training_metrics.get("holdout_sharpe", -1.0))
        )
        effective_holdout_sharpe = (
            float(self._effective_holdout_sharpe_metric(holdout_sharpe))
            if holdout_available
            else float(self.config.min_holdout_sharpe)
        )
        self.training_metrics["effective_holdout_sharpe_gate_metric"] = effective_holdout_sharpe
        holdout_sharpe_consistency = (
            float(holdout_sharpe - float(self.training_metrics.get("mean_sharpe", holdout_sharpe)))
            if holdout_available
            else 0.0
        )
        holdout_sharpe_consistency_floor = -float(
            max(0.75, abs(float(self.config.min_holdout_sharpe)) + 0.15)
        )
        self.training_metrics["effective_holdout_sharpe_consistency_metric"] = (
            holdout_sharpe_consistency
        )
        holdout_trade_count_metric = float(effective_holdout.get("trade_count", np.nan))
        if not np.isfinite(holdout_trade_count_metric):
            holdout_trade_count_metric = float(
                self.training_metrics.get(
                    "holdout_trade_return_observations",
                    1.0 if holdout_available else 0.0,
                )
            )
        holdout_active_signal_rate_metric = float(
            effective_holdout.get("active_signal_rate", np.nan)
        )
        if not np.isfinite(holdout_active_signal_rate_metric):
            holdout_active_signal_rate_metric = (
                1.0 if holdout_available and holdout_trade_count_metric > 0.0 else 0.0
            )
        holdout_group_source = str(
            self.training_metrics.get(
                "effective_holdout_group_metric_source",
                effective_holdout.get("source", "base"),
            )
        ).strip() or "base"
        self.training_metrics["effective_holdout_group_metric_source"] = holdout_group_source
        holdout_worst_regime_sharpe = float(
            self.training_metrics.get(
                (
                    "expected_edge_holdout_worst_regime_sharpe"
                    if holdout_group_source == "expected_edge_policy"
                    else "holdout_worst_regime_sharpe"
                ),
                self.training_metrics.get("holdout_worst_regime_sharpe", holdout_sharpe),
            )
        )
        holdout_drawdown = float(effective_holdout.get("max_drawdown", 1.0))
        holdout_symbol_coverage = float(
            self.training_metrics.get(
                (
                    "expected_edge_holdout_symbol_coverage_ratio"
                    if holdout_group_source == "expected_edge_policy"
                    else "holdout_symbol_coverage_ratio"
                ),
                self.training_metrics.get(
                    "holdout_symbol_coverage_ratio",
                    self.config.min_holdout_symbol_coverage,
                ),
            )
        )
        holdout_symbol_sharpe_p25 = float(
            self.training_metrics.get(
                (
                    "expected_edge_holdout_symbol_sharpe_p25"
                    if holdout_group_source == "expected_edge_policy"
                    else "holdout_symbol_sharpe_p25"
                ),
                self.training_metrics.get(
                    "holdout_symbol_sharpe_p25",
                    self.config.min_holdout_symbol_p25_sharpe,
                ),
            )
        )
        holdout_symbol_underwater_ratio = float(
            self.training_metrics.get(
                (
                    "expected_edge_holdout_symbol_underwater_ratio"
                    if holdout_group_source == "expected_edge_policy"
                    else "holdout_symbol_underwater_ratio"
                ),
                self.training_metrics.get(
                    "holdout_symbol_underwater_ratio",
                    self.config.max_holdout_symbol_underwater_ratio,
                ),
            )
        )
        self.training_metrics["effective_holdout_symbol_coverage_metric"] = holdout_symbol_coverage
        self.training_metrics["effective_holdout_symbol_p25_sharpe_metric"] = (
            holdout_symbol_sharpe_p25
        )
        self.training_metrics["effective_holdout_symbol_underwater_ratio_metric"] = (
            holdout_symbol_underwater_ratio
        )
        effective_pbo_metric = float(
            self.training_metrics.get(
                "effective_pbo_gate_metric",
                self.training_metrics.get("pbo", 1.0),
            )
        )
        self.training_metrics["effective_pbo_gate_metric"] = effective_pbo_metric
        symbol_concentration_metric = float(
            max(
                self.training_metrics.get("mean_symbol_concentration_hhi", 0.0),
                self.training_metrics.get("holdout_symbol_concentration_hhi", 0.0),
            )
        )
        self.training_metrics["effective_symbol_concentration_gate_metric"] = (
            symbol_concentration_metric
        )
        effective_max_pbo = float(
            self.training_metrics.get("effective_max_pbo_gate", self.config.max_pbo)
        )
        regime_shift_metric = float(
            max(
                self.training_metrics.get("mean_regime_shift", 0.0),
                self.training_metrics.get("holdout_regime_shift", 0.0),
            )
        )
        oof_raw_summary = {
            "finite_count": self.training_metrics.get(
                "oof_raw_probability_distribution_finite_count", 0.0
            ),
            "std": self.training_metrics.get("oof_raw_probability_distribution_std", 0.0),
            "span": self.training_metrics.get("oof_raw_probability_distribution_span", 0.0),
        }
        holdout_raw_summary = {
            "finite_count": self.training_metrics.get(
                "holdout_raw_probability_distribution_finite_count",
                self.training_metrics.get("holdout_rows", 0.0),
            ),
            "std": self.training_metrics.get("holdout_raw_probability_distribution_std", 0.0),
            "span": self.training_metrics.get("holdout_raw_probability_distribution_span", 0.0),
        }
        lightgbm_structure_gate_passed = (not self._is_lightgbm_family_model()) or (
            float(self.training_metrics.get("final_model_structure_split_count", 0.0)) > 0.0
        )
        oof_dispersion_gate_passed = (
            not self._is_lightgbm_family_model()
        ) or not self._is_dead_probability_summary(oof_raw_summary)
        holdout_dispersion_gate_passed = (
            (not self._is_lightgbm_family_model())
            or (not holdout_available)
            or not self._is_dead_probability_summary(holdout_raw_summary)
        )
        self.training_metrics["lightgbm_structure_gate_passed"] = float(
            lightgbm_structure_gate_passed
        )
        self.training_metrics["lightgbm_oof_dispersion_gate_passed"] = float(
            oof_dispersion_gate_passed
        )
        self.training_metrics["lightgbm_holdout_dispersion_gate_passed"] = float(
            holdout_dispersion_gate_passed
        )
        if not holdout_available:
            self.logger.warning(
                "Holdout block unavailable for this validation pass; "
                "full training pipeline now fails earlier during holdout construction."
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
            "lightgbm_booster_has_splits": (
                lightgbm_structure_gate_passed,
                1.0 if lightgbm_structure_gate_passed else 0.0,
                1.0,
            ),
            "lightgbm_oof_score_dispersion": (
                oof_dispersion_gate_passed,
                1.0 if oof_dispersion_gate_passed else 0.0,
                1.0,
            ),
            "oof_prediction_coverage": (
                (
                    self.training_metrics.get("oof_prediction_coverage", 0.0) >= 0.60
                    if "oof_prediction_coverage" in self.training_metrics
                    else (not self.config.use_meta_labeling)
                ),
                self.training_metrics.get(
                    "oof_prediction_coverage", 1.0 if not self.config.use_meta_labeling else 0.0
                ),
                0.60 if self.config.use_meta_labeling else 0.0,
            ),
            "min_deflated_sharpe": (
                self.training_metrics.get(
                    "deflated_sharpe",
                    self.training_metrics.get("mean_sharpe", 0.0),
                )
                >= self.config.min_deflated_sharpe,
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
                effective_pbo_metric <= effective_max_pbo,
                effective_pbo_metric,
                effective_max_pbo,
            ),
            "min_white_reality_stat": (
                self.training_metrics.get("white_reality_stat", self.config.min_white_reality_stat)
                >= self.config.min_white_reality_stat,
                self.training_metrics.get("white_reality_stat", self.config.min_white_reality_stat),
                self.config.min_white_reality_stat,
            ),
            "max_white_reality_pvalue": (
                self.training_metrics.get(
                    "white_reality_pvalue",
                    self.config.max_white_reality_pvalue,
                )
                <= self.config.max_white_reality_pvalue,
                self.training_metrics.get(
                    "white_reality_pvalue",
                    self.config.max_white_reality_pvalue,
                ),
                self.config.max_white_reality_pvalue,
            ),
            "holdout_trade_count_positive": (
                ((holdout_trade_count_metric > 0.0) if holdout_available else True),
                holdout_trade_count_metric if holdout_available else 1.0,
                0.0,
            ),
            "lightgbm_holdout_score_dispersion": (
                holdout_dispersion_gate_passed if holdout_available else True,
                (1.0 if (holdout_dispersion_gate_passed if holdout_available else True) else 0.0),
                1.0,
            ),
            "holdout_active_signal_rate_positive": (
                ((holdout_active_signal_rate_metric > 0.0) if holdout_available else True),
                holdout_active_signal_rate_metric if holdout_available else 1.0,
                0.0,
            ),
            "min_holdout_sharpe": (
                (
                    (effective_holdout_sharpe >= self.config.min_holdout_sharpe)
                    if holdout_available
                    else True
                ),
                effective_holdout_sharpe if holdout_available else self.config.min_holdout_sharpe,
                self.config.min_holdout_sharpe,
            ),
            "holdout_sharpe_consistency": (
                (
                    (holdout_sharpe_consistency >= holdout_sharpe_consistency_floor)
                    if holdout_available
                    else True
                ),
                (
                    holdout_sharpe_consistency
                    if holdout_available
                    else holdout_sharpe_consistency_floor
                ),
                holdout_sharpe_consistency_floor,
            ),
            "max_holdout_drawdown": (
                (
                    (holdout_drawdown <= self.config.max_holdout_drawdown)
                    if holdout_available
                    else True
                ),
                holdout_drawdown if holdout_available else self.config.max_holdout_drawdown,
                self.config.max_holdout_drawdown,
            ),
            "min_holdout_regime_sharpe": (
                (
                    (holdout_worst_regime_sharpe >= self.config.min_holdout_regime_sharpe)
                    if holdout_available
                    else True
                ),
                (
                    holdout_worst_regime_sharpe
                    if holdout_available
                    else self.config.min_holdout_regime_sharpe
                ),
                self.config.min_holdout_regime_sharpe,
            ),
            "min_holdout_symbol_coverage": (
                (
                    (holdout_symbol_coverage >= self.config.min_holdout_symbol_coverage)
                    if holdout_available
                    else True
                ),
                (
                    holdout_symbol_coverage
                    if holdout_available
                    else self.config.min_holdout_symbol_coverage
                ),
                self.config.min_holdout_symbol_coverage,
            ),
            "min_holdout_symbol_p25_sharpe": (
                (
                    (holdout_symbol_sharpe_p25 >= self.config.min_holdout_symbol_p25_sharpe)
                    if holdout_available
                    else True
                ),
                (
                    holdout_symbol_sharpe_p25
                    if holdout_available
                    else self.config.min_holdout_symbol_p25_sharpe
                ),
                self.config.min_holdout_symbol_p25_sharpe,
            ),
            "max_holdout_symbol_underwater_ratio": (
                (
                    (
                        holdout_symbol_underwater_ratio
                        <= self.config.max_holdout_symbol_underwater_ratio
                    )
                    if holdout_available
                    else True
                ),
                (
                    holdout_symbol_underwater_ratio
                    if holdout_available
                    else self.config.max_holdout_symbol_underwater_ratio
                ),
                self.config.max_holdout_symbol_underwater_ratio,
            ),
            "max_regime_shift": (
                regime_shift_metric <= self.config.max_regime_shift,
                regime_shift_metric,
                self.config.max_regime_shift,
            ),
            "max_symbol_concentration_hhi": (
                symbol_concentration_metric <= self.config.max_symbol_concentration_hhi,
                symbol_concentration_metric,
                self.config.max_symbol_concentration_hhi,
            ),
            "nested_walk_forward_trace": (
                (
                    (
                        (len(self.nested_cv_trace) > 0)
                        or (
                            isinstance(self.training_metrics.get("nested_cv_trace"), list)
                            and len(self.training_metrics.get("nested_cv_trace", [])) > 0
                        )
                    )
                    if self.config.require_nested_trace_for_promotion
                    else True
                ),
                float(
                    len(self.nested_cv_trace)
                    if self.nested_cv_trace
                    else len(self.training_metrics.get("nested_cv_trace", []))
                ),
                1.0 if self.config.require_nested_trace_for_promotion else 0.0,
            ),
            "objective_breakdown_present": (
                (
                    (
                        isinstance(self.training_metrics.get("objective_component_summary"), dict)
                        and len(self.training_metrics.get("objective_component_summary", {})) > 0
                    )
                    if self.config.require_objective_breakdown_for_promotion
                    else True
                ),
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
        for gate_name, (passed, actual, threshold) in gates.items():
            passed_bool = bool(passed)
            actual_value = float(actual)
            threshold_value = float(threshold)
            normalized_gates[gate_name] = (passed_bool, actual_value, threshold_value)

            status = "PASS" if passed_bool else "FAIL"
            self.logger.info(
                f"  {gate_name}: {status} (actual={actual_value:.4f}, threshold={threshold_value:.4f})"
            )

        layer_definitions: dict[str, tuple[str, ...]] = {
            "model_utility": (
                "min_sharpe_ratio",
                "min_accuracy",
                "min_win_rate",
                "risk_adjusted_positive",
                "lightgbm_booster_has_splits",
                "lightgbm_oof_score_dispersion",
                "min_deflated_sharpe",
                "max_deflated_sharpe_pvalue",
                "max_pbo",
                "min_white_reality_stat",
                "max_white_reality_pvalue",
                "min_holdout_sharpe",
                "holdout_sharpe_consistency",
                "nested_walk_forward_trace",
                "objective_breakdown_present",
            ),
            "execution_robustness": (
                "max_drawdown",
                "max_holdout_drawdown",
                "min_trades",
                "lightgbm_holdout_score_dispersion",
                "holdout_trade_count_positive",
                "holdout_active_signal_rate_positive",
                "max_regime_shift",
                "max_symbol_concentration_hhi",
                "min_holdout_regime_sharpe",
            ),
            "cross_symbol_robustness": (
                "min_holdout_symbol_coverage",
                "min_holdout_symbol_p25_sharpe",
                "max_holdout_symbol_underwater_ratio",
                "oof_prediction_coverage",
            ),
        }

        layer_results: dict[str, dict[str, Any]] = {}
        for layer_name, layer_gates in layer_definitions.items():
            failed_gates = [
                gate_name
                for gate_name in layer_gates
                if gate_name in normalized_gates and not normalized_gates[gate_name][0]
            ]
            passed = len(failed_gates) == 0
            layer_results[layer_name] = {
                "passed": bool(passed),
                "failed_gate_count": int(len(failed_gates)),
                "failed_gates": failed_gates,
                "gate_count": int(len(layer_gates)),
            }
            self.training_metrics[f"layer_{layer_name}_passed"] = 1.0 if passed else 0.0
            self.logger.info(
                "  layer:%s: %s (failed=%s)",
                layer_name,
                "PASS" if passed else "FAIL",
                ",".join(failed_gates) if failed_gates else "none",
            )

        all_gates_passed = all(gate_tuple[0] for gate_tuple in normalized_gates.values())
        all_layers_passed = all(layer_payload["passed"] for layer_payload in layer_results.values())
        all_passed = bool(all_gates_passed and all_layers_passed)
        self.training_metrics["validation_layer_all_passed"] = 1.0 if all_layers_passed else 0.0

        self.validation_results = {
            "gates": normalized_gates,
            "layers": layer_results,
            "all_passed": bool(all_passed),
            "all_gates_passed": bool(all_gates_passed),
            "all_layers_passed": bool(all_layers_passed),
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
        layer_summary: dict[str, dict[str, Any]] = {}
        layers = self.validation_results.get("layers", {})
        if isinstance(layers, dict):
            for layer_name, layer_payload in layers.items():
                if not isinstance(layer_payload, dict):
                    continue
                layer_summary[str(layer_name)] = {
                    "passed": bool(layer_payload.get("passed", False)),
                    "failed_gate_count": int(layer_payload.get("failed_gate_count", 0)),
                    "failed_gates": list(layer_payload.get("failed_gates", [])),
                    "gate_count": int(layer_payload.get("gate_count", 0)),
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
                "mean_symbol_concentration_hhi": float(
                    self.training_metrics.get("mean_symbol_concentration_hhi", 0.0)
                ),
                "holdout_sharpe": float(self.training_metrics.get("holdout_sharpe", 0.0)),
                "holdout_max_drawdown": float(
                    self.training_metrics.get("holdout_max_drawdown", 1.0)
                ),
                "holdout_regime_shift": float(
                    self.training_metrics.get("holdout_regime_shift", 0.0)
                ),
                "holdout_symbol_concentration_hhi": float(
                    self.training_metrics.get("holdout_symbol_concentration_hhi", 0.0)
                ),
                "holdout_worst_regime_sharpe": float(
                    self.training_metrics.get("holdout_worst_regime_sharpe", 0.0)
                ),
                "holdout_symbol_coverage_ratio": float(
                    self.training_metrics.get("holdout_symbol_coverage_ratio", 0.0)
                ),
                "holdout_symbol_sharpe_p25": float(
                    self.training_metrics.get("holdout_symbol_sharpe_p25", 0.0)
                ),
                "holdout_symbol_underwater_ratio": float(
                    self.training_metrics.get("holdout_symbol_underwater_ratio", 0.0)
                ),
                "deflated_sharpe": float(self.training_metrics.get("deflated_sharpe", 0.0)),
                "deflated_sharpe_p_value": float(
                    self.training_metrics.get("deflated_sharpe_p_value", 1.0)
                ),
                "pbo": float(self.training_metrics.get("pbo", 1.0)),
                "white_reality_stat": float(self.training_metrics.get("white_reality_stat", 0.0)),
                "white_reality_pvalue": float(
                    self.training_metrics.get("white_reality_pvalue", 1.0)
                ),
            },
            "data": {
                "development_rows": int(self.training_metrics.get("development_rows", 0)),
                "holdout_rows": int(self.training_metrics.get("holdout_rows", 0)),
                "holdout_symbol_count_total": int(
                    self.training_metrics.get("holdout_symbol_count_total", 0)
                ),
                "holdout_symbol_count_evaluated": int(
                    self.training_metrics.get("holdout_symbol_count_evaluated", 0)
                ),
                "regime_distribution_train": regime_mix,
                "regime_distribution_holdout": holdout_regime_mix,
            },
            "gates": gate_summary,
            "gate_layers": layer_summary,
            "top_features": top_features,
        }

    def _build_deployment_plan(self, passed_validation: bool) -> dict[str, Any]:
        """Build champion/challenger canary rollout and runtime risk guardrails."""
        total_cost_bps = float(self.config.to_trading_cost_model().execution_cost_bps)
        layer_payload = self.validation_results.get("layers", {})
        if isinstance(layer_payload, dict):
            layers = {
                str(layer_name): {
                    "passed": bool(layer_state.get("passed", False)),
                    "failed_gates": list(layer_state.get("failed_gates", [])),
                    "failed_gate_count": int(layer_state.get("failed_gate_count", 0)),
                }
                for layer_name, layer_state in layer_payload.items()
                if isinstance(layer_state, dict)
            }
        else:
            layers = {}
        all_layers_passed = bool(layers) and all(
            layer.get("passed", False) for layer in layers.values()
        )
        effective_ready = bool(passed_validation and (all_layers_passed or not layers))

        return {
            "schema_version": "1.0.0",
            "ready_for_production": effective_ready,
            "promotion_strategy": {
                "mode": "champion_challenger_canary",
                "champion_retention_if_fail": True,
                "challenger_activation_requires_all_gates": True,
                "challenger_activation_requires_layers": [
                    "model_utility",
                    "execution_robustness",
                    "cross_symbol_robustness",
                ],
            },
            "promotion_layers": {
                "all_layers_passed": bool(all_layers_passed),
                "layers": layers,
            },
            "canary_rollout": [
                {"phase": 1, "capital_fraction": 0.05, "min_trades": 50},
                {"phase": 2, "capital_fraction": 0.15, "min_trades": 120},
                {"phase": 3, "capital_fraction": 0.35, "min_trades": 250},
                {"phase": 4, "capital_fraction": 1.00, "min_trades": 400},
            ],
            "kill_switch_guardrails": {
                "max_intraday_drawdown": float(
                    min(self.config.max_drawdown, self.config.max_holdout_drawdown)
                ),
                "max_regime_shift": float(self.config.max_regime_shift),
                "min_live_sharpe_rolling": float(
                    max(self.config.min_holdout_sharpe, self.config.min_holdout_regime_sharpe, 0.0)
                ),
                "min_holdout_symbol_coverage": float(self.config.min_holdout_symbol_coverage),
                "min_holdout_symbol_p25_sharpe": float(self.config.min_holdout_symbol_p25_sharpe),
                "max_holdout_symbol_underwater_ratio": float(
                    self.config.max_holdout_symbol_underwater_ratio
                ),
                "min_trade_count_rolling": int(self.config.min_trades),
                "max_symbol_concentration_hhi": float(self.config.max_symbol_concentration_hhi),
            },
            "tca_guardrails": {
                "expected_cost_bps": total_cost_bps,
                "max_slippage_bps": float(total_cost_bps * 1.5),
                "max_turnover": float(self.config.execution_turnover_cap),
                "execution_cooldown_bars": int(self.config.execution_cooldown_bars),
                "max_symbol_entry_share": float(self.config.execution_max_symbol_entry_share),
            },
        }

    def _record_pre_promotion_checklist(self) -> dict[str, Any]:
        """Persist research-readiness evidence used to authorize promotion runs."""
        checklist = _build_pre_promotion_checklist(
            {
                "model_name": self.config.model_name,
                "model_type": self.config.model_type,
                "training_profile": self.config.training_profile,
                "snapshot_id": (
                    str(self.snapshot_manifest.get("snapshot_id"))
                    if isinstance(self.snapshot_manifest, dict)
                    else None
                ),
                "dataset_bundle_hash": (
                    str(self.dataset_snapshot_bundle_manifest.get("bundle_hash"))
                    if isinstance(self.dataset_snapshot_bundle_manifest, dict)
                    else (
                        str(self.snapshot_manifest.get("dataset_bundle_hash"))
                        if isinstance(self.snapshot_manifest, dict)
                        else None
                    )
                ),
                "data_quality_report_hash": self.data_quality_report_hash,
                "training_metrics": self.training_metrics,
                "validation_results": self.validation_results,
            }
        )
        self.training_metrics["pre_promotion_checklist"] = self._json_safe(checklist)
        self.training_metrics["pre_promotion_ready"] = 1.0 if checklist.get("ready", False) else 0.0
        self.training_metrics["pre_promotion_failed_checks"] = list(
            checklist.get("failed_checks", [])
        )
        return checklist

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

            primary_predictions = np.asarray(self.oof_primary_proba[oof_mask], dtype=float).reshape(
                -1
            )
            used_close = self.close_prices.iloc[oof_mask].reset_index(drop=True)
            used_features = self.features.iloc[oof_mask].copy()
            self.logger.info(
                f"Meta-labeling uses OOF predictions: {usable_count}/{len(self.features)} samples"
            )

            meta_dynamic_threshold = bool(self.config.meta_label_dynamic_threshold)
            if meta_dynamic_threshold:
                meta_min_confidence = self._meta_confidence_for_horizon()
            else:
                meta_min_confidence = float(
                    np.clip(self.config.meta_label_min_confidence, 0.45, 0.95)
                )
            self.training_metrics["meta_label_min_confidence"] = meta_min_confidence
            self.training_metrics["meta_label_dynamic_threshold"] = (
                1.0 if meta_dynamic_threshold else 0.0
            )
            self.training_metrics["meta_label_primary_horizon"] = float(self._primary_horizon())

            meta_cfg = MetaLabelConfig(
                model_type=self.config.meta_model_type,
                min_confidence=meta_min_confidence,
                dynamic_threshold=meta_dynamic_threshold,
            )
            self.logger.info(
                "Meta-label threshold policy: min_confidence=%.3f dynamic=%s horizon=%d",
                meta_min_confidence,
                str(meta_dynamic_threshold).lower(),
                self._primary_horizon(),
            )
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

        use_expected_edge_returns = bool(self.training_metrics.get("expected_edge_policy_enabled", 0.0)) and (
            bool(self.expected_edge_cv_active_return_series) or bool(self.expected_edge_cv_return_series)
        )
        if use_expected_edge_returns:
            sharpe = float(
                self.training_metrics.get(
                    "expected_edge_training_policy_sharpe",
                    self.training_metrics.get("mean_sharpe", 0.0),
                )
            )
            active_returns = (
                np.concatenate(self.expected_edge_cv_active_return_series)
                if self.expected_edge_cv_active_return_series
                else np.array([], dtype=float)
            )
            full_returns = (
                np.concatenate(self.expected_edge_cv_return_series)
                if self.expected_edge_cv_return_series
                else np.array([], dtype=float)
            )
            return_source_prefix = "expected_edge_policy_"
        else:
            sharpe = float(self.training_metrics.get("mean_sharpe", 0.0))
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
            return_source_prefix = ""
        use_active_returns = int(active_returns.size) >= 30
        returns = active_returns if use_active_returns else full_returns
        self.training_metrics["multiple_testing_strategy_source"] = (
            "expected_edge_policy" if use_expected_edge_returns else "base"
        )
        self.training_metrics["multiple_testing_return_source"] = (
            f"{return_source_prefix}{'active_returns' if use_active_returns else 'all_returns'}"
        )
        self.training_metrics["multiple_testing_return_observations"] = float(returns.size)
        n_trials = max(
            int(getattr(self.config, "n_trials", 1)),
            len(self.cv_results) if self.cv_results else 1,
            1,
        )
        bootstrap_reps = int(np.clip(max(400, n_trials * 30), 400, 3000))
        block_size = int(np.clip(np.sqrt(float(max(returns.size, 1))), 3.0, 32.0))
        annualization_factor = float(self._annualization_periods())
        white_stat, white_p_value, white_interpretation = calculate_white_reality_check(
            returns=returns,
            n_bootstrap=bootstrap_reps,
            block_size=block_size,
            random_seed=int(self.config.seed),
            annualization_factor=annualization_factor,
        )
        self.training_metrics["white_reality_stat"] = float(white_stat)
        self.training_metrics["white_reality_pvalue"] = float(white_p_value)
        self.training_metrics["white_reality_interpretation"] = str(white_interpretation)
        self.training_metrics["white_reality_bootstrap_reps"] = float(bootstrap_reps)
        self.training_metrics["white_reality_block_size"] = float(block_size)
        self.training_metrics["white_reality_annualization_factor"] = float(annualization_factor)
        self.logger.info(
            "White Reality Check: stat=%.4f p=%.4f (%s)",
            float(white_stat),
            float(white_p_value),
            str(white_interpretation),
        )

        if self.config.correction_method == "deflated_sharpe":
            deflated_sharpe, p_value = calculate_deflated_sharpe_ratio(
                observed_sharpe=sharpe,
                returns=returns,
                n_trials=n_trials,
            )
            pbo, interpretation, pbo_diagnostics = calculate_probability_of_backtest_overfitting(
                returns,
                return_diagnostics=True,
            )
            deflation = deflated_sharpe / sharpe if abs(sharpe) > 1e-12 else 0.0
            pbo = float(np.nan_to_num(pbo, nan=1.0, posinf=1.0, neginf=1.0))
            pbo_upper_95 = float(np.clip(pbo_diagnostics.get("pbo_ci_upper_95", pbo), 0.0, 1.0))
            pbo_reliability = float(np.clip(pbo_diagnostics.get("pbo_reliability", 0.0), 0.0, 1.0))
            effective_holdout = self._resolve_effective_holdout_metric_payload()
            holdout_available = float(self.training_metrics.get("holdout_rows", 0.0)) > 0.0
            holdout_sharpe = float(
                effective_holdout.get(
                    "sharpe",
                    self.training_metrics.get("mean_sharpe", 0.0),
                )
            )
            self.training_metrics["pbo_holdout_metric_source"] = str(
                effective_holdout.get("source", "base")
            )
            sharpe_gap_baseline = max(abs(sharpe), 0.25)
            holdout_gap_ratio = (
                float(
                    np.clip(
                        (sharpe - holdout_sharpe) / sharpe_gap_baseline,
                        0.0,
                        1.5,
                    )
                )
                if holdout_available
                else 0.0
            )
            effective_pbo_metric = self._effective_pbo_gate_metric(
                base_pbo=pbo,
                pbo_upper_95=pbo_upper_95,
                pbo_reliability=pbo_reliability,
                holdout_gap_ratio=holdout_gap_ratio,
            )
            self.training_metrics["deflated_sharpe"] = deflated_sharpe
            self.training_metrics["deflated_sharpe_p_value"] = p_value
            self.training_metrics["sharpe_deflation_factor"] = deflation
            self.training_metrics["pbo"] = pbo
            self.training_metrics["pbo_interpretation"] = interpretation
            self.training_metrics["pbo_ci_upper_95"] = pbo_upper_95
            self.training_metrics["pbo_ci_lower_95"] = float(
                np.clip(pbo_diagnostics.get("pbo_ci_lower_95", pbo), 0.0, 1.0)
            )
            self.training_metrics["pbo_probability_stability"] = float(
                np.clip(pbo_diagnostics.get("pbo_probability_stability", 0.0), 0.0, 1.0)
            )
            self.training_metrics["pbo_reliability"] = pbo_reliability
            self.training_metrics["pbo_holdout_gap_ratio"] = holdout_gap_ratio
            self.training_metrics["effective_pbo_gate_metric"] = float(effective_pbo_metric)
            self.training_metrics["effective_max_pbo_gate"] = self._effective_pbo_threshold(
                sample_size=int(returns.size),
                interpretation=str(interpretation),
                holdout_gap_ratio=holdout_gap_ratio,
                pbo_reliability=(pbo_reliability if int(returns.size) >= 80 else None),
            )

            self.logger.info(
                f"Deflated Sharpe: {deflated_sharpe:.4f} "
                f"(original: {sharpe:.4f}, p-value: {p_value:.4f}, deflation: {deflation:.4f})"
            )
            self.logger.info(
                "PBO: %.2f%% (gate_metric=%.2f%%, ci95_upper=%.2f%%, reliability=%.3f, holdout_gap=%.3f) %s",
                pbo * 100.0,
                effective_pbo_metric * 100.0,
                pbo_upper_95 * 100.0,
                pbo_reliability,
                holdout_gap_ratio,
                interpretation,
            )

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

    def _effective_pbo_threshold(
        self,
        sample_size: int,
        interpretation: str,
        holdout_gap_ratio: float = 0.0,
        pbo_reliability: float | None = None,
    ) -> float:
        """Adjust PBO gate for low-sample statistical uncertainty."""
        base_threshold = float(np.clip(self.config.max_pbo, 0.05, 0.95))
        if sample_size <= 0:
            return base_threshold

        reliability = float(np.clip((float(sample_size) - 80.0) / 320.0, 0.0, 1.0))
        uncertainty_slack = (1.0 - reliability) * 0.10
        if interpretation.lower().startswith("insufficient"):
            uncertainty_slack = max(uncertainty_slack, 0.10)
        threshold = float(min(0.60, max(base_threshold, base_threshold + uncertainty_slack)))

        holdout_penalty = float(np.clip(holdout_gap_ratio, 0.0, 1.5)) * 0.08
        reliability_penalty = 0.0
        if pbo_reliability is not None:
            reliability_penalty = float(np.clip((0.50 - float(pbo_reliability)), 0.0, 0.50)) * 0.08

        threshold -= holdout_penalty + reliability_penalty
        return float(np.clip(threshold, 0.15, 0.60))

    @staticmethod
    def _effective_pbo_gate_metric(
        base_pbo: float,
        pbo_upper_95: float,
        pbo_reliability: float,
        holdout_gap_ratio: float,
    ) -> float:
        """Build conservative PBO gate metric from uncertainty and OOS degradation."""
        pbo = float(np.clip(base_pbo, 0.0, 1.0))
        pbo_ci_upper = float(np.clip(pbo_upper_95, pbo, 1.0))
        reliability = float(np.clip(pbo_reliability, 0.0, 1.0))
        holdout_gap = float(np.clip(holdout_gap_ratio, 0.0, 1.5))

        uncertainty_lift = float(1.0 - reliability) * 0.10
        ci_lift = float(np.clip(pbo_ci_upper - pbo, 0.0, 1.0)) * 0.55
        holdout_lift = holdout_gap * 0.12
        return float(np.clip(pbo + uncertainty_lift + ci_lift + holdout_lift, 0.0, 1.0))

    def _effective_holdout_sharpe_metric(self, holdout_sharpe: float) -> float:
        """Shrink holdout Sharpe by confidence and CV->holdout deterioration."""
        holdout_rows = max(0.0, float(self.training_metrics.get("holdout_rows", 0.0)))
        effective_holdout = self._resolve_effective_holdout_metric_payload()
        holdout_trade_obs = max(0.0, float(effective_holdout.get("trade_return_observations", 0.0)))
        holdout_sharpe_confidence = float(
            np.clip(effective_holdout.get("sharpe_observation_confidence", 0.0), 0.0, 1.0)
        )
        expected_trade_target = float(
            self._effective_trade_target(int(round(holdout_rows)) if holdout_rows > 0 else None)
        )
        obs_coverage = float(np.clip(holdout_trade_obs / max(expected_trade_target, 1.0), 0.0, 1.0))

        train_regimes = self._regime_distribution(self.regimes)
        regime_target = max(1.0, float(len(train_regimes)))
        holdout_regime_count = max(
            0.0,
            float(self.training_metrics.get("holdout_regime_count_evaluated", 0.0)),
        )
        regime_coverage = float(np.clip(holdout_regime_count / regime_target, 0.0, 1.0))
        confidence = float(
            np.clip(
                (0.55 * holdout_sharpe_confidence)
                + (0.30 * obs_coverage)
                + (0.15 * regime_coverage),
                0.0,
                1.0,
            )
        )

        mean_sharpe = float(self.training_metrics.get("mean_sharpe", 0.0))
        sharpe_gap_baseline = max(abs(mean_sharpe), 0.25)
        degradation_ratio = float(
            np.clip((mean_sharpe - holdout_sharpe) / sharpe_gap_baseline, 0.0, 1.5)
        )
        uncertainty_penalty = (1.0 - confidence) * 0.22
        degradation_penalty = degradation_ratio * 0.18
        adjusted_holdout_sharpe = float(holdout_sharpe - uncertainty_penalty - degradation_penalty)

        self.training_metrics["holdout_sharpe_gate_confidence"] = float(confidence)
        self.training_metrics["holdout_sharpe_gate_obs_coverage"] = float(obs_coverage)
        self.training_metrics["holdout_sharpe_gate_regime_coverage"] = float(regime_coverage)
        self.training_metrics["holdout_sharpe_gap_ratio"] = float(degradation_ratio)
        self.training_metrics["holdout_sharpe_gate_metric_source"] = str(
            effective_holdout.get("source", "base")
        )

        return adjusted_holdout_sharpe

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
            is_sequence_model = self.config.model_type in sequence_model_types or hasattr(
                self.model, "lookback_window"
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

            if (
                is_sequence_model
                and hasattr(self.model, "_network")
                and hasattr(self.model, "device")
            ):
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

            if self.config.model_type in [
                "xgboost",
                "lightgbm",
                "lightgbm_ranker",
                "xgboost_regressor",
                "lightgbm_regressor",
                "random_forest",
            ]:
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

                explainer = shap.KernelExplainer(_predict_for_shap, background)

            if self.config.model_type in [
                "xgboost",
                "lightgbm",
                "lightgbm_ranker",
                "xgboost_regressor",
                "lightgbm_regressor",
                "random_forest",
            ]:
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
            "dataset_snapshot_bundle_path": (
                str(self.dataset_snapshot_bundle_manifest_path)
                if self.dataset_snapshot_bundle_manifest_path
                else None
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
        artifacts_path = output_dir / f"{self.config.model_name}_artifacts.json"
        meta_model_path = (
            output_dir / f"{self.config.model_name}_meta.pkl"
            if self.meta_model is not None
            else None
        )
        expected_edge_model_path = (
            output_dir / f"{self.config.model_name}_expected_edge.pkl"
            if self.expected_edge_model is not None
            else None
        )
        selected_features = list(
            self.training_metrics.get(
                "feature_selection_selected_features", self.feature_names or []
            )
        )
        cost_model = self.config.to_trading_cost_model()

        payload = {
            "schema_version": PROMOTION_PACKAGE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_name": self.config.model_name,
            "model_type": self.config.model_type,
            "model_path": str(model_path),
            "meta_model_path": str(meta_model_path) if meta_model_path is not None else None,
            "expected_edge_model_path": (
                str(expected_edge_model_path) if expected_edge_model_path is not None else None
            ),
            "artifacts_path": str(artifacts_path),
            "training_config": self.config.__dict__,
            "training_provenance": {
                "git_commit": self.training_metrics.get(
                    "git_commit",
                    self.training_metrics.get("git_commit_hash"),
                ),
                "git_commit_hash": self.training_metrics.get("git_commit_hash"),
                "git_branch": self.training_metrics.get("git_branch"),
                "git_dirty": self.training_metrics.get("git_dirty"),
                "environment_lock_hash": self.training_metrics.get(
                    "environment_lock_hash",
                    self.training_metrics.get("dependency_lock_hash"),
                ),
                "dependency_lock_hash": self.training_metrics.get("dependency_lock_hash"),
                "training_host": self.training_metrics.get(
                    "training_host",
                    self.training_metrics.get("host"),
                ),
                "wsl_distro": self.training_metrics.get(
                    "training_wsl_distro",
                    self.training_metrics.get("wsl_distro"),
                ),
                "repo_provenance_ready": bool(
                    self.training_metrics.get("training_repo_provenance_ready", 0.0)
                ),
            },
            "feature_schema_version": (
                str(self.snapshot_manifest.get("feature_schema_version"))
                if isinstance(self.snapshot_manifest, dict)
                else None
            ),
            "feature_contract": {
                "feature_schema_version": (
                    str(self.snapshot_manifest.get("feature_schema_version"))
                    if isinstance(self.snapshot_manifest, dict)
                    else None
                ),
                "feature_names": list(self.feature_names or []),
                "selected_features": selected_features,
                "feature_groups": sorted(str(group) for group in self.config.feature_groups),
                "enable_cross_sectional": bool(self.config.enable_cross_sectional),
                "enable_reference_features": bool(self.config.enable_reference_features),
                "reference_feature_sources": list(self.config.reference_feature_sources),
                "enable_tick_microstructure_features": bool(
                    self.config.enable_tick_microstructure_features
                ),
                "timeframe": str(self.config.timeframe),
                "timeframes": list(self.config.timeframes),
            },
            "feature_selection_contract": {
                "effective_binding": float(self.training_metrics.get("feature_selection_binding", 0.0)),
                "development_binding": float(
                    self.training_metrics.get("feature_selection_development_binding", 0.0)
                ),
                "final_refit_binding": float(
                    self.training_metrics.get("feature_selection_final_refit_binding", 0.0)
                ),
                "development_initial_feature_count": float(
                    self.training_metrics.get(
                        "feature_selection_development_initial_feature_count",
                        self.training_metrics.get("feature_selection_initial_feature_count", 0.0),
                    )
                ),
                "development_selected_feature_count": float(
                    self.training_metrics.get(
                        "feature_selection_development_selected_feature_count",
                        self.training_metrics.get("feature_selection_selected_feature_count", 0.0),
                    )
                ),
                "final_refit_initial_feature_count": float(
                    self.training_metrics.get(
                        "feature_selection_final_refit_initial_feature_count",
                        self.training_metrics.get("feature_selection_selected_feature_count", 0.0),
                    )
                ),
                "final_refit_selected_feature_count": float(
                    self.training_metrics.get(
                        "feature_selection_final_refit_selected_feature_count",
                        self.training_metrics.get("feature_selection_selected_feature_count", 0.0),
                    )
                ),
            },
            "signal_policy": {
                "long_threshold": self.training_metrics.get("holdout_long_threshold"),
                "short_threshold": self.training_metrics.get("holdout_short_threshold"),
                "horizon_bars": int(self.config.primary_label_horizon),
                "max_holding_bars": int(self.config.label_max_holding_period),
                "take_profit_pct": float(self.config.label_profit_taking_threshold),
                "stop_loss_pct": float(abs(self.config.label_stop_loss_threshold)),
                "meta_label_enabled": bool(self.meta_model is not None),
                "meta_label_threshold": self.training_metrics.get("meta_label_min_confidence"),
                "meta_label_dynamic_threshold": bool(
                    self.training_metrics.get("meta_label_dynamic_threshold", 0.0)
                ),
                "probability_calibration_enabled": bool(
                    self.training_metrics.get("probability_calibration_enabled", 0.0)
                ),
                "probability_calibration_method": self.training_metrics.get(
                    "probability_calibration_method",
                    self.training_metrics.get("probability_calibration_requested_method"),
                ),
                "long_side_policy": (
                    self.training_metrics.get("holdout_side_policy", {}).get("long", {})
                    if isinstance(self.training_metrics.get("holdout_side_policy"), dict)
                    else {}
                ),
                "short_side_policy": (
                    self.training_metrics.get("holdout_side_policy", {}).get("short", {})
                    if isinstance(self.training_metrics.get("holdout_side_policy"), dict)
                    else {}
                ),
                "model_source": f"promotion_package:{self.config.model_name}",
            },
            "signal_activity_contract": self.training_metrics.get(
                "holdout_signal_activity_summary",
                {},
            ),
            "execution_cost_model": {
                "spread_bps": float(cost_model.spread_bps),
                "slippage_bps": float(cost_model.slippage_bps),
                "impact_bps": float(cost_model.impact_bps),
            },
            "position_sizing_policy": {
                "use_portfolio_target_sizing": True,
                "max_position_pct": 0.10,
                "max_total_positions": 20,
                "confidence_position_sizing": True,
                "min_confidence_position_scale": float(self.config.min_confidence_position_scale),
            },
            "expected_edge_policy": {
                "enabled": bool(self.expected_edge_model is not None),
                "min_pass_probability": float(self.config.expected_edge_min_pass_probability),
                "min_expected_edge": float(self.config.expected_edge_min_expected_edge),
                "min_signal_scale": float(self.config.expected_edge_min_signal_scale),
                "max_signal_scale": float(self.config.expected_edge_max_signal_scale),
                "use_symbol_priors": bool(self.config.expected_edge_use_symbol_priors),
                "selected_context_features": list(
                    self.training_metrics.get(
                        "expected_edge_training_selected_context_features", []
                    )
                ),
                "training_summary": self.training_metrics.get("expected_edge_training_summary", {}),
                "holdout_summary": self.training_metrics.get("expected_edge_holdout_summary", {}),
                "edge_reference": self.training_metrics.get(
                    "expected_edge_training_edge_reference",
                    self.training_metrics.get("expected_edge_holdout_edge_reference"),
                ),
                "regime_conditioned_policy": self.training_metrics.get(
                    "expected_edge_regime_policy", {}
                ),
            },
            "ranker_scoring": (self._ranker_scoring_contract() if self._is_ranker_model() else {}),
            "promotion_passed": bool(self.validation_results.get("all_passed", False)),
            "snapshot_id": (
                self.snapshot_manifest.get("snapshot_id")
                if isinstance(self.snapshot_manifest, dict)
                else None
            ),
            "snapshot_manifest_path": (
                str(self.snapshot_manifest_path) if self.snapshot_manifest_path else None
            ),
            "dataset_snapshot_bundle_path": (
                str(self.dataset_snapshot_bundle_manifest_path)
                if self.dataset_snapshot_bundle_manifest_path
                else None
            ),
            "data_quality_report_path": (
                str(self.data_quality_report_path) if self.data_quality_report_path else None
            ),
            "data_quality_report_hash": self.data_quality_report_hash,
            "universe_quality_policy": {
                "enabled": bool(self.config.enable_symbol_quality_filter),
                "min_rows": int(self.config.symbol_quality_min_rows),
                "min_symbols": int(self.config.symbol_quality_min_symbols),
                "max_missing_ratio": float(self.config.symbol_quality_max_missing_ratio),
                "max_extreme_move_ratio": float(self.config.symbol_quality_max_extreme_move_ratio),
                "max_corporate_action_ratio": float(
                    self.config.symbol_quality_max_corporate_action_ratio
                ),
                "min_median_dollar_volume": float(
                    self.config.symbol_quality_min_median_dollar_volume
                ),
                "selected_symbols": list(self.training_metrics.get("symbol_quality_universe", [])),
                "dropped_symbols": list(
                    self.training_metrics.get("symbol_quality_dropped_list", [])
                ),
                "symbol_quality_report": self.training_metrics.get("symbol_quality_report", {}),
            },
            "validation_results": self.validation_results,
            "objective_component_summary": self.training_metrics.get(
                "objective_component_summary", {}
            ),
            "feature_quality_contract": {
                "summary": self.training_metrics.get("feature_quality_summary", {}),
                "family_summary": self.training_metrics.get("feature_quality_family_summary", {}),
                "worst_null_features": self.training_metrics.get(
                    "feature_quality_worst_null_features",
                    [],
                ),
                "worst_clip_features": self.training_metrics.get(
                    "feature_quality_worst_clip_features",
                    [],
                ),
            },
            "model_card": self.training_metrics.get("model_card", {}),
            "deployment_plan": self.training_metrics.get("deployment_plan", {}),
            "nested_cv_trace": self.training_metrics.get("nested_cv_trace", self.nested_cv_trace),
            "statistical_validity": {
                "deflated_sharpe": self.training_metrics.get("deflated_sharpe"),
                "deflated_sharpe_p_value": self.training_metrics.get("deflated_sharpe_p_value"),
                "pbo": self.training_metrics.get("pbo"),
                "pbo_interpretation": self.training_metrics.get("pbo_interpretation"),
                "white_reality_stat": self.training_metrics.get("white_reality_stat"),
                "white_reality_pvalue": self.training_metrics.get("white_reality_pvalue"),
                "white_reality_interpretation": self.training_metrics.get(
                    "white_reality_interpretation"
                ),
            },
            "promotion_thresholds": {
                "min_sharpe_ratio": self.config.min_sharpe_ratio,
                "min_accuracy": self.config.min_accuracy,
                "max_drawdown": self.config.max_drawdown,
                "min_win_rate": self.config.min_win_rate,
                "min_trades": self.config.min_trades,
                "min_holdout_sharpe": self.config.min_holdout_sharpe,
                "min_holdout_regime_sharpe": self.config.min_holdout_regime_sharpe,
                "min_holdout_symbol_coverage": self.config.min_holdout_symbol_coverage,
                "min_holdout_symbol_p25_sharpe": self.config.min_holdout_symbol_p25_sharpe,
                "max_holdout_symbol_underwater_ratio": self.config.max_holdout_symbol_underwater_ratio,
                "max_holdout_drawdown": self.config.max_holdout_drawdown,
                "max_regime_shift": self.config.max_regime_shift,
                "max_symbol_concentration_hhi": self.config.max_symbol_concentration_hhi,
                "min_deflated_sharpe": self.config.min_deflated_sharpe,
                "max_deflated_sharpe_pvalue": self.config.max_deflated_sharpe_pvalue,
                "max_pbo": self.config.max_pbo,
                "min_white_reality_stat": self.config.min_white_reality_stat,
                "max_white_reality_pvalue": self.config.max_white_reality_pvalue,
            },
            "replay_manifest_path": str(replay_manifest_path) if replay_manifest_path else None,
        }

        package_path = package_dir / f"{self.config.model_name}.promotion_package.json"
        with package_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=True, default=str)

        self.logger.info(f"Promotion package saved to: {package_path}")
        return package_path

    def _snapshot_bundle_cv_splits(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return auditable CV splits for immutable dataset snapshot bundles."""
        if self.cached_cv_splits:
            return self.cached_cv_splits
        if self.features is None or self.labels is None:
            return []
        try:
            self.cached_cv_splits = self._generate_cv_splits(
                self.features.values,
                self.labels.values,
            )
        except Exception as exc:
            self.logger.warning("Failed to capture CV splits for snapshot bundle: %s", exc)
            self.cached_cv_splits = []
        return self.cached_cv_splits

    def _build_snapshot_review(self) -> dict[str, Any]:
        """Build a review payload for Task 1 snapshot-quality sign-off."""
        data_quality_passed = bool(
            isinstance(self.data_quality_report, dict)
            and self.data_quality_report.get("passed", False)
        )
        dropped_symbols = [
            str(symbol)
            for symbol in self.training_metrics.get("symbol_quality_dropped_list", [])
            if str(symbol).strip()
        ]
        selected_symbols = [
            str(symbol)
            for symbol in self.training_metrics.get("symbol_quality_universe", [])
            if str(symbol).strip()
        ]
        checks = {
            "data_quality_passed": {
                "passed": data_quality_passed,
                "actual": data_quality_passed,
                "threshold": True,
                "comparator": "is",
            },
            "no_silent_symbol_drop": {
                "passed": len(dropped_symbols) == 0,
                "actual": len(dropped_symbols),
                "threshold": 0,
                "comparator": "<=",
            },
        }
        failed_checks = [name for name, payload in checks.items() if not bool(payload["passed"])]
        dataset_bundle_manifest_path = (
            str(self.dataset_snapshot_bundle_manifest_path)
            if self.dataset_snapshot_bundle_manifest_path is not None
            else None
        )
        return {
            "ready": len(failed_checks) == 0,
            "failed_checks": failed_checks,
            "checks": checks,
            "data_quality_passed": data_quality_passed,
            "no_silent_symbol_drop": len(dropped_symbols) == 0,
            "snapshot_id": (
                str(self.snapshot_manifest.get("snapshot_id"))
                if isinstance(self.snapshot_manifest, dict)
                else None
            ),
            "run_id": self.run_id,
            "model_type": self.config.model_type,
            "model_name": self.config.model_name,
            "training_profile": self.config.training_profile,
            "data_quality_report_hash": self.data_quality_report_hash,
            "data_quality_summary": self._json_safe(
                self.data_quality_report.get("summary", {})
                if isinstance(self.data_quality_report, dict)
                else {}
            ),
            "data_quality_threshold_breaches": self._json_safe(
                self.data_quality_report.get("threshold_breaches", {})
                if isinstance(self.data_quality_report, dict)
                else {}
            ),
            "top_missing_bar_symbols": self._json_safe(
                self.data_quality_report.get("top_missing_bar_symbols", [])
                if isinstance(self.data_quality_report, dict)
                else []
            ),
            "top_missing_bar_windows": self._json_safe(
                self.data_quality_report.get("top_missing_bar_windows", [])
                if isinstance(self.data_quality_report, dict)
                else []
            ),
            "symbol_quality": {
                "input_symbols": int(
                    round(float(self.training_metrics.get("symbol_quality_input_symbols", 0.0)))
                ),
                "selected_symbols": int(
                    round(float(self.training_metrics.get("symbol_quality_selected_symbols", 0.0)))
                ),
                "dropped_symbols": int(
                    round(float(self.training_metrics.get("symbol_quality_dropped_symbols", 0.0)))
                ),
                "selected_universe": selected_symbols,
                "dropped_list": dropped_symbols,
                "report": self._json_safe(self.training_metrics.get("symbol_quality_report", {})),
            },
            "market_session_filter": self._json_safe(
                self.training_metrics.get("market_session_filter_summary", {})
            ),
            "dataset_snapshot_bundle_path": dataset_bundle_manifest_path,
            "dataset_bundle_manifest_path": dataset_bundle_manifest_path,
            "dataset_bundle_hash": (
                str(self.dataset_snapshot_bundle_manifest.get("bundle_hash"))
                if isinstance(self.dataset_snapshot_bundle_manifest, dict)
                else (
                    str(self.snapshot_manifest.get("dataset_bundle_hash"))
                    if isinstance(self.snapshot_manifest, dict)
                    else None
                )
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
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def _persist_snapshot_artifacts(self, output_dir: Path) -> dict[str, Any]:
        """Persist snapshot lineage artifacts without requiring a trained model."""
        if self.snapshot_manifest is None:
            raise RuntimeError(
                "Snapshot manifest missing at artifact stage. Institutional mode requires snapshot lineage."
            )
        if self.data_quality_report is None:
            raise RuntimeError(
                "Data quality report missing at artifact stage. Institutional mode requires quality review."
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
        if self.dataset_snapshot_bundle_manifest_path is None:
            bundle_manifest_path, bundle_manifest = persist_dataset_snapshot_bundle(
                output_dir=output_dir,
                snapshot_manifest=self.snapshot_manifest,
                raw_ohlcv_data=self.data,
                development_frame=(
                    self.development_frame if self.development_frame is not None else self.features
                ),
                holdout_frame=self.holdout_frame,
                feature_names=self.feature_names,
                training_scope=_build_snapshot_training_scope(self.config),
                data_quality_report=self.data_quality_report,
                development_sample_weights=self.sample_weights,
                holdout_sample_weights=self.holdout_sample_weights,
                cv_splits=self._snapshot_bundle_cv_splits(),
                label_diagnostics=self.label_diagnostics,
                feature_selection_summary={
                    "binding": float(self.training_metrics.get("feature_selection_binding", 0.0)),
                    "development_binding": float(
                        self.training_metrics.get("feature_selection_development_binding", 0.0)
                    ),
                    "final_refit_binding": float(
                        self.training_metrics.get("feature_selection_final_refit_binding", 0.0)
                    ),
                    "development_initial_feature_count": float(
                        self.training_metrics.get(
                            "feature_selection_development_initial_feature_count",
                            self.training_metrics.get("feature_selection_initial_feature_count", 0.0),
                        )
                    ),
                    "development_selected_feature_count": float(
                        self.training_metrics.get(
                            "feature_selection_development_selected_feature_count",
                            self.training_metrics.get(
                                "feature_selection_selected_feature_count",
                                0.0,
                            ),
                        )
                    ),
                    "final_refit_initial_feature_count": float(
                        self.training_metrics.get(
                            "feature_selection_final_refit_initial_feature_count",
                            self.training_metrics.get(
                                "feature_selection_selected_feature_count",
                                0.0,
                            ),
                        )
                    ),
                    "final_refit_selected_feature_count": float(
                        self.training_metrics.get(
                            "feature_selection_final_refit_selected_feature_count",
                            self.training_metrics.get(
                                "feature_selection_selected_feature_count",
                                0.0,
                            ),
                        )
                    ),
                    "audit": self._json_safe(self.training_metrics.get("feature_selection_audit", {})),
                },
            )
            self.dataset_snapshot_bundle_manifest_path = bundle_manifest_path
            self.dataset_snapshot_bundle_manifest = bundle_manifest
        if isinstance(self.snapshot_manifest, dict):
            self.snapshot_manifest["dataset_bundle_manifest_path"] = (
                str(self.dataset_snapshot_bundle_manifest_path)
                if self.dataset_snapshot_bundle_manifest_path
                else None
            )
            self.snapshot_manifest["dataset_bundle_hash"] = (
                self.dataset_snapshot_bundle_manifest.get("bundle_hash")
                if isinstance(self.dataset_snapshot_bundle_manifest, dict)
                else None
            )
        if self.snapshot_manifest_path is not None and isinstance(self.snapshot_manifest, dict):
            self.snapshot_manifest_path.write_text(
                json.dumps(
                    self.snapshot_manifest,
                    indent=2,
                    ensure_ascii=True,
                    sort_keys=True,
                    default=str,
                ),
                encoding="utf-8",
            )

        review_payload = self._build_snapshot_review()
        review_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.config.model_name).strip("_")
        if not review_stem:
            review_stem = "snapshot"
        self.snapshot_review_path = output_dir / f"{review_stem}.snapshot_review.json"
        review_payload["snapshot_review_path"] = str(self.snapshot_review_path)
        self.snapshot_review_path.write_text(
            json.dumps(review_payload, indent=2, ensure_ascii=True, sort_keys=True, default=str),
            encoding="utf-8",
        )
        self.training_metrics["snapshot_review"] = review_payload
        self.training_metrics["snapshot_review_ready"] = bool(review_payload.get("ready", False))
        self.training_metrics["snapshot_review_failed_checks"] = list(
            review_payload.get("failed_checks", [])
        )
        return review_payload

    def _persist_snapshot_review_only(self, output_dir: Path) -> dict[str, Any]:
        """Persist a fast-fail Task 1 review artifact before feature materialization."""
        output_dir.mkdir(parents=True, exist_ok=True)
        review_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.config.model_name).strip("_")
        if not review_stem:
            review_stem = "snapshot"

        if self.data_quality_report is not None:
            preflight_quality_path = output_dir / f"{review_stem}.preflight_quality.json"
            preflight_quality_path.write_text(
                json.dumps(
                    self.data_quality_report,
                    indent=2,
                    ensure_ascii=True,
                    sort_keys=True,
                    default=str,
                ),
                encoding="utf-8",
            )
            self.data_quality_report_path = preflight_quality_path

        review_payload = self._build_snapshot_review()
        review_payload["preflight_only"] = True
        self.snapshot_review_path = output_dir / f"{review_stem}.snapshot_review.json"
        review_payload["snapshot_review_path"] = str(self.snapshot_review_path)
        self.snapshot_review_path.write_text(
            json.dumps(review_payload, indent=2, ensure_ascii=True, sort_keys=True, default=str),
            encoding="utf-8",
        )
        self.training_metrics["snapshot_review"] = review_payload
        self.training_metrics["snapshot_review_ready"] = bool(review_payload.get("ready", False))
        self.training_metrics["snapshot_review_failed_checks"] = list(
            review_payload.get("failed_checks", [])
        )
        return review_payload

    def _build_snapshot_only_results(
        self,
        review_payload: dict[str, Any],
        duration: float,
    ) -> dict[str, Any]:
        """Build a stable result payload for snapshot-only execution paths."""
        return {
            "success": bool(review_payload.get("ready", False)),
            "snapshot_only": True,
            "preflight_only": bool(review_payload.get("preflight_only", False)),
            "run_id": self.run_id,
            "model_path": None,
            "model_type": self.config.model_type,
            "model_name": self.config.model_name,
            "training_duration_seconds": duration,
            "cv_results": [],
            "validation_results": {},
            "training_metrics": self.training_metrics,
            "passed_validation_gates": None,
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
            "data_quality_report_passed": (
                bool(self.data_quality_report.get("passed", False))
                if isinstance(self.data_quality_report, dict)
                else None
            ),
            "feature_schema_version": (
                str(self.snapshot_manifest.get("feature_schema_version"))
                if isinstance(self.snapshot_manifest, dict)
                else None
            ),
            "dataset_snapshot_bundle_path": (
                str(self.dataset_snapshot_bundle_manifest_path)
                if self.dataset_snapshot_bundle_manifest_path is not None
                else None
            ),
            "dataset_bundle_hash": (
                str(self.dataset_snapshot_bundle_manifest.get("bundle_hash"))
                if isinstance(self.dataset_snapshot_bundle_manifest, dict)
                else (
                    str(self.snapshot_manifest.get("dataset_bundle_hash"))
                    if isinstance(self.snapshot_manifest, dict)
                    else None
                )
            ),
            "snapshot_review_path": (
                str(self.snapshot_review_path) if self.snapshot_review_path is not None else None
            ),
            "snapshot_review": review_payload,
            "label_diagnostics": self.label_diagnostics,
            "nested_cv_trace": self.nested_cv_trace,
            "replay_manifest_path": None,
            "promotion_package_path": None,
            "artifacts_path": None,
            "run_event_index_path": (
                str(self.run_event_index_path) if self.run_event_index_path is not None else None
            ),
            "model_card": {},
            "deployment_plan": {},
            "pre_promotion_checklist": {},
        }

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
            self._persist_snapshot_artifacts(output_dir)
            self.replay_manifest_path = self._write_replay_manifest(output_dir, model_path)
            self.promotion_package_path = self._write_promotion_package(
                output_dir=output_dir,
                model_path=model_path,
                replay_manifest_path=self.replay_manifest_path,
            )

        # Save training artifacts
        if self.config.save_artifacts:
            artifacts = {
                "run_id": self.run_id,
                "config": self.config.__dict__,
                "cv_results": self.cv_results,
                "validation_results": self.validation_results,
                "training_metrics": {
                    k: v for k, v in self.training_metrics.items() if not isinstance(v, np.ndarray)
                },
                "feature_names": self.features.columns.tolist(),
                "snapshot_id": self.snapshot_manifest.get("snapshot_id"),
                "snapshot_manifest": self.snapshot_manifest,
                "snapshot_manifest_path": str(self.snapshot_manifest_path),
                "dataset_snapshot_bundle_path": (
                    str(self.dataset_snapshot_bundle_manifest_path)
                    if self.dataset_snapshot_bundle_manifest_path
                    else None
                ),
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
                "run_event_index_path": (
                    str(self.run_event_index_path) if self.run_event_index_path else None
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            artifacts_path = output_dir / f"{self.config.model_name}_artifacts.json"
            with open(artifacts_path, "w") as f:
                json.dump(artifacts, f, indent=2, default=str)

            self.artifacts_path = artifacts_path
            self.logger.info(f"Artifacts saved to: {artifacts_path}")

            # Save meta-model if trained
            if self.meta_model is not None:
                meta_path = output_dir / f"{self.config.model_name}_meta.pkl"
                with open(meta_path, "wb") as f:
                    pickle.dump(self.meta_model, f)
                self.logger.info(f"Meta-model saved to: {meta_path}")

            if self.expected_edge_model is not None:
                expected_edge_path = output_dir / f"{self.config.model_name}_expected_edge.pkl"
                with open(expected_edge_path, "wb") as f:
                    pickle.dump(self.expected_edge_model, f)
                self.logger.info(f"Expected-edge model saved to: {expected_edge_path}")

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
            self.logger.info(
                f"Trained {model_type}: Sharpe={result['training_metrics'].get('mean_sharpe', 0):.4f}"
            )

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
        model_name = (
            self.config.model_name
            or f"ensemble_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
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


def _payload_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Return training_metrics when available, otherwise an empty mapping."""
    metrics = payload.get("training_metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _payload_metric(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Read a numeric metric from either training_metrics or a flattened benchmark row."""
    metrics = _payload_metrics(payload)
    if key in metrics:
        return _metric_as_float(metrics, key, default)
    return _metric_as_float(payload, key, default)


def _payload_bool(payload: dict[str, Any], key: str, default: bool = False) -> bool:
    """Read a boolean value from either training_metrics or a flattened benchmark row."""
    metrics = _payload_metrics(payload)
    if key in metrics:
        return bool(metrics.get(key, default))
    return bool(payload.get(key, default))


def _payload_list(payload: dict[str, Any], key: str) -> list[Any]:
    """Read a list payload from either training_metrics or a flattened benchmark row."""
    metrics = _payload_metrics(payload)
    value = metrics.get(key) if key in metrics else payload.get(key)
    return list(value) if isinstance(value, list) else []


def _snapshot_comparison_key(payload: dict[str, Any]) -> str:
    """Choose the strongest available immutable snapshot identity for comparisons."""
    for key in ("dataset_bundle_hash", "data_quality_report_hash", "snapshot_id"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return "unversioned_snapshot"


def _extract_validation_layer_states(payload: dict[str, Any]) -> dict[str, bool]:
    """Extract normalized layer pass/fail states from result payload."""
    if not isinstance(payload, dict):
        return {}
    validation_results = payload.get("validation_results", {})
    if not isinstance(validation_results, dict):
        validation_results = {}
    raw_layers = validation_results.get("layers", {})
    if not isinstance(raw_layers, dict):
        raw_layers = {}
    layer_states: dict[str, bool] = {}
    for layer_name in ("model_utility", "execution_robustness", "cross_symbol_robustness"):
        layer_payload = raw_layers.get(layer_name, {})
        if isinstance(layer_payload, dict):
            layer_states[layer_name] = bool(layer_payload.get("passed", False))
    return layer_states


def _build_pre_promotion_checklist(payload: dict[str, Any]) -> dict[str, Any]:
    """Build the work-plan checklist used to authorize promotion runs."""
    layer_states = _extract_validation_layer_states(payload)
    validation_layers_ready = (
        all(layer_states.values()) if layer_states else bool(payload.get("success", False))
    )
    snapshot_id = str(payload.get("snapshot_id") or "").strip()
    dataset_bundle_hash = str(payload.get("dataset_bundle_hash") or "").strip()
    data_quality_report_hash = str(payload.get("data_quality_report_hash") or "").strip()
    metrics = _payload_metrics(payload)
    expected_edge_reason = str(
        metrics.get("expected_edge_policy_reason", payload.get("expected_edge_policy_reason", ""))
    ).strip()
    expected_edge_trained = bool(
        _payload_metric(payload, "expected_edge_policy_enabled", 0.0) > 0.0
        or expected_edge_reason == "trained"
    )
    dropped_symbols = max(
        int(round(_payload_metric(payload, "symbol_quality_dropped_symbols", 0.0))),
        len(_payload_list(payload, "symbol_quality_dropped_list")),
    )
    expected_edge_candidate_floor_payload = metrics.get(
        "expected_edge_candidate_floor",
        payload.get("expected_edge_candidate_floor", {}),
    )
    expected_edge_candidate_count = _payload_metric(
        payload,
        "expected_edge_candidate_floor_candidate_count",
        np.nan,
    )
    if not np.isfinite(expected_edge_candidate_count) and isinstance(
        expected_edge_candidate_floor_payload, dict
    ):
        expected_edge_candidate_count = float(
            expected_edge_candidate_floor_payload.get("candidate_count", 0.0)
        )
    if not np.isfinite(expected_edge_candidate_count):
        expected_edge_candidate_count = 0.0
    holdout_trade_count = _payload_metric(
        payload,
        "effective_holdout_trade_count_metric",
        np.nan,
    )
    if not np.isfinite(holdout_trade_count):
        holdout_trade_count = _payload_metric(payload, "holdout_trade_count", 0.0)
    holdout_active_signal_rate = _payload_metric(
        payload,
        "effective_holdout_active_signal_rate_metric",
        np.nan,
    )
    if not np.isfinite(holdout_active_signal_rate):
        holdout_active_signal_rate = _payload_metric(payload, "holdout_active_signal_rate", 0.0)

    checks = {
        "snapshot_lineage_present": {
            "passed": bool(snapshot_id),
            "actual": bool(snapshot_id),
            "threshold": True,
            "comparator": "is",
        },
        "dataset_bundle_hash_present": {
            "passed": bool(dataset_bundle_hash),
            "actual": bool(dataset_bundle_hash),
            "threshold": True,
            "comparator": "is",
        },
        "data_quality_passed": {
            "passed": _payload_bool(payload, "data_quality_report_passed", False),
            "actual": _payload_bool(payload, "data_quality_report_passed", False),
            "threshold": True,
            "comparator": "is",
        },
        "validation_layers_ready": {
            "passed": bool(validation_layers_ready),
            "actual": bool(validation_layers_ready),
            "threshold": True,
            "comparator": "is",
        },
        "no_silent_symbol_drop": {
            "passed": int(dropped_symbols) == 0,
            "actual": int(dropped_symbols),
            "threshold": 0,
            "comparator": "<=",
        },
        "expected_edge_trained": {
            "passed": bool(expected_edge_trained),
            "actual": bool(expected_edge_trained),
            "threshold": True,
            "comparator": "is",
            "reason": expected_edge_reason or None,
        },
        "expected_edge_candidate_floor_positive": {
            "passed": expected_edge_candidate_count > 0.0,
            "actual": expected_edge_candidate_count,
            "threshold": 0.0,
            "comparator": ">",
            "reason": expected_edge_reason or None,
        },
        "mean_trade_count": {
            "passed": _payload_metric(payload, "mean_trade_count", 0.0)
            >= PRE_PROMOTION_WORK_PLAN_THRESHOLDS["mean_trade_count_min"],
            "actual": _payload_metric(payload, "mean_trade_count", 0.0),
            "threshold": PRE_PROMOTION_WORK_PLAN_THRESHOLDS["mean_trade_count_min"],
            "comparator": ">=",
        },
        "mean_risk_adjusted_score": {
            "passed": _payload_metric(payload, "mean_risk_adjusted_score", -1.0)
            > PRE_PROMOTION_WORK_PLAN_THRESHOLDS["mean_risk_adjusted_score_min"],
            "actual": _payload_metric(payload, "mean_risk_adjusted_score", -1.0),
            "threshold": PRE_PROMOTION_WORK_PLAN_THRESHOLDS["mean_risk_adjusted_score_min"],
            "comparator": ">",
        },
        "holdout_trade_count_positive": {
            "passed": holdout_trade_count > 0.0,
            "actual": holdout_trade_count,
            "threshold": 0.0,
            "comparator": ">",
        },
        "holdout_active_signal_rate_positive": {
            "passed": holdout_active_signal_rate > 0.0,
            "actual": holdout_active_signal_rate,
            "threshold": 0.0,
            "comparator": ">",
        },
        "holdout_max_drawdown": {
            "passed": _payload_metric(payload, "holdout_max_drawdown", 1.0)
            <= PRE_PROMOTION_WORK_PLAN_THRESHOLDS["holdout_max_drawdown_max"],
            "actual": _payload_metric(payload, "holdout_max_drawdown", 1.0),
            "threshold": PRE_PROMOTION_WORK_PLAN_THRESHOLDS["holdout_max_drawdown_max"],
            "comparator": "<=",
        },
        "holdout_symbol_sharpe_p25": {
            "passed": _payload_metric(payload, "holdout_symbol_sharpe_p25", -1.0)
            >= PRE_PROMOTION_WORK_PLAN_THRESHOLDS["holdout_symbol_sharpe_p25_min"],
            "actual": _payload_metric(payload, "holdout_symbol_sharpe_p25", -1.0),
            "threshold": PRE_PROMOTION_WORK_PLAN_THRESHOLDS["holdout_symbol_sharpe_p25_min"],
            "comparator": ">=",
        },
        "pbo": {
            "passed": _payload_metric(payload, "pbo", 1.0)
            <= PRE_PROMOTION_WORK_PLAN_THRESHOLDS["pbo_max"],
            "actual": _payload_metric(payload, "pbo", 1.0),
            "threshold": PRE_PROMOTION_WORK_PLAN_THRESHOLDS["pbo_max"],
            "comparator": "<=",
        },
        "white_reality_pvalue": {
            "passed": _payload_metric(payload, "white_reality_pvalue", 1.0)
            <= PRE_PROMOTION_WORK_PLAN_THRESHOLDS["white_reality_pvalue_max"],
            "actual": _payload_metric(payload, "white_reality_pvalue", 1.0),
            "threshold": PRE_PROMOTION_WORK_PLAN_THRESHOLDS["white_reality_pvalue_max"],
            "comparator": "<=",
        },
    }
    failed_checks = [name for name, state in checks.items() if not bool(state.get("passed", False))]

    return {
        "schema_version": PRE_PROMOTION_CHECKLIST_SCHEMA_VERSION,
        "ready": not failed_checks,
        "failed_checks": failed_checks,
        "model_name": str(payload.get("model_name") or ""),
        "model_type": str(payload.get("model_type") or ""),
        "training_profile": str(
            payload.get("training_profile") or metrics.get("training_profile") or ""
        ),
        "primary_label_horizon": int(
            max(_payload_metric(payload, "primary_label_horizon", 0.0), 0.0)
        ),
        "snapshot_id": snapshot_id or None,
        "dataset_bundle_hash": dataset_bundle_hash or None,
        "data_quality_report_hash": data_quality_report_hash or None,
        "checks": checks,
        "validation_layers": layer_states,
    }


def _build_training_comparison_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build explicit baseline-vs-challenger deltas on the same snapshot identity."""
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        try:
            horizon = int(float(row.get("primary_label_horizon", 0)))
        except (TypeError, ValueError):
            horizon = 0
        grouped.setdefault((horizon, _snapshot_comparison_key(row)), []).append(row)

    summary: list[dict[str, Any]] = []
    for (horizon, snapshot_key), group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        ordered = sorted(
            group_rows,
            key=lambda row: float(row.get("governance_score", -1e9)),
            reverse=True,
        )
        if not ordered:
            continue
        baseline = ordered[0]
        baseline_view = {
            "run_id": str(baseline.get("run_id") or ""),
            "model_name": str(baseline.get("model_name") or ""),
            "model_type": str(baseline.get("model_type") or ""),
            "governance_score": float(baseline.get("governance_score", 0.0)),
            "pre_promotion_ready": bool(baseline.get("pre_promotion_ready", 0.0)),
        }
        challengers = []
        for challenger in ordered[1:]:
            challengers.append(
                {
                    "run_id": str(challenger.get("run_id") or ""),
                    "model_name": str(challenger.get("model_name") or ""),
                    "model_type": str(challenger.get("model_type") or ""),
                    "governance_score_delta": float(challenger.get("governance_score", 0.0))
                    - float(baseline.get("governance_score", 0.0)),
                    "mean_trade_count_delta": float(challenger.get("mean_trade_count", 0.0))
                    - float(baseline.get("mean_trade_count", 0.0)),
                    "mean_risk_adjusted_score_delta": float(
                        challenger.get("mean_risk_adjusted_score", 0.0)
                    )
                    - float(baseline.get("mean_risk_adjusted_score", 0.0)),
                    "holdout_sharpe_delta": float(challenger.get("holdout_sharpe", 0.0))
                    - float(baseline.get("holdout_sharpe", 0.0)),
                    "holdout_max_drawdown_delta": float(challenger.get("holdout_max_drawdown", 0.0))
                    - float(baseline.get("holdout_max_drawdown", 0.0)),
                    "pbo_delta": float(challenger.get("pbo", 0.0))
                    - float(baseline.get("pbo", 0.0)),
                    "white_reality_pvalue_delta": float(challenger.get("white_reality_pvalue", 0.0))
                    - float(baseline.get("white_reality_pvalue", 0.0)),
                    "pre_promotion_ready": bool(challenger.get("pre_promotion_ready", 0.0)),
                    "failed_checks": list(challenger.get("pre_promotion_failed_checks", [])),
                }
            )
        summary.append(
            {
                "comparison_group_key": f"h{horizon}:{snapshot_key[:16]}",
                "primary_label_horizon": int(max(horizon, 0)),
                "snapshot_id": baseline.get("snapshot_id") or None,
                "dataset_bundle_hash": baseline.get("dataset_bundle_hash") or None,
                "data_quality_report_hash": baseline.get("data_quality_report_hash") or None,
                "training_profile": str(baseline.get("training_profile") or ""),
                "baseline": baseline_view,
                "challengers": challengers,
            }
        )
    return summary


def _load_latest_training_matrix_rows(output_dir: Path) -> tuple[list[dict[str, Any]], Path | None]:
    """Load the newest persisted training matrix for promotion preflight checks."""
    bench_dir = Path(output_dir) / "benchmarks"
    if not bench_dir.exists():
        return [], None
    for matrix_path in sorted(bench_dir.glob("training_matrix_*.json"), reverse=True):
        try:
            payload = json.loads(matrix_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = payload.get("rows", [])
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)], matrix_path
    return [], None


def _load_snapshot_bundle_identity(bundle_path: str | Path | None) -> dict[str, str]:
    """Extract immutable snapshot identifiers from a dataset bundle manifest."""
    if not bundle_path:
        return {}
    path = Path(bundle_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    snapshot_manifest = payload.get("snapshot_manifest", {})
    if not isinstance(snapshot_manifest, dict):
        snapshot_manifest = {}
    return {
        "snapshot_id": str(
            payload.get("snapshot_id") or snapshot_manifest.get("snapshot_id") or ""
        ),
        "dataset_bundle_hash": str(payload.get("bundle_hash") or ""),
        "data_quality_report_hash": str(
            snapshot_manifest.get("data_quality_report_hash")
            or payload.get("data_quality_report_hash")
            or ""
        ),
    }


def _match_pre_promotion_rows(
    rows: list[dict[str, Any]],
    *,
    model_type: str,
    primary_label_horizon: int,
    bundle_identity: dict[str, str],
) -> list[dict[str, Any]]:
    """Filter persisted benchmark rows down to the relevant research baseline candidates."""
    filtered = [
        row
        for row in rows
        if str(row.get("model_type") or "") == str(model_type)
        and int(float(row.get("primary_label_horizon", 0) or 0)) == int(primary_label_horizon)
    ]
    if not filtered:
        return []

    research_rows = [
        row for row in filtered if str(row.get("training_profile") or "") == "research"
    ]
    if research_rows:
        filtered = research_rows

    for key in ("dataset_bundle_hash", "snapshot_id", "data_quality_report_hash"):
        expected = str(bundle_identity.get(key) or "").strip()
        if not expected:
            continue
        exact = [row for row in filtered if str(row.get(key) or "").strip() == expected]
        if exact:
            filtered = exact
            break

    filtered.sort(
        key=lambda row: (
            float(row.get("pre_promotion_ready", 0.0)),
            float(row.get("governance_score", -1e9)),
            -float(row.get("rank", 1e9)),
        ),
        reverse=True,
    )
    return filtered


def _evaluate_promotion_precheck(
    *,
    output_dir: Path,
    model_type: str,
    primary_label_horizon: int,
    dataset_snapshot_bundle_path: str,
    bypass: bool = False,
) -> dict[str, Any]:
    """Decide whether a strict-snapshot promotion run is allowed to start."""
    bundle_identity = _load_snapshot_bundle_identity(dataset_snapshot_bundle_path)
    rows, matrix_path = _load_latest_training_matrix_rows(output_dir)
    if bypass:
        return {
            "blocked": False,
            "bypassed": True,
            "reason": "operator_bypass",
            "matrix_path": str(matrix_path) if matrix_path is not None else None,
            "bundle_identity": bundle_identity,
        }
    if not rows:
        return {
            "blocked": True,
            "bypassed": False,
            "reason": "missing_training_matrix",
            "message": "No prior training matrix found for promotion precheck.",
            "matrix_path": None,
            "bundle_identity": bundle_identity,
        }

    matched_rows = _match_pre_promotion_rows(
        rows,
        model_type=model_type,
        primary_label_horizon=primary_label_horizon,
        bundle_identity=bundle_identity,
    )
    if not matched_rows:
        return {
            "blocked": True,
            "bypassed": False,
            "reason": "no_matching_research_candidate",
            "message": "No matching research candidate was found for the requested promotion snapshot.",
            "matrix_path": str(matrix_path) if matrix_path is not None else None,
            "bundle_identity": bundle_identity,
        }

    candidate = matched_rows[0]
    checklist = candidate.get("pre_promotion_checklist", {})
    if not isinstance(checklist, dict) or not checklist:
        checklist = _build_pre_promotion_checklist(candidate)
    blocked = not bool(checklist.get("ready", False))
    return {
        "blocked": blocked,
        "bypassed": False,
        "reason": "failed_checklist" if blocked else "passed",
        "message": (
            "Research candidate did not clear the pre-promotion checklist."
            if blocked
            else "Research candidate cleared the pre-promotion checklist."
        ),
        "matrix_path": str(matrix_path) if matrix_path is not None else None,
        "bundle_identity": bundle_identity,
        "candidate": candidate,
        "checklist": checklist,
    }


def _governance_score(result: dict[str, Any]) -> float:
    """Composite score used for benchmark ranking and champion selection."""
    metrics = result.get("training_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    score = 0.0
    score += _metric_as_float(metrics, "mean_risk_adjusted_score", -1.0) * 1.0
    score += (
        _metric_as_float(metrics, "holdout_sharpe", _metric_as_float(metrics, "mean_sharpe", 0.0))
        * 0.6
    )
    score += (
        _metric_as_float(
            metrics,
            "holdout_worst_regime_sharpe",
            _metric_as_float(metrics, "holdout_sharpe", 0.0),
        )
        * 0.35
    )
    score += (
        _metric_as_float(metrics, "deflated_sharpe", _metric_as_float(metrics, "mean_sharpe", 0.0))
        * 0.4
    )
    score += _metric_as_float(metrics, "white_reality_stat", 0.0) * 0.15
    score -= (
        _metric_as_float(
            metrics, "holdout_max_drawdown", _metric_as_float(metrics, "mean_max_drawdown", 1.0)
        )
        * 0.75
    )
    score -= _metric_as_float(metrics, "mean_symbol_concentration_hhi", 0.0) * 0.35
    score -= _metric_as_float(metrics, "holdout_symbol_concentration_hhi", 0.0) * 0.20
    score -= _metric_as_float(metrics, "pbo", 1.0) * 0.25
    score -= _metric_as_float(metrics, "white_reality_pvalue", 1.0) * 0.30
    layer_states = _extract_validation_layer_states(result)
    if layer_states:
        layer_failures = sum(1 for passed in layer_states.values() if not passed)
        score -= float(layer_failures) * 1.5
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
        layer_states = _extract_validation_layer_states(result)
        pre_promotion_checklist = result.get("pre_promotion_checklist", {})
        if not isinstance(pre_promotion_checklist, dict) or not pre_promotion_checklist:
            pre_promotion_checklist = _build_pre_promotion_checklist(result)
        deployment_plan = result.get("deployment_plan", {})
        if not isinstance(deployment_plan, dict):
            deployment_plan = {}
        all_layers_passed = (
            all(layer_states.values()) if layer_states else bool(result.get("success", False))
        )
        ready_for_production = bool(
            deployment_plan.get("ready_for_production", False) and all_layers_passed
        )
        raw_primary_horizon = result.get(
            "primary_label_horizon", metrics.get("primary_label_horizon", 0)
        )
        try:
            primary_horizon = int(float(raw_primary_horizon))
        except (TypeError, ValueError):
            primary_horizon = 0
        rows.append(
            {
                "model_type": str(model_type),
                "primary_label_horizon": int(max(primary_horizon, 0)),
                "training_profile": str(
                    result.get("training_profile") or metrics.get("training_profile") or ""
                ),
                "run_id": str(result.get("run_id") or ""),
                "snapshot_id": str(result.get("snapshot_id") or ""),
                "data_quality_report_hash": str(result.get("data_quality_report_hash") or ""),
                "data_quality_report_passed": (
                    1.0
                    if bool(
                        result.get(
                            "data_quality_report_passed",
                            metrics.get("data_quality_report_passed", False),
                        )
                    )
                    else 0.0
                ),
                "dataset_bundle_hash": str(result.get("dataset_bundle_hash") or ""),
                "success": bool(result.get("success", False)),
                "governance_score": _governance_score(result),
                "mean_accuracy": _metric_as_float(metrics, "mean_accuracy", 0.0),
                "mean_sharpe": _metric_as_float(metrics, "mean_sharpe", 0.0),
                "mean_risk_adjusted_score": _metric_as_float(
                    metrics, "mean_risk_adjusted_score", -1.0
                ),
                "deflated_sharpe": _metric_as_float(metrics, "deflated_sharpe", 0.0),
                "deflated_sharpe_p_value": _metric_as_float(
                    metrics, "deflated_sharpe_p_value", 1.0
                ),
                "pbo": _metric_as_float(metrics, "pbo", 1.0),
                "white_reality_stat": _metric_as_float(metrics, "white_reality_stat", 0.0),
                "white_reality_pvalue": _metric_as_float(metrics, "white_reality_pvalue", 1.0),
                "holdout_sharpe": _metric_as_float(metrics, "holdout_sharpe", 0.0),
                "holdout_max_drawdown": _metric_as_float(metrics, "holdout_max_drawdown", 1.0),
                "holdout_regime_shift": _metric_as_float(metrics, "holdout_regime_shift", 0.0),
                "holdout_symbol_sharpe_p25": _metric_as_float(
                    metrics, "holdout_symbol_sharpe_p25", 0.0
                ),
                "layer_model_utility_pass": (
                    1.0 if layer_states.get("model_utility", False) else 0.0
                ),
                "layer_execution_robustness_pass": (
                    1.0 if layer_states.get("execution_robustness", False) else 0.0
                ),
                "layer_cross_symbol_robustness_pass": (
                    1.0 if layer_states.get("cross_symbol_robustness", False) else 0.0
                ),
                "layer_all_passed": 1.0 if all_layers_passed else 0.0,
                "ready_for_production": 1.0 if ready_for_production else 0.0,
                "holdout_worst_regime_sharpe": _metric_as_float(
                    metrics, "holdout_worst_regime_sharpe", 0.0
                ),
                "mean_symbol_concentration_hhi": _metric_as_float(
                    metrics, "mean_symbol_concentration_hhi", 0.0
                ),
                "holdout_symbol_concentration_hhi": _metric_as_float(
                    metrics, "holdout_symbol_concentration_hhi", 0.0
                ),
                "mean_trade_count": _metric_as_float(metrics, "mean_trade_count", 0.0),
                "expected_edge_policy_trained": (
                    1.0
                    if (
                        _metric_as_float(metrics, "expected_edge_policy_enabled", 0.0) > 0.0
                        or str(metrics.get("expected_edge_policy_reason", "")).strip() == "trained"
                    )
                    else 0.0
                ),
                "symbol_quality_dropped_symbols": _metric_as_float(
                    metrics, "symbol_quality_dropped_symbols", 0.0
                ),
                "pre_promotion_ready": (
                    1.0 if bool(pre_promotion_checklist.get("ready", False)) else 0.0
                ),
                "pre_promotion_failed_check_count": float(
                    len(pre_promotion_checklist.get("failed_checks", []))
                ),
                "pre_promotion_failed_checks": list(
                    pre_promotion_checklist.get("failed_checks", [])
                ),
                "pre_promotion_checklist": pre_promotion_checklist,
                "registry_version_id": str(result.get("registry_version_id") or ""),
                "model_name": str(result.get("model_name") or model_type),
                "model_path": str(result.get("model_path") or ""),
                "promotion_package_path": str(result.get("promotion_package_path") or ""),
                "replay_manifest_path": str(result.get("replay_manifest_path") or ""),
                "validation_layers": layer_states,
                "deployment_plan": deployment_plan,
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
        "comparison_summary": _build_training_comparison_summary(rows),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=True, default=str)

    flat_rows = []
    for row in rows:
        flat_row = {
            k: v
            for k, v in row.items()
            if k
            not in {
                "deployment_plan",
                "validation_layers",
                "pre_promotion_checklist",
                "pre_promotion_failed_checks",
            }
        }
        flat_rows.append(flat_row)
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    return json_path, csv_path


def _row_is_promotion_ready(row: dict[str, Any]) -> bool:
    """Return True when a benchmark row is eligible for promotion."""
    if not bool(row.get("success", False)):
        return False
    if not str(row.get("registry_version_id", "")).strip():
        return False
    if not str(row.get("promotion_package_path", "")).strip():
        return False
    deployment_plan = row.get("deployment_plan", {})
    if not isinstance(deployment_plan, dict):
        return False
    if not bool(deployment_plan.get("ready_for_production", False)):
        return False
    layers = row.get("validation_layers", {})
    if isinstance(layers, dict) and layers:
        if not all(bool(v) for v in layers.values()):
            return False
    return True


def _build_horizon_leaderboards(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build per-horizon leaderboard with champion/challenger candidates."""
    horizon_groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        try:
            horizon = int(float(row.get("primary_label_horizon", 0)))
        except (TypeError, ValueError):
            horizon = 0
        horizon_groups.setdefault(max(horizon, 0), []).append(row)

    payload: dict[str, dict[str, Any]] = {}
    for horizon, group_rows in sorted(horizon_groups.items(), key=lambda kv: kv[0]):
        sorted_rows = sorted(
            group_rows,
            key=lambda item: float(item.get("governance_score", -1e9)),
            reverse=True,
        )
        eligible = [row for row in sorted_rows if _row_is_promotion_ready(row)]
        champion = eligible[0] if eligible else None
        challenger = eligible[1] if len(eligible) > 1 else None
        payload[str(horizon)] = {
            "eligible_count": int(len(eligible)),
            "champion": champion,
            "challenger": challenger,
            "top_rows": [
                {
                    "model_type": str(row.get("model_type", "")),
                    "run_id": str(row.get("run_id", "")),
                    "governance_score": float(row.get("governance_score", -1e9)),
                    "registry_version_id": str(row.get("registry_version_id", "")),
                    "ready_for_production": bool(_row_is_promotion_ready(row)),
                }
                for row in sorted_rows[:8]
            ],
        }
    return payload


def _auto_select_champion_and_challenger(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Promote best valid candidate and persist champion/challenger snapshot."""
    registry_root = Path(output_dir) / "registry"
    models_root = Path(output_dir)
    current_entries = load_registry_entries(registry_root)

    eligible = [row for row in rows if _row_is_promotion_ready(row)]
    horizon_leaderboards = _build_horizon_leaderboards(rows)

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
                    json.dump(
                        canary_payload, f, indent=2, ensure_ascii=True, sort_keys=True, default=str
                    )

    snapshot = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "promoted": bool(promoted),
        "pointer_path": str(pointer_path) if pointer_path else None,
        "canary_plan_path": str(canary_plan_path) if canary_plan_path else None,
        "champion": champion,
        "challenger": challenger,
        "horizon_leaderboards": horizon_leaderboards,
        "eligible_count": len(eligible),
        "registry_entries_seen": len(current_entries),
        "selection_policy": {
            "requires_validation_pass": True,
            "requires_layered_promotion_gates": True,
            "horizon_specific_leaderboards": True,
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
    logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine.base.Engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)

    logger.info("=" * 80)
    logger.info("ALPHATRADE MODEL TRAINING PIPELINE")
    logger.info("=" * 80)

    try:
        if getattr(args, "gpu", False) or getattr(args, "use_gpu", False):
            logger.warning(
                "`--gpu/--use-gpu` is deprecated; use `--require-model-gpu` instead."
            )
        if getattr(args, "no_database", False) or getattr(args, "no_redis_cache", False):
            raise ValueError("Institutional mode does not allow disabling PostgreSQL or Redis.")
        if getattr(args, "disable_nested_walk_forward", False):
            raise ValueError("Institutional mode requires nested walk-forward optimization trace.")

        replay_bundle: dict[str, Any] | None = None
        replay_config: dict[str, Any] = {}
        replay_snapshot_bundle_path = ""
        replay_manifest_arg = getattr(args, "replay_manifest", None)
        if replay_manifest_arg:
            replay_bundle = _load_replay_manifest(Path(replay_manifest_arg))
            replay_config = replay_bundle["training_config"]
            replay_manifest_payload = replay_bundle.get("manifest", {})
            if isinstance(replay_manifest_payload, dict):
                replay_snapshot_bundle_path = str(
                    replay_manifest_payload.get("dataset_snapshot_bundle_path") or ""
                ).strip()
                if not replay_snapshot_bundle_path and isinstance(
                    replay_manifest_payload.get("snapshot_manifest"), dict
                ):
                    replay_snapshot_bundle_path = str(
                        replay_manifest_payload["snapshot_manifest"].get(
                            "dataset_bundle_manifest_path"
                        )
                        or ""
                    ).strip()
            logger.info(f"Replay mode enabled from manifest: {replay_bundle['manifest_path']}")

        training_profile = _normalize_training_profile(
            replay_config.get("training_profile") or getattr(args, "training_profile", "promotion")
        )
        if getattr(args, "no_shap", False) and training_profile == "promotion":
            raise ValueError("Promotion profile requires SHAP explainability.")
        if getattr(args, "disable_meta_labeling", False) and training_profile == "promotion":
            raise ValueError("Promotion profile requires meta-labeling.")

        requested_model = str(
            replay_config.get("model_type") or getattr(args, "model", DEFAULT_PRIMARY_MODEL)
        )
        model_list = (
            [
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
            if requested_model == "all"
            else (
                ["lightgbm", "tcn"]
                if requested_model == PRIMARY_CHALLENGER_MODEL_ALIAS
                else [requested_model]
            )
        )
        if training_profile == "research" and requested_model == "all":
            logger.warning(
                "Research profile is intended for focused candidate runs. "
                "`--model all` will still execute the full broad sweep."
            )

        global_seed = int(replay_config.get("seed", getattr(args, "seed", 42)))
        set_global_seed(global_seed)

        require_gpu_requested = bool(
            replay_config.get("require_gpu", False)
            or getattr(args, "require_gpu", False)
            or getattr(args, "gpu", False)
            or getattr(args, "use_gpu", False)
        )
        require_feature_gpu_requested = bool(
            replay_config.get("require_feature_gpu", False)
            or getattr(args, "require_feature_gpu", False)
        )
        resolved_use_gpu = _verify_gpu_stack(model_list)
        if require_gpu_requested and not resolved_use_gpu:
            raise RuntimeError(
                "GPU acceleration was required, but the requested training stack is not GPU-ready."
            )

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
        configured_name = str(getattr(args, "name", "") or "").strip()
        profile_defaults = TRAINING_PROFILE_DEFAULTS[training_profile]

        def _cfg_value(key: str, fallback: Any) -> Any:
            value = replay_config.get(key, fallback)
            return fallback if value is None else value

        cli_symbols = list(getattr(args, "symbols", []) or [])
        symbols_file_arg = getattr(args, "symbols_file", None)
        if symbols_file_arg:
            cli_symbols.extend(_load_symbols_file(Path(symbols_file_arg)))

        raw_primary_horizon_sweep = _cfg_value(
            "primary_horizon_sweep",
            getattr(args, "primary_horizon_sweep", []),
        )
        if isinstance(raw_primary_horizon_sweep, (list, tuple, set)):
            primary_horizon_sweep = sorted(
                {
                    int(h)
                    for h in raw_primary_horizon_sweep
                    if isinstance(h, (int, float, np.integer, np.floating)) and int(h) > 0
                }
            )
        elif raw_primary_horizon_sweep:
            try:
                parsed_h = int(raw_primary_horizon_sweep)
                primary_horizon_sweep = [parsed_h] if parsed_h > 0 else []
            except (TypeError, ValueError):
                primary_horizon_sweep = []
        else:
            primary_horizon_sweep = []

        base_primary_horizon = int(
            _cfg_value("primary_label_horizon", getattr(args, "primary_horizon", 5))
        )
        if base_primary_horizon <= 0:
            base_primary_horizon = 5
        if not primary_horizon_sweep:
            primary_horizon_sweep = [base_primary_horizon]

        run_specs = [
            {"model_type": model_type, "primary_horizon": int(primary_horizon)}
            for model_type in model_list
            for primary_horizon in primary_horizon_sweep
        ]
        is_multi_run = len(run_specs) > 1
        logger.info(
            "Training sweep configured: %d run(s), models=%s, primary_horizons=%s",
            len(run_specs),
            ",".join(model_list),
            primary_horizon_sweep,
        )

        for run_index, run_spec in enumerate(run_specs, start=1):
            model_type = str(run_spec["model_type"])
            primary_horizon = int(run_spec["primary_horizon"])
            model_params = _load_model_defaults(
                model_type=model_type,
                use_gpu=resolved_use_gpu,
            )
            replay_model_params = replay_config.get("model_params")
            if isinstance(replay_model_params, dict):
                model_params = {**model_params, **replay_model_params}

            run_name_suffix = f"{model_type}_h{primary_horizon}"
            if configured_name:
                model_name = (
                    configured_name if not is_multi_run else f"{configured_name}_{run_name_suffix}"
                )
            elif replay_config:
                replay_base_name = str(replay_config.get("model_name") or model_type).strip()
                if is_multi_run:
                    model_name = f"{replay_base_name}_{run_name_suffix}_replay_{replay_timestamp}"
                else:
                    model_name = f"{replay_base_name}_replay_{replay_timestamp}"
            else:
                model_name = "" if not is_multi_run else f"{run_name_suffix}_{replay_timestamp}"

            raw_horizons = _cfg_value("label_horizons", getattr(args, "label_horizons", [1, 5, 20]))
            if isinstance(raw_horizons, (list, tuple)):
                label_horizons = [int(v) for v in raw_horizons]
            else:
                label_horizons = [1, 5, 20]

            raw_symbols = _cfg_value("symbols", cli_symbols)
            if isinstance(raw_symbols, (list, tuple, set)):
                symbols = [str(s).strip().upper() for s in raw_symbols if str(s).strip()]
            elif raw_symbols:
                symbols = [str(raw_symbols).strip().upper()]
            else:
                symbols = []

            raw_feature_groups = _cfg_value(
                "feature_groups",
                getattr(
                    args,
                    "feature_groups",
                    ["technical", "statistical", "microstructure", "cross_sectional"],
                ),
            )
            if isinstance(raw_feature_groups, (list, tuple, set)):
                feature_groups = [
                    str(g).strip().lower() for g in raw_feature_groups if str(g).strip()
                ]
            elif raw_feature_groups:
                feature_groups = [str(raw_feature_groups).strip().lower()]
            else:
                feature_groups = ["technical", "statistical", "microstructure", "cross_sectional"]

            use_meta_labeling = bool(
                _resolve_profiled_value(
                    replay_config=replay_config,
                    config_key="use_meta_labeling",
                    args=args,
                    profile_defaults=profile_defaults,
                    fallback=True,
                    arg_dest="disable_meta_labeling",
                    arg_transform=lambda disabled: not bool(disabled),
                )
            )
            compute_shap = bool(
                _resolve_profiled_value(
                    replay_config=replay_config,
                    config_key="compute_shap",
                    args=args,
                    profile_defaults=profile_defaults,
                    fallback=True,
                    arg_dest="no_shap",
                    arg_transform=lambda disabled: not bool(disabled),
                )
            )
            n_splits = int(
                _resolve_profiled_value(
                    replay_config=replay_config,
                    config_key="n_splits",
                    args=args,
                    profile_defaults=profile_defaults,
                    fallback=5,
                    arg_dest="n_splits",
                )
            )
            n_trials = int(
                _resolve_profiled_value(
                    replay_config=replay_config,
                    config_key="n_trials",
                    args=args,
                    profile_defaults=profile_defaults,
                    fallback=100,
                    arg_dest="n_trials",
                )
            )
            nested_outer_splits = int(
                _resolve_profiled_value(
                    replay_config=replay_config,
                    config_key="nested_outer_splits",
                    args=args,
                    profile_defaults=profile_defaults,
                    fallback=4,
                    arg_dest="nested_outer_splits",
                )
            )
            nested_inner_splits = int(
                _resolve_profiled_value(
                    replay_config=replay_config,
                    config_key="nested_inner_splits",
                    args=args,
                    profile_defaults=profile_defaults,
                    fallback=3,
                    arg_dest="nested_inner_splits",
                )
            )
            auto_live_profile_enabled = bool(
                _resolve_profiled_value(
                    replay_config=replay_config,
                    config_key="auto_live_profile_enabled",
                    args=args,
                    profile_defaults=profile_defaults,
                    fallback=True,
                    arg_dest="disable_auto_live_profile",
                    arg_transform=lambda disabled: not bool(disabled),
                )
            )
            feature_selection_stability_iterations = int(
                _resolve_profiled_value(
                    replay_config=replay_config,
                    config_key="feature_selection_stability_iterations",
                    args=args,
                    profile_defaults=profile_defaults,
                    fallback=16,
                    arg_dest="feature_selection_stability_iterations",
                )
            )
            require_nested_trace_for_promotion = bool(
                _resolve_profiled_value(
                    replay_config=replay_config,
                    config_key="require_nested_trace_for_promotion",
                    args=args,
                    profile_defaults=profile_defaults,
                    fallback=True,
                )
            )
            require_objective_breakdown_for_promotion = bool(
                _resolve_profiled_value(
                    replay_config=replay_config,
                    config_key="require_objective_breakdown_for_promotion",
                    args=args,
                    profile_defaults=profile_defaults,
                    fallback=True,
                )
            )

            config = TrainingConfig(
                model_type=model_type,
                model_name=model_name,
                training_profile=training_profile,
                symbols=symbols,
                start_date=_cfg_value(
                    "start_date",
                    getattr(args, "start_date", "") or getattr(args, "start", ""),
                ),
                end_date=_cfg_value(
                    "end_date",
                    getattr(args, "end_date", "") or getattr(args, "end", ""),
                ),
                timeframe=_cfg_value(
                    "timeframe",
                    getattr(args, "timeframe", DEFAULT_TIMEFRAME),
                ),
                include_premarket=bool(
                    _cfg_value(
                        "include_premarket",
                        getattr(args, "include_premarket", False),
                    )
                ),
                include_postmarket=bool(
                    _cfg_value(
                        "include_postmarket",
                        getattr(args, "include_postmarket", False),
                    )
                ),
                timeframes=list(
                    _cfg_value(
                        "timeframes",
                        getattr(args, "timeframes", []),
                    )
                    or []
                ),
                cv_method=_cfg_value("cv_method", getattr(args, "cv_method", "purged_kfold")),
                n_splits=n_splits,
                embargo_pct=max(
                    float(_cfg_value("embargo_pct", getattr(args, "embargo_pct", 0.01))), 0.01
                ),  # P1-H1: Min 1%
                optimize=True,
                optimizer="optuna",
                n_trials=n_trials,
                n_jobs=int(
                    _cfg_value(
                        "n_jobs",
                        1 if (model_type == "random_forest" and os.name == "nt") else -1,
                    )
                ),
                optuna_storage_dir=str(
                    _cfg_value(
                        "optuna_storage_dir",
                        getattr(args, "optuna_storage_dir", "") or "",
                    )
                ).strip(),
                optuna_resume_enabled=bool(
                    _cfg_value(
                        "optuna_resume_enabled",
                        not bool(getattr(args, "disable_optuna_resume", False)),
                    )
                ),
                seed=int(_cfg_value("seed", getattr(args, "seed", global_seed))),
                use_nested_walk_forward=True,
                nested_outer_splits=nested_outer_splits,
                nested_inner_splits=nested_inner_splits,
                nested_outer_stability_ratio_cap=float(
                    _cfg_value(
                        "nested_outer_stability_ratio_cap",
                        getattr(args, "nested_outer_stability_ratio_cap", 1.25),
                    )
                ),
                nested_outer_stability_min_trials=int(
                    _cfg_value(
                        "nested_outer_stability_min_trials",
                        getattr(args, "nested_outer_stability_min_trials", 8),
                    )
                ),
                require_nested_trace_for_promotion=require_nested_trace_for_promotion,
                require_objective_breakdown_for_promotion=require_objective_breakdown_for_promotion,
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
                        getattr(args, "objective_weight_turnover", 0.2),
                    )
                ),
                objective_weight_calibration=float(
                    _cfg_value(
                        "objective_weight_calibration",
                        getattr(args, "objective_weight_calibration", 0.35),
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
                objective_skew_penalty_cap=float(
                    _cfg_value(
                        "objective_skew_penalty_cap",
                        getattr(args, "objective_skew_penalty_cap", 1.5),
                    )
                ),
                objective_weight_tail_risk=float(
                    _cfg_value(
                        "objective_weight_tail_risk",
                        getattr(args, "objective_weight_tail_risk", 0.35),
                    )
                ),
                objective_weight_symbol_concentration=float(
                    _cfg_value(
                        "objective_weight_symbol_concentration",
                        getattr(args, "objective_weight_symbol_concentration", 0.20),
                    )
                ),
                objective_expected_shortfall_cap=float(
                    _cfg_value(
                        "objective_expected_shortfall_cap",
                        getattr(args, "objective_expected_shortfall_cap", 0.012),
                    )
                ),
                allow_unstable_outer_fold_fallback=bool(
                    _resolve_profiled_value(
                        replay_config=replay_config,
                        config_key="allow_unstable_outer_fold_fallback",
                        args=args,
                        profile_defaults=profile_defaults,
                        fallback=False,
                        arg_dest="allow_unstable_outer_fold_fallback",
                    )
                ),
                epochs=int(_cfg_value("epochs", getattr(args, "epochs", 100))),
                batch_size=int(_cfg_value("batch_size", getattr(args, "batch_size", 64))),
                learning_rate=float(
                    _cfg_value("learning_rate", getattr(args, "learning_rate", 0.001))
                ),
                primary_horizon_sweep=primary_horizon_sweep,
                use_meta_labeling=use_meta_labeling,
                meta_label_min_confidence=float(
                    _cfg_value(
                        "meta_label_min_confidence",
                        getattr(args, "meta_label_min_confidence", 0.55),
                    )
                ),
                meta_label_dynamic_threshold=bool(
                    _cfg_value(
                        "meta_label_dynamic_threshold",
                        not bool(getattr(args, "disable_meta_dynamic_threshold", False)),
                    )
                ),
                enable_probability_calibration=bool(
                    _cfg_value(
                        "enable_probability_calibration",
                        not bool(getattr(args, "disable_probability_calibration", False)),
                    )
                ),
                enable_ranker_probability_calibration=bool(
                    _cfg_value(
                        "enable_ranker_probability_calibration",
                        bool(getattr(args, "enable_ranker_probability_calibration", False)),
                    )
                ),
                expected_edge_use_symbol_priors=bool(
                    _cfg_value(
                        "expected_edge_use_symbol_priors",
                        bool(getattr(args, "enable_expected_edge_symbol_priors", False)),
                    )
                ),
                expected_edge_min_coverage=float(
                    _cfg_value(
                        "expected_edge_min_coverage",
                        getattr(args, "expected_edge_min_coverage", 0.55),
                    )
                ),
                probability_calibration_method=str(
                    _cfg_value(
                        "probability_calibration_method",
                        getattr(args, "probability_calibration_method", "isotonic"),
                    )
                ),
                apply_multiple_testing=True,
                correction_method="deflated_sharpe",
                use_gpu=resolved_use_gpu,
                require_gpu=require_gpu_requested,
                require_feature_gpu=require_feature_gpu_requested,
                use_database=True,
                use_redis_cache=True,
                compute_shap=compute_shap,
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
                min_white_reality_stat=float(
                    _cfg_value(
                        "min_white_reality_stat",
                        getattr(args, "min_white_reality_stat", 0.0),
                    )
                ),
                max_white_reality_pvalue=float(
                    _cfg_value(
                        "max_white_reality_pvalue",
                        getattr(args, "max_white_reality_pvalue", 0.10),
                    )
                ),
                min_holdout_sharpe=float(
                    _cfg_value(
                        "min_holdout_sharpe",
                        getattr(args, "min_holdout_sharpe", 0.0),
                    )
                ),
                min_holdout_regime_sharpe=float(
                    _cfg_value(
                        "min_holdout_regime_sharpe",
                        getattr(args, "min_holdout_regime_sharpe", -0.10),
                    )
                ),
                min_holdout_symbol_coverage=float(
                    _cfg_value(
                        "min_holdout_symbol_coverage",
                        getattr(args, "min_holdout_symbol_coverage", 0.60),
                    )
                ),
                min_holdout_symbol_p25_sharpe=float(
                    _cfg_value(
                        "min_holdout_symbol_p25_sharpe",
                        getattr(args, "min_holdout_symbol_p25_sharpe", -0.10),
                    )
                ),
                max_holdout_symbol_underwater_ratio=float(
                    _cfg_value(
                        "max_holdout_symbol_underwater_ratio",
                        getattr(args, "max_holdout_symbol_underwater_ratio", 0.55),
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
                max_symbol_concentration_hhi=float(
                    _cfg_value(
                        "max_symbol_concentration_hhi",
                        getattr(args, "max_symbol_concentration_hhi", 0.65),
                    )
                ),
                label_horizons=label_horizons,
                primary_label_horizon=primary_horizon,
                label_profit_taking_threshold=float(
                    _cfg_value(
                        "label_profit_taking_threshold", getattr(args, "profit_taking", 0.015)
                    )
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
                        getattr(args, "label_min_signal_abs_return_bps", 10.0),
                    )
                ),
                label_neutral_buffer_bps=float(
                    _cfg_value(
                        "label_neutral_buffer_bps",
                        getattr(args, "label_neutral_buffer_bps", 5.0),
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
                        getattr(args, "label_edge_cost_buffer_bps", 3.0),
                    )
                ),
                label_apply_uniqueness_weighting=bool(
                    _cfg_value(
                        "label_apply_uniqueness_weighting",
                        not bool(getattr(args, "disable_uniqueness_weighting", False)),
                    )
                ),
                label_uniqueness_weight_floor=float(
                    _cfg_value(
                        "label_uniqueness_weight_floor",
                        getattr(args, "label_uniqueness_weight_floor", 0.25),
                    )
                ),
                label_apply_volatility_inverse_weighting=bool(
                    _cfg_value(
                        "label_apply_volatility_inverse_weighting",
                        not bool(getattr(args, "disable_volatility_inverse_weighting", False)),
                    )
                ),
                label_volatility_weight_cap=float(
                    _cfg_value(
                        "label_volatility_weight_cap",
                        getattr(args, "label_volatility_weight_cap", 2.5),
                    )
                ),
                feature_groups=feature_groups,
                training_bar_mode=str(
                    _cfg_value(
                        "training_bar_mode",
                        getattr(args, "training_bar_mode", "time"),
                    )
                ),
                intrinsic_bar_type=str(
                    _cfg_value(
                        "intrinsic_bar_type",
                        getattr(args, "intrinsic_bar_type", "volume"),
                    )
                ),
                intrinsic_bar_threshold=float(
                    _cfg_value(
                        "intrinsic_bar_threshold",
                        getattr(args, "intrinsic_bar_threshold", 0.0),
                    )
                ),
                intrinsic_target_bars_per_day=int(
                    _cfg_value(
                        "intrinsic_target_bars_per_day",
                        getattr(args, "intrinsic_target_bars_per_day", 100),
                    )
                ),
                enable_symbol_quality_filter=bool(
                    _cfg_value(
                        "enable_symbol_quality_filter",
                        not bool(getattr(args, "disable_symbol_quality_filter", False)),
                    )
                ),
                target_universe_size=int(
                    _cfg_value(
                        "target_universe_size",
                        getattr(args, "target_universe_size", 0),
                    )
                ),
                universe_selection_buffer_size=int(
                    _cfg_value(
                        "universe_selection_buffer_size",
                        getattr(args, "universe_selection_buffer_size", 24),
                    )
                ),
                symbol_quality_min_rows=int(
                    _cfg_value(
                        "symbol_quality_min_rows",
                        getattr(args, "symbol_quality_min_rows", 1200),
                    )
                ),
                symbol_quality_min_symbols=int(
                    _cfg_value(
                        "symbol_quality_min_symbols",
                        getattr(args, "symbol_quality_min_symbols", 8),
                    )
                ),
                symbol_quality_max_missing_ratio=float(
                    _cfg_value(
                        "symbol_quality_max_missing_ratio",
                        getattr(args, "symbol_quality_max_missing_ratio", 0.12),
                    )
                ),
                symbol_quality_max_extreme_move_ratio=float(
                    _cfg_value(
                        "symbol_quality_max_extreme_move_ratio",
                        getattr(args, "symbol_quality_max_extreme_move_ratio", 0.08),
                    )
                ),
                symbol_quality_max_corporate_action_ratio=float(
                    _cfg_value(
                        "symbol_quality_max_corporate_action_ratio",
                        getattr(args, "symbol_quality_max_corporate_action_ratio", 0.02),
                    )
                ),
                symbol_quality_min_median_dollar_volume=float(
                    _cfg_value(
                        "symbol_quality_min_median_dollar_volume",
                        getattr(args, "symbol_quality_min_median_dollar_volume", 1_000_000.0),
                    )
                ),
                enable_cross_sectional=bool(
                    _cfg_value(
                        "enable_cross_sectional",
                        not bool(getattr(args, "disable_cross_sectional", False)),
                    )
                ),
                enable_reference_features=bool(
                    _cfg_value(
                        "enable_reference_features",
                        not bool(getattr(args, "disable_reference_features", False)),
                    )
                ),
                reference_feature_sources=list(
                    _cfg_value(
                        "reference_feature_sources",
                        getattr(args, "reference_feature_sources", []),
                    )
                    or []
                ),
                enable_tick_microstructure_features=bool(
                    _cfg_value(
                        "enable_tick_microstructure_features",
                        not bool(getattr(args, "disable_tick_microstructure_features", False)),
                    )
                ),
                cross_sectional_user_locked=bool(
                    ("enable_cross_sectional" in replay_config)
                    or bool(getattr(args, "disable_cross_sectional", False))
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
                        getattr(args, "max_cross_sectional_rows", 500000),
                    )
                ),
                allow_feature_group_fallback=bool(
                    _cfg_value(
                        "allow_feature_group_fallback",
                        bool(getattr(args, "allow_partial_feature_fallback", False))
                        and not bool(getattr(args, "strict_feature_groups", False)),
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
                feature_set_id=str(
                    _cfg_value(
                        "feature_set_id",
                        getattr(args, "feature_set_id", "default"),
                    )
                ),
                enable_feature_selection=bool(
                    _cfg_value(
                        "enable_feature_selection",
                        not bool(getattr(args, "disable_feature_selection", False)),
                    )
                ),
                feature_selection_min_ic=float(
                    _cfg_value(
                        "feature_selection_min_ic",
                        getattr(args, "feature_selection_min_ic", 0.015),
                    )
                ),
                feature_selection_max_corr=float(
                    _cfg_value(
                        "feature_selection_max_corr",
                        getattr(args, "feature_selection_max_corr", 0.90),
                    )
                ),
                feature_selection_max_features=int(
                    _cfg_value(
                        "feature_selection_max_features",
                        getattr(args, "feature_selection_max_features", 180),
                    )
                ),
                feature_selection_stability_iterations=feature_selection_stability_iterations,
                feature_selection_min_stability_support=float(
                    _cfg_value(
                        "feature_selection_min_stability_support",
                        getattr(args, "feature_selection_min_stability_support", 0.60),
                    )
                ),
                windows_force_fallback_features=bool(
                    _cfg_value(
                        "windows_force_fallback_features",
                        bool(getattr(args, "windows_fallback_features", False)),
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
                        getattr(args, "execution_turnover_cap", 0.60),
                    )
                ),
                execution_cooldown_bars=int(
                    _cfg_value(
                        "execution_cooldown_bars",
                        getattr(args, "execution_cooldown_bars", 2),
                    )
                ),
                execution_max_symbol_entry_share=float(
                    _cfg_value(
                        "execution_max_symbol_entry_share",
                        getattr(args, "execution_max_symbol_entry_share", 0.68),
                    )
                ),
                min_confidence_position_scale=float(
                    _cfg_value(
                        "min_confidence_position_scale",
                        getattr(args, "min_confidence_position_scale", 0.20),
                    )
                ),
                lightgbm_use_monotonic_constraints=bool(
                    _cfg_value(
                        "lightgbm_use_monotonic_constraints",
                        not bool(getattr(args, "disable_lightgbm_monotonic_constraints", False)),
                    )
                ),
                auto_live_profile_enabled=auto_live_profile_enabled,
                auto_live_profile_symbol_threshold=int(
                    _cfg_value(
                        "auto_live_profile_symbol_threshold",
                        getattr(args, "auto_live_profile_symbol_threshold", 40),
                    )
                ),
                auto_live_profile_min_years=float(
                    _cfg_value(
                        "auto_live_profile_min_years",
                        getattr(args, "auto_live_profile_min_years", 4.0),
                    )
                ),
                output_dir=str(getattr(args, "output_dir", "models")),
                warm_start_model_path=str(
                    _cfg_value(
                        "warm_start_model_path",
                        getattr(args, "warm_start_model", "") or "",
                    )
                ).strip(),
                dataset_snapshot_bundle_path=str(
                    _cfg_value(
                        "dataset_snapshot_bundle_path",
                        getattr(args, "dataset_snapshot_bundle", "") or replay_snapshot_bundle_path,
                    )
                ).strip(),
                strict_snapshot_replay=bool(
                    _cfg_value(
                        "strict_snapshot_replay",
                        bool(getattr(args, "strict_snapshot_replay", False))
                        or bool(replay_snapshot_bundle_path),
                    )
                ),
                auto_snapshot_reuse_enabled=bool(
                    _cfg_value(
                        "auto_snapshot_reuse_enabled",
                        not bool(getattr(args, "disable_auto_snapshot_reuse", False)),
                    )
                ),
                snapshot_only=bool(
                    _cfg_value(
                        "snapshot_only",
                        bool(getattr(args, "snapshot_only", False)),
                    )
                ),
                model_params=model_params,
            )
            if (
                not config.dataset_snapshot_bundle_path
                and config.auto_snapshot_reuse_enabled
                and not replay_config
            ):
                reusable_snapshot_bundle = _find_reusable_snapshot_bundle(config)
                if reusable_snapshot_bundle is not None:
                    config.dataset_snapshot_bundle_path = str(reusable_snapshot_bundle)
                    logger.info(
                        "Auto-reusing dataset snapshot bundle for matching training scope: %s",
                        reusable_snapshot_bundle,
                    )
            if config.training_profile == "promotion":
                if not _promotion_repo_provenance_ready(PROJECT_ROOT):
                    logger.error(
                        "Promotion profile requires a real Git checkout/worktree with repository provenance."
                    )
                    if requested_model == "all" or is_multi_run:
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
                                    "primary_label_horizon": int(config.primary_label_horizon),
                                    "run_id": f"{config.model_type}_h{int(config.primary_label_horizon)}",
                                    "error": "promotion_repo_provenance_missing",
                                },
                            )
                        )
                        overall_success = False
                        continue
                    return 1
                if not config.dataset_snapshot_bundle_path:
                    logger.error(
                        "Promotion profile requires an immutable dataset snapshot bundle. "
                        "Provide --dataset-snapshot-bundle or enable auto snapshot reuse from a "
                        "matching research candidate."
                    )
                    if requested_model == "all" or is_multi_run:
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
                                    "primary_label_horizon": int(config.primary_label_horizon),
                                    "run_id": f"{config.model_type}_h{int(config.primary_label_horizon)}",
                                    "error": "promotion_snapshot_bundle_required",
                                },
                            )
                        )
                        overall_success = False
                        continue
                    return 1
                config.strict_snapshot_replay = True
            _verify_institutional_infra(config)
            _log_training_environment_guidance(config.training_profile)

            logger.info("-" * 80)
            logger.info(f"Run {run_index}/{len(run_specs)}")
            logger.info(f"Model type: {config.model_type}")
            logger.info(f"Training profile: {config.training_profile}")
            logger.info(f"Primary horizon: {config.primary_label_horizon}")
            logger.info(f"Symbols: {config.symbols or 'all available'}")
            logger.info("Data source: PostgreSQL (mandatory)")
            logger.info("Redis cache: enabled (mandatory)")
            if config.dataset_snapshot_bundle_path:
                logger.info(f"Dataset snapshot bundle: {config.dataset_snapshot_bundle_path}")
            if config.snapshot_only:
                logger.info("Snapshot-only mode: enabled")
            logger.info(f"CV method: {config.cv_method} ({config.n_splits} splits)")
            logger.info(f"Embargo: {config.embargo_pct * 100:.1f}%")
            logger.info(f"GPU acceleration enabled: {config.use_gpu}")
            logger.info(f"Model GPU required: {config.require_gpu}")
            logger.info(f"Feature GPU required: {config.require_feature_gpu}")
            if config.require_gpu and not config.require_feature_gpu:
                logger.info(
                    "Feature materialization may still use CPU-optimized paths; "
                    "use --require-feature-gpu only after full CUDA family coverage exists."
                )
            logger.info(
                "Training stack: Nested Walk-Forward + Optuna + Meta-Labeling=%s + "
                "Multiple-Testing + SHAP=%s + Leak-Validation",
                "on" if config.use_meta_labeling else "off",
                "on" if config.compute_shap else "off",
            )
            logger.info(
                "Optuna resume: %s (%s)",
                "enabled" if config.optuna_resume_enabled else "disabled",
                config.optuna_storage_dir or str(Path(config.output_dir) / "optuna_state"),
            )
            if config.optuna_resume_enabled:
                logger.info(
                    "Optuna resume procedure: rerun the same command with the same --name "
                    "to reuse saved studies after interruption."
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
            if (
                config.training_profile == "promotion"
                and config.strict_snapshot_replay
                and config.dataset_snapshot_bundle_path
                and not replay_config
            ):
                promotion_precheck = _evaluate_promotion_precheck(
                    output_dir=Path(config.output_dir),
                    model_type=config.model_type,
                    primary_label_horizon=int(config.primary_label_horizon),
                    dataset_snapshot_bundle_path=str(config.dataset_snapshot_bundle_path),
                    bypass=bool(getattr(args, "force_promotion_precheck_bypass", False)),
                )
                if promotion_precheck.get("bypassed", False):
                    logger.warning(
                        "Promotion precheck bypassed by operator for strict snapshot replay: %s",
                        config.dataset_snapshot_bundle_path,
                    )
                elif promotion_precheck.get("blocked", False):
                    logger.error(
                        "Promotion blocked before training: %s",
                        promotion_precheck.get(
                            "message",
                            "Research candidate did not clear the pre-promotion checklist.",
                        ),
                    )
                    if promotion_precheck.get("matrix_path"):
                        logger.error(
                            "Source training matrix: %s",
                            promotion_precheck.get("matrix_path"),
                        )
                    checklist = promotion_precheck.get("checklist", {})
                    failed_checks = (
                        checklist.get("failed_checks", []) if isinstance(checklist, dict) else []
                    )
                    if failed_checks:
                        logger.error("Failed checklist items: %s", ", ".join(failed_checks))
                    candidate = promotion_precheck.get("candidate", {})
                    if isinstance(candidate, dict) and candidate:
                        logger.error(
                            "Best matching research candidate: run_id=%s governance=%.2f pre_ready=%s",
                            candidate.get("run_id", ""),
                            float(candidate.get("governance_score", 0.0)),
                            bool(candidate.get("pre_promotion_ready", 0.0)),
                        )
                    if requested_model == "all" or is_multi_run:
                        results.append(
                            (
                                model_type,
                                {
                                    "success": False,
                                    "model_type": config.model_type,
                                    "model_name": config.model_name,
                                    "model_path": None,
                                    "training_duration_seconds": 0.0,
                                    "training_metrics": {
                                        "pre_promotion_checklist": checklist,
                                        "pre_promotion_failed_checks": failed_checks,
                                    },
                                    "primary_label_horizon": int(config.primary_label_horizon),
                                    "run_id": f"{config.model_type}_h{int(config.primary_label_horizon)}",
                                    "error": "promotion_precheck_blocked",
                                },
                            )
                        )
                        overall_success = False
                        continue
                    return 1
                else:
                    candidate = promotion_precheck.get("candidate", {})
                    if isinstance(candidate, dict) and candidate:
                        logger.info(
                            "Promotion precheck passed using research candidate %s (governance=%.2f)",
                            candidate.get("run_id", ""),
                            float(candidate.get("governance_score", 0.0)),
                        )

            if audit_logger is not None:
                audit_logger.log_model_training_started(
                    model_name=config.model_type,
                    model_version=config.model_name,
                    config={
                        "training_profile": config.training_profile,
                        "cv_method": config.cv_method,
                        "n_splits": config.n_splits,
                        "embargo_pct": config.embargo_pct,
                        "n_trials": config.n_trials,
                        "use_gpu": config.use_gpu,
                        "require_gpu": config.require_gpu,
                        "require_feature_gpu": config.require_feature_gpu,
                        "use_meta_labeling": config.use_meta_labeling,
                        "compute_shap": config.compute_shap,
                        "holdout_pct": config.holdout_pct,
                        "execution_turnover_cap": config.execution_turnover_cap,
                        "execution_cooldown_bars": config.execution_cooldown_bars,
                        "execution_max_symbol_entry_share": config.execution_max_symbol_entry_share,
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
                if requested_model == "all" or is_multi_run:
                    logger.error(
                        "Model %s horizon=%s failed during sweep; continuing. Error: %s",
                        config.model_type,
                        config.primary_label_horizon,
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
                                "primary_label_horizon": int(config.primary_label_horizon),
                                "run_id": f"{config.model_type}_h{int(config.primary_label_horizon)}",
                                "error": str(model_exc),
                            },
                        )
                    )
                    overall_success = False
                    continue
                raise

            if not isinstance(result, dict):
                result = {
                    "success": False,
                    "training_metrics": {},
                    "error": "invalid_result_payload",
                }
            result.setdefault("model_type", config.model_type)
            result.setdefault("model_name", config.model_name)
            result["primary_label_horizon"] = int(config.primary_label_horizon)
            result.setdefault("run_id", f"{config.model_type}_h{int(config.primary_label_horizon)}")
            metrics_payload = result.get("training_metrics")
            if isinstance(metrics_payload, dict):
                metrics_payload.setdefault(
                    "primary_label_horizon", float(config.primary_label_horizon)
                )

            registry_entry: dict[str, Any] | None = None
            model_path = result.get("model_path")
            if isinstance(model_path, str) and model_path.strip():
                tags = [
                    "training_pipeline",
                    "institutional_mode",
                    f"cv:{config.cv_method}",
                    (
                        "validation:passed"
                        if bool(result.get("success", False))
                        else "validation:failed"
                    ),
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
                        trainer.snapshot_manifest if isinstance(trainer, ModelTrainer) else None
                    ),
                    training_config=config.__dict__,
                    project_root=PROJECT_ROOT,
                    model_card=result.get("model_card"),
                    deployment_plan=result.get("deployment_plan"),
                )
                result["registry_version_id"] = registry_entry.get("version_id")
                result["registry_path"] = str(
                    Path(config.output_dir) / "registry" / "registry.json"
                )

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

        benchmark_json_path: Path | None = None
        benchmark_csv_path: Path | None = None
        champion_snapshot: dict[str, Any] | None = None
        has_persisted_models = any(
            isinstance(result.get("model_path"), str)
            and bool(str(result.get("model_path")).strip())
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

                promoted = (
                    bool(champion_snapshot.get("promoted", False)) if champion_snapshot else False
                )
                champion_payload = (
                    champion_snapshot.get("champion", {}) if champion_snapshot else {}
                )
                if promoted and audit_logger is not None and isinstance(champion_payload, dict):
                    champion_metrics = {
                        "governance_score": float(champion_payload.get("governance_score", 0.0)),
                        "mean_sharpe": float(champion_payload.get("mean_sharpe", 0.0)),
                        "holdout_sharpe": float(champion_payload.get("holdout_sharpe", 0.0)),
                        "pbo": float(champion_payload.get("pbo", 1.0)),
                        "white_reality_pvalue": float(
                            champion_payload.get("white_reality_pvalue", 1.0)
                        ),
                    }
                    audit_logger.log_model_deployed(
                        model_name=str(champion_payload.get("model_type", "")),
                        model_version=str(champion_payload.get("registry_version_id", "")),
                        metrics=champion_metrics,
                        promotion_mode="auto_benchmark_promotion",
                        champion_snapshot_path=str(champion_snapshot.get("snapshot_path", "")),
                    )
            except Exception as governance_exc:
                logger.warning(
                    f"Benchmark/champion governance post-processing failed: {governance_exc}"
                )

        # Report results
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        snapshot_only_mode_only = bool(results) and all(
            bool(result.get("snapshot_only", False)) for _, result in results
        )
        for model_type, result in results:
            logger.info("-" * 80)
            logger.info(f"{model_type.upper()} RESULT")
            if result.get("snapshot_only", False):
                logger.info("Snapshot-only run: yes")
                logger.info(f"Snapshot review: {result.get('snapshot_review_path', 'N/A')}")
                logger.info(
                    "Snapshot readiness: %s",
                    "PASSED" if bool(result.get("success", False)) else "FAILED",
                )
            else:
                logger.info(f"Model saved to: {result.get('model_path', 'N/A')}")
            logger.info(f"Training duration: {result.get('training_duration_seconds', 0):.1f}s")
            metrics = result.get("training_metrics", {})
            if metrics:
                if result.get("snapshot_only", False):
                    review_payload = metrics.get("snapshot_review", {})
                    if isinstance(review_payload, dict) and review_payload:
                        failed_checks = review_payload.get("failed_checks", [])
                        if isinstance(failed_checks, list) and failed_checks:
                            logger.info(
                                "Snapshot blocked by: %s",
                                ", ".join(str(item) for item in failed_checks),
                            )
                else:
                    logger.info(f"Mean accuracy: {metrics.get('mean_accuracy', 0):.4f}")
                    logger.info(f"Mean Sharpe: {metrics.get('mean_sharpe', 0):.4f}")
                    if "deflated_sharpe" in metrics:
                        logger.info(f"Deflated Sharpe: {metrics['deflated_sharpe']:.4f}")
                    checklist = metrics.get("pre_promotion_checklist", {})
                    if isinstance(checklist, dict) and checklist:
                        logger.info(
                            "Pre-promotion checklist: %s",
                            "PASSED" if checklist.get("ready", False) else "FAILED",
                        )
                        failed_checks = checklist.get("failed_checks", [])
                        if isinstance(failed_checks, list) and failed_checks:
                            logger.info(
                                "Blocked by: %s", ", ".join(str(item) for item in failed_checks)
                            )
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
            if snapshot_only_mode_only:
                logger.info("Snapshot preparation gates: PASSED")
            else:
                logger.info("Validation gates: PASSED")
            return 0

        if snapshot_only_mode_only:
            logger.warning("Snapshot preparation gates: FAILED for one or more runs")
            logger.warning("Review the snapshot review artifact before launching research training")
        else:
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
        artifacts_path = (
            Path(model_path).with_suffix(".json").with_stem(Path(model_path).stem + "_artifacts")
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
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_PRIMARY_MODEL,
        choices=SUPPORTED_MODELS + ["all", PRIMARY_CHALLENGER_MODEL_ALIAS],
    )
    parser.add_argument("--name", type=str, default="")
    parser.add_argument(
        "--training-profile",
        type=str,
        choices=list(SUPPORTED_TRAINING_PROFILES),
        default="promotion",
        help="Named training preset: promotion for final candidates, research for fast iteration.",
    )
    parser.add_argument("--symbols", nargs="+", default=[])
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Load symbols from a newline/comma separated text file or JSON payload.",
    )
    parser.add_argument("--start", "--start-date", dest="start_date", type=str, default="")
    parser.add_argument("--end", "--end-date", dest="end_date", type=str, default="")
    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME)
    parser.add_argument(
        "--include-premarket",
        action="store_true",
        help="Keep premarket bars in intraday training snapshots instead of filtering to regular session.",
    )
    parser.add_argument(
        "--include-postmarket",
        action="store_true",
        help="Keep postmarket bars in intraday training snapshots instead of filtering to regular session.",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=[],
        help="Optional multi-timeframe fusion scope (for example: 15Min 1Hour 1Day).",
    )
    parser.add_argument(
        "--cv-method",
        choices=["purged_kfold", "combinatorial", "walk_forward"],
        default="purged_kfold",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--embargo-pct", type=float, default=0.01)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument(
        "--optuna-storage-dir",
        type=Path,
        default=None,
        help="Directory for persistent Optuna SQLite studies used for resume-safe training.",
    )
    parser.add_argument(
        "--disable-optuna-resume",
        action="store_true",
        help="Use fresh in-memory Optuna studies instead of resumable SQLite storage.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nested-outer-splits", type=int, default=4)
    parser.add_argument("--nested-inner-splits", type=int, default=3)
    parser.add_argument("--nested-outer-stability-ratio-cap", type=float, default=1.25)
    parser.add_argument("--nested-outer-stability-min-trials", type=int, default=8)
    parser.add_argument(
        "--allow-unstable-outer-fold-fallback",
        action="store_true",
        help=(
            "Allow research runs to fall back to unstable outer-fold candidates when no stable "
            "candidate passes the Optuna surface gate."
        ),
    )
    parser.add_argument(
        "--disable-nested-walk-forward",
        action="store_true",
        help="Forbidden in institutional mode; nested walk-forward is mandatory.",
    )
    parser.add_argument("--objective-weight-sharpe", type=float, default=1.0)
    parser.add_argument("--objective-weight-drawdown", type=float, default=0.5)
    parser.add_argument("--objective-weight-turnover", type=float, default=0.2)
    parser.add_argument("--objective-weight-calibration", type=float, default=0.35)
    parser.add_argument("--objective-weight-trade-activity", type=float, default=1.0)
    parser.add_argument("--objective-weight-cvar", type=float, default=0.4)
    parser.add_argument("--objective-weight-skew", type=float, default=0.1)
    parser.add_argument("--objective-skew-penalty-cap", type=float, default=1.5)
    parser.add_argument("--objective-weight-tail-risk", type=float, default=0.35)
    parser.add_argument("--objective-weight-symbol-concentration", type=float, default=0.20)
    parser.add_argument("--objective-expected-shortfall-cap", type=float, default=0.012)
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
        "--dataset-snapshot-bundle",
        type=Path,
        default=None,
        help="Load immutable training dataset bundle directly instead of rebuilding from live data.",
    )
    parser.add_argument(
        "--strict-snapshot-replay",
        action="store_true",
        help="Fail instead of falling back when dataset snapshot replay bundle is missing.",
    )
    parser.add_argument(
        "--force-promotion-precheck-bypass",
        action="store_true",
        help="Allow strict-snapshot promotion to start without a matching pre-promotion-ready research candidate.",
    )
    parser.add_argument(
        "--disable-auto-snapshot-reuse",
        action="store_true",
        help="Force live dataset rebuild even when a matching local dataset snapshot bundle exists.",
    )
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help=(
            "Stop after persisting the dataset snapshot bundle, data-quality report, "
            "and dropped-symbol review artifact."
        ),
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
    parser.add_argument("--min-holdout-regime-sharpe", type=float, default=-0.10)
    parser.add_argument("--min-holdout-symbol-coverage", type=float, default=0.60)
    parser.add_argument("--min-holdout-symbol-p25-sharpe", type=float, default=-0.10)
    parser.add_argument("--max-holdout-symbol-underwater-ratio", type=float, default=0.55)
    parser.add_argument("--max-holdout-drawdown", type=float, default=0.35)
    parser.add_argument("--max-regime-shift", type=float, default=0.35)
    parser.add_argument("--max-symbol-concentration-hhi", type=float, default=0.65)
    parser.add_argument(
        "--disable-auto-live-profile",
        action="store_true",
        help="Disable automatic institutional profile tuning for large multi-symbol datasets.",
    )
    parser.add_argument("--auto-live-profile-symbol-threshold", type=int, default=40)
    parser.add_argument("--auto-live-profile-min-years", type=float, default=4.0)
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Deprecated: training auto-detects GPU acceleration.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Deprecated alias for --require-model-gpu.",
    )
    parser.add_argument(
        "--require-model-gpu",
        "--require-gpu",
        dest="require_gpu",
        action="store_true",
        help=(
            "Fail fast unless the requested model stack can run on GPU. "
            "Feature materialization may still use CPU-optimized paths unless "
            "--require-feature-gpu is also set."
        ),
    )
    parser.add_argument(
        "--require-feature-gpu",
        action="store_true",
        help=(
            "Fail fast unless all requested feature groups have complete CUDA/cuDF "
            "materialization. Partial technical acceleration does not satisfy this flag."
        ),
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
        help="Disable SHAP explainability. Allowed only in research profile.",
    )
    parser.add_argument("--label-horizons", nargs="+", type=int, default=[1, 5, 20])
    parser.add_argument("--primary-horizon", type=int, default=5)
    parser.add_argument(
        "--primary-horizon-sweep",
        nargs="+",
        type=int,
        default=[],
        help="Optional sweep of primary horizons (bars) for model-horizon benchmark matrix.",
    )
    parser.add_argument("--profit-taking", type=float, default=0.015)
    parser.add_argument("--stop-loss", type=float, default=0.010)
    parser.add_argument("--max-holding", type=int, default=20)
    parser.add_argument("--spread-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--impact-bps", type=float, default=2.0)
    parser.add_argument("--label-min-signal-abs-return-bps", type=float, default=10.0)
    parser.add_argument("--label-neutral-buffer-bps", type=float, default=5.0)
    parser.add_argument("--label-edge-cost-buffer-bps", type=float, default=3.0)
    parser.add_argument("--label-max-abs-forward-return", type=float, default=0.35)
    parser.add_argument("--label-signal-volatility-floor-mult", type=float, default=0.50)
    parser.add_argument("--label-volatility-lookback", type=int, default=20)
    parser.add_argument("--label-regime-lookback", type=int, default=30)
    parser.add_argument("--label-temporal-weight-decay", type=float, default=0.999)
    parser.add_argument(
        "--disable-uniqueness-weighting",
        action="store_true",
        help="Disable event-overlap uniqueness weighting during target engineering.",
    )
    parser.add_argument("--label-uniqueness-weight-floor", type=float, default=0.25)
    parser.add_argument(
        "--disable-volatility-inverse-weighting",
        action="store_true",
        help="Disable inverse-volatility sample reweighting during target engineering.",
    )
    parser.add_argument("--label-volatility-weight-cap", type=float, default=2.5)
    parser.add_argument(
        "--meta-label-min-confidence",
        type=float,
        default=0.55,
        help="Base minimum confidence for meta-label filtering.",
    )
    parser.add_argument(
        "--disable-meta-dynamic-threshold",
        action="store_true",
        help="Disable horizon/regime-adaptive meta-label confidence thresholding.",
    )
    parser.add_argument(
        "--disable-meta-labeling",
        action="store_true",
        help="Disable meta-labeling. Allowed only in research profile.",
    )
    parser.add_argument(
        "--disable-probability-calibration",
        action="store_true",
        help="Disable post-hoc probability calibration on out-of-fold predictions.",
    )
    parser.add_argument(
        "--enable-ranker-probability-calibration",
        action="store_true",
        help="Allow post-hoc probability calibration for ranker models using normalized OOF scores.",
    )
    parser.add_argument(
        "--probability-calibration-method",
        choices=["isotonic", "sigmoid"],
        default="isotonic",
        help="Post-hoc probability calibration method (default: isotonic).",
    )
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        default=["technical", "statistical", "microstructure", "cross_sectional"],
        help="Feature groups to compute (default: technical statistical microstructure cross_sectional)",
    )
    parser.add_argument(
        "--training-bar-mode",
        choices=["time", "intrinsic"],
        default="time",
        help="Input bar representation for training (default: time).",
    )
    parser.add_argument(
        "--intrinsic-bar-type",
        choices=[
            "tick",
            "volume",
            "dollar",
            "tick_imbalance",
            "volume_imbalance",
            "tick_run",
            "volume_run",
        ],
        default="volume",
        help="Intrinsic bar family when --training-bar-mode intrinsic.",
    )
    parser.add_argument("--intrinsic-bar-threshold", type=float, default=0.0)
    parser.add_argument("--intrinsic-target-bars-per-day", type=int, default=100)
    parser.add_argument(
        "--disable-symbol-quality-filter",
        action="store_true",
        help="Disable symbol-level quality universe filtering before feature generation.",
    )
    parser.add_argument(
        "--target-universe-size",
        type=int,
        default=0,
        help=(
            "Optional database-first universe cap when symbols are not provided. "
            "Selects the most liquid/high-coverage symbols before loading full panels."
        ),
    )
    parser.add_argument(
        "--universe-selection-buffer-size",
        type=int,
        default=24,
        help="Extra candidate symbols to keep before post-load quality filtering (default: 24).",
    )
    parser.add_argument("--symbol-quality-min-rows", type=int, default=1200)
    parser.add_argument("--symbol-quality-min-symbols", type=int, default=8)
    parser.add_argument("--symbol-quality-max-missing-ratio", type=float, default=0.12)
    parser.add_argument("--symbol-quality-max-extreme-move-ratio", type=float, default=0.08)
    parser.add_argument("--symbol-quality-max-corporate-action-ratio", type=float, default=0.02)
    parser.add_argument(
        "--symbol-quality-min-median-dollar-volume", type=float, default=1_000_000.0
    )
    parser.add_argument(
        "--disable-cross-sectional",
        action="store_true",
        help="Disable cross-sectional features explicitly.",
    )
    parser.add_argument(
        "--disable-reference-features",
        action="store_true",
        help="Disable point-in-time reference/event feature augmentation.",
    )
    parser.add_argument(
        "--reference-feature-sources",
        nargs="+",
        default=[],
        help=(
            "Optional reference feature source allowlist. Supported values: "
            "macro short_sale news sec_filings fundamentals earnings ftd corporate_actions"
        ),
    )
    parser.add_argument(
        "--disable-tick-microstructure-features",
        action="store_true",
        help="Disable stock_quotes/stock_trades microstructure enrichment.",
    )
    parser.add_argument(
        "--strict-feature-groups",
        action="store_true",
        help="Fail run if any requested feature group cannot be materialized.",
    )
    parser.add_argument(
        "--allow-partial-feature-fallback",
        action="store_true",
        help=(
            "Allow partial feature fallback (for example cross-sectional skip or "
            "Windows emergency fallback). Disabled by default for full-feature enforcement."
        ),
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
        default=500000,
        help=(
            "Adaptive guardrail: disable cross-sectional when dataset rows exceed this value. "
            "Default 500000 keeps the Wave 1 multi-symbol 15Min scope on the full feature stack."
        ),
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
    parser.add_argument(
        "--feature-set-id",
        type=str,
        default="default",
        help="Optional namespace seed for feature cache scoping.",
    )
    parser.add_argument(
        "--disable-feature-selection",
        action="store_true",
        help="Disable IC/correlation/stability feature selection on the development set.",
    )
    parser.add_argument("--feature-selection-min-ic", type=float, default=0.015)
    parser.add_argument("--feature-selection-max-corr", type=float, default=0.90)
    parser.add_argument("--feature-selection-max-features", type=int, default=180)
    parser.add_argument("--feature-selection-stability-iterations", type=int, default=16)
    parser.add_argument("--feature-selection-min-stability-support", type=float, default=0.60)
    parser.add_argument(
        "--windows-fallback-features",
        action="store_true",
        help="Force deterministic basic feature fallback on Windows instead of full feature pipeline.",
    )
    parser.add_argument("--disable-dynamic-no-trade-band", action="store_true")
    parser.add_argument("--execution-vol-target-daily", type=float, default=0.012)
    parser.add_argument("--execution-turnover-cap", type=float, default=0.60)
    parser.add_argument("--execution-cooldown-bars", type=int, default=2)
    parser.add_argument("--execution-max-symbol-entry-share", type=float, default=0.68)
    parser.add_argument(
        "--min-confidence-position-scale",
        type=float,
        default=0.20,
        help="Minimum fractional position scale for lower-confidence signals (default: 0.20).",
    )
    parser.add_argument(
        "--enable-expected-edge-symbol-priors",
        action="store_true",
        help="Opt in to symbol-level priors inside the expected-edge selector.",
    )
    parser.add_argument("--expected-edge-min-coverage", type=float, default=0.55)
    parser.add_argument(
        "--disable-lightgbm-monotonic-constraints",
        action="store_true",
        help="Disable automatic LightGBM monotonic constraints inferred from feature names.",
    )
    parser.add_argument("--min-deflated-sharpe", type=float, default=0.10)
    parser.add_argument("--max-deflated-sharpe-pvalue", type=float, default=0.10)
    parser.add_argument("--max-pbo", type=float, default=0.45)
    parser.add_argument("--min-white-reality-stat", type=float, default=0.0)
    parser.add_argument("--max-white-reality-pvalue", type=float, default=0.10)
    parser.add_argument(
        "--warm-start-model",
        type=Path,
        default=None,
        help="Optional prior model artifact used only during final production refit.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Console entrypoint for quant-train."""
    parser = build_parser()
    normalized_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(normalized_argv)
    _attach_explicit_profile_overrides(parser, args, normalized_argv)
    return run_training(args)


if __name__ == "__main__":
    sys.exit(main())
