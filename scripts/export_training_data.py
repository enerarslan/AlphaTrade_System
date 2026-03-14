#!/usr/bin/env python3
"""
Export PostgreSQL data to Parquet files for ML training.

This script is part of the hybrid architecture where PostgreSQL + TimescaleDB
is the source of truth, and Parquet files are used for ML training (faster I/O).

Usage:
    python scripts/export_training_data.py --output data/training
    python scripts/export_training_data.py --symbols AAPL MSFT --start-date 2024-01-01
    python scripts/export_training_data.py --features-only --output data/training/features

@agent: @data, @mlquant
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from quant_trading_system.data.db_loader import get_db_loader
from quant_trading_system.data.db_feature_store import get_db_feature_store
from quant_trading_system.data.timeframe import DEFAULT_TIMEFRAME, normalize_timeframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _load_symbols_file(symbols_file: Path) -> list[str]:
    """Load symbols from newline/comma separated text or JSON payloads."""
    raw_text = Path(symbols_file).read_text(encoding="utf-8")
    stripped = raw_text.strip()
    if not stripped:
        return []

    payload: Any | None = None
    if stripped[0] in "[{":
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None

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


def _normalize_symbols(symbols: list[str] | None) -> list[str] | None:
    if symbols is None:
        return None
    normalized: list[str] = []
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        if symbol and symbol not in normalized:
            normalized.append(symbol)
    return normalized


def resolve_training_feature_set_id(
    *,
    symbols: list[str],
    timeframe: str,
    feature_set_id: str,
    enable_reference_features: bool = True,
    feature_groups: list[str] | None = None,
) -> tuple[str, list[str], bool]:
    """Resolve the deterministic feature scope id used by the training pipeline."""
    from scripts.train import ModelTrainer, TrainingConfig

    resolved_groups = list(feature_groups or ["technical", "statistical", "microstructure", "cross_sectional"])
    enable_cross_sectional = len(symbols) > 1
    if not enable_cross_sectional:
        resolved_groups = [group for group in resolved_groups if group != "cross_sectional"]

    config = TrainingConfig(
        model_type="xgboost",
        symbols=symbols,
        timeframe=timeframe,
        optimize=False,
        use_meta_labeling=False,
        apply_multiple_testing=False,
        compute_shap=False,
        save_artifacts=False,
        feature_groups=resolved_groups,
        enable_reference_features=enable_reference_features,
        enable_cross_sectional=enable_cross_sectional,
        cross_sectional_user_locked=not enable_cross_sectional,
        persist_features_to_postgres=False,
        feature_set_id=feature_set_id,
    )
    trainer = ModelTrainer(config)
    resolved_id = trainer._resolve_feature_set_id(symbols)
    return resolved_id, resolved_groups, enable_cross_sectional


def materialize_missing_features(
    *,
    output_dir: Path,
    symbols: list[str],
    start_date: datetime | None,
    end_date: datetime | None,
    timeframe: str,
    feature_set_id: str,
    enable_reference_features: bool = True,
    allow_feature_group_fallback: bool = False,
    windows_force_fallback_features: bool = False,
    feature_groups: list[str] | None = None,
) -> str:
    """Compute and persist missing training features before export."""
    if not symbols:
        return []

    from scripts.train import ModelTrainer, TrainingConfig

    resolved_feature_set_id, resolved_groups, enable_cross_sectional = resolve_training_feature_set_id(
        symbols=symbols,
        timeframe=timeframe,
        feature_set_id=feature_set_id,
        enable_reference_features=enable_reference_features,
        feature_groups=feature_groups,
    )

    config = TrainingConfig(
        model_type="xgboost",
        model_name=f"feature_materializer_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        symbols=symbols,
        start_date=start_date.isoformat() if start_date is not None else "",
        end_date=end_date.isoformat() if end_date is not None else "",
        timeframe=timeframe,
        optimize=False,
        use_meta_labeling=False,
        apply_multiple_testing=False,
        compute_shap=False,
        save_artifacts=False,
        use_database=True,
        use_redis_cache=True,
        feature_groups=resolved_groups,
        enable_reference_features=enable_reference_features,
        enable_cross_sectional=enable_cross_sectional,
        cross_sectional_user_locked=not enable_cross_sectional,
        allow_feature_group_fallback=allow_feature_group_fallback,
        persist_features_to_postgres=True,
        feature_set_id=feature_set_id,
        windows_force_fallback_features=windows_force_fallback_features,
        output_dir=str(output_dir / "_feature_materialization"),
        enable_symbol_quality_filter=False,
        symbol_quality_min_symbols=1,
    )
    trainer = ModelTrainer(config)
    trainer._load_data()
    trainer._compute_features()
    if trainer.features is None or trainer.features.empty:
        raise RuntimeError("Feature materialization completed without producing a feature matrix")
    return resolved_feature_set_id


def export_ohlcv_data(
    output_dir: Path,
    symbols: list[str] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> dict[str, Path]:
    """
    Export OHLCV data from PostgreSQL to Parquet files.

    Args:
        output_dir: Output directory for Parquet files.
        symbols: List of symbols to export. If None, exports all.
        start_date: Start date filter.
        end_date: End date filter.

    Returns:
        Dictionary mapping symbol to output path.
    """
    db_loader = get_db_loader()
    ohlcv_dir = output_dir / "ohlcv"
    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    timeframe = normalize_timeframe(timeframe)
    symbols = _normalize_symbols(symbols)

    if symbols is None:
        symbols = db_loader.get_available_symbols(timeframe=timeframe)
        logger.info(f"Found {len(symbols)} symbols in database for timeframe={timeframe}")

    results = {}
    for symbol in symbols:
        try:
            output_path = ohlcv_dir / f"{symbol}.parquet"
            db_loader.export_to_parquet(
                symbol,
                output_path,
                start_date,
                end_date,
                timeframe=timeframe,
            )
            results[symbol] = output_path
            logger.info(f"Exported OHLCV for {symbol}")
        except Exception as e:
            logger.error(f"Failed to export OHLCV for {symbol}: {e}")

    return results


def export_features_data(
    output_dir: Path,
    symbols: list[str] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    timeframe: str = DEFAULT_TIMEFRAME,
    feature_set_id: str = "default",
    materialize_missing: bool = False,
    enable_reference_features: bool = True,
    allow_feature_group_fallback: bool = False,
    windows_force_fallback_features: bool = False,
    feature_groups: list[str] | None = None,
) -> dict[str, Path]:
    """
    Export feature data from PostgreSQL to Parquet files.

    Args:
        output_dir: Output directory for Parquet files.
        symbols: List of symbols to export. If None, exports all.
        start_date: Start date filter.
        end_date: End date filter.

    Returns:
        Dictionary mapping symbol to output path.
    """
    feature_store = get_db_feature_store()
    db_loader = get_db_loader()
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    timeframe = normalize_timeframe(timeframe)
    feature_set_id = str(feature_set_id or "default").strip() or "default"

    resolved_symbols = _normalize_symbols(symbols)
    resolved_feature_set_id = feature_set_id
    resolved_groups = list(feature_groups or ["technical", "statistical", "microstructure", "cross_sectional"])
    if resolved_symbols:
        (
            resolved_feature_set_id,
            resolved_groups,
            _,
        ) = resolve_training_feature_set_id(
            symbols=resolved_symbols,
            timeframe=timeframe,
            feature_set_id=feature_set_id,
            enable_reference_features=enable_reference_features,
            feature_groups=feature_groups,
        )
    if resolved_symbols is None:
        resolved_symbols = feature_store.get_available_symbols(
            timeframe=timeframe,
            feature_set_id=resolved_feature_set_id,
        )
        if materialize_missing and not resolved_symbols:
            resolved_symbols = db_loader.get_available_symbols(timeframe=timeframe)
            if resolved_symbols:
                (
                    resolved_feature_set_id,
                    resolved_groups,
                    _,
                ) = resolve_training_feature_set_id(
                    symbols=resolved_symbols,
                    timeframe=timeframe,
                    feature_set_id=feature_set_id,
                    enable_reference_features=enable_reference_features,
                    feature_groups=feature_groups,
                )
    elif materialize_missing:
        available_features = set(
            feature_store.get_available_symbols(
                timeframe=timeframe,
                feature_set_id=resolved_feature_set_id,
            )
        )
        missing_symbols = [symbol for symbol in resolved_symbols if symbol not in available_features]
        if missing_symbols:
            logger.info(
                "Materializing missing features for %d symbols (%s, feature_set_id=%s)",
                len(missing_symbols),
                timeframe,
                resolved_feature_set_id,
            )
            resolved_feature_set_id = materialize_missing_features(
                output_dir=output_dir,
                symbols=resolved_symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                feature_set_id=feature_set_id,
                enable_reference_features=enable_reference_features,
                allow_feature_group_fallback=allow_feature_group_fallback,
                windows_force_fallback_features=windows_force_fallback_features,
                feature_groups=resolved_groups,
            )

    if resolved_symbols is None:
        resolved_symbols = []
    logger.info(
        "Found %d symbols with exportable features in database for timeframe=%s feature_set_id=%s",
        len(resolved_symbols),
        timeframe,
        resolved_feature_set_id,
    )

    results = {}
    for symbol in resolved_symbols:
        try:
            output_path = features_dir / f"{symbol}_features.parquet"
            feature_store.export_to_parquet(
                symbol,
                output_path,
                start_date,
                end_date,
                timeframe=timeframe,
                feature_set_id=resolved_feature_set_id,
            )
            results[symbol] = output_path
            logger.info(f"Exported features for {symbol}")
        except Exception as e:
            logger.error(f"Failed to export features for {symbol}: {e}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export PostgreSQL data to Parquet for ML training"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/training"),
        help="Output directory for Parquet files",
    )
    parser.add_argument(
        "--symbols", "-s",
        nargs="+",
        help="Symbols to export (default: all)",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Load symbols from a newline/comma separated text file or JSON payload.",
    )
    parser.add_argument(
        "--start-date",
        type=lambda s: datetime.fromisoformat(s),
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=lambda s: datetime.fromisoformat(s),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--ohlcv-only",
        action="store_true",
        help="Export only OHLCV data",
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Export only feature data",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=DEFAULT_TIMEFRAME,
        help="Bar timeframe namespace for OHLCV/features (default: %(default)s)",
    )
    parser.add_argument(
        "--feature-set-id",
        type=str,
        default="default",
        help="Feature namespace to export (default: %(default)s)",
    )
    parser.add_argument(
        "--materialize-missing-features",
        action="store_true",
        help="Compute and persist missing feature rows before export.",
    )
    parser.add_argument(
        "--disable-reference-features",
        action="store_true",
        help="Disable point-in-time reference feature augmentation during materialization.",
    )
    parser.add_argument(
        "--allow-partial-feature-fallback",
        action="store_true",
        help="Allow deterministic fallback if a requested feature group cannot be materialized.",
    )
    parser.add_argument(
        "--windows-fallback-features",
        action="store_true",
        help="Force the basic Windows-safe fallback feature pipeline during materialization.",
    )

    args = parser.parse_args()
    symbols = list(args.symbols or [])
    if args.symbols_file is not None:
        symbols.extend(_load_symbols_file(args.symbols_file))
    normalized_symbols = _normalize_symbols(symbols)
    args.symbols = normalized_symbols if normalized_symbols else None

    logger.info(f"Exporting training data to {args.output}")

    total_exported = 0

    # Export OHLCV data
    if not args.features_only:
        ohlcv_results = export_ohlcv_data(
            args.output,
            args.symbols,
            args.start_date,
            args.end_date,
            timeframe=args.timeframe,
        )
        total_exported += len(ohlcv_results)
        logger.info(f"Exported OHLCV data for {len(ohlcv_results)} symbols")

    # Export feature data
    if not args.ohlcv_only:
        features_results = export_features_data(
            args.output,
            args.symbols,
            args.start_date,
            args.end_date,
            timeframe=args.timeframe,
            feature_set_id=args.feature_set_id,
            materialize_missing=args.materialize_missing_features,
            enable_reference_features=not args.disable_reference_features,
            allow_feature_group_fallback=args.allow_partial_feature_fallback,
            windows_force_fallback_features=args.windows_fallback_features,
        )
        total_exported += len(features_results)
        logger.info(f"Exported features for {len(features_results)} symbols")

    logger.info(f"Export complete. Total files: {total_exported}")


if __name__ == "__main__":
    main()
