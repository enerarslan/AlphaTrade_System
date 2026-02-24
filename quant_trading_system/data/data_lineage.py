"""Deprecated compatibility layer for `quant_trading_system.data.lineage`.

This module preserves legacy imports while delegating implementation to
`lineage.py`. New code should import directly from `quant_trading_system.data.lineage`.
"""

from __future__ import annotations

import warnings
from enum import Enum
from pathlib import Path

from quant_trading_system.data.lineage import (
    DataLineageTracker as _DataLineageTracker,
    DataQualityFlag,
    LineageEdge,
    LineageEvent,
    LineageEventType,
    LineageNode,
    PersistentDataLineageTracker,
    get_lineage_tracker,
)

warnings.warn(
    "`quant_trading_system.data.data_lineage` is deprecated; use "
    "`quant_trading_system.data.lineage` instead.",
    DeprecationWarning,
    stacklevel=2,
)


class DataSourceType(str, Enum):
    """Legacy source type enum retained for backward compatibility."""

    ALPACA_API = "alpaca_api"
    CSV_FILE = "csv_file"
    PARQUET_FILE = "parquet_file"
    DATABASE = "database"
    FEATURE_STORE = "feature_store"
    REAL_TIME_FEED = "real_time_feed"
    COMPUTED = "computed"
    EXTERNAL_API = "external_api"


class TransformationType(str, Enum):
    """Legacy transformation enum retained for backward compatibility."""

    LOAD = "load"
    CLEAN = "clean"
    NORMALIZE = "normalize"
    FEATURE_ENGINEERING = "feature_engineering"
    AGGREGATION = "aggregation"
    FILTER = "filter"
    JOIN = "join"
    RESAMPLE = "resample"
    IMPUTE = "impute"
    SCALE = "scale"
    ENCODE = "encode"
    SPLIT = "split"


class DataLineageTracker(_DataLineageTracker):
    """Legacy constructor adapter around the canonical tracker."""

    def __init__(
        self,
        persist_path: str | Path | None = None,
        auto_persist: bool = True,
    ):
        super().__init__(storage_path=persist_path)
        self.auto_persist = auto_persist


__all__ = [
    "DataLineageTracker",
    "DataQualityFlag",
    "DataSourceType",
    "LineageEdge",
    "LineageEvent",
    "LineageEventType",
    "LineageNode",
    "PersistentDataLineageTracker",
    "TransformationType",
    "get_lineage_tracker",
]
