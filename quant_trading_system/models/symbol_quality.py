"""Shared symbol-level quality assessment for training and promotion inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_trading_system.models.training_lineage import estimate_missing_bars_count


@dataclass(frozen=True, slots=True)
class SymbolQualityThresholds:
    """Thresholds for symbol-level universe eligibility."""

    min_rows: int = 1200
    max_missing_ratio: float = 0.12
    max_extreme_move_ratio: float = 0.08
    max_corporate_action_ratio: float = 0.02
    min_median_dollar_volume: float = 1_000_000.0


def assess_symbol_quality(
    frame: pd.DataFrame,
    thresholds: SymbolQualityThresholds | None = None,
) -> dict[str, Any]:
    """Assess raw OHLCV quality for one symbol stream."""
    rules = thresholds or SymbolQualityThresholds()
    if frame is None or frame.empty:
        return {
            "rows": 0,
            "missing_ratio": 1.0,
            "extreme_move_ratio": 1.0,
            "corporate_action_ratio": 1.0,
            "median_dollar_volume": 0.0,
            "passes": False,
            "quality_score": 0.0,
            "reasons": ["empty_frame"],
        }

    working = frame.copy()
    if "timestamp" not in working.columns:
        working = working.reset_index()
        first_column = working.columns[0]
        if str(first_column) != "timestamp":
            working = working.rename(columns={first_column: "timestamp"})

    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    working["close"] = pd.to_numeric(working.get("close"), errors="coerce")
    working["volume"] = pd.to_numeric(working.get("volume"), errors="coerce")
    working = working.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)

    rows = int(len(working))
    if rows == 0:
        return {
            "rows": 0,
            "missing_ratio": 1.0,
            "extreme_move_ratio": 1.0,
            "corporate_action_ratio": 1.0,
            "median_dollar_volume": 0.0,
            "passes": False,
            "quality_score": 0.0,
            "reasons": ["no_valid_rows"],
        }

    timestamps = working["timestamp"].dropna().sort_values()
    if len(timestamps) >= 2:
        missing_count, inferred_bar_seconds = estimate_missing_bars_count(timestamps)
    else:
        missing_count, inferred_bar_seconds = 0, None
    expected_rows = float(rows + max(0, int(missing_count)))
    missing_ratio = float(np.clip(max(0, int(missing_count)) / max(expected_rows, 1.0), 0.0, 1.0))

    returns = working["close"].pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    ret_obs = int(len(returns))
    extreme_ratio = float((returns.abs() > 0.20).mean()) if ret_obs > 0 else 1.0
    corporate_ratio = float((returns.abs() > 0.35).mean()) if ret_obs > 0 else 1.0

    dollar_volume = np.nan_to_num(
        working["close"].to_numpy(dtype=float) * working["volume"].fillna(0.0).to_numpy(dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    median_dollar_volume = float(np.nanmedian(dollar_volume)) if dollar_volume.size > 0 else 0.0

    passes = bool(
        rows >= int(rules.min_rows)
        and missing_ratio <= float(rules.max_missing_ratio)
        and extreme_ratio <= float(rules.max_extreme_move_ratio)
        and corporate_ratio <= float(rules.max_corporate_action_ratio)
        and median_dollar_volume >= float(rules.min_median_dollar_volume)
    )

    score = (
        min(1.0, float(rows) / float(max(1, int(rules.min_rows)))) * 0.35
        + max(0.0, 1.0 - (missing_ratio / max(1e-9, float(rules.max_missing_ratio)))) * 0.25
        + max(0.0, 1.0 - (extreme_ratio / max(1e-9, float(rules.max_extreme_move_ratio)))) * 0.20
        + max(0.0, 1.0 - (corporate_ratio / max(1e-9, float(rules.max_corporate_action_ratio)))) * 0.10
        + min(1.0, median_dollar_volume / max(1.0, float(rules.min_median_dollar_volume))) * 0.10
    )
    quality_score = float(np.clip(score, 0.0, 1.0))

    reasons: list[str] = []
    if rows < int(rules.min_rows):
        reasons.append("insufficient_rows")
    if missing_ratio > float(rules.max_missing_ratio):
        reasons.append("missing_bar_ratio")
    if extreme_ratio > float(rules.max_extreme_move_ratio):
        reasons.append("extreme_move_ratio")
    if corporate_ratio > float(rules.max_corporate_action_ratio):
        reasons.append("corporate_action_ratio")
    if median_dollar_volume < float(rules.min_median_dollar_volume):
        reasons.append("median_dollar_volume")

    return {
        "rows": rows,
        "missing_ratio": float(missing_ratio),
        "extreme_move_ratio": float(extreme_ratio),
        "corporate_action_ratio": float(corporate_ratio),
        "median_dollar_volume": float(median_dollar_volume),
        "inferred_bar_seconds": (
            float(inferred_bar_seconds) if inferred_bar_seconds not in {None, 0.0} else None
        ),
        "passes": passes,
        "quality_score": quality_score,
        "reasons": reasons,
    }
