"""
Multi-timeframe OHLCV resampling and point-in-time feature alignment utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from quant_trading_system.data.timeframe import DEFAULT_TIMEFRAME, normalize_timeframe

_US_TRADING_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

_TIMEFRAME_RULES: dict[str, str] = {
    "1Min": "1min",
    "5Min": "5min",
    "15Min": "15min",
    "30Min": "30min",
    "1Hour": "1h",
    "4Hour": "4h",
    "1Day": "1D",
}

_TIMEFRAME_PRIORITY: dict[str, int] = {
    "1Min": 1,
    "5Min": 5,
    "15Min": 15,
    "30Min": 30,
    "1Hour": 60,
    "4Hour": 240,
    "1Day": 1440,
}

_TIMEFRAME_PREFIXES: dict[str, str] = {
    "1Min": "tf1m",
    "5Min": "tf5m",
    "15Min": "tf15m",
    "30Min": "tf30m",
    "1Hour": "tf1h",
    "4Hour": "tf4h",
    "1Day": "tf1d",
}

_BASE_COLUMNS = {"symbol", "timestamp"}
_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _coerce_utc_ns(values: pd.Series | pd.DatetimeIndex) -> pd.Series:
    timestamps = pd.to_datetime(values, utc=True, errors="coerce")
    return pd.Series(
        pd.DatetimeIndex(timestamps).as_unit("ns"),
        index=getattr(timestamps, "index", None),
        dtype="datetime64[ns, UTC]",
    )


def timeframe_to_rule(timeframe: str) -> str:
    canonical = normalize_timeframe(timeframe, default=DEFAULT_TIMEFRAME)
    try:
        return _TIMEFRAME_RULES[canonical]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe for resampling: {canonical}") from exc


def timeframe_to_prefix(timeframe: str) -> str:
    canonical = normalize_timeframe(timeframe, default=DEFAULT_TIMEFRAME)
    try:
        return _TIMEFRAME_PREFIXES[canonical]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe prefix: {canonical}") from exc


def normalize_timeframes(base_timeframe: str, timeframes: list[str] | None) -> list[str]:
    canonical_base = normalize_timeframe(base_timeframe, default=DEFAULT_TIMEFRAME)
    requested = [canonical_base]
    for raw in timeframes or []:
        normalized = normalize_timeframe(raw, default=canonical_base)
        if normalized not in requested:
            requested.append(normalized)
    return sorted(
        requested,
        key=lambda value: (_TIMEFRAME_PRIORITY.get(value, 10_000), value),
    )


def _next_business_midnight(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return pd.Timestamp(ts.date(), tz="UTC") + _US_TRADING_DAY


def availability_timestamps(
    timestamps: pd.Series | pd.DatetimeIndex,
    timeframe: str,
) -> pd.Series:
    canonical = normalize_timeframe(timeframe, default=DEFAULT_TIMEFRAME)
    ts = _coerce_utc_ns(timestamps)
    if canonical == "1Day":
        return _coerce_utc_ns(ts.map(_next_business_midnight))
    return ts


def resample_ohlcv(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample one or more symbol streams into the requested timeframe.

    The output timestamp represents the completed bar close, which is later used as
    the point-in-time availability anchor during merge-asof alignment.
    """
    canonical = normalize_timeframe(timeframe, default=DEFAULT_TIMEFRAME)
    rule = timeframe_to_rule(canonical)
    required = {"symbol", "timestamp", *_OHLCV_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Cannot resample OHLCV without required columns: {missing}")

    working = frame.copy()
    working["symbol"] = working["symbol"].astype(str).str.upper().str.strip()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    working = working.dropna(subset=["symbol", "timestamp", *_OHLCV_COLUMNS]).copy()
    if working.empty:
        return pd.DataFrame(columns=["symbol", "timestamp", *_OHLCV_COLUMNS])

    parts: list[pd.DataFrame] = []
    for symbol, symbol_frame in working.groupby("symbol", sort=False):
        sdf = symbol_frame.sort_values("timestamp").copy()
        sdf = sdf.set_index("timestamp")
        aggregated = (
            sdf.resample(rule, label="right", closed="right", origin="start_day")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(subset=["open", "high", "low", "close"])
        )
        if canonical == "1Day" and not aggregated.empty:
            aggregated.index = aggregated.index - pd.Timedelta(days=1)
        if aggregated.empty:
            continue
        aggregated["symbol"] = symbol
        parts.append(aggregated.reset_index())

    if not parts:
        return pd.DataFrame(columns=["symbol", "timestamp", *_OHLCV_COLUMNS])

    result = pd.concat(parts, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    return result[["symbol", "timestamp", *_OHLCV_COLUMNS]].sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)


def _symbol_merge_asof(
    base_frame: pd.DataFrame,
    aligned_frame: pd.DataFrame,
    *,
    right_on: str,
) -> pd.DataFrame:
    if aligned_frame.empty:
        return base_frame

    merged_parts: list[pd.DataFrame] = []
    right_only_cols = [column for column in aligned_frame.columns if column not in {"symbol"}]
    for symbol, base_symbol in base_frame.groupby("symbol", sort=False):
        left = base_symbol.sort_values("timestamp").copy()
        right = aligned_frame[aligned_frame["symbol"] == symbol].copy()
        if right.empty:
            for column in right_only_cols:
                if column not in left.columns:
                    left[column] = np.nan
            merged_parts.append(left)
            continue

        left["timestamp"] = _coerce_utc_ns(left["timestamp"])
        right[right_on] = _coerce_utc_ns(right[right_on])
        right = right.dropna(subset=[right_on]).sort_values(right_on).drop(columns=["symbol"])
        merged_parts.append(
            pd.merge_asof(
                left,
                right,
                left_on="timestamp",
                right_on=right_on,
                direction="backward",
                allow_exact_matches=True,
            )
        )

    return pd.concat(merged_parts, ignore_index=True).sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)


@dataclass(slots=True)
class MultiTimeframeFeatureEngine:
    """
    Build deterministic multi-timeframe training matrices without look-ahead.
    """

    base_timeframe: str = DEFAULT_TIMEFRAME
    timeframes: list[str] | None = None

    def normalized_timeframes(self) -> list[str]:
        return normalize_timeframes(self.base_timeframe, self.timeframes)

    def build_ohlcv_frames(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        canonical_base = normalize_timeframe(self.base_timeframe, default=DEFAULT_TIMEFRAME)
        frames: dict[str, pd.DataFrame] = {}
        for timeframe in self.normalized_timeframes():
            if timeframe == canonical_base:
                sdf = frame.copy()
                sdf["symbol"] = sdf["symbol"].astype(str).str.upper().str.strip()
                sdf["timestamp"] = pd.to_datetime(sdf["timestamp"], utc=True, errors="coerce")
                frames[timeframe] = (
                    sdf.dropna(subset=["symbol", "timestamp", *_OHLCV_COLUMNS])
                    .sort_values(["symbol", "timestamp"])
                    .reset_index(drop=True)
                )
            else:
                frames[timeframe] = resample_ohlcv(frame, timeframe)
        return frames

    def align_feature_frames(
        self,
        *,
        base_frame: pd.DataFrame,
        feature_frames: dict[str, pd.DataFrame],
        include_resampled_ohlcv: bool = True,
    ) -> pd.DataFrame:
        working = base_frame.copy()
        working["timestamp"] = _coerce_utc_ns(working["timestamp"])
        working = working.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        base_prefix = timeframe_to_prefix(self.base_timeframe)
        for column in _OHLCV_COLUMNS:
            prefixed = f"{base_prefix}_{column}"
            if column in working.columns and prefixed not in working.columns:
                working[prefixed] = pd.to_numeric(working[column], errors="coerce")

        for timeframe in self.normalized_timeframes():
            if timeframe == normalize_timeframe(self.base_timeframe, default=DEFAULT_TIMEFRAME):
                continue
            frame = feature_frames.get(timeframe)
            if frame is None or frame.empty:
                continue

            prefix = timeframe_to_prefix(timeframe)
            candidate = frame.copy()
            candidate["timestamp"] = _coerce_utc_ns(candidate["timestamp"])
            candidate["feature_timestamp"] = candidate["timestamp"]
            candidate["availability_timestamp"] = availability_timestamps(
                candidate["timestamp"], timeframe
            )

            rename_map: dict[str, str] = {}
            for column in list(candidate.columns):
                if column in _BASE_COLUMNS or column in {"feature_timestamp", "availability_timestamp"}:
                    continue
                if column in _OHLCV_COLUMNS and not include_resampled_ohlcv:
                    candidate = candidate.drop(columns=[column])
                    continue
                rename_map[column] = f"{prefix}_{column}"
            candidate = candidate.rename(columns=rename_map)

            merged = _symbol_merge_asof(
                working,
                candidate[["symbol", "availability_timestamp", *rename_map.values()]],
                right_on="availability_timestamp",
            )
            working = merged.drop(columns=["availability_timestamp"], errors="ignore")

        return self._derive_cross_timeframe_features(working)

    def _derive_cross_timeframe_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy()
        for timeframe in self.normalized_timeframes():
            prefix = timeframe_to_prefix(timeframe)
            open_col = f"{prefix}_open"
            high_col = f"{prefix}_high"
            low_col = f"{prefix}_low"
            close_col = f"{prefix}_close"
            volume_col = f"{prefix}_volume"
            if {open_col, high_col, low_col, close_col}.issubset(enriched.columns):
                close = pd.to_numeric(enriched[close_col], errors="coerce")
                opened = pd.to_numeric(enriched[open_col], errors="coerce").replace(0.0, np.nan)
                high = pd.to_numeric(enriched[high_col], errors="coerce")
                low = pd.to_numeric(enriched[low_col], errors="coerce")
                enriched[f"{prefix}_bar_return"] = (
                    (close / opened) - 1.0
                ).replace([np.inf, -np.inf], np.nan)
                enriched[f"{prefix}_range_pct"] = (
                    (high - low) / close.replace(0.0, np.nan)
                ).replace([np.inf, -np.inf], np.nan)
            if volume_col in enriched.columns:
                volume = pd.to_numeric(enriched[volume_col], errors="coerce")
                enriched[f"{prefix}_volume_log1p"] = np.log1p(volume.clip(lower=0.0))

        base_prefix = timeframe_to_prefix(self.base_timeframe)
        base_close_col = f"{base_prefix}_close"
        base_return_col = f"{base_prefix}_bar_return"
        base_volume_col = f"{base_prefix}_volume"

        for timeframe in self.normalized_timeframes():
            if timeframe == normalize_timeframe(self.base_timeframe, default=DEFAULT_TIMEFRAME):
                continue
            prefix = timeframe_to_prefix(timeframe)
            higher_close_col = f"{prefix}_close"
            higher_return_col = f"{prefix}_bar_return"
            higher_volume_col = f"{prefix}_volume"

            if {base_close_col, higher_close_col}.issubset(enriched.columns):
                base_close = pd.to_numeric(enriched[base_close_col], errors="coerce")
                higher_close = pd.to_numeric(enriched[higher_close_col], errors="coerce").replace(
                    0.0, np.nan
                )
                enriched[f"mtf_close_ratio_{base_prefix}_{prefix}"] = (
                    (base_close / higher_close) - 1.0
                ).replace([np.inf, -np.inf], np.nan)

            if {base_return_col, higher_return_col}.issubset(enriched.columns):
                base_return = pd.to_numeric(enriched[base_return_col], errors="coerce").fillna(0.0)
                higher_return = pd.to_numeric(enriched[higher_return_col], errors="coerce").fillna(
                    0.0
                )
                enriched[f"mtf_return_spread_{base_prefix}_{prefix}"] = base_return - higher_return
                enriched[f"mtf_trend_agreement_{base_prefix}_{prefix}"] = (
                    np.sign(base_return) * np.sign(higher_return)
                ).astype(float)

            if {base_volume_col, higher_volume_col}.issubset(enriched.columns):
                base_volume = pd.to_numeric(enriched[base_volume_col], errors="coerce")
                higher_volume = pd.to_numeric(
                    enriched[higher_volume_col], errors="coerce"
                ).replace(0.0, np.nan)
                enriched[f"mtf_volume_ratio_{base_prefix}_{prefix}"] = (
                    (base_volume / higher_volume) - 1.0
                ).replace([np.inf, -np.inf], np.nan)

        return enriched
