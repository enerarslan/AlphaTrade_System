"""
Point-in-time reference and event feature augmentation for training.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, time, timezone, timedelta
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from sqlalchemy import or_, select

from quant_trading_system.database.connection import DatabaseManager, get_db_manager
from quant_trading_system.database.models import (
    CorporateAction,
    EarningsEvent,
    FailsToDeliver,
    FundamentalSnapshot,
    MacroObservation,
    MacroVintageObservation,
    NewsArticle,
    SECFiling,
    ShortSaleVolume,
)
from quant_trading_system.database.schema_sync import ensure_reference_schema_extensions

logger = logging.getLogger(__name__)

_US_TRADING_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
_DAY_NS = int(pd.Timedelta(days=1).value)
_HOUR_NS = int(pd.Timedelta(hours=1).value)
_EVENT_AGE_FILL = 9999.0
_DAILY_MACRO_SERIES = ("VIX", "VIX9D", "VVIX", "OVX", "GVZ", "DGS2", "DGS10")
_PIT_VINTAGE_SERIES = ("FEDFUNDS", "CPIAUCSL", "UNRATE", "PAYEMS", "RSAFS", "DGORDER", "GDPC1")
_FTD_ZIP_PATTERN = re.compile(r"cnsfails(?P<year>\d{4})(?P<month>\d{2})(?P<half>[ab])", re.IGNORECASE)


@dataclass(slots=True)
class ReferenceFeatureConfig:
    enable_macro_features: bool = True
    enable_short_sale_features: bool = True
    enable_news_features: bool = True
    enable_sec_filing_features: bool = True
    enable_fundamental_features: bool = True
    enable_earnings_features: bool = True
    enable_ftd_features: bool = True
    enable_corporate_action_features: bool = True


def _coerce_utc_timestamp(values: Any) -> pd.Series:
    timestamps = pd.to_datetime(values, utc=True, errors="coerce")
    if isinstance(timestamps, pd.Series):
        return pd.Series(
            pd.DatetimeIndex(timestamps).as_unit("ns"),
            index=timestamps.index,
            dtype="datetime64[ns, UTC]",
        )
    return pd.Series(pd.DatetimeIndex(timestamps).as_unit("ns"), dtype="datetime64[ns, UTC]")


def _timestamp_ns_array(values: Any) -> np.ndarray:
    """Convert timezone-aware timestamps to nanosecond epoch integers."""
    timestamps = _coerce_utc_timestamp(values)
    return pd.DatetimeIndex(timestamps).as_unit("ns").asi8


def _series_to_float(values: Any) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if isinstance(numeric, pd.Series):
        return numeric.astype(float)
    if isinstance(numeric, pd.Index):
        return pd.Series(numeric, dtype=float)
    return pd.Series(numeric, dtype=float)


def _next_business_midnight(value: Any) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return pd.Timestamp(ts.date(), tz="UTC") + _US_TRADING_DAY


def _month_end(day: date) -> date:
    anchor = pd.Timestamp(day).replace(day=1) + pd.offsets.MonthEnd(0)
    return anchor.date()


def _date_midnight(value: Any) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    return pd.Timestamp(pd.Timestamp(value).date(), tz="UTC")


def _date_end_of_day(value: Any) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    return pd.Timestamp.combine(
        pd.Timestamp(value).date(),
        time(hour=23, minute=59, second=59),
        tzinfo=timezone.utc,
    )


def _safe_days_since(current_ts: pd.Series, last_ts: pd.Series) -> np.ndarray:
    age = np.full(len(current_ts), _EVENT_AGE_FILL, dtype=float)
    valid = current_ts.notna() & last_ts.notna()
    if valid.any():
        delta = current_ts.loc[valid] - last_ts.loc[valid]
        age[valid.to_numpy()] = np.maximum(delta.dt.total_seconds().to_numpy() / 86400.0, 0.0)
    return age


def _normalize_symbol_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip().upper() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        for delimiter in ("[", "]", "\"", "'"):
            stripped = stripped.replace(delimiter, "")
        if "," in stripped:
            return [part.strip().upper() for part in stripped.split(",") if part.strip()]
        return [stripped.upper()]
    return []


def _infer_ftd_publish_timestamp(
    settlement_date: Any,
    metadata: dict[str, Any] | None,
) -> pd.Timestamp:
    """Infer a conservative publication timestamp for SEC FTD batches."""
    candidate = ""
    if isinstance(metadata, dict):
        candidate = str(metadata.get("zip_file") or metadata.get("member") or "").strip()
    match = _FTD_ZIP_PATTERN.search(candidate)
    if match:
        year = int(match.group("year"))
        month = int(match.group("month"))
        half = match.group("half").lower()
        if half == "a":
            publish_anchor = _month_end(date(year, month, 1))
        else:
            next_month = pd.Timestamp(date(year, month, 1)) + pd.offsets.MonthBegin(1)
            publish_anchor = date(next_month.year, next_month.month, 15)
        return _next_business_midnight(publish_anchor)

    parsed_settlement = pd.Timestamp(settlement_date).date() if settlement_date is not None else None
    if parsed_settlement is None:
        return pd.NaT
    # Conservative fallback when batch metadata is unavailable.
    return _next_business_midnight(parsed_settlement + timedelta(days=21))


def _count_events_in_windows(
    bar_ns: np.ndarray,
    event_ns: np.ndarray,
    windows_ns: dict[str, int],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    counts = {label: np.zeros(len(bar_ns), dtype=float) for label in windows_ns}
    age_days = np.full(len(bar_ns), _EVENT_AGE_FILL, dtype=float)
    if len(bar_ns) == 0 or len(event_ns) == 0:
        return counts, age_days

    event_ns = np.sort(event_ns.astype(np.int64, copy=False))
    right = np.searchsorted(event_ns, bar_ns, side="right")
    valid = right > 0
    if np.any(valid):
        age_days[valid] = np.maximum((bar_ns[valid] - event_ns[right[valid] - 1]) / _DAY_NS, 0.0)

    for label, window_ns in windows_ns.items():
        left = np.searchsorted(event_ns, bar_ns - window_ns, side="left")
        counts[label] = (right - left).astype(float)
    return counts, age_days


class ReferenceFeatureBuilder:
    def __init__(
        self,
        db_manager: DatabaseManager | None = None,
        config: ReferenceFeatureConfig | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        self.db_manager = db_manager or get_db_manager()
        if hasattr(self.db_manager, "engine"):
            ensure_reference_schema_extensions(self.db_manager)
        self.config = config or ReferenceFeatureConfig()
        self.logger = logger_ or logger
        self._warned_sources: set[str] = set()

    def augment(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """Augment raw training feature matrix with reference/event features."""
        if feature_matrix is None or feature_matrix.empty:
            return feature_matrix

        base = feature_matrix.copy()
        base["timestamp"] = _coerce_utc_timestamp(base["timestamp"])
        base = base.dropna(subset=["symbol", "timestamp"]).copy()
        if base.empty:
            return feature_matrix

        base["symbol"] = base["symbol"].astype(str).str.upper().str.strip()
        base["__row_id"] = np.arange(len(base), dtype=np.int64)
        base = base.sort_values(["symbol", "timestamp", "__row_id"]).reset_index(drop=True)

        symbols = sorted({symbol for symbol in base["symbol"].tolist() if symbol})
        min_timestamp = base["timestamp"].min()
        max_timestamp = base["timestamp"].max()
        min_trade_date = pd.Timestamp(min_timestamp).date()
        max_trade_date = pd.Timestamp(max_timestamp).date()

        if self.config.enable_macro_features:
            base = self._augment_macro_features(base, max_trade_date=max_trade_date)
        if self.config.enable_short_sale_features:
            base = self._augment_short_sale_features(
                base,
                symbols=symbols,
                min_trade_date=min_trade_date,
                max_trade_date=max_trade_date,
            )
        if self.config.enable_news_features:
            base = self._augment_news_features(
                base,
                symbols=symbols,
                min_timestamp=min_timestamp,
                max_timestamp=max_timestamp,
            )
        if self.config.enable_sec_filing_features:
            base = self._augment_sec_filing_features(
                base,
                symbols=symbols,
                min_timestamp=min_timestamp,
                max_timestamp=max_timestamp,
            )
        if self.config.enable_fundamental_features:
            base = self._augment_fundamental_features(
                base,
                symbols=symbols,
                max_trade_date=max_trade_date,
            )
        if self.config.enable_earnings_features:
            base = self._augment_earnings_features(
                base,
                symbols=symbols,
                min_timestamp=min_timestamp,
                max_timestamp=max_timestamp,
            )
        if self.config.enable_ftd_features:
            base = self._augment_fails_to_deliver_features(
                base,
                symbols=symbols,
                min_trade_date=min_trade_date,
                max_trade_date=max_trade_date,
            )
        if self.config.enable_corporate_action_features:
            base = self._augment_corporate_action_features(
                base,
                symbols=symbols,
                max_trade_date=max_trade_date,
            )

        self._warn_unsafe_sources()
        return (
            base.sort_values("__row_id")
            .drop(columns=["__row_id"], errors="ignore")
            .reset_index(drop=True)
        )

    def _warn_once(self, source: str, message: str) -> None:
        if source in self._warned_sources:
            return
        self.logger.warning(message)
        self._warned_sources.add(source)

    def _warn_unsafe_sources(self) -> None:
        return None

    def _read_rows(self, statement: Any) -> list[tuple[Any, ...]]:
        with self.db_manager.session() as session:
            return list(session.execute(statement).all())

    def _merge_global_asof(self, base: pd.DataFrame, ref: pd.DataFrame, *, right_on: str) -> pd.DataFrame:
        if ref.empty:
            return base
        left = base.sort_values(["timestamp", "__row_id"]).copy()
        right = ref.sort_values(right_on).copy()
        left["timestamp"] = _coerce_utc_timestamp(left["timestamp"])
        right[right_on] = _coerce_utc_timestamp(right[right_on])
        merged = pd.merge_asof(
            left,
            right,
            left_on="timestamp",
            right_on=right_on,
            direction="backward",
            allow_exact_matches=True,
        )
        return merged.sort_values("__row_id").reset_index(drop=True)

    def _merge_symbol_asof(self, base: pd.DataFrame, ref: pd.DataFrame, *, right_on: str) -> pd.DataFrame:
        if ref.empty:
            return base
        right_only_cols = [column for column in ref.columns if column != "symbol"]
        merged_parts: list[pd.DataFrame] = []

        for symbol, left_symbol in base.groupby("symbol", sort=False):
            left = left_symbol.sort_values(["timestamp", "__row_id"]).copy()
            left["timestamp"] = _coerce_utc_timestamp(left["timestamp"])

            right = ref[ref["symbol"] == symbol].copy()
            if right.empty:
                for column in right_only_cols:
                    if column not in left.columns:
                        left[column] = np.nan
                merged_parts.append(left)
                continue

            right[right_on] = _coerce_utc_timestamp(right[right_on])
            right = right.dropna(subset=[right_on]).drop(columns=["symbol"], errors="ignore").sort_values(right_on)
            if right.empty:
                for column in right_only_cols:
                    if column not in left.columns:
                        left[column] = np.nan
                merged_parts.append(left)
                continue
            merged_symbol = pd.merge_asof(
                left,
                right,
                left_on="timestamp",
                right_on=right_on,
                direction="backward",
                allow_exact_matches=True,
            )
            merged_parts.append(merged_symbol)

        return pd.concat(merged_parts, ignore_index=True).sort_values("__row_id").reset_index(drop=True)

    def _augment_macro_features(self, base: pd.DataFrame, *, max_trade_date: date) -> pd.DataFrame:
        daily_rows = self._read_rows(
            select(
                MacroObservation.series_id,
                MacroObservation.observation_date,
                MacroObservation.value,
            ).where(
                MacroObservation.series_id.in_(_DAILY_MACRO_SERIES),
                MacroObservation.observation_date <= max_trade_date,
            )
        )
        vintage_rows = self._read_rows(
            select(
                MacroVintageObservation.series_id,
                MacroVintageObservation.observation_date,
                MacroVintageObservation.realtime_start,
                MacroVintageObservation.value,
            ).where(
                MacroVintageObservation.series_id.in_(_PIT_VINTAGE_SERIES),
                MacroVintageObservation.realtime_start <= max_trade_date,
                MacroVintageObservation.observation_date <= max_trade_date,
            )
        )

        release_frames: list[pd.DataFrame] = []

        if daily_rows:
            daily = pd.DataFrame(daily_rows, columns=["series_id", "observation_date", "value"])
            daily["value"] = _series_to_float(daily["value"])
            daily = daily.dropna(subset=["series_id", "observation_date", "value"]).copy()
            if not daily.empty:
                daily["release_timestamp"] = _coerce_utc_timestamp(
                    daily["observation_date"].map(_next_business_midnight)
                )
                daily_pivot = (
                    daily.pivot_table(
                        index="release_timestamp",
                        columns="series_id",
                        values="value",
                        aggfunc="last",
                    )
                    .sort_index()
                    .ffill()
                )
                daily_pivot = daily_pivot.rename(
                    columns={series: f"macro_{series.lower()}_level" for series in daily_pivot.columns}
                )
                for column in list(daily_pivot.columns):
                    daily_pivot[f"{column}_delta"] = daily_pivot[column].diff()
                if {"macro_dgs10_level", "macro_dgs2_level"}.issubset(daily_pivot.columns):
                    daily_pivot["macro_term_spread_10y_2y"] = (
                        daily_pivot["macro_dgs10_level"] - daily_pivot["macro_dgs2_level"]
                    )
                if {"macro_vix_level", "macro_vix9d_level"}.issubset(daily_pivot.columns):
                    daily_pivot["macro_vix_curve"] = (
                        daily_pivot["macro_vix_level"] - daily_pivot["macro_vix9d_level"]
                    )
                if {"macro_vvix_level", "macro_vix_level"}.issubset(daily_pivot.columns):
                    denominator = daily_pivot["macro_vix_level"].replace(0.0, np.nan)
                    daily_pivot["macro_vol_of_vol_ratio"] = (
                        daily_pivot["macro_vvix_level"] / denominator
                    ).replace([np.inf, -np.inf], np.nan)
                release_frames.append(daily_pivot.reset_index())

        if vintage_rows:
            vintage = pd.DataFrame(
                vintage_rows,
                columns=["series_id", "observation_date", "realtime_start", "value"],
            )
            vintage["value"] = _series_to_float(vintage["value"])
            vintage = vintage.dropna(subset=["series_id", "observation_date", "realtime_start", "value"]).copy()
            if not vintage.empty:
                latest_prints = (
                    vintage.sort_values(["series_id", "realtime_start", "observation_date"])
                    .groupby(["series_id", "realtime_start"], as_index=False)
                    .tail(1)
                    .copy()
                )
                latest_prints["release_timestamp"] = _coerce_utc_timestamp(
                    latest_prints["realtime_start"].map(_next_business_midnight)
                )
                vintage_pivot = (
                    latest_prints.pivot_table(
                        index="release_timestamp",
                        columns="series_id",
                        values="value",
                        aggfunc="last",
                    )
                    .sort_index()
                    .ffill()
                )
                vintage_pivot = vintage_pivot.rename(
                    columns={series: f"macro_{series.lower()}_pit_level" for series in vintage_pivot.columns}
                )
                for column in list(vintage_pivot.columns):
                    vintage_pivot[f"{column}_delta"] = vintage_pivot[column].diff()
                release_frames.append(vintage_pivot.reset_index())

        if not release_frames:
            return base

        macro_ref = (
            pd.concat(release_frames, ignore_index=True, sort=False)
            .sort_values("release_timestamp")
            .groupby("release_timestamp", as_index=False)
            .last()
        )
        merged = self._merge_global_asof(base, macro_ref, right_on="release_timestamp")
        merged["macro_days_since_release"] = _safe_days_since(merged["timestamp"], merged["release_timestamp"])
        feature_cols = [
            col
            for col in merged.columns
            if col.startswith("macro_") and col != "macro_days_since_release"
        ]
        for column in feature_cols:
            merged[column] = _series_to_float(merged[column]).fillna(0.0)
        merged["macro_days_since_release"] = _series_to_float(merged["macro_days_since_release"]).fillna(
            _EVENT_AGE_FILL
        )
        return merged.drop(columns=["release_timestamp"], errors="ignore")

    def _augment_short_sale_features(
        self,
        base: pd.DataFrame,
        *,
        symbols: list[str],
        min_trade_date: date,
        max_trade_date: date,
    ) -> pd.DataFrame:
        rows = self._read_rows(
            select(
                ShortSaleVolume.symbol,
                ShortSaleVolume.trade_date,
                ShortSaleVolume.short_volume,
                ShortSaleVolume.short_exempt_volume,
                ShortSaleVolume.total_volume,
            ).where(
                ShortSaleVolume.symbol.in_(symbols),
                ShortSaleVolume.trade_date >= min_trade_date - timedelta(days=45),
                ShortSaleVolume.trade_date <= max_trade_date,
            )
        )
        if not rows:
            return base

        short_df = pd.DataFrame(
            rows,
            columns=["symbol", "trade_date", "short_volume", "short_exempt_volume", "total_volume"],
        )
        short_df["symbol"] = short_df["symbol"].astype(str).str.upper().str.strip()
        for column in ("short_volume", "short_exempt_volume", "total_volume"):
            short_df[column] = _series_to_float(short_df[column]).fillna(0.0)
        short_df = (
            short_df.groupby(["symbol", "trade_date"], as_index=False)[
                ["short_volume", "short_exempt_volume", "total_volume"]
            ]
            .sum()
        )
        denominator = short_df["total_volume"].replace(0.0, np.nan)
        short_df["ref_short_volume_ratio"] = (
            (short_df["short_volume"] + short_df["short_exempt_volume"]) / denominator
        ).replace([np.inf, -np.inf], np.nan)
        short_df["ref_short_exempt_ratio"] = (
            short_df["short_exempt_volume"] / denominator
        ).replace([np.inf, -np.inf], np.nan)
        short_df["ref_short_total_volume_log1p"] = np.log1p(short_df["total_volume"].clip(lower=0.0))
        short_df["effective_timestamp"] = _coerce_utc_timestamp(
            short_df["trade_date"].map(_next_business_midnight)
        )
        short_df = short_df.sort_values(["symbol", "effective_timestamp"]).copy()
        short_df["ref_short_volume_ratio_delta"] = (
            short_df.groupby("symbol", sort=False)["ref_short_volume_ratio"].diff().fillna(0.0)
        )

        ref = short_df[
            [
                "symbol",
                "effective_timestamp",
                "ref_short_volume_ratio",
                "ref_short_exempt_ratio",
                "ref_short_total_volume_log1p",
                "ref_short_volume_ratio_delta",
            ]
        ].copy()
        merged = self._merge_symbol_asof(base, ref, right_on="effective_timestamp")
        merged["ref_short_days_since_update"] = _safe_days_since(merged["timestamp"], merged["effective_timestamp"])
        for column in (
            "ref_short_volume_ratio",
            "ref_short_exempt_ratio",
            "ref_short_total_volume_log1p",
            "ref_short_volume_ratio_delta",
        ):
            merged[column] = _series_to_float(merged[column]).fillna(0.0)
        merged["ref_short_days_since_update"] = _series_to_float(merged["ref_short_days_since_update"]).fillna(
            _EVENT_AGE_FILL
        )
        return merged.drop(columns=["effective_timestamp"], errors="ignore")

    def _load_news_articles(
        self,
        *,
        min_timestamp: pd.Timestamp,
        max_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        rows = self._read_rows(
            select(
                NewsArticle.article_id,
                NewsArticle.created_at_source,
                NewsArticle.symbols,
                NewsArticle.sentiment,
            ).where(
                NewsArticle.created_at_source >= (min_timestamp - pd.Timedelta(days=8)).to_pydatetime(),
                NewsArticle.created_at_source <= max_timestamp.to_pydatetime(),
            )
        )
        if not rows:
            return pd.DataFrame(columns=["article_id", "event_timestamp", "symbol", "sentiment"])

        news_df = pd.DataFrame(rows, columns=["article_id", "created_at_source", "symbols", "sentiment"])
        news_df["event_timestamp"] = _coerce_utc_timestamp(news_df["created_at_source"])
        news_df["symbols"] = news_df["symbols"].map(_normalize_symbol_list)
        news_df = news_df.dropna(subset=["event_timestamp"]).copy()
        news_df = news_df[news_df["symbols"].map(bool)].copy()
        if news_df.empty:
            return pd.DataFrame(columns=["article_id", "event_timestamp", "symbol", "sentiment"])
        news_df = news_df.explode("symbols").rename(columns={"symbols": "symbol"})
        news_df["symbol"] = news_df["symbol"].astype(str).str.upper().str.strip()
        news_df["sentiment"] = _series_to_float(news_df["sentiment"])
        return news_df[["article_id", "event_timestamp", "symbol", "sentiment"]].copy()

    def _augment_news_features(
        self,
        base: pd.DataFrame,
        *,
        symbols: list[str],
        min_timestamp: pd.Timestamp,
        max_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        base["ref_news_count_6h"] = 0.0
        base["ref_news_count_1d"] = 0.0
        base["ref_news_count_7d"] = 0.0
        base["ref_news_days_since_last"] = _EVENT_AGE_FILL

        news_df = self._load_news_articles(min_timestamp=min_timestamp, max_timestamp=max_timestamp)
        if news_df.empty:
            return base

        news_df = news_df[news_df["symbol"].isin(set(symbols))].copy()
        if news_df.empty:
            return base

        windows_ns = {"6h": 6 * _HOUR_NS, "1d": 24 * _HOUR_NS, "7d": 7 * _DAY_NS}
        for symbol, row_index in base.groupby("symbol", sort=False).groups.items():
            event_times = news_df.loc[news_df["symbol"] == symbol, "event_timestamp"].sort_values()
            if event_times.empty:
                continue
            bar_ns = _timestamp_ns_array(base.loc[row_index, "timestamp"])
            counts, age_days = _count_events_in_windows(
                bar_ns,
                _timestamp_ns_array(event_times),
                windows_ns,
            )
            base.loc[row_index, "ref_news_count_6h"] = counts["6h"]
            base.loc[row_index, "ref_news_count_1d"] = counts["1d"]
            base.loc[row_index, "ref_news_count_7d"] = counts["7d"]
            base.loc[row_index, "ref_news_days_since_last"] = age_days
        return base

    def _load_sec_filings(
        self,
        *,
        symbols: list[str],
        min_timestamp: pd.Timestamp,
        max_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        rows = self._read_rows(
            select(
                SECFiling.symbol,
                SECFiling.form,
                SECFiling.accepted_at,
                SECFiling.filed_date,
            ).where(
                SECFiling.symbol.in_(symbols),
                SECFiling.filed_date.is_not(None),
                SECFiling.filed_date >= (min_timestamp - pd.Timedelta(days=370)).date(),
                SECFiling.filed_date <= max_timestamp.date(),
            )
        )
        if not rows:
            return pd.DataFrame(columns=["symbol", "form", "event_timestamp"])

        filings = pd.DataFrame(rows, columns=["symbol", "form", "accepted_at", "filed_date"])
        filings["symbol"] = filings["symbol"].astype(str).str.upper().str.strip()
        filings["accepted_at"] = _coerce_utc_timestamp(filings["accepted_at"])
        filings["event_timestamp"] = filings["accepted_at"]
        missing_accept = filings["event_timestamp"].isna()
        if missing_accept.any():
            filings.loc[missing_accept, "event_timestamp"] = _coerce_utc_timestamp(
                filings.loc[missing_accept, "filed_date"].map(_date_end_of_day)
            )
        filings["form"] = filings["form"].astype(str).str.upper().str.strip()
        filings = filings.dropna(subset=["event_timestamp", "symbol"]).copy()
        return filings[["symbol", "form", "event_timestamp"]].copy()

    def _augment_sec_filing_features(
        self,
        base: pd.DataFrame,
        *,
        symbols: list[str],
        min_timestamp: pd.Timestamp,
        max_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        default_columns = {
            "ref_filing_count_1d": 0.0,
            "ref_filing_count_7d": 0.0,
            "ref_filing_count_30d": 0.0,
            "ref_filing_8k_count_30d": 0.0,
            "ref_filing_10q_count_120d": 0.0,
            "ref_filing_10k_count_365d": 0.0,
            "ref_filing_form4_count_30d": 0.0,
            "ref_filing_days_since_last": _EVENT_AGE_FILL,
        }
        for column, default_value in default_columns.items():
            base[column] = default_value

        filings = self._load_sec_filings(symbols=symbols, min_timestamp=min_timestamp, max_timestamp=max_timestamp)
        if filings.empty:
            return base

        general_windows = {"1d": _DAY_NS, "7d": 7 * _DAY_NS, "30d": 30 * _DAY_NS}
        form_windows = {
            "8-K": ("ref_filing_8k_count_30d", 30 * _DAY_NS),
            "10-Q": ("ref_filing_10q_count_120d", 120 * _DAY_NS),
            "10-K": ("ref_filing_10k_count_365d", 365 * _DAY_NS),
            "4": ("ref_filing_form4_count_30d", 30 * _DAY_NS),
        }

        for symbol, row_index in base.groupby("symbol", sort=False).groups.items():
            symbol_filings = filings[filings["symbol"] == symbol]
            if symbol_filings.empty:
                continue

            bar_ns = _timestamp_ns_array(base.loc[row_index, "timestamp"])
            filing_ns = _timestamp_ns_array(symbol_filings["event_timestamp"].sort_values())
            counts, age_days = _count_events_in_windows(bar_ns, filing_ns, general_windows)
            base.loc[row_index, "ref_filing_count_1d"] = counts["1d"]
            base.loc[row_index, "ref_filing_count_7d"] = counts["7d"]
            base.loc[row_index, "ref_filing_count_30d"] = counts["30d"]
            base.loc[row_index, "ref_filing_days_since_last"] = age_days

            for form_value, (column_name, window_ns) in form_windows.items():
                form_times = symbol_filings.loc[symbol_filings["form"] == form_value, "event_timestamp"].sort_values()
                if form_times.empty:
                    continue
                form_counts, _ = _count_events_in_windows(
                    bar_ns,
                    _timestamp_ns_array(form_times),
                    {column_name: window_ns},
                )
                base.loc[row_index, column_name] = form_counts[column_name]
        return base

    def _load_fundamental_snapshots(self, *, symbols: list[str], max_trade_date: date) -> pd.DataFrame:
        rows = self._read_rows(
            select(
                FundamentalSnapshot.symbol,
                FundamentalSnapshot.as_of_date,
                FundamentalSnapshot.created_at,
                FundamentalSnapshot.market_cap,
                FundamentalSnapshot.shares_outstanding,
                FundamentalSnapshot.pe_ratio,
                FundamentalSnapshot.price_to_book,
                FundamentalSnapshot.dividend_per_share,
                FundamentalSnapshot.dividend_yield,
                FundamentalSnapshot.revenue_ttm,
                FundamentalSnapshot.operating_margin_ttm,
                FundamentalSnapshot.profit_margin,
                FundamentalSnapshot.beta,
                FundamentalSnapshot.week_52_high,
                FundamentalSnapshot.week_52_low,
                FundamentalSnapshot.analyst_target_price,
            ).where(
                FundamentalSnapshot.symbol.in_(symbols),
                FundamentalSnapshot.as_of_date <= max_trade_date,
            )
        )
        if not rows:
            return pd.DataFrame()

        frame = pd.DataFrame(
            rows,
            columns=[
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
            ],
        )
        if frame.empty:
            return frame

        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
        for column in frame.columns:
            if column in {"symbol", "as_of_date", "created_at"}:
                continue
            frame[column] = _series_to_float(frame[column])
        availability_anchor = _coerce_utc_timestamp(frame["created_at"])
        if availability_anchor.notna().any():
            self._warn_once(
                "fundamental_created_at_gate",
                "fundamental_snapshots are gated by immutable created_at next-business-day availability "
                "to prevent current-state backfill leakage.",
            )
            frame["effective_timestamp"] = _coerce_utc_timestamp(
                availability_anchor.map(_next_business_midnight)
            )
        else:
            frame["effective_timestamp"] = _coerce_utc_timestamp(
                frame["as_of_date"].map(_next_business_midnight)
            )
        frame = (
            frame.dropna(subset=["symbol", "effective_timestamp"])
            .sort_values(["symbol", "effective_timestamp", "as_of_date"])
            .groupby(["symbol", "effective_timestamp"], as_index=False)
            .last()
        )
        return frame

    def _augment_fundamental_features(
        self,
        base: pd.DataFrame,
        *,
        symbols: list[str],
        max_trade_date: date,
    ) -> pd.DataFrame:
        default_columns = {
            "ref_market_cap_log1p": 0.0,
            "ref_shares_outstanding_log1p": 0.0,
            "ref_revenue_ttm_log1p": 0.0,
            "ref_pe_ratio": 0.0,
            "ref_price_to_book": 0.0,
            "ref_dividend_per_share": 0.0,
            "ref_dividend_yield": 0.0,
            "ref_operating_margin_ttm": 0.0,
            "ref_profit_margin": 0.0,
            "ref_beta": 0.0,
            "ref_analyst_target_upside": 0.0,
            "ref_price_to_52w_high": 0.0,
            "ref_price_to_52w_low": 0.0,
            "ref_fundamental_days_since_snapshot": _EVENT_AGE_FILL,
        }
        for column, default_value in default_columns.items():
            base[column] = default_value

        fundamentals = self._load_fundamental_snapshots(symbols=symbols, max_trade_date=max_trade_date)
        if fundamentals.empty:
            return base

        if fundamentals.groupby("symbol")["as_of_date"].nunique().max() <= 1:
            self._warn_once(
                "fundamental_snapshots_sparse",
                "fundamental_snapshots joined in point-in-time safe mode, but current dataset is sparse/current-only; "
                "historical coverage before snapshot dates will remain empty until richer PIT ingestion is added.",
            )

        merged = self._merge_symbol_asof(base, fundamentals, right_on="effective_timestamp")
        market_cap = _series_to_float(merged.get("market_cap")).clip(lower=0.0)
        shares_outstanding = _series_to_float(merged.get("shares_outstanding")).clip(lower=0.0)
        revenue_ttm = _series_to_float(merged.get("revenue_ttm")).clip(lower=0.0)
        merged["ref_market_cap_log1p"] = np.log1p(market_cap).fillna(0.0)
        merged["ref_shares_outstanding_log1p"] = np.log1p(shares_outstanding).fillna(0.0)
        merged["ref_revenue_ttm_log1p"] = np.log1p(revenue_ttm).fillna(0.0)
        for source_col, target_col in (
            ("pe_ratio", "ref_pe_ratio"),
            ("price_to_book", "ref_price_to_book"),
            ("dividend_per_share", "ref_dividend_per_share"),
            ("dividend_yield", "ref_dividend_yield"),
            ("operating_margin_ttm", "ref_operating_margin_ttm"),
            ("profit_margin", "ref_profit_margin"),
            ("beta", "ref_beta"),
        ):
            merged[target_col] = _series_to_float(merged.get(source_col)).fillna(0.0)

        merged["ref_fundamental_days_since_snapshot"] = _safe_days_since(
            merged["timestamp"],
            merged["effective_timestamp"],
        )

        close = _series_to_float(merged.get("close"))
        if close.isna().all():
            close = (market_cap / shares_outstanding.replace(0.0, np.nan)).astype(float)
        close = close.replace(0.0, np.nan)
        analyst_target = _series_to_float(merged.get("analyst_target_price"))
        week_52_high = _series_to_float(merged.get("week_52_high")).replace(0.0, np.nan)
        week_52_low = _series_to_float(merged.get("week_52_low")).replace(0.0, np.nan)
        merged["ref_analyst_target_upside"] = (
            analyst_target / close - 1.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        merged["ref_price_to_52w_high"] = (
            close / week_52_high - 1.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        merged["ref_price_to_52w_low"] = (
            close / week_52_low - 1.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        merged["ref_fundamental_days_since_snapshot"] = _series_to_float(
            merged["ref_fundamental_days_since_snapshot"]
        ).fillna(_EVENT_AGE_FILL)
        return merged.drop(
            columns=[
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
                "effective_timestamp",
            ],
            errors="ignore",
        )

    def _load_earnings_events(
        self,
        *,
        symbols: list[str],
        min_timestamp: pd.Timestamp,
        max_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        rows = self._read_rows(
            select(
                EarningsEvent.symbol,
                EarningsEvent.fiscal_date_ending,
                EarningsEvent.reported_date,
                EarningsEvent.announcement_timestamp,
                EarningsEvent.availability_timestamp,
                EarningsEvent.first_seen_at,
                EarningsEvent.created_at,
                EarningsEvent.reported_eps,
                EarningsEvent.estimated_eps,
                EarningsEvent.surprise,
                EarningsEvent.surprise_pct,
            ).where(
                EarningsEvent.symbol.in_(symbols),
                or_(
                    EarningsEvent.announcement_timestamp <= max_timestamp + pd.Timedelta(days=7),
                    EarningsEvent.availability_timestamp <= max_timestamp + pd.Timedelta(days=7),
                    EarningsEvent.first_seen_at <= max_timestamp + pd.Timedelta(days=7),
                    EarningsEvent.created_at <= max_timestamp + pd.Timedelta(days=7),
                ),
            )
        )
        if not rows:
            return pd.DataFrame()

        frame = pd.DataFrame(
            rows,
            columns=[
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
            ],
        )
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
        for column in ("reported_eps", "estimated_eps", "surprise", "surprise_pct"):
            frame[column] = _series_to_float(frame[column])
        announcement_ts = _coerce_utc_timestamp(frame["announcement_timestamp"])
        availability_ts = _coerce_utc_timestamp(frame["availability_timestamp"])
        first_seen_ts = _coerce_utc_timestamp(frame["first_seen_at"])
        legacy_created_ts = _coerce_utc_timestamp(frame["created_at"])
        fallback_first_seen = first_seen_ts.where(first_seen_ts.notna(), legacy_created_ts)

        frame["event_timestamp"] = announcement_ts
        missing_event_ts = frame["event_timestamp"].isna() & availability_ts.notna()
        if missing_event_ts.any():
            frame.loc[missing_event_ts, "event_timestamp"] = availability_ts.loc[missing_event_ts]

        fallback_mask = frame["event_timestamp"].isna() & fallback_first_seen.notna()
        if fallback_mask.any():
            self._warn_once(
                "earnings_first_seen_fallback",
                "earnings_events without explicit announcement/availability timestamps are gated by "
                "immutable first_seen_at next-business-day fallback.",
            )
            frame.loc[fallback_mask, "event_timestamp"] = _coerce_utc_timestamp(
                fallback_first_seen.loc[fallback_mask].map(_next_business_midnight)
            )
        frame = frame.dropna(subset=["symbol", "event_timestamp"]).copy()
        frame = frame.sort_values(["symbol", "event_timestamp", "fiscal_date_ending"]).copy()
        frame["ref_earnings_surprise_abs"] = frame["surprise_pct"].abs().fillna(0.0)
        frame["ref_earnings_positive_surprise"] = (
            frame["surprise_pct"].fillna(0.0) > 0.0
        ).astype(float)
        frame["ref_earnings_surprise_pct_mean_4q"] = (
            frame.groupby("symbol", sort=False)["surprise_pct"]
            .transform(lambda series: series.fillna(0.0).rolling(4, min_periods=1).mean())
            .fillna(0.0)
        )
        return frame

    def _augment_earnings_features(
        self,
        base: pd.DataFrame,
        *,
        symbols: list[str],
        min_timestamp: pd.Timestamp,
        max_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        default_columns = {
            "ref_earnings_count_30d": 0.0,
            "ref_earnings_count_120d": 0.0,
            "ref_earnings_count_365d": 0.0,
            "ref_days_since_earnings": _EVENT_AGE_FILL,
            "ref_last_reported_eps": 0.0,
            "ref_last_estimated_eps": 0.0,
            "ref_last_earnings_surprise": 0.0,
            "ref_last_earnings_surprise_pct": 0.0,
            "ref_last_earnings_surprise_abs": 0.0,
            "ref_last_earnings_positive_surprise": 0.0,
            "ref_earnings_surprise_pct_mean_4q": 0.0,
        }
        for column, default_value in default_columns.items():
            base[column] = default_value

        earnings = self._load_earnings_events(
            symbols=symbols,
            min_timestamp=min_timestamp,
            max_timestamp=max_timestamp,
        )
        if earnings.empty:
            return base

        windows_ns = {"30d": 30 * _DAY_NS, "120d": 120 * _DAY_NS, "365d": 365 * _DAY_NS}
        for symbol, row_index in base.groupby("symbol", sort=False).groups.items():
            symbol_events = earnings[earnings["symbol"] == symbol].copy()
            if symbol_events.empty:
                continue

            bar_ns = _timestamp_ns_array(base.loc[row_index, "timestamp"])
            event_ns = _timestamp_ns_array(symbol_events["event_timestamp"].sort_values())
            counts, age_days = _count_events_in_windows(bar_ns, event_ns, windows_ns)
            base.loc[row_index, "ref_earnings_count_30d"] = counts["30d"]
            base.loc[row_index, "ref_earnings_count_120d"] = counts["120d"]
            base.loc[row_index, "ref_earnings_count_365d"] = counts["365d"]
            base.loc[row_index, "ref_days_since_earnings"] = age_days

            latest = self._merge_symbol_asof(
                base.loc[row_index].drop(
                    columns=[
                        "ref_last_reported_eps",
                        "ref_last_estimated_eps",
                        "ref_last_earnings_surprise",
                        "ref_last_earnings_surprise_pct",
                        "ref_last_earnings_surprise_abs",
                        "ref_last_earnings_positive_surprise",
                        "ref_earnings_surprise_pct_mean_4q",
                    ],
                    errors="ignore",
                ).copy(),
                symbol_events[
                    [
                        "symbol",
                        "event_timestamp",
                        "reported_eps",
                        "estimated_eps",
                        "surprise",
                        "surprise_pct",
                        "ref_earnings_surprise_abs",
                        "ref_earnings_positive_surprise",
                        "ref_earnings_surprise_pct_mean_4q",
                    ]
                ].rename(
                    columns={
                        "reported_eps": "ref_last_reported_eps",
                        "estimated_eps": "ref_last_estimated_eps",
                        "surprise": "ref_last_earnings_surprise",
                        "surprise_pct": "ref_last_earnings_surprise_pct",
                        "ref_earnings_surprise_abs": "ref_last_earnings_surprise_abs",
                        "ref_earnings_positive_surprise": "ref_last_earnings_positive_surprise",
                    }
                ),
                right_on="event_timestamp",
            )
            for column in (
                "ref_last_reported_eps",
                "ref_last_estimated_eps",
                "ref_last_earnings_surprise",
                "ref_last_earnings_surprise_pct",
                "ref_last_earnings_surprise_abs",
                "ref_last_earnings_positive_surprise",
                "ref_earnings_surprise_pct_mean_4q",
            ):
                base.loc[row_index, column] = _series_to_float(latest[column]).fillna(0.0).to_numpy()

        return base

    def _load_fails_to_deliver(
        self,
        *,
        symbols: list[str],
        min_trade_date: date,
        max_trade_date: date,
    ) -> pd.DataFrame:
        rows = self._read_rows(
            select(
                FailsToDeliver.symbol,
                FailsToDeliver.settlement_date,
                FailsToDeliver.quantity,
                FailsToDeliver.price,
                FailsToDeliver.ftd_metadata,
            ).where(
                FailsToDeliver.symbol.in_(symbols),
                FailsToDeliver.settlement_date >= min_trade_date - timedelta(days=180),
                FailsToDeliver.settlement_date <= max_trade_date,
            )
        )
        if not rows:
            return pd.DataFrame()

        frame = pd.DataFrame(
            rows,
            columns=["symbol", "settlement_date", "quantity", "price", "ftd_metadata"],
        )
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
        frame["quantity"] = _series_to_float(frame["quantity"]).fillna(0.0)
        frame["price"] = _series_to_float(frame["price"]).fillna(0.0)
        frame["notional"] = frame["quantity"].clip(lower=0.0) * frame["price"].clip(lower=0.0)
        frame["event_timestamp"] = _coerce_utc_timestamp(
            [
                _infer_ftd_publish_timestamp(settlement_date, metadata)
                for settlement_date, metadata in zip(
                    frame["settlement_date"],
                    frame["ftd_metadata"],
                    strict=False,
                )
            ]
        )
        frame = frame.dropna(subset=["symbol", "event_timestamp"]).copy()
        frame = frame.sort_values(["symbol", "event_timestamp", "settlement_date"]).copy()
        frame["ref_ftd_log_quantity"] = np.log1p(frame["quantity"].clip(lower=0.0))
        frame["ref_ftd_notional_log1p"] = np.log1p(frame["notional"].clip(lower=0.0))
        return frame

    def _augment_fails_to_deliver_features(
        self,
        base: pd.DataFrame,
        *,
        symbols: list[str],
        min_trade_date: date,
        max_trade_date: date,
    ) -> pd.DataFrame:
        default_columns = {
            "ref_ftd_count_30d": 0.0,
            "ref_ftd_count_90d": 0.0,
            "ref_ftd_days_since_last": _EVENT_AGE_FILL,
            "ref_ftd_log_quantity": 0.0,
            "ref_ftd_notional_log1p": 0.0,
            "ref_ftd_price": 0.0,
        }
        for column, default_value in default_columns.items():
            base[column] = default_value

        ftd = self._load_fails_to_deliver(
            symbols=symbols,
            min_trade_date=min_trade_date,
            max_trade_date=max_trade_date,
        )
        if ftd.empty:
            return base

        windows_ns = {"30d": 30 * _DAY_NS, "90d": 90 * _DAY_NS}
        for symbol, row_index in base.groupby("symbol", sort=False).groups.items():
            symbol_ftd = ftd[ftd["symbol"] == symbol].copy()
            if symbol_ftd.empty:
                continue

            bar_ns = _timestamp_ns_array(base.loc[row_index, "timestamp"])
            event_ns = _timestamp_ns_array(symbol_ftd["event_timestamp"].sort_values())
            counts, age_days = _count_events_in_windows(bar_ns, event_ns, windows_ns)
            base.loc[row_index, "ref_ftd_count_30d"] = counts["30d"]
            base.loc[row_index, "ref_ftd_count_90d"] = counts["90d"]
            base.loc[row_index, "ref_ftd_days_since_last"] = age_days

            latest = self._merge_symbol_asof(
                base.loc[row_index].drop(
                    columns=[
                        "ref_ftd_log_quantity",
                        "ref_ftd_notional_log1p",
                        "ref_ftd_price",
                    ],
                    errors="ignore",
                ).copy(),
                symbol_ftd[
                    ["symbol", "event_timestamp", "ref_ftd_log_quantity", "ref_ftd_notional_log1p", "price"]
                ].rename(columns={"price": "ref_ftd_price"}),
                right_on="event_timestamp",
            )
            for column in ("ref_ftd_log_quantity", "ref_ftd_notional_log1p", "ref_ftd_price"):
                base.loc[row_index, column] = _series_to_float(latest[column]).fillna(0.0).to_numpy()

        return base

    def _load_corporate_actions(self, *, symbols: list[str], max_trade_date: date) -> pd.DataFrame:
        rows = self._read_rows(
            select(
                CorporateAction.symbol,
                CorporateAction.action_type,
                CorporateAction.ex_date,
                CorporateAction.amount,
                CorporateAction.split_from,
                CorporateAction.split_to,
            ).where(
                CorporateAction.symbol.in_(symbols),
                CorporateAction.ex_date >= max_trade_date - timedelta(days=365 * 2),
                CorporateAction.ex_date <= max_trade_date,
            )
        )
        if not rows:
            return pd.DataFrame(columns=["symbol", "action_type", "event_timestamp", "amount", "split_ratio"])

        actions = pd.DataFrame(
            rows,
            columns=["symbol", "action_type", "ex_date", "amount", "split_from", "split_to"],
        )
        actions["symbol"] = actions["symbol"].astype(str).str.upper().str.strip()
        actions["action_type"] = actions["action_type"].astype(str).str.upper().str.strip()
        actions["event_timestamp"] = _coerce_utc_timestamp(actions["ex_date"].map(_date_midnight))
        actions["amount"] = _series_to_float(actions["amount"]).fillna(0.0)
        split_from = _series_to_float(actions["split_from"])
        split_to = _series_to_float(actions["split_to"])
        denominator = split_from.replace(0.0, np.nan)
        actions["split_ratio"] = (split_to / denominator).replace([np.inf, -np.inf], np.nan)
        missing_ratio = actions["split_ratio"].isna()
        actions.loc[missing_ratio, "split_ratio"] = split_to[missing_ratio]
        actions["split_ratio"] = actions["split_ratio"].fillna(0.0)
        return actions[["symbol", "action_type", "event_timestamp", "amount", "split_ratio"]].copy()

    def _augment_corporate_action_features(
        self,
        base: pd.DataFrame,
        *,
        symbols: list[str],
        max_trade_date: date,
    ) -> pd.DataFrame:
        default_columns = {
            "ref_corporate_action_count_30d": 0.0,
            "ref_corporate_action_count_365d": 0.0,
            "ref_dividend_count_365d": 0.0,
            "ref_split_count_365d": 0.0,
            "ref_days_since_dividend": _EVENT_AGE_FILL,
            "ref_days_since_split": _EVENT_AGE_FILL,
            "ref_last_dividend_amount": 0.0,
            "ref_last_split_ratio": 0.0,
        }
        for column, default_value in default_columns.items():
            base[column] = default_value

        actions = self._load_corporate_actions(symbols=symbols, max_trade_date=max_trade_date)
        if actions.empty:
            return base

        any_windows = {"30d": 30 * _DAY_NS, "365d": 365 * _DAY_NS}
        annual_window = {"365d": 365 * _DAY_NS}

        for symbol, row_index in base.groupby("symbol", sort=False).groups.items():
            symbol_actions = actions[actions["symbol"] == symbol]
            if symbol_actions.empty:
                continue

            bar_ns = _timestamp_ns_array(base.loc[row_index, "timestamp"])
            any_counts, _ = _count_events_in_windows(
                bar_ns,
                _timestamp_ns_array(symbol_actions["event_timestamp"].sort_values()),
                any_windows,
            )
            base.loc[row_index, "ref_corporate_action_count_30d"] = any_counts["30d"]
            base.loc[row_index, "ref_corporate_action_count_365d"] = any_counts["365d"]

            dividends = symbol_actions[symbol_actions["action_type"] == "DIVIDEND"].copy()
            if not dividends.empty:
                div_counts, div_age = _count_events_in_windows(
                    bar_ns,
                    _timestamp_ns_array(dividends["event_timestamp"].sort_values()),
                    annual_window,
                )
                base.loc[row_index, "ref_dividend_count_365d"] = div_counts["365d"]
                base.loc[row_index, "ref_days_since_dividend"] = div_age
                div_merge = self._merge_symbol_asof(
                    base.loc[row_index].drop(columns=["ref_last_dividend_amount"], errors="ignore").copy(),
                    dividends[["symbol", "event_timestamp", "amount"]].rename(
                        columns={"amount": "ref_last_dividend_amount"}
                    ),
                    right_on="event_timestamp",
                )
                base.loc[row_index, "ref_last_dividend_amount"] = _series_to_float(
                    div_merge["ref_last_dividend_amount"]
                ).fillna(0.0).to_numpy()

            splits = symbol_actions[symbol_actions["action_type"] == "SPLIT"].copy()
            if not splits.empty:
                split_counts, split_age = _count_events_in_windows(
                    bar_ns,
                    _timestamp_ns_array(splits["event_timestamp"].sort_values()),
                    annual_window,
                )
                base.loc[row_index, "ref_split_count_365d"] = split_counts["365d"]
                base.loc[row_index, "ref_days_since_split"] = split_age
                split_merge = self._merge_symbol_asof(
                    base.loc[row_index].drop(columns=["ref_last_split_ratio"], errors="ignore").copy(),
                    splits[["symbol", "event_timestamp", "split_ratio"]].rename(
                        columns={"split_ratio": "ref_last_split_ratio"}
                    ),
                    right_on="event_timestamp",
                )
                base.loc[row_index, "ref_last_split_ratio"] = _series_to_float(
                    split_merge["ref_last_split_ratio"]
                ).fillna(0.0).to_numpy()
        return base
