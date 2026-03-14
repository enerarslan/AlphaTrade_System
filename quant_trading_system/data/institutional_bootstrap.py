"""Multi-source free-data bootstrap for institutional research workflows."""

from __future__ import annotations

import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import requests
import yfinance as yf
import yaml
from dotenv import load_dotenv
from sqlalchemy import case, func
from sqlalchemy.dialects.postgresql import insert

from quant_trading_system.database.connection import get_db_manager
from quant_trading_system.database.models import (
    Base,
    CorporateAction,
    EarningsEvent,
    FundamentalSnapshot,
    MacroObservation,
    NewsArticle,
    SECFiling,
    SecurityMaster,
)
from quant_trading_system.database.schema_sync import ensure_reference_schema_extensions
from quant_trading_system.data.db_loader import get_db_loader
from quant_trading_system.execution.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

load_dotenv(override=True)

ETF_BENCHMARKS = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "TLT",
    "GLD",
    "USO",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
    "XLB",
    "XLC",
    "XLRE",
]

FRED_SERIES = {
    "DGS2": {"name": "US Treasury 2Y", "interval": "daily", "unit": "percent"},
    "DGS10": {"name": "US Treasury 10Y", "interval": "daily", "unit": "percent"},
    "FEDFUNDS": {"name": "Federal Funds Rate", "interval": "monthly", "unit": "percent"},
    "CPIAUCSL": {"name": "CPI Urban Consumers", "interval": "monthly", "unit": "index"},
    "UNRATE": {"name": "US Unemployment Rate", "interval": "monthly", "unit": "percent"},
    "PAYEMS": {"name": "Nonfarm Payrolls", "interval": "monthly", "unit": "thousands"},
    "RSAFS": {"name": "Retail Sales", "interval": "monthly", "unit": "millions_usd"},
    "DGORDER": {"name": "Durable Goods Orders", "interval": "monthly", "unit": "millions_usd"},
    "GDPC1": {"name": "Real GDP", "interval": "quarterly", "unit": "billions_2017_usd"},
}

CBOE_SERIES = {
    "VIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "VIX9D": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv",
    "VVIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv",
    "OVX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/OVX_History.csv",
    "GVZ": "https://cdn.cboe.com/api/global/us_indices/daily_prices/GVZ_History.csv",
}

_EARNINGS_TIMESTAMP_KEYS = (
    "datetime",
    "dateTime",
    "reportDateTime",
    "reportedDateTime",
    "announcementTimestamp",
    "announcementTime",
    "reportTimestamp",
)
_EARNINGS_DATE_KEYS = (
    "date",
    "reportDate",
    "report_date",
    "reportedDate",
    "announcementDate",
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_date(value: Any) -> date | None:
    if value in (None, "", "None", "null", "NaN"):
        return None
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def _parse_datetime_utc(value: Any) -> datetime | None:
    if value in (None, "", "None", "null", "NaN"):
        return None
    try:
        ts = pd.to_datetime(value, utc=True)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def _extract_earnings_timestamp(event: dict[str, Any]) -> datetime | None:
    for key in _EARNINGS_TIMESTAMP_KEYS:
        value = event.get(key)
        parsed = _parse_datetime_utc(value)
        if parsed is not None:
            return parsed
    return None


def _extract_earnings_reported_date(event: dict[str, Any]) -> date | None:
    for key in _EARNINGS_DATE_KEYS:
        value = event.get(key)
        parsed = _parse_date(value)
        if parsed is not None:
            return parsed
    return None


def _resolve_earnings_availability(event: dict[str, Any], *, captured_at: datetime) -> dict[str, Any]:
    """Resolve immutable first-seen and earliest trustworthy availability fields."""
    announcement_timestamp = _extract_earnings_timestamp(event)
    reported_date = _extract_earnings_reported_date(event)
    availability_source = "first_seen_at"
    availability_timestamp = None

    if announcement_timestamp is not None:
        availability_timestamp = announcement_timestamp
        if reported_date is None:
            reported_date = announcement_timestamp.date()
        availability_source = "announcement_timestamp"

    return {
        "reported_date": reported_date,
        "announcement_timestamp": announcement_timestamp,
        "availability_timestamp": availability_timestamp,
        "first_seen_at": captured_at,
        "availability_source": availability_source,
    }


def _parse_decimal(value: Any, multiplier: Decimal | None = None) -> Decimal | None:
    if value in (None, "", "None", "null", "NaN"):
        return None
    try:
        number = Decimal(str(value))
        return number * multiplier if multiplier else number
    except (InvalidOperation, ValueError, TypeError):
        return None


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime()
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray, Decimal, date, datetime)):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _clean_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean_json_value(item) for item in value]

    value = _coerce_scalar(value)
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _clean_record_for_db(record: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (dict, list, tuple, set)):
            cleaned[key] = _clean_json_value(value)
            continue

        value = _coerce_scalar(value)
        if value is None:
            cleaned[key] = None
            continue

        try:
            cleaned[key] = None if pd.isna(value) else value
        except Exception:
            cleaned[key] = value
    return cleaned


def _clean_record_for_export(record: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (dict, list, tuple, set)):
            cleaned[key] = _clean_json_value(value)
            continue

        value = _coerce_scalar(value)
        if value is None:
            cleaned[key] = None
            continue

        try:
            if pd.isna(value):
                cleaned[key] = None
                continue
        except Exception:
            pass

        if isinstance(value, Decimal):
            cleaned[key] = str(value)
        elif isinstance(value, (datetime, date)):
            cleaned[key] = value.isoformat()
        else:
            cleaned[key] = value
    return cleaned


def _batched(items: Sequence[str], size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _batched_records(records: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for record in records:
        batch.append(record)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _safe_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _yf_symbol(symbol: str) -> str:
    return _safe_symbol(symbol).replace(".", "-")


@dataclass
class InstitutionalBootstrapConfig:
    """Configuration for the institutional free-data bootstrap."""

    broad_universe_size: int = 250
    output_root: Path = Path("data")
    sync_db: bool = True
    daily_start: date = field(default_factory=lambda: date.today() - timedelta(days=365 * 15))
    intraday_15m_start: date = field(default_factory=lambda: date.today() - timedelta(days=365 * 5))
    intraday_1m_start: date = field(default_factory=lambda: date.today() - timedelta(days=365 * 2))
    news_start: date = field(default_factory=lambda: date.today() - timedelta(days=365))
    core_universe_path: Path = Path("quant_trading_system/config/symbols.yaml")
    include_news: bool = True
    include_sec: bool = True
    include_macro: bool = True
    include_corporate_actions: bool = True
    include_fundamentals: bool = True
    include_daily_bars: bool = True
    include_intraday_bars: bool = True


class _ProviderGate:
    """Simple per-provider pacing to stay under free-plan limits."""

    def __init__(self) -> None:
        self._last_call: dict[str, float] = {}
        self._min_interval = {
            "alpha_vantage": 1.2,
            "finnhub": 1.10,
            "sec": 0.15,
            "alpaca": 0.10,
            "yfinance": 0.0,
            "fred": 0.0,
            "cboe": 0.0,
        }

    def wait(self, provider: str) -> None:
        minimum = self._min_interval.get(provider, 0.0)
        if minimum <= 0:
            return
        now = time.monotonic()
        last = self._last_call.get(provider, 0.0)
        elapsed = now - last
        if elapsed < minimum:
            time.sleep(minimum - elapsed)
        self._last_call[provider] = time.monotonic()


class InstitutionalDataBootstrapper:
    """Downloads a broad institutional-style research dataset from free sources."""

    def __init__(self, config: InstitutionalBootstrapConfig) -> None:
        self.config = config
        self.output_root = config.output_root
        self.reference_dir = self.output_root / "reference"
        self.raw_dir = self.output_root / "raw"
        self.news_dir = self.output_root / "alternative" / "news"
        self.manifest_path = self.output_root / "export" / "institutional_bootstrap_manifest.json"

        for path in (
            self.reference_dir,
            self.raw_dir,
            self.news_dir,
            self.output_root / "export",
        ):
            path.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "AlphaTradeSystem research@alphatrade.local"})
        self.alpaca_headers = {
            "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
            "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET", ""),
        }
        self.finnhub_token = os.getenv("FINNHUB_API_KEY", "").strip()
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()

        self.provider_gate = _ProviderGate()
        self.db_loader = get_db_loader() if self.config.sync_db else None
        self.db_manager = get_db_manager() if self.config.sync_db else None
        self.profile_cache: dict[str, dict[str, Any]] = {}
        self.metric_cache: dict[str, dict[str, Any]] = {}
        self.alpha_vantage_earnings_cache: dict[str, dict[date, dict[str, Any]]] = {}
        self.sec_ticker_map: dict[str, dict[str, Any]] = {}
        self.manifest: dict[str, Any] = {
            "generated_at": _now_utc().isoformat(),
            "datasets": {},
        }

    def _record_dataset(self, name: str, **details: Any) -> None:
        self.manifest["datasets"][name] = details

    def _get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        provider: str,
        timeout: int = 60,
    ) -> Any:
        for attempt in range(5):
            self.provider_gate.wait(provider)
            response = self.session.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 429:
                time.sleep(min(2 ** attempt, 15))
                continue
            response.raise_for_status()
            data = response.json()
            if provider == "alpha_vantage" and isinstance(data, dict):
                message = str(data.get("Note") or data.get("Information") or "")
                if "rate limit" in message.lower():
                    time.sleep(max(10.0, 2 ** attempt))
                    continue
            return data
        raise RuntimeError(f"{provider} request failed after retries: {url}")

    def _get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        provider: str,
        headers: dict[str, str] | None = None,
        timeout: int = 60,
    ) -> str:
        for attempt in range(5):
            self.provider_gate.wait(provider)
            response = self.session.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 429:
                time.sleep(min(2 ** attempt, 15))
                continue
            response.raise_for_status()
            return response.text
        raise RuntimeError(f"{provider} text request failed after retries: {url}")

    def run(self) -> dict[str, Any]:
        """Execute the bootstrap workflow and return a manifest summary."""
        if self.config.sync_db and self.db_manager is not None:
            with self.db_manager.engine.begin() as conn:
                Base.metadata.create_all(bind=conn)
            ensure_reference_schema_extensions(self.db_manager, force=True)

        core_symbols = self._load_core_symbols()
        broad_symbols = self._build_broad_universe(core_symbols)
        self._write_universe_files(core_symbols, broad_symbols)

        security_records = self.download_security_master(broad_symbols)
        self._upsert_records(SecurityMaster, security_records)
        self._save_table_export("security_master", security_records)

        if self.config.include_fundamentals:
            fundamentals, earnings = self.download_finnhub_company_data(broad_symbols)
            self._upsert_records(FundamentalSnapshot, fundamentals)
            self._upsert_records(EarningsEvent, earnings)
            self._save_table_export("fundamental_snapshots", fundamentals)
            self._save_table_export("earnings_events", earnings)

        if self.config.include_daily_bars:
            self.download_daily_bars_yfinance(broad_symbols)

        if self.config.include_intraday_bars:
            self.download_intraday_bars_alpaca(
                core_symbols,
                timeframe="15Min",
                start_date=self.config.intraday_15m_start,
            )
            self.download_intraday_bars_alpaca(
                core_symbols,
                timeframe="1Min",
                start_date=self.config.intraday_1m_start,
            )

        if self.config.include_corporate_actions:
            actions = self.download_corporate_actions_yfinance(broad_symbols)
            self._upsert_records(CorporateAction, actions)
            self._save_table_export("corporate_actions", actions)

        if self.config.include_sec:
            filings = self.download_sec_filings(broad_symbols)
            self._upsert_records(SECFiling, filings)
            self._save_table_export("sec_filings", filings)

        if self.config.include_macro:
            macro = self.download_macro_series()
            self._upsert_records(MacroObservation, macro)
            self._save_table_export("macro_observations", macro)

        if self.config.include_news:
            news_records = self.download_alpaca_news(core_symbols)
            self._upsert_records(NewsArticle, news_records)
            self._save_table_export("news_articles", news_records)

        self.manifest["generated_at"] = _now_utc().isoformat()
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        return self.manifest

    def _load_core_symbols(self) -> list[str]:
        data = yaml.safe_load(self.config.core_universe_path.read_text(encoding="utf-8"))
        symbols = [_safe_symbol(s) for s in data.get("symbols", [])]
        return [s for s in symbols if s]

    def _fetch_sp100_symbols(self) -> list[str]:
        url = "https://en.wikipedia.org/wiki/S%26P_100"
        try:
            html = self._get_text(url, provider="yfinance")
            tables = pd.read_html(io.StringIO(html))
            for table in tables:
                if "Symbol" in table.columns:
                    symbols = [_safe_symbol(v) for v in table["Symbol"].astype(str).tolist()]
                    symbols = [s for s in symbols if s]
                    if symbols:
                        return symbols
        except Exception as exc:
            logger.warning(f"Could not fetch S&P 100 universe: {exc}")
        return []

    def _fetch_sp500_symbols(self) -> list[str]:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            html = self._get_text(url, provider="yfinance")
            tables = pd.read_html(io.StringIO(html))
            for table in tables:
                if "Symbol" in table.columns:
                    symbols = [_safe_symbol(v) for v in table["Symbol"].astype(str).tolist()]
                    symbols = [s for s in symbols if s]
                    if symbols:
                        return symbols
        except Exception as exc:
            logger.warning(f"Could not fetch S&P 500 universe: {exc}")
        return []

    def _build_broad_universe(self, core_symbols: list[str]) -> list[str]:
        broad = []
        candidate_symbols = (
            core_symbols
            + self._fetch_sp500_symbols()
            + self._fetch_sp100_symbols()
            + ETF_BENCHMARKS
        )
        for symbol in candidate_symbols:
            normalized = _safe_symbol(symbol)
            if normalized and normalized not in broad:
                broad.append(normalized)
            if len(broad) >= max(self.config.broad_universe_size, len(core_symbols)):
                break
        return broad

    def _write_universe_files(self, core_symbols: list[str], broad_symbols: list[str]) -> None:
        payload = {
            "generated_at": _now_utc().isoformat(),
            "core_symbols": core_symbols,
            "broad_symbols": broad_symbols,
        }
        path = self.reference_dir / "institutional_universe.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._record_dataset(
            "universe",
            core_symbol_count=len(core_symbols),
            broad_symbol_count=len(broad_symbols),
            path=str(path),
        )

    def _load_sector_map(self) -> dict[str, str]:
        data = yaml.safe_load(self.config.core_universe_path.read_text(encoding="utf-8"))
        sector_map: dict[str, str] = {}
        for sector_name, symbols in (data.get("sector_mappings") or {}).items():
            for symbol in symbols or []:
                sector_map[_safe_symbol(symbol)] = str(sector_name).upper()
        return sector_map

    def _fetch_finnhub_profile(self, symbol: str) -> dict[str, Any]:
        symbol = _safe_symbol(symbol)
        if symbol in self.profile_cache:
            return self.profile_cache[symbol]
        if not self.finnhub_token:
            return {}
        try:
            data = self._get_json(
                "https://finnhub.io/api/v1/stock/profile2",
                params={"symbol": symbol, "token": self.finnhub_token},
                provider="finnhub",
            )
        except Exception as exc:
            logger.warning(f"Finnhub profile failed for {symbol}: {exc}")
            data = {}
        self.profile_cache[symbol] = data if isinstance(data, dict) else {}
        return self.profile_cache[symbol]

    def _fetch_finnhub_metric(self, symbol: str) -> dict[str, Any]:
        symbol = _safe_symbol(symbol)
        if symbol in self.metric_cache:
            return self.metric_cache[symbol]
        if not self.finnhub_token:
            return {}
        try:
            data = self._get_json(
                "https://finnhub.io/api/v1/stock/metric",
                params={"symbol": symbol, "metric": "all", "token": self.finnhub_token},
                provider="finnhub",
            )
        except Exception as exc:
            logger.warning(f"Finnhub metric failed for {symbol}: {exc}")
            data = {}
        metric = data.get("metric", {}) if isinstance(data, dict) else {}
        self.metric_cache[symbol] = metric if isinstance(metric, dict) else {}
        return self.metric_cache[symbol]

    def _fetch_alpha_vantage_earnings(self, symbol: str) -> dict[date, dict[str, Any]]:
        symbol = _safe_symbol(symbol)
        if symbol in self.alpha_vantage_earnings_cache:
            return self.alpha_vantage_earnings_cache[symbol]
        if not self.alpha_vantage_key:
            return {}
        try:
            payload = self._get_json(
                "https://www.alphavantage.co/query",
                params={
                    "function": "EARNINGS",
                    "symbol": symbol,
                    "apikey": self.alpha_vantage_key,
                },
                provider="alpha_vantage",
            )
        except Exception as exc:
            logger.warning(f"Alpha Vantage earnings failed for {symbol}: {exc}")
            payload = {}

        mapped: dict[date, dict[str, Any]] = {}
        quarterly = payload.get("quarterlyEarnings", []) if isinstance(payload, dict) else []
        if isinstance(quarterly, list):
            for item in quarterly:
                if not isinstance(item, dict):
                    continue
                fiscal_date = _parse_date(item.get("fiscalDateEnding"))
                if fiscal_date is None:
                    continue
                mapped[fiscal_date] = item
        self.alpha_vantage_earnings_cache[symbol] = mapped
        return mapped

    def download_security_master(self, symbols: list[str]) -> list[dict[str, Any]]:
        """Build a filtered security master using listing + profile data."""
        listing_text = self._get_text(
            "https://www.alphavantage.co/query",
            params={
                "function": "LISTING_STATUS",
                "state": "active",
                "apikey": self.alpha_vantage_key,
            },
            provider="alpha_vantage",
        )
        listing_df = pd.read_csv(io.StringIO(listing_text))
        listing_df["symbol"] = listing_df["symbol"].astype(str).map(_safe_symbol)
        listing_df = listing_df[listing_df["symbol"].isin(symbols)].copy()

        sec_map = self._get_json(
            "https://www.sec.gov/files/company_tickers.json",
            provider="sec",
        )
        if isinstance(sec_map, dict):
            for payload in sec_map.values():
                ticker = _safe_symbol(payload.get("ticker"))
                if ticker:
                    self.sec_ticker_map[ticker] = payload

        sector_map = self._load_sector_map()
        records: list[dict[str, Any]] = []
        now = _now_utc()
        for symbol in symbols:
            row = listing_df.loc[listing_df["symbol"] == symbol]
            row_data = row.iloc[0].to_dict() if not row.empty else {}
            profile = self._fetch_finnhub_profile(symbol)
            sec_payload = self.sec_ticker_map.get(symbol, {})

            records.append(
                {
                    "symbol": symbol,
                    "name": profile.get("name") or sec_payload.get("title") or row_data.get("name"),
                    "exchange": profile.get("exchange") or row_data.get("exchange"),
                    "asset_type": row_data.get("assetType") or "Stock",
                    "status": row_data.get("status") or "Active",
                    "ipo_date": _parse_date(profile.get("ipo") or row_data.get("ipoDate")),
                    "delisting_date": _parse_date(row_data.get("delistingDate")),
                    "currency": profile.get("currency"),
                    "country": profile.get("country"),
                    "sector": sector_map.get(symbol),
                    "industry": profile.get("finnhubIndustry"),
                    "market_cap": _parse_decimal(
                        profile.get("marketCapitalization"),
                        multiplier=Decimal("1000000"),
                    ),
                    "shares_outstanding": _parse_decimal(
                        profile.get("shareOutstanding"),
                        multiplier=Decimal("1000000"),
                    ),
                    "cik": str(sec_payload.get("cik_str", "")).zfill(10) if sec_payload else None,
                    "source": "mixed",
                    "security_metadata": {
                        "listing_status": {k: row_data.get(k) for k in row_data.keys()},
                        "finnhub_profile": profile,
                    },
                    "created_at": now,
                    "updated_at": now,
                }
            )

        self._record_dataset("security_master", rows=len(records))
        return records

    def download_finnhub_company_data(
        self,
        symbols: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Fetch current fundamental snapshots and earnings history from Finnhub."""
        fundamentals: list[dict[str, Any]] = []
        earnings_records: list[dict[str, Any]] = []
        as_of_date = date.today()
        now = _now_utc()

        for symbol in symbols:
            profile = self._fetch_finnhub_profile(symbol)
            metric = self._fetch_finnhub_metric(symbol)
            alpha_vantage_earnings = self._fetch_alpha_vantage_earnings(symbol)
            earnings: list[dict[str, Any]] = []
            if self.finnhub_token:
                try:
                    raw_earnings = self._get_json(
                        "https://finnhub.io/api/v1/stock/earnings",
                        params={"symbol": symbol, "token": self.finnhub_token},
                        provider="finnhub",
                    )
                    if isinstance(raw_earnings, list):
                        earnings = raw_earnings
                except Exception as exc:
                    logger.warning(f"Finnhub earnings failed for {symbol}: {exc}")

            shares_outstanding = _parse_decimal(
                metric.get("sharesOutstanding") or profile.get("shareOutstanding"),
                multiplier=Decimal("1000000"),
            )
            revenue_per_share = _parse_decimal(metric.get("revenuePerShareTTM"))
            revenue_ttm = (
                revenue_per_share * shares_outstanding
                if revenue_per_share is not None and shares_outstanding is not None
                else None
            )
            fundamentals.append(
                {
                    "symbol": symbol,
                    "as_of_date": as_of_date,
                    "source": "finnhub",
                    "market_cap": _parse_decimal(
                        metric.get("marketCapitalization") or profile.get("marketCapitalization"),
                        multiplier=Decimal("1000000"),
                    ),
                    "shares_outstanding": shares_outstanding,
                    "pe_ratio": metric.get("peTTM"),
                    "peg_ratio": metric.get("pegAnnual"),
                    "price_to_book": metric.get("pbAnnual"),
                    "eps": metric.get("epsInclExtraItemsTTM"),
                    "book_value": metric.get("bookValuePerShareQuarterly"),
                    "dividend_per_share": metric.get("dividendPerShareAnnual"),
                    "dividend_yield": metric.get("currentDividendYieldTTM"),
                    "revenue_ttm": revenue_ttm,
                    "gross_profit_ttm": None,
                    "operating_margin_ttm": metric.get("operatingMarginTTM"),
                    "profit_margin": metric.get("netMargin"),
                    "beta": metric.get("beta"),
                    "week_52_high": metric.get("52WeekHigh"),
                    "week_52_low": metric.get("52WeekLow"),
                    "analyst_target_price": metric.get("targetPrice"),
                    "fundamentals_metadata": {
                        "profile": profile,
                        "metric": metric,
                        "ingested_at_utc": now.isoformat(),
                    },
                    "created_at": now,
                    "updated_at": now,
                }
            )

            for event in earnings:
                fiscal_date = _parse_date(event.get("period"))
                if fiscal_date is None:
                    continue
                alpha_vantage_event = alpha_vantage_earnings.get(fiscal_date, {})
                enriched_event = {
                    **alpha_vantage_event,
                    **event,
                }
                availability = _resolve_earnings_availability(enriched_event, captured_at=now)
                earnings_records.append(
                    {
                        "symbol": symbol,
                        "fiscal_date_ending": fiscal_date,
                        "source": "finnhub",
                        "reported_date": availability["reported_date"],
                        "announcement_timestamp": availability["announcement_timestamp"],
                        "availability_timestamp": availability["availability_timestamp"],
                        "first_seen_at": availability["first_seen_at"],
                        "reported_eps": (
                            event.get("actual")
                            if event.get("actual") is not None
                            else alpha_vantage_event.get("reportedEPS")
                        ),
                        "estimated_eps": (
                            event.get("estimate")
                            if event.get("estimate") is not None
                            else alpha_vantage_event.get("estimatedEPS")
                        ),
                        "surprise": (
                            event.get("surprise")
                            if event.get("surprise") is not None
                            else alpha_vantage_event.get("surprise")
                        ),
                        "surprise_pct": (
                            event.get("surprisePercent")
                            if event.get("surprisePercent") is not None
                            else alpha_vantage_event.get("surprisePercentage")
                        ),
                        "earnings_metadata": {
                            **event,
                            "alpha_vantage": alpha_vantage_event or None,
                            "availability_source": availability["availability_source"],
                            "reported_date": (
                                availability["reported_date"].isoformat()
                                if availability["reported_date"] is not None
                                else None
                            ),
                            "announcement_timestamp_utc": (
                                availability["announcement_timestamp"].isoformat()
                                if availability["announcement_timestamp"] is not None
                                else None
                            ),
                            "availability_timestamp_utc": (
                                availability["availability_timestamp"].isoformat()
                                if availability["availability_timestamp"] is not None
                                else None
                            ),
                            "first_seen_at_utc": availability["first_seen_at"].isoformat(),
                            "ingested_at_utc": now.isoformat(),
                        },
                        "created_at": now,
                        "updated_at": now,
                    }
                )

        self._record_dataset(
            "finnhub_company_data",
            fundamental_rows=len(fundamentals),
            earnings_rows=len(earnings_records),
        )
        return fundamentals, earnings_records

    def download_daily_bars_yfinance(self, symbols: list[str]) -> None:
        """Download broad daily history with yfinance and persist to OHLCV storage."""
        start = pd.Timestamp(self.config.daily_start)
        end = pd.Timestamp(date.today() + timedelta(days=1))
        total_rows = 0
        source_dir = self.raw_dir / "yfinance" / "1Day"
        source_dir.mkdir(parents=True, exist_ok=True)

        for batch in _batched(symbols, 25):
            batch_map = {_yf_symbol(symbol): symbol for symbol in batch}
            downloaded = yf.download(
                tickers=list(batch_map.keys()),
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            if downloaded is None or downloaded.empty:
                continue

            for yf_symbol, original_symbol in batch_map.items():
                frame = self._extract_yfinance_symbol_frame(downloaded, yf_symbol)
                if frame.empty:
                    continue
                normalized = self._normalize_yfinance_daily_frame(original_symbol, frame)
                if normalized.empty:
                    continue
                total_rows += len(normalized)
                self._save_bar_frame(normalized, source_dir / f"{original_symbol}.parquet")
                self._upsert_ohlcv(normalized, symbol=original_symbol, timeframe="1Day")

        self._record_dataset(
            "daily_bars_yfinance",
            symbols=len(symbols),
            timeframe="1Day",
            start=str(self.config.daily_start),
            rows=total_rows,
        )

    def _extract_yfinance_symbol_frame(self, downloaded: pd.DataFrame, yf_symbol: str) -> pd.DataFrame:
        if isinstance(downloaded.columns, pd.MultiIndex):
            if yf_symbol not in downloaded.columns.get_level_values(0):
                return pd.DataFrame()
            frame = downloaded[yf_symbol].copy()
        else:
            frame = downloaded.copy()
        if frame.empty:
            return pd.DataFrame()
        return frame.reset_index()

    def _normalize_yfinance_daily_frame(self, symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
        timestamp_col = "Date" if "Date" in frame.columns else frame.columns[0]
        normalized = pd.DataFrame()
        normalized["timestamp"] = pd.to_datetime(frame[timestamp_col], errors="coerce").dt.strftime("%Y-%m-%d")
        normalized["timestamp"] = pd.to_datetime(
            normalized["timestamp"] + " 16:00:00",
            utc=False,
            errors="coerce",
        )
        normalized["timestamp"] = (
            normalized["timestamp"]
            .dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
            .dt.tz_convert("UTC")
        )
        normalized["open"] = pd.to_numeric(frame.get("Open"), errors="coerce")
        normalized["high"] = pd.to_numeric(frame.get("High"), errors="coerce")
        normalized["low"] = pd.to_numeric(frame.get("Low"), errors="coerce")
        normalized["close"] = pd.to_numeric(frame.get("Close"), errors="coerce")
        normalized["volume"] = pd.to_numeric(frame.get("Volume"), errors="coerce").fillna(0).astype("int64")
        normalized["symbol"] = symbol
        normalized = normalized.dropna(subset=["timestamp", "open", "high", "low", "close"])
        normalized = normalized.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        return normalized[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]

    def download_intraday_bars_alpaca(self, symbols: list[str], *, timeframe: str, start_date: date) -> None:
        """Download intraday bars from Alpaca with raw IEX feed."""
        end_dt = datetime.now(timezone.utc)
        start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
        output_dir = self.raw_dir / "alpaca" / timeframe
        output_dir.mkdir(parents=True, exist_ok=True)
        total_rows = 0

        async def _download_all() -> None:
            nonlocal total_rows
            client = AlpacaClient()
            for symbol in symbols:
                chunks: list[pd.DataFrame] = []
                page_token: str | None = None
                while True:
                    page = await client.get_bars_page(
                        symbol=symbol,
                        timeframe=timeframe,
                        start=start_dt,
                        end=end_dt,
                        limit=10_000,
                        adjustment="raw",
                        page_token=page_token,
                        feed=os.getenv("ALPACA_DATA_FEED", "iex"),
                    )
                    bars = page.get("bars", []) if isinstance(page, dict) else []
                    normalized = self._normalize_alpaca_bars(symbol, bars)
                    if normalized.empty:
                        break
                    chunks.append(normalized)
                    page_token = page.get("next_page_token") if isinstance(page, dict) else None
                    if not page_token:
                        break

                if not chunks:
                    continue
                frame = pd.concat(chunks, ignore_index=True)
                frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
                total_rows += len(frame)
                self._save_bar_frame(frame, output_dir / f"{symbol}.parquet")
                self._upsert_ohlcv(frame, symbol=symbol, timeframe=timeframe)

        import asyncio

        asyncio.run(_download_all())
        self._record_dataset(
            f"intraday_bars_{timeframe}",
            symbols=len(symbols),
            timeframe=timeframe,
            start=str(start_date),
            rows=total_rows,
        )

    def _normalize_alpaca_bars(self, symbol: str, bars: list[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(bars or [])
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
        frame = frame.rename(
            columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "vw": "vwap",
                "n": "trade_count",
            }
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        for col in ("open", "high", "low", "close", "volume", "vwap", "trade_count"):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["symbol"] = symbol
        frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
        frame["volume"] = frame["volume"].astype("int64")
        ordered = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        for col in ("vwap", "trade_count"):
            if col in frame.columns:
                ordered.append(col)
        return frame[ordered]

    def download_corporate_actions_yfinance(self, symbols: list[str]) -> list[dict[str, Any]]:
        """Fetch dividend/split history with yfinance."""
        records: list[dict[str, Any]] = []
        now = _now_utc()
        for symbol in symbols:
            try:
                actions = yf.Ticker(_yf_symbol(symbol)).actions
            except Exception as exc:
                logger.warning(f"yfinance actions failed for {symbol}: {exc}")
                continue
            if actions is None or actions.empty:
                continue
            actions = actions.reset_index()
            date_col = actions.columns[0]
            for _, row in actions.iterrows():
                ex_date = _parse_date(row[date_col])
                if ex_date is None:
                    continue
                dividend = row.get("Dividends")
                split = row.get("Stock Splits")
                if pd.notna(dividend) and float(dividend) != 0.0:
                    records.append(
                        {
                            "symbol": symbol,
                            "action_type": "DIVIDEND",
                            "ex_date": ex_date,
                            "record_date": None,
                            "payment_date": None,
                            "declaration_date": None,
                            "amount": _parse_decimal(dividend),
                            "split_from": None,
                            "split_to": None,
                            "currency": "USD",
                            "source": "yfinance",
                            "action_metadata": {"dividend": float(dividend)},
                            "created_at": now,
                            "updated_at": now,
                        }
                    )
                if pd.notna(split) and float(split) != 0.0:
                    records.append(
                        {
                            "symbol": symbol,
                            "action_type": "SPLIT",
                            "ex_date": ex_date,
                            "record_date": None,
                            "payment_date": None,
                            "declaration_date": None,
                            "amount": None,
                            "split_from": Decimal("1"),
                            "split_to": Decimal(str(split)),
                            "currency": None,
                            "source": "yfinance",
                            "action_metadata": {"split_ratio": float(split)},
                            "created_at": now,
                            "updated_at": now,
                        }
                    )
        self._record_dataset("corporate_actions", rows=len(records))
        return records

    def download_sec_filings(self, symbols: list[str]) -> list[dict[str, Any]]:
        """Download recent SEC submission metadata for the broad universe."""
        records: list[dict[str, Any]] = []
        now = _now_utc()
        for symbol in symbols:
            sec_payload = self.sec_ticker_map.get(symbol)
            if not sec_payload:
                continue
            cik = str(sec_payload.get("cik_str", "")).zfill(10)
            try:
                data = self._get_json(
                    f"https://data.sec.gov/submissions/CIK{cik}.json",
                    provider="sec",
                )
            except Exception as exc:
                logger.warning(f"SEC submissions failed for {symbol}: {exc}")
                continue

            recent = ((data or {}).get("filings") or {}).get("recent") or {}
            accessions = recent.get("accessionNumber") or []
            for idx, accession in enumerate(accessions):
                accession = str(accession)
                if not accession:
                    continue
                filing_date = _parse_date(
                    (recent.get("filingDate") or [None])[idx]
                    if idx < len(recent.get("filingDate") or [])
                    else None
                )
                accepted = _parse_datetime_utc(
                    (recent.get("acceptanceDateTime") or [None])[idx]
                    if idx < len(recent.get("acceptanceDateTime") or [])
                    else None
                )
                report_date = _parse_date(
                    (recent.get("reportDate") or [None])[idx]
                    if idx < len(recent.get("reportDate") or [])
                    else None
                )
                primary_document = (
                    (recent.get("primaryDocument") or [None])[idx]
                    if idx < len(recent.get("primaryDocument") or [])
                    else None
                )
                accession_clean = accession.replace("-", "")
                filing_url = None
                if primary_document:
                    filing_url = (
                        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_clean}/{primary_document}"
                    )
                records.append(
                    {
                        "accession_number": accession,
                        "symbol": symbol,
                        "cik": cik,
                        "form": (recent.get("form") or [None])[idx]
                        if idx < len(recent.get("form") or [])
                        else None,
                        "filed_date": filing_date,
                        "accepted_at": accepted,
                        "report_date": report_date,
                        "filing_url": filing_url,
                        "report_url": filing_url,
                        "filing_metadata": {
                            "primary_document": primary_document,
                            "primary_doc_description": (recent.get("primaryDocDescription") or [None])[idx]
                            if idx < len(recent.get("primaryDocDescription") or [])
                            else None,
                            "is_xbrl": (recent.get("isXBRL") or [None])[idx]
                            if idx < len(recent.get("isXBRL") or [])
                            else None,
                        },
                        "created_at": now,
                        "updated_at": now,
                    }
                )
        self._record_dataset("sec_filings", rows=len(records))
        return records

    def download_macro_series(self) -> list[dict[str, Any]]:
        """Download macro and volatility series from FRED and CBOE."""
        records: list[dict[str, Any]] = []
        now = _now_utc()

        for series_id, meta in FRED_SERIES.items():
            csv_text = self._get_text(
                f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
                provider="fred",
            )
            frame = pd.read_csv(io.StringIO(csv_text))
            value_col = frame.columns[-1]
            for _, row in frame.iterrows():
                obs_date = _parse_date(row.get("observation_date"))
                value = row.get(value_col)
                if obs_date is None or value in (".", None, ""):
                    continue
                records.append(
                    {
                        "series_id": series_id,
                        "observation_date": obs_date,
                        "source": "fred",
                        "name": meta["name"],
                        "interval": meta["interval"],
                        "unit": meta["unit"],
                        "value": _parse_decimal(value),
                        "macro_metadata": {"raw_series": series_id},
                        "created_at": now,
                        "updated_at": now,
                    }
                )

        for series_id, url in CBOE_SERIES.items():
            csv_text = self._get_text(url, provider="cboe")
            frame = pd.read_csv(io.StringIO(csv_text))
            date_col = frame.columns[0]
            value_col = next(
                (col for col in frame.columns if col.lower().strip() == "close"),
                frame.columns[-1],
            )
            for _, row in frame.iterrows():
                obs_date = _parse_date(row.get(date_col))
                value = row.get(value_col)
                if obs_date is None or value in (None, "", "."):
                    continue
                records.append(
                    {
                        "series_id": series_id,
                        "observation_date": obs_date,
                        "source": "cboe",
                        "name": series_id,
                        "interval": "daily",
                        "unit": "index",
                        "value": _parse_decimal(value),
                        "macro_metadata": {"url": url},
                        "created_at": now,
                        "updated_at": now,
                    }
                )

        self._record_dataset("macro_observations", rows=len(records))
        return records

    def download_alpaca_news(self, symbols: list[str]) -> list[dict[str, Any]]:
        """Download news for the core trading universe."""
        records: list[dict[str, Any]] = []
        now = _now_utc()
        seen: set[str] = set()
        start_iso = datetime.combine(
            self.config.news_start,
            datetime.min.time(),
            tzinfo=timezone.utc,
        ).isoformat().replace("+00:00", "Z")
        end_iso = _now_utc().isoformat().replace("+00:00", "Z")

        for batch in _batched(symbols, 8):
            page_token: str | None = None
            while True:
                params: dict[str, Any] = {
                    "symbols": ",".join(batch),
                    "start": start_iso,
                    "end": end_iso,
                    "limit": 50,
                    "sort": "asc",
                }
                if page_token:
                    params["page_token"] = page_token
                try:
                    data = self._get_json(
                        "https://data.alpaca.markets/v1beta1/news",
                        params=params,
                        headers=self.alpaca_headers,
                        provider="alpaca",
                    )
                except Exception as exc:
                    logger.warning(f"Alpaca news failed for {batch}: {exc}")
                    break

                articles = data.get("news", []) if isinstance(data, dict) else []
                if not articles:
                    break
                for article in articles:
                    article_id = str(article.get("id") or "")
                    if not article_id or article_id in seen:
                        continue
                    seen.add(article_id)
                    records.append(
                        {
                            "article_id": article_id,
                            "source": "alpaca",
                            "headline": article.get("headline"),
                            "author": article.get("author"),
                            "created_at_source": _parse_datetime_utc(article.get("created_at")),
                            "updated_at_source": _parse_datetime_utc(article.get("updated_at")),
                            "summary": article.get("summary"),
                            "url": article.get("url"),
                            "symbols": article.get("symbols"),
                            "sentiment": None,
                            "news_metadata": {
                                "source": article.get("source"),
                                "images": article.get("images"),
                            },
                            "created_at": now,
                            "updated_at": now,
                        }
                    )
                page_token = data.get("next_page_token") if isinstance(data, dict) else None
                if not page_token:
                    break

        self._record_dataset("news_articles", rows=len(records))
        return records

    def _save_bar_frame(self, frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    def _save_table_export(self, name: str, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        path = self.reference_dir / f"{name}.parquet"
        frame = pd.DataFrame(_clean_record_for_export(record) for record in records)
        for column in (
            "security_metadata",
            "action_metadata",
            "fundamentals_metadata",
            "earnings_metadata",
            "filing_metadata",
            "macro_metadata",
            "news_metadata",
            "symbols",
        ):
            if column in frame.columns:
                frame[column] = frame[column].map(
                    lambda v: json.dumps(v, ensure_ascii=True) if isinstance(v, (dict, list)) else v
                )
        frame.to_parquet(path, index=False)
        self.manifest["datasets"].setdefault(name, {})
        self.manifest["datasets"][name]["path"] = str(path)

    def _upsert_ohlcv(self, frame: pd.DataFrame, *, symbol: str, timeframe: str) -> None:
        if self.db_loader is None or frame.empty:
            return
        self.db_loader.upsert_dataframe(
            df=frame.copy(),
            symbol=symbol,
            timeframe=timeframe,
            batch_size=10000,
            source_timezone="UTC",
        )

    def _upsert_records(self, model: type[Any], records: list[dict[str, Any]]) -> None:
        if self.db_manager is None or not records:
            return
        table = model.__table__
        conflict_columns = {
            "corporate_actions": ["symbol", "action_type", "ex_date", "source"],
        }.get(table.name, [col.name for col in table.primary_key.columns])
        update_columns = [
            col.name
            for col in table.columns
            if col.name not in conflict_columns and not col.primary_key and col.name != "created_at"
        ]
        cleaned_records = [_clean_record_for_db(record) for record in records]
        with self.db_manager.engine.begin() as conn:
            for chunk in _batched_records(cleaned_records, 500):
                stmt = insert(table).values(chunk)
                set_map = {col: getattr(stmt.excluded, col) for col in update_columns}
                if table.name == "earnings_events":
                    set_map["first_seen_at"] = func.coalesce(
                        table.c.first_seen_at,
                        stmt.excluded.first_seen_at,
                    )
                    for column_name in ("announcement_timestamp", "availability_timestamp"):
                        existing_value = getattr(table.c, column_name)
                        incoming_value = getattr(stmt.excluded, column_name)
                        set_map[column_name] = case(
                            (existing_value.is_(None), incoming_value),
                            (incoming_value.is_(None), existing_value),
                            else_=func.least(existing_value, incoming_value),
                        )
                upsert_stmt = stmt.on_conflict_do_update(
                    index_elements=conflict_columns,
                    set_=set_map,
                )
                conn.execute(upsert_stmt)
