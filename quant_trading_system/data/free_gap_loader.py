"""
Free data gap bootstrap for institutional-style research extensions.

Captures the highest-value free layers missing from the initial bootstrap:
- Historical SIP top-of-book quotes and trades from Alpaca
- FINRA Reg SHO short-sale volume
- SEC fails-to-deliver data
- ALFRED point-in-time macro vintages
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from sqlalchemy.dialects.postgresql import insert

from quant_trading_system.data.institutional_bootstrap import (
    FRED_SERIES,
    _batched_records,
    _clean_record_for_db,
    _clean_record_for_export,
    _now_utc,
    _parse_date,
    _parse_datetime_utc,
    _parse_decimal,
    _safe_symbol,
)
from quant_trading_system.database.connection import DatabaseManager
from quant_trading_system.database.models import (
    Base,
    FailsToDeliver,
    MacroVintageObservation,
    ShortSaleVolume,
    StockQuote,
    StockTrade,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=REPO_ROOT / ".env", override=True)

logger = logging.getLogger(__name__)

MICROSTRUCTURE_PRIORITY = [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "TSLA",
    "IWM",
    "TLT",
    "GLD",
    "JPM",
    "AMD",
    "GOOGL",
    "BAC",
    "V",
    "MA",
    "XOM",
    "CVX",
    "UNH",
]


@dataclass
class FreeGapBootstrapConfig:
    """Configuration for the free market-data gap loader."""

    output_root: Path = Path("data")
    sync_db: bool = True
    quote_trade_days: int = 1
    quote_trade_symbol_limit: int | None = 12
    short_sale_days: int = 30
    ftd_periods: int = 4
    alfred_years: int = 15
    universe_path: Path = Path("data/reference/institutional_universe.json")
    core_universe_path: Path = Path("quant_trading_system/config/symbols.yaml")
    include_quotes_trades: bool = True
    include_short_sale: bool = True
    include_ftd: bool = True
    include_alfred: bool = True
    fred_series_ids: tuple[str, ...] = field(default_factory=lambda: tuple(FRED_SERIES.keys()))


class FreeGapDataBootstrapper:
    """Downloads and persists the missing free data layers."""

    def __init__(self, config: FreeGapBootstrapConfig) -> None:
        self.config = config
        self.output_root = Path(config.output_root)
        self.raw_dir = self.output_root / "raw"
        self.reference_dir = self.output_root / "reference"
        self.export_dir = self.output_root / "export"
        for path in (self.raw_dir, self.reference_dir, self.export_dir):
            path.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.export_dir / "free_gap_manifest.json"
        self.manifest: dict[str, Any] = {
            "started_at": _now_utc().isoformat(),
            "datasets": {},
        }

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "AlphaTradeSystem research@alphatrade.local"})
        self.alpaca_headers = {
            "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
            "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET", ""),
        }
        self.fred_api_key = os.getenv("FRED_API_KEY", "")
        self._provider_last_call: dict[str, float] = {}
        self._provider_min_interval = {
            "alpaca": 0.08,
            "finra": 0.10,
            "sec": 0.20,
            "alfred": 0.25,
        }

        self.db_manager = DatabaseManager() if config.sync_db else None

    def run(self) -> dict[str, Any]:
        """Execute the gap bootstrap and persist a manifest."""
        if self.config.sync_db and self.db_manager is not None:
            with self.db_manager.engine.begin() as conn:
                Base.metadata.create_all(bind=conn)

        core_symbols, broad_symbols = self._load_universe()
        broad_symbol_set = set(broad_symbols)

        if self.config.include_quotes_trades:
            selected_symbols = self._select_microstructure_symbols(core_symbols)
            self.download_stock_microstructure(selected_symbols)

        if self.config.include_short_sale:
            short_sale_records = self.download_short_sale_volume(broad_symbol_set)
            self._upsert_records(ShortSaleVolume, short_sale_records)
            self._save_table_export("short_sale_volumes", short_sale_records)

        if self.config.include_ftd:
            ftd_records = self.download_fails_to_deliver(broad_symbol_set)
            self._upsert_records(FailsToDeliver, ftd_records)
            self._save_table_export("fails_to_deliver", ftd_records)

        if self.config.include_alfred:
            vintage_records = self.download_alfred_vintages()
            self._upsert_records(MacroVintageObservation, vintage_records)
            self._save_table_export("macro_vintage_observations", vintage_records)

        self.manifest["generated_at"] = _now_utc().isoformat()
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        return self.manifest

    def _load_universe(self) -> tuple[list[str], list[str]]:
        if self.config.universe_path.exists():
            payload = json.loads(self.config.universe_path.read_text(encoding="utf-8"))
            core = [_safe_symbol(symbol) for symbol in payload.get("core_symbols", [])]
            broad = [_safe_symbol(symbol) for symbol in payload.get("broad_symbols", [])]
            core = [symbol for symbol in core if symbol]
            broad = [symbol for symbol in broad if symbol]
            if core and broad:
                return core, broad

        data = yaml.safe_load(self.config.core_universe_path.read_text(encoding="utf-8")) or {}
        core = [_safe_symbol(symbol) for symbol in data.get("symbols", [])]
        core = [symbol for symbol in core if symbol]
        return core, core

    def _select_microstructure_symbols(self, core_symbols: list[str]) -> list[str]:
        ordered: list[str] = []
        for symbol in MICROSTRUCTURE_PRIORITY + core_symbols:
            normalized = _safe_symbol(symbol)
            if normalized and normalized in core_symbols and normalized not in ordered:
                ordered.append(normalized)

        limit = self.config.quote_trade_symbol_limit
        if limit is not None and limit > 0:
            return ordered[:limit]
        return ordered

    def _pace(self, provider: str) -> None:
        min_interval = self._provider_min_interval.get(provider, 0.0)
        if min_interval <= 0:
            return
        now = time.monotonic()
        last = self._provider_last_call.get(provider)
        if last is not None:
            wait_seconds = min_interval - (now - last)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
        self._provider_last_call[provider] = time.monotonic()

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        provider: str,
        timeout: int = 120,
    ) -> requests.Response:
        merged_headers = dict(self.session.headers)
        if headers:
            merged_headers.update(headers)

        last_error: Exception | None = None
        for attempt in range(4):
            try:
                self._pace(provider)
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=merged_headers,
                    timeout=timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"HTTP {response.status_code} from {provider}: {response.text[:200]}",
                        response=response,
                    )
                response.raise_for_status()
                return response
            except Exception as exc:
                last_error = exc
                if attempt >= 3:
                    break
                time.sleep(min(8.0, 1.5 * (attempt + 1)))
        raise RuntimeError(f"{provider} request failed after retries: {url}") from last_error

    def _request_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        provider: str,
    ) -> dict[str, Any]:
        response = self._request("GET", url, params=params, headers=headers, provider=provider)
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def _request_alfred_json(self, params: dict[str, Any]) -> dict[str, Any]:
        merged_headers = dict(self.session.headers)
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                self._pace("alfred")
                response = self.session.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params=params,
                    headers=merged_headers,
                    timeout=120,
                )
                if response.status_code == 400 and "No vintage dates exist" in response.text:
                    return {}
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"HTTP {response.status_code} from alfred: {response.text[:200]}",
                        response=response,
                    )
                response.raise_for_status()
                payload = response.json()
                return payload if isinstance(payload, dict) else {}
            except Exception as exc:
                last_error = exc
                if attempt >= 3:
                    break
                time.sleep(min(8.0, 1.5 * (attempt + 1)))
        raise RuntimeError(
            "alfred request failed after retries: https://api.stlouisfed.org/fred/series/observations"
        ) from last_error

    def _request_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        provider: str,
    ) -> str:
        response = self._request("GET", url, params=params, headers=headers, provider=provider)
        return response.text

    def _request_bytes(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        provider: str,
    ) -> bytes:
        response = self._request("GET", url, params=params, headers=headers, provider=provider)
        return response.content

    def _record_dataset(self, name: str, **metadata: Any) -> None:
        self.manifest["datasets"][name] = {
            **metadata,
            "captured_at": _now_utc().isoformat(),
        }

    def _save_records_parquet(self, records: list[dict[str, Any]], path: Path) -> None:
        if not records:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(_clean_record_for_export(record) for record in records)
        for column in frame.columns:
            sample = frame[column].dropna().head(5)
            if not sample.empty and any(isinstance(value, (dict, list)) for value in sample):
                frame[column] = frame[column].map(
                    lambda value: json.dumps(value, ensure_ascii=True)
                    if isinstance(value, (dict, list))
                    else value
                )
        frame.to_parquet(path, index=False)

    def _save_table_export(self, name: str, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        path = self.reference_dir / f"{name}.parquet"
        self._save_records_parquet(records, path)
        self.manifest["datasets"].setdefault(name, {})
        self.manifest["datasets"][name]["path"] = str(path)
        self.manifest["datasets"][name]["rows"] = len(records)

    def _upsert_records(self, model: type[Any], records: list[dict[str, Any]]) -> None:
        if self.db_manager is None or not records:
            return

        table = model.__table__
        conflict_columns = {
            "stock_quotes": [
                "symbol",
                "timestamp",
                "bid_price",
                "bid_size",
                "bid_exchange",
                "ask_price",
                "ask_size",
                "ask_exchange",
                "source",
            ],
            "stock_trades": ["symbol", "timestamp", "trade_id", "exchange", "source"],
            "short_sale_volumes": ["symbol", "trade_date", "market", "source"],
            "fails_to_deliver": ["settlement_date", "cusip", "symbol", "source"],
            "macro_vintage_observations": [
                "series_id",
                "observation_date",
                "realtime_start",
                "realtime_end",
                "source",
            ],
        }.get(table.name, [column.name for column in table.primary_key.columns])

        cleaned_records = [_clean_record_for_db(record) for record in records]
        with self.db_manager.engine.begin() as conn:
            for chunk in _batched_records(cleaned_records, 1000):
                stmt = insert(table).values(chunk)
                upsert_stmt = stmt.on_conflict_do_nothing(index_elements=conflict_columns)
                conn.execute(upsert_stmt)

    def _recent_trading_dates(self, count: int) -> list[date]:
        dates: list[date] = []
        cursor = date.today() - timedelta(days=1)
        while len(dates) < max(count, 0):
            if cursor.weekday() < 5:
                dates.append(cursor)
            cursor -= timedelta(days=1)
        dates.reverse()
        return dates

    def _build_realtime_windows(
        self,
        start_date: date,
        end_date: date,
        *,
        years_per_window: int = 5,
    ) -> list[tuple[date, date]]:
        windows: list[tuple[date, date]] = []
        cursor = start_date
        while cursor <= end_date:
            window_end = min(cursor + timedelta(days=365 * years_per_window - 1), end_date)
            windows.append((cursor, window_end))
            cursor = window_end + timedelta(days=1)
        return windows

    def _normalize_condition_codes(self, value: Any) -> list[str] | None:
        if value in (None, "", []):
            return None
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return [str(value)]

    def _normalize_quote_records(self, symbol: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = _now_utc()
        records: list[dict[str, Any]] = []
        for row in rows:
            timestamp = _parse_datetime_utc(row.get("t"))
            if timestamp is None:
                continue
            records.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "source": "alpaca_sip",
                    "bid_price": _parse_decimal(row.get("bp")),
                    "bid_size": int(row["bs"]) if row.get("bs") not in (None, "") else None,
                    "bid_exchange": str(row["bx"]) if row.get("bx") not in (None, "") else None,
                    "ask_price": _parse_decimal(row.get("ap")),
                    "ask_size": int(row["as"]) if row.get("as") not in (None, "") else None,
                    "ask_exchange": str(row["ax"]) if row.get("ax") not in (None, "") else None,
                    "tape": str(row["z"]) if row.get("z") not in (None, "") else None,
                    "condition_codes": self._normalize_condition_codes(row.get("c")),
                    "quote_metadata": {
                        key: value
                        for key, value in row.items()
                        if key not in {"t", "bp", "bs", "bx", "ap", "as", "ax", "z", "c"}
                    }
                    or None,
                    "created_at": now,
                    "updated_at": now,
                }
            )
        return records

    def _normalize_trade_records(self, symbol: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = _now_utc()
        records: list[dict[str, Any]] = []
        for row in rows:
            timestamp = _parse_datetime_utc(row.get("t"))
            if timestamp is None:
                continue
            records.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "source": "alpaca_sip",
                    "trade_id": str(row["i"]) if row.get("i") not in (None, "") else None,
                    "price": _parse_decimal(row.get("p")),
                    "size": int(row["s"]) if row.get("s") not in (None, "") else None,
                    "exchange": str(row["x"]) if row.get("x") not in (None, "") else None,
                    "tape": str(row["z"]) if row.get("z") not in (None, "") else None,
                    "condition_codes": self._normalize_condition_codes(row.get("c")),
                    "trade_metadata": {
                        key: value
                        for key, value in row.items()
                        if key not in {"t", "i", "p", "s", "x", "z", "c"}
                    }
                    or None,
                    "created_at": now,
                    "updated_at": now,
                }
            )
        return records

    def _download_alpaca_endpoint(
        self,
        symbol: str,
        trading_day: date,
        *,
        endpoint: str,
        model: type[Any],
    ) -> tuple[int, int]:
        eastern = ZoneInfo("America/New_York")
        start_dt = datetime.combine(trading_day, dt_time(9, 30), tzinfo=eastern).astimezone(timezone.utc)
        end_dt = datetime.combine(trading_day, dt_time(16, 0), tzinfo=eastern).astimezone(timezone.utc)
        url = f"https://data.alpaca.markets/v2/stocks/{symbol}/{endpoint}"
        page_token: str | None = None
        page_count = 0
        row_count = 0
        output_dir = self.raw_dir / "alpaca_sip" / endpoint / symbol

        while True:
            params: dict[str, Any] = {
                "start": start_dt.isoformat().replace("+00:00", "Z"),
                "end": end_dt.isoformat().replace("+00:00", "Z"),
                "feed": "sip",
                "limit": 10_000,
                "sort": "asc",
            }
            if page_token:
                params["page_token"] = page_token

            payload = self._request_json(
                url,
                params=params,
                headers=self.alpaca_headers,
                provider="alpaca",
            )
            rows = payload.get(endpoint, []) if isinstance(payload, dict) else []
            if not rows:
                break

            if endpoint == "quotes":
                records = self._normalize_quote_records(symbol, rows)
            else:
                records = self._normalize_trade_records(symbol, rows)

            if records:
                page_count += 1
                row_count += len(records)
                self._upsert_records(model, records)
                part_path = output_dir / f"{trading_day.isoformat()}_part{page_count:05d}.parquet"
                self._save_records_parquet(records, part_path)

            page_token = payload.get("next_page_token") if isinstance(payload, dict) else None
            if not page_token:
                break

        return row_count, page_count

    def download_stock_microstructure(self, symbols: list[str]) -> None:
        """Download recent SIP quotes and trades for the selected core universe."""
        if not symbols:
            return
        if not all(self.alpaca_headers.values()):
            raise RuntimeError("Alpaca API credentials are required for quote/trade capture.")

        trading_dates = self._recent_trading_dates(self.config.quote_trade_days)
        quote_rows = 0
        quote_pages = 0
        trade_rows = 0
        trade_pages = 0

        logger.info(
            "Starting SIP quote/trade capture for %s symbols across %s trading day(s)",
            len(symbols),
            len(trading_dates),
        )

        for trading_day in trading_dates:
            for symbol in symbols:
                symbol_quote_rows, symbol_quote_pages = self._download_alpaca_endpoint(
                    symbol,
                    trading_day,
                    endpoint="quotes",
                    model=StockQuote,
                )
                quote_rows += symbol_quote_rows
                quote_pages += symbol_quote_pages

                symbol_trade_rows, symbol_trade_pages = self._download_alpaca_endpoint(
                    symbol,
                    trading_day,
                    endpoint="trades",
                    model=StockTrade,
                )
                trade_rows += symbol_trade_rows
                trade_pages += symbol_trade_pages

                logger.info(
                    "Captured %s quotes / %s trades for %s on %s",
                    symbol_quote_rows,
                    symbol_trade_rows,
                    symbol,
                    trading_day.isoformat(),
                )

        self._record_dataset(
            "stock_quotes",
            symbols=len(symbols),
            trading_days=[day.isoformat() for day in trading_dates],
            rows=quote_rows,
            pages=quote_pages,
            path=str(self.raw_dir / "alpaca_sip" / "quotes"),
        )
        self._record_dataset(
            "stock_trades",
            symbols=len(symbols),
            trading_days=[day.isoformat() for day in trading_dates],
            rows=trade_rows,
            pages=trade_pages,
            path=str(self.raw_dir / "alpaca_sip" / "trades"),
        )

    def _discover_finra_short_sale_links(self) -> list[tuple[str, date, str]]:
        page_url = (
            "https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/"
            "daily-short-sale-volume-files"
        )
        html = self._request_text(page_url, provider="finra")
        matches = sorted(
            set(
                re.findall(
                    r"(https://cdn\.finra\.org/equity/regsho/daily/([A-Z]+)shvol(\d{8})\.txt)",
                    html,
                )
            )
        )
        links: list[tuple[str, date, str]] = []
        for url, market, datestr in matches:
            trade_date = _parse_date(datestr)
            if trade_date is None:
                continue
            links.append((url, trade_date, market))
        return links

    def download_short_sale_volume(self, broad_symbol_set: set[str]) -> list[dict[str, Any]]:
        """Download recent FINRA Reg SHO short-sale volume files."""
        records: list[dict[str, Any]] = []
        now = _now_utc()
        links = self._discover_finra_short_sale_links()
        selected_dates = {
            item.isoformat()
            for item in sorted({trade_date for _, trade_date, _ in links})[-self.config.short_sale_days :]
        }

        chosen_links = [
            (url, trade_date, market)
            for url, trade_date, market in links
            if trade_date.isoformat() in selected_dates
        ]

        for url, trade_date, market in chosen_links:
            filename = Path(url).name
            text = self._request_text(url, provider="finra")
            raw_path = self.raw_dir / "finra_regsho" / filename
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(text, encoding="utf-8")

            frame = pd.read_csv(io.StringIO(text), sep="|", dtype=str)
            for _, row in frame.iterrows():
                symbol = _safe_symbol(row.get("Symbol"))
                if not symbol or symbol not in broad_symbol_set:
                    continue
                records.append(
                    {
                        "symbol": symbol,
                        "trade_date": _parse_date(row.get("Date")) or trade_date,
                        "market": str(row.get("Market") or market),
                        "source": "finra_regsho",
                        "short_volume": _parse_decimal(row.get("ShortVolume")),
                        "short_exempt_volume": _parse_decimal(row.get("ShortExemptVolume")),
                        "total_volume": _parse_decimal(row.get("TotalVolume")),
                        "volume_metadata": {"file": filename, "market_code": market},
                        "created_at": now,
                        "updated_at": now,
                    }
                )

            logger.info("Captured FINRA Reg SHO file %s", filename)

        self._record_dataset(
            "short_sale_volumes",
            files=len(chosen_links),
            rows=len(records),
            trading_dates=sorted(selected_dates),
            path=str(self.raw_dir / "finra_regsho"),
        )
        return records

    def _discover_ftd_links(self) -> list[str]:
        html = self._request_text(
            "https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data",
            provider="sec",
        )
        links = sorted(
            set(re.findall(r"/files/data/fails-deliver-data/cnsfails\d{6}[ab]\.zip", html))
        )
        return [f"https://www.sec.gov{link}" for link in links]

    def download_fails_to_deliver(self, broad_symbol_set: set[str]) -> list[dict[str, Any]]:
        """Download recent SEC fails-to-deliver periods for the broad universe."""
        records: list[dict[str, Any]] = []
        now = _now_utc()
        links = self._discover_ftd_links()
        selected_links = links[-self.config.ftd_periods :]

        for url in selected_links:
            filename = Path(url).name
            raw_bytes = self._request_bytes(url, provider="sec")
            raw_path = self.raw_dir / "sec_ftd" / filename
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(raw_bytes)

            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
                for member in archive.namelist():
                    content = archive.read(member)
                    try:
                        text = content.decode("utf-8")
                    except UnicodeDecodeError:
                        text = content.decode("latin-1")
                    frame = pd.read_csv(io.StringIO(text), sep="|", dtype=str)
                    for _, row in frame.iterrows():
                        symbol = _safe_symbol(row.get("SYMBOL"))
                        if not symbol or symbol not in broad_symbol_set:
                            continue
                        settlement_date = _parse_date(row.get("SETTLEMENT DATE"))
                        cusip = str(row.get("CUSIP") or "").strip()
                        if settlement_date is None or not cusip:
                            continue
                        quantity_raw = row.get("QUANTITY (FAILS)")
                        quantity = int(quantity_raw) if quantity_raw not in (None, "", "NaN") else None
                        records.append(
                            {
                                "settlement_date": settlement_date,
                                "cusip": cusip,
                                "symbol": symbol,
                                "source": "sec_ftd",
                                "quantity": quantity,
                                "description": row.get("DESCRIPTION"),
                                "price": _parse_decimal(row.get("PRICE")),
                                "ftd_metadata": {"zip_file": filename, "member": member},
                                "created_at": now,
                                "updated_at": now,
                            }
                        )

            logger.info("Captured SEC FTD period %s", filename)

        self._record_dataset(
            "fails_to_deliver",
            periods=len(selected_links),
            rows=len(records),
            path=str(self.raw_dir / "sec_ftd"),
        )
        return records

    def download_alfred_vintages(self) -> list[dict[str, Any]]:
        """Download revision-aware ALFRED vintages for core macro series."""
        if not self.fred_api_key:
            raise RuntimeError("FRED_API_KEY is required for ALFRED vintage capture.")

        now = _now_utc()
        observation_start = date.today() - timedelta(days=365 * self.config.alfred_years)
        realtime_end = max(observation_start, date.today() - timedelta(days=1))
        realtime_windows = self._build_realtime_windows(observation_start, realtime_end, years_per_window=5)
        records: list[dict[str, Any]] = []

        for series_id in self.config.fred_series_ids:
            meta = FRED_SERIES.get(series_id, {"name": series_id, "interval": None, "unit": None})
            total_rows = 0

            for realtime_start_window, realtime_end_window in realtime_windows:
                offset = 0
                while True:
                    payload = self._request_alfred_json(
                        {
                            "series_id": series_id,
                            "api_key": self.fred_api_key,
                            "file_type": "json",
                            "realtime_start": realtime_start_window.isoformat(),
                            "realtime_end": realtime_end_window.isoformat(),
                            "observation_start": observation_start.isoformat(),
                            "output_type": 4,
                            "sort_order": "asc",
                            "limit": 100000,
                            "offset": offset,
                        }
                    )
                    observations = payload.get("observations", []) if isinstance(payload, dict) else []
                    if not observations:
                        break

                    for observation in observations:
                        value = observation.get("value")
                        if value in (None, "", "."):
                            continue
                        observation_date = _parse_date(observation.get("date"))
                        realtime_start = _parse_date(observation.get("realtime_start"))
                        realtime_end = _parse_date(observation.get("realtime_end"))
                        if observation_date is None or realtime_start is None or realtime_end is None:
                            continue
                        records.append(
                            {
                                "series_id": series_id,
                                "observation_date": observation_date,
                                "realtime_start": realtime_start,
                                "realtime_end": realtime_end,
                                "source": "alfred",
                                "name": meta.get("name"),
                                "interval": meta.get("interval"),
                                "unit": meta.get("unit"),
                                "value": _parse_decimal(value),
                                "vintage_metadata": {
                                    "output_type": 4,
                                    "window_start": realtime_start_window.isoformat(),
                                    "window_end": realtime_end_window.isoformat(),
                                },
                                "created_at": now,
                                "updated_at": now,
                            }
                        )

                    total_rows += len(observations)
                    count = int(payload.get("count", total_rows))
                    offset += len(observations)
                    if offset >= count:
                        break

            logger.info("Captured %s ALFRED vintages for %s", total_rows, series_id)

        self._record_dataset(
            "macro_vintage_observations",
            rows=len(records),
            series=list(self.config.fred_series_ids),
            observation_start=observation_start.isoformat(),
        )
        return records
