"""Unit tests for scripts/data.py."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from datetime import datetime, timedelta, timezone

import pandas as pd

from scripts import data as data_script
from quant_trading_system.execution import alpaca_client as alpaca_client_module


class _DummyDataManager:
    instances: list["_DummyDataManager"] = []
    latest_timestamp: datetime | None = None

    def __init__(self, config):
        self.config = config
        self.download_calls: list[tuple[list[str], datetime, datetime]] = []
        self.sync_calls: list[tuple[list[str], int]] = []
        _DummyDataManager.instances.append(self)

    @staticmethod
    def _parse_datetime_utc(value, end_of_day: bool = False):
        if isinstance(value, datetime):
            parsed = value
        elif value:
            parsed = datetime.fromisoformat(str(value))
        else:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        if end_of_day and isinstance(value, str) and len(value.strip()) <= 10:
            parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
        return parsed

    @staticmethod
    def _timeframe_to_timedelta(_timeframe: str) -> timedelta:
        return timedelta(minutes=15)

    def get_latest_database_timestamp(self, _symbol: str) -> datetime | None:
        return self.latest_timestamp

    async def download_data(self, symbols, start_date, end_date):
        self.download_calls.append((list(symbols), start_date, end_date))
        rows = {
            "symbol": [symbols[0]],
            "timestamp": [start_date],
            "open": [100.0],
            "high": [101.0],
            "low": [99.5],
            "close": [100.5],
            "volume": [1000],
        }
        return {symbols[0]: pd.DataFrame(rows)}

    def save_data(self, _data, output_dir=None, format: str = "csv"):
        return [f"{output_dir or 'data/raw'}/dummy.{format}"]

    async def sync_to_database(self, data, batch_size: int = 5000):
        self.sync_calls.append((list(data.keys()), batch_size))
        return sum(len(df) for df in data.values())


def test_run_data_command_supports_load_alias(monkeypatch):
    observed = {}

    def _fake_download(args):
        observed["command"] = args.data_command
        return 0

    monkeypatch.setattr(data_script, "cmd_download", _fake_download)
    exit_code = data_script.run_data_command(Namespace(data_command="load"))

    assert exit_code == 0
    assert observed["command"] == "load"


def test_cmd_download_uses_start_end_alias(monkeypatch):
    _DummyDataManager.instances.clear()
    _DummyDataManager.latest_timestamp = None
    monkeypatch.setattr(data_script, "DataManager", _DummyDataManager)

    args = Namespace(
        source="alpaca",
        symbols=["AAPL"],
        start="2024-01-01",
        end="2024-01-10",
        start_date="",
        end_date="",
        timeframe="15Min",
        output_dir="data/raw",
        sync_db=False,
        incremental=False,
        batch_size=5000,
    )

    exit_code = data_script.cmd_download(args)
    mgr = _DummyDataManager.instances[-1]
    _, start_dt, end_dt = mgr.download_calls[0]

    assert exit_code == 0
    assert start_dt.isoformat().startswith("2024-01-01T00:00:00")
    assert end_dt.isoformat().startswith("2024-01-10T23:59:59")
    assert mgr.sync_calls == []


def test_cmd_download_incremental_sync_uses_latest_db_timestamp(monkeypatch):
    _DummyDataManager.instances.clear()
    _DummyDataManager.latest_timestamp = datetime(2025, 1, 3, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(data_script, "DataManager", _DummyDataManager)

    args = Namespace(
        source="alpaca",
        symbols=["MSFT"],
        start="2024-01-01",
        end="2025-01-05",
        start_date="",
        end_date="",
        timeframe="15Min",
        output_dir="data/raw",
        sync_db=True,
        incremental=True,
        batch_size=2048,
    )

    exit_code = data_script.cmd_download(args)
    mgr = _DummyDataManager.instances[-1]
    _, start_dt, _ = mgr.download_calls[0]

    assert exit_code == 0
    assert start_dt == _DummyDataManager.latest_timestamp + timedelta(minutes=15)
    assert mgr.sync_calls == [(["MSFT"], 2048)]


def test_download_data_uses_alpaca_next_page_token(monkeypatch):
    pages = [
        {
            "bars": [
                {
                    "t": "2026-01-02T14:00:00Z",
                    "o": 100.0,
                    "h": 101.0,
                    "l": 99.0,
                    "c": 100.5,
                    "v": 1000,
                }
            ],
            "next_page_token": "tok_1",
        },
        {
            "bars": [
                {
                    "t": "2026-01-02T14:15:00Z",
                    "o": 100.5,
                    "h": 101.5,
                    "l": 100.0,
                    "c": 101.0,
                    "v": 1200,
                }
            ],
            "next_page_token": None,
        },
    ]

    class _FakeAlpacaClient:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        async def get_bars_page(self, **kwargs):
            self.calls.append(kwargs)
            return pages[len(self.calls) - 1]

    fake_client = _FakeAlpacaClient()
    monkeypatch.setattr(alpaca_client_module, "AlpacaClient", lambda: fake_client)

    manager = data_script.DataManager(
        data_script.DataConfig(
            symbols=["AAPL"],
            timeframe="15Min",
        )
    )

    result = asyncio.run(
        manager.download_data(
            symbols=["AAPL"],
            start_date="2026-01-02",
            end_date="2026-01-03",
        )
    )

    assert "AAPL" in result
    assert len(result["AAPL"]) == 2
    assert fake_client.calls[0].get("page_token") is None
    assert fake_client.calls[1].get("page_token") == "tok_1"
