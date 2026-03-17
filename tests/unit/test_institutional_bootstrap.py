from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from quant_trading_system.data.institutional_bootstrap import (
    InstitutionalBootstrapConfig,
    InstitutionalDataBootstrapper,
    _resolve_earnings_availability,
)


def test_resolve_earnings_availability_prefers_explicit_announcement_timestamp() -> None:
    captured_at = datetime(2026, 3, 14, 10, 30, tzinfo=timezone.utc)
    payload = _resolve_earnings_availability(
        {
            "reportDate": "2026-01-28",
            "reportDateTime": "2026-01-28T12:45:00Z",
        },
        captured_at=captured_at,
    )

    assert payload["reported_date"].isoformat() == "2026-01-28"
    assert payload["announcement_timestamp"].isoformat() == "2026-01-28T12:45:00+00:00"
    assert payload["availability_timestamp"].isoformat() == "2026-01-28T12:45:00+00:00"
    assert payload["first_seen_at"] == captured_at
    assert payload["availability_source"] == "announcement_timestamp"


def test_resolve_earnings_availability_falls_back_to_first_seen() -> None:
    captured_at = datetime(2026, 3, 14, 10, 30, tzinfo=timezone.utc)
    payload = _resolve_earnings_availability(
        {
            "period": "2025-12-31",
            "actual": 2.84,
        },
        captured_at=captured_at,
    )

    assert payload["reported_date"] is None
    assert payload["announcement_timestamp"] is None
    assert payload["availability_timestamp"] is None
    assert payload["first_seen_at"] == captured_at
    assert payload["availability_source"] == "first_seen_at"


def test_bootstrap_run_uses_alembic_migrations_when_sync_db(monkeypatch, tmp_path) -> None:
    import quant_trading_system.data.institutional_bootstrap as bootstrap_module

    fake_loader = MagicMock()
    fake_db_manager = MagicMock()
    fake_db_manager.engine.url.render_as_string.return_value = (
        "postgresql://user:pass@localhost:5432/test_db"
    )
    captured: dict[str, str] = {}

    monkeypatch.setattr(bootstrap_module, "get_db_loader", lambda: fake_loader)
    monkeypatch.setattr(bootstrap_module, "get_db_manager", lambda: fake_db_manager)
    monkeypatch.setattr(
        bootstrap_module,
        "run_database_migrations",
        lambda *, database_url: captured.setdefault("database_url", database_url),
    )

    config = InstitutionalBootstrapConfig(
        output_root=tmp_path,
        sync_db=True,
        include_news=False,
        include_sec=False,
        include_macro=False,
        include_corporate_actions=False,
        include_fundamentals=False,
        include_daily_bars=False,
        include_intraday_bars=False,
    )
    bootstrapper = InstitutionalDataBootstrapper(config)

    monkeypatch.setattr(bootstrapper, "_load_core_symbols", lambda: ["AAPL"])
    monkeypatch.setattr(
        bootstrapper, "_build_broad_universe", lambda core_symbols: list(core_symbols)
    )
    monkeypatch.setattr(bootstrapper, "_write_universe_files", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        bootstrapper,
        "download_security_master",
        lambda broad_symbols: [{"symbol": "AAPL", "name": "Apple"}],
    )
    monkeypatch.setattr(bootstrapper, "_upsert_records", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bootstrapper, "_save_table_export", lambda *_args, **_kwargs: None)

    manifest = bootstrapper.run()

    assert captured["database_url"] == "postgresql://user:pass@localhost:5432/test_db"
    assert isinstance(manifest, dict)
