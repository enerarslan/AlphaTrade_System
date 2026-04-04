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


def test_download_alpha_vantage_news_uses_symbol_specific_sentiment(tmp_path) -> None:
    config = InstitutionalBootstrapConfig(
        output_root=tmp_path,
        sync_db=False,
        news_start=datetime(2026, 3, 1, tzinfo=timezone.utc).date(),
        alpha_vantage_news_window_days=30,
    )
    bootstrapper = InstitutionalDataBootstrapper(config)
    bootstrapper.alpha_vantage_key = "test-key"

    bootstrapper._get_json = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "feed": [
            {
                "title": "Apple supplier update",
                "summary": "Supplier checks improved.",
                "url": "https://example.com/apple-supplier",
                "time_published": "20260314T120000",
                "source": "Reuters",
                "source_domain": "reuters.com",
                "overall_sentiment_score": "0.10",
                "overall_sentiment_label": "Neutral",
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "ticker_sentiment_score": "0.62",
                        "ticker_sentiment_label": "Bullish",
                        "relevance_score": "0.87",
                    },
                    {
                        "ticker": "MSFT",
                        "ticker_sentiment_score": "-0.31",
                        "ticker_sentiment_label": "Bearish",
                        "relevance_score": "0.21",
                    },
                ],
            }
        ]
    }

    records = bootstrapper.download_alpha_vantage_news(["AAPL"])

    assert len(records) == 1
    assert records[0]["article_id"].startswith("av::AAPL::")
    assert records[0]["symbols"] == ["AAPL"]
    assert records[0]["sentiment"] == 0.62
    assert records[0]["news_metadata"]["ticker_sentiment_label"] == "Bullish"
    assert records[0]["news_metadata"]["overall_sentiment_score"] == 0.10


def test_bootstrap_run_combines_alpaca_and_alpha_vantage_news(monkeypatch, tmp_path) -> None:
    config = InstitutionalBootstrapConfig(
        output_root=tmp_path,
        sync_db=False,
        include_news=True,
        include_sec=False,
        include_macro=False,
        include_corporate_actions=False,
        include_fundamentals=False,
        include_daily_bars=False,
        include_intraday_bars=False,
    )
    bootstrapper = InstitutionalDataBootstrapper(config)

    captured_upserts: list[tuple[object, list[dict[str, object]]]] = []

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
    monkeypatch.setattr(
        bootstrapper,
        "download_alpaca_news",
        lambda core_symbols: [{"article_id": "alp-1", "source": "alpaca"}],
    )
    monkeypatch.setattr(
        bootstrapper,
        "download_alpha_vantage_news",
        lambda core_symbols: [{"article_id": "av::AAPL::1", "source": "alpha_vantage"}],
    )
    monkeypatch.setattr(
        bootstrapper,
        "_upsert_records",
        lambda model, records: captured_upserts.append((model, list(records))),
    )
    monkeypatch.setattr(bootstrapper, "_save_table_export", lambda *_args, **_kwargs: None)

    manifest = bootstrapper.run()

    assert any(model.__name__ == "NewsArticle" and len(records) == 2 for model, records in captured_upserts)
    assert manifest["datasets"]["news_articles"]["providers"] == {
        "alpaca": 1,
        "alpha_vantage": 1,
    }


def test_get_json_rotates_alpha_vantage_keys_when_one_is_limited(tmp_path, monkeypatch) -> None:
    config = InstitutionalBootstrapConfig(output_root=tmp_path, sync_db=False)
    bootstrapper = InstitutionalDataBootstrapper(config)
    bootstrapper.alpha_vantage_keys = ["key_a", "key_b"]
    bootstrapper.alpha_vantage_key = "key_a"

    class _Response:
        def __init__(self, payload: dict[str, object]) -> None:
            self.status_code = 200
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    seen_keys: list[str] = []

    def _fake_get(_url, *, params=None, headers=None, timeout=60):
        seen_keys.append(params["apikey"])
        if params["apikey"] == "key_a":
            return _Response({"Information": "standard API rate limit is 25 requests per day"})
        return _Response({"feed": []})

    monkeypatch.setattr(bootstrapper.provider_gate, "wait", lambda provider: None)
    monkeypatch.setattr(bootstrapper.session, "get", _fake_get)
    monkeypatch.setattr("quant_trading_system.data.institutional_bootstrap.time.sleep", lambda _x: None)

    payload = bootstrapper._get_json(
        "https://www.alphavantage.co/query",
        params={"function": "NEWS_SENTIMENT"},
        provider="alpha_vantage",
    )

    assert payload == {"feed": []}
    assert seen_keys[:2] == ["key_a", "key_b"]
