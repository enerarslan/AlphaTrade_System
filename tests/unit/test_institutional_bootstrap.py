from __future__ import annotations

from datetime import datetime, timezone

from quant_trading_system.data.institutional_bootstrap import _resolve_earnings_availability


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
