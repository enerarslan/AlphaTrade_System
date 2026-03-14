from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import pandas as pd

from quant_trading_system.features.reference import (
    ReferenceFeatureBuilder,
    _infer_ftd_publish_timestamp,
)


def _builder() -> ReferenceFeatureBuilder:
    return ReferenceFeatureBuilder(db_manager=SimpleNamespace(), logger_=logging.getLogger("test.reference"))


def _base_frame(timestamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL"] * len(timestamps),
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "__row_id": list(range(len(timestamps))),
        }
    )


def test_short_sale_features_activate_next_business_day(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder()
    base = _base_frame(
        [
            "2025-01-02T21:00:00Z",
            "2025-01-03T21:00:00Z",
        ]
    )
    base["close"] = [100.0, 100.0]

    monkeypatch.setattr(
        builder,
        "_read_rows",
        lambda statement: [
            ("AAPL", date(2025, 1, 2), 60.0, 0.0, 100.0),
        ],
    )

    result = builder._augment_short_sale_features(
        base.copy(),
        symbols=["AAPL"],
        min_trade_date=date(2025, 1, 2),
        max_trade_date=date(2025, 1, 3),
    )

    assert result.loc[0, "ref_short_volume_ratio"] == pytest.approx(0.0)
    assert result.loc[1, "ref_short_volume_ratio"] == pytest.approx(0.6)
    assert result.loc[0, "ref_short_days_since_update"] > 1000.0


def test_news_features_respect_event_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder()
    base = _base_frame(
        [
            "2025-01-02T14:00:00Z",
            "2025-01-02T16:00:00Z",
            "2025-01-03T12:00:00Z",
        ]
    )
    news = pd.DataFrame(
        {
            "article_id": ["n1", "n2"],
            "event_timestamp": pd.to_datetime(
                [
                    "2025-01-02T15:00:00Z",
                    "2025-01-03T10:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["AAPL", "AAPL"],
            "sentiment": [None, None],
        }
    )
    monkeypatch.setattr(builder, "_load_news_articles", lambda **kwargs: news)

    result = builder._augment_news_features(
        base.copy(),
        symbols=["AAPL"],
        min_timestamp=base["timestamp"].min(),
        max_timestamp=base["timestamp"].max(),
    )

    assert result["ref_news_count_6h"].tolist() == [0.0, 1.0, 1.0]
    assert result["ref_news_count_1d"].tolist() == [0.0, 1.0, 2.0]
    assert result.loc[1, "ref_news_days_since_last"] == pytest.approx(1.0 / 24.0)


def test_corporate_action_features_carry_last_values(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder()
    base = _base_frame(
        [
            "2025-01-09T21:00:00Z",
            "2025-01-10T21:00:00Z",
            "2025-02-18T21:00:00Z",
        ]
    )
    actions = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "action_type": ["DIVIDEND", "SPLIT"],
            "event_timestamp": pd.to_datetime(
                [
                    "2025-01-10T00:00:00Z",
                    "2025-02-15T00:00:00Z",
                ],
                utc=True,
            ),
            "amount": [0.24, 0.0],
            "split_ratio": [0.0, 4.0],
        }
    )
    monkeypatch.setattr(builder, "_load_corporate_actions", lambda **kwargs: actions)

    result = builder._augment_corporate_action_features(
        base.copy(),
        symbols=["AAPL"],
        max_trade_date=date(2025, 2, 18),
    )

    assert result.loc[0, "ref_last_dividend_amount"] == pytest.approx(0.0)
    assert result.loc[1, "ref_last_dividend_amount"] == pytest.approx(0.24)
    assert result.loc[2, "ref_last_split_ratio"] == pytest.approx(4.0)
    assert result.loc[2, "ref_dividend_count_365d"] == pytest.approx(1.0)


def test_fundamental_features_activate_next_business_day(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder()
    base = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "timestamp": pd.to_datetime(
                ["2025-01-02T21:00:00Z", "2025-01-03T21:00:00Z"],
                utc=True,
            ),
            "close": [100.0, 100.0],
            "__row_id": [0, 1],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "as_of_date": [date(2025, 1, 2)],
            "market_cap": [1000.0],
            "shares_outstanding": [10.0],
            "pe_ratio": [20.0],
            "price_to_book": [5.0],
            "dividend_per_share": [1.0],
            "dividend_yield": [0.02],
            "revenue_ttm": [2000.0],
            "operating_margin_ttm": [0.30],
            "profit_margin": [0.20],
            "beta": [1.1],
            "week_52_high": [120.0],
            "week_52_low": [80.0],
            "analyst_target_price": [110.0],
            "effective_timestamp": pd.to_datetime(["2025-01-03T00:00:00Z"], utc=True),
        }
    )
    monkeypatch.setattr(builder, "_load_fundamental_snapshots", lambda **kwargs: fundamentals)

    result = builder._augment_fundamental_features(
        base.copy(),
        symbols=["AAPL"],
        max_trade_date=date(2025, 1, 3),
    )

    assert result.loc[0, "ref_market_cap_log1p"] == pytest.approx(0.0)
    assert result.loc[1, "ref_market_cap_log1p"] == pytest.approx(np.log1p(1000.0))
    assert result.loc[1, "ref_analyst_target_upside"] == pytest.approx(0.10)


def test_load_fundamental_snapshots_uses_created_at_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder()
    monkeypatch.setattr(
        builder,
        "_read_rows",
        lambda statement: [
            (
                "AAPL",
                date(2024, 12, 31),
                datetime(2025, 1, 2, 14, 0, tzinfo=timezone.utc),
                1000.0,
                10.0,
                20.0,
                5.0,
                1.0,
                0.02,
                2000.0,
                0.30,
                0.20,
                1.1,
                120.0,
                80.0,
                110.0,
            ),
        ],
    )

    result = builder._load_fundamental_snapshots(
        symbols=["AAPL"],
        max_trade_date=date(2025, 1, 3),
    )

    assert result.loc[0, "effective_timestamp"] == pd.Timestamp("2025-01-03T00:00:00Z")


def test_earnings_features_activate_after_first_seen(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder()
    base = _base_frame(
        [
            "2025-01-02T21:00:00Z",
            "2025-01-03T21:00:00Z",
        ]
    )
    earnings = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "fiscal_date_ending": [date(2024, 12, 31)],
            "reported_date": [date(2025, 1, 2)],
            "reported_eps": [1.2],
            "estimated_eps": [1.0],
            "surprise": [0.2],
            "surprise_pct": [20.0],
            "event_timestamp": pd.to_datetime(["2025-01-03T00:00:00Z"], utc=True),
            "ref_earnings_surprise_abs": [20.0],
            "ref_earnings_positive_surprise": [1.0],
            "ref_earnings_surprise_pct_mean_4q": [20.0],
        }
    )
    monkeypatch.setattr(builder, "_load_earnings_events", lambda **kwargs: earnings)

    result = builder._augment_earnings_features(
        base.copy(),
        symbols=["AAPL"],
        min_timestamp=base["timestamp"].min(),
        max_timestamp=base["timestamp"].max(),
    )

    assert result.loc[0, "ref_earnings_count_30d"] == pytest.approx(0.0)
    assert result.loc[1, "ref_earnings_count_30d"] == pytest.approx(1.0)
    assert result.loc[1, "ref_last_earnings_surprise_pct"] == pytest.approx(20.0)
    assert result.loc[1, "ref_last_earnings_positive_surprise"] == pytest.approx(1.0)


def test_load_earnings_events_uses_created_at_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder()
    monkeypatch.setattr(
        builder,
        "_read_rows",
        lambda statement: [
            (
                "AAPL",
                date(2024, 12, 31),
                date(2024, 12, 31),
                None,
                None,
                datetime(2025, 1, 2, 14, 0, tzinfo=timezone.utc),
                datetime(2025, 1, 2, 14, 0, tzinfo=timezone.utc),
                1.2,
                1.0,
                0.2,
                20.0,
            ),
        ],
    )

    result = builder._load_earnings_events(
        symbols=["AAPL"],
        min_timestamp=pd.Timestamp("2025-01-01T00:00:00Z"),
        max_timestamp=pd.Timestamp("2025-01-03T23:59:59Z"),
    )

    assert result.loc[0, "event_timestamp"] == pd.Timestamp("2025-01-03T00:00:00Z")


def test_load_earnings_events_prefers_announcement_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder()
    monkeypatch.setattr(
        builder,
        "_read_rows",
        lambda statement: [
            (
                "AAPL",
                date(2024, 12, 31),
                date(2025, 1, 2),
                datetime(2025, 1, 2, 12, 30, tzinfo=timezone.utc),
                datetime(2025, 1, 2, 12, 30, tzinfo=timezone.utc),
                datetime(2025, 1, 2, 14, 0, tzinfo=timezone.utc),
                datetime(2025, 1, 2, 14, 0, tzinfo=timezone.utc),
                1.2,
                1.0,
                0.2,
                20.0,
            ),
        ],
    )

    result = builder._load_earnings_events(
        symbols=["AAPL"],
        min_timestamp=pd.Timestamp("2025-01-01T00:00:00Z"),
        max_timestamp=pd.Timestamp("2025-01-03T23:59:59Z"),
    )

    assert result.loc[0, "event_timestamp"] == pd.Timestamp("2025-01-02T12:30:00Z")


def test_ftd_publish_timestamp_is_conservative() -> None:
    ts = _infer_ftd_publish_timestamp(
        date(2025, 1, 14),
        {"zip_file": "cnsfails202501a.zip"},
    )
    assert ts == pd.Timestamp("2025-02-03T00:00:00Z")


def test_ftd_features_activate_after_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _builder()
    base = _base_frame(
        [
            "2025-01-31T21:00:00Z",
            "2025-02-03T21:00:00Z",
        ]
    )
    ftd = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "settlement_date": [date(2025, 1, 14)],
            "quantity": [1000.0],
            "price": [50.0],
            "ftd_metadata": [{"zip_file": "cnsfails202501a.zip"}],
            "event_timestamp": pd.to_datetime(["2025-02-03T00:00:00Z"], utc=True),
            "ref_ftd_log_quantity": [np.log1p(1000.0)],
            "ref_ftd_notional_log1p": [np.log1p(50000.0)],
        }
    )
    monkeypatch.setattr(builder, "_load_fails_to_deliver", lambda **kwargs: ftd)

    result = builder._augment_fails_to_deliver_features(
        base.copy(),
        symbols=["AAPL"],
        min_trade_date=date(2025, 1, 31),
        max_trade_date=date(2025, 2, 3),
    )

    assert result.loc[0, "ref_ftd_count_30d"] == pytest.approx(0.0)
    assert result.loc[1, "ref_ftd_count_30d"] == pytest.approx(1.0)
    assert result.loc[1, "ref_ftd_log_quantity"] == pytest.approx(np.log1p(1000.0))
