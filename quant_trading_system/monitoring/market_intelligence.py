"""Database-backed market intelligence helpers for dashboard APIs.

This module provides read-only market data snapshots used by the dashboard UI.
The PostgreSQL/TimescaleDB runtime remains the primary source of truth.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from statistics import mean
from typing import Any, Iterable

from sqlalchemy import text
from sqlalchemy.engine import Connection

from quant_trading_system.database.connection import get_engine

logger = logging.getLogger(__name__)

DEFAULT_MACRO_SERIES = ("DGS10", "FEDFUNDS", "VIX", "UNRATE", "PAYEMS")

SERIES_IMPACT = {
    "FEDFUNDS": "HIGH",
    "DGS10": "HIGH",
    "VIX": "HIGH",
    "UNRATE": "HIGH",
    "PAYEMS": "HIGH",
    "CPIAUCSL": "HIGH",
    "RSAFS": "MEDIUM",
    "DGORDER": "MEDIUM",
    "GDPC1": "HIGH",
    "DGS2": "MEDIUM",
}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    """Convert database numeric values to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert database integer values to int."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_iso(value: Any) -> str | None:
    """Convert datetime/date values to ISO-8601."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _to_unix_seconds(value: Any) -> int | None:
    """Convert datetime/date values to Unix seconds."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt_value = value
    elif isinstance(value, date):
        dt_value = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    else:
        return None

    if dt_value.tzinfo is None:
        dt_value = dt_value.replace(tzinfo=timezone.utc)
    return int(dt_value.timestamp())


def _normalize_symbols(symbols: str | Iterable[str] | None) -> list[str]:
    """Parse, uppercase, and deduplicate symbols while preserving order."""
    if symbols is None:
        return []

    if isinstance(symbols, str):
        raw_items = symbols.split(",")
    else:
        raw_items = list(symbols)

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        symbol = str(item).strip().upper()
        if not symbol or symbol in seen:
            continue
        normalized.append(symbol)
        seen.add(symbol)
    return normalized


def _build_in_clause(param_prefix: str, values: list[str]) -> tuple[str, dict[str, Any]]:
    """Build a safe SQL IN clause with bound parameters."""
    params: dict[str, Any] = {}
    placeholders: list[str] = []
    for idx, value in enumerate(values):
        key = f"{param_prefix}_{idx}"
        placeholders.append(f":{key}")
        params[key] = value
    return ", ".join(placeholders), params


def _compute_pct_change(current: float | None, previous: float | None) -> float | None:
    """Compute percentage change."""
    if current is None or previous in (None, 0):
        return None
    return ((current - previous) / abs(previous)) * 100.0


def _compute_trend_pct(current: float | None, baseline: list[float]) -> float:
    """Compute trend against a prior baseline window."""
    baseline_values = [value for value in baseline if value is not None]
    if current is None or not baseline_values:
        return 0.0
    baseline_mean = mean(baseline_values)
    if baseline_mean == 0:
        return 0.0
    return ((current - baseline_mean) / abs(baseline_mean)) * 100.0


def _compute_midpoint(bid_price: float | None, ask_price: float | None) -> float | None:
    """Compute mid-price from best bid/ask when available."""
    if bid_price is not None and ask_price is not None:
        return (bid_price + ask_price) / 2.0
    return bid_price if bid_price is not None else ask_price


def _compute_rsi_series(closes: list[float], period: int = 14) -> list[float | None]:
    """Compute RSI values for a close series."""
    if len(closes) < 2:
        return [None] * len(closes)

    deltas = [closes[idx] - closes[idx - 1] for idx in range(1, len(closes))]
    gains = [max(delta, 0.0) for delta in deltas]
    losses = [abs(min(delta, 0.0)) for delta in deltas]
    rsi: list[float | None] = [None] * len(closes)

    if len(deltas) < period:
        return rsi

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    if avg_loss == 0 and avg_gain == 0:
        rsi[period] = 50.0
    elif avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for idx in range(period + 1, len(closes)):
        gain = gains[idx - 1]
        loss = losses[idx - 1]
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period

        if avg_loss == 0 and avg_gain == 0:
            rsi[idx] = 50.0
        elif avg_loss == 0:
            rsi[idx] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[idx] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def _classify_sentiment(score: float | None) -> str:
    """Map numeric sentiment to a display bucket."""
    if score is None:
        return "neutral"
    if score >= 0.25:
        return "bullish"
    if score <= -0.25:
        return "bearish"
    return "neutral"


def _load_quotes(conn: Connection, symbols: list[str]) -> list[dict[str, Any]]:
    """Load latest watchlist quotes from quotes + daily bars."""
    if not symbols:
        return []

    symbol_clause, symbol_params = _build_in_clause("symbol", symbols)

    latest_quotes = conn.execute(
        text(
            f"""
            SELECT DISTINCT ON (q.symbol)
                   q.symbol,
                   q.timestamp,
                   q.bid_price,
                   q.ask_price,
                   q.bid_size,
                   q.ask_size
            FROM stock_quotes q
            WHERE q.symbol IN ({symbol_clause})
            ORDER BY q.symbol, q.timestamp DESC
            """
        ),
        symbol_params,
    ).mappings()

    quote_by_symbol = {str(row["symbol"]).upper(): dict(row) for row in latest_quotes}

    daily_rows = conn.execute(
        text(
            f"""
            SELECT ranked.symbol,
                   ranked.timestamp,
                   ranked.close,
                   ranked.high,
                   ranked.low,
                   ranked.volume,
                   ranked.rn
            FROM (
                SELECT o.symbol,
                       o.timestamp,
                       o.close,
                       o.high,
                       o.low,
                       o.volume,
                       ROW_NUMBER() OVER (PARTITION BY o.symbol ORDER BY o.timestamp DESC) AS rn
                FROM ohlcv_bars o
                WHERE o.symbol IN ({symbol_clause})
                  AND o.timeframe = :timeframe
            ) ranked
            WHERE ranked.rn <= 2
            """
        ),
        {**symbol_params, "timeframe": "1Day"},
    ).mappings()

    bars_by_symbol: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in daily_rows:
        bars_by_symbol[str(row["symbol"]).upper()][int(row["rn"])] = dict(row)

    sector_rows = conn.execute(
        text(
            f"""
            SELECT symbol, sector
            FROM security_master
            WHERE symbol IN ({symbol_clause})
            """
        ),
        symbol_params,
    ).mappings()
    sectors = {str(row["symbol"]).upper(): row.get("sector") or "other" for row in sector_rows}

    results: list[dict[str, Any]] = []
    for symbol in symbols:
        quote_row = quote_by_symbol.get(symbol, {})
        current_bar = bars_by_symbol.get(symbol, {}).get(1, {})
        previous_bar = bars_by_symbol.get(symbol, {}).get(2, {})

        bid_price = _safe_float(quote_row.get("bid_price"))
        ask_price = _safe_float(quote_row.get("ask_price"))
        midpoint = _compute_midpoint(bid_price, ask_price)
        current_close = _safe_float(current_bar.get("close"))
        previous_close = _safe_float(previous_bar.get("close"))
        price = midpoint if midpoint is not None else current_close

        if price is None:
            continue

        change = (price - previous_close) if previous_close is not None else 0.0
        change_percent = _compute_pct_change(price, previous_close) or 0.0

        results.append(
            {
                "symbol": symbol,
                "price": price,
                "change": change,
                "changePercent": change_percent,
                "volume": _safe_int(current_bar.get("volume")),
                "high": _safe_float(current_bar.get("high"), price) or price,
                "low": _safe_float(current_bar.get("low"), price) or price,
                "sector": str(sectors.get(symbol, "other")).lower(),
                "asOf": _to_iso(quote_row.get("timestamp") or current_bar.get("timestamp")),
                "bidPrice": bid_price,
                "askPrice": ask_price,
                "bidSize": _safe_int(quote_row.get("bid_size")),
                "askSize": _safe_int(quote_row.get("ask_size")),
            }
        )

    return results


def _load_news(conn: Connection, limit: int) -> list[dict[str, Any]]:
    """Load recent news articles from the database."""
    result = conn.execute(
        text(
            """
            SELECT article_id,
                   source,
                   headline,
                   summary,
                   url,
                   symbols,
                   sentiment,
                   created_at_source
            FROM news_articles
            ORDER BY created_at_source DESC NULLS LAST
            LIMIT :limit
            """
        ),
        {"limit": min(max(limit, 1), 100)},
    ).mappings()

    rows: list[dict[str, Any]] = []
    for row in result:
        published_at = row.get("created_at_source")
        rows.append(
            {
                "id": str(row.get("article_id") or ""),
                "headline": row.get("headline") or "",
                "summary": row.get("summary") or "",
                "url": row.get("url") or "",
                "source": row.get("source") or "database",
                "symbols": list(row.get("symbols") or []),
                "sentiment": _safe_float(row.get("sentiment")),
                "classification": _classify_sentiment(_safe_float(row.get("sentiment"))),
                "datetime": _to_unix_seconds(published_at),
                "publishedAt": _to_iso(published_at),
                "image": None,
            }
        )
    return rows


def _load_macro_histories(
    conn: Connection,
    series_ids: list[str],
    history_limit: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    """Load recent macro observation history by series."""
    normalized = _normalize_symbols(series_ids)
    if not normalized:
        return {}

    series_clause, series_params = _build_in_clause("series", normalized)
    result = conn.execute(
        text(
            f"""
            SELECT ranked.series_id,
                   ranked.observation_date,
                   ranked.source,
                   ranked.name,
                   ranked.interval,
                   ranked.unit,
                   ranked.value
            FROM (
                SELECT m.series_id,
                       m.observation_date,
                       m.source,
                       m.name,
                       m.interval,
                       m.unit,
                       m.value,
                       ROW_NUMBER() OVER (
                           PARTITION BY m.series_id
                           ORDER BY m.observation_date DESC
                       ) AS rn
                FROM macro_observations m
                WHERE m.series_id IN ({series_clause})
            ) ranked
            WHERE ranked.rn <= :history_limit
            ORDER BY ranked.series_id, ranked.observation_date DESC
            """
        ),
        {**series_params, "history_limit": min(max(history_limit, 2), 60)},
    ).mappings()

    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result:
        series_id = str(row.get("series_id") or "").upper()
        histories[series_id].append(
            {
                "seriesId": series_id,
                "observationDate": _to_iso(row.get("observation_date")),
                "source": row.get("source") or "macro",
                "name": row.get("name") or series_id,
                "interval": row.get("interval") or "",
                "unit": row.get("unit") or "",
                "value": _safe_float(row.get("value")),
            }
        )
    return histories


def _build_macro_series(histories: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Build summary cards for key macro series."""
    rows: list[dict[str, Any]] = []
    for series_id in DEFAULT_MACRO_SERIES:
        series_history = histories.get(series_id, [])
        if not series_history:
            continue

        latest = series_history[0]
        previous = series_history[1] if len(series_history) > 1 else None
        spark = [
            value["value"]
            for value in reversed(series_history)
            if value.get("value") is not None
        ]
        current_value = latest.get("value")
        previous_value = previous.get("value") if previous else None
        change = (
            current_value - previous_value
            if current_value is not None and previous_value is not None
            else None
        )

        rows.append(
            {
                "seriesId": series_id,
                "name": latest.get("name") or series_id,
                "unit": latest.get("unit") or "",
                "observationDate": latest.get("observationDate"),
                "value": current_value,
                "previousValue": previous_value,
                "change": change,
                "changePercent": _compute_pct_change(current_value, previous_value),
                "spark": spark,
                "impact": SERIES_IMPACT.get(series_id, "LOW"),
            }
        )

    return rows


def _build_macro_board(histories: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Build a recent macro release board from series histories."""
    rows: list[dict[str, Any]] = []
    for series_id, series_history in histories.items():
        for index, item in enumerate(series_history[:4]):
            previous_value = None
            if index + 1 < len(series_history):
                previous_value = series_history[index + 1].get("value")
            current_value = item.get("value")
            change = (
                current_value - previous_value
                if current_value is not None and previous_value is not None
                else None
            )
            observation_date = item.get("observationDate")
            rows.append(
                {
                    "id": f"{series_id}:{observation_date}:{index}",
                    "seriesId": series_id,
                    "name": item.get("name") or series_id,
                    "observationDate": observation_date,
                    "source": item.get("source") or "macro",
                    "unit": item.get("unit") or "",
                    "value": current_value,
                    "previousValue": previous_value,
                    "change": change,
                    "changePercent": _compute_pct_change(current_value, previous_value),
                    "impact": SERIES_IMPACT.get(series_id, "LOW"),
                }
            )

    rows.sort(
        key=lambda item: (
            item.get("observationDate") or "",
            0 if item.get("impact") == "HIGH" else 1 if item.get("impact") == "MEDIUM" else 2,
            item.get("seriesId") or "",
        ),
        reverse=True,
    )
    return rows[:12]


def _load_alternative_metrics(conn: Connection) -> dict[str, Any]:
    """Load alternative/news-derived summary metrics."""
    velocity_rows = list(
        conn.execute(
            text(
                """
                SELECT CAST(created_at_source AT TIME ZONE 'UTC' AS date) AS bucket,
                       COUNT(*) AS article_count,
                       AVG(sentiment) AS avg_sentiment
                FROM news_articles
                WHERE created_at_source >= NOW() - INTERVAL '14 days'
                GROUP BY bucket
                ORDER BY bucket DESC
                """
            )
        ).mappings()
    )

    breadth_rows = list(
        conn.execute(
            text(
                """
                SELECT expanded.bucket,
                       COUNT(DISTINCT expanded.symbol) AS breadth
                FROM (
                    SELECT CAST(created_at_source AT TIME ZONE 'UTC' AS date) AS bucket,
                           jsonb_array_elements_text(COALESCE(symbols, '[]'::jsonb)) AS symbol
                    FROM news_articles
                    WHERE created_at_source >= NOW() - INTERVAL '14 days'
                ) expanded
                GROUP BY expanded.bucket
                ORDER BY expanded.bucket DESC
                """
            )
        ).mappings()
    )

    velocity_by_bucket = {
        str(row.get("bucket")): _safe_int(row.get("article_count")) for row in velocity_rows
    }
    sentiment_by_bucket = {
        str(row.get("bucket")): _safe_float(row.get("avg_sentiment"), 0.0) or 0.0
        for row in velocity_rows
    }
    breadth_by_bucket = {
        str(row.get("bucket")): _safe_int(row.get("breadth")) for row in breadth_rows
    }

    buckets = sorted(
        set(velocity_by_bucket) | set(sentiment_by_bucket) | set(breadth_by_bucket),
        reverse=True,
    )
    if not buckets:
        return {
            "sentimentScore": 50.0,
            "sentimentTrendPct": 0.0,
            "sentimentSpark": [],
            "headlineVelocity24h": 0,
            "headlineVelocityTrendPct": 0.0,
            "velocitySpark": [],
            "coverageBreadth24h": 0,
            "coverageBreadthTrendPct": 0.0,
            "breadthSpark": [],
        }

    sentiment_series = [sentiment_by_bucket.get(bucket, 0.0) for bucket in buckets]
    velocity_series = [velocity_by_bucket.get(bucket, 0) for bucket in buckets]
    breadth_series = [breadth_by_bucket.get(bucket, 0) for bucket in buckets]

    current_sentiment = ((sentiment_series[0] + 1.0) / 2.0) * 100.0
    baseline_sentiment = [((value + 1.0) / 2.0) * 100.0 for value in sentiment_series[1:8]]

    return {
        "sentimentScore": current_sentiment,
        "sentimentTrendPct": _compute_trend_pct(current_sentiment, baseline_sentiment),
        "sentimentSpark": [((value + 1.0) / 2.0) * 100.0 for value in reversed(sentiment_series)],
        "headlineVelocity24h": velocity_series[0],
        "headlineVelocityTrendPct": _compute_trend_pct(float(velocity_series[0]), velocity_series[1:8]),
        "velocitySpark": list(reversed(velocity_series)),
        "coverageBreadth24h": breadth_series[0],
        "coverageBreadthTrendPct": _compute_trend_pct(float(breadth_series[0]), breadth_series[1:8]),
        "breadthSpark": list(reversed(breadth_series)),
    }


def get_market_intelligence(symbols: list[str], news_limit: int = 20) -> dict[str, Any]:
    """Return watchlist, news, macro, and alternative data summary."""
    normalized_symbols = _normalize_symbols(symbols)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "quotes": [],
        "news": [],
        "macroSeries": [],
        "macroBoard": [],
        "alternativeMetrics": {
            "sentimentScore": 50.0,
            "sentimentTrendPct": 0.0,
            "sentimentSpark": [],
            "headlineVelocity24h": 0,
            "headlineVelocityTrendPct": 0.0,
            "velocitySpark": [],
            "coverageBreadth24h": 0,
            "coverageBreadthTrendPct": 0.0,
            "breadthSpark": [],
        },
    }

    try:
        engine = get_engine()
        with engine.connect() as conn:
            payload["quotes"] = _load_quotes(conn, normalized_symbols)
            payload["news"] = _load_news(conn, news_limit)
            histories = _load_macro_histories(conn, list(DEFAULT_MACRO_SERIES))
            payload["macroSeries"] = _build_macro_series(histories)
            payload["macroBoard"] = _build_macro_board(histories)
            payload["alternativeMetrics"] = _load_alternative_metrics(conn)
    except Exception as exc:
        logger.warning("Market intelligence query failed: %s", exc)

    return payload


def get_chart_snapshot(symbol: str, timeframe: str = "1Day", limit: int = 200) -> dict[str, Any]:
    """Return OHLCV chart data with RSI and recent signal markers."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "candles": [],
        "volumes": [],
        "rsi": [],
        "markers": [],
    }

    try:
        engine = get_engine()
        with engine.connect() as conn:
            candle_rows = list(
                conn.execute(
                    text(
                        """
                        SELECT timestamp, open, high, low, close, volume
                        FROM ohlcv_bars
                        WHERE symbol = :symbol
                          AND timeframe = :timeframe
                        ORDER BY timestamp DESC
                        LIMIT :limit
                        """
                    ),
                    {
                        "symbol": symbol.upper(),
                        "timeframe": timeframe,
                        "limit": min(max(limit, 10), 500),
                    },
                ).mappings()
            )

            candles = list(reversed(candle_rows))
            close_values = [_safe_float(row.get("close"), 0.0) or 0.0 for row in candles]
            rsi_values = _compute_rsi_series(close_values, period=14)

            for index, row in enumerate(candles):
                candle_time = row.get("timestamp")
                time_key = _to_iso(candle_time)
                payload["candles"].append(
                    {
                        "time": str(time_key).split("T", 1)[0] if time_key else "",
                        "open": _safe_float(row.get("open"), 0.0) or 0.0,
                        "high": _safe_float(row.get("high"), 0.0) or 0.0,
                        "low": _safe_float(row.get("low"), 0.0) or 0.0,
                        "close": _safe_float(row.get("close"), 0.0) or 0.0,
                    }
                )
                payload["volumes"].append(
                    {
                        "time": str(time_key).split("T", 1)[0] if time_key else "",
                        "value": _safe_int(row.get("volume")),
                    }
                )
                payload["rsi"].append(
                    {
                        "time": str(time_key).split("T", 1)[0] if time_key else "",
                        "value": rsi_values[index],
                    }
                )

            try:
                marker_rows = conn.execute(
                    text(
                        """
                        SELECT signal_id, timestamp, direction, strength
                        FROM signals
                        WHERE symbol = :symbol
                        ORDER BY timestamp DESC
                        LIMIT 20
                        """
                    ),
                    {"symbol": symbol.upper()},
                ).mappings()
                seen_dates: set[str] = set()
                for row in marker_rows:
                    signal_time = _to_iso(row.get("timestamp"))
                    if not signal_time:
                        continue
                    date_key = signal_time.split("T", 1)[0]
                    if date_key in seen_dates:
                        continue
                    seen_dates.add(date_key)
                    direction = str(row.get("direction") or "").upper()
                    side = "BUY" if ("LONG" in direction or "BUY" in direction) else "SELL"
                    payload["markers"].append(
                        {
                            "time": date_key,
                            "side": side,
                            "text": f"{side} {(_safe_float(row.get('strength'), 0.0) or 0.0):.2f}",
                        }
                    )
            except Exception as exc:
                logger.debug("Chart marker query failed for %s: %s", symbol, exc)

    except Exception as exc:
        logger.warning("Chart snapshot query failed for %s: %s", symbol, exc)

    return payload


def get_top_book(symbol: str, limit: int = 15) -> dict[str, Any]:
    """Return recent top-of-book quote ladder for a symbol."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol.upper(),
        "lastPrice": None,
        "spread": None,
        "levels": [],
    }

    try:
        engine = get_engine()
        with engine.connect() as conn:
            rows = list(
                conn.execute(
                    text(
                        """
                        SELECT timestamp,
                               bid_price,
                               bid_size,
                               bid_exchange,
                               ask_price,
                               ask_size,
                               ask_exchange
                        FROM stock_quotes
                        WHERE symbol = :symbol
                        ORDER BY timestamp DESC
                        LIMIT :limit
                        """
                    ),
                    {"symbol": symbol.upper(), "limit": min(max(limit, 5), 30)},
                ).mappings()
            )

            if rows:
                latest = rows[0]
                bid_price = _safe_float(latest.get("bid_price"))
                ask_price = _safe_float(latest.get("ask_price"))
                payload["lastPrice"] = _compute_midpoint(bid_price, ask_price)
                if bid_price is not None and ask_price is not None:
                    payload["spread"] = ask_price - bid_price

            for row in rows:
                bid_price = _safe_float(row.get("bid_price"))
                ask_price = _safe_float(row.get("ask_price"))
                payload["levels"].append(
                    {
                        "timestamp": _to_iso(row.get("timestamp")),
                        "bidPrice": bid_price,
                        "bidSize": _safe_int(row.get("bid_size")),
                        "bidExchange": row.get("bid_exchange") or "",
                        "askPrice": ask_price,
                        "askSize": _safe_int(row.get("ask_size")),
                        "askExchange": row.get("ask_exchange") or "",
                        "midPrice": _compute_midpoint(bid_price, ask_price),
                        "spread": (
                            ask_price - bid_price
                            if bid_price is not None and ask_price is not None
                            else None
                        ),
                    }
                )
    except Exception as exc:
        logger.warning("Top-of-book query failed for %s: %s", symbol, exc)

    return payload


def get_trade_tape(symbol: str, limit: int = 50) -> dict[str, Any]:
    """Return recent trade prints for a symbol."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol.upper(),
        "trades": [],
    }

    try:
        engine = get_engine()
        with engine.connect() as conn:
            rows = list(
                conn.execute(
                    text(
                        """
                        SELECT timestamp,
                               trade_id,
                               price,
                               size,
                               exchange,
                               tape
                        FROM stock_trades
                        WHERE symbol = :symbol
                        ORDER BY timestamp DESC
                        LIMIT :limit
                        """
                    ),
                    {"symbol": symbol.upper(), "limit": min(max(limit, 10), 100)},
                ).mappings()
            )

            for index, row in enumerate(rows):
                price = _safe_float(row.get("price"))
                next_price = (
                    _safe_float(rows[index + 1].get("price"))
                    if index + 1 < len(rows)
                    else price
                )
                if price is None:
                    continue
                direction = "FLAT"
                if next_price is not None:
                    if price > next_price:
                        direction = "UP"
                    elif price < next_price:
                        direction = "DOWN"

                payload["trades"].append(
                    {
                        "id": str(
                            row.get("trade_id")
                            or f"{symbol.upper()}:{_to_iso(row.get('timestamp'))}:{index}"
                        ),
                        "timestamp": _to_iso(row.get("timestamp")),
                        "price": price,
                        "size": _safe_int(row.get("size")),
                        "exchange": row.get("exchange") or "",
                        "tape": row.get("tape") or "",
                        "direction": direction,
                    }
                )
    except Exception as exc:
        logger.warning("Trade tape query failed for %s: %s", symbol, exc)

    return payload


def get_venue_flow(symbols: list[str] | None = None, limit: int = 60) -> dict[str, Any]:
    """Return recent venue flow across recent market trades."""
    normalized_symbols = _normalize_symbols(symbols)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "totalVolume": 0,
            "dominantVenue": None,
            "dominantVenueShare": 0.0,
            "blockCount": 0,
            "uniqueSymbols": 0,
        },
        "prints": [],
    }

    try:
        engine = get_engine()
        with engine.connect() as conn:
            params: dict[str, Any] = {"limit": min(max(limit, 10), 120)}
            where_sql = ""
            if normalized_symbols:
                symbol_clause, symbol_params = _build_in_clause("venue_symbol", normalized_symbols)
                where_sql = f"WHERE symbol IN ({symbol_clause})"
                params.update(symbol_params)

            rows = list(
                conn.execute(
                    text(
                        f"""
                        SELECT symbol,
                               timestamp,
                               trade_id,
                               price,
                               size,
                               exchange,
                               tape
                        FROM stock_trades
                        {where_sql}
                        ORDER BY timestamp DESC
                        LIMIT :limit
                        """
                    ),
                    params,
                ).mappings()
            )

            sizes = sorted(_safe_int(row.get("size")) for row in rows if row.get("size") is not None)
            threshold = 5000
            if sizes:
                percentile_index = min(len(sizes) - 1, math.floor((len(sizes) - 1) * 0.85))
                threshold = max(threshold, sizes[percentile_index])

            total_volume = 0
            venue_volume: dict[str, int] = defaultdict(int)
            unique_symbols: set[str] = set()

            for index, row in enumerate(rows):
                size = _safe_int(row.get("size"))
                price = _safe_float(row.get("price"))
                if price is None:
                    continue
                venue = str(row.get("exchange") or row.get("tape") or "UNKNOWN").upper()
                total_volume += size
                venue_volume[venue] += size
                unique_symbols.add(str(row.get("symbol") or "").upper())
                is_block = size >= threshold

                payload["prints"].append(
                    {
                        "id": str(
                            row.get("trade_id")
                            or f"{row.get('symbol')}:{_to_iso(row.get('timestamp'))}:{index}"
                        ),
                        "timestamp": _to_iso(row.get("timestamp")),
                        "symbol": str(row.get("symbol") or "").upper(),
                        "price": price,
                        "size": size,
                        "venue": venue,
                        "tape": str(row.get("tape") or "").upper(),
                        "exchange": str(row.get("exchange") or "").upper(),
                        "isBlock": is_block,
                    }
                )

            dominant_venue = None
            dominant_share = 0.0
            if venue_volume and total_volume > 0:
                dominant_venue, dominant_volume = max(venue_volume.items(), key=lambda item: item[1])
                dominant_share = (dominant_volume / total_volume) * 100.0

            payload["stats"] = {
                "totalVolume": total_volume,
                "dominantVenue": dominant_venue,
                "dominantVenueShare": dominant_share,
                "blockCount": sum(1 for item in payload["prints"] if item["isBlock"]),
                "uniqueSymbols": len(unique_symbols),
            }
    except Exception as exc:
        logger.warning("Venue flow query failed: %s", exc)

    return payload
