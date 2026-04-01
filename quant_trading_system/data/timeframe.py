"""
Timeframe normalization utilities shared by database and script layers.
"""

from __future__ import annotations

import math
from pathlib import Path

DEFAULT_TIMEFRAME = "15Min"
TRADING_DAYS_PER_YEAR = 252
US_EQUITY_REGULAR_SESSION_MINUTES = 390

_TIMEFRAME_ALIASES = {
    "1min": "1Min",
    "1m": "1Min",
    "5min": "5Min",
    "5m": "5Min",
    "15min": "15Min",
    "15m": "15Min",
    "30min": "30Min",
    "30m": "30Min",
    "60min": "1Hour",
    "1h": "1Hour",
    "1hour": "1Hour",
    "hour": "1Hour",
    "240min": "4Hour",
    "4h": "4Hour",
    "4hour": "4Hour",
    "1d": "1Day",
    "1day": "1Day",
    "day": "1Day",
    "daily": "1Day",
}

_FILENAME_SUFFIXES = {
    "_1MIN": "1Min",
    "_5MIN": "5Min",
    "_15MIN": "15Min",
    "_15MINUTE": "15Min",
    "_15M": "15Min",
    "_30MIN": "30Min",
    "_1H": "1Hour",
    "_1HOUR": "1Hour",
    "_HOURLY": "1Hour",
    "_4H": "4Hour",
    "_4HOUR": "4Hour",
    "_1D": "1Day",
    "_1DAY": "1Day",
    "_DAY": "1Day",
    "_DAILY": "1Day",
}

_TIMEFRAME_MINUTES = {
    "1Min": 1,
    "5Min": 5,
    "15Min": 15,
    "30Min": 30,
    "1Hour": 60,
    "4Hour": 240,
    "1Day": US_EQUITY_REGULAR_SESSION_MINUTES,
}


def normalize_timeframe(timeframe: str | None, default: str = DEFAULT_TIMEFRAME) -> str:
    """Normalize user/file/database timeframe labels to a canonical representation."""
    if timeframe is None:
        return default
    raw = str(timeframe).strip()
    if not raw:
        return default
    key = raw.replace(" ", "").replace("-", "").replace("_", "").lower()
    return _TIMEFRAME_ALIASES.get(key, raw)


def timeframe_slug(timeframe: str | None, default: str = DEFAULT_TIMEFRAME) -> str:
    """Build a stable lowercase slug for cache keys and file names."""
    canonical = normalize_timeframe(timeframe, default=default)
    return canonical.lower()


def timeframe_to_minutes(timeframe: str | None, default: str = DEFAULT_TIMEFRAME) -> int:
    """Return the canonical timeframe length in minutes."""
    canonical = normalize_timeframe(timeframe, default=default)
    return int(_TIMEFRAME_MINUTES.get(canonical, _TIMEFRAME_MINUTES[DEFAULT_TIMEFRAME]))


def estimate_periods_per_year(
    timeframe: str | None,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    session_minutes: int = US_EQUITY_REGULAR_SESSION_MINUTES,
) -> int:
    """Estimate annualization periods for US-equity regular-session bars."""
    canonical = normalize_timeframe(timeframe, default=DEFAULT_TIMEFRAME)
    if canonical == "1Day":
        return int(trading_days_per_year)

    minutes = timeframe_to_minutes(canonical, default=DEFAULT_TIMEFRAME)
    bars_per_session = max(1, int(math.ceil(float(session_minutes) / float(minutes))))
    return int(trading_days_per_year) * bars_per_session


def infer_symbol_and_timeframe(
    source: str | Path,
    default_timeframe: str = DEFAULT_TIMEFRAME,
) -> tuple[str, str]:
    """Infer ticker symbol and timeframe from a file stem."""
    stem = Path(source).stem
    normalized_stem = stem.strip()
    stem_upper = normalized_stem.upper()

    for suffix, timeframe in sorted(_FILENAME_SUFFIXES.items(), key=lambda item: len(item[0]), reverse=True):
        if stem_upper.endswith(suffix):
            base = normalized_stem[: -len(suffix)]
            return base.upper(), timeframe

    return normalized_stem.upper(), normalize_timeframe(default_timeframe)
