"""
Timeframe normalization utilities shared by database and script layers.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_TIMEFRAME = "15Min"

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
    "_1D": "1Day",
    "_1DAY": "1Day",
    "_DAY": "1Day",
    "_DAILY": "1Day",
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
