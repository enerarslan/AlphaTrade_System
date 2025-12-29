"""
Shared utilities for the trading system.

Provides common functions for:
- Time and date handling
- Numeric operations and validation
- String formatting
- Caching and memoization
- Retry logic
- Performance timing
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Awaitable, Callable, Generator, TypeVar
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Time and Date Utilities
# =============================================================================

# Common timezone references
UTC = ZoneInfo("UTC")
EASTERN = ZoneInfo("America/New_York")


def utc_now() -> datetime:
    """Get current UTC datetime with timezone info."""
    return datetime.now(UTC)


def eastern_now() -> datetime:
    """Get current Eastern time datetime."""
    return datetime.now(EASTERN)


def to_utc(dt: datetime) -> datetime:
    """Convert datetime to UTC.

    Args:
        dt: Datetime to convert. If naive, assumes UTC.

    Returns:
        UTC datetime.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def to_eastern(dt: datetime) -> datetime:
    """Convert datetime to Eastern time.

    Args:
        dt: Datetime to convert. If naive, assumes UTC.

    Returns:
        Eastern time datetime.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(EASTERN)


def is_market_open(dt: datetime | None = None) -> bool:
    """Check if US stock market is open.

    Args:
        dt: Datetime to check. If None, uses current time.

    Returns:
        True if market is open (simplified check, no holidays).
    """
    if dt is None:
        dt = eastern_now()
    else:
        dt = to_eastern(dt)

    # Weekend check
    if dt.weekday() >= 5:
        return False

    # Time check (9:30 AM - 4:00 PM ET)
    market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= dt <= market_close


def get_market_open_time(date: datetime | None = None) -> datetime:
    """Get market open time for a given date.

    Args:
        date: Date to get open time for. If None, uses today.

    Returns:
        Market open datetime in Eastern time.
    """
    if date is None:
        date = eastern_now()
    else:
        date = to_eastern(date)
    return date.replace(hour=9, minute=30, second=0, microsecond=0)


def get_market_close_time(date: datetime | None = None) -> datetime:
    """Get market close time for a given date.

    Args:
        date: Date to get close time for. If None, uses today.

    Returns:
        Market close datetime in Eastern time.
    """
    if date is None:
        date = eastern_now()
    else:
        date = to_eastern(date)
    return date.replace(hour=16, minute=0, second=0, microsecond=0)


def bars_to_timedelta(bars: int, timeframe: str = "15Min") -> timedelta:
    """Convert number of bars to timedelta.

    Args:
        bars: Number of bars.
        timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day).

    Returns:
        Corresponding timedelta.
    """
    timeframe_map = {
        "1Min": timedelta(minutes=1),
        "5Min": timedelta(minutes=5),
        "15Min": timedelta(minutes=15),
        "1Hour": timedelta(hours=1),
        "1Day": timedelta(days=1),
    }
    base = timeframe_map.get(timeframe, timedelta(minutes=15))
    return base * bars


def timedelta_to_bars(td: timedelta, timeframe: str = "15Min") -> int:
    """Convert timedelta to number of bars.

    Args:
        td: Timedelta to convert.
        timeframe: Bar timeframe.

    Returns:
        Number of bars.
    """
    timeframe_minutes = {
        "1Min": 1,
        "5Min": 5,
        "15Min": 15,
        "1Hour": 60,
        "1Day": 1440,
    }
    minutes_per_bar = timeframe_minutes.get(timeframe, 15)
    total_minutes = td.total_seconds() / 60
    return int(total_minutes / minutes_per_bar)


# =============================================================================
# Numeric Utilities
# =============================================================================


def round_decimal(value: Decimal, places: int = 2) -> Decimal:
    """Round decimal to specified places.

    Args:
        value: Decimal value to round.
        places: Number of decimal places.

    Returns:
        Rounded decimal.
    """
    quantize_str = "0." + "0" * places if places > 0 else "1"
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def to_decimal(value: float | int | str | Decimal, places: int = 8) -> Decimal:
    """Convert value to Decimal with specified precision.

    Args:
        value: Value to convert.
        places: Decimal places for rounding.

    Returns:
        Decimal value.
    """
    if isinstance(value, Decimal):
        return round_decimal(value, places)
    return round_decimal(Decimal(str(value)), places)


def calculate_return(
    entry_price: Decimal | float,
    exit_price: Decimal | float,
) -> float:
    """Calculate simple return between two prices.

    Args:
        entry_price: Entry price.
        exit_price: Exit price.

    Returns:
        Return as decimal (e.g., 0.05 for 5%).
    """
    entry = float(entry_price)
    exit_ = float(exit_price)
    if entry == 0:
        return 0.0
    return (exit_ - entry) / entry


def calculate_log_return(
    entry_price: Decimal | float,
    exit_price: Decimal | float,
) -> float:
    """Calculate log return between two prices.

    Args:
        entry_price: Entry price.
        exit_price: Exit price.

    Returns:
        Log return.
    """
    import math
    entry = float(entry_price)
    exit_ = float(exit_price)
    if entry <= 0 or exit_ <= 0:
        return 0.0
    return math.log(exit_ / entry)


def basis_points_to_decimal(bps: int | float) -> float:
    """Convert basis points to decimal.

    Args:
        bps: Basis points (e.g., 50 for 0.5%).

    Returns:
        Decimal value (e.g., 0.005).
    """
    return bps / 10000


def decimal_to_basis_points(value: float) -> float:
    """Convert decimal to basis points.

    Args:
        value: Decimal value (e.g., 0.005).

    Returns:
        Basis points (e.g., 50).
    """
    return value * 10000


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to specified range.

    Args:
        value: Value to clamp.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.

    Returns:
        Clamped value.
    """
    return max(min_val, min(max_val, value))


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """Safely divide two numbers.

    Args:
        numerator: Numerator.
        denominator: Denominator.
        default: Value to return if denominator is zero.

    Returns:
        Division result or default.
    """
    if denominator == 0:
        return default
    return numerator / denominator


# =============================================================================
# String Utilities
# =============================================================================


def normalize_symbol(symbol: str) -> str:
    """Normalize a stock symbol.

    Args:
        symbol: Symbol to normalize.

    Returns:
        Uppercase, stripped symbol.
    """
    return symbol.upper().strip()


def generate_id(prefix: str = "", length: int = 8) -> str:
    """Generate a unique ID string.

    Args:
        prefix: Optional prefix for the ID.
        length: Length of random portion.

    Returns:
        Generated ID string.
    """
    import uuid
    random_part = uuid.uuid4().hex[:length]
    if prefix:
        return f"{prefix}_{random_part}"
    return random_part


def hash_string(s: str) -> str:
    """Generate SHA-256 hash of a string.

    Args:
        s: String to hash.

    Returns:
        Hex digest of hash.
    """
    return hashlib.sha256(s.encode()).hexdigest()


# =============================================================================
# Caching and Memoization
# =============================================================================


def memoize(func: F) -> F:
    """Simple memoization decorator for functions with hashable arguments.

    Args:
        func: Function to memoize.

    Returns:
        Memoized function.
    """
    cache: dict[tuple, Any] = {}

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    wrapper.cache = cache  # type: ignore
    wrapper.clear_cache = cache.clear  # type: ignore
    return wrapper  # type: ignore


def timed_cache(seconds: int) -> Callable[[F], F]:
    """Decorator for time-based caching.

    Args:
        seconds: Cache TTL in seconds.

    Returns:
        Decorator function.
    """
    def decorator(func: F) -> F:
        cache: dict[tuple, tuple[Any, float]] = {}

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()

            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < seconds:
                    return result

            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result

        wrapper.cache = cache  # type: ignore
        wrapper.clear_cache = cache.clear  # type: ignore
        return wrapper  # type: ignore

    return decorator


# =============================================================================
# Retry Logic
# =============================================================================


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator for retry logic with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts.
        delay: Initial delay between retries in seconds.
        backoff: Multiplier for delay after each retry.
        exceptions: Tuple of exceptions to catch.

    Returns:
        Decorator function.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed")

            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic failed unexpectedly")

        return wrapper  # type: ignore

    return decorator


async def retry_async(
    func: Callable[..., Awaitable[T]],
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Async retry with exponential backoff.

    Args:
        func: Async function to retry.
        max_attempts: Maximum number of attempts.
        delay: Initial delay between retries.
        backoff: Multiplier for delay.
        exceptions: Exceptions to catch.

    Returns:
        Function result.
    """
    current_delay = delay
    last_exception: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                logger.warning(
                    f"Async attempt {attempt + 1}/{max_attempts} failed: {e}. "
                    f"Retrying in {current_delay:.1f}s"
                )
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(f"All {max_attempts} async attempts failed")

    if last_exception:
        raise last_exception
    raise RuntimeError("Async retry logic failed unexpectedly")


# =============================================================================
# Performance Timing
# =============================================================================


@contextmanager
def timer(name: str = "Operation") -> Generator[dict[str, float], None, None]:
    """Context manager for timing operations.

    Args:
        name: Name of the operation being timed.

    Yields:
        Dictionary that will contain 'elapsed' time after context exits.

    Example:
        with timer("Data loading") as t:
            load_data()
        print(f"Took {t['elapsed']:.2f}s")
    """
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result["elapsed"] = elapsed
        logger.debug(f"{name} took {elapsed:.4f}s")


def timed(func: F) -> F:
    """Decorator to log function execution time.

    Args:
        func: Function to time.

    Returns:
        Wrapped function that logs execution time.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger.debug(f"{func.__name__} took {elapsed:.4f}s")

    return wrapper  # type: ignore


# =============================================================================
# Validation Utilities
# =============================================================================


def validate_symbol(symbol: str) -> bool:
    """Validate a stock symbol format.

    Args:
        symbol: Symbol to validate.

    Returns:
        True if valid symbol format.
    """
    if not symbol or not isinstance(symbol, str):
        return False
    symbol = symbol.strip()
    if not 1 <= len(symbol) <= 10:
        return False
    return symbol.isalpha() or (symbol.isalnum() and symbol[0].isalpha())


def validate_quantity(quantity: Decimal | float | int) -> bool:
    """Validate order quantity.

    Args:
        quantity: Quantity to validate.

    Returns:
        True if valid positive quantity.
    """
    try:
        q = float(quantity)
        return q > 0
    except (ValueError, TypeError):
        return False


def validate_price(price: Decimal | float | int) -> bool:
    """Validate price value.

    Args:
        price: Price to validate.

    Returns:
        True if valid non-negative price.
    """
    try:
        p = float(price)
        return p >= 0
    except (ValueError, TypeError):
        return False
