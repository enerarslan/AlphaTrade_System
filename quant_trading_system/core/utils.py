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


# =============================================================================
# US Market Holiday Calendar
# =============================================================================


def get_us_market_holidays(year: int) -> set[datetime]:
    """Get US stock market holidays for a given year.

    Includes all NYSE/NASDAQ holidays:
    - New Year's Day
    - Martin Luther King Jr. Day (3rd Monday in January)
    - Presidents' Day (3rd Monday in February)
    - Good Friday (Friday before Easter Sunday)
    - Memorial Day (Last Monday in May)
    - Juneteenth (June 19, observed)
    - Independence Day (July 4, observed)
    - Labor Day (1st Monday in September)
    - Thanksgiving Day (4th Thursday in November)
    - Christmas Day (December 25, observed)

    Args:
        year: Year to get holidays for.

    Returns:
        Set of holiday dates (datetime with time at midnight ET).
    """
    holidays = set()

    # Helper to get nth weekday of month
    def nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime:
        """Get the nth occurrence of a weekday in a month."""
        first_day = datetime(year, month, 1, tzinfo=EASTERN)
        first_weekday = first_day.weekday()
        days_until = (weekday - first_weekday) % 7
        return first_day + timedelta(days=days_until + (n - 1) * 7)

    def last_weekday(year: int, month: int, weekday: int) -> datetime:
        """Get the last occurrence of a weekday in a month."""
        # Start from 5th week and go back if needed
        for n in range(5, 0, -1):
            try:
                date = nth_weekday(year, month, weekday, n)
                if date.month == month:
                    return date
            except ValueError:
                continue
        return nth_weekday(year, month, weekday, 1)

    def observed_holiday(date: datetime) -> datetime:
        """Adjust for weekend observation (Friday if Saturday, Monday if Sunday)."""
        if date.weekday() == 5:  # Saturday -> Friday
            return date - timedelta(days=1)
        elif date.weekday() == 6:  # Sunday -> Monday
            return date + timedelta(days=1)
        return date

    # New Year's Day (January 1)
    new_years = datetime(year, 1, 1, tzinfo=EASTERN)
    holidays.add(observed_holiday(new_years))

    # MLK Day (3rd Monday in January)
    mlk_day = nth_weekday(year, 1, 0, 3)  # Monday = 0
    holidays.add(mlk_day)

    # Presidents' Day (3rd Monday in February)
    presidents_day = nth_weekday(year, 2, 0, 3)
    holidays.add(presidents_day)

    # Good Friday (requires Easter calculation)
    def calculate_easter(year: int) -> datetime:
        """Calculate Easter Sunday using the Anonymous Gregorian algorithm."""
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return datetime(year, month, day, tzinfo=EASTERN)

    easter = calculate_easter(year)
    good_friday = easter - timedelta(days=2)
    holidays.add(good_friday)

    # Memorial Day (Last Monday in May)
    memorial_day = last_weekday(year, 5, 0)
    holidays.add(memorial_day)

    # Juneteenth (June 19) - observed since 2021
    if year >= 2021:
        juneteenth = datetime(year, 6, 19, tzinfo=EASTERN)
        holidays.add(observed_holiday(juneteenth))

    # Independence Day (July 4)
    independence_day = datetime(year, 7, 4, tzinfo=EASTERN)
    holidays.add(observed_holiday(independence_day))

    # Labor Day (1st Monday in September)
    labor_day = nth_weekday(year, 9, 0, 1)
    holidays.add(labor_day)

    # Thanksgiving (4th Thursday in November)
    thanksgiving = nth_weekday(year, 11, 3, 4)  # Thursday = 3
    holidays.add(thanksgiving)

    # Christmas Day (December 25)
    christmas = datetime(year, 12, 25, tzinfo=EASTERN)
    holidays.add(observed_holiday(christmas))

    return holidays


# Cache holidays by year
_holiday_cache: dict[int, set[datetime]] = {}


def is_market_holiday(dt: datetime | None = None) -> bool:
    """Check if a date is a US market holiday.

    Args:
        dt: Date to check. If None, uses current date.

    Returns:
        True if the date is a market holiday.
    """
    if dt is None:
        dt = eastern_now()
    else:
        dt = to_eastern(dt)

    year = dt.year

    # Cache holidays for the year
    if year not in _holiday_cache:
        _holiday_cache[year] = get_us_market_holidays(year)

    # Check if date matches any holiday (compare date only, not time)
    dt_date = dt.replace(hour=0, minute=0, second=0, microsecond=0)

    for holiday in _holiday_cache[year]:
        holiday_date = holiday.replace(hour=0, minute=0, second=0, microsecond=0)
        if dt_date == holiday_date:
            return True

    return False


def get_next_trading_day(dt: datetime | None = None) -> datetime:
    """Get the next trading day (excludes weekends and holidays).

    Args:
        dt: Starting date. If None, uses current date.

    Returns:
        Next trading day at market open time.
    """
    if dt is None:
        dt = eastern_now()
    else:
        dt = to_eastern(dt)

    # Start from next day
    next_day = dt + timedelta(days=1)
    next_day = next_day.replace(hour=9, minute=30, second=0, microsecond=0)

    # Skip weekends and holidays
    while next_day.weekday() >= 5 or is_market_holiday(next_day):
        next_day += timedelta(days=1)

    return next_day


def get_previous_trading_day(dt: datetime | None = None) -> datetime:
    """Get the previous trading day (excludes weekends and holidays).

    Args:
        dt: Starting date. If None, uses current date.

    Returns:
        Previous trading day at market close time.
    """
    if dt is None:
        dt = eastern_now()
    else:
        dt = to_eastern(dt)

    # Start from previous day
    prev_day = dt - timedelta(days=1)
    prev_day = prev_day.replace(hour=16, minute=0, second=0, microsecond=0)

    # Skip weekends and holidays
    while prev_day.weekday() >= 5 or is_market_holiday(prev_day):
        prev_day -= timedelta(days=1)

    return prev_day


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

    COMPLETE implementation including:
    - Weekend check
    - Holiday check (full NYSE/NASDAQ calendar)
    - Trading hours check (9:30 AM - 4:00 PM ET)

    Args:
        dt: Datetime to check. If None, uses current time.

    Returns:
        True if market is currently open.
    """
    if dt is None:
        dt = eastern_now()
    else:
        dt = to_eastern(dt)

    # Weekend check
    if dt.weekday() >= 5:
        return False

    # Holiday check (uses full holiday calendar)
    if is_market_holiday(dt):
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


def memoize(func: F, maxsize: int = 1024) -> F:
    """Simple memoization decorator for functions with hashable arguments.

    Uses an LRU eviction policy to bound memory usage.

    Args:
        func: Function to memoize.
        maxsize: Maximum cache size (default 1024). Oldest entries evicted when full.

    Returns:
        Memoized function.
    """
    from collections import OrderedDict
    cache: OrderedDict[tuple, Any] = OrderedDict()

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = (args, tuple(sorted(kwargs.items())))
        if key in cache:
            # Move to end (most recently used)
            cache.move_to_end(key)
            return cache[key]

        result = func(*args, **kwargs)
        cache[key] = result

        # Evict oldest if over maxsize
        while len(cache) > maxsize:
            cache.popitem(last=False)

        return result

    wrapper.cache = cache  # type: ignore
    wrapper.clear_cache = cache.clear  # type: ignore
    wrapper.cache_info = lambda: {"size": len(cache), "maxsize": maxsize}  # type: ignore
    return wrapper  # type: ignore


def timed_cache(seconds: int, maxsize: int = 512) -> Callable[[F], F]:
    """Decorator for time-based caching with bounded size.

    Uses LRU eviction when cache is full. Entries also expire after TTL.

    Args:
        seconds: Cache TTL in seconds.
        maxsize: Maximum cache size (default 512).

    Returns:
        Decorator function.
    """
    def decorator(func: F) -> F:
        from collections import OrderedDict
        cache: OrderedDict[tuple, tuple[Any, float]] = OrderedDict()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()

            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < seconds:
                    cache.move_to_end(key)  # Mark as recently used
                    return result
                else:
                    # Expired, remove it
                    del cache[key]

            result = func(*args, **kwargs)
            cache[key] = (result, now)

            # Evict oldest if over maxsize
            while len(cache) > maxsize:
                cache.popitem(last=False)

            return result

        wrapper.cache = cache  # type: ignore
        wrapper.clear_cache = cache.clear  # type: ignore
        wrapper.cache_info = lambda: {"size": len(cache), "maxsize": maxsize, "ttl": seconds}  # type: ignore
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
