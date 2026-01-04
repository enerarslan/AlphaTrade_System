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
import threading
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
# Backtest Mode - Prevents Look-Ahead Bias
# =============================================================================

# Thread-local storage for backtest mode state
_backtest_mode_storage = threading.local()


class BacktestModeError(Exception):
    """Raised when time functions are called with None during backtest mode.

    This prevents look-ahead bias by forcing explicit datetime specification
    during backtesting instead of using real-world current time.
    """
    pass


def is_backtest_mode() -> bool:
    """Check if currently in backtest mode.

    Returns:
        True if backtest mode is active for current thread.
    """
    return getattr(_backtest_mode_storage, "active", False)


def get_backtest_time() -> datetime | None:
    """Get the current simulated time during backtest.

    Returns:
        Current simulated datetime if in backtest mode, None otherwise.
    """
    if not is_backtest_mode():
        return None
    return getattr(_backtest_mode_storage, "current_time", None)


def set_backtest_time(dt: datetime) -> None:
    """Set the current simulated time during backtest.

    Args:
        dt: The simulated datetime to set.

    Raises:
        RuntimeError: If not in backtest mode.
    """
    if not is_backtest_mode():
        raise RuntimeError("Cannot set backtest time outside of backtest mode")
    _backtest_mode_storage.current_time = dt


@contextmanager
def backtest_mode(start_time: datetime | None = None) -> Generator[None, None, None]:
    """Context manager to enable backtest mode.

    When backtest mode is active, time utility functions that would normally
    default to current real-world time will instead:
    1. Raise BacktestModeError if called with dt=None
    2. Allow explicit use of set_backtest_time() to set simulated time

    This CRITICAL feature prevents look-ahead bias in backtesting by ensuring
    all time comparisons use the simulated backtest time, not real-world time.

    Args:
        start_time: Optional initial simulated time. Can be updated via set_backtest_time().

    Usage:
        with backtest_mode(start_time=datetime(2024, 1, 1)):
            for bar in bars:
                set_backtest_time(bar.timestamp)
                if is_market_open():  # Uses simulated time
                    process_bar(bar)

    Yields:
        None

    Note:
        Backtest mode is thread-local, so multiple threads can run backtests
        independently without interference.
    """
    previous_active = getattr(_backtest_mode_storage, "active", False)
    previous_time = getattr(_backtest_mode_storage, "current_time", None)

    try:
        _backtest_mode_storage.active = True
        _backtest_mode_storage.current_time = start_time
        yield
    finally:
        _backtest_mode_storage.active = previous_active
        _backtest_mode_storage.current_time = previous_time


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


# Cache holidays by year with thread-safe access
_holiday_cache: dict[int, set[datetime]] = {}
_holiday_cache_lock = threading.RLock()


def is_market_holiday(dt: datetime | None = None) -> bool:
    """Check if a date is a US market holiday.

    THREAD-SAFE: Uses lock to protect holiday cache access.

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

    # THREAD SAFETY: Use lock to protect cache access
    with _holiday_cache_lock:
        # Cache holidays for the year
        if year not in _holiday_cache:
            _holiday_cache[year] = get_us_market_holidays(year)

        holidays = _holiday_cache[year]

    # Check if date matches any holiday (compare date only, not time)
    dt_date = dt.replace(hour=0, minute=0, second=0, microsecond=0)

    for holiday in holidays:
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
    - BACKTEST MODE SUPPORT: Uses simulated time to prevent look-ahead bias

    Args:
        dt: Datetime to check. If None:
            - In backtest mode: uses simulated backtest time
            - Outside backtest mode: uses current real-world time

    Returns:
        True if market is currently open.

    Raises:
        BacktestModeError: If dt is None during backtest mode and no
            simulated time has been set via set_backtest_time().
    """
    if dt is None:
        # CRITICAL: Prevent look-ahead bias during backtesting
        if is_backtest_mode():
            backtest_time = get_backtest_time()
            if backtest_time is None:
                raise BacktestModeError(
                    "is_market_open() called with dt=None during backtest mode, "
                    "but no simulated time has been set. Either pass an explicit "
                    "datetime or call set_backtest_time() first to prevent look-ahead bias."
                )
            dt = to_eastern(backtest_time)
        else:
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

    BACKTEST MODE SUPPORT: Uses simulated time to prevent look-ahead bias.

    Args:
        date: Date to get open time for. If None:
            - In backtest mode: uses simulated backtest date
            - Outside backtest mode: uses today's date

    Returns:
        Market open datetime in Eastern time (9:30 AM).

    Raises:
        BacktestModeError: If date is None during backtest mode and no
            simulated time has been set via set_backtest_time().
    """
    if date is None:
        # CRITICAL: Prevent look-ahead bias during backtesting
        if is_backtest_mode():
            backtest_time = get_backtest_time()
            if backtest_time is None:
                raise BacktestModeError(
                    "get_market_open_time() called with date=None during backtest mode, "
                    "but no simulated time has been set. Either pass an explicit "
                    "datetime or call set_backtest_time() first to prevent look-ahead bias."
                )
            date = to_eastern(backtest_time)
        else:
            date = eastern_now()
    else:
        date = to_eastern(date)
    return date.replace(hour=9, minute=30, second=0, microsecond=0)


def get_market_close_time(date: datetime | None = None) -> datetime:
    """Get market close time for a given date.

    BACKTEST MODE SUPPORT: Uses simulated time to prevent look-ahead bias.

    Args:
        date: Date to get close time for. If None:
            - In backtest mode: uses simulated backtest date
            - Outside backtest mode: uses today's date

    Returns:
        Market close datetime in Eastern time (4:00 PM).

    Raises:
        BacktestModeError: If date is None during backtest mode and no
            simulated time has been set via set_backtest_time().
    """
    if date is None:
        # CRITICAL: Prevent look-ahead bias during backtesting
        if is_backtest_mode():
            backtest_time = get_backtest_time()
            if backtest_time is None:
                raise BacktestModeError(
                    "get_market_close_time() called with date=None during backtest mode, "
                    "but no simulated time has been set. Either pass an explicit "
                    "datetime or call set_backtest_time() first to prevent look-ahead bias."
                )
            date = to_eastern(backtest_time)
        else:
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

    P1 FIX: Preserves Decimal precision through calculation, only converting
    to float at the final output. This prevents compounding rounding errors
    in PnL calculations.

    Args:
        entry_price: Entry price.
        exit_price: Exit price.

    Returns:
        Return as decimal (e.g., 0.05 for 5%).
    """
    # P1 FIX: Convert to Decimal first to preserve precision
    # Use str() conversion to avoid float precision issues
    entry = Decimal(str(entry_price)) if not isinstance(entry_price, Decimal) else entry_price
    exit_ = Decimal(str(exit_price)) if not isinstance(exit_price, Decimal) else exit_price

    if entry == 0:
        return 0.0

    # Calculate in Decimal, convert only at output
    return float((exit_ - entry) / entry)


def calculate_log_return(
    entry_price: Decimal | float,
    exit_price: Decimal | float,
) -> float:
    """Calculate log return between two prices.

    P1 FIX: Preserves Decimal precision through calculation, only converting
    to float at the final output.

    Args:
        entry_price: Entry price.
        exit_price: Exit price.

    Returns:
        Log return.
    """
    import math

    # P1 FIX: Convert to Decimal first to preserve precision
    entry = Decimal(str(entry_price)) if not isinstance(entry_price, Decimal) else entry_price
    exit_ = Decimal(str(exit_price)) if not isinstance(exit_price, Decimal) else exit_price

    if entry <= 0 or exit_ <= 0:
        return 0.0

    # For log, we need to convert to float for math.log
    # but the ratio calculation preserves Decimal precision
    ratio = float(exit_ / entry)
    return math.log(ratio)


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


# =============================================================================
# Validation Decorators
# =============================================================================


# Import ValidationError from exceptions module for consistency
# We also define it here for backward compatibility
from quant_trading_system.core.exceptions import ValidationError


def validate_order_params(
    symbol_param: str = "symbol",
    quantity_param: str | None = "quantity",
    price_param: str | None = "price",
    limit_price_param: str | None = "limit_price",
    stop_price_param: str | None = "stop_price",
    strict: bool = True,
) -> Callable[[F], F]:
    """Decorator to validate order parameters.

    This decorator provides reusable validation for order-related methods,
    ensuring that symbols, quantities, and prices are valid before execution.

    Args:
        symbol_param: Name of the symbol parameter to validate.
        quantity_param: Name of the quantity parameter (None to skip).
        price_param: Name of the price parameter (None to skip).
        limit_price_param: Name of the limit price parameter (None to skip).
        stop_price_param: Name of the stop price parameter (None to skip).
        strict: If True, raise ValidationError; if False, log warning and continue.

    Returns:
        Decorator function.

    Example:
        @validate_order_params(symbol_param="ticker", quantity_param="qty")
        def submit_order(self, ticker: str, qty: Decimal, price: Decimal):
            ...

    Raises:
        ValidationError: If strict=True and validation fails.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build param dict from args and kwargs
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            params = dict(bound.arguments)

            errors: list[str] = []

            # Validate symbol
            if symbol_param and symbol_param in params:
                symbol = params[symbol_param]
                if symbol is not None and not validate_symbol(str(symbol)):
                    errors.append(f"Invalid symbol format: {symbol}")

            # Validate quantity
            if quantity_param and quantity_param in params:
                quantity = params[quantity_param]
                if quantity is not None and not validate_quantity(quantity):
                    errors.append(f"Invalid quantity (must be positive): {quantity}")

            # Validate price
            if price_param and price_param in params:
                price = params[price_param]
                if price is not None and not validate_price(price):
                    errors.append(f"Invalid price (must be non-negative): {price}")

            # Validate limit price
            if limit_price_param and limit_price_param in params:
                limit_price = params[limit_price_param]
                if limit_price is not None and not validate_price(limit_price):
                    errors.append(f"Invalid limit_price (must be non-negative): {limit_price}")

            # Validate stop price
            if stop_price_param and stop_price_param in params:
                stop_price = params[stop_price_param]
                if stop_price is not None and not validate_price(stop_price):
                    errors.append(f"Invalid stop_price (must be non-negative): {stop_price}")

            if errors:
                error_msg = f"Order parameter validation failed: {'; '.join(errors)}"
                if strict:
                    raise ValidationError(error_msg)
                else:
                    logger.warning(error_msg)

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def validate_model_input(
    features_param: str = "X",
    min_samples: int = 1,
    max_features: int | None = None,
    check_nan: bool = True,
    check_inf: bool = True,
) -> Callable[[F], F]:
    """Decorator to validate ML model input arrays.

    Ensures input data meets requirements before model training/prediction,
    preventing cryptic errors from underlying ML libraries.

    Args:
        features_param: Name of the features parameter (numpy array).
        min_samples: Minimum number of samples required.
        max_features: Maximum number of features allowed (None for unlimited).
        check_nan: Check for NaN values in input.
        check_inf: Check for infinite values in input.

    Returns:
        Decorator function.

    Example:
        @validate_model_input(features_param="X", min_samples=10)
        def fit(self, X: np.ndarray, y: np.ndarray):
            ...

    Raises:
        ValidationError: If validation fails.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import inspect
            import numpy as np

            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            params = dict(bound.arguments)

            if features_param not in params:
                return func(*args, **kwargs)

            X = params[features_param]
            if X is None:
                raise ValidationError(f"{features_param} cannot be None")

            # Convert to numpy if needed
            if hasattr(X, 'values'):  # DataFrame/Series
                X = X.values

            if not isinstance(X, np.ndarray):
                raise ValidationError(
                    f"{features_param} must be a numpy array, got {type(X).__name__}"
                )

            # Check dimensions
            if X.ndim < 1:
                raise ValidationError(f"{features_param} must have at least 1 dimension")

            n_samples = X.shape[0] if X.ndim >= 1 else 1
            n_features = X.shape[1] if X.ndim >= 2 else 1

            # Check minimum samples
            if n_samples < min_samples:
                raise ValidationError(
                    f"{features_param} has {n_samples} samples, "
                    f"minimum required is {min_samples}"
                )

            # Check maximum features
            if max_features is not None and n_features > max_features:
                raise ValidationError(
                    f"{features_param} has {n_features} features, "
                    f"maximum allowed is {max_features}"
                )

            # Check for NaN
            if check_nan and np.isnan(X).any():
                nan_count = np.isnan(X).sum()
                raise ValidationError(
                    f"{features_param} contains {nan_count} NaN values. "
                    "Please clean data before model input."
                )

            # Check for infinite values
            if check_inf and np.isinf(X).any():
                inf_count = np.isinf(X).sum()
                raise ValidationError(
                    f"{features_param} contains {inf_count} infinite values. "
                    "Please clean data before model input."
                )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def validate_positive(
    *param_names: str,
    allow_zero: bool = False,
) -> Callable[[F], F]:
    """Decorator to validate that parameters are positive numbers.

    Args:
        *param_names: Names of parameters that must be positive.
        allow_zero: If True, zero values are allowed.

    Returns:
        Decorator function.

    Example:
        @validate_positive("quantity", "price", allow_zero=False)
        def create_order(self, quantity: Decimal, price: Decimal):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            params = dict(bound.arguments)

            errors: list[str] = []
            for param_name in param_names:
                if param_name not in params:
                    continue
                value = params[param_name]
                if value is None:
                    continue

                try:
                    num_value = float(value)
                    if allow_zero:
                        if num_value < 0:
                            errors.append(f"{param_name} must be non-negative, got {value}")
                    else:
                        if num_value <= 0:
                            errors.append(f"{param_name} must be positive, got {value}")
                except (ValueError, TypeError):
                    errors.append(f"{param_name} must be a number, got {type(value).__name__}")

            if errors:
                raise ValidationError(f"Validation failed: {'; '.join(errors)}")

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def validate_range(
    param_name: str,
    min_value: float | None = None,
    max_value: float | None = None,
    inclusive: bool = True,
) -> Callable[[F], F]:
    """Decorator to validate that a parameter is within a specified range.

    Args:
        param_name: Name of the parameter to validate.
        min_value: Minimum allowed value (None for no minimum).
        max_value: Maximum allowed value (None for no maximum).
        inclusive: If True, boundary values are allowed.

    Returns:
        Decorator function.

    Example:
        @validate_range("confidence", min_value=0.0, max_value=1.0)
        def set_signal(self, confidence: float):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            params = dict(bound.arguments)

            if param_name not in params:
                return func(*args, **kwargs)

            value = params[param_name]
            if value is None:
                return func(*args, **kwargs)

            try:
                num_value = float(value)
            except (ValueError, TypeError):
                raise ValidationError(
                    f"{param_name} must be a number, got {type(value).__name__}"
                )

            if min_value is not None:
                if inclusive and num_value < min_value:
                    raise ValidationError(
                        f"{param_name} must be >= {min_value}, got {num_value}"
                    )
                elif not inclusive and num_value <= min_value:
                    raise ValidationError(
                        f"{param_name} must be > {min_value}, got {num_value}"
                    )

            if max_value is not None:
                if inclusive and num_value > max_value:
                    raise ValidationError(
                        f"{param_name} must be <= {max_value}, got {num_value}"
                    )
                elif not inclusive and num_value >= max_value:
                    raise ValidationError(
                        f"{param_name} must be < {max_value}, got {num_value}"
                    )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# JPMorgan-level Enhancement: Circuit Breaker Pattern
# =============================================================================


class CircuitBreakerState:
    """Circuit breaker states.

    P3 DEPRECATION: This class is deprecated. Use core/circuit_breaker.py instead:
        from quant_trading_system.core.circuit_breaker import CircuitState
    """
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open.

    P3 DEPRECATION: This class is deprecated. Use core/circuit_breaker.py instead:
        from quant_trading_system.core.circuit_breaker import CircuitBreakerException
    """

    def __init__(self, message: str, service_name: str, time_until_retry: float):
        import warnings
        warnings.warn(
            "CircuitBreakerError from utils.py is deprecated. "
            "Use quant_trading_system.core.circuit_breaker.CircuitBreakerException instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(message)
        self.service_name = service_name
        self.time_until_retry = time_until_retry


class CircuitBreaker:
    """Circuit breaker for protecting external service calls.

    P3 DEPRECATION: This class is deprecated. Use core/circuit_breaker.py instead:
        from quant_trading_system.core.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            circuit_breaker_registry
        )

    The dedicated circuit_breaker.py module provides:
    - Configurable failure/success thresholds
    - Half-open state strategies
    - Centralized registry
    - Decorator for easy integration
    - Comprehensive statistics tracking

    JPMorgan-level enhancement: Implements the circuit breaker pattern
    to prevent cascading failures and provide graceful degradation
    when external services are unavailable.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests are blocked
    - HALF_OPEN: Testing if service has recovered

    Example:
        >>> breaker = CircuitBreaker("alpaca_api", failure_threshold=5)
        >>> try:
        ...     async with breaker:
        ...         await alpaca_client.get_account()
        ... except CircuitBreakerError as e:
        ...     logger.warning(f"Service unavailable: {e}")
        ...     use_cached_data()
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        """Initialize circuit breaker.

        Args:
            name: Name of the service being protected.
            failure_threshold: Number of failures before opening circuit.
            success_threshold: Number of successes in half-open to close circuit.
            timeout_seconds: Time to wait before trying half-open.
            half_open_max_calls: Max concurrent calls in half-open state.
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = threading.RLock()

        # Metrics
        self._total_failures = 0
        self._total_successes = 0
        self._times_opened = 0

    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    def _check_state_transition(self) -> None:
        """Check and perform state transitions based on timeout."""
        if self._state == CircuitBreakerState.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.timeout_seconds:
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"Circuit breaker '{self.name}' entering half-open state")

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._total_successes += 1

            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitBreakerState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' closed after recovery")
            elif self._state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        with self._lock:
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitBreakerState.HALF_OPEN:
                # Failure during half-open, go back to open
                self._state = CircuitBreakerState.OPEN
                self._success_count = 0
                logger.warning(
                    f"Circuit breaker '{self.name}' reopened after half-open failure: {error}"
                )
            elif self._state == CircuitBreakerState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitBreakerState.OPEN
                    self._times_opened += 1
                    logger.error(
                        f"Circuit breaker '{self.name}' opened after {self._failure_count} failures"
                    )

    def _can_execute(self) -> bool:
        """Check if a call can be executed."""
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitBreakerState.CLOSED:
                return True
            elif self._state == CircuitBreakerState.OPEN:
                return False
            elif self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    def _get_time_until_retry(self) -> float:
        """Get time remaining until circuit breaker will try half-open."""
        if self._last_failure_time is None:
            return 0.0
        elapsed = time.time() - self._last_failure_time
        return max(0.0, self.timeout_seconds - elapsed)

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a function through the circuit breaker.

        Args:
            func: Function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Function result if successful.

        Raises:
            CircuitBreakerError: If circuit is open.
            Exception: Re-raises any exception from the function.
        """
        if not self._can_execute():
            time_until_retry = self._get_time_until_retry()
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is open. "
                f"Retry in {time_until_retry:.1f}s",
                self.name,
                time_until_retry,
            )

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    async def call_async(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute an async function through the circuit breaker.

        Args:
            func: Async function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Function result if successful.

        Raises:
            CircuitBreakerError: If circuit is open.
            Exception: Re-raises any exception from the function.
        """
        if not self._can_execute():
            time_until_retry = self._get_time_until_retry()
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is open. "
                f"Retry in {time_until_retry:.1f}s",
                self.name,
                time_until_retry,
            )

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    def __enter__(self) -> "CircuitBreaker":
        """Context manager entry for sync calls."""
        if not self._can_execute():
            time_until_retry = self._get_time_until_retry()
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is open. "
                f"Retry in {time_until_retry:.1f}s",
                self.name,
                time_until_retry,
            )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit for sync calls."""
        if exc_type is None:
            self._record_success()
        elif exc_type is not CircuitBreakerError:
            self._record_failure(exc_val)  # type: ignore

    async def __aenter__(self) -> "CircuitBreaker":
        """Async context manager entry."""
        if not self._can_execute():
            time_until_retry = self._get_time_until_retry()
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is open. "
                f"Retry in {time_until_retry:.1f}s",
                self.name,
                time_until_retry,
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        if exc_type is None:
            self._record_success()
        elif exc_type is not CircuitBreakerError:
            self._record_failure(exc_val)  # type: ignore

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0
            logger.info(f"Circuit breaker '{self.name}' manually reset")

    def get_metrics(self) -> dict[str, Any]:
        """Get circuit breaker metrics.

        Returns:
            Dictionary with circuit breaker statistics.
        """
        with self._lock:
            return {
                "name": self.name,
                "state": self._state,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_failures": self._total_failures,
                "total_successes": self._total_successes,
                "times_opened": self._times_opened,
                "time_until_retry": self._get_time_until_retry(),
            }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers.

    JPMorgan-level enhancement: Centralized management of circuit
    breakers for all external services.
    """

    _instance: "CircuitBreakerRegistry | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "CircuitBreakerRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._breakers = {}
                    instance._default_config = {
                        "failure_threshold": 5,
                        "success_threshold": 2,
                        "timeout_seconds": 60.0,
                    }
                    cls._instance = instance
        return cls._instance

    @classmethod
    def get_instance(cls) -> "CircuitBreakerRegistry":
        """Get the singleton instance."""
        return cls()

    def get_or_create(
        self,
        name: str,
        failure_threshold: int | None = None,
        success_threshold: int | None = None,
        timeout_seconds: float | None = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker by name.

        Args:
            name: Name of the service.
            failure_threshold: Override default failure threshold.
            success_threshold: Override default success threshold.
            timeout_seconds: Override default timeout.

        Returns:
            CircuitBreaker instance.
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold or self._default_config["failure_threshold"],
                success_threshold=success_threshold or self._default_config["success_threshold"],
                timeout_seconds=timeout_seconds or self._default_config["timeout_seconds"],
            )
        return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """Get a circuit breaker by name.

        Args:
            name: Name of the service.

        Returns:
            CircuitBreaker if exists, None otherwise.
        """
        return self._breakers.get(name)

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all circuit breakers.

        Returns:
            Dictionary mapping service names to metrics.
        """
        return {name: cb.get_metrics() for name, cb in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for cb in self._breakers.values():
            cb.reset()


def get_circuit_breaker(name: str, **kwargs: Any) -> CircuitBreaker:
    """Get or create a circuit breaker from the global registry.

    Args:
        name: Name of the service.
        **kwargs: Additional circuit breaker configuration.

    Returns:
        CircuitBreaker instance.
    """
    return CircuitBreakerRegistry.get_instance().get_or_create(name, **kwargs)
