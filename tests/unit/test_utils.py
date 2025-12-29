"""
Unit tests for core/utils.py
"""

import asyncio
import time
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from quant_trading_system.core.utils import (
    EASTERN,
    UTC,
    bars_to_timedelta,
    basis_points_to_decimal,
    calculate_log_return,
    calculate_return,
    clamp,
    decimal_to_basis_points,
    eastern_now,
    generate_id,
    get_market_close_time,
    get_market_open_time,
    hash_string,
    is_market_open,
    memoize,
    normalize_symbol,
    retry,
    retry_async,
    round_decimal,
    safe_divide,
    timed,
    timed_cache,
    timedelta_to_bars,
    timer,
    to_decimal,
    to_eastern,
    to_utc,
    utc_now,
    validate_price,
    validate_quantity,
    validate_symbol,
)


class TestTimeUtilities:
    """Tests for time and date utilities."""

    def test_utc_now(self):
        """Test getting current UTC time."""
        now = utc_now()
        assert now.tzinfo == UTC
        # Should be close to current time
        assert abs((now - datetime.now(UTC)).total_seconds()) < 1

    def test_eastern_now(self):
        """Test getting current Eastern time."""
        now = eastern_now()
        assert now.tzinfo == EASTERN

    def test_to_utc_with_naive_datetime(self):
        """Test converting naive datetime to UTC."""
        naive = datetime(2024, 1, 15, 10, 30, 0)
        result = to_utc(naive)
        assert result.tzinfo == UTC
        assert result.hour == 10  # Assumed UTC, no conversion

    def test_to_utc_with_aware_datetime(self):
        """Test converting aware datetime to UTC."""
        eastern = datetime(2024, 1, 15, 10, 30, 0, tzinfo=EASTERN)
        result = to_utc(eastern)
        assert result.tzinfo == UTC
        # Eastern is UTC-5 in January, so 10:30 ET = 15:30 UTC
        assert result.hour == 15

    def test_to_eastern(self):
        """Test converting to Eastern time."""
        utc_dt = datetime(2024, 1, 15, 15, 30, 0, tzinfo=UTC)
        result = to_eastern(utc_dt)
        assert result.tzinfo == EASTERN
        # 15:30 UTC = 10:30 ET in January
        assert result.hour == 10

    def test_is_market_open_during_trading_hours(self):
        """Test market open check during trading hours."""
        # Tuesday at 10:30 ET
        trading_time = datetime(2024, 1, 16, 10, 30, 0, tzinfo=EASTERN)
        assert is_market_open(trading_time) is True

    def test_is_market_open_before_open(self):
        """Test market open check before market opens."""
        # Tuesday at 8:00 ET
        early_time = datetime(2024, 1, 16, 8, 0, 0, tzinfo=EASTERN)
        assert is_market_open(early_time) is False

    def test_is_market_open_after_close(self):
        """Test market open check after market closes."""
        # Tuesday at 17:00 ET
        late_time = datetime(2024, 1, 16, 17, 0, 0, tzinfo=EASTERN)
        assert is_market_open(late_time) is False

    def test_is_market_open_weekend(self):
        """Test market open check on weekend."""
        # Saturday at noon ET
        weekend_time = datetime(2024, 1, 20, 12, 0, 0, tzinfo=EASTERN)
        assert is_market_open(weekend_time) is False

    def test_get_market_open_time(self):
        """Test getting market open time."""
        date = datetime(2024, 1, 15, 14, 0, 0, tzinfo=EASTERN)
        open_time = get_market_open_time(date)
        assert open_time.hour == 9
        assert open_time.minute == 30

    def test_get_market_close_time(self):
        """Test getting market close time."""
        date = datetime(2024, 1, 15, 14, 0, 0, tzinfo=EASTERN)
        close_time = get_market_close_time(date)
        assert close_time.hour == 16
        assert close_time.minute == 0

    def test_bars_to_timedelta(self):
        """Test converting bars to timedelta."""
        assert bars_to_timedelta(1, "15Min") == timedelta(minutes=15)
        assert bars_to_timedelta(4, "15Min") == timedelta(hours=1)
        assert bars_to_timedelta(2, "1Hour") == timedelta(hours=2)
        assert bars_to_timedelta(1, "1Day") == timedelta(days=1)

    def test_timedelta_to_bars(self):
        """Test converting timedelta to bars."""
        assert timedelta_to_bars(timedelta(minutes=15), "15Min") == 1
        assert timedelta_to_bars(timedelta(hours=1), "15Min") == 4
        assert timedelta_to_bars(timedelta(hours=2), "1Hour") == 2


class TestNumericUtilities:
    """Tests for numeric utilities."""

    def test_round_decimal(self):
        """Test decimal rounding."""
        assert round_decimal(Decimal("3.14159"), 2) == Decimal("3.14")
        assert round_decimal(Decimal("3.145"), 2) == Decimal("3.15")
        assert round_decimal(Decimal("3.144"), 2) == Decimal("3.14")

    def test_to_decimal(self):
        """Test conversion to Decimal."""
        assert to_decimal(3.14159, 2) == Decimal("3.14")
        assert to_decimal(100, 2) == Decimal("100.00")
        assert to_decimal("123.456", 2) == Decimal("123.46")
        assert to_decimal(Decimal("1.5"), 1) == Decimal("1.5")

    def test_calculate_return(self):
        """Test simple return calculation."""
        assert calculate_return(100, 110) == 0.1  # 10% gain
        assert calculate_return(100, 90) == -0.1  # 10% loss
        assert calculate_return(0, 100) == 0.0  # Division by zero protection

    def test_calculate_log_return(self):
        """Test log return calculation."""
        import math

        result = calculate_log_return(100, 110)
        expected = math.log(110 / 100)
        assert abs(result - expected) < 0.0001

        # Handle invalid prices
        assert calculate_log_return(0, 100) == 0.0
        assert calculate_log_return(100, 0) == 0.0

    def test_basis_points_to_decimal(self):
        """Test basis points conversion."""
        assert basis_points_to_decimal(100) == 0.01  # 100 bps = 1%
        assert basis_points_to_decimal(50) == 0.005  # 50 bps = 0.5%
        assert basis_points_to_decimal(10) == 0.001  # 10 bps = 0.1%

    def test_decimal_to_basis_points(self):
        """Test decimal to basis points conversion."""
        assert decimal_to_basis_points(0.01) == 100
        assert decimal_to_basis_points(0.005) == 50
        assert decimal_to_basis_points(0.001) == 10

    def test_clamp(self):
        """Test value clamping."""
        assert clamp(5, 0, 10) == 5  # Within range
        assert clamp(-5, 0, 10) == 0  # Below min
        assert clamp(15, 0, 10) == 10  # Above max

    def test_safe_divide(self):
        """Test safe division."""
        assert safe_divide(10, 2) == 5
        assert safe_divide(10, 0) == 0.0  # Default
        assert safe_divide(10, 0, default=-1) == -1


class TestStringUtilities:
    """Tests for string utilities."""

    def test_normalize_symbol(self):
        """Test symbol normalization."""
        assert normalize_symbol("aapl") == "AAPL"
        assert normalize_symbol("  MSFT  ") == "MSFT"
        assert normalize_symbol("googl") == "GOOGL"

    def test_generate_id(self):
        """Test ID generation."""
        id1 = generate_id()
        id2 = generate_id()
        assert id1 != id2
        assert len(id1) == 8

        prefixed = generate_id(prefix="ORD", length=6)
        assert prefixed.startswith("ORD_")
        assert len(prefixed) == 10  # ORD_ + 6 chars

    def test_hash_string(self):
        """Test string hashing."""
        hash1 = hash_string("test")
        hash2 = hash_string("test")
        hash3 = hash_string("different")

        assert hash1 == hash2  # Same input = same hash
        assert hash1 != hash3  # Different input = different hash
        assert len(hash1) == 64  # SHA-256 produces 64 hex chars


class TestCaching:
    """Tests for caching utilities."""

    def test_memoize(self):
        """Test memoization decorator."""
        call_count = [0]

        @memoize
        def expensive_func(x, y):
            call_count[0] += 1
            return x + y

        result1 = expensive_func(1, 2)
        result2 = expensive_func(1, 2)
        result3 = expensive_func(2, 3)

        assert result1 == 3
        assert result2 == 3
        assert result3 == 5
        assert call_count[0] == 2  # Only called twice (2 unique inputs)

    def test_memoize_clear_cache(self):
        """Test clearing memoized cache."""
        @memoize
        def func(x):
            return x * 2

        func(1)
        assert len(func.cache) == 1

        func.clear_cache()
        assert len(func.cache) == 0

    def test_timed_cache(self):
        """Test time-based caching."""
        call_count = [0]

        @timed_cache(seconds=1)
        def cached_func():
            call_count[0] += 1
            return call_count[0]

        result1 = cached_func()
        result2 = cached_func()

        assert result1 == 1
        assert result2 == 1  # Cached
        assert call_count[0] == 1

        # Wait for cache to expire
        time.sleep(1.1)

        result3 = cached_func()
        assert result3 == 2  # New call
        assert call_count[0] == 2


class TestRetryLogic:
    """Tests for retry utilities."""

    def test_retry_success_first_attempt(self):
        """Test retry when function succeeds immediately."""
        call_count = [0]

        @retry(max_attempts=3, delay=0.01)
        def succeeds():
            call_count[0] += 1
            return "success"

        result = succeeds()
        assert result == "success"
        assert call_count[0] == 1

    def test_retry_success_after_failures(self):
        """Test retry when function succeeds after failures."""
        attempts = [0]

        @retry(max_attempts=3, delay=0.01)
        def fails_twice():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("Not yet")
            return "success"

        result = fails_twice()
        assert result == "success"
        assert attempts[0] == 3

    def test_retry_all_attempts_fail(self):
        """Test retry when all attempts fail."""
        @retry(max_attempts=3, delay=0.01)
        def always_fails():
            raise RuntimeError("Always fails")

        with pytest.raises(RuntimeError):
            always_fails()

    def test_retry_specific_exceptions(self):
        """Test retry with specific exception types."""
        attempts = [0]

        @retry(max_attempts=3, delay=0.01, exceptions=(ValueError,))
        def raises_type_error():
            attempts[0] += 1
            raise TypeError("Not retried")

        with pytest.raises(TypeError):
            raises_type_error()

        assert attempts[0] == 1  # Not retried


class TestAsyncRetry:
    """Tests for async retry."""

    def test_retry_async_success(self):
        """Test async retry with success."""
        async def run_test():
            attempts = [0]

            async def succeeds():
                attempts[0] += 1
                return "async success"

            result = await retry_async(succeeds, max_attempts=3, delay=0.01)
            return result == "async success" and attempts[0] == 1

        assert asyncio.run(run_test())

    def test_retry_async_with_failures(self):
        """Test async retry with initial failures."""
        async def run_test():
            attempts = [0]

            async def fails_once():
                attempts[0] += 1
                if attempts[0] == 1:
                    raise ValueError("First attempt")
                return "success"

            result = await retry_async(fails_once, max_attempts=3, delay=0.01)
            return result == "success" and attempts[0] == 2

        assert asyncio.run(run_test())


class TestPerformanceTiming:
    """Tests for performance timing utilities."""

    def test_timer_context_manager(self):
        """Test timer context manager."""
        with timer("Test operation") as t:
            time.sleep(0.1)

        assert "elapsed" in t
        assert t["elapsed"] >= 0.1
        assert t["elapsed"] < 0.2

    def test_timed_decorator(self):
        """Test timed decorator."""
        @timed
        def slow_function():
            time.sleep(0.05)
            return "done"

        result = slow_function()
        assert result == "done"


class TestValidation:
    """Tests for validation utilities."""

    def test_validate_symbol_valid(self):
        """Test valid symbol validation."""
        assert validate_symbol("AAPL") is True
        assert validate_symbol("A") is True
        assert validate_symbol("GOOGL") is True
        assert validate_symbol("BRK.A") is False  # Contains dot

    def test_validate_symbol_invalid(self):
        """Test invalid symbol validation."""
        assert validate_symbol("") is False
        assert validate_symbol("   ") is False
        assert validate_symbol("12345") is False  # All numbers
        assert validate_symbol("VERYLONGSYMBOLNAME") is False  # Too long
        assert validate_symbol(None) is False

    def test_validate_quantity_valid(self):
        """Test valid quantity validation."""
        assert validate_quantity(100) is True
        assert validate_quantity(Decimal("50.5")) is True
        assert validate_quantity(0.001) is True

    def test_validate_quantity_invalid(self):
        """Test invalid quantity validation."""
        assert validate_quantity(0) is False
        assert validate_quantity(-100) is False
        assert validate_quantity("not a number") is False

    def test_validate_price_valid(self):
        """Test valid price validation."""
        assert validate_price(185.50) is True
        assert validate_price(Decimal("100.00")) is True
        assert validate_price(0) is True  # Zero is valid

    def test_validate_price_invalid(self):
        """Test invalid price validation."""
        assert validate_price(-10) is False
        assert validate_price("not a price") is False
