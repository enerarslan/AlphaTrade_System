"""
Circuit Breaker Pattern for External API Resilience.

P2-M5 Enhancement: Comprehensive circuit breaker implementation for preventing
cascading failures when external APIs become unavailable or slow.

Note: This is the primary circuit breaker implementation. A simpler version
exists in core/utils.py for basic use cases. This module provides:
- Configurable failure/success thresholds
- Half-open state strategies
- Centralized registry
- Decorator for easy integration
- Comprehensive statistics tracking

Key concepts:
- CLOSED: Normal operation, requests pass through
- OPEN: Failures exceeded threshold, requests fail immediately
- HALF_OPEN: Testing if service recovered, allow limited requests

Benefits:
- Prevents cascading failures
- Fails fast when service is down
- Automatic recovery when service returns
- Reduces load on struggling services

Based on:
- Martin Fowler's Circuit Breaker pattern
- Netflix Hystrix implementation

Author: AlphaTrade System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    # Failure thresholds
    failure_threshold: int = 5  # Failures to trigger OPEN
    success_threshold: int = 3  # Successes in HALF_OPEN to close

    # Timing
    open_timeout_seconds: float = 30.0  # Time to stay OPEN before HALF_OPEN
    half_open_max_calls: int = 3  # Max calls allowed in HALF_OPEN

    # Slow call handling
    slow_call_duration_seconds: float = 5.0  # Duration to consider "slow"
    slow_call_threshold: int = 3  # Slow calls to count as failure

    # Recording window
    window_size_seconds: float = 60.0  # Window for counting failures

    # Fallback
    fallback_value: Any = None  # Value to return when circuit is OPEN


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""

    state: CircuitState
    total_calls: int
    successful_calls: int
    failed_calls: int
    slow_calls: int
    rejected_calls: int  # Calls rejected due to OPEN state
    last_failure_time: datetime | None
    last_success_time: datetime | None
    state_change_time: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "slow_calls": self.slow_calls,
            "rejected_calls": self.rejected_calls,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "state_change_time": self.state_change_time.isoformat(),
            "failure_rate": self.failed_calls / self.total_calls if self.total_calls > 0 else 0.0,
        }


class CircuitOpenError(Exception):
    """Exception raised when circuit is open."""

    def __init__(self, circuit_name: str, remaining_seconds: float):
        self.circuit_name = circuit_name
        self.remaining_seconds = remaining_seconds
        super().__init__(
            f"Circuit '{circuit_name}' is OPEN. "
            f"Retry after {remaining_seconds:.1f} seconds."
        )


class CircuitBreaker:
    """
    P2-M5 Enhancement: Circuit Breaker for External API Resilience.

    Wraps external API calls to prevent cascading failures when services
    become unavailable. Implements the standard circuit breaker pattern
    with CLOSED, OPEN, and HALF_OPEN states.

    State transitions:
    - CLOSED -> OPEN: When failure_threshold is exceeded
    - OPEN -> HALF_OPEN: After open_timeout_seconds
    - HALF_OPEN -> CLOSED: When success_threshold is reached
    - HALF_OPEN -> OPEN: On any failure

    Example:
        >>> breaker = CircuitBreaker("news_api", config)
        >>> try:
        ...     result = await breaker.call(async_api_function, *args)
        ... except CircuitOpenError:
        ...     # Use fallback
        ...     result = get_cached_data()

    Thread-safe for use in async and sync contexts.
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        on_state_change: Callable[[str, CircuitState, CircuitState], None] | None = None,
    ):
        """Initialize Circuit Breaker.

        Args:
            name: Identifier for this circuit (e.g., "news_api").
            config: Circuit breaker configuration.
            on_state_change: Callback when state changes (name, old_state, new_state).
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._on_state_change = on_state_change

        # State
        self._state = CircuitState.CLOSED
        self._state_change_time = datetime.now(timezone.utc)

        # Counters (within current window)
        self._failure_count = 0
        self._success_count = 0
        self._slow_call_count = 0
        self._half_open_calls = 0

        # Lifetime counters
        self._total_calls = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_slow = 0
        self._rejected_calls = 0

        # Timestamps
        self._last_failure_time: datetime | None = None
        self._last_success_time: datetime | None = None
        self._window_start = datetime.now(timezone.utc)

        # Thread safety
        self._lock = threading.RLock()

        logger.info(
            f"CircuitBreaker '{name}' initialized: "
            f"failure_threshold={self.config.failure_threshold}, "
            f"open_timeout={self.config.open_timeout_seconds}s"
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state_timeout()
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self.state == CircuitState.OPEN

    def _check_state_timeout(self) -> None:
        """Check if OPEN state has timed out and should transition to HALF_OPEN."""
        if self._state == CircuitState.OPEN:
            elapsed = (datetime.now(timezone.utc) - self._state_change_time).total_seconds()
            if elapsed >= self.config.open_timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self._state_change_time = datetime.now(timezone.utc)

        # Reset counters for new state
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        elif new_state == CircuitState.CLOSED:
            self._reset_window()

        logger.info(f"CircuitBreaker '{self.name}': {old_state.value} -> {new_state.value}")

        # Notify callback
        if self._on_state_change:
            try:
                self._on_state_change(self.name, old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

    def _reset_window(self) -> None:
        """Reset the failure counting window."""
        self._failure_count = 0
        self._success_count = 0
        self._slow_call_count = 0
        self._window_start = datetime.now(timezone.utc)

    def _check_window_expiry(self) -> None:
        """Check if the counting window has expired and reset if needed."""
        elapsed = (datetime.now(timezone.utc) - self._window_start).total_seconds()
        if elapsed >= self.config.window_size_seconds:
            self._reset_window()

    def _record_success(self, duration: float) -> None:
        """Record a successful call."""
        with self._lock:
            self._check_window_expiry()

            self._total_calls += 1
            self._total_successes += 1
            self._success_count += 1
            self._last_success_time = datetime.now(timezone.utc)

            # Check if call was slow
            if duration >= self.config.slow_call_duration_seconds:
                self._slow_call_count += 1
                self._total_slow += 1

            # Handle HALF_OPEN state
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        with self._lock:
            self._check_window_expiry()

            self._total_calls += 1
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = datetime.now(timezone.utc)

            logger.warning(f"CircuitBreaker '{self.name}' recorded failure: {error}")

            # Handle state transitions
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN -> OPEN
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                # Check if threshold exceeded
                total_failures = self._failure_count + (
                    self._slow_call_count if self._slow_call_count >= self.config.slow_call_threshold else 0
                )
                if total_failures >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _can_execute(self) -> bool:
        """Check if a call can be executed."""
        with self._lock:
            self._check_state_timeout()

            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                return False
            elif self._state == CircuitState.HALF_OPEN:
                # Allow limited calls in HALF_OPEN
                return self._half_open_calls < self.config.half_open_max_calls

            return False

    def _get_remaining_open_time(self) -> float:
        """Get remaining time in OPEN state."""
        elapsed = (datetime.now(timezone.utc) - self._state_change_time).total_seconds()
        remaining = self.config.open_timeout_seconds - elapsed
        return max(0.0, remaining)

    async def call_async(
        self,
        func: Callable[..., Any],
        *args: Any,
        fallback: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute async function with circuit breaker protection.

        Args:
            func: Async function to call.
            *args: Positional arguments for func.
            fallback: Optional fallback function if circuit is open.
            **kwargs: Keyword arguments for func.

        Returns:
            Result of func or fallback.

        Raises:
            CircuitOpenError: If circuit is OPEN and no fallback provided.
        """
        with self._lock:
            if not self._can_execute():
                self._rejected_calls += 1
                remaining = self._get_remaining_open_time()

                if fallback is not None:
                    logger.debug(f"Circuit '{self.name}' OPEN, using fallback")
                    return fallback(*args, **kwargs)
                elif self.config.fallback_value is not None:
                    return self.config.fallback_value
                else:
                    raise CircuitOpenError(self.name, remaining)

        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            duration = time.time() - start_time
            self._record_success(duration)
            return result

        except Exception as e:
            self._record_failure(e)
            raise

    def call(
        self,
        func: Callable[..., T],
        *args: Any,
        fallback: Callable[..., T] | None = None,
        **kwargs: Any,
    ) -> T:
        """Execute sync function with circuit breaker protection.

        Args:
            func: Function to call.
            *args: Positional arguments for func.
            fallback: Optional fallback function if circuit is open.
            **kwargs: Keyword arguments for func.

        Returns:
            Result of func or fallback.

        Raises:
            CircuitOpenError: If circuit is OPEN and no fallback provided.
        """
        with self._lock:
            if not self._can_execute():
                self._rejected_calls += 1
                remaining = self._get_remaining_open_time()

                if fallback is not None:
                    logger.debug(f"Circuit '{self.name}' OPEN, using fallback")
                    return fallback(*args, **kwargs)
                elif self.config.fallback_value is not None:
                    return self.config.fallback_value
                else:
                    raise CircuitOpenError(self.name, remaining)

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            self._record_success(duration)
            return result

        except Exception as e:
            self._record_failure(e)
            raise

    def force_open(self) -> None:
        """Force circuit to OPEN state (for testing or manual intervention)."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)
            logger.warning(f"CircuitBreaker '{self.name}' forced OPEN")

    def force_close(self) -> None:
        """Force circuit to CLOSED state (for recovery)."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            logger.info(f"CircuitBreaker '{self.name}' forced CLOSED")

    def reset(self) -> None:
        """Reset circuit to initial state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._state_change_time = datetime.now(timezone.utc)
            self._reset_window()
            self._half_open_calls = 0
            self._rejected_calls = 0
            logger.info(f"CircuitBreaker '{self.name}' reset")

    def get_stats(self) -> CircuitStats:
        """Get circuit breaker statistics."""
        with self._lock:
            self._check_state_timeout()
            return CircuitStats(
                state=self._state,
                total_calls=self._total_calls,
                successful_calls=self._total_successes,
                failed_calls=self._total_failures,
                slow_calls=self._total_slow,
                rejected_calls=self._rejected_calls,
                last_failure_time=self._last_failure_time,
                last_success_time=self._last_success_time,
                state_change_time=self._state_change_time,
            )


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized access to all circuit breakers in the system.
    """

    _instance: "CircuitBreakerRegistry | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "CircuitBreakerRegistry":
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._breakers = {}
                cls._instance._registry_lock = threading.RLock()
            return cls._instance

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one.

        Args:
            name: Circuit breaker name.
            config: Configuration (used only if creating new).

        Returns:
            CircuitBreaker instance.
        """
        with self._registry_lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """Get circuit breaker by name.

        Args:
            name: Circuit breaker name.

        Returns:
            CircuitBreaker or None if not found.
        """
        with self._registry_lock:
            return self._breakers.get(name)

    def get_all_stats(self) -> dict[str, CircuitStats]:
        """Get stats for all circuit breakers.

        Returns:
            Dictionary of name -> stats.
        """
        with self._registry_lock:
            return {
                name: breaker.get_stats()
                for name, breaker in self._breakers.items()
            }

    def get_all_breakers(self) -> dict[str, CircuitBreaker]:
        """Get all registered circuit breakers.

        Returns:
            Dictionary of name -> CircuitBreaker.
        """
        with self._registry_lock:
            return dict(self._breakers)

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._registry_lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info(f"Reset {len(self._breakers)} circuit breakers")


def circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
    fallback: Callable[..., Any] | None = None,
) -> Callable:
    """Decorator to wrap function with circuit breaker.

    Args:
        name: Circuit breaker name.
        config: Circuit breaker configuration.
        fallback: Optional fallback function.

    Returns:
        Decorated function.

    Example:
        >>> @circuit_breaker("news_api", fallback=get_cached_news)
        ... async def fetch_news(symbol: str) -> dict:
        ...     return await external_api.get_news(symbol)
    """
    registry = CircuitBreakerRegistry()
    breaker = registry.get_or_create(name, config)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await breaker.call_async(func, *args, fallback=fallback, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return breaker.call(func, *args, fallback=fallback, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def get_circuit_breaker(name: str) -> CircuitBreaker | None:
    """Get circuit breaker from global registry.

    Args:
        name: Circuit breaker name.

    Returns:
        CircuitBreaker or None.
    """
    return CircuitBreakerRegistry().get(name)


def get_all_circuit_stats() -> dict[str, dict[str, Any]]:
    """Get stats for all circuit breakers.

    Returns:
        Dictionary of stats.
    """
    stats = CircuitBreakerRegistry().get_all_stats()
    return {name: s.to_dict() for name, s in stats.items()}
