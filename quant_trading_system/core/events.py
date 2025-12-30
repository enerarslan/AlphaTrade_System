"""
Event system for message passing between components.

Implements an event-driven architecture with:
- Typed event definitions
- Async event dispatching
- Priority queues for critical events
- Dead letter queue for failed handlers
- Event persistence for replay
- Metrics on event processing latency
- THREAD-SAFE singleton pattern for EventBus
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Event type enumeration for all system events."""

    # Market Data Events
    BAR_UPDATE = "market.bar_update"
    QUOTE_UPDATE = "market.quote_update"
    TRADE_UPDATE = "market.trade_update"

    # Signal Events
    SIGNAL_GENERATED = "signal.generated"
    SIGNAL_EXPIRED = "signal.expired"
    SIGNAL_CANCELLED = "signal.cancelled"

    # Order Events
    ORDER_SUBMITTED = "order.submitted"
    ORDER_ACCEPTED = "order.accepted"
    ORDER_FILLED = "order.filled"
    ORDER_PARTIAL = "order.partial"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_REJECTED = "order.rejected"
    ORDER_EXPIRED = "order.expired"

    # Risk Events
    LIMIT_BREACH = "risk.limit_breach"
    DRAWDOWN_WARNING = "risk.drawdown_warning"
    EXPOSURE_WARNING = "risk.exposure_warning"
    KILL_SWITCH_TRIGGERED = "risk.kill_switch"

    # System Events
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    MODEL_RELOAD = "system.model_reload"
    CONFIG_RELOAD = "system.config_reload"
    HEARTBEAT = "system.heartbeat"

    # Portfolio Events
    POSITION_OPENED = "portfolio.position_opened"
    POSITION_CLOSED = "portfolio.position_closed"
    POSITION_UPDATED = "portfolio.position_updated"
    PORTFOLIO_REBALANCED = "portfolio.rebalanced"


class EventPriority(int, Enum):
    """Event priority levels for queue ordering."""

    CRITICAL = 0  # Risk events, kill switch
    HIGH = 1  # Order events, limit breaches
    NORMAL = 2  # Signals, market data
    LOW = 3  # System events, heartbeats


@dataclass
class Event:
    """Base event class for all system events.

    Attributes:
        event_id: Unique identifier for the event.
        event_type: Type of the event from EventType enum.
        timestamp: When the event was created.
        data: Event payload data.
        source: Component that generated the event.
        priority: Event priority for queue ordering.
        metadata: Additional event metadata.
    """

    event_type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    event_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    priority: EventPriority = EventPriority.NORMAL
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "Event") -> bool:
        """Compare events by priority for queue ordering."""
        if not isinstance(other, Event):
            return NotImplemented
        # Lower priority value = higher priority
        if self.priority != other.priority:
            return self.priority < other.priority
        # Same priority, earlier timestamp first
        return self.timestamp < other.timestamp

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source,
            "priority": self.priority.name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            event_type=EventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data.get("data", {}),
            source=data.get("source", ""),
            priority=EventPriority[data.get("priority", "NORMAL")],
            metadata=data.get("metadata", {}),
        )


# Type alias for event handlers
EventHandler = Callable[[Event], Awaitable[None] | None]


@dataclass
class DeadLetterEntry:
    """Entry in the dead letter queue for failed event handling."""

    event: Event
    handler_name: str
    error: Exception
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retry_count: int = 0


class EventBus:
    """Central event bus for message passing between components.

    SINGLETON PATTERN: Use get_event_bus() to get the global instance.

    Supports:
    - Synchronous and asynchronous event handlers
    - Priority-based event processing
    - Dead letter queue for failed handlers
    - Event metrics tracking
    - Event filtering by type
    - THREAD-SAFE operations with RLock
    """

    # Singleton instance and lock
    _instance: "EventBus | None" = None
    _instance_lock: threading.Lock = threading.Lock()

    def __new__(cls, max_queue_size: int = 10000) -> "EventBus":
        """Ensure singleton pattern with thread-safe instantiation."""
        if cls._instance is None:
            with cls._instance_lock:
                # Double-check locking pattern
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self, max_queue_size: int = 10000) -> None:
        """Initialize the event bus.

        Args:
            max_queue_size: Maximum size of the event queue.
        """
        # Prevent re-initialization of singleton
        if getattr(self, "_initialized", False):
            return

        # THREAD SAFETY: Use RLock for all mutable state
        self._lock = threading.RLock()

        self._handlers: dict[EventType, list[tuple[str, EventHandler, int]]] = defaultdict(list)
        self._queue: asyncio.PriorityQueue[tuple[int, float, Event]] = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._dead_letter_queue: list[DeadLetterEntry] = []
        self._running = False
        self._metrics: dict[str, Any] = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "total_latency_ms": 0.0,
            "handler_errors": defaultdict(int),
        }
        self._event_history: list[Event] = []
        self._max_history_size = 1000
        self._global_handlers: list[tuple[str, EventHandler, int]] = []
        self._initialized = True

    @classmethod
    def get_instance(cls, max_queue_size: int = 10000) -> "EventBus":
        """Get the singleton EventBus instance.

        Args:
            max_queue_size: Maximum queue size (only used on first call).

        Returns:
            The singleton EventBus instance.
        """
        return cls(max_queue_size)

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing only).

        WARNING: This should only be used in tests. Using this in production
        may cause lost events and coordination failures.
        """
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance._running = False
            cls._instance = None

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
        handler_name: str | None = None,
        priority: int = 0,
    ) -> None:
        """Subscribe a handler to an event type.

        THREAD-SAFE: Uses lock to protect handler registration.

        Args:
            event_type: Type of event to subscribe to.
            handler: Callback function to handle the event.
            handler_name: Optional name for the handler (for logging).
            priority: Handler priority (lower = called first).
        """
        name = handler_name or handler.__name__
        with self._lock:
            self._handlers[event_type].append((name, handler, priority))
            # Sort handlers by priority
            self._handlers[event_type].sort(key=lambda x: x[2])
        logger.debug(f"Handler '{name}' subscribed to {event_type.value}")

    def subscribe_all(
        self,
        handler: EventHandler,
        handler_name: str | None = None,
        priority: int = 0,
    ) -> None:
        """Subscribe a handler to all event types.

        THREAD-SAFE: Uses lock to protect handler registration.

        Args:
            handler: Callback function to handle events.
            handler_name: Optional name for the handler.
            priority: Handler priority.
        """
        name = handler_name or handler.__name__
        with self._lock:
            self._global_handlers.append((name, handler, priority))
            self._global_handlers.sort(key=lambda x: x[2])
        logger.debug(f"Global handler '{name}' subscribed to all events")

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> bool:
        """Unsubscribe a handler from an event type.

        THREAD-SAFE: Uses lock to protect handler removal.

        Args:
            event_type: Type of event to unsubscribe from.
            handler: Handler to remove.

        Returns:
            True if handler was found and removed, False otherwise.
        """
        with self._lock:
            handlers = self._handlers[event_type]
            for i, (name, h, _) in enumerate(handlers):
                if h == handler:
                    handlers.pop(i)
                    logger.debug(f"Handler '{name}' unsubscribed from {event_type.value}")
                    return True
        return False

    def publish(self, event: Event) -> None:
        """Publish an event to the bus (synchronous).

        THREAD-SAFE: Uses lock to protect metrics and history updates.

        Args:
            event: Event to publish.
        """
        with self._lock:
            self._metrics["events_published"] += 1
            self._add_to_history(event)
            # Get snapshot of handlers under lock
            handlers = list(self._handlers.get(event.event_type, [])) + list(self._global_handlers)

        for handler_name, handler, _ in handlers:
            try:
                result = handler(event)
                # If handler is async, we need special handling
                if asyncio.iscoroutine(result):
                    # Create task if event loop is running
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(self._run_async_handler(handler_name, handler, event))
                    except RuntimeError:
                        # No event loop running, run synchronously with new loop
                        asyncio.run(result)
                # RACE CONDITION FIX: Update metrics inside lock
                with self._lock:
                    self._metrics["events_processed"] += 1
            except Exception as e:
                self._handle_error(event, handler_name, e)

    async def publish_async(self, event: Event) -> None:
        """Publish an event to the bus (asynchronous).

        RACE CONDITION FIX: All metrics and history updates are now inside lock.

        Args:
            event: Event to publish.
        """
        with self._lock:
            self._metrics["events_published"] += 1
            self._add_to_history(event)
            # Get snapshot of handlers under lock
            handlers = list(self._handlers.get(event.event_type, [])) + list(self._global_handlers)

        for handler_name, handler, _ in handlers:
            start_time = time.time()
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
                latency_ms = (time.time() - start_time) * 1000
                # RACE CONDITION FIX: Update metrics inside lock
                with self._lock:
                    self._metrics["total_latency_ms"] += latency_ms
                    self._metrics["events_processed"] += 1
            except Exception as e:
                self._handle_error(event, handler_name, e)

    async def enqueue(self, event: Event) -> None:
        """Add an event to the priority queue for async processing.

        Args:
            event: Event to enqueue.
        """
        # Priority queue ordering: (priority, timestamp, event)
        await self._queue.put((event.priority.value, event.timestamp.timestamp(), event))
        # RACE CONDITION FIX: Update metrics inside lock
        with self._lock:
            self._metrics["events_published"] += 1

    async def _run_async_handler(
        self,
        handler_name: str,
        handler: EventHandler,
        event: Event,
    ) -> None:
        """Run an async handler and handle errors.

        RACE CONDITION FIX: Metrics updates are now inside lock.
        """
        start_time = time.time()
        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
            latency_ms = (time.time() - start_time) * 1000
            # RACE CONDITION FIX: Update metrics inside lock
            with self._lock:
                self._metrics["total_latency_ms"] += latency_ms
                self._metrics["events_processed"] += 1
        except Exception as e:
            self._handle_error(event, handler_name, e)

    def _handle_error(self, event: Event, handler_name: str, error: Exception) -> None:
        """Handle a handler error by logging and adding to dead letter queue.

        THREAD-SAFE: Uses lock to protect metrics and dead letter queue.
        """
        with self._lock:
            self._metrics["events_failed"] += 1
            self._metrics["handler_errors"][handler_name] += 1

            entry = DeadLetterEntry(
                event=event,
                handler_name=handler_name,
                error=error,
            )
            self._dead_letter_queue.append(entry)

        logger.error(
            f"Handler '{handler_name}' failed for event {event.event_type.value}: {error}",
            exc_info=True,
        )

    def _add_to_history(self, event: Event) -> None:
        """Add event to history, maintaining max size."""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size :]

    async def start_processing(self) -> None:
        """Start processing events from the queue."""
        self._running = True
        logger.info("Event bus started processing")

        while self._running:
            try:
                # Wait for event with timeout to allow checking _running flag
                try:
                    _, _, event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                await self.publish_async(event)
                self._queue.task_done()

            except Exception as e:
                logger.error(f"Error in event processing loop: {e}", exc_info=True)

    def stop_processing(self) -> None:
        """Stop processing events."""
        self._running = False
        logger.info("Event bus stopped processing")

    def get_metrics(self) -> dict[str, Any]:
        """Get event bus metrics.

        Returns:
            Dictionary with event bus metrics.
        """
        avg_latency = 0.0
        if self._metrics["events_processed"] > 0:
            avg_latency = self._metrics["total_latency_ms"] / self._metrics["events_processed"]

        return {
            "events_published": self._metrics["events_published"],
            "events_processed": self._metrics["events_processed"],
            "events_failed": self._metrics["events_failed"],
            "avg_latency_ms": avg_latency,
            "queue_size": self._queue.qsize(),
            "dead_letter_count": len(self._dead_letter_queue),
            "handler_errors": dict(self._metrics["handler_errors"]),
        }

    def get_dead_letter_queue(self) -> list[DeadLetterEntry]:
        """Get the dead letter queue entries.

        Returns:
            List of failed event entries.
        """
        return self._dead_letter_queue.copy()

    def clear_dead_letter_queue(self) -> int:
        """Clear the dead letter queue.

        Returns:
            Number of entries cleared.
        """
        count = len(self._dead_letter_queue)
        self._dead_letter_queue.clear()
        return count

    def get_event_history(
        self,
        event_type: EventType | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """Get event history, optionally filtered by type.

        Args:
            event_type: Optional type to filter by.
            limit: Maximum number of events to return.

        Returns:
            List of events from history.
        """
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if limit:
            events = events[-limit:]
        return events

    def get_handler_count(self, event_type: EventType | None = None) -> int:
        """Get the number of registered handlers.

        Args:
            event_type: Optional type to count handlers for.

        Returns:
            Number of handlers.
        """
        if event_type:
            return len(self._handlers.get(event_type, [])) + len(self._global_handlers)
        total = sum(len(handlers) for handlers in self._handlers.values())
        return total + len(self._global_handlers)


# Factory functions for common events
def create_bar_event(symbol: str, bar_data: dict[str, Any], source: str = "") -> Event:
    """Create a bar update event."""
    return Event(
        event_type=EventType.BAR_UPDATE,
        data={"symbol": symbol, "bar": bar_data},
        source=source,
        priority=EventPriority.NORMAL,
    )


def create_signal_event(
    symbol: str,
    signal_data: dict[str, Any],
    source: str = "",
) -> Event:
    """Create a signal generated event."""
    return Event(
        event_type=EventType.SIGNAL_GENERATED,
        data={"symbol": symbol, "signal": signal_data},
        source=source,
        priority=EventPriority.NORMAL,
    )


def create_order_event(
    event_type: EventType,
    order_data: dict[str, Any],
    source: str = "",
) -> Event:
    """Create an order event."""
    return Event(
        event_type=event_type,
        data={"order": order_data},
        source=source,
        priority=EventPriority.HIGH,
    )


def create_risk_event(
    event_type: EventType,
    risk_data: dict[str, Any],
    source: str = "",
) -> Event:
    """Create a risk event."""
    priority = EventPriority.CRITICAL if event_type == EventType.KILL_SWITCH_TRIGGERED else EventPriority.HIGH
    return Event(
        event_type=event_type,
        data=risk_data,
        source=source,
        priority=priority,
    )


def create_system_event(
    event_type: EventType,
    message: str,
    details: dict[str, Any] | None = None,
    source: str = "",
) -> Event:
    """Create a system event."""
    return Event(
        event_type=event_type,
        data={"message": message, "details": details or {}},
        source=source,
        priority=EventPriority.LOW,
    )


def get_event_bus() -> EventBus:
    """Get the global EventBus singleton instance.

    This is the recommended way to get the EventBus in application code.
    Using this ensures all components share the same event bus.

    Returns:
        The singleton EventBus instance.

    Example:
        >>> from quant_trading_system.core.events import get_event_bus, EventType
        >>> bus = get_event_bus()
        >>> bus.subscribe(EventType.ORDER_FILLED, my_handler)
    """
    return EventBus.get_instance()
