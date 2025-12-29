"""
Unit tests for core/events.py
"""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from quant_trading_system.core.events import (
    Event,
    EventBus,
    EventPriority,
    EventType,
    create_bar_event,
    create_order_event,
    create_risk_event,
    create_signal_event,
    create_system_event,
)


class TestEventType:
    """Tests for EventType enum."""

    def test_market_data_events(self):
        """Test market data event types."""
        assert EventType.BAR_UPDATE.value == "market.bar_update"
        assert EventType.QUOTE_UPDATE.value == "market.quote_update"
        assert EventType.TRADE_UPDATE.value == "market.trade_update"

    def test_signal_events(self):
        """Test signal event types."""
        assert EventType.SIGNAL_GENERATED.value == "signal.generated"
        assert EventType.SIGNAL_EXPIRED.value == "signal.expired"
        assert EventType.SIGNAL_CANCELLED.value == "signal.cancelled"

    def test_order_events(self):
        """Test order event types."""
        assert EventType.ORDER_SUBMITTED.value == "order.submitted"
        assert EventType.ORDER_FILLED.value == "order.filled"
        assert EventType.ORDER_CANCELLED.value == "order.cancelled"
        assert EventType.ORDER_REJECTED.value == "order.rejected"

    def test_risk_events(self):
        """Test risk event types."""
        assert EventType.LIMIT_BREACH.value == "risk.limit_breach"
        assert EventType.DRAWDOWN_WARNING.value == "risk.drawdown_warning"
        assert EventType.KILL_SWITCH_TRIGGERED.value == "risk.kill_switch"

    def test_system_events(self):
        """Test system event types."""
        assert EventType.SYSTEM_START.value == "system.start"
        assert EventType.SYSTEM_STOP.value == "system.stop"
        assert EventType.MODEL_RELOAD.value == "system.model_reload"


class TestEventPriority:
    """Tests for EventPriority enum."""

    def test_priority_ordering(self):
        """Test priority values are ordered correctly."""
        assert EventPriority.CRITICAL < EventPriority.HIGH
        assert EventPriority.HIGH < EventPriority.NORMAL
        assert EventPriority.NORMAL < EventPriority.LOW


class TestEvent:
    """Tests for Event class."""

    def test_event_creation(self):
        """Test basic event creation."""
        event = Event(
            event_type=EventType.BAR_UPDATE,
            data={"symbol": "AAPL", "price": 185.50},
            source="market_data",
        )
        assert event.event_type == EventType.BAR_UPDATE
        assert event.data["symbol"] == "AAPL"
        assert event.source == "market_data"
        assert event.priority == EventPriority.NORMAL
        assert event.event_id is not None
        assert event.timestamp is not None

    def test_event_priority_comparison(self):
        """Test event comparison by priority."""
        critical_event = Event(
            event_type=EventType.KILL_SWITCH_TRIGGERED,
            priority=EventPriority.CRITICAL,
        )
        normal_event = Event(
            event_type=EventType.BAR_UPDATE,
            priority=EventPriority.NORMAL,
        )
        # Critical should be "less than" (higher priority) normal
        assert critical_event < normal_event

    def test_event_to_dict(self):
        """Test event serialization to dictionary."""
        event = Event(
            event_type=EventType.ORDER_FILLED,
            data={"order_id": "123"},
            source="execution",
            priority=EventPriority.HIGH,
        )
        d = event.to_dict()
        assert d["event_type"] == "order.filled"
        assert d["data"]["order_id"] == "123"
        assert d["source"] == "execution"
        assert d["priority"] == "HIGH"
        assert "event_id" in d
        assert "timestamp" in d

    def test_event_from_dict(self):
        """Test event deserialization from dictionary."""
        data = {
            "event_id": "550e8400-e29b-41d4-a716-446655440000",
            "event_type": "signal.generated",
            "timestamp": "2024-01-15T10:30:00",
            "data": {"signal_strength": 0.8},
            "source": "model",
            "priority": "HIGH",
            "metadata": {},
        }
        event = Event.from_dict(data)
        assert event.event_type == EventType.SIGNAL_GENERATED
        assert event.data["signal_strength"] == 0.8
        assert event.priority == EventPriority.HIGH


class TestEventBus:
    """Tests for EventBus class."""

    def test_subscribe_and_publish(self, event_bus):
        """Test basic subscription and publishing."""
        received_events = []

        def handler(event):
            received_events.append(event)

        event_bus.subscribe(EventType.BAR_UPDATE, handler)
        event = Event(event_type=EventType.BAR_UPDATE, data={"test": True})
        event_bus.publish(event)

        assert len(received_events) == 1
        assert received_events[0].data["test"] is True

    def test_multiple_handlers(self, event_bus):
        """Test multiple handlers for same event type."""
        handler1_calls = []
        handler2_calls = []

        def handler1(event):
            handler1_calls.append(event)

        def handler2(event):
            handler2_calls.append(event)

        event_bus.subscribe(EventType.BAR_UPDATE, handler1)
        event_bus.subscribe(EventType.BAR_UPDATE, handler2)

        event = Event(event_type=EventType.BAR_UPDATE)
        event_bus.publish(event)

        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1

    def test_subscribe_all(self, event_bus):
        """Test subscribing to all event types."""
        received_events = []

        def global_handler(event):
            received_events.append(event)

        event_bus.subscribe_all(global_handler)

        event_bus.publish(Event(event_type=EventType.BAR_UPDATE))
        event_bus.publish(Event(event_type=EventType.ORDER_FILLED))

        assert len(received_events) == 2

    def test_unsubscribe(self, event_bus):
        """Test unsubscribing a handler."""
        received = []

        def handler(event):
            received.append(event)

        event_bus.subscribe(EventType.BAR_UPDATE, handler)
        event_bus.publish(Event(event_type=EventType.BAR_UPDATE))
        assert len(received) == 1

        result = event_bus.unsubscribe(EventType.BAR_UPDATE, handler)
        assert result is True

        event_bus.publish(Event(event_type=EventType.BAR_UPDATE))
        assert len(received) == 1  # No new events

    def test_unsubscribe_nonexistent(self, event_bus):
        """Test unsubscribing a handler that doesn't exist."""
        def handler(event):
            pass

        result = event_bus.unsubscribe(EventType.BAR_UPDATE, handler)
        assert result is False

    def test_handler_priority(self, event_bus):
        """Test handler execution order by priority."""
        call_order = []

        def handler_low(event):
            call_order.append("low")

        def handler_high(event):
            call_order.append("high")

        # Subscribe in reverse priority order
        event_bus.subscribe(EventType.BAR_UPDATE, handler_low, priority=10)
        event_bus.subscribe(EventType.BAR_UPDATE, handler_high, priority=1)

        event_bus.publish(Event(event_type=EventType.BAR_UPDATE))

        # High priority should be called first
        assert call_order == ["high", "low"]

    def test_handler_error_handling(self, event_bus):
        """Test that handler errors don't stop other handlers."""
        good_handler_called = []

        def bad_handler(event):
            raise ValueError("Handler error")

        def good_handler(event):
            good_handler_called.append(True)

        event_bus.subscribe(EventType.BAR_UPDATE, bad_handler)
        event_bus.subscribe(EventType.BAR_UPDATE, good_handler)

        event_bus.publish(Event(event_type=EventType.BAR_UPDATE))

        # Good handler should still be called
        assert len(good_handler_called) == 1

    def test_dead_letter_queue(self, event_bus):
        """Test that failed events go to dead letter queue."""
        def bad_handler(event):
            raise RuntimeError("Intentional error")

        event_bus.subscribe(EventType.BAR_UPDATE, bad_handler, handler_name="bad")
        event_bus.publish(Event(event_type=EventType.BAR_UPDATE))

        dlq = event_bus.get_dead_letter_queue()
        assert len(dlq) == 1
        assert dlq[0].handler_name == "bad"
        assert isinstance(dlq[0].error, RuntimeError)

    def test_clear_dead_letter_queue(self, event_bus):
        """Test clearing the dead letter queue."""
        def bad_handler(event):
            raise RuntimeError()

        event_bus.subscribe(EventType.BAR_UPDATE, bad_handler)
        event_bus.publish(Event(event_type=EventType.BAR_UPDATE))

        count = event_bus.clear_dead_letter_queue()
        assert count == 1
        assert len(event_bus.get_dead_letter_queue()) == 0

    def test_get_metrics(self, event_bus):
        """Test event bus metrics."""
        def handler(event):
            pass

        event_bus.subscribe(EventType.BAR_UPDATE, handler)
        event_bus.publish(Event(event_type=EventType.BAR_UPDATE))
        event_bus.publish(Event(event_type=EventType.BAR_UPDATE))

        metrics = event_bus.get_metrics()
        assert metrics["events_published"] == 2
        assert metrics["events_processed"] == 2
        assert metrics["events_failed"] == 0

    def test_get_event_history(self, event_bus):
        """Test retrieving event history."""
        event_bus.publish(Event(event_type=EventType.BAR_UPDATE, data={"n": 1}))
        event_bus.publish(Event(event_type=EventType.ORDER_FILLED, data={"n": 2}))
        event_bus.publish(Event(event_type=EventType.BAR_UPDATE, data={"n": 3}))

        # All events
        history = event_bus.get_event_history()
        assert len(history) == 3

        # Filter by type
        bar_history = event_bus.get_event_history(event_type=EventType.BAR_UPDATE)
        assert len(bar_history) == 2

        # Limit
        limited = event_bus.get_event_history(limit=2)
        assert len(limited) == 2

    def test_get_handler_count(self, event_bus):
        """Test getting handler count."""
        def handler(event):
            pass

        event_bus.subscribe(EventType.BAR_UPDATE, handler)
        event_bus.subscribe(EventType.BAR_UPDATE, handler)
        event_bus.subscribe(EventType.ORDER_FILLED, handler)

        assert event_bus.get_handler_count(EventType.BAR_UPDATE) == 2
        assert event_bus.get_handler_count(EventType.ORDER_FILLED) == 1
        assert event_bus.get_handler_count() >= 3


class TestEventBusAsync:
    """Async tests for EventBus."""

    def test_publish_async(self):
        """Test async event publishing."""
        async def run_test():
            event_bus = EventBus()
            received = []

            async def async_handler(event):
                await asyncio.sleep(0.01)
                received.append(event)

            event_bus.subscribe(EventType.BAR_UPDATE, async_handler)
            await event_bus.publish_async(Event(event_type=EventType.BAR_UPDATE))

            return len(received) == 1

        result = asyncio.run(run_test())
        assert result

    def test_enqueue_and_process(self):
        """Test event queue processing."""
        async def run_test():
            event_bus = EventBus()
            received = []

            def handler(event):
                received.append(event)

            event_bus.subscribe(EventType.BAR_UPDATE, handler)

            # Enqueue events
            await event_bus.enqueue(Event(event_type=EventType.BAR_UPDATE, data={"n": 1}))
            await event_bus.enqueue(Event(event_type=EventType.BAR_UPDATE, data={"n": 2}))

            # Start processing in background
            process_task = asyncio.create_task(event_bus.start_processing())

            # Wait a bit for processing
            await asyncio.sleep(0.1)

            # Stop processing
            event_bus.stop_processing()
            process_task.cancel()

            try:
                await process_task
            except asyncio.CancelledError:
                pass

            return len(received) >= 1

        result = asyncio.run(run_test())
        assert result


class TestEventFactories:
    """Tests for event factory functions."""

    def test_create_bar_event(self):
        """Test creating bar update event."""
        event = create_bar_event(
            symbol="AAPL",
            bar_data={"open": 185.0, "close": 186.0},
            source="alpaca",
        )
        assert event.event_type == EventType.BAR_UPDATE
        assert event.data["symbol"] == "AAPL"
        assert event.data["bar"]["open"] == 185.0
        assert event.source == "alpaca"
        assert event.priority == EventPriority.NORMAL

    def test_create_signal_event(self):
        """Test creating signal event."""
        event = create_signal_event(
            symbol="MSFT",
            signal_data={"direction": "LONG", "strength": 0.8},
            source="xgboost",
        )
        assert event.event_type == EventType.SIGNAL_GENERATED
        assert event.data["symbol"] == "MSFT"
        assert event.data["signal"]["direction"] == "LONG"

    def test_create_order_event(self):
        """Test creating order event."""
        event = create_order_event(
            event_type=EventType.ORDER_FILLED,
            order_data={"order_id": "123", "symbol": "AAPL"},
            source="execution",
        )
        assert event.event_type == EventType.ORDER_FILLED
        assert event.priority == EventPriority.HIGH

    def test_create_risk_event(self):
        """Test creating risk event."""
        # Kill switch has CRITICAL priority
        kill_event = create_risk_event(
            event_type=EventType.KILL_SWITCH_TRIGGERED,
            risk_data={"reason": "max_drawdown"},
            source="risk_manager",
        )
        assert kill_event.priority == EventPriority.CRITICAL

        # Other risk events have HIGH priority
        warning_event = create_risk_event(
            event_type=EventType.DRAWDOWN_WARNING,
            risk_data={"current": 0.08},
            source="risk_manager",
        )
        assert warning_event.priority == EventPriority.HIGH

    def test_create_system_event(self):
        """Test creating system event."""
        event = create_system_event(
            event_type=EventType.SYSTEM_START,
            message="System initialized",
            details={"version": "1.0.0"},
            source="main",
        )
        assert event.event_type == EventType.SYSTEM_START
        assert event.data["message"] == "System initialized"
        assert event.data["details"]["version"] == "1.0.0"
        assert event.priority == EventPriority.LOW
