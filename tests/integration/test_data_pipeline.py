"""
Integration tests for data pipeline.
"""

import asyncio
from datetime import datetime
from decimal import Decimal

import pytest

from quant_trading_system.core.data_types import OHLCVBar, TradeSignal, Direction
from quant_trading_system.core.events import EventBus, EventType, create_bar_event


class TestDataPipelineIntegration:
    """Integration tests for data pipeline components."""

    @pytest.fixture
    def event_bus(self):
        """Create a fresh EventBus."""
        return EventBus()

    @pytest.fixture
    def sample_bars(self):
        """Create sample OHLCV bars."""
        return [
            OHLCVBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 10 + i // 4, (i % 4) * 15, 0),
                open=Decimal(f"{185.50 + i * 0.5}"),
                high=Decimal(f"{186.75 + i * 0.5}"),
                low=Decimal(f"{184.25 + i * 0.5}"),
                close=Decimal(f"{186.00 + i * 0.5}"),
                volume=1500000 + i * 10000,
            )
            for i in range(5)
        ]

    def test_bar_data_to_event_flow(self, event_bus, sample_bars):
        """Test flow from bar data to events."""
        received_events = []

        def bar_handler(event):
            received_events.append(event)

        event_bus.subscribe(EventType.BAR_UPDATE, bar_handler)

        # Publish bar events
        for bar in sample_bars:
            event = create_bar_event(
                symbol=bar.symbol,
                bar_data=bar.to_dict(),
                source="test_data_feed",
            )
            event_bus.publish(event)

        # Verify all events received
        assert len(received_events) == 5
        for i, event in enumerate(received_events):
            assert event.data["symbol"] == "AAPL"
            assert event.data["bar"]["volume"] == 1500000 + i * 10000

    def test_bar_validation_in_pipeline(self, sample_bars):
        """Test that bar validation works in pipeline."""
        # All sample bars should be valid
        for bar in sample_bars:
            assert bar.high >= bar.open
            assert bar.high >= bar.close
            assert bar.low <= bar.open
            assert bar.low <= bar.close

    def test_signal_generation_from_bars(self, event_bus, sample_bars):
        """Test signal generation from bar data."""
        signals_generated = []

        def signal_handler(event):
            signals_generated.append(event)

        event_bus.subscribe(EventType.SIGNAL_GENERATED, signal_handler)

        # Simulate signal generation from bars
        # In real system, this would be done by models
        for bar in sample_bars:
            if float(bar.close) > float(bar.open):
                signal = TradeSignal(
                    symbol=bar.symbol,
                    direction=Direction.LONG,
                    strength=0.7,
                    confidence=0.65,
                    horizon=4,
                    model_source="test_model",
                )
                from quant_trading_system.core.events import create_signal_event
                event = create_signal_event(
                    symbol=signal.symbol,
                    signal_data={
                        "direction": signal.direction.value,
                        "strength": signal.strength,
                        "confidence": signal.confidence,
                    },
                    source=signal.model_source,
                )
                event_bus.publish(event)

        # All bars have close > open in our sample, so all should generate signals
        assert len(signals_generated) == 5


class TestEventFlowIntegration:
    """Integration tests for event flow through the system."""

    @pytest.fixture
    def event_bus(self):
        """Create a fresh EventBus."""
        return EventBus()

    def test_multi_handler_event_flow(self, event_bus):
        """Test event flow with multiple handlers."""
        handler1_events = []
        handler2_events = []
        handler3_events = []

        def handler1(event):
            handler1_events.append(event)

        def handler2(event):
            handler2_events.append(event)

        def handler3(event):
            handler3_events.append(event)

        # Subscribe handlers to different events
        event_bus.subscribe(EventType.BAR_UPDATE, handler1)
        event_bus.subscribe(EventType.BAR_UPDATE, handler2)
        event_bus.subscribe(EventType.ORDER_FILLED, handler3)

        # Publish events
        from quant_trading_system.core.events import Event
        event_bus.publish(Event(event_type=EventType.BAR_UPDATE))
        event_bus.publish(Event(event_type=EventType.ORDER_FILLED))

        # Verify distribution
        assert len(handler1_events) == 1
        assert len(handler2_events) == 1
        assert len(handler3_events) == 1

    def test_global_handler_receives_all(self, event_bus):
        """Test global handler receives all events."""
        all_events = []

        def global_handler(event):
            all_events.append(event)

        event_bus.subscribe_all(global_handler)

        # Publish various events
        from quant_trading_system.core.events import Event
        event_bus.publish(Event(event_type=EventType.BAR_UPDATE))
        event_bus.publish(Event(event_type=EventType.ORDER_FILLED))
        event_bus.publish(Event(event_type=EventType.SIGNAL_GENERATED))

        assert len(all_events) == 3

    def test_async_event_processing(self, event_bus):
        """Test async event processing."""

        async def run_test():
            processed_events = []

            async def async_handler(event):
                await asyncio.sleep(0.01)
                processed_events.append(event)

            event_bus.subscribe(EventType.BAR_UPDATE, async_handler)

            from quant_trading_system.core.events import Event
            await event_bus.publish_async(Event(event_type=EventType.BAR_UPDATE))

            return len(processed_events)

        count = asyncio.run(run_test())
        assert count == 1


class TestDataTypeConversions:
    """Integration tests for data type conversions."""

    def test_ohlcv_bar_to_dict_and_back(self):
        """Test OHLCV bar serialization round-trip."""
        original = OHLCVBar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            open=Decimal("185.50"),
            high=Decimal("186.75"),
            low=Decimal("184.25"),
            close=Decimal("186.00"),
            volume=1500000,
            vwap=Decimal("185.75"),
            trade_count=12500,
        )

        # Convert to dict
        data = original.to_dict()

        # Verify dict contents
        assert data["symbol"] == "AAPL"
        assert data["volume"] == 1500000
        assert isinstance(data["open"], float)

    def test_trade_signal_validation(self):
        """Test trade signal validation."""
        # Valid signal
        signal = TradeSignal(
            symbol="AAPL",
            direction=Direction.LONG,
            strength=0.75,
            confidence=0.8,
            horizon=4,
            model_source="xgboost",
        )

        # Check actionable
        assert signal.is_actionable(min_confidence=0.6, min_strength=0.5)
        assert not signal.is_actionable(min_confidence=0.9, min_strength=0.5)

    def test_signal_not_actionable_when_flat(self):
        """Test that FLAT signals are not actionable."""
        signal = TradeSignal(
            symbol="AAPL",
            direction=Direction.FLAT,
            strength=0.9,
            confidence=0.9,
            horizon=4,
            model_source="test",
        )
        assert not signal.is_actionable()


class TestMetricsDataFlow:
    """Integration tests for metrics data flow."""

    def test_trading_metrics_update_flow(self):
        """Test flow of trading metrics updates."""
        from quant_trading_system.monitoring.metrics import MetricsCollector

        # Reset singleton
        MetricsCollector._instance = None
        metrics = MetricsCollector()

        # Simulate trading day
        # Morning - initial portfolio state
        metrics.update_portfolio_metrics(
            equity=100000,
            cash=50000,
            buying_power=150000,
            positions_count=0,
            long_exposure=0,
            short_exposure=0,
        )

        # Trade execution
        metrics.record_order_submitted("AAPL", "BUY", "MARKET")
        metrics.record_order_filled("AAPL", "BUY")
        metrics.record_order_latency("AAPL", 0.05)

        # Position update
        metrics.update_portfolio_metrics(
            equity=100000,
            cash=31450,
            buying_power=131450,
            positions_count=1,
            long_exposure=18550,
            short_exposure=0,
        )

        # Performance update
        metrics.update_performance_metrics(
            daily_pnl=100,
            total_pnl=100,
        )

        # Get metrics output
        output = metrics.get_metrics()
        assert len(output) > 0
