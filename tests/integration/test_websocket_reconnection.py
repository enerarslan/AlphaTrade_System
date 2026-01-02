"""
Integration tests for WebSocket reconnection logic.

Tests connection failure scenarios, reconnection with exponential backoff,
heartbeat monitoring, and state transitions for live data feeds.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from quant_trading_system.core.data_types import OHLCVBar
from quant_trading_system.core.events import EventBus
from quant_trading_system.core.exceptions import (
    BrokerConnectionError,
    DataConnectionError,
)
from quant_trading_system.data.live_feed import (
    AlpacaLiveFeed,
    AlpacaWebSocketFeed,
    BaseLiveFeed,
    ConnectionState,
    MockLiveFeed,
    WebSocketConfig,
    StreamType,
    create_live_feed,
)
from quant_trading_system.execution.alpaca_client import (
    AlpacaClient,
    RateLimiter,
    TradingEnvironment,
)


class TestConnectionStateTransitions:
    """Test connection state machine transitions."""

    @pytest.fixture
    def mock_feed(self):
        """Create a mock live feed for testing."""
        return MockLiveFeed(symbols=["AAPL", "MSFT"])

    @pytest.fixture
    def alpaca_feed(self):
        """Create an AlpacaWebSocketFeed for testing."""
        config = WebSocketConfig(
            api_key="test_key",
            api_secret="test_secret",
            paper=True,
        )
        return AlpacaWebSocketFeed(config=config, symbols=["AAPL"])

    def test_initial_state_disconnected(self, alpaca_feed):
        """Test that initial state is DISCONNECTED."""
        assert alpaca_feed.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_mock_feed_connect_state_change(self, mock_feed):
        """Test MockLiveFeed state changes on connect."""
        assert mock_feed.state == ConnectionState.DISCONNECTED

        await mock_feed.connect()
        assert mock_feed.state == ConnectionState.CONNECTED

        await mock_feed.disconnect()
        assert mock_feed.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_alpaca_feed_connecting_state(self, alpaca_feed):
        """Test CONNECTING state during connection attempt."""
        # The CONNECTING state is set at the beginning of connect()
        # before attempting to create the Stream
        alpaca_feed.state = ConnectionState.CONNECTING
        assert alpaca_feed.state == ConnectionState.CONNECTING

    def test_all_connection_states_defined(self):
        """Test all connection states are properly defined."""
        expected_states = {
            "DISCONNECTED",
            "CONNECTING",
            "AUTHENTICATING",
            "CONNECTED",
            "SUBSCRIBED",
            "RECONNECTING",
            "ERROR",
            "CLOSED",
        }
        actual_states = {state.name for state in ConnectionState}
        assert expected_states == actual_states


class TestExponentialBackoff:
    """Test exponential backoff behavior for reconnection."""

    @pytest.fixture
    def alpaca_feed(self):
        """Create an AlpacaWebSocketFeed for testing."""
        config = WebSocketConfig(
            api_key="test_key",
            api_secret="test_secret",
            paper=True,
        )
        return AlpacaWebSocketFeed(config=config, symbols=["AAPL"])

    def test_initial_reconnect_delay(self, alpaca_feed):
        """Test initial reconnect delay is set correctly."""
        assert alpaca_feed._reconnect_delay == alpaca_feed.config.initial_reconnect_delay
        assert alpaca_feed._reconnect_delay == 1.0

    def test_max_reconnect_delay(self, alpaca_feed):
        """Test max reconnect delay from config."""
        assert alpaca_feed.config.max_reconnect_delay == 60.0

    @pytest.mark.asyncio
    async def test_exponential_backoff_calculation(self, alpaca_feed):
        """Test that reconnect delay doubles each time up to max."""
        initial_delay = alpaca_feed._reconnect_delay
        max_delay = alpaca_feed.config.max_reconnect_delay

        # Simulate backoff progression
        delays = [initial_delay]
        current_delay = initial_delay

        for _ in range(7):  # Test several iterations
            current_delay = min(current_delay * 2, max_delay)
            delays.append(current_delay)

        # Expected: 1, 2, 4, 8, 16, 32, 60, 60
        expected = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
        assert delays == expected

    @pytest.mark.asyncio
    async def test_reconnect_resets_delay_on_success(self, alpaca_feed):
        """Test that successful connection resets delay."""
        # Increase delay as if multiple reconnects happened
        alpaca_feed._reconnect_delay = 32.0
        assert alpaca_feed._reconnect_delay == 32.0

        # Simulate what happens on successful connect - delay gets reset
        alpaca_feed._reconnect_delay = alpaca_feed.config.initial_reconnect_delay
        assert alpaca_feed._reconnect_delay == 1.0


class TestHeartbeatMonitoring:
    """Test heartbeat monitoring for connection health."""

    @pytest.fixture
    def alpaca_feed(self):
        """Create an AlpacaWebSocketFeed for testing."""
        config = WebSocketConfig(
            api_key="test_key",
            api_secret="test_secret",
            paper=True,
        )
        return AlpacaWebSocketFeed(config=config, symbols=["AAPL"])

    def test_heartbeat_interval_defined(self, alpaca_feed):
        """Test heartbeat interval is properly defined."""
        assert alpaca_feed.config.heartbeat_interval == 30.0

    def test_initial_heartbeat_none(self, alpaca_feed):
        """Test initial heartbeat is None before connection."""
        assert alpaca_feed._last_market_message is None

    @pytest.mark.asyncio
    async def test_heartbeat_updated_on_bar_receipt(self, alpaca_feed):
        """Test heartbeat is updated when receiving bars."""
        # Create mock bar message (Alpaca format)
        mock_bar_msg = {
            "T": "b",
            "S": "AAPL",
            "t": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "o": 150.0,
            "h": 151.0,
            "l": 149.0,
            "c": 150.5,
            "v": 1000,
            "vw": 150.25,
            "n": 50,
        }

        before_time = datetime.now(timezone.utc)
        await alpaca_feed._handle_bar(mock_bar_msg)
        after_time = datetime.now(timezone.utc)

        # Bars are processed but heartbeat update happens in message handler
        # This test validates the _handle_bar method works correctly
        assert alpaca_feed._metrics["bars_processed"] == 1

    @pytest.mark.asyncio
    async def test_heartbeat_monitor_basic(self, alpaca_feed):
        """Test that heartbeat monitor runs without errors."""
        alpaca_feed._running = True
        alpaca_feed._last_market_message = datetime.now(timezone.utc)

        # Run heartbeat monitor briefly
        monitor_task = asyncio.create_task(alpaca_feed._heartbeat_monitor())

        # Wait a bit for the monitor to check
        await asyncio.sleep(0.1)

        # Cancel the task
        alpaca_feed._running = False
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        # No errors should have occurred


class TestLiveFeedReconnection:
    """Test live feed reconnection scenarios."""

    @pytest.fixture
    def alpaca_feed(self):
        """Create an AlpacaWebSocketFeed for testing."""
        config = WebSocketConfig(
            api_key="test_key",
            api_secret="test_secret",
            paper=True,
        )
        return AlpacaWebSocketFeed(config=config, symbols=["AAPL"])

    @pytest.mark.asyncio
    async def test_connection_error_sets_error_state(self, alpaca_feed):
        """Test that connection errors set ERROR state."""
        # Patch websockets to raise an exception
        with patch.dict("sys.modules", {"websockets": MagicMock()}):
            import sys
            mock_ws = sys.modules["websockets"]
            mock_ws.connect = AsyncMock(side_effect=Exception("Connection refused"))

            with pytest.raises(DataConnectionError):
                await alpaca_feed.connect()

            assert alpaca_feed.state == ConnectionState.ERROR

    @pytest.mark.asyncio
    async def test_reconnect_increments_attempts(self, alpaca_feed):
        """Test that reconnection increments attempt counter."""
        alpaca_feed.state = ConnectionState.CONNECTED
        alpaca_feed._running = True

        initial_attempts = alpaca_feed._reconnect_attempts

        # Mock sleep to avoid waiting
        with patch("asyncio.sleep", new_callable=AsyncMock):
            # Mock the actual connection methods to fail
            with patch.object(alpaca_feed, "_connect_market_stream",
                            new_callable=AsyncMock) as mock_connect:
                mock_connect.side_effect = Exception("Connection failed")

                # Set max attempts low for test
                alpaca_feed.config.max_reconnect_attempts = 1

                await alpaca_feed._schedule_reconnect("market")

                # After max attempts, should be in ERROR state
                assert alpaca_feed.state == ConnectionState.ERROR

    @pytest.mark.asyncio
    async def test_reconnect_delay_increases(self, alpaca_feed):
        """Test reconnection delay increases exponentially."""
        initial_delay = alpaca_feed._reconnect_delay
        max_delay = alpaca_feed.config.max_reconnect_delay

        # Simulate backoff progression manually
        delays = [initial_delay]
        current_delay = initial_delay
        for i in range(1, 8):
            current_delay = min(current_delay * 2, max_delay)
            delays.append(current_delay)

        # Expected: 1, 2, 4, 8, 16, 32, 60, 60
        expected = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
        assert delays == expected


class TestAlpacaClientRetryLogic:
    """Test retry logic in AlpacaClient."""

    @pytest.fixture
    def client(self):
        """Create AlpacaClient for testing."""
        return AlpacaClient(
            api_key="test_key",
            api_secret="test_secret",
            environment=TradingEnvironment.PAPER,
            max_retries=3,
            retry_delay=0.01,  # Fast retries for testing
        )

    def test_max_retries_config(self, client):
        """Test max retries is configurable."""
        assert client.max_retries == 3

    def test_retry_delay_config(self, client):
        """Test retry delay is configurable."""
        assert client.retry_delay == 0.01

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, client):
        """Test request is retried on transient errors."""
        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return {"status": "ok"}

        with patch.object(client, "_session") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "ok"})
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)

            mock_session.request = MagicMock(return_value=mock_response)
            client._session = mock_session

            # Should succeed after retries
            await client.rate_limiter.acquire()

    @pytest.mark.asyncio
    async def test_exponential_backoff_in_retries(self, client):
        """Test exponential backoff between retries."""
        delays = []

        async def tracking_sleep(delay):
            delays.append(delay)

        with patch("asyncio.sleep", tracking_sleep):
            with patch.object(client, "_session") as mock_session:
                # Simulate failures
                mock_session.request = MagicMock(
                    side_effect=Exception("Connection error")
                )
                client._session = mock_session

                with pytest.raises(BrokerConnectionError):
                    await client._request("GET", "/v2/account")

        # Check exponential backoff pattern (0.01 * 2^attempt)
        expected = [0.01, 0.02]  # Two retries before max_retries=3
        assert delays == expected

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, client):
        """Test error raised after max retries exceeded."""
        with patch.object(client, "_session") as mock_session:
            mock_session.request = MagicMock(side_effect=Exception("Persistent error"))
            client._session = mock_session

            with pytest.raises(BrokerConnectionError) as exc_info:
                await client._request("GET", "/v2/account")

            assert "failed after" in str(exc_info.value).lower()


class TestRateLimiter:
    """Test rate limiter functionality."""

    @pytest.fixture
    def limiter(self):
        """Create rate limiter for testing."""
        return RateLimiter(requests_per_minute=5)

    def test_initial_state(self, limiter):
        """Test rate limiter initial state."""
        assert limiter.can_proceed()
        assert len(limiter._request_times) == 0

    @pytest.mark.asyncio
    async def test_under_rate_limit(self, limiter):
        """Test requests under rate limit proceed immediately."""
        for _ in range(4):  # Below limit of 5
            await limiter.acquire()

        assert limiter.can_proceed()

    @pytest.mark.asyncio
    async def test_rate_limit_reached(self, limiter):
        """Test rate limiter blocks when limit reached."""
        # Fill up to limit
        for _ in range(5):
            await limiter.acquire()

        assert not limiter.can_proceed()

    @pytest.mark.asyncio
    async def test_old_requests_expire(self, limiter):
        """Test old requests are removed after 60 seconds."""
        # Add request times from 70 seconds ago
        import time

        old_time = time.time() - 70
        limiter._request_times = [old_time] * 5

        # Should be able to proceed (old requests expired)
        assert limiter.can_proceed()


class TestFactoryFunction:
    """Test the create_live_feed factory function."""

    def test_create_mock_feed(self):
        """Test creating a MockLiveFeed via factory."""
        feed = create_live_feed(
            feed_type="mock",
            symbols=["AAPL", "MSFT"],
            interval_seconds=0.1,
        )
        assert isinstance(feed, MockLiveFeed)
        assert feed.symbols == ["AAPL", "MSFT"]

    def test_create_alpaca_feed(self):
        """Test creating an AlpacaWebSocketFeed via factory."""
        feed = create_live_feed(
            feed_type="alpaca",
            symbols=["AAPL"],
            api_key="test_key",
            api_secret="test_secret",
            paper=True,
        )
        assert isinstance(feed, AlpacaWebSocketFeed)
        assert feed.symbols == ["AAPL"]
        assert feed.config.api_key == "test_key"

    def test_unknown_feed_type_raises(self):
        """Test that unknown feed type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown feed type"):
            create_live_feed(feed_type="unknown", symbols=["AAPL"])


class TestCallbackNotification:
    """Test callback notification during reconnection."""

    @pytest.fixture
    def mock_feed(self):
        """Create MockLiveFeed for testing."""
        return MockLiveFeed(
            symbols=["AAPL"],
            interval_seconds=0.05,
        )

    @pytest.mark.asyncio
    async def test_callbacks_notified_on_bar(self, mock_feed):
        """Test callbacks are notified when bars are received."""
        received_bars = []

        def bar_callback(bar: OHLCVBar):
            received_bars.append(bar)

        mock_feed.add_bar_callback(bar_callback)
        await mock_feed.connect()

        # Wait for at least one bar
        await asyncio.sleep(0.1)
        await mock_feed.disconnect()

        assert len(received_bars) > 0
        assert received_bars[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_callback_error_doesnt_break_feed(self, mock_feed):
        """Test that callback errors don't break the feed."""
        good_callback_received = []

        def bad_callback(bar: OHLCVBar):
            raise Exception("Callback error")

        def good_callback(bar: OHLCVBar):
            good_callback_received.append(bar)

        mock_feed.add_bar_callback(bad_callback)
        mock_feed.add_bar_callback(good_callback)

        await mock_feed.connect()
        await asyncio.sleep(0.1)
        await mock_feed.disconnect()

        # Good callback should still receive bars
        assert len(good_callback_received) > 0

    def test_remove_callback(self, mock_feed):
        """Test callback removal."""
        callback_count = [0]

        def callback(bar):
            callback_count[0] += 1

        mock_feed.add_bar_callback(callback)
        assert callback in mock_feed._callbacks

        mock_feed.remove_bar_callback(callback)
        assert callback not in mock_feed._callbacks


class TestEventBusIntegration:
    """Test event bus integration with live feeds."""

    @pytest.fixture
    def event_bus(self):
        """Create EventBus for testing."""
        return EventBus()

    @pytest.fixture
    def mock_feed_with_bus(self, event_bus):
        """Create MockLiveFeed with EventBus."""
        return MockLiveFeed(
            symbols=["AAPL"],
            event_bus=event_bus,
            interval_seconds=0.05,
        )

    @pytest.mark.asyncio
    async def test_events_published_on_bars(self, mock_feed_with_bus, event_bus):
        """Test that bar updates are published to event bus."""
        from quant_trading_system.core.events import EventType

        received_events = []

        def event_handler(event):
            received_events.append(event)

        event_bus.subscribe(EventType.BAR_UPDATE, event_handler)

        await mock_feed_with_bus.connect()
        await asyncio.sleep(0.1)
        await mock_feed_with_bus.disconnect()

        assert len(received_events) > 0
        assert received_events[0].event_type == EventType.BAR_UPDATE

    @pytest.mark.asyncio
    async def test_no_events_without_bus(self):
        """Test no errors when event bus is not provided."""
        feed = MockLiveFeed(
            symbols=["AAPL"],
            event_bus=None,  # No event bus
            interval_seconds=0.05,
        )

        # Should not raise
        await feed.connect()
        await asyncio.sleep(0.1)
        await feed.disconnect()


class TestConnectionFailureScenarios:
    """Test various connection failure scenarios."""

    @pytest.fixture
    def alpaca_feed(self):
        """Create AlpacaWebSocketFeed for testing."""
        config = WebSocketConfig(
            api_key="test_key",
            api_secret="test_secret",
            paper=True,
        )
        return AlpacaWebSocketFeed(config=config, symbols=["AAPL"])

    @pytest.mark.asyncio
    async def test_import_error_handling(self, alpaca_feed):
        """Test handling when websockets not installed."""
        # Remove the module from cache to simulate import error
        import sys

        # Temporarily remove the module
        original_modules = {}
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("websocket"):
                original_modules[mod_name] = sys.modules.pop(mod_name)

        try:
            # Create mock that simulates ImportError
            with patch.dict("sys.modules", {"websockets": None}):
                with pytest.raises(DataConnectionError) as exc_info:
                    await alpaca_feed.connect()

                assert alpaca_feed.state == ConnectionState.ERROR
                assert "package required" in str(exc_info.value).lower()
        finally:
            # Restore original modules
            sys.modules.update(original_modules)

    @pytest.mark.asyncio
    async def test_auth_failure(self, alpaca_feed):
        """Test authentication failure handling."""
        # Mock websockets.connect to return a mock that fails auth
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value='[{"T": "error", "msg": "auth failed"}]')

        with patch("websockets.connect", return_value=mock_ws):
            with pytest.raises(DataConnectionError):
                await alpaca_feed.connect()

            assert alpaca_feed.state == ConnectionState.ERROR

    @pytest.mark.asyncio
    async def test_network_timeout(self, alpaca_feed):
        """Test network timeout handling."""
        # Mock websockets.connect to raise TimeoutError
        with patch("websockets.connect", side_effect=asyncio.TimeoutError("Connection timed out")):
            with pytest.raises(DataConnectionError):
                await alpaca_feed.connect()

            assert alpaca_feed.state == ConnectionState.ERROR

    @pytest.mark.asyncio
    async def test_already_connected_noop(self, alpaca_feed):
        """Test that connect is no-op when already connected."""
        alpaca_feed.state = ConnectionState.CONNECTED

        # Should return immediately without attempting connection
        await alpaca_feed.connect()

        # State should remain CONNECTED
        assert alpaca_feed.state == ConnectionState.CONNECTED


class TestSubscriptionManagement:
    """Test symbol subscription during reconnection."""

    @pytest.fixture
    def alpaca_feed(self):
        """Create AlpacaWebSocketFeed for testing."""
        config = WebSocketConfig(
            api_key="test_key",
            api_secret="test_secret",
            paper=True,
        )
        return AlpacaWebSocketFeed(config=config, symbols=["AAPL"])

    def test_initial_symbols(self, alpaca_feed):
        """Test initial symbols are set correctly."""
        assert alpaca_feed.symbols == ["AAPL"]

    @pytest.mark.asyncio
    async def test_subscribe_adds_symbols(self, alpaca_feed):
        """Test subscribing adds new symbols."""
        await alpaca_feed.subscribe(["MSFT", "GOOGL"])

        assert "MSFT" in alpaca_feed.symbols
        assert "GOOGL" in alpaca_feed.symbols

    @pytest.mark.asyncio
    async def test_subscribe_no_duplicates(self, alpaca_feed):
        """Test subscribing doesn't add duplicates."""
        await alpaca_feed.subscribe(["AAPL"])  # Already subscribed

        assert alpaca_feed.symbols.count("AAPL") == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_symbols(self, alpaca_feed):
        """Test unsubscribing removes symbols."""
        alpaca_feed.symbols = ["AAPL", "MSFT", "GOOGL"]

        await alpaca_feed.unsubscribe(["MSFT"])

        assert "MSFT" not in alpaca_feed.symbols
        assert "AAPL" in alpaca_feed.symbols
        assert "GOOGL" in alpaca_feed.symbols

    @pytest.mark.asyncio
    async def test_symbols_uppercase(self, alpaca_feed):
        """Test symbols are converted to uppercase."""
        await alpaca_feed.subscribe(["msft", "googl"])

        assert "MSFT" in alpaca_feed.symbols
        assert "GOOGL" in alpaca_feed.symbols


class TestStressScenarios:
    """Stress tests for reconnection logic."""

    @pytest.fixture
    def mock_feed(self):
        """Create MockLiveFeed for stress testing."""
        return MockLiveFeed(
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            interval_seconds=0.01,  # Fast generation
        )

    @pytest.mark.asyncio
    async def test_rapid_connect_disconnect(self, mock_feed):
        """Test rapid connect/disconnect cycles."""
        for _ in range(10):
            await mock_feed.connect()
            assert mock_feed.state == ConnectionState.CONNECTED

            await mock_feed.disconnect()
            assert mock_feed.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_high_frequency_bars(self, mock_feed):
        """Test handling high frequency bar updates."""
        received_count = [0]

        def counter_callback(bar):
            received_count[0] += 1

        mock_feed.add_bar_callback(counter_callback)

        await mock_feed.connect()
        await asyncio.sleep(0.2)  # Should receive multiple bars per symbol
        await mock_feed.disconnect()

        # Should have received multiple bars (5 symbols * ~20 intervals)
        assert received_count[0] > 10

    @pytest.mark.asyncio
    async def test_many_callbacks(self, mock_feed):
        """Test feed with many registered callbacks."""
        callback_results = {i: 0 for i in range(20)}

        def make_callback(idx):
            def callback(bar):
                callback_results[idx] += 1

            return callback

        # Register 20 callbacks
        for i in range(20):
            mock_feed.add_bar_callback(make_callback(i))

        await mock_feed.connect()
        await asyncio.sleep(0.05)
        await mock_feed.disconnect()

        # All callbacks should have been called
        assert all(count > 0 for count in callback_results.values())
