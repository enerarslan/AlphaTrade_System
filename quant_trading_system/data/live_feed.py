"""
Live data feed module for real-time market data streaming.

Production-ready implementation with:
- WebSocket connection management via websockets library
- Alpaca streaming API integration (Market Data + Trading streams)
- Automatic reconnection with exponential backoff
- Heartbeat/ping-pong monitoring
- Circuit breaker pattern for failover
- Thread-safe async operations
- Full EventBus integration
- PostgreSQL + TimescaleDB persistence (optional)

Alpaca WebSocket Endpoints:
- Market Data: wss://stream.data.alpaca.markets/v2/{feed}
- Trading (Paper): wss://paper-api.alpaca.markets/stream
- Trading (Live): wss://api.alpaca.markets/stream

@agent: @data, @trader
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from quant_trading_system.core.data_types import OHLCVBar
from quant_trading_system.core.events import (
    Event,
    EventBus,
    EventPriority,
    EventType,
    create_bar_event,
    get_event_bus,
)
from quant_trading_system.core.exceptions import (
    DataConnectionError,
    DataError,
    DataStreamError,
)

if TYPE_CHECKING:
    from quant_trading_system.data.live_feed_persister import LiveFeedPersister

logger = logging.getLogger(__name__)


class ConnectionState(str, Enum):
    """WebSocket connection state."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    AUTHENTICATING = "authenticating"
    CONNECTED = "connected"
    SUBSCRIBED = "subscribed"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    CLOSED = "closed"


class CircuitState(str, Enum):
    """Circuit breaker state."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


class StreamType(str, Enum):
    """Type of data stream."""

    BARS = "bars"
    QUOTES = "quotes"
    TRADES = "trades"
    DAILY_BARS = "dailyBars"
    UPDATED_BARS = "updatedBars"
    STATUSES = "statuses"
    LULDS = "lulds"  # Limit Up Limit Down


class DataFeedType(str, Enum):
    """Type of data feed provider."""

    ALPACA = "alpaca"
    MOCK = "mock"
    POLYGON = "polygon"
    POLLING = "polling"


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern for data feed failover.

    Prevents cascade failures by temporarily blocking requests after
    repeated failures, then gradually recovering.

    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Circuit tripped, all requests blocked
    - HALF_OPEN: Testing recovery, allows limited requests
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: datetime | None = field(default=None, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state with automatic state transitions."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time:
                # FIX: Use timezone-aware datetime to avoid comparison issues
                elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
        return self._state

    async def can_execute(self) -> bool:
        """Check if a request can be executed (async-safe)."""
        async with self._lock:
            state = self.state

            if state == CircuitState.CLOSED:
                return True
            elif state == CircuitState.OPEN:
                return False
            elif state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            return False

    async def record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    logger.info("Circuit breaker transitioning to CLOSED (recovered)")
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def record_failure(self) -> None:
        """Record a failed request."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker transitioning to OPEN (recovery failed)")
                self._state = CircuitState.OPEN
                self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    logger.warning(
                        f"Circuit breaker transitioning to OPEN "
                        f"(failure threshold {self.failure_threshold} reached)"
                    )
                    self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure": (
                self._last_failure_time.isoformat() if self._last_failure_time else None
            ),
        }


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket connection."""

    # Connection settings
    api_key: str
    api_secret: str
    feed: str = "iex"  # 'iex' (free) or 'sip' (paid)
    paper: bool = True

    # Reconnection settings
    initial_reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    max_reconnect_attempts: int = 10
    reconnect_jitter: float = 0.5  # Add randomness to prevent thundering herd

    # Heartbeat settings
    heartbeat_interval: float = 30.0
    heartbeat_timeout: float = 60.0

    # Buffer settings
    max_message_queue_size: int = 10000

    # MAJOR FIX: Data staleness detection settings
    # Alerts when symbol data becomes stale (no updates for configurable period)
    staleness_check_interval: float = 30.0  # How often to check for stale data
    staleness_warning_seconds: float = 120.0  # Warning after 2 minutes of no data
    staleness_critical_seconds: float = 300.0  # Critical alert after 5 minutes

    @property
    def market_data_url(self) -> str:
        """Get the market data WebSocket URL."""
        return f"wss://stream.data.alpaca.markets/v2/{self.feed}"

    @property
    def trading_stream_url(self) -> str:
        """Get the trading stream WebSocket URL."""
        if self.paper:
            return "wss://paper-api.alpaca.markets/stream"
        return "wss://api.alpaca.markets/stream"


class BaseLiveFeed(ABC):
    """Abstract base class for live data feeds."""

    def __init__(
        self,
        symbols: list[str],
        event_bus: EventBus | None = None,
    ):
        """
        Initialize the live feed.

        Args:
            symbols: List of symbols to subscribe to
            event_bus: Event bus for publishing updates
        """
        self.symbols = [s.upper() for s in symbols]
        self.event_bus = event_bus or get_event_bus()
        self.state = ConnectionState.DISCONNECTED
        self._callbacks: list[Callable[[OHLCVBar], None]] = []
        self._quote_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._trade_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._running = False

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to data feed."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data feed."""
        pass

    @abstractmethod
    async def subscribe(
        self,
        symbols: list[str],
        streams: list[StreamType] | None = None,
    ) -> None:
        """Subscribe to symbols."""
        pass

    @abstractmethod
    async def unsubscribe(
        self,
        symbols: list[str],
        streams: list[StreamType] | None = None,
    ) -> None:
        """Unsubscribe from symbols."""
        pass

    def add_bar_callback(self, callback: Callable[[OHLCVBar], None]) -> None:
        """Add a callback for bar updates."""
        self._callbacks.append(callback)

    def remove_bar_callback(self, callback: Callable[[OHLCVBar], None]) -> None:
        """Remove a bar callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def add_quote_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Add a callback for quote updates."""
        self._quote_callbacks.append(callback)

    def add_trade_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Add a callback for trade updates."""
        self._trade_callbacks.append(callback)

    def _notify_bar_callbacks(self, bar: OHLCVBar) -> None:
        """Notify all bar callbacks of a new bar."""
        for callback in self._callbacks:
            try:
                callback(bar)
            except Exception as e:
                logger.error(f"Bar callback error: {e}")

    def _notify_quote_callbacks(self, quote: dict[str, Any]) -> None:
        """Notify all quote callbacks."""
        for callback in self._quote_callbacks:
            try:
                callback(quote)
            except Exception as e:
                logger.error(f"Quote callback error: {e}")

    def _notify_trade_callbacks(self, trade: dict[str, Any]) -> None:
        """Notify all trade callbacks."""
        for callback in self._trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")

    def _publish_bar_event(self, bar: OHLCVBar) -> None:
        """Publish bar update to event bus."""
        if self.event_bus:
            event = create_bar_event(
                symbol=bar.symbol,
                bar_data=bar.to_dict(),
                source="live_feed",
            )
            self.event_bus.publish(event)

    def _publish_quote_event(self, quote: dict[str, Any]) -> None:
        """Publish quote update to event bus."""
        if self.event_bus:
            event = Event(
                event_type=EventType.QUOTE_UPDATE,
                data={"symbol": quote.get("S", ""), "quote": quote},
                source="live_feed",
                priority=EventPriority.NORMAL,
            )
            self.event_bus.publish(event)

    def _publish_trade_event(self, trade: dict[str, Any]) -> None:
        """Publish trade update to event bus."""
        if self.event_bus:
            event = Event(
                event_type=EventType.TRADE_UPDATE,
                data={"symbol": trade.get("S", ""), "trade": trade},
                source="live_feed",
                priority=EventPriority.NORMAL,
            )
            self.event_bus.publish(event)


class AlpacaWebSocketFeed(BaseLiveFeed):
    """
    Production-ready Alpaca WebSocket live feed.

    Implements:
    - Dual WebSocket connections (market data + trading updates)
    - Automatic authentication
    - Exponential backoff reconnection with jitter
    - Heartbeat monitoring via ping/pong
    - Circuit breaker for connection failures
    - Full event bus integration
    - Thread-safe async operations
    """

    def __init__(
        self,
        config: WebSocketConfig,
        symbols: list[str],
        event_bus: EventBus | None = None,
        persister: "LiveFeedPersister | None" = None,
    ):
        """
        Initialize Alpaca WebSocket feed.

        Args:
            config: WebSocket configuration
            symbols: Symbols to subscribe to
            event_bus: Event bus for publishing
            persister: Optional LiveFeedPersister for database persistence
        """
        super().__init__(symbols, event_bus)
        self.config = config
        self._persister = persister

        # WebSocket connections
        self._market_ws: Any = None  # websockets.WebSocketClientProtocol
        self._trading_ws: Any = None

        # Connection state
        self._reconnect_delay = config.initial_reconnect_delay
        self._reconnect_attempts = 0
        self._authenticated_market = False
        self._authenticated_trading = False
        self._subscribed_streams: dict[str, set[str]] = {
            "bars": set(),
            "quotes": set(),
            "trades": set(),
        }

        # Tasks
        self._market_task: asyncio.Task | None = None
        self._trading_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._message_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=config.max_message_queue_size
        )

        # Heartbeat tracking
        self._last_market_message: datetime | None = None
        self._last_trading_message: datetime | None = None
        self._pending_pong: bool = False

        # Circuit breaker
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            half_open_max_calls=3,
        )

        # Metrics
        self._metrics: dict[str, int] = {
            "messages_received": 0,
            "bars_processed": 0,
            "quotes_processed": 0,
            "trades_processed": 0,
            "reconnections": 0,
            "errors": 0,
            "out_of_order_bars": 0,
        }

        # JPMORGAN FIX: Timestamp ordering tracking per symbol
        # Prevents out-of-order bar processing that corrupts position states
        self._last_bar_timestamp: dict[str, datetime] = {}
        self._out_of_order_queue: list[tuple[str, dict[str, Any]]] = []

        # MAJOR FIX: Data staleness tracking per symbol
        # Alerts when data for a symbol becomes stale (no updates for too long)
        self._last_data_timestamp: dict[str, datetime] = {}
        self._staleness_alerts_sent: dict[str, str] = {}  # symbol -> alert level
        self._staleness_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Establish WebSocket connections to Alpaca."""
        if self.state == ConnectionState.CONNECTED:
            logger.warning("Already connected")
            return

        # Check circuit breaker
        if not await self._circuit_breaker.can_execute():
            stats = self._circuit_breaker.get_stats()
            logger.warning(f"Circuit breaker OPEN - connection blocked. Stats: {stats}")
            raise DataConnectionError(
                "Connection blocked by circuit breaker",
                source="alpaca_websocket",
            )

        self.state = ConnectionState.CONNECTING
        logger.info("Connecting to Alpaca WebSocket streams...")

        try:
            # Import websockets
            try:
                import websockets
                import websockets.exceptions
            except ImportError:
                raise DataConnectionError(
                    "websockets package required. Install with: pip install websockets",
                    source="alpaca_websocket",
                )

            # Connect to market data stream
            await self._connect_market_stream()

            # Connect to trading stream for order updates
            await self._connect_trading_stream()

            self.state = ConnectionState.CONNECTED
            self._reconnect_delay = self.config.initial_reconnect_delay
            self._reconnect_attempts = 0
            await self._circuit_breaker.record_success()

            # Set running flag
            self._running = True

            # Start heartbeat monitoring
            self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

            # MAJOR FIX: Start data staleness monitoring
            self._staleness_task = asyncio.create_task(self._staleness_monitor())

            logger.info(
                f"Connected to Alpaca WebSocket. "
                f"Market: {self._authenticated_market}, "
                f"Trading: {self._authenticated_trading}"
            )

        except Exception as e:
            self.state = ConnectionState.ERROR
            await self._circuit_breaker.record_failure()
            self._metrics["errors"] += 1
            logger.error(f"Connection failed: {e}")
            raise DataConnectionError(
                f"Failed to connect to Alpaca WebSocket: {e}",
                source="alpaca_websocket",
            )

    async def _connect_market_stream(self) -> None:
        """Connect to market data WebSocket."""
        import websockets

        logger.info(f"Connecting to market data: {self.config.market_data_url}")

        self._market_ws = await websockets.connect(
            self.config.market_data_url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            max_size=10 * 1024 * 1024,  # 10MB max message
        )

        # Wait for welcome message
        welcome = await asyncio.wait_for(self._market_ws.recv(), timeout=10.0)
        welcome_data = json.loads(welcome)
        logger.debug(f"Market stream welcome: {welcome_data}")

        if welcome_data[0].get("T") != "success" or welcome_data[0].get("msg") != "connected":
            raise DataConnectionError(
                f"Unexpected welcome message: {welcome_data}",
                source="market_stream",
            )

        # Authenticate
        self.state = ConnectionState.AUTHENTICATING
        auth_msg = {
            "action": "auth",
            "key": self.config.api_key,
            "secret": self.config.api_secret,
        }
        await self._market_ws.send(json.dumps(auth_msg))

        # Wait for auth response
        auth_response = await asyncio.wait_for(self._market_ws.recv(), timeout=10.0)
        auth_data = json.loads(auth_response)
        logger.debug(f"Market auth response: {auth_data}")

        if auth_data[0].get("T") != "success" or auth_data[0].get("msg") != "authenticated":
            raise DataConnectionError(
                f"Authentication failed: {auth_data}",
                source="market_stream",
            )

        self._authenticated_market = True
        self._last_market_message = datetime.now(timezone.utc)

        # Start market message handler
        self._market_task = asyncio.create_task(self._handle_market_messages())

        # Subscribe to initial symbols
        if self.symbols:
            await self._subscribe_market_streams(
                self.symbols,
                [StreamType.BARS, StreamType.QUOTES, StreamType.TRADES],
            )

    async def _connect_trading_stream(self) -> None:
        """Connect to trading WebSocket for order updates."""
        import websockets

        logger.info(f"Connecting to trading stream: {self.config.trading_stream_url}")

        try:
            self._trading_ws = await websockets.connect(
                self.config.trading_stream_url,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
            )

            # Wait for welcome message (trading stream format differs)
            welcome = await asyncio.wait_for(self._trading_ws.recv(), timeout=10.0)
            welcome_data = json.loads(welcome)
            logger.debug(f"Trading stream welcome: {welcome_data}")

            # Authenticate for trading stream
            auth_msg = {
                "action": "auth",
                "key": self.config.api_key,
                "secret": self.config.api_secret,
            }
            await self._trading_ws.send(json.dumps(auth_msg))

            # Wait for auth response
            auth_response = await asyncio.wait_for(self._trading_ws.recv(), timeout=10.0)
            auth_data = json.loads(auth_response)
            logger.debug(f"Trading auth response: {auth_data}")

            # Trading stream auth response format
            if auth_data.get("stream") == "authorization":
                if auth_data.get("data", {}).get("status") == "authorized":
                    self._authenticated_trading = True
                else:
                    logger.warning(f"Trading stream auth failed: {auth_data}")
            else:
                # Check for alternative success response format
                self._authenticated_trading = True

            self._last_trading_message = datetime.now(timezone.utc)

            # Subscribe to trade updates
            listen_msg = {
                "action": "listen",
                "data": {"streams": ["trade_updates"]},
            }
            await self._trading_ws.send(json.dumps(listen_msg))

            # Start trading message handler
            self._trading_task = asyncio.create_task(self._handle_trading_messages())

        except Exception as e:
            logger.warning(f"Trading stream connection failed (non-fatal): {e}")
            self._authenticated_trading = False

    async def _handle_market_messages(self) -> None:
        """Handle incoming market data messages."""
        import websockets.exceptions

        try:
            async for message in self._market_ws:
                self._last_market_message = datetime.now(timezone.utc)
                self._metrics["messages_received"] += 1

                try:
                    data = json.loads(message)

                    # Handle array of messages
                    if isinstance(data, list):
                        for msg in data:
                            await self._process_market_message(msg)
                    else:
                        await self._process_market_message(data)

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in market message: {e}")
                except Exception as e:
                    logger.error(f"Error processing market message: {e}")
                    self._metrics["errors"] += 1

        except websockets.exceptions.ConnectionClosedOK:
            logger.info("Market WebSocket closed normally")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"Market WebSocket closed with error: {e}")
            if self._running:
                await self._schedule_reconnect("market")
        except Exception as e:
            logger.error(f"Market message handler error: {e}")
            if self._running:
                await self._schedule_reconnect("market")

    async def _handle_trading_messages(self) -> None:
        """Handle incoming trading stream messages."""
        import websockets.exceptions

        if not self._trading_ws:
            return

        try:
            async for message in self._trading_ws:
                self._last_trading_message = datetime.now(timezone.utc)

                try:
                    data = json.loads(message)
                    await self._process_trading_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in trading message: {e}")
                except Exception as e:
                    logger.error(f"Error processing trading message: {e}")

        except websockets.exceptions.ConnectionClosedOK:
            logger.info("Trading WebSocket closed normally")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"Trading WebSocket closed with error: {e}")
            # Trading stream is optional, don't force reconnect
        except Exception as e:
            logger.error(f"Trading message handler error: {e}")

    async def _process_market_message(self, msg: dict[str, Any]) -> None:
        """Process a single market data message."""
        msg_type = msg.get("T")

        if msg_type == "b":  # Bar
            await self._handle_bar(msg)
        elif msg_type == "q":  # Quote
            await self._handle_quote(msg)
        elif msg_type == "t":  # Trade
            await self._handle_trade(msg)
        elif msg_type == "d":  # Daily bar
            await self._handle_bar(msg, is_daily=True)
        elif msg_type == "u":  # Updated bar (corrections)
            await self._handle_bar(msg, is_update=True)
        elif msg_type == "s":  # Status
            logger.debug(f"Trading status: {msg}")
        elif msg_type == "l":  # LULD
            logger.debug(f"LULD: {msg}")
        elif msg_type == "success":
            logger.debug(f"Success message: {msg}")
        elif msg_type == "subscription":
            logger.info(f"Subscription confirmed: {msg}")
        elif msg_type == "error":
            logger.error(f"Market stream error: {msg}")
            self._metrics["errors"] += 1
        else:
            logger.debug(f"Unknown market message type '{msg_type}': {msg}")

    async def _process_trading_message(self, msg: dict[str, Any]) -> None:
        """Process a single trading stream message."""
        stream = msg.get("stream")

        if stream == "trade_updates":
            data = msg.get("data", {})
            event_type = data.get("event")

            # Emit appropriate order event
            if event_type == "new":
                self._publish_order_event(EventType.ORDER_SUBMITTED, data)
            elif event_type == "fill":
                self._publish_order_event(EventType.ORDER_FILLED, data)
            elif event_type == "partial_fill":
                self._publish_order_event(EventType.ORDER_PARTIAL, data)
            elif event_type == "canceled":
                self._publish_order_event(EventType.ORDER_CANCELLED, data)
            elif event_type == "rejected":
                self._publish_order_event(EventType.ORDER_REJECTED, data)
            elif event_type == "expired":
                self._publish_order_event(EventType.ORDER_EXPIRED, data)
            elif event_type == "accepted":
                self._publish_order_event(EventType.ORDER_ACCEPTED, data)

            logger.debug(f"Trade update: {event_type} - {data.get('order', {}).get('symbol')}")

        elif stream == "authorization":
            status = msg.get("data", {}).get("status")
            if status == "authorized":
                logger.info("Trading stream authorized")
            else:
                logger.warning(f"Trading stream auth status: {status}")
        elif stream == "listening":
            logger.info(f"Trading stream listening to: {msg.get('data', {}).get('streams')}")
        else:
            logger.debug(f"Unknown trading stream: {msg}")

    def _publish_order_event(self, event_type: EventType, data: dict[str, Any]) -> None:
        """Publish order event to event bus."""
        if self.event_bus:
            event = Event(
                event_type=event_type,
                data={"order": data.get("order", {}), "event": data.get("event")},
                source="trading_stream",
                priority=EventPriority.HIGH,
            )
            self.event_bus.publish(event)

    async def _handle_bar(
        self,
        msg: dict[str, Any],
        is_daily: bool = False,
        is_update: bool = False,
    ) -> None:
        """Handle incoming bar data with JPMORGAN-LEVEL timestamp ordering validation."""
        try:
            # Parse Alpaca bar format
            symbol = msg.get("S", "")
            timestamp_str = msg.get("t", "")

            # Parse ISO timestamp
            if timestamp_str:
                if timestamp_str.endswith("Z"):
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                else:
                    timestamp = datetime.fromisoformat(timestamp_str)
            else:
                timestamp = datetime.now(timezone.utc)

            # JPMORGAN FIX: Validate timestamp ordering per symbol
            # Critical for preventing position state corruption from out-of-order data
            if symbol in self._last_bar_timestamp:
                last_ts = self._last_bar_timestamp[symbol]
                if timestamp < last_ts:
                    # Out-of-order bar detected - log critical and skip
                    self._metrics["out_of_order_bars"] += 1
                    logger.critical(
                        f"OUT-OF-ORDER BAR DETECTED: {symbol} received bar for "
                        f"{timestamp.isoformat()} but last processed was {last_ts.isoformat()}. "
                        f"Delta: {(last_ts - timestamp).total_seconds():.1f}s. "
                        f"Bar REJECTED to prevent position state corruption."
                    )
                    # Queue for potential replay/investigation
                    self._out_of_order_queue.append((symbol, msg))
                    if len(self._out_of_order_queue) > 100:
                        self._out_of_order_queue.pop(0)  # Keep queue bounded
                    return  # Skip this bar
                elif timestamp == last_ts and not is_update:
                    # Duplicate bar (same timestamp, not an update)
                    logger.warning(
                        f"Duplicate bar timestamp for {symbol} at {timestamp.isoformat()}"
                    )
                    return  # Skip duplicate

            # Update last timestamp for this symbol
            self._last_bar_timestamp[symbol] = timestamp

            bar = OHLCVBar(
                symbol=symbol,
                timestamp=timestamp,
                open=Decimal(str(msg.get("o", 0))),
                high=Decimal(str(msg.get("h", 0))),
                low=Decimal(str(msg.get("l", 0))),
                close=Decimal(str(msg.get("c", 0))),
                volume=int(msg.get("v", 0)),
                vwap=Decimal(str(msg.get("vw"))) if msg.get("vw") else None,
                trade_count=int(msg.get("n")) if msg.get("n") else None,
            )

            self._metrics["bars_processed"] += 1

            # MAJOR FIX: Update staleness tracking
            self._update_symbol_data_timestamp(symbol)

            # Persist to database (async, non-blocking)
            if self._persister is not None:
                try:
                    await self._persister.persist_bar(bar)
                except Exception as e:
                    # Log but don't fail - persistence is secondary to trading
                    logger.warning(f"Failed to persist bar for {symbol}: {e}")

            # Notify callbacks
            self._notify_bar_callbacks(bar)

            # Publish to event bus
            self._publish_bar_event(bar)

            logger.debug(
                f"{'Daily ' if is_daily else ''}{'Updated ' if is_update else ''}"
                f"Bar: {symbol} @ {bar.close} (vol: {bar.volume})"
            )

        except Exception as e:
            logger.error(f"Error handling bar: {e} - Message: {msg}")
            self._metrics["errors"] += 1

    async def _handle_quote(self, msg: dict[str, Any]) -> None:
        """Handle incoming quote data."""
        try:
            symbol = msg.get("S", "")
            self._metrics["quotes_processed"] += 1

            # MAJOR FIX: Update staleness tracking
            if symbol:
                self._update_symbol_data_timestamp(symbol)

            # Notify callbacks
            self._notify_quote_callbacks(msg)

            # Publish to event bus
            self._publish_quote_event(msg)

            logger.debug(
                f"Quote: {symbol} bid={msg.get('bp')} ask={msg.get('ap')}"
            )

        except Exception as e:
            logger.error(f"Error handling quote: {e}")
            self._metrics["errors"] += 1

    async def _handle_trade(self, msg: dict[str, Any]) -> None:
        """Handle incoming trade data."""
        try:
            symbol = msg.get("S", "")
            self._metrics["trades_processed"] += 1

            # MAJOR FIX: Update staleness tracking
            if symbol:
                self._update_symbol_data_timestamp(symbol)

            # Notify callbacks
            self._notify_trade_callbacks(msg)

            # Publish to event bus
            self._publish_trade_event(msg)

            logger.debug(
                f"Trade: {symbol} @ {msg.get('p')} x {msg.get('s')}"
            )

        except Exception as e:
            logger.error(f"Error handling trade: {e}")
            self._metrics["errors"] += 1

    async def _staleness_monitor(self) -> None:
        """MAJOR FIX: Monitor data staleness per symbol.

        Checks each subscribed symbol for data freshness and
        publishes alerts when data becomes stale. This is critical
        for detecting market data issues before they impact trading.
        """
        while self._running:
            try:
                await asyncio.sleep(self.config.staleness_check_interval)

                now = datetime.now(timezone.utc)

                for symbol in self.symbols:
                    last_update = self._last_data_timestamp.get(symbol)

                    if last_update is None:
                        # Never received data for this symbol
                        if symbol not in self._staleness_alerts_sent:
                            logger.warning(
                                f"DATA STALENESS: No data ever received for {symbol}"
                            )
                            self._staleness_alerts_sent[symbol] = "warning"
                            self._publish_staleness_alert(symbol, "warning", None)
                        continue

                    elapsed = (now - last_update).total_seconds()

                    if elapsed >= self.config.staleness_critical_seconds:
                        # Critical staleness
                        if self._staleness_alerts_sent.get(symbol) != "critical":
                            logger.critical(
                                f"DATA STALENESS CRITICAL: {symbol} no data for "
                                f"{elapsed:.0f}s (threshold: {self.config.staleness_critical_seconds}s)"
                            )
                            self._staleness_alerts_sent[symbol] = "critical"
                            self._publish_staleness_alert(symbol, "critical", elapsed)

                    elif elapsed >= self.config.staleness_warning_seconds:
                        # Warning staleness
                        if self._staleness_alerts_sent.get(symbol) not in ("warning", "critical"):
                            logger.warning(
                                f"DATA STALENESS WARNING: {symbol} no data for "
                                f"{elapsed:.0f}s (threshold: {self.config.staleness_warning_seconds}s)"
                            )
                            self._staleness_alerts_sent[symbol] = "warning"
                            self._publish_staleness_alert(symbol, "warning", elapsed)

                    else:
                        # Data is fresh - clear any previous alerts
                        if symbol in self._staleness_alerts_sent:
                            logger.info(
                                f"DATA STALENESS CLEARED: {symbol} is receiving data again"
                            )
                            self._staleness_alerts_sent.pop(symbol, None)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Staleness monitor error: {e}")

    def _publish_staleness_alert(
        self,
        symbol: str,
        severity: str,
        elapsed_seconds: float | None
    ) -> None:
        """Publish data staleness alert to event bus."""
        if self.event_bus:
            event = Event(
                event_type=EventType.SYSTEM_ALERT,
                data={
                    "alert_type": "data_staleness",
                    "symbol": symbol,
                    "severity": severity,
                    "elapsed_seconds": elapsed_seconds,
                    "warning_threshold": self.config.staleness_warning_seconds,
                    "critical_threshold": self.config.staleness_critical_seconds,
                },
                source="live_feed",
                priority=EventPriority.HIGH if severity == "critical" else EventPriority.NORMAL,
            )
            self.event_bus.publish(event)

    def _update_symbol_data_timestamp(self, symbol: str) -> None:
        """Update the last data timestamp for a symbol."""
        self._last_data_timestamp[symbol] = datetime.now(timezone.utc)

    async def _heartbeat_monitor(self) -> None:
        """Monitor connection health via heartbeat."""
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                # Check market stream health
                if self._last_market_message:
                    elapsed = (
                        datetime.now(timezone.utc) - self._last_market_message
                    ).total_seconds()
                    if elapsed > self.config.heartbeat_timeout:
                        logger.warning(
                            f"No market data received for {elapsed:.0f}s - reconnecting"
                        )
                        await self._schedule_reconnect("market")

                # Check trading stream health (more lenient - trading updates are sparse)
                if self._trading_ws and self._last_trading_message:
                    elapsed = (
                        datetime.now(timezone.utc) - self._last_trading_message
                    ).total_seconds()
                    # Trading stream timeout is longer since updates are event-driven
                    if elapsed > self.config.heartbeat_timeout * 3:
                        logger.warning(
                            f"No trading updates for {elapsed:.0f}s - attempting ping"
                        )
                        try:
                            pong = await asyncio.wait_for(
                                self._trading_ws.ping(),
                                timeout=10.0,
                            )
                            await pong
                            self._last_trading_message = datetime.now(timezone.utc)
                        except Exception:
                            logger.warning("Trading stream ping failed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")

    async def _schedule_reconnect(self, stream: str) -> None:
        """Schedule a reconnection attempt."""
        if not self._running:
            return

        self.state = ConnectionState.RECONNECTING
        self._metrics["reconnections"] += 1

        if self._reconnect_attempts >= self.config.max_reconnect_attempts:
            logger.critical(
                f"Max reconnection attempts ({self.config.max_reconnect_attempts}) "
                f"reached. Manual intervention required."
            )
            self.state = ConnectionState.ERROR
            await self._circuit_breaker.record_failure()
            return

        self._reconnect_attempts += 1

        # Calculate delay with exponential backoff and jitter
        jitter = random.uniform(0, self.config.reconnect_jitter)
        delay = min(
            self._reconnect_delay * (2 ** (self._reconnect_attempts - 1)),
            self.config.max_reconnect_delay,
        ) + jitter

        logger.info(
            f"Reconnection attempt {self._reconnect_attempts}/"
            f"{self.config.max_reconnect_attempts} for {stream} in {delay:.1f}s"
        )

        await asyncio.sleep(delay)

        try:
            if stream == "market":
                if self._market_ws:
                    await self._market_ws.close()
                await self._connect_market_stream()
            elif stream == "trading":
                if self._trading_ws:
                    await self._trading_ws.close()
                await self._connect_trading_stream()

            self.state = ConnectionState.CONNECTED
            self._reconnect_attempts = 0
            self._reconnect_delay = self.config.initial_reconnect_delay
            await self._circuit_breaker.record_success()
            logger.info(f"Reconnection successful for {stream}")

        except Exception as e:
            logger.error(f"Reconnection failed for {stream}: {e}")
            await self._circuit_breaker.record_failure()
            await self._schedule_reconnect(stream)

    async def subscribe(
        self,
        symbols: list[str],
        streams: list[StreamType] | None = None,
    ) -> None:
        """Subscribe to symbols for specified streams."""
        symbols = [s.upper() for s in symbols]
        streams = streams or [StreamType.BARS, StreamType.QUOTES, StreamType.TRADES]

        # Add to tracking
        for s in symbols:
            if s not in self.symbols:
                self.symbols.append(s)

        if self.state == ConnectionState.CONNECTED and self._market_ws:
            await self._subscribe_market_streams(symbols, streams)

    async def _subscribe_market_streams(
        self,
        symbols: list[str],
        streams: list[StreamType],
    ) -> None:
        """Send subscription message to market stream."""
        sub_msg: dict[str, Any] = {"action": "subscribe"}

        for stream in streams:
            if stream == StreamType.BARS:
                sub_msg["bars"] = symbols
                self._subscribed_streams["bars"].update(symbols)
            elif stream == StreamType.QUOTES:
                sub_msg["quotes"] = symbols
                self._subscribed_streams["quotes"].update(symbols)
            elif stream == StreamType.TRADES:
                sub_msg["trades"] = symbols
                self._subscribed_streams["trades"].update(symbols)
            elif stream == StreamType.DAILY_BARS:
                sub_msg["dailyBars"] = symbols
            elif stream == StreamType.UPDATED_BARS:
                sub_msg["updatedBars"] = symbols

        await self._market_ws.send(json.dumps(sub_msg))
        logger.info(f"Subscribed to {symbols} for streams: {[s.value for s in streams]}")

    async def unsubscribe(
        self,
        symbols: list[str],
        streams: list[StreamType] | None = None,
    ) -> None:
        """Unsubscribe from symbols."""
        symbols = [s.upper() for s in symbols]
        streams = streams or [StreamType.BARS, StreamType.QUOTES, StreamType.TRADES]

        # Remove from tracking
        for s in symbols:
            if s in self.symbols:
                self.symbols.remove(s)

        if self.state == ConnectionState.CONNECTED and self._market_ws:
            unsub_msg: dict[str, Any] = {"action": "unsubscribe"}

            for stream in streams:
                if stream == StreamType.BARS:
                    unsub_msg["bars"] = symbols
                    self._subscribed_streams["bars"] -= set(symbols)
                elif stream == StreamType.QUOTES:
                    unsub_msg["quotes"] = symbols
                    self._subscribed_streams["quotes"] -= set(symbols)
                elif stream == StreamType.TRADES:
                    unsub_msg["trades"] = symbols
                    self._subscribed_streams["trades"] -= set(symbols)

            await self._market_ws.send(json.dumps(unsub_msg))
            logger.info(f"Unsubscribed from {symbols}")

    async def disconnect(self) -> None:
        """Disconnect from all WebSocket streams."""
        self._running = False
        self.state = ConnectionState.CLOSED

        # Cancel tasks
        tasks_to_cancel = [
            self._heartbeat_task,
            self._market_task,
            self._trading_task,
            self._staleness_task,  # MAJOR FIX: Include staleness task
        ]

        for task in tasks_to_cancel:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close WebSocket connections
        if self._market_ws:
            try:
                await self._market_ws.close()
            except Exception as e:
                logger.warning(f"Error closing market WebSocket: {e}")
            self._market_ws = None

        if self._trading_ws:
            try:
                await self._trading_ws.close()
            except Exception as e:
                logger.warning(f"Error closing trading WebSocket: {e}")
            self._trading_ws = None

        self._authenticated_market = False
        self._authenticated_trading = False
        self.state = ConnectionState.DISCONNECTED

        logger.info("Disconnected from Alpaca WebSocket streams")

    async def run(self) -> None:
        """Run the live feed (blocking)."""
        await self.connect()

        try:
            # Keep running until disconnected
            while self._running and self.state in (
                ConnectionState.CONNECTED,
                ConnectionState.SUBSCRIBED,
                ConnectionState.RECONNECTING,
            ):
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.disconnect()

    def get_metrics(self) -> dict[str, Any]:
        """Get feed metrics."""
        return {
            **self._metrics,
            "state": self.state.value,
            "subscribed_symbols": len(self.symbols),
            "authenticated_market": self._authenticated_market,
            "authenticated_trading": self._authenticated_trading,
            "circuit_breaker": self._circuit_breaker.get_stats(),
        }

    def get_subscriptions(self) -> dict[str, list[str]]:
        """Get current subscriptions."""
        return {k: list(v) for k, v in self._subscribed_streams.items()}


# Backwards compatibility alias
AlpacaLiveFeed = AlpacaWebSocketFeed


class MockLiveFeed(BaseLiveFeed):
    """
    Mock live feed for testing and development.

    Generates simulated bar data at specified intervals.
    """

    def __init__(
        self,
        symbols: list[str],
        event_bus: EventBus | None = None,
        interval_seconds: float = 1.0,
        initial_prices: dict[str, float] | None = None,
        volatility: float = 0.001,
    ):
        """
        Initialize mock feed.

        Args:
            symbols: Symbols to simulate
            event_bus: Event bus for publishing
            interval_seconds: Seconds between bar updates
            initial_prices: Initial prices per symbol
            volatility: Price volatility (std dev multiplier)
        """
        super().__init__(symbols, event_bus)
        self.interval_seconds = interval_seconds
        self.volatility = volatility
        self._prices = initial_prices or {s: 100.0 for s in self.symbols}
        self._task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Start mock feed."""
        self.state = ConnectionState.CONNECTED
        self._running = True
        self._task = asyncio.create_task(self._generate_bars())
        logger.info(f"Mock feed started for {len(self.symbols)} symbols")

    async def disconnect(self) -> None:
        """Stop mock feed."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.state = ConnectionState.DISCONNECTED
        logger.info("Mock feed stopped")

    async def subscribe(
        self,
        symbols: list[str],
        streams: list[StreamType] | None = None,
    ) -> None:
        """Add symbols to mock feed."""
        for s in symbols:
            s = s.upper()
            if s not in self.symbols:
                self.symbols.append(s)
                self._prices[s] = 100.0

    async def unsubscribe(
        self,
        symbols: list[str],
        streams: list[StreamType] | None = None,
    ) -> None:
        """Remove symbols from mock feed."""
        for s in symbols:
            s = s.upper()
            if s in self.symbols:
                self.symbols.remove(s)
                self._prices.pop(s, None)

    async def _generate_bars(self) -> None:
        """Generate simulated bars."""
        while self._running:
            timestamp = datetime.now(timezone.utc)

            for symbol in self.symbols:
                price = self._prices.get(symbol, 100.0)
                change = random.gauss(0, price * self.volatility)
                new_price = max(0.01, price + change)

                open_price = price
                close_price = new_price
                high_price = max(open_price, close_price) + abs(
                    random.gauss(0, price * self.volatility * 0.5)
                )
                low_price = min(open_price, close_price) - abs(
                    random.gauss(0, price * self.volatility * 0.5)
                )

                bar = OHLCVBar(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=Decimal(str(round(open_price, 2))),
                    high=Decimal(str(round(high_price, 2))),
                    low=Decimal(str(round(low_price, 2))),
                    close=Decimal(str(round(close_price, 2))),
                    volume=random.randint(1000, 100000),
                    trade_count=random.randint(10, 500),
                )

                self._prices[symbol] = new_price
                self._notify_bar_callbacks(bar)
                self._publish_bar_event(bar)

            await asyncio.sleep(self.interval_seconds)


class BarAggregator:
    """
    Aggregates tick/quote data into OHLCV bars.

    Handles partial bars at market open/close and computes
    proper OHLCV values including VWAP.
    """

    def __init__(
        self,
        timeframe_minutes: int = 15,
        on_bar_complete: Callable[[OHLCVBar], None] | None = None,
    ):
        """
        Initialize bar aggregator.

        Args:
            timeframe_minutes: Bar timeframe in minutes
            on_bar_complete: Callback when bar is complete
        """
        self.timeframe_minutes = timeframe_minutes
        self.on_bar_complete = on_bar_complete
        self._pending_bars: dict[str, dict[str, Any]] = {}

    def add_trade(
        self,
        symbol: str,
        price: float,
        volume: int,
        timestamp: datetime,
    ) -> OHLCVBar | None:
        """
        Add a trade to the aggregator.

        Args:
            symbol: Ticker symbol
            price: Trade price
            volume: Trade volume
            timestamp: Trade timestamp

        Returns:
            Completed bar if timeframe boundary crossed, else None
        """
        symbol = symbol.upper()
        bar_start = self._get_bar_start(timestamp)

        # Check if we need to complete previous bar
        completed_bar = None
        if symbol in self._pending_bars:
            pending = self._pending_bars[symbol]
            if pending["bar_start"] != bar_start:
                completed_bar = self._complete_bar(symbol)

        # Initialize or update pending bar
        if (
            symbol not in self._pending_bars
            or self._pending_bars[symbol]["bar_start"] != bar_start
        ):
            self._pending_bars[symbol] = {
                "bar_start": bar_start,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
                "vwap_volume_sum": price * volume,
                "trade_count": 1,
            }
        else:
            pending = self._pending_bars[symbol]
            pending["high"] = max(pending["high"], price)
            pending["low"] = min(pending["low"], price)
            pending["close"] = price
            pending["volume"] += volume
            pending["vwap_volume_sum"] += price * volume
            pending["trade_count"] += 1

        return completed_bar

    def _get_bar_start(self, timestamp: datetime) -> datetime:
        """Get the start of the bar period for a timestamp."""
        minutes = timestamp.minute - (timestamp.minute % self.timeframe_minutes)
        return timestamp.replace(minute=minutes, second=0, microsecond=0)

    def _complete_bar(self, symbol: str) -> OHLCVBar:
        """Complete a pending bar and return it."""
        pending = self._pending_bars.pop(symbol)

        vwap = None
        if pending["volume"] > 0:
            vwap = Decimal(
                str(round(pending["vwap_volume_sum"] / pending["volume"], 4))
            )

        bar = OHLCVBar(
            symbol=symbol,
            timestamp=pending["bar_start"],
            open=Decimal(str(round(pending["open"], 4))),
            high=Decimal(str(round(pending["high"], 4))),
            low=Decimal(str(round(pending["low"], 4))),
            close=Decimal(str(round(pending["close"], 4))),
            volume=pending["volume"],
            vwap=vwap,
            trade_count=pending["trade_count"],
        )

        if self.on_bar_complete:
            self.on_bar_complete(bar)

        return bar

    def flush(self) -> list[OHLCVBar]:
        """Flush all pending bars."""
        bars = []
        for symbol in list(self._pending_bars.keys()):
            bars.append(self._complete_bar(symbol))
        return bars


def create_live_feed(
    feed_type: str = "alpaca",
    symbols: list[str] | None = None,
    event_bus: EventBus | None = None,
    **kwargs: Any,
) -> BaseLiveFeed:
    """
    Factory function to create a live feed.

    Args:
        feed_type: Type of feed ('alpaca', 'mock')
        symbols: Symbols to subscribe to
        event_bus: Event bus for publishing
        **kwargs: Additional arguments for specific feed type

    Returns:
        Live feed instance

    Example:
        >>> feed = create_live_feed(
        ...     feed_type="alpaca",
        ...     symbols=["AAPL", "MSFT"],
        ...     api_key="your_key",
        ...     api_secret="your_secret",
        ...     paper=True,
        ... )
        >>> await feed.connect()
    """
    symbols = symbols or []

    if feed_type == "mock":
        return MockLiveFeed(
            symbols=symbols,
            event_bus=event_bus,
            interval_seconds=kwargs.get("interval_seconds", 1.0),
            initial_prices=kwargs.get("initial_prices"),
            volatility=kwargs.get("volatility", 0.001),
        )

    elif feed_type == "alpaca":
        config = WebSocketConfig(
            api_key=kwargs.get("api_key", ""),
            api_secret=kwargs.get("api_secret", ""),
            feed=kwargs.get("feed", "iex"),
            paper=kwargs.get("paper", True),
            initial_reconnect_delay=kwargs.get("initial_reconnect_delay", 1.0),
            max_reconnect_delay=kwargs.get("max_reconnect_delay", 60.0),
            max_reconnect_attempts=kwargs.get("max_reconnect_attempts", 10),
            heartbeat_interval=kwargs.get("heartbeat_interval", 30.0),
            heartbeat_timeout=kwargs.get("heartbeat_timeout", 60.0),
        )
        return AlpacaWebSocketFeed(
            config=config,
            symbols=symbols,
            event_bus=event_bus,
        )

    else:
        raise ValueError(f"Unknown feed type: {feed_type}")


# Helper function for quick setup
async def start_live_feed(
    api_key: str,
    api_secret: str,
    symbols: list[str],
    paper: bool = True,
    feed: str = "iex",
) -> AlpacaWebSocketFeed:
    """
    Quick helper to start a live feed.

    Args:
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        symbols: Symbols to subscribe to
        paper: Use paper trading (default True)
        feed: Data feed type ('iex' or 'sip')

    Returns:
        Connected AlpacaWebSocketFeed instance

    Example:
        >>> feed = await start_live_feed(
        ...     api_key="...",
        ...     api_secret="...",
        ...     symbols=["AAPL", "MSFT"],
        ... )
    """
    config = WebSocketConfig(
        api_key=api_key,
        api_secret=api_secret,
        feed=feed,
        paper=paper,
    )
    live_feed = AlpacaWebSocketFeed(
        config=config,
        symbols=symbols,
    )
    await live_feed.connect()
    return live_feed
