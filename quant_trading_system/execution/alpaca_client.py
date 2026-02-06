"""
Alpaca API client for trade execution.

Provides a unified interface to Alpaca's REST API and WebSocket streams:
- Account management (balance, buying power, positions)
- Order submission, cancellation, and modification
- Position management
- Market data retrieval
- Real-time trade and account updates via WebSocket

Includes rate limiting, retry logic, and error handling.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable
from uuid import UUID

from quant_trading_system.core.data_types import (
    OHLCVBar,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)
from quant_trading_system.core.exceptions import (
    BrokerConnectionError,
    InsufficientFundsError,
    OrderCancellationError,
    OrderSubmissionError,
)

logger = logging.getLogger(__name__)


class TradingEnvironment(str, Enum):
    """Trading environment (paper vs live)."""

    PAPER = "paper"
    LIVE = "live"


class OrderClass(str, Enum):
    """Alpaca order class types."""

    SIMPLE = "simple"
    BRACKET = "bracket"
    OTO = "oto"  # One-Triggers-Other
    OCO = "oco"  # One-Cancels-Other


@dataclass
class RateLimiter:
    """Rate limiter for API requests."""

    requests_per_minute: int = 200
    _request_times: list[float] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def acquire(self) -> None:
        """Wait if rate limit would be exceeded."""
        async with self._lock:
            now = time.time()
            # Remove requests older than 1 minute
            self._request_times = [t for t in self._request_times if now - t < 60]

            if len(self._request_times) >= self.requests_per_minute:
                # Wait until oldest request is more than 1 minute old
                wait_time = 60 - (now - self._request_times[0])
                if wait_time > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)

            self._request_times.append(time.time())

    def can_proceed(self) -> bool:
        """Check if request can proceed without waiting."""
        now = time.time()
        recent = [t for t in self._request_times if now - t < 60]
        return len(recent) < self.requests_per_minute


@dataclass
class AccountInfo:
    """Alpaca account information."""

    account_id: str
    status: str
    currency: str
    cash: Decimal
    portfolio_value: Decimal
    buying_power: Decimal
    daytrading_buying_power: Decimal
    regt_buying_power: Decimal
    equity: Decimal
    last_equity: Decimal
    long_market_value: Decimal
    short_market_value: Decimal
    initial_margin: Decimal
    maintenance_margin: Decimal
    multiplier: int
    pattern_day_trader: bool
    trading_blocked: bool
    transfers_blocked: bool
    account_blocked: bool
    created_at: datetime
    sma: Decimal | None = None
    daytrade_count: int = 0

    @classmethod
    def from_alpaca(cls, data: dict[str, Any]) -> "AccountInfo":
        """Create from Alpaca API response.

        CRITICAL FIX: All monetary values converted through str() to avoid
        float precision issues. Decimal(0.1) != Decimal("0.1").
        """
        return cls(
            account_id=data["id"],
            status=data["status"],
            currency=data["currency"],
            cash=Decimal(str(data["cash"])),
            portfolio_value=Decimal(str(data["portfolio_value"])),
            buying_power=Decimal(str(data["buying_power"])),
            daytrading_buying_power=Decimal(str(data.get("daytrading_buying_power", "0"))),
            regt_buying_power=Decimal(str(data.get("regt_buying_power", "0"))),
            equity=Decimal(str(data["equity"])),
            last_equity=Decimal(str(data["last_equity"])),
            long_market_value=Decimal(str(data["long_market_value"])),
            short_market_value=Decimal(str(data["short_market_value"])),
            initial_margin=Decimal(str(data["initial_margin"])),
            maintenance_margin=Decimal(str(data["maintenance_margin"])),
            multiplier=int(data.get("multiplier", 1)),
            pattern_day_trader=data.get("pattern_day_trader", False),
            trading_blocked=data.get("trading_blocked", False),
            transfers_blocked=data.get("transfers_blocked", False),
            account_blocked=data.get("account_blocked", False),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            sma=Decimal(str(data["sma"])) if data.get("sma") else None,
            daytrade_count=data.get("daytrade_count", 0),
        )


@dataclass
class AlpacaPosition:
    """Alpaca position data."""

    symbol: str
    quantity: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: Decimal
    cost_basis: Decimal
    asset_class: str
    asset_id: str
    side: str  # long or short
    exchange: str
    change_today: Decimal | None = None

    @classmethod
    def from_alpaca(cls, data: dict[str, Any]) -> "AlpacaPosition":
        """Create from Alpaca API response.

        CRITICAL FIX: All monetary values converted through str() to avoid
        float precision issues. Decimal(0.1) != Decimal("0.1").
        """
        return cls(
            symbol=data["symbol"],
            quantity=Decimal(str(data["qty"])),
            avg_entry_price=Decimal(str(data["avg_entry_price"])),
            current_price=Decimal(str(data["current_price"])),
            market_value=Decimal(str(data["market_value"])),
            unrealized_pnl=Decimal(str(data["unrealized_pl"])),
            unrealized_pnl_pct=Decimal(str(data["unrealized_plpc"])),
            cost_basis=Decimal(str(data["cost_basis"])),
            asset_class=data["asset_class"],
            asset_id=data["asset_id"],
            side=data["side"],
            exchange=data["exchange"],
            change_today=Decimal(str(data["change_today"])) if data.get("change_today") else None,
        )

    def to_position(self) -> Position:
        """Convert to internal Position model."""
        qty = self.quantity if self.side == "long" else -self.quantity
        return Position(
            symbol=self.symbol,
            quantity=qty,
            avg_entry_price=self.avg_entry_price,
            current_price=self.current_price,
            cost_basis=self.cost_basis,
            market_value=self.market_value,
            unrealized_pnl=self.unrealized_pnl,
        )


@dataclass
class AlpacaOrder:
    """Alpaca order data."""

    order_id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    time_in_force: str
    quantity: Decimal
    filled_qty: Decimal
    filled_avg_price: Decimal | None
    limit_price: Decimal | None
    stop_price: Decimal | None
    status: str
    created_at: datetime
    updated_at: datetime
    submitted_at: datetime | None
    filled_at: datetime | None
    expired_at: datetime | None
    cancelled_at: datetime | None
    asset_class: str
    order_class: str
    extended_hours: bool
    legs: list["AlpacaOrder"] | None = None

    @classmethod
    def from_alpaca(cls, data: dict[str, Any]) -> "AlpacaOrder":
        """Create from Alpaca API response."""

        def parse_time(ts: str | None) -> datetime | None:
            if ts is None:
                return None
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))

        legs = None
        if data.get("legs"):
            legs = [cls.from_alpaca(leg) for leg in data["legs"]]

        # CRITICAL FIX: Convert through str() to avoid float precision issues
        # Decimal(0.1) != Decimal("0.1") due to floating point representation
        return cls(
            order_id=data["id"],
            client_order_id=data["client_order_id"],
            symbol=data["symbol"],
            side=data["side"],
            order_type=data["type"],
            time_in_force=data["time_in_force"],
            quantity=Decimal(str(data["qty"])),
            filled_qty=Decimal(str(data["filled_qty"])),
            filled_avg_price=Decimal(str(data["filled_avg_price"])) if data.get("filled_avg_price") else None,
            limit_price=Decimal(str(data["limit_price"])) if data.get("limit_price") else None,
            stop_price=Decimal(str(data["stop_price"])) if data.get("stop_price") else None,
            status=data["status"],
            created_at=parse_time(data["created_at"]) or datetime.now(timezone.utc),
            updated_at=parse_time(data["updated_at"]) or datetime.now(timezone.utc),
            submitted_at=parse_time(data.get("submitted_at")),
            filled_at=parse_time(data.get("filled_at")),
            expired_at=parse_time(data.get("expired_at")),
            cancelled_at=parse_time(data.get("cancelled_at")),
            asset_class=data.get("asset_class", "us_equity"),
            order_class=data.get("order_class", "simple"),
            extended_hours=data.get("extended_hours", False),
            legs=legs,
        )

    def to_order(self) -> Order:
        """Convert to internal Order model."""
        status_map = {
            "new": OrderStatus.SUBMITTED,
            "accepted": OrderStatus.ACCEPTED,
            "pending_new": OrderStatus.PENDING,
            "filled": OrderStatus.FILLED,
            "partially_filled": OrderStatus.PARTIAL_FILLED,
            "canceled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
            "pending_cancel": OrderStatus.SUBMITTED,
            "pending_replace": OrderStatus.SUBMITTED,
        }

        type_map = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP,
            "stop_limit": OrderType.STOP_LIMIT,
            "trailing_stop": OrderType.TRAILING_STOP,
        }

        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
            "opg": TimeInForce.OPG,
            "cls": TimeInForce.CLS,
        }

        return Order(
            client_order_id=self.client_order_id,
            broker_order_id=self.order_id,
            symbol=self.symbol,
            side=OrderSide.BUY if self.side == "buy" else OrderSide.SELL,
            order_type=type_map.get(self.order_type, OrderType.MARKET),
            quantity=self.quantity,
            limit_price=self.limit_price,
            stop_price=self.stop_price,
            time_in_force=tif_map.get(self.time_in_force, TimeInForce.DAY),
            status=status_map.get(self.status, OrderStatus.PENDING),
            filled_qty=self.filled_qty,
            filled_avg_price=self.filled_avg_price,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class AlpacaClient:
    """Alpaca API client for trading operations.

    Provides methods for:
    - Account information retrieval
    - Order submission, modification, and cancellation
    - Position management
    - Market data access

    Includes automatic rate limiting, retry logic, and error handling.
    """

    # API Base URLs
    PAPER_BASE_URL = "https://paper-api.alpaca.markets"
    LIVE_BASE_URL = "https://api.alpaca.markets"
    DATA_BASE_URL = "https://data.alpaca.markets"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        environment: TradingEnvironment = TradingEnvironment.PAPER,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize Alpaca client.

        Args:
            api_key: Alpaca API key (or from ALPACA_API_KEY env var).
            api_secret: Alpaca API secret (or from ALPACA_API_SECRET env var).
            environment: Paper or live trading environment.
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Base delay between retries (exponential backoff).
        """
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("ALPACA_API_SECRET", "")
        self.environment = environment
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Set base URL based on environment
        self.base_url = (
            self.PAPER_BASE_URL if environment == TradingEnvironment.PAPER else self.LIVE_BASE_URL
        )

        # Rate limiter
        self.rate_limiter = RateLimiter()

        # Connection state
        self._connected = False
        self._session: Any = None  # aiohttp.ClientSession

        # WebSocket callbacks
        self._trade_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._account_callbacks: list[Callable[[dict[str, Any]], None]] = []

        # WebSocket streaming state
        self._stream_running = False

        logger.info(f"AlpacaClient initialized for {environment.value} trading")

    @property
    def headers(self) -> dict[str, str]:
        """Get authentication headers."""
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        }

    async def connect(self) -> None:
        """Establish connection to Alpaca API."""
        try:
            import aiohttp

            self._session = aiohttp.ClientSession(headers=self.headers)
            # Verify connection by getting account
            await self.get_account()
            self._connected = True
            logger.info(f"Connected to Alpaca {self.environment.value} API")
        except ImportError:
            # aiohttp not available, will use sync mode
            logger.warning("aiohttp not available, async operations will fail")
            self._connected = True
        except Exception as e:
            raise BrokerConnectionError(
                f"Failed to connect to Alpaca: {e}",
                broker="alpaca",
            )

    async def disconnect(self) -> None:
        """Close connection to Alpaca API."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        logger.info("Disconnected from Alpaca API")

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        """Make an API request with rate limiting and retry logic.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.).
            endpoint: API endpoint path.
            params: Query parameters.
            json_data: JSON body data.
            base_url: Override base URL.

        Returns:
            API response as dictionary.

        Raises:
            BrokerConnectionError: If request fails after retries.
        """
        await self.rate_limiter.acquire()

        url = f"{base_url or self.base_url}{endpoint}"
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                if self._session is None:
                    # Fallback to synchronous requests
                    return await self._sync_request(method, url, params, json_data)

                async with self._session.request(
                    method,
                    url,
                    params=params,
                    json=json_data,
                ) as response:
                    if response.status == 429:
                        # Rate limited - wait and retry
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue

                    response_data = await response.json()

                    if response.status >= 400:
                        error_msg = response_data.get("message", str(response_data))
                        if response.status == 403:
                            raise BrokerConnectionError(
                                f"Authentication failed: {error_msg}",
                                broker="alpaca",
                            )
                        elif response.status == 422:
                            raise OrderSubmissionError(
                                f"Invalid order: {error_msg}",
                                reason=error_msg,
                            )
                        else:
                            raise BrokerConnectionError(
                                f"API error ({response.status}): {error_msg}",
                                broker="alpaca",
                            )

                    return response_data

            except (BrokerConnectionError, OrderSubmissionError):
                raise
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(f"Request failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)

        raise BrokerConnectionError(
            f"Request failed after {self.max_retries} attempts: {last_error}",
            broker="alpaca",
        )

    async def _sync_request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None,
        json_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Make synchronous HTTP request (fallback when aiohttp unavailable).

        P0-3 FIX (January 2026 Audit): Wrap blocking urllib call in asyncio.to_thread()
        to prevent blocking the event loop. Previously this blocked the entire event
        loop which could cause timeouts and missed updates in live trading.
        """
        import json
        import urllib.request

        def _blocking_request() -> dict[str, Any]:
            """Execute blocking HTTP request in thread pool."""
            req_url = url
            if params:
                query = "&".join(f"{k}={v}" for k, v in params.items())
                req_url = f"{url}?{query}"

            data = json.dumps(json_data).encode() if json_data else None
            request = urllib.request.Request(req_url, data=data, method=method)

            for key, value in self.headers.items():
                request.add_header(key, value)

            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode())

        # Run blocking code in thread pool to avoid blocking event loop
        return await asyncio.to_thread(_blocking_request)

    # Account Methods

    async def get_account(self) -> AccountInfo:
        """Get account information.

        Returns:
            AccountInfo with current account state.
        """
        data = await self._request("GET", "/v2/account")
        return AccountInfo.from_alpaca(data)

    async def get_portfolio_history(
        self,
        period: str = "1M",
        timeframe: str = "1D",
        date_start: str | None = None,
        date_end: str | None = None,
        extended_hours: bool = False,
    ) -> dict[str, Any]:
        """Get portfolio performance history.

        Args:
            period: Period of history (1D, 1W, 1M, 3M, 1A, all).
            timeframe: Timeframe for data points (1Min, 5Min, 15Min, 1H, 1D).
            date_start: Start date (YYYY-MM-DD).
            date_end: End date (YYYY-MM-DD).
            extended_hours: Include extended hours data.

        Returns:
            Portfolio history data.
        """
        params: dict[str, Any] = {
            "period": period,
            "timeframe": timeframe,
            "extended_hours": str(extended_hours).lower(),
        }
        if date_start:
            params["date_start"] = date_start
        if date_end:
            params["date_end"] = date_end

        return await self._request("GET", "/v2/account/portfolio/history", params=params)

    # Order Methods

    def _is_transient_error(self, error: Exception) -> bool:
        """Check if an error is transient and can be retried.

        Transient errors include:
        - Network timeouts
        - Connection errors
        - Rate limit errors (429)
        - Server errors (5xx)

        Non-transient errors (should NOT be retried):
        - Authentication errors (401, 403)
        - Invalid order errors (422)
        - Insufficient funds
        - Order already exists

        Args:
            error: The exception to check.

        Returns:
            True if the error is transient and can be retried.
        """
        error_str = str(error).lower()

        # Non-transient errors - do not retry
        non_transient_patterns = [
            "insufficient",
            "authentication",
            "forbidden",
            "invalid order",
            "already exists",
            "not found",
            "422",
            "401",
            "403",
        ]

        for pattern in non_transient_patterns:
            if pattern in error_str:
                return False

        # Transient error patterns - safe to retry
        transient_patterns = [
            "timeout",
            "connection",
            "network",
            "rate limit",
            "429",
            "500",
            "502",
            "503",
            "504",
            "temporary",
            "unavailable",
        ]

        for pattern in transient_patterns:
            if pattern in error_str:
                return True

        # For generic BrokerConnectionError, assume transient unless specifically non-transient
        if isinstance(error, BrokerConnectionError):
            return True

        return False

    async def submit_order(
        self,
        symbol: str,
        qty: Decimal,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        time_in_force: TimeInForce = TimeInForce.DAY,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
        client_order_id: str | None = None,
        extended_hours: bool = False,
        order_class: OrderClass = OrderClass.SIMPLE,
        take_profit: dict[str, Any] | None = None,
        stop_loss: dict[str, Any] | None = None,
    ) -> AlpacaOrder:
        """Submit an order with retry logic for transient failures.

        Implements exponential backoff retry for transient errors:
        - Max 3 retries with backoff: 1s, 2s, 4s
        - Only retries on transient failures (network issues, rate limits, server errors)
        - Does NOT retry on non-transient errors (insufficient funds, invalid orders)

        Args:
            symbol: Stock symbol.
            qty: Order quantity.
            side: Buy or sell.
            order_type: Market, limit, stop, etc.
            time_in_force: DAY, GTC, IOC, etc.
            limit_price: Limit price for limit orders.
            stop_price: Stop price for stop orders.
            client_order_id: Client-specified order ID.
            extended_hours: Allow extended hours trading.
            order_class: Simple, bracket, OTO, OCO.
            take_profit: Take profit leg for bracket orders.
            stop_loss: Stop loss leg for bracket orders.

        Returns:
            Submitted order.

        Raises:
            OrderSubmissionError: If order submission fails after all retries.
            InsufficientFundsError: If insufficient buying power (not retried).
        """
        # Order submission retry configuration
        ORDER_MAX_RETRIES = 3
        ORDER_BASE_DELAY = 1.0  # Base delay in seconds (1s, 2s, 4s)

        type_map = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop",
            OrderType.STOP_LIMIT: "stop_limit",
            OrderType.TRAILING_STOP: "trailing_stop",
        }

        tif_map = {
            TimeInForce.DAY: "day",
            TimeInForce.GTC: "gtc",
            TimeInForce.IOC: "ioc",
            TimeInForce.FOK: "fok",
            TimeInForce.OPG: "opg",
            TimeInForce.CLS: "cls",
        }

        order_data: dict[str, Any] = {
            "symbol": symbol.upper(),
            "qty": str(qty),
            "side": side.value.lower(),
            "type": type_map[order_type],
            "time_in_force": tif_map[time_in_force],
            "extended_hours": extended_hours,
        }

        if limit_price is not None:
            order_data["limit_price"] = str(limit_price)
        if stop_price is not None:
            order_data["stop_price"] = str(stop_price)
        if client_order_id:
            order_data["client_order_id"] = client_order_id
        if order_class != OrderClass.SIMPLE:
            order_data["order_class"] = order_class.value
        if take_profit:
            order_data["take_profit"] = take_profit
        if stop_loss:
            order_data["stop_loss"] = stop_loss

        last_error: Exception | None = None

        for attempt in range(ORDER_MAX_RETRIES):
            try:
                data = await self._request("POST", "/v2/orders", json_data=order_data)
                order = AlpacaOrder.from_alpaca(data)
                if attempt > 0:
                    logger.info(f"Order submitted after {attempt + 1} attempts: {order.order_id} {symbol} {side.value} {qty}")
                else:
                    logger.info(f"Order submitted: {order.order_id} {symbol} {side.value} {qty}")
                return order

            except InsufficientFundsError:
                # Non-transient error - do not retry
                logger.error(f"Insufficient funds for order: {symbol} {side.value} {qty}")
                raise

            except (BrokerConnectionError, OrderSubmissionError) as e:
                last_error = e

                # Check if error is transient and can be retried
                if not self._is_transient_error(e):
                    logger.error(f"Non-transient error submitting order, not retrying: {e}")
                    if "insufficient" in str(e).lower():
                        raise InsufficientFundsError(
                            f"Insufficient funds for order: {e}",
                            symbol=symbol,
                        )
                    raise OrderSubmissionError(
                        f"Failed to submit order: {e}",
                        symbol=symbol,
                    )

                # Transient error - retry with exponential backoff
                if attempt < ORDER_MAX_RETRIES - 1:
                    delay = ORDER_BASE_DELAY * (2 ** attempt)  # 1s, 2s, 4s
                    logger.warning(
                        f"Transient error submitting order (attempt {attempt + 1}/{ORDER_MAX_RETRIES}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Order submission failed after {ORDER_MAX_RETRIES} attempts: {e}"
                    )

            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error submitting order: {e}")
                # For unexpected errors, do not retry
                raise OrderSubmissionError(
                    f"Failed to submit order: {e}",
                    symbol=symbol,
                )

        # All retries exhausted
        raise OrderSubmissionError(
            f"Failed to submit order after {ORDER_MAX_RETRIES} attempts: {last_error}",
            symbol=symbol,
        )

    async def get_order(self, order_id: str, by_client_id: bool = False) -> AlpacaOrder:
        """Get order by ID.

        Args:
            order_id: Order ID or client order ID.
            by_client_id: If True, lookup by client order ID.

        Returns:
            Order details.
        """
        endpoint = f"/v2/orders:by_client_order_id" if by_client_id else f"/v2/orders/{order_id}"
        params = {"client_order_id": order_id} if by_client_id else None
        data = await self._request("GET", endpoint, params=params)
        return AlpacaOrder.from_alpaca(data)

    async def get_orders(
        self,
        status: str = "open",
        symbols: list[str] | None = None,
        limit: int = 500,
        after: datetime | None = None,
        until: datetime | None = None,
        direction: str = "desc",
    ) -> list[AlpacaOrder]:
        """Get orders with optional filtering.

        Args:
            status: open, closed, all.
            symbols: Filter by symbols.
            limit: Maximum orders to return.
            after: Get orders after this time.
            until: Get orders until this time.
            direction: Sort direction (asc, desc).

        Returns:
            List of orders.
        """
        params: dict[str, Any] = {
            "status": status,
            "limit": limit,
            "direction": direction,
        }
        if symbols:
            params["symbols"] = ",".join(symbols)
        if after:
            params["after"] = after.isoformat()
        if until:
            params["until"] = until.isoformat()

        data = await self._request("GET", "/v2/orders", params=params)
        return [AlpacaOrder.from_alpaca(o) for o in data]

    async def cancel_order(self, order_id: str) -> None:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel.

        Raises:
            OrderCancellationError: If cancellation fails.
        """
        try:
            await self._request("DELETE", f"/v2/orders/{order_id}")
            logger.info(f"Order cancelled: {order_id}")
        except BrokerConnectionError as e:
            raise OrderCancellationError(
                f"Failed to cancel order: {e}",
                order_id=order_id,
            )

    async def cancel_all_orders(self) -> int:
        """Cancel all open orders.

        Returns:
            Number of orders cancelled.
        """
        data = await self._request("DELETE", "/v2/orders")
        count = len(data) if isinstance(data, list) else 0
        logger.info(f"Cancelled {count} orders")
        return count

    async def replace_order(
        self,
        order_id: str,
        qty: Decimal | None = None,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
        time_in_force: TimeInForce | None = None,
        client_order_id: str | None = None,
    ) -> AlpacaOrder:
        """Replace (modify) an existing order.

        Args:
            order_id: Order ID to replace.
            qty: New quantity.
            limit_price: New limit price.
            stop_price: New stop price.
            time_in_force: New time in force.
            client_order_id: New client order ID.

        Returns:
            Replaced order.
        """
        tif_map = {
            TimeInForce.DAY: "day",
            TimeInForce.GTC: "gtc",
            TimeInForce.IOC: "ioc",
            TimeInForce.FOK: "fok",
            TimeInForce.OPG: "opg",
            TimeInForce.CLS: "cls",
        }

        order_data: dict[str, Any] = {}
        if qty is not None:
            order_data["qty"] = str(qty)
        if limit_price is not None:
            order_data["limit_price"] = str(limit_price)
        if stop_price is not None:
            order_data["stop_price"] = str(stop_price)
        if time_in_force is not None:
            order_data["time_in_force"] = tif_map[time_in_force]
        if client_order_id:
            order_data["client_order_id"] = client_order_id

        data = await self._request("PATCH", f"/v2/orders/{order_id}", json_data=order_data)
        return AlpacaOrder.from_alpaca(data)

    # Position Methods

    async def get_positions(self) -> list[AlpacaPosition]:
        """Get all open positions.

        Returns:
            List of positions.
        """
        data = await self._request("GET", "/v2/positions")
        return [AlpacaPosition.from_alpaca(p) for p in data]

    async def get_position(self, symbol: str) -> AlpacaPosition | None:
        """Get position for a specific symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Position or None if no position.
        """
        try:
            data = await self._request("GET", f"/v2/positions/{symbol.upper()}")
            return AlpacaPosition.from_alpaca(data)
        except BrokerConnectionError as e:
            if "not found" in str(e).lower() or "404" in str(e):
                return None
            raise

    async def close_position(
        self,
        symbol: str,
        qty: Decimal | None = None,
        percentage: float | None = None,
    ) -> AlpacaOrder:
        """Close a position.

        Args:
            symbol: Stock symbol.
            qty: Quantity to close (None = all).
            percentage: Percentage to close (0-100).

        Returns:
            Close order.
        """
        params: dict[str, Any] = {}
        if qty is not None:
            params["qty"] = str(qty)
        elif percentage is not None:
            params["percentage"] = str(percentage)

        data = await self._request(
            "DELETE",
            f"/v2/positions/{symbol.upper()}",
            params=params if params else None,
        )
        return AlpacaOrder.from_alpaca(data)

    async def close_all_positions(self, cancel_orders: bool = True) -> list[AlpacaOrder]:
        """Close all positions.

        Args:
            cancel_orders: Also cancel pending orders.

        Returns:
            List of close orders.
        """
        params = {"cancel_orders": str(cancel_orders).lower()}
        data = await self._request("DELETE", "/v2/positions", params=params)
        return [AlpacaOrder.from_alpaca(o) for o in data] if isinstance(data, list) else []

    # Market Data Methods

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 1000,
        adjustment: str = "raw",
    ) -> list[dict[str, Any]]:
        """Get historical bars.

        Args:
            symbol: Stock symbol.
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day).
            start: Start datetime.
            end: End datetime.
            limit: Maximum bars to return.
            adjustment: Price adjustment (raw, split, dividend, all).

        Returns:
            List of bar data.
        """
        params: dict[str, Any] = {
            "timeframe": timeframe,
            "limit": limit,
            "adjustment": adjustment,
        }
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()

        data = await self._request(
            "GET",
            f"/v2/stocks/{symbol.upper()}/bars",
            params=params,
            base_url=self.DATA_BASE_URL,
        )
        return data.get("bars", [])

    async def get_latest_bar(self, symbol: str) -> dict[str, Any] | None:
        """Get the latest bar for a symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Latest bar data or None.
        """
        data = await self._request(
            "GET",
            f"/v2/stocks/{symbol.upper()}/bars/latest",
            base_url=self.DATA_BASE_URL,
        )
        return data.get("bar")

    async def get_snapshot(self, symbol: str) -> dict[str, Any]:
        """Get market snapshot for a symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Snapshot data (latest trade, quote, bar).
        """
        return await self._request(
            "GET",
            f"/v2/stocks/{symbol.upper()}/snapshot",
            base_url=self.DATA_BASE_URL,
        )

    async def get_trades(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get historical trades.

        Args:
            symbol: Stock symbol.
            start: Start datetime.
            end: End datetime.
            limit: Maximum trades to return.

        Returns:
            List of trade data.
        """
        params: dict[str, Any] = {"limit": limit}
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()

        data = await self._request(
            "GET",
            f"/v2/stocks/{symbol.upper()}/trades",
            params=params,
            base_url=self.DATA_BASE_URL,
        )
        return data.get("trades", [])

    # Clock and Calendar

    async def get_clock(self) -> dict[str, Any]:
        """Get market clock (open/close times, trading status).

        Returns:
            Clock data.
        """
        return await self._request("GET", "/v2/clock")

    async def is_market_open(self) -> bool:
        """Check if market is currently open.

        Returns:
            True if market is open.
        """
        clock = await self.get_clock()
        return clock.get("is_open", False)

    async def get_calendar(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get market calendar.

        Args:
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).

        Returns:
            List of calendar days.
        """
        params: dict[str, Any] = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        return await self._request("GET", "/v2/calendar", params=params)

    # WebSocket Callback Registration

    def on_trade_update(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register callback for trade updates.

        Args:
            callback: Function to call on trade updates.
        """
        self._trade_callbacks.append(callback)

    def on_account_update(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register callback for account updates.

        Args:
            callback: Function to call on account updates.
        """
        self._account_callbacks.append(callback)

    def _dispatch_trade_update(self, data: dict[str, Any]) -> None:
        """Dispatch trade update to registered callbacks."""
        for callback in self._trade_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")

    def _dispatch_account_update(self, data: dict[str, Any]) -> None:
        """Dispatch account update to registered callbacks."""
        for callback in self._account_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Account callback error: {e}")

    # WebSocket Streaming Methods

    async def start_trade_stream(self) -> None:
        """Start WebSocket stream for real-time trade/order updates.

        Connects to Alpaca's trading stream to receive real-time updates on:
        - Order status changes (new, fill, partial_fill, canceled, rejected, expired)
        - Account updates

        The stream will automatically reconnect on disconnection.
        """
        try:
            import websockets
            import json as json_module
        except ImportError:
            logger.error("websockets package required for trade streaming")
            return

        stream_url = (
            "wss://paper-api.alpaca.markets/stream"
            if self.environment == TradingEnvironment.PAPER
            else "wss://api.alpaca.markets/stream"
        )

        logger.info(f"Connecting to trade stream: {stream_url}")

        self._stream_running = True
        reconnect_delay = 1.0
        max_reconnect_delay = 60.0

        while self._stream_running:
            try:
                async with websockets.connect(
                    stream_url,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    # Wait for welcome
                    welcome = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    logger.debug(f"Trade stream welcome: {welcome}")

                    # Authenticate
                    auth_msg = json_module.dumps({
                        "action": "auth",
                        "key": self.api_key,
                        "secret": self.api_secret,
                    })
                    await ws.send(auth_msg)

                    auth_response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    auth_data = json_module.loads(auth_response)
                    logger.debug(f"Trade stream auth: {auth_data}")

                    # Subscribe to trade updates
                    listen_msg = json_module.dumps({
                        "action": "listen",
                        "data": {"streams": ["trade_updates"]},
                    })
                    await ws.send(listen_msg)

                    logger.info("Trade stream connected and authenticated")
                    reconnect_delay = 1.0  # Reset on successful connection

                    # Handle messages
                    async for message in ws:
                        try:
                            data = json_module.loads(message)
                            await self._handle_stream_message(data)
                        except json_module.JSONDecodeError:
                            logger.error(f"Invalid JSON in trade stream: {message}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._stream_running:
                    logger.warning(f"Trade stream error: {e}, reconnecting in {reconnect_delay}s")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

        logger.info("Trade stream stopped")

    async def _handle_stream_message(self, data: dict[str, Any]) -> None:
        """Handle incoming trade stream message."""
        stream = data.get("stream")

        if stream == "trade_updates":
            trade_data = data.get("data", {})
            event = trade_data.get("event")

            logger.info(
                f"Trade update: {event} - "
                f"{trade_data.get('order', {}).get('symbol')} "
                f"qty={trade_data.get('order', {}).get('qty')}"
            )

            # Dispatch to callbacks
            self._dispatch_trade_update(trade_data)

        elif stream == "authorization":
            status = data.get("data", {}).get("status")
            if status != "authorized":
                logger.error(f"Trade stream authorization failed: {status}")

        elif stream == "listening":
            streams = data.get("data", {}).get("streams", [])
            logger.info(f"Trade stream subscribed to: {streams}")

    def stop_trade_stream(self) -> None:
        """Stop the trade stream."""
        self._stream_running = False

    async def start_with_stream(self) -> None:
        """Connect to Alpaca API and start trade stream.

        Convenience method that connects and starts the WebSocket stream
        for real-time trade updates.
        """
        await self.connect()
        # Start stream in background task
        asyncio.create_task(self.start_trade_stream())


# Convenience functions for order creation


def create_market_order(
    symbol: str,
    qty: Decimal,
    side: OrderSide,
    time_in_force: TimeInForce = TimeInForce.DAY,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    """Create market order parameters."""
    return {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "order_type": OrderType.MARKET,
        "time_in_force": time_in_force,
        "client_order_id": client_order_id,
    }


def create_limit_order(
    symbol: str,
    qty: Decimal,
    side: OrderSide,
    limit_price: Decimal,
    time_in_force: TimeInForce = TimeInForce.DAY,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    """Create limit order parameters."""
    return {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "order_type": OrderType.LIMIT,
        "limit_price": limit_price,
        "time_in_force": time_in_force,
        "client_order_id": client_order_id,
    }


def create_stop_order(
    symbol: str,
    qty: Decimal,
    side: OrderSide,
    stop_price: Decimal,
    time_in_force: TimeInForce = TimeInForce.DAY,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    """Create stop order parameters."""
    return {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "order_type": OrderType.STOP,
        "stop_price": stop_price,
        "time_in_force": time_in_force,
        "client_order_id": client_order_id,
    }


def create_bracket_order(
    symbol: str,
    qty: Decimal,
    side: OrderSide,
    take_profit_price: Decimal,
    stop_loss_price: Decimal,
    limit_price: Decimal | None = None,
    time_in_force: TimeInForce = TimeInForce.DAY,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    """Create bracket order parameters."""
    order_params: dict[str, Any] = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "order_type": OrderType.LIMIT if limit_price else OrderType.MARKET,
        "time_in_force": time_in_force,
        "order_class": OrderClass.BRACKET,
        "take_profit": {"limit_price": str(take_profit_price)},
        "stop_loss": {"stop_price": str(stop_loss_price)},
        "client_order_id": client_order_id,
    }
    if limit_price:
        order_params["limit_price"] = limit_price
    return order_params
