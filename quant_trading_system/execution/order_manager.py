"""
Order management module.

Handles the full order lifecycle:
- Order creation and validation
- Order submission and monitoring
- Order modification and cancellation
- Fill tracking and reconciliation
- Smart order routing

Integrates with Alpaca client for execution and event bus for notifications.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable
from uuid import UUID, uuid4

from quant_trading_system.core.data_types import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    TimeInForce,
)
from quant_trading_system.core.events import (
    Event,
    EventBus,
    EventType,
    create_order_event,
)
from quant_trading_system.core.exceptions import (
    ExecutionError,
    InsufficientFundsError,
    OrderCancellationError,
    OrderSubmissionError,
)
from quant_trading_system.execution.alpaca_client import (
    AlpacaClient,
    AlpacaOrder,
    OrderClass,
)

logger = logging.getLogger(__name__)


class OrderState(str, Enum):
    """Internal order state tracking."""

    CREATED = "created"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    PENDING = "pending"
    ACCEPTED = "accepted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


class OrderPriority(int, Enum):
    """Order priority for submission queue."""

    CRITICAL = 0  # Risk reduction orders
    HIGH = 1  # Stop losses, take profits
    NORMAL = 2  # Regular orders
    LOW = 3  # Rebalancing orders


@dataclass
class OrderRequest:
    """Order request before submission."""

    request_id: UUID = field(default_factory=uuid4)
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: Decimal = Decimal("0")
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    signal_id: UUID | None = None
    strategy_id: str = ""
    priority: OrderPriority = OrderPriority.NORMAL
    extended_hours: bool = False
    take_profit_price: Decimal | None = None
    stop_loss_price: Decimal | None = None
    notes: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ManagedOrder:
    """Order with full lifecycle tracking."""

    order: Order
    request: OrderRequest
    state: OrderState = OrderState.CREATED
    broker_order: AlpacaOrder | None = None
    submission_time: datetime | None = None
    fill_time: datetime | None = None
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retry_count: int = 0
    error_message: str | None = None
    fill_events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.state in (
            OrderState.CREATED,
            OrderState.VALIDATED,
            OrderState.SUBMITTED,
            OrderState.PENDING,
            OrderState.ACCEPTED,
            OrderState.PARTIAL_FILLED,
        )

    @property
    def is_terminal(self) -> bool:
        """Check if order has reached terminal state."""
        return self.state in (
            OrderState.FILLED,
            OrderState.REJECTED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
            OrderState.FAILED,
        )

    @property
    def fill_progress(self) -> float:
        """Get fill progress as percentage."""
        if self.order.quantity == 0:
            return 0.0
        return float(self.order.filled_qty / self.order.quantity) * 100


@dataclass
class OrderValidationResult:
    """Result of order validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class OrderValidator:
    """Validates orders before submission."""

    def __init__(
        self,
        max_order_value: Decimal = Decimal("100000"),
        max_position_pct: float = 0.10,
        min_order_value: Decimal = Decimal("1"),
        allowed_symbols: set[str] | None = None,
        blocked_symbols: set[str] | None = None,
    ) -> None:
        """Initialize validator.

        Args:
            max_order_value: Maximum order value in dollars.
            max_position_pct: Maximum position as percentage of portfolio.
            min_order_value: Minimum order value.
            allowed_symbols: If set, only these symbols allowed.
            blocked_symbols: Symbols that cannot be traded.
        """
        self.max_order_value = max_order_value
        self.max_position_pct = max_position_pct
        self.min_order_value = min_order_value
        self.allowed_symbols = allowed_symbols
        self.blocked_symbols = blocked_symbols or set()

    def validate(
        self,
        request: OrderRequest,
        portfolio: Portfolio | None = None,
        current_price: Decimal | None = None,
    ) -> OrderValidationResult:
        """Validate an order request.

        Args:
            request: Order request to validate.
            portfolio: Current portfolio state.
            current_price: Current market price.

        Returns:
            Validation result with errors/warnings.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Symbol validation
        if not request.symbol:
            errors.append("Symbol is required")
        elif self.allowed_symbols and request.symbol not in self.allowed_symbols:
            errors.append(f"Symbol {request.symbol} not in allowed list")
        elif request.symbol in self.blocked_symbols:
            errors.append(f"Symbol {request.symbol} is blocked")

        # Quantity validation
        if request.quantity <= 0:
            errors.append("Quantity must be positive")

        # Price validation for limit orders
        if request.order_type == OrderType.LIMIT and request.limit_price is None:
            errors.append("Limit price required for limit orders")
        if request.order_type == OrderType.STOP and request.stop_price is None:
            errors.append("Stop price required for stop orders")
        if request.order_type == OrderType.STOP_LIMIT:
            if request.limit_price is None:
                errors.append("Limit price required for stop-limit orders")
            if request.stop_price is None:
                errors.append("Stop price required for stop-limit orders")

        # Value checks if price available
        if current_price and current_price > 0:
            order_value = request.quantity * current_price

            if order_value < self.min_order_value:
                errors.append(f"Order value ${order_value:.2f} below minimum ${self.min_order_value}")

            if order_value > self.max_order_value:
                errors.append(f"Order value ${order_value:.2f} exceeds maximum ${self.max_order_value}")

            # Portfolio checks
            if portfolio and portfolio.equity > 0:
                position_pct = float(order_value / portfolio.equity)
                if position_pct > self.max_position_pct:
                    errors.append(
                        f"Order would create position of {position_pct:.1%}, "
                        f"exceeds maximum {self.max_position_pct:.1%}"
                    )

                # Buying power check for buys
                if request.side == OrderSide.BUY and order_value > portfolio.buying_power:
                    errors.append(
                        f"Insufficient buying power: need ${order_value:.2f}, "
                        f"have ${portfolio.buying_power:.2f}"
                    )

        # Warnings
        if request.extended_hours:
            warnings.append("Extended hours trading has reduced liquidity")
        if request.time_in_force == TimeInForce.GTC:
            warnings.append("GTC orders remain open until filled or cancelled")

        return OrderValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


class SmartOrderRouter:
    """Routes orders to optimal execution strategy."""

    def __init__(
        self,
        twap_threshold: Decimal = Decimal("10000"),
        vwap_threshold: Decimal = Decimal("50000"),
        urgency_threshold: float = 0.8,
    ) -> None:
        """Initialize router.

        Args:
            twap_threshold: Order value threshold for TWAP.
            vwap_threshold: Order value threshold for VWAP.
            urgency_threshold: Urgency level above which to use market orders.
        """
        self.twap_threshold = twap_threshold
        self.vwap_threshold = vwap_threshold
        self.urgency_threshold = urgency_threshold

    def select_algorithm(
        self,
        request: OrderRequest,
        order_value: Decimal,
        urgency: float = 0.5,
        volatility: float = 0.0,
    ) -> str:
        """Select execution algorithm.

        Args:
            request: Order request.
            order_value: Estimated order value.
            urgency: Urgency level (0-1).
            volatility: Current market volatility.

        Returns:
            Algorithm name: "market", "limit", "twap", "vwap", "adaptive".
        """
        # High urgency or small orders -> market
        if urgency >= self.urgency_threshold or request.priority == OrderPriority.CRITICAL:
            return "market"

        # Large orders need algorithmic execution
        if order_value >= self.vwap_threshold:
            return "vwap"
        elif order_value >= self.twap_threshold:
            return "twap"

        # High volatility suggests limit orders
        if volatility > 0.03:  # 3% daily volatility
            return "limit"

        # Default to market for simplicity
        return "market"

    def calculate_limit_price(
        self,
        side: OrderSide,
        current_price: Decimal,
        spread: Decimal = Decimal("0.01"),
        buffer_pct: float = 0.001,
    ) -> Decimal:
        """Calculate limit price with buffer.

        Args:
            side: Buy or sell.
            current_price: Current market price.
            spread: Bid-ask spread.
            buffer_pct: Buffer percentage from mid.

        Returns:
            Calculated limit price.
        """
        buffer = current_price * Decimal(str(buffer_pct))

        if side == OrderSide.BUY:
            # Buy slightly above mid for better fill probability
            return current_price + buffer
        else:
            # Sell slightly below mid
            return current_price - buffer


class OrderManager:
    """Manages order lifecycle from creation to completion.

    Handles:
    - Order validation and risk checks
    - Order submission with smart routing
    - Order monitoring and updates
    - Order modification and cancellation
    - Fill tracking and reconciliation
    """

    def __init__(
        self,
        client: AlpacaClient,
        event_bus: EventBus | None = None,
        validator: OrderValidator | None = None,
        router: SmartOrderRouter | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize order manager.

        Args:
            client: Alpaca client for execution.
            event_bus: Event bus for order events.
            validator: Order validator.
            router: Smart order router.
            max_retries: Maximum retry attempts.
            retry_delay: Base delay between retries.
        """
        self.client = client
        self.event_bus = event_bus
        self.validator = validator or OrderValidator()
        self.router = router or SmartOrderRouter()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Order tracking
        self._orders: dict[UUID, ManagedOrder] = {}
        self._client_id_map: dict[str, UUID] = {}
        self._broker_id_map: dict[str, UUID] = {}

        # Thread safety - protect shared state from race conditions
        self._lock = asyncio.Lock()

        # Callbacks
        self._fill_callbacks: list[Callable[[ManagedOrder], None]] = []
        self._rejection_callbacks: list[Callable[[ManagedOrder], None]] = []

        # Monitoring
        self._monitor_task: asyncio.Task | None = None
        self._running = False

        # Register for trade updates
        client.on_trade_update(self._handle_trade_update)

    async def start(self) -> None:
        """Start order monitoring."""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_orders())
        logger.info("OrderManager started")

    async def stop(self) -> None:
        """Stop order monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("OrderManager stopped")

    def create_order(
        self,
        request: OrderRequest,
        portfolio: Portfolio | None = None,
        current_price: Decimal | None = None,
    ) -> ManagedOrder:
        """Create a new managed order.

        Args:
            request: Order request.
            portfolio: Current portfolio for validation.
            current_price: Current price for validation.

        Returns:
            Created managed order.

        Raises:
            OrderSubmissionError: If validation fails.
        """
        # Validate order
        validation = self.validator.validate(request, portfolio, current_price)
        if not validation.is_valid:
            raise OrderSubmissionError(
                f"Order validation failed: {'; '.join(validation.errors)}",
                symbol=request.symbol,
            )

        # Log warnings
        for warning in validation.warnings:
            logger.warning(f"Order warning: {warning}")

        # Create internal order
        order = Order(
            client_order_id=f"OM-{request.request_id.hex[:8]}-{int(datetime.now(timezone.utc).timestamp())}",
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            limit_price=request.limit_price,
            stop_price=request.stop_price,
            time_in_force=request.time_in_force,
            status=OrderStatus.PENDING,
            signal_id=request.signal_id,
        )

        # Create managed order
        managed = ManagedOrder(
            order=order,
            request=request,
            state=OrderState.VALIDATED,
        )

        # Track order - use synchronous lock acquisition for create_order
        # (runs in sync context, but dict access is atomic in CPython)
        self._orders[order.order_id] = managed
        self._client_id_map[order.client_order_id] = order.order_id

        logger.info(f"Created order {order.order_id}: {request.symbol} {request.side.value} {request.quantity}")

        return managed

    async def submit_order(
        self,
        managed: ManagedOrder,
        current_price: Decimal | None = None,
    ) -> ManagedOrder:
        """Submit an order to the broker.

        Args:
            managed: Managed order to submit.
            current_price: Current market price.

        Returns:
            Updated managed order.

        Raises:
            OrderSubmissionError: If submission fails.
        """
        if managed.is_terminal:
            raise OrderSubmissionError(
                f"Cannot submit order in terminal state: {managed.state}",
                order_id=str(managed.order.order_id),
            )

        try:
            # Determine execution parameters
            order_type = managed.order.order_type
            limit_price = managed.order.limit_price
            stop_price = managed.order.stop_price

            # Check if we need bracket order
            take_profit = None
            stop_loss = None
            order_class = OrderClass.SIMPLE

            if managed.request.take_profit_price and managed.request.stop_loss_price:
                order_class = OrderClass.BRACKET
                take_profit = {"limit_price": str(managed.request.take_profit_price)}
                stop_loss = {"stop_price": str(managed.request.stop_loss_price)}

            # Submit to broker
            managed.state = OrderState.SUBMITTED
            managed.submission_time = datetime.now(timezone.utc)

            broker_order = await self.client.submit_order(
                symbol=managed.order.symbol,
                qty=managed.order.quantity,
                side=managed.order.side,
                order_type=order_type,
                time_in_force=managed.order.time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
                client_order_id=managed.order.client_order_id,
                extended_hours=managed.request.extended_hours,
                order_class=order_class,
                take_profit=take_profit,
                stop_loss=stop_loss,
            )

            # Update managed order
            managed.broker_order = broker_order
            managed.order = managed.order.model_copy(
                update={
                    "broker_order_id": broker_order.order_id,
                    "status": OrderStatus.SUBMITTED,
                }
            )
            managed.state = OrderState.PENDING
            managed.last_update = datetime.now(timezone.utc)

            # Track by broker ID with lock protection
            async with self._lock:
                self._broker_id_map[broker_order.order_id] = managed.order.order_id

            # Publish event
            self._publish_order_event(EventType.ORDER_SUBMITTED, managed)

            logger.info(f"Submitted order {managed.order.order_id} -> broker {broker_order.order_id}")

            return managed

        except InsufficientFundsError:
            managed.state = OrderState.REJECTED
            managed.error_message = "Insufficient funds"
            self._publish_order_event(EventType.ORDER_REJECTED, managed)
            raise
        except OrderSubmissionError as e:
            managed.retry_count += 1
            if managed.retry_count >= self.max_retries:
                managed.state = OrderState.FAILED
                managed.error_message = str(e)
                self._publish_order_event(EventType.ORDER_REJECTED, managed)
            raise
        except Exception as e:
            managed.state = OrderState.FAILED
            managed.error_message = str(e)
            self._publish_order_event(EventType.ORDER_REJECTED, managed)
            raise OrderSubmissionError(
                f"Order submission failed: {e}",
                order_id=str(managed.order.order_id),
                symbol=managed.order.symbol,
            )

    async def cancel_order(self, order_id: UUID) -> ManagedOrder:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel.

        Returns:
            Updated managed order.

        Raises:
            OrderCancellationError: If cancellation fails.
        """
        managed = self._orders.get(order_id)
        if not managed:
            raise OrderCancellationError(
                f"Order not found: {order_id}",
                order_id=str(order_id),
            )

        if managed.is_terminal:
            raise OrderCancellationError(
                f"Cannot cancel order in terminal state: {managed.state}",
                order_id=str(order_id),
            )

        try:
            if managed.broker_order:
                await self.client.cancel_order(managed.broker_order.order_id)

            managed.state = OrderState.CANCELLED
            managed.order = managed.order.model_copy(update={"status": OrderStatus.CANCELLED})
            managed.last_update = datetime.now(timezone.utc)

            self._publish_order_event(EventType.ORDER_CANCELLED, managed)

            logger.info(f"Cancelled order {order_id}")

            return managed

        except Exception as e:
            raise OrderCancellationError(
                f"Failed to cancel order: {e}",
                order_id=str(order_id),
            )

    async def cancel_all_orders(self, symbol: str | None = None) -> list[ManagedOrder]:
        """Cancel all active orders.

        Args:
            symbol: Optional symbol filter.

        Returns:
            List of cancelled orders.
        """
        cancelled = []

        for managed in list(self._orders.values()):
            if not managed.is_active:
                continue
            if symbol and managed.order.symbol != symbol:
                continue

            try:
                await self.cancel_order(managed.order.order_id)
                cancelled.append(managed)
            except OrderCancellationError as e:
                logger.warning(f"Failed to cancel order {managed.order.order_id}: {e}")

        return cancelled

    async def modify_order(
        self,
        order_id: UUID,
        quantity: Decimal | None = None,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> ManagedOrder:
        """Modify an existing order.

        Args:
            order_id: Order ID to modify.
            quantity: New quantity.
            limit_price: New limit price.
            stop_price: New stop price.

        Returns:
            Updated managed order.
        """
        managed = self._orders.get(order_id)
        if not managed:
            raise OrderCancellationError(
                f"Order not found: {order_id}",
                order_id=str(order_id),
            )

        if managed.is_terminal:
            raise ExecutionError(
                f"Cannot modify order in terminal state: {managed.state}",
            )

        if not managed.broker_order:
            raise ExecutionError("Order not yet submitted to broker")

        try:
            new_broker_order = await self.client.replace_order(
                order_id=managed.broker_order.order_id,
                qty=quantity,
                limit_price=limit_price,
                stop_price=stop_price,
            )

            # Update tracking
            managed.broker_order = new_broker_order
            managed.order = new_broker_order.to_order()
            managed.last_update = datetime.now(timezone.utc)

            # Update broker ID map with lock protection
            async with self._lock:
                self._broker_id_map[new_broker_order.order_id] = managed.order.order_id

            logger.info(f"Modified order {order_id}")

            return managed

        except Exception as e:
            raise ExecutionError(f"Failed to modify order: {e}")

    def get_order(self, order_id: UUID) -> ManagedOrder | None:
        """Get managed order by ID."""
        return self._orders.get(order_id)

    def get_order_by_client_id(self, client_order_id: str) -> ManagedOrder | None:
        """Get managed order by client order ID."""
        order_id = self._client_id_map.get(client_order_id)
        if order_id:
            return self._orders.get(order_id)
        return None

    def get_order_by_broker_id(self, broker_order_id: str) -> ManagedOrder | None:
        """Get managed order by broker order ID."""
        order_id = self._broker_id_map.get(broker_order_id)
        if order_id:
            return self._orders.get(order_id)
        return None

    def get_active_orders(self, symbol: str | None = None) -> list[ManagedOrder]:
        """Get all active orders.

        Args:
            symbol: Optional symbol filter.

        Returns:
            List of active managed orders.
        """
        orders = [o for o in self._orders.values() if o.is_active]
        if symbol:
            orders = [o for o in orders if o.order.symbol == symbol]
        return orders

    def get_filled_orders(
        self,
        symbol: str | None = None,
        since: datetime | None = None,
    ) -> list[ManagedOrder]:
        """Get filled orders.

        Args:
            symbol: Optional symbol filter.
            since: Optional time filter.

        Returns:
            List of filled managed orders.
        """
        orders = [o for o in self._orders.values() if o.state == OrderState.FILLED]
        if symbol:
            orders = [o for o in orders if o.order.symbol == symbol]
        if since:
            orders = [o for o in orders if o.fill_time and o.fill_time >= since]
        return orders

    def on_fill(self, callback: Callable[[ManagedOrder], None]) -> None:
        """Register callback for order fills."""
        self._fill_callbacks.append(callback)

    def on_rejection(self, callback: Callable[[ManagedOrder], None]) -> None:
        """Register callback for order rejections."""
        self._rejection_callbacks.append(callback)

    def _handle_trade_update(self, data: dict[str, Any]) -> None:
        """Handle trade update from WebSocket."""
        event_type = data.get("event")
        order_data = data.get("order", {})
        broker_id = order_data.get("id")

        if not broker_id:
            return

        managed = self.get_order_by_broker_id(broker_id)
        if not managed:
            logger.debug(f"Received update for unknown order {broker_id}")
            return

        # Update from broker
        broker_order = AlpacaOrder.from_alpaca(order_data)
        managed.broker_order = broker_order
        managed.order = broker_order.to_order()
        managed.last_update = datetime.now(timezone.utc)

        # Handle state transitions
        if event_type == "fill":
            self._handle_fill(managed, data)
        elif event_type == "partial_fill":
            self._handle_partial_fill(managed, data)
        elif event_type == "canceled":
            self._handle_cancellation(managed)
        elif event_type == "rejected":
            self._handle_rejection(managed, data)
        elif event_type == "expired":
            self._handle_expiration(managed)

    def _handle_fill(self, managed: ManagedOrder, data: dict[str, Any]) -> None:
        """Handle order fill event."""
        managed.state = OrderState.FILLED
        managed.fill_time = datetime.now(timezone.utc)
        managed.fill_events.append(data)

        self._publish_order_event(EventType.ORDER_FILLED, managed)

        # Invoke callbacks
        for callback in self._fill_callbacks:
            try:
                callback(managed)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")

        logger.info(
            f"Order {managed.order.order_id} filled: "
            f"{managed.order.symbol} {managed.order.side.value} "
            f"{managed.order.filled_qty} @ {managed.order.filled_avg_price}"
        )

    def _handle_partial_fill(self, managed: ManagedOrder, data: dict[str, Any]) -> None:
        """Handle partial fill event."""
        managed.state = OrderState.PARTIAL_FILLED
        managed.fill_events.append(data)

        self._publish_order_event(EventType.ORDER_PARTIAL, managed)

        logger.info(
            f"Order {managed.order.order_id} partial fill: "
            f"{managed.order.filled_qty}/{managed.order.quantity}"
        )

    def _handle_cancellation(self, managed: ManagedOrder) -> None:
        """Handle order cancellation event."""
        managed.state = OrderState.CANCELLED
        self._publish_order_event(EventType.ORDER_CANCELLED, managed)
        logger.info(f"Order {managed.order.order_id} cancelled")

    def _handle_rejection(self, managed: ManagedOrder, data: dict[str, Any]) -> None:
        """Handle order rejection event."""
        managed.state = OrderState.REJECTED
        managed.error_message = data.get("message", "Order rejected")

        self._publish_order_event(EventType.ORDER_REJECTED, managed)

        # Invoke callbacks
        for callback in self._rejection_callbacks:
            try:
                callback(managed)
            except Exception as e:
                logger.error(f"Rejection callback error: {e}")

        logger.warning(f"Order {managed.order.order_id} rejected: {managed.error_message}")

    def _handle_expiration(self, managed: ManagedOrder) -> None:
        """Handle order expiration event."""
        managed.state = OrderState.EXPIRED
        self._publish_order_event(EventType.ORDER_EXPIRED, managed)
        logger.info(f"Order {managed.order.order_id} expired")

    async def _monitor_orders(self) -> None:
        """Background task to monitor order states."""
        while self._running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds

                # Get open orders from broker
                broker_orders = await self.client.get_orders(status="open")
                broker_ids = {o.order_id for o in broker_orders}

                # Check for stale orders
                for managed in list(self._orders.values()):
                    if not managed.is_active:
                        continue

                    if managed.broker_order and managed.broker_order.order_id not in broker_ids:
                        # Order no longer open - fetch latest state
                        try:
                            latest = await self.client.get_order(managed.broker_order.order_id)
                            self._handle_trade_update({
                                "event": latest.status.lower().replace("_", ""),
                                "order": {
                                    "id": latest.order_id,
                                    "client_order_id": latest.client_order_id,
                                    "symbol": latest.symbol,
                                    "side": latest.side,
                                    "type": latest.order_type,
                                    "qty": str(latest.quantity),
                                    "filled_qty": str(latest.filled_qty),
                                    "filled_avg_price": str(latest.filled_avg_price) if latest.filled_avg_price else None,
                                    "status": latest.status,
                                    "created_at": latest.created_at.isoformat(),
                                    "updated_at": latest.updated_at.isoformat(),
                                }
                            })
                        except Exception as e:
                            logger.warning(f"Failed to fetch order state: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Order monitoring error: {e}")

    async def reconcile_with_broker(self) -> dict[str, Any]:
        """Reconcile internal state with broker.

        Returns:
            Reconciliation report.
        """
        report = {
            "broker_orders": 0,
            "internal_orders": len(self._orders),
            "matched": 0,
            "missing_internal": [],
            "missing_broker": [],
            "state_mismatches": [],
        }

        # Get all broker orders
        broker_orders = await self.client.get_orders(status="all", limit=500)
        report["broker_orders"] = len(broker_orders)

        broker_map = {o.order_id: o for o in broker_orders}

        # Check internal orders against broker
        for managed in self._orders.values():
            if managed.broker_order:
                broker_order = broker_map.get(managed.broker_order.order_id)
                if broker_order:
                    report["matched"] += 1
                    # Check for state mismatch
                    if managed.broker_order.status != broker_order.status:
                        report["state_mismatches"].append({
                            "order_id": str(managed.order.order_id),
                            "internal_state": managed.state.value,
                            "broker_state": broker_order.status,
                        })
                else:
                    report["missing_broker"].append(str(managed.order.order_id))

        logger.info(f"Reconciliation complete: {report['matched']} matched, "
                   f"{len(report['state_mismatches'])} mismatches")

        return report

    def _publish_order_event(self, event_type: EventType, managed: ManagedOrder) -> None:
        """Publish order event to event bus."""
        if not self.event_bus:
            return

        event = create_order_event(
            event_type=event_type,
            order_data={
                "order_id": str(managed.order.order_id),
                "client_order_id": managed.order.client_order_id,
                "broker_order_id": managed.order.broker_order_id,
                "symbol": managed.order.symbol,
                "side": managed.order.side.value,
                "quantity": str(managed.order.quantity),
                "filled_qty": str(managed.order.filled_qty),
                "filled_avg_price": str(managed.order.filled_avg_price) if managed.order.filled_avg_price else None,
                "status": managed.order.status.value,
                "state": managed.state.value,
            },
            source="OrderManager",
        )
        self.event_bus.publish(event)

    def get_statistics(self) -> dict[str, Any]:
        """Get order manager statistics.

        Returns:
            Dictionary with order statistics.
        """
        total = len(self._orders)
        by_state = {}
        for managed in self._orders.values():
            state = managed.state.value
            by_state[state] = by_state.get(state, 0) + 1

        filled_orders = [o for o in self._orders.values() if o.state == OrderState.FILLED]
        avg_fill_time = 0.0
        if filled_orders:
            fill_times = []
            for o in filled_orders:
                if o.submission_time and o.fill_time:
                    delta = (o.fill_time - o.submission_time).total_seconds()
                    fill_times.append(delta)
            if fill_times:
                avg_fill_time = sum(fill_times) / len(fill_times)

        return {
            "total_orders": total,
            "by_state": by_state,
            "active_orders": len([o for o in self._orders.values() if o.is_active]),
            "fill_rate": len(filled_orders) / total if total > 0 else 0.0,
            "avg_fill_time_seconds": avg_fill_time,
            "retry_count": sum(o.retry_count for o in self._orders.values()),
        }
