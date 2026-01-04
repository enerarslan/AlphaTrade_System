"""
Position tracking module.

Tracks portfolio positions and cash with:
- Real-time position updates from fills
- Price updates and P&L calculation
- Broker reconciliation
- Settlement tracking
- P&L attribution by strategy/model

Maintains consistency between internal state and broker positions.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable
from uuid import UUID

from quant_trading_system.core.data_types import (
    Order,
    OrderSide,
    Portfolio,
    Position,
)
from quant_trading_system.core.events import (
    Event,
    EventBus,
    EventType,
)
from quant_trading_system.core.exceptions import ExecutionError
from quant_trading_system.execution.alpaca_client import AlpacaClient, AlpacaPosition

logger = logging.getLogger(__name__)


class SettlementStatus(str, Enum):
    """Trade settlement status."""

    PENDING = "pending"
    SETTLED = "settled"


@dataclass
class TradeRecord:
    """Record of an executed trade."""

    trade_id: UUID
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    order_id: UUID | None = None
    commission: Decimal = Decimal("0")
    settlement_date: datetime | None = None
    settlement_status: SettlementStatus = SettlementStatus.PENDING
    strategy_id: str | None = None
    model_id: str | None = None
    realized_pnl: Decimal = Decimal("0")
    notes: str = ""


@dataclass
class CashState:
    """Cash balance tracking."""

    total_cash: Decimal = Decimal("0")
    settled_cash: Decimal = Decimal("0")
    pending_settlement: Decimal = Decimal("0")
    buying_power: Decimal = Decimal("0")
    margin_used: Decimal = Decimal("0")
    margin_available: Decimal = Decimal("0")
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PnLRecord:
    """P&L record for attribution."""

    date: datetime
    symbol: str | None
    strategy_id: str | None
    model_id: str | None
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    dividend_income: Decimal = Decimal("0")
    interest: Decimal = Decimal("0")

    @property
    def net_pnl(self) -> Decimal:
        """Net P&L after costs."""
        return self.realized_pnl + self.unrealized_pnl - self.commission + self.dividend_income - self.interest


@dataclass
class PositionState:
    """Extended position state with tracking metadata."""

    position: Position
    strategy_id: str | None = None
    model_id: str | None = None
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trade_count: int = 0
    total_commission: Decimal = Decimal("0")
    high_water_mark: Decimal = Decimal("0")  # Best unrealized P&L
    low_water_mark: Decimal = Decimal("0")  # Worst unrealized P&L
    last_trade: datetime | None = None


@dataclass
class ReconciliationResult:
    """Result of position reconciliation."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    matches: int = 0
    discrepancies: list[dict[str, Any]] = field(default_factory=list)
    missing_internal: list[str] = field(default_factory=list)
    missing_broker: list[str] = field(default_factory=list)
    auto_corrected: list[str] = field(default_factory=list)

    @property
    def is_consistent(self) -> bool:
        """Check if positions are consistent."""
        return len(self.discrepancies) == 0 and len(self.missing_internal) == 0 and len(self.missing_broker) == 0


class PositionTracker:
    """Tracks portfolio positions, cash, and P&L.

    Maintains:
    - Current positions with real-time updates
    - Cash balances and buying power
    - Trade history for audit trail
    - P&L attribution by strategy/model
    - Broker reconciliation
    """

    def __init__(
        self,
        client: AlpacaClient,
        event_bus: EventBus | None = None,
        settlement_days: int = 2,
        auto_reconcile: bool = True,
        reconcile_interval: int = 60,
    ) -> None:
        """Initialize position tracker.

        Args:
            client: Alpaca client for broker queries.
            event_bus: Event bus for position events.
            settlement_days: T+N settlement days.
            auto_reconcile: Automatically reconcile with broker.
            reconcile_interval: Seconds between reconciliations.
        """
        self.client = client
        self.event_bus = event_bus
        self.settlement_days = settlement_days
        self.auto_reconcile = auto_reconcile
        self.reconcile_interval = reconcile_interval

        # State
        self._positions: dict[str, PositionState] = {}
        self._cash: CashState = CashState()
        self._trades: list[TradeRecord] = []
        self._pnl_records: list[PnLRecord] = []

        # Thread safety - protect shared state from race conditions
        # Use asyncio.Lock for async methods and threading.RLock for sync methods
        self._async_lock = asyncio.Lock()
        self._sync_lock = threading.RLock()  # RLock allows reentrant locking

        # Callbacks
        self._position_callbacks: list[Callable[[str, Position], None]] = []

        # Background tasks
        self._reconcile_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start position tracking."""
        self._running = True

        # Initial sync with broker
        await self.sync_with_broker()

        # Start reconciliation loop
        if self.auto_reconcile:
            self._reconcile_task = asyncio.create_task(self._reconcile_loop())

        logger.info("PositionTracker started")

    async def stop(self) -> None:
        """Stop position tracking."""
        self._running = False
        if self._reconcile_task:
            self._reconcile_task.cancel()
            try:
                await self._reconcile_task
            except asyncio.CancelledError:
                pass
        logger.info("PositionTracker stopped")

    async def sync_with_broker(self) -> None:
        """Sync position state with broker."""
        try:
            # Get account info
            account = await self.client.get_account()
            self._cash = CashState(
                total_cash=account.cash,
                settled_cash=account.cash,  # Alpaca doesn't expose pending separately
                buying_power=account.buying_power,
                margin_used=account.initial_margin,
                margin_available=account.buying_power,
                last_updated=datetime.now(timezone.utc),
            )

            # Get positions with lock protection
            broker_positions = await self.client.get_positions()
            async with self._async_lock:
                for bp in broker_positions:
                    self._positions[bp.symbol] = PositionState(
                        position=bp.to_position(),
                    )

            logger.info(f"Synced with broker: {len(self._positions)} positions, ${self._cash.total_cash} cash")

        except Exception as e:
            logger.error(f"Failed to sync with broker: {e}")
            raise

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Position or None if no position.
        """
        state = self._positions.get(symbol.upper())
        return state.position if state else None

    def get_position_state(self, symbol: str) -> PositionState | None:
        """Get full position state for a symbol."""
        return self._positions.get(symbol.upper())

    def get_all_positions(self) -> dict[str, Position]:
        """Get all positions."""
        return {s: ps.position for s, ps in self._positions.items() if not ps.position.is_flat}

    def get_portfolio(self) -> Portfolio:
        """Get current portfolio state.

        Returns:
            Portfolio with current positions and cash.
        """
        positions = self.get_all_positions()

        total_market_value = sum(p.market_value for p in positions.values())
        equity = self._cash.total_cash + total_market_value

        return Portfolio(
            equity=equity,
            cash=self._cash.total_cash,
            buying_power=self._cash.buying_power,
            positions=positions,
            margin_used=self._cash.margin_used,
        )

    def update_from_fill(
        self,
        order: Order,
        fill_qty: Decimal,
        fill_price: Decimal,
        commission: Decimal = Decimal("0"),
        strategy_id: str | None = None,
        model_id: str | None = None,
    ) -> Position:
        """Update position from order fill.

        THREAD-SAFE: Uses RLock to prevent race conditions when
        multiple fills arrive concurrently from WebSocket callbacks.

        Args:
            order: Filled order.
            fill_qty: Quantity filled.
            fill_price: Average fill price.
            commission: Commission paid.
            strategy_id: Strategy that generated order.
            model_id: Model that generated signal.

        Returns:
            Updated position.
        """
        # CRITICAL FIX: Acquire lock to prevent race conditions
        with self._sync_lock:
            return self._update_from_fill_impl(
                order, fill_qty, fill_price, commission, strategy_id, model_id
            )

    def _update_from_fill_impl(
        self,
        order: Order,
        fill_qty: Decimal,
        fill_price: Decimal,
        commission: Decimal,
        strategy_id: str | None,
        model_id: str | None,
    ) -> Position:
        """Internal implementation of update_from_fill (called with lock held)."""
        symbol = order.symbol.upper()
        state = self._positions.get(symbol)

        # Calculate new position
        if state is None:
            # New position
            if order.side == OrderSide.BUY:
                qty = fill_qty
            else:
                qty = -fill_qty

            position = Position(
                symbol=symbol,
                quantity=qty,
                avg_entry_price=fill_price,
                current_price=fill_price,
                cost_basis=qty * fill_price,
                market_value=qty * fill_price,
            )
            state = PositionState(
                position=position,
                strategy_id=strategy_id,
                model_id=model_id,
            )
        else:
            # Update existing position
            position = state.position
            old_qty = position.quantity
            old_cost = position.cost_basis

            # BUG FIX: Initialize realized_pnl to avoid undefined variable issue
            realized_pnl = Decimal("0")

            if order.side == OrderSide.BUY:
                new_qty = old_qty + fill_qty
                new_cost = old_cost + (fill_qty * fill_price)
            else:
                new_qty = old_qty - fill_qty
                # Closing position - calculate realized P&L
                if abs(fill_qty) <= abs(old_qty):
                    # Partial or full close
                    realized_pnl = fill_qty * (fill_price - position.avg_entry_price)
                    if old_qty < 0:
                        # SHORT position: profit when exit < entry
                        realized_pnl = -realized_pnl

                new_cost = new_qty * position.avg_entry_price if new_qty != 0 else Decimal("0")

            # Calculate new average entry price
            if new_qty != 0:
                if (old_qty > 0 and order.side == OrderSide.BUY) or (old_qty < 0 and order.side == OrderSide.SELL):
                    # Adding to position
                    new_avg_price = new_cost / new_qty
                else:
                    # Reducing position - keep same avg entry
                    new_avg_price = position.avg_entry_price
            else:
                new_avg_price = Decimal("0")

            position = Position(
                symbol=symbol,
                quantity=new_qty,
                avg_entry_price=new_avg_price,
                current_price=fill_price,
                cost_basis=abs(new_qty * new_avg_price) if new_qty != 0 else Decimal("0"),
                market_value=new_qty * fill_price,
                unrealized_pnl=(fill_price - new_avg_price) * new_qty if new_qty != 0 else Decimal("0"),
                realized_pnl=state.position.realized_pnl + realized_pnl,
            )
            state.position = position

        # Update state
        state.trade_count += 1
        state.total_commission += commission
        state.last_trade = datetime.now(timezone.utc)

        # Track water marks
        if position.unrealized_pnl > state.high_water_mark:
            state.high_water_mark = position.unrealized_pnl
        if position.unrealized_pnl < state.low_water_mark:
            state.low_water_mark = position.unrealized_pnl

        self._positions[symbol] = state

        # Record trade
        trade = TradeRecord(
            trade_id=order.order_id,
            symbol=symbol,
            side=order.side,
            quantity=fill_qty,
            price=fill_price,
            timestamp=datetime.now(timezone.utc),
            order_id=order.order_id,
            commission=commission,
            settlement_date=datetime.now(timezone.utc) + timedelta(days=self.settlement_days),
            strategy_id=strategy_id,
            model_id=model_id,
        )
        self._trades.append(trade)

        # Update cash
        trade_value = fill_qty * fill_price
        if order.side == OrderSide.BUY:
            self._cash.total_cash -= trade_value + commission
            self._cash.pending_settlement += trade_value
        else:
            self._cash.total_cash += trade_value - commission
            self._cash.pending_settlement -= trade_value

        # Publish event
        self._publish_position_event(symbol, position)

        # Invoke callbacks
        for callback in self._position_callbacks:
            try:
                callback(symbol, position)
            except Exception as e:
                logger.error(f"Position callback error: {e}")

        logger.info(
            f"Position updated: {symbol} {position.quantity} @ {position.avg_entry_price} "
            f"(unrealized: ${position.unrealized_pnl:.2f})"
        )

        return position

    def update_price(self, symbol: str, price: Decimal) -> Position | None:
        """Update position with new price.

        Args:
            symbol: Stock symbol.
            price: New market price.

        Returns:
            Updated position or None.
        """
        state = self._positions.get(symbol.upper())
        if not state or state.position.is_flat:
            return None

        old_position = state.position
        new_position = old_position.update_price(price)
        state.position = new_position

        # Track water marks
        if new_position.unrealized_pnl > state.high_water_mark:
            state.high_water_mark = new_position.unrealized_pnl
        if new_position.unrealized_pnl < state.low_water_mark:
            state.low_water_mark = new_position.unrealized_pnl

        return new_position

    def update_prices(self, prices: dict[str, Decimal]) -> None:
        """Update multiple position prices.

        Args:
            prices: Symbol to price mapping.
        """
        for symbol, price in prices.items():
            self.update_price(symbol, price)

    def get_cash(self) -> CashState:
        """Get current cash state."""
        return self._cash

    def get_buying_power(self) -> Decimal:
        """Get available buying power."""
        return self._cash.buying_power

    def get_total_exposure(self) -> Decimal:
        """Get total market exposure (absolute value)."""
        return sum(abs(ps.position.market_value) for ps in self._positions.values())

    def get_net_exposure(self) -> Decimal:
        """Get net market exposure (long - short)."""
        return sum(ps.position.market_value for ps in self._positions.values())

    def get_sector_exposure(self, sector_map: dict[str, str]) -> dict[str, Decimal]:
        """Get exposure by sector.

        Args:
            sector_map: Symbol to sector mapping.

        Returns:
            Sector to exposure mapping.
        """
        exposures: dict[str, Decimal] = {}
        for symbol, state in self._positions.items():
            sector = sector_map.get(symbol, "UNKNOWN")
            if sector not in exposures:
                exposures[sector] = Decimal("0")
            exposures[sector] += abs(state.position.market_value)
        return exposures

    def get_trades(
        self,
        symbol: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        strategy_id: str | None = None,
    ) -> list[TradeRecord]:
        """Get trade records with optional filters.

        Args:
            symbol: Filter by symbol.
            since: Start time filter.
            until: End time filter.
            strategy_id: Filter by strategy.

        Returns:
            Filtered trade records.
        """
        trades = self._trades

        if symbol:
            trades = [t for t in trades if t.symbol == symbol.upper()]
        if since:
            trades = [t for t in trades if t.timestamp >= since]
        if until:
            trades = [t for t in trades if t.timestamp <= until]
        if strategy_id:
            trades = [t for t in trades if t.strategy_id == strategy_id]

        return trades

    def get_realized_pnl(
        self,
        symbol: str | None = None,
        strategy_id: str | None = None,
        since: datetime | None = None,
    ) -> Decimal:
        """Get total realized P&L.

        Args:
            symbol: Filter by symbol.
            strategy_id: Filter by strategy.
            since: Start time filter.

        Returns:
            Total realized P&L.
        """
        total = Decimal("0")
        for state in self._positions.values():
            if symbol and state.position.symbol != symbol:
                continue
            if strategy_id and state.strategy_id != strategy_id:
                continue
            total += state.position.realized_pnl
        return total

    def get_unrealized_pnl(
        self,
        symbol: str | None = None,
        strategy_id: str | None = None,
    ) -> Decimal:
        """Get total unrealized P&L.

        Args:
            symbol: Filter by symbol.
            strategy_id: Filter by strategy.

        Returns:
            Total unrealized P&L.
        """
        total = Decimal("0")
        for state in self._positions.values():
            if symbol and state.position.symbol != symbol:
                continue
            if strategy_id and state.strategy_id != strategy_id:
                continue
            total += state.position.unrealized_pnl
        return total

    def get_pnl_attribution(self) -> dict[str, dict[str, Decimal]]:
        """Get P&L attribution by strategy and model.

        Returns:
            Nested dict of strategy -> model -> P&L.
        """
        attribution: dict[str, dict[str, Decimal]] = {}

        for state in self._positions.values():
            strategy = state.strategy_id or "UNKNOWN"
            model = state.model_id or "UNKNOWN"

            if strategy not in attribution:
                attribution[strategy] = {}
            if model not in attribution[strategy]:
                attribution[strategy][model] = Decimal("0")

            attribution[strategy][model] += state.position.unrealized_pnl + state.position.realized_pnl

        return attribution

    async def reconcile_with_broker(
        self,
        auto_correct: bool = False,
    ) -> ReconciliationResult:
        """Reconcile positions with broker.

        Args:
            auto_correct: Automatically correct discrepancies.

        Returns:
            Reconciliation result.
        """
        result = ReconciliationResult()

        try:
            # Get broker positions
            broker_positions = await self.client.get_positions()
            broker_map = {p.symbol: p for p in broker_positions}

            # Check internal positions against broker
            for symbol, state in self._positions.items():
                if state.position.is_flat:
                    continue

                broker_pos = broker_map.get(symbol)
                if broker_pos:
                    # Compare quantities
                    internal_qty = state.position.quantity
                    broker_qty = broker_pos.quantity
                    if broker_pos.side == "short":
                        broker_qty = -broker_qty

                    if internal_qty != broker_qty:
                        discrepancy = {
                            "symbol": symbol,
                            "internal_qty": str(internal_qty),
                            "broker_qty": str(broker_qty),
                            "diff": str(internal_qty - broker_qty),
                        }
                        result.discrepancies.append(discrepancy)

                        if auto_correct:
                            # P1 FIX: Validate that the position delta is reasonable
                            # Don't blindly accept broker state that could be corrupt
                            actual_delta = abs(broker_qty - internal_qty)
                            max_reasonable_delta = Decimal("1000")  # Max shares to auto-correct

                            if actual_delta > max_reasonable_delta:
                                # Large discrepancy - refuse to auto-correct
                                logger.error(
                                    f"Reconciliation blocked for {symbol}: delta {actual_delta} "
                                    f"exceeds max reasonable delta {max_reasonable_delta}. "
                                    "Manual intervention required."
                                )
                                discrepancy["auto_correct_blocked"] = True
                                discrepancy["reason"] = f"Delta too large: {actual_delta}"
                            else:
                                # Small discrepancy - allow auto-correct with warning
                                logger.warning(
                                    f"Auto-correcting {symbol} position from "
                                    f"{internal_qty} to {broker_qty} (delta: {actual_delta})"
                                )
                                state.position = broker_pos.to_position()
                                result.auto_corrected.append(symbol)
                    else:
                        result.matches += 1
                else:
                    result.missing_broker.append(symbol)

            # Check for positions in broker but not internal
            for symbol in broker_map:
                if symbol not in self._positions or self._positions[symbol].position.is_flat:
                    result.missing_internal.append(symbol)

                    if auto_correct:
                        self._positions[symbol] = PositionState(
                            position=broker_map[symbol].to_position(),
                        )
                        result.auto_corrected.append(symbol)

            # Also reconcile cash
            account = await self.client.get_account()
            if self._cash.total_cash != account.cash:
                result.discrepancies.append({
                    "type": "cash",
                    "internal": str(self._cash.total_cash),
                    "broker": str(account.cash),
                })

                if auto_correct:
                    self._cash.total_cash = account.cash
                    self._cash.buying_power = account.buying_power

            logger.info(
                f"Reconciliation: {result.matches} matches, "
                f"{len(result.discrepancies)} discrepancies, "
                f"{len(result.missing_internal)} missing internal, "
                f"{len(result.missing_broker)} missing broker"
            )

        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            raise

        return result

    async def _reconcile_loop(self) -> None:
        """Background reconciliation loop."""
        while self._running:
            try:
                await asyncio.sleep(self.reconcile_interval)
                await self.reconcile_with_broker(auto_correct=True)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Reconciliation loop error: {e}")

    def on_position_change(self, callback: Callable[[str, Position], None]) -> None:
        """Register callback for position changes.

        Args:
            callback: Function(symbol, position) to call.
        """
        self._position_callbacks.append(callback)

    def _publish_position_event(self, symbol: str, position: Position) -> None:
        """Publish position event to event bus."""
        if not self.event_bus:
            return

        if position.is_flat:
            event_type = EventType.POSITION_CLOSED
        elif self._positions.get(symbol) and self._positions[symbol].trade_count == 1:
            event_type = EventType.POSITION_OPENED
        else:
            event_type = EventType.POSITION_UPDATED

        event = Event(
            event_type=event_type,
            data={
                "symbol": symbol,
                "quantity": str(position.quantity),
                "avg_entry_price": str(position.avg_entry_price),
                "current_price": str(position.current_price),
                "market_value": str(position.market_value),
                "unrealized_pnl": str(position.unrealized_pnl),
            },
            source="PositionTracker",
        )
        self.event_bus.publish(event)

    def get_statistics(self) -> dict[str, Any]:
        """Get position tracker statistics.

        Returns:
            Dictionary with statistics.
        """
        positions = [ps for ps in self._positions.values() if not ps.position.is_flat]

        return {
            "num_positions": len(positions),
            "total_market_value": float(sum(ps.position.market_value for ps in positions)),
            "total_unrealized_pnl": float(sum(ps.position.unrealized_pnl for ps in positions)),
            "total_realized_pnl": float(sum(ps.position.realized_pnl for ps in positions)),
            "num_trades": len(self._trades),
            "total_commission": float(sum(ps.total_commission for ps in positions)),
            "cash": float(self._cash.total_cash),
            "buying_power": float(self._cash.buying_power),
            "long_positions": len([ps for ps in positions if ps.position.is_long]),
            "short_positions": len([ps for ps in positions if ps.position.is_short]),
        }

    def export_trades(self, format: str = "dict") -> list[dict[str, Any]]:
        """Export trade records.

        Args:
            format: Output format ("dict").

        Returns:
            List of trade records.
        """
        return [
            {
                "trade_id": str(t.trade_id),
                "symbol": t.symbol,
                "side": t.side.value,
                "quantity": str(t.quantity),
                "price": str(t.price),
                "timestamp": t.timestamp.isoformat(),
                "order_id": str(t.order_id) if t.order_id else None,
                "commission": str(t.commission),
                "settlement_date": t.settlement_date.isoformat() if t.settlement_date else None,
                "settlement_status": t.settlement_status.value,
                "strategy_id": t.strategy_id,
                "model_id": t.model_id,
                "realized_pnl": str(t.realized_pnl),
                "notes": t.notes,
            }
            for t in self._trades
        ]
