"""
Risk limits and controls module.

Implements pre-trade, intra-trade, and post-trade risk checks:
- Pre-trade: buying power, position limits, concentration, blacklist
- Intra-trade: slippage monitoring, partial fill handling, stuck orders
- Post-trade: execution quality, cost validation, reconciliation

Includes kill switch functionality for emergency situations.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from quant_trading_system.core.data_types import Order, OrderSide, OrderStatus, Portfolio, Position
from quant_trading_system.core.events import EventBus, EventType, create_risk_event
from quant_trading_system.core.exceptions import (
    DrawdownLimitError,
    ExposureLimitError,
    InsufficientFundsError,
    PositionLimitError,
    RiskError,
)

logger = logging.getLogger(__name__)


class CheckResult(str, Enum):
    """Result of a risk check."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


class KillSwitchReason(str, Enum):
    """Reason for kill switch activation."""

    MAX_DRAWDOWN = "max_drawdown"
    RAPID_PNL_DECLINE = "rapid_pnl_decline"
    SYSTEM_ERROR = "system_error"
    DATA_FEED_FAILURE = "data_feed_failure"
    BROKER_CONNECTION_LOSS = "broker_connection_loss"
    MANUAL_ACTIVATION = "manual_activation"
    LIMIT_BREACH = "limit_breach"


@dataclass
class RiskCheckResult:
    """Result of a risk check operation."""

    check_name: str
    result: CheckResult
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def passed(self) -> bool:
        """Check if the result passed."""
        return self.result == CheckResult.PASSED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_name": self.check_name,
            "result": self.result.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class KillSwitchState:
    """Current state of the kill switch."""

    is_active: bool = False
    activated_at: datetime | None = None
    reason: KillSwitchReason | None = None
    trigger_value: float | None = None
    orders_cancelled: int = 0
    positions_closed: int = 0
    activated_by: str | None = None

    def activate(
        self,
        reason: KillSwitchReason,
        trigger_value: float | None = None,
        activated_by: str = "system",
    ) -> None:
        """Activate the kill switch."""
        self.is_active = True
        self.activated_at = datetime.now(timezone.utc)
        self.reason = reason
        self.trigger_value = trigger_value
        self.activated_by = activated_by

    def reset(self) -> None:
        """Reset the kill switch."""
        self.is_active = False
        self.activated_at = None
        self.reason = None
        self.trigger_value = None
        self.orders_cancelled = 0
        self.positions_closed = 0
        self.activated_by = None


class RiskLimitsConfig(BaseModel):
    """Configuration for risk limits."""

    # Loss limits
    daily_loss_limit_pct: float = Field(default=0.02, ge=0, le=1.0, description="Daily loss limit as % of equity")
    weekly_loss_limit_pct: float = Field(default=0.05, ge=0, le=1.0, description="Weekly loss limit")
    monthly_loss_limit_pct: float = Field(default=0.10, ge=0, le=1.0, description="Monthly loss limit")
    per_trade_loss_limit_pct: float = Field(default=0.01, ge=0, le=1.0, description="Per-trade loss limit")

    # Position limits
    max_position_value: Decimal = Field(default=Decimal("100000"), ge=0, description="Max $ per position")
    max_position_pct: float = Field(default=0.10, ge=0, le=1.0, description="Max % of equity per position")
    max_total_positions: int = Field(default=20, ge=1, description="Max number of positions")
    max_sector_exposure_pct: float = Field(default=0.25, ge=0, le=1.0, description="Max sector exposure")
    max_correlated_exposure_pct: float = Field(default=0.30, ge=0, le=1.0, description="Max correlated exposure")

    # Volatility limits
    max_portfolio_volatility: float = Field(default=0.25, ge=0, description="Max annualized volatility")
    high_vix_threshold: float = Field(default=30.0, ge=0, description="VIX threshold for risk reduction")
    high_vix_position_multiplier: float = Field(default=0.5, ge=0, le=1.0, description="Position size mult in high VIX")

    # Drawdown limits
    drawdown_warning_threshold: float = Field(default=0.05, ge=0, le=1.0, description="Drawdown warning level")
    drawdown_reduce_threshold: float = Field(default=0.10, ge=0, le=1.0, description="Drawdown position reduction level")
    drawdown_halt_threshold: float = Field(default=0.15, ge=0, le=1.0, description="Drawdown trading halt level")

    # Trading controls
    max_daily_trades: int = Field(default=100, ge=0, description="Max trades per day")
    max_daily_turnover_pct: float = Field(default=1.0, ge=0, description="Max daily turnover")
    min_order_value: Decimal = Field(default=Decimal("100"), ge=0, description="Min order value")
    max_order_value: Decimal = Field(default=Decimal("50000"), ge=0, description="Max single order value")

    # Blacklist
    symbol_blacklist: list[str] = Field(default_factory=list, description="Symbols not allowed to trade")

    # Trading hours (24h format)
    trading_start_hour: int = Field(default=9, ge=0, le=23, description="Trading start hour")
    trading_start_minute: int = Field(default=30, ge=0, le=59, description="Trading start minute")
    trading_end_hour: int = Field(default=16, ge=0, le=23, description="Trading end hour")
    trading_end_minute: int = Field(default=0, ge=0, le=59, description="Trading end minute")


class PreTradeRiskChecker:
    """Pre-trade risk checks before order submission.

    THREAD-SAFE: Uses RLock to protect daily counters.
    """

    def __init__(
        self,
        config: RiskLimitsConfig | None = None,
    ) -> None:
        """Initialize pre-trade checker.

        Args:
            config: Risk limits configuration.
        """
        self.config = config or RiskLimitsConfig()
        # THREAD SAFETY: Use lock for daily counter operations
        self._lock = threading.RLock()
        self._daily_trades: int = 0
        self._daily_turnover: Decimal = Decimal("0")
        self._last_reset_date: datetime = datetime.now(timezone.utc).date()  # type: ignore

    def _reset_daily_counters_if_needed(self) -> None:
        """Reset daily counters if it's a new day.

        THREAD-SAFE: Must be called with lock held or call with lock.
        """
        today = datetime.now(timezone.utc).date()
        with self._lock:
            if today != self._last_reset_date:
                self._daily_trades = 0
                self._daily_turnover = Decimal("0")
                self._last_reset_date = today  # type: ignore

    def check_all(
        self,
        order: Order,
        portfolio: Portfolio,
        current_price: Decimal,
    ) -> list[RiskCheckResult]:
        """Run all pre-trade checks.

        Args:
            order: Order to check.
            portfolio: Current portfolio state.
            current_price: Current price of the symbol.

        Returns:
            List of check results.
        """
        self._reset_daily_counters_if_needed()

        results = [
            self.check_buying_power(order, portfolio, current_price),
            self.check_position_limit(order, portfolio, current_price),
            self.check_concentration(order, portfolio, current_price),
            self.check_blacklist(order),
            self.check_trading_hours(),
            self.check_order_size(order, current_price),
            self.check_daily_limits(order, portfolio, current_price),
        ]

        return results

    def check_buying_power(
        self,
        order: Order,
        portfolio: Portfolio,
        current_price: Decimal,
    ) -> RiskCheckResult:
        """Check if there's sufficient buying power."""
        if order.side == OrderSide.SELL:
            # Check if we have the position to sell
            position = portfolio.get_position(order.symbol)
            if position is None or position.quantity < order.quantity:
                return RiskCheckResult(
                    check_name="buying_power",
                    result=CheckResult.FAILED,
                    message=f"Insufficient position to sell {order.quantity} shares of {order.symbol}",
                    details={"required": float(order.quantity), "available": float(position.quantity) if position else 0},
                )
            return RiskCheckResult(
                check_name="buying_power",
                result=CheckResult.PASSED,
                message="Sufficient position for sell order",
            )

        # For buy orders, check buying power
        order_value = order.quantity * current_price
        if order_value > portfolio.buying_power:
            return RiskCheckResult(
                check_name="buying_power",
                result=CheckResult.FAILED,
                message=f"Insufficient buying power for {order.symbol}",
                details={"required": float(order_value), "available": float(portfolio.buying_power)},
            )

        return RiskCheckResult(
            check_name="buying_power",
            result=CheckResult.PASSED,
            message="Sufficient buying power",
        )

    def check_position_limit(
        self,
        order: Order,
        portfolio: Portfolio,
        current_price: Decimal,
    ) -> RiskCheckResult:
        """Check position size limits.

        CRITICAL FIX: Handle BUY and SELL orders differently.
        - BUY: Check if new position would exceed limits
        - SELL: Check if we're reducing position (no limit check needed for reduction)
        """
        order_value = order.quantity * current_price
        existing_position = portfolio.get_position(order.symbol)
        existing_value = abs(existing_position.market_value) if existing_position else Decimal("0")

        # CRITICAL FIX: For SELL orders, we're reducing the position
        # Position limit checks only apply when INCREASING position
        if order.side == OrderSide.SELL:
            # For sells, check if we're closing or reducing a long position
            if existing_position and existing_position.is_long:
                # This is a position reduction, no limit check needed
                return RiskCheckResult(
                    check_name="position_limit",
                    result=CheckResult.PASSED,
                    message="Sell order reduces existing long position",
                )
            elif existing_position and existing_position.is_short:
                # Selling while short means adding to short position - check limits
                total_position_value = existing_value + order_value
            else:
                # Opening new short position
                total_position_value = order_value
        else:  # BUY order
            if existing_position and existing_position.is_short:
                # Buying while short means reducing short position - no limit check needed
                return RiskCheckResult(
                    check_name="position_limit",
                    result=CheckResult.PASSED,
                    message="Buy order reduces existing short position",
                )
            else:
                # Adding to long or opening new long
                total_position_value = existing_value + order_value

        # Check absolute limit
        if order_value > self.config.max_position_value:
            return RiskCheckResult(
                check_name="position_limit",
                result=CheckResult.FAILED,
                message=f"Order value exceeds max position value",
                details={
                    "order_value": float(order_value),
                    "max_value": float(self.config.max_position_value),
                },
            )

        # Check percentage limit
        max_value_by_pct = portfolio.equity * Decimal(str(self.config.max_position_pct))

        if total_position_value > max_value_by_pct:
            return RiskCheckResult(
                check_name="position_limit",
                result=CheckResult.FAILED,
                message=f"Position would exceed max % of equity",
                details={
                    "total_value": float(total_position_value),
                    "max_value": float(max_value_by_pct),
                    "max_pct": self.config.max_position_pct,
                },
            )

        # Check total number of positions (only for new positions)
        if existing_position is None and portfolio.position_count >= self.config.max_total_positions:
            return RiskCheckResult(
                check_name="position_limit",
                result=CheckResult.FAILED,
                message=f"Max number of positions reached",
                details={
                    "current_positions": portfolio.position_count,
                    "max_positions": self.config.max_total_positions,
                },
            )

        return RiskCheckResult(
            check_name="position_limit",
            result=CheckResult.PASSED,
            message="Position within limits",
        )

    def check_concentration(
        self,
        order: Order,
        portfolio: Portfolio,
        current_price: Decimal,
    ) -> RiskCheckResult:
        """Check portfolio concentration limits."""
        order_value = order.quantity * current_price
        existing_position = portfolio.get_position(order.symbol)
        existing_value = abs(existing_position.market_value) if existing_position else Decimal("0")
        total_position_value = existing_value + order_value

        if portfolio.equity > 0:
            concentration = float(total_position_value / portfolio.equity)
            if concentration > self.config.max_position_pct:
                return RiskCheckResult(
                    check_name="concentration",
                    result=CheckResult.WARNING,
                    message=f"High concentration in {order.symbol}",
                    details={
                        "concentration": concentration,
                        "threshold": self.config.max_position_pct,
                    },
                )

        return RiskCheckResult(
            check_name="concentration",
            result=CheckResult.PASSED,
            message="Concentration within limits",
        )

    def check_blacklist(self, order: Order) -> RiskCheckResult:
        """Check if symbol is on blacklist."""
        if order.symbol.upper() in [s.upper() for s in self.config.symbol_blacklist]:
            return RiskCheckResult(
                check_name="blacklist",
                result=CheckResult.FAILED,
                message=f"{order.symbol} is on the blacklist",
                details={"symbol": order.symbol},
            )

        return RiskCheckResult(
            check_name="blacklist",
            result=CheckResult.PASSED,
            message="Symbol not on blacklist",
        )

    def check_trading_hours(self) -> RiskCheckResult:
        """Check if within trading hours.

        CRITICAL FIX: Trading hours are in US Eastern Time, not UTC.
        The configured hours (9:30 AM - 4:00 PM) are Eastern Time.
        """
        # Convert current time to Eastern Time for proper comparison
        eastern = ZoneInfo("America/New_York")
        now_eastern = datetime.now(eastern)

        start_time = now_eastern.replace(
            hour=self.config.trading_start_hour,
            minute=self.config.trading_start_minute,
            second=0,
            microsecond=0,
        )
        end_time = now_eastern.replace(
            hour=self.config.trading_end_hour,
            minute=self.config.trading_end_minute,
            second=0,
            microsecond=0,
        )

        # Check weekends (Saturday=5, Sunday=6)
        if now_eastern.weekday() >= 5:
            return RiskCheckResult(
                check_name="trading_hours",
                result=CheckResult.WARNING,
                message="Market closed - weekend",
                details={
                    "current_time_eastern": now_eastern.isoformat(),
                    "day_of_week": now_eastern.strftime("%A"),
                },
            )

        # Simple check - doesn't account for holidays
        if not (start_time <= now_eastern <= end_time):
            return RiskCheckResult(
                check_name="trading_hours",
                result=CheckResult.WARNING,
                message="Outside regular trading hours",
                details={
                    "current_time_eastern": now_eastern.isoformat(),
                    "trading_start": start_time.strftime("%H:%M ET"),
                    "trading_end": end_time.strftime("%H:%M ET"),
                },
            )

        return RiskCheckResult(
            check_name="trading_hours",
            result=CheckResult.PASSED,
            message="Within trading hours",
        )

    def check_order_size(
        self,
        order: Order,
        current_price: Decimal,
    ) -> RiskCheckResult:
        """Check order size limits."""
        order_value = order.quantity * current_price

        if order_value < self.config.min_order_value:
            return RiskCheckResult(
                check_name="order_size",
                result=CheckResult.FAILED,
                message=f"Order value below minimum",
                details={
                    "order_value": float(order_value),
                    "min_value": float(self.config.min_order_value),
                },
            )

        if order_value > self.config.max_order_value:
            return RiskCheckResult(
                check_name="order_size",
                result=CheckResult.FAILED,
                message=f"Order value exceeds maximum",
                details={
                    "order_value": float(order_value),
                    "max_value": float(self.config.max_order_value),
                },
            )

        return RiskCheckResult(
            check_name="order_size",
            result=CheckResult.PASSED,
            message="Order size within limits",
        )

    def check_daily_limits(
        self,
        order: Order,
        portfolio: Portfolio,
        current_price: Decimal,
    ) -> RiskCheckResult:
        """Check daily trading limits."""
        # Check trade count
        if self._daily_trades >= self.config.max_daily_trades:
            return RiskCheckResult(
                check_name="daily_limits",
                result=CheckResult.FAILED,
                message="Daily trade limit reached",
                details={
                    "trades_today": self._daily_trades,
                    "max_trades": self.config.max_daily_trades,
                },
            )

        # Check turnover
        order_value = order.quantity * current_price
        new_turnover = self._daily_turnover + order_value
        max_turnover = portfolio.equity * Decimal(str(self.config.max_daily_turnover_pct))

        if new_turnover > max_turnover:
            return RiskCheckResult(
                check_name="daily_limits",
                result=CheckResult.FAILED,
                message="Daily turnover limit would be exceeded",
                details={
                    "current_turnover": float(self._daily_turnover),
                    "order_value": float(order_value),
                    "max_turnover": float(max_turnover),
                },
            )

        return RiskCheckResult(
            check_name="daily_limits",
            result=CheckResult.PASSED,
            message="Within daily limits",
        )

    def record_trade(self, order_value: Decimal) -> None:
        """Record a completed trade for daily limit tracking.

        THREAD-SAFE: Uses lock to protect counter updates.
        """
        self._reset_daily_counters_if_needed()
        with self._lock:
            self._daily_trades += 1
            self._daily_turnover += order_value


class IntraTradeMonitor:
    """Monitor orders during execution.

    THREAD-SAFE: Uses RLock to protect order tracking dictionaries.
    """

    def __init__(
        self,
        max_slippage_bps: float = 50.0,
        stuck_order_timeout_seconds: int = 300,
    ) -> None:
        """Initialize intra-trade monitor.

        Args:
            max_slippage_bps: Maximum allowed slippage in basis points.
            stuck_order_timeout_seconds: Time before order considered stuck.
        """
        self.max_slippage_bps = max_slippage_bps
        self.stuck_order_timeout_seconds = stuck_order_timeout_seconds
        # THREAD SAFETY: Use lock for dictionary operations
        self._lock = threading.RLock()
        self._order_submit_times: dict[str, datetime] = {}
        self._order_expected_prices: dict[str, Decimal] = {}

    def register_order(self, order: Order, expected_price: Decimal) -> None:
        """Register an order for monitoring.

        THREAD-SAFE: Uses lock to protect dictionary updates.

        Args:
            order: Order to monitor.
            expected_price: Expected execution price.
        """
        order_id = str(order.order_id)
        with self._lock:
            self._order_submit_times[order_id] = datetime.now(timezone.utc)
            self._order_expected_prices[order_id] = expected_price

    def check_slippage(
        self,
        order: Order,
        fill_price: Decimal,
    ) -> RiskCheckResult:
        """Check execution slippage.

        Args:
            order: Filled order.
            fill_price: Actual fill price.

        Returns:
            Check result.
        """
        order_id = str(order.order_id)
        expected_price = self._order_expected_prices.get(order_id)

        if expected_price is None or expected_price == 0:
            return RiskCheckResult(
                check_name="slippage",
                result=CheckResult.WARNING,
                message="No expected price recorded for slippage check",
            )

        slippage_pct = abs(float((fill_price - expected_price) / expected_price)) * 10000  # bps

        if slippage_pct > self.max_slippage_bps:
            return RiskCheckResult(
                check_name="slippage",
                result=CheckResult.WARNING,
                message=f"Slippage exceeded threshold",
                details={
                    "slippage_bps": slippage_pct,
                    "max_slippage_bps": self.max_slippage_bps,
                    "expected_price": float(expected_price),
                    "fill_price": float(fill_price),
                },
            )

        return RiskCheckResult(
            check_name="slippage",
            result=CheckResult.PASSED,
            message=f"Slippage within limits ({slippage_pct:.1f} bps)",
        )

    def check_stuck_orders(self, active_orders: list[Order]) -> list[RiskCheckResult]:
        """Check for stuck orders.

        Args:
            active_orders: List of currently active orders.

        Returns:
            List of check results for stuck orders.
        """
        results = []
        now = datetime.now(timezone.utc)

        for order in active_orders:
            order_id = str(order.order_id)
            submit_time = self._order_submit_times.get(order_id)

            if submit_time is None:
                continue

            elapsed = (now - submit_time).total_seconds()

            if elapsed > self.stuck_order_timeout_seconds:
                results.append(RiskCheckResult(
                    check_name="stuck_order",
                    result=CheckResult.WARNING,
                    message=f"Order {order_id} may be stuck",
                    details={
                        "order_id": order_id,
                        "symbol": order.symbol,
                        "elapsed_seconds": elapsed,
                        "timeout_seconds": self.stuck_order_timeout_seconds,
                    },
                ))

        return results

    def cleanup_order(self, order_id: str) -> None:
        """Clean up tracking data for a completed order.

        THREAD-SAFE: Uses lock to protect dictionary updates.
        """
        with self._lock:
            self._order_submit_times.pop(order_id, None)
            self._order_expected_prices.pop(order_id, None)


class PostTradeValidator:
    """Post-trade validation and reconciliation."""

    def __init__(self, max_cost_deviation_pct: float = 0.10) -> None:
        """Initialize post-trade validator.

        Args:
            max_cost_deviation_pct: Max deviation from estimated costs.
        """
        self.max_cost_deviation_pct = max_cost_deviation_pct

    def validate_execution(
        self,
        order: Order,
        estimated_cost: Decimal,
        actual_cost: Decimal,
    ) -> RiskCheckResult:
        """Validate execution quality.

        Args:
            order: Completed order.
            estimated_cost: Estimated transaction cost.
            actual_cost: Actual transaction cost.

        Returns:
            Check result.
        """
        if estimated_cost == 0:
            return RiskCheckResult(
                check_name="execution_quality",
                result=CheckResult.PASSED,
                message="No cost estimate for comparison",
            )

        deviation = abs(float((actual_cost - estimated_cost) / estimated_cost))

        if deviation > self.max_cost_deviation_pct:
            return RiskCheckResult(
                check_name="execution_quality",
                result=CheckResult.WARNING,
                message="Transaction cost deviation exceeded threshold",
                details={
                    "estimated_cost": float(estimated_cost),
                    "actual_cost": float(actual_cost),
                    "deviation_pct": deviation * 100,
                    "threshold_pct": self.max_cost_deviation_pct * 100,
                },
            )

        return RiskCheckResult(
            check_name="execution_quality",
            result=CheckResult.PASSED,
            message=f"Execution quality acceptable ({deviation * 100:.1f}% deviation)",
        )

    def reconcile_position(
        self,
        expected_position: Position,
        actual_position: Position,
    ) -> RiskCheckResult:
        """Reconcile expected vs actual position.

        Args:
            expected_position: Expected position state.
            actual_position: Actual position from broker.

        Returns:
            Check result.
        """
        qty_mismatch = expected_position.quantity != actual_position.quantity
        price_mismatch = abs(expected_position.avg_entry_price - actual_position.avg_entry_price) > Decimal("0.01")

        if qty_mismatch or price_mismatch:
            return RiskCheckResult(
                check_name="position_reconciliation",
                result=CheckResult.WARNING,
                message=f"Position mismatch for {expected_position.symbol}",
                details={
                    "expected_qty": float(expected_position.quantity),
                    "actual_qty": float(actual_position.quantity),
                    "expected_price": float(expected_position.avg_entry_price),
                    "actual_price": float(actual_position.avg_entry_price),
                },
            )

        return RiskCheckResult(
            check_name="position_reconciliation",
            result=CheckResult.PASSED,
            message="Position reconciled successfully",
        )


class KillSwitch:
    """Emergency kill switch for halting all trading.

    THREAD-SAFE: Uses RLock to ensure atomic state transitions.

    SAFETY FEATURE: Implements a 30-minute cooldown period before the kill switch
    can be reset. This prevents premature re-enabling of trading after a risk event.
    """

    # Cooldown period before kill switch can be reset (30 minutes)
    COOLDOWN_MINUTES: int = 30

    def __init__(
        self,
        config: RiskLimitsConfig | None = None,
        event_bus: EventBus | None = None,
        cancel_orders_callback: Callable[[], int] | None = None,
        close_positions_callback: Callable[[], int] | None = None,
        cooldown_minutes: int | None = None,
    ) -> None:
        """Initialize kill switch.

        Args:
            config: Risk limits configuration.
            event_bus: Event bus for publishing events.
            cancel_orders_callback: Callback to cancel all orders.
            close_positions_callback: Callback to close all positions.
            cooldown_minutes: Override default cooldown period (default: 30 minutes).
        """
        self.config = config or RiskLimitsConfig()
        self.event_bus = event_bus
        self.cancel_orders_callback = cancel_orders_callback
        self.close_positions_callback = close_positions_callback
        self.state = KillSwitchState()
        # CRITICAL: Use RLock for thread-safe atomic operations
        self._lock = threading.RLock()
        # Cooldown period configuration
        self._cooldown_minutes = cooldown_minutes if cooldown_minutes is not None else self.COOLDOWN_MINUTES

    def check_conditions(
        self,
        current_drawdown: float,
        pnl_decline_rate: float | None = None,
        has_system_error: bool = False,
        has_data_feed_failure: bool = False,
        has_broker_connection_loss: bool = False,
    ) -> tuple[bool, KillSwitchReason | None]:
        """Check if kill switch should be triggered.

        Args:
            current_drawdown: Current drawdown as decimal.
            pnl_decline_rate: Rate of P&L decline (if available).
            has_system_error: Whether there's a system error.
            has_data_feed_failure: Whether data feed has failed.
            has_broker_connection_loss: Whether broker connection is lost.

        Returns:
            Tuple of (should_trigger, reason).
        """
        # Max drawdown check
        if current_drawdown >= self.config.drawdown_halt_threshold:
            return True, KillSwitchReason.MAX_DRAWDOWN

        # Rapid P&L decline
        if pnl_decline_rate is not None and pnl_decline_rate < -0.05:  # 5% decline in short period
            return True, KillSwitchReason.RAPID_PNL_DECLINE

        # System conditions
        if has_system_error:
            return True, KillSwitchReason.SYSTEM_ERROR

        if has_data_feed_failure:
            return True, KillSwitchReason.DATA_FEED_FAILURE

        if has_broker_connection_loss:
            return True, KillSwitchReason.BROKER_CONNECTION_LOSS

        return False, None

    def activate(
        self,
        reason: KillSwitchReason,
        trigger_value: float | None = None,
        activated_by: str = "system",
        flatten_positions: bool = True,
    ) -> KillSwitchState:
        """Activate the kill switch.

        THREAD-SAFE: Uses lock to ensure atomic activation.

        Args:
            reason: Reason for activation.
            trigger_value: Value that triggered activation.
            activated_by: Who/what activated the switch.
            flatten_positions: Whether to close all positions.

        Returns:
            Kill switch state after activation.
        """
        # CRITICAL FIX: Use lock for atomic state transition
        with self._lock:
            # Check if already active to prevent duplicate activations
            if self.state.is_active:
                logger.warning(f"Kill switch already active, ignoring activation request")
                return self.state

            self.state.activate(reason, trigger_value, activated_by)

            logger.critical(f"KILL SWITCH ACTIVATED: {reason.value} by {activated_by}")

            # Cancel all orders (inside lock to protect state updates)
            if self.cancel_orders_callback:
                try:
                    self.state.orders_cancelled = self.cancel_orders_callback()
                    logger.info(f"Cancelled {self.state.orders_cancelled} orders")
                except Exception as e:
                    logger.error(f"Error cancelling orders: {e}")

            # Close all positions
            if flatten_positions and self.close_positions_callback:
                try:
                    self.state.positions_closed = self.close_positions_callback()
                    logger.info(f"Closed {self.state.positions_closed} positions")
                except Exception as e:
                    logger.error(f"Error closing positions: {e}")

            # Publish event
            if self.event_bus:
                event = create_risk_event(
                    event_type=EventType.KILL_SWITCH_TRIGGERED,
                    risk_data={
                        "reason": reason.value,
                        "trigger_value": trigger_value,
                        "activated_by": activated_by,
                        "orders_cancelled": self.state.orders_cancelled,
                        "positions_closed": self.state.positions_closed,
                    },
                    source="KillSwitch",
                )
                self.event_bus.publish(event)

            return self.state

    def manual_activate(self, activated_by: str = "manual") -> KillSwitchState:
        """Manually activate the kill switch.

        Args:
            activated_by: Identifier of who activated.

        Returns:
            Kill switch state.
        """
        return self.activate(
            reason=KillSwitchReason.MANUAL_ACTIVATION,
            activated_by=activated_by,
        )

    def get_cooldown_remaining(self) -> timedelta | None:
        """Get remaining cooldown time before kill switch can be reset.

        THREAD-SAFE: Uses lock for consistent read.

        Returns:
            Remaining cooldown time, or None if not active or cooldown has passed.
        """
        with self._lock:
            if not self.state.is_active or self.state.activated_at is None:
                return None

            elapsed = datetime.now(timezone.utc) - self.state.activated_at
            cooldown_duration = timedelta(minutes=self._cooldown_minutes)

            if elapsed >= cooldown_duration:
                return None

            return cooldown_duration - elapsed

    def can_reset(self) -> tuple[bool, str | None]:
        """Check if the kill switch can be reset (cooldown has passed).

        THREAD-SAFE: Uses lock for consistent read.

        Returns:
            Tuple of (can_reset, reason_if_not).
        """
        with self._lock:
            if not self.state.is_active:
                return False, "Kill switch is not active"

            if self.state.activated_at is None:
                # This shouldn't happen, but handle it safely
                return True, None

            elapsed = datetime.now(timezone.utc) - self.state.activated_at
            cooldown_duration = timedelta(minutes=self._cooldown_minutes)

            if elapsed < cooldown_duration:
                remaining = cooldown_duration - elapsed
                minutes_remaining = int(remaining.total_seconds() / 60)
                seconds_remaining = int(remaining.total_seconds() % 60)
                return False, (
                    f"Cooldown period not elapsed. "
                    f"Remaining: {minutes_remaining}m {seconds_remaining}s"
                )

            return True, None

    def reset(
        self,
        authorized_by: str,
        force: bool = False,
        override_code: str | None = None,
    ) -> tuple[bool, str]:
        """Reset the kill switch with cooldown enforcement.

        THREAD-SAFE: Uses lock for atomic reset.

        SAFETY: The kill switch cannot be reset until the cooldown period (30 minutes)
        has elapsed since activation. This prevents premature re-enabling of trading
        after a risk event.

        CRITICAL FIX: Force reset now requires an override_code that must match
        the KILL_SWITCH_OVERRIDE_CODE environment variable. This implements
        2-factor authorization for emergency overrides.

        Args:
            authorized_by: Who authorized the reset.
            force: If True, bypass cooldown check (DANGEROUS - use only in emergencies).
                   Requires explicit acknowledgment and will be logged at CRITICAL level.
            override_code: Required for force=True. Must match KILL_SWITCH_OVERRIDE_CODE
                          environment variable.

        Returns:
            Tuple of (success, message).
        """
        import os

        with self._lock:
            if not self.state.is_active:
                return False, "Kill switch is not active"

            # Check cooldown unless force override is specified
            if not force:
                can_reset, reason = self.can_reset()
                if not can_reset:
                    logger.warning(
                        f"Kill switch reset DENIED for {authorized_by}: {reason}"
                    )
                    return False, reason or "Cannot reset"

            # Force override - requires 2-factor authorization
            if force:
                # CRITICAL: Verify override code for force reset
                expected_code = os.environ.get("KILL_SWITCH_OVERRIDE_CODE")
                if not expected_code:
                    logger.error(
                        f"Force reset DENIED for {authorized_by}: "
                        "KILL_SWITCH_OVERRIDE_CODE environment variable not set"
                    )
                    return False, "Force reset not configured - contact system administrator"

                if override_code != expected_code:
                    logger.critical(
                        f"SECURITY ALERT: Force reset with INVALID override code "
                        f"attempted by {authorized_by}"
                    )
                    return False, "Invalid override code - force reset denied"

                logger.critical(
                    f"KILL SWITCH FORCE RESET by {authorized_by} - "
                    f"Cooldown bypassed with valid override code! "
                    f"Original activation: {self.state.reason.value if self.state.reason else 'unknown'} "
                    f"at {self.state.activated_at.isoformat() if self.state.activated_at else 'unknown'}"
                )

            # Calculate how long the kill switch was active
            active_duration = None
            if self.state.activated_at:
                active_duration = datetime.now(timezone.utc) - self.state.activated_at

            logger.warning(
                f"Kill switch reset authorized by: {authorized_by} "
                f"(was active for {active_duration if active_duration else 'unknown duration'})"
            )
            self.state.reset()

            # Publish reset event
            if self.event_bus:
                event = create_risk_event(
                    event_type=EventType.KILL_SWITCH_RESET,
                    risk_data={
                        "authorized_by": authorized_by,
                        "force_override": force,
                        "active_duration_seconds": active_duration.total_seconds() if active_duration else None,
                    },
                    source="KillSwitch",
                )
                self.event_bus.publish(event)

            return True, "Kill switch reset successfully"

    def is_active(self) -> bool:
        """Check if kill switch is active.

        THREAD-SAFE: Read-only check with lock for memory visibility.
        """
        with self._lock:
            return self.state.is_active

    def can_trade(self) -> RiskCheckResult:
        """Check if trading is allowed.

        THREAD-SAFE: Uses lock to ensure consistent read of state.
        CRITICAL FIX: Must use lock to prevent TOCTOU race condition.

        Returns:
            Check result indicating if trading is allowed.
        """
        with self._lock:
            if self.state.is_active:
                return RiskCheckResult(
                    check_name="kill_switch",
                    result=CheckResult.FAILED,
                    message="Trading halted - kill switch is active",
                    details={
                        "reason": self.state.reason.value if self.state.reason else "unknown",
                        "activated_at": self.state.activated_at.isoformat() if self.state.activated_at else None,
                    },
                )

            return RiskCheckResult(
                check_name="kill_switch",
                result=CheckResult.PASSED,
                message="Trading allowed",
            )


class RiskLimitsManager:
    """Central manager for all risk limit checks."""

    def __init__(
        self,
        config: RiskLimitsConfig | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize risk limits manager.

        Args:
            config: Risk limits configuration.
            event_bus: Event bus for publishing events.
        """
        self.config = config or RiskLimitsConfig()
        self.event_bus = event_bus

        self.pre_trade = PreTradeRiskChecker(config)
        self.intra_trade = IntraTradeMonitor()
        self.post_trade = PostTradeValidator()
        self.kill_switch = KillSwitch(config, event_bus)

    def pre_trade_check(
        self,
        order: Order,
        portfolio: Portfolio,
        current_price: Decimal,
    ) -> tuple[bool, list[RiskCheckResult]]:
        """Run all pre-trade checks.

        Args:
            order: Order to check.
            portfolio: Current portfolio.
            current_price: Current price.

        Returns:
            Tuple of (all_passed, list of results).
        """
        # First check kill switch
        kill_switch_result = self.kill_switch.can_trade()
        if not kill_switch_result.passed:
            return False, [kill_switch_result]

        # Run all pre-trade checks
        results = self.pre_trade.check_all(order, portfolio, current_price)

        # Check if any failed
        all_passed = all(r.result != CheckResult.FAILED for r in results)

        return all_passed, results

    def on_order_submitted(self, order: Order, expected_price: Decimal) -> None:
        """Handle order submission for monitoring.

        Args:
            order: Submitted order.
            expected_price: Expected fill price.
        """
        self.intra_trade.register_order(order, expected_price)

    def on_order_filled(
        self,
        order: Order,
        fill_price: Decimal,
        estimated_cost: Decimal,
        actual_cost: Decimal,
    ) -> list[RiskCheckResult]:
        """Handle order fill with validation.

        Args:
            order: Filled order.
            fill_price: Actual fill price.
            estimated_cost: Estimated cost.
            actual_cost: Actual cost.

        Returns:
            List of validation results.
        """
        results = []

        # Check slippage
        slippage_result = self.intra_trade.check_slippage(order, fill_price)
        results.append(slippage_result)

        # Validate execution quality
        execution_result = self.post_trade.validate_execution(order, estimated_cost, actual_cost)
        results.append(execution_result)

        # Record trade for daily limits
        order_value = order.quantity * fill_price
        self.pre_trade.record_trade(order_value)

        # Cleanup monitoring
        self.intra_trade.cleanup_order(str(order.order_id))

        return results

    def check_drawdown_limits(self, current_drawdown: float) -> RiskCheckResult:
        """Check drawdown against limits.

        Args:
            current_drawdown: Current drawdown as decimal.

        Returns:
            Check result with any required actions.
        """
        if current_drawdown >= self.config.drawdown_halt_threshold:
            # Trigger kill switch
            should_trigger, reason = self.kill_switch.check_conditions(current_drawdown)
            if should_trigger and reason:
                self.kill_switch.activate(reason, current_drawdown)
            return RiskCheckResult(
                check_name="drawdown",
                result=CheckResult.FAILED,
                message="Drawdown halt threshold breached - trading halted",
                details={
                    "current_drawdown": current_drawdown,
                    "halt_threshold": self.config.drawdown_halt_threshold,
                },
            )

        if current_drawdown >= self.config.drawdown_reduce_threshold:
            return RiskCheckResult(
                check_name="drawdown",
                result=CheckResult.WARNING,
                message="Drawdown reduction threshold breached - reduce position sizes",
                details={
                    "current_drawdown": current_drawdown,
                    "reduce_threshold": self.config.drawdown_reduce_threshold,
                    "recommended_action": "reduce_position_sizes",
                },
            )

        if current_drawdown >= self.config.drawdown_warning_threshold:
            return RiskCheckResult(
                check_name="drawdown",
                result=CheckResult.WARNING,
                message="Drawdown warning threshold breached",
                details={
                    "current_drawdown": current_drawdown,
                    "warning_threshold": self.config.drawdown_warning_threshold,
                },
            )

        return RiskCheckResult(
            check_name="drawdown",
            result=CheckResult.PASSED,
            message="Drawdown within limits",
        )
