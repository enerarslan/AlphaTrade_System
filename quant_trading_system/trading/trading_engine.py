"""
Main trading engine module.

Orchestrates the complete trading workflow:
- Pre-market initialization
- Market hours trading loop
- Post-market reconciliation
- State management and recovery

Coordinates all components: data, models, signals, execution.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable
from uuid import UUID

from quant_trading_system.core.data_types import (
    FeatureVector,
    ModelPrediction,
    OHLCVBar,
    Portfolio,
)
from quant_trading_system.core.events import (
    Event,
    EventBus,
    EventType,
    create_system_event,
)
from quant_trading_system.core.exceptions import ExecutionError, TradingSystemError
from quant_trading_system.execution.alpaca_client import AlpacaClient
from quant_trading_system.execution.order_manager import ManagedOrder, OrderManager, OrderRequest
from quant_trading_system.execution.position_tracker import PositionTracker
from quant_trading_system.trading.portfolio_manager import PortfolioManager
from quant_trading_system.trading.signal_generator import EnrichedSignal, SignalGenerator

logger = logging.getLogger(__name__)


class TradingMode(str, Enum):
    """Trading mode."""

    LIVE = "live"  # Real money trading
    PAPER = "paper"  # Paper trading with broker
    DRY_RUN = "dry_run"  # Signals only, no orders


class EngineState(str, Enum):
    """Engine state."""

    STOPPED = "stopped"
    STARTING = "starting"
    PRE_MARKET = "pre_market"
    MARKET_HOURS = "market_hours"
    POST_MARKET = "post_market"
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class TradingEngineConfig:
    """Trading engine configuration."""

    mode: TradingMode = TradingMode.PAPER
    symbols: list[str] = field(default_factory=list)
    bar_interval: str = "1Min"  # 1Min, 5Min, 15Min, 1Hour
    trading_start: time = time(9, 30)
    trading_end: time = time(16, 0)
    pre_market_minutes: int = 30
    post_market_minutes: int = 30
    max_daily_trades: int = 100
    max_daily_loss_pct: float = 0.02  # 2% max daily loss
    kill_switch_drawdown: float = 0.05  # 5% drawdown triggers kill switch
    enable_risk_checks: bool = True
    enable_reconciliation: bool = True
    heartbeat_interval: int = 60  # seconds
    state_save_interval: int = 300  # seconds


@dataclass
class TradingSession:
    """Trading session state."""

    session_id: str
    date: datetime
    state: EngineState = EngineState.STOPPED
    start_time: datetime | None = None
    end_time: datetime | None = None
    start_equity: Decimal = Decimal("0")
    current_equity: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    daily_pnl_pct: float = 0.0
    trades_today: int = 0
    signals_generated: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    errors: list[str] = field(default_factory=list)
    kill_switch_triggered: bool = False

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.state in (
            EngineState.PRE_MARKET,
            EngineState.MARKET_HOURS,
            EngineState.POST_MARKET,
        )


@dataclass
class EngineMetrics:
    """Trading engine metrics."""

    uptime_seconds: float = 0.0
    bars_processed: int = 0
    signals_generated: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    total_pnl: Decimal = Decimal("0")
    max_drawdown: float = 0.0  # Peak-to-trough drawdown (updated continuously)
    current_drawdown: float = 0.0  # Current drawdown from peak
    peak_equity: Decimal = Decimal("0")  # Highest equity level observed
    avg_latency_ms: float = 0.0
    errors_count: int = 0
    last_heartbeat: datetime | None = None


class TradingEngine:
    """Main trading engine orchestrating all components.

    Manages the complete trading workflow:
    1. Initialization and connection
    2. Pre-market preparation
    3. Market hours trading loop
    4. Post-market reconciliation
    5. State persistence and recovery
    """

    def __init__(
        self,
        config: TradingEngineConfig,
        client: AlpacaClient,
        order_manager: OrderManager,
        position_tracker: PositionTracker,
        signal_generator: SignalGenerator,
        portfolio_manager: PortfolioManager,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize trading engine.

        Args:
            config: Engine configuration.
            client: Alpaca API client.
            order_manager: Order management system.
            position_tracker: Position tracking system.
            signal_generator: Signal generation system.
            portfolio_manager: Portfolio management system.
            event_bus: Event bus for system events.
        """
        self.config = config
        self.client = client
        self.order_manager = order_manager
        self.position_tracker = position_tracker
        self.signal_generator = signal_generator
        self.portfolio_manager = portfolio_manager
        self.event_bus = event_bus

        # State
        self._state = EngineState.STOPPED
        self._session: TradingSession | None = None
        self._metrics = EngineMetrics()
        self._start_time: datetime | None = None

        # Tasks
        self._main_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._state_save_task: asyncio.Task | None = None

        # Data state
        self._latest_bars: dict[str, OHLCVBar] = {}
        self._latest_features: dict[str, FeatureVector] = {}
        self._latest_predictions: dict[str, ModelPrediction] = {}
        self._prices: dict[str, Decimal] = {}

        # Callbacks
        self._bar_callbacks: list[Callable[[dict[str, OHLCVBar]], None]] = []
        self._signal_callbacks: list[Callable[[list[EnrichedSignal]], None]] = []

        # Register for order events
        order_manager.on_fill(self._on_order_fill)
        order_manager.on_rejection(self._on_order_rejection)

    @property
    def state(self) -> EngineState:
        """Get current engine state."""
        return self._state

    @property
    def session(self) -> TradingSession | None:
        """Get current trading session."""
        return self._session

    @property
    def metrics(self) -> EngineMetrics:
        """Get engine metrics."""
        return self._metrics

    async def start(self) -> None:
        """Start the trading engine."""
        if self._state != EngineState.STOPPED:
            raise TradingSystemError("Engine is already running")

        logger.info(f"Starting trading engine in {self.config.mode.value} mode")
        self._state = EngineState.STARTING
        self._start_time = datetime.now(timezone.utc)

        try:
            # Connect to broker
            await self.client.connect()

            # Start components
            await self.order_manager.start()
            await self.position_tracker.start()

            # Initialize session
            self._session = TradingSession(
                session_id=f"SESSION-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
                date=datetime.now(timezone.utc).date(),
            )

            # Get initial portfolio state
            portfolio = self.position_tracker.get_portfolio()
            self._session.start_equity = portfolio.equity
            self._session.current_equity = portfolio.equity

            # Initialize drawdown tracking with starting equity as peak
            self._metrics.peak_equity = portfolio.equity
            self._metrics.current_drawdown = 0.0
            self._metrics.max_drawdown = 0.0
            logger.info(f"Initialized drawdown tracking with peak equity: ${portfolio.equity:.2f}")

            # Start background tasks
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._state_save_task = asyncio.create_task(self._state_save_loop())

            # Start main trading loop
            self._main_task = asyncio.create_task(self._main_loop())

            self._publish_event(EventType.SYSTEM_START, "Trading engine started")
            logger.info("Trading engine started successfully")

        except Exception as e:
            self._state = EngineState.ERROR
            logger.error(f"Failed to start trading engine: {e}")
            raise

    async def stop(self) -> None:
        """Stop the trading engine."""
        if self._state == EngineState.STOPPED:
            return

        logger.info("Stopping trading engine")
        self._state = EngineState.SHUTTING_DOWN

        # Cancel open orders in dry run mode is a no-op
        if self.config.mode != TradingMode.DRY_RUN:
            try:
                await self.order_manager.cancel_all_orders()
            except Exception as e:
                logger.warning(f"Error cancelling orders: {e}")

        # Stop tasks
        for task in [self._main_task, self._heartbeat_task, self._state_save_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop components
        await self.order_manager.stop()
        await self.position_tracker.stop()
        await self.client.disconnect()

        # Finalize session
        if self._session:
            self._session.end_time = datetime.now(timezone.utc)
            self._session.state = EngineState.STOPPED

        self._state = EngineState.STOPPED
        self._publish_event(EventType.SYSTEM_STOP, "Trading engine stopped")
        logger.info("Trading engine stopped")

    async def pause(self) -> None:
        """Pause trading (no new orders)."""
        if self._state == EngineState.MARKET_HOURS:
            self._state = EngineState.PAUSED
            logger.info("Trading engine paused")

    async def resume(self) -> None:
        """Resume trading."""
        if self._state == EngineState.PAUSED:
            self._state = EngineState.MARKET_HOURS
            logger.info("Trading engine resumed")

    def trigger_kill_switch(self, reason: str) -> None:
        """Trigger kill switch to stop all trading.

        Args:
            reason: Reason for triggering kill switch.
        """
        logger.critical(f"KILL SWITCH TRIGGERED: {reason}")

        if self._session:
            self._session.kill_switch_triggered = True
            self._session.errors.append(f"Kill switch: {reason}")

        self._state = EngineState.PAUSED

        # Publish critical event
        self._publish_event(
            EventType.KILL_SWITCH_TRIGGERED,
            reason,
            details={"timestamp": datetime.now(timezone.utc).isoformat()},
        )

    async def _main_loop(self) -> None:
        """Main trading loop."""
        while self._state not in (EngineState.STOPPED, EngineState.SHUTTING_DOWN):
            try:
                now = datetime.now(timezone.utc)
                current_time = now.time()

                # Determine market state
                if self._is_pre_market(current_time):
                    if self._state != EngineState.PRE_MARKET:
                        await self._enter_pre_market()
                    await self._pre_market_tasks()

                elif self._is_market_hours(current_time):
                    if self._state != EngineState.MARKET_HOURS and self._state != EngineState.PAUSED:
                        await self._enter_market_hours()
                    if self._state == EngineState.MARKET_HOURS:
                        await self._market_hours_iteration()

                elif self._is_post_market(current_time):
                    if self._state != EngineState.POST_MARKET:
                        await self._enter_post_market()
                    await self._post_market_tasks()

                else:
                    # Outside trading hours
                    await asyncio.sleep(60)

                # Short sleep between iterations
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                self._metrics.errors_count += 1
                if self._session:
                    self._session.errors.append(str(e))
                await asyncio.sleep(5)

    async def _enter_pre_market(self) -> None:
        """Enter pre-market state."""
        logger.info("Entering pre-market")
        self._state = EngineState.PRE_MARKET
        if self._session:
            self._session.state = EngineState.PRE_MARKET
            self._session.start_time = datetime.now(timezone.utc)

    async def _pre_market_tasks(self) -> None:
        """Execute pre-market tasks."""
        # Check system health
        try:
            account = await self.client.get_account()
            if account.trading_blocked:
                self.trigger_kill_switch("Trading blocked on account")
                return

            # Sync positions
            await self.position_tracker.sync_with_broker()

            # Update session equity
            if self._session:
                portfolio = self.position_tracker.get_portfolio()
                self._session.start_equity = portfolio.equity
                self._session.current_equity = portfolio.equity

        except Exception as e:
            logger.error(f"Pre-market check failed: {e}")

        await asyncio.sleep(30)  # Check every 30 seconds

    async def _enter_market_hours(self) -> None:
        """Enter market hours state."""
        logger.info("Entering market hours")
        self._state = EngineState.MARKET_HOURS
        if self._session:
            self._session.state = EngineState.MARKET_HOURS

    async def _market_hours_iteration(self) -> None:
        """Single iteration of market hours trading loop."""
        iteration_start = datetime.now(timezone.utc)

        try:
            # 1. Get latest market data
            await self._update_market_data()

            # 2. Check risk limits
            if self.config.enable_risk_checks:
                if not self._check_risk_limits():
                    return

            # 3. Generate signals
            portfolio = self.position_tracker.get_portfolio()
            signals = self.signal_generator.generate_signals(
                bars=self._latest_bars,
                features=self._latest_features,
                predictions=self._latest_predictions,
                portfolio=portfolio,
            )

            if self._session:
                self._session.signals_generated += len(signals)
            self._metrics.signals_generated += len(signals)

            # Invoke callbacks
            for callback in self._signal_callbacks:
                try:
                    callback(signals)
                except Exception as e:
                    logger.error(f"Signal callback error: {e}")

            # 4. Build target portfolio
            if signals:
                target = self.portfolio_manager.build_target_portfolio(
                    signals=signals,
                    portfolio=portfolio,
                    prices=self._prices,
                )

                # 5. Check if rebalancing needed
                needs_rebalance, reason = self.portfolio_manager.check_rebalance_needed(
                    target=target,
                    portfolio=portfolio,
                    prices=self._prices,
                )

                # 6. Generate and execute trades
                if needs_rebalance:
                    await self._execute_rebalance(target, portfolio)

            # Update metrics
            self._metrics.bars_processed += len(self._latest_bars)

            # Calculate latency
            latency_ms = (datetime.now(timezone.utc) - iteration_start).total_seconds() * 1000
            self._update_avg_latency(latency_ms)

        except Exception as e:
            logger.error(f"Market hours iteration error: {e}")
            self._metrics.errors_count += 1

        # Wait for next bar interval
        await self._wait_for_next_bar()

    async def _execute_rebalance(
        self,
        target: Any,
        portfolio: Portfolio,
    ) -> None:
        """Execute rebalancing trades.

        Args:
            target: Target portfolio.
            portfolio: Current portfolio.
        """
        if self.config.mode == TradingMode.DRY_RUN:
            logger.info("DRY RUN: Would execute rebalance trades")
            return

        # Check trade limits
        if self._session and self._session.trades_today >= self.config.max_daily_trades:
            logger.warning("Daily trade limit reached")
            return

        # Generate trades
        trades = self.portfolio_manager.generate_rebalance_trades(
            target=target,
            portfolio=portfolio,
            prices=self._prices,
        )

        if not trades:
            return

        # Convert to order requests
        order_requests = self.portfolio_manager.create_order_requests(
            trades=trades,
            strategy_id="TradingEngine",
        )

        # Submit orders
        for request in order_requests:
            try:
                managed = self.order_manager.create_order(
                    request=request,
                    portfolio=portfolio,
                    current_price=self._prices.get(request.symbol),
                )
                await self.order_manager.submit_order(
                    managed=managed,
                    current_price=self._prices.get(request.symbol),
                )

                if self._session:
                    self._session.orders_submitted += 1
                self._metrics.orders_submitted += 1

            except Exception as e:
                logger.error(f"Failed to submit order for {request.symbol}: {e}")

        self.portfolio_manager.clear_pending_trades()

    async def _enter_post_market(self) -> None:
        """Enter post-market state."""
        logger.info("Entering post-market")
        self._state = EngineState.POST_MARKET
        if self._session:
            self._session.state = EngineState.POST_MARKET

    async def _post_market_tasks(self) -> None:
        """Execute post-market tasks."""
        try:
            # Reconcile positions
            if self.config.enable_reconciliation:
                result = await self.position_tracker.reconcile_with_broker(
                    auto_correct=True
                )
                if not result.is_consistent:
                    logger.warning(f"Position discrepancies found: {result.discrepancies}")

            # Calculate daily P&L
            if self._session:
                portfolio = self.position_tracker.get_portfolio()
                self._session.current_equity = portfolio.equity
                self._session.daily_pnl = portfolio.equity - self._session.start_equity
                if self._session.start_equity > 0:
                    self._session.daily_pnl_pct = float(
                        self._session.daily_pnl / self._session.start_equity
                    )

                logger.info(
                    f"Daily P&L: ${self._session.daily_pnl:.2f} "
                    f"({self._session.daily_pnl_pct:.2%})"
                )

            # Generate end-of-day report
            await self._generate_eod_report()

        except Exception as e:
            logger.error(f"Post-market tasks error: {e}")

        await asyncio.sleep(60)

    async def _update_market_data(self) -> None:
        """Update market data for all symbols."""
        for symbol in self.config.symbols:
            try:
                # Get latest bar from Alpaca
                bar_data = await self.client.get_latest_bar(symbol)
                if bar_data:
                    bar = OHLCVBar(
                        symbol=symbol,
                        timestamp=datetime.fromisoformat(
                            bar_data["t"].replace("Z", "+00:00")
                        ),
                        open=Decimal(str(bar_data["o"])),
                        high=Decimal(str(bar_data["h"])),
                        low=Decimal(str(bar_data["l"])),
                        close=Decimal(str(bar_data["c"])),
                        volume=int(bar_data["v"]),
                        vwap=Decimal(str(bar_data.get("vw", bar_data["c"]))),
                    )
                    self._latest_bars[symbol] = bar
                    self._prices[symbol] = bar.close

            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {e}")

        # Update position prices
        self.position_tracker.update_prices(self._prices)

        # Invoke bar callbacks
        for callback in self._bar_callbacks:
            try:
                callback(self._latest_bars)
            except Exception as e:
                logger.error(f"Bar callback error: {e}")

    def _check_risk_limits(self) -> bool:
        """Check risk limits.

        Returns:
            True if trading can continue.
        """
        if not self._session:
            return True

        portfolio = self.position_tracker.get_portfolio()
        self._session.current_equity = portfolio.equity

        # Calculate current P&L
        self._session.daily_pnl = portfolio.equity - self._session.start_equity
        if self._session.start_equity > 0:
            self._session.daily_pnl_pct = float(
                self._session.daily_pnl / self._session.start_equity
            )

        # UPDATE DRAWDOWN METRICS (Critical fix: max_drawdown was never calculated before)
        # This must be called BEFORE checking drawdown limit
        self._update_drawdown(portfolio.equity)

        # Check daily loss limit
        if self._session.daily_pnl_pct < -self.config.max_daily_loss_pct:
            self.trigger_kill_switch(
                f"Daily loss limit exceeded: {self._session.daily_pnl_pct:.2%}"
            )
            return False

        # Check drawdown limit (now actually works since max_drawdown is calculated)
        if self._metrics.max_drawdown > self.config.kill_switch_drawdown:
            self.trigger_kill_switch(
                f"Max drawdown exceeded: {self._metrics.max_drawdown:.2%} "
                f"(threshold: {self.config.kill_switch_drawdown:.2%})"
            )
            return False

        return True

    async def _wait_for_next_bar(self) -> None:
        """Wait for next bar based on interval."""
        interval_map = {
            "1Min": 60,
            "5Min": 300,
            "15Min": 900,
            "1Hour": 3600,
        }
        interval_seconds = interval_map.get(self.config.bar_interval, 60)

        now = datetime.now(timezone.utc)
        seconds_into_interval = now.timestamp() % interval_seconds
        wait_time = interval_seconds - seconds_into_interval

        await asyncio.sleep(wait_time)

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat task."""
        while self._state not in (EngineState.STOPPED, EngineState.SHUTTING_DOWN):
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                self._metrics.last_heartbeat = datetime.now(timezone.utc)
                if self._start_time:
                    self._metrics.uptime_seconds = (
                        datetime.now(timezone.utc) - self._start_time
                    ).total_seconds()

                self._publish_event(EventType.HEARTBEAT, "Engine heartbeat")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _state_save_loop(self) -> None:
        """Background state persistence task."""
        while self._state not in (EngineState.STOPPED, EngineState.SHUTTING_DOWN):
            try:
                await asyncio.sleep(self.config.state_save_interval)
                await self._save_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"State save error: {e}")

    async def _save_state(self) -> None:
        """Save engine state for recovery."""
        state = {
            "session": {
                "session_id": self._session.session_id if self._session else None,
                "state": self._state.value,
                "start_equity": str(self._session.start_equity) if self._session else "0",
                "current_equity": str(self._session.current_equity) if self._session else "0",
                "trades_today": self._session.trades_today if self._session else 0,
            },
            "metrics": {
                "bars_processed": self._metrics.bars_processed,
                "signals_generated": self._metrics.signals_generated,
                "orders_submitted": self._metrics.orders_submitted,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # In production, would persist to file/database
        logger.debug(f"State saved: {state}")

    async def _generate_eod_report(self) -> None:
        """Generate end-of-day report."""
        if not self._session:
            return

        report = {
            "date": self._session.date.isoformat(),
            "session_id": self._session.session_id,
            "start_equity": str(self._session.start_equity),
            "end_equity": str(self._session.current_equity),
            "daily_pnl": str(self._session.daily_pnl),
            "daily_pnl_pct": f"{self._session.daily_pnl_pct:.2%}",
            "signals_generated": self._session.signals_generated,
            "orders_submitted": self._session.orders_submitted,
            "orders_filled": self._session.orders_filled,
            "trades": self._session.trades_today,
            "errors": len(self._session.errors),
        }

        logger.info(f"End of Day Report: {report}")

    def _on_order_fill(self, managed: ManagedOrder) -> None:
        """Handle order fill event."""
        if self._session:
            self._session.orders_filled += 1
            self._session.trades_today += 1
        self._metrics.orders_filled += 1

        # Update position tracker
        self.position_tracker.update_from_fill(
            order=managed.order,
            fill_qty=managed.order.filled_qty,
            fill_price=managed.order.filled_avg_price or Decimal("0"),
            strategy_id=managed.request.strategy_id,
        )

    def _on_order_rejection(self, managed: ManagedOrder) -> None:
        """Handle order rejection event."""
        self._metrics.orders_rejected += 1
        if self._session:
            self._session.errors.append(
                f"Order rejected: {managed.order.symbol} - {managed.error_message}"
            )

    def _is_pre_market(self, current_time: time) -> bool:
        """Check if in pre-market period."""
        pre_market_start = (
            datetime.combine(datetime.today(), self.config.trading_start)
            - timedelta(minutes=self.config.pre_market_minutes)
        ).time()
        return pre_market_start <= current_time < self.config.trading_start

    def _is_market_hours(self, current_time: time) -> bool:
        """Check if during market hours."""
        return self.config.trading_start <= current_time < self.config.trading_end

    def _is_post_market(self, current_time: time) -> bool:
        """Check if in post-market period."""
        post_market_end = (
            datetime.combine(datetime.today(), self.config.trading_end)
            + timedelta(minutes=self.config.post_market_minutes)
        ).time()
        return self.config.trading_end <= current_time < post_market_end

    def _update_avg_latency(self, latency_ms: float) -> None:
        """Update average latency with exponential moving average."""
        alpha = 0.1
        if self._metrics.avg_latency_ms == 0:
            self._metrics.avg_latency_ms = latency_ms
        else:
            self._metrics.avg_latency_ms = (
                alpha * latency_ms + (1 - alpha) * self._metrics.avg_latency_ms
            )

    def _update_drawdown(self, current_equity: Decimal) -> None:
        """Update drawdown metrics based on current equity.

        Calculates:
        - peak_equity: Highest equity observed during session
        - current_drawdown: Current drawdown from peak (as decimal, e.g., 0.05 = 5%)
        - max_drawdown: Maximum drawdown observed during session

        Args:
            current_equity: Current portfolio equity value.
        """
        if current_equity <= 0:
            # Avoid division by zero or negative equity edge cases
            logger.warning(f"Invalid equity value for drawdown calculation: {current_equity}")
            return

        # Update peak equity if current equity is higher
        if current_equity > self._metrics.peak_equity:
            self._metrics.peak_equity = current_equity

        # Calculate current drawdown from peak
        # Drawdown = (Peak - Current) / Peak
        if self._metrics.peak_equity > 0:
            drawdown = float(
                (self._metrics.peak_equity - current_equity) / self._metrics.peak_equity
            )
            self._metrics.current_drawdown = max(0.0, drawdown)  # Ensure non-negative

            # Update max drawdown if current is higher
            if self._metrics.current_drawdown > self._metrics.max_drawdown:
                self._metrics.max_drawdown = self._metrics.current_drawdown
                logger.info(
                    f"New max drawdown: {self._metrics.max_drawdown:.2%} "
                    f"(Peak: ${self._metrics.peak_equity:.2f}, Current: ${current_equity:.2f})"
                )

    def _publish_event(
        self,
        event_type: EventType,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Publish system event."""
        if not self.event_bus:
            return

        event = create_system_event(
            event_type=event_type,
            message=message,
            details=details,
            source="TradingEngine",
        )
        self.event_bus.publish(event)

    def on_bar(self, callback: Callable[[dict[str, OHLCVBar]], None]) -> None:
        """Register callback for bar updates."""
        self._bar_callbacks.append(callback)

    def on_signal(self, callback: Callable[[list[EnrichedSignal]], None]) -> None:
        """Register callback for signal generation."""
        self._signal_callbacks.append(callback)

    def get_statistics(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "state": self._state.value,
            "mode": self.config.mode.value,
            "uptime_seconds": self._metrics.uptime_seconds,
            "session": {
                "session_id": self._session.session_id if self._session else None,
                "start_equity": str(self._session.start_equity) if self._session else "0",
                "current_equity": str(self._session.current_equity) if self._session else "0",
                "daily_pnl": str(self._session.daily_pnl) if self._session else "0",
                "daily_pnl_pct": self._session.daily_pnl_pct if self._session else 0.0,
                "trades_today": self._session.trades_today if self._session else 0,
                "kill_switch": self._session.kill_switch_triggered if self._session else False,
            },
            "metrics": {
                "bars_processed": self._metrics.bars_processed,
                "signals_generated": self._metrics.signals_generated,
                "orders_submitted": self._metrics.orders_submitted,
                "orders_filled": self._metrics.orders_filled,
                "orders_rejected": self._metrics.orders_rejected,
                "avg_latency_ms": self._metrics.avg_latency_ms,
                "errors_count": self._metrics.errors_count,
                "max_drawdown": self._metrics.max_drawdown,
                "current_drawdown": self._metrics.current_drawdown,
                "peak_equity": str(self._metrics.peak_equity),
            },
            "components": {
                "order_manager": self.order_manager.get_statistics(),
                "position_tracker": self.position_tracker.get_statistics(),
                "signal_generator": self.signal_generator.get_statistics(),
                "portfolio_manager": self.portfolio_manager.get_statistics(),
            },
        }


# Factory function

def create_trading_engine(
    mode: TradingMode = TradingMode.PAPER,
    symbols: list[str] | None = None,
    api_key: str | None = None,
    api_secret: str | None = None,
) -> TradingEngine:
    """Create a trading engine with default components.

    Args:
        mode: Trading mode.
        symbols: Symbols to trade.
        api_key: Alpaca API key.
        api_secret: Alpaca API secret.

    Returns:
        Configured trading engine.
    """
    from quant_trading_system.execution.alpaca_client import TradingEnvironment

    # Create event bus
    event_bus = EventBus()

    # Create client
    environment = (
        TradingEnvironment.LIVE if mode == TradingMode.LIVE
        else TradingEnvironment.PAPER
    )
    client = AlpacaClient(
        api_key=api_key,
        api_secret=api_secret,
        environment=environment,
    )

    # Create components
    order_manager = OrderManager(client=client, event_bus=event_bus)
    position_tracker = PositionTracker(client=client, event_bus=event_bus)
    signal_generator = SignalGenerator(event_bus=event_bus)
    portfolio_manager = PortfolioManager(event_bus=event_bus)

    # Create config
    config = TradingEngineConfig(
        mode=mode,
        symbols=symbols or [],
    )

    return TradingEngine(
        config=config,
        client=client,
        order_manager=order_manager,
        position_tracker=position_tracker,
        signal_generator=signal_generator,
        portfolio_manager=portfolio_manager,
        event_bus=event_bus,
    )
