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
import signal
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable
from uuid import UUID
from zoneinfo import ZoneInfo

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
from quant_trading_system.core.exceptions import ExecutionError, TradingSystemError, StrategyError
from quant_trading_system.risk.limits import (
    CheckResult,
    KillSwitch,
    KillSwitchReason,
    PreTradeRiskChecker,
    RiskLimitsManager,
    RiskLimitsConfig,
)
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
    # CRITICAL FIX: enable_risk_checks removed - risk checks are now MANDATORY
    # Risk checks can NEVER be disabled in production. This prevents accidental
    # or intentional bypass of safety controls.
    enable_reconciliation: bool = True
    heartbeat_interval: int = 60  # seconds
    state_save_interval: int = 300  # seconds

    # MISSING FEATURE: Watchdog timer configuration
    watchdog_timeout: int = 120  # seconds - max time without heartbeat before alert
    watchdog_enabled: bool = True  # Enable watchdog timer for detecting stuck loops

    # MAJOR FIX: Strategy exception escalation configuration
    # Tracks consecutive strategy exceptions and triggers kill switch after threshold
    max_consecutive_strategy_exceptions: int = 3  # Kill switch after N consecutive exceptions
    strategy_exception_reset_interval: int = 300  # Reset counter after N seconds of success


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

        # CRITICAL: Thread safety for shared state modifications
        # Order fill/rejection callbacks may be called from broker websocket threads
        # while the main trading loop is also modifying session/metrics
        self._state_lock = threading.RLock()

        # MISSING FEATURE: Watchdog timer for detecting stuck main loop
        self._watchdog_task: asyncio.Task | None = None
        self._last_loop_heartbeat: datetime | None = None
        self._shutdown_event = asyncio.Event()

        # MISSING FEATURE: OS signal handlers for graceful shutdown
        self._signal_handlers_installed = False

        # MAJOR FIX: Strategy exception tracking for kill switch escalation
        self._consecutive_strategy_exceptions: int = 0
        self._last_successful_iteration: datetime | None = None

        # Use OrderManager's kill switch when available so submit/modify checks
        # and engine-level checks are guaranteed to share the same state.
        shared_kill_switch = getattr(order_manager, "kill_switch", None)
        self._global_kill_switch = (
            shared_kill_switch if isinstance(shared_kill_switch, KillSwitch) else KillSwitch()
        )

        # Centralized risk manager wiring (shared kill switch + pre-trade checks).
        risk_config = RiskLimitsConfig(
            daily_loss_limit_pct=self.config.max_daily_loss_pct,
            drawdown_halt_threshold=self.config.kill_switch_drawdown,
            max_daily_trades=self.config.max_daily_trades,
        )
        self._risk_checker = PreTradeRiskChecker(risk_config)
        self._risk_limits_manager = RiskLimitsManager(risk_config, event_bus)
        # Enforce a single kill switch state and pre-trade checker across components.
        self._risk_limits_manager.kill_switch = self._global_kill_switch
        self._risk_limits_manager.pre_trade = self._risk_checker

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

            # MISSING FEATURE: Start watchdog timer
            if self.config.watchdog_enabled:
                self._watchdog_task = asyncio.create_task(self._watchdog_loop())
                logger.info(f"Watchdog timer started (timeout: {self.config.watchdog_timeout}s)")

            # MISSING FEATURE: Install OS signal handlers for graceful shutdown
            self._install_signal_handlers()

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

        # Set shutdown event to signal watchdog
        self._shutdown_event.set()

        # Stop tasks (including watchdog)
        for task in [self._main_task, self._heartbeat_task, self._state_save_task, self._watchdog_task]:
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

    # =========================================================================
    # MISSING FEATURE: OS Signal Handlers for Graceful Shutdown
    # =========================================================================

    def _install_signal_handlers(self) -> None:
        """Install OS signal handlers for graceful shutdown.

        Handles:
        - SIGTERM: Graceful shutdown (used by systemd, Docker, Kubernetes)
        - SIGINT: Keyboard interrupt (Ctrl+C)

        On Windows, only SIGINT is reliably available.
        """
        if self._signal_handlers_installed:
            return

        def signal_handler(signum: int, frame: Any) -> None:
            """Handle OS signals by triggering graceful shutdown."""
            sig_name = signal.Signals(signum).name
            logger.warning(f"Received {sig_name} signal - initiating graceful shutdown")

            # Set shutdown event
            self._shutdown_event.set()

            # Create shutdown task if we have a running loop
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._graceful_shutdown(sig_name))
            except RuntimeError:
                # No running loop - just log
                logger.warning("No running event loop for graceful shutdown")

        # Install handlers based on platform
        if sys.platform != "win32":
            # Unix - handle both SIGTERM and SIGINT
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            logger.info("Installed signal handlers for SIGTERM and SIGINT")
        else:
            # Windows - only SIGINT is reliably available
            signal.signal(signal.SIGINT, signal_handler)
            logger.info("Installed signal handler for SIGINT (Windows)")

        self._signal_handlers_installed = True

    async def _graceful_shutdown(self, reason: str) -> None:
        """Perform graceful shutdown.

        1. Stop accepting new orders
        2. Cancel pending orders
        3. Save state
        4. Stop engine

        Args:
            reason: Reason for shutdown.
        """
        logger.info(f"Graceful shutdown initiated: {reason}")

        # First pause to stop new orders
        await self.pause()

        # Give pending orders time to fill (max 30 seconds)
        await asyncio.sleep(5)

        # Now fully stop
        await self.stop()

        logger.info("Graceful shutdown complete")

    # =========================================================================
    # MISSING FEATURE: Watchdog Timer for Stuck Loop Detection
    # =========================================================================

    async def _watchdog_loop(self) -> None:
        """Watchdog timer to detect stuck main loop.

        Monitors the main trading loop heartbeat and triggers alerts
        if the loop appears stuck (no heartbeat for watchdog_timeout seconds).
        """
        logger.info("Watchdog timer started")

        while not self._shutdown_event.is_set():
            try:
                # Wait for the watchdog interval
                await asyncio.sleep(self.config.watchdog_timeout / 2)

                # Check if shutdown was requested
                if self._shutdown_event.is_set():
                    break

                # Check main loop heartbeat
                if self._last_loop_heartbeat:
                    elapsed = (datetime.now(timezone.utc) - self._last_loop_heartbeat).total_seconds()

                    if elapsed > self.config.watchdog_timeout:
                        logger.critical(
                            f"WATCHDOG ALERT: Main loop stuck! "
                            f"No heartbeat for {elapsed:.1f} seconds"
                        )

                        # Publish critical alert
                        self._publish_event(
                            EventType.SYSTEM_ALERT,
                            "Main loop stuck - watchdog timeout exceeded",
                            details={
                                "last_heartbeat": self._last_loop_heartbeat.isoformat(),
                                "elapsed_seconds": elapsed,
                                "timeout_seconds": self.config.watchdog_timeout,
                            },
                        )

                        # Trigger kill switch to prevent trading in stuck state
                        self.trigger_kill_switch(
                            f"Watchdog timeout - no heartbeat for {elapsed:.1f}s"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Watchdog error: {e}")

        logger.info("Watchdog timer stopped")

    def _update_loop_heartbeat(self) -> None:
        """Update the main loop heartbeat timestamp.

        Called at the end of each main loop iteration to signal
        the watchdog that the loop is still running.
        """
        self._last_loop_heartbeat = datetime.now(timezone.utc)

    def trigger_kill_switch(self, reason: str) -> None:
        """Trigger kill switch to stop all trading.

        P0 FIX (January 2026 Audit): Now activates BOTH local and global kill switches.
        Previously only set a local flag, allowing orders to be submitted when global
        kill switch should have blocked them.

        THREAD SAFETY: May be called from multiple contexts (risk checks,
        callbacks, external triggers). Uses lock for state consistency.

        Args:
            reason: Reason for triggering kill switch.
        """
        logger.critical(f"KILL SWITCH TRIGGERED: {reason}")

        # P0 FIX: Activate global kill switch singleton
        # This ensures ALL components respect the kill switch, not just this engine
        # FIX: Use correct parameter name 'activated_by' instead of 'message'
        self._global_kill_switch.activate(
            reason=KillSwitchReason.MANUAL_ACTIVATION,
            activated_by=reason,
        )

        with self._state_lock:
            if self._session:
                self._session.kill_switch_triggered = True
                self._session.errors.append(f"Kill switch: {reason}")

            self._state = EngineState.PAUSED

        # Publish critical event (outside lock to avoid deadlock with event handlers)
        self._publish_event(
            EventType.KILL_SWITCH_TRIGGERED,
            reason,
            details={"timestamp": datetime.now(timezone.utc).isoformat()},
        )

    async def _main_loop(self) -> None:
        """Main trading loop."""
        while self._state not in (EngineState.STOPPED, EngineState.SHUTTING_DOWN):
            try:
                # CRITICAL FIX: Use Eastern Time for market hours detection
                # US market hours are defined in Eastern Time, not UTC
                current_time = self._get_eastern_time()

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

                # Update watchdog heartbeat to signal loop is alive
                self._update_loop_heartbeat()

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

            # 2. Check risk limits - MANDATORY (cannot be disabled)
            # CRITICAL FIX: Risk checks always run. Safety controls are non-negotiable.
            if not self._check_risk_limits():
                logger.warning("Risk limits check failed - skipping this iteration")
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

            # 4. Build target portfolio (including empty-signal case)
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

            # MAJOR FIX: Reset exception counter on successful iteration
            self._record_successful_iteration()

        except (StrategyError, ExecutionError) as e:
            # MAJOR FIX: Strategy-specific exceptions trigger escalation tracking
            logger.error(f"Strategy/execution error in iteration: {e}")
            self._metrics.errors_count += 1
            self._handle_strategy_exception(e)

        except Exception as e:
            logger.error(f"Market hours iteration error: {e}")
            self._metrics.errors_count += 1
            # Non-strategy exceptions also tracked for safety
            self._handle_strategy_exception(e)

        # Wait for next bar interval
        await self._wait_for_next_bar()

    async def _execute_rebalance(
        self,
        target: Any,
        portfolio: Portfolio,
    ) -> None:
        """Execute rebalancing trades.

        P0 FIX (January 2026 Audit): Added kill switch check and PreTradeRiskChecker.
        Previously, orders could be submitted even when kill switch was active or
        without proper risk validation.

        Args:
            target: Target portfolio.
            portfolio: Current portfolio.
        """
        # P0 FIX: Check global kill switch FIRST - before any order processing
        if self._global_kill_switch.is_active():
            # FIX: Use state.reason instead of non-existent _reason attribute
            reason = self._global_kill_switch.state.reason
            logger.warning(
                f"Kill switch active - blocking order submission: "
                f"{reason.value if reason else 'unknown'}"
            )
            return

        # P0 FIX: Also check local kill switch flag for redundancy
        # P0-7 FIX: Acquire lock when reading session state
        with self._state_lock:
            if self._session and self._session.kill_switch_triggered:
                logger.warning("Local kill switch triggered - blocking order submission")
                return

        if self.config.mode == TradingMode.DRY_RUN:
            logger.info("DRY RUN: Would execute rebalance trades")
            return

        # Check trade limits
        # P0-7 FIX: Acquire lock when reading session state
        with self._state_lock:
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

        # Submit orders with P0 FIX: PreTradeRiskChecker validation
        for request in order_requests:
            try:
                current_price = self._prices.get(request.symbol)

                # Create managed order first (required for risk check)
                managed = self.order_manager.create_order(
                    request=request,
                    portfolio=portfolio,
                    current_price=current_price,
                )

                # P0 FIX: Pre-trade risk check with portfolio lock for atomicity
                # FIX: Use check_all() instead of non-existent check_order()
                with self._risk_checker.portfolio_lock:
                    can_submit, risk_results = self._risk_limits_manager.pre_trade_check(
                        managed.order, portfolio, current_price or Decimal("0")
                    )

                    # Check if any risk check failed
                    failed_checks = [r for r in risk_results if r.result == CheckResult.FAILED]
                    warning_checks = [r for r in risk_results if r.result == CheckResult.WARNING]
                    if not can_submit or failed_checks:
                        # P0 FIX (January 2026 Audit): Use 'message' attribute instead of 'reason'
                        # RiskCheckResult dataclass has 'message' field, not 'reason'
                        reasons = "; ".join(r.message for r in failed_checks if r.message)
                        logger.warning(
                            f"Order rejected by PreTradeRiskChecker: {request.symbol} "
                            f"- {reasons}"
                        )
                        continue
                    if warning_checks:
                        warning_text = "; ".join(w.message for w in warning_checks if w.message)
                        logger.warning(
                            f"Order risk warnings for {request.symbol}: {warning_text}"
                        )

                    await self.order_manager.submit_order(
                        managed=managed,
                        current_price=current_price,
                    )

                # P0-7 FIX: Acquire lock when modifying session state
                with self._state_lock:
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

        P0-7 FIX (January 2026 Audit): Added lock to prevent race condition
        with websocket callbacks that also modify session state.
        """
        if not self._session:
            return True

        portfolio = self.position_tracker.get_portfolio()

        # P0-7 FIX: Acquire lock before modifying session state
        with self._state_lock:
            self._session.current_equity = portfolio.equity

            # Calculate current P&L
            self._session.daily_pnl = portfolio.equity - self._session.start_equity
            if self._session.start_equity > 0:
                self._session.daily_pnl_pct = float(
                    self._session.daily_pnl / self._session.start_equity
                )

            # Capture values we need outside the lock
            daily_pnl_pct = self._session.daily_pnl_pct

        # UPDATE DRAWDOWN METRICS (Critical fix: max_drawdown was never calculated before)
        # This must be called BEFORE checking drawdown limit
        self._update_drawdown(portfolio.equity)

        # Check daily loss limit
        if daily_pnl_pct < -self.config.max_daily_loss_pct:
            self.trigger_kill_switch(
                f"Daily loss limit exceeded: {daily_pnl_pct:.2%}"
            )
            return False

        # Centralized drawdown policy through risk manager.
        drawdown_result = self._risk_limits_manager.check_drawdown_limits(self._metrics.max_drawdown)
        if drawdown_result.result == CheckResult.FAILED:
            self.trigger_kill_switch(drawdown_result.message)
            return False
        if drawdown_result.result == CheckResult.WARNING:
            logger.warning(drawdown_result.message)

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
        """Handle order fill event.

        THREAD SAFETY: This callback is invoked from broker websocket threads.
        Uses lock to prevent race conditions when modifying shared state.
        """
        with self._state_lock:
            if self._session:
                self._session.orders_filled += 1
                self._session.trades_today += 1
            self._metrics.orders_filled += 1

        # Update position tracker (has its own thread safety)
        self.position_tracker.update_from_fill(
            order=managed.order,
            fill_qty=managed.order.filled_qty,
            fill_price=managed.order.filled_avg_price or Decimal("0"),
            strategy_id=managed.request.strategy_id,
        )

    def _on_order_rejection(self, managed: ManagedOrder) -> None:
        """Handle order rejection event.

        THREAD SAFETY: This callback is invoked from broker websocket threads.
        Uses lock to prevent race conditions when modifying shared state.
        """
        with self._state_lock:
            self._metrics.orders_rejected += 1
            if self._session:
                self._session.errors.append(
                    f"Order rejected: {managed.order.symbol} - {managed.error_message}"
                )

    def _is_pre_market(self, current_time: time) -> bool:
        """Check if in pre-market period.

        CRITICAL: Trading hours are in US Eastern Time.
        The current_time parameter is now expected to be Eastern Time.
        """
        pre_market_start = (
            datetime.combine(datetime.today(), self.config.trading_start)
            - timedelta(minutes=self.config.pre_market_minutes)
        ).time()
        return pre_market_start <= current_time < self.config.trading_start

    def _is_market_hours(self, current_time: time) -> bool:
        """Check if during market hours.

        CRITICAL: Trading hours are in US Eastern Time.
        """
        return self.config.trading_start <= current_time < self.config.trading_end

    def _is_post_market(self, current_time: time) -> bool:
        """Check if in post-market period.

        CRITICAL: Trading hours are in US Eastern Time.
        """
        post_market_end = (
            datetime.combine(datetime.today(), self.config.trading_end)
            + timedelta(minutes=self.config.post_market_minutes)
        ).time()
        return self.config.trading_end <= current_time < post_market_end

    def _get_eastern_time(self) -> time:
        """Get current time in US Eastern timezone.

        This is critical for correct trading hours detection since
        US market hours are defined in Eastern Time.

        JPMorgan-level enhancement: Explicit DST handling to avoid
        edge cases during spring forward/fall back transitions.
        """
        eastern = ZoneInfo("America/New_York")
        now_eastern = datetime.now(eastern)
        return now_eastern.time()

    def _get_eastern_datetime(self) -> datetime:
        """Get current datetime in US Eastern timezone.

        JPMorgan-level enhancement: Full datetime with DST awareness.

        Returns:
            Timezone-aware datetime in America/New_York timezone.
        """
        eastern = ZoneInfo("America/New_York")
        return datetime.now(eastern)

    def _is_dst_transition_day(self) -> bool:
        """Check if today is a DST transition day.

        JPMorgan-level enhancement: Detect DST transition days where
        trading hours may be affected by time zone changes.

        US DST rules (since 2007):
        - Spring forward: 2nd Sunday in March at 2 AM → 3 AM
        - Fall back: 1st Sunday in November at 2 AM → 1 AM

        Returns:
            True if today is a DST transition day.
        """
        eastern = ZoneInfo("America/New_York")
        now = datetime.now(eastern)

        # Check if DST status changes today
        # Get the start and end of today
        today_start = datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=eastern)
        today_end = datetime(now.year, now.month, now.day, 23, 59, 59, tzinfo=eastern)

        # DST at start and end of day
        dst_start = today_start.dst()
        dst_end = today_end.dst()

        # If DST status is different, it's a transition day
        if dst_start != dst_end:
            logger.info(
                f"DST transition detected today ({now.date()}): "
                f"{'Spring Forward' if dst_end > dst_start else 'Fall Back'}"
            )
            return True

        return False

    def _get_utc_offset_hours(self) -> float:
        """Get current UTC offset for Eastern Time in hours.

        JPMorgan-level enhancement: Provides offset for UTC-based
        systems that need to convert to Eastern Time.

        Returns:
            UTC offset in hours (negative for behind UTC).
            -5.0 for EST (standard time)
            -4.0 for EDT (daylight time)
        """
        eastern = ZoneInfo("America/New_York")
        now = datetime.now(eastern)
        offset = now.utcoffset()

        if offset is not None:
            return offset.total_seconds() / 3600
        return -5.0  # Default to EST if somehow offset is unavailable

    def _is_in_trading_day(self, dt: datetime | None = None) -> bool:
        """Check if a datetime is within a valid trading day.

        JPMorgan-level enhancement: Accounts for DST transitions
        and market holidays.

        Args:
            dt: Datetime to check (defaults to current Eastern time).

        Returns:
            True if the datetime falls within market hours.
        """
        if dt is None:
            dt = self._get_eastern_datetime()

        current_time = dt.time()

        # Check basic market hours
        is_trading_time = (
            self.config.trading_start <= current_time < self.config.trading_end
        )

        # On DST transition days, be more conservative
        if self._is_dst_transition_day():
            # Log for awareness
            logger.debug(
                f"DST transition day check: time={current_time}, "
                f"is_trading={is_trading_time}"
            )

        return is_trading_time

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

    # =========================================================================
    # MAJOR FIX: Strategy Exception Escalation Tracking
    # =========================================================================

    def _record_successful_iteration(self) -> None:
        """Record a successful iteration for exception tracking.

        Resets the consecutive exception counter if we've had enough
        success time since the last exception.
        """
        now = datetime.now(timezone.utc)

        # If we've had enough successful time, reset the counter
        if self._last_successful_iteration:
            elapsed = (now - self._last_successful_iteration).total_seconds()
            if elapsed >= self.config.strategy_exception_reset_interval:
                if self._consecutive_strategy_exceptions > 0:
                    logger.info(
                        f"Strategy exception counter reset after {elapsed:.0f}s "
                        f"of successful operation"
                    )
                    self._consecutive_strategy_exceptions = 0

        self._last_successful_iteration = now

    def _handle_strategy_exception(self, exception: Exception) -> None:
        """Handle a strategy/execution exception with escalation logic.

        Tracks consecutive exceptions and triggers kill switch if threshold
        is exceeded. This prevents a malfunctioning strategy from causing
        continuous damage.

        Args:
            exception: The exception that occurred.
        """
        self._consecutive_strategy_exceptions += 1

        logger.warning(
            f"Strategy exception {self._consecutive_strategy_exceptions}/"
            f"{self.config.max_consecutive_strategy_exceptions}: {exception}"
        )

        # Record in session
        if self._session:
            self._session.errors.append(
                f"Strategy exception ({self._consecutive_strategy_exceptions}): {str(exception)[:200]}"
            )

        # Check if we've hit the threshold
        if self._consecutive_strategy_exceptions >= self.config.max_consecutive_strategy_exceptions:
            reason = (
                f"STRATEGY EXCEPTION ESCALATION: {self._consecutive_strategy_exceptions} "
                f"consecutive strategy exceptions. Last error: {str(exception)[:100]}"
            )
            self.trigger_kill_switch(reason)

            # Publish critical alert
            self._publish_event(
                EventType.SYSTEM_ERROR,
                reason,
                details={
                    "consecutive_exceptions": self._consecutive_strategy_exceptions,
                    "threshold": self.config.max_consecutive_strategy_exceptions,
                    "last_exception": str(exception),
                    "exception_type": type(exception).__name__,
                },
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
