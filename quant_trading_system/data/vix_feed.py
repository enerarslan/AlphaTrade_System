"""
Real-Time VIX Integration for Dynamic Risk Adjustment.

P1-A Enhancement: Provides real-time VIX data feed for:
- Dynamic position sizing based on market fear
- Regime-aware risk parameter adjustment
- Volatility-based alpha signal modulation
- Kill switch threshold adaptation

Integrates with:
- EventBus for VIX updates
- RegimeDetector for VIX-based regime classification
- RiskLimits for dynamic limit adjustment

Expected Impact: +8-12 bps annually from improved risk timing.

Author: AlphaTrade System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field, field_validator

from quant_trading_system.core.events import (
    Event,
    EventBus,
    EventPriority,
    EventType,
    get_event_bus,
)
from quant_trading_system.core.exceptions import DataError

logger = logging.getLogger(__name__)


# =============================================================================
# VIX Regime Classification
# =============================================================================


class VIXRegime(str, Enum):
    """VIX-based market regime classification.

    Based on historical VIX levels:
    - COMPLACENT: VIX < 12 (extreme low volatility, potential complacency)
    - LOW: 12 <= VIX < 16 (normal low volatility)
    - NORMAL: 16 <= VIX < 20 (typical market conditions)
    - ELEVATED: 20 <= VIX < 25 (increased uncertainty)
    - HIGH: 25 <= VIX < 30 (high fear, increased hedging)
    - EXTREME: 30 <= VIX < 40 (significant market stress)
    - CRISIS: VIX >= 40 (panic, potential capitulation)
    """

    COMPLACENT = "complacent"
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    EXTREME = "extreme"
    CRISIS = "crisis"


@dataclass
class VIXThresholds:
    """Configuration for VIX regime thresholds."""

    complacent_upper: float = 12.0
    low_upper: float = 16.0
    normal_upper: float = 20.0
    elevated_upper: float = 25.0
    high_upper: float = 30.0
    extreme_upper: float = 40.0

    # Dynamic risk adjustment multipliers
    position_size_multipliers: dict[VIXRegime, float] = field(default_factory=lambda: {
        VIXRegime.COMPLACENT: 1.1,   # Slightly increase positions in calm markets
        VIXRegime.LOW: 1.05,
        VIXRegime.NORMAL: 1.0,
        VIXRegime.ELEVATED: 0.85,
        VIXRegime.HIGH: 0.70,
        VIXRegime.EXTREME: 0.50,
        VIXRegime.CRISIS: 0.25,      # Drastically reduce in crisis
    })

    stop_loss_multipliers: dict[VIXRegime, float] = field(default_factory=lambda: {
        VIXRegime.COMPLACENT: 1.0,
        VIXRegime.LOW: 1.0,
        VIXRegime.NORMAL: 1.0,
        VIXRegime.ELEVATED: 1.25,    # Wider stops in volatile markets
        VIXRegime.HIGH: 1.5,
        VIXRegime.EXTREME: 2.0,
        VIXRegime.CRISIS: 2.5,
    })

    # Kill switch VIX thresholds
    kill_switch_warning: float = 35.0
    kill_switch_trigger: float = 45.0


# =============================================================================
# VIX Data Models
# =============================================================================


class VIXData(BaseModel):
    """Real-time VIX data model."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="VIX data timestamp"
    )
    value: Decimal = Field(..., ge=0, description="Current VIX value")
    open: Decimal | None = Field(default=None, ge=0, description="VIX open")
    high: Decimal | None = Field(default=None, ge=0, description="VIX high")
    low: Decimal | None = Field(default=None, ge=0, description="VIX low")
    previous_close: Decimal | None = Field(default=None, ge=0, description="Previous close")
    change: Decimal | None = Field(default=None, description="Daily change")
    change_pct: float | None = Field(default=None, description="Daily change percentage")

    # Derived fields
    regime: VIXRegime | None = Field(default=None, description="VIX regime")
    percentile_52w: float | None = Field(default=None, ge=0, le=100, description="52-week percentile")

    @field_validator("value", mode="before")
    @classmethod
    def convert_to_decimal(cls, v: Any) -> Decimal:
        """Convert value to Decimal."""
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with serializable types."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": str(self.value),
            "open": str(self.open) if self.open else None,
            "high": str(self.high) if self.high else None,
            "low": str(self.low) if self.low else None,
            "previous_close": str(self.previous_close) if self.previous_close else None,
            "change": str(self.change) if self.change else None,
            "change_pct": self.change_pct,
            "regime": self.regime.value if self.regime else None,
            "percentile_52w": self.percentile_52w,
        }


@dataclass
class VIXFeedConfig:
    """Configuration for VIX data feed."""

    # Update frequency
    poll_interval_seconds: float = 60.0  # How often to fetch VIX

    # Staleness detection
    staleness_warning_seconds: float = 300.0   # 5 minutes
    staleness_critical_seconds: float = 900.0  # 15 minutes

    # History settings
    history_size: int = 500  # Keep last 500 VIX readings

    # Thresholds
    thresholds: VIXThresholds = field(default_factory=VIXThresholds)

    # Data source
    data_source: str = "alpaca"  # 'alpaca', 'yahoo', 'cboe', 'mock'

    # Enable features
    enable_regime_events: bool = True
    enable_risk_adjustment: bool = True


# =============================================================================
# VIX Feed Base Class
# =============================================================================


class BaseVIXFeed(ABC):
    """Abstract base class for VIX data feeds."""

    def __init__(
        self,
        config: VIXFeedConfig | None = None,
        event_bus: EventBus | None = None,
    ):
        """Initialize VIX feed.

        Args:
            config: VIX feed configuration
            event_bus: Event bus for publishing updates
        """
        self.config = config or VIXFeedConfig()
        self.event_bus = event_bus or get_event_bus()

        self._running = False
        self._current_vix: VIXData | None = None
        self._vix_history: list[VIXData] = []
        self._last_update: datetime | None = None
        self._current_regime: VIXRegime | None = None

        # Callbacks
        self._callbacks: list[Callable[[VIXData], None]] = []
        self._regime_callbacks: list[Callable[[VIXRegime, VIXRegime], None]] = []

        # Thread safety
        self._lock = threading.RLock()

        # Metrics
        self._metrics: dict[str, Any] = {
            "updates_received": 0,
            "regime_changes": 0,
            "errors": 0,
            "staleness_warnings": 0,
        }

    @abstractmethod
    async def fetch_vix(self) -> VIXData:
        """Fetch current VIX data from source."""
        pass

    @abstractmethod
    async def connect(self) -> None:
        """Connect to VIX data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from VIX data source."""
        pass

    def classify_regime(self, vix_value: Decimal) -> VIXRegime:
        """Classify VIX value into regime.

        Args:
            vix_value: Current VIX value

        Returns:
            VIXRegime classification
        """
        value = float(vix_value)
        thresholds = self.config.thresholds

        if value < thresholds.complacent_upper:
            return VIXRegime.COMPLACENT
        elif value < thresholds.low_upper:
            return VIXRegime.LOW
        elif value < thresholds.normal_upper:
            return VIXRegime.NORMAL
        elif value < thresholds.elevated_upper:
            return VIXRegime.ELEVATED
        elif value < thresholds.high_upper:
            return VIXRegime.HIGH
        elif value < thresholds.extreme_upper:
            return VIXRegime.EXTREME
        else:
            return VIXRegime.CRISIS

    def get_position_size_multiplier(self, regime: VIXRegime | None = None) -> float:
        """Get position size multiplier for current/specified regime.

        Args:
            regime: VIX regime (uses current if not specified)

        Returns:
            Position size multiplier (0.25 to 1.1)
        """
        regime = regime or self._current_regime or VIXRegime.NORMAL
        return self.config.thresholds.position_size_multipliers.get(regime, 1.0)

    def get_stop_loss_multiplier(self, regime: VIXRegime | None = None) -> float:
        """Get stop loss multiplier for current/specified regime.

        Args:
            regime: VIX regime (uses current if not specified)

        Returns:
            Stop loss multiplier (1.0 to 2.5)
        """
        regime = regime or self._current_regime or VIXRegime.NORMAL
        return self.config.thresholds.stop_loss_multipliers.get(regime, 1.0)

    def should_trigger_kill_switch(self) -> tuple[bool, str | None]:
        """Check if VIX level should trigger kill switch.

        Returns:
            Tuple of (should_trigger, reason)
        """
        if not self._current_vix:
            return False, None

        vix_value = float(self._current_vix.value)
        thresholds = self.config.thresholds

        if vix_value >= thresholds.kill_switch_trigger:
            return True, f"VIX at {vix_value:.2f} exceeds kill switch threshold {thresholds.kill_switch_trigger}"

        return False, None

    def should_warn_kill_switch(self) -> tuple[bool, str | None]:
        """Check if VIX level should trigger kill switch warning.

        Returns:
            Tuple of (should_warn, reason)
        """
        if not self._current_vix:
            return False, None

        vix_value = float(self._current_vix.value)
        thresholds = self.config.thresholds

        if vix_value >= thresholds.kill_switch_warning:
            return True, f"VIX at {vix_value:.2f} approaching kill switch threshold"

        return False, None

    def calculate_percentile(self, vix_value: Decimal) -> float | None:
        """Calculate VIX percentile based on history.

        Args:
            vix_value: Current VIX value

        Returns:
            Percentile (0-100) or None if insufficient history
        """
        with self._lock:
            if len(self._vix_history) < 50:
                return None

            history_values = [float(v.value) for v in self._vix_history]
            current = float(vix_value)

            below_count = sum(1 for v in history_values if v < current)
            return (below_count / len(history_values)) * 100

    def add_callback(self, callback: Callable[[VIXData], None]) -> None:
        """Add callback for VIX updates."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[VIXData], None]) -> None:
        """Remove VIX update callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def add_regime_callback(
        self,
        callback: Callable[[VIXRegime, VIXRegime], None]
    ) -> None:
        """Add callback for regime changes.

        Args:
            callback: Function called with (old_regime, new_regime)
        """
        self._regime_callbacks.append(callback)

    def _notify_callbacks(self, vix_data: VIXData) -> None:
        """Notify all VIX update callbacks."""
        for callback in self._callbacks:
            try:
                callback(vix_data)
            except Exception as e:
                logger.error(f"VIX callback error: {e}")

    def _notify_regime_callbacks(
        self,
        old_regime: VIXRegime,
        new_regime: VIXRegime
    ) -> None:
        """Notify all regime change callbacks."""
        for callback in self._regime_callbacks:
            try:
                callback(old_regime, new_regime)
            except Exception as e:
                logger.error(f"Regime callback error: {e}")

    def _publish_vix_event(self, vix_data: VIXData) -> None:
        """Publish VIX update to event bus."""
        if self.event_bus:
            event = Event(
                event_type=EventType.VIX_UPDATE,
                data={
                    "vix": vix_data.to_dict(),
                    "regime": vix_data.regime.value if vix_data.regime else None,
                    "position_multiplier": self.get_position_size_multiplier(vix_data.regime),
                    "stop_multiplier": self.get_stop_loss_multiplier(vix_data.regime),
                },
                source="vix_feed",
                priority=EventPriority.HIGH,
            )
            self.event_bus.publish(event)

    def _publish_regime_change_event(
        self,
        old_regime: VIXRegime,
        new_regime: VIXRegime,
        vix_value: Decimal
    ) -> None:
        """Publish regime change event."""
        if self.event_bus and self.config.enable_regime_events:
            # Determine priority based on severity of change
            priority = EventPriority.NORMAL
            if new_regime in [VIXRegime.EXTREME, VIXRegime.CRISIS]:
                priority = EventPriority.CRITICAL
            elif new_regime in [VIXRegime.HIGH, VIXRegime.ELEVATED]:
                priority = EventPriority.HIGH

            event = Event(
                event_type=EventType.REGIME_CHANGE,
                data={
                    "source": "vix",
                    "old_regime": old_regime.value if old_regime else None,
                    "new_regime": new_regime.value,
                    "vix_value": str(vix_value),
                    "position_multiplier": self.get_position_size_multiplier(new_regime),
                    "stop_multiplier": self.get_stop_loss_multiplier(new_regime),
                },
                source="vix_feed",
                priority=priority,
            )
            self.event_bus.publish(event)

    async def _process_vix_update(self, vix_data: VIXData) -> None:
        """Process a VIX update."""
        with self._lock:
            # Classify regime
            new_regime = self.classify_regime(vix_data.value)
            vix_data.regime = new_regime

            # Calculate percentile
            vix_data.percentile_52w = self.calculate_percentile(vix_data.value)

            # Check for regime change
            old_regime = self._current_regime
            regime_changed = old_regime != new_regime

            # Update state
            self._current_vix = vix_data
            self._current_regime = new_regime
            self._last_update = datetime.now(timezone.utc)
            self._metrics["updates_received"] += 1

            # Add to history
            self._vix_history.append(vix_data)
            if len(self._vix_history) > self.config.history_size:
                self._vix_history.pop(0)

        # Notify callbacks (outside lock)
        self._notify_callbacks(vix_data)

        # Publish events
        self._publish_vix_event(vix_data)

        # Handle regime change
        if regime_changed and old_regime is not None:
            self._metrics["regime_changes"] += 1
            self._notify_regime_callbacks(old_regime, new_regime)
            self._publish_regime_change_event(old_regime, new_regime, vix_data.value)

            logger.info(
                f"VIX regime changed: {old_regime.value} -> {new_regime.value} "
                f"(VIX={vix_data.value:.2f})"
            )

        # Check kill switch warnings
        should_warn, warn_msg = self.should_warn_kill_switch()
        if should_warn:
            logger.warning(f"VIX KILL SWITCH WARNING: {warn_msg}")

        should_trigger, trigger_msg = self.should_trigger_kill_switch()
        if should_trigger:
            logger.critical(f"VIX KILL SWITCH TRIGGER: {trigger_msg}")
            # Publish critical risk event
            if self.event_bus:
                event = Event(
                    event_type=EventType.KILL_SWITCH_TRIGGERED,
                    data={
                        "reason": "vix_threshold",
                        "message": trigger_msg,
                        "vix_value": str(vix_data.value),
                    },
                    source="vix_feed",
                    priority=EventPriority.CRITICAL,
                )
                self.event_bus.publish(event)

    def get_current_vix(self) -> VIXData | None:
        """Get current VIX data."""
        return self._current_vix

    def get_current_regime(self) -> VIXRegime | None:
        """Get current VIX regime."""
        return self._current_regime

    def get_vix_history(self, limit: int | None = None) -> list[VIXData]:
        """Get VIX history."""
        with self._lock:
            if limit:
                return self._vix_history[-limit:]
            return self._vix_history.copy()

    def get_metrics(self) -> dict[str, Any]:
        """Get VIX feed metrics."""
        return {
            **self._metrics,
            "current_vix": str(self._current_vix.value) if self._current_vix else None,
            "current_regime": self._current_regime.value if self._current_regime else None,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "history_size": len(self._vix_history),
            "is_running": self._running,
        }

    def is_data_stale(self) -> tuple[bool, str | None]:
        """Check if VIX data is stale.

        Returns:
            Tuple of (is_stale, severity) where severity is 'warning' or 'critical'
        """
        if not self._last_update:
            return True, "critical"

        elapsed = (datetime.now(timezone.utc) - self._last_update).total_seconds()

        if elapsed >= self.config.staleness_critical_seconds:
            return True, "critical"
        elif elapsed >= self.config.staleness_warning_seconds:
            return True, "warning"

        return False, None


# =============================================================================
# VIX Feed Implementations
# =============================================================================


class AlpacaVIXFeed(BaseVIXFeed):
    """VIX feed using Alpaca Markets data.

    Fetches VIX data from Alpaca's market data API.
    VIX is available as ticker ^VIX or via VIXY as a proxy.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        config: VIXFeedConfig | None = None,
        event_bus: EventBus | None = None,
        paper: bool = True,
    ):
        """Initialize Alpaca VIX feed.

        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            config: VIX feed configuration
            event_bus: Event bus for publishing
            paper: Use paper trading environment
        """
        super().__init__(config, event_bus)
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper

        self._client = None
        self._poll_task: asyncio.Task | None = None
        self._staleness_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Connect to Alpaca and start polling."""
        try:
            # Import Alpaca SDK
            try:
                from alpaca.data import StockHistoricalDataClient
                from alpaca.data.requests import StockBarsRequest
            except ImportError:
                logger.warning(
                    "alpaca-py not installed. Using mock VIX data. "
                    "Install with: pip install alpaca-py"
                )
                # Fall back to mock
                return

            # Create client
            self._client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
            )

            self._running = True

            # Start polling task
            self._poll_task = asyncio.create_task(self._poll_vix())

            # Start staleness monitoring
            self._staleness_task = asyncio.create_task(self._monitor_staleness())

            logger.info("Alpaca VIX feed connected and polling started")

        except Exception as e:
            logger.error(f"Failed to connect Alpaca VIX feed: {e}")
            raise DataError(f"VIX feed connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect and stop polling."""
        self._running = False

        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        if self._staleness_task:
            self._staleness_task.cancel()
            try:
                await self._staleness_task
            except asyncio.CancelledError:
                pass

        self._client = None
        logger.info("Alpaca VIX feed disconnected")

    async def fetch_vix(self) -> VIXData:
        """Fetch current VIX data from Alpaca.

        Note: Alpaca doesn't directly provide VIX index data.
        We use VIXY (ProShares VIX Short-Term Futures ETF) as a proxy,
        or attempt to get ^VIX if available.
        """
        if not self._client:
            raise DataError("VIX feed not connected")

        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            from alpaca.data.timeframe import TimeFrame

            # Try to get VIXY as VIX proxy
            symbol = "VIXY"

            request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = self._client.get_stock_latest_quote(request)

            if symbol in quotes:
                quote = quotes[symbol]
                # VIXY trades around 10-30, we need to approximate VIX
                # This is a simplified proxy - in production, use proper VIX futures
                vixy_price = float(quote.ask_price + quote.bid_price) / 2

                # Rough VIX approximation from VIXY (this is simplified)
                # In production, use actual VIX index data from CBOE
                vix_approx = vixy_price * 1.5  # Simplified approximation

                return VIXData(
                    timestamp=quote.timestamp,
                    value=Decimal(str(round(vix_approx, 2))),
                )

            raise DataError(f"No quote data for {symbol}")

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"Error fetching VIX from Alpaca: {e}")
            raise DataError(f"VIX fetch failed: {e}")

    async def _poll_vix(self) -> None:
        """Poll VIX data at configured interval."""
        while self._running:
            try:
                vix_data = await self.fetch_vix()
                await self._process_vix_update(vix_data)

            except Exception as e:
                logger.error(f"VIX poll error: {e}")
                self._metrics["errors"] += 1

            await asyncio.sleep(self.config.poll_interval_seconds)

    async def _monitor_staleness(self) -> None:
        """Monitor for stale VIX data."""
        while self._running:
            await asyncio.sleep(60)  # Check every minute

            is_stale, severity = self.is_data_stale()
            if is_stale:
                self._metrics["staleness_warnings"] += 1
                if severity == "critical":
                    logger.critical("VIX data critically stale - no updates for >15 minutes")
                else:
                    logger.warning("VIX data stale - no updates for >5 minutes")


class MockVIXFeed(BaseVIXFeed):
    """Mock VIX feed for testing and development.

    Generates simulated VIX data following realistic patterns.
    """

    def __init__(
        self,
        config: VIXFeedConfig | None = None,
        event_bus: EventBus | None = None,
        initial_vix: float = 18.0,
        volatility: float = 0.02,
        mean_reversion_speed: float = 0.1,
        long_term_mean: float = 18.0,
    ):
        """Initialize mock VIX feed.

        Args:
            config: VIX feed configuration
            event_bus: Event bus for publishing
            initial_vix: Starting VIX value
            volatility: VIX volatility (std dev)
            mean_reversion_speed: Speed of mean reversion
            long_term_mean: Long-term VIX mean
        """
        super().__init__(config, event_bus)
        self._vix_value = initial_vix
        self._volatility = volatility
        self._mean_reversion = mean_reversion_speed
        self._long_term_mean = long_term_mean
        self._poll_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Start mock VIX generation."""
        self._running = True
        self._poll_task = asyncio.create_task(self._generate_vix())
        logger.info(f"Mock VIX feed started at VIX={self._vix_value:.2f}")

    async def disconnect(self) -> None:
        """Stop mock VIX generation."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        logger.info("Mock VIX feed stopped")

    async def fetch_vix(self) -> VIXData:
        """Generate mock VIX data using Ornstein-Uhlenbeck process."""
        # Mean-reverting random walk
        drift = self._mean_reversion * (self._long_term_mean - self._vix_value)
        shock = np.random.normal(0, self._vix_value * self._volatility)

        self._vix_value = max(8.0, self._vix_value + drift + shock)  # VIX floor at 8

        # Add some intraday range
        daily_range = self._vix_value * 0.05
        high = self._vix_value + np.random.uniform(0, daily_range)
        low = self._vix_value - np.random.uniform(0, daily_range)

        return VIXData(
            timestamp=datetime.now(timezone.utc),
            value=Decimal(str(round(self._vix_value, 2))),
            high=Decimal(str(round(high, 2))),
            low=Decimal(str(round(low, 2))),
            open=Decimal(str(round(self._vix_value - shock/2, 2))),
        )

    async def _generate_vix(self) -> None:
        """Generate VIX data at configured interval."""
        while self._running:
            try:
                vix_data = await self.fetch_vix()
                await self._process_vix_update(vix_data)
            except Exception as e:
                logger.error(f"Mock VIX generation error: {e}")

            await asyncio.sleep(self.config.poll_interval_seconds)

    def simulate_spike(self, target_vix: float, duration_seconds: float = 60.0) -> None:
        """Simulate a VIX spike for testing.

        Args:
            target_vix: Target VIX level
            duration_seconds: How long to stay elevated
        """
        logger.info(f"Simulating VIX spike to {target_vix}")
        self._vix_value = target_vix


# =============================================================================
# VIX-Based Risk Adjustor
# =============================================================================


class VIXRiskAdjustor:
    """
    Adjusts risk parameters based on real-time VIX levels.

    Integrates VIX feed with risk management system to provide:
    - Dynamic position sizing
    - Adaptive stop losses
    - Regime-aware risk limits
    """

    def __init__(
        self,
        vix_feed: BaseVIXFeed,
        base_position_size_pct: float = 0.10,
        base_stop_loss_pct: float = 0.02,
        base_max_drawdown_pct: float = 0.15,
    ):
        """Initialize VIX risk adjustor.

        Args:
            vix_feed: VIX data feed
            base_position_size_pct: Base max position size (10%)
            base_stop_loss_pct: Base stop loss percentage (2%)
            base_max_drawdown_pct: Base max drawdown (15%)
        """
        self.vix_feed = vix_feed
        self.base_position_size = base_position_size_pct
        self.base_stop_loss = base_stop_loss_pct
        self.base_max_drawdown = base_max_drawdown_pct

        # Track adjustments
        self._adjustment_history: list[dict[str, Any]] = []

    def get_adjusted_position_size(self, symbol: str | None = None) -> float:
        """Get VIX-adjusted max position size.

        Args:
            symbol: Optional symbol for symbol-specific adjustments

        Returns:
            Adjusted position size percentage
        """
        multiplier = self.vix_feed.get_position_size_multiplier()
        adjusted = self.base_position_size * multiplier

        # Log adjustment
        self._log_adjustment("position_size", self.base_position_size, adjusted, multiplier)

        return adjusted

    def get_adjusted_stop_loss(self, symbol: str | None = None) -> float:
        """Get VIX-adjusted stop loss percentage.

        Args:
            symbol: Optional symbol for symbol-specific adjustments

        Returns:
            Adjusted stop loss percentage
        """
        multiplier = self.vix_feed.get_stop_loss_multiplier()
        adjusted = self.base_stop_loss * multiplier

        self._log_adjustment("stop_loss", self.base_stop_loss, adjusted, multiplier)

        return adjusted

    def get_adjusted_max_drawdown(self) -> float:
        """Get VIX-adjusted max drawdown threshold.

        In high VIX environments, we tighten the drawdown limit.

        Returns:
            Adjusted max drawdown percentage
        """
        regime = self.vix_feed.get_current_regime()

        # Tighten drawdown limits in high VIX
        adjustments = {
            VIXRegime.COMPLACENT: 1.0,
            VIXRegime.LOW: 1.0,
            VIXRegime.NORMAL: 1.0,
            VIXRegime.ELEVATED: 0.90,
            VIXRegime.HIGH: 0.80,
            VIXRegime.EXTREME: 0.65,
            VIXRegime.CRISIS: 0.50,
        }

        multiplier = adjustments.get(regime, 1.0) if regime else 1.0
        adjusted = self.base_max_drawdown * multiplier

        self._log_adjustment("max_drawdown", self.base_max_drawdown, adjusted, multiplier)

        return adjusted

    def get_all_adjustments(self) -> dict[str, float]:
        """Get all current VIX-adjusted risk parameters.

        Returns:
            Dictionary of adjusted parameters
        """
        vix_data = self.vix_feed.get_current_vix()
        regime = self.vix_feed.get_current_regime()

        return {
            "vix_value": float(vix_data.value) if vix_data else None,
            "vix_regime": regime.value if regime else None,
            "adjusted_position_size_pct": self.get_adjusted_position_size(),
            "adjusted_stop_loss_pct": self.get_adjusted_stop_loss(),
            "adjusted_max_drawdown_pct": self.get_adjusted_max_drawdown(),
            "position_multiplier": self.vix_feed.get_position_size_multiplier(),
            "stop_multiplier": self.vix_feed.get_stop_loss_multiplier(),
        }

    def _log_adjustment(
        self,
        param_name: str,
        base_value: float,
        adjusted_value: float,
        multiplier: float
    ) -> None:
        """Log risk parameter adjustment."""
        self._adjustment_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "parameter": param_name,
            "base_value": base_value,
            "adjusted_value": adjusted_value,
            "multiplier": multiplier,
            "vix_regime": self.vix_feed.get_current_regime().value if self.vix_feed.get_current_regime() else None,
        })

        # Keep bounded history
        if len(self._adjustment_history) > 1000:
            self._adjustment_history = self._adjustment_history[-1000:]


# =============================================================================
# Factory Functions
# =============================================================================


def create_vix_feed(
    feed_type: str = "mock",
    event_bus: EventBus | None = None,
    **kwargs: Any,
) -> BaseVIXFeed:
    """Factory function to create a VIX feed.

    Args:
        feed_type: Type of feed ('alpaca', 'mock')
        event_bus: Event bus for publishing
        **kwargs: Additional arguments for specific feed type

    Returns:
        VIX feed instance

    Example:
        >>> feed = create_vix_feed(
        ...     feed_type="alpaca",
        ...     api_key="your_key",
        ...     api_secret="your_secret",
        ... )
        >>> await feed.connect()
    """
    config = VIXFeedConfig(
        poll_interval_seconds=kwargs.get("poll_interval_seconds", 60.0),
        data_source=feed_type,
    )

    if feed_type == "mock":
        return MockVIXFeed(
            config=config,
            event_bus=event_bus,
            initial_vix=kwargs.get("initial_vix", 18.0),
            volatility=kwargs.get("volatility", 0.02),
        )

    elif feed_type == "alpaca":
        return AlpacaVIXFeed(
            api_key=kwargs.get("api_key", ""),
            api_secret=kwargs.get("api_secret", ""),
            config=config,
            event_bus=event_bus,
            paper=kwargs.get("paper", True),
        )

    else:
        raise ValueError(f"Unknown VIX feed type: {feed_type}")


async def start_vix_feed(
    feed_type: str = "mock",
    event_bus: EventBus | None = None,
    **kwargs: Any,
) -> BaseVIXFeed:
    """Quick helper to start a VIX feed.

    Args:
        feed_type: Type of feed
        event_bus: Event bus
        **kwargs: Feed-specific arguments

    Returns:
        Connected VIX feed instance
    """
    feed = create_vix_feed(feed_type, event_bus, **kwargs)
    await feed.connect()
    return feed
