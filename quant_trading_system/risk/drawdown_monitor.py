"""
Intraday Drawdown Monitor with Real-Time Alerts.

P2-E Enhancement: Real-time drawdown monitoring with multi-channel alerting:
- Continuous intraday drawdown tracking
- Multi-threshold alert system (warning, critical, kill switch)
- Slack/PagerDuty/Email push notifications
- Rolling and peak-to-trough drawdown calculations
- Integration with kill switch for automatic trading halt

Expected Impact: Reduced tail risk through faster response to drawdown events.

Author: AlphaTrade System
Version: 1.0.0
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

import numpy as np
from pydantic import BaseModel, Field

from quant_trading_system.core.events import (
    Event,
    EventBus,
    EventPriority,
    EventType,
    get_event_bus,
)
from quant_trading_system.monitoring.alerting import (
    AlertManager,
    AlertType,
    AlertSeverity,
    get_alert_manager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Drawdown Types
# =============================================================================


class DrawdownSeverity(str, Enum):
    """Drawdown severity levels."""

    NORMAL = "normal"
    ELEVATED = "elevated"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class DrawdownType(str, Enum):
    """Types of drawdown calculations."""

    PEAK_TO_TROUGH = "peak_to_trough"
    ROLLING_WINDOW = "rolling_window"
    INTRADAY = "intraday"
    SESSION_OPEN = "session_open"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class DrawdownState:
    """Current drawdown state."""

    current_equity: Decimal
    peak_equity: Decimal
    session_start_equity: Decimal

    # Drawdown values (as positive percentages)
    peak_to_trough_pct: float = 0.0
    intraday_pct: float = 0.0
    rolling_1h_pct: float = 0.0
    rolling_4h_pct: float = 0.0

    # Timing
    peak_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trough_timestamp: datetime | None = None
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # State
    severity: DrawdownSeverity = DrawdownSeverity.NORMAL
    is_recovering: bool = False
    recovery_start: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_equity": str(self.current_equity),
            "peak_equity": str(self.peak_equity),
            "peak_to_trough_pct": self.peak_to_trough_pct,
            "intraday_pct": self.intraday_pct,
            "rolling_1h_pct": self.rolling_1h_pct,
            "rolling_4h_pct": self.rolling_4h_pct,
            "severity": self.severity.value,
            "is_recovering": self.is_recovering,
            "last_update": self.last_update.isoformat(),
        }


@dataclass
class DrawdownAlert:
    """Drawdown alert record."""

    alert_id: str
    severity: DrawdownSeverity
    drawdown_type: DrawdownType
    drawdown_pct: float
    threshold_pct: float
    equity: Decimal
    peak_equity: Decimal
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "drawdown_type": self.drawdown_type.value,
            "drawdown_pct": self.drawdown_pct,
            "threshold_pct": self.threshold_pct,
            "equity": str(self.equity),
            "peak_equity": str(self.peak_equity),
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


class DrawdownMonitorConfig(BaseModel):
    """Configuration for drawdown monitoring."""

    # Alert thresholds (as percentages)
    warning_threshold_pct: float = Field(default=3.0, description="Warning alert at 3%")
    critical_threshold_pct: float = Field(default=5.0, description="Critical alert at 5%")
    emergency_threshold_pct: float = Field(default=8.0, description="Emergency at 8%")
    kill_switch_threshold_pct: float = Field(default=10.0, description="Kill switch at 10%")

    # Intraday limits
    max_intraday_loss_pct: float = Field(default=5.0, description="Max intraday loss %")

    # Rolling window thresholds
    rolling_1h_warning_pct: float = Field(default=2.0, description="1-hour rolling warning")
    rolling_4h_warning_pct: float = Field(default=4.0, description="4-hour rolling warning")

    # Monitoring settings
    update_interval_seconds: int = Field(default=5, description="Update frequency")
    equity_history_size: int = Field(default=1000, description="Equity history buffer size")

    # Alert cooldowns (prevent alert fatigue)
    alert_cooldown_minutes: int = Field(default=5, description="Cooldown between same alerts")

    # Kill switch integration
    enable_kill_switch: bool = Field(default=True, description="Auto-activate kill switch")


# =============================================================================
# Intraday Drawdown Monitor
# =============================================================================


class IntradayDrawdownMonitor:
    """
    P2-E Enhancement: Real-time intraday drawdown monitoring.

    Features:
    - Continuous equity tracking
    - Multiple drawdown calculations (peak-to-trough, intraday, rolling)
    - Multi-threshold alerting (warning, critical, emergency)
    - Slack/PagerDuty/Email push notifications via AlertManager
    - Kill switch integration for automatic trading halt
    - Recovery tracking and notifications
    """

    def __init__(
        self,
        config: DrawdownMonitorConfig | None = None,
        alert_manager: AlertManager | None = None,
        event_bus: EventBus | None = None,
        kill_switch_callback: Callable[[], None] | None = None,
    ):
        """Initialize drawdown monitor.

        Args:
            config: Monitor configuration
            alert_manager: Alert manager for notifications
            event_bus: Event bus for publishing events
            kill_switch_callback: Callback to trigger kill switch
        """
        self.config = config or DrawdownMonitorConfig()
        self.alert_manager = alert_manager or get_alert_manager()
        self.event_bus = event_bus or get_event_bus()
        self.kill_switch_callback = kill_switch_callback

        # State
        self._lock = threading.RLock()
        self._equity_history: list[tuple[datetime, Decimal]] = []
        self._peak_equity: Decimal = Decimal("0")
        self._session_start_equity: Decimal = Decimal("0")
        self._current_equity: Decimal = Decimal("0")
        self._state: DrawdownState | None = None

        # Alert tracking
        self._alert_history: list[DrawdownAlert] = []
        self._last_alert_times: dict[str, datetime] = {}
        self._alert_counter = 0
        self._kill_switch_triggered = False

        # Monitoring task
        self._running = False
        self._monitor_task: asyncio.Task | None = None

    def start_session(self, initial_equity: Decimal) -> None:
        """Start a new trading session.

        Args:
            initial_equity: Starting equity for the session
        """
        with self._lock:
            self._session_start_equity = initial_equity
            self._peak_equity = initial_equity
            self._current_equity = initial_equity
            self._kill_switch_triggered = False
            self._equity_history.clear()
            self._last_alert_times.clear()

            self._equity_history.append((datetime.now(timezone.utc), initial_equity))

            self._state = DrawdownState(
                current_equity=initial_equity,
                peak_equity=initial_equity,
                session_start_equity=initial_equity,
            )

            logger.info(f"Drawdown monitor session started with equity ${initial_equity:,.2f}")

    def update_equity(self, equity: Decimal) -> DrawdownState:
        """Update current equity and calculate drawdown.

        Args:
            equity: Current portfolio equity

        Returns:
            Updated DrawdownState
        """
        now = datetime.now(timezone.utc)

        with self._lock:
            self._current_equity = equity

            # Update peak
            if equity > self._peak_equity:
                self._peak_equity = equity
                if self._state:
                    self._state.peak_timestamp = now
                    self._state.is_recovering = False

            # Store in history
            self._equity_history.append((now, equity))

            # Trim history
            if len(self._equity_history) > self.config.equity_history_size:
                self._equity_history = self._equity_history[-self.config.equity_history_size:]

            # Calculate drawdowns
            state = self._calculate_drawdowns(equity, now)

        # FIX: Create async task OUTSIDE the lock to prevent potential deadlock
        asyncio.create_task(self._check_thresholds(state))

        return state

    def _calculate_drawdowns(self, equity: Decimal, now: datetime) -> DrawdownState:
        """Calculate all drawdown metrics.

        Args:
            equity: Current equity
            now: Current timestamp

        Returns:
            Updated DrawdownState
        """
        # Peak-to-trough drawdown
        if self._peak_equity > 0:
            peak_to_trough = float((self._peak_equity - equity) / self._peak_equity * 100)
        else:
            peak_to_trough = 0.0

        # Intraday drawdown (from session start)
        if self._session_start_equity > 0:
            intraday = float((self._session_start_equity - equity) / self._session_start_equity * 100)
        else:
            intraday = 0.0

        # Rolling drawdowns
        rolling_1h = self._calculate_rolling_drawdown(hours=1)
        rolling_4h = self._calculate_rolling_drawdown(hours=4)

        # Determine severity
        severity = self._determine_severity(max(peak_to_trough, intraday))

        # Check for recovery
        is_recovering = False
        recovery_start = None
        if self._state and self._state.peak_to_trough_pct > peak_to_trough:
            if not self._state.is_recovering:
                is_recovering = True
                recovery_start = now
            else:
                is_recovering = self._state.is_recovering
                recovery_start = self._state.recovery_start

        self._state = DrawdownState(
            current_equity=equity,
            peak_equity=self._peak_equity,
            session_start_equity=self._session_start_equity,
            peak_to_trough_pct=max(0, peak_to_trough),
            intraday_pct=max(0, intraday),
            rolling_1h_pct=max(0, rolling_1h),
            rolling_4h_pct=max(0, rolling_4h),
            severity=severity,
            is_recovering=is_recovering,
            recovery_start=recovery_start,
            last_update=now,
        )

        return self._state

    def _calculate_rolling_drawdown(self, hours: int) -> float:
        """Calculate rolling window drawdown.

        Args:
            hours: Lookback hours

        Returns:
            Rolling drawdown percentage
        """
        if not self._equity_history:
            return 0.0

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        relevant = [(t, e) for t, e in self._equity_history if t >= cutoff]

        if not relevant:
            return 0.0

        equities = [float(e) for _, e in relevant]
        peak = max(equities)
        current = equities[-1]

        if peak > 0:
            return ((peak - current) / peak) * 100
        return 0.0

    def _determine_severity(self, drawdown_pct: float) -> DrawdownSeverity:
        """Determine severity based on drawdown.

        Args:
            drawdown_pct: Current drawdown percentage

        Returns:
            DrawdownSeverity level
        """
        if drawdown_pct >= self.config.emergency_threshold_pct:
            return DrawdownSeverity.EMERGENCY
        elif drawdown_pct >= self.config.critical_threshold_pct:
            return DrawdownSeverity.CRITICAL
        elif drawdown_pct >= self.config.warning_threshold_pct:
            return DrawdownSeverity.WARNING
        elif drawdown_pct >= self.config.warning_threshold_pct * 0.5:
            return DrawdownSeverity.ELEVATED
        return DrawdownSeverity.NORMAL

    async def _check_thresholds(self, state: DrawdownState) -> None:
        """Check drawdown thresholds and trigger alerts.

        Args:
            state: Current drawdown state
        """
        max_dd = max(state.peak_to_trough_pct, state.intraday_pct)

        # Kill switch check
        if max_dd >= self.config.kill_switch_threshold_pct:
            if not self._kill_switch_triggered and self.config.enable_kill_switch:
                await self._trigger_kill_switch(state)

        # Emergency alert
        elif max_dd >= self.config.emergency_threshold_pct:
            await self._send_alert(
                severity=DrawdownSeverity.EMERGENCY,
                drawdown_type=DrawdownType.PEAK_TO_TROUGH,
                drawdown_pct=max_dd,
                threshold_pct=self.config.emergency_threshold_pct,
                state=state,
            )

        # Critical alert
        elif max_dd >= self.config.critical_threshold_pct:
            await self._send_alert(
                severity=DrawdownSeverity.CRITICAL,
                drawdown_type=DrawdownType.PEAK_TO_TROUGH,
                drawdown_pct=max_dd,
                threshold_pct=self.config.critical_threshold_pct,
                state=state,
            )

        # Warning alert
        elif max_dd >= self.config.warning_threshold_pct:
            await self._send_alert(
                severity=DrawdownSeverity.WARNING,
                drawdown_type=DrawdownType.PEAK_TO_TROUGH,
                drawdown_pct=max_dd,
                threshold_pct=self.config.warning_threshold_pct,
                state=state,
            )

        # Rolling window alerts
        if state.rolling_1h_pct >= self.config.rolling_1h_warning_pct:
            await self._send_alert(
                severity=DrawdownSeverity.WARNING,
                drawdown_type=DrawdownType.ROLLING_WINDOW,
                drawdown_pct=state.rolling_1h_pct,
                threshold_pct=self.config.rolling_1h_warning_pct,
                state=state,
                extra_info="1-hour rolling",
            )

    async def _send_alert(
        self,
        severity: DrawdownSeverity,
        drawdown_type: DrawdownType,
        drawdown_pct: float,
        threshold_pct: float,
        state: DrawdownState,
        extra_info: str = "",
    ) -> None:
        """Send drawdown alert via AlertManager.

        Args:
            severity: Alert severity
            drawdown_type: Type of drawdown
            drawdown_pct: Current drawdown percentage
            threshold_pct: Threshold that was breached
            state: Current state
            extra_info: Additional info for message
        """
        # Check cooldown
        alert_key = f"{severity.value}_{drawdown_type.value}"
        now = datetime.now(timezone.utc)

        with self._lock:
            last_alert = self._last_alert_times.get(alert_key)
            if last_alert:
                cooldown = timedelta(minutes=self.config.alert_cooldown_minutes)
                if now - last_alert < cooldown:
                    return

            self._last_alert_times[alert_key] = now
            self._alert_counter += 1
            alert_id = f"DD-{self._alert_counter:06d}"

        # Build message
        type_str = extra_info if extra_info else drawdown_type.value.replace("_", " ").title()
        message = (
            f"DRAWDOWN ALERT: {type_str} drawdown of {drawdown_pct:.2f}% "
            f"has exceeded {threshold_pct:.1f}% threshold.\n"
            f"Current Equity: ${float(state.current_equity):,.2f}\n"
            f"Peak Equity: ${float(state.peak_equity):,.2f}\n"
            f"Loss: ${float(state.peak_equity - state.current_equity):,.2f}"
        )

        # Map to AlertManager severity
        alert_severity_map = {
            DrawdownSeverity.EMERGENCY: AlertSeverity.CRITICAL,
            DrawdownSeverity.CRITICAL: AlertSeverity.CRITICAL,
            DrawdownSeverity.WARNING: AlertSeverity.HIGH,
            DrawdownSeverity.ELEVATED: AlertSeverity.MEDIUM,
        }

        alert_type_map = {
            DrawdownSeverity.EMERGENCY: AlertType.MAX_DRAWDOWN_BREACHED,
            DrawdownSeverity.CRITICAL: AlertType.MAX_DRAWDOWN_BREACHED,
            DrawdownSeverity.WARNING: AlertType.RISK_LIMIT_APPROACHING,
            DrawdownSeverity.ELEVATED: AlertType.RISK_LIMIT_APPROACHING,
        }

        # Send via AlertManager (triggers Slack/PagerDuty/Email)
        try:
            await self.alert_manager.create_alert(
                alert_type=alert_type_map.get(severity, AlertType.RISK_LIMIT_APPROACHING),
                title=f"Drawdown Alert: {severity.value.upper()} ({drawdown_pct:.2f}%)",
                message=message,
                context={
                    "drawdown_pct": drawdown_pct,
                    "threshold_pct": threshold_pct,
                    "current_equity": str(state.current_equity),
                    "peak_equity": str(state.peak_equity),
                    "intraday_pct": state.intraday_pct,
                    "rolling_1h_pct": state.rolling_1h_pct,
                },
                severity=alert_severity_map.get(severity, AlertSeverity.HIGH),
                suggested_action=(
                    "1. Review current positions. "
                    "2. Consider reducing position sizes. "
                    "3. Check for correlated losses. "
                    "4. Prepare to close high-risk positions."
                ),
            )

            logger.warning(f"Drawdown alert sent: {severity.value} - {drawdown_pct:.2f}%")

        except Exception as e:
            logger.error(f"Failed to send drawdown alert: {e}")

        # Store alert
        alert = DrawdownAlert(
            alert_id=alert_id,
            severity=severity,
            drawdown_type=drawdown_type,
            drawdown_pct=drawdown_pct,
            threshold_pct=threshold_pct,
            equity=state.current_equity,
            peak_equity=state.peak_equity,
            message=message,
        )

        with self._lock:
            self._alert_history.append(alert)
            if len(self._alert_history) > 1000:
                self._alert_history = self._alert_history[-1000:]

        # Publish event
        self.event_bus.publish(Event(
            event_type=EventType.DRAWDOWN_WARNING,
            data=alert.to_dict(),
            priority=EventPriority.HIGH if severity in (DrawdownSeverity.CRITICAL, DrawdownSeverity.EMERGENCY) else EventPriority.NORMAL,
        ))

    async def _trigger_kill_switch(self, state: DrawdownState) -> None:
        """Trigger the kill switch for extreme drawdown.

        Args:
            state: Current drawdown state
        """
        with self._lock:
            if self._kill_switch_triggered:
                return
            self._kill_switch_triggered = True

        max_dd = max(state.peak_to_trough_pct, state.intraday_pct)

        message = (
            f"KILL SWITCH TRIGGERED: Drawdown of {max_dd:.2f}% "
            f"has exceeded {self.config.kill_switch_threshold_pct:.1f}% limit.\n"
            f"ALL TRADING HALTED IMMEDIATELY.\n"
            f"Current Equity: ${float(state.current_equity):,.2f}\n"
            f"Peak Equity: ${float(state.peak_equity):,.2f}\n"
            f"Total Loss: ${float(state.peak_equity - state.current_equity):,.2f}"
        )

        # Send critical alert via AlertManager
        try:
            await self.alert_manager.create_alert(
                alert_type=AlertType.KILL_SWITCH_TRIGGERED,
                title=f"KILL SWITCH: Max Drawdown {max_dd:.2f}%",
                message=message,
                context={
                    "drawdown_pct": max_dd,
                    "threshold_pct": self.config.kill_switch_threshold_pct,
                    "current_equity": str(state.current_equity),
                    "peak_equity": str(state.peak_equity),
                    "loss_amount": str(state.peak_equity - state.current_equity),
                },
                severity=AlertSeverity.CRITICAL,
                suggested_action=(
                    "1. DO NOT RESET kill switch without thorough investigation. "
                    "2. Review all positions and market conditions. "
                    "3. Document the incident. "
                    "4. Get approval from risk management before resuming."
                ),
            )
        except Exception as e:
            logger.error(f"Failed to send kill switch alert: {e}")

        # Trigger callback
        if self.kill_switch_callback:
            try:
                self.kill_switch_callback()
            except Exception as e:
                logger.error(f"Kill switch callback failed: {e}")

        logger.critical(f"KILL SWITCH TRIGGERED: Drawdown {max_dd:.2f}%")

        # Publish event
        self.event_bus.publish(Event(
            event_type=EventType.KILL_SWITCH_TRIGGERED,
            data={
                "reason": "max_drawdown",
                "drawdown_pct": max_dd,
                "threshold_pct": self.config.kill_switch_threshold_pct,
                "equity": str(state.current_equity),
                "peak_equity": str(state.peak_equity),
            },
            priority=EventPriority.CRITICAL,
        ))

    async def start_monitoring(self, equity_provider: Callable[[], Decimal]) -> None:
        """Start continuous monitoring.

        Args:
            equity_provider: Callback to get current equity
        """
        if self._running:
            logger.warning("Drawdown monitor already running")
            return

        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitoring_loop(equity_provider)
        )
        logger.info("Drawdown monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("Drawdown monitoring stopped")

    async def _monitoring_loop(self, equity_provider: Callable[[], Decimal]) -> None:
        """Continuous monitoring loop.

        Args:
            equity_provider: Callback to get current equity
        """
        while self._running:
            try:
                equity = equity_provider()
                self.update_equity(equity)
            except Exception as e:
                logger.error(f"Error in drawdown monitoring: {e}")

            await asyncio.sleep(self.config.update_interval_seconds)

    def get_current_state(self) -> DrawdownState | None:
        """Get current drawdown state.

        Returns:
            Current DrawdownState or None
        """
        with self._lock:
            return self._state

    def get_alert_history(
        self,
        severity: DrawdownSeverity | None = None,
        limit: int = 50,
    ) -> list[DrawdownAlert]:
        """Get alert history.

        Args:
            severity: Filter by severity
            limit: Max alerts to return

        Returns:
            List of DrawdownAlert
        """
        with self._lock:
            alerts = self._alert_history

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts[-limit:]

    def reset_kill_switch(self, authorized_by: str) -> bool:
        """Reset the kill switch after investigation.

        Args:
            authorized_by: User authorizing the reset

        Returns:
            True if reset successful
        """
        with self._lock:
            if not self._kill_switch_triggered:
                return False

            self._kill_switch_triggered = False
            self._peak_equity = self._current_equity  # Reset peak

            logger.warning(f"Kill switch reset by: {authorized_by}")

            return True

    def get_stats(self) -> dict[str, Any]:
        """Get monitor statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            state = self._state

        return {
            "is_running": self._running,
            "kill_switch_triggered": self._kill_switch_triggered,
            "current_state": state.to_dict() if state else None,
            "total_alerts": len(self._alert_history),
            "alert_counts": {
                sev.value: len([a for a in self._alert_history if a.severity == sev])
                for sev in DrawdownSeverity
            },
            "config": {
                "warning_threshold": self.config.warning_threshold_pct,
                "critical_threshold": self.config.critical_threshold_pct,
                "kill_switch_threshold": self.config.kill_switch_threshold_pct,
            },
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_drawdown_monitor(
    config: DrawdownMonitorConfig | None = None,
    alert_manager: AlertManager | None = None,
    kill_switch_callback: Callable[[], None] | None = None,
) -> IntradayDrawdownMonitor:
    """Factory function to create drawdown monitor.

    Args:
        config: Monitor configuration
        alert_manager: Alert manager
        kill_switch_callback: Kill switch callback

    Returns:
        Configured IntradayDrawdownMonitor
    """
    return IntradayDrawdownMonitor(
        config=config,
        alert_manager=alert_manager,
        kill_switch_callback=kill_switch_callback,
    )
