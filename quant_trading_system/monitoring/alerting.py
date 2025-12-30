"""
Alert management system for the trading system.

Provides:
- Alert definitions by severity (Critical, Warning, Info)
- Multi-channel notification routing (Email, SMS, Slack, PagerDuty)
- Alert deduplication and rate limiting
- Escalation handling and acknowledgment tracking
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .logger import get_logger, LogCategory


logger = get_logger("alerting", LogCategory.ALERT)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class AlertChannel(str, Enum):
    """Alert notification channels."""

    EMAIL = "EMAIL"
    SMS = "SMS"
    SLACK = "SLACK"
    PAGERDUTY = "PAGERDUTY"
    DASHBOARD = "DASHBOARD"
    WEBHOOK = "WEBHOOK"


class AlertStatus(str, Enum):
    """Alert status."""

    FIRING = "FIRING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    SUPPRESSED = "SUPPRESSED"


class AlertType(str, Enum):
    """Predefined alert types."""

    # Critical alerts
    SYSTEM_DOWN = "SYSTEM_DOWN"
    BROKER_CONNECTION_LOST = "BROKER_CONNECTION_LOST"
    DATA_FEED_FAILURE = "DATA_FEED_FAILURE"
    KILL_SWITCH_TRIGGERED = "KILL_SWITCH_TRIGGERED"
    MAX_DRAWDOWN_BREACHED = "MAX_DRAWDOWN_BREACHED"
    DATABASE_FAILURE = "DATABASE_FAILURE"

    # Warning alerts
    HIGH_LATENCY = "HIGH_LATENCY"
    MODEL_ACCURACY_DEGRADED = "MODEL_ACCURACY_DEGRADED"
    RISK_LIMIT_APPROACHING = "RISK_LIMIT_APPROACHING"
    UNUSUAL_VOLATILITY = "UNUSUAL_VOLATILITY"
    POSITION_CONCENTRATION_HIGH = "POSITION_CONCENTRATION_HIGH"
    SLIPPAGE_ABOVE_THRESHOLD = "SLIPPAGE_ABOVE_THRESHOLD"

    # Info alerts
    DAILY_SUMMARY = "DAILY_SUMMARY"
    WEEKLY_PERFORMANCE_REPORT = "WEEKLY_PERFORMANCE_REPORT"
    MODEL_RETRAINING_COMPLETE = "MODEL_RETRAINING_COMPLETE"
    NEW_POSITIONS_OPENED = "NEW_POSITIONS_OPENED"
    SYSTEM_MAINTENANCE_SCHEDULED = "SYSTEM_MAINTENANCE_SCHEDULED"


# Default severity mapping for alert types
DEFAULT_SEVERITY_MAP: dict[AlertType, AlertSeverity] = {
    AlertType.SYSTEM_DOWN: AlertSeverity.CRITICAL,
    AlertType.BROKER_CONNECTION_LOST: AlertSeverity.CRITICAL,
    AlertType.DATA_FEED_FAILURE: AlertSeverity.CRITICAL,
    AlertType.KILL_SWITCH_TRIGGERED: AlertSeverity.CRITICAL,
    AlertType.MAX_DRAWDOWN_BREACHED: AlertSeverity.CRITICAL,
    AlertType.DATABASE_FAILURE: AlertSeverity.CRITICAL,
    AlertType.HIGH_LATENCY: AlertSeverity.WARNING,
    AlertType.MODEL_ACCURACY_DEGRADED: AlertSeverity.WARNING,
    AlertType.RISK_LIMIT_APPROACHING: AlertSeverity.WARNING,
    AlertType.UNUSUAL_VOLATILITY: AlertSeverity.WARNING,
    AlertType.POSITION_CONCENTRATION_HIGH: AlertSeverity.WARNING,
    AlertType.SLIPPAGE_ABOVE_THRESHOLD: AlertSeverity.WARNING,
    AlertType.DAILY_SUMMARY: AlertSeverity.INFO,
    AlertType.WEEKLY_PERFORMANCE_REPORT: AlertSeverity.INFO,
    AlertType.MODEL_RETRAINING_COMPLETE: AlertSeverity.INFO,
    AlertType.NEW_POSITIONS_OPENED: AlertSeverity.INFO,
    AlertType.SYSTEM_MAINTENANCE_SCHEDULED: AlertSeverity.INFO,
}


class Alert(BaseModel):
    """Alert model."""

    alert_id: UUID = Field(default_factory=uuid4)
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)
    suggested_action: str | None = None
    runbook_link: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: AlertStatus = AlertStatus.FIRING
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None
    fingerprint: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Generate fingerprint after initialization."""
        if not self.fingerprint:
            self.fingerprint = self._generate_fingerprint()

    def _generate_fingerprint(self) -> str:
        """Generate a unique fingerprint for deduplication."""
        content = f"{self.alert_type.value}:{self.title}:{json.dumps(self.context, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def acknowledge(self, acknowledged_by: str) -> None:
        """Acknowledge the alert.

        Args:
            acknowledged_by: User who acknowledged the alert.
        """
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_by = acknowledged_by
        self.acknowledged_at = datetime.now(timezone.utc)

    def resolve(self) -> None:
        """Resolve the alert."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": str(self.alert_id),
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "context": self.context,
            "suggested_action": self.suggested_action,
            "runbook_link": self.runbook_link,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "fingerprint": self.fingerprint,
        }


@dataclass
class AlertRule:
    """Alert routing rule."""

    severity: AlertSeverity
    channels: list[AlertChannel]
    quiet_hours_start: int | None = None  # Hour (0-23)
    quiet_hours_end: int | None = None  # Hour (0-23)
    escalation_minutes: int | None = None
    escalation_channels: list[AlertChannel] | None = None


class AlertNotifier(ABC):
    """Abstract base class for alert notifiers."""

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send an alert notification.

        Args:
            alert: Alert to send.

        Returns:
            True if sent successfully.
        """
        pass

    @abstractmethod
    def get_channel(self) -> AlertChannel:
        """Get the notification channel type.

        Returns:
            AlertChannel type.
        """
        pass


class EmailNotifier(AlertNotifier):
    """Email alert notifier."""

    def __init__(
        self,
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        username: str = "",
        password: str = "",
        from_address: str = "",
        to_addresses: list[str] | None = None,
    ) -> None:
        """Initialize email notifier.

        Args:
            smtp_host: SMTP server host.
            smtp_port: SMTP server port.
            username: SMTP username.
            password: SMTP password.
            from_address: Sender email address.
            to_addresses: List of recipient email addresses.
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_address = from_address
        self.to_addresses = to_addresses or []

    async def send(self, alert: Alert) -> bool:
        """Send email notification."""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart()
            msg["From"] = self.from_address
            msg["To"] = ", ".join(self.to_addresses)
            msg["Subject"] = f"[{alert.severity.value}] {alert.title}"

            body = f"""
Alert: {alert.title}
Severity: {alert.severity.value}
Type: {alert.alert_type.value}
Time: {alert.timestamp.isoformat()}

Message:
{alert.message}

Context:
{json.dumps(alert.context, indent=2)}

Suggested Action:
{alert.suggested_action or 'N/A'}

Runbook: {alert.runbook_link or 'N/A'}
"""
            msg.attach(MIMEText(body, "plain"))

            def send_sync() -> bool:
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    if self.username and self.password:
                        server.starttls()
                        server.login(self.username, self.password)
                    server.send_message(msg)
                return True

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, send_sync)

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def get_channel(self) -> AlertChannel:
        return AlertChannel.EMAIL


class SlackNotifier(AlertNotifier):
    """Slack alert notifier."""

    def __init__(self, webhook_url: str, channel: str = "#alerts") -> None:
        """Initialize Slack notifier.

        Args:
            webhook_url: Slack webhook URL.
            channel: Slack channel.
        """
        self.webhook_url = webhook_url
        self.channel = channel

    async def send(self, alert: Alert) -> bool:
        """Send Slack notification."""
        try:
            import aiohttp

            severity_emoji = {
                AlertSeverity.CRITICAL: ":rotating_light:",
                AlertSeverity.WARNING: ":warning:",
                AlertSeverity.INFO: ":information_source:",
            }

            color = {
                AlertSeverity.CRITICAL: "#FF0000",
                AlertSeverity.WARNING: "#FFA500",
                AlertSeverity.INFO: "#0000FF",
            }

            payload = {
                "channel": self.channel,
                "attachments": [
                    {
                        "color": color.get(alert.severity, "#808080"),
                        "title": f"{severity_emoji.get(alert.severity, '')} {alert.title}",
                        "text": alert.message,
                        "fields": [
                            {"title": "Severity", "value": alert.severity.value, "short": True},
                            {"title": "Type", "value": alert.alert_type.value, "short": True},
                            {"title": "Time", "value": alert.timestamp.isoformat(), "short": True},
                        ],
                        "footer": "Trading System Alert",
                        "ts": int(alert.timestamp.timestamp()),
                    }
                ],
            }

            if alert.suggested_action:
                payload["attachments"][0]["fields"].append(
                    {"title": "Suggested Action", "value": alert.suggested_action, "short": False}
                )

            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    def get_channel(self) -> AlertChannel:
        return AlertChannel.SLACK


class WebhookNotifier(AlertNotifier):
    """Generic webhook notifier."""

    def __init__(
        self,
        webhook_url: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize webhook notifier.

        Args:
            webhook_url: Webhook URL.
            headers: Optional HTTP headers.
        """
        self.webhook_url = webhook_url
        self.headers = headers or {}

    async def send(self, alert: Alert) -> bool:
        """Send webhook notification."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=alert.to_dict(),
                    headers=self.headers,
                ) as response:
                    return response.status in (200, 201, 202)

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False

    def get_channel(self) -> AlertChannel:
        return AlertChannel.WEBHOOK


class DashboardNotifier(AlertNotifier):
    """Dashboard notifier - stores alerts for dashboard display."""

    def __init__(self, max_alerts: int = 1000) -> None:
        """Initialize dashboard notifier.

        Args:
            max_alerts: Maximum number of alerts to store.
        """
        self.max_alerts = max_alerts
        self.alerts: list[Alert] = []

    async def send(self, alert: Alert) -> bool:
        """Store alert for dashboard display."""
        self.alerts.append(alert)
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts :]
        return True

    def get_channel(self) -> AlertChannel:
        return AlertChannel.DASHBOARD

    def get_alerts(
        self,
        severity: AlertSeverity | None = None,
        status: AlertStatus | None = None,
        limit: int = 100,
    ) -> list[Alert]:
        """Get stored alerts.

        Args:
            severity: Optional severity filter.
            status: Optional status filter.
            limit: Maximum number of alerts to return.

        Returns:
            List of alerts.
        """
        alerts = self.alerts
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if status:
            alerts = [a for a in alerts if a.status == status]
        return alerts[-limit:]


@dataclass
class RateLimitEntry:
    """Rate limit tracking entry."""

    count: int = 0
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AlertManager:
    """Central alert manager for the trading system.

    Handles:
    - Alert creation and routing
    - Deduplication and rate limiting
    - Multi-channel notification
    - Escalation handling
    - Acknowledgment tracking
    """

    def __init__(
        self,
        rate_limit_window_seconds: int = 300,
        rate_limit_max_alerts: int = 5,
        dedup_window_seconds: int = 600,
    ) -> None:
        """Initialize the alert manager.

        Args:
            rate_limit_window_seconds: Rate limit window in seconds.
            rate_limit_max_alerts: Maximum alerts per fingerprint in window.
            dedup_window_seconds: Deduplication window in seconds.
        """
        self.rate_limit_window = timedelta(seconds=rate_limit_window_seconds)
        self.rate_limit_max = rate_limit_max_alerts
        self.dedup_window = timedelta(seconds=dedup_window_seconds)

        self._notifiers: dict[AlertChannel, AlertNotifier] = {}
        self._routing_rules: dict[AlertSeverity, AlertRule] = {}
        self._rate_limits: dict[str, RateLimitEntry] = defaultdict(RateLimitEntry)
        self._active_alerts: dict[str, Alert] = {}
        self._alert_history: list[Alert] = []
        self._max_history = 10000
        self._suppressed_fingerprints: set[str] = set()

        # Default routing rules
        self._setup_default_routing()

        # Dashboard notifier always enabled
        self._dashboard_notifier = DashboardNotifier()
        self._notifiers[AlertChannel.DASHBOARD] = self._dashboard_notifier

    def _setup_default_routing(self) -> None:
        """Set up default routing rules."""
        self._routing_rules = {
            AlertSeverity.CRITICAL: AlertRule(
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.DASHBOARD],
                escalation_minutes=5,
                escalation_channels=[AlertChannel.SMS],
            ),
            AlertSeverity.WARNING: AlertRule(
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.SLACK, AlertChannel.DASHBOARD],
                quiet_hours_start=22,
                quiet_hours_end=7,
            ),
            AlertSeverity.INFO: AlertRule(
                severity=AlertSeverity.INFO,
                channels=[AlertChannel.DASHBOARD],
                quiet_hours_start=20,
                quiet_hours_end=8,
            ),
        }

    def register_notifier(self, notifier: AlertNotifier) -> None:
        """Register an alert notifier.

        Args:
            notifier: Notifier instance to register.
        """
        self._notifiers[notifier.get_channel()] = notifier

    def set_routing_rule(self, rule: AlertRule) -> None:
        """Set a routing rule.

        Args:
            rule: Routing rule to set.
        """
        self._routing_rules[rule.severity] = rule

    def suppress_alert(self, fingerprint: str) -> None:
        """Suppress alerts with the given fingerprint.

        Args:
            fingerprint: Alert fingerprint to suppress.
        """
        self._suppressed_fingerprints.add(fingerprint)

    def unsuppress_alert(self, fingerprint: str) -> None:
        """Unsuppress alerts with the given fingerprint.

        Args:
            fingerprint: Alert fingerprint to unsuppress.
        """
        self._suppressed_fingerprints.discard(fingerprint)

    def _is_rate_limited(self, fingerprint: str) -> bool:
        """Check if an alert is rate limited.

        Args:
            fingerprint: Alert fingerprint.

        Returns:
            True if rate limited.
        """
        now = datetime.now(timezone.utc)
        entry = self._rate_limits[fingerprint]

        # Reset if window expired
        if now - entry.first_seen > self.rate_limit_window:
            entry.count = 0
            entry.first_seen = now

        entry.count += 1
        entry.last_seen = now

        return entry.count > self.rate_limit_max

    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if an alert is a duplicate.

        Args:
            alert: Alert to check.

        Returns:
            True if duplicate.
        """
        if alert.fingerprint not in self._active_alerts:
            return False

        existing = self._active_alerts[alert.fingerprint]
        return (
            existing.status in (AlertStatus.FIRING, AlertStatus.ACKNOWLEDGED)
            and alert.timestamp - existing.timestamp < self.dedup_window
        )

    def _is_in_quiet_hours(self, rule: AlertRule) -> bool:
        """Check if current time is in quiet hours.

        Args:
            rule: Alert routing rule.

        Returns:
            True if in quiet hours.
        """
        if rule.quiet_hours_start is None or rule.quiet_hours_end is None:
            return False

        current_hour = datetime.now(timezone.utc).hour
        start = rule.quiet_hours_start
        end = rule.quiet_hours_end

        if start < end:
            return start <= current_hour < end
        else:  # Overnight quiet hours (e.g., 22-7)
            return current_hour >= start or current_hour < end

    async def create_alert(
        self,
        alert_type: AlertType,
        title: str,
        message: str,
        context: dict[str, Any] | None = None,
        suggested_action: str | None = None,
        runbook_link: str | None = None,
        severity: AlertSeverity | None = None,
    ) -> Alert | None:
        """Create and dispatch an alert.

        Args:
            alert_type: Type of alert.
            title: Alert title.
            message: Alert message.
            context: Additional context.
            suggested_action: Suggested action to take.
            runbook_link: Link to runbook.
            severity: Optional severity override.

        Returns:
            Alert if created and sent, None if suppressed/deduplicated.
        """
        # Determine severity
        if severity is None:
            severity = DEFAULT_SEVERITY_MAP.get(alert_type, AlertSeverity.WARNING)

        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            context=context or {},
            suggested_action=suggested_action,
            runbook_link=runbook_link,
        )

        # Check suppression
        if alert.fingerprint in self._suppressed_fingerprints:
            alert.status = AlertStatus.SUPPRESSED
            logger.debug(f"Alert suppressed: {alert.title}")
            return None

        # Check for duplicates
        if self._is_duplicate(alert):
            logger.debug(f"Alert deduplicated: {alert.title}")
            return None

        # Check rate limiting
        if self._is_rate_limited(alert.fingerprint):
            logger.warning(f"Alert rate limited: {alert.title}")
            return None

        # Store alert
        self._active_alerts[alert.fingerprint] = alert
        self._add_to_history(alert)

        # Get routing rule
        rule = self._routing_rules.get(alert.severity)
        if not rule:
            logger.warning(f"No routing rule for severity {alert.severity}")
            rule = AlertRule(severity=alert.severity, channels=[AlertChannel.DASHBOARD])

        # Send to channels
        await self._send_to_channels(alert, rule)

        logger.info(
            f"Alert created: {alert.title}",
            extra={"extra_data": {"alert_id": str(alert.alert_id), "severity": alert.severity.value}},
        )

        return alert

    async def _send_to_channels(self, alert: Alert, rule: AlertRule) -> None:
        """Send alert to configured channels.

        Args:
            alert: Alert to send.
            rule: Routing rule.
        """
        in_quiet_hours = self._is_in_quiet_hours(rule)

        for channel in rule.channels:
            # Skip non-dashboard channels during quiet hours for non-critical alerts
            if in_quiet_hours and channel != AlertChannel.DASHBOARD and alert.severity != AlertSeverity.CRITICAL:
                continue

            notifier = self._notifiers.get(channel)
            if notifier:
                try:
                    success = await notifier.send(alert)
                    if not success:
                        logger.warning(f"Failed to send alert to {channel.value}")
                except Exception as e:
                    logger.error(f"Error sending alert to {channel.value}: {e}")

    def _add_to_history(self, alert: Alert) -> None:
        """Add alert to history."""
        self._alert_history.append(alert)
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history :]

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge.
            acknowledged_by: User acknowledging the alert.

        Returns:
            True if acknowledged.
        """
        for fingerprint, alert in self._active_alerts.items():
            if str(alert.alert_id) == alert_id:
                alert.acknowledge(acknowledged_by)
                logger.info(f"Alert acknowledged: {alert.title} by {acknowledged_by}")
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.

        Args:
            alert_id: Alert ID to resolve.

        Returns:
            True if resolved.
        """
        for fingerprint, alert in list(self._active_alerts.items()):
            if str(alert.alert_id) == alert_id:
                alert.resolve()
                del self._active_alerts[fingerprint]
                logger.info(f"Alert resolved: {alert.title}")
                return True
        return False

    def resolve_by_type(self, alert_type: AlertType) -> int:
        """Resolve all alerts of a given type.

        Args:
            alert_type: Alert type to resolve.

        Returns:
            Number of alerts resolved.
        """
        resolved = 0
        for fingerprint, alert in list(self._active_alerts.items()):
            if alert.alert_type == alert_type:
                alert.resolve()
                del self._active_alerts[fingerprint]
                resolved += 1
        logger.info(f"Resolved {resolved} alerts of type {alert_type.value}")
        return resolved

    def get_active_alerts(
        self,
        severity: AlertSeverity | None = None,
        alert_type: AlertType | None = None,
    ) -> list[Alert]:
        """Get active alerts.

        Args:
            severity: Optional severity filter.
            alert_type: Optional type filter.

        Returns:
            List of active alerts.
        """
        alerts = list(self._active_alerts.values())

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_alert_history(
        self,
        limit: int = 100,
        severity: AlertSeverity | None = None,
        alert_type: AlertType | None = None,
    ) -> list[Alert]:
        """Get alert history.

        Args:
            limit: Maximum alerts to return.
            severity: Optional severity filter.
            alert_type: Optional type filter.

        Returns:
            List of historical alerts.
        """
        alerts = self._alert_history

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        return alerts[-limit:]

    def get_dashboard_alerts(self, limit: int = 100) -> list[Alert]:
        """Get alerts for dashboard display.

        Args:
            limit: Maximum alerts to return.

        Returns:
            List of alerts for dashboard.
        """
        return self._dashboard_notifier.get_alerts(limit=limit)

    def get_stats(self) -> dict[str, Any]:
        """Get alert statistics.

        Returns:
            Dictionary with alert stats.
        """
        active = list(self._active_alerts.values())
        return {
            "active_count": len(active),
            "active_by_severity": {
                severity.value: len([a for a in active if a.severity == severity])
                for severity in AlertSeverity
            },
            "history_count": len(self._alert_history),
            "suppressed_count": len(self._suppressed_fingerprints),
            "rate_limited_fingerprints": sum(
                1
                for entry in self._rate_limits.values()
                if entry.count > self.rate_limit_max
            ),
        }


# Singleton alert manager
_alert_manager: AlertManager | None = None


def get_alert_manager() -> AlertManager:
    """Get the singleton alert manager instance.

    Returns:
        AlertManager instance.
    """
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


# Convenience functions for creating common alerts
async def alert_critical(
    alert_type: AlertType,
    title: str,
    message: str,
    **context: Any,
) -> Alert | None:
    """Create a critical alert."""
    manager = get_alert_manager()
    return await manager.create_alert(
        alert_type=alert_type,
        title=title,
        message=message,
        context=context,
        severity=AlertSeverity.CRITICAL,
    )


async def alert_warning(
    alert_type: AlertType,
    title: str,
    message: str,
    **context: Any,
) -> Alert | None:
    """Create a warning alert."""
    manager = get_alert_manager()
    return await manager.create_alert(
        alert_type=alert_type,
        title=title,
        message=message,
        context=context,
        severity=AlertSeverity.WARNING,
    )


async def alert_info(
    alert_type: AlertType,
    title: str,
    message: str,
    **context: Any,
) -> Alert | None:
    """Create an info alert."""
    manager = get_alert_manager()
    return await manager.create_alert(
        alert_type=alert_type,
        title=title,
        message=message,
        context=context,
        severity=AlertSeverity.INFO,
    )
