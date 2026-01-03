"""
Alert management system for the trading system.

Provides:
- Alert definitions by severity (Critical, High, Medium, Low)
- Multi-channel notification routing (Email, SMS, Slack, PagerDuty)
- Alert deduplication and rate limiting
- Escalation handling and acknowledgment tracking
- Professional alert templates for institutional trading

PRODUCTION-READY ALERTING SYSTEM
================================
This module implements a comprehensive alerting system designed for
institutional-grade trading operations. It supports multiple notification
channels with intelligent routing based on alert severity.

Alert Severity Routing:
- CRITICAL -> PagerDuty + Slack + Email (immediate)
- HIGH     -> Slack + Email
- MEDIUM   -> Slack only
- LOW      -> Email only (digest/batched)

Environment Variables Required:
- SLACK_WEBHOOK_URL: Slack incoming webhook URL
- SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD: Email configuration
- SMTP_FROM_ADDRESS, SMTP_TO_ADDRESSES: Email addresses
- PAGERDUTY_SERVICE_KEY: PagerDuty Events API v2 integration key
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN: For SMS alerts (optional)
- TWILIO_FROM_NUMBER, TWILIO_TO_NUMBERS: SMS phone numbers (optional)
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from string import Template
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .logger import get_logger, LogCategory


logger = get_logger("alerting", LogCategory.ALERT)


# =============================================================================
# Alert Configuration from Environment
# =============================================================================

def _get_env_list(key: str, default: list[str] | None = None) -> list[str]:
    """Get a comma-separated list from environment variable."""
    value = os.environ.get(key, "")
    if not value:
        return default or []
    return [item.strip() for item in value.split(",") if item.strip()]


class AlertConfig:
    """Alert configuration loaded from environment variables.

    This class provides centralized configuration for all notification channels.
    Values are loaded from environment variables with sensible defaults.
    """

    # Slack Configuration
    SLACK_WEBHOOK_URL: str = os.environ.get("SLACK_WEBHOOK_URL", "")
    SLACK_CHANNEL: str = os.environ.get("SLACK_CHANNEL", "#trading-alerts")
    SLACK_CRITICAL_CHANNEL: str = os.environ.get("SLACK_CRITICAL_CHANNEL", "#trading-critical")
    SLACK_USERNAME: str = os.environ.get("SLACK_USERNAME", "AlphaTrade Alert Bot")

    # Email (SMTP) Configuration
    SMTP_HOST: str = os.environ.get("SMTP_HOST", "")
    SMTP_PORT: int = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_USER: str = os.environ.get("SMTP_USER", "")
    SMTP_PASSWORD: str = os.environ.get("SMTP_PASSWORD", "")
    SMTP_FROM_ADDRESS: str = os.environ.get("SMTP_FROM_ADDRESS", "")
    SMTP_TO_ADDRESSES: list[str] = _get_env_list("SMTP_TO_ADDRESSES")
    SMTP_USE_TLS: bool = os.environ.get("SMTP_USE_TLS", "true").lower() == "true"

    # PagerDuty Configuration
    PAGERDUTY_SERVICE_KEY: str = os.environ.get("PAGERDUTY_SERVICE_KEY", "")
    PAGERDUTY_API_URL: str = os.environ.get(
        "PAGERDUTY_API_URL",
        "https://events.pagerduty.com/v2/enqueue"
    )

    # Twilio (SMS) Configuration
    TWILIO_ACCOUNT_SID: str = os.environ.get("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.environ.get("TWILIO_AUTH_TOKEN", "")
    TWILIO_FROM_NUMBER: str = os.environ.get("TWILIO_FROM_NUMBER", "")
    TWILIO_TO_NUMBERS: list[str] = _get_env_list("TWILIO_TO_NUMBERS")

    # General Alert Settings
    ALERT_ENVIRONMENT: str = os.environ.get("APP_ENV", "development")
    ALERT_SYSTEM_NAME: str = os.environ.get("ALERT_SYSTEM_NAME", "AlphaTrade")
    RUNBOOK_BASE_URL: str = os.environ.get("RUNBOOK_BASE_URL", "")

    @classmethod
    def is_slack_configured(cls) -> bool:
        """Check if Slack notifications are properly configured."""
        return bool(cls.SLACK_WEBHOOK_URL)

    @classmethod
    def is_email_configured(cls) -> bool:
        """Check if email notifications are properly configured."""
        return bool(cls.SMTP_HOST and cls.SMTP_FROM_ADDRESS and cls.SMTP_TO_ADDRESSES)

    @classmethod
    def is_pagerduty_configured(cls) -> bool:
        """Check if PagerDuty notifications are properly configured."""
        return bool(cls.PAGERDUTY_SERVICE_KEY)

    @classmethod
    def is_sms_configured(cls) -> bool:
        """Check if SMS notifications are properly configured."""
        return bool(
            cls.TWILIO_ACCOUNT_SID and
            cls.TWILIO_AUTH_TOKEN and
            cls.TWILIO_FROM_NUMBER and
            cls.TWILIO_TO_NUMBERS
        )

    @classmethod
    def get_configuration_status(cls) -> dict[str, bool]:
        """Get status of all notification channel configurations."""
        return {
            "slack": cls.is_slack_configured(),
            "email": cls.is_email_configured(),
            "pagerduty": cls.is_pagerduty_configured(),
            "sms": cls.is_sms_configured(),
        }


class AlertSeverity(str, Enum):
    """Alert severity levels.

    Severity levels determine notification routing:
    - CRITICAL: Immediate escalation to all channels (PagerDuty + Slack + Email)
    - HIGH: Urgent notification to Slack + Email
    - MEDIUM: Standard notification to Slack only
    - LOW: Low priority, Email only (can be batched/digested)
    """

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    # Legacy aliases for backward compatibility
    WARNING = "HIGH"  # Maps to HIGH
    INFO = "LOW"  # Maps to LOW


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
    """Predefined alert types for the trading system.

    Alert types are organized by category and mapped to default severity levels.
    Each alert type has an associated runbook and suggested action.
    """

    # =========================================================================
    # CRITICAL Alerts - Require immediate attention (PagerDuty + Slack + Email)
    # =========================================================================
    SYSTEM_DOWN = "SYSTEM_DOWN"
    BROKER_CONNECTION_LOST = "BROKER_CONNECTION_LOST"
    DATA_FEED_FAILURE = "DATA_FEED_FAILURE"
    KILL_SWITCH_TRIGGERED = "KILL_SWITCH_TRIGGERED"
    MAX_DRAWDOWN_BREACHED = "MAX_DRAWDOWN_BREACHED"
    DATABASE_FAILURE = "DATABASE_FAILURE"
    DAILY_LOSS_LIMIT_BREACHED = "DAILY_LOSS_LIMIT_BREACHED"
    AUTHENTICATION_FAILURE = "AUTHENTICATION_FAILURE"
    ORDER_REJECTION_CRITICAL = "ORDER_REJECTION_CRITICAL"
    MARGIN_CALL = "MARGIN_CALL"

    # =========================================================================
    # HIGH Alerts - Urgent issues (Slack + Email)
    # =========================================================================
    HIGH_LATENCY = "HIGH_LATENCY"
    MODEL_ACCURACY_DEGRADED = "MODEL_ACCURACY_DEGRADED"
    RISK_LIMIT_APPROACHING = "RISK_LIMIT_APPROACHING"
    UNUSUAL_VOLATILITY = "UNUSUAL_VOLATILITY"
    POSITION_CONCENTRATION_HIGH = "POSITION_CONCENTRATION_HIGH"
    SLIPPAGE_ABOVE_THRESHOLD = "SLIPPAGE_ABOVE_THRESHOLD"
    CONNECTION_DEGRADED = "CONNECTION_DEGRADED"
    MEMORY_PRESSURE = "MEMORY_PRESSURE"
    CPU_PRESSURE = "CPU_PRESSURE"
    ORDER_FILL_DELAYED = "ORDER_FILL_DELAYED"

    # =========================================================================
    # MEDIUM Alerts - Attention needed (Slack only)
    # =========================================================================
    MODEL_PREDICTION_ANOMALY = "MODEL_PREDICTION_ANOMALY"
    FEATURE_DATA_STALE = "FEATURE_DATA_STALE"
    WEBSOCKET_RECONNECTION = "WEBSOCKET_RECONNECTION"
    POSITION_SIZE_ADJUSTED = "POSITION_SIZE_ADJUSTED"
    SIGNAL_QUALITY_LOW = "SIGNAL_QUALITY_LOW"
    BACKTEST_DEVIATION = "BACKTEST_DEVIATION"
    RATE_LIMIT_WARNING = "RATE_LIMIT_WARNING"

    # =========================================================================
    # LOW Alerts - Informational (Email only, can be batched)
    # =========================================================================
    DAILY_SUMMARY = "DAILY_SUMMARY"
    WEEKLY_PERFORMANCE_REPORT = "WEEKLY_PERFORMANCE_REPORT"
    MODEL_RETRAINING_COMPLETE = "MODEL_RETRAINING_COMPLETE"
    NEW_POSITIONS_OPENED = "NEW_POSITIONS_OPENED"
    SYSTEM_MAINTENANCE_SCHEDULED = "SYSTEM_MAINTENANCE_SCHEDULED"
    POSITION_CLOSED = "POSITION_CLOSED"
    MODEL_DEPLOYED = "MODEL_DEPLOYED"
    CONFIG_CHANGE = "CONFIG_CHANGE"
    HEARTBEAT = "HEARTBEAT"


# Default severity mapping for alert types
DEFAULT_SEVERITY_MAP: dict[AlertType, AlertSeverity] = {
    # CRITICAL
    AlertType.SYSTEM_DOWN: AlertSeverity.CRITICAL,
    AlertType.BROKER_CONNECTION_LOST: AlertSeverity.CRITICAL,
    AlertType.DATA_FEED_FAILURE: AlertSeverity.CRITICAL,
    AlertType.KILL_SWITCH_TRIGGERED: AlertSeverity.CRITICAL,
    AlertType.MAX_DRAWDOWN_BREACHED: AlertSeverity.CRITICAL,
    AlertType.DATABASE_FAILURE: AlertSeverity.CRITICAL,
    AlertType.DAILY_LOSS_LIMIT_BREACHED: AlertSeverity.CRITICAL,
    AlertType.AUTHENTICATION_FAILURE: AlertSeverity.CRITICAL,
    AlertType.ORDER_REJECTION_CRITICAL: AlertSeverity.CRITICAL,
    AlertType.MARGIN_CALL: AlertSeverity.CRITICAL,

    # HIGH
    AlertType.HIGH_LATENCY: AlertSeverity.HIGH,
    AlertType.MODEL_ACCURACY_DEGRADED: AlertSeverity.HIGH,
    AlertType.RISK_LIMIT_APPROACHING: AlertSeverity.HIGH,
    AlertType.UNUSUAL_VOLATILITY: AlertSeverity.HIGH,
    AlertType.POSITION_CONCENTRATION_HIGH: AlertSeverity.HIGH,
    AlertType.SLIPPAGE_ABOVE_THRESHOLD: AlertSeverity.HIGH,
    AlertType.CONNECTION_DEGRADED: AlertSeverity.HIGH,
    AlertType.MEMORY_PRESSURE: AlertSeverity.HIGH,
    AlertType.CPU_PRESSURE: AlertSeverity.HIGH,
    AlertType.ORDER_FILL_DELAYED: AlertSeverity.HIGH,

    # MEDIUM
    AlertType.MODEL_PREDICTION_ANOMALY: AlertSeverity.MEDIUM,
    AlertType.FEATURE_DATA_STALE: AlertSeverity.MEDIUM,
    AlertType.WEBSOCKET_RECONNECTION: AlertSeverity.MEDIUM,
    AlertType.POSITION_SIZE_ADJUSTED: AlertSeverity.MEDIUM,
    AlertType.SIGNAL_QUALITY_LOW: AlertSeverity.MEDIUM,
    AlertType.BACKTEST_DEVIATION: AlertSeverity.MEDIUM,
    AlertType.RATE_LIMIT_WARNING: AlertSeverity.MEDIUM,

    # LOW
    AlertType.DAILY_SUMMARY: AlertSeverity.LOW,
    AlertType.WEEKLY_PERFORMANCE_REPORT: AlertSeverity.LOW,
    AlertType.MODEL_RETRAINING_COMPLETE: AlertSeverity.LOW,
    AlertType.NEW_POSITIONS_OPENED: AlertSeverity.LOW,
    AlertType.SYSTEM_MAINTENANCE_SCHEDULED: AlertSeverity.LOW,
    AlertType.POSITION_CLOSED: AlertSeverity.LOW,
    AlertType.MODEL_DEPLOYED: AlertSeverity.LOW,
    AlertType.CONFIG_CHANGE: AlertSeverity.LOW,
    AlertType.HEARTBEAT: AlertSeverity.LOW,
}


# Runbook URL mappings for alert types
ALERT_RUNBOOK_MAP: dict[AlertType, str] = {
    AlertType.KILL_SWITCH_TRIGGERED: "runbooks/kill-switch-recovery",
    AlertType.MAX_DRAWDOWN_BREACHED: "runbooks/drawdown-breach-response",
    AlertType.DAILY_LOSS_LIMIT_BREACHED: "runbooks/daily-loss-limit-response",
    AlertType.BROKER_CONNECTION_LOST: "runbooks/broker-connection-recovery",
    AlertType.DATA_FEED_FAILURE: "runbooks/data-feed-recovery",
    AlertType.DATABASE_FAILURE: "runbooks/database-recovery",
    AlertType.SYSTEM_DOWN: "runbooks/system-recovery",
    AlertType.MODEL_ACCURACY_DEGRADED: "runbooks/model-performance-investigation",
    AlertType.HIGH_LATENCY: "runbooks/latency-investigation",
}


# Suggested actions for alert types
ALERT_SUGGESTED_ACTIONS: dict[AlertType, str] = {
    AlertType.KILL_SWITCH_TRIGGERED: "1. Review kill switch trigger reason. 2. Check recent trades for anomalies. 3. Verify market conditions. 4. Reset kill switch only after thorough investigation.",
    AlertType.MAX_DRAWDOWN_BREACHED: "1. HALT all trading immediately. 2. Review position P&L. 3. Close losing positions if appropriate. 4. Investigate root cause. 5. Consult risk management.",
    AlertType.DAILY_LOSS_LIMIT_BREACHED: "1. Trading is paused for the day. 2. Review all trades made today. 3. Analyze what went wrong. 4. Prepare incident report.",
    AlertType.BROKER_CONNECTION_LOST: "1. Check network connectivity. 2. Verify Alpaca API status. 3. Check API credentials. 4. Monitor pending orders. 5. Initiate reconnection.",
    AlertType.DATA_FEED_FAILURE: "1. Check WebSocket connection status. 2. Verify market hours. 3. Check data provider status. 4. Switch to backup data source if available.",
    AlertType.DATABASE_FAILURE: "1. Check database server status. 2. Verify connection pool. 3. Check disk space. 4. Review recent migrations. 5. Contact DBA if needed.",
    AlertType.MODEL_ACCURACY_DEGRADED: "1. Review recent predictions vs actual. 2. Check for data quality issues. 3. Consider model retraining. 4. Evaluate market regime change.",
    AlertType.HIGH_LATENCY: "1. Check system resources. 2. Review network conditions. 3. Check for resource contention. 4. Consider scaling up infrastructure.",
    AlertType.MODEL_PREDICTION_ANOMALY: "1. Review input features. 2. Check for data anomalies. 3. Compare with historical predictions. 4. Consider halting model temporarily.",
    AlertType.MARGIN_CALL: "1. URGENT: Deposit additional funds or close positions. 2. Review margin requirements. 3. Close highest-risk positions first. 4. Contact broker if needed.",
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


# =============================================================================
# Professional Alert Message Templates
# =============================================================================

class AlertTemplates:
    """Professional alert message templates for different channels.

    Templates are designed to be clear, actionable, and include all
    necessary information for rapid incident response.
    """

    # Slack Block Kit template for rich formatting
    SLACK_BLOCK_TEMPLATE = {
        "critical": {
            "color": "#FF0000",
            "emoji": ":rotating_light:",
            "header_emoji": ":fire:",
        },
        "high": {
            "color": "#FF6600",
            "emoji": ":warning:",
            "header_emoji": ":warning:",
        },
        "medium": {
            "color": "#FFCC00",
            "emoji": ":large_yellow_circle:",
            "header_emoji": ":bell:",
        },
        "low": {
            "color": "#0066FF",
            "emoji": ":information_source:",
            "header_emoji": ":memo:",
        },
    }

    # HTML Email template
    EMAIL_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden; }}
        .header {{ padding: 20px; color: white; }}
        .header.critical {{ background: linear-gradient(135deg, #FF0000, #CC0000); }}
        .header.high {{ background: linear-gradient(135deg, #FF6600, #CC5200); }}
        .header.medium {{ background: linear-gradient(135deg, #FFCC00, #E6B800); color: #333; }}
        .header.low {{ background: linear-gradient(135deg, #0066FF, #0052CC); }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        .header .severity {{ font-size: 14px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9; }}
        .content {{ padding: 20px; }}
        .section {{ margin-bottom: 20px; }}
        .section-title {{ font-size: 12px; text-transform: uppercase; color: #666; letter-spacing: 1px; margin-bottom: 8px; }}
        .section-content {{ font-size: 14px; color: #333; line-height: 1.6; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
        .metric {{ background: #f8f9fa; padding: 12px; border-radius: 4px; }}
        .metric-label {{ font-size: 11px; color: #666; text-transform: uppercase; }}
        .metric-value {{ font-size: 18px; font-weight: 600; color: #333; }}
        .action-box {{ background: #FFF3CD; border-left: 4px solid #FFC107; padding: 15px; border-radius: 4px; }}
        .action-box.critical {{ background: #F8D7DA; border-left-color: #DC3545; }}
        .footer {{ padding: 20px; background: #f8f9fa; text-align: center; font-size: 12px; color: #666; }}
        .btn {{ display: inline-block; padding: 10px 20px; background: #0066FF; color: white; text-decoration: none; border-radius: 4px; margin-top: 10px; }}
        .timestamp {{ font-size: 12px; color: #999; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header {severity_lower}">
            <div class="severity">{severity} ALERT</div>
            <h1>{title}</h1>
        </div>
        <div class="content">
            <div class="section">
                <div class="section-title">Alert Details</div>
                <div class="section-content">
                    <p><strong>Type:</strong> {alert_type}</p>
                    <p><strong>Time:</strong> {timestamp}</p>
                    <p><strong>Environment:</strong> {environment}</p>
                    <p><strong>Alert ID:</strong> {alert_id}</p>
                </div>
            </div>

            <div class="section">
                <div class="section-title">Description</div>
                <div class="section-content">{message}</div>
            </div>

            {metrics_section}

            <div class="section">
                <div class="action-box {action_class}">
                    <div class="section-title">Suggested Action</div>
                    <div class="section-content">{suggested_action}</div>
                </div>
            </div>

            {runbook_section}
        </div>
        <div class="footer">
            <p>{system_name} Alert System</p>
            <p class="timestamp">Generated at {timestamp}</p>
        </div>
    </div>
</body>
</html>
'''

    # Plain text email template
    EMAIL_TEXT_TEMPLATE = '''
================================================================================
{severity} ALERT - {title}
================================================================================

Alert Type: {alert_type}
Severity: {severity}
Time: {timestamp}
Environment: {environment}
Alert ID: {alert_id}

--------------------------------------------------------------------------------
DESCRIPTION
--------------------------------------------------------------------------------
{message}

--------------------------------------------------------------------------------
CONTEXT / METRICS
--------------------------------------------------------------------------------
{metrics_text}

--------------------------------------------------------------------------------
SUGGESTED ACTION
--------------------------------------------------------------------------------
{suggested_action}

{runbook_text}
================================================================================
{system_name} Alert System
================================================================================
'''

    # PagerDuty event template
    PAGERDUTY_TEMPLATE = {
        "routing_key": "{routing_key}",
        "event_action": "trigger",
        "dedup_key": "{fingerprint}",
        "payload": {
            "summary": "[{severity}] {title}",
            "source": "{system_name}",
            "severity": "{pd_severity}",
            "timestamp": "{timestamp}",
            "component": "{component}",
            "group": "{alert_type}",
            "class": "trading_alert",
            "custom_details": {}
        }
    }

    @classmethod
    def format_slack_message(cls, alert: "Alert") -> dict[str, Any]:
        """Format alert as a Slack Block Kit message.

        Args:
            alert: Alert to format.

        Returns:
            Slack message payload with rich formatting.
        """
        severity_key = alert.severity.value.lower()
        if severity_key not in cls.SLACK_BLOCK_TEMPLATE:
            severity_key = "medium"

        config = cls.SLACK_BLOCK_TEMPLATE[severity_key]
        env = AlertConfig.ALERT_ENVIRONMENT.upper()

        # Build blocks for Block Kit
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{config['header_emoji']} {alert.severity.value} Alert - {alert.title}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Type:*\n{alert.alert_type.value}"},
                    {"type": "mrkdwn", "text": f"*Severity:*\n{config['emoji']} {alert.severity.value}"},
                    {"type": "mrkdwn", "text": f"*Time:*\n{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"},
                    {"type": "mrkdwn", "text": f"*Environment:*\n{env}"},
                ]
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Description:*\n{alert.message}"}
            },
        ]

        # Add context metrics if present
        if alert.context:
            metrics_text = "\n".join([f"- *{k}:* `{v}`" for k, v in alert.context.items()])
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Metrics:*\n{metrics_text}"}
            })

        # Add suggested action
        if alert.suggested_action:
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f":point_right: *Suggested Action:*\n{alert.suggested_action}"}
            })

        # Add runbook link
        if alert.runbook_link:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f":book: <{alert.runbook_link}|View Runbook>"}
            })

        # Add footer with alert ID
        blocks.append({
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"Alert ID: `{alert.alert_id}` | Fingerprint: `{alert.fingerprint}`"}
            ]
        })

        return {
            "blocks": blocks,
            "attachments": [{
                "color": config["color"],
                "fallback": f"[{alert.severity.value}] {alert.title}: {alert.message}",
            }]
        }

    @classmethod
    def format_email_html(cls, alert: "Alert") -> str:
        """Format alert as HTML email.

        Args:
            alert: Alert to format.

        Returns:
            HTML formatted email string.
        """
        # Build metrics section
        metrics_html = ""
        if alert.context:
            metrics_items = "".join([
                f'<div class="metric"><div class="metric-label">{k}</div><div class="metric-value">{v}</div></div>'
                for k, v in alert.context.items()
            ])
            metrics_html = f'''
            <div class="section">
                <div class="section-title">Metrics / Context</div>
                <div class="metrics-grid">{metrics_items}</div>
            </div>
            '''

        # Build runbook section
        runbook_html = ""
        if alert.runbook_link:
            runbook_html = f'''
            <div class="section" style="text-align: center;">
                <a href="{html.escape(alert.runbook_link)}" class="btn">View Runbook</a>
            </div>
            '''

        action_class = "critical" if alert.severity == AlertSeverity.CRITICAL else ""

        return cls.EMAIL_HTML_TEMPLATE.format(
            severity=alert.severity.value,
            severity_lower=alert.severity.value.lower(),
            title=html.escape(alert.title),
            alert_type=alert.alert_type.value,
            timestamp=alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            environment=AlertConfig.ALERT_ENVIRONMENT.upper(),
            alert_id=str(alert.alert_id),
            message=html.escape(alert.message).replace("\n", "<br>"),
            metrics_section=metrics_html,
            suggested_action=html.escape(alert.suggested_action or "No specific action required.").replace("\n", "<br>"),
            action_class=action_class,
            runbook_section=runbook_html,
            system_name=AlertConfig.ALERT_SYSTEM_NAME,
        )

    @classmethod
    def format_email_text(cls, alert: "Alert") -> str:
        """Format alert as plain text email.

        Args:
            alert: Alert to format.

        Returns:
            Plain text formatted email string.
        """
        # Build metrics text
        metrics_text = "N/A"
        if alert.context:
            metrics_text = "\n".join([f"  {k}: {v}" for k, v in alert.context.items()])

        # Build runbook text
        runbook_text = ""
        if alert.runbook_link:
            runbook_text = f"\nRUNBOOK: {alert.runbook_link}\n"

        return cls.EMAIL_TEXT_TEMPLATE.format(
            severity=alert.severity.value,
            title=alert.title,
            alert_type=alert.alert_type.value,
            timestamp=alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            environment=AlertConfig.ALERT_ENVIRONMENT.upper(),
            alert_id=str(alert.alert_id),
            message=alert.message,
            metrics_text=metrics_text,
            suggested_action=alert.suggested_action or "No specific action required.",
            runbook_text=runbook_text,
            system_name=AlertConfig.ALERT_SYSTEM_NAME,
        )

    @classmethod
    def format_pagerduty_event(cls, alert: "Alert", routing_key: str) -> dict[str, Any]:
        """Format alert as PagerDuty Events API v2 payload.

        Args:
            alert: Alert to format.
            routing_key: PagerDuty integration routing key.

        Returns:
            PagerDuty event payload.
        """
        # Map severity to PagerDuty severity
        severity_map = {
            AlertSeverity.CRITICAL: "critical",
            AlertSeverity.HIGH: "error",
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.LOW: "info",
        }

        # Determine component from alert type
        component = "trading-system"
        if "MODEL" in alert.alert_type.value:
            component = "ml-models"
        elif "DATA" in alert.alert_type.value:
            component = "data-pipeline"
        elif "BROKER" in alert.alert_type.value or "ORDER" in alert.alert_type.value:
            component = "execution"
        elif "RISK" in alert.alert_type.value or "DRAWDOWN" in alert.alert_type.value or "LOSS" in alert.alert_type.value:
            component = "risk-management"

        return {
            "routing_key": routing_key,
            "event_action": "trigger",
            "dedup_key": alert.fingerprint,
            "payload": {
                "summary": f"[{alert.severity.value}] {alert.title}",
                "source": AlertConfig.ALERT_SYSTEM_NAME,
                "severity": severity_map.get(alert.severity, "warning"),
                "timestamp": alert.timestamp.isoformat(),
                "component": component,
                "group": alert.alert_type.value,
                "class": "trading_alert",
                "custom_details": {
                    "message": alert.message,
                    "alert_id": str(alert.alert_id),
                    "environment": AlertConfig.ALERT_ENVIRONMENT,
                    "suggested_action": alert.suggested_action,
                    "runbook": alert.runbook_link,
                    **alert.context,
                },
            },
            "links": [{"href": alert.runbook_link, "text": "Runbook"}] if alert.runbook_link else [],
            "images": [],
        }


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
    """Email alert notifier with professional HTML formatting.

    Sends professionally formatted email alerts with HTML and plain text
    alternatives. Supports configuration via environment variables.

    Environment Variables:
        SMTP_HOST: SMTP server hostname
        SMTP_PORT: SMTP server port (default: 587)
        SMTP_USER: SMTP authentication username
        SMTP_PASSWORD: SMTP authentication password
        SMTP_FROM_ADDRESS: Sender email address
        SMTP_TO_ADDRESSES: Comma-separated list of recipient addresses
        SMTP_USE_TLS: Whether to use TLS (default: true)
    """

    def __init__(
        self,
        smtp_host: str | None = None,
        smtp_port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        from_address: str | None = None,
        to_addresses: list[str] | None = None,
        use_tls: bool | None = None,
    ) -> None:
        """Initialize email notifier.

        Args:
            smtp_host: SMTP server host (default: from env SMTP_HOST).
            smtp_port: SMTP server port (default: from env SMTP_PORT or 587).
            username: SMTP username (default: from env SMTP_USER).
            password: SMTP password (default: from env SMTP_PASSWORD).
            from_address: Sender email address (default: from env SMTP_FROM_ADDRESS).
            to_addresses: List of recipient email addresses (default: from env SMTP_TO_ADDRESSES).
            use_tls: Whether to use TLS (default: from env SMTP_USE_TLS or True).
        """
        self.smtp_host = smtp_host or AlertConfig.SMTP_HOST
        self.smtp_port = smtp_port if smtp_port is not None else AlertConfig.SMTP_PORT
        self.username = username or AlertConfig.SMTP_USER
        self.password = password or AlertConfig.SMTP_PASSWORD
        self.from_address = from_address or AlertConfig.SMTP_FROM_ADDRESS
        self.to_addresses = to_addresses or AlertConfig.SMTP_TO_ADDRESSES
        self.use_tls = use_tls if use_tls is not None else AlertConfig.SMTP_USE_TLS

    @classmethod
    def from_config(cls) -> "EmailNotifier":
        """Create EmailNotifier from environment configuration.

        Returns:
            Configured EmailNotifier instance.

        Raises:
            ValueError: If required configuration is missing.
        """
        if not AlertConfig.is_email_configured():
            raise ValueError(
                "Email not configured. Set SMTP_HOST, SMTP_FROM_ADDRESS, and SMTP_TO_ADDRESSES."
            )
        return cls()

    def is_configured(self) -> bool:
        """Check if the notifier is properly configured."""
        return bool(self.smtp_host and self.from_address and self.to_addresses)

    async def send(self, alert: Alert) -> bool:
        """Send email notification with professional HTML formatting.

        Args:
            alert: Alert to send.

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self.is_configured():
            logger.warning("Email notifier not configured, skipping send")
            return False

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart("alternative")
            msg["From"] = self.from_address
            msg["To"] = ", ".join(self.to_addresses)
            msg["Subject"] = f"[{alert.severity.value}] {AlertConfig.ALERT_SYSTEM_NAME}: {alert.title}"
            msg["X-Priority"] = "1" if alert.severity == AlertSeverity.CRITICAL else "3"

            # Add plain text version
            text_body = AlertTemplates.format_email_text(alert)
            msg.attach(MIMEText(text_body, "plain", "utf-8"))

            # Add HTML version (preferred by most clients)
            html_body = AlertTemplates.format_email_html(alert)
            msg.attach(MIMEText(html_body, "html", "utf-8"))

            def send_sync() -> bool:
                try:
                    with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as server:
                        server.ehlo()
                        if self.use_tls:
                            server.starttls()
                            server.ehlo()
                        if self.username and self.password:
                            server.login(self.username, self.password)
                        server.send_message(msg)
                    return True
                except smtplib.SMTPException as e:
                    logger.error(f"SMTP error: {e}")
                    return False

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, send_sync)

            if result:
                logger.info(f"Email alert sent: {alert.title}", extra={"extra_data": {"recipients": len(self.to_addresses)}})
            return result

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}", extra={"extra_data": {"error": str(e)}})
            return False

    def get_channel(self) -> AlertChannel:
        return AlertChannel.EMAIL


class SlackNotifier(AlertNotifier):
    """Slack alert notifier with Block Kit formatting.

    Sends richly formatted Slack messages using Block Kit for better
    readability and actionability. Supports different channels for
    different severity levels.

    Environment Variables:
        SLACK_WEBHOOK_URL: Slack incoming webhook URL
        SLACK_CHANNEL: Default channel for alerts (default: #trading-alerts)
        SLACK_CRITICAL_CHANNEL: Channel for critical alerts (default: #trading-critical)
        SLACK_USERNAME: Bot username (default: AlphaTrade Alert Bot)
    """

    def __init__(
        self,
        webhook_url: str | None = None,
        channel: str | None = None,
        critical_channel: str | None = None,
        username: str | None = None,
    ) -> None:
        """Initialize Slack notifier.

        Args:
            webhook_url: Slack webhook URL (default: from env SLACK_WEBHOOK_URL).
            channel: Default Slack channel (default: from env SLACK_CHANNEL).
            critical_channel: Channel for critical alerts (default: from env SLACK_CRITICAL_CHANNEL).
            username: Bot username (default: from env SLACK_USERNAME).
        """
        self.webhook_url = webhook_url or AlertConfig.SLACK_WEBHOOK_URL
        self.channel = channel or AlertConfig.SLACK_CHANNEL
        self.critical_channel = critical_channel or AlertConfig.SLACK_CRITICAL_CHANNEL
        self.username = username or AlertConfig.SLACK_USERNAME

    @classmethod
    def from_config(cls) -> "SlackNotifier":
        """Create SlackNotifier from environment configuration.

        Returns:
            Configured SlackNotifier instance.

        Raises:
            ValueError: If required configuration is missing.
        """
        if not AlertConfig.is_slack_configured():
            raise ValueError("Slack not configured. Set SLACK_WEBHOOK_URL environment variable.")
        return cls()

    def is_configured(self) -> bool:
        """Check if the notifier is properly configured."""
        return bool(self.webhook_url)

    async def send(self, alert: Alert) -> bool:
        """Send Slack notification with Block Kit formatting.

        Args:
            alert: Alert to send.

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self.is_configured():
            logger.warning("Slack notifier not configured, skipping send")
            return False

        try:
            import aiohttp

            # Use critical channel for critical alerts
            target_channel = self.critical_channel if alert.severity == AlertSeverity.CRITICAL else self.channel

            # Get formatted message from templates
            payload = AlertTemplates.format_slack_message(alert)
            payload["channel"] = target_channel
            payload["username"] = self.username

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    success = response.status == 200
                    if success:
                        logger.info(f"Slack alert sent: {alert.title}", extra={"extra_data": {"channel": target_channel}})
                    else:
                        response_text = await response.text()
                        logger.error(f"Slack API error: {response.status} - {response_text}")
                    return success

        except asyncio.TimeoutError:
            logger.error("Slack notification timed out")
            return False
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


class PagerDutyNotifier(AlertNotifier):
    """PagerDuty alert notifier for critical incident management.

    Integrates with PagerDuty Events API v2 for incident creation and
    management. Supports automatic deduplication and resolution.

    Environment Variables:
        PAGERDUTY_SERVICE_KEY: PagerDuty Events API v2 integration key
        PAGERDUTY_API_URL: API endpoint (default: https://events.pagerduty.com/v2/enqueue)
    """

    def __init__(
        self,
        routing_key: str | None = None,
        api_url: str | None = None,
    ) -> None:
        """Initialize PagerDuty notifier.

        Args:
            routing_key: PagerDuty Events API v2 routing key (default: from env).
            api_url: PagerDuty Events API endpoint (default: from env).
        """
        self.routing_key = routing_key or AlertConfig.PAGERDUTY_SERVICE_KEY
        self.api_url = api_url or AlertConfig.PAGERDUTY_API_URL

    @classmethod
    def from_config(cls) -> "PagerDutyNotifier":
        """Create PagerDutyNotifier from environment configuration.

        Returns:
            Configured PagerDutyNotifier instance.

        Raises:
            ValueError: If required configuration is missing.
        """
        if not AlertConfig.is_pagerduty_configured():
            raise ValueError("PagerDuty not configured. Set PAGERDUTY_SERVICE_KEY environment variable.")
        return cls()

    def is_configured(self) -> bool:
        """Check if the notifier is properly configured."""
        return bool(self.routing_key)

    async def send(self, alert: Alert) -> bool:
        """Send PagerDuty notification using professional template.

        Args:
            alert: Alert to send.

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self.is_configured():
            logger.warning("PagerDuty notifier not configured, skipping send")
            return False

        try:
            import aiohttp

            # Use professional template for payload
            payload = AlertTemplates.format_pagerduty_event(alert, self.routing_key)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status in (200, 201, 202):
                        response_data = await response.json()
                        dedup_key = response_data.get("dedup_key", alert.fingerprint)
                        logger.info(
                            f"PagerDuty incident created: {alert.title}",
                            extra={"extra_data": {"dedup_key": dedup_key}}
                        )
                        return True
                    else:
                        response_text = await response.text()
                        logger.error(f"PagerDuty API error: {response.status} - {response_text}")
                        return False

        except asyncio.TimeoutError:
            logger.error("PagerDuty notification timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False

    async def resolve(self, fingerprint: str) -> bool:
        """Resolve an existing PagerDuty incident.

        Args:
            fingerprint: Alert fingerprint (dedup_key) to resolve.

        Returns:
            True if resolved successfully, False otherwise.
        """
        if not self.is_configured():
            return False

        try:
            import aiohttp

            payload = {
                "routing_key": self.routing_key,
                "event_action": "resolve",
                "dedup_key": fingerprint,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status in (200, 201, 202):
                        logger.info(f"PagerDuty incident resolved: {fingerprint}")
                        return True
                    return False

        except Exception as e:
            logger.error(f"Failed to resolve PagerDuty incident: {e}")
            return False

    def get_channel(self) -> AlertChannel:
        return AlertChannel.PAGERDUTY


class SMSNotifier(AlertNotifier):
    """SMS alert notifier using Twilio.

    Sends SMS alerts via Twilio API for urgent notifications.
    Typically used for CRITICAL alerts as an escalation channel.

    Environment Variables:
        TWILIO_ACCOUNT_SID: Twilio account SID
        TWILIO_AUTH_TOKEN: Twilio auth token
        TWILIO_FROM_NUMBER: Twilio phone number to send from
        TWILIO_TO_NUMBERS: Comma-separated list of phone numbers to send to
    """

    def __init__(
        self,
        account_sid: str | None = None,
        auth_token: str | None = None,
        from_number: str | None = None,
        to_numbers: list[str] | None = None,
    ) -> None:
        """Initialize SMS notifier.

        Args:
            account_sid: Twilio account SID (default: from env).
            auth_token: Twilio auth token (default: from env).
            from_number: Twilio phone number to send from (default: from env).
            to_numbers: List of phone numbers to send to (default: from env).
        """
        self.account_sid = account_sid or AlertConfig.TWILIO_ACCOUNT_SID
        self.auth_token = auth_token or AlertConfig.TWILIO_AUTH_TOKEN
        self.from_number = from_number or AlertConfig.TWILIO_FROM_NUMBER
        self.to_numbers = to_numbers or AlertConfig.TWILIO_TO_NUMBERS

    @classmethod
    def from_config(cls) -> "SMSNotifier":
        """Create SMSNotifier from environment configuration.

        Returns:
            Configured SMSNotifier instance.

        Raises:
            ValueError: If required configuration is missing.
        """
        if not AlertConfig.is_sms_configured():
            raise ValueError(
                "SMS not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, "
                "TWILIO_FROM_NUMBER, and TWILIO_TO_NUMBERS environment variables."
            )
        return cls()

    def is_configured(self) -> bool:
        """Check if the notifier is properly configured."""
        return bool(
            self.account_sid and
            self.auth_token and
            self.from_number and
            self.to_numbers
        )

    async def send(self, alert: Alert) -> bool:
        """Send SMS notification via Twilio.

        Args:
            alert: Alert to send.

        Returns:
            True if sent successfully to at least one recipient, False otherwise.
        """
        if not self.is_configured():
            logger.warning("SMS notifier not configured, skipping send")
            return False

        try:
            import aiohttp
            from base64 import b64encode

            # Construct SMS message (optimized for single SMS)
            env_tag = f"[{AlertConfig.ALERT_ENVIRONMENT.upper()[:3]}]" if AlertConfig.ALERT_ENVIRONMENT != "production" else ""
            message = f"{env_tag}[{alert.severity.value}] {AlertConfig.ALERT_SYSTEM_NAME}: {alert.title[:60]}"

            if alert.suggested_action:
                remaining = 155 - len(message)
                if remaining > 20:
                    action_preview = alert.suggested_action.split(".")[0][:remaining]
                    message += f" - {action_preview}"

            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
            auth = b64encode(f"{self.account_sid}:{self.auth_token}".encode()).decode()

            success_count = 0
            async with aiohttp.ClientSession() as session:
                for to_number in self.to_numbers:
                    data = {
                        "From": self.from_number,
                        "To": to_number,
                        "Body": message,
                    }
                    headers = {
                        "Authorization": f"Basic {auth}",
                        "Content-Type": "application/x-www-form-urlencoded",
                    }

                    try:
                        async with session.post(
                            url,
                            data=data,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as response:
                            if response.status in (200, 201):
                                success_count += 1
                                logger.info(f"SMS sent to {to_number[-4:]}****")
                            else:
                                response_text = await response.text()
                                logger.warning(f"Failed to send SMS to {to_number}: {response.status} - {response_text}")
                    except asyncio.TimeoutError:
                        logger.warning(f"SMS to {to_number} timed out")

            if success_count > 0:
                logger.info(f"SMS alert sent: {alert.title}", extra={"extra_data": {"recipients": success_count}})
            return success_count > 0

        except Exception as e:
            logger.error(f"Failed to send SMS alert: {e}")
            return False

    def get_channel(self) -> AlertChannel:
        return AlertChannel.SMS


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
    - Escalation handling with timeout
    - Acknowledgment tracking

    JPMorgan-level enhancement: Added escalation timeout functionality
    to automatically escalate unacknowledged alerts.
    """

    def __init__(
        self,
        rate_limit_window_seconds: int = 300,
        rate_limit_max_alerts: int = 5,
        dedup_window_seconds: int = 600,
        escalation_check_interval_seconds: int = 60,
    ) -> None:
        """Initialize the alert manager.

        Args:
            rate_limit_window_seconds: Rate limit window in seconds.
            rate_limit_max_alerts: Maximum alerts per fingerprint in window.
            dedup_window_seconds: Deduplication window in seconds.
            escalation_check_interval_seconds: How often to check for escalations.
        """
        self.rate_limit_window = timedelta(seconds=rate_limit_window_seconds)
        self.rate_limit_max = rate_limit_max_alerts
        self.dedup_window = timedelta(seconds=dedup_window_seconds)
        self.escalation_check_interval = escalation_check_interval_seconds

        self._notifiers: dict[AlertChannel, AlertNotifier] = {}
        self._routing_rules: dict[AlertSeverity, AlertRule] = {}
        self._rate_limits: dict[str, RateLimitEntry] = defaultdict(RateLimitEntry)
        self._active_alerts: dict[str, Alert] = {}
        self._alert_history: list[Alert] = []
        self._max_history = 10000
        self._suppressed_fingerprints: set[str] = set()

        # JPMorgan-level enhancement: Escalation tracking
        self._escalated_alerts: set[str] = set()  # Alert fingerprints that have been escalated
        self._escalation_task: asyncio.Task | None = None
        self._escalation_running = False

        # Default routing rules
        self._setup_default_routing()

        # Dashboard notifier always enabled
        self._dashboard_notifier = DashboardNotifier()
        self._notifiers[AlertChannel.DASHBOARD] = self._dashboard_notifier

    def _setup_default_routing(self) -> None:
        """Set up default routing rules.

        Routing follows institutional alerting best practices:
        - CRITICAL: PagerDuty + Slack + Email (immediate, no quiet hours)
        - HIGH: Slack + Email
        - MEDIUM: Slack only (with quiet hours)
        - LOW: Email only (can be batched, with quiet hours)
        """
        self._routing_rules = {
            AlertSeverity.CRITICAL: AlertRule(
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.PAGERDUTY, AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.DASHBOARD],
                escalation_minutes=5,
                escalation_channels=[AlertChannel.SMS],
            ),
            AlertSeverity.HIGH: AlertRule(
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.DASHBOARD],
                quiet_hours_start=None,  # No quiet hours for HIGH
                quiet_hours_end=None,
            ),
            AlertSeverity.MEDIUM: AlertRule(
                severity=AlertSeverity.MEDIUM,
                channels=[AlertChannel.SLACK, AlertChannel.DASHBOARD],
                quiet_hours_start=22,  # 10 PM
                quiet_hours_end=7,     # 7 AM
            ),
            AlertSeverity.LOW: AlertRule(
                severity=AlertSeverity.LOW,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD],
                quiet_hours_start=20,  # 8 PM
                quiet_hours_end=8,     # 8 AM
            ),
        }

    def setup_notifiers_from_config(self) -> dict[str, bool]:
        """Set up all notifiers from environment configuration.

        Automatically registers notifiers that are properly configured
        via environment variables.

        Returns:
            Dictionary of channel names and their configuration status.
        """
        status = {}

        # Try to set up Slack notifier
        if AlertConfig.is_slack_configured():
            try:
                self.register_notifier(SlackNotifier.from_config())
                status["slack"] = True
                logger.info("Slack notifier configured successfully")
            except Exception as e:
                status["slack"] = False
                logger.warning(f"Failed to configure Slack notifier: {e}")
        else:
            status["slack"] = False
            logger.debug("Slack not configured (SLACK_WEBHOOK_URL not set)")

        # Try to set up Email notifier
        if AlertConfig.is_email_configured():
            try:
                self.register_notifier(EmailNotifier.from_config())
                status["email"] = True
                logger.info("Email notifier configured successfully")
            except Exception as e:
                status["email"] = False
                logger.warning(f"Failed to configure Email notifier: {e}")
        else:
            status["email"] = False
            logger.debug("Email not configured (SMTP settings not set)")

        # Try to set up PagerDuty notifier
        if AlertConfig.is_pagerduty_configured():
            try:
                self.register_notifier(PagerDutyNotifier.from_config())
                status["pagerduty"] = True
                logger.info("PagerDuty notifier configured successfully")
            except Exception as e:
                status["pagerduty"] = False
                logger.warning(f"Failed to configure PagerDuty notifier: {e}")
        else:
            status["pagerduty"] = False
            logger.debug("PagerDuty not configured (PAGERDUTY_SERVICE_KEY not set)")

        # Try to set up SMS notifier
        if AlertConfig.is_sms_configured():
            try:
                self.register_notifier(SMSNotifier.from_config())
                status["sms"] = True
                logger.info("SMS notifier configured successfully")
            except Exception as e:
                status["sms"] = False
                logger.warning(f"Failed to configure SMS notifier: {e}")
        else:
            status["sms"] = False
            logger.debug("SMS not configured (Twilio settings not set)")

        return status

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

    # =========================================================================
    # JPMorgan-level Enhancement: Alert Escalation Timeout
    # =========================================================================

    async def start_escalation_monitor(self) -> None:
        """Start the background escalation monitoring task.

        JPMorgan-level enhancement: Automatically escalates unacknowledged
        alerts after the configured timeout period.
        """
        if self._escalation_running:
            logger.warning("Escalation monitor already running")
            return

        self._escalation_running = True
        self._escalation_task = asyncio.create_task(self._escalation_loop())
        logger.info(
            f"Started escalation monitor (check interval: {self.escalation_check_interval}s)"
        )

    async def stop_escalation_monitor(self) -> None:
        """Stop the background escalation monitoring task."""
        self._escalation_running = False

        if self._escalation_task:
            self._escalation_task.cancel()
            try:
                await self._escalation_task
            except asyncio.CancelledError:
                pass
            self._escalation_task = None

        logger.info("Stopped escalation monitor")

    async def _escalation_loop(self) -> None:
        """Background loop that checks for alerts needing escalation."""
        while self._escalation_running:
            try:
                await self._check_escalations()
            except Exception as e:
                logger.error(f"Error in escalation check: {e}")

            await asyncio.sleep(self.escalation_check_interval)

    async def _check_escalations(self) -> None:
        """Check all active alerts for escalation needs.

        Escalates alerts that:
        1. Are still in FIRING status (not acknowledged)
        2. Have exceeded their escalation timeout
        3. Have not already been escalated
        """
        now = datetime.now(timezone.utc)

        for fingerprint, alert in list(self._active_alerts.items()):
            # Skip already escalated or acknowledged alerts
            if fingerprint in self._escalated_alerts:
                continue
            if alert.status != AlertStatus.FIRING:
                continue

            # Get routing rule for this severity
            rule = self._routing_rules.get(alert.severity)
            if not rule or rule.escalation_minutes is None:
                continue

            # Check if escalation timeout exceeded
            escalation_threshold = timedelta(minutes=rule.escalation_minutes)
            time_since_alert = now - alert.timestamp

            if time_since_alert >= escalation_threshold:
                await self._escalate_alert(alert, rule)

    async def _escalate_alert(self, alert: Alert, rule: AlertRule) -> None:
        """Escalate an alert to additional channels.

        Args:
            alert: The alert to escalate.
            rule: The routing rule with escalation channels.
        """
        if not rule.escalation_channels:
            return

        logger.warning(
            f"Escalating unacknowledged alert: {alert.title} "
            f"(after {rule.escalation_minutes} minutes)"
        )

        # Mark as escalated
        self._escalated_alerts.add(alert.fingerprint)

        # Update alert metadata
        alert.context["escalated"] = True
        alert.context["escalated_at"] = datetime.now(timezone.utc).isoformat()
        alert.context["escalation_reason"] = f"Unacknowledged after {rule.escalation_minutes} minutes"

        # Create escalation message
        escalation_title = f"[ESCALATED] {alert.title}"
        escalation_message = (
            f"ALERT ESCALATION: This alert has been unacknowledged for "
            f"{rule.escalation_minutes} minutes.\n\n"
            f"Original message: {alert.message}"
        )

        # Send to escalation channels
        for channel in rule.escalation_channels:
            notifier = self._notifiers.get(channel)
            if notifier:
                try:
                    # Create escalated version of the alert
                    escalated_alert = Alert(
                        alert_type=alert.alert_type,
                        severity=AlertSeverity.CRITICAL,  # Escalated alerts are critical
                        title=escalation_title,
                        message=escalation_message,
                        context=alert.context,
                        suggested_action=alert.suggested_action,
                        runbook_link=alert.runbook_link,
                    )
                    success = await notifier.send(escalated_alert)
                    if success:
                        logger.info(f"Escalation sent to {channel.value}")
                    else:
                        logger.warning(f"Failed to send escalation to {channel.value}")
                except Exception as e:
                    logger.error(f"Error sending escalation to {channel.value}: {e}")

    def get_pending_escalations(self) -> list[tuple[Alert, int]]:
        """Get alerts pending escalation with time remaining.

        Returns:
            List of (alert, seconds_until_escalation) tuples.
        """
        now = datetime.now(timezone.utc)
        pending = []

        for fingerprint, alert in self._active_alerts.items():
            # Skip already escalated or acknowledged
            if fingerprint in self._escalated_alerts:
                continue
            if alert.status != AlertStatus.FIRING:
                continue

            rule = self._routing_rules.get(alert.severity)
            if not rule or rule.escalation_minutes is None:
                continue

            escalation_threshold = timedelta(minutes=rule.escalation_minutes)
            time_since_alert = now - alert.timestamp
            time_remaining = escalation_threshold - time_since_alert

            if time_remaining.total_seconds() > 0:
                pending.append((alert, int(time_remaining.total_seconds())))

        return sorted(pending, key=lambda x: x[1])

    def reset_escalation(self, fingerprint: str) -> bool:
        """Reset escalation status for an alert.

        Args:
            fingerprint: Alert fingerprint to reset.

        Returns:
            True if reset was successful.
        """
        if fingerprint in self._escalated_alerts:
            self._escalated_alerts.discard(fingerprint)
            logger.info(f"Escalation reset for alert: {fingerprint}")
            return True
        return False

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
