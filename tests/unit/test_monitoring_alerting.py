"""
Unit tests for monitoring/alerting.py
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quant_trading_system.monitoring.alerting import (
    Alert,
    AlertChannel,
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    AlertType,
    DashboardNotifier,
    EmailNotifier,
    SlackNotifier,
    WebhookNotifier,
    get_alert_manager,
    alert_critical,
    alert_warning,
    alert_info,
    DEFAULT_SEVERITY_MAP,
)


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_severity_values(self):
        """Test severity enum values."""
        assert AlertSeverity.CRITICAL == "CRITICAL"
        assert AlertSeverity.WARNING == "WARNING"
        assert AlertSeverity.INFO == "INFO"


class TestAlertChannel:
    """Tests for AlertChannel enum."""

    def test_channel_values(self):
        """Test channel enum values."""
        assert AlertChannel.EMAIL == "EMAIL"
        assert AlertChannel.SMS == "SMS"
        assert AlertChannel.SLACK == "SLACK"
        assert AlertChannel.PAGERDUTY == "PAGERDUTY"
        assert AlertChannel.DASHBOARD == "DASHBOARD"
        assert AlertChannel.WEBHOOK == "WEBHOOK"


class TestAlertType:
    """Tests for AlertType enum."""

    def test_critical_alert_types(self):
        """Test critical alert types."""
        assert AlertType.SYSTEM_DOWN == "SYSTEM_DOWN"
        assert AlertType.BROKER_CONNECTION_LOST == "BROKER_CONNECTION_LOST"
        assert AlertType.KILL_SWITCH_TRIGGERED == "KILL_SWITCH_TRIGGERED"

    def test_warning_alert_types(self):
        """Test warning alert types."""
        assert AlertType.HIGH_LATENCY == "HIGH_LATENCY"
        assert AlertType.MODEL_ACCURACY_DEGRADED == "MODEL_ACCURACY_DEGRADED"

    def test_info_alert_types(self):
        """Test info alert types."""
        assert AlertType.DAILY_SUMMARY == "DAILY_SUMMARY"
        assert AlertType.MODEL_RETRAINING_COMPLETE == "MODEL_RETRAINING_COMPLETE"


class TestDefaultSeverityMap:
    """Tests for default severity mapping."""

    def test_critical_mappings(self):
        """Test that critical alerts are mapped correctly."""
        assert DEFAULT_SEVERITY_MAP[AlertType.SYSTEM_DOWN] == AlertSeverity.CRITICAL
        assert DEFAULT_SEVERITY_MAP[AlertType.KILL_SWITCH_TRIGGERED] == AlertSeverity.CRITICAL

    def test_warning_mappings(self):
        """Test that warning alerts are mapped correctly."""
        assert DEFAULT_SEVERITY_MAP[AlertType.HIGH_LATENCY] == AlertSeverity.WARNING
        assert DEFAULT_SEVERITY_MAP[AlertType.SLIPPAGE_ABOVE_THRESHOLD] == AlertSeverity.WARNING

    def test_info_mappings(self):
        """Test that info alerts are mapped correctly."""
        assert DEFAULT_SEVERITY_MAP[AlertType.DAILY_SUMMARY] == AlertSeverity.INFO


class TestAlert:
    """Tests for Alert model."""

    def test_create_alert(self):
        """Test creating an alert."""
        alert = Alert(
            alert_type=AlertType.HIGH_LATENCY,
            severity=AlertSeverity.WARNING,
            title="High Latency Detected",
            message="Order execution latency exceeded threshold",
            context={"latency_ms": 500, "threshold_ms": 200},
            suggested_action="Check network connectivity",
        )
        assert alert.alert_type == AlertType.HIGH_LATENCY
        assert alert.severity == AlertSeverity.WARNING
        assert alert.status == AlertStatus.FIRING
        assert alert.fingerprint is not None

    def test_alert_fingerprint_generation(self):
        """Test that fingerprint is generated consistently."""
        alert1 = Alert(
            alert_type=AlertType.HIGH_LATENCY,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="Test",
            context={"key": "value"},
        )
        alert2 = Alert(
            alert_type=AlertType.HIGH_LATENCY,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="Test",
            context={"key": "value"},
        )
        assert alert1.fingerprint == alert2.fingerprint

    def test_alert_acknowledge(self):
        """Test acknowledging an alert."""
        alert = Alert(
            alert_type=AlertType.SYSTEM_DOWN,
            severity=AlertSeverity.CRITICAL,
            title="System Down",
            message="Main service unavailable",
        )
        alert.acknowledge("admin@example.com")
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == "admin@example.com"
        assert alert.acknowledged_at is not None

    def test_alert_resolve(self):
        """Test resolving an alert."""
        alert = Alert(
            alert_type=AlertType.DATABASE_FAILURE,
            severity=AlertSeverity.CRITICAL,
            title="DB Connection Lost",
            message="Cannot connect to database",
        )
        alert.resolve()
        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_at is not None

    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        alert = Alert(
            alert_type=AlertType.DAILY_SUMMARY,
            severity=AlertSeverity.INFO,
            title="Daily Summary",
            message="Daily trading summary",
            context={"trades": 10, "pnl": 500.0},
        )
        d = alert.to_dict()
        assert d["alert_type"] == "DAILY_SUMMARY"
        assert d["severity"] == "INFO"
        assert d["title"] == "Daily Summary"
        assert d["context"]["trades"] == 10


class TestAlertRule:
    """Tests for AlertRule class."""

    def test_create_rule(self):
        """Test creating an alert rule."""
        rule = AlertRule(
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            quiet_hours_start=22,
            quiet_hours_end=7,
            escalation_minutes=5,
            escalation_channels=[AlertChannel.SMS],
        )
        assert rule.severity == AlertSeverity.CRITICAL
        assert len(rule.channels) == 2
        assert rule.escalation_minutes == 5


class TestDashboardNotifier:
    """Tests for DashboardNotifier class."""

    def test_send_alert(self):
        """Test sending alert to dashboard."""
        notifier = DashboardNotifier(max_alerts=100)
        alert = Alert(
            alert_type=AlertType.HIGH_LATENCY,
            severity=AlertSeverity.WARNING,
            title="Test",
            message="Test alert",
        )

        async def run_test():
            result = await notifier.send(alert)
            return result

        result = asyncio.run(run_test())
        assert result is True
        assert len(notifier.alerts) == 1

    def test_get_channel(self):
        """Test getting channel type."""
        notifier = DashboardNotifier()
        assert notifier.get_channel() == AlertChannel.DASHBOARD

    def test_get_alerts_with_filter(self):
        """Test getting alerts with filters."""
        notifier = DashboardNotifier()

        async def run_test():
            await notifier.send(Alert(
                alert_type=AlertType.HIGH_LATENCY,
                severity=AlertSeverity.WARNING,
                title="Warning 1",
                message="Test",
            ))
            await notifier.send(Alert(
                alert_type=AlertType.SYSTEM_DOWN,
                severity=AlertSeverity.CRITICAL,
                title="Critical 1",
                message="Test",
            ))
            return notifier.get_alerts(severity=AlertSeverity.CRITICAL)

        alerts = asyncio.run(run_test())
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_max_alerts_limit(self):
        """Test that max alerts limit is enforced."""
        notifier = DashboardNotifier(max_alerts=5)

        async def run_test():
            for i in range(10):
                await notifier.send(Alert(
                    alert_type=AlertType.HIGH_LATENCY,
                    severity=AlertSeverity.WARNING,
                    title=f"Alert {i}",
                    message="Test",
                ))
            return len(notifier.alerts)

        count = asyncio.run(run_test())
        assert count == 5


class TestWebhookNotifier:
    """Tests for WebhookNotifier class."""

    def test_get_channel(self):
        """Test getting channel type."""
        notifier = WebhookNotifier(webhook_url="https://example.com/webhook")
        assert notifier.get_channel() == AlertChannel.WEBHOOK


class TestEmailNotifier:
    """Tests for EmailNotifier class."""

    def test_get_channel(self):
        """Test getting channel type."""
        notifier = EmailNotifier()
        assert notifier.get_channel() == AlertChannel.EMAIL


class TestSlackNotifier:
    """Tests for SlackNotifier class."""

    def test_get_channel(self):
        """Test getting channel type."""
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        assert notifier.get_channel() == AlertChannel.SLACK


class TestAlertManager:
    """Tests for AlertManager class."""

    @pytest.fixture
    def alert_manager(self):
        """Create a fresh AlertManager for testing."""
        return AlertManager()

    def test_create_alert(self, alert_manager):
        """Test creating an alert through the manager."""

        async def run_test():
            alert = await alert_manager.create_alert(
                alert_type=AlertType.HIGH_LATENCY,
                title="Test Alert",
                message="Test message",
                context={"latency": 500},
            )
            return alert

        alert = asyncio.run(run_test())
        assert alert is not None
        assert alert.alert_type == AlertType.HIGH_LATENCY

    def test_create_alert_with_severity_override(self, alert_manager):
        """Test creating alert with severity override."""

        async def run_test():
            alert = await alert_manager.create_alert(
                alert_type=AlertType.DAILY_SUMMARY,
                title="Important Summary",
                message="Critical summary",
                severity=AlertSeverity.CRITICAL,
            )
            return alert

        alert = asyncio.run(run_test())
        assert alert.severity == AlertSeverity.CRITICAL

    def test_suppress_alert(self, alert_manager):
        """Test suppressing alerts by fingerprint."""

        async def run_test():
            # Create an alert to get its fingerprint
            alert = await alert_manager.create_alert(
                alert_type=AlertType.HIGH_LATENCY,
                title="Suppress Test",
                message="Test",
            )
            fingerprint = alert.fingerprint

            # Suppress this fingerprint
            alert_manager.suppress_alert(fingerprint)

            # Try to create the same alert again
            suppressed = await alert_manager.create_alert(
                alert_type=AlertType.HIGH_LATENCY,
                title="Suppress Test",
                message="Test",
            )
            return suppressed

        result = asyncio.run(run_test())
        assert result is None

    def test_acknowledge_alert(self, alert_manager):
        """Test acknowledging an alert."""

        async def run_test():
            alert = await alert_manager.create_alert(
                alert_type=AlertType.HIGH_LATENCY,
                title="Ack Test",
                message="Test",
            )
            result = alert_manager.acknowledge_alert(str(alert.alert_id), "admin")
            return result, alert

        result, alert = asyncio.run(run_test())
        assert result is True
        assert alert.status == AlertStatus.ACKNOWLEDGED

    def test_resolve_alert(self, alert_manager):
        """Test resolving an alert."""

        async def run_test():
            alert = await alert_manager.create_alert(
                alert_type=AlertType.HIGH_LATENCY,
                title="Resolve Test",
                message="Test",
            )
            result = alert_manager.resolve_alert(str(alert.alert_id))
            return result

        result = asyncio.run(run_test())
        assert result is True

    def test_resolve_by_type(self, alert_manager):
        """Test resolving alerts by type."""

        async def run_test():
            await alert_manager.create_alert(
                alert_type=AlertType.HIGH_LATENCY,
                title="Test 1",
                message="Test",
            )
            await alert_manager.create_alert(
                alert_type=AlertType.HIGH_LATENCY,
                title="Test 2",
                message="Test",
            )
            count = alert_manager.resolve_by_type(AlertType.HIGH_LATENCY)
            return count

        count = asyncio.run(run_test())
        assert count == 2

    def test_get_active_alerts(self, alert_manager):
        """Test getting active alerts."""

        async def run_test():
            await alert_manager.create_alert(
                alert_type=AlertType.HIGH_LATENCY,
                title="Active 1",
                message="Test",
            )
            await alert_manager.create_alert(
                alert_type=AlertType.SYSTEM_DOWN,
                title="Active 2",
                message="Test",
            )
            return alert_manager.get_active_alerts()

        alerts = asyncio.run(run_test())
        assert len(alerts) == 2

    def test_get_active_alerts_filtered(self, alert_manager):
        """Test getting active alerts with filters."""

        async def run_test():
            await alert_manager.create_alert(
                alert_type=AlertType.HIGH_LATENCY,
                title="Warning",
                message="Test",
                severity=AlertSeverity.WARNING,
            )
            await alert_manager.create_alert(
                alert_type=AlertType.SYSTEM_DOWN,
                title="Critical",
                message="Test",
                severity=AlertSeverity.CRITICAL,
            )
            return alert_manager.get_active_alerts(severity=AlertSeverity.CRITICAL)

        alerts = asyncio.run(run_test())
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_get_stats(self, alert_manager):
        """Test getting alert statistics."""

        async def run_test():
            await alert_manager.create_alert(
                alert_type=AlertType.HIGH_LATENCY,
                title="Test",
                message="Test",
            )
            return alert_manager.get_stats()

        stats = asyncio.run(run_test())
        assert "active_count" in stats
        assert "active_by_severity" in stats
        assert stats["active_count"] == 1

    def test_register_notifier(self, alert_manager):
        """Test registering a notifier."""
        notifier = DashboardNotifier()
        alert_manager.register_notifier(notifier)
        assert AlertChannel.DASHBOARD in alert_manager._notifiers

    def test_set_routing_rule(self, alert_manager):
        """Test setting a routing rule."""
        rule = AlertRule(
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.SLACK],
        )
        alert_manager.set_routing_rule(rule)
        assert AlertSeverity.CRITICAL in alert_manager._routing_rules


class TestConvenienceAlertFunctions:
    """Tests for convenience alert functions."""

    def test_alert_critical(self):
        """Test alert_critical function."""

        async def run_test():
            # Reset singleton
            from quant_trading_system.monitoring import alerting
            alerting._alert_manager = None

            alert = await alert_critical(
                AlertType.SYSTEM_DOWN,
                "System Down",
                "Main service unavailable",
                component="trading_engine",
            )
            return alert

        alert = asyncio.run(run_test())
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL

    def test_alert_warning(self):
        """Test alert_warning function."""

        async def run_test():
            from quant_trading_system.monitoring import alerting
            alerting._alert_manager = None

            alert = await alert_warning(
                AlertType.HIGH_LATENCY,
                "High Latency",
                "Latency exceeded threshold",
                latency_ms=500,
            )
            return alert

        alert = asyncio.run(run_test())
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING

    def test_alert_info(self):
        """Test alert_info function."""

        async def run_test():
            from quant_trading_system.monitoring import alerting
            alerting._alert_manager = None

            alert = await alert_info(
                AlertType.DAILY_SUMMARY,
                "Daily Summary",
                "Trading day completed",
                trades=50,
            )
            return alert

        alert = asyncio.run(run_test())
        assert alert is not None
        assert alert.severity == AlertSeverity.INFO


class TestGetAlertManager:
    """Tests for get_alert_manager function."""

    def test_singleton_pattern(self):
        """Test that get_alert_manager returns singleton."""
        from quant_trading_system.monitoring import alerting
        alerting._alert_manager = None

        manager1 = get_alert_manager()
        manager2 = get_alert_manager()
        assert manager1 is manager2
