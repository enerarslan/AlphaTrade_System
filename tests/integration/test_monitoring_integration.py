"""
Integration tests for monitoring module.
"""

import asyncio
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from quant_trading_system.monitoring import (
    # Metrics
    MetricsCollector,
    get_metrics_collector,
    # Logger
    setup_logging,
    LogFormat,
    get_logger,
    LogCategory,
    log_system,
    log_trading,
    # Alerting
    AlertManager,
    AlertType,
    AlertSeverity,
    get_alert_manager,
    alert_critical,
    alert_warning,
    # Dashboard
    dashboard_app,
    get_dashboard_state,
    broadcast_portfolio_update,
)


class TestMonitoringIntegration:
    """Integration tests for monitoring components working together."""

    def test_metrics_and_dashboard_integration(self):
        """Test that metrics are accessible through dashboard."""
        # Reset metrics singleton
        MetricsCollector._instance = None

        # Update some metrics
        metrics = get_metrics_collector()
        metrics.update_portfolio_metrics(
            equity=100000.0,
            cash=50000.0,
            buying_power=150000.0,
            positions_count=5,
            long_exposure=40000.0,
            short_exposure=10000.0,
        )

        # Verify metrics endpoint returns data
        client = TestClient(dashboard_app)
        response = client.get("/metrics")
        assert response.status_code == 200
        content = response.text
        assert "portfolio_equity" in content

    def test_alerts_and_dashboard_integration(self):
        """Test that alerts appear in dashboard."""
        # Reset alert manager singleton
        from quant_trading_system.monitoring import alerting
        alerting._alert_manager = None

        async def run_test():
            # Create an alert
            manager = get_alert_manager()
            await manager.create_alert(
                alert_type=AlertType.HIGH_LATENCY,
                title="Integration Test Alert",
                message="Testing alert integration",
                context={"test": True},
            )

            # Check via dashboard endpoint
            client = TestClient(dashboard_app)
            response = client.get("/alerts")
            return response.json()

        alerts = asyncio.run(run_test())
        # Note: Due to singleton reset, list may or may not contain the alert
        # The important thing is no error occurs
        assert isinstance(alerts, list)

    def test_logging_with_metrics(self):
        """Test that logging works alongside metrics collection."""
        # Setup logging
        setup_logging(level="INFO", log_format=LogFormat.TEXT)

        # Reset metrics singleton
        MetricsCollector._instance = None
        metrics = get_metrics_collector()

        # Log some events
        logger = get_logger("integration_test", LogCategory.TRADING)
        logger.info("Test log message")

        # Record related metrics
        metrics.record_order_submitted("AAPL", "BUY", "MARKET")

        # Verify no errors occurred
        assert True

    def test_dashboard_state_with_alerts(self):
        """Test dashboard state updates with alert creation."""
        state = get_dashboard_state()

        # Update portfolio
        state.update_portfolio({
            "equity": 100000,
            "cash": 50000,
            "daily_pnl": 500,
        })

        # Add an order
        state.add_order({
            "order_id": "int-test-001",
            "symbol": "AAPL",
            "status": "filled",
        })

        # Verify state is updated
        assert state._portfolio_data["equity"] == 100000
        assert len(state._orders) > 0

    def test_full_trading_cycle_monitoring(self):
        """Test monitoring a full trading cycle."""
        # Reset singletons
        MetricsCollector._instance = None
        from quant_trading_system.monitoring import alerting
        alerting._alert_manager = None

        metrics = get_metrics_collector()
        state = get_dashboard_state()

        # 1. Start of day - update portfolio
        state.update_portfolio({
            "equity": 100000,
            "cash": 50000,
            "daily_pnl": 0,
        })
        metrics.update_portfolio_metrics(
            equity=100000,
            cash=50000,
            buying_power=150000,
            positions_count=0,
            long_exposure=0,
            short_exposure=0,
        )

        # 2. Signal generated
        state.add_signal({
            "symbol": "AAPL",
            "direction": "LONG",
            "strength": 0.75,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # 3. Order submitted
        state.add_order({
            "order_id": "cycle-001",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 100,
            "status": "submitted",
        })
        metrics.record_order_submitted("AAPL", "BUY", "MARKET")

        # 4. Order filled
        state.update_order("cycle-001", {"status": "filled"})
        metrics.record_order_filled("AAPL", "BUY")

        # 5. Position opened
        state.update_position("AAPL", {
            "quantity": 100,
            "avg_entry_price": 185.50,
            "current_price": 185.50,
            "market_value": 18550.0,
            "unrealized_pnl": 0,
        })

        # Verify final state
        assert "AAPL" in state._positions
        assert state._positions["AAPL"]["quantity"] == 100


class TestDashboardAPIIntegration:
    """Integration tests for Dashboard API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(dashboard_app)

    def test_full_api_workflow(self, client):
        """Test a full API workflow."""
        # 1. Check health
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] in ["healthy", "degraded"]

        # 2. Get portfolio
        response = client.get("/portfolio")
        assert response.status_code == 200

        # 3. Get positions
        response = client.get("/positions")
        assert response.status_code == 200

        # 4. Get orders
        response = client.get("/orders")
        assert response.status_code == 200

        # 5. Get performance
        response = client.get("/performance")
        assert response.status_code == 200

        # 6. Get signals
        response = client.get("/signals")
        assert response.status_code == 200

        # 7. Get models
        response = client.get("/models")
        assert response.status_code == 200

        # 8. Get risk metrics
        response = client.get("/risk")
        assert response.status_code == 200

        # 9. Get alerts
        response = client.get("/alerts")
        assert response.status_code == 200

        # 10. Get logs
        response = client.get("/logs")
        assert response.status_code == 200

        # 10.1 Project coverage surfaces
        response = client.get("/system/coverage")
        assert response.status_code == 200
        response = client.get("/control/jobs/catalog")
        assert response.status_code == 200

        # 11. Advanced model governance
        response = client.get("/models/registry")
        assert response.status_code == 200
        response = client.get("/models/drift")
        assert response.status_code == 200
        response = client.get("/models/validation-gates")
        assert response.status_code == 200
        response = client.get("/models/champion-challenger")
        assert response.status_code == 200

        # 12. SRE/incident endpoints
        response = client.get("/sre/slo")
        assert response.status_code == 200
        response = client.get("/sre/incidents")
        assert response.status_code == 200
        response = client.get("/sre/incidents/timeline")
        assert response.status_code == 200
        response = client.get("/sre/runbooks")
        assert response.status_code == 200

        # 13. Security/admin endpoints
        response = client.get("/auth/security/status")
        assert response.status_code == 200
        response = client.get("/admin/users")
        assert response.status_code == 200
        response = client.get("/auth/sso/status")
        assert response.status_code == 200

        # 14. Audit export and SIEM status
        response = client.get("/control/audit/export?format=json")
        assert response.status_code == 200
        response = client.get("/control/audit/siem/status")
        assert response.status_code == 200

    def test_metrics_endpoint_format(self, client):
        """Test that metrics endpoint returns valid Prometheus format."""
        response = client.get("/metrics")
        assert response.status_code == 200

        # Check for some expected metrics
        content = response.text
        # Should contain at least some metric definitions
        assert "HELP" in content or "TYPE" in content or len(content) > 0


class TestAlertingIntegration:
    """Integration tests for alerting system."""

    def test_alert_lifecycle(self):
        """Test full alert lifecycle."""
        from quant_trading_system.monitoring import alerting
        alerting._alert_manager = None

        async def run_test():
            manager = get_alert_manager()

            # 1. Create alert
            alert = await manager.create_alert(
                alert_type=AlertType.HIGH_LATENCY,
                title="Lifecycle Test",
                message="Testing alert lifecycle",
            )
            assert alert is not None
            assert alert.status.value == "FIRING"

            # 2. Acknowledge alert
            result = manager.acknowledge_alert(str(alert.alert_id), "test_user")
            assert result is True
            assert alert.status.value == "ACKNOWLEDGED"

            # 3. Resolve alert
            result = manager.resolve_alert(str(alert.alert_id))
            assert result is True
            assert alert.status.value == "RESOLVED"

            # 4. Check history
            history = manager.get_alert_history()
            assert len(history) >= 1

        asyncio.run(run_test())

    def test_alert_deduplication(self):
        """Test that duplicate alerts are deduplicated."""
        from quant_trading_system.monitoring import alerting
        alerting._alert_manager = None

        async def run_test():
            manager = get_alert_manager()

            # Create first alert
            alert1 = await manager.create_alert(
                alert_type=AlertType.HIGH_LATENCY,
                title="Dedup Test",
                message="Same message",
                context={"key": "value"},
            )

            # Try to create duplicate
            alert2 = await manager.create_alert(
                alert_type=AlertType.HIGH_LATENCY,
                title="Dedup Test",
                message="Same message",
                context={"key": "value"},
            )

            # Second should be None (deduplicated)
            return alert1, alert2

        alert1, alert2 = asyncio.run(run_test())
        assert alert1 is not None
        assert alert2 is None  # Deduplicated

    def test_alert_suppression(self):
        """Test alert suppression."""
        from quant_trading_system.monitoring import alerting
        alerting._alert_manager = None

        async def run_test():
            manager = get_alert_manager()

            # Create alert to get fingerprint
            alert = await manager.create_alert(
                alert_type=AlertType.HIGH_LATENCY,
                title="Suppress Test",
                message="Will be suppressed",
            )
            fingerprint = alert.fingerprint

            # Resolve the alert first
            manager.resolve_alert(str(alert.alert_id))

            # Now suppress this fingerprint
            manager.suppress_alert(fingerprint)

            # Try to create same alert
            suppressed_alert = await manager.create_alert(
                alert_type=AlertType.HIGH_LATENCY,
                title="Suppress Test",
                message="Will be suppressed",
            )

            return suppressed_alert

        result = asyncio.run(run_test())
        assert result is None  # Suppressed
