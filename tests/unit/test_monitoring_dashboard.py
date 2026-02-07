"""
Unit tests for monitoring/dashboard.py
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from quant_trading_system.monitoring.dashboard import (
    app,
    DashboardState,
    ConnectionManager,
    get_dashboard_state,
    get_connection_manager,
    create_dashboard_app,
    get_dashboard_app,
    broadcast_portfolio_update,
    broadcast_order_update,
    broadcast_signal,
    broadcast_alert,
)


class TestDashboardState:
    """Tests for DashboardState class."""

    def test_create_state(self):
        """Test creating dashboard state."""
        state = DashboardState()
        assert state.start_time is not None
        assert state._portfolio_data == {}
        assert state._positions == {}

    def test_get_uptime(self):
        """Test getting uptime."""
        state = DashboardState()
        uptime = state.get_uptime()
        assert uptime >= 0

    def test_update_portfolio(self):
        """Test updating portfolio data."""
        state = DashboardState()
        data = {"equity": 100000, "cash": 50000}
        state.update_portfolio(data)
        assert state._portfolio_data == data

    def test_update_position(self):
        """Test updating position data."""
        state = DashboardState()
        state.update_position("AAPL", {"quantity": 100, "price": 185.0})
        assert "AAPL" in state._positions
        assert state._positions["AAPL"]["quantity"] == 100

    def test_remove_position(self):
        """Test removing a position."""
        state = DashboardState()
        state.update_position("AAPL", {"quantity": 100})
        state.remove_position("AAPL")
        assert "AAPL" not in state._positions

    def test_add_order(self):
        """Test adding an order."""
        state = DashboardState()
        state.add_order({"order_id": "123", "symbol": "AAPL"})
        assert len(state._orders) == 1

    def test_add_order_limit(self):
        """Test order list size limit."""
        state = DashboardState()
        for i in range(1100):
            state.add_order({"order_id": str(i)})
        assert len(state._orders) == 1000

    def test_update_order(self):
        """Test updating an order."""
        state = DashboardState()
        state.add_order({"order_id": "123", "status": "pending"})
        state.update_order("123", {"status": "filled"})
        assert state._orders[0]["status"] == "filled"

    def test_add_signal(self):
        """Test adding a signal."""
        state = DashboardState()
        state.add_signal({"symbol": "AAPL", "strength": 0.8})
        assert len(state._signals) == 1

    def test_update_model_status(self):
        """Test updating model status."""
        state = DashboardState()
        state.update_model_status("xgboost", {"accuracy": 0.65})
        assert "xgboost" in state._model_status

    def test_add_log(self):
        """Test adding a log entry."""
        state = DashboardState()
        state.add_log({"level": "INFO", "message": "Test"})
        assert len(state._logs) == 1

    def test_set_component_health(self):
        """Test setting component health."""
        state = DashboardState()
        state.set_component_health("database", False)
        assert state._components_healthy["database"] is False


class TestConnectionManager:
    """Tests for ConnectionManager class."""

    def test_create_manager(self):
        """Test creating connection manager."""
        manager = ConnectionManager()
        assert "portfolio" in manager._active_connections
        assert "orders" in manager._active_connections

    def test_get_connection_count_empty(self):
        """Test getting connection count when empty."""
        manager = ConnectionManager()
        assert manager.get_connection_count() == 0
        assert manager.get_connection_count("portfolio") == 0


class TestDashboardEndpoints:
    """Tests for dashboard REST endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "components" in data

    def test_metrics_endpoint(self, client):
        """Test /metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"] or "openmetrics" in response.headers["content-type"]

    def test_portfolio_endpoint(self, client):
        """Test /portfolio endpoint."""
        response = client.get("/portfolio")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "equity" in data
        assert "cash" in data

    def test_positions_endpoint(self, client):
        """Test /positions endpoint."""
        response = client.get("/positions")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_positions_symbol_not_found(self, client):
        """Test /positions/{symbol} for non-existent position."""
        response = client.get("/positions/INVALID")
        assert response.status_code == 404

    def test_orders_endpoint(self, client):
        """Test /orders endpoint."""
        response = client.get("/orders")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_orders_with_filters(self, client):
        """Test /orders endpoint with filters."""
        response = client.get("/orders?symbol=AAPL&limit=50")
        assert response.status_code == 200

    def test_performance_endpoint(self, client):
        """Test /performance endpoint."""
        response = client.get("/performance")
        assert response.status_code == 200
        data = response.json()
        assert "daily_pnl" in data
        assert "total_pnl" in data

    def test_signals_endpoint(self, client):
        """Test /signals endpoint."""
        response = client.get("/signals")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_models_endpoint(self, client):
        """Test /models endpoint."""
        response = client.get("/models")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_auth_mfa_status_endpoint(self, client):
        """Test /auth/mfa/status endpoint."""
        response = client.get("/auth/mfa/status")
        assert response.status_code == 200
        data = response.json()
        assert "username" in data
        assert "role" in data
        assert "mfa_enabled" in data

    def test_auth_sso_status_endpoint(self, client):
        """Test /auth/sso/status endpoint."""
        response = client.get("/auth/sso/status")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "configured" in data
        assert "username_claim" in data

    def test_auth_sso_start_disabled_endpoint(self, client):
        """Test /auth/sso/start when SSO is disabled."""
        response = client.get("/auth/sso/start")
        assert response.status_code in (404, 503)

    def test_auth_mfa_enrollment_init_endpoint(self, client):
        """Test /auth/mfa/enroll/init endpoint."""
        response = client.post("/auth/mfa/enroll/init")
        assert response.status_code == 200
        data = response.json()
        assert "secret" in data
        assert "provisioning_uri" in data

    def test_auth_security_status_endpoint(self, client):
        """Test /auth/security/status endpoint."""
        response = client.get("/auth/security/status")
        assert response.status_code == 200
        data = response.json()
        assert "jwt_key_count" in data
        assert "active_key_fingerprint" in data

    def test_auth_security_rotate_jwt_endpoint(self, client):
        """Test /auth/security/rotate-jwt endpoint."""
        response = client.post(
            "/auth/security/rotate-jwt",
            json={"new_secret": "x" * 40},
        )
        assert response.status_code == 200

    def test_admin_users_endpoints(self, client):
        """Test admin users list and role update endpoints."""
        response = client.get("/admin/users")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        response = client.post(
            "/admin/users/test_user/role",
            json={"role": "operator", "reason": "unit_test"},
        )
        assert response.status_code == 200

    def test_risk_endpoint(self, client):
        """Test /risk endpoint."""
        response = client.get("/risk")
        assert response.status_code == 200
        data = response.json()
        assert "portfolio_var_95" in data
        assert "current_drawdown" in data

    def test_execution_quality_endpoint(self, client):
        """Test /execution/quality endpoint."""
        response = client.get("/execution/quality")
        assert response.status_code == 200
        data = response.json()
        assert "arrival_price_delta_bps" in data
        assert "latency_distribution_ms" in data

    def test_advanced_risk_endpoints(self, client):
        """Test advanced risk endpoints."""
        for endpoint in ["/risk/concentration", "/risk/correlation", "/risk/stress", "/risk/attribution"]:
            response = client.get(endpoint)
            assert response.status_code == 200

    def test_control_audit_endpoint(self, client):
        """Test /control/audit endpoint."""
        response = client.get("/control/audit")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_control_audit_export_json_endpoint(self, client):
        """Test /control/audit/export endpoint (json)."""
        response = client.get("/control/audit/export?format=json")
        assert response.status_code == 200
        data = response.json()
        assert "generated_at" in data
        assert "records" in data

    def test_control_audit_export_jsonl_endpoint(self, client):
        """Test /control/audit/export endpoint (jsonl)."""
        response = client.get("/control/audit/export?format=jsonl")
        assert response.status_code == 200
        assert "application/x-ndjson" in response.headers.get("content-type", "")

    def test_control_audit_siem_status_endpoint(self, client):
        """Test /control/audit/siem/status endpoint."""
        response = client.get("/control/audit/siem/status")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "queue_depth" in data

    def test_control_audit_siem_flush_endpoint(self, client):
        """Test /control/audit/siem/flush endpoint."""
        response = client.post("/control/audit/siem/flush")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "flushed"

    def test_control_jobs_catalog_endpoint(self, client):
        """Test /control/jobs/catalog endpoint."""
        response = client.get("/control/jobs/catalog")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert any(item.get("command") == "health" for item in data)
        assert all("script_path" in item for item in data)

    def test_system_coverage_endpoint(self, client):
        """Test /system/coverage endpoint."""
        response = client.get("/system/coverage")
        assert response.status_code == 200
        data = response.json()
        assert "main_entrypoint" in data
        assert "command_catalog" in data
        assert "scripts" in data
        assert "domains" in data
        assert "data_assets" in data

    def test_model_governance_endpoints(self, client):
        """Test model governance endpoints."""
        response = client.get("/models/registry")
        assert response.status_code == 200
        response = client.get("/models/drift")
        assert response.status_code == 200
        response = client.get("/models/validation-gates")
        assert response.status_code == 200
        response = client.get("/models/champion-challenger")
        assert response.status_code == 200

    def test_model_promote_endpoint_not_found_without_registry(self, client):
        """Promotion should fail cleanly when requested version does not exist."""
        response = client.post(
            "/models/champion/promote",
            json={
                "model_name": "nonexistent",
                "version_id": "v0",
                "reason": "unit_test",
            },
        )
        assert response.status_code in (404, 422)

    def test_sre_endpoints(self, client):
        """Test SRE endpoints."""
        response = client.get("/sre/slo")
        assert response.status_code == 200
        response = client.get("/sre/incidents")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        response = client.get("/sre/incidents/timeline")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        response = client.get("/sre/runbooks")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_sre_runbook_execute_endpoint(self, client):
        """Test runbook execution endpoint."""
        response = client.post("/sre/runbooks/execute", json={"action": "health_check"})
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data

    def test_alerts_endpoint(self, client):
        """Test /alerts endpoint."""
        response = client.get("/alerts")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_alerts_with_filters(self, client):
        """Test /alerts endpoint with filters."""
        response = client.get("/alerts?severity=WARNING&limit=50")
        assert response.status_code == 200

    def test_acknowledge_alert_not_found(self, client):
        """Test acknowledging non-existent alert."""
        response = client.post(
            "/alerts/nonexistent/acknowledge?acknowledged_by=admin"
        )
        assert response.status_code == 404

    def test_resolve_alert_not_found(self, client):
        """Test resolving non-existent alert."""
        response = client.post("/alerts/nonexistent/resolve")
        assert response.status_code == 404

    def test_logs_endpoint(self, client):
        """Test /logs endpoint."""
        response = client.get("/logs")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_logs_with_filters(self, client):
        """Test /logs endpoint with filters."""
        response = client.get("/logs?level=ERROR&limit=50")
        assert response.status_code == 200


class TestDashboardFactoryFunctions:
    """Tests for dashboard factory functions."""

    def test_create_dashboard_app(self):
        """Test create_dashboard_app function."""
        dashboard = create_dashboard_app()
        assert dashboard is app

    def test_get_dashboard_app(self):
        """Test get_dashboard_app function."""
        dashboard = get_dashboard_app()
        assert dashboard is app

    def test_get_dashboard_state(self):
        """Test get_dashboard_state function."""
        state = get_dashboard_state()
        assert isinstance(state, DashboardState)

    def test_get_connection_manager(self):
        """Test get_connection_manager function."""
        manager = get_connection_manager()
        assert isinstance(manager, ConnectionManager)


class TestBroadcastFunctions:
    """Tests for broadcast functions."""

    def test_broadcast_portfolio_update(self):
        """Test broadcast_portfolio_update function."""

        async def run_test():
            # This won't actually broadcast (no connections)
            # but should not raise an error
            await broadcast_portfolio_update({"equity": 100000})

        asyncio.run(run_test())

    def test_broadcast_order_update(self):
        """Test broadcast_order_update function."""

        async def run_test():
            await broadcast_order_update({"order_id": "123", "status": "filled"})

        asyncio.run(run_test())

    def test_broadcast_signal(self):
        """Test broadcast_signal function."""

        async def run_test():
            await broadcast_signal({"symbol": "AAPL", "direction": "LONG"})

        asyncio.run(run_test())

    def test_broadcast_alert(self):
        """Test broadcast_alert function."""

        async def run_test():
            await broadcast_alert({"title": "Test Alert", "severity": "WARNING"})

        asyncio.run(run_test())
