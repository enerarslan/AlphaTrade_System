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

    def test_risk_endpoint(self, client):
        """Test /risk endpoint."""
        response = client.get("/risk")
        assert response.status_code == 200
        data = response.json()
        assert "portfolio_var_95" in data
        assert "current_drawdown" in data

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
