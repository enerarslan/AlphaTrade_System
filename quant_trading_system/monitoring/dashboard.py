"""
FastAPI dashboard for the trading system.

Provides REST API endpoints and WebSocket streams for:
- System health and status
- Portfolio and position data
- Order history and execution
- Performance metrics
- Signal and model status
- Risk monitoring
- Real-time updates
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
import os
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from .alerting import AlertSeverity, AlertStatus, get_alert_manager
from .metrics import get_metrics_collector
from .logger import get_logger, LogCategory


logger = get_logger("dashboard", LogCategory.SYSTEM)

# Create FastAPI app
app = FastAPI(
    title="Quant Trading System Dashboard",
    description="Real-time monitoring and control dashboard for the trading system",
    version="1.0.0",
)

# CORS middleware for frontend access
# SECURITY: Restrict CORS to specific origins instead of "*"
# Get allowed origins from environment variable or use safe defaults
_allowed_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
_allowed_origins = [origin.strip() for origin in _allowed_origins if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Only needed methods
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)


# =============================================================================
# Response Models
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    uptime_seconds: float
    components: dict[str, str]


class PortfolioResponse(BaseModel):
    """Portfolio state response."""

    timestamp: str
    equity: float
    cash: float
    buying_power: float
    positions_count: int
    long_exposure: float
    short_exposure: float
    net_exposure: float
    gross_exposure: float
    daily_pnl: float
    total_pnl: float


class PositionResponse(BaseModel):
    """Position response."""

    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    cost_basis: float


class OrderResponse(BaseModel):
    """Order response."""

    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    filled_qty: float
    limit_price: float | None
    stop_price: float | None
    status: str
    created_at: str
    updated_at: str


class PerformanceResponse(BaseModel):
    """Performance metrics response."""

    daily_pnl: float
    total_pnl: float
    sharpe_ratio_30d: float | None
    sortino_ratio_30d: float | None
    max_drawdown_30d: float | None
    current_drawdown: float | None
    win_rate_30d: float | None
    profit_factor: float | None
    avg_trade_pnl: float | None


class SignalResponse(BaseModel):
    """Signal response."""

    signal_id: str
    timestamp: str
    symbol: str
    direction: str
    strength: float
    confidence: float
    model_source: str


class ModelStatusResponse(BaseModel):
    """Model status response."""

    model_name: str
    status: str
    accuracy: float | None
    auc: float | None
    sharpe: float | None
    last_prediction_time: str | None
    prediction_count: int
    error_count: int


class RiskMetricsResponse(BaseModel):
    """Risk metrics response."""

    portfolio_var_95: float
    portfolio_var_99: float | None
    current_drawdown: float
    max_drawdown_30d: float
    largest_position_pct: float
    sector_exposures: dict[str, float]
    beta_exposure: float | None
    correlation_risk: float | None


class AlertResponse(BaseModel):
    """Alert response."""

    alert_id: str
    alert_type: str
    severity: str
    title: str
    message: str
    status: str
    timestamp: str
    context: dict[str, Any]


class LogEntryResponse(BaseModel):
    """Log entry response."""

    timestamp: str
    level: str
    category: str
    message: str
    extra: dict[str, Any] | None


# =============================================================================
# State Management (placeholder - would be injected in real implementation)
# =============================================================================


class DashboardState:
    """Dashboard state management.

    In a real implementation, this would be connected to the actual
    trading system components via dependency injection.
    """

    def __init__(self) -> None:
        """Initialize dashboard state."""
        self.start_time = datetime.now(timezone.utc)
        self._portfolio_data: dict[str, Any] = {}
        self._positions: dict[str, dict[str, Any]] = {}
        self._orders: list[dict[str, Any]] = []
        self._signals: list[dict[str, Any]] = []
        self._model_status: dict[str, dict[str, Any]] = {}
        self._logs: list[dict[str, Any]] = []
        self._components_healthy: dict[str, bool] = {
            "database": True,
            "redis": True,
            "broker": True,
            "data_feed": True,
            "models": True,
        }

    def get_uptime(self) -> float:
        """Get uptime in seconds."""
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()

    def update_portfolio(self, data: dict[str, Any]) -> None:
        """Update portfolio data."""
        self._portfolio_data = data

    def update_position(self, symbol: str, data: dict[str, Any]) -> None:
        """Update position data."""
        self._positions[symbol] = data

    def remove_position(self, symbol: str) -> None:
        """Remove a position."""
        self._positions.pop(symbol, None)

    def add_order(self, order: dict[str, Any]) -> None:
        """Add an order."""
        self._orders.append(order)
        if len(self._orders) > 1000:
            self._orders = self._orders[-1000:]

    def update_order(self, order_id: str, updates: dict[str, Any]) -> None:
        """Update an order."""
        for order in self._orders:
            if order.get("order_id") == order_id:
                order.update(updates)
                break

    def add_signal(self, signal: dict[str, Any]) -> None:
        """Add a signal."""
        self._signals.append(signal)
        if len(self._signals) > 500:
            self._signals = self._signals[-500:]

    def update_model_status(self, model_name: str, status: dict[str, Any]) -> None:
        """Update model status."""
        self._model_status[model_name] = status

    def add_log(self, log: dict[str, Any]) -> None:
        """Add a log entry."""
        self._logs.append(log)
        if len(self._logs) > 1000:
            self._logs = self._logs[-1000:]

    def set_component_health(self, component: str, healthy: bool) -> None:
        """Set component health status."""
        self._components_healthy[component] = healthy


# Global state instance
_dashboard_state = DashboardState()


def get_dashboard_state() -> DashboardState:
    """Get the dashboard state instance."""
    return _dashboard_state


# =============================================================================
# WebSocket Connection Manager
# =============================================================================


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        """Initialize connection manager."""
        self._active_connections: dict[str, list[WebSocket]] = {
            "portfolio": [],
            "orders": [],
            "signals": [],
            "alerts": [],
        }

    async def connect(self, websocket: WebSocket, channel: str) -> None:
        """Accept and track a WebSocket connection.

        Args:
            websocket: WebSocket connection.
            channel: Channel name.
        """
        await websocket.accept()
        if channel not in self._active_connections:
            self._active_connections[channel] = []
        self._active_connections[channel].append(websocket)
        logger.info(f"WebSocket connected to {channel}")

    def disconnect(self, websocket: WebSocket, channel: str) -> None:
        """Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection.
            channel: Channel name.
        """
        if channel in self._active_connections:
            try:
                self._active_connections[channel].remove(websocket)
            except ValueError:
                pass
        logger.info(f"WebSocket disconnected from {channel}")

    async def broadcast(self, channel: str, message: dict[str, Any]) -> None:
        """Broadcast a message to all connections on a channel.

        Args:
            channel: Channel name.
            message: Message to broadcast.
        """
        if channel not in self._active_connections:
            return

        dead_connections = []
        for connection in self._active_connections[channel]:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)

        # Clean up dead connections
        for conn in dead_connections:
            self.disconnect(conn, channel)

    def get_connection_count(self, channel: str | None = None) -> int:
        """Get number of active connections.

        Args:
            channel: Optional channel to count.

        Returns:
            Number of connections.
        """
        if channel:
            return len(self._active_connections.get(channel, []))
        return sum(len(conns) for conns in self._active_connections.values())


_connection_manager = ConnectionManager()


def get_connection_manager() -> ConnectionManager:
    """Get the connection manager instance."""
    return _connection_manager


# =============================================================================
# REST API Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Get system health status."""
    state = get_dashboard_state()
    components = {k: "healthy" if v else "unhealthy" for k, v in state._components_healthy.items()}

    overall_status = "healthy" if all(state._components_healthy.values()) else "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime_seconds=state.get_uptime(),
        components=components,
    )


@app.get("/metrics")
async def get_metrics() -> Response:
    """Get Prometheus metrics."""
    metrics = get_metrics_collector()
    return Response(
        content=metrics.get_metrics(),
        media_type=metrics.get_content_type(),
    )


@app.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio() -> PortfolioResponse:
    """Get current portfolio state."""
    state = get_dashboard_state()
    data = state._portfolio_data

    return PortfolioResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        equity=data.get("equity", 0.0),
        cash=data.get("cash", 0.0),
        buying_power=data.get("buying_power", 0.0),
        positions_count=len(state._positions),
        long_exposure=data.get("long_exposure", 0.0),
        short_exposure=data.get("short_exposure", 0.0),
        net_exposure=data.get("net_exposure", 0.0),
        gross_exposure=data.get("gross_exposure", 0.0),
        daily_pnl=data.get("daily_pnl", 0.0),
        total_pnl=data.get("total_pnl", 0.0),
    )


@app.get("/positions", response_model=list[PositionResponse])
async def get_positions() -> list[PositionResponse]:
    """Get current positions."""
    state = get_dashboard_state()

    return [
        PositionResponse(
            symbol=symbol,
            quantity=pos.get("quantity", 0),
            avg_entry_price=pos.get("avg_entry_price", 0),
            current_price=pos.get("current_price", 0),
            market_value=pos.get("market_value", 0),
            unrealized_pnl=pos.get("unrealized_pnl", 0),
            unrealized_pnl_pct=pos.get("unrealized_pnl_pct", 0),
            cost_basis=pos.get("cost_basis", 0),
        )
        for symbol, pos in state._positions.items()
    ]


@app.get("/positions/{symbol}", response_model=PositionResponse)
async def get_position(symbol: str) -> PositionResponse:
    """Get a specific position."""
    state = get_dashboard_state()
    pos = state._positions.get(symbol.upper())

    if not pos:
        raise HTTPException(status_code=404, detail=f"Position not found: {symbol}")

    return PositionResponse(
        symbol=symbol.upper(),
        quantity=pos.get("quantity", 0),
        avg_entry_price=pos.get("avg_entry_price", 0),
        current_price=pos.get("current_price", 0),
        market_value=pos.get("market_value", 0),
        unrealized_pnl=pos.get("unrealized_pnl", 0),
        unrealized_pnl_pct=pos.get("unrealized_pnl_pct", 0),
        cost_basis=pos.get("cost_basis", 0),
    )


@app.get("/orders", response_model=list[OrderResponse])
async def get_orders(
    symbol: str | None = Query(None, description="Filter by symbol"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum orders to return"),
) -> list[OrderResponse]:
    """Get order history."""
    state = get_dashboard_state()
    orders = state._orders

    if symbol:
        orders = [o for o in orders if o.get("symbol") == symbol.upper()]
    if status:
        orders = [o for o in orders if o.get("status") == status.upper()]

    orders = orders[-limit:]

    return [
        OrderResponse(
            order_id=o.get("order_id", ""),
            symbol=o.get("symbol", ""),
            side=o.get("side", ""),
            order_type=o.get("order_type", ""),
            quantity=o.get("quantity", 0),
            filled_qty=o.get("filled_qty", 0),
            limit_price=o.get("limit_price"),
            stop_price=o.get("stop_price"),
            status=o.get("status", ""),
            created_at=o.get("created_at", ""),
            updated_at=o.get("updated_at", ""),
        )
        for o in orders
    ]


@app.get("/performance", response_model=PerformanceResponse)
async def get_performance() -> PerformanceResponse:
    """Get performance metrics."""
    state = get_dashboard_state()
    data = state._portfolio_data

    return PerformanceResponse(
        daily_pnl=data.get("daily_pnl", 0.0),
        total_pnl=data.get("total_pnl", 0.0),
        sharpe_ratio_30d=data.get("sharpe_ratio_30d"),
        sortino_ratio_30d=data.get("sortino_ratio_30d"),
        max_drawdown_30d=data.get("max_drawdown_30d"),
        current_drawdown=data.get("current_drawdown"),
        win_rate_30d=data.get("win_rate_30d"),
        profit_factor=data.get("profit_factor"),
        avg_trade_pnl=data.get("avg_trade_pnl"),
    )


@app.get("/signals", response_model=list[SignalResponse])
async def get_signals(
    symbol: str | None = Query(None, description="Filter by symbol"),
    limit: int = Query(100, ge=1, le=500, description="Maximum signals to return"),
) -> list[SignalResponse]:
    """Get latest signals."""
    state = get_dashboard_state()
    signals = state._signals

    if symbol:
        signals = [s for s in signals if s.get("symbol") == symbol.upper()]

    signals = signals[-limit:]

    return [
        SignalResponse(
            signal_id=s.get("signal_id", ""),
            timestamp=s.get("timestamp", ""),
            symbol=s.get("symbol", ""),
            direction=s.get("direction", ""),
            strength=s.get("strength", 0),
            confidence=s.get("confidence", 0),
            model_source=s.get("model_source", ""),
        )
        for s in signals
    ]


@app.get("/models", response_model=list[ModelStatusResponse])
async def get_model_status() -> list[ModelStatusResponse]:
    """Get model status."""
    state = get_dashboard_state()

    return [
        ModelStatusResponse(
            model_name=name,
            status=status.get("status", "unknown"),
            accuracy=status.get("accuracy"),
            auc=status.get("auc"),
            sharpe=status.get("sharpe"),
            last_prediction_time=status.get("last_prediction_time"),
            prediction_count=status.get("prediction_count", 0),
            error_count=status.get("error_count", 0),
        )
        for name, status in state._model_status.items()
    ]


@app.get("/risk", response_model=RiskMetricsResponse)
async def get_risk_metrics() -> RiskMetricsResponse:
    """Get risk metrics."""
    state = get_dashboard_state()
    data = state._portfolio_data

    return RiskMetricsResponse(
        portfolio_var_95=data.get("portfolio_var_95", 0.0),
        portfolio_var_99=data.get("portfolio_var_99"),
        current_drawdown=data.get("current_drawdown", 0.0),
        max_drawdown_30d=data.get("max_drawdown_30d", 0.0),
        largest_position_pct=data.get("largest_position_pct", 0.0),
        sector_exposures=data.get("sector_exposures", {}),
        beta_exposure=data.get("beta_exposure"),
        correlation_risk=data.get("correlation_risk"),
    )


@app.get("/alerts", response_model=list[AlertResponse])
async def get_alerts(
    severity: str | None = Query(None, description="Filter by severity"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=500, description="Maximum alerts to return"),
) -> list[AlertResponse]:
    """Get alerts."""
    manager = get_alert_manager()

    severity_filter = None
    if severity:
        try:
            severity_filter = AlertSeverity(severity.upper())
        except ValueError:
            pass

    alerts = manager.get_dashboard_alerts(limit=limit)

    if severity_filter:
        alerts = [a for a in alerts if a.severity == severity_filter]
    if status:
        try:
            status_filter = AlertStatus(status.upper())
            alerts = [a for a in alerts if a.status == status_filter]
        except ValueError:
            pass

    return [
        AlertResponse(
            alert_id=str(a.alert_id),
            alert_type=a.alert_type.value,
            severity=a.severity.value,
            title=a.title,
            message=a.message,
            status=a.status.value,
            timestamp=a.timestamp.isoformat(),
            context=a.context,
        )
        for a in alerts
    ]


@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, acknowledged_by: str = Query(...)) -> dict[str, str]:
    """Acknowledge an alert."""
    manager = get_alert_manager()

    if manager.acknowledge_alert(alert_id, acknowledged_by):
        return {"status": "acknowledged"}

    raise HTTPException(status_code=404, detail="Alert not found")


@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str) -> dict[str, str]:
    """Resolve an alert."""
    manager = get_alert_manager()

    if manager.resolve_alert(alert_id):
        return {"status": "resolved"}

    raise HTTPException(status_code=404, detail="Alert not found")


@app.get("/logs", response_model=list[LogEntryResponse])
async def get_logs(
    level: str | None = Query(None, description="Filter by level"),
    category: str | None = Query(None, description="Filter by category"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum logs to return"),
) -> list[LogEntryResponse]:
    """Get recent logs."""
    state = get_dashboard_state()
    logs = state._logs

    if level:
        logs = [log for log in logs if log.get("level") == level.upper()]
    if category:
        logs = [log for log in logs if log.get("category") == category.upper()]

    logs = logs[-limit:]

    return [
        LogEntryResponse(
            timestamp=log.get("timestamp", ""),
            level=log.get("level", ""),
            category=log.get("category", ""),
            message=log.get("message", ""),
            extra=log.get("extra"),
        )
        for log in logs
    ]


# =============================================================================
# WebSocket Endpoints
# =============================================================================


@app.websocket("/ws/portfolio")
async def websocket_portfolio(websocket: WebSocket) -> None:
    """WebSocket for real-time portfolio updates."""
    manager = get_connection_manager()
    await manager.connect(websocket, "portfolio")

    try:
        while True:
            # Keep connection alive and wait for messages
            data = await websocket.receive_text()
            # Echo back or handle commands
            await websocket.send_json({"type": "ack", "message": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket, "portfolio")


@app.websocket("/ws/orders")
async def websocket_orders(websocket: WebSocket) -> None:
    """WebSocket for real-time order updates."""
    manager = get_connection_manager()
    await manager.connect(websocket, "orders")

    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "ack", "message": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket, "orders")


@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket) -> None:
    """WebSocket for real-time signal updates."""
    manager = get_connection_manager()
    await manager.connect(websocket, "signals")

    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "ack", "message": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket, "signals")


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket) -> None:
    """WebSocket for real-time alert updates."""
    manager = get_connection_manager()
    await manager.connect(websocket, "alerts")

    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "ack", "message": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket, "alerts")


# =============================================================================
# Broadcast Functions
# =============================================================================


async def broadcast_portfolio_update(data: dict[str, Any]) -> None:
    """Broadcast portfolio update to all connected clients.

    Args:
        data: Portfolio data to broadcast.
    """
    manager = get_connection_manager()
    await manager.broadcast("portfolio", {"type": "portfolio_update", "data": data})


async def broadcast_order_update(order: dict[str, Any]) -> None:
    """Broadcast order update to all connected clients.

    Args:
        order: Order data to broadcast.
    """
    manager = get_connection_manager()
    await manager.broadcast("orders", {"type": "order_update", "data": order})


async def broadcast_signal(signal: dict[str, Any]) -> None:
    """Broadcast signal to all connected clients.

    Args:
        signal: Signal data to broadcast.
    """
    manager = get_connection_manager()
    await manager.broadcast("signals", {"type": "signal", "data": signal})


async def broadcast_alert(alert: dict[str, Any]) -> None:
    """Broadcast alert to all connected clients.

    Args:
        alert: Alert data to broadcast.
    """
    manager = get_connection_manager()
    await manager.broadcast("alerts", {"type": "alert", "data": alert})


# =============================================================================
# Dashboard App Factory
# =============================================================================


def create_dashboard_app() -> FastAPI:
    """Create and configure the dashboard FastAPI application.

    Returns:
        Configured FastAPI application.
    """
    return app


def get_dashboard_app() -> FastAPI:
    """Get the dashboard FastAPI application.

    Returns:
        FastAPI application instance.
    """
    return app
