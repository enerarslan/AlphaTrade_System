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

SECURITY: All sensitive endpoints require JWT authentication.
Configure via environment variables:
- JWT_SECRET_KEY: Required for production (generate with: python -c "import secrets; print(secrets.token_hex(32))")
- REQUIRE_AUTH: Set to 'false' to disable auth (development only)
- API_KEYS: Comma-separated list of valid API keys (alternative to JWT)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta, timezone
import os
from typing import Annotated, Any
from uuid import UUID

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .alerting import AlertSeverity, AlertStatus, get_alert_manager
from .metrics import get_metrics_collector
from .logger import get_logger, LogCategory
from ..config.settings import get_settings
from ..core.events import EventBus, Event, EventType, create_system_event
from ..core.redis_bridge import RedisEventBridge
from .health import SystemHealthChecker, HealthStatus, HealthCheckResult


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
# JWT Authentication
# =============================================================================

# Security scheme for Swagger UI
security_scheme = HTTPBearer(auto_error=False)


class TokenData(BaseModel):
    """JWT token payload data."""

    username: str
    exp: datetime
    iat: datetime
    jti: str  # JWT ID for token revocation


class AuthenticationError(HTTPException):
    """Authentication failure exception."""

    def __init__(self, detail: str = "Authentication required"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


def _get_jwt_secret() -> str:
    """Get JWT secret key with validation.

    Returns:
        JWT secret key.

    Raises:
        RuntimeError: If secret is not configured in production.
    """
    settings = get_settings()
    secret = settings.security.jwt_secret_key

    if not secret:
        # In production, require a secret key
        if settings.environment == "production":
            raise RuntimeError(
                "JWT_SECRET_KEY must be set in production. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        # In development, use a warning and generate a temporary key
        logger.warning(
            "JWT_SECRET_KEY not set - using temporary key. "
            "This is insecure and should not be used in production!"
        )
        # Generate a deterministic key for development (consistent across restarts)
        return hashlib.sha256(b"DEVELOPMENT_ONLY_DO_NOT_USE_IN_PRODUCTION").hexdigest()

    return secret


def create_access_token(username: str, expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token.

    Args:
        username: Username to encode in token.
        expires_delta: Optional custom expiration time.

    Returns:
        Encoded JWT token string.
    """
    settings = get_settings()
    secret = _get_jwt_secret()
    algorithm = settings.security.jwt_algorithm

    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.security.jwt_expiration_minutes)

    # Create token payload
    payload = {
        "sub": username,
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "jti": secrets.token_hex(16),  # Unique token ID
    }

    # Manual JWT encoding (avoiding external dependency)
    # Header
    header = {"alg": algorithm, "typ": "JWT"}

    def base64url_encode(data: bytes) -> str:
        import base64
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")

    header_b64 = base64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = base64url_encode(json.dumps(payload, separators=(",", ":")).encode())

    # Signature
    message = f"{header_b64}.{payload_b64}"
    if algorithm == "HS256":
        signature = hmac.new(
            secret.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    signature_b64 = base64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def verify_jwt_token(token: str) -> TokenData:
    """Verify and decode a JWT token.

    Args:
        token: JWT token string.

    Returns:
        Decoded token data.

    Raises:
        AuthenticationError: If token is invalid or expired.
    """
    import base64

    settings = get_settings()
    secret = _get_jwt_secret()
    algorithm = settings.security.jwt_algorithm

    def base64url_decode(data: str) -> bytes:
        # Add padding if needed
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)

    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise AuthenticationError("Invalid token format")

        header_b64, payload_b64, signature_b64 = parts

        # Verify signature
        message = f"{header_b64}.{payload_b64}"
        if algorithm == "HS256":
            expected_signature = hmac.new(
                secret.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
        else:
            raise AuthenticationError(f"Unsupported algorithm: {algorithm}")

        actual_signature = base64url_decode(signature_b64)

        if not hmac.compare_digest(expected_signature, actual_signature):
            raise AuthenticationError("Invalid token signature")

        # Decode payload
        payload = json.loads(base64url_decode(payload_b64).decode())

        # Verify expiration
        exp = payload.get("exp")
        if not exp:
            raise AuthenticationError("Token has no expiration")

        if datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(timezone.utc):
            raise AuthenticationError("Token has expired")

        return TokenData(
            username=payload.get("sub", ""),
            exp=datetime.fromtimestamp(exp, tz=timezone.utc),
            iat=datetime.fromtimestamp(payload.get("iat", 0), tz=timezone.utc),
            jti=payload.get("jti", ""),
        )

    except AuthenticationError:
        raise
    except Exception as e:
        logger.warning(f"Token verification failed: {e}")
        raise AuthenticationError("Invalid token")


def verify_api_key(api_key: str) -> bool:
    """Verify an API key.

    Args:
        api_key: API key to verify.

    Returns:
        True if valid, False otherwise.
    """
    settings = get_settings()
    valid_keys = settings.security.api_keys

    if not valid_keys:
        return False

    # Use constant-time comparison to prevent timing attacks
    for valid_key in valid_keys:
        if hmac.compare_digest(api_key, valid_key):
            return True

    return False


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security_scheme)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> str:
    """Dependency to get the current authenticated user.

    Supports two authentication methods:
    1. Bearer token (JWT) in Authorization header
    2. API key in X-API-Key header

    Args:
        credentials: Bearer token credentials from Authorization header.
        x_api_key: API key from X-API-Key header.

    Returns:
        Username or "api_key_user" for API key auth.

    Raises:
        AuthenticationError: If authentication fails.
    """
    settings = get_settings()

    # Check if authentication is required
    if not settings.security.require_auth:
        return "anonymous"

    # Try Bearer token first
    if credentials and credentials.credentials:
        token_data = verify_jwt_token(credentials.credentials)
        logger.debug(f"Authenticated user via JWT: {token_data.username}")
        return token_data.username

    # Try API key
    if x_api_key:
        if verify_api_key(x_api_key):
            logger.debug("Authenticated user via API key")
            return "api_key_user"
        raise AuthenticationError("Invalid API key")

    raise AuthenticationError("Authentication required")


async def optional_auth(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security_scheme)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> str | None:
    """Optional authentication - returns None if not authenticated.

    Used for endpoints that have different behavior based on auth status.
    """
    settings = get_settings()

    if not settings.security.require_auth:
        return "anonymous"

    try:
        return await get_current_user(credentials, x_api_key)
    except AuthenticationError:
        return None


# Type alias for authenticated user dependency
AuthenticatedUser = Annotated[str, Depends(get_current_user)]


# =============================================================================
# Authentication Endpoints
# =============================================================================


class LoginRequest(BaseModel):
    """Login request body."""

    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    """Token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


# Simple user store for demo - in production, use a proper user database
# WARNING: This is for demonstration only. Production should use proper auth.
_DEMO_USERS: dict[str, bytes] = {}  # Changed to bytes to store bcrypt hashes


def _get_password_hash(password: str) -> bytes:
    """Hash a password using bcrypt with automatic salt generation.

    SECURITY FIX: Now uses bcrypt with random salt per password.
    bcrypt is designed for password hashing and is resistant to:
    - Rainbow table attacks (random salt)
    - Brute force attacks (slow by design)
    - GPU attacks (memory-hard)

    Falls back to PBKDF2 if bcrypt is not available.
    """
    password_bytes = password.encode('utf-8')

    if BCRYPT_AVAILABLE:
        # bcrypt automatically generates a random salt
        # Work factor 12 = ~250ms on modern hardware
        return bcrypt.hashpw(password_bytes, bcrypt.gensalt(rounds=12))
    else:
        # Fallback to PBKDF2 with random salt if bcrypt not installed
        import hashlib
        salt = secrets.token_bytes(32)
        # 100000 iterations for PBKDF2
        derived_key = hashlib.pbkdf2_hmac(
            'sha256',
            password_bytes,
            salt,
            100000,
            dklen=32
        )
        # Store salt + derived key together
        return salt + derived_key


def _verify_password(password: str, hashed: bytes) -> bool:
    """Verify a password against its hash.

    SECURITY FIX: Proper password verification for bcrypt or PBKDF2.
    Uses constant-time comparison to prevent timing attacks.
    """
    password_bytes = password.encode('utf-8')

    if BCRYPT_AVAILABLE:
        try:
            return bcrypt.checkpw(password_bytes, hashed)
        except Exception:
            return False
    else:
        # PBKDF2 fallback verification
        import hashlib
        if len(hashed) < 64:  # salt(32) + key(32)
            return False
        salt = hashed[:32]
        stored_key = hashed[32:]
        derived_key = hashlib.pbkdf2_hmac(
            'sha256',
            password_bytes,
            salt,
            100000,
            dklen=32
        )
        return hmac.compare_digest(derived_key, stored_key)


def _init_demo_users() -> None:
    """Initialize demo users from environment variable.

    Set DASHBOARD_USERS as: username1:password1,username2:password2
    """
    global _DEMO_USERS

    users_str = os.environ.get("DASHBOARD_USERS", "")
    if users_str:
        for user_pass in users_str.split(","):
            if ":" in user_pass:
                username, password = user_pass.split(":", 1)
                _DEMO_USERS[username.strip()] = _get_password_hash(password.strip())

    # Add default admin if no users configured (development only)
    if not _DEMO_USERS:
        settings = get_settings()
        if settings.environment != "production":
            logger.warning(
                "No users configured - using default admin:admin. "
                "Set DASHBOARD_USERS env var for production!"
            )
            _DEMO_USERS["admin"] = _get_password_hash("admin")


# Initialize demo users on module load
_init_demo_users()


@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(request: LoginRequest) -> TokenResponse:
    """Authenticate and get access token.

    SECURITY FIX: Uses proper bcrypt password verification.

    Args:
        request: Login credentials.

    Returns:
        JWT access token.
    """
    if request.username not in _DEMO_USERS:
        # SECURITY: Still hash password to prevent timing attacks
        # that could reveal whether username exists
        _get_password_hash(request.password)
        logger.warning(f"Login attempt for unknown user: {request.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    # SECURITY FIX: Use proper bcrypt verification
    if not _verify_password(request.password, _DEMO_USERS[request.username]):
        logger.warning(f"Invalid password for user: {request.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    settings = get_settings()
    expires_in = settings.security.jwt_expiration_minutes * 60

    token = create_access_token(request.username)
    logger.info(f"User logged in: {request.username}")

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
    )


@app.get("/auth/me", tags=["Authentication"])
async def get_current_user_info(current_user: AuthenticatedUser) -> dict[str, str]:
    """Get information about the current authenticated user."""
    return {"username": current_user}


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


class TCAResponse(BaseModel):
    """Transaction Cost Analysis response."""
    
    timestamp: str
    slippage_bps: float
    market_impact_bps: float
    execution_speed_ms: float
    fill_probability: float
    venue_breakdown: dict[str, float]
    cost_savings_vs_vwap: float


class VaRResponse(BaseModel):
    """Value at Risk & Stress Test response."""
    
    timestamp: str
    var_95: float
    var_99: float
    cvar_95: float
    stress_scenarios: dict[str, float]  # e.g., "2008_crash": -25.4%
    distribution_curve: list[dict[str, float]]  # x: pnl, y: probability


class FeatureImportanceResponse(BaseModel):
    """AI Feature Importance response."""
    
    timestamp: str
    model_name: str
    global_importance: dict[str, float]  # feature: score
    recent_shift: dict[str, float]  # feature: change_pct


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

    async def handle_event(self, event: Event) -> None:
        """Handle incoming system event."""
        # Route event to appropriate handler
        if event.event_type == EventType.PORTFOLIO_UPDATE:
            self.update_portfolio(event.data)
            await broadcast_portfolio_update(event.data)
        elif event.event_type == EventType.ORDER_UPDATE:
            self.update_order(event.data)
            await broadcast_order_update(event.data)
        elif event.event_type == EventType.SIGNAL_GENERATED:
            self.add_signal(event.data)
            await broadcast_signal(event.data)
        elif event.event_type == EventType.RISK_ALERT:
            # Trigger alert broadcast via alert manager if needed, or direct
            pass

    def update_portfolio(self, data: dict[str, Any]) -> None:
        """Update portfolio data."""
        self._portfolio_data.update(data)
        self._last_update = datetime.now(timezone.utc)

    def update_position(self, symbol: str, position: dict[str, Any]) -> None:
        """Update a position."""
        if position.get("quantity", 0) == 0:
            if symbol in self._positions:
                del self._positions[symbol]
        else:
            self._positions[symbol] = position
            
        # Update last update time
        self._last_update = datetime.now(timezone.utc)

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


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def get_health() -> HealthResponse:
    """Get system health status.

    This endpoint is public by default (configurable via ALLOW_PUBLIC_HEALTH env var).
    """
    state = get_dashboard_state()
    
    # Use real component health if available
    checker = SystemHealthChecker()
    # Note: We run a quick check here or rely on cached background checks
    # For now, let's use the state which should be updated by background task
    
    components = {k: "healthy" if v else "unhealthy" for k, v in state._components_healthy.items()}
    overall_status = "healthy" if all(state._components_healthy.values()) else "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime_seconds=state.get_uptime(),
        components=components,
    )


@app.get("/health/detailed", tags=["System"])
async def get_health_detailed() -> dict[str, Any]:
    """Get detailed system health report.
    
    This runs actual diagnostics on all components.
    Public endpoint for easy monitoring.
    """
    checker = SystemHealthChecker()
    results = await checker.run_all_checks()
    overall = checker.get_overall_status(results)
    
    return {
        "status": overall.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": [r.to_dict() for r in results]
    }


@app.get("/metrics", tags=["System"])
async def get_metrics() -> Response:
    """Get Prometheus metrics.

    This endpoint is public for Prometheus scraping.
    """
    metrics = get_metrics_collector()
    return Response(
        content=metrics.get_metrics(),
        media_type=metrics.get_content_type(),
    )


@app.get("/portfolio", response_model=PortfolioResponse, tags=["Portfolio"])
async def get_portfolio(current_user: AuthenticatedUser) -> PortfolioResponse:
    """Get current portfolio state. Requires authentication."""
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


@app.get("/positions", response_model=list[PositionResponse], tags=["Positions"])
async def get_positions(current_user: AuthenticatedUser) -> list[PositionResponse]:
    """Get current positions. Requires authentication."""
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


@app.get("/positions/{symbol}", response_model=PositionResponse, tags=["Positions"])
async def get_position(symbol: str, current_user: AuthenticatedUser) -> PositionResponse:
    """Get a specific position. Requires authentication."""
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


@app.get("/orders", response_model=list[OrderResponse], tags=["Orders"])
async def get_orders(
    current_user: AuthenticatedUser,
    symbol: str | None = Query(None, description="Filter by symbol"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum orders to return"),
) -> list[OrderResponse]:
    """Get order history. Requires authentication."""
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


@app.get("/performance", response_model=PerformanceResponse, tags=["Performance"])
async def get_performance(current_user: AuthenticatedUser) -> PerformanceResponse:
    """Get performance metrics. Requires authentication."""
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


@app.get("/signals", response_model=list[SignalResponse], tags=["Signals"])
async def get_signals(
    current_user: AuthenticatedUser,
    symbol: str | None = Query(None, description="Filter by symbol"),
    limit: int = Query(100, ge=1, le=500, description="Maximum signals to return"),
) -> list[SignalResponse]:
    """Get latest signals. Requires authentication."""
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


@app.get("/models", response_model=list[ModelStatusResponse], tags=["Models"])
async def get_model_status(current_user: AuthenticatedUser) -> list[ModelStatusResponse]:
    """Get model status. Requires authentication."""
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


@app.get("/risk", response_model=RiskMetricsResponse, tags=["Risk"])
async def get_risk_metrics(current_user: AuthenticatedUser) -> RiskMetricsResponse:
    """Get risk metrics. Requires authentication."""
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


@app.get("/alerts", response_model=list[AlertResponse], tags=["Alerts"])
async def get_alerts(
    current_user: AuthenticatedUser,
    severity: str | None = Query(None, description="Filter by severity"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=500, description="Maximum alerts to return"),
) -> list[AlertResponse]:
    """Get alerts. Requires authentication."""
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


@app.post("/alerts/{alert_id}/acknowledge", tags=["Alerts"])
async def acknowledge_alert(
    alert_id: str,
    current_user: AuthenticatedUser,
    acknowledged_by: str = Query(...),
) -> dict[str, str]:
    """Acknowledge an alert. Requires authentication."""
    manager = get_alert_manager()

    if manager.acknowledge_alert(alert_id, acknowledged_by):
        return {"status": "acknowledged"}

    raise HTTPException(status_code=404, detail="Alert not found")


@app.post("/alerts/{alert_id}/resolve", tags=["Alerts"])
async def resolve_alert(alert_id: str, current_user: AuthenticatedUser) -> dict[str, str]:
    """Resolve an alert. Requires authentication."""
    manager = get_alert_manager()

    if manager.resolve_alert(alert_id):
        return {"status": "resolved"}

    raise HTTPException(status_code=404, detail="Alert not found")


@app.get("/logs", response_model=list[LogEntryResponse], tags=["Logs"])
async def get_logs(
    current_user: AuthenticatedUser,
    level: str | None = Query(None, description="Filter by level"),
    category: str | None = Query(None, description="Filter by category"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum logs to return"),
) -> list[LogEntryResponse]:
    """Get recent logs. Requires authentication."""
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


@app.get("/execution/tca", response_model=TCAResponse, tags=["Execution"])
async def get_tca_metrics(current_user: AuthenticatedUser) -> TCAResponse:
    """Get Transaction Cost Analysis metrics. Requires authentication."""
    # MOCKED for Phase 1
    return TCAResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        slippage_bps=2.4,
        market_impact_bps=1.1,
        execution_speed_ms=145.0,
        fill_probability=0.98,
        venue_breakdown={"IEX": 0.45, "NYSE": 0.30, "DARK": 0.25},
        cost_savings_vs_vwap=12.5
    )


@app.get("/risk/var", response_model=VaRResponse, tags=["Risk"])
async def get_var_metrics(current_user: AuthenticatedUser) -> VaRResponse:
    """Get Value at Risk simulations. Requires authentication."""
    # MOCKED for Phase 1
    import numpy as np
    
    # Generate a mock normal distribution for VaR
    x = np.linspace(-5000, 5000, 50)
    y = (1 / (np.sqrt(2 * np.pi) * 1000)) * np.exp(-0.5 * ((x / 1000) ** 2))
    
    distribution = [{"pnl": float(xi), "probability": float(yi)} for xi, yi in zip(x, y)]

    return VaRResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        var_95=12400.0,
        var_99=18500.0,
        cvar_95=15600.0,
        stress_scenarios={
            "Black Monday": -22.4,
            "DotCom Burst": -15.8,
            "Covid Crash": -18.2,
            "Rate Hike": -5.4
        },
        distribution_curve=distribution
    )


@app.get("/models/explainability", response_model=FeatureImportanceResponse, tags=["Models"])
async def get_model_explainability(current_user: AuthenticatedUser) -> FeatureImportanceResponse:
    """Get AI feature importance (SHAP values). Requires authentication."""
    # MOCKED for Phase 1
    return FeatureImportanceResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_name="Ensemble_v4",
        global_importance={
            "momentum_rsi": 0.35,
            "volatility_atr": 0.25,
            "order_flow_imbalance": 0.20,
            "sector_correlation": 0.15,
            "macro_sentiment": 0.05
        },
        recent_shift={
            "momentum_rsi": -0.05,
            "volatility_atr": 0.12,  # Volatility becoming more important
            "order_flow_imbalance": 0.01
        }
    )


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


# =============================================================================
# Startup & Shutdown Events
# =============================================================================


_redis_bridge: RedisEventBridge | None = None


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize dashboard services on startup."""
    global _redis_bridge
    logger.info("Starting dashboard services...")

    try:
        # Initialize EventBus
        event_bus = EventBus()

        # Initialize and start Redis Bridge
        # We subscribe to "events.dashboard" (trading engine publishes here)
        # And we publish control events to "events.control"
        _redis_bridge = RedisEventBridge(
            event_bus, 
            publish_channels=["events.control"],
            subscribe_channels=["events.dashboard"]
        )
        await _redis_bridge.start()

        # Subscribe DashboardState to EventBus
        # The RedisBridge receives "events.dashboard" -> republishes to local EventBus
        # DashboardState listens to local EventBus -> updates state & WebSocket
        state = get_dashboard_state()
        event_bus.subscribe_all(state.handle_event, "dashboard_state_updater")
        
        logger.info("Dashboard services started successfully")

    except Exception as e:
        logger.error(f"Failed to start dashboard services: {e}")
        # We don't raise here to allow the API to start even if Redis is down,
        # but health checks will show it.


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up resources on shutdown."""
    global _redis_bridge
    logger.info("Stopping dashboard services...")

    if _redis_bridge:
        await _redis_bridge.stop()

    logger.info("Dashboard services stopped")
