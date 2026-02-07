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
import base64
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import ssl
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any
from urllib.parse import urlencode, quote, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from uuid import UUID, uuid4

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

try:
    import redis as redis_sync
except ImportError:
    redis_sync = None

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .alerting import (
    ALERT_RUNBOOK_MAP,
    ALERT_SUGGESTED_ACTIONS,
    AlertSeverity,
    AlertStatus,
    AlertType,
    get_alert_manager,
)
from .metrics import get_metrics_collector
from .logger import get_logger, LogCategory
from ..config.settings import get_settings
from ..core.events import EventBus, Event, EventType, create_system_event
from ..core.redis_bridge import RedisEventBridge
from .health import SystemHealthChecker, HealthStatus, HealthCheckResult


logger = get_logger("dashboard", LogCategory.SYSTEM)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_REGISTRY_ROOT = PROJECT_ROOT / "models" / "registry"
MODEL_REGISTRY_FILE = MODEL_REGISTRY_ROOT / "registry.json"
MODEL_ACTIVE_POINTER_FILE = PROJECT_ROOT / "models" / "active_model.json"
SECURITY_RUNTIME_STATE_FILE = PROJECT_ROOT / "logs" / "dashboard_security_state.json"
AUDIT_EXPORT_ROOT = PROJECT_ROOT / "logs" / "audit_exports"

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
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Request-ID",
        "X-API-Key",
        "Idempotency-Key",
        "X-MFA-Code",
    ],
)

_rate_limit_lock = threading.RLock()
_rate_limit_records: dict[str, list[float]] = {}


def _rate_limit_enabled() -> bool:
    """Resolve whether HTTP rate limiting is enabled."""
    return os.environ.get("DASHBOARD_RATE_LIMIT_ENABLED", "true").strip().lower() == "true"


def _rate_limit_window_seconds() -> int:
    """Resolve rate limit window size."""
    try:
        return max(1, int(os.environ.get("DASHBOARD_RATE_LIMIT_WINDOW_SECONDS", "60")))
    except Exception:
        return 60


def _rate_limit_limit_per_window() -> int:
    """Resolve request limit per window."""
    try:
        return max(5, int(os.environ.get("DASHBOARD_RATE_LIMIT_PER_WINDOW", "180")))
    except Exception:
        return 180


def _rate_limit_is_bypassed(path: str) -> bool:
    """Check if a path bypasses rate limiting."""
    default_bypass = "/health,/metrics,/ws,/openapi.json,/docs,/redoc"
    configured = os.environ.get("DASHBOARD_RATE_LIMIT_BYPASS_PATHS", default_bypass)
    prefixes = [x.strip() for x in configured.split(",") if x.strip()]
    return any(path.startswith(prefix) for prefix in prefixes)


def _rate_limit_identity(request: Request) -> str:
    """Build stable identity key for rate limiting."""
    client_host = request.client.host if request.client else "unknown"
    auth_hint = request.headers.get("Authorization") or request.headers.get("X-API-Key") or ""
    if not auth_hint:
        return client_host
    digest = hashlib.sha256(auth_hint.encode("utf-8")).hexdigest()[:16]
    return f"{client_host}:{digest}"


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply coarse-grained HTTP rate limiting for abuse protection."""
    if (not _rate_limit_enabled()) or _rate_limit_is_bypassed(request.url.path):
        return await call_next(request)

    window_seconds = _rate_limit_window_seconds()
    limit = _rate_limit_limit_per_window()
    identity = _rate_limit_identity(request)
    now = time.time()
    key = f"{identity}:{request.url.path}"

    with _rate_limit_lock:
        timestamps = _rate_limit_records.get(key, [])
        cutoff = now - window_seconds
        timestamps = [ts for ts in timestamps if ts >= cutoff]
        if len(timestamps) >= limit:
            retry_after = max(1, int(window_seconds - (now - min(timestamps))))
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "limit": limit,
                    "window_seconds": window_seconds,
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )
        timestamps.append(now)
        _rate_limit_records[key] = timestamps

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Window-Seconds"] = str(window_seconds)
    return response


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


_security_runtime_lock = threading.RLock()
_security_runtime_enabled = "PYTEST_CURRENT_TEST" not in os.environ
_role_overrides: dict[str, str] = {}
_dynamic_mfa_secrets: dict[str, str] = {}
_pending_mfa_enrollments: dict[str, dict[str, Any]] = {}
_jwt_runtime_key_ring: list[str] = []


def _load_security_runtime_state() -> None:
    """Load runtime security state from disk when enabled."""
    global _role_overrides, _dynamic_mfa_secrets, _jwt_runtime_key_ring
    if not _security_runtime_enabled or not SECURITY_RUNTIME_STATE_FILE.exists():
        return
    try:
        with SECURITY_RUNTIME_STATE_FILE.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return
        role_overrides = payload.get("role_overrides", {})
        mfa = payload.get("dynamic_mfa_secrets", {})
        key_ring = payload.get("jwt_key_ring", [])
        if isinstance(role_overrides, dict):
            _role_overrides = {str(k): str(v).lower() for k, v in role_overrides.items()}
        if isinstance(mfa, dict):
            _dynamic_mfa_secrets = {str(k): str(v) for k, v in mfa.items()}
        if isinstance(key_ring, list):
            _jwt_runtime_key_ring = [str(x) for x in key_ring if isinstance(x, str) and x]
    except Exception as exc:
        logger.warning(f"Failed to load dashboard security runtime state: {exc}")


def _save_security_runtime_state() -> None:
    """Persist runtime security state."""
    if not _security_runtime_enabled:
        return
    SECURITY_RUNTIME_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "role_overrides": _role_overrides,
        "dynamic_mfa_secrets": _dynamic_mfa_secrets,
        "jwt_key_ring": _jwt_runtime_key_ring[:5],
    }
    tmp_file = SECURITY_RUNTIME_STATE_FILE.with_suffix(".tmp")
    try:
        with tmp_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        tmp_file.replace(SECURITY_RUNTIME_STATE_FILE)
    except Exception as exc:
        logger.warning(f"Failed to persist dashboard security runtime state: {exc}")
        try:
            tmp_file.unlink(missing_ok=True)
        except Exception:
            pass


_load_security_runtime_state()


def _get_jwt_signing_secrets() -> list[str]:
    """Resolve JWT key ring for signing and verification."""
    with _security_runtime_lock:
        if _jwt_runtime_key_ring:
            return [x for x in _jwt_runtime_key_ring if x]
    configured_ring = [x.strip() for x in os.environ.get("JWT_SECRET_KEYS", "").split(",") if x.strip()]
    if configured_ring:
        return configured_ring
    return [_get_jwt_secret()]


def create_access_token(username: str, expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token.

    Args:
        username: Username to encode in token.
        expires_delta: Optional custom expiration time.

    Returns:
        Encoded JWT token string.
    """
    settings = get_settings()
    signing_secrets = _get_jwt_signing_secrets()
    secret = signing_secrets[0]
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
        "kid": hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12],
    }

    # Manual JWT encoding (avoiding external dependency)
    # Header
    header = {"alg": algorithm, "typ": "JWT", "kid": payload["kid"]}

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
    secrets_ring = _get_jwt_signing_secrets()
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
            expected_signature = None
            for secret in secrets_ring:
                candidate = hmac.new(
                    secret.encode(),
                    message.encode(),
                    hashlib.sha256
                ).digest()
                if hmac.compare_digest(candidate, actual_signature := base64url_decode(signature_b64)):
                    expected_signature = candidate
                    break
            if expected_signature is None:
                raise AuthenticationError("Invalid token signature")
        else:
            raise AuthenticationError(f"Unsupported algorithm: {algorithm}")

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


class MFAStatusResponse(BaseModel):
    """MFA status for authenticated user."""

    username: str
    role: str
    mfa_enabled: bool
    mfa_required_for_privileged_actions: bool = True


class MFAEnrollInitResponse(BaseModel):
    """MFA enrollment initialization response."""

    username: str
    secret: str
    provisioning_uri: str
    issuer: str


class MFAEnrollVerifyRequest(BaseModel):
    """Request to verify and activate enrolled MFA secret."""

    code: str = Field(..., min_length=6, max_length=10)


class MFADisableRequest(BaseModel):
    """Request to disable MFA for current user."""

    code: str = Field(..., min_length=6, max_length=10)


class SecurityStatusResponse(BaseModel):
    """Authentication security configuration status."""

    jwt_key_count: int
    active_key_fingerprint: str
    require_auth: bool
    api_key_enabled: bool
    mfa_enabled_users: int
    role_overrides_count: int
    rate_limit_enabled: bool
    rate_limit_limit: int
    rate_limit_window_seconds: int


class SSOStatusResponse(BaseModel):
    """OIDC/SSO runtime configuration status."""

    enabled: bool
    configured: bool
    issuer: str | None
    authorization_endpoint: str | None
    token_endpoint: str | None
    userinfo_endpoint: str | None
    redirect_uri: str | None
    username_claim: str
    role_claim: str
    scopes: list[str]


class SSOStartResponse(BaseModel):
    """SSO authentication start response."""

    auth_url: str
    state: str
    expires_in_seconds: int


class JwtRotateRequest(BaseModel):
    """Request to rotate active JWT signing key."""

    new_secret: str = Field(..., min_length=32)
    mfa_code: str | None = None


class AdminUserRoleUpdateRequest(BaseModel):
    """Request to update a user's role."""

    role: str = Field(..., pattern="^(viewer|operator|risk|admin)$")
    reason: str = Field(default="admin_update")
    mfa_code: str | None = None


class AdminUserRecordResponse(BaseModel):
    """Admin view of user/role/security profile."""

    username: str
    role: str
    has_mfa: bool
    role_source: str


class ControlActionResponse(BaseModel):
    """Response for control actions."""

    status: str
    detail: str
    timestamp: str


class SIEMStatusResponse(BaseModel):
    """SIEM forwarder health and delivery status."""

    enabled: bool
    endpoint: str | None
    queue_depth: int
    total_enqueued: int
    total_delivered: int
    total_failed: int
    last_flush_at: str | None
    last_success_at: str | None
    last_error: str | None


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


_oidc_lock = threading.RLock()
_oidc_discovery_cache: dict[str, Any] = {}
_oidc_discovery_cache_ts: float = 0.0
_oidc_pending_states: dict[str, dict[str, Any]] = {}


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse boolean flag from environment variable."""
    raw = os.environ.get(name, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _oidc_enabled() -> bool:
    """Return whether OIDC/SSO is enabled."""
    return _env_flag("OIDC_ENABLED", False)


def _oidc_issuer() -> str:
    """Return OIDC issuer URL."""
    return os.environ.get("OIDC_ISSUER_URL", "").strip().rstrip("/")


def _oidc_client_id() -> str:
    """Return OIDC client id."""
    return os.environ.get("OIDC_CLIENT_ID", "").strip()


def _oidc_client_secret() -> str:
    """Return OIDC client secret (optional for public clients)."""
    return os.environ.get("OIDC_CLIENT_SECRET", "").strip()


def _oidc_redirect_uri() -> str:
    """Return OIDC redirect URI."""
    configured = os.environ.get("OIDC_REDIRECT_URI", "").strip()
    if configured:
        return configured
    return "http://127.0.0.1:8000/auth/sso/callback"


def _oidc_scopes() -> list[str]:
    """Return normalized OIDC scopes."""
    raw = os.environ.get("OIDC_SCOPES", "openid profile email")
    scopes = [x.strip() for x in raw.replace(",", " ").split() if x.strip()]
    if "openid" not in scopes:
        scopes.insert(0, "openid")
    return scopes


def _oidc_username_claim() -> str:
    """Claim to extract username from userinfo."""
    return os.environ.get("OIDC_USERNAME_CLAIM", "preferred_username").strip() or "preferred_username"


def _oidc_role_claim() -> str:
    """Claim to extract role/group from userinfo."""
    return os.environ.get("OIDC_ROLE_CLAIM", "groups").strip() or "groups"


def _oidc_default_role() -> str:
    """Default role applied when SSO claim mapping is missing."""
    candidate = os.environ.get("OIDC_DEFAULT_ROLE", "viewer").strip().lower() or "viewer"
    return candidate if candidate in ROLE_PERMISSIONS else "viewer"


def _oidc_state_ttl_seconds() -> int:
    """Pending OIDC state TTL."""
    try:
        return max(60, int(os.environ.get("OIDC_STATE_TTL_SECONDS", "600")))
    except Exception:
        return 600


def _oidc_discovery_ttl_seconds() -> int:
    """OIDC discovery cache TTL."""
    try:
        return max(60, int(os.environ.get("OIDC_DISCOVERY_TTL_SECONDS", "900")))
    except Exception:
        return 900


def _oidc_frontend_base_url() -> str:
    """Frontend base URL for callback redirects."""
    base = os.environ.get("FRONTEND_BASE_URL", "http://localhost:5173").strip()
    return base.rstrip("/") if base else "http://localhost:5173"


def _oidc_frontend_login_url() -> str:
    """Frontend login route used for successful/failed callback landing."""
    return f"{_oidc_frontend_base_url()}/login"


def _oidc_allowed_return_origins() -> set[str]:
    """Allowed origins for post-SSO browser redirects."""
    default = _oidc_frontend_base_url()
    raw = os.environ.get("OIDC_ALLOWED_RETURN_ORIGINS", default).strip()
    origins = {x.strip().rstrip("/") for x in raw.split(",") if x.strip()}
    if default:
        origins.add(default.rstrip("/"))
    return origins


def _sanitize_sso_return_to(return_to: str | None) -> str:
    """Validate post-login return URL to prevent open redirects."""
    fallback = _oidc_frontend_login_url()
    candidate = (return_to or fallback).strip()
    if not candidate:
        return fallback
    if candidate.startswith("/"):
        candidate = f"{_oidc_frontend_base_url()}{candidate}"
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return fallback
    origin = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
    if origin not in _oidc_allowed_return_origins():
        return fallback
    return candidate


def _oidc_ssl_context() -> ssl.SSLContext | None:
    """Build TLS context for outbound OIDC/SIEM calls."""
    if _env_flag("OIDC_VERIFY_TLS", True):
        return None
    return ssl._create_unverified_context()


def _http_get_json(url: str, timeout: float = 10.0, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """Fetch JSON payload over HTTP."""
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    req = Request(url=url, method="GET", headers=request_headers)
    try:
        with urlopen(req, timeout=timeout, context=_oidc_ssl_context()) as response:
            content = response.read().decode("utf-8")
        payload = json.loads(content)
        if isinstance(payload, dict):
            return payload
        raise ValueError("Expected object payload")
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail=f"Upstream HTTP failure: {exc}") from exc


def _http_post_form(
    url: str,
    form_data: dict[str, str],
    timeout: float = 10.0,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """POST form-encoded data and parse JSON response."""
    request_headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    if headers:
        request_headers.update(headers)
    encoded = urlencode(form_data).encode("utf-8")
    req = Request(url=url, data=encoded, method="POST", headers=request_headers)
    try:
        with urlopen(req, timeout=timeout, context=_oidc_ssl_context()) as response:
            content = response.read().decode("utf-8")
        payload = json.loads(content)
        if isinstance(payload, dict):
            return payload
        raise ValueError("Expected object payload")
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail=f"Upstream HTTP failure: {exc}") from exc


def _oidc_discover(force_refresh: bool = False) -> dict[str, Any]:
    """Resolve OIDC discovery metadata with TTL cache."""
    global _oidc_discovery_cache_ts
    if not _oidc_enabled():
        return {}
    issuer = _oidc_issuer()
    if not issuer:
        return {}
    now = time.time()
    ttl = _oidc_discovery_ttl_seconds()
    with _oidc_lock:
        if (not force_refresh) and _oidc_discovery_cache and (now - _oidc_discovery_cache_ts <= ttl):
            return dict(_oidc_discovery_cache)
    metadata = _http_get_json(f"{issuer}/.well-known/openid-configuration")
    with _oidc_lock:
        _oidc_discovery_cache.clear()
        _oidc_discovery_cache.update(metadata)
        _oidc_discovery_cache_ts = now
    return metadata


def _oidc_is_configured() -> bool:
    """Return whether OIDC has minimum required configuration."""
    if not _oidc_enabled():
        return False
    if not _oidc_issuer() or not _oidc_client_id() or not _oidc_redirect_uri():
        return False
    try:
        metadata = _oidc_discover(force_refresh=False)
    except HTTPException:
        return False
    return bool(metadata.get("authorization_endpoint")) and bool(metadata.get("token_endpoint"))


def _pkce_verifier() -> str:
    """Generate RFC7636 code verifier."""
    raw = secrets.token_urlsafe(72)
    cleaned = "".join(ch for ch in raw if ch.isalnum() or ch in "-._~")
    if len(cleaned) < 43:
        cleaned += secrets.token_urlsafe(43 - len(cleaned))
    return cleaned[:96]


def _pkce_challenge(verifier: str) -> str:
    """Generate RFC7636 code challenge from verifier."""
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("utf-8")


def _oidc_cleanup_pending_states() -> None:
    """Prune expired OIDC login states."""
    ttl = _oidc_state_ttl_seconds()
    now = time.time()
    with _oidc_lock:
        stale = []
        for state, payload in _oidc_pending_states.items():
            created_at = float(payload.get("created_at", 0.0))
            if now - created_at > ttl:
                stale.append(state)
        for state in stale:
            _oidc_pending_states.pop(state, None)


def _oidc_store_state(state: str, code_verifier: str, return_to: str) -> None:
    """Store temporary OIDC state record."""
    _oidc_cleanup_pending_states()
    with _oidc_lock:
        _oidc_pending_states[state] = {
            "created_at": time.time(),
            "code_verifier": code_verifier,
            "return_to": return_to,
        }


def _oidc_pop_state(state: str) -> dict[str, Any] | None:
    """Pop and return temporary OIDC state record."""
    _oidc_cleanup_pending_states()
    with _oidc_lock:
        payload = _oidc_pending_states.pop(state, None)
    return dict(payload) if isinstance(payload, dict) else None


def _oidc_extract_username(userinfo: dict[str, Any]) -> str:
    """Extract username from userinfo using configured/fallback claims."""
    claims = [
        _oidc_username_claim(),
        "preferred_username",
        "email",
        "upn",
        "sub",
    ]
    for claim in claims:
        value = userinfo.get(claim)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise HTTPException(status_code=502, detail="OIDC userinfo does not contain a valid username claim")


def _oidc_claim_values(userinfo: dict[str, Any], claim_name: str) -> list[str]:
    """Normalize OIDC claim values into string list."""
    value = userinfo.get(claim_name)
    if isinstance(value, str):
        raw_values = value.replace(";", ",").replace("|", ",").split(",")
        return [x.strip() for x in raw_values if x.strip()]
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


def _oidc_map_role(userinfo: dict[str, Any]) -> str:
    """Map OIDC claim values to internal dashboard role."""
    role_map_raw = _parse_mapping_env(os.environ.get("OIDC_ROLE_MAP", ""))
    role_map = {str(k).strip().lower(): str(v).strip().lower() for k, v in role_map_raw.items()}
    claim_name = _oidc_role_claim()
    candidates = _oidc_claim_values(userinfo, claim_name)

    for candidate in candidates:
        direct = candidate.lower()
        mapped = role_map.get(direct, role_map.get(candidate))
        if mapped and mapped in ROLE_PERMISSIONS:
            return mapped
        if direct in ROLE_PERMISSIONS:
            return direct
    return _oidc_default_role()


def _oidc_decode_id_token_unverified(id_token: str) -> dict[str, Any]:
    """Decode JWT payload without signature validation as userinfo fallback."""
    parts = id_token.split(".")
    if len(parts) != 3:
        return {}
    payload_b64 = parts[1]
    padding = 4 - (len(payload_b64) % 4)
    if padding != 4:
        payload_b64 += "=" * padding
    try:
        decoded = base64.urlsafe_b64decode(payload_b64.encode("utf-8"))
        payload = json.loads(decoded.decode("utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _build_fragment_redirect(base_url: str, params: dict[str, str]) -> str:
    """Build redirect URL with fragment payload (not sent to server logs)."""
    encoded = urlencode(params)
    safe_base = base_url.split("#", 1)[0]
    return f"{safe_base}#{encoded}"

def _parse_mapping_env(value: str) -> dict[str, str]:
    """Parse env string in form 'key1:val1,key2:val2'."""
    result: dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        key, mapped = item.split(":", 1)
        key = key.strip()
        mapped = mapped.strip()
        if key and mapped:
            result[key] = mapped
    return result


def _get_user_role(username: str) -> str:
    """Resolve dashboard role for a username."""
    settings = get_settings()
    if not settings.security.require_auth and username == "anonymous":
        return "admin"
    with _security_runtime_lock:
        override = _role_overrides.get(username)
    if override:
        return override
    raw_map = os.environ.get("DASHBOARD_USER_ROLES", "")
    role_map = _parse_mapping_env(raw_map)
    if username == "api_key_user":
        return os.environ.get("DASHBOARD_API_KEY_ROLE", "viewer").strip().lower() or "viewer"
    if username == "admin" and "admin" not in role_map:
        return "admin"
    return role_map.get(username, os.environ.get("DASHBOARD_DEFAULT_ROLE", "viewer")).strip().lower() or "viewer"


ROLE_PERMISSIONS: dict[str, set[str]] = {
    "viewer": {
        "read.basic",
        "auth.mfa.self_manage",
    },
    "operator": {
        "read.basic",
        "alerts.manage",
        "control.trading.status",
        "control.trading.start",
        "control.trading.stop",
        "control.trading.restart",
        "control.jobs.create",
        "control.jobs.read",
        "control.jobs.cancel",
        "models.governance.read",
        "operations.sre.read",
        "operations.runbooks.execute",
        "auth.mfa.self_manage",
    },
    "risk": {
        "read.basic",
        "alerts.manage",
        "risk.advanced.read",
        "control.audit.read",
        "control.trading.status",
        "control.risk.kill_switch.activate",
        "control.risk.kill_switch.reset",
        "control.jobs.read",
        "models.governance.read",
        "operations.sre.read",
        "auth.mfa.self_manage",
    },
    "admin": {
        "*",
        "models.governance.write",
        "auth.security.rotate",
        "admin.users.manage",
    },
}


def _has_permission(username: str, permission: str) -> bool:
    """Check whether user has requested permission."""
    role = _get_user_role(username)
    grants = ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS["viewer"])
    if "*" in grants or permission in grants:
        return True
    if permission.startswith("read.") and "read.basic" in grants:
        return True
    return False


def _require_permission(username: str, permission: str) -> None:
    """Raise 403 if user does not have permission."""
    if not _has_permission(username, permission):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permission: {permission}",
        )


def _get_user_mfa_secret(username: str) -> str | None:
    """Get user's TOTP secret from environment mapping."""
    with _security_runtime_lock:
        dynamic_secret = _dynamic_mfa_secrets.get(username)
    if dynamic_secret:
        return dynamic_secret
    mapping = _parse_mapping_env(os.environ.get("DASHBOARD_MFA_SECRETS", ""))
    secret = mapping.get(username)
    return secret.strip() if secret else None


def _base32_decode(secret: str) -> bytes:
    """Decode base32 secret with padding normalization."""
    normalized = "".join(secret.upper().split())
    missing_padding = len(normalized) % 8
    if missing_padding:
        normalized += "=" * (8 - missing_padding)
    return base64.b32decode(normalized, casefold=True)


def _generate_totp(secret: str, timestamp: int | None = None, period: int = 30, digits: int = 6) -> str:
    """Generate TOTP code for a given timestamp."""
    current_ts = int(timestamp if timestamp is not None else datetime.now(timezone.utc).timestamp())
    counter = current_ts // period
    key = _base32_decode(secret)
    msg = counter.to_bytes(8, byteorder="big")
    digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code_int = (
        ((digest[offset] & 0x7F) << 24)
        | (digest[offset + 1] << 16)
        | (digest[offset + 2] << 8)
        | digest[offset + 3]
    )
    return str(code_int % (10**digits)).zfill(digits)


def _verify_totp(secret: str, code: str, window: int = 1) -> bool:
    """Verify a TOTP code with small time drift window."""
    clean_code = "".join(ch for ch in code if ch.isdigit())
    if len(clean_code) != 6:
        return False
    now = int(datetime.now(timezone.utc).timestamp())
    for step in range(-window, window + 1):
        ts = now + (step * 30)
        if hmac.compare_digest(_generate_totp(secret, timestamp=ts), clean_code):
            return True
    return False


def _require_mfa_if_configured(username: str, action: str, mfa_code: str | None) -> None:
    """Enforce MFA only when user has configured secret."""
    secret = _get_user_mfa_secret(username)
    if not secret:
        return
    if not mfa_code:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"MFA code required for action: {action}",
        )
    try:
        valid = _verify_totp(secret, mfa_code)
    except Exception:
        valid = False
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid MFA code",
        )


def _generate_base32_secret(length: int = 20) -> str:
    """Generate RFC4648 base32 secret for TOTP."""
    raw = secrets.token_bytes(length)
    return base64.b32encode(raw).decode("utf-8").rstrip("=")


def _build_totp_provisioning_uri(username: str, secret: str) -> str:
    """Build otpauth URI for authenticator enrollment."""
    issuer = os.environ.get("DASHBOARD_MFA_ISSUER", "AlphaTrade")
    label = f"{issuer}:{username}"
    return f"otpauth://totp/{label}?secret={secret}&issuer={issuer}&algorithm=SHA1&digits=6&period=30"


def _active_jwt_fingerprint() -> str:
    """Return short fingerprint of current signing key."""
    secrets_ring = _get_jwt_signing_secrets()
    active = secrets_ring[0] if secrets_ring else "none"
    return hashlib.sha256(active.encode("utf-8")).hexdigest()[:12]


def _role_source_for_user(username: str) -> str:
    """Resolve role source for diagnostics."""
    with _security_runtime_lock:
        if username in _role_overrides:
            return "runtime_override"
    env_map = _parse_mapping_env(os.environ.get("DASHBOARD_USER_ROLES", ""))
    if username in env_map:
        return "env_mapping"
    if username == "api_key_user":
        return "api_key_role"
    if username == "admin":
        return "default_admin"
    return "default_role"


def _list_known_users() -> list[str]:
    """Return known users from configured sources."""
    users = set(_DEMO_USERS.keys())
    env_role_users = _parse_mapping_env(os.environ.get("DASHBOARD_USER_ROLES", "")).keys()
    users.update(env_role_users)
    with _security_runtime_lock:
        users.update(_role_overrides.keys())
        users.update(_dynamic_mfa_secrets.keys())
        users.update(_pending_mfa_enrollments.keys())
    users.add("api_key_user")
    return sorted(u for u in users if u)


def _serialize_admin_users() -> list[AdminUserRecordResponse]:
    """Serialize known users for admin panel."""
    records = []
    for username in _list_known_users():
        records.append(
            AdminUserRecordResponse(
                username=username,
                role=_get_user_role(username),
                has_mfa=bool(_get_user_mfa_secret(username)),
                role_source=_role_source_for_user(username),
            )
        )
    return records


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
    _record_signed_audit(
        action="auth.login",
        user=request.username,
        status_value="success",
        details={"role": _get_user_role(request.username)},
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
    )


@app.get("/auth/sso/status", response_model=SSOStatusResponse, tags=["Authentication"])
async def get_sso_status() -> SSOStatusResponse:
    """Return OIDC SSO status and discovery availability."""
    enabled = _oidc_enabled()
    metadata: dict[str, Any] = {}
    if enabled:
        try:
            metadata = _oidc_discover(force_refresh=False)
        except HTTPException:
            metadata = {}
    return SSOStatusResponse(
        enabled=enabled,
        configured=_oidc_is_configured(),
        issuer=_oidc_issuer() or None,
        authorization_endpoint=str(metadata.get("authorization_endpoint")) if metadata.get("authorization_endpoint") else None,
        token_endpoint=str(metadata.get("token_endpoint")) if metadata.get("token_endpoint") else None,
        userinfo_endpoint=str(metadata.get("userinfo_endpoint")) if metadata.get("userinfo_endpoint") else None,
        redirect_uri=_oidc_redirect_uri() if enabled else None,
        username_claim=_oidc_username_claim(),
        role_claim=_oidc_role_claim(),
        scopes=_oidc_scopes(),
    )


@app.get("/auth/sso/start", response_model=SSOStartResponse, tags=["Authentication"])
async def start_sso_auth(
    return_to: str | None = Query(default=None),
    redirect: bool = Query(default=True),
) -> SSOStartResponse | RedirectResponse:
    """Initialize OIDC authorization code flow with PKCE."""
    if not _oidc_enabled():
        raise HTTPException(status_code=404, detail="SSO is disabled")
    if not _oidc_is_configured():
        raise HTTPException(status_code=503, detail="SSO is enabled but not fully configured")

    metadata = _oidc_discover(force_refresh=False)
    authorization_endpoint = str(metadata.get("authorization_endpoint", "")).strip()
    if not authorization_endpoint:
        raise HTTPException(status_code=503, detail="OIDC discovery missing authorization_endpoint")

    state = secrets.token_urlsafe(32)
    verifier = _pkce_verifier()
    challenge = _pkce_challenge(verifier)
    safe_return_to = _sanitize_sso_return_to(return_to)
    _oidc_store_state(state, verifier, safe_return_to)

    query: dict[str, str] = {
        "response_type": "code",
        "client_id": _oidc_client_id(),
        "redirect_uri": _oidc_redirect_uri(),
        "scope": " ".join(_oidc_scopes()),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    audience = os.environ.get("OIDC_AUDIENCE", "").strip()
    if audience:
        query["audience"] = audience
    prompt = os.environ.get("OIDC_PROMPT", "").strip()
    if prompt:
        query["prompt"] = prompt

    auth_url = f"{authorization_endpoint}?{urlencode(query, quote_via=quote)}"
    payload = SSOStartResponse(
        auth_url=auth_url,
        state=state,
        expires_in_seconds=_oidc_state_ttl_seconds(),
    )
    if redirect:
        return RedirectResponse(url=auth_url, status_code=307)
    return payload


@app.get("/auth/sso/callback", tags=["Authentication"])
async def handle_sso_callback(
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
    error_description: str | None = Query(default=None),
) -> RedirectResponse:
    """Handle OIDC callback, mint local JWT, and redirect browser to frontend."""
    fallback_return = _oidc_frontend_login_url()
    if not _oidc_enabled():
        return RedirectResponse(
            url=_build_fragment_redirect(
                fallback_return,
                {"error": "sso_disabled", "error_description": "SSO is disabled"},
            ),
            status_code=302,
        )

    pending = _oidc_pop_state(state or "")
    return_to = _sanitize_sso_return_to((pending or {}).get("return_to"))
    if error:
        return RedirectResponse(
            url=_build_fragment_redirect(
                return_to,
                {
                    "error": error,
                    "error_description": error_description or "OIDC provider returned an error",
                },
            ),
            status_code=302,
        )
    if not code or not state:
        return RedirectResponse(
            url=_build_fragment_redirect(
                return_to,
                {
                    "error": "invalid_callback",
                    "error_description": "Missing code/state in callback",
                },
            ),
            status_code=302,
        )
    if not pending:
        return RedirectResponse(
            url=_build_fragment_redirect(
                return_to,
                {
                    "error": "invalid_state",
                    "error_description": "SSO state is missing or expired",
                },
            ),
            status_code=302,
        )

    try:
        metadata = _oidc_discover(force_refresh=False)
        token_endpoint = str(metadata.get("token_endpoint", "")).strip()
        if not token_endpoint:
            raise HTTPException(status_code=503, detail="OIDC discovery missing token_endpoint")

        token_request: dict[str, str] = {
            "grant_type": "authorization_code",
            "client_id": _oidc_client_id(),
            "code": code,
            "redirect_uri": _oidc_redirect_uri(),
            "code_verifier": str(pending.get("code_verifier", "")),
        }
        client_secret = _oidc_client_secret()
        if client_secret:
            token_request["client_secret"] = client_secret

        token_payload = _http_post_form(token_endpoint, token_request, timeout=12.0)
        oidc_access_token = str(token_payload.get("access_token", "")).strip()
        oidc_id_token = str(token_payload.get("id_token", "")).strip()

        userinfo: dict[str, Any] = {}
        userinfo_endpoint = str(metadata.get("userinfo_endpoint", "")).strip()
        if userinfo_endpoint and oidc_access_token:
            userinfo = _http_get_json(
                userinfo_endpoint,
                timeout=12.0,
                headers={"Authorization": f"Bearer {oidc_access_token}"},
            )
        if not userinfo and oidc_id_token:
            userinfo = _oidc_decode_id_token_unverified(oidc_id_token)
        if not userinfo:
            raise HTTPException(status_code=502, detail="Unable to fetch OIDC user profile")

        username = _oidc_extract_username(userinfo)
        role = _oidc_map_role(userinfo)
        with _security_runtime_lock:
            _role_overrides[username] = role
            _save_security_runtime_state()

        settings = get_settings()
        expires_in = settings.security.jwt_expiration_minutes * 60
        local_token = create_access_token(username)
        _record_signed_audit(
            action="auth.sso.login",
            user=username,
            status_value="success",
            details={
                "role": role,
                "issuer": _oidc_issuer(),
                "subject": str(userinfo.get("sub", "")),
                "username_claim": _oidc_username_claim(),
                "role_claim": _oidc_role_claim(),
            },
        )
        return RedirectResponse(
            url=_build_fragment_redirect(
                return_to,
                {
                    "access_token": local_token,
                    "token_type": "bearer",
                    "expires_in": str(expires_in),
                    "username": username,
                    "role": role,
                    "sso": "1",
                },
            ),
            status_code=302,
        )
    except HTTPException as exc:
        _record_signed_audit(
            action="auth.sso.login",
            user="unknown",
            status_value="failed",
            details={"reason": str(exc.detail)},
        )
        return RedirectResponse(
            url=_build_fragment_redirect(
                return_to,
                {
                    "error": "oidc_exchange_failed",
                    "error_description": str(exc.detail),
                },
            ),
            status_code=302,
        )


@app.get("/auth/me", tags=["Authentication"])
async def get_current_user_info(current_user: AuthenticatedUser) -> dict[str, Any]:
    """Get information about the current authenticated user."""
    role = _get_user_role(current_user)
    return {
        "username": current_user,
        "role": role,
        "mfa_enabled": bool(_get_user_mfa_secret(current_user)),
    }


@app.get("/auth/mfa/status", response_model=MFAStatusResponse, tags=["Authentication"])
async def get_mfa_status(current_user: AuthenticatedUser) -> MFAStatusResponse:
    """Get MFA policy status for current user."""
    role = _get_user_role(current_user)
    return MFAStatusResponse(
        username=current_user,
        role=role,
        mfa_enabled=bool(_get_user_mfa_secret(current_user)),
    )


@app.post("/auth/mfa/enroll/init", response_model=MFAEnrollInitResponse, tags=["Authentication"])
async def init_mfa_enrollment(current_user: AuthenticatedUser) -> MFAEnrollInitResponse:
    """Initialize MFA enrollment for the current user."""
    _require_permission(current_user, "auth.mfa.self_manage")
    with _security_runtime_lock:
        if current_user in _dynamic_mfa_secrets:
            raise HTTPException(status_code=409, detail="MFA already enrolled for this user")
        secret = _generate_base32_secret()
        _pending_mfa_enrollments[current_user] = {
            "secret": secret,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    issuer = os.environ.get("DASHBOARD_MFA_ISSUER", "AlphaTrade")
    provisioning_uri = _build_totp_provisioning_uri(current_user, secret)
    _record_signed_audit(
        action="auth.mfa.enroll.init",
        user=current_user,
        status_value="initiated",
        details={"issuer": issuer},
    )
    return MFAEnrollInitResponse(
        username=current_user,
        secret=secret,
        provisioning_uri=provisioning_uri,
        issuer=issuer,
    )


@app.post("/auth/mfa/enroll/verify", response_model=MFAStatusResponse, tags=["Authentication"])
async def verify_mfa_enrollment(
    request: MFAEnrollVerifyRequest,
    current_user: AuthenticatedUser,
) -> MFAStatusResponse:
    """Verify MFA enrollment code and activate secret."""
    _require_permission(current_user, "auth.mfa.self_manage")
    with _security_runtime_lock:
        pending = _pending_mfa_enrollments.get(current_user)
        if not pending:
            raise HTTPException(status_code=400, detail="No pending MFA enrollment found")
        secret = str(pending.get("secret") or "")

    if not _verify_totp(secret, request.code):
        raise HTTPException(status_code=401, detail="Invalid MFA code")

    with _security_runtime_lock:
        _dynamic_mfa_secrets[current_user] = secret
        _pending_mfa_enrollments.pop(current_user, None)
        _save_security_runtime_state()

    _record_signed_audit(
        action="auth.mfa.enroll.verify",
        user=current_user,
        status_value="enabled",
        details={},
    )
    return MFAStatusResponse(
        username=current_user,
        role=_get_user_role(current_user),
        mfa_enabled=True,
    )


@app.post("/auth/mfa/disable", response_model=MFAStatusResponse, tags=["Authentication"])
async def disable_mfa(
    request: MFADisableRequest,
    current_user: AuthenticatedUser,
) -> MFAStatusResponse:
    """Disable MFA for the current user with valid TOTP challenge."""
    _require_permission(current_user, "auth.mfa.self_manage")
    secret = _get_user_mfa_secret(current_user)
    if not secret:
        return MFAStatusResponse(
            username=current_user,
            role=_get_user_role(current_user),
            mfa_enabled=False,
        )
    if not _verify_totp(secret, request.code):
        raise HTTPException(status_code=401, detail="Invalid MFA code")

    with _security_runtime_lock:
        _dynamic_mfa_secrets.pop(current_user, None)
        _pending_mfa_enrollments.pop(current_user, None)
        _save_security_runtime_state()

    _record_signed_audit(
        action="auth.mfa.disable",
        user=current_user,
        status_value="disabled",
        details={},
    )
    return MFAStatusResponse(
        username=current_user,
        role=_get_user_role(current_user),
        mfa_enabled=bool(_get_user_mfa_secret(current_user)),
    )


@app.get("/auth/security/status", response_model=SecurityStatusResponse, tags=["Authentication"])
async def get_security_status(current_user: AuthenticatedUser) -> SecurityStatusResponse:
    """Get current authentication and security posture."""
    _require_permission(current_user, "auth.security.rotate")
    settings = get_settings()
    with _security_runtime_lock:
        mfa_count = len(_dynamic_mfa_secrets)
        role_override_count = len(_role_overrides)
    secrets_ring = _get_jwt_signing_secrets()
    return SecurityStatusResponse(
        jwt_key_count=len(secrets_ring),
        active_key_fingerprint=_active_jwt_fingerprint(),
        require_auth=settings.security.require_auth,
        api_key_enabled=bool(settings.security.api_keys),
        mfa_enabled_users=mfa_count,
        role_overrides_count=role_override_count,
        rate_limit_enabled=_rate_limit_enabled(),
        rate_limit_limit=_rate_limit_limit_per_window(),
        rate_limit_window_seconds=_rate_limit_window_seconds(),
    )


@app.post("/auth/security/rotate-jwt", response_model=ControlActionResponse, tags=["Authentication"])
async def rotate_jwt_secret(
    request: JwtRotateRequest,
    current_user: AuthenticatedUser,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    x_mfa_code: Annotated[str | None, Header(alias="X-MFA-Code")] = None,
) -> ControlActionResponse:
    """Rotate active JWT secret key and keep fallback verification keys."""
    _require_permission(current_user, "auth.security.rotate")
    _require_mfa_if_configured(
        current_user,
        action="auth.security.rotate-jwt",
        mfa_code=request.mfa_code or x_mfa_code,
    )

    payload = request.model_dump(mode="json")
    cached, fingerprint = _check_idempotent_replay(
        action="auth.security.rotate-jwt",
        user=current_user,
        idempotency_key=idempotency_key,
        payload=payload,
    )
    if cached:
        _record_signed_audit(
            action="auth.security.rotate-jwt",
            user=current_user,
            status_value="replayed",
            details={"cached": True},
            idempotency_key=idempotency_key,
        )
        return ControlActionResponse(**cached)

    with _security_runtime_lock:
        previous = _get_jwt_signing_secrets()
        next_ring = [request.new_secret, *[x for x in previous if x != request.new_secret]]
        _jwt_runtime_key_ring.clear()
        _jwt_runtime_key_ring.extend(next_ring[:5])
        _save_security_runtime_state()

    response = ControlActionResponse(
        status="rotated",
        detail="JWT signing key rotated successfully",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    _store_idempotent_response(
        action="auth.security.rotate-jwt",
        idempotency_key=idempotency_key,
        fingerprint=fingerprint,
        response_payload=response.model_dump(mode="json"),
    )
    _record_signed_audit(
        action="auth.security.rotate-jwt",
        user=current_user,
        status_value="success",
        details={"active_key_fingerprint": _active_jwt_fingerprint(), "ring_size": len(_get_jwt_signing_secrets())},
        idempotency_key=idempotency_key,
    )
    return response


@app.get("/admin/users", response_model=list[AdminUserRecordResponse], tags=["Admin"])
async def list_admin_users(current_user: AuthenticatedUser) -> list[AdminUserRecordResponse]:
    """List users with role and MFA status."""
    _require_permission(current_user, "admin.users.manage")
    return _serialize_admin_users()


@app.post("/admin/users/{username}/role", response_model=ControlActionResponse, tags=["Admin"])
async def update_admin_user_role(
    username: str,
    request: AdminUserRoleUpdateRequest,
    current_user: AuthenticatedUser,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    x_mfa_code: Annotated[str | None, Header(alias="X-MFA-Code")] = None,
) -> ControlActionResponse:
    """Update user role override."""
    _require_permission(current_user, "admin.users.manage")
    _require_mfa_if_configured(
        current_user,
        action="admin.users.role_update",
        mfa_code=request.mfa_code or x_mfa_code,
    )
    payload = {
        "username": username,
        **request.model_dump(mode="json"),
    }
    cached, fingerprint = _check_idempotent_replay(
        action="admin.users.role_update",
        user=current_user,
        idempotency_key=idempotency_key,
        payload=payload,
    )
    if cached:
        return ControlActionResponse(**cached)

    with _security_runtime_lock:
        _role_overrides[username] = request.role.lower()
        _save_security_runtime_state()

    response = ControlActionResponse(
        status="updated",
        detail=f"Role for {username} set to {request.role}",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    _store_idempotent_response(
        action="admin.users.role_update",
        idempotency_key=idempotency_key,
        fingerprint=fingerprint,
        response_payload=response.model_dump(mode="json"),
    )
    _record_signed_audit(
        action="admin.users.role_update",
        user=current_user,
        status_value="success",
        details={"target_user": username, "role": request.role, "reason": request.reason},
        idempotency_key=idempotency_key,
    )
    return response


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


class ExecutionQualityResponse(BaseModel):
    """Execution quality decomposition response."""

    timestamp: str
    arrival_price_delta_bps: float
    rejection_rate: float
    fill_rate_buy: float
    fill_rate_sell: float
    venue_slippage_buckets: dict[str, float]
    latency_distribution_ms: dict[str, float]


class VaRResponse(BaseModel):
    """Value at Risk & Stress Test response."""
    
    timestamp: str
    var_95: float
    var_99: float
    cvar_95: float
    stress_scenarios: dict[str, float]  # e.g., "2008_crash": -25.4%
    distribution_curve: list[dict[str, float]]  # x: pnl, y: probability


class RiskConcentrationResponse(BaseModel):
    """Concentration risk response."""

    timestamp: str
    largest_symbol_pct: float
    top3_symbols_pct: float
    hhi_symbol: float
    hhi_sector: float
    symbol_weights: dict[str, float]
    sector_weights: dict[str, float]


class RiskCorrelationResponse(BaseModel):
    """Correlation risk response."""

    timestamp: str
    average_pairwise_correlation: float
    cluster_risk_score: float
    beta_weighted_exposure: float
    matrix: list[dict[str, Any]]


class RiskStressResponse(BaseModel):
    """Stress test response."""

    timestamp: str
    scenarios: dict[str, float]
    worst_case_loss: float
    resilience_score: float


class RiskAttributionResponse(BaseModel):
    """Pre-trade and post-trade risk attribution."""

    timestamp: str
    pre_trade_checks: list[dict[str, Any]]
    post_trade_findings: list[dict[str, Any]]
    breaches_count: int


class FeatureImportanceResponse(BaseModel):
    """AI Feature Importance response."""
    
    timestamp: str
    model_name: str
    global_importance: dict[str, float]  # feature: score
    recent_shift: dict[str, float]  # feature: change_pct


class TradingStartRequest(BaseModel):
    """Start/Restart request for the trading process."""

    mode: str = Field(default="paper", pattern="^(live|paper|dry-run)$")
    symbols: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
    strategy: str = Field(default="momentum")
    capital: float = Field(default=100000.0, gt=0)
    mfa_code: str | None = None


class CommandJobRequest(BaseModel):
    """Request to run an operational command asynchronously."""

    command: str = Field(
        ...,
        description="One of: backtest, train, data, features, health, deploy",
    )
    args: list[str] = Field(default_factory=list)


class CommandJobResponse(BaseModel):
    """Async job status response."""

    job_id: str
    command: str
    args: list[str]
    status: str
    created_at: str
    started_at: str | None
    ended_at: str | None
    exit_code: int | None
    output: str


class CommandCatalogEntryResponse(BaseModel):
    """Allowlisted dashboard job command metadata."""

    command: str
    summary: str
    script_path: str
    script_exists: bool
    sample_args: list[str]
    risk_level: str


class KillSwitchActivateRequest(BaseModel):
    """Request for kill-switch activation."""

    reason: str = Field(default="manual_activation")
    mfa_code: str | None = None


class KillSwitchResetRequest(BaseModel):
    """Request for kill-switch reset."""

    authorized_by: str = Field(..., min_length=1)
    force: bool = Field(default=False)
    override_code: str | None = None
    mfa_code: str | None = None


class AuditRecordResponse(BaseModel):
    """Signed audit record response."""

    timestamp: str
    action: str
    user: str
    status: str
    details: dict[str, Any]
    prev_hash: str
    record_hash: str


class ModelRegistryEntryResponse(BaseModel):
    """Model registry entry."""

    model_name: str
    version_id: str
    model_version: str
    model_type: str
    registered_at: str
    metrics: dict[str, Any]
    tags: list[str]
    is_active: bool
    path: str


class ModelRegistryResponse(BaseModel):
    """Model registry snapshot."""

    timestamp: str
    model_count: int
    versions_count: int
    active_model: str | None
    entries: list[ModelRegistryEntryResponse]


class ModelDriftResponse(BaseModel):
    """Model drift monitoring response."""

    timestamp: str
    model_name: str
    drift_score: float
    drift_status: str
    staleness_reason: str | None
    recommendation: str
    feature_shift: dict[str, float]


class ModelValidationGateResponse(BaseModel):
    """Validation gate decision response."""

    timestamp: str
    model_name: str
    passed: bool
    gates: list[dict[str, Any]]
    failed_gates: list[str]
    decision: str


class ModelChampionChallengerResponse(BaseModel):
    """Champion/challenger state response."""

    timestamp: str
    champion: str | None
    challenger: str | None
    comparison: dict[str, float]
    recommendation: str


class ModelPromotionRequest(BaseModel):
    """Request for champion promotion."""

    model_name: str = Field(..., min_length=1)
    version_id: str = Field(..., min_length=1)
    reason: str = Field(default="manual_promotion")
    mfa_code: str | None = None


class SLOStatusResponse(BaseModel):
    """SLO and burn-rate status."""

    timestamp: str
    availability: float
    error_budget_remaining_pct: float
    p95_action_latency_ms: float
    p99_action_latency_ms: float
    burn_rate_1h: float
    burn_rate_6h: float
    status: str


class IncidentRecordResponse(BaseModel):
    """Incident response record."""

    incident_id: str
    severity: str
    title: str
    status: str
    created_at: str
    runbook_link: str | None
    suggested_action: str | None


class IncidentTimelineEventResponse(BaseModel):
    """Incident timeline event."""

    timestamp: str
    source: str
    event_type: str
    severity: str
    message: str
    context: dict[str, Any]


class RunbookRecordResponse(BaseModel):
    """Runbook metadata response."""

    alert_type: str
    runbook_path: str
    suggested_action: str | None


class RunbookExecutionRequest(BaseModel):
    """Request to execute runbook automation command."""

    action: str = Field(..., pattern="^(health_check|validate_env|gpu_check|data_feed_check|broker_reconnect_dryrun)$")
    mfa_code: str | None = None


class ScriptInventoryEntryResponse(BaseModel):
    """Inventory record for one scripts/* module."""

    script_name: str
    path: str
    exists: bool
    linked_command: str | None
    entrypoint: str | None


class DomainCoverageEntryResponse(BaseModel):
    """Coverage summary for one quant_trading_system domain package."""

    domain: str
    path: str
    module_count: int
    subpackage_count: int


class DataCoverageResponse(BaseModel):
    """Dataset inventory summary used by dashboard system coverage."""

    root: str
    exists: bool
    total_files: int
    csv_files: int
    parquet_files: int
    json_files: int
    other_files: int


class SystemCoverageResponse(BaseModel):
    """Project-wide coverage snapshot for control-plane visibility."""

    timestamp: str
    main_entrypoint: str
    main_entrypoint_exists: bool
    command_catalog: list[CommandCatalogEntryResponse]
    scripts: list[ScriptInventoryEntryResponse]
    domains: list[DomainCoverageEntryResponse]
    data_assets: DataCoverageResponse


# =============================================================================
# Persistence and Audit Infrastructure
# =============================================================================


class PersistentControlStateStore:
    """Persistent store for dashboard control state with Redis + file fallback."""

    def __init__(self) -> None:
        test_mode = "PYTEST_CURRENT_TEST" in os.environ
        default_persist = "false" if test_mode else "true"
        self._enabled = os.environ.get("DASHBOARD_PERSIST_STATE", default_persist).strip().lower() == "true"
        self._backend = os.environ.get("DASHBOARD_STATE_BACKEND", "redis").strip().lower()
        self._redis_key = os.environ.get("DASHBOARD_STATE_REDIS_KEY", "dashboard:control_state:v1").strip()
        self._lock = threading.RLock()
        self._file_path = PROJECT_ROOT / "logs" / "dashboard_control_state.json"
        self._redis_client: Any | None = None

        if self._enabled:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_redis_client()

    def _init_redis_client(self) -> None:
        """Initialize optional Redis client."""
        if self._backend != "redis" or redis_sync is None:
            return
        try:
            settings = get_settings()
            self._redis_client = redis_sync.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                db=settings.redis.db,
                socket_timeout=2,
                decode_responses=True,
            )
            self._redis_client.ping()
            logger.info("Control state persistence backend: redis")
        except Exception as exc:
            self._redis_client = None
            logger.warning(f"Redis unavailable for control state persistence, using file fallback: {exc}")

    def load(self) -> dict[str, Any]:
        """Load persisted state."""
        if not self._enabled:
            return {}
        with self._lock:
            if self._redis_client:
                try:
                    raw = self._redis_client.get(self._redis_key)
                    if raw:
                        payload = json.loads(raw)
                        if isinstance(payload, dict):
                            return payload
                except Exception as exc:
                    logger.warning(f"Failed to load control state from Redis: {exc}")
            return self._load_file()

    def save(self, payload: dict[str, Any]) -> None:
        """Save persisted state."""
        if not self._enabled:
            return
        serialized = dict(payload)
        serialized["updated_at"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if self._redis_client:
                try:
                    self._redis_client.set(self._redis_key, json.dumps(serialized, ensure_ascii=True))
                except Exception as exc:
                    logger.warning(f"Failed to save control state to Redis: {exc}")
            self._save_file(serialized)

    def _load_file(self) -> dict[str, Any]:
        """Load state from fallback JSON file."""
        if not self._file_path.exists():
            return {}
        try:
            with self._file_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            logger.warning(f"Failed to load control state file {self._file_path}: {exc}")
            return {}

    def _save_file(self, payload: dict[str, Any]) -> None:
        """Save state to fallback JSON file."""
        tmp_path = self._file_path.with_suffix(".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True)
            tmp_path.replace(self._file_path)
        except Exception as exc:
            logger.warning(f"Failed to save control state file {self._file_path}: {exc}")
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def _get_audit_secret() -> str:
    """Get audit signing secret."""
    configured = os.environ.get("DASHBOARD_AUDIT_SECRET", "").strip()
    if configured:
        return configured
    logger.warning(
        "DASHBOARD_AUDIT_SECRET is not set, using development-only fallback secret for signed audit records."
    )
    return "DEV_ONLY_AUDIT_SECRET_CHANGE_ME"


class AuditTrailSigner:
    """Append-only signed audit trail using hash chaining."""

    def __init__(self, path: Path, secret: str) -> None:
        self._path = path
        self._secret = secret.encode("utf-8")
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = self._load_last_hash()

    def _load_last_hash(self) -> str:
        """Load last hash from existing log file."""
        if not self._path.exists():
            return "GENESIS"
        last_hash = "GENESIS"
        try:
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        candidate = payload.get("record_hash")
                        if isinstance(candidate, str) and candidate:
                            last_hash = candidate
        except Exception as exc:
            logger.warning(f"Failed to read audit trail hash chain: {exc}")
        return last_hash

    def append(
        self,
        action: str,
        user: str,
        status_value: str,
        details: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Append a signed audit record and return it."""
        timestamp = datetime.now(timezone.utc).isoformat()
        base_record = {
            "timestamp": timestamp,
            "action": action,
            "user": user,
            "status": status_value,
            "details": details,
            "idempotency_key": idempotency_key,
        }
        with self._lock:
            canonical = json.dumps(base_record, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            digest_payload = f"{self._last_hash}|{canonical}".encode("utf-8")
            record_hash = hmac.new(self._secret, digest_payload, hashlib.sha256).hexdigest()
            signed = {
                **base_record,
                "prev_hash": self._last_hash,
                "record_hash": record_hash,
            }
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(signed, ensure_ascii=True))
                f.write("\n")
            self._last_hash = record_hash
            return signed

    def list_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return most recent signed audit records."""
        if not self._path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        records.append(payload)
        except Exception as exc:
            logger.warning(f"Failed reading audit records: {exc}")
            return []
        return records[-limit:]


class AuditSIEMForwarder:
    """Best-effort asynchronous forwarding of signed audit records to SIEM."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._queue: list[dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._total_enqueued = 0
        self._total_delivered = 0
        self._total_failed = 0
        self._last_flush_at: str | None = None
        self._last_success_at: str | None = None
        self._last_error: str | None = None

    @staticmethod
    def _enabled() -> bool:
        endpoint = os.environ.get("DASHBOARD_SIEM_ENDPOINT", "").strip()
        return _env_flag("DASHBOARD_SIEM_ENABLED", False) and bool(endpoint)

    @staticmethod
    def _endpoint() -> str:
        return os.environ.get("DASHBOARD_SIEM_ENDPOINT", "").strip()

    @staticmethod
    def _batch_size() -> int:
        try:
            return max(1, int(os.environ.get("DASHBOARD_SIEM_BATCH_SIZE", "25")))
        except Exception:
            return 25

    @staticmethod
    def _flush_interval_seconds() -> float:
        try:
            return max(0.5, float(os.environ.get("DASHBOARD_SIEM_FLUSH_INTERVAL_SECONDS", "5")))
        except Exception:
            return 5.0

    @staticmethod
    def _max_queue() -> int:
        try:
            return max(100, int(os.environ.get("DASHBOARD_SIEM_MAX_QUEUE", "5000")))
        except Exception:
            return 5000

    @staticmethod
    def _timeout_seconds() -> float:
        try:
            return max(1.0, float(os.environ.get("DASHBOARD_SIEM_TIMEOUT_SECONDS", "6")))
        except Exception:
            return 6.0

    @staticmethod
    def _ssl_context() -> ssl.SSLContext | None:
        if _env_flag("DASHBOARD_SIEM_VERIFY_TLS", True):
            return None
        return ssl._create_unverified_context()

    @staticmethod
    def _headers() -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        api_key = os.environ.get("DASHBOARD_SIEM_API_KEY", "").strip()
        if api_key:
            header_name = os.environ.get("DASHBOARD_SIEM_API_KEY_HEADER", "Authorization").strip() or "Authorization"
            if header_name.lower() == "authorization":
                headers[header_name] = f"Bearer {api_key}"
            else:
                headers[header_name] = api_key
        return headers

    @staticmethod
    def _payload(batch: list[dict[str, Any]]) -> bytes:
        fmt = os.environ.get("DASHBOARD_SIEM_FORMAT", "generic").strip().lower()
        if fmt == "splunk_hec":
            events = [
                {
                    "time": record.get("timestamp"),
                    "host": os.environ.get("HOSTNAME", "alphatrade-dashboard"),
                    "source": "alphatrade.dashboard.audit",
                    "event": record,
                }
                for record in batch
            ]
            return json.dumps(events, ensure_ascii=True).encode("utf-8")
        envelope = {
            "source": "alphatrade.dashboard.audit",
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "records": batch,
        }
        return json.dumps(envelope, ensure_ascii=True).encode("utf-8")

    def enqueue(self, record: dict[str, Any]) -> None:
        """Queue one signed audit record for SIEM forwarding."""
        if not self._enabled():
            return
        with self._lock:
            self._queue.append(dict(record))
            self._total_enqueued += 1
            max_queue = self._max_queue()
            if len(self._queue) > max_queue:
                overflow = len(self._queue) - max_queue
                del self._queue[:overflow]
                self._total_failed += overflow
                self._last_error = f"SIEM queue overflow: dropped {overflow} records"

    def _send_batch(self, batch: list[dict[str, Any]]) -> tuple[bool, str | None]:
        endpoint = self._endpoint()
        if not endpoint:
            return False, "SIEM endpoint is not configured"
        payload = self._payload(batch)
        req = Request(
            url=endpoint,
            data=payload,
            method="POST",
            headers=self._headers(),
        )
        try:
            with urlopen(req, timeout=self._timeout_seconds(), context=self._ssl_context()) as response:
                code = int(getattr(response, "status", 200))
            if 200 <= code < 300:
                return True, None
            return False, f"SIEM HTTP status {code}"
        except Exception as exc:
            return False, str(exc)

    def flush(self, max_batches: int = 1) -> int:
        """Flush up to max_batches from local queue."""
        if not self._enabled():
            return 0
        sent_total = 0
        for _ in range(max_batches):
            with self._lock:
                batch = self._queue[: self._batch_size()]
                self._queue = self._queue[self._batch_size() :]
            if not batch:
                break

            self._last_flush_at = datetime.now(timezone.utc).isoformat()
            success, error = self._send_batch(batch)
            if success:
                with self._lock:
                    self._total_delivered += len(batch)
                    self._last_success_at = datetime.now(timezone.utc).isoformat()
                    self._last_error = None
                sent_total += len(batch)
                continue

            with self._lock:
                self._queue = batch + self._queue
                self._total_failed += len(batch)
                self._last_error = error or "Unknown SIEM forwarding failure"
            break
        return sent_total

    def _run_loop(self) -> None:
        """Background loop to flush queue periodically."""
        while not self._stop_event.wait(self._flush_interval_seconds()):
            try:
                self.flush(max_batches=3)
            except Exception as exc:
                with self._lock:
                    self._last_error = f"SIEM forwarder loop error: {exc}"

    def start(self) -> None:
        """Start background forwarder loop if SIEM is enabled."""
        if not self._enabled():
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="dashboard-siem-forwarder",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop background loop and perform a final best-effort flush."""
        with self._lock:
            thread = self._thread
        if thread and thread.is_alive():
            self._stop_event.set()
            thread.join(timeout=3)
        try:
            self.flush(max_batches=10)
        except Exception:
            pass
        with self._lock:
            self._thread = None

    def status(self) -> SIEMStatusResponse:
        """Return current forwarder status."""
        with self._lock:
            return SIEMStatusResponse(
                enabled=self._enabled(),
                endpoint=self._endpoint() or None,
                queue_depth=len(self._queue),
                total_enqueued=self._total_enqueued,
                total_delivered=self._total_delivered,
                total_failed=self._total_failed,
                last_flush_at=self._last_flush_at,
                last_success_at=self._last_success_at,
                last_error=self._last_error,
            )


_control_state_store = PersistentControlStateStore()
_audit_signer = AuditTrailSigner(PROJECT_ROOT / "logs" / "dashboard_audit.log", _get_audit_secret())
_audit_siem_forwarder = AuditSIEMForwarder()


def _record_signed_audit(
    action: str,
    user: str,
    status_value: str,
    details: dict[str, Any],
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Write signed audit record and mirror to runtime logs."""
    record = _audit_signer.append(
        action=action,
        user=user,
        status_value=status_value,
        details=details,
        idempotency_key=idempotency_key,
    )
    state = get_dashboard_state()
    state.add_log(
        {
            "timestamp": record["timestamp"],
            "level": "INFO",
            "category": "AUDIT",
            "message": f"{action} -> {status_value}",
            "extra": {
                "user": user,
                "record_hash": record["record_hash"],
                "idempotency_key": idempotency_key,
            },
        }
    )
    _audit_siem_forwarder.enqueue(record)
    return record


# =============================================================================
# State Management
# =============================================================================


class DashboardState:
    """Dashboard state management.

    In a real implementation, this would be connected to the actual
    trading system components via dependency injection.
    """

    def __init__(self) -> None:
        """Initialize dashboard state."""
        self.start_time = datetime.now(timezone.utc)
        self._last_update = self.start_time
        self._portfolio_data: dict[str, Any] = {}
        self._positions: dict[str, dict[str, Any]] = {}
        self._orders: list[dict[str, Any]] = []
        self._signals: list[dict[str, Any]] = []
        self._model_status: dict[str, dict[str, Any]] = {}
        self._logs: list[dict[str, Any]] = []
        self._event_bus: EventBus | None = None
        self._trading_process: subprocess.Popen[str] | None = None
        self._trading_started_at: datetime | None = None
        self._command_jobs: dict[str, dict[str, Any]] = {}
        self._idempotency_records: dict[str, dict[str, Any]] = {}
        self._job_tasks: dict[str, asyncio.Task] = {}
        self._components_healthy: dict[str, bool] = {
            "database": True,
            "redis": True,
            "broker": True,
            "data_feed": True,
            "models": True,
        }
        self._state_store = _control_state_store
        self._runtime_persistence_enabled = "PYTEST_CURRENT_TEST" not in os.environ
        if self._runtime_persistence_enabled:
            self._restore_persisted_state()

    def get_uptime(self) -> float:
        """Get uptime in seconds."""
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()

    @staticmethod
    def _to_iso_local(value: datetime | None) -> str | None:
        """Serialize datetime to iso string."""
        return value.isoformat() if value else None

    @staticmethod
    def _parse_dt_local(value: str | None) -> datetime | None:
        """Parse timestamp string to datetime."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None

    def _restore_persisted_state(self) -> None:
        """Restore persisted control-plane state."""
        payload = self._state_store.load()
        if not payload:
            return

        raw_jobs = payload.get("command_jobs", {})
        if isinstance(raw_jobs, dict):
            for job_id, raw in raw_jobs.items():
                if not isinstance(raw, dict):
                    continue
                restored = {
                    "job_id": str(raw.get("job_id", job_id)),
                    "command": str(raw.get("command", "")),
                    "args": [str(x) for x in raw.get("args", [])] if isinstance(raw.get("args"), list) else [],
                    "status": str(raw.get("status", "unknown")),
                    "created_at": self._parse_dt_local(raw.get("created_at")) or self.start_time,
                    "started_at": self._parse_dt_local(raw.get("started_at")),
                    "ended_at": self._parse_dt_local(raw.get("ended_at")),
                    "exit_code": raw.get("exit_code"),
                    "output": str(raw.get("output", ""))[-20000:],
                }
                if restored["status"] in {"queued", "running"}:
                    restored["status"] = "interrupted"
                    restored["ended_at"] = datetime.now(timezone.utc)
                    restored["exit_code"] = -1
                    restored["output"] = (
                        f"{restored['output']}\n[dashboard restart] job marked as interrupted"
                    ).strip()
                self._command_jobs[restored["job_id"]] = restored

        raw_logs = payload.get("logs", [])
        if isinstance(raw_logs, list):
            self._logs = [x for x in raw_logs if isinstance(x, dict)][-1000:]

        persisted_started_at = self._parse_dt_local(payload.get("trading_started_at"))
        self._trading_started_at = persisted_started_at

        raw_idempotency = payload.get("idempotency", {})
        if isinstance(raw_idempotency, dict):
            for key, value in raw_idempotency.items():
                if not isinstance(value, dict):
                    continue
                created_at = self._parse_dt_local(value.get("created_at"))
                if not created_at:
                    continue
                self._idempotency_records[key] = {
                    "fingerprint": str(value.get("fingerprint", "")),
                    "response": value.get("response", {}),
                    "status_code": int(value.get("status_code", 200)),
                    "created_at": created_at,
                }
        self._cleanup_idempotency_records()

    def _serialize_persistable_state(self) -> dict[str, Any]:
        """Serialize persisted control-plane state."""
        jobs: dict[str, dict[str, Any]] = {}
        for job_id, job in self._command_jobs.items():
            jobs[job_id] = {
                "job_id": job["job_id"],
                "command": job["command"],
                "args": job["args"],
                "status": job["status"],
                "created_at": self._to_iso_local(job.get("created_at")),
                "started_at": self._to_iso_local(job.get("started_at")),
                "ended_at": self._to_iso_local(job.get("ended_at")),
                "exit_code": job.get("exit_code"),
                "output": str(job.get("output", ""))[-20000:],
            }

        idempotency: dict[str, dict[str, Any]] = {}
        for key, value in self._idempotency_records.items():
            idempotency[key] = {
                "fingerprint": value.get("fingerprint", ""),
                "response": value.get("response", {}),
                "status_code": value.get("status_code", 200),
                "created_at": self._to_iso_local(value.get("created_at")),
            }

        return {
            "command_jobs": jobs,
            "idempotency": idempotency,
            "logs": self._logs[-300:],
            "trading_started_at": self._to_iso_local(self._trading_started_at),
            "last_update": self._to_iso_local(self._last_update),
        }

    def _persist_control_state(self) -> None:
        """Persist runtime control-plane state."""
        if not self._runtime_persistence_enabled:
            return
        self._cleanup_idempotency_records()
        self._state_store.save(self._serialize_persistable_state())

    def _cleanup_idempotency_records(self, ttl_seconds: int = 3600) -> None:
        """Remove stale idempotency records."""
        now = datetime.now(timezone.utc)
        stale_keys = []
        for key, value in self._idempotency_records.items():
            created_at = value.get("created_at")
            if not isinstance(created_at, datetime):
                stale_keys.append(key)
                continue
            if (now - created_at).total_seconds() > ttl_seconds:
                stale_keys.append(key)
        for key in stale_keys:
            self._idempotency_records.pop(key, None)

    def check_idempotency(
        self,
        action: str,
        key: str | None,
        fingerprint: str,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Check idempotency cache.

        Returns (cached_response, conflict_flag).
        """
        if not key:
            return None, False
        storage_key = f"{action}:{key}"
        self._cleanup_idempotency_records()
        cached = self._idempotency_records.get(storage_key)
        if not cached:
            return None, False
        if cached.get("fingerprint") != fingerprint:
            return None, True
        payload = cached.get("response")
        return payload if isinstance(payload, dict) else None, False

    def store_idempotency(
        self,
        action: str,
        key: str | None,
        fingerprint: str,
        response_payload: dict[str, Any],
        status_code: int = 200,
    ) -> None:
        """Store idempotent response payload."""
        if not key:
            return
        storage_key = f"{action}:{key}"
        self._idempotency_records[storage_key] = {
            "fingerprint": fingerprint,
            "response": response_payload,
            "status_code": status_code,
            "created_at": datetime.now(timezone.utc),
        }
        self._persist_control_state()

    def bind_event_bus(self, event_bus: EventBus) -> None:
        """Bind the shared event bus for control-plane publishing."""
        self._event_bus = event_bus

    async def handle_event(self, event: Event) -> None:
        """Handle incoming system events and map them to dashboard state."""
        event_type = event.event_type.value

        if event_type.startswith("portfolio.") or event_type == EventType.PORTFOLIO_REBALANCED.value:
            if "symbol" in event.data and "quantity" in event.data:
                self.update_position(event.data["symbol"], event.data)
            self.update_portfolio(event.data)
            await broadcast_portfolio_update(self._portfolio_data)
            return

        if event_type in {
            EventType.ORDER_SUBMITTED.value,
            EventType.ORDER_ACCEPTED.value,
            EventType.ORDER_FILLED.value,
            EventType.ORDER_PARTIAL.value,
            EventType.ORDER_CANCELLED.value,
            EventType.ORDER_REJECTED.value,
            EventType.ORDER_EXPIRED.value,
        }:
            self.update_order(event.data)
            await broadcast_order_update(event.data)
            return

        if event_type == EventType.SIGNAL_GENERATED.value:
            self.add_signal(event.data)
            await broadcast_signal(event.data)
            return

        if event_type in {
            EventType.KILL_SWITCH_TRIGGERED.value,
            EventType.LIMIT_BREACH.value,
            EventType.DRAWDOWN_WARNING.value,
            EventType.SYSTEM_ALERT.value,
            EventType.SYSTEM_ERROR.value,
        }:
            alert_payload = {
                "event_type": event_type,
                "timestamp": event.timestamp.isoformat(),
                "message": event.data.get("message") or event.data.get("reason", event_type),
                "data": event.data,
            }
            await broadcast_alert(alert_payload)
            self.add_log(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "level": "WARNING",
                    "category": "RISK" if event_type.startswith("risk.") else "SYSTEM",
                    "message": alert_payload["message"],
                    "extra": event.data,
                }
            )
            return

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
        self._last_update = datetime.now(timezone.utc)

    def update_order(self, order_or_id: str | dict[str, Any], updates: dict[str, Any] | None = None) -> None:
        """Update an order, or upsert when given a full order payload."""
        if isinstance(order_or_id, dict):
            payload = dict(order_or_id)
            order_id = payload.get("order_id") or payload.get("id")
            if not order_id:
                return
            updates = payload
        else:
            order_id = order_or_id
            updates = updates or {}

        for order in self._orders:
            if order.get("order_id") == order_id or order.get("id") == order_id:
                order.update(updates)
                break
        else:
            new_order = {"order_id": order_id, **updates}
            self.add_order(new_order)

        self._last_update = datetime.now(timezone.utc)

    def add_signal(self, signal: dict[str, Any]) -> None:
        """Add a signal."""
        self._signals.append(signal)
        if len(self._signals) > 500:
            self._signals = self._signals[-500:]
        self._last_update = datetime.now(timezone.utc)

    def update_model_status(self, model_name: str, status: dict[str, Any]) -> None:
        """Update model status."""
        self._model_status[model_name] = status
        self._last_update = datetime.now(timezone.utc)

    def add_log(self, log: dict[str, Any]) -> None:
        """Add a log entry."""
        self._logs.append(log)
        if len(self._logs) > 1000:
            self._logs = self._logs[-1000:]
        self._last_update = datetime.now(timezone.utc)
        self._persist_control_state()

    def set_component_health(self, component: str, healthy: bool) -> None:
        """Set component health status."""
        self._components_healthy[component] = healthy
        self._last_update = datetime.now(timezone.utc)

    def get_trading_status(self) -> dict[str, Any]:
        """Get the currently tracked trading process status."""
        running = bool(self._trading_process and self._trading_process.poll() is None)
        return {
            "running": running,
            "pid": self._trading_process.pid if running and self._trading_process else None,
            "started_at": self._trading_started_at.isoformat() if self._trading_started_at else None,
        }

    def set_trading_process(self, process: subprocess.Popen[str]) -> None:
        """Track active trading process."""
        self._trading_process = process
        self._trading_started_at = datetime.now(timezone.utc)
        self._persist_control_state()

    def clear_trading_process(self) -> None:
        """Clear tracked trading process."""
        self._trading_process = None
        self._trading_started_at = None
        self._persist_control_state()

    def create_job(self, command: str, args: list[str]) -> dict[str, Any]:
        """Create an async command job entry."""
        now = datetime.now(timezone.utc)
        job_id = str(uuid4())
        job = {
            "job_id": job_id,
            "command": command,
            "args": args,
            "status": "queued",
            "created_at": now,
            "started_at": None,
            "ended_at": None,
            "exit_code": None,
            "output": "",
        }
        self._command_jobs[job_id] = job
        self._persist_control_state()
        return job

    def update_job(self, job_id: str, **updates: Any) -> dict[str, Any] | None:
        """Update an existing command job."""
        job = self._command_jobs.get(job_id)
        if not job:
            return None
        job.update(updates)
        self._persist_control_state()
        return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get a job by id."""
        return self._command_jobs.get(job_id)

    def get_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent jobs."""
        jobs = sorted(
            self._command_jobs.values(),
            key=lambda x: x["created_at"],
            reverse=True,
        )
        return jobs[:limit]


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


def _to_iso(dt: datetime | None) -> str | None:
    """Serialize datetime to ISO string."""
    return dt.isoformat() if dt else None


def _parse_iso_dt(value: str | None) -> datetime | None:
    """Best-effort parse of timestamp strings."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _serialize_job(job: dict[str, Any]) -> CommandJobResponse:
    """Serialize internal job structure to API response."""
    return CommandJobResponse(
        job_id=job["job_id"],
        command=job["command"],
        args=job["args"],
        status=job["status"],
        created_at=_to_iso(job["created_at"]) or "",
        started_at=_to_iso(job["started_at"]),
        ended_at=_to_iso(job["ended_at"]),
        exit_code=job["exit_code"],
        output=job["output"],
    )


def _build_request_fingerprint(user: str, payload: dict[str, Any]) -> str:
    """Build deterministic fingerprint for idempotent requests."""
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(f"{user}|{canonical}".encode("utf-8")).hexdigest()


def _check_idempotent_replay(
    action: str,
    user: str,
    idempotency_key: str | None,
    payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Check idempotency cache and return cached payload if available."""
    fingerprint = _build_request_fingerprint(user, payload)
    state = get_dashboard_state()
    cached, conflict = state.check_idempotency(action, idempotency_key, fingerprint)
    if conflict:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Idempotency key already used with a different payload",
        )
    return cached, fingerprint


def _store_idempotent_response(
    action: str,
    idempotency_key: str | None,
    fingerprint: str,
    response_payload: dict[str, Any],
) -> None:
    """Store idempotent response payload."""
    state = get_dashboard_state()
    state.store_idempotency(
        action=action,
        key=idempotency_key,
        fingerprint=fingerprint,
        response_payload=response_payload,
        status_code=200,
    )


JOB_COMMAND_CATALOG: dict[str, dict[str, Any]] = {
    "backtest": {
        "summary": "Historical simulation and attribution run.",
        "script_path": "scripts/backtest.py",
        "sample_args": ["--start", "2025-01-01", "--end", "2025-12-31", "--symbols", "AAPL", "MSFT"],
        "risk_level": "medium",
    },
    "train": {
        "summary": "Model training with validation gates.",
        "script_path": "scripts/train.py",
        "sample_args": ["--model", "xgboost", "--symbols", "AAPL", "MSFT"],
        "risk_level": "medium",
    },
    "data": {
        "summary": "Data loading, validation, and export workflows.",
        "script_path": "scripts/data.py",
        "sample_args": ["validate", "--symbols", "AAPL", "MSFT"],
        "risk_level": "low",
    },
    "features": {
        "summary": "Feature pipeline computation and diagnostics.",
        "script_path": "scripts/features.py",
        "sample_args": ["compute", "--symbols", "AAPL", "MSFT"],
        "risk_level": "low",
    },
    "health": {
        "summary": "Operational health and dependency diagnostics.",
        "script_path": "scripts/health.py",
        "sample_args": ["check", "--full"],
        "risk_level": "low",
    },
    "deploy": {
        "summary": "Environment, docker, and deployment checks.",
        "script_path": "scripts/deploy.py",
        "sample_args": ["env", "check"],
        "risk_level": "high",
    },
}


def _build_command_catalog() -> list[CommandCatalogEntryResponse]:
    """Build sorted command catalog from allowlisted job definitions."""
    records: list[CommandCatalogEntryResponse] = []
    for command, metadata in sorted(JOB_COMMAND_CATALOG.items(), key=lambda x: x[0]):
        rel_path = str(metadata.get("script_path", "")).replace("\\", "/")
        script_exists = bool(rel_path) and (PROJECT_ROOT / rel_path).exists()
        records.append(
            CommandCatalogEntryResponse(
                command=command,
                summary=str(metadata.get("summary", "")),
                script_path=rel_path,
                script_exists=script_exists,
                sample_args=[str(x) for x in metadata.get("sample_args", [])],
                risk_level=str(metadata.get("risk_level", "low")),
            )
        )
    return records


def _extract_script_entrypoint(script_path: Path) -> str | None:
    """Best-effort detection of `run_*` entrypoint in scripts file."""
    try:
        source = script_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    match = re.search(r"def\s+(run_[A-Za-z0-9_]+)\s*\(", source)
    return match.group(1) if match else None


def _build_script_inventory() -> list[ScriptInventoryEntryResponse]:
    """Build scripts directory inventory for dashboard coverage view."""
    scripts_root = PROJECT_ROOT / "scripts"
    if not scripts_root.exists():
        return []

    command_by_script = {
        str(metadata.get("script_path", "")).replace("\\", "/"): command
        for command, metadata in JOB_COMMAND_CATALOG.items()
    }

    records: list[ScriptInventoryEntryResponse] = []
    for script_path in sorted(scripts_root.glob("*.py")):
        if script_path.name.startswith("__"):
            continue
        rel_path = script_path.relative_to(PROJECT_ROOT).as_posix()
        records.append(
            ScriptInventoryEntryResponse(
                script_name=script_path.stem,
                path=rel_path,
                exists=script_path.exists(),
                linked_command=command_by_script.get(rel_path),
                entrypoint=_extract_script_entrypoint(script_path),
            )
        )
    return records


def _build_domain_coverage() -> list[DomainCoverageEntryResponse]:
    """Summarize quant_trading_system package/domain module coverage."""
    package_root = PROJECT_ROOT / "quant_trading_system"
    if not package_root.exists():
        return []

    records: list[DomainCoverageEntryResponse] = []
    for domain_path in sorted(package_root.iterdir(), key=lambda x: x.name):
        if not domain_path.is_dir() or domain_path.name.startswith("_"):
            continue
        if not (domain_path / "__init__.py").exists():
            continue

        module_count = len([p for p in domain_path.rglob("*.py") if p.name != "__init__.py"])
        subpackage_count = len(
            [p for p in domain_path.iterdir() if p.is_dir() and (p / "__init__.py").exists()]
        )
        records.append(
            DomainCoverageEntryResponse(
                domain=domain_path.name,
                path=domain_path.relative_to(PROJECT_ROOT).as_posix(),
                module_count=module_count,
                subpackage_count=subpackage_count,
            )
        )
    return records


def _build_data_coverage() -> DataCoverageResponse:
    """Summarize local data assets for operator situational awareness."""
    data_root = PROJECT_ROOT / "data"
    if not data_root.exists():
        return DataCoverageResponse(
            root="data",
            exists=False,
            total_files=0,
            csv_files=0,
            parquet_files=0,
            json_files=0,
            other_files=0,
        )

    counts = {
        "total_files": 0,
        "csv_files": 0,
        "parquet_files": 0,
        "json_files": 0,
        "other_files": 0,
    }
    for candidate in data_root.rglob("*"):
        if not candidate.is_file():
            continue
        counts["total_files"] += 1
        suffix = candidate.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            counts["csv_files"] += 1
        elif suffix in {".parquet", ".pq", ".feather"}:
            counts["parquet_files"] += 1
        elif suffix in {".json", ".jsonl", ".ndjson"}:
            counts["json_files"] += 1
        else:
            counts["other_files"] += 1

    return DataCoverageResponse(
        root=data_root.relative_to(PROJECT_ROOT).as_posix(),
        exists=True,
        total_files=counts["total_files"],
        csv_files=counts["csv_files"],
        parquet_files=counts["parquet_files"],
        json_files=counts["json_files"],
        other_files=counts["other_files"],
    )


def _build_main_command(command: str, args: list[str]) -> list[str]:
    """Build executable CLI command with safety allowlist."""
    if command not in JOB_COMMAND_CATALOG:
        raise HTTPException(status_code=400, detail=f"Unsupported command: {command}")
    return [sys.executable, "main.py", command, *args]


async def _run_command_job(job_id: str) -> None:
    """Background worker for async command execution."""
    state = get_dashboard_state()
    job = state.get_job(job_id)
    if not job:
        return

    state.update_job(job_id, status="running", started_at=datetime.now(timezone.utc))
    cmd = _build_main_command(job["command"], job["args"])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        state.add_log(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "INFO",
                "category": "SYSTEM",
                "message": f"Started job {job_id}: {' '.join(cmd)}",
                "extra": {"job_id": job_id},
            }
        )

        output_bytes, _ = await proc.communicate()
        output = output_bytes.decode("utf-8", errors="replace")[-20000:]

        status_value = "succeeded" if proc.returncode == 0 else "failed"
        state.update_job(
            job_id,
            status=status_value,
            ended_at=datetime.now(timezone.utc),
            exit_code=proc.returncode,
            output=output,
        )
    except asyncio.CancelledError:
        state.update_job(
            job_id,
            status="cancelled",
            ended_at=datetime.now(timezone.utc),
            exit_code=-1,
        )
        raise
    except Exception as exc:
        state.update_job(
            job_id,
            status="failed",
            ended_at=datetime.now(timezone.utc),
            exit_code=1,
            output=f"Job execution failed: {exc}",
        )


def _calculate_tca_from_orders(orders: list[dict[str, Any]]) -> TCAResponse:
    """Compute TCA metrics from current order history."""
    now = datetime.now(timezone.utc).isoformat()
    if not orders:
        return TCAResponse(
            timestamp=now,
            slippage_bps=0.0,
            market_impact_bps=0.0,
            execution_speed_ms=0.0,
            fill_probability=0.0,
            venue_breakdown={},
            cost_savings_vs_vwap=0.0,
        )

    total = len(orders)
    filled = 0
    slippage_values: list[float] = []
    impact_values: list[float] = []
    speed_values: list[float] = []
    venue_count: dict[str, int] = {}
    savings_values: list[float] = []

    for order in orders:
        venue = str(order.get("venue") or order.get("exchange") or "UNKNOWN").upper()
        venue_count[venue] = venue_count.get(venue, 0) + 1

        status_value = str(order.get("status", "")).upper()
        if status_value in {"FILLED", "PARTIAL_FILLED"}:
            filled += 1

        slip = order.get("slippage_bps")
        if isinstance(slip, (int, float)):
            slippage_values.append(float(slip))

        impact = order.get("market_impact_bps")
        if isinstance(impact, (int, float)):
            impact_values.append(float(impact))

        savings = order.get("savings_vs_vwap_bps")
        if isinstance(savings, (int, float)):
            savings_values.append(float(savings))

        created_at = _parse_iso_dt(order.get("created_at"))
        filled_at = _parse_iso_dt(order.get("filled_at") or order.get("updated_at"))
        if created_at and filled_at and filled_at >= created_at:
            speed_values.append((filled_at - created_at).total_seconds() * 1000.0)

    venue_breakdown = {
        venue: count / total
        for venue, count in sorted(venue_count.items(), key=lambda x: x[1], reverse=True)
    }

    avg_slip = sum(slippage_values) / len(slippage_values) if slippage_values else 0.0
    avg_impact = sum(impact_values) / len(impact_values) if impact_values else 0.0
    avg_speed = sum(speed_values) / len(speed_values) if speed_values else 0.0
    avg_savings = sum(savings_values) / len(savings_values) if savings_values else 0.0

    return TCAResponse(
        timestamp=now,
        slippage_bps=round(avg_slip, 4),
        market_impact_bps=round(avg_impact, 4),
        execution_speed_ms=round(avg_speed, 2),
        fill_probability=round(filled / total, 6) if total else 0.0,
        venue_breakdown=venue_breakdown,
        cost_savings_vs_vwap=round(avg_savings, 4),
    )


def _percentile(values: list[float], pct: float) -> float:
    """Compute percentile for non-empty values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(0.0, min(1.0, pct)) * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _calculate_execution_quality(orders: list[dict[str, Any]]) -> ExecutionQualityResponse:
    """Compute execution quality decomposition from order stream."""
    now = datetime.now(timezone.utc).isoformat()
    if not orders:
        return ExecutionQualityResponse(
            timestamp=now,
            arrival_price_delta_bps=0.0,
            rejection_rate=0.0,
            fill_rate_buy=0.0,
            fill_rate_sell=0.0,
            venue_slippage_buckets={},
            latency_distribution_ms={"p50": 0.0, "p90": 0.0, "p99": 0.0},
        )

    arrival_deltas: list[float] = []
    latencies: list[float] = []
    venue_slippage: dict[str, list[float]] = {}
    total_orders = len(orders)
    rejected = 0
    side_totals = {"BUY": 0, "SELL": 0}
    side_filled = {"BUY": 0, "SELL": 0}

    for order in orders:
        side = str(order.get("side", "")).upper()
        if side in side_totals:
            side_totals[side] += 1

        status_value = str(order.get("status", "")).upper()
        if status_value in {"REJECTED", "CANCELLED"}:
            rejected += 1
        if status_value in {"FILLED", "PARTIAL_FILLED", "PARTIAL"} and side in side_filled:
            side_filled[side] += 1

        arrival_delta = order.get("arrival_price_delta_bps")
        if isinstance(arrival_delta, (int, float)):
            arrival_deltas.append(float(arrival_delta))
        else:
            arrival_price = order.get("arrival_price")
            executed_price = order.get("filled_avg_price") or order.get("executed_price")
            if isinstance(arrival_price, (int, float)) and isinstance(executed_price, (int, float)) and arrival_price:
                bps = ((float(executed_price) - float(arrival_price)) / float(arrival_price)) * 10000.0
                arrival_deltas.append(bps)

        created_at = _parse_iso_dt(order.get("created_at"))
        finished_at = _parse_iso_dt(order.get("filled_at") or order.get("updated_at"))
        if created_at and finished_at and finished_at >= created_at:
            latencies.append((finished_at - created_at).total_seconds() * 1000.0)

        venue = str(order.get("venue") or order.get("exchange") or "UNKNOWN").upper()
        slip = order.get("slippage_bps")
        if isinstance(slip, (int, float)):
            venue_slippage.setdefault(venue, []).append(float(slip))

    venue_slippage_buckets = {
        venue: round(sum(values) / len(values), 4)
        for venue, values in sorted(venue_slippage.items(), key=lambda x: len(x[1]), reverse=True)
    }

    fill_rate_buy = (side_filled["BUY"] / side_totals["BUY"]) if side_totals["BUY"] else 0.0
    fill_rate_sell = (side_filled["SELL"] / side_totals["SELL"]) if side_totals["SELL"] else 0.0

    return ExecutionQualityResponse(
        timestamp=now,
        arrival_price_delta_bps=round(sum(arrival_deltas) / len(arrival_deltas), 4) if arrival_deltas else 0.0,
        rejection_rate=round(rejected / total_orders, 6) if total_orders else 0.0,
        fill_rate_buy=round(fill_rate_buy, 6),
        fill_rate_sell=round(fill_rate_sell, 6),
        venue_slippage_buckets=venue_slippage_buckets,
        latency_distribution_ms={
            "p50": round(_percentile(latencies, 0.50), 3),
            "p90": round(_percentile(latencies, 0.90), 3),
            "p99": round(_percentile(latencies, 0.99), 3),
        },
    )


def _calculate_var_from_portfolio(equity: float, drawdown: float) -> VaRResponse:
    """Compute VaR/CVaR and a synthetic probability curve from portfolio state."""
    now = datetime.now(timezone.utc).isoformat()
    safe_equity = max(equity, 1.0)
    sigma = max(0.005, min(0.08, drawdown * 1.5 + 0.015))

    z95 = 1.6448536269514722
    z99 = 2.3263478740408408
    var_95 = safe_equity * sigma * z95
    var_99 = safe_equity * sigma * z99
    cvar_95 = safe_equity * sigma * 2.0627128075074257

    points = []
    std_abs = safe_equity * sigma
    for i in range(-24, 25):
        pnl = i * std_abs / 4
        x = pnl / std_abs
        prob = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-(x * x) / 2)
        points.append({"pnl": float(pnl), "probability": float(prob)})

    stress = {
        "GFC_2008": round(-(sigma * 6.0 * 100), 2),
        "Covid_2020": round(-(sigma * 5.0 * 100), 2),
        "Rates_Shock": round(-(sigma * 2.2 * 100), 2),
        "Liquidity_Event": round(-(sigma * 3.3 * 100), 2),
    }

    return VaRResponse(
        timestamp=now,
        var_95=round(var_95, 2),
        var_99=round(var_99, 2),
        cvar_95=round(cvar_95, 2),
        stress_scenarios=stress,
        distribution_curve=points,
    )


def _calculate_risk_concentration(
    positions: dict[str, dict[str, Any]],
    portfolio: dict[str, Any],
) -> RiskConcentrationResponse:
    """Compute concentration risk decomposition."""
    now = datetime.now(timezone.utc).isoformat()
    if not positions:
        return RiskConcentrationResponse(
            timestamp=now,
            largest_symbol_pct=0.0,
            top3_symbols_pct=0.0,
            hhi_symbol=0.0,
            hhi_sector=0.0,
            symbol_weights={},
            sector_weights={},
        )

    absolute_values: dict[str, float] = {}
    sector_values: dict[str, float] = {}
    for symbol, pos in positions.items():
        market_value = pos.get("market_value")
        if not isinstance(market_value, (int, float)):
            qty = float(pos.get("quantity", 0.0))
            px = float(pos.get("current_price", pos.get("avg_entry_price", 0.0)))
            market_value = qty * px
        abs_value = abs(float(market_value))
        absolute_values[symbol] = abs_value
        sector = str(pos.get("sector") or "UNCLASSIFIED").upper()
        sector_values[sector] = sector_values.get(sector, 0.0) + abs_value

    total = sum(absolute_values.values())
    if total <= 0:
        total = max(float(portfolio.get("gross_exposure", 1.0)), 1.0)

    symbol_weights = {k: v / total for k, v in sorted(absolute_values.items(), key=lambda x: x[1], reverse=True)}
    sector_weights = {k: v / total for k, v in sorted(sector_values.items(), key=lambda x: x[1], reverse=True)}

    ordered_symbol_weights = list(symbol_weights.values())
    top3 = sum(ordered_symbol_weights[:3]) if ordered_symbol_weights else 0.0
    hhi_symbol = sum(w * w for w in symbol_weights.values())
    hhi_sector = sum(w * w for w in sector_weights.values())

    return RiskConcentrationResponse(
        timestamp=now,
        largest_symbol_pct=round((ordered_symbol_weights[0] if ordered_symbol_weights else 0.0) * 100.0, 4),
        top3_symbols_pct=round(top3 * 100.0, 4),
        hhi_symbol=round(hhi_symbol, 6),
        hhi_sector=round(hhi_sector, 6),
        symbol_weights={k: round(v, 6) for k, v in symbol_weights.items()},
        sector_weights={k: round(v, 6) for k, v in sector_weights.items()},
    )


def _calculate_risk_correlation(
    positions: dict[str, dict[str, Any]],
    portfolio: dict[str, Any],
) -> RiskCorrelationResponse:
    """Compute synthetic correlation risk from position concentration."""
    now = datetime.now(timezone.utc).isoformat()
    symbols = list(positions.keys())
    if len(symbols) < 2:
        beta_exposure = float(portfolio.get("beta_exposure", 0.0) or 0.0)
        return RiskCorrelationResponse(
            timestamp=now,
            average_pairwise_correlation=0.0,
            cluster_risk_score=0.0,
            beta_weighted_exposure=round(beta_exposure, 6),
            matrix=[],
        )

    pairs: list[dict[str, Any]] = []
    values = []
    sector_lookup = {sym: str(positions[sym].get("sector") or "UNCLASSIFIED").upper() for sym in symbols}
    for idx, sym_a in enumerate(symbols):
        for sym_b in symbols[idx + 1 :]:
            seed = int(hashlib.sha256(f"{sym_a}|{sym_b}".encode("utf-8")).hexdigest()[:8], 16)
            corr = 0.20 + ((seed % 5600) / 10000.0)  # 0.20 .. 0.76
            if sector_lookup.get(sym_a) == sector_lookup.get(sym_b):
                corr = min(0.95, corr + 0.14)
            corr = round(corr, 4)
            values.append(corr)
            if len(pairs) < 20:
                pairs.append({"symbol_a": sym_a, "symbol_b": sym_b, "correlation": corr})

    avg_corr = sum(values) / len(values) if values else 0.0
    sector_weights = _calculate_risk_concentration(positions, portfolio).sector_weights
    max_sector_weight = max(sector_weights.values()) if sector_weights else 0.0
    cluster_score = min(1.0, avg_corr * (1.0 + max_sector_weight))

    total_abs_mv = 0.0
    beta_weighted = 0.0
    for pos in positions.values():
        mv = abs(float(pos.get("market_value", 0.0) or 0.0))
        beta = float(pos.get("beta", portfolio.get("beta_exposure", 1.0) or 1.0))
        total_abs_mv += mv
        beta_weighted += beta * mv
    beta_weighted_exposure = (beta_weighted / total_abs_mv) if total_abs_mv else float(portfolio.get("beta_exposure", 0.0) or 0.0)

    return RiskCorrelationResponse(
        timestamp=now,
        average_pairwise_correlation=round(avg_corr, 6),
        cluster_risk_score=round(cluster_score, 6),
        beta_weighted_exposure=round(beta_weighted_exposure, 6),
        matrix=pairs,
    )


def _calculate_risk_stress(portfolio: dict[str, Any]) -> RiskStressResponse:
    """Compute stress scenarios from current portfolio state."""
    now = datetime.now(timezone.utc).isoformat()
    equity = max(float(portfolio.get("equity", 0.0) or 0.0), 1.0)
    gross_exposure = float(portfolio.get("gross_exposure", 0.0) or 0.0)
    drawdown = max(float(portfolio.get("current_drawdown", 0.0) or 0.0), 0.0)
    beta = abs(float(portfolio.get("beta_exposure", 1.0) or 1.0))
    leverage = max(gross_exposure / equity, 0.0)

    scenario_pct = {
        "equity_gap_down": -(0.07 + drawdown * 1.4 + leverage * 0.03),
        "rates_shock_parallel_up": -(0.02 + beta * 0.01 + leverage * 0.01),
        "credit_spread_widening": -(0.03 + drawdown * 0.8 + leverage * 0.015),
        "liquidity_freeze": -(0.04 + leverage * 0.025 + drawdown * 0.6),
    }

    scenario_loss = {name: round(equity * pct, 2) for name, pct in scenario_pct.items()}
    worst_case = min(scenario_loss.values()) if scenario_loss else 0.0
    resilience = max(0.0, 100.0 - (abs(worst_case) / equity) * 180.0)

    return RiskStressResponse(
        timestamp=now,
        scenarios=scenario_loss,
        worst_case_loss=round(worst_case, 2),
        resilience_score=round(resilience, 4),
    )


def _calculate_risk_attribution(
    portfolio: dict[str, Any],
    orders: list[dict[str, Any]],
    positions: dict[str, dict[str, Any]],
) -> RiskAttributionResponse:
    """Build pre-trade and post-trade risk attribution views."""
    now = datetime.now(timezone.utc).isoformat()
    equity = max(float(portfolio.get("equity", 0.0) or 0.0), 1.0)
    gross_exposure = float(portfolio.get("gross_exposure", 0.0) or 0.0)
    leverage = gross_exposure / equity if equity else 0.0

    concentration = _calculate_risk_concentration(positions, portfolio)
    concentration_pct = concentration.largest_symbol_pct / 100.0

    pre_trade_checks: list[dict[str, Any]] = []
    post_trade_findings: list[dict[str, Any]] = []

    for order in orders[-50:]:
        qty = float(order.get("quantity", 0.0) or 0.0)
        price = float(order.get("limit_price") or order.get("filled_avg_price") or order.get("current_price") or 0.0)
        notional = abs(qty * price)
        notional_limit_ok = notional <= equity * 0.20
        leverage_ok = leverage <= 2.0
        concentration_ok = concentration_pct <= 0.30

        status_value = "pass" if (notional_limit_ok and leverage_ok and concentration_ok) else "breach"
        pre_trade_checks.append(
            {
                "order_id": str(order.get("order_id") or ""),
                "symbol": str(order.get("symbol") or ""),
                "notional": round(notional, 4),
                "notional_limit_ok": notional_limit_ok,
                "leverage_ok": leverage_ok,
                "concentration_ok": concentration_ok,
                "status": status_value,
            }
        )

        order_status = str(order.get("status", "")).upper()
        slippage = float(order.get("slippage_bps", 0.0) or 0.0)
        if order_status in {"REJECTED", "CANCELLED"} or abs(slippage) > 30:
            finding_type = "execution_rejection" if order_status in {"REJECTED", "CANCELLED"} else "slippage_spike"
            post_trade_findings.append(
                {
                    "order_id": str(order.get("order_id") or ""),
                    "symbol": str(order.get("symbol") or ""),
                    "finding": finding_type,
                    "status": "breach",
                    "details": {
                        "order_status": order_status,
                        "slippage_bps": slippage,
                    },
                }
            )

    breaches_count = sum(1 for x in pre_trade_checks if x["status"] == "breach") + len(post_trade_findings)

    return RiskAttributionResponse(
        timestamp=now,
        pre_trade_checks=pre_trade_checks[-25:],
        post_trade_findings=post_trade_findings[-25:],
        breaches_count=breaches_count,
    )


def _load_latest_model_explainability() -> FeatureImportanceResponse:
    """Load explainability from latest model artifacts if available."""
    models_dir = PROJECT_ROOT / "models"
    candidate_files = sorted(
        models_dir.glob("*_artifacts.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    now = datetime.now(timezone.utc).isoformat()
    if not candidate_files:
        return FeatureImportanceResponse(
            timestamp=now,
            model_name="unknown",
            global_importance={},
            recent_shift={},
        )

    latest = candidate_files[0]
    try:
        with latest.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        metrics = payload.get("training_metrics", {}) if isinstance(payload, dict) else {}
        shap_data = metrics.get("shap_importance", {})
        global_importance = (
            {str(k): float(v) for k, v in shap_data.items()}
            if isinstance(shap_data, dict)
            else {}
        )
        sorted_items = sorted(global_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        recent_shift = {name: round(value * 0.05, 6) for name, value in sorted_items}

        return FeatureImportanceResponse(
            timestamp=now,
            model_name=str(payload.get("model_name") or latest.stem.replace("_artifacts", "")),
            global_importance=global_importance,
            recent_shift=recent_shift,
        )
    except Exception as exc:
        logger.warning(f"Failed to load explainability artifacts from {latest}: {exc}")
        return FeatureImportanceResponse(
            timestamp=now,
            model_name="unknown",
            global_importance={},
            recent_shift={},
        )


def _load_model_registry_entries() -> list[dict[str, Any]]:
    """Load flattened model registry entries."""
    if not MODEL_REGISTRY_FILE.exists():
        return []
    try:
        with MODEL_REGISTRY_FILE.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return []
        entries: list[dict[str, Any]] = []
        for model_name, versions in payload.items():
            if not isinstance(versions, list):
                continue
            for version in versions:
                if not isinstance(version, dict):
                    continue
                item = dict(version)
                item["model_name"] = str(item.get("model_name") or model_name)
                entries.append(item)
        entries.sort(
            key=lambda x: _parse_iso_dt(str(x.get("registered_at") or "")) or datetime.fromtimestamp(0, tz=timezone.utc),
            reverse=True,
        )
        return entries
    except Exception as exc:
        logger.warning(f"Failed to load model registry: {exc}")
        return []


def _latest_model_artifact_payload() -> tuple[dict[str, Any], str]:
    """Load latest model artifact payload and resolve model name."""
    models_dir = PROJECT_ROOT / "models"
    artifacts = sorted(models_dir.glob("*_artifacts.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not artifacts:
        return {}, "unknown"
    latest = artifacts[0]
    try:
        with latest.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return {}, latest.stem.replace("_artifacts", "")
        model_name = str(payload.get("model_name") or latest.stem.replace("_artifacts", ""))
        return payload, model_name
    except Exception as exc:
        logger.warning(f"Failed to read model artifact {latest}: {exc}")
        return {}, latest.stem.replace("_artifacts", "")


def _get_active_model_pointer(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve current champion pointer from file or registry active flag."""
    pointer: dict[str, Any] = {}
    if MODEL_ACTIVE_POINTER_FILE.exists():
        try:
            with MODEL_ACTIVE_POINTER_FILE.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                pointer = payload
        except Exception as exc:
            logger.warning(f"Failed to read active model pointer: {exc}")

    if pointer.get("model_name") and pointer.get("version_id"):
        return pointer

    for entry in entries:
        if bool(entry.get("is_active", False)):
            return {
                "model_name": str(entry.get("model_name", "")),
                "version_id": str(entry.get("version_id", "")),
                "updated_at": str(entry.get("registered_at", "")),
                "updated_by": "registry",
                "reason": "active_flag",
            }

    if entries:
        first = entries[0]
        return {
            "model_name": str(first.get("model_name", "")),
            "version_id": str(first.get("version_id", "")),
            "updated_at": str(first.get("registered_at", "")),
            "updated_by": "registry",
            "reason": "latest_version",
        }
    return {"model_name": None, "version_id": None, "updated_at": None, "updated_by": None, "reason": "none"}


def _save_active_model_pointer(
    model_name: str,
    version_id: str,
    updated_by: str,
    reason: str,
) -> None:
    """Persist active model pointer for champion/challenger control."""
    MODEL_ACTIVE_POINTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "version_id": version_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "updated_by": updated_by,
        "reason": reason,
    }
    with MODEL_ACTIVE_POINTER_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _build_model_registry_snapshot() -> ModelRegistryResponse:
    """Build model registry snapshot response."""
    entries = _load_model_registry_entries()
    active = _get_active_model_pointer(entries)
    serialized = [
        ModelRegistryEntryResponse(
            model_name=str(entry.get("model_name", "")),
            version_id=str(entry.get("version_id", "")),
            model_version=str(entry.get("model_version", "")),
            model_type=str(entry.get("model_type", "")),
            registered_at=str(entry.get("registered_at", "")),
            metrics=entry.get("metrics", {}) if isinstance(entry.get("metrics"), dict) else {},
            tags=[str(x) for x in entry.get("tags", [])] if isinstance(entry.get("tags"), list) else [],
            is_active=bool(entry.get("is_active", False)),
            path=str(entry.get("path", "")),
        )
        for entry in entries[:300]
    ]
    model_names = {entry.model_name for entry in serialized if entry.model_name}
    return ModelRegistryResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_count=len(model_names),
        versions_count=len(serialized),
        active_model=str(active.get("model_name")) if active.get("model_name") else None,
        entries=serialized,
    )


def _build_model_drift_snapshot() -> ModelDriftResponse:
    """Build model drift response from explainability/staleness artifacts."""
    explainability = _load_latest_model_explainability()
    shift_values = list(explainability.recent_shift.values())
    drift_score = sum(abs(v) for v in shift_values)
    if drift_score >= 0.20:
        drift_status = "critical"
    elif drift_score >= 0.10:
        drift_status = "warning"
    else:
        drift_status = "stable"

    staleness_reason = None
    recommendation = "continue_monitoring"
    staleness_file = PROJECT_ROOT / "models" / "staleness_status.json"
    if staleness_file.exists():
        try:
            with staleness_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                staleness_reason = payload.get("reason")
                recommendation = str(payload.get("recommendation") or recommendation)
        except Exception as exc:
            logger.warning(f"Failed to read staleness status: {exc}")
    else:
        if drift_status == "critical":
            recommendation = "quarantine_model_and_retrain"
        elif drift_status == "warning":
            recommendation = "increase_monitoring_and_run_backtest"

    return ModelDriftResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_name=explainability.model_name,
        drift_score=round(drift_score, 6),
        drift_status=drift_status,
        staleness_reason=str(staleness_reason) if staleness_reason else None,
        recommendation=recommendation,
        feature_shift={k: round(v, 6) for k, v in explainability.recent_shift.items()},
    )


def _build_model_validation_gates_snapshot() -> ModelValidationGateResponse:
    """Build model validation gate decision from artifacts and risk state."""
    now = datetime.now(timezone.utc)
    artifact_payload, model_name = _latest_model_artifact_payload()
    state = get_dashboard_state()
    portfolio = state._portfolio_data
    drawdown = float(portfolio.get("current_drawdown", 0.0) or 0.0)

    metrics = artifact_payload.get("training_metrics", {}) if isinstance(artifact_payload, dict) else {}
    sharpe = float(metrics.get("sharpe_ratio", metrics.get("sharpe", 0.0)) or 0.0)
    auc = float(metrics.get("auc", 0.0) or 0.0)

    artifact_ts = None
    latest_files = sorted((PROJECT_ROOT / "models").glob("*_artifacts.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if latest_files:
        artifact_ts = datetime.fromtimestamp(latest_files[0].stat().st_mtime, tz=timezone.utc)
    artifact_fresh = bool(artifact_ts and (now - artifact_ts).total_seconds() <= 72 * 3600)

    drift = _build_model_drift_snapshot()
    drift_ok = drift.drift_score < 0.25
    sharpe_ok = sharpe >= 0.75
    auc_ok = auc >= 0.52 if auc > 0 else True
    risk_impact_ok = drawdown <= 0.12

    gates = [
        {"gate": "artifact_freshness", "passed": artifact_fresh, "value": artifact_ts.isoformat() if artifact_ts else None},
        {"gate": "drift_threshold", "passed": drift_ok, "value": drift.drift_score},
        {"gate": "minimum_sharpe", "passed": sharpe_ok, "value": sharpe},
        {"gate": "auc_floor", "passed": auc_ok, "value": auc},
        {"gate": "portfolio_risk_impact", "passed": risk_impact_ok, "value": drawdown},
    ]
    failed = [x["gate"] for x in gates if not x["passed"]]
    passed = not failed
    decision = "approved_for_promotion" if passed else "blocked_requires_review"

    return ModelValidationGateResponse(
        timestamp=now.isoformat(),
        model_name=model_name,
        passed=passed,
        gates=gates,
        failed_gates=failed,
        decision=decision,
    )


def _extract_metric(entry: dict[str, Any], key: str) -> float:
    """Extract numeric metric from registry entry."""
    metrics = entry.get("metrics", {})
    if not isinstance(metrics, dict):
        return 0.0
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _build_champion_challenger_snapshot() -> ModelChampionChallengerResponse:
    """Build champion/challenger comparison from model registry."""
    entries = _load_model_registry_entries()
    pointer = _get_active_model_pointer(entries)
    champion_name = pointer.get("model_name")
    champion_version = pointer.get("version_id")

    champion_entry = None
    for entry in entries:
        if entry.get("model_name") == champion_name and entry.get("version_id") == champion_version:
            champion_entry = entry
            break
    if not champion_entry and entries:
        champion_entry = entries[0]

    challenger_entry = None
    for entry in entries:
        if not champion_entry:
            challenger_entry = entry
            break
        if entry.get("version_id") != champion_entry.get("version_id"):
            challenger_entry = entry
            break

    champion_label = None
    challenger_label = None
    comparison: dict[str, float] = {}
    recommendation = "insufficient_data"

    if champion_entry:
        champion_label = f"{champion_entry.get('model_name')}:{champion_entry.get('version_id')}"
    if challenger_entry:
        challenger_label = f"{challenger_entry.get('model_name')}:{challenger_entry.get('version_id')}"

    if champion_entry and challenger_entry:
        c_sharpe = _extract_metric(champion_entry, "sharpe_ratio") or _extract_metric(champion_entry, "sharpe")
        ch_sharpe = _extract_metric(challenger_entry, "sharpe_ratio") or _extract_metric(challenger_entry, "sharpe")
        c_auc = _extract_metric(champion_entry, "auc")
        ch_auc = _extract_metric(challenger_entry, "auc")
        comparison = {
            "champion_sharpe": round(c_sharpe, 6),
            "challenger_sharpe": round(ch_sharpe, 6),
            "sharpe_delta": round(ch_sharpe - c_sharpe, 6),
            "champion_auc": round(c_auc, 6),
            "challenger_auc": round(ch_auc, 6),
            "auc_delta": round(ch_auc - c_auc, 6),
        }
        recommendation = "promote_challenger" if comparison["sharpe_delta"] > 0.05 else "retain_champion"
    elif champion_entry:
        recommendation = "retain_champion_no_challenger"

    return ModelChampionChallengerResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        champion=champion_label,
        challenger=challenger_label,
        comparison=comparison,
        recommendation=recommendation,
    )


def _compute_slo_status() -> SLOStatusResponse:
    """Compute SLO and burn-rate status from runtime health and incidents."""
    state = get_dashboard_state()
    manager = get_alert_manager()
    alerts = manager.get_dashboard_alerts(limit=500)
    unresolved = [a for a in alerts if a.status != AlertStatus.RESOLVED]
    critical = len([a for a in unresolved if a.severity == AlertSeverity.CRITICAL])
    high = len([a for a in unresolved if a.severity == AlertSeverity.HIGH])

    healthy_components = sum(1 for x in state._components_healthy.values() if x)
    total_components = max(len(state._components_healthy), 1)
    availability = healthy_components / total_components

    now = datetime.now(timezone.utc)
    logs_recent_1h = [
        log for log in state._logs
        if (ts := _parse_iso_dt(str(log.get("timestamp") or ""))) and (now - ts).total_seconds() <= 3600
    ]
    logs_recent_6h = [
        log for log in state._logs
        if (ts := _parse_iso_dt(str(log.get("timestamp") or ""))) and (now - ts).total_seconds() <= 21600
    ]
    err1 = len([x for x in logs_recent_1h if str(x.get("level", "")).upper() in {"ERROR", "CRITICAL"}])
    err6 = len([x for x in logs_recent_6h if str(x.get("level", "")).upper() in {"ERROR", "CRITICAL"}])

    burn_rate_1h = min(20.0, (critical * 2.5) + (high * 1.1) + (err1 * 0.2))
    burn_rate_6h = min(20.0, (critical * 1.8) + (high * 0.9) + (err6 * 0.08))

    latencies = []
    for job in state._command_jobs.values():
        started = job.get("started_at")
        ended = job.get("ended_at")
        if isinstance(started, datetime) and isinstance(ended, datetime) and ended >= started:
            latencies.append((ended - started).total_seconds() * 1000.0)
    p95 = _percentile(latencies, 0.95) if latencies else 250.0
    p99 = _percentile(latencies, 0.99) if latencies else 400.0

    error_budget_remaining = max(0.0, 100.0 - (burn_rate_1h * 2.2 + burn_rate_6h * 1.0))
    if burn_rate_1h >= 8 or burn_rate_6h >= 6:
        status_value = "critical"
    elif burn_rate_1h >= 4 or burn_rate_6h >= 3:
        status_value = "warning"
    else:
        status_value = "healthy"

    return SLOStatusResponse(
        timestamp=now.isoformat(),
        availability=round(availability, 6),
        error_budget_remaining_pct=round(error_budget_remaining, 4),
        p95_action_latency_ms=round(p95, 4),
        p99_action_latency_ms=round(p99, 4),
        burn_rate_1h=round(burn_rate_1h, 4),
        burn_rate_6h=round(burn_rate_6h, 4),
        status=status_value,
    )


def _build_incident_records(limit: int = 100) -> list[IncidentRecordResponse]:
    """Build incident records from alert history."""
    manager = get_alert_manager()
    history = manager.get_alert_history(limit=limit)
    incidents = sorted(history, key=lambda a: a.timestamp, reverse=True)
    return [
        IncidentRecordResponse(
            incident_id=str(alert.alert_id),
            severity=alert.severity.value,
            title=alert.title,
            status=alert.status.value,
            created_at=alert.timestamp.isoformat(),
            runbook_link=alert.runbook_link,
            suggested_action=alert.suggested_action,
        )
        for alert in incidents
    ]


def _build_incident_timeline(limit: int = 250) -> list[IncidentTimelineEventResponse]:
    """Build timeline by merging alerts, audit trail, and runtime logs."""
    state = get_dashboard_state()
    manager = get_alert_manager()
    events: list[IncidentTimelineEventResponse] = []

    for alert in manager.get_alert_history(limit=limit):
        events.append(
            IncidentTimelineEventResponse(
                timestamp=alert.timestamp.isoformat(),
                source="alerting",
                event_type=alert.alert_type.value,
                severity=alert.severity.value,
                message=alert.message,
                context=alert.context,
            )
        )

    for audit in _audit_signer.list_recent(limit=limit):
        events.append(
            IncidentTimelineEventResponse(
                timestamp=str(audit.get("timestamp", "")),
                source="audit",
                event_type=str(audit.get("action", "")),
                severity="INFO",
                message=str(audit.get("status", "")),
                context=audit.get("details", {}) if isinstance(audit.get("details"), dict) else {},
            )
        )

    for log in state._logs[-limit:]:
        level = str(log.get("level", "")).upper()
        if level not in {"ERROR", "CRITICAL", "WARNING"}:
            continue
        events.append(
            IncidentTimelineEventResponse(
                timestamp=str(log.get("timestamp", "")),
                source="runtime_log",
                event_type=str(log.get("category", "")),
                severity=level,
                message=str(log.get("message", "")),
                context=log.get("extra", {}) if isinstance(log.get("extra"), dict) else {},
            )
        )

    return sorted(events, key=lambda x: _parse_iso_dt(x.timestamp) or datetime.fromtimestamp(0, tz=timezone.utc), reverse=True)[:limit]


def _list_runbook_records() -> list[RunbookRecordResponse]:
    """Return known runbook mappings."""
    records: list[RunbookRecordResponse] = []
    for alert_type, path in ALERT_RUNBOOK_MAP.items():
        records.append(
            RunbookRecordResponse(
                alert_type=alert_type.value,
                runbook_path=path,
                suggested_action=ALERT_SUGGESTED_ACTIONS.get(alert_type),
            )
        )
    records.sort(key=lambda x: x.alert_type)
    return records


def _map_runbook_action(action: str) -> tuple[str, list[str]]:
    """Map runbook automation action to allowlisted command."""
    mapping: dict[str, tuple[str, list[str]]] = {
        "health_check": ("health", ["check", "--full"]),
        "validate_env": ("deploy", ["env", "check"]),
        "gpu_check": ("deploy", ["gpu", "check"]),
        "data_feed_check": ("health", ["check", "--component", "data_feed"]),
        "broker_reconnect_dryrun": ("health", ["check", "--component", "broker"]),
    }
    if action not in mapping:
        raise HTTPException(status_code=400, detail=f"Unsupported runbook action: {action}")
    return mapping[action]


async def _publish_control_event(
    event_type: EventType,
    data: dict[str, Any],
    retries: int = 3,
    base_delay: float = 0.15,
) -> bool:
    """Publish a control event with bounded retry + exponential backoff."""
    state = get_dashboard_state()
    if not state._event_bus:
        return False

    for attempt in range(retries + 1):
        try:
            event = Event(
                event_type=event_type,
                data=data,
                source="dashboard.control",
            )
            state._event_bus.publish(event)
            return True
        except Exception as exc:
            if attempt >= retries:
                logger.error(
                    f"Control event publish failed after {retries + 1} attempts: {event_type.value} -> {exc}"
                )
                return False
            delay = base_delay * (2**attempt)
            await asyncio.sleep(delay)
    return False


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


@app.get("/system/coverage", response_model=SystemCoverageResponse, tags=["System"])
async def get_system_coverage(current_user: AuthenticatedUser) -> SystemCoverageResponse:
    """Return project-wide dashboard coverage for main/scripts/domains/data."""
    _require_permission(current_user, "read.basic")
    return SystemCoverageResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        main_entrypoint="main.py",
        main_entrypoint_exists=(PROJECT_ROOT / "main.py").exists(),
        command_catalog=_build_command_catalog(),
        scripts=_build_script_inventory(),
        domains=_build_domain_coverage(),
        data_assets=_build_data_coverage(),
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
    _require_permission(current_user, "alerts.manage")
    manager = get_alert_manager()

    if manager.acknowledge_alert(alert_id, acknowledged_by):
        _record_signed_audit(
            action="alerts.acknowledge",
            user=current_user,
            status_value="success",
            details={"alert_id": alert_id, "acknowledged_by": acknowledged_by},
        )
        return {"status": "acknowledged"}

    raise HTTPException(status_code=404, detail="Alert not found")


@app.post("/alerts/{alert_id}/resolve", tags=["Alerts"])
async def resolve_alert(alert_id: str, current_user: AuthenticatedUser) -> dict[str, str]:
    """Resolve an alert. Requires authentication."""
    _require_permission(current_user, "alerts.manage")
    manager = get_alert_manager()

    if manager.resolve_alert(alert_id):
        _record_signed_audit(
            action="alerts.resolve",
            user=current_user,
            status_value="success",
            details={"alert_id": alert_id},
        )
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


@app.get("/control/trading/status", tags=["Control"])
async def get_trading_process_status(current_user: AuthenticatedUser) -> dict[str, Any]:
    """Get status of the tracked trading process."""
    _require_permission(current_user, "control.trading.status")
    state = get_dashboard_state()
    return state.get_trading_status()


@app.post("/control/trading/start", response_model=ControlActionResponse, tags=["Control"])
async def start_trading_process(
    request: TradingStartRequest,
    current_user: AuthenticatedUser,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    x_mfa_code: Annotated[str | None, Header(alias="X-MFA-Code")] = None,
) -> ControlActionResponse:
    """Start a managed trading process."""
    _require_permission(current_user, "control.trading.start")

    payload = request.model_dump(mode="json")
    cached, fingerprint = _check_idempotent_replay(
        action="control.trading.start",
        user=current_user,
        idempotency_key=idempotency_key,
        payload=payload,
    )
    if cached:
        _record_signed_audit(
            action="control.trading.start",
            user=current_user,
            status_value="replayed",
            details={"cached": True},
            idempotency_key=idempotency_key,
        )
        return ControlActionResponse(**cached)

    if not request.symbols:
        raise HTTPException(status_code=400, detail="At least one symbol is required")
    if request.mode == "live":
        _require_mfa_if_configured(
            current_user,
            action="control.trading.start.live",
            mfa_code=request.mfa_code or x_mfa_code,
        )

    state = get_dashboard_state()
    status_state = state.get_trading_status()
    if status_state["running"]:
        raise HTTPException(status_code=409, detail="Trading process already running")

    command = [
        sys.executable,
        "main.py",
        "trade",
        "--mode",
        request.mode,
        "--strategy",
        request.strategy,
        "--capital",
        str(request.capital),
        "--symbols",
        *request.symbols,
    ]

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    runtime_log_path = log_dir / "dashboard_trading_runtime.log"
    log_file = runtime_log_path.open("a", encoding="utf-8")

    creation_flags = 0
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    process = subprocess.Popen(  # noqa: S603,S607 - controlled command allowlist
        command,
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=creation_flags,
    )
    log_file.close()
    state.set_trading_process(process)
    state.add_log(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "category": "TRADING",
            "message": f"Trading process started by {current_user}",
            "extra": {
                "pid": process.pid,
                "mode": request.mode,
                "symbols": request.symbols,
                "strategy": request.strategy,
            },
        }
    )

    ok = await _publish_control_event(
        EventType.SYSTEM_START,
        {
            "component": "trading_process",
            "pid": process.pid,
            "requested_by": current_user,
            "mode": request.mode,
        },
    )
    if not ok:
        raise HTTPException(status_code=503, detail="Control bus unavailable")

    response = ControlActionResponse(
        status="started",
        detail=f"Trading process started with PID {process.pid}",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    response_payload = response.model_dump(mode="json")
    _store_idempotent_response(
        action="control.trading.start",
        idempotency_key=idempotency_key,
        fingerprint=fingerprint,
        response_payload=response_payload,
    )
    _record_signed_audit(
        action="control.trading.start",
        user=current_user,
        status_value="success",
        details={"pid": process.pid, "mode": request.mode, "symbols": request.symbols},
        idempotency_key=idempotency_key,
    )
    return response


@app.post("/control/trading/stop", response_model=ControlActionResponse, tags=["Control"])
async def stop_trading_process(
    current_user: AuthenticatedUser,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> ControlActionResponse:
    """Stop the managed trading process."""
    _require_permission(current_user, "control.trading.stop")
    cached, fingerprint = _check_idempotent_replay(
        action="control.trading.stop",
        user=current_user,
        idempotency_key=idempotency_key,
        payload={},
    )
    if cached:
        _record_signed_audit(
            action="control.trading.stop",
            user=current_user,
            status_value="replayed",
            details={"cached": True},
            idempotency_key=idempotency_key,
        )
        return ControlActionResponse(**cached)

    state = get_dashboard_state()
    process = state._trading_process

    if not process or process.poll() is not None:
        state.clear_trading_process()
        response = ControlActionResponse(
            status="stopped",
            detail="No active trading process",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        _store_idempotent_response(
            action="control.trading.stop",
            idempotency_key=idempotency_key,
            fingerprint=fingerprint,
            response_payload=response.model_dump(mode="json"),
        )
        _record_signed_audit(
            action="control.trading.stop",
            user=current_user,
            status_value="noop",
            details={"detail": "No active process"},
            idempotency_key=idempotency_key,
        )
        return response

    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)

    state.add_log(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "WARNING",
            "category": "TRADING",
            "message": f"Trading process stopped by {current_user}",
            "extra": {"pid": process.pid},
        }
    )
    state.clear_trading_process()

    ok = await _publish_control_event(
        EventType.SYSTEM_STOP,
        {
            "component": "trading_process",
            "requested_by": current_user,
            "pid": process.pid,
        },
    )
    if not ok:
        raise HTTPException(status_code=503, detail="Control bus unavailable")

    response = ControlActionResponse(
        status="stopped",
        detail=f"Trading process {process.pid} stopped",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    _store_idempotent_response(
        action="control.trading.stop",
        idempotency_key=idempotency_key,
        fingerprint=fingerprint,
        response_payload=response.model_dump(mode="json"),
    )
    _record_signed_audit(
        action="control.trading.stop",
        user=current_user,
        status_value="success",
        details={"pid": process.pid},
        idempotency_key=idempotency_key,
    )
    return response


@app.post("/control/trading/restart", response_model=ControlActionResponse, tags=["Control"])
async def restart_trading_process(
    request: TradingStartRequest,
    current_user: AuthenticatedUser,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    x_mfa_code: Annotated[str | None, Header(alias="X-MFA-Code")] = None,
) -> ControlActionResponse:
    """Restart the managed trading process with new config."""
    _require_permission(current_user, "control.trading.restart")
    payload = request.model_dump(mode="json")
    cached, fingerprint = _check_idempotent_replay(
        action="control.trading.restart",
        user=current_user,
        idempotency_key=idempotency_key,
        payload=payload,
    )
    if cached:
        _record_signed_audit(
            action="control.trading.restart",
            user=current_user,
            status_value="replayed",
            details={"cached": True},
            idempotency_key=idempotency_key,
        )
        return ControlActionResponse(**cached)

    if request.mode == "live":
        _require_mfa_if_configured(
            current_user,
            action="control.trading.restart.live",
            mfa_code=request.mfa_code or x_mfa_code,
        )

    await stop_trading_process(current_user, idempotency_key=None)
    start_response = await start_trading_process(request, current_user, idempotency_key=None, x_mfa_code=x_mfa_code)
    response = ControlActionResponse(
        status="restarted",
        detail=start_response.detail,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    _store_idempotent_response(
        action="control.trading.restart",
        idempotency_key=idempotency_key,
        fingerprint=fingerprint,
        response_payload=response.model_dump(mode="json"),
    )
    _record_signed_audit(
        action="control.trading.restart",
        user=current_user,
        status_value="success",
        details={"mode": request.mode, "symbols": request.symbols},
        idempotency_key=idempotency_key,
    )
    return response


@app.post("/control/risk/kill-switch/activate", response_model=ControlActionResponse, tags=["Control"])
async def activate_kill_switch(
    request: KillSwitchActivateRequest,
    current_user: AuthenticatedUser,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    x_mfa_code: Annotated[str | None, Header(alias="X-MFA-Code")] = None,
) -> ControlActionResponse:
    """Publish kill-switch activation control event."""
    _require_permission(current_user, "control.risk.kill_switch.activate")
    payload = request.model_dump(mode="json")
    cached, fingerprint = _check_idempotent_replay(
        action="control.risk.kill_switch.activate",
        user=current_user,
        idempotency_key=idempotency_key,
        payload=payload,
    )
    if cached:
        _record_signed_audit(
            action="control.risk.kill_switch.activate",
            user=current_user,
            status_value="replayed",
            details={"cached": True},
            idempotency_key=idempotency_key,
        )
        return ControlActionResponse(**cached)

    _require_mfa_if_configured(
        current_user,
        action="control.risk.kill_switch.activate",
        mfa_code=request.mfa_code or x_mfa_code,
    )

    ok = await _publish_control_event(
        EventType.KILL_SWITCH_TRIGGERED,
        {
            "action": "activate",
            "reason": request.reason,
            "requested_by": current_user,
        },
    )
    if not ok:
        raise HTTPException(status_code=503, detail="Control bus unavailable")

    response = ControlActionResponse(
        status="accepted",
        detail="Kill-switch activation event published",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    _store_idempotent_response(
        action="control.risk.kill_switch.activate",
        idempotency_key=idempotency_key,
        fingerprint=fingerprint,
        response_payload=response.model_dump(mode="json"),
    )
    _record_signed_audit(
        action="control.risk.kill_switch.activate",
        user=current_user,
        status_value="success",
        details={"reason": request.reason},
        idempotency_key=idempotency_key,
    )
    return response


@app.post("/control/risk/kill-switch/reset", response_model=ControlActionResponse, tags=["Control"])
async def reset_kill_switch(
    request: KillSwitchResetRequest,
    current_user: AuthenticatedUser,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    x_mfa_code: Annotated[str | None, Header(alias="X-MFA-Code")] = None,
) -> ControlActionResponse:
    """Publish kill-switch reset control event."""
    _require_permission(current_user, "control.risk.kill_switch.reset")
    _require_mfa_if_configured(
        current_user,
        action="control.risk.kill_switch.reset",
        mfa_code=request.mfa_code or x_mfa_code,
    )

    payload = request.model_dump(mode="json")
    cached, fingerprint = _check_idempotent_replay(
        action="control.risk.kill_switch.reset",
        user=current_user,
        idempotency_key=idempotency_key,
        payload=payload,
    )
    if cached:
        _record_signed_audit(
            action="control.risk.kill_switch.reset",
            user=current_user,
            status_value="replayed",
            details={"cached": True},
            idempotency_key=idempotency_key,
        )
        return ControlActionResponse(**cached)

    ok = await _publish_control_event(
        EventType.KILL_SWITCH_RESET,
        {
            "action": "reset",
            "authorized_by": request.authorized_by,
            "force": request.force,
            "override_code": request.override_code,
            "requested_by": current_user,
        },
    )
    if not ok:
        raise HTTPException(status_code=503, detail="Control bus unavailable")

    response = ControlActionResponse(
        status="accepted",
        detail="Kill-switch reset event published",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    _store_idempotent_response(
        action="control.risk.kill_switch.reset",
        idempotency_key=idempotency_key,
        fingerprint=fingerprint,
        response_payload=response.model_dump(mode="json"),
    )
    _record_signed_audit(
        action="control.risk.kill_switch.reset",
        user=current_user,
        status_value="success",
        details={
            "authorized_by": request.authorized_by,
            "force": request.force,
        },
        idempotency_key=idempotency_key,
    )
    return response


@app.get("/control/jobs/catalog", response_model=list[CommandCatalogEntryResponse], tags=["Control"])
async def get_command_job_catalog(current_user: AuthenticatedUser) -> list[CommandCatalogEntryResponse]:
    """List allowlisted async command jobs exposed to dashboard operators."""
    _require_permission(current_user, "control.jobs.read")
    return _build_command_catalog()


@app.post("/control/jobs", response_model=CommandJobResponse, tags=["Control"])
async def create_command_job(
    request: CommandJobRequest,
    current_user: AuthenticatedUser,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> CommandJobResponse:
    """Create and enqueue an operational command job."""
    _require_permission(current_user, "control.jobs.create")
    payload = request.model_dump(mode="json")
    cached, fingerprint = _check_idempotent_replay(
        action="control.jobs.create",
        user=current_user,
        idempotency_key=idempotency_key,
        payload=payload,
    )
    if cached:
        _record_signed_audit(
            action="control.jobs.create",
            user=current_user,
            status_value="replayed",
            details={"cached": True},
            idempotency_key=idempotency_key,
        )
        return CommandJobResponse(**cached)

    _build_main_command(request.command, request.args)  # Validate allowlist

    state = get_dashboard_state()
    job = state.create_job(request.command, request.args)
    task = asyncio.create_task(_run_command_job(job["job_id"]))
    state._job_tasks[job["job_id"]] = task

    state.add_log(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "category": "SYSTEM",
            "message": f"Job queued by {current_user}",
            "extra": {"job_id": job["job_id"], "command": request.command, "args": request.args},
        }
    )

    response = _serialize_job(job)
    _store_idempotent_response(
        action="control.jobs.create",
        idempotency_key=idempotency_key,
        fingerprint=fingerprint,
        response_payload=response.model_dump(mode="json"),
    )
    _record_signed_audit(
        action="control.jobs.create",
        user=current_user,
        status_value="accepted",
        details={"job_id": job["job_id"], "command": request.command},
        idempotency_key=idempotency_key,
    )
    return response


@app.get("/control/jobs", response_model=list[CommandJobResponse], tags=["Control"])
async def list_command_jobs(
    current_user: AuthenticatedUser,
    limit: int = Query(50, ge=1, le=200),
) -> list[CommandJobResponse]:
    """List recent operational command jobs."""
    _require_permission(current_user, "control.jobs.read")
    state = get_dashboard_state()
    return [_serialize_job(job) for job in state.get_jobs(limit=limit)]


@app.get("/control/jobs/{job_id}", response_model=CommandJobResponse, tags=["Control"])
async def get_command_job(job_id: str, current_user: AuthenticatedUser) -> CommandJobResponse:
    """Get a single command job."""
    _require_permission(current_user, "control.jobs.read")
    state = get_dashboard_state()
    job = state.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _serialize_job(job)


@app.post("/control/jobs/{job_id}/cancel", response_model=ControlActionResponse, tags=["Control"])
async def cancel_command_job(
    job_id: str,
    current_user: AuthenticatedUser,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> ControlActionResponse:
    """Cancel a queued/running command job."""
    _require_permission(current_user, "control.jobs.cancel")
    cached, fingerprint = _check_idempotent_replay(
        action="control.jobs.cancel",
        user=current_user,
        idempotency_key=idempotency_key,
        payload={"job_id": job_id},
    )
    if cached:
        _record_signed_audit(
            action="control.jobs.cancel",
            user=current_user,
            status_value="replayed",
            details={"job_id": job_id},
            idempotency_key=idempotency_key,
        )
        return ControlActionResponse(**cached)

    state = get_dashboard_state()
    task = state._job_tasks.get(job_id)
    job = state.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if task and not task.done():
        task.cancel()
        state.update_job(
            job_id,
            status="cancelled",
            ended_at=datetime.now(timezone.utc),
            exit_code=-1,
        )
        response = ControlActionResponse(
            status="cancelled",
            detail=f"Job {job_id} cancelled",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        _store_idempotent_response(
            action="control.jobs.cancel",
            idempotency_key=idempotency_key,
            fingerprint=fingerprint,
            response_payload=response.model_dump(mode="json"),
        )
        _record_signed_audit(
            action="control.jobs.cancel",
            user=current_user,
            status_value="success",
            details={"job_id": job_id},
            idempotency_key=idempotency_key,
        )
        return response

    response = ControlActionResponse(
        status="noop",
        detail=f"Job {job_id} already completed",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    _store_idempotent_response(
        action="control.jobs.cancel",
        idempotency_key=idempotency_key,
        fingerprint=fingerprint,
        response_payload=response.model_dump(mode="json"),
    )
    _record_signed_audit(
        action="control.jobs.cancel",
        user=current_user,
        status_value="noop",
        details={"job_id": job_id},
        idempotency_key=idempotency_key,
    )
    return response


@app.get("/execution/tca", response_model=TCAResponse, tags=["Execution"])
async def get_tca_metrics(current_user: AuthenticatedUser) -> TCAResponse:
    """Get Transaction Cost Analysis metrics. Requires authentication."""
    state = get_dashboard_state()
    return _calculate_tca_from_orders(state._orders)


@app.get("/execution/quality", response_model=ExecutionQualityResponse, tags=["Execution"])
async def get_execution_quality(current_user: AuthenticatedUser) -> ExecutionQualityResponse:
    """Get execution quality decomposition metrics."""
    _require_permission(current_user, "read.basic")
    state = get_dashboard_state()
    return _calculate_execution_quality(state._orders)


@app.get("/risk/var", response_model=VaRResponse, tags=["Risk"])
async def get_var_metrics(current_user: AuthenticatedUser) -> VaRResponse:
    """Get Value at Risk simulations. Requires authentication."""
    state = get_dashboard_state()
    portfolio = state._portfolio_data
    equity = float(portfolio.get("equity", 100000.0))
    drawdown = float(portfolio.get("current_drawdown", 0.0))
    return _calculate_var_from_portfolio(equity=equity, drawdown=drawdown)


@app.get("/risk/concentration", response_model=RiskConcentrationResponse, tags=["Risk"])
async def get_risk_concentration(current_user: AuthenticatedUser) -> RiskConcentrationResponse:
    """Get concentration risk decomposition."""
    _require_permission(current_user, "risk.advanced.read")
    state = get_dashboard_state()
    return _calculate_risk_concentration(state._positions, state._portfolio_data)


@app.get("/risk/correlation", response_model=RiskCorrelationResponse, tags=["Risk"])
async def get_risk_correlation(current_user: AuthenticatedUser) -> RiskCorrelationResponse:
    """Get correlation and clustering risk."""
    _require_permission(current_user, "risk.advanced.read")
    state = get_dashboard_state()
    return _calculate_risk_correlation(state._positions, state._portfolio_data)


@app.get("/risk/stress", response_model=RiskStressResponse, tags=["Risk"])
async def get_risk_stress(current_user: AuthenticatedUser) -> RiskStressResponse:
    """Get stress test scenarios."""
    _require_permission(current_user, "risk.advanced.read")
    state = get_dashboard_state()
    return _calculate_risk_stress(state._portfolio_data)


@app.get("/risk/attribution", response_model=RiskAttributionResponse, tags=["Risk"])
async def get_risk_attribution(current_user: AuthenticatedUser) -> RiskAttributionResponse:
    """Get pre-trade and post-trade risk attribution."""
    _require_permission(current_user, "risk.advanced.read")
    state = get_dashboard_state()
    return _calculate_risk_attribution(state._portfolio_data, state._orders, state._positions)


@app.get("/models/explainability", response_model=FeatureImportanceResponse, tags=["Models"])
async def get_model_explainability(current_user: AuthenticatedUser) -> FeatureImportanceResponse:
    """Get AI feature importance (SHAP values). Requires authentication."""
    return _load_latest_model_explainability()


@app.get("/models/registry", response_model=ModelRegistryResponse, tags=["Models"])
async def get_model_registry(current_user: AuthenticatedUser) -> ModelRegistryResponse:
    """Get model registry snapshot."""
    _require_permission(current_user, "models.governance.read")
    return _build_model_registry_snapshot()


@app.get("/models/drift", response_model=ModelDriftResponse, tags=["Models"])
async def get_model_drift(current_user: AuthenticatedUser) -> ModelDriftResponse:
    """Get model drift status."""
    _require_permission(current_user, "models.governance.read")
    return _build_model_drift_snapshot()


@app.get("/models/validation-gates", response_model=ModelValidationGateResponse, tags=["Models"])
async def get_model_validation_gates(current_user: AuthenticatedUser) -> ModelValidationGateResponse:
    """Get model validation gate decision."""
    _require_permission(current_user, "models.governance.read")
    return _build_model_validation_gates_snapshot()


@app.get("/models/champion-challenger", response_model=ModelChampionChallengerResponse, tags=["Models"])
async def get_model_champion_challenger(current_user: AuthenticatedUser) -> ModelChampionChallengerResponse:
    """Get champion/challenger comparison."""
    _require_permission(current_user, "models.governance.read")
    return _build_champion_challenger_snapshot()


@app.post("/models/champion/promote", response_model=ControlActionResponse, tags=["Models"])
async def promote_model_champion(
    request: ModelPromotionRequest,
    current_user: AuthenticatedUser,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    x_mfa_code: Annotated[str | None, Header(alias="X-MFA-Code")] = None,
) -> ControlActionResponse:
    """Promote model version to champion pointer."""
    _require_permission(current_user, "models.governance.write")
    _require_mfa_if_configured(
        current_user,
        action="models.champion.promote",
        mfa_code=request.mfa_code or x_mfa_code,
    )

    payload = request.model_dump(mode="json")
    cached, fingerprint = _check_idempotent_replay(
        action="models.champion.promote",
        user=current_user,
        idempotency_key=idempotency_key,
        payload=payload,
    )
    if cached:
        _record_signed_audit(
            action="models.champion.promote",
            user=current_user,
            status_value="replayed",
            details={"cached": True},
            idempotency_key=idempotency_key,
        )
        return ControlActionResponse(**cached)

    entries = _load_model_registry_entries()
    matched = [
        entry
        for entry in entries
        if str(entry.get("model_name", "")) == request.model_name and str(entry.get("version_id", "")) == request.version_id
    ]
    if not matched:
        raise HTTPException(status_code=404, detail="Requested model/version not found in registry")

    _save_active_model_pointer(
        model_name=request.model_name,
        version_id=request.version_id,
        updated_by=current_user,
        reason=request.reason,
    )

    response = ControlActionResponse(
        status="promoted",
        detail=f"{request.model_name}:{request.version_id} promoted as champion",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    _store_idempotent_response(
        action="models.champion.promote",
        idempotency_key=idempotency_key,
        fingerprint=fingerprint,
        response_payload=response.model_dump(mode="json"),
    )
    _record_signed_audit(
        action="models.champion.promote",
        user=current_user,
        status_value="success",
        details={
            "model_name": request.model_name,
            "version_id": request.version_id,
            "reason": request.reason,
        },
        idempotency_key=idempotency_key,
    )
    return response


@app.get("/sre/slo", response_model=SLOStatusResponse, tags=["SRE"])
async def get_slo_status(current_user: AuthenticatedUser) -> SLOStatusResponse:
    """Get SLO and burn-rate status."""
    _require_permission(current_user, "operations.sre.read")
    return _compute_slo_status()


@app.get("/sre/incidents", response_model=list[IncidentRecordResponse], tags=["SRE"])
async def get_incident_records(
    current_user: AuthenticatedUser,
    limit: int = Query(100, ge=1, le=500),
) -> list[IncidentRecordResponse]:
    """Get recent incidents."""
    _require_permission(current_user, "operations.sre.read")
    return _build_incident_records(limit=limit)


@app.get("/sre/incidents/timeline", response_model=list[IncidentTimelineEventResponse], tags=["SRE"])
async def get_incident_timeline(
    current_user: AuthenticatedUser,
    limit: int = Query(250, ge=1, le=1000),
) -> list[IncidentTimelineEventResponse]:
    """Get incident timeline events."""
    _require_permission(current_user, "operations.sre.read")
    return _build_incident_timeline(limit=limit)


@app.get("/sre/runbooks", response_model=list[RunbookRecordResponse], tags=["SRE"])
async def get_runbooks(current_user: AuthenticatedUser) -> list[RunbookRecordResponse]:
    """Get mapped runbooks and suggested actions."""
    _require_permission(current_user, "operations.sre.read")
    return _list_runbook_records()


@app.post("/sre/runbooks/execute", response_model=CommandJobResponse, tags=["SRE"])
async def execute_runbook_action(
    request: RunbookExecutionRequest,
    current_user: AuthenticatedUser,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    x_mfa_code: Annotated[str | None, Header(alias="X-MFA-Code")] = None,
) -> CommandJobResponse:
    """Execute runbook automation action as an operational command job."""
    _require_permission(current_user, "operations.runbooks.execute")
    _require_mfa_if_configured(
        current_user,
        action="operations.runbooks.execute",
        mfa_code=request.mfa_code or x_mfa_code,
    )

    payload = request.model_dump(mode="json")
    cached, fingerprint = _check_idempotent_replay(
        action="sre.runbooks.execute",
        user=current_user,
        idempotency_key=idempotency_key,
        payload=payload,
    )
    if cached:
        _record_signed_audit(
            action="sre.runbooks.execute",
            user=current_user,
            status_value="replayed",
            details={"cached": True},
            idempotency_key=idempotency_key,
        )
        return CommandJobResponse(**cached)

    command, args = _map_runbook_action(request.action)
    _build_main_command(command, args)
    state = get_dashboard_state()
    job = state.create_job(command, args)
    task = asyncio.create_task(_run_command_job(job["job_id"]))
    state._job_tasks[job["job_id"]] = task

    response = _serialize_job(job)
    _store_idempotent_response(
        action="sre.runbooks.execute",
        idempotency_key=idempotency_key,
        fingerprint=fingerprint,
        response_payload=response.model_dump(mode="json"),
    )
    _record_signed_audit(
        action="sre.runbooks.execute",
        user=current_user,
        status_value="accepted",
        details={"action": request.action, "job_id": job["job_id"], "command": command, "args": args},
        idempotency_key=idempotency_key,
    )
    return response


@app.get("/control/audit", response_model=list[AuditRecordResponse], tags=["Control"])
async def get_control_audit(
    current_user: AuthenticatedUser,
    limit: int = Query(100, ge=1, le=500),
) -> list[AuditRecordResponse]:
    """Get signed control-plane audit records."""
    _require_permission(current_user, "control.audit.read")
    records = _audit_signer.list_recent(limit=limit)
    return _serialize_signed_audit_records(records)


def _serialize_signed_audit_records(records: list[dict[str, Any]]) -> list[AuditRecordResponse]:
    """Convert raw signed audit records to response model."""
    return [
        AuditRecordResponse(
            timestamp=str(record.get("timestamp", "")),
            action=str(record.get("action", "")),
            user=str(record.get("user", "")),
            status=str(record.get("status", "")),
            details=record.get("details", {}) if isinstance(record.get("details"), dict) else {},
            prev_hash=str(record.get("prev_hash", "")),
            record_hash=str(record.get("record_hash", "")),
        )
        for record in records
    ]


@app.get("/control/audit/export", response_model=None, tags=["Control"])
async def export_control_audit(
    current_user: AuthenticatedUser,
    limit: int = Query(1000, ge=1, le=20000),
    format: str = Query("jsonl", pattern="^(json|jsonl)$"),
) -> dict[str, Any] | Response:
    """Export signed audit trail in JSON or NDJSON format."""
    _require_permission(current_user, "control.audit.read")
    records = _audit_signer.list_recent(limit=limit)
    serialized = [item.model_dump(mode="json") for item in _serialize_signed_audit_records(records)]

    _record_signed_audit(
        action="control.audit.export",
        user=current_user,
        status_value="success",
        details={"format": format, "count": len(serialized)},
    )

    if format == "json":
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(serialized),
            "records": serialized,
        }

    lines = [json.dumps(record, ensure_ascii=True) for record in serialized]
    payload = "\n".join(lines) + ("\n" if lines else "")

    # Persist latest export snapshot for central retrieval/forensics.
    try:
        AUDIT_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
        export_name = datetime.now(timezone.utc).strftime("dashboard_audit_export_%Y%m%dT%H%M%SZ.ndjson")
        (AUDIT_EXPORT_ROOT / export_name).write_text(payload, encoding="utf-8")
    except Exception as exc:
        logger.warning(f"Failed writing audit export snapshot: {exc}")

    return Response(
        content=payload,
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": "attachment; filename=dashboard_audit_export.ndjson",
            "Cache-Control": "no-store",
        },
    )


@app.get("/control/audit/siem/status", response_model=SIEMStatusResponse, tags=["Control"])
async def get_audit_siem_status(current_user: AuthenticatedUser) -> SIEMStatusResponse:
    """Get SIEM forwarder status."""
    _require_permission(current_user, "control.audit.read")
    return _audit_siem_forwarder.status()


@app.post("/control/audit/siem/flush", response_model=ControlActionResponse, tags=["Control"])
async def flush_audit_siem(
    current_user: AuthenticatedUser,
    max_batches: int = Query(3, ge=1, le=100),
) -> ControlActionResponse:
    """Force immediate SIEM flush for queued audit records."""
    _require_permission(current_user, "control.audit.manage")
    sent_count = _audit_siem_forwarder.flush(max_batches=max_batches)
    status_snapshot = _audit_siem_forwarder.status()
    _record_signed_audit(
        action="control.audit.siem.flush",
        user=current_user,
        status_value="success",
        details={
            "sent_count": sent_count,
            "queue_depth": status_snapshot.queue_depth,
            "max_batches": max_batches,
        },
    )
    return ControlActionResponse(
        status="flushed",
        detail=f"SIEM flush completed. Delivered {sent_count} queued audit records.",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# WebSocket Endpoints
# =============================================================================


async def _stream_channel(
    websocket: WebSocket,
    channel: str,
    initial_payload: dict[str, Any] | None = None,
) -> None:
    """Common websocket stream loop with heartbeat and optional initial snapshot."""
    manager = get_connection_manager()
    await manager.connect(websocket, channel)

    try:
        if initial_payload is not None:
            await websocket.send_json(initial_payload)

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=25.0)
                if data.strip().lower() == "ping":
                    await websocket.send_json(
                        {"type": "pong", "timestamp": datetime.now(timezone.utc).isoformat()}
                    )
                else:
                    await websocket.send_json({"type": "ack", "message": data})
            except asyncio.TimeoutError:
                await websocket.send_json(
                    {"type": "heartbeat", "timestamp": datetime.now(timezone.utc).isoformat()}
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel)


@app.websocket("/ws/portfolio")
async def websocket_portfolio(websocket: WebSocket) -> None:
    """WebSocket for real-time portfolio updates."""
    state = get_dashboard_state()
    await _stream_channel(
        websocket,
        "portfolio",
        {"type": "snapshot", "data": state._portfolio_data},
    )


@app.websocket("/ws/orders")
async def websocket_orders(websocket: WebSocket) -> None:
    """WebSocket for real-time order updates."""
    state = get_dashboard_state()
    await _stream_channel(
        websocket,
        "orders",
        {"type": "snapshot", "data": state._orders[-100:]},
    )


@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket) -> None:
    """WebSocket for real-time signal updates."""
    state = get_dashboard_state()
    await _stream_channel(
        websocket,
        "signals",
        {"type": "snapshot", "data": state._signals[-100:]},
    )


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket) -> None:
    """WebSocket for real-time alert updates."""
    manager = get_alert_manager()
    alerts = manager.get_dashboard_alerts(limit=100)
    payload = [
        {
            "alert_id": str(a.alert_id),
            "alert_type": a.alert_type.value,
            "severity": a.severity.value,
            "status": a.status.value,
            "title": a.title,
            "message": a.message,
            "timestamp": a.timestamp.isoformat(),
        }
        for a in alerts
    ]
    await _stream_channel(websocket, "alerts", {"type": "snapshot", "data": payload})


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
_event_bus: EventBus | None = None


async def startup_event() -> None:
    """Initialize dashboard services on startup."""
    global _redis_bridge, _event_bus
    logger.info("Starting dashboard services...")

    try:
        # Initialize EventBus
        _event_bus = EventBus()
        event_bus = _event_bus

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
        state.bind_event_bus(event_bus)
        event_bus.subscribe_all(state.handle_event, "dashboard_state_updater")
        _audit_siem_forwarder.start()
        
        logger.info("Dashboard services started successfully")

    except Exception as e:
        logger.error(f"Failed to start dashboard services: {e}")
        # We don't raise here to allow the API to start even if Redis is down,
        # but health checks will show it.


async def shutdown_event() -> None:
    """Clean up resources on shutdown."""
    global _redis_bridge, _event_bus
    logger.info("Stopping dashboard services...")

    state = get_dashboard_state()
    for task in list(state._job_tasks.values()):
        if not task.done():
            task.cancel()

    process = state._trading_process
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except Exception:
            process.kill()
    state.clear_trading_process()

    if _redis_bridge:
        await _redis_bridge.stop()
    _audit_siem_forwarder.stop()
    _event_bus = None

    logger.info("Dashboard services stopped")


@asynccontextmanager
async def _dashboard_lifespan(_app: FastAPI):
    """FastAPI lifespan hook for startup/shutdown orchestration."""
    await startup_event()
    try:
        yield
    finally:
        await shutdown_event()


app.router.lifespan_context = _dashboard_lifespan
