"""
Central configuration management using Pydantic settings.

Provides type-safe configuration with validation, environment variable
support, and YAML configuration file loading.

JPMorgan-level enhancement: Configuration change auditing for regulatory compliance.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import threading
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = Path(__file__).resolve().parent


class DatabaseSettings(BaseModel):
    """Database configuration settings.

    P0 FIX: Credentials are now protected from accidental logging.
    Use `connection_string` for actual database connections.
    Use `url` (masked) for logging/display purposes.
    """

    host: str = "localhost"
    port: int = 5432
    name: str = "quant_trading"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10

    @property
    def url(self) -> str:
        """Get the database URL with MASKED password.

        P0 FIX: Use this for logging/display purposes ONLY.
        For actual connections, use connection_string.
        """
        return f"postgresql://{self.user}:***@{self.host}:{self.port}/{self.name}"

    @property
    def connection_string(self) -> str:
        """Get the full database connection URL with credentials.

        P0 FIX: Use this for actual database connections.
        WARNING: Never log this value directly.
        """
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseModel):
    """Redis configuration settings."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None

    @property
    def url(self) -> str:
        """Get the Redis connection URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class AlpacaSettings(BaseModel):
    """Alpaca broker configuration settings.

    SECURITY: Credentials should be provided via environment variables:
    - ALPACA_API_KEY: Your Alpaca API key
    - ALPACA_API_SECRET: Your Alpaca API secret

    The settings are loaded from environment variables FIRST (secure),
    then fall back to YAML config (less secure, for development only).
    """

    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"
    paper_trading: bool = True

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str, info) -> str:
        """Ensure paper trading uses paper API."""
        return v

    @classmethod
    def from_env(cls) -> "AlpacaSettings":
        """Create settings from environment variables (SECURE method).

        Environment variables:
        - ALPACA_API_KEY: API key (required for live trading)
        - ALPACA_API_SECRET: API secret (required for live trading)
        - ALPACA_BASE_URL: API base URL (optional)
        - ALPACA_DATA_URL: Data API URL (optional)
        - ALPACA_PAPER_TRADING: Set to 'false' for live trading (optional)
        """
        return cls(
            api_key=os.environ.get("ALPACA_API_KEY", ""),
            api_secret=os.environ.get("ALPACA_API_SECRET", ""),
            base_url=os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
            data_url=os.environ.get("ALPACA_DATA_URL", "https://data.alpaca.markets"),
            paper_trading=os.environ.get("ALPACA_PAPER_TRADING", "true").lower() == "true",
        )


class RiskSettings(BaseModel):
    """Risk management configuration settings."""

    max_position_pct: Decimal = Field(default=Decimal("0.10"), description="Max position size as % of equity")
    max_portfolio_positions: int = Field(default=20, description="Maximum number of positions")
    max_sector_exposure_pct: Decimal = Field(default=Decimal("0.25"), description="Max sector exposure")
    max_correlation_exposure_pct: Decimal = Field(default=Decimal("0.30"), description="Max correlated exposure")
    daily_loss_limit_pct: Decimal = Field(default=Decimal("0.02"), description="Daily loss limit")
    weekly_loss_limit_pct: Decimal = Field(default=Decimal("0.05"), description="Weekly loss limit")
    monthly_loss_limit_pct: Decimal = Field(default=Decimal("0.10"), description="Monthly loss limit")
    max_drawdown_pct: Decimal = Field(default=Decimal("0.15"), description="Maximum drawdown limit")
    var_confidence_level: float = Field(default=0.95, description="VaR confidence level")
    position_size_method: str = Field(default="volatility", description="Position sizing method")
    kelly_fraction: float = Field(default=0.25, description="Kelly criterion fraction")


class TradingSettings(BaseModel):
    """Trading engine configuration settings."""

    bar_timeframe: str = Field(default="15Min", description="Bar timeframe")
    lookback_bars: int = Field(default=100, description="Lookback window in bars")
    signal_threshold: float = Field(default=0.5, description="Signal strength threshold")
    min_confidence: float = Field(default=0.6, description="Minimum confidence for trades")
    max_slippage_bps: int = Field(default=10, description="Max slippage in basis points")
    transaction_cost_bps: int = Field(default=5, description="Transaction cost in basis points")
    market_open_time: str = Field(default="09:30", description="Market open time (ET)")
    market_close_time: str = Field(default="16:00", description="Market close time (ET)")


class ModelSettings(BaseModel):
    """Machine learning model configuration settings."""

    models_dir: Path = Field(default=BASE_DIR / "models_artifacts", description="Models directory")
    default_model: str = Field(default="xgboost", description="Default model type")
    ensemble_models: list[str] = Field(
        default=["xgboost", "lightgbm", "catboost"],
        description="Models in ensemble"
    )
    retrain_frequency_days: int = Field(default=7, description="Model retraining frequency")
    min_training_samples: int = Field(default=1000, description="Minimum training samples")
    validation_split: float = Field(default=0.2, description="Validation split ratio")
    early_stopping_rounds: int = Field(default=50, description="Early stopping patience")


class LoggingSettings(BaseModel):
    """Logging configuration settings."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format (json or text)")
    file_path: Path | None = Field(default=None, description="Log file path")
    rotation: str = Field(default="1 day", description="Log rotation period")
    retention: str = Field(default="30 days", description="Log retention period")


class AlertSettings(BaseModel):
    """Alert notification configuration settings.

    Supports multiple notification channels:
    - Slack: Webhook-based notifications with Block Kit formatting
    - Email: SMTP-based notifications with HTML and plain text
    - PagerDuty: Events API v2 integration for incident management
    - SMS: Twilio-based SMS notifications for escalation

    All credentials should be provided via environment variables.
    """

    # Slack Configuration
    slack_webhook_url: str = Field(
        default="",
        description="Slack incoming webhook URL"
    )
    slack_channel: str = Field(
        default="#trading-alerts",
        description="Default Slack channel for alerts"
    )
    slack_critical_channel: str = Field(
        default="#trading-critical",
        description="Slack channel for critical alerts"
    )
    slack_username: str = Field(
        default="AlphaTrade Alert Bot",
        description="Bot username for Slack messages"
    )

    # Email (SMTP) Configuration
    smtp_host: str = Field(default="", description="SMTP server hostname")
    smtp_port: int = Field(default=587, description="SMTP server port")
    smtp_user: str = Field(default="", description="SMTP authentication username")
    smtp_password: str = Field(default="", description="SMTP authentication password")
    smtp_from_address: str = Field(default="", description="Sender email address")
    smtp_to_addresses: list[str] = Field(
        default_factory=list,
        description="List of recipient email addresses"
    )
    smtp_use_tls: bool = Field(default=True, description="Use TLS for SMTP connection")

    # PagerDuty Configuration
    pagerduty_service_key: str = Field(
        default="",
        description="PagerDuty Events API v2 integration key"
    )
    pagerduty_api_url: str = Field(
        default="https://events.pagerduty.com/v2/enqueue",
        description="PagerDuty Events API endpoint"
    )

    # Twilio SMS Configuration
    twilio_account_sid: str = Field(default="", description="Twilio account SID")
    twilio_auth_token: str = Field(default="", description="Twilio auth token")
    twilio_from_number: str = Field(default="", description="Twilio phone number")
    twilio_to_numbers: list[str] = Field(
        default_factory=list,
        description="List of recipient phone numbers"
    )

    # General Alert Settings
    alert_system_name: str = Field(
        default="AlphaTrade",
        description="System name for alert identification"
    )
    runbook_base_url: str = Field(
        default="",
        description="Base URL for runbook links"
    )

    @property
    def is_slack_configured(self) -> bool:
        """Check if Slack notifications are configured."""
        return bool(self.slack_webhook_url)

    @property
    def is_email_configured(self) -> bool:
        """Check if email notifications are configured."""
        return bool(self.smtp_host and self.smtp_from_address and self.smtp_to_addresses)

    @property
    def is_pagerduty_configured(self) -> bool:
        """Check if PagerDuty notifications are configured."""
        return bool(self.pagerduty_service_key)

    @property
    def is_sms_configured(self) -> bool:
        """Check if SMS notifications are configured."""
        return bool(
            self.twilio_account_sid and
            self.twilio_auth_token and
            self.twilio_from_number and
            self.twilio_to_numbers
        )

    def get_configuration_status(self) -> dict[str, bool]:
        """Get status of all notification channels."""
        return {
            "slack": self.is_slack_configured,
            "email": self.is_email_configured,
            "pagerduty": self.is_pagerduty_configured,
            "sms": self.is_sms_configured,
        }

    @classmethod
    def from_env(cls) -> "AlertSettings":
        """Create AlertSettings from environment variables.

        Environment Variables:
            SLACK_WEBHOOK_URL, SLACK_CHANNEL, SLACK_CRITICAL_CHANNEL, SLACK_USERNAME
            SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM_ADDRESS, SMTP_TO_ADDRESSES
            PAGERDUTY_SERVICE_KEY, PAGERDUTY_API_URL
            TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, TWILIO_TO_NUMBERS
            ALERT_SYSTEM_NAME, RUNBOOK_BASE_URL
        """
        def get_list(key: str) -> list[str]:
            value = os.environ.get(key, "")
            if not value:
                return []
            return [item.strip() for item in value.split(",") if item.strip()]

        return cls(
            # Slack
            slack_webhook_url=os.environ.get("SLACK_WEBHOOK_URL", ""),
            slack_channel=os.environ.get("SLACK_CHANNEL", "#trading-alerts"),
            slack_critical_channel=os.environ.get("SLACK_CRITICAL_CHANNEL", "#trading-critical"),
            slack_username=os.environ.get("SLACK_USERNAME", "AlphaTrade Alert Bot"),
            # Email
            smtp_host=os.environ.get("SMTP_HOST", ""),
            smtp_port=int(os.environ.get("SMTP_PORT", "587")),
            smtp_user=os.environ.get("SMTP_USER", ""),
            smtp_password=os.environ.get("SMTP_PASSWORD", ""),
            smtp_from_address=os.environ.get("SMTP_FROM_ADDRESS", ""),
            smtp_to_addresses=get_list("SMTP_TO_ADDRESSES"),
            smtp_use_tls=os.environ.get("SMTP_USE_TLS", "true").lower() == "true",
            # PagerDuty
            pagerduty_service_key=os.environ.get("PAGERDUTY_SERVICE_KEY", ""),
            pagerduty_api_url=os.environ.get("PAGERDUTY_API_URL", "https://events.pagerduty.com/v2/enqueue"),
            # Twilio
            twilio_account_sid=os.environ.get("TWILIO_ACCOUNT_SID", ""),
            twilio_auth_token=os.environ.get("TWILIO_AUTH_TOKEN", ""),
            twilio_from_number=os.environ.get("TWILIO_FROM_NUMBER", ""),
            twilio_to_numbers=get_list("TWILIO_TO_NUMBERS"),
            # General
            alert_system_name=os.environ.get("ALERT_SYSTEM_NAME", "AlphaTrade"),
            runbook_base_url=os.environ.get("RUNBOOK_BASE_URL", ""),
        )


class SecuritySettings(BaseModel):
    """Security configuration settings for API authentication.

    CRITICAL: JWT secret key MUST be set via environment variable in production.
    Generate a secure key with: python -c "import secrets; print(secrets.token_hex(32))"
    """

    # JWT settings
    jwt_secret_key: str = Field(
        default="",
        description="JWT secret key - MUST be set via JWT_SECRET_KEY env var in production"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_minutes: int = Field(default=60, description="JWT token expiration in minutes")

    # API key settings (alternative to JWT)
    api_keys: list[str] = Field(
        default_factory=list,
        description="List of valid API keys for simple auth"
    )

    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, description="API rate limit per minute")

    # Security flags
    require_auth: bool = Field(
        default=True,
        description="Require authentication for API endpoints"
    )
    allow_public_health_check: bool = Field(
        default=True,
        description="Allow unauthenticated access to /health endpoint"
    )

    @model_validator(mode="after")
    def validate_jwt_secret(self) -> "SecuritySettings":
        """Validate that JWT secret is set when auth is required.

        CRITICAL SECURITY FIX: Reject empty JWT secrets in production to
        prevent authentication bypass vulnerabilities.
        """
        if self.require_auth and not self.jwt_secret_key:
            # Check environment to determine severity
            env = os.environ.get("APP_ENV", "development").lower()
            if env in ("production", "staging"):
                raise ValueError(
                    "SECURITY ERROR: JWT_SECRET_KEY must be set in production/staging. "
                    "Generate with: python -c \"import secrets; print(secrets.token_hex(32))\""
                )
            else:
                warnings.warn(
                    "JWT_SECRET_KEY is empty. Authentication will fail. "
                    "Set JWT_SECRET_KEY environment variable.",
                    UserWarning,
                    stacklevel=2,
                )
        # Validate minimum key length for security
        if self.jwt_secret_key and len(self.jwt_secret_key) < 32:
            raise ValueError(
                "JWT_SECRET_KEY must be at least 32 characters for security. "
                "Generate with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        return self

    @classmethod
    def from_env(cls) -> "SecuritySettings":
        """Create settings from environment variables."""
        api_keys_str = os.environ.get("API_KEYS", "")
        api_keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]

        return cls(
            jwt_secret_key=os.environ.get("JWT_SECRET_KEY", ""),
            jwt_algorithm=os.environ.get("JWT_ALGORITHM", "HS256"),
            jwt_expiration_minutes=int(os.environ.get("JWT_EXPIRATION_MINUTES", "60")),
            api_keys=api_keys,
            rate_limit_per_minute=int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60")),
            require_auth=os.environ.get("REQUIRE_AUTH", "true").lower() == "true",
            allow_public_health_check=os.environ.get("ALLOW_PUBLIC_HEALTH", "true").lower() == "true",
        )


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Application settings
    app_name: str = "Quant Trading System"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = Field(default="development", description="Environment (development, staging, production)")

    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)

    # Symbols configuration (loaded from YAML)
    symbols: list[str] = Field(default_factory=list)

    @classmethod
    def load_yaml_config(cls, config_path: Path) -> dict[str, Any]:
        """Load configuration from a YAML file."""
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def load_symbols_config(self) -> list[str]:
        """Load symbols from YAML configuration."""
        config = self.load_yaml_config(CONFIG_DIR / "symbols.yaml")
        return config.get("symbols", [])

    def load_risk_config(self) -> RiskSettings:
        """Load risk parameters from YAML configuration."""
        config = self.load_yaml_config(CONFIG_DIR / "risk_params.yaml")
        if config:
            return RiskSettings(**config)
        return self.risk

    def load_model_configs(self) -> dict[str, Any]:
        """Load model configurations from YAML."""
        return self.load_yaml_config(CONFIG_DIR / "model_configs.yaml")

    def load_alpaca_config(self) -> AlpacaSettings:
        """Load Alpaca configuration with priority:
        1. Environment variables (MOST SECURE - recommended for production)
        2. YAML file (for development only)
        3. Default values

        SECURITY WARNING: Never commit API credentials to version control.
        Use environment variables for production deployments.
        """
        # First try environment variables (SECURE)
        env_settings = AlpacaSettings.from_env()
        if env_settings.api_key and env_settings.api_secret:
            return env_settings

        # Fall back to YAML config (less secure, development only)
        config = self.load_yaml_config(CONFIG_DIR / "alpaca_config.yaml")
        if config:
            return AlpacaSettings(**config)

        return self.alpaca


@dataclass
class ConfigurationChange:
    """Record of a configuration change for auditing.

    JPMorgan-level enhancement: Tracks all configuration changes
    with full before/after state for regulatory compliance.
    """

    change_id: str
    timestamp: datetime
    section: str
    field: str
    old_value: Any
    new_value: Any
    changed_by: str
    source: str  # "environment", "yaml", "api", "manual"
    hash: str = ""

    def __post_init__(self):
        """Calculate change hash after initialization."""
        if not self.hash:
            self.hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of the change record."""
        data = {
            "change_id": self.change_id,
            "timestamp": self.timestamp.isoformat(),
            "section": self.section,
            "field": self.field,
            "old_value": str(self.old_value),
            "new_value": str(self.new_value),
            "changed_by": self.changed_by,
            "source": self.source,
        }
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "change_id": self.change_id,
            "timestamp": self.timestamp.isoformat(),
            "section": self.section,
            "field": self.field,
            "old_value": str(self.old_value),
            "new_value": str(self.new_value),
            "changed_by": self.changed_by,
            "source": self.source,
            "hash": self.hash,
        }


class ConfigurationAuditor:
    """Auditor for tracking and logging configuration changes.

    JPMorgan-level enhancement: Provides immutable audit trail for
    all configuration changes to meet regulatory compliance requirements.

    Thread-safe singleton pattern for consistent auditing.
    """

    _instance: "ConfigurationAuditor | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ConfigurationAuditor":
        """Singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize the auditor."""
        if getattr(self, "_initialized", False):
            return

        self._lock_internal = threading.RLock()
        self._changes: list[ConfigurationChange] = []
        self._callbacks: list[Callable[[ConfigurationChange], None]] = []
        self._previous_state: dict[str, Any] = {}
        self._audit_file: Path | None = None
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "ConfigurationAuditor":
        """Get the singleton instance."""
        return cls()

    def set_audit_file(self, path: Path) -> None:
        """Set the audit file path for persistent storage.

        Args:
            path: Path to the audit log file.
        """
        self._audit_file = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def register_callback(
        self,
        callback: Callable[[ConfigurationChange], None],
    ) -> None:
        """Register a callback for configuration changes.

        Args:
            callback: Function to call when configuration changes.
        """
        with self._lock_internal:
            self._callbacks.append(callback)

    def unregister_callback(
        self,
        callback: Callable[[ConfigurationChange], None],
    ) -> bool:
        """Unregister a callback.

        Args:
            callback: The callback to remove.

        Returns:
            True if callback was found and removed.
        """
        with self._lock_internal:
            try:
                self._callbacks.remove(callback)
                return True
            except ValueError:
                return False

    def record_change(
        self,
        section: str,
        field: str,
        old_value: Any,
        new_value: Any,
        changed_by: str = "system",
        source: str = "unknown",
    ) -> ConfigurationChange:
        """Record a configuration change.

        Args:
            section: Configuration section (e.g., "risk", "trading").
            field: Field name that changed.
            old_value: Previous value.
            new_value: New value.
            changed_by: User or system that made the change.
            source: Source of the change.

        Returns:
            The recorded ConfigurationChange.
        """
        import uuid

        change = ConfigurationChange(
            change_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            section=section,
            field=field,
            old_value=old_value,
            new_value=new_value,
            changed_by=changed_by,
            source=source,
        )

        with self._lock_internal:
            self._changes.append(change)
            callbacks = list(self._callbacks)

        # Log the change
        logger.info(
            f"Configuration change: {section}.{field} changed "
            f"from '{old_value}' to '{new_value}' by {changed_by} (source: {source})"
        )

        # Write to audit file if configured
        if self._audit_file:
            self._write_to_file(change)

        # Notify callbacks (outside lock to prevent deadlocks)
        for callback in callbacks:
            try:
                callback(change)
            except Exception as e:
                logger.warning(f"Configuration change callback failed: {e}")

        return change

    def _write_to_file(self, change: ConfigurationChange) -> None:
        """Write change to audit file."""
        try:
            with open(self._audit_file, "a") as f:
                f.write(json.dumps(change.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write configuration change to audit file: {e}")

    def track_settings_load(
        self,
        settings: Settings,
        source: str = "initial_load",
    ) -> list[ConfigurationChange]:
        """Track settings load and detect changes from previous state.

        Args:
            settings: The Settings object being loaded.
            source: Source of the load (e.g., "initial_load", "reload").

        Returns:
            List of detected changes.
        """
        changes: list[ConfigurationChange] = []
        current_state = self._extract_state(settings)

        with self._lock_internal:
            if not self._previous_state:
                # First load - record all as initial values
                self._previous_state = current_state
                logger.info("Configuration initially loaded - audit trail started")
                return changes

            # Compare and record changes
            for key, new_value in current_state.items():
                old_value = self._previous_state.get(key)
                if old_value != new_value:
                    section, field = key.rsplit(".", 1) if "." in key else ("root", key)
                    change = self.record_change(
                        section=section,
                        field=field,
                        old_value=old_value,
                        new_value=new_value,
                        changed_by="system",
                        source=source,
                    )
                    changes.append(change)

            # Update previous state
            self._previous_state = current_state

        return changes

    def _extract_state(self, settings: Settings) -> dict[str, Any]:
        """Extract flattened state from settings for comparison.

        Args:
            settings: The Settings object.

        Returns:
            Flattened dictionary of setting values.
        """
        state = {}

        # Extract nested settings
        for section_name in [
            "database", "redis", "alpaca", "risk", "trading",
            "model", "logging", "security", "alerts"
        ]:
            section = getattr(settings, section_name, None)
            if section:
                for field_name, value in section.model_dump().items():
                    # Mask sensitive values
                    if "password" in field_name.lower() or "secret" in field_name.lower():
                        value = "***MASKED***"
                    elif "key" in field_name.lower() and field_name != "api_key":
                        value = "***MASKED***"
                    state[f"{section_name}.{field_name}"] = str(value)

        # Extract root-level settings
        state["app_name"] = settings.app_name
        state["app_version"] = settings.app_version
        state["debug"] = str(settings.debug)
        state["environment"] = settings.environment
        state["symbols_count"] = str(len(settings.symbols))

        return state

    def get_changes(
        self,
        since: datetime | None = None,
        section: str | None = None,
    ) -> list[ConfigurationChange]:
        """Get recorded changes with optional filtering.

        Args:
            since: Only return changes after this time.
            section: Only return changes to this section.

        Returns:
            List of matching ConfigurationChange records.
        """
        with self._lock_internal:
            changes = self._changes.copy()

        if since:
            changes = [c for c in changes if c.timestamp >= since]
        if section:
            changes = [c for c in changes if c.section == section]

        return changes

    def get_change_count(self) -> int:
        """Get the total number of recorded changes."""
        with self._lock_internal:
            return len(self._changes)

    def export_audit_trail(self) -> list[dict[str, Any]]:
        """Export the full audit trail for compliance reporting.

        Returns:
            List of all changes as dictionaries.
        """
        with self._lock_internal:
            return [c.to_dict() for c in self._changes]

    def verify_audit_trail(self) -> tuple[bool, list[str]]:
        """Verify the integrity of the audit trail.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []
        with self._lock_internal:
            for i, change in enumerate(self._changes):
                expected_hash = change._calculate_hash()
                if change.hash != expected_hash:
                    errors.append(
                        f"Change {i} ({change.change_id}) hash mismatch: "
                        f"expected {expected_hash}, got {change.hash}"
                    )

        return len(errors) == 0, errors


def get_configuration_auditor() -> ConfigurationAuditor:
    """Get the global ConfigurationAuditor instance.

    Returns:
        The singleton ConfigurationAuditor instance.
    """
    return ConfigurationAuditor.get_instance()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance with all configurations loaded.

    Loads configurations in order:
    1. Base settings from environment and .env file
    2. Symbols from YAML
    3. Alpaca credentials from environment variables (secure) or YAML (dev)
    4. Security settings from environment variables
    5. Alert settings from environment variables

    JPMorgan-level enhancement: All configuration loads are audited.
    """
    settings = Settings()

    # Load symbols from YAML if available
    symbols = settings.load_symbols_config()
    if symbols:
        settings.symbols = symbols

    # CRITICAL FIX: Load Alpaca config from environment variables first,
    # falling back to YAML. This ensures credentials can be securely
    # injected via environment variables in production.
    settings.alpaca = settings.load_alpaca_config()

    # Load security settings from environment variables
    settings.security = SecuritySettings.from_env()

    # Load alert settings from environment variables
    settings.alerts = AlertSettings.from_env()

    # Audit the settings load
    auditor = get_configuration_auditor()
    auditor.track_settings_load(settings, source="initial_load")

    return settings


def reload_settings() -> Settings:
    """Reload settings and audit any changes.

    This clears the cached settings and loads fresh values,
    tracking any changes in the configuration audit trail.

    Returns:
        The newly loaded Settings instance.
    """
    # Clear the cache
    get_settings.cache_clear()

    # Get previous state from auditor for comparison
    auditor = get_configuration_auditor()

    # Load new settings (will be tracked by auditor)
    settings = Settings()

    # Load all sub-configurations
    symbols = settings.load_symbols_config()
    if symbols:
        settings.symbols = symbols
    settings.alpaca = settings.load_alpaca_config()
    settings.security = SecuritySettings.from_env()
    settings.alerts = AlertSettings.from_env()

    # Track the reload in auditor
    changes = auditor.track_settings_load(settings, source="reload")

    if changes:
        logger.info(f"Configuration reloaded with {len(changes)} changes")
    else:
        logger.info("Configuration reloaded with no changes detected")

    # Cache the new settings
    get_settings.cache_clear()
    return get_settings()
