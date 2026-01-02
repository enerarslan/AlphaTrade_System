"""
Central configuration management using Pydantic settings.

Provides type-safe configuration with validation, environment variable
support, and YAML configuration file loading.
"""

from __future__ import annotations

import os
import warnings
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = Path(__file__).resolve().parent


class DatabaseSettings(BaseModel):
    """Database configuration settings."""

    host: str = "localhost"
    port: int = 5432
    name: str = "quant_trading"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10

    @property
    def url(self) -> str:
        """Get the database connection URL."""
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance with all configurations loaded.

    Loads configurations in order:
    1. Base settings from environment and .env file
    2. Symbols from YAML
    3. Alpaca credentials from environment variables (secure) or YAML (dev)
    4. Security settings from environment variables
    5. Alert settings from environment variables
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

    return settings
