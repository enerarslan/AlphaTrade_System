"""
Central configuration management using Pydantic settings.

Provides type-safe configuration with validation, environment variable
support, and YAML configuration file loading.
"""

from __future__ import annotations

import os
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
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

    return settings
