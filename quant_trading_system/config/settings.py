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
    """Alpaca broker configuration settings."""

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
        """Load Alpaca configuration from YAML."""
        config = self.load_yaml_config(CONFIG_DIR / "alpaca_config.yaml")
        if config:
            return AlpacaSettings(**config)
        return self.alpaca


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()

    # Load symbols from YAML if available
    symbols = settings.load_symbols_config()
    if symbols:
        settings.symbols = symbols

    return settings
