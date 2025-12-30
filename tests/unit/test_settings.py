"""
Unit tests for config/settings.py
"""

import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from quant_trading_system.config.settings import (
    DatabaseSettings,
    RedisSettings,
    AlpacaSettings,
    RiskSettings,
    TradingSettings,
    ModelSettings,
    LoggingSettings,
    Settings,
    get_settings,
)


class TestDatabaseSettings:
    """Tests for DatabaseSettings model."""

    def test_default_values(self):
        """Test default database settings."""
        settings = DatabaseSettings()
        assert settings.host == "localhost"
        assert settings.port == 5432
        assert settings.name == "quant_trading"
        assert settings.user == "postgres"
        assert settings.pool_size == 5

    def test_url_property(self):
        """Test database URL generation."""
        settings = DatabaseSettings(
            host="db.example.com",
            port=5433,
            name="trading_db",
            user="admin",
            password="secret",
        )
        url = settings.url
        assert "postgresql://" in url
        assert "admin:secret" in url
        assert "db.example.com:5433" in url
        assert "trading_db" in url


class TestRedisSettings:
    """Tests for RedisSettings model."""

    def test_default_values(self):
        """Test default Redis settings."""
        settings = RedisSettings()
        assert settings.host == "localhost"
        assert settings.port == 6379
        assert settings.db == 0

    def test_url_without_password(self):
        """Test Redis URL without password."""
        settings = RedisSettings()
        url = settings.url
        assert url == "redis://localhost:6379/0"

    def test_url_with_password(self):
        """Test Redis URL with password."""
        settings = RedisSettings(password="secret")
        url = settings.url
        assert ":secret@" in url


class TestAlpacaSettings:
    """Tests for AlpacaSettings model."""

    def test_default_values(self):
        """Test default Alpaca settings."""
        settings = AlpacaSettings()
        assert settings.paper_trading is True
        assert "paper-api" in settings.base_url

    def test_custom_values(self):
        """Test custom Alpaca settings."""
        settings = AlpacaSettings(
            api_key="test_key",
            api_secret="test_secret",
            paper_trading=False,
        )
        assert settings.api_key == "test_key"
        assert settings.api_secret == "test_secret"


class TestRiskSettings:
    """Tests for RiskSettings model."""

    def test_default_values(self):
        """Test default risk settings."""
        settings = RiskSettings()
        assert settings.max_position_pct == Decimal("0.10")
        assert settings.max_portfolio_positions == 20
        assert settings.daily_loss_limit_pct == Decimal("0.02")
        assert settings.var_confidence_level == 0.95

    def test_custom_values(self):
        """Test custom risk settings."""
        settings = RiskSettings(
            max_position_pct=Decimal("0.05"),
            max_portfolio_positions=10,
            kelly_fraction=0.5,
        )
        assert settings.max_position_pct == Decimal("0.05")
        assert settings.max_portfolio_positions == 10
        assert settings.kelly_fraction == 0.5


class TestTradingSettings:
    """Tests for TradingSettings model."""

    def test_default_values(self):
        """Test default trading settings."""
        settings = TradingSettings()
        assert settings.bar_timeframe == "15Min"
        assert settings.lookback_bars == 100
        assert settings.signal_threshold == 0.5
        assert settings.min_confidence == 0.6

    def test_market_hours(self):
        """Test market hours settings."""
        settings = TradingSettings()
        assert settings.market_open_time == "09:30"
        assert settings.market_close_time == "16:00"


class TestModelSettings:
    """Tests for ModelSettings model."""

    def test_default_values(self):
        """Test default model settings."""
        settings = ModelSettings()
        assert settings.default_model == "xgboost"
        assert "xgboost" in settings.ensemble_models
        assert settings.validation_split == 0.2

    def test_ensemble_models(self):
        """Test ensemble models configuration."""
        settings = ModelSettings()
        assert len(settings.ensemble_models) == 3
        assert "lightgbm" in settings.ensemble_models
        assert "catboost" in settings.ensemble_models


class TestLoggingSettings:
    """Tests for LoggingSettings model."""

    def test_default_values(self):
        """Test default logging settings."""
        settings = LoggingSettings()
        assert settings.level == "INFO"
        assert settings.format == "json"
        assert settings.rotation == "1 day"

    def test_custom_level(self):
        """Test custom log level."""
        settings = LoggingSettings(level="DEBUG")
        assert settings.level == "DEBUG"


class TestSettings:
    """Tests for main Settings class."""

    @patch.dict("os.environ", {"DEBUG": "false"}, clear=False)
    def test_default_values(self):
        """Test default settings."""
        settings = Settings(_env_file=None)  # Ignore .env file for test
        assert settings.app_name == "Quant Trading System"
        assert settings.debug is False
        assert settings.environment == "development"

    def test_nested_settings(self):
        """Test nested settings models."""
        settings = Settings()
        assert isinstance(settings.database, DatabaseSettings)
        assert isinstance(settings.redis, RedisSettings)
        assert isinstance(settings.alpaca, AlpacaSettings)
        assert isinstance(settings.risk, RiskSettings)
        assert isinstance(settings.trading, TradingSettings)
        assert isinstance(settings.model, ModelSettings)
        assert isinstance(settings.logging, LoggingSettings)

    def test_load_yaml_config_nonexistent(self):
        """Test loading non-existent YAML config."""
        config = Settings.load_yaml_config(Path("/nonexistent/config.yaml"))
        assert config == {}

    def test_load_yaml_config_valid(self):
        """Test loading valid YAML config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"test_key": "test_value"}, f)
            f.flush()
            config = Settings.load_yaml_config(Path(f.name))
            assert config["test_key"] == "test_value"

    def test_load_symbols_config(self):
        """Test loading symbols config."""
        settings = Settings()
        # This will return empty list if symbols.yaml doesn't exist
        symbols = settings.load_symbols_config()
        assert isinstance(symbols, list)


class TestGetSettings:
    """Tests for get_settings function."""

    def test_returns_settings_instance(self):
        """Test that get_settings returns Settings instance."""
        # Clear cache
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_caching(self):
        """Test that get_settings is cached."""
        get_settings.cache_clear()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
