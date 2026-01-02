"""
Pytest fixtures for the Quant Trading System tests.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV data for testing."""
    from datetime import datetime
    from decimal import Decimal

    return {
        "symbol": "AAPL",
        "timestamp": datetime(2024, 1, 15, 10, 30, 0),
        "open": Decimal("185.50"),
        "high": Decimal("186.75"),
        "low": Decimal("184.25"),
        "close": Decimal("186.00"),
        "volume": 1500000,
        "vwap": Decimal("185.75"),
        "trade_count": 12500,
    }


@pytest.fixture
def sample_order_data():
    """Sample order data for testing."""
    from decimal import Decimal

    return {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": Decimal("100"),
        "order_type": "MARKET",
        "time_in_force": "DAY",
    }


@pytest.fixture
def sample_position_data():
    """Sample position data for testing."""
    from decimal import Decimal

    return {
        "symbol": "AAPL",
        "quantity": Decimal("100"),
        "avg_entry_price": Decimal("185.50"),
        "current_price": Decimal("186.00"),
        "cost_basis": Decimal("18550.00"),
        "market_value": Decimal("18600.00"),
        "unrealized_pnl": Decimal("50.00"),
        "realized_pnl": Decimal("0"),
    }


@pytest.fixture
def event_bus():
    """Create a fresh EventBus instance for testing.

    Resets the singleton instance before each test to ensure
    clean state and prevent test pollution.
    """
    from quant_trading_system.core.events import EventBus

    # Reset the singleton to ensure fresh instance
    EventBus.reset_instance()
    bus = EventBus()

    yield bus

    # Clean up after test
    EventBus.reset_instance()


@pytest.fixture
def component_registry():
    """Create a fresh ComponentRegistry instance for testing."""
    from quant_trading_system.core.registry import ComponentRegistry

    return ComponentRegistry()


@pytest.fixture
def sample_bars():
    """Create sample OHLCV bars for testing."""
    from datetime import datetime
    from decimal import Decimal
    from quant_trading_system.core.data_types import OHLCVBar

    return [
        OHLCVBar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 15, 10 + i // 4, (i % 4) * 15, 0),
            open=Decimal(f"{185.50 + i * 0.5}"),
            high=Decimal(f"{186.75 + i * 0.5}"),
            low=Decimal(f"{184.25 + i * 0.5}"),
            close=Decimal(f"{186.00 + i * 0.5}"),
            volume=1500000 + i * 10000,
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_signal():
    """Create a sample trade signal for testing."""
    from quant_trading_system.core.data_types import TradeSignal, Direction

    return TradeSignal(
        symbol="AAPL",
        direction=Direction.LONG,
        strength=0.75,
        confidence=0.8,
        horizon=4,
        model_source="xgboost",
    )


@pytest.fixture
def metrics_collector():
    """Create a fresh MetricsCollector for testing."""
    from quant_trading_system.monitoring.metrics import MetricsCollector

    # Reset singleton
    MetricsCollector._instance = None
    return MetricsCollector()


@pytest.fixture
def alert_manager():
    """Create a fresh AlertManager for testing."""
    from quant_trading_system.monitoring.alerting import AlertManager
    from quant_trading_system.monitoring import alerting

    # Reset singleton
    alerting._alert_manager = None
    return AlertManager()


@pytest.fixture
def dashboard_state():
    """Create a fresh DashboardState for testing."""
    from quant_trading_system.monitoring.dashboard import DashboardState

    return DashboardState()


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    from unittest.mock import MagicMock

    settings = MagicMock()
    settings.app_name = "Quant Trading System"
    settings.app_version = "1.0.0"
    settings.environment = "test"
    settings.debug = False
    settings.symbols = ["AAPL", "MSFT", "GOOGL"]

    # Database settings
    settings.database.host = "localhost"
    settings.database.port = 5432
    settings.database.name = "test_db"
    settings.database.url = "postgresql://test@localhost/test_db"

    # Redis settings
    settings.redis.host = "localhost"
    settings.redis.port = 6379
    settings.redis.url = "redis://localhost:6379/0"

    # Risk settings
    settings.risk.max_position_pct = 0.10
    settings.risk.max_portfolio_positions = 20

    # Trading settings
    settings.trading.bar_timeframe = "15Min"
    settings.trading.signal_threshold = 0.5

    return settings


@pytest.fixture
def temp_log_dir(tmp_path):
    """Create a temporary directory for log files."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration between tests."""
    import logging

    # Store original handlers
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers.copy()
    original_level = root_logger.level

    yield

    # Restore original configuration
    root_logger.handlers = original_handlers
    root_logger.level = original_level


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    from decimal import Decimal
    from quant_trading_system.core.data_types import Portfolio, Position

    positions = {
        "AAPL": Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            avg_entry_price=Decimal("185.50"),
            current_price=Decimal("186.00"),
            cost_basis=Decimal("18550.00"),
            market_value=Decimal("18600.00"),
            unrealized_pnl=Decimal("50.00"),
        ),
        "MSFT": Position(
            symbol="MSFT",
            quantity=Decimal("50"),
            avg_entry_price=Decimal("380.00"),
            current_price=Decimal("382.00"),
            cost_basis=Decimal("19000.00"),
            market_value=Decimal("19100.00"),
            unrealized_pnl=Decimal("100.00"),
        ),
    }

    return Portfolio(
        equity=Decimal("100000.00"),
        cash=Decimal("62850.00"),
        buying_power=Decimal("162850.00"),
        positions=positions,
        daily_pnl=Decimal("150.00"),
        total_pnl=Decimal("500.00"),
    )


@pytest.fixture
def sample_risk_metrics():
    """Create sample risk metrics for testing."""
    from decimal import Decimal
    from quant_trading_system.core.data_types import RiskMetrics

    return RiskMetrics(
        portfolio_var_95=Decimal("5000.00"),
        portfolio_var_99=Decimal("7500.00"),
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        max_drawdown=0.08,
        current_drawdown=0.03,
        beta=1.1,
        volatility_annual=0.15,
    )


@pytest.fixture(autouse=True)
def disable_dashboard_auth(monkeypatch):
    """Disable dashboard authentication for all tests.

    This fixture automatically runs for all tests and sets the
    REQUIRE_AUTH environment variable to 'false' so dashboard
    endpoints don't require authentication during testing.
    """
    import os

    # Set environment variables BEFORE clearing the settings cache
    monkeypatch.setenv("REQUIRE_AUTH", "false")
    # Also set a test JWT secret to prevent warnings
    if not os.environ.get("JWT_SECRET_KEY"):
        monkeypatch.setenv("JWT_SECRET_KEY", "test_secret_key_for_testing_only")

    # Clear the settings cache so it picks up the new environment variables
    from quant_trading_system.config.settings import get_settings
    get_settings.cache_clear()

    yield

    # Clear cache again after test to not affect subsequent tests
    get_settings.cache_clear()
