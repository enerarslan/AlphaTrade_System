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
    """Create a fresh EventBus instance for testing."""
    from quant_trading_system.core.events import EventBus

    return EventBus()


@pytest.fixture
def component_registry():
    """Create a fresh ComponentRegistry instance for testing."""
    from quant_trading_system.core.registry import ComponentRegistry

    return ComponentRegistry()
