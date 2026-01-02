"""
Unit tests for the database module.

Tests database connection management, ORM models, and repositories.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Check if redis is available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class TestDatabaseManager:
    """Tests for DatabaseManager class."""

    def test_database_manager_initialization(self):
        """Test DatabaseManager initializes correctly."""
        with patch("quant_trading_system.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.database = MagicMock(
                url="postgresql://user:pass@localhost:5432/test",
                pool_size=5,
                max_overflow=10,
            )
            mock_settings.return_value.debug = False

            from quant_trading_system.database.connection import DatabaseManager

            manager = DatabaseManager(mock_settings.return_value)

            assert manager._engine is None
            assert manager._async_engine is None

    def test_database_manager_sync_session_context(self):
        """Test sync session context manager."""
        with patch("quant_trading_system.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.database = MagicMock(
                url="postgresql://user:pass@localhost:5432/test",
                pool_size=5,
                max_overflow=10,
            )
            mock_settings.return_value.debug = False

            from quant_trading_system.database.connection import DatabaseManager

            manager = DatabaseManager(mock_settings.return_value)

            # Create mock session and session factory
            mock_session = MagicMock()
            mock_session_factory = MagicMock(return_value=mock_session)
            manager._session_factory = mock_session_factory

            with manager.session() as session:
                assert session is mock_session

            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()


class TestRedisManager:
    """Tests for RedisManager class."""

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not installed")
    def test_redis_manager_initialization(self):
        """Test RedisManager initializes correctly."""
        with patch("quant_trading_system.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.redis = MagicMock(
                host="localhost",
                port=6379,
                db=0,
                password=None,
            )

            from quant_trading_system.database.connection import RedisManager

            manager = RedisManager(mock_settings.return_value)

            assert manager._client is None
            assert manager._pool is None

    def test_redis_manager_requires_redis_package(self):
        """Test RedisManager raises ImportError when redis not available."""
        if REDIS_AVAILABLE:
            pytest.skip("Redis is installed, skipping import test")

        with patch("quant_trading_system.database.connection.REDIS_AVAILABLE", False):
            from quant_trading_system.database.connection import RedisManager

            with pytest.raises(ImportError):
                RedisManager()


class TestOHLCVBarModel:
    """Tests for OHLCVBar SQLAlchemy model."""

    def test_ohlcv_bar_creation(self):
        """Test OHLCVBar model creation."""
        from quant_trading_system.database.models import OHLCVBar

        bar = OHLCVBar(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("151.00"),
            volume=1000000,
            vwap=Decimal("150.50"),
            trade_count=5000,
        )

        assert bar.symbol == "AAPL"
        assert bar.open == Decimal("150.00")
        assert bar.volume == 1000000

    def test_ohlcv_bar_to_dict(self):
        """Test OHLCVBar to_dict method."""
        from quant_trading_system.database.models import OHLCVBar

        timestamp = datetime.now(timezone.utc)
        bar = OHLCVBar(
            symbol="AAPL",
            timestamp=timestamp,
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("151.00"),
            volume=1000000,
        )

        # Test with float conversion (preserve_decimal_precision=False)
        result = bar.to_dict(preserve_decimal_precision=False)

        assert result["symbol"] == "AAPL"
        assert result["timestamp"] == timestamp.isoformat()
        assert result["open"] == 150.00

    def test_ohlcv_bar_to_dict_preserves_precision(self):
        """Test OHLCVBar to_dict method preserves decimal precision by default."""
        from quant_trading_system.database.models import OHLCVBar

        timestamp = datetime.now(timezone.utc)
        bar = OHLCVBar(
            symbol="AAPL",
            timestamp=timestamp,
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("151.00"),
            volume=1000000,
        )

        # Default preserves precision as strings
        result = bar.to_dict()

        assert result["symbol"] == "AAPL"
        assert result["timestamp"] == timestamp.isoformat()
        assert result["open"] == "150.00"


class TestOrderModel:
    """Tests for Order SQLAlchemy model."""

    def test_order_creation(self):
        """Test Order model creation."""
        from quant_trading_system.database.models import Order

        order_id = uuid4()
        order = Order(
            order_id=str(order_id),
            client_order_id="test-001",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
            status="PENDING",
        )

        assert order.symbol == "AAPL"
        assert order.quantity == Decimal("100")
        assert order.status == "PENDING"

    def test_order_to_dict(self):
        """Test Order to_dict method."""
        from quant_trading_system.database.models import Order

        order = Order(
            order_id=str(uuid4()),
            client_order_id="test-001",
            symbol="AAPL",
            side="BUY",
            order_type="MARKET",
            quantity=Decimal("100"),
            status="FILLED",
        )

        # Test with float conversion (preserve_decimal_precision=False)
        result = order.to_dict(preserve_decimal_precision=False)

        assert result["symbol"] == "AAPL"
        assert result["side"] == "BUY"
        assert result["quantity"] == 100.0

    def test_order_to_dict_preserves_precision(self):
        """Test Order to_dict method preserves decimal precision by default."""
        from quant_trading_system.database.models import Order

        order = Order(
            order_id=str(uuid4()),
            client_order_id="test-001",
            symbol="AAPL",
            side="BUY",
            order_type="MARKET",
            quantity=Decimal("100"),
            status="FILLED",
        )

        # Default preserves precision as strings
        result = order.to_dict()

        assert result["symbol"] == "AAPL"
        assert result["side"] == "BUY"
        assert result["quantity"] == "100"


class TestTradeModel:
    """Tests for Trade SQLAlchemy model."""

    def test_trade_creation(self):
        """Test Trade model creation."""
        from quant_trading_system.database.models import Trade

        trade = Trade(
            trade_id=str(uuid4()),
            order_id=str(uuid4()),
            symbol="AAPL",
            side="BUY",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("1.00"),
            executed_at=datetime.now(timezone.utc),
        )

        assert trade.symbol == "AAPL"
        assert trade.quantity == Decimal("100")
        assert trade.price == Decimal("150.00")


class TestPositionModel:
    """Tests for Position SQLAlchemy model."""

    def test_position_creation(self):
        """Test Position model creation."""
        from quant_trading_system.database.models import Position

        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            avg_entry_price=Decimal("150.00"),
            cost_basis=Decimal("15000.00"),
        )

        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")
        assert position.cost_basis == Decimal("15000.00")


class TestSignalModel:
    """Tests for Signal SQLAlchemy model."""

    def test_signal_creation(self):
        """Test Signal model creation."""
        from quant_trading_system.database.models import Signal

        signal = Signal(
            signal_id=str(uuid4()),
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            confidence=0.9,
            horizon=4,
            model_source="xgboost_v1",
        )

        assert signal.symbol == "AAPL"
        assert signal.direction == "LONG"
        assert signal.strength == 0.8


class TestDailyPerformanceModel:
    """Tests for DailyPerformance SQLAlchemy model."""

    def test_daily_performance_creation(self):
        """Test DailyPerformance model creation."""
        from quant_trading_system.database.models import DailyPerformance

        perf = DailyPerformance(
            date=datetime.now(timezone.utc).date(),
            starting_equity=Decimal("100000.00"),
            ending_equity=Decimal("101000.00"),
            pnl=Decimal("1000.00"),
            pnl_percent=1.0,
            trades_count=10,
            win_count=6,
            loss_count=4,
        )

        assert perf.pnl == Decimal("1000.00")
        assert perf.trades_count == 10


class TestAlertModel:
    """Tests for Alert SQLAlchemy model."""

    def test_alert_creation(self):
        """Test Alert model creation."""
        from quant_trading_system.database.models import Alert

        alert = Alert(
            alert_id=str(uuid4()),
            severity="WARNING",
            title="Test Alert",
            message="This is a test alert",
        )

        assert alert.severity == "WARNING"
        assert alert.title == "Test Alert"


class TestBaseRepository:
    """Tests for BaseRepository class."""

    def test_base_repository_initialization(self):
        """Test BaseRepository initialization."""
        from quant_trading_system.database.repository import OHLCVRepository

        mock_db_manager = MagicMock()
        repo = OHLCVRepository(mock_db_manager)

        assert repo._db is mock_db_manager


class TestOHLCVRepository:
    """Tests for OHLCVRepository class."""

    def test_ohlcv_repository_initialization(self):
        """Test OHLCVRepository initialization."""
        from quant_trading_system.database.repository import OHLCVRepository

        mock_db_manager = MagicMock()
        repo = OHLCVRepository(mock_db_manager)

        assert repo._db is mock_db_manager


class TestOrderRepository:
    """Tests for OrderRepository class."""

    def test_order_repository_initialization(self):
        """Test OrderRepository initialization."""
        from quant_trading_system.database.repository import OrderRepository

        mock_db_manager = MagicMock()
        repo = OrderRepository(mock_db_manager)

        assert repo._db is mock_db_manager


class TestTradeRepository:
    """Tests for TradeRepository class."""

    def test_trade_repository_initialization(self):
        """Test TradeRepository initialization."""
        from quant_trading_system.database.repository import TradeRepository

        mock_db_manager = MagicMock()
        repo = TradeRepository(mock_db_manager)

        assert repo._db is mock_db_manager


class TestPositionRepository:
    """Tests for PositionRepository class."""

    def test_position_repository_initialization(self):
        """Test PositionRepository initialization."""
        from quant_trading_system.database.repository import PositionRepository

        mock_db_manager = MagicMock()
        repo = PositionRepository(mock_db_manager)

        assert repo._db is mock_db_manager


class TestSignalRepository:
    """Tests for SignalRepository class."""

    def test_signal_repository_initialization(self):
        """Test SignalRepository initialization."""
        from quant_trading_system.database.repository import SignalRepository

        mock_db_manager = MagicMock()
        repo = SignalRepository(mock_db_manager)

        assert repo._db is mock_db_manager


class TestAlertRepository:
    """Tests for AlertRepository class."""

    def test_alert_repository_initialization(self):
        """Test AlertRepository initialization."""
        from quant_trading_system.database.repository import AlertRepository

        mock_db_manager = MagicMock()
        repo = AlertRepository(mock_db_manager)

        assert repo._db is mock_db_manager


class TestDailyPerformanceRepository:
    """Tests for DailyPerformanceRepository class."""

    def test_daily_performance_repository_initialization(self):
        """Test DailyPerformanceRepository initialization."""
        from quant_trading_system.database.repository import DailyPerformanceRepository

        mock_db_manager = MagicMock()
        repo = DailyPerformanceRepository(mock_db_manager)

        assert repo._db is mock_db_manager
