"""
Unit tests for execution module.

Tests for:
- AlpacaClient
- OrderManager
- ExecutionAlgorithms (TWAP, VWAP)
- PositionTracker
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from quant_trading_system.core.data_types import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    TimeInForce,
)
from quant_trading_system.core.events import EventBus
from quant_trading_system.execution.alpaca_client import (
    AccountInfo,
    AlpacaClient,
    AlpacaOrder,
    AlpacaPosition,
    OrderClass,
    RateLimiter,
    TradingEnvironment,
    create_bracket_order,
    create_limit_order,
    create_market_order,
)
from quant_trading_system.execution.order_manager import (
    ManagedOrder,
    OrderManager,
    OrderPriority,
    OrderRequest,
    OrderState,
    OrderValidator,
    OrderValidationResult,
    SmartOrderRouter,
)
from quant_trading_system.execution.execution_algo import (
    AlgoExecutionEngine,
    AlgoExecutionState,
    AlgoStatus,
    AlgoType,
    ImplementationShortfallAlgorithm,
    SliceOrder,
    TWAPAlgorithm,
    VWAPAlgorithm,
)
from quant_trading_system.execution.position_tracker import (
    CashState,
    PositionState,
    PositionTracker,
    ReconciliationResult,
    SettlementStatus,
    TradeRecord,
)


# =============================================================================
# Rate Limiter Tests
# =============================================================================

class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.mark.asyncio
    async def test_can_proceed_when_under_limit(self):
        """Test that requests proceed when under limit."""
        limiter = RateLimiter(requests_per_minute=10)
        assert limiter.can_proceed() is True

    @pytest.mark.asyncio
    async def test_acquire_tracks_requests(self):
        """Test that acquire tracks request times."""
        limiter = RateLimiter(requests_per_minute=100)
        await limiter.acquire()
        assert len(limiter._request_times) == 1


# =============================================================================
# AlpacaClient Tests
# =============================================================================

class TestAlpacaClient:
    """Tests for AlpacaClient."""

    def test_initialization_paper_mode(self):
        """Test client initialization in paper mode."""
        client = AlpacaClient(
            api_key="test_key",
            api_secret="test_secret",
            environment=TradingEnvironment.PAPER,
        )
        assert client.environment == TradingEnvironment.PAPER
        assert client.base_url == AlpacaClient.PAPER_BASE_URL

    def test_initialization_live_mode(self):
        """Test client initialization in live mode."""
        client = AlpacaClient(
            api_key="test_key",
            api_secret="test_secret",
            environment=TradingEnvironment.LIVE,
        )
        assert client.environment == TradingEnvironment.LIVE
        assert client.base_url == AlpacaClient.LIVE_BASE_URL

    def test_headers_contain_auth(self):
        """Test that headers contain authentication."""
        client = AlpacaClient(api_key="key123", api_secret="secret456")
        headers = client.headers
        assert headers["APCA-API-KEY-ID"] == "key123"
        assert headers["APCA-API-SECRET-KEY"] == "secret456"


class TestAccountInfo:
    """Tests for AccountInfo dataclass."""

    def test_from_alpaca_creates_instance(self):
        """Test creating AccountInfo from Alpaca response."""
        data = {
            "id": "acc123",
            "status": "ACTIVE",
            "currency": "USD",
            "cash": "10000.00",
            "portfolio_value": "15000.00",
            "buying_power": "20000.00",
            "daytrading_buying_power": "40000.00",
            "regt_buying_power": "20000.00",
            "equity": "15000.00",
            "last_equity": "14500.00",
            "long_market_value": "5000.00",
            "short_market_value": "0",
            "initial_margin": "2500.00",
            "maintenance_margin": "2000.00",
            "multiplier": "2",
            "pattern_day_trader": False,
            "trading_blocked": False,
            "transfers_blocked": False,
            "account_blocked": False,
            "created_at": "2024-01-01T00:00:00Z",
        }
        account = AccountInfo.from_alpaca(data)
        assert account.account_id == "acc123"
        assert account.cash == Decimal("10000.00")
        assert account.buying_power == Decimal("20000.00")


class TestAlpacaPosition:
    """Tests for AlpacaPosition dataclass."""

    def test_from_alpaca_creates_instance(self):
        """Test creating AlpacaPosition from Alpaca response."""
        data = {
            "symbol": "AAPL",
            "qty": "100",
            "avg_entry_price": "150.00",
            "current_price": "155.00",
            "market_value": "15500.00",
            "unrealized_pl": "500.00",
            "unrealized_plpc": "0.0333",
            "cost_basis": "15000.00",
            "asset_class": "us_equity",
            "asset_id": "asset123",
            "side": "long",
            "exchange": "NASDAQ",
        }
        position = AlpacaPosition.from_alpaca(data)
        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")
        assert position.current_price == Decimal("155.00")

    def test_to_position_converts_correctly(self):
        """Test conversion to internal Position model."""
        alpaca_pos = AlpacaPosition(
            symbol="AAPL",
            quantity=Decimal("100"),
            avg_entry_price=Decimal("150"),
            current_price=Decimal("155"),
            market_value=Decimal("15500"),
            unrealized_pnl=Decimal("500"),
            unrealized_pnl_pct=Decimal("0.0333"),
            cost_basis=Decimal("15000"),
            asset_class="us_equity",
            asset_id="asset123",
            side="long",
            exchange="NASDAQ",
        )
        position = alpaca_pos.to_position()
        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")


class TestAlpacaOrder:
    """Tests for AlpacaOrder dataclass."""

    def test_to_order_converts_correctly(self):
        """Test conversion to internal Order model."""
        alpaca_order = AlpacaOrder(
            order_id="order123",
            client_order_id="client123",
            symbol="AAPL",
            side="buy",
            order_type="market",
            time_in_force="day",
            quantity=Decimal("100"),
            filled_qty=Decimal("50"),
            filled_avg_price=Decimal("150"),
            limit_price=None,
            stop_price=None,
            status="partially_filled",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            submitted_at=datetime.now(timezone.utc),
            filled_at=None,
            expired_at=None,
            cancelled_at=None,
            asset_class="us_equity",
            order_class="simple",
            extended_hours=False,
        )
        order = alpaca_order.to_order()
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.status == OrderStatus.PARTIAL_FILLED


# =============================================================================
# Order Helper Function Tests
# =============================================================================

class TestOrderHelpers:
    """Tests for order creation helper functions."""

    def test_create_market_order(self):
        """Test market order creation."""
        params = create_market_order(
            symbol="AAPL",
            qty=Decimal("100"),
            side=OrderSide.BUY,
        )
        assert params["symbol"] == "AAPL"
        assert params["order_type"] == OrderType.MARKET

    def test_create_limit_order(self):
        """Test limit order creation."""
        params = create_limit_order(
            symbol="AAPL",
            qty=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150"),
        )
        assert params["limit_price"] == Decimal("150")
        assert params["order_type"] == OrderType.LIMIT

    def test_create_bracket_order(self):
        """Test bracket order creation."""
        params = create_bracket_order(
            symbol="AAPL",
            qty=Decimal("100"),
            side=OrderSide.BUY,
            take_profit_price=Decimal("160"),
            stop_loss_price=Decimal("145"),
        )
        assert params["order_class"] == OrderClass.BRACKET
        assert "take_profit" in params
        assert "stop_loss" in params


# =============================================================================
# OrderValidator Tests
# =============================================================================

class TestOrderValidator:
    """Tests for OrderValidator."""

    def test_validate_valid_order(self):
        """Test validation of a valid order."""
        validator = OrderValidator(
            max_order_value=Decimal("100000"),
            max_position_pct=0.20,  # Allow up to 20% position
        )
        request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET,
        )
        portfolio = Portfolio(
            equity=Decimal("100000"),
            cash=Decimal("50000"),
            buying_power=Decimal("50000"),
        )
        result = validator.validate(request, portfolio, Decimal("150"))
        # 100 shares * $150 = $15,000 = 15% of $100,000 portfolio < 20% max
        assert result.is_valid is True

    def test_validate_missing_symbol(self):
        """Test validation fails for missing symbol."""
        validator = OrderValidator()
        request = OrderRequest(
            symbol="",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
        )
        result = validator.validate(request)
        assert result.is_valid is False
        assert "Symbol is required" in result.errors

    def test_validate_limit_order_without_price(self):
        """Test validation fails for limit order without price."""
        validator = OrderValidator()
        request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.LIMIT,
            limit_price=None,
        )
        result = validator.validate(request)
        assert result.is_valid is False
        assert "Limit price required" in result.errors[0]

    def test_validate_exceeds_max_position(self):
        """Test validation fails when exceeding max position."""
        validator = OrderValidator(max_position_pct=0.05)
        request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
        )
        portfolio = Portfolio(
            equity=Decimal("10000"),
            cash=Decimal("10000"),
            buying_power=Decimal("10000"),
        )
        result = validator.validate(request, portfolio, Decimal("100"))  # 100 shares * $100 = $10,000 = 100%
        assert result.is_valid is False


# =============================================================================
# SmartOrderRouter Tests
# =============================================================================

class TestSmartOrderRouter:
    """Tests for SmartOrderRouter."""

    def test_select_market_for_urgent_orders(self):
        """Test market order selection for urgent orders."""
        router = SmartOrderRouter(urgency_threshold=0.8)
        request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
        )
        algo = router.select_algorithm(
            request=request,
            order_value=Decimal("1000"),
            urgency=0.9,
        )
        assert algo == "market"

    def test_select_twap_for_large_orders(self):
        """Test TWAP selection for large orders."""
        router = SmartOrderRouter(
            twap_threshold=Decimal("10000"),
            vwap_threshold=Decimal("50000"),
        )
        request = OrderRequest(symbol="AAPL", side=OrderSide.BUY, quantity=Decimal("100"))
        algo = router.select_algorithm(
            request=request,
            order_value=Decimal("25000"),
            urgency=0.3,
        )
        assert algo == "twap"

    def test_calculate_limit_price_buy(self):
        """Test limit price calculation for buy."""
        router = SmartOrderRouter()
        limit = router.calculate_limit_price(
            side=OrderSide.BUY,
            current_price=Decimal("100"),
        )
        assert limit > Decimal("100")

    def test_calculate_limit_price_sell(self):
        """Test limit price calculation for sell."""
        router = SmartOrderRouter()
        limit = router.calculate_limit_price(
            side=OrderSide.SELL,
            current_price=Decimal("100"),
        )
        assert limit < Decimal("100")


# =============================================================================
# ManagedOrder Tests
# =============================================================================

class TestManagedOrder:
    """Tests for ManagedOrder dataclass."""

    def test_is_active_for_pending_state(self):
        """Test is_active for pending state."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
        )
        managed = ManagedOrder(
            order=order,
            request=OrderRequest(symbol="AAPL", side=OrderSide.BUY, quantity=Decimal("100")),
            state=OrderState.PENDING,
        )
        assert managed.is_active is True

    def test_is_terminal_for_filled_state(self):
        """Test is_terminal for filled state."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
        )
        managed = ManagedOrder(
            order=order,
            request=OrderRequest(symbol="AAPL", side=OrderSide.BUY, quantity=Decimal("100")),
            state=OrderState.FILLED,
        )
        assert managed.is_terminal is True

    def test_fill_progress_calculation(self):
        """Test fill progress calculation."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            filled_qty=Decimal("50"),
        )
        managed = ManagedOrder(
            order=order,
            request=OrderRequest(symbol="AAPL", side=OrderSide.BUY, quantity=Decimal("100")),
        )
        assert managed.fill_progress == 50.0


# =============================================================================
# TWAP Algorithm Tests
# =============================================================================

class TestTWAPAlgorithm:
    """Tests for TWAPAlgorithm."""

    def test_calculate_slices_even_distribution(self):
        """Test TWAP calculates evenly distributed slices."""
        mock_order_manager = MagicMock()
        mock_client = MagicMock()

        algo = TWAPAlgorithm(mock_order_manager, mock_client, randomize_timing=False)
        slices = algo.calculate_slices(
            total_quantity=Decimal("100"),
            duration_minutes=60,
            interval_minutes=10,
        )

        # Should have 6 slices (60 / 10)
        assert len(slices) == 6

        # Total quantity should match
        total = sum(s.quantity for s in slices)
        assert total == Decimal("100")

    def test_calculate_slices_handles_remainder(self):
        """Test TWAP handles remainder in slice calculation."""
        mock_order_manager = MagicMock()
        mock_client = MagicMock()

        algo = TWAPAlgorithm(mock_order_manager, mock_client, randomize_timing=False)
        slices = algo.calculate_slices(
            total_quantity=Decimal("17"),  # Not evenly divisible
            duration_minutes=30,
            interval_minutes=10,
        )

        # Total should still match
        total = sum(s.quantity for s in slices)
        assert total == Decimal("17")


# =============================================================================
# VWAP Algorithm Tests
# =============================================================================

class TestVWAPAlgorithm:
    """Tests for VWAPAlgorithm."""

    def test_calculate_slices_volume_weighted(self):
        """Test VWAP calculates volume-weighted slices."""
        mock_order_manager = MagicMock()
        mock_client = MagicMock()

        algo = VWAPAlgorithm(mock_order_manager, mock_client)
        slices = algo.calculate_slices(
            total_quantity=Decimal("100"),
            duration_minutes=60,
            interval_minutes=10,
        )

        # Should have slices
        assert len(slices) > 0

        # Total quantity should match
        total = sum(s.quantity for s in slices)
        assert total == Decimal("100")


# =============================================================================
# AlgoExecutionState Tests
# =============================================================================

class TestAlgoExecutionState:
    """Tests for AlgoExecutionState dataclass."""

    def test_fill_progress_calculation(self):
        """Test fill progress calculation."""
        state = AlgoExecutionState(
            total_quantity=Decimal("100"),
            filled_quantity=Decimal("50"),
            remaining_quantity=Decimal("50"),
        )
        assert state.fill_progress == 50.0

    def test_vwap_calculation(self):
        """Test VWAP calculation from slices."""
        state = AlgoExecutionState(
            total_quantity=Decimal("100"),
            slices=[
                SliceOrder(quantity=Decimal("50"), executed_qty=Decimal("50"), executed_price=Decimal("100")),
                SliceOrder(quantity=Decimal("50"), executed_qty=Decimal("50"), executed_price=Decimal("102")),
            ],
        )
        vwap = state.vwap
        assert vwap == Decimal("101")  # (50*100 + 50*102) / 100


# =============================================================================
# PositionTracker Tests
# =============================================================================

class TestPositionTracker:
    """Tests for PositionTracker."""

    def test_get_portfolio_empty(self):
        """Test getting empty portfolio."""
        mock_client = MagicMock()
        tracker = PositionTracker(client=mock_client)

        portfolio = tracker.get_portfolio()
        assert portfolio.position_count == 0

    def test_update_from_fill_creates_position(self):
        """Test that fill creates new position."""
        mock_client = MagicMock()
        tracker = PositionTracker(client=mock_client)

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
        )

        position = tracker.update_from_fill(
            order=order,
            fill_qty=Decimal("100"),
            fill_price=Decimal("150"),
        )

        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")
        assert position.avg_entry_price == Decimal("150")

    def test_update_price(self):
        """Test price update for position."""
        mock_client = MagicMock()
        tracker = PositionTracker(client=mock_client)

        # Create initial position
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=Decimal("100"))
        tracker.update_from_fill(order, Decimal("100"), Decimal("150"))

        # Update price
        updated = tracker.update_price("AAPL", Decimal("155"))

        assert updated is not None
        assert updated.current_price == Decimal("155")
        assert updated.unrealized_pnl == Decimal("500")  # 100 * (155 - 150)


class TestTradeRecord:
    """Tests for TradeRecord dataclass."""

    def test_trade_record_creation(self):
        """Test creating a trade record."""
        record = TradeRecord(
            trade_id=uuid4(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150"),
            timestamp=datetime.now(timezone.utc),
        )
        assert record.symbol == "AAPL"
        assert record.settlement_status == SettlementStatus.PENDING


class TestReconciliationResult:
    """Tests for ReconciliationResult dataclass."""

    def test_is_consistent_when_no_issues(self):
        """Test is_consistent with no issues."""
        result = ReconciliationResult(matches=5)
        assert result.is_consistent is True

    def test_is_consistent_with_discrepancies(self):
        """Test is_consistent with discrepancies."""
        result = ReconciliationResult(
            matches=3,
            discrepancies=[{"symbol": "AAPL", "diff": "10"}],
        )
        assert result.is_consistent is False


# =============================================================================
# Integration-style Tests
# =============================================================================

class TestOrderManagerIntegration:
    """Integration tests for OrderManager."""

    @pytest.mark.asyncio
    async def test_create_and_validate_order(self):
        """Test creating and validating an order."""
        mock_client = MagicMock(spec=AlpacaClient)
        event_bus = EventBus()

        # Use a validator with higher position limit
        validator = OrderValidator(max_position_pct=0.20)
        manager = OrderManager(
            client=mock_client,
            event_bus=event_bus,
            validator=validator,
        )

        request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
        )

        portfolio = Portfolio(
            equity=Decimal("100000"),
            cash=Decimal("50000"),
            buying_power=Decimal("50000"),
        )

        managed = manager.create_order(
            request=request,
            portfolio=portfolio,
            current_price=Decimal("150"),
        )

        assert managed.state == OrderState.VALIDATED
        assert managed.order.symbol == "AAPL"

    def test_get_active_orders_empty(self):
        """Test getting active orders when empty."""
        mock_client = MagicMock(spec=AlpacaClient)
        manager = OrderManager(client=mock_client)

        active = manager.get_active_orders()
        assert len(active) == 0

    def test_get_statistics(self):
        """Test getting order statistics."""
        mock_client = MagicMock(spec=AlpacaClient)
        manager = OrderManager(client=mock_client)

        stats = manager.get_statistics()
        assert "total_orders" in stats
        assert "by_state" in stats
