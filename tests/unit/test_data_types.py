"""
Unit tests for core/data_types.py
"""

from datetime import datetime
from decimal import Decimal
from uuid import UUID

import pytest

from quant_trading_system.core.data_types import (
    Direction,
    OHLCVBar,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    RiskMetrics,
    TimeInForce,
    TradeSignal,
)


class TestOHLCVBar:
    """Tests for OHLCVBar model."""

    def test_valid_ohlcv_bar(self, sample_ohlcv_data):
        """Test creating a valid OHLCV bar."""
        bar = OHLCVBar(**sample_ohlcv_data)
        assert bar.symbol == "AAPL"
        assert bar.open == Decimal("185.50")
        assert bar.high == Decimal("186.75")
        assert bar.low == Decimal("184.25")
        assert bar.close == Decimal("186.00")
        assert bar.volume == 1500000

    def test_symbol_normalization(self):
        """Test that symbol is normalized to uppercase."""
        bar = OHLCVBar(
            symbol="  aapl  ",
            timestamp=datetime.now(),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100.50"),
            volume=1000,
        )
        assert bar.symbol == "AAPL"

    def test_invalid_ohlc_relationship_low_gt_high(self):
        """Test that low > high raises validation error."""
        with pytest.raises(ValueError, match="Low price"):
            OHLCVBar(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=Decimal("100"),
                high=Decimal("99"),  # Invalid: high < low
                low=Decimal("101"),
                close=Decimal("100"),
                volume=1000,
            )

    def test_invalid_ohlc_relationship_low_gt_open(self):
        """Test that low > open raises validation error."""
        with pytest.raises(ValueError, match="Low price"):
            OHLCVBar(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=Decimal("100"),
                high=Decimal("105"),
                low=Decimal("102"),  # Invalid: low > open
                close=Decimal("103"),
                volume=1000,
            )

    def test_to_dict(self, sample_ohlcv_data):
        """Test conversion to dictionary."""
        bar = OHLCVBar(**sample_ohlcv_data)
        # Test with float conversion (preserve_decimal_precision=False)
        d = bar.to_dict(preserve_decimal_precision=False)
        assert d["symbol"] == "AAPL"
        assert d["open"] == 185.50
        assert d["volume"] == 1500000
        assert "timestamp" in d

    def test_to_dict_preserves_precision(self, sample_ohlcv_data):
        """Test conversion to dictionary preserves decimal precision."""
        bar = OHLCVBar(**sample_ohlcv_data)
        # Default behavior preserves precision as strings
        d = bar.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["open"] == "185.50"
        assert d["volume"] == 1500000
        assert "timestamp" in d


class TestTradeSignal:
    """Tests for TradeSignal model."""

    def test_valid_trade_signal(self):
        """Test creating a valid trade signal."""
        signal = TradeSignal(
            symbol="AAPL",
            direction=Direction.LONG,
            strength=0.75,
            confidence=0.85,
            horizon=5,
            model_source="xgboost_v1",
        )
        assert signal.symbol == "AAPL"
        assert signal.direction == Direction.LONG
        assert signal.strength == 0.75
        assert signal.confidence == 0.85
        assert isinstance(signal.signal_id, UUID)

    def test_signal_strength_bounds(self):
        """Test signal strength validation bounds."""
        # Valid bounds
        signal = TradeSignal(
            symbol="AAPL",
            direction=Direction.SHORT,
            strength=-1.0,
            confidence=0.5,
            horizon=1,
            model_source="test",
        )
        assert signal.strength == -1.0

        # Out of bounds
        with pytest.raises(ValueError):
            TradeSignal(
                symbol="AAPL",
                direction=Direction.LONG,
                strength=1.5,  # Invalid: > 1.0
                confidence=0.5,
                horizon=1,
                model_source="test",
            )

    def test_is_actionable(self):
        """Test signal actionability check."""
        # Actionable signal
        signal = TradeSignal(
            symbol="AAPL",
            direction=Direction.LONG,
            strength=0.6,
            confidence=0.7,
            horizon=5,
            model_source="test",
        )
        assert signal.is_actionable(min_confidence=0.5, min_strength=0.3)

        # Not actionable due to low confidence
        assert not signal.is_actionable(min_confidence=0.8)

        # FLAT direction is never actionable
        flat_signal = TradeSignal(
            symbol="AAPL",
            direction=Direction.FLAT,
            strength=0.6,
            confidence=0.9,
            horizon=5,
            model_source="test",
        )
        assert not flat_signal.is_actionable()


class TestOrder:
    """Tests for Order model."""

    def test_valid_market_order(self, sample_order_data):
        """Test creating a valid market order."""
        order = Order(
            symbol=sample_order_data["symbol"],
            side=OrderSide.BUY,
            quantity=sample_order_data["quantity"],
        )
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING

    def test_limit_order_requires_price(self):
        """Test that limit orders require limit price."""
        with pytest.raises(ValueError, match="Limit orders require"):
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=Decimal("100"),
                order_type=OrderType.LIMIT,
                # Missing limit_price
            )

    def test_stop_order_requires_price(self):
        """Test that stop orders require stop price."""
        with pytest.raises(ValueError, match="Stop orders require"):
            Order(
                symbol="AAPL",
                side=OrderSide.SELL,
                quantity=Decimal("100"),
                order_type=OrderType.STOP,
                # Missing stop_price
            )

    def test_stop_limit_order_requires_both_prices(self):
        """Test that stop-limit orders require both prices."""
        with pytest.raises(ValueError, match="Stop-limit orders require both"):
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=Decimal("100"),
                order_type=OrderType.STOP_LIMIT,
                limit_price=Decimal("185.00"),
                # Missing stop_price
            )

    def test_order_status_properties(self):
        """Test order status helper properties."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            status=OrderStatus.SUBMITTED,
        )
        assert order.is_active
        assert not order.is_terminal

        filled_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            status=OrderStatus.FILLED,
        )
        assert not filled_order.is_active
        assert filled_order.is_terminal

    def test_remaining_quantity(self):
        """Test remaining quantity calculation."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            filled_qty=Decimal("60"),
        )
        assert order.remaining_qty == Decimal("40")


class TestPosition:
    """Tests for Position model."""

    def test_valid_position(self, sample_position_data):
        """Test creating a valid position."""
        position = Position(**sample_position_data)
        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")
        assert position.is_long
        assert not position.is_short
        assert not position.is_flat

    def test_short_position(self):
        """Test short position detection."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("-50"),
            avg_entry_price=Decimal("185.00"),
            current_price=Decimal("183.00"),
            cost_basis=Decimal("-9250.00"),
            market_value=Decimal("-9150.00"),
        )
        assert position.is_short
        assert not position.is_long

    def test_unrealized_pnl_percent(self, sample_position_data):
        """Test unrealized P&L percentage calculation."""
        position = Position(**sample_position_data)
        pnl_pct = position.unrealized_pnl_pct
        expected = (50.0 / 18550.0) * 100  # ~0.27%
        assert abs(pnl_pct - expected) < 0.01

    def test_update_price(self, sample_position_data):
        """Test position price update."""
        position = Position(**sample_position_data)
        new_price = Decimal("190.00")
        updated = position.update_price(new_price)

        assert updated.current_price == new_price
        assert updated.market_value == Decimal("100") * new_price
        # Original position unchanged
        assert position.current_price == Decimal("186.00")


class TestPortfolio:
    """Tests for Portfolio model."""

    def test_empty_portfolio(self):
        """Test empty portfolio creation."""
        portfolio = Portfolio(
            equity=Decimal("100000"),
            cash=Decimal("100000"),
            buying_power=Decimal("200000"),
        )
        assert portfolio.position_count == 0
        assert portfolio.total_market_value == Decimal("0")

    def test_portfolio_with_positions(self, sample_position_data):
        """Test portfolio with positions."""
        position = Position(**sample_position_data)
        portfolio = Portfolio(
            equity=Decimal("100000"),
            cash=Decimal("81400"),  # 100000 - 18600
            buying_power=Decimal("162800"),
            positions={"AAPL": position},
        )
        assert portfolio.position_count == 1
        assert portfolio.total_market_value == Decimal("18600.00")
        assert portfolio.get_position("AAPL") is not None
        assert portfolio.get_position("MSFT") is None

    def test_exposure_calculations(self, sample_position_data):
        """Test portfolio exposure calculations."""
        long_position = Position(**sample_position_data)
        short_position = Position(
            symbol="MSFT",
            quantity=Decimal("-50"),
            avg_entry_price=Decimal("370.00"),
            current_price=Decimal("375.00"),
            cost_basis=Decimal("-18500.00"),
            market_value=Decimal("-18750.00"),
        )
        portfolio = Portfolio(
            equity=Decimal("100000"),
            cash=Decimal("62650"),
            buying_power=Decimal("125300"),
            positions={"AAPL": long_position, "MSFT": short_position},
        )
        assert portfolio.long_exposure == Decimal("18600.00")
        assert portfolio.short_exposure == Decimal("18750.00")
        assert portfolio.net_exposure == Decimal("18600.00") - Decimal("18750.00")


class TestRiskMetrics:
    """Tests for RiskMetrics model."""

    def test_valid_risk_metrics(self):
        """Test creating valid risk metrics."""
        metrics = RiskMetrics(
            portfolio_var_95=Decimal("5000"),
            portfolio_var_99=Decimal("7500"),
            sharpe_ratio=1.8,
            max_drawdown=0.12,
            current_drawdown=0.05,
        )
        assert metrics.sharpe_ratio == 1.8
        assert metrics.max_drawdown == 0.12

    def test_is_within_limits(self):
        """Test risk limit checking."""
        metrics = RiskMetrics(
            portfolio_var_95=Decimal("5000"),
            current_drawdown=0.08,
            volatility_annual=0.15,
        )

        # All within limits
        assert metrics.is_within_limits(
            max_var_95=Decimal("10000"),
            max_drawdown=0.15,
            max_volatility=0.20,
        )

        # Drawdown exceeds limit
        assert not metrics.is_within_limits(max_drawdown=0.05)

        # VaR exceeds limit
        assert not metrics.is_within_limits(max_var_95=Decimal("4000"))


class TestEnums:
    """Tests for enum types."""

    def test_direction_enum(self):
        """Test Direction enum values."""
        assert Direction.LONG.value == "LONG"
        assert Direction.SHORT.value == "SHORT"
        assert Direction.FLAT.value == "FLAT"

    def test_order_side_enum(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"

    def test_order_type_enum(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.STOP.value == "STOP"

    def test_order_status_enum(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "PENDING"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.CANCELLED.value == "CANCELLED"

    def test_time_in_force_enum(self):
        """Test TimeInForce enum values."""
        assert TimeInForce.DAY.value == "DAY"
        assert TimeInForce.GTC.value == "GTC"
        assert TimeInForce.IOC.value == "IOC"
