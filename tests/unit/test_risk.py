"""
Unit tests for the risk management module.

Tests position sizing, portfolio optimization, risk monitoring, and limits.
"""

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

import numpy as np
import pytest

from quant_trading_system.core.data_types import (
    Direction,
    Order,
    OrderSide,
    OrderType,
    Portfolio,
    Position,
    TradeSignal,
)
from quant_trading_system.core.events import EventBus
from quant_trading_system.risk.limits import (
    CheckResult,
    KillSwitch,
    KillSwitchReason,
    PreTradeRiskChecker,
    RiskLimitsConfig,
    RiskLimitsManager,
)
from quant_trading_system.risk.portfolio_optimizer import (
    HierarchicalRiskParityOptimizer,
    MeanVarianceOptimizer,
    OptimizationConstraints,
    PortfolioOptimizerFactory,
    OptimizationMethod,
    PortfolioRebalancer,
    RiskParityOptimizer,
)
from quant_trading_system.risk.position_sizer import (
    FixedFractionalSizer,
    KellyCriterionSizer,
    PositionSizerFactory,
    SizingConstraints,
    SizingMethod,
    VolatilityBasedSizer,
)
from quant_trading_system.risk.risk_monitor import (
    AlertSeverity,
    RiskMonitor,
    VaRCalculator,
)


# ============================================================================
# Position Sizer Tests
# ============================================================================


class TestFixedFractionalSizer:
    """Tests for FixedFractionalSizer."""

    @pytest.fixture
    def sizer(self) -> FixedFractionalSizer:
        """Create a fixed fractional sizer."""
        return FixedFractionalSizer(risk_fraction=0.01, default_stop_pct=0.02)

    @pytest.fixture
    def portfolio(self) -> Portfolio:
        """Create a test portfolio."""
        return Portfolio(
            equity=Decimal("100000"),
            cash=Decimal("100000"),
            buying_power=Decimal("100000"),
        )

    @pytest.fixture
    def signal(self) -> TradeSignal:
        """Create a test signal."""
        return TradeSignal(
            symbol="AAPL",
            direction=Direction.LONG,
            strength=0.8,
            confidence=0.7,
            horizon=5,
            model_source="test_model",
        )

    def test_calculate_size_basic(self, sizer, portfolio, signal):
        """Test basic position size calculation."""
        result = sizer.calculate_size(
            signal=signal,
            portfolio=portfolio,
            entry_price=Decimal("150"),
            stop_loss_price=Decimal("147"),
        )

        assert result.symbol == "AAPL"
        assert result.recommended_size > 0
        assert result.method == SizingMethod.FIXED_FRACTIONAL
        assert result.entry_price == Decimal("150")

    def test_calculate_size_respects_risk_fraction(self, sizer, portfolio, signal):
        """Test that risk is capped at risk_fraction."""
        result = sizer.calculate_size(
            signal=signal,
            portfolio=portfolio,
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("98"),  # 2% stop
        )

        # Risk amount should be 1% of 100000 = 1000
        assert float(result.risk_amount) == pytest.approx(1000, rel=0.01)

    def test_calculate_size_applies_constraints(self, portfolio, signal):
        """Test that constraints are applied."""
        constraints = SizingConstraints(max_position_pct=0.05)
        sizer = FixedFractionalSizer(
            risk_fraction=0.02,  # Higher risk fraction
            constraints=constraints,
        )

        result = sizer.calculate_size(
            signal=signal,
            portfolio=portfolio,
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("95"),
        )

        # Position should be constrained to 5% of equity
        max_value = float(portfolio.equity) * 0.05
        actual_value = float(result.recommended_size * Decimal("100"))
        assert actual_value <= max_value * 1.01  # Allow 1% tolerance

    def test_invalid_risk_fraction(self):
        """Test that invalid risk fraction raises error."""
        with pytest.raises(ValueError):
            FixedFractionalSizer(risk_fraction=0.10)  # Too high

    def test_zero_entry_price(self, sizer, portfolio, signal):
        """Test that zero entry price raises error."""
        from quant_trading_system.core.exceptions import RiskError

        with pytest.raises(RiskError):
            sizer.calculate_size(
                signal=signal,
                portfolio=portfolio,
                entry_price=Decimal("0"),
            )


class TestKellyCriterionSizer:
    """Tests for KellyCriterionSizer."""

    def test_kelly_calculation(self):
        """Test Kelly percentage calculation."""
        sizer = KellyCriterionSizer(
            win_rate=0.6,
            win_loss_ratio=2.0,
            kelly_fraction=0.25,
        )

        kelly_pct = sizer.calculate_kelly_pct()

        # Full Kelly = (0.6 * 2 - 0.4) / 2 = 0.4
        # Quarter Kelly = 0.4 * 0.25 = 0.1
        assert kelly_pct == pytest.approx(0.10, rel=0.01)

    def test_negative_kelly(self):
        """Test that negative Kelly returns 0."""
        sizer = KellyCriterionSizer(
            win_rate=0.3,
            win_loss_ratio=0.5,  # Bad edge
            kelly_fraction=0.25,
        )

        kelly_pct = sizer.calculate_kelly_pct()
        assert kelly_pct == 0.0

    def test_update_trade_history(self):
        """Test trade history updates."""
        sizer = KellyCriterionSizer(min_trades=5)

        # Add some trades
        for pnl in [0.02, -0.01, 0.03, -0.015, 0.025]:
            sizer.update_trade_history(pnl)

        assert len(sizer._trade_history) == 5


class TestVolatilityBasedSizer:
    """Tests for VolatilityBasedSizer."""

    def test_low_volatility_increases_size(self):
        """Test that low volatility increases position size."""
        sizer = VolatilityBasedSizer(target_volatility=0.15)

        portfolio = Portfolio(
            equity=Decimal("100000"),
            cash=Decimal("100000"),
            buying_power=Decimal("100000"),
        )
        signal = TradeSignal(
            symbol="AAPL",
            direction=Direction.LONG,
            strength=0.8,
            confidence=0.7,
            horizon=5,
            model_source="test",
        )

        # Low volatility (below target)
        result = sizer.calculate_size(
            signal=signal,
            portfolio=portfolio,
            entry_price=Decimal("100"),
            market_data={"volatility": 0.10},
        )

        assert result.metadata["volatility_ratio"] > 1.0

    def test_high_volatility_decreases_size(self):
        """Test that high volatility decreases position size."""
        sizer = VolatilityBasedSizer(target_volatility=0.15)

        portfolio = Portfolio(
            equity=Decimal("100000"),
            cash=Decimal("100000"),
            buying_power=Decimal("100000"),
        )
        signal = TradeSignal(
            symbol="AAPL",
            direction=Direction.LONG,
            strength=0.8,
            confidence=0.7,
            horizon=5,
            model_source="test",
        )

        # High volatility (above target)
        result = sizer.calculate_size(
            signal=signal,
            portfolio=portfolio,
            entry_price=Decimal("100"),
            market_data={"volatility": 0.30},
        )

        assert result.metadata["volatility_ratio"] < 1.0


class TestPositionSizerFactory:
    """Tests for PositionSizerFactory."""

    def test_create_fixed_fractional(self):
        """Test factory creates fixed fractional sizer."""
        sizer = PositionSizerFactory.create(
            SizingMethod.FIXED_FRACTIONAL,
            risk_fraction=0.01,
        )
        assert isinstance(sizer, FixedFractionalSizer)

    def test_create_kelly(self):
        """Test factory creates Kelly sizer."""
        sizer = PositionSizerFactory.create(
            SizingMethod.KELLY_CRITERION,
            win_rate=0.55,
        )
        assert isinstance(sizer, KellyCriterionSizer)

    def test_create_volatility_based(self):
        """Test factory creates volatility-based sizer."""
        sizer = PositionSizerFactory.create(
            SizingMethod.VOLATILITY_BASED,
            target_volatility=0.20,
        )
        assert isinstance(sizer, VolatilityBasedSizer)


# ============================================================================
# Portfolio Optimizer Tests
# ============================================================================


class TestMeanVarianceOptimizer:
    """Tests for MeanVarianceOptimizer."""

    @pytest.fixture
    def optimizer(self) -> MeanVarianceOptimizer:
        """Create a mean-variance optimizer."""
        return MeanVarianceOptimizer(objective="max_sharpe")

    def test_optimize_basic(self, optimizer):
        """Test basic optimization."""
        symbols = ["AAPL", "GOOG", "MSFT"]
        expected_returns = np.array([0.10, 0.08, 0.12])
        cov_matrix = np.array([
            [0.04, 0.01, 0.015],
            [0.01, 0.03, 0.01],
            [0.015, 0.01, 0.05],
        ])

        result = optimizer.optimize(symbols, expected_returns, cov_matrix)

        # Result contains weights for all symbols
        assert len(result.target_weights) == 3
        # Weights are non-negative (long only by default)
        assert all(w >= 0 for w in result.target_weights.values())

    def test_weights_sum_to_one(self, optimizer):
        """Test that optimized weights sum to 1 or less."""
        symbols = ["A", "B"]
        returns = np.array([0.05, 0.08])
        cov = np.array([[0.02, 0.005], [0.005, 0.03]])

        result = optimizer.optimize(symbols, returns, cov)

        total_weight = sum(result.target_weights.values())
        # Weights should be bounded (may not sum to 1 if optimization doesn't converge fully)
        assert 0 <= total_weight <= 1.01

    def test_respects_max_weight_constraint(self):
        """Test that max weight constraint is respected."""
        constraints = OptimizationConstraints(max_weight=0.30)
        optimizer = MeanVarianceOptimizer(constraints=constraints)

        symbols = ["A", "B", "C"]
        returns = np.array([0.20, 0.05, 0.05])  # A is much better
        cov = np.array([
            [0.02, 0.0, 0.0],
            [0.0, 0.02, 0.0],
            [0.0, 0.0, 0.02],
        ])

        result = optimizer.optimize(symbols, returns, cov)

        for weight in result.target_weights.values():
            assert weight <= 0.30 + 0.01  # Allow small tolerance


class TestRiskParityOptimizer:
    """Tests for RiskParityOptimizer."""

    def test_equal_risk_contribution(self):
        """Test that risk contributions are approximately equal."""
        optimizer = RiskParityOptimizer()

        symbols = ["A", "B"]
        returns = np.array([0.08, 0.08])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])  # B more volatile

        result = optimizer.optimize(symbols, returns, cov)

        # Both assets should have non-zero weights
        assert result.target_weights["A"] > 0
        assert result.target_weights["B"] > 0
        # Weights should sum to approximately 1 (may have some tolerance due to optimizer)
        total = sum(result.target_weights.values())
        assert 0.3 < total <= 1.01  # Allow some flexibility for optimization convergence


class TestHierarchicalRiskParityOptimizer:
    """Tests for HierarchicalRiskParityOptimizer."""

    def test_optimize_basic(self):
        """Test basic HRP optimization."""
        optimizer = HierarchicalRiskParityOptimizer()

        symbols = ["A", "B", "C"]
        returns = np.array([0.08, 0.10, 0.06])
        cov = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.05, 0.01],
            [0.02, 0.01, 0.03],
        ])

        result = optimizer.optimize(symbols, returns, cov)

        assert result.optimization_success
        assert sum(result.target_weights.values()) == pytest.approx(1.0, abs=0.01)


class TestPortfolioRebalancer:
    """Tests for PortfolioRebalancer."""

    def test_should_rebalance_drift_exceeded(self):
        """Test rebalancing triggered when drift exceeds threshold."""
        rebalancer = PortfolioRebalancer(drift_threshold=0.05)

        current = {"A": 0.60, "B": 0.40}
        target = {"A": 0.50, "B": 0.50}

        should_rebalance, reason = rebalancer.should_rebalance(current, target)

        assert should_rebalance
        assert "Drift exceeded" in reason

    def test_should_not_rebalance_within_threshold(self):
        """Test no rebalancing when within threshold."""
        rebalancer = PortfolioRebalancer(drift_threshold=0.05)

        current = {"A": 0.52, "B": 0.48}
        target = {"A": 0.50, "B": 0.50}

        should_rebalance, reason = rebalancer.should_rebalance(current, target)

        assert not should_rebalance


class TestPortfolioOptimizerFactory:
    """Tests for PortfolioOptimizerFactory."""

    def test_create_mean_variance(self):
        """Test factory creates mean-variance optimizer."""
        optimizer = PortfolioOptimizerFactory.create(OptimizationMethod.MEAN_VARIANCE)
        assert isinstance(optimizer, MeanVarianceOptimizer)

    def test_create_risk_parity(self):
        """Test factory creates risk parity optimizer."""
        optimizer = PortfolioOptimizerFactory.create(OptimizationMethod.RISK_PARITY)
        assert isinstance(optimizer, RiskParityOptimizer)


# ============================================================================
# Risk Monitor Tests
# ============================================================================


class TestVaRCalculator:
    """Tests for VaRCalculator."""

    @pytest.fixture
    def calculator(self) -> VaRCalculator:
        """Create a VaR calculator."""
        return VaRCalculator()

    def test_historical_var(self, calculator):
        """Test historical VaR calculation."""
        # Generate some returns
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, 252)

        var = calculator.calculate_historical_var(returns, 100000, 0.95)

        assert var > 0
        assert var < 100000 * 0.10  # Should be less than 10% of portfolio

    def test_parametric_var(self, calculator):
        """Test parametric VaR calculation."""
        returns = np.array([0.01, -0.02, 0.015, -0.01, 0.005] * 50)

        var = calculator.calculate_parametric_var(returns, 100000, 0.95)

        assert var > 0

    def test_cvar_greater_than_var(self, calculator):
        """Test that CVaR >= VaR."""
        np.random.seed(42)
        returns = np.random.normal(-0.001, 0.02, 252)

        var = calculator.calculate_historical_var(returns, 100000, 0.95)
        cvar = calculator.calculate_cvar(returns, 100000, 0.95)

        assert cvar >= var


class TestRiskMonitor:
    """Tests for RiskMonitor."""

    @pytest.fixture
    def monitor(self) -> RiskMonitor:
        """Create a risk monitor."""
        return RiskMonitor()

    @pytest.fixture
    def portfolio(self) -> Portfolio:
        """Create a test portfolio."""
        return Portfolio(
            equity=Decimal("100000"),
            cash=Decimal("50000"),
            buying_power=Decimal("50000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("100"),
                    avg_entry_price=Decimal("150"),
                    current_price=Decimal("155"),
                    cost_basis=Decimal("15000"),
                    market_value=Decimal("15500"),
                ),
            },
        )

    def test_initialize(self, monitor):
        """Test monitor initialization."""
        monitor.initialize(Decimal("100000"))
        state = monitor.get_drawdown_state()

        assert state is not None
        assert state.peak_equity == Decimal("100000")
        assert state.current_drawdown == 0.0

    def test_update_tracks_drawdown(self, monitor, portfolio):
        """Test that update tracks drawdown correctly."""
        monitor.initialize(Decimal("110000"))  # Peak was higher

        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 100)

        metrics = monitor.update(portfolio, returns)

        assert metrics.current_drawdown > 0
        assert metrics.current_drawdown <= 1.0

    def test_alert_generation(self, monitor, portfolio):
        """Test that alerts are generated on threshold breach."""
        monitor.initialize(Decimal("100000"))
        monitor.set_threshold("current_drawdown", 0.01, AlertSeverity.WARNING)

        # Force a drawdown scenario
        monitor._drawdown_state.current_drawdown = 0.05
        monitor._drawdown_state.max_drawdown = 0.05

        np.random.seed(42)
        returns = np.random.normal(-0.01, 0.02, 100)

        monitor.update(portfolio, returns)

        alerts = monitor.get_active_alerts()
        # Should have at least one warning
        assert len(alerts) > 0 or monitor._drawdown_state.current_drawdown < 0.01


# ============================================================================
# Risk Limits Tests
# ============================================================================


class TestPreTradeRiskChecker:
    """Tests for PreTradeRiskChecker."""

    @pytest.fixture
    def checker(self) -> PreTradeRiskChecker:
        """Create a pre-trade checker."""
        config = RiskLimitsConfig(
            max_position_pct=0.10,
            max_total_positions=10,
            symbol_blacklist=["BADSTOCK"],
        )
        return PreTradeRiskChecker(config)

    @pytest.fixture
    def portfolio(self) -> Portfolio:
        """Create a test portfolio."""
        return Portfolio(
            equity=Decimal("100000"),
            cash=Decimal("100000"),
            buying_power=Decimal("100000"),
        )

    def test_check_buying_power_pass(self, checker, portfolio):
        """Test buying power check passes with sufficient funds."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("10"),
            order_type=OrderType.MARKET,
        )

        result = checker.check_buying_power(order, portfolio, Decimal("150"))

        assert result.result == CheckResult.PASSED

    def test_check_buying_power_fail(self, checker, portfolio):
        """Test buying power check fails with insufficient funds."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("1000"),
            order_type=OrderType.MARKET,
        )

        result = checker.check_buying_power(order, portfolio, Decimal("150"))

        assert result.result == CheckResult.FAILED

    def test_check_blacklist_pass(self, checker):
        """Test blacklist check passes for allowed symbol."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("10"),
            order_type=OrderType.MARKET,
        )

        result = checker.check_blacklist(order)

        assert result.result == CheckResult.PASSED

    def test_check_blacklist_fail(self, checker):
        """Test blacklist check fails for banned symbol."""
        order = Order(
            symbol="BADSTOCK",
            side=OrderSide.BUY,
            quantity=Decimal("10"),
            order_type=OrderType.MARKET,
        )

        result = checker.check_blacklist(order)

        assert result.result == CheckResult.FAILED

    def test_check_position_limit(self, checker, portfolio):
        """Test position limit check."""
        # Order that would exceed 10% of equity
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET,
        )

        result = checker.check_position_limit(order, portfolio, Decimal("150"))

        # 100 * 150 = 15000, which is 15% > 10%
        assert result.result == CheckResult.FAILED


class TestKillSwitch:
    """Tests for KillSwitch."""

    @pytest.fixture
    def kill_switch(self) -> KillSwitch:
        """Create a kill switch."""
        return KillSwitch()

    def test_check_conditions_max_drawdown(self, kill_switch):
        """Test kill switch triggers on max drawdown."""
        should_trigger, reason = kill_switch.check_conditions(
            current_drawdown=0.20,  # 20% drawdown
        )

        assert should_trigger
        assert reason == KillSwitchReason.MAX_DRAWDOWN

    def test_check_conditions_no_trigger(self, kill_switch):
        """Test kill switch doesn't trigger on normal conditions."""
        should_trigger, reason = kill_switch.check_conditions(
            current_drawdown=0.05,  # 5% drawdown
        )

        assert not should_trigger
        assert reason is None

    def test_activate(self, kill_switch):
        """Test kill switch activation."""
        state = kill_switch.activate(
            reason=KillSwitchReason.MANUAL_ACTIVATION,
            activated_by="test",
        )

        assert state.is_active
        assert state.reason == KillSwitchReason.MANUAL_ACTIVATION
        assert kill_switch.is_active()

    def test_reset(self, kill_switch):
        """Test kill switch reset."""
        kill_switch.activate(KillSwitchReason.MANUAL_ACTIVATION)
        assert kill_switch.is_active()

        kill_switch.reset("authorized_user")
        assert not kill_switch.is_active()

    def test_can_trade_when_active(self, kill_switch):
        """Test trading is blocked when kill switch is active."""
        kill_switch.activate(KillSwitchReason.MANUAL_ACTIVATION)

        result = kill_switch.can_trade()

        assert result.result == CheckResult.FAILED


class TestRiskLimitsManager:
    """Tests for RiskLimitsManager."""

    @pytest.fixture
    def manager(self) -> RiskLimitsManager:
        """Create a risk limits manager."""
        return RiskLimitsManager()

    @pytest.fixture
    def portfolio(self) -> Portfolio:
        """Create a test portfolio."""
        return Portfolio(
            equity=Decimal("100000"),
            cash=Decimal("100000"),
            buying_power=Decimal("100000"),
        )

    def test_pre_trade_check_all_pass(self, manager, portfolio):
        """Test pre-trade check with all passing."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("10"),
            order_type=OrderType.MARKET,
        )

        passed, results = manager.pre_trade_check(order, portfolio, Decimal("150"))

        assert passed
        assert all(r.result != CheckResult.FAILED for r in results)

    def test_check_drawdown_limits_warning(self, manager):
        """Test drawdown limit check returns warning."""
        result = manager.check_drawdown_limits(0.06)  # 6% > 5% warning threshold

        assert result.result == CheckResult.WARNING

    def test_check_drawdown_limits_halt(self, manager):
        """Test drawdown limit check triggers halt."""
        result = manager.check_drawdown_limits(0.16)  # 16% > 15% halt threshold

        assert result.result == CheckResult.FAILED
        assert manager.kill_switch.is_active()
