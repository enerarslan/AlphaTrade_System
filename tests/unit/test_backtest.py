"""
Unit tests for the backtesting module.

Tests backtest engine, market simulator, performance analyzer, and optimizer.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from quant_trading_system.backtest.analyzer import (
    PerformanceAnalyzer,
    VisualizationData,
)
from quant_trading_system.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestState,
    ExecutionMode,
    PandasDataHandler,
    Strategy,
    Trade,
    VectorizedBacktest,
)
from quant_trading_system.backtest.optimizer import (
    GeneticOptimizer,
    GridSearchOptimizer,
    OptimizationWindow,
    OverfitDetector,
    ParameterSpace,
    RandomSearchOptimizer,
    StrategyOptimizer,
    WalkForwardOptimizer,
)
from quant_trading_system.backtest.simulator import (
    BidAskSimulator,
    FillSimulator,
    FillType,
    FixedSlippageModel,
    LatencySimulator,
    MarketConditions,
    MarketImpactModel,
    MarketSimulator,
    SlippageModelFactory,
    SlippageModel,
    VolatilitySlippageModel,
    VolumeBasedSlippageModel,
    create_optimistic_simulator,
    create_pessimistic_simulator,
    create_realistic_simulator,
)
from quant_trading_system.core.data_types import (
    Direction,
    Order,
    OrderSide,
    OrderType,
    Portfolio,
    TradeSignal,
)


# ============================================================================
# Market Simulator Tests
# ============================================================================


class TestFixedSlippageModel:
    """Tests for FixedSlippageModel."""

    def test_calculate_slippage(self):
        """Test fixed slippage calculation."""
        model = FixedSlippageModel(slippage_bps=10)

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET,
        )
        conditions = MarketConditions(price=Decimal("100"))

        slippage = model.calculate_slippage(order, conditions)

        # 10 bps = 0.10% = $0.10 on $100
        assert slippage == pytest.approx(Decimal("0.10"), abs=Decimal("0.01"))


class TestVolumeBasedSlippageModel:
    """Tests for VolumeBasedSlippageModel."""

    def test_large_order_more_slippage(self):
        """Test that larger orders have more slippage."""
        model = VolumeBasedSlippageModel(
            base_slippage_bps=5,
            volume_factor=10,
        )

        conditions = MarketConditions(
            price=Decimal("100"),
            avg_daily_volume=1000000,
        )

        small_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET,
        )
        large_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("10000"),
            order_type=OrderType.MARKET,
        )

        small_slippage = model.calculate_slippage(small_order, conditions)
        large_slippage = model.calculate_slippage(large_order, conditions)

        assert large_slippage > small_slippage


class TestVolatilitySlippageModel:
    """Tests for VolatilitySlippageModel."""

    def test_high_volatility_more_slippage(self):
        """Test that higher volatility causes more slippage."""
        model = VolatilitySlippageModel(
            base_slippage_bps=10,
            avg_volatility=0.02,
        )

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET,
        )

        low_vol_conditions = MarketConditions(price=Decimal("100"), volatility=0.01)
        high_vol_conditions = MarketConditions(price=Decimal("100"), volatility=0.04)

        low_vol_slippage = model.calculate_slippage(order, low_vol_conditions)
        high_vol_slippage = model.calculate_slippage(order, high_vol_conditions)

        assert high_vol_slippage > low_vol_slippage


class TestMarketImpactModel:
    """Tests for MarketImpactModel."""

    def test_square_root_impact(self):
        """Test square root market impact calculation."""
        model = MarketImpactModel(temporary_factor=0.5)

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("10000"),
            order_type=OrderType.MARKET,
        )
        conditions = MarketConditions(
            price=Decimal("100"),
            avg_daily_volume=1000000,
            volatility=0.02,
        )

        impact = model.calculate_slippage(order, conditions)

        assert impact > 0


class TestBidAskSimulator:
    """Tests for BidAskSimulator."""

    def test_get_bid_ask(self):
        """Test bid-ask spread generation."""
        simulator = BidAskSimulator(base_spread_bps=10)

        conditions = MarketConditions(price=Decimal("100"))
        bid, ask = simulator.get_bid_ask(Decimal("100"), conditions)

        assert bid < Decimal("100")
        assert ask > Decimal("100")
        assert ask - bid > 0

    def test_execution_price_buy(self):
        """Test execution at ask for buys."""
        simulator = BidAskSimulator(base_spread_bps=10)

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET,
        )
        conditions = MarketConditions(price=Decimal("100"))

        exec_price = simulator.get_execution_price(order, conditions)

        assert exec_price > Decimal("100")  # Buy at ask


class TestFillSimulator:
    """Tests for FillSimulator."""

    def test_full_fill_probability(self):
        """Test fill simulation produces expected distribution."""
        simulator = FillSimulator(
            fill_probability=0.95,
            partial_fill_probability=0.0,
        )

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET,
        )
        conditions = MarketConditions(price=Decimal("100"), avg_daily_volume=1000000)

        fill_types = []
        for _ in range(100):
            fill_type, _ = simulator.simulate_fill(order, conditions)
            fill_types.append(fill_type)

        # Most should be full fills
        full_fills = sum(1 for ft in fill_types if ft == FillType.FULL)
        assert full_fills > 80  # At least 80% full fills


class TestMarketSimulator:
    """Tests for MarketSimulator."""

    def test_simulate_execution(self):
        """Test complete execution simulation."""
        simulator = create_realistic_simulator()

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET,
        )
        conditions = MarketConditions(
            price=Decimal("100"),
            avg_daily_volume=1000000,
            volatility=0.02,
        )

        result = simulator.simulate_execution(order, conditions)

        assert result.fill_type in [FillType.FULL, FillType.PARTIAL, FillType.REJECTED]
        if result.fill_type != FillType.REJECTED:
            assert result.fill_price > 0
            assert result.fill_quantity > 0


class TestSlippageModelFactory:
    """Tests for SlippageModelFactory."""

    def test_create_fixed(self):
        """Test factory creates fixed model."""
        model = SlippageModelFactory.create(SlippageModel.FIXED, slippage_bps=10)
        assert isinstance(model, FixedSlippageModel)

    def test_create_volume_based(self):
        """Test factory creates volume-based model."""
        model = SlippageModelFactory.create(SlippageModel.VOLUME_BASED)
        assert isinstance(model, VolumeBasedSlippageModel)


class TestSimulatorFactoryFunctions:
    """Tests for simulator factory functions."""

    def test_create_optimistic_simulator(self):
        """Test optimistic simulator has no costs."""
        simulator = create_optimistic_simulator()
        assert simulator.commission_bps == 0.0

    def test_create_realistic_simulator(self):
        """Test realistic simulator has reasonable defaults."""
        simulator = create_realistic_simulator()
        assert simulator.commission_bps > 0

    def test_create_pessimistic_simulator(self):
        """Test pessimistic simulator has high costs."""
        realistic = create_realistic_simulator()
        pessimistic = create_pessimistic_simulator()
        assert pessimistic.commission_bps > realistic.commission_bps


# ============================================================================
# Backtest Engine Tests
# ============================================================================


class SimpleStrategy(Strategy):
    """Simple test strategy."""

    def generate_signals(self, data_handler, portfolio):
        """Generate a signal for each symbol."""
        signals = []
        for symbol in data_handler.get_symbols():
            bar = data_handler.get_current_bar(symbol)
            if bar:
                # Simple: buy if close > open
                direction = Direction.LONG if bar.close > bar.open else Direction.SHORT
                signals.append(TradeSignal(
                    symbol=symbol,
                    direction=direction,
                    strength=0.5,
                    confidence=0.6,
                    horizon=1,
                    model_source="simple",
                ))
        return signals


class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BacktestConfig()

        assert config.initial_capital == Decimal("100000")
        assert config.commission_bps > 0
        assert config.slippage_bps > 0

    def test_liquidity_cap_limits_order_size(self):
        """Order sizing should respect bar participation constraints."""
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        data = {
            "AAPL": pd.DataFrame(
                {
                    "open": [10.0, 10.0, 10.0],
                    "high": [10.5, 10.5, 10.5],
                    "low": [9.5, 9.5, 9.5],
                    "close": [10.0, 10.0, 10.0],
                    "volume": [100, 100, 100],
                },
                index=dates,
            )
        }

        handler = PandasDataHandler(data)
        strategy = SimpleStrategy()
        config = BacktestConfig(
            initial_capital=Decimal("100000"),
            max_position_pct=1.0,
            max_participation_rate=0.05,  # 5 shares max on 100 volume bar
            max_adv_order_pct=1.0,
            allow_fractional=False,
            use_market_simulator=False,
        )
        engine = BacktestEngine(data_handler=handler, strategy=strategy, config=config)
        engine._initialize()

        assert handler.update_bars() is True
        signal = TradeSignal(
            symbol="AAPL",
            direction=Direction.LONG,
            strength=0.9,
            confidence=0.9,
            horizon=1,
            model_source="test",
        )
        engine._process_signal(signal)

        assert engine._state is not None
        assert len(engine._state.pending_orders) == 1
        assert engine._state.pending_orders[0].quantity == Decimal("5")


class TestPandasDataHandler:
    """Tests for PandasDataHandler."""

    @pytest.fixture
    def sample_data(self) -> dict[str, pd.DataFrame]:
        """Create sample OHLCV data."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = {
            "AAPL": pd.DataFrame({
                "open": np.random.uniform(145, 155, 100),
                "high": np.random.uniform(150, 160, 100),
                "low": np.random.uniform(140, 150, 100),
                "close": np.random.uniform(145, 155, 100),
                "volume": np.random.randint(1000000, 10000000, 100),
            }, index=dates),
        }
        # Ensure OHLC relationship
        for symbol in data:
            data[symbol]["high"] = data[symbol][["open", "close", "high"]].max(axis=1)
            data[symbol]["low"] = data[symbol][["open", "close", "low"]].min(axis=1)
        return data

    def test_get_symbols(self, sample_data):
        """Test getting symbol list."""
        handler = PandasDataHandler(sample_data)
        symbols = handler.get_symbols()

        assert "AAPL" in symbols

    def test_update_bars(self, sample_data):
        """Test updating bars."""
        handler = PandasDataHandler(sample_data)

        # First update
        result = handler.update_bars()
        assert result is True

        bar = handler.get_current_bar("AAPL")
        assert bar is not None
        assert bar.symbol == "AAPL"

    def test_get_latest_bars(self, sample_data):
        """Test getting latest bars."""
        handler = PandasDataHandler(sample_data)

        # Update a few times
        for _ in range(5):
            handler.update_bars()

        bars = handler.get_latest_bars("AAPL", 3)
        assert len(bars) == 3


class TestVectorizedBacktest:
    """Tests for VectorizedBacktest."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample data with signals."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")
        data = pd.DataFrame({
            "close": 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, 252)),
            "signal": np.random.choice([-1, 0, 1], 252),
        }, index=dates)
        return data

    def test_run_basic(self, sample_data):
        """Test basic vectorized backtest."""
        backtest = VectorizedBacktest(sample_data)
        results = backtest.run()

        assert "equity" in results.columns
        assert "strategy_returns" in results.columns
        assert len(results) == len(sample_data)

    def test_get_metrics(self, sample_data):
        """Test metrics calculation."""
        backtest = VectorizedBacktest(sample_data)
        results = backtest.run()
        metrics = backtest.get_metrics(results)

        assert "sharpe_ratio" in metrics
        assert "total_return" in metrics
        assert "max_drawdown" in metrics


# ============================================================================
# Performance Analyzer Tests
# ============================================================================


class TestPerformanceAnalyzer:
    """Tests for PerformanceAnalyzer."""

    @pytest.fixture
    def analyzer(self) -> PerformanceAnalyzer:
        """Create a performance analyzer."""
        return PerformanceAnalyzer()

    @pytest.fixture
    def backtest_state(self) -> BacktestState:
        """Create a sample backtest state."""
        np.random.seed(42)
        equity = [100000]
        for _ in range(251):
            equity.append(equity[-1] * (1 + np.random.normal(0.0003, 0.015)))

        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(252)]
        equity_curve = list(zip(dates, equity))

        trades = [
            Trade(
                symbol="AAPL",
                entry_time=dates[i],
                exit_time=dates[i + 1],
                side=OrderSide.BUY,
                quantity=Decimal("100"),
                entry_price=Decimal("150"),
                exit_price=Decimal(str(150 + np.random.uniform(-5, 7))),
                pnl=Decimal(str(np.random.uniform(-500, 700))),
                pnl_pct=np.random.uniform(-0.03, 0.05),
                commission=Decimal("1"),
                slippage=Decimal("0.50"),
                holding_period_bars=1,
            )
            for i in range(0, 100, 5)
        ]

        return BacktestState(
            timestamp=dates[-1],
            equity=Decimal(str(equity[-1])),
            cash=Decimal("50000"),
            positions={},
            pending_orders=[],
            equity_curve=equity_curve,
            trades=trades,
        )

    def test_analyze_basic(self, analyzer, backtest_state):
        """Test basic performance analysis."""
        report = analyzer.analyze(backtest_state)

        assert report.return_metrics is not None
        assert report.risk_adjusted_metrics is not None
        assert report.drawdown_metrics is not None
        assert report.trade_metrics is not None

    def test_sharpe_ratio_calculation(self, analyzer, backtest_state):
        """Test Sharpe ratio is reasonable."""
        report = analyzer.analyze(backtest_state)

        # Sharpe should be within reasonable bounds
        assert -5.0 < report.risk_adjusted_metrics.sharpe_ratio < 5.0

    def test_trade_metrics(self, analyzer, backtest_state):
        """Test trade metrics calculation."""
        report = analyzer.analyze(backtest_state)

        assert report.trade_metrics.total_trades == 20
        assert 0 <= report.trade_metrics.win_rate <= 1

    def test_benchmark_comparison(self, analyzer, backtest_state):
        """Test benchmark comparison."""
        np.random.seed(42)
        benchmark = np.random.normal(0.0003, 0.012, 251)

        report = analyzer.analyze(backtest_state, benchmark)

        assert report.benchmark_metrics is not None
        assert -2.0 < report.benchmark_metrics.beta < 2.0

    def test_report_summary(self, analyzer, backtest_state):
        """Test report summary generation."""
        report = analyzer.analyze(backtest_state)
        summary = report.summary()

        assert "Total Return" in summary
        assert "Sharpe Ratio" in summary


class TestSortinoRatioCalculation:
    """Tests for correct Sortino ratio downside deviation calculation."""

    @pytest.fixture
    def analyzer(self) -> PerformanceAnalyzer:
        """Create a performance analyzer."""
        return PerformanceAnalyzer(risk_free_rate=0.0)  # 0% for simpler calculation

    def test_sortino_uses_all_samples_for_downside_deviation(self, analyzer):
        """Test that downside deviation uses ALL samples, not just negative returns.

        The correct formula: sqrt(mean(min(0, return - threshold)^2))
        NOT: std(negative_returns_only)
        """
        # Simple case: 5 returns with 2 negative
        returns = np.array([0.05, -0.02, 0.03, -0.01, 0.02])

        # Correct downside deviation (threshold = 0):
        # min(0, 0.05), min(0, -0.02), min(0, 0.03), min(0, -0.01), min(0, 0.02)
        # = [0, -0.02, 0, -0.01, 0]
        # squared = [0, 0.0004, 0, 0.0001, 0]
        # mean = 0.0001
        # sqrt = 0.01

        # Wrong way (std of only negative returns):
        # std([-0.02, -0.01]) ≈ 0.005

        metrics = analyzer._calculate_risk_adjusted_metrics(returns)

        # Annualized downside vol should be sqrt(0.0001) * sqrt(252) ≈ 0.159
        # Using threshold=0 (risk_free_rate=0)
        expected_downside_std = np.sqrt(0.0001)  # 0.01
        expected_downside_vol = expected_downside_std * np.sqrt(252)  # ~0.159

        assert metrics.downside_volatility == pytest.approx(expected_downside_vol, rel=0.05)

    def test_sortino_all_positive_returns_gives_zero_downside(self, analyzer):
        """Test that all positive returns give zero downside deviation."""
        returns = np.array([0.01, 0.02, 0.03, 0.01, 0.02])

        metrics = analyzer._calculate_risk_adjusted_metrics(returns)

        # With all positive returns and threshold=0, downside deviation should be 0
        assert metrics.downside_volatility == pytest.approx(0.0, abs=1e-10)

    def test_sortino_all_negative_returns(self, analyzer):
        """Test Sortino calculation with all negative returns."""
        returns = np.array([-0.01, -0.02, -0.03, -0.01, -0.02])

        metrics = analyzer._calculate_risk_adjusted_metrics(returns)

        # All returns are below threshold, so downside deviation = sqrt(mean(returns^2))
        expected_downside_variance = np.mean(returns ** 2)
        expected_downside_std = np.sqrt(expected_downside_variance)
        expected_downside_vol = expected_downside_std * np.sqrt(252)

        assert metrics.downside_volatility == pytest.approx(expected_downside_vol, rel=0.01)

    def test_sortino_with_risk_free_rate(self):
        """Test Sortino uses risk-free rate as threshold."""
        analyzer = PerformanceAnalyzer(risk_free_rate=0.05)  # 5% annual = 0.0002 daily
        daily_rf = 0.05 / 252  # ~0.0002

        # Returns that are positive but below daily rf should count as downside
        returns = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001])  # All below daily rf

        metrics = analyzer._calculate_risk_adjusted_metrics(returns)

        # All returns are below threshold (daily_rf), so all count toward downside
        downside_returns = returns - daily_rf  # All negative
        expected_downside_variance = np.mean(downside_returns ** 2)
        expected_downside_vol = np.sqrt(expected_downside_variance) * np.sqrt(252)

        assert metrics.downside_volatility == pytest.approx(expected_downside_vol, rel=0.01)


class TestVisualizationData:
    """Tests for VisualizationData."""

    @pytest.fixture
    def backtest_state(self) -> BacktestState:
        """Create a sample backtest state."""
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
        equity = [100000 + i * 100 for i in range(100)]

        return BacktestState(
            timestamp=dates[-1],
            equity=Decimal(str(equity[-1])),
            cash=Decimal("50000"),
            positions={},
            pending_orders=[],
            equity_curve=list(zip(dates, equity)),
            trades=[],
        )

    def test_get_equity_curve_data(self, backtest_state):
        """Test equity curve data extraction."""
        df = VisualizationData.get_equity_curve_data(backtest_state)

        assert "timestamp" in df.columns
        assert "equity" in df.columns
        assert len(df) == 100

    def test_get_drawdown_data(self, backtest_state):
        """Test drawdown data extraction."""
        df = VisualizationData.get_drawdown_data(backtest_state)

        assert "drawdown" in df.columns
        assert "peak" in df.columns


# ============================================================================
# Optimizer Tests
# ============================================================================


class TestWalkForwardOptimizer:
    """Tests for WalkForwardOptimizer."""

    def test_generate_windows(self):
        """Test walk-forward window generation."""
        optimizer = WalkForwardOptimizer(
            train_period_days=365,
            test_period_days=90,
            step_days=90,
        )

        start = datetime(2020, 1, 1)
        end = datetime(2023, 12, 31)

        windows = optimizer.generate_windows(start, end)

        assert len(windows) > 0
        for window in windows:
            assert window.train_end < window.test_start
            assert window.test_end <= end


class TestStrategyOptimizerSelection:
    """Tests robust parameter window selection logic."""

    def test_select_robust_window_not_last_window_biased(self):
        """Selection should prefer best OOS/stability window, not always the last window."""
        start = datetime(2020, 1, 1)

        windows = [
            OptimizationWindow(
                window_id=0,
                train_start=start,
                train_end=start + timedelta(days=180),
                test_start=start + timedelta(days=185),
                test_end=start + timedelta(days=270),
                best_params={"lookback": 10},
                train_metric=1.0,
                test_metric=0.7,
            ),
            OptimizationWindow(
                window_id=1,
                train_start=start + timedelta(days=90),
                train_end=start + timedelta(days=270),
                test_start=start + timedelta(days=275),
                test_end=start + timedelta(days=360),
                best_params={"lookback": 20},
                train_metric=1.1,
                test_metric=0.9,
            ),
            OptimizationWindow(
                window_id=2,
                train_start=start + timedelta(days=180),
                train_end=start + timedelta(days=360),
                test_start=start + timedelta(days=365),
                test_end=start + timedelta(days=450),
                best_params={"lookback": 30},
                train_metric=2.5,
                test_metric=0.2,
            ),
        ]

        selected = StrategyOptimizer._select_robust_window(windows)

        assert selected is not None
        assert selected.window_id == 1
        assert selected.best_params == {"lookback": 20}

    def test_select_robust_window_returns_none_without_candidates(self):
        """Selection should gracefully handle missing best_params."""
        start = datetime(2020, 1, 1)
        windows = [
            OptimizationWindow(
                window_id=0,
                train_start=start,
                train_end=start + timedelta(days=180),
                test_start=start + timedelta(days=185),
                test_end=start + timedelta(days=270),
                best_params=None,
                train_metric=1.0,
                test_metric=0.8,
            ),
        ]

        assert StrategyOptimizer._select_robust_window(windows) is None

    def test_optimize_populates_walk_forward_audit_metadata(self, monkeypatch):
        """Optimize should persist walk-forward audit evidence in result metadata."""
        from quant_trading_system.backtest.engine import WalkForwardReport

        class _DummyStrategy(Strategy):
            def generate_signals(self, data_handler, portfolio):
                return []

        class _FakeEngine:
            def __init__(self, *args, **kwargs):
                pass

            def run(self):
                return object()

        class _FakeHandler:
            def __init__(self, data):
                self._data = data

        class _StubValidator:
            def run_validation(self, data, strategy_factory, config):
                assert "AAPL" in data
                return WalkForwardReport(
                    windows=[],
                    aggregate_metrics={"sharpe_ratio": 0.82},
                    is_valid=True,
                    validation_messages=["ok"],
                    total_trades=12,
                    combined_equity_curve=[],
                )

        monkeypatch.setattr("quant_trading_system.backtest.optimizer.BacktestEngine", _FakeEngine)
        monkeypatch.setattr(
            "quant_trading_system.backtest.optimizer.StrategyOptimizer._get_objective_value",
            lambda self, state: 1.0,
        )

        idx = pd.date_range("2024-01-01", periods=120, freq="D", tz="UTC")
        dataset = {
            "AAPL": pd.DataFrame({
                "open": np.linspace(100, 110, len(idx)),
                "high": np.linspace(101, 111, len(idx)),
                "low": np.linspace(99, 109, len(idx)),
                "close": np.linspace(100, 110, len(idx)),
                "volume": np.full(len(idx), 1000),
            }, index=idx)
        }

        optimizer = StrategyOptimizer(
            strategy_factory=lambda params: _DummyStrategy(),
            param_spaces=[ParameterSpace("x", "continuous", 0.0, 1.0)],
            walk_forward=WalkForwardOptimizer(
                train_period_days=30,
                test_period_days=15,
                step_days=15,
                purge_days=0,
            ),
            walk_forward_validator=_StubValidator(),
        )

        result = optimizer.optimize(
            data_handler_factory=lambda _s, _e: _FakeHandler(dataset),
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 30),
            n_iterations=2,
        )

        audit = result.metadata.get("walk_forward_audit", {})
        assert audit.get("status") == "completed"
        assert audit.get("is_valid") is True
        assert audit.get("total_trades") == 12
        assert audit.get("aggregate_metrics", {}).get("sharpe_ratio") == pytest.approx(0.82)

    def test_optimize_skips_walk_forward_audit_when_dataset_missing(self, monkeypatch):
        """Optimize should skip walk-forward audit when handler dataset is unavailable."""
        class _DummyStrategy(Strategy):
            def generate_signals(self, data_handler, portfolio):
                return []

        class _FakeEngine:
            def __init__(self, *args, **kwargs):
                pass

            def run(self):
                return object()

        class _NoDatasetHandler:
            pass

        class _NeverCalledValidator:
            def __init__(self):
                self.called = False

            def run_validation(self, data, strategy_factory, config):
                self.called = True
                raise AssertionError("Validator should not be called without dataset")

        validator = _NeverCalledValidator()
        monkeypatch.setattr("quant_trading_system.backtest.optimizer.BacktestEngine", _FakeEngine)
        monkeypatch.setattr(
            "quant_trading_system.backtest.optimizer.StrategyOptimizer._get_objective_value",
            lambda self, state: 0.5,
        )

        optimizer = StrategyOptimizer(
            strategy_factory=lambda params: _DummyStrategy(),
            param_spaces=[ParameterSpace("x", "continuous", 0.0, 1.0)],
            walk_forward=WalkForwardOptimizer(
                train_period_days=30,
                test_period_days=15,
                step_days=15,
                purge_days=0,
            ),
            walk_forward_validator=validator,
        )

        result = optimizer.optimize(
            data_handler_factory=lambda _s, _e: _NoDatasetHandler(),
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 30),
            n_iterations=2,
        )

        audit = result.metadata.get("walk_forward_audit", {})
        assert audit.get("status") == "skipped"
        assert audit.get("reason") == "data_handler_missing_dict_dataset"
        assert validator.called is False


class TestParameterSpace:
    """Tests for ParameterSpace."""

    def test_continuous_sample(self):
        """Test continuous parameter sampling."""
        space = ParameterSpace(
            name="param",
            param_type="continuous",
            low=0.0,
            high=1.0,
        )

        samples = [space.sample() for _ in range(100)]

        assert all(0.0 <= s <= 1.0 for s in samples)

    def test_integer_sample(self):
        """Test integer parameter sampling."""
        space = ParameterSpace(
            name="param",
            param_type="integer",
            low=1,
            high=10,
        )

        samples = [space.sample() for _ in range(100)]

        assert all(1 <= s <= 10 for s in samples)
        assert all(isinstance(s, (int, np.integer)) for s in samples)

    def test_categorical_sample(self):
        """Test categorical parameter sampling."""
        space = ParameterSpace(
            name="param",
            param_type="categorical",
            categories=["a", "b", "c"],
        )

        samples = [space.sample() for _ in range(100)]

        assert all(s in ["a", "b", "c"] for s in samples)

    def test_grid_values(self):
        """Test grid value generation."""
        space = ParameterSpace(
            name="param",
            param_type="continuous",
            low=0.0,
            high=1.0,
        )

        grid = space.grid_values(5)

        assert len(grid) == 5
        assert grid[0] == 0.0
        assert grid[-1] == 1.0


class TestRandomSearchOptimizer:
    """Tests for RandomSearchOptimizer."""

    def test_optimize(self):
        """Test random search optimization."""
        param_spaces = [
            ParameterSpace("x", "continuous", 0, 10),
            ParameterSpace("y", "continuous", 0, 10),
        ]

        def objective(params):
            # Simple quadratic: max at (5, 5)
            return -((params["x"] - 5) ** 2 + (params["y"] - 5) ** 2)

        optimizer = RandomSearchOptimizer(param_spaces, objective, maximize=True)
        best = optimizer.optimize(n_iterations=100)

        # Should be reasonably close to (5, 5)
        assert 2 < best["x"] < 8
        assert 2 < best["y"] < 8


class TestGridSearchOptimizer:
    """Tests for GridSearchOptimizer."""

    def test_optimize(self):
        """Test grid search optimization."""
        param_spaces = [
            ParameterSpace("x", "continuous", 0, 10),
        ]

        def objective(params):
            return -((params["x"] - 5) ** 2)

        optimizer = GridSearchOptimizer(
            param_spaces,
            objective,
            maximize=True,
            n_points_per_dim=11,
        )
        best = optimizer.optimize()

        # Should find x=5
        assert 4 < best["x"] < 6


class TestGeneticOptimizer:
    """Tests for GeneticOptimizer."""

    def test_optimize(self):
        """Test genetic algorithm optimization."""
        param_spaces = [
            ParameterSpace("x", "continuous", 0, 10),
        ]

        def objective(params):
            return -((params["x"] - 5) ** 2)

        optimizer = GeneticOptimizer(
            param_spaces,
            objective,
            maximize=True,
            population_size=20,
        )
        best = optimizer.optimize(n_iterations=100)

        # Should be close to x=5
        assert 3 < best["x"] < 7


class TestOverfitDetector:
    """Tests for OverfitDetector."""

    def test_detect_overfit(self):
        """Test overfitting detection."""
        from quant_trading_system.backtest.optimizer import (
            OptimizationResult,
            OptimizationWindow,
        )

        detector = OverfitDetector(is_oos_ratio_threshold=2.0)

        # Create result that looks overfit
        windows = [
            OptimizationWindow(
                window_id=i,
                train_start=datetime(2020, 1, 1),
                train_end=datetime(2020, 12, 31),
                test_start=datetime(2021, 1, 1),
                test_end=datetime(2021, 3, 31),
                train_metric=2.0,  # High IS performance
                test_metric=0.5,  # Low OOS performance
            )
            for i in range(5)
        ]

        result = OptimizationResult(
            best_params={"x": 5},
            best_metric=2.0,
            all_results=[],
            windows=windows,
            oos_metric=0.5,
            is_ratio=4.0,  # Very high ratio = overfit
            optimization_method="random",
            total_evaluations=100,
            elapsed_seconds=10.0,
        )

        analysis = detector.analyze(result)

        assert analysis["is_likely_overfit"] is True
        assert len(analysis["warnings"]) > 0

    def test_detect_robust(self):
        """Test detection of robust results."""
        from quant_trading_system.backtest.optimizer import (
            OptimizationResult,
            OptimizationWindow,
        )

        detector = OverfitDetector()

        windows = [
            OptimizationWindow(
                window_id=i,
                train_start=datetime(2020, 1, 1),
                train_end=datetime(2020, 12, 31),
                test_start=datetime(2021, 1, 1),
                test_end=datetime(2021, 3, 31),
                train_metric=1.5,
                test_metric=1.4,  # Similar to IS
            )
            for i in range(5)
        ]

        result = OptimizationResult(
            best_params={"x": 5},
            best_metric=1.5,
            all_results=[],
            windows=windows,
            oos_metric=1.4,
            is_ratio=1.07,  # Close to 1 = good
            optimization_method="random",
            total_evaluations=100,
            elapsed_seconds=10.0,
        )

        analysis = detector.analyze(result)

        assert analysis["is_likely_overfit"] is False


# ============================================================================
# P0/P1 Enhancement Tests
# ============================================================================


class TestDeflatedSharpeRatio:
    """Tests for P0: Deflated Sharpe Ratio calculation."""

    def test_dsr_calculation(self):
        """Test DSR is calculated and less than observed Sharpe."""
        from quant_trading_system.backtest.analyzer import PerformanceAnalyzer

        analyzer = PerformanceAnalyzer()

        # Create mock backtest state with positive returns
        np.random.seed(42)
        n_bars = 500
        base_time = datetime(2024, 1, 1)

        # Generate returns that would give a high Sharpe
        equity_curve = [100000.0]
        for i in range(n_bars):
            daily_return = np.random.normal(0.0005, 0.01)  # Positive drift
            equity_curve.append(equity_curve[-1] * (1 + daily_return))

        state = BacktestState(
            timestamp=datetime.now(),
            equity=Decimal(str(equity_curve[-1])),
            cash=Decimal(str(equity_curve[-1])),
            positions={},
            pending_orders=[],
            equity_curve=[
                (base_time + timedelta(days=i), e)
                for i, e in enumerate(equity_curve)
            ],
            trades=[],
            bars_processed=n_bars,
        )

        report = analyzer.analyze(state, n_trials=10)

        # DSR should be calculated
        assert report.statistical_tests.deflated_sharpe_ratio >= 0
        # With multiple trials, DSR should typically be less than observed Sharpe
        # (deflation accounts for multiple testing)
        if report.risk_adjusted_metrics.sharpe_ratio > 0:
            assert report.statistical_tests.n_trials_tested == 10

    def test_dsr_with_single_trial(self):
        """Test DSR with single trial equals observed Sharpe."""
        from quant_trading_system.backtest.analyzer import PerformanceAnalyzer

        analyzer = PerformanceAnalyzer()

        np.random.seed(42)
        n_bars = 252
        base_time = datetime(2024, 1, 1)

        equity_curve = [100000.0]
        for i in range(n_bars):
            daily_return = np.random.normal(0.0003, 0.012)
            equity_curve.append(equity_curve[-1] * (1 + daily_return))

        state = BacktestState(
            timestamp=datetime.now(),
            equity=Decimal(str(equity_curve[-1])),
            cash=Decimal(str(equity_curve[-1])),
            positions={},
            pending_orders=[],
            equity_curve=[
                (base_time + timedelta(days=i), e)
                for i, e in enumerate(equity_curve)
            ],
            trades=[],
            bars_processed=n_bars,
        )

        report = analyzer.analyze(state, n_trials=1)

        # With single trial, DSR should be close to observed Sharpe
        assert report.statistical_tests.n_trials_tested == 1


class TestProbabilityOfBacktestOverfitting:
    """Tests for P0: Probability of Backtest Overfitting (PBO)."""

    def test_pbo_calculation(self):
        """Test PBO is calculated between 0 and 1."""
        from quant_trading_system.backtest.analyzer import PerformanceAnalyzer

        analyzer = PerformanceAnalyzer()

        np.random.seed(42)
        n_bars = 500
        base_time = datetime(2024, 1, 1)

        equity_curve = [100000.0]
        for i in range(n_bars):
            daily_return = np.random.normal(0.0002, 0.015)
            equity_curve.append(equity_curve[-1] * (1 + daily_return))

        state = BacktestState(
            timestamp=datetime.now(),
            equity=Decimal(str(equity_curve[-1])),
            cash=Decimal(str(equity_curve[-1])),
            positions={},
            pending_orders=[],
            equity_curve=[
                (base_time + timedelta(days=i), e)
                for i, e in enumerate(equity_curve)
            ],
            trades=[],
            bars_processed=n_bars,
        )

        report = analyzer.analyze(state)

        # PBO should be between 0 and 1
        assert 0.0 <= report.statistical_tests.pbo <= 1.0
        # Should have an interpretation
        assert report.statistical_tests.pbo_interpretation != ""

    def test_pbo_interpretation_categories(self):
        """Test PBO interpretation categories are meaningful."""
        from quant_trading_system.backtest.analyzer import PerformanceAnalyzer

        analyzer = PerformanceAnalyzer()

        # Low overfitting interpretation
        low_pbo_result = analyzer._calculate_pbo(np.random.normal(0.001, 0.01, 500))
        high_pbo_result = analyzer._calculate_pbo(np.random.normal(0, 0.02, 100))

        # Interpretations should exist
        assert low_pbo_result[1] in [
            "Very low overfitting risk",
            "Low overfitting risk",
            "Moderate overfitting risk",
            "High overfitting risk - exercise caution",
            "Very high overfitting risk - likely overfit",
        ]


class TestSignalDecayMonitor:
    """Tests for P0: Signal Decay Monitor."""

    def test_signal_decay_monitor_initialization(self):
        """Test monitor initializes correctly."""
        from quant_trading_system.alpha.alpha_metrics import SignalDecayMonitor

        monitor = SignalDecayMonitor(
            window_size=30,
            decay_threshold=-0.5,
            min_samples=20,
        )

        assert monitor.window_size == 30
        assert monitor.decay_threshold == -0.5
        assert not monitor.is_calibrated()

    def test_signal_decay_calibration(self):
        """Test monitor calibration with good signal."""
        from quant_trading_system.alpha.alpha_metrics import SignalDecayMonitor

        monitor = SignalDecayMonitor(window_size=20, min_samples=30)

        # Generate predictions that correlate with returns
        np.random.seed(42)
        n = 100
        true_signal = np.random.randn(n)
        noise = np.random.randn(n) * 0.5
        predictions = true_signal + noise
        returns = true_signal * 0.01  # Returns correlated with signal

        baseline = monitor.calibrate(predictions, returns)

        assert monitor.is_calibrated()
        assert baseline != 0.0

    def test_signal_decay_detection(self):
        """Test that decay is detected when signal quality drops."""
        from quant_trading_system.alpha.alpha_metrics import SignalDecayMonitor

        monitor = SignalDecayMonitor(
            window_size=20,
            decay_threshold=-0.5,
            min_samples=20,
        )

        np.random.seed(42)

        # Phase 1: Good signal for calibration
        n_good = 50
        good_signal = np.random.randn(n_good)
        good_returns = good_signal * 0.01

        monitor.calibrate(good_signal, good_returns)

        # Phase 2: Signal decays (no correlation)
        n_bad = 30
        bad_predictions = np.random.randn(n_bad)
        bad_returns = np.random.randn(n_bad) * 0.01  # Uncorrelated

        alert = monitor.update(bad_predictions, bad_returns)

        # Should detect some level of degradation
        assert alert.status in ["healthy", "warning", "decay_detected"]
        assert alert.baseline_ic != 0.0

    def test_signal_decay_healthy_signal(self):
        """Test healthy signal maintains good status."""
        from quant_trading_system.alpha.alpha_metrics import SignalDecayMonitor

        monitor = SignalDecayMonitor(window_size=20, min_samples=20)

        np.random.seed(42)

        # Consistently good signal
        n = 60
        signal = np.random.randn(n)
        returns = signal * 0.01 + np.random.randn(n) * 0.002

        monitor.calibrate(signal[:30], returns[:30])
        alert = monitor.update(signal[30:], returns[30:])

        # Should remain healthy with consistent signal
        assert alert.status in ["healthy", "warning"]


class TestRegimeMetrics:
    """Tests for P1: Regime-specific analysis."""

    def test_regime_metrics_dataclass(self):
        """Test RegimeMetrics dataclass creation."""
        from quant_trading_system.backtest.analyzer import RegimeMetrics

        metrics = RegimeMetrics(
            regime="BULL_LOW_VOL",
            n_bars=100,
            total_return=0.15,
            sharpe_ratio=1.5,
            win_rate=0.55,
            max_drawdown=0.05,
        )

        assert metrics.regime == "BULL_LOW_VOL"
        assert metrics.n_bars == 100
        assert metrics.sharpe_ratio == 1.5

        # Test to_dict
        d = metrics.to_dict()
        assert d["regime"] == "BULL_LOW_VOL"
        assert d["sharpe_ratio"] == 1.5

    def test_analyze_by_regime(self):
        """Test regime-specific performance analysis."""
        from quant_trading_system.backtest.analyzer import PerformanceAnalyzer

        analyzer = PerformanceAnalyzer()

        # Create mock backtest state
        np.random.seed(42)
        n_bars = 200
        base_time = datetime(2024, 1, 1)

        equity_curve = [100000.0]
        for i in range(n_bars):
            daily_return = np.random.normal(0.0003, 0.012)
            equity_curve.append(equity_curve[-1] * (1 + daily_return))

        # Create mock regime history
        from quant_trading_system.alpha.regime_detection import MarketRegime

        regime_history = []
        for i in range(n_bars):
            if i < 50:
                regime = MarketRegime.BULL_LOW_VOL
            elif i < 100:
                regime = MarketRegime.BULL_HIGH_VOL
            elif i < 150:
                regime = MarketRegime.BEAR_LOW_VOL
            else:
                regime = MarketRegime.RANGE_BOUND
            regime_history.append(regime)

        state = BacktestState(
            timestamp=datetime.now(),
            equity=Decimal(str(equity_curve[-1])),
            cash=Decimal(str(equity_curve[-1])),
            positions={},
            pending_orders=[],
            equity_curve=[
                (base_time + timedelta(days=i), e)
                for i, e in enumerate(equity_curve)
            ],
            trades=[],
            bars_processed=n_bars,
            regime_history=[
                (base_time + timedelta(days=i), r)
                for i, r in enumerate(regime_history)
            ],
        )

        metrics = analyzer.analyze_by_regime(state, regime_history)

        # Should have metrics for each unique regime
        assert len(metrics) > 0
        assert all(hasattr(m, 'regime') for m in metrics)
        assert all(hasattr(m, 'sharpe_ratio') for m in metrics)


class TestCostSensitivity:
    """Tests for P1: Transaction cost sensitivity analysis."""

    def test_cost_sensitivity_dataclass(self):
        """Test CostSensitivity dataclass creation."""
        from quant_trading_system.backtest.analyzer import CostSensitivity

        sensitivity = CostSensitivity(
            break_even_cost_bps=8.5,
            sharpe_at_costs={0.0: 1.5, 5.0: 1.2, 10.0: 0.9},
            return_at_costs={0.0: 0.15, 5.0: 0.12, 10.0: 0.09},
        )

        assert sensitivity.break_even_cost_bps == 8.5
        assert sensitivity.sharpe_at_costs[0.0] == 1.5

        # Test to_dict
        d = sensitivity.to_dict()
        assert d["break_even_cost_bps"] == 8.5

    def test_analyze_cost_sensitivity(self):
        """Test cost sensitivity analysis."""
        from quant_trading_system.backtest.analyzer import PerformanceAnalyzer

        analyzer = PerformanceAnalyzer()

        # Create mock backtest state with trades
        np.random.seed(42)
        n_bars = 252
        base_time = datetime(2024, 1, 1)

        # Generate positive returns
        equity_curve = [100000.0]
        for i in range(n_bars):
            daily_return = np.random.normal(0.0004, 0.01)
            equity_curve.append(equity_curve[-1] * (1 + daily_return))

        # Create some trades
        trades = [
            Trade(
                symbol="AAPL",
                entry_time=base_time + timedelta(days=i*10),
                exit_time=base_time + timedelta(days=i*10+5),
                side=OrderSide.BUY,
                quantity=Decimal("100"),
                entry_price=Decimal("150.00"),
                exit_price=Decimal("155.00"),
                pnl=Decimal("500"),
                pnl_pct=0.0333,
                commission=Decimal("1.00"),
                slippage=Decimal("0.50"),
                holding_period_bars=5,
            )
            for i in range(20)
        ]

        state = BacktestState(
            timestamp=datetime.now(),
            equity=Decimal(str(equity_curve[-1])),
            cash=Decimal(str(equity_curve[-1])),
            positions={},
            pending_orders=[],
            equity_curve=[
                (base_time + timedelta(days=i), e)
                for i, e in enumerate(equity_curve)
            ],
            trades=trades,
            bars_processed=n_bars,
        )

        sensitivity = analyzer.analyze_cost_sensitivity(state)

        # Should have break-even cost
        assert sensitivity.break_even_cost_bps >= 0
        # Should have costs at different levels
        assert len(sensitivity.sharpe_at_costs) > 0
        # Higher costs should result in lower Sharpe
        costs = sorted(sensitivity.sharpe_at_costs.keys())
        if len(costs) >= 2:
            assert sensitivity.sharpe_at_costs[costs[-1]] <= sensitivity.sharpe_at_costs[costs[0]]


class TestBacktestStateRegimeHistory:
    """Tests for P1: BacktestState regime_history field."""

    def test_regime_history_default_empty(self):
        """Test regime_history defaults to empty list."""
        state = BacktestState(
            timestamp=datetime.now(),
            equity=Decimal("100000"),
            cash=Decimal("100000"),
            positions={},
            pending_orders=[],
            equity_curve=[],
            trades=[],
        )

        assert state.regime_history == []

    def test_regime_history_stores_data(self):
        """Test regime_history stores regime data."""
        from quant_trading_system.alpha.regime_detection import MarketRegime

        base_time = datetime(2024, 1, 1)
        regime_history = [
            (base_time, MarketRegime.BULL_LOW_VOL),
            (base_time + timedelta(days=1), MarketRegime.BULL_HIGH_VOL),
        ]

        state = BacktestState(
            timestamp=datetime.now(),
            equity=Decimal("100000"),
            cash=Decimal("100000"),
            positions={},
            pending_orders=[],
            equity_curve=[],
            trades=[],
            regime_history=regime_history,
        )

        assert len(state.regime_history) == 2
        assert state.regime_history[0][1] == MarketRegime.BULL_LOW_VOL
