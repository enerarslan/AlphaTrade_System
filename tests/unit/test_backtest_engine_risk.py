"""Risk-integration tests for the event-driven backtest engine."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pandas as pd

from quant_trading_system.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    PandasDataHandler,
    Strategy,
)
from quant_trading_system.backtest.simulator import FillResult, FillType
from quant_trading_system.core.data_types import Direction, OrderSide, TradeSignal
from quant_trading_system.risk.limits import KillSwitchReason


class OneShotStrategy(Strategy):
    """Emit one signal once, then stay flat."""

    def __init__(self, direction: Direction = Direction.LONG) -> None:
        self.direction = direction
        self._emitted = False

    def generate_signals(self, data_handler, portfolio):
        if self._emitted:
            return []
        self._emitted = True
        symbol = data_handler.get_symbols()[0]
        return [
            TradeSignal(
                symbol=symbol,
                direction=self.direction,
                strength=1.0,
                confidence=1.0,
                horizon=1,
                model_source="unit_test",
            )
        ]


class NoSignalStrategy(Strategy):
    """Never emits signals."""

    def generate_signals(self, data_handler, portfolio):
        return []


class AlwaysLongStrategy(Strategy):
    """Emit long signal on every bar."""

    def generate_signals(self, data_handler, portfolio):
        symbol = data_handler.get_symbols()[0]
        return [
            TradeSignal(
                symbol=symbol,
                direction=Direction.LONG,
                strength=1.0,
                confidence=1.0,
                horizon=1,
                model_source="unit_test",
            )
        ]


def _sample_handler() -> PandasDataHandler:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    data = {
        "AAPL": pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 50.0, 50.0],
                "high": [101.0, 101.0, 101.0, 51.0, 51.0],
                "low": [99.0, 99.0, 49.0, 49.0, 49.0],
                "close": [100.0, 100.0, 50.0, 50.0, 50.0],
                "volume": [1_000_000, 1_000_000, 1_000_000, 1_000_000, 1_000_000],
            },
            index=dates,
        )
    }
    return PandasDataHandler(data)


def test_backtest_engine_allows_short_signal_when_shorting_enabled():
    handler = _sample_handler()
    engine = BacktestEngine(
        data_handler=handler,
        strategy=OneShotStrategy(Direction.SHORT),
        config=BacktestConfig(allow_short=True, use_market_simulator=False),
    )
    engine._initialize()
    assert handler.update_bars() is True

    engine._process_signal(
        TradeSignal(
            symbol="AAPL",
            direction=Direction.SHORT,
            strength=1.0,
            confidence=1.0,
            horizon=1,
            model_source="unit_test",
        )
    )

    assert len(engine._state.pending_orders) == 1
    assert engine._state.pending_orders[0].side == OrderSide.SELL


def test_pandas_data_handler_uses_union_timestamp_stream_for_multi_symbol_panels():
    t0 = pd.Timestamp("2024-01-02 14:30:00+00:00")
    t1 = pd.Timestamp("2024-01-02 14:45:00+00:00")
    handler = PandasDataHandler(
        {
            "AAPL": pd.DataFrame(
                {
                    "open": [100.0, 101.0],
                    "high": [101.0, 102.0],
                    "low": [99.0, 100.0],
                    "close": [100.5, 101.5],
                    "volume": [1_000_000, 1_000_000],
                },
                index=pd.DatetimeIndex([t0, t1]),
            ),
            "MSFT": pd.DataFrame(
                {
                    "open": [200.0],
                    "high": [201.0],
                    "low": [199.0],
                    "close": [200.5],
                    "volume": [1_000_000],
                },
                index=pd.DatetimeIndex([t1]),
            ),
        }
    )

    assert handler.update_bars() is True
    assert handler.get_current_timestamp() == t0.to_pydatetime()
    assert handler.get_updated_symbols() == ["AAPL"]

    assert handler.update_bars() is True
    assert set(handler.get_updated_symbols()) == {"AAPL", "MSFT"}


def test_backtest_engine_risk_limit_uses_configured_max_position_pct():
    engine = BacktestEngine(
        data_handler=_sample_handler(),
        strategy=NoSignalStrategy(),
        config=BacktestConfig(
            initial_capital=Decimal("100000"),
            max_position_pct=0.10,
            use_market_simulator=False,
        ),
    )

    risk_config = engine._build_risk_limits_config()

    assert risk_config.max_position_value == Decimal("10000")


def test_backtest_engine_blocks_new_orders_when_kill_switch_active():
    handler = _sample_handler()
    engine = BacktestEngine(
        data_handler=handler,
        strategy=OneShotStrategy(Direction.LONG),
        config=BacktestConfig(use_market_simulator=False),
    )
    engine._initialize()
    assert handler.update_bars() is True
    engine._risk_limits_manager.kill_switch.manual_activate(activated_by="unit_test")

    engine._process_signal(
        TradeSignal(
            symbol="AAPL",
            direction=Direction.LONG,
            strength=1.0,
            confidence=1.0,
            horizon=1,
            model_source="unit_test",
        )
    )

    assert engine._state.pending_orders == []


def test_backtest_engine_halts_on_drawdown_via_risk_manager():
    handler = _sample_handler()
    strategy = OneShotStrategy(Direction.LONG)
    config = BacktestConfig(
        initial_capital=Decimal("100000"),
        max_position_pct=0.9,
        max_drawdown_halt=0.10,
        use_market_simulator=False,
        enforce_market_calendar=False,
        allow_fractional=False,
    )
    engine = BacktestEngine(data_handler=handler, strategy=strategy, config=config)

    state = engine.run()

    assert state.bars_processed < 5
    assert engine._risk_limits_manager.kill_switch.is_active()


def test_backtest_engine_respects_early_close_session_filter():
    dates = pd.DatetimeIndex(
        [
            "2024-11-29 12:00:00-05:00",  # Day after Thanksgiving (early close day)
            "2024-11-29 14:00:00-05:00",  # Outside 1pm close
        ]
    )
    data = {
        "AAPL": pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1_000_000, 1_000_000],
            },
            index=dates,
        )
    }

    engine = BacktestEngine(
        data_handler=PandasDataHandler(data),
        strategy=NoSignalStrategy(),
        config=BacktestConfig(use_market_simulator=False, enforce_market_calendar=True),
    )
    state = engine.run()

    assert state.bars_processed == 1
    assert len(state.equity_curve) == 1


def test_backtest_engine_execution_quality_gate_triggers_kill_switch():
    class RejectingSimulator:
        def simulate_execution(self, order, conditions):
            return FillResult(
                fill_type=FillType.REJECTED,
                fill_price=Decimal("0"),
                fill_quantity=Decimal("0"),
                slippage=Decimal("0"),
                market_impact=Decimal("0"),
                commission=Decimal("0"),
                latency_ms=0.0,
                timestamp=conditions.timestamp,
                rejection_reason="forced_rejection",
            )

    dates = pd.date_range("2024-01-02 10:00:00", periods=8, freq="15min", tz="America/New_York")
    data = {
        "AAPL": pd.DataFrame(
            {
                "open": [100.0] * len(dates),
                "high": [101.0] * len(dates),
                "low": [99.0] * len(dates),
                "close": [100.0] * len(dates),
                "volume": [1_000_000] * len(dates),
            },
            index=dates,
        )
    }

    config = BacktestConfig(
        use_market_simulator=True,
        min_orders_for_execution_gate=3,
        max_fill_rejection_rate=0.10,
        max_avg_slippage_bps=1_000.0,
        enforce_market_calendar=True,
    )
    engine = BacktestEngine(
        data_handler=PandasDataHandler(data),
        strategy=AlwaysLongStrategy(),
        config=config,
        market_simulator=RejectingSimulator(),
    )
    state = engine.run()

    assert state.bars_processed < len(dates)
    assert engine._risk_limits_manager.kill_switch.is_active()
    assert engine._risk_limits_manager.kill_switch.state.reason == KillSwitchReason.LIMIT_BREACH


def test_backtest_engine_dynamic_capacity_multiplier_reduces_size_after_poor_execution():
    handler = _sample_handler()
    config = BacktestConfig(
        use_market_simulator=False,
        capacity_safety_buffer=0.5,
        min_orders_for_execution_gate=1,
        max_fill_rejection_rate=0.50,
        max_avg_slippage_bps=20.0,
        min_capacity_multiplier=0.25,
    )
    engine = BacktestEngine(data_handler=handler, strategy=NoSignalStrategy(), config=config)
    engine._initialize()

    engine._orders_submitted_total = 10
    engine._orders_rejected_total = 7  # 70% rejection -> severe degradation
    engine._recent_slippage_bps.extend([50.0, 45.0, 55.0])  # also above threshold

    cap = engine._get_liquidity_cap_quantity(bar_volume=1_000, avg_daily_volume=1_000_000)
    assert cap < Decimal("50")  # 100 * 0.5 base * degraded factor


def test_backtest_engine_defers_latency_fills_until_eligible_bar():
    class DelayedFillSimulator:
        def simulate_execution(self, order, conditions):
            return FillResult(
                fill_type=FillType.FULL,
                fill_price=conditions.price,
                fill_quantity=order.quantity,
                slippage=Decimal("0"),
                market_impact=Decimal("0"),
                commission=Decimal("0"),
                latency_ms=86_400_000.0,
                timestamp=conditions.timestamp + timedelta(days=1),
            )

    handler = _sample_handler()
    engine = BacktestEngine(
        data_handler=handler,
        strategy=NoSignalStrategy(),
        config=BacktestConfig(use_market_simulator=True, simulate_latency=True),
        market_simulator=DelayedFillSimulator(),
    )
    engine._initialize()

    assert handler.update_bars() is True
    first_bar = handler.get_current_bar("AAPL")
    assert first_bar is not None
    engine._update_position_prices("AAPL", first_bar)
    engine._process_signal(
        TradeSignal(
            symbol="AAPL",
            direction=Direction.LONG,
            strength=1.0,
            confidence=1.0,
            horizon=1,
            model_source="unit_test",
        )
    )
    engine._process_pending_orders("AAPL", first_bar)
    assert engine._state.orders_filled == 0
    assert len(engine._state.pending_orders) == 1

    assert handler.update_bars() is True
    second_bar = handler.get_current_bar("AAPL")
    assert second_bar is not None
    engine._update_position_prices("AAPL", second_bar)
    engine._process_pending_orders("AAPL", second_bar)
    assert engine._state.orders_filled == 1
    assert len(engine._state.pending_orders) == 0


def test_backtest_engine_partial_remainder_not_reprocessed_same_bar():
    class OnePartialThenFullSimulator:
        def __init__(self):
            self.calls = 0

        def simulate_execution(self, order, conditions):
            self.calls += 1
            if self.calls == 1:
                partial_qty = order.quantity / Decimal("2")
                return FillResult(
                    fill_type=FillType.PARTIAL,
                    fill_price=conditions.price,
                    fill_quantity=partial_qty,
                    slippage=Decimal("0"),
                    market_impact=Decimal("0"),
                    commission=Decimal("0"),
                    latency_ms=0.0,
                    timestamp=conditions.timestamp,
                    partial_remaining=order.quantity - partial_qty,
                )
            return FillResult(
                fill_type=FillType.FULL,
                fill_price=conditions.price,
                fill_quantity=order.quantity,
                slippage=Decimal("0"),
                market_impact=Decimal("0"),
                commission=Decimal("0"),
                latency_ms=0.0,
                timestamp=conditions.timestamp,
            )

    simulator = OnePartialThenFullSimulator()
    handler = _sample_handler()
    engine = BacktestEngine(
        data_handler=handler,
        strategy=NoSignalStrategy(),
        config=BacktestConfig(use_market_simulator=True, simulate_latency=False),
        market_simulator=simulator,
    )
    engine._initialize()

    assert handler.update_bars() is True
    bar = handler.get_current_bar("AAPL")
    assert bar is not None
    engine._update_position_prices("AAPL", bar)
    engine._process_signal(
        TradeSignal(
            symbol="AAPL",
            direction=Direction.LONG,
            strength=1.0,
            confidence=1.0,
            horizon=1,
            model_source="unit_test",
        )
    )
    engine._process_pending_orders("AAPL", bar)

    # New remainder must stay pending and not get filled in this same pass.
    assert simulator.calls == 1
    assert engine._state.orders_filled == 1
    assert len(engine._state.pending_orders) == 1
