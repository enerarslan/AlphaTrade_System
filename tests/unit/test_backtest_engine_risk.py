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
from quant_trading_system.core.data_types import Direction, OrderSide, Position, TradeSignal
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


class OneShotMetadataStrategy(Strategy):
    """Emit one signal once with custom metadata and confidence."""

    def __init__(
        self,
        *,
        direction: Direction = Direction.LONG,
        confidence: float = 1.0,
        horizon: int = 1,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.direction = direction
        self.confidence = confidence
        self.horizon = horizon
        self.metadata = metadata or {}
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
                strength=1.0 if self.direction == Direction.LONG else -1.0,
                confidence=self.confidence,
                horizon=self.horizon,
                model_source="unit_test_metadata",
                metadata=dict(self.metadata),
            )
        ]


class BatchSignalStrategy(Strategy):
    """Emit a predefined batch of signals once."""

    def __init__(self, signals: list[TradeSignal]) -> None:
        self.signals = signals
        self._emitted = False

    def generate_signals(self, data_handler, portfolio):
        if self._emitted:
            return []
        self._emitted = True
        return list(self.signals)


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


def test_backtest_engine_scales_position_size_by_signal_confidence():
    handler = _sample_handler()
    engine = BacktestEngine(
        data_handler=handler,
        strategy=NoSignalStrategy(),
        config=BacktestConfig(
            initial_capital=Decimal("100000"),
            max_position_pct=0.10,
            use_market_simulator=False,
            allow_fractional=False,
        ),
    )
    engine._initialize()
    assert handler.update_bars() is True

    engine._process_signal(
        TradeSignal(
            symbol="AAPL",
            direction=Direction.LONG,
            strength=1.0,
            confidence=0.50,
            horizon=1,
            model_source="unit_test",
        )
    )

    assert len(engine._state.pending_orders) == 1
    assert engine._state.pending_orders[0].quantity == Decimal("50")


def test_backtest_engine_target_sizing_does_not_stack_same_direction_when_at_target():
    handler = _sample_handler()
    engine = BacktestEngine(
        data_handler=handler,
        strategy=NoSignalStrategy(),
        config=BacktestConfig(
            initial_capital=Decimal("100000"),
            max_position_pct=0.10,
            use_market_simulator=False,
            allow_fractional=False,
            use_portfolio_target_sizing=True,
        ),
    )
    engine._initialize()
    assert handler.update_bars() is True

    engine._state.positions["AAPL"] = Position(
        symbol="AAPL",
        quantity=Decimal("100"),
        avg_entry_price=Decimal("100"),
        current_price=Decimal("100"),
        cost_basis=Decimal("10000"),
        market_value=Decimal("10000"),
        unrealized_pnl=Decimal("0"),
        realized_pnl=Decimal("0"),
    )

    plans = engine._build_target_position_plans(
        [
            TradeSignal(
                symbol="AAPL",
                direction=Direction.LONG,
                strength=1.0,
                confidence=1.0,
                horizon=1,
                model_source="unit_test",
            )
        ]
    )

    assert plans == []


def test_backtest_engine_target_sizing_can_disable_confidence_scaling():
    handler = _sample_handler()
    engine = BacktestEngine(
        data_handler=handler,
        strategy=NoSignalStrategy(),
        config=BacktestConfig(
            initial_capital=Decimal("100000"),
            max_position_pct=0.10,
            use_market_simulator=False,
            allow_fractional=False,
            use_portfolio_target_sizing=True,
            confidence_position_sizing=False,
        ),
    )
    engine._initialize()
    assert handler.update_bars() is True

    plans = engine._build_target_position_plans(
        [
            TradeSignal(
                symbol="AAPL",
                direction=Direction.LONG,
                strength=1.0,
                confidence=0.25,
                horizon=1,
                model_source="unit_test",
            )
        ]
    )

    assert len(plans) == 1
    assert plans[0][2] == Decimal("100")


def test_backtest_engine_target_sizing_normalizes_multi_signal_gross_exposure():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    symbols = ["AAPL", "MSFT", "NVDA", "AMD", "META"]
    handler = PandasDataHandler(
        {
            symbol: pd.DataFrame(
                {
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.0],
                    "volume": [1_000_000],
                },
                index=dates,
            )
            for symbol in symbols
        }
    )
    signals = [
        TradeSignal(
            symbol=symbol,
            direction=Direction.LONG,
            strength=1.0,
            confidence=1.0,
            horizon=2,
            model_source="unit_test",
        )
        for symbol in symbols
    ]
    engine = BacktestEngine(
        data_handler=handler,
        strategy=BatchSignalStrategy(signals),
        config=BacktestConfig(
            initial_capital=Decimal("100000"),
            max_position_pct=0.50,
            max_leverage=3.0,
            use_market_simulator=False,
            allow_fractional=False,
            enforce_market_calendar=False,
            use_portfolio_target_sizing=True,
        ),
    )
    engine._initialize()
    assert handler.update_bars() is True

    plans = engine._build_target_position_plans(signals)

    assert len(plans) == 5
    assert {symbol for symbol, _, _, _ in [(p[0].symbol, p[1], p[2], p[3]) for p in plans]} == set(
        symbols
    )
    assert all(quantity == Decimal("400") for _, _, quantity, _ in plans)


def test_backtest_engine_target_sizing_limits_signal_batch_to_max_total_positions():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    handler = PandasDataHandler(
        {
            "AAPL": pd.DataFrame(
                {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1_000_000]},
                index=dates,
            ),
            "MSFT": pd.DataFrame(
                {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1_000_000]},
                index=dates,
            ),
            "NVDA": pd.DataFrame(
                {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1_000_000]},
                index=dates,
            ),
        }
    )
    signals = [
        TradeSignal(
            symbol="AAPL",
            direction=Direction.LONG,
            strength=1.0,
            confidence=0.90,
            horizon=2,
            model_source="unit_test",
        ),
        TradeSignal(
            symbol="MSFT",
            direction=Direction.LONG,
            strength=1.0,
            confidence=0.80,
            horizon=2,
            model_source="unit_test",
        ),
        TradeSignal(
            symbol="NVDA",
            direction=Direction.LONG,
            strength=1.0,
            confidence=0.70,
            horizon=2,
            model_source="unit_test",
        ),
    ]
    engine = BacktestEngine(
        data_handler=handler,
        strategy=BatchSignalStrategy(signals),
        config=BacktestConfig(
            use_market_simulator=False,
            allow_fractional=False,
            enforce_market_calendar=False,
            use_portfolio_target_sizing=True,
            max_total_positions=2,
        ),
    )
    engine._initialize()
    assert handler.update_bars() is True

    plans = engine._build_target_position_plans(signals)

    assert len(plans) == 2
    assert [signal.symbol for signal, _, _, _ in plans] == ["AAPL", "MSFT"]


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


def test_backtest_engine_exits_position_on_stop_loss_reason():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    data = {
        "AAPL": pd.DataFrame(
            {
                "open": [100.0, 100.0, 90.0, 90.0],
                "high": [101.0, 101.0, 91.0, 91.0],
                "low": [99.0, 99.0, 89.0, 89.0],
                "close": [100.0, 100.0, 90.0, 90.0],
                "volume": [1_000_000] * 4,
            },
            index=dates,
        )
    }
    engine = BacktestEngine(
        data_handler=PandasDataHandler(data),
        strategy=OneShotMetadataStrategy(
            direction=Direction.LONG,
            confidence=1.0,
            horizon=5,
            metadata={"stop_loss_pct": 0.05, "max_holding_bars": 10},
        ),
        config=BacktestConfig(
            use_market_simulator=False,
            allow_fractional=False,
            enforce_market_calendar=False,
            max_position_pct=0.50,
        ),
    )

    state = engine.run()

    assert len(state.trades) == 1
    assert state.trades[0].exit_reason == "stop_loss"


def test_backtest_engine_exits_position_on_max_holding_reason():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    data = {
        "AAPL": pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 100.0, 100.0],
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.0] * 5,
                "volume": [1_000_000] * 5,
            },
            index=dates,
        )
    }
    engine = BacktestEngine(
        data_handler=PandasDataHandler(data),
        strategy=OneShotMetadataStrategy(
            direction=Direction.LONG,
            confidence=1.0,
            horizon=5,
            metadata={"max_holding_bars": 2},
        ),
        config=BacktestConfig(
            use_market_simulator=False,
            allow_fractional=False,
            enforce_market_calendar=False,
            max_position_pct=0.50,
        ),
    )

    state = engine.run()

    assert len(state.trades) == 1
    assert state.trades[0].exit_reason == "max_holding"


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


def test_backtest_engine_execution_slo_snapshot_exposes_candidate_and_direction_metrics():
    engine = BacktestEngine(
        data_handler=_sample_handler(),
        strategy=NoSignalStrategy(),
        config=BacktestConfig(use_market_simulator=False),
    )
    engine._initialize()
    engine._raw_candidates_total = 4
    engine._meta_passed_total = 3
    engine._edge_passed_total = 2
    engine._short_rejected_total = 1
    engine._long_orders_submitted_total = 2
    engine._short_orders_submitted_total = 1

    snapshot = engine.get_execution_slo_snapshot()

    assert snapshot["raw_candidates"] == 4
    assert snapshot["meta_passed"] == 3
    assert snapshot["edge_passed"] == 2
    assert snapshot["short_rejected"] == 1
    assert snapshot["fills"] == 0
    assert snapshot["long_short_split"] == {"long": 2, "short": 1}
