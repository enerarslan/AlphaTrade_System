"""Unit tests for scripts/backtest.py regression paths."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd

from quant_trading_system.backtest.engine import BacktestState, PandasDataHandler, Trade
from quant_trading_system.backtest.performance_attribution import PerformanceAttributionService
from quant_trading_system.core.data_types import Direction, OrderSide, Portfolio
from scripts.backtest import BacktestRunner, BacktestSession, SignalBasedStrategy, run_backtest


def test_performance_attribution_compute_attribution_accepts_backtest_state():
    """BacktestRunner attribution call should work with BacktestState payloads."""
    t0 = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=15)

    trade = Trade(
        symbol="AAPL",
        entry_time=t0,
        exit_time=t1,
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        entry_price=Decimal("100"),
        exit_price=Decimal("101"),
        pnl=Decimal("8"),
        pnl_pct=0.008,
        commission=Decimal("1"),
        slippage=Decimal("1"),
        holding_period_bars=1,
    )

    result = BacktestState(
        timestamp=t1,
        equity=Decimal("100800"),
        cash=Decimal("100800"),
        positions={},
        pending_orders=[],
        equity_curve=[(t0, 100000.0), (t1, 100800.0)],
        trades=[trade],
    )

    report = PerformanceAttributionService().compute_attribution(result)

    assert "summary" in report
    assert "trade_attribution" in report
    assert report["trade_attribution"]["total_trades"] == 1


def test_backtest_runner_monte_carlo_handles_equity_curve_tuples(capsys):
    """Monte Carlo should handle (timestamp, equity) tuples without attribute errors."""
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 2, tzinfo=timezone.utc)
    session = BacktestSession(
        start_date=start,
        end_date=end,
        symbols=["AAPL"],
        initial_capital=Decimal("100000"),
        monte_carlo_sims=10,
    )
    runner = BacktestRunner(session)

    result = SimpleNamespace(
        equity_curve=[
            (start, 100000.0),
            (start + timedelta(minutes=15), 101000.0),
            (start + timedelta(minutes=30), 99500.0),
            (start + timedelta(minutes=45), 102250.0),
        ]
    )

    runner._run_monte_carlo(result)
    output = capsys.readouterr().out
    assert "Monte Carlo Analysis" in output


def test_backtest_runner_passes_benchmark_returns_to_analyzer(monkeypatch):
    """Analyzer should receive benchmark return series when benchmark is configured."""
    import scripts.backtest as backtest_script

    captured: dict[str, object] = {}

    class _Metric:
        def to_dict(self):
            return {}

    class _Report:
        return_metrics = _Metric()
        risk_adjusted_metrics = _Metric()
        drawdown_metrics = _Metric()
        trade_metrics = _Metric()
        benchmark_metrics = None
        statistical_tests = _Metric()

    class _Analyzer:
        def analyze(self, _result, benchmark_returns=None):
            captured["benchmark_returns"] = benchmark_returns
            return _Report()

    monkeypatch.setattr(backtest_script, "PerformanceAnalyzer", lambda: _Analyzer())

    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=15)
    session = BacktestSession(
        start_date=t0,
        end_date=t1,
        symbols=["AAPL"],
        initial_capital=Decimal("100000"),
        benchmark_symbol="SPY",
    )
    runner = BacktestRunner(session)
    runner.data["SPY"] = pd.DataFrame({"close": [100.0, 101.0, 99.0]})

    result = BacktestState(
        timestamp=t1,
        equity=Decimal("100000"),
        cash=Decimal("100000"),
        positions={},
        pending_orders=[],
        equity_curve=[(t0, 100000.0), (t1, 100100.0)],
        trades=[],
    )
    runner._analyze_results(result)

    benchmark_returns = captured.get("benchmark_returns")
    assert benchmark_returns is not None
    assert len(benchmark_returns) == 2


def test_signal_based_strategy_uses_bar_timestamp_lookup():
    t0 = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=15)
    handler = PandasDataHandler(
        {
            "AAPL": pd.DataFrame(
                {
                    "open": [100.0, 100.0],
                    "high": [101.0, 101.0],
                    "low": [99.0, 99.0],
                    "close": [100.0, 100.0],
                    "volume": [1_000_000, 1_000_000],
                },
                index=pd.DatetimeIndex([t0, t1]),
            )
        }
    )
    strategy = SignalBasedStrategy(
        {
            "AAPL": pd.DataFrame(
                {"signal": [0.0, -0.8]},
                index=pd.DatetimeIndex([t0, t1]),
            )
        },
        signal_threshold=0.1,
    )
    portfolio = Portfolio(
        equity=Decimal("100000"),
        cash=Decimal("100000"),
        buying_power=Decimal("100000"),
        positions={},
    )

    assert handler.update_bars() is True
    assert strategy.generate_signals(handler, portfolio) == []

    assert handler.update_bars() is True
    signals = strategy.generate_signals(handler, portfolio)

    assert len(signals) == 1
    assert signals[0].direction == Direction.SHORT
    assert signals[0].timestamp == t1


def test_run_backtest_rejects_csv_mode():
    args = SimpleNamespace(
        start="2025-01-01",
        end="2025-01-02",
        symbols=["AAPL"],
        capital=100000.0,
        strategy="momentum",
        execution_mode="realistic",
        gpu=False,
        timeframe="15Min",
        use_database=True,
        no_database=True,
        benchmark=None,
        slippage_bps=5.0,
        commission_bps=1.0,
        monte_carlo=0,
        output=None,
        log_level="INFO",
    )

    assert run_backtest(args) == 1
