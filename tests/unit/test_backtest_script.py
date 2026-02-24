"""Unit tests for scripts/backtest.py regression paths."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd

from quant_trading_system.backtest.engine import BacktestState, Trade
from quant_trading_system.backtest.performance_attribution import PerformanceAttributionService
from quant_trading_system.core.data_types import OrderSide
from scripts.backtest import BacktestRunner, BacktestSession


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
