"""Unit tests for deterministic replay + SLO gates."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from quant_trading_system.backtest.engine import ExecutionMode
from quant_trading_system.backtest.replay import (
    ReplaySLOGates,
    ReplayScenario,
    ReplaySignalConfig,
    run_replay_scenario,
    run_replay_suite,
)


def _ohlcv(prices: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-02 14:30:00", periods=len(prices), freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": prices,
            "high": [price * 1.001 for price in prices],
            "low": [price * 0.999 for price in prices],
            "close": prices,
            "volume": [1_000_000] * len(prices),
        },
        index=index,
    )


def _scenario(
    allow_short: bool = True,
    backtest_overrides: dict[str, object] | None = None,
) -> ReplayScenario:
    return ReplayScenario(
        scenario_id="unit_replay",
        symbols=["AAPL"],
        start_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end_date=datetime(2024, 1, 5, tzinfo=timezone.utc),
        initial_capital=Decimal("100000"),
        execution_mode=ExecutionMode.REALISTIC,
        slippage_bps=1.0,
        commission_bps=0.5,
        allow_short=allow_short,
        signal=ReplaySignalConfig(return_threshold_bps=5.0, confidence=0.75, horizon_bars=1),
        backtest_overrides=backtest_overrides or {},
    )


def test_replay_scenario_passes_lenient_slo_gates():
    data = {"AAPL": _ohlcv([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])}
    outcome = run_replay_scenario(
        data=data,
        scenario=_scenario(),
        slo_gates=ReplaySLOGates(
            max_drawdown=0.50,
            max_rejection_rate=1.0,
            max_avg_slippage_bps=500.0,
            max_risk_escalations=5,
            max_escalation_latency_ms=10_000.0,
            min_orders_for_gate=1,
        ),
    )

    assert outcome.passed is True
    assert outcome.violations == []
    assert outcome.bars_processed > 0
    assert outcome.execution_slo["orders_submitted"] > 0
    assert "risk_playbook" in outcome.execution_slo


def test_replay_scenario_fails_drawdown_gate_when_shorting_disabled():
    data = {"AAPL": _ohlcv([100.0, 110.0, 80.0, 70.0, 65.0, 60.0])}
    outcome = run_replay_scenario(
        data=data,
        scenario=_scenario(allow_short=False, backtest_overrides={"max_position_pct": 0.90}),
        slo_gates=ReplaySLOGates(
            max_drawdown=0.01,
            max_rejection_rate=1.0,
            max_avg_slippage_bps=500.0,
            max_risk_escalations=5,
            max_escalation_latency_ms=10_000.0,
            min_orders_for_gate=1,
        ),
    )

    assert outcome.passed is False
    assert any("max_drawdown=" in violation for violation in outcome.violations)


def test_replay_scenario_raises_on_missing_ohlcv_columns():
    bad_data = {
        "AAPL": pd.DataFrame(
            {"close": [100.0, 101.0]},
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
        )
    }
    with pytest.raises(ValueError, match="missing columns"):
        run_replay_scenario(
            data=bad_data,
            scenario=_scenario(),
            slo_gates=ReplaySLOGates(),
        )


def test_replay_suite_aggregates_scenario_status():
    scenario = _scenario()

    def _provider(_scenario_input: ReplayScenario) -> dict[str, pd.DataFrame]:
        return {"AAPL": _ohlcv([100.0, 101.0, 102.0, 103.0, 104.0])}

    report = run_replay_suite(
        scenarios=[scenario],
        data_provider=_provider,
        slo_gates=ReplaySLOGates(
            max_drawdown=0.50,
            max_rejection_rate=1.0,
            max_avg_slippage_bps=500.0,
            max_risk_escalations=5,
            max_escalation_latency_ms=10_000.0,
            min_orders_for_gate=1,
        ),
    )

    assert report.passed is True
    payload = report.to_dict()
    assert payload["passed"] is True
    assert payload["total_scenarios"] == 1
