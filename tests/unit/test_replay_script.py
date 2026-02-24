"""Unit tests for scripts/replay.py."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from quant_trading_system.backtest.replay import ReplayOutcome
from scripts import replay as replay_script


def _args(tmp_path: Path, **overrides):
    base = {
        "scenario_id": "script_replay",
        "start": "2024-01-01",
        "end": "2024-01-10",
        "symbols": ["AAPL"],
        "capital": 100000.0,
        "execution_mode": "realistic",
        "slippage_bps": 5.0,
        "commission_bps": 1.0,
        "return_threshold_bps": 5.0,
        "signal_confidence": 0.75,
        "signal_horizon": 1,
        "allow_short": True,
        "max_drawdown": 0.20,
        "max_rejection_rate": 0.20,
        "max_avg_slippage_bps": 35.0,
        "max_risk_escalations": 0,
        "max_escalation_latency_ms": 2000.0,
        "min_orders_for_slo": 1,
        "no_fail_on_kill_switch": False,
        "use_database": True,
        "no_database": True,
        "data_dir": Path("data/raw"),
        "output": tmp_path / "replay_result.json",
        "format": "json",
        "fail_on_slo_breach": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _outcome(passed: bool) -> ReplayOutcome:
    return ReplayOutcome(
        scenario_id="script_replay",
        passed=passed,
        violations=[] if passed else ["max_drawdown=0.3000 > allowed=0.2000"],
        bars_processed=10,
        trades_closed=4,
        total_return=0.05,
        max_drawdown=0.03,
        execution_slo={"orders_submitted": 8, "rejection_rate": 0.0, "avg_slippage_bps": 2.0},
        started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        completed_at=datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
    )


def test_run_replay_returns_zero_and_writes_output(monkeypatch, tmp_path):
    monkeypatch.setattr(replay_script, "_load_replay_data", lambda **_: {"AAPL": object()})
    monkeypatch.setattr(replay_script, "run_replay_scenario", lambda **_: _outcome(True))

    args = _args(tmp_path)
    code = replay_script.run_replay(args)

    assert code == 0
    payload = json.loads(args.output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["scenario_id"] == "script_replay"


def test_run_replay_returns_two_when_slo_breach(monkeypatch, tmp_path):
    monkeypatch.setattr(replay_script, "_load_replay_data", lambda **_: {"AAPL": object()})
    monkeypatch.setattr(replay_script, "run_replay_scenario", lambda **_: _outcome(False))

    args = _args(tmp_path, fail_on_slo_breach=True, format="text")
    code = replay_script.run_replay(args)

    assert code == 2


def test_run_replay_returns_one_on_bad_dates(tmp_path):
    args = _args(tmp_path, start="2024-99-99")
    code = replay_script.run_replay(args)
    assert code == 1
