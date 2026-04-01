"""Unit tests for scripts/backtest.py regression paths."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_trading_system.backtest.engine import BacktestState, PandasDataHandler, Trade
from quant_trading_system.backtest.promotion import PromotionSignalAdapter, load_promotion_package
from quant_trading_system.backtest.performance_attribution import PerformanceAttributionService
from quant_trading_system.core.data_types import Direction, OrderSide, Portfolio
from scripts.backtest import BacktestRunner, BacktestSession, SignalBasedStrategy, run_backtest


class DummyProbabilityModel:
    feature_names = ["feat_a", "feat_b"]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        prob = np.clip(X[:, 0], 0.0, 1.0)
        return np.column_stack([1.0 - prob, prob])


class DummyOffsetCalibrator:
    def predict(self, values):
        values = np.asarray(values, dtype=float)
        return np.clip(values + 0.15, 0.0, 1.0)


class DummyMetaModel:
    def filter_signals(self, X, signals, prices=None, threshold=None):
        threshold = 0.60 if threshold is None else float(threshold)
        signals = pd.Series(signals, dtype=float)
        confidence = pd.Series(np.where(signals != 0.0, 0.90, 0.10), dtype=float)
        filtered = signals.where(confidence >= threshold, 0.0)
        return pd.DataFrame(
            {
                "original_signal": signals,
                "confidence": confidence,
                "filtered_signal": filtered,
                "passed_filter": confidence >= threshold,
            }
        )


class DummyRankerModel:
    def predict(self, X):
        del X
        return np.array([-2.0, 0.0, 2.0], dtype=float)


def test_promotion_signal_adapter_uses_training_parity_for_ranker_scores():
    adapter = PromotionSignalAdapter(SimpleNamespace(model_type="lightgbm_ranker"))

    probabilities, raw_scores = adapter._predict_model_probabilities(
        DummyRankerModel(),
        np.zeros((3, 1), dtype=float),
    )

    expected = 1.0 / (1.0 + np.exp(-np.array([-2.0, 0.0, 2.0], dtype=float)))
    np.testing.assert_allclose(raw_scores, np.array([-2.0, 0.0, 2.0], dtype=float))
    np.testing.assert_allclose(probabilities, expected)


def test_promotion_signal_adapter_applies_attached_probability_calibration():
    adapter = PromotionSignalAdapter(SimpleNamespace(model_type="lightgbm"))
    model = DummyProbabilityModel()
    model._alphatrade_probability_calibration = {
        "method": "isotonic",
        "calibrator": DummyOffsetCalibrator(),
    }

    probabilities, raw_scores = adapter._predict_model_probabilities(
        model,
        np.array([[0.50, 0.0], [0.35, 0.0]], dtype=float),
    )

    np.testing.assert_allclose(raw_scores, np.array([0.50, 0.35], dtype=float))
    np.testing.assert_allclose(probabilities, np.array([0.65, 0.50], dtype=float))


def test_promotion_signal_adapter_computes_multi_timeframe_feature_layers():
    contract = SimpleNamespace(
        feature_groups=("statistical",),
        enable_cross_sectional=False,
        timeframe="15Min",
        timeframes=("15Min", "1Hour"),
        enable_reference_features=False,
        enable_tick_microstructure_features=False,
    )
    adapter = PromotionSignalAdapter(contract)
    timestamps = pd.date_range("2025-01-02T09:15:00Z", periods=12, freq="15min")
    frame = pd.DataFrame(
        {
            "open": np.arange(100.0, 112.0, dtype=float),
            "high": np.arange(101.0, 113.0, dtype=float),
            "low": np.arange(99.5, 111.5, dtype=float),
            "close": np.arange(100.5, 112.5, dtype=float),
            "volume": np.arange(10.0, 130.0, 10.0, dtype=float),
        },
        index=pd.DatetimeIndex(timestamps),
    )

    feature_frames = adapter.compute_features({"AAPL": frame})

    assert "AAPL" in feature_frames
    symbol_frame = feature_frames["AAPL"]
    assert "tf1h_close" in symbol_frame.columns
    assert "tf1h_bar_return" in symbol_frame.columns
    assert symbol_frame.loc[4, "tf1h_close"] == 103.5
    assert pd.notna(symbol_frame.loc[4, "tf1h_bar_return"])


def test_promotion_signal_adapter_applies_asymmetric_side_policy():
    adapter = PromotionSignalAdapter(
        SimpleNamespace(
            model_type="lightgbm",
            feature_names=("feat_a",),
            long_threshold=0.60,
            short_threshold=0.40,
            horizon_bars=5,
            model_source="promotion_package:test_model",
            meta_label_enabled=False,
            meta_label_threshold=None,
            take_profit_pct=0.03,
            stop_loss_pct=0.02,
            max_holding_bars=6,
            long_side_policy={
                "enabled": True,
                "signal_scale": 0.8,
                "confidence_scale": 0.9,
                "threshold_adjustment": 0.02,
            },
            short_side_policy={
                "enabled": False,
                "signal_scale": 0.0,
                "confidence_scale": 0.0,
                "threshold_adjustment": -0.03,
            },
            enable_universe_quality_gate=False,
            universe_quality_policy={},
        )
    )
    adapter._model = DummyProbabilityModel()
    timestamps = pd.to_datetime(
        ["2025-01-02T14:30:00Z", "2025-01-02T14:45:00Z"],
        utc=True,
    )
    features = {"AAPL": pd.DataFrame({"timestamp": timestamps, "feat_a": [0.65, 0.20]})}
    raw_frame = pd.DataFrame(
        {
            "open": [100.0, 100.5],
            "high": [101.0, 101.5],
            "low": [99.0, 99.5],
            "close": [100.5, 100.0],
            "volume": [1000.0, 1200.0],
        },
        index=pd.DatetimeIndex(timestamps),
    )
    signal_frames = adapter.generate_signal_frames({"AAPL": raw_frame}, features=features)

    signal_frame = signal_frames["AAPL"]
    assert signal_frame.loc[0, "long_threshold"] == pytest.approx(0.62)
    assert signal_frame.loc[0, "signal"] == pytest.approx(((0.65 - 0.62) / 0.38) * 0.8)
    assert signal_frame.loc[0, "confidence"] == pytest.approx(abs(0.65 - 0.5) * 2.0 * 0.9)
    assert signal_frame.loc[1, "short_threshold"] == pytest.approx(0.37)
    assert signal_frame.loc[1, "signal"] == pytest.approx(0.0)
    assert signal_frame.loc[1, "confidence"] == pytest.approx(0.0)


def test_promotion_signal_adapter_blocks_ineligible_universe_symbol():
    adapter = PromotionSignalAdapter(
        SimpleNamespace(
            model_type="lightgbm",
            feature_names=("feat_a",),
            long_threshold=0.60,
            short_threshold=0.40,
            horizon_bars=5,
            model_source="promotion_package:test_model",
            meta_label_enabled=False,
            meta_label_threshold=None,
            take_profit_pct=0.03,
            stop_loss_pct=0.02,
            max_holding_bars=6,
            long_side_policy={},
            short_side_policy={},
            enable_universe_quality_gate=True,
            universe_quality_policy={
                "enabled": True,
                "min_rows": 3,
                "max_missing_ratio": 1.0,
                "max_extreme_move_ratio": 1.0,
                "max_corporate_action_ratio": 1.0,
                "min_median_dollar_volume": 1_000_000.0,
                "selected_symbols": ["AAPL"],
            },
            symbols=("AAPL",),
            raw_payload={},
        )
    )
    adapter._model = DummyProbabilityModel()
    timestamps = pd.to_datetime(
        ["2025-01-02T14:30:00Z", "2025-01-02T14:45:00Z"],
        utc=True,
    )
    features = {"AAPL": pd.DataFrame({"timestamp": timestamps, "feat_a": [0.8, 0.8]})}
    raw_frame = pd.DataFrame(
        {
            "open": [100.0, 100.5],
            "high": [101.0, 101.5],
            "low": [99.0, 99.5],
            "close": [100.5, 101.0],
            "volume": [100.0, 120.0],
        },
        index=pd.DatetimeIndex(timestamps),
    )

    signal_frames = adapter.generate_signal_frames({"AAPL": raw_frame}, features=features)

    signal_frame = signal_frames["AAPL"]
    assert signal_frame["signal"].eq(0.0).all()
    assert signal_frame["confidence"].eq(0.0).all()
    assert bool(signal_frame.loc[0, "universe_eligible"]) is False
    assert "insufficient_rows" in signal_frame.loc[0, "universe_quality_reasons"]


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

    monkeypatch.setattr(
        backtest_script,
        "PerformanceAnalyzer",
        lambda *args, **kwargs: _Analyzer(),
    )

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


def test_signal_based_strategy_prefers_confidence_and_horizon_columns():
    t0 = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)
    handler = PandasDataHandler(
        {
            "AAPL": pd.DataFrame(
                {
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.0],
                    "volume": [1_000_000],
                },
                index=pd.DatetimeIndex([t0]),
            )
        }
    )
    strategy = SignalBasedStrategy(
        {
            "AAPL": pd.DataFrame(
                {
                    "signal": [0.75],
                    "confidence": [0.88],
                    "horizon": [5],
                    "model_source": ["promotion_package:test_model"],
                    "probability": [0.87],
                },
                index=pd.DatetimeIndex([t0]),
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
    signals = strategy.generate_signals(handler, portfolio)

    assert len(signals) == 1
    assert signals[0].confidence == 0.88
    assert signals[0].horizon == 5
    assert signals[0].model_source == "promotion_package:test_model"
    assert signals[0].metadata["probability"] == 0.87


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


def test_load_promotion_package_uses_artifacts_fallback(tmp_path: Path):
    model_path = tmp_path / "test_model.pkl"
    artifacts_path = tmp_path / "test_model_artifacts.json"
    package_path = tmp_path / "test_model.promotion_package.json"

    with model_path.open("wb") as handle:
        pickle.dump(DummyProbabilityModel(), handle)

    artifacts_payload = {
        "feature_names": ["feat_a", "feat_b"],
        "training_metrics": {
            "holdout_long_threshold": 0.65,
            "holdout_short_threshold": 0.35,
            "meta_label_min_confidence": 0.70,
        },
        "snapshot_manifest": {"feature_schema_version": "schema-123"},
    }
    artifacts_path.write_text(json.dumps(artifacts_payload), encoding="utf-8")

    package_payload = {
        "schema_version": "1.0.0",
        "model_name": "test_model",
        "model_type": "lightgbm",
        "model_path": str(model_path),
        "signal_policy": {
            "long_side_policy": {"enabled": True, "signal_scale": 1.1},
            "short_side_policy": {"enabled": False, "signal_scale": 0.0},
        },
        "universe_quality_policy": {
            "enabled": True,
            "min_rows": 900,
            "selected_symbols": ["AAPL"],
        },
        "training_config": {
            "symbols": ["AAPL"],
            "timeframe": "15Min",
            "timeframes": ["15Min", "1Hour"],
            "feature_groups": ["technical", "statistical"],
            "meta_label_min_confidence": 0.66,
        },
        "position_sizing_policy": {
            "use_portfolio_target_sizing": True,
            "max_position_pct": 0.12,
            "max_total_positions": 8,
            "confidence_position_sizing": True,
            "min_confidence_position_scale": 0.15,
        },
    }
    package_path.write_text(json.dumps(package_payload), encoding="utf-8")

    package = load_promotion_package(package_path)

    assert package.feature_names == ("feat_a", "feat_b")
    assert package.long_threshold == 0.65
    assert package.short_threshold == 0.35
    assert package.meta_label_threshold == 0.70
    assert package.feature_schema_version == "schema-123"
    assert package.timeframes == ("15Min", "1Hour")
    assert package.max_position_pct == 0.12
    assert package.max_total_positions == 8
    assert package.min_confidence_position_scale == 0.15
    assert package.long_side_policy["signal_scale"] == pytest.approx(1.1)
    assert package.short_side_policy["enabled"] is False
    assert package.enable_universe_quality_gate is True
    assert package.universe_quality_policy["min_rows"] == 900


def test_backtest_runner_generates_signals_from_promotion_package(tmp_path: Path):
    model_path = tmp_path / "pkg_model.pkl"
    meta_model_path = tmp_path / "pkg_model_meta.pkl"
    package_path = tmp_path / "pkg_model.promotion_package.json"

    with model_path.open("wb") as handle:
        pickle.dump(DummyProbabilityModel(), handle)
    with meta_model_path.open("wb") as handle:
        pickle.dump(DummyMetaModel(), handle)

    package_payload = {
        "schema_version": "1.1.0",
        "model_name": "pkg_model",
        "model_type": "lightgbm",
        "model_path": str(model_path),
        "meta_model_path": str(meta_model_path),
        "feature_contract": {
            "selected_features": ["feat_a", "feat_b"],
            "feature_groups": ["technical", "statistical"],
        },
        "signal_policy": {
            "long_threshold": 0.60,
            "short_threshold": 0.40,
            "horizon_bars": 5,
            "meta_label_enabled": True,
            "meta_label_threshold": 0.60,
            "model_source": "promotion_package:pkg_model",
        },
        "position_sizing_policy": {
            "use_portfolio_target_sizing": True,
            "max_position_pct": 0.12,
            "max_total_positions": 6,
            "confidence_position_sizing": True,
            "min_confidence_position_scale": 0.20,
        },
        "training_config": {
            "symbols": ["AAPL"],
            "timeframe": "15Min",
            "timeframes": ["15Min"],
        },
    }
    package_path.write_text(json.dumps(package_payload), encoding="utf-8")

    t0 = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=15)
    session = BacktestSession(
        start_date=t0,
        end_date=t1,
        symbols=["AAPL"],
        initial_capital=Decimal("100000"),
        timeframe="15Min",
        promotion_package_path=package_path,
    )
    runner = BacktestRunner(session)
    assert runner.session.max_position_pct == 0.12
    assert runner.session.max_total_positions == 6
    assert runner.session.min_confidence_position_scale == 0.20
    runner.data["AAPL"] = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1000, 1000],
        },
        index=pd.DatetimeIndex([t0, t1]),
    )
    runner.features["AAPL"] = pd.DataFrame(
        {
            "timestamp": [t0, t1],
            "feat_a": [0.80, 0.20],
            "feat_b": [0.10, 0.20],
        }
    )

    runner._generate_signals()

    signal_frame = runner.signals["AAPL"]
    assert list(signal_frame["horizon"]) == [5, 5]
    assert list(signal_frame["model_source"]) == ["promotion_package:pkg_model"] * 2
    assert signal_frame["signal"].iloc[0] > 0.0
    assert signal_frame["signal"].iloc[1] < 0.0
    assert signal_frame["confidence"].iloc[0] == 0.90
    assert signal_frame["confidence"].iloc[1] == 0.90


def test_backtest_runner_uses_timeframe_aware_annualization(monkeypatch):
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
        def __init__(self, periods_per_year):
            captured["periods_per_year"] = periods_per_year

        def analyze(self, _result, benchmark_returns=None):
            return _Report()

    monkeypatch.setattr(
        backtest_script,
        "PerformanceAnalyzer",
        lambda periods_per_year=252, risk_free_rate=0.02: _Analyzer(periods_per_year),
    )

    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=15)
    session = BacktestSession(
        start_date=t0,
        end_date=t1,
        symbols=["AAPL"],
        initial_capital=Decimal("100000"),
        timeframe="15Min",
    )
    runner = BacktestRunner(session)
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

    assert captured["periods_per_year"] == 26 * 252
