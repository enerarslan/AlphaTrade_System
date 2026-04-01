"""Regression tests for scripts/trade.py."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

import scripts.trade as trade_script
from quant_trading_system.core.data_types import Direction, OHLCVBar
from quant_trading_system.trading.portfolio_manager import RebalanceMethod


def test_configure_engine_uses_promotion_package_runtime_settings(monkeypatch):
    engine = SimpleNamespace(
        config=SimpleNamespace(symbols=[], kill_switch_drawdown=0.0, bar_interval="1Min"),
        portfolio_manager=SimpleNamespace(
            position_sizer=SimpleNamespace(
                config=SimpleNamespace(
                    percent_of_equity=0.0,
                    max_position_pct=0.0,
                    max_total_positions=0,
                    confidence_position_sizing=False,
                    min_confidence_position_scale=0.0,
                )
            ),
            rebalance_config=SimpleNamespace(method=None),
        ),
        signal_generator=MagicMock(),
    )
    monkeypatch.setattr(
        trade_script,
        "get_settings",
        lambda: SimpleNamespace(alpaca=SimpleNamespace(api_key="key", api_secret="secret")),
    )
    monkeypatch.setattr(
        trade_script,
        "create_trading_engine",
        lambda mode, symbols, api_key, api_secret: engine,
    )
    source_sentinel = SimpleNamespace(generate=MagicMock())
    monkeypatch.setattr(
        trade_script,
        "PromotionPackageSignalSource",
        lambda contract: source_sentinel,
    )

    args = SimpleNamespace(
        mode="paper",
        symbols=list(trade_script.DEFAULT_TRADE_SYMBOLS),
        kill_switch_drawdown=0.08,
        strategy="momentum",
        max_positions=10,
    )
    contract = SimpleNamespace(
        symbols=("AAPL", "MSFT"),
        timeframe="15Min",
        max_position_pct=0.12,
        max_total_positions=6,
        confidence_position_sizing=True,
        min_confidence_position_scale=0.25,
        model_name="pkg_model",
    )

    configured_engine, symbols = trade_script._configure_engine(args, contract)

    assert configured_engine is engine
    assert symbols == ["AAPL", "MSFT"]
    assert engine.config.symbols == ["AAPL", "MSFT"]
    assert engine.config.kill_switch_drawdown == 0.08
    assert engine.config.bar_interval == "15Min"
    assert engine.portfolio_manager.position_sizer.config.percent_of_equity == 0.12
    assert engine.portfolio_manager.position_sizer.config.max_position_pct == 0.12
    assert engine.portfolio_manager.position_sizer.config.max_total_positions == 6
    assert engine.portfolio_manager.position_sizer.config.confidence_position_sizing is True
    assert (
        engine.portfolio_manager.position_sizer.config.min_confidence_position_scale == 0.25
    )
    assert engine.portfolio_manager.rebalance_config.method == RebalanceMethod.SIGNAL_DRIVEN
    engine.signal_generator.add_external_source.assert_called_once_with(
        "promotion_package:pkg_model",
        source_sentinel.generate,
    )


def test_promotion_package_signal_source_dedupes_bars_and_sets_hold_contract_metadata(
    monkeypatch,
):
    timestamp = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)

    class _Adapter:
        def __init__(self, contract, use_gpu=False, logger_=None):
            self.contract = contract

        def compute_features(self, data):
            frame = next(iter(data.values()))
            return {"AAPL": pd.DataFrame({"timestamp": frame.index, "feat_a": [0.8]})}

        def generate_signal_frames(self, data, features=None):
            del data, features
            return {
                "AAPL": pd.DataFrame(
                    {
                        "timestamp": [timestamp],
                        "signal": [0.8],
                        "confidence": [0.9],
                        "horizon": [3],
                        "model_source": ["promotion_package:pkg_model"],
                        "probability": [0.82],
                        "raw_prediction": [0.67],
                        "meta_confidence": [0.88],
                        "max_holding_bars": [4],
                        "take_profit_pct": [0.05],
                        "stop_loss_pct": [0.02],
                        "universe_eligible": [True],
                        "universe_quality_score": [0.91],
                        "long_side_policy_scale": [1.05],
                        "expected_edge": [0.012],
                        "edge_pass_probability": [0.9],
                        "edge_policy_scale": [0.5],
                    }
                )
            }

    monkeypatch.setattr(trade_script, "PromotionSignalAdapter", _Adapter)

    source = trade_script.PromotionPackageSignalSource(
        SimpleNamespace(
            symbols=("AAPL",),
            model_source="promotion_package:pkg_model",
            model_name="pkg_model",
        )
    )
    bar = OHLCVBar(
        symbol="AAPL",
        timestamp=timestamp,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=1000,
    )

    signals = source.generate({"AAPL": bar})

    assert len(signals) == 1
    signal = signals[0]
    assert signal.direction == Direction.LONG
    assert signal.timestamp == timestamp
    assert signal.metadata["hold_contract_enabled"] is True
    assert signal.metadata["prediction_horizon_bars"] == 3
    assert signal.metadata["max_holding_bars"] == 4
    assert signal.metadata["take_profit_pct"] == 0.05
    assert signal.metadata["stop_loss_pct"] == 0.02
    assert signal.metadata["probability"] == 0.82
    assert signal.metadata["raw_prediction"] == 0.67
    assert signal.metadata["meta_confidence"] == 0.88
    assert signal.metadata["universe_eligible"] is True
    assert signal.metadata["universe_quality_score"] == 0.91
    assert signal.metadata["long_side_policy_scale"] == 1.05
    assert signal.metadata["expected_edge"] == 0.012
    assert signal.metadata["edge_pass_probability"] == 0.9
    assert signal.metadata["edge_policy_scale"] == 0.5

    assert source.generate({"AAPL": bar}) == []
