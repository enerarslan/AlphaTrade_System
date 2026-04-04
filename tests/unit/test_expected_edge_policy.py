from __future__ import annotations

import numpy as np
import pandas as pd

from quant_trading_system.models.expected_edge_policy import (
    ExpectedEdgePolicyConfig,
    ExpectedEdgePolicyModel,
    build_expected_edge_feature_frame,
)


def test_build_expected_edge_feature_frame_adds_symbol_prior_columns() -> None:
    frame = pd.DataFrame({"feat_a": [0.72, 0.72], "flow_imbalance": [0.1, 0.1]})
    symbol_priors = {
        "__GLOBAL__": {
            "edge_prior": 0.001,
            "pass_prior": 0.55,
            "tail_prior": -0.002,
            "downside_prior": 0.45,
            "sample_confidence": 0.0,
        },
        "AAPL": {
            "edge_prior": 0.006,
            "pass_prior": 0.72,
            "tail_prior": 0.001,
            "downside_prior": 0.28,
            "sample_confidence": 0.80,
        },
    }

    policy_frame = build_expected_edge_feature_frame(
        frame,
        probabilities=np.array([0.72, 0.72], dtype=float),
        long_threshold=0.60,
        short_threshold=0.40,
        symbols=np.array(["AAPL", "UNKNOWN"], dtype=object),
        symbol_priors=symbol_priors,
    )

    assert policy_frame["policy_symbol_edge_prior"].tolist() == [0.006, 0.001]
    assert policy_frame["policy_symbol_pass_prior"].tolist() == [0.72, 0.55]
    assert policy_frame["policy_symbol_tail_prior"].tolist() == [0.001, -0.002]
    assert policy_frame["policy_symbol_downside_prior"].tolist() == [0.28, 0.45]
    assert policy_frame["policy_symbol_sample_confidence"].tolist() == [0.80, 0.0]


def test_expected_edge_policy_uses_symbol_priors_for_identical_contexts() -> None:
    rows_per_symbol = 80
    symbols = np.array(
        ["AAPL"] * rows_per_symbol + ["MSFT"] * rows_per_symbol,
        dtype=object,
    )
    feature_frame = pd.DataFrame(
        {
            "feat_a": np.full(rows_per_symbol * 2, 0.72, dtype=float),
            "flow_imbalance_signal": np.zeros(rows_per_symbol * 2, dtype=float),
        }
    )
    probabilities = np.full(rows_per_symbol * 2, 0.72, dtype=float)
    trade_returns = np.concatenate(
        [
            np.full(rows_per_symbol, 0.018, dtype=float),
            np.full(rows_per_symbol, -0.018, dtype=float),
        ]
    )

    model = ExpectedEdgePolicyModel(
        ExpectedEdgePolicyConfig(
            min_samples=20,
            min_coverage=0.50,
            max_context_features=4,
            random_state=7,
        )
    )
    summary = model.fit(
        feature_frame,
        probabilities=probabilities,
        trade_returns=trade_returns,
        long_threshold=0.60,
        short_threshold=0.40,
        symbols=symbols,
    )

    result = model.predict_policy(
        pd.DataFrame(
            {
                "feat_a": [0.72, 0.72],
                "flow_imbalance_signal": [0.0, 0.0],
            }
        ),
        probabilities=np.array([0.72, 0.72], dtype=float),
        long_threshold=0.60,
        short_threshold=0.40,
        symbols=np.array(["AAPL", "MSFT"], dtype=object),
    )

    assert summary["symbol_prior_summary"]["AAPL"]["edge_prior"] > 0.0
    assert summary["symbol_prior_summary"]["MSFT"]["edge_prior"] < 0.0
    assert result.loc[0, "expected_edge"] > result.loc[1, "expected_edge"]
    assert result.loc[0, "edge_pass_probability"] > result.loc[1, "edge_pass_probability"]
    assert bool(result.loc[0, "edge_policy_pass"]) is True
    assert bool(result.loc[1, "edge_policy_pass"]) is False
