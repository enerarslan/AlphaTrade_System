"""Unit tests for training feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_trading_system.models.feature_selection import (
    FeatureSelectionConfig,
    compute_information_coefficients,
    select_training_features,
)


def test_compute_information_coefficients_ranks_signal_above_noise() -> None:
    rng = np.random.default_rng(42)
    n = 240
    signal = rng.normal(size=n)
    label = (signal + rng.normal(scale=0.1, size=n) > 0.0).astype(float)
    features = pd.DataFrame(
        {
            "signal_feature": signal,
            "noise_feature": rng.normal(size=n),
        }
    )

    scores = compute_information_coefficients(features, label)

    assert scores.index[0] == "signal_feature"
    assert float(scores["signal_feature"]) > float(scores["noise_feature"])


def test_select_training_features_prunes_highly_correlated_duplicate() -> None:
    rng = np.random.default_rng(7)
    n = 260
    signal = rng.normal(size=n)
    duplicate = signal * 0.999 + rng.normal(scale=1e-3, size=n)
    label = (signal > 0.0).astype(float)
    features = pd.DataFrame(
        {
            "signal_feature": signal,
            "duplicate_feature": duplicate,
            "noise_feature": rng.normal(size=n),
        }
    )

    result = select_training_features(
        features,
        label,
        FeatureSelectionConfig(
            min_information_coefficient=0.01,
            max_correlation=0.90,
            max_features=5,
            stability_iterations=4,
            min_stability_support=0.0,
        ),
    )

    assert "signal_feature" in result.selected_features
    assert "duplicate_feature" in result.correlation_pruned_features
    assert "duplicate_feature" not in result.selected_features
