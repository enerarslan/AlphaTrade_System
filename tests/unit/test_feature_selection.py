"""Unit tests for training feature selection."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from quant_trading_system.models.feature_selection import (
    FeatureSelectionConfig,
    compute_information_coefficients,
    select_training_features,
)
import quant_trading_system.models.feature_selection as feature_selection_module


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


def test_compute_information_coefficients_skips_constant_inputs_without_warning() -> None:
    features = pd.DataFrame(
        {
            "constant_feature": np.ones(64, dtype=float),
            "signal_feature": np.linspace(-1.0, 1.0, 64),
        }
    )
    target = np.linspace(-0.5, 0.5, 64)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scores = compute_information_coefficients(features, target)

    assert float(scores["constant_feature"]) == 0.0
    assert float(scores["signal_feature"]) > 0.0


def test_compute_information_coefficients_supports_group_aware_rank_ic() -> None:
    rng = np.random.default_rng(123)
    group_size = 6
    group_count = 48
    groups = np.repeat(np.arange(group_count), group_size)
    signal = rng.normal(size=group_size * group_count)
    noise = rng.normal(size=group_size * group_count)
    target = signal + rng.normal(scale=0.15, size=group_size * group_count)
    features = pd.DataFrame(
        {
            "signal_feature": signal,
            "noise_feature": noise,
        }
    )

    scores = compute_information_coefficients(
        features,
        target,
        groups=groups,
        min_group_size=4,
    )

    assert scores.index[0] == "signal_feature"
    assert float(scores["signal_feature"]) > float(scores["noise_feature"])


def test_group_aware_information_coefficients_match_reference_with_missing_values() -> None:
    rng = np.random.default_rng(9)
    group_size = 5
    group_count = 24
    groups = np.repeat(np.arange(group_count), group_size)
    base_signal = rng.normal(size=group_size * group_count)
    features = pd.DataFrame(
        {
            "signal_feature": base_signal,
            "partially_missing_feature": base_signal * 0.6 + rng.normal(scale=0.2, size=base_signal.size),
            "noise_feature": rng.normal(size=base_signal.size),
        }
    )
    features.loc[features.index % 7 == 0, "partially_missing_feature"] = np.nan
    target = base_signal + rng.normal(scale=0.25, size=base_signal.size)

    scores = compute_information_coefficients(
        features,
        target,
        groups=groups,
        min_group_size=3,
    )

    numeric = feature_selection_module._finite_feature_matrix(features)
    label = pd.Series(pd.to_numeric(target, errors="coerce"), index=numeric.index)
    group_series = pd.Series(groups, index=numeric.index, dtype="object")
    reference_scores: dict[str, float] = {}
    for column in numeric.columns:
        values = numeric[column]
        valid = values.notna() & label.notna() & group_series.notna()
        if int(valid.sum()) < 20:
            reference_scores[str(column)] = 0.0
            continue
        reference_scores[str(column)] = feature_selection_module._groupwise_information_coefficient(
            values.loc[valid],
            label.loc[valid],
            group_series.loc[valid],
            min_group_size=3,
        )

    for feature_name, expected in reference_scores.items():
        assert float(scores[feature_name]) == pytest.approx(float(expected), abs=1e-12)


def test_select_training_features_emits_progress_callbacks() -> None:
    rng = np.random.default_rng(21)
    n = 180
    signal = rng.normal(size=n)
    features = pd.DataFrame(
        {
            "signal_feature": signal,
            "signal_feature_2": signal * 0.8 + rng.normal(scale=0.1, size=n),
            "noise_feature": rng.normal(size=n),
        }
    )
    target = (signal > 0.0).astype(float)
    events: list[str] = []

    result = select_training_features(
        features,
        target,
        FeatureSelectionConfig(
            min_information_coefficient=0.01,
            max_correlation=0.999,
            max_features=3,
            stability_iterations=3,
            min_stability_support=0.0,
        ),
        progress_callback=lambda stage, payload: events.append(stage),
    )

    assert result.selected_features
    assert "information_coefficient_start" in events
    assert "information_coefficient_progress" in events
    assert "information_coefficient_screen_complete" in events
    assert "correlation_prune_complete" in events
    assert "stability_selection_start" in events
    assert "stability_selection_progress" in events
    assert "stability_selection_complete" in events
    assert "selection_complete" in events
