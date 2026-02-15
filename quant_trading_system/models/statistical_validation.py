"""Statistical validation helpers for model promotion gates."""

from __future__ import annotations

from itertools import combinations
from typing import Literal, overload

import numpy as np
from scipy import stats

from quant_trading_system.models.purged_cv import MultipleTestingCorrector


def calculate_deflated_sharpe_ratio(
    observed_sharpe: float,
    returns: np.ndarray,
    n_trials: int,
) -> tuple[float, float]:
    """Calculate deflated Sharpe ratio and p-value."""
    arr = np.asarray(returns, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size < 10:
        return float(observed_sharpe), 0.5

    skewness = float(stats.skew(arr)) if arr.size > 2 else 0.0
    kurtosis = float(stats.kurtosis(arr)) + 3.0 if arr.size > 3 else 3.0

    corrector = MultipleTestingCorrector(n_trials=max(1, int(n_trials)))
    dsr, p_value = corrector.deflated_sharpe_ratio(
        observed_sharpe=float(observed_sharpe),
        n_trials=max(1, int(n_trials)),
        skewness=skewness,
        kurtosis=kurtosis,
        n_returns=int(arr.size),
    )
    return float(dsr), float(p_value)


@overload
def calculate_probability_of_backtest_overfitting(
    returns: np.ndarray,
    n_partitions: int = 16,
    max_combinations: int = 100,
    random_seed: int = 42,
    annualization_factor: float = 252.0,
    return_diagnostics: Literal[False] = False,
) -> tuple[float, str]:
    ...


@overload
def calculate_probability_of_backtest_overfitting(
    returns: np.ndarray,
    n_partitions: int = 16,
    max_combinations: int = 100,
    random_seed: int = 42,
    annualization_factor: float = 252.0,
    return_diagnostics: Literal[True] = True,
) -> tuple[float, str, dict[str, float]]:
    ...


def calculate_probability_of_backtest_overfitting(
    returns: np.ndarray,
    n_partitions: int = 16,
    max_combinations: int = 100,
    random_seed: int = 42,
    annualization_factor: float = 252.0,
    return_diagnostics: bool = False,
) -> tuple[float, str] | tuple[float, str, dict[str, float]]:
    """Estimate Probability of Backtest Overfitting (PBO) from return path."""
    arr = np.asarray(returns, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    n = arr.size
    diagnostics: dict[str, float] = {
        "pbo_ci_lower_95": 0.0,
        "pbo_ci_upper_95": 1.0,
        "pbo_probability_std": 0.5,
        "pbo_probability_stability": 0.0,
        "pbo_reliability": 0.0,
        "pbo_combinations_used": 0.0,
        "pbo_partition_size": 0.0,
    }

    if n < n_partitions * 5:
        result = (0.5, "Insufficient data for PBO calculation")
        return (*result, diagnostics) if return_diagnostics else result

    n_partitions = int(n_partitions)
    n_partitions = n_partitions if n_partitions % 2 == 0 else n_partitions - 1
    if n_partitions < 4:
        result = (0.5, "Too few partitions for PBO")
        return (*result, diagnostics) if return_diagnostics else result

    partition_size = n // n_partitions
    if partition_size <= 1:
        result = (0.5, "Too few observations per partition")
        return (*result, diagnostics) if return_diagnostics else result

    partitions = [
        arr[i * partition_size:(i + 1) * partition_size]
        for i in range(n_partitions)
    ]

    half = n_partitions // 2
    train_combos = list(combinations(range(n_partitions), half))
    if len(train_combos) > max_combinations:
        rng = np.random.default_rng(random_seed)
        choice = rng.choice(len(train_combos), max_combinations, replace=False)
        train_combos = [train_combos[i] for i in choice]

    overfit_probabilities: list[float] = []
    ann = np.sqrt(float(annualization_factor))
    diagnostics["pbo_combinations_used"] = float(len(train_combos))
    diagnostics["pbo_partition_size"] = float(partition_size)

    for train_indices in train_combos:
        test_indices = [i for i in range(n_partitions) if i not in train_indices]
        train_returns = np.concatenate([partitions[i] for i in train_indices])
        test_returns = np.concatenate([partitions[i] for i in test_indices])

        train_std = float(np.std(train_returns))
        test_std = float(np.std(test_returns))
        train_sharpe = float(np.mean(train_returns) / train_std * ann) if train_std > 1e-12 else 0.0
        test_sharpe = float(np.mean(test_returns) / test_std * ann) if test_std > 1e-12 else 0.0

        if train_sharpe <= 0.05:
            continue

        # Map train-vs-test Sharpe generalization gap into a continuous overfitting probability.
        gap = float(train_sharpe - test_sharpe)
        scale = float(max(0.35, abs(train_sharpe)))
        gap_score = float(np.clip(gap / scale, -4.0, 4.0))
        prob_overfit = float(1.0 / (1.0 + np.exp(-gap_score)))
        if test_sharpe < 0.0:
            prob_overfit = float(max(prob_overfit, 0.70))
        overfit_probabilities.append(float(np.clip(prob_overfit, 0.01, 0.99)))

    if not overfit_probabilities:
        result = (0.5, "Could not compute PBO probabilities")
        return (*result, diagnostics) if return_diagnostics else result

    probability_array = np.asarray(overfit_probabilities, dtype=float)
    pbo = float(np.mean(probability_array))
    pbo_std = float(np.std(probability_array, ddof=1)) if probability_array.size > 1 else 0.0
    ci_half_width = (
        float(1.96 * pbo_std / np.sqrt(float(probability_array.size)))
        if probability_array.size > 1
        else 0.0
    )
    pbo_ci_lower = float(np.clip(pbo - ci_half_width, 0.0, 1.0))
    pbo_ci_upper = float(np.clip(pbo + ci_half_width, 0.0, 1.0))
    stability = float(np.clip(1.0 - (pbo_std / 0.35), 0.0, 1.0))
    partition_reliability = float(np.clip((float(partition_size) - 8.0) / 32.0, 0.0, 1.0))
    combo_reliability = float(np.clip((float(probability_array.size) - 12.0) / 48.0, 0.0, 1.0))
    pbo_reliability = float(np.clip((0.55 * combo_reliability) + (0.45 * partition_reliability), 0.0, 1.0))

    diagnostics["pbo_ci_lower_95"] = pbo_ci_lower
    diagnostics["pbo_ci_upper_95"] = pbo_ci_upper
    diagnostics["pbo_probability_std"] = pbo_std
    diagnostics["pbo_probability_stability"] = stability
    diagnostics["pbo_reliability"] = pbo_reliability

    if pbo < 0.10:
        interpretation = "Very low overfitting risk"
    elif pbo < 0.25:
        interpretation = "Low overfitting risk"
    elif pbo < 0.45:
        interpretation = "Moderate overfitting risk"
    elif pbo < 0.70:
        interpretation = "High overfitting risk - exercise caution"
    else:
        interpretation = "Very high overfitting risk - likely overfit"

    if pbo_reliability < 0.35:
        interpretation = f"{interpretation} (low statistical confidence)"

    result = (pbo, interpretation)
    return (*result, diagnostics) if return_diagnostics else result
