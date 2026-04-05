"""
Training-time feature selection for institutional model governance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(slots=True)
class FeatureSelectionConfig:
    min_information_coefficient: float = 0.01
    max_correlation: float = 0.95
    max_features: int = 250
    stability_iterations: int = 16
    stability_subsample_ratio: float = 0.65
    min_stability_support: float = 0.55
    group_aware_ic: bool = False
    min_group_size: int = 3
    random_state: int = 42


@dataclass(slots=True)
class FeatureSelectionResult:
    selected_features: list[str]
    information_coefficients: dict[str, float]
    correlation_pruned_features: list[str]
    stability_scores: dict[str, float]
    diagnostics: dict[str, float | int]


FeatureSelectionProgressCallback = Callable[[str, dict[str, Any]], None]


def _finite_feature_matrix(
    features: pd.DataFrame,
) -> pd.DataFrame:
    working = features.copy()
    working = working.replace([np.inf, -np.inf], np.nan)
    numeric = working.apply(pd.to_numeric, errors="coerce")
    return numeric


def _is_effectively_constant(values: pd.Series | np.ndarray) -> bool:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size <= 1:
        return True
    return bool((np.nanmax(finite) - np.nanmin(finite)) <= 1e-12)


def _groupwise_information_coefficient(
    values: pd.Series,
    label: pd.Series,
    groups: pd.Series,
    *,
    min_group_size: int,
) -> float:
    working = pd.DataFrame(
        {
            "feature": pd.to_numeric(values, errors="coerce"),
            "label": pd.to_numeric(label, errors="coerce"),
            "group": groups,
        }
    ).dropna(subset=["feature", "label", "group"])
    if len(working) < 20:
        return 0.0

    group_scores: list[float] = []
    group_weights: list[int] = []
    for _, group_frame in working.groupby("group", sort=False):
        if len(group_frame) < max(3, int(min_group_size)):
            continue
        feature_values = group_frame["feature"].to_numpy(dtype=float)
        label_values = group_frame["label"].to_numpy(dtype=float)
        if _is_effectively_constant(feature_values) or _is_effectively_constant(label_values):
            continue
        ic = stats.spearmanr(feature_values, label_values, nan_policy="omit")[0]
        if np.isfinite(ic):
            group_scores.append(abs(float(ic)))
            group_weights.append(int(len(group_frame)))

    if not group_scores:
        return 0.0
    return float(np.average(group_scores, weights=group_weights))


def compute_information_coefficients(
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    *,
    groups: pd.Series | np.ndarray | None = None,
    min_group_size: int = 3,
    progress_callback: FeatureSelectionProgressCallback | None = None,
) -> pd.Series:
    numeric = _finite_feature_matrix(features)
    label = pd.Series(pd.to_numeric(target, errors="coerce"), index=numeric.index)
    group_series = (
        pd.Series(groups, index=numeric.index, dtype="object") if groups is not None else None
    )
    scores: dict[str, float] = {}
    total_features = int(len(numeric.columns))
    progress_step = max(1, total_features // 6) if total_features > 0 else 1
    if progress_callback is not None:
        progress_callback(
            "information_coefficient_start",
            {
                "feature_count": total_features,
                "group_aware": bool(group_series is not None),
            },
        )
    for idx, column in enumerate(numeric.columns, start=1):
        values = numeric[column]
        valid = values.notna() & label.notna()
        if group_series is not None:
            valid = valid & group_series.notna()
        if int(valid.sum()) < 20:
            scores[column] = 0.0
        else:
            if group_series is not None:
                ic_value = _groupwise_information_coefficient(
                    values.loc[valid],
                    label.loc[valid],
                    group_series.loc[valid],
                    min_group_size=min_group_size,
                )
            else:
                feature_values = values.loc[valid].to_numpy(dtype=float)
                label_values = label.loc[valid].to_numpy(dtype=float)
                if _is_effectively_constant(feature_values) or _is_effectively_constant(label_values):
                    scores[column] = 0.0
                    if progress_callback is not None and (
                        idx == total_features or idx % progress_step == 0
                    ):
                        progress_callback(
                            "information_coefficient_progress",
                            {
                                "processed_features": idx,
                                "total_features": total_features,
                            },
                        )
                    continue
                ic = stats.spearmanr(feature_values, label_values, nan_policy="omit")[0]
                ic_value = float(abs(ic)) if np.isfinite(ic) else 0.0
            scores[column] = float(ic_value)
        if progress_callback is not None and (idx == total_features or idx % progress_step == 0):
            progress_callback(
                "information_coefficient_progress",
                {
                    "processed_features": idx,
                    "total_features": total_features,
                },
            )
    ranked = pd.Series(scores, dtype=float).sort_values(ascending=False)
    if progress_callback is not None:
        progress_callback(
            "information_coefficient_complete",
            {
                "ranked_feature_count": int(len(ranked)),
            },
        )
    return ranked


def _prune_correlated_features(
    features: pd.DataFrame,
    ranked_features: pd.Series,
    *,
    max_correlation: float,
) -> tuple[list[str], list[str]]:
    if ranked_features.empty:
        return [], []

    numeric = _finite_feature_matrix(features[ranked_features.index.tolist()])
    corr = numeric.corr(method="spearman").abs().fillna(0.0)
    selected: list[str] = []
    pruned: list[str] = []

    for feature_name in ranked_features.index.tolist():
        if not selected:
            selected.append(feature_name)
            continue
        is_correlated = False
        for retained in selected:
            if float(corr.loc[feature_name, retained]) >= max_correlation:
                is_correlated = True
                pruned.append(feature_name)
                break
        if not is_correlated:
            selected.append(feature_name)

    return selected, pruned


def _stability_selection_scores(
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    *,
    config: FeatureSelectionConfig,
    progress_callback: FeatureSelectionProgressCallback | None = None,
) -> dict[str, float]:
    if features.empty:
        if progress_callback is not None:
            progress_callback(
                "stability_selection_complete",
                {
                    "successful_iterations": 0,
                    "total_iterations": 0,
                    "skipped": True,
                },
            )
        return {}

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    numeric = _finite_feature_matrix(features)
    numeric = numeric.fillna(0.0)
    label = pd.Series(pd.to_numeric(target, errors="coerce"), index=numeric.index).fillna(0.0)
    unique_values = set(label.dropna().unique().tolist())
    if unique_values.difference({0.0, 1.0}):
        median = float(label.median()) if len(label) else 0.0
        label = (label > median).astype(float)

    total_iterations = max(1, int(config.stability_iterations))
    if progress_callback is not None:
        progress_callback(
            "stability_selection_start",
            {
                "feature_count": int(numeric.shape[1]),
                "sample_size": (
                    int(max(50, int(round(len(numeric) * config.stability_subsample_ratio))))
                    if len(numeric) > 0
                    else 0
                ),
                "total_iterations": total_iterations,
            },
        )

    if numeric.shape[0] < 100 or numeric.shape[1] <= 1:
        if progress_callback is not None:
            progress_callback(
                "stability_selection_complete",
                {
                    "successful_iterations": 0,
                    "total_iterations": int(total_iterations),
                    "skipped": True,
                },
            )
        return {column: 1.0 for column in numeric.columns}

    rng = np.random.default_rng(config.random_state)
    sample_size = max(50, int(round(len(numeric) * config.stability_subsample_ratio)))
    active_counts = dict.fromkeys(numeric.columns.tolist(), 0)
    successful_iterations = 0
    progress_step = max(1, total_iterations // 4)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "selector",
                LogisticRegression(
                    solver="saga",
                    l1_ratio=1.0,
                    C=0.15,
                    max_iter=2000,
                    random_state=config.random_state,
                ),
            ),
        ]
    )

    for iteration in range(1, total_iterations + 1):
        sample_idx = rng.choice(len(numeric), size=sample_size, replace=False)
        X_sample = numeric.iloc[sample_idx]
        y_sample = label.iloc[sample_idx]
        if y_sample.nunique(dropna=True) < 2:
            if progress_callback is not None and (
                iteration == total_iterations or iteration % progress_step == 0
            ):
                progress_callback(
                    "stability_selection_progress",
                    {
                        "completed_iterations": int(iteration),
                        "successful_iterations": int(successful_iterations),
                        "total_iterations": int(total_iterations),
                    },
                )
            continue
        model.fit(X_sample, y_sample)
        coefs = np.asarray(model.named_steps["selector"].coef_).reshape(-1)
        successful_iterations += 1
        for column, coef in zip(numeric.columns, coefs, strict=False):
            if abs(float(coef)) > 1e-8:
                active_counts[column] = int(active_counts[column]) + 1
        if progress_callback is not None and (
            iteration == total_iterations or iteration % progress_step == 0
        ):
            progress_callback(
                "stability_selection_progress",
                {
                    "completed_iterations": int(iteration),
                    "successful_iterations": int(successful_iterations),
                    "total_iterations": int(total_iterations),
                },
            )

    if successful_iterations <= 0:
        if progress_callback is not None:
            progress_callback(
                "stability_selection_complete",
                {
                    "successful_iterations": 0,
                    "total_iterations": int(total_iterations),
                },
            )
        return {column: 1.0 for column in numeric.columns}

    scores = {
        column: float(int(active_counts[column]) / successful_iterations)
        for column in numeric.columns
    }
    if progress_callback is not None:
        progress_callback(
            "stability_selection_complete",
            {
                "successful_iterations": int(successful_iterations),
                "total_iterations": int(total_iterations),
            },
        )
    return scores


def select_training_features(
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    config: FeatureSelectionConfig,
    *,
    groups: pd.Series | np.ndarray | None = None,
    progress_callback: FeatureSelectionProgressCallback | None = None,
) -> FeatureSelectionResult:
    use_group_aware_ic = bool(config.group_aware_ic and groups is not None)
    ranked = compute_information_coefficients(
        features,
        target,
        groups=(groups if use_group_aware_ic else None),
        min_group_size=int(config.min_group_size),
        progress_callback=progress_callback,
    )
    screened = ranked[ranked >= float(config.min_information_coefficient)]
    if progress_callback is not None:
        progress_callback(
            "information_coefficient_screen_complete",
            {
                "ranked_feature_count": int(len(ranked)),
                "screened_feature_count": int(len(screened)),
                "min_information_coefficient": float(config.min_information_coefficient),
            },
        )
    if screened.empty:
        screened = ranked.head(min(int(config.max_features), len(ranked)))
    if screened.empty:
        return FeatureSelectionResult(
            selected_features=list(features.columns),
            information_coefficients={},
            correlation_pruned_features=[],
            stability_scores={},
            diagnostics={
                "initial_feature_count": float(features.shape[1]),
                "selected_feature_count": float(features.shape[1]),
            },
        )

    corr_selected, corr_pruned = _prune_correlated_features(
        features,
        screened,
        max_correlation=float(config.max_correlation),
    )
    if not corr_selected:
        corr_selected = screened.index.tolist()

    if len(corr_selected) > int(config.max_features):
        corr_selected = corr_selected[: int(config.max_features)]

    if progress_callback is not None:
        progress_callback(
            "correlation_prune_complete",
            {
                "candidate_feature_count": int(len(corr_selected)),
                "pruned_feature_count": int(len(corr_pruned)),
                "max_features": int(config.max_features),
            },
        )

    stability_scores = _stability_selection_scores(
        features[corr_selected],
        target,
        config=config,
        progress_callback=progress_callback,
    )
    stable_selected = [
        column
        for column in corr_selected
        if float(stability_scores.get(column, 1.0)) >= float(config.min_stability_support)
    ]
    if stable_selected:
        final_selected = stable_selected[: int(config.max_features)]
    else:
        final_selected = corr_selected[: int(config.max_features)]

    result = FeatureSelectionResult(
        selected_features=final_selected,
        information_coefficients={key: float(value) for key, value in screened.items()},
        correlation_pruned_features=corr_pruned,
        stability_scores=stability_scores,
        diagnostics={
            "initial_feature_count": float(features.shape[1]),
            "ic_screened_feature_count": float(len(screened)),
            "correlation_pruned_count": float(len(corr_pruned)),
            "stability_selected_count": float(len(stable_selected)),
            "selected_feature_count": float(len(final_selected)),
            "group_aware_ic_enabled": float(use_group_aware_ic),
        },
    )
    if progress_callback is not None:
        progress_callback(
            "selection_complete",
            {
                "selected_feature_count": int(len(final_selected)),
                "screened_feature_count": int(len(screened)),
                "candidate_feature_count": int(len(corr_selected)),
            },
        )
    return result
