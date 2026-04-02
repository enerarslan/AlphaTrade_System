"""Helpers for preserving fitted feature schema during inference."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def _normalize_feature_names(values: Any) -> tuple[str, ...]:
    """Normalize arbitrary feature-name payloads into a stable tuple."""
    if values is None:
        return ()
    if isinstance(values, (list, tuple, set, np.ndarray, pd.Index)):
        names = [str(value).strip() for value in values if str(value).strip()]
        return tuple(names)
    value = str(values).strip()
    return (value,) if value else ()


def resolve_model_feature_names(
    model: Any,
    *,
    fallback_feature_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Resolve the fitted feature schema exposed by an estimator or fallback contract."""
    candidates = [
        model,
        getattr(model, "_model", None),
        getattr(model, "booster_", None),
        getattr(getattr(model, "_model", None), "booster_", None),
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        for attr_name in ("feature_names_in_", "feature_name_", "feature_names"):
            feature_names = _normalize_feature_names(getattr(candidate, attr_name, None))
            if feature_names:
                return feature_names
        feature_name_getter = getattr(candidate, "feature_name", None)
        if callable(feature_name_getter):
            try:
                feature_names = _normalize_feature_names(feature_name_getter())
            except Exception:
                feature_names = ()
            if feature_names:
                return feature_names

    return _normalize_feature_names(fallback_feature_names)


def align_inference_features(
    X: Any,
    *,
    feature_names: Sequence[str] | None = None,
) -> Any:
    """Align inference input to a fitted feature schema when one is available."""
    resolved_feature_names = _normalize_feature_names(feature_names)
    if not resolved_feature_names or X is None:
        return X

    if isinstance(X, pd.DataFrame):
        frame = X.copy()
        if tuple(str(column) for column in frame.columns) == resolved_feature_names:
            return frame
        if all(name in frame.columns for name in resolved_feature_names):
            return frame.loc[:, list(resolved_feature_names)].copy()
        if frame.shape[1] == len(resolved_feature_names):
            frame.columns = list(resolved_feature_names)
            return frame
        return X

    if not hasattr(X, "shape") or len(getattr(X, "shape", ())) != 2:
        return X
    if int(X.shape[1]) != len(resolved_feature_names):
        return X

    return pd.DataFrame(np.asarray(X), columns=list(resolved_feature_names))


def prepare_model_inference_input(
    model: Any,
    X: Any,
    *,
    fallback_feature_names: Sequence[str] | None = None,
) -> Any:
    """Prepare inference input using estimator-exposed schema or a contract fallback."""
    return align_inference_features(
        X,
        feature_names=resolve_model_feature_names(
            model,
            fallback_feature_names=fallback_feature_names,
        ),
    )
