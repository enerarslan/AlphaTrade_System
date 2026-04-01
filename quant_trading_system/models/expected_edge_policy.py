"""OOF-safe expected-edge policy model shared by training and promotion inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

_CONTEXT_TOKENS = (
    "vol",
    "regime",
    "spread",
    "imbalance",
    "flow",
    "sentiment",
    "news",
    "earnings",
    "filing",
    "macro",
    "relative",
    "strength",
    "short",
    "ftd",
    "liquid",
    "vwap",
    "pressure",
    "quality",
    "breadth",
    "momentum",
)
_DERIVED_POLICY_COLUMNS = (
    "policy_probability",
    "policy_probability_centered",
    "policy_probability_confidence",
    "policy_long_margin",
    "policy_short_margin",
    "policy_abs_threshold_margin",
    "policy_signal_strength",
    "policy_signal_direction",
    "policy_is_long",
    "policy_is_short",
    "policy_raw_prediction",
)


def _safe_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize arbitrary feature frames into finite numeric columns."""
    numeric = (
        frame.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .copy()
    )
    return numeric


def _normalize_vector(values: Any, length: int, fill_value: float = np.nan) -> np.ndarray:
    """Align arbitrary inputs to a dense float vector."""
    if values is None:
        return np.full(length, fill_value, dtype=float)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == length:
        return arr.astype(float)
    if arr.size < length:
        aligned = np.full(length, fill_value, dtype=float)
        aligned[-arr.size :] = arr.astype(float)
        return aligned
    return arr[-length:].astype(float)


def derive_base_signal(
    probabilities: np.ndarray | list[float],
    *,
    long_threshold: float,
    short_threshold: float,
) -> np.ndarray:
    """Convert probabilities into long/short/flat base signals."""
    probability = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), 0.0, 1.0)
    return np.where(
        probability >= float(long_threshold),
        1.0,
        np.where(probability <= float(short_threshold), -1.0, 0.0),
    ).astype(float)


def build_expected_edge_feature_frame(
    feature_frame: pd.DataFrame,
    *,
    probabilities: np.ndarray | list[float],
    long_threshold: float,
    short_threshold: float,
    raw_predictions: np.ndarray | list[float] | None = None,
    signal_values: np.ndarray | list[float] | None = None,
    confidence: np.ndarray | list[float] | None = None,
    context_features: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Build the deterministic feature frame consumed by the edge policy model."""
    numeric_frame = _safe_numeric_frame(feature_frame.reset_index(drop=True))
    length = len(numeric_frame)
    probability = np.clip(_normalize_vector(probabilities, length, fill_value=np.nan), 0.0, 1.0)
    if signal_values is None:
        signal = derive_base_signal(
            probability,
            long_threshold=float(long_threshold),
            short_threshold=float(short_threshold),
        )
    else:
        signal = _normalize_vector(signal_values, length, fill_value=0.0)
    if confidence is None:
        resolved_confidence = np.clip(np.abs(probability - 0.5) * 2.0, 0.0, 1.0)
    else:
        resolved_confidence = np.clip(
            _normalize_vector(confidence, length, fill_value=0.0),
            0.0,
            1.0,
        )
    raw_prediction = _normalize_vector(raw_predictions, length, fill_value=np.nan)

    frame = pd.DataFrame(
        {
            "policy_probability": probability,
            "policy_probability_centered": probability - 0.5,
            "policy_probability_confidence": resolved_confidence,
            "policy_long_margin": probability - float(long_threshold),
            "policy_short_margin": float(short_threshold) - probability,
            "policy_abs_threshold_margin": np.maximum(
                probability - float(long_threshold),
                float(short_threshold) - probability,
            ),
            "policy_signal_strength": np.abs(signal),
            "policy_signal_direction": np.sign(signal),
            "policy_is_long": (signal > 0.0).astype(float),
            "policy_is_short": (signal < 0.0).astype(float),
            "policy_raw_prediction": raw_prediction,
        },
        index=numeric_frame.index,
    )

    selected_context = [name for name in list(context_features or []) if name in numeric_frame.columns]
    if selected_context:
        frame = pd.concat([frame, numeric_frame.loc[:, selected_context]], axis=1)
    return frame.replace([np.inf, -np.inf], np.nan)


@dataclass(slots=True)
class ExpectedEdgePolicyConfig:
    """Configuration for the second-stage expected-edge policy model."""

    min_samples: int = 120
    min_coverage: float = 0.55
    max_context_features: int = 24
    min_pass_probability: float = 0.53
    min_expected_edge: float = 0.0
    min_signal_scale: float = 0.25
    max_signal_scale: float = 1.10
    random_state: int = 42


class ExpectedEdgePolicyModel:
    """Learn a cost-aware trade admission and sizing policy from OOF-safe data."""

    def __init__(self, config: ExpectedEdgePolicyConfig | None = None) -> None:
        self.config = config or ExpectedEdgePolicyConfig()
        self.selected_context_features_: list[str] = []
        self.policy_feature_names_: list[str] = []
        self.feature_scores_: dict[str, float] = {}
        self.edge_reference_: float = 0.001
        self.training_summary_: dict[str, Any] = {}
        self._edge_model: Any | None = None
        self._pass_model: Any | None = None

    @staticmethod
    def _feature_score(series: pd.Series, target: np.ndarray) -> float:
        """Score candidate context features with stable finite correlation."""
        values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() < 40:
            return 0.0
        valid = values.notna().to_numpy(dtype=bool) & np.isfinite(target)
        if int(np.count_nonzero(valid)) < 40:
            return 0.0
        aligned_values = values.to_numpy(dtype=float)[valid]
        aligned_target = target[valid]
        if np.nanstd(aligned_values) < 1e-9:
            return 0.0
        corr = np.corrcoef(aligned_values, aligned_target)[0, 1]
        if not np.isfinite(corr):
            return 0.0
        return float(abs(corr))

    def _select_context_features(
        self,
        feature_frame: pd.DataFrame,
        target: np.ndarray,
    ) -> list[str]:
        """Select a compact high-signal context feature slice for policy learning."""
        numeric_frame = _safe_numeric_frame(feature_frame)
        scored: list[tuple[int, float, str]] = []
        for column in numeric_frame.columns:
            if column in _DERIVED_POLICY_COLUMNS:
                continue
            score = self._feature_score(numeric_frame[column], target)
            token_bonus = int(any(token in column.lower() for token in _CONTEXT_TOKENS))
            if score <= 0.0 and token_bonus <= 0:
                continue
            scored.append((token_bonus, score, str(column)))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        selected = [column for _, _, column in scored[: int(self.config.max_context_features)]]
        self.feature_scores_ = {
            column: float(score)
            for _, score, column in scored[: int(self.config.max_context_features)]
        }
        return selected

    @staticmethod
    def _coerce_sample_weights(
        weights: np.ndarray | pd.Series | None,
        mask: np.ndarray,
    ) -> np.ndarray | None:
        """Align optional sample weights to the active candidate mask."""
        if weights is None:
            return None
        arr = np.asarray(weights, dtype=float).reshape(-1)
        if arr.size != mask.size:
            return None
        selected = arr[mask]
        return None if selected.size == 0 else selected

    @staticmethod
    def _safe_positive_reference(values: np.ndarray) -> float:
        """Derive a stable positive edge anchor for downstream sizing."""
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.001
        positive = np.abs(finite[finite > 0.0])
        if positive.size == 0:
            positive = np.abs(finite)
        reference = float(np.quantile(positive, 0.65)) if positive.size else 0.001
        return float(max(reference, 1e-4))

    def fit(
        self,
        feature_frame: pd.DataFrame,
        *,
        probabilities: np.ndarray | list[float],
        trade_returns: np.ndarray | list[float],
        long_threshold: float,
        short_threshold: float,
        raw_predictions: np.ndarray | list[float] | None = None,
        signal_values: np.ndarray | list[float] | None = None,
        confidence: np.ndarray | list[float] | None = None,
        sample_weights: np.ndarray | pd.Series | None = None,
    ) -> dict[str, Any]:
        """Fit the expected-edge admission/sizing policy on candidate trades only."""
        trade_return_values = np.asarray(trade_returns, dtype=float).reshape(-1)
        numeric_frame = _safe_numeric_frame(feature_frame.reset_index(drop=True))
        if len(numeric_frame) != trade_return_values.size:
            raise ValueError("Expected-edge policy feature/target length mismatch.")

        selected_context = self._select_context_features(numeric_frame, trade_return_values)
        policy_frame = build_expected_edge_feature_frame(
            numeric_frame,
            probabilities=probabilities,
            long_threshold=float(long_threshold),
            short_threshold=float(short_threshold),
            raw_predictions=raw_predictions,
            signal_values=signal_values,
            confidence=confidence,
            context_features=selected_context,
        )

        signal = policy_frame["policy_signal_direction"].to_numpy(dtype=float)
        candidate_mask = np.isfinite(trade_return_values) & (np.abs(signal) > 1e-8)
        candidate_count = int(np.count_nonzero(candidate_mask))
        coverage = float(candidate_count / max(len(policy_frame), 1))
        if candidate_count < int(self.config.min_samples):
            raise ValueError(
                f"Expected-edge policy needs at least {self.config.min_samples} candidate trades; "
                f"received {candidate_count}."
            )
        if coverage < float(self.config.min_coverage):
            raise ValueError(
                f"Expected-edge policy coverage too low ({coverage:.2%} < "
                f"{float(self.config.min_coverage):.2%})."
            )

        X_train = (
            policy_frame.loc[candidate_mask]
            .fillna(0.0)
            .reset_index(drop=True)
        )
        y_edge = trade_return_values[candidate_mask]
        y_pass = (y_edge > 0.0).astype(int)
        weights = self._coerce_sample_weights(sample_weights, candidate_mask)

        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        self._edge_model = RandomForestRegressor(
            n_estimators=160,
            max_depth=6,
            min_samples_leaf=24,
            random_state=int(self.config.random_state),
            n_jobs=-1,
        )
        self._edge_model.fit(X_train, y_edge, sample_weight=weights)

        if len(np.unique(y_pass)) <= 1:
            self._pass_model = DummyClassifier(strategy="constant", constant=int(y_pass[0]))
            self._pass_model.fit(X_train, y_pass, sample_weight=weights)
        else:
            self._pass_model = RandomForestClassifier(
                n_estimators=160,
                max_depth=6,
                min_samples_leaf=24,
                random_state=int(self.config.random_state),
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
            self._pass_model.fit(X_train, y_pass, sample_weight=weights)

        self.selected_context_features_ = list(selected_context)
        self.policy_feature_names_ = list(X_train.columns)
        self.edge_reference_ = self._safe_positive_reference(y_edge)

        training_policy = self.predict_policy(
            numeric_frame,
            probabilities=probabilities,
            long_threshold=float(long_threshold),
            short_threshold=float(short_threshold),
            raw_predictions=raw_predictions,
            signal_values=signal_values,
            confidence=confidence,
        )
        selected_mask = training_policy["edge_policy_pass"].to_numpy(dtype=bool)
        selected_edge = trade_return_values[selected_mask & np.isfinite(trade_return_values)]
        candidate_edge = trade_return_values[candidate_mask]
        self.training_summary_ = {
            "candidate_count": int(candidate_count),
            "candidate_rate": float(coverage),
            "selected_count": int(np.count_nonzero(selected_mask)),
            "selected_rate": float(np.mean(selected_mask)) if len(selected_mask) else 0.0,
            "candidate_mean_trade_return": float(np.mean(candidate_edge)) if candidate_edge.size else 0.0,
            "selected_mean_trade_return": float(np.mean(selected_edge)) if selected_edge.size else 0.0,
            "selected_win_rate": float(np.mean(selected_edge > 0.0)) if selected_edge.size else 0.0,
            "edge_reference": float(self.edge_reference_),
            "selected_context_features": list(self.selected_context_features_),
        }
        return dict(self.training_summary_)

    def _predict_pass_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the profitable-trade probability from the fitted classifier."""
        if self._pass_model is None:
            return np.zeros(len(X), dtype=float)
        if hasattr(self._pass_model, "predict_proba"):
            proba = np.asarray(self._pass_model.predict_proba(X), dtype=float)
            if proba.ndim == 2:
                if proba.shape[1] >= 2:
                    return np.clip(proba[:, -1], 0.0, 1.0)
                return np.clip(proba[:, 0], 0.0, 1.0)
        prediction = np.asarray(self._pass_model.predict(X), dtype=float).reshape(-1)
        return np.clip(prediction, 0.0, 1.0)

    def predict_policy(
        self,
        feature_frame: pd.DataFrame,
        *,
        probabilities: np.ndarray | list[float],
        long_threshold: float,
        short_threshold: float,
        raw_predictions: np.ndarray | list[float] | None = None,
        signal_values: np.ndarray | list[float] | None = None,
        confidence: np.ndarray | list[float] | None = None,
    ) -> pd.DataFrame:
        """Predict expected edge, trade pass probability, and signal scale."""
        if self._edge_model is None or self._pass_model is None:
            raise RuntimeError("Expected-edge policy model has not been fitted.")

        numeric_frame = _safe_numeric_frame(feature_frame.reset_index(drop=True))
        policy_frame = build_expected_edge_feature_frame(
            numeric_frame,
            probabilities=probabilities,
            long_threshold=float(long_threshold),
            short_threshold=float(short_threshold),
            raw_predictions=raw_predictions,
            signal_values=signal_values,
            confidence=confidence,
            context_features=self.selected_context_features_,
        )
        for column in self.policy_feature_names_:
            if column not in policy_frame.columns:
                policy_frame[column] = 0.0
        X = policy_frame.loc[:, self.policy_feature_names_].fillna(0.0)
        signal = policy_frame["policy_signal_direction"].to_numpy(dtype=float)
        candidate_mask = np.abs(signal) > 1e-8

        expected_edge = np.zeros(len(X), dtype=float)
        pass_probability = np.zeros(len(X), dtype=float)
        if bool(np.any(candidate_mask)):
            candidate_features = X.loc[candidate_mask]
            expected_edge[candidate_mask] = np.asarray(
                self._edge_model.predict(candidate_features),
                dtype=float,
            ).reshape(-1)
            pass_probability[candidate_mask] = self._predict_pass_probability(candidate_features)

        min_edge = float(self.config.min_expected_edge)
        min_pass_probability = float(self.config.min_pass_probability)
        edge_range = max(float(self.edge_reference_) - min_edge, 1e-6)
        edge_component = np.clip((expected_edge - min_edge) / edge_range, 0.0, 1.5)
        pass_component = np.clip(
            (pass_probability - min_pass_probability) / max(1.0 - min_pass_probability, 1e-6),
            0.0,
            1.0,
        )
        scale = float(self.config.min_signal_scale) + (
            float(self.config.max_signal_scale) - float(self.config.min_signal_scale)
        ) * ((0.55 * pass_component) + (0.45 * np.clip(edge_component, 0.0, 1.0)))
        scale = np.clip(scale, float(self.config.min_signal_scale), float(self.config.max_signal_scale))
        policy_pass = (
            candidate_mask
            & (expected_edge >= min_edge)
            & (pass_probability >= min_pass_probability)
        )
        scale = np.where(policy_pass, scale, 0.0)

        return pd.DataFrame(
            {
                "expected_edge": expected_edge,
                "edge_pass_probability": np.clip(pass_probability, 0.0, 1.0),
                "edge_loss_probability": np.clip(1.0 - pass_probability, 0.0, 1.0),
                "edge_policy_pass": policy_pass.astype(bool),
                "edge_policy_scale": scale,
            },
            index=policy_frame.index,
        )
