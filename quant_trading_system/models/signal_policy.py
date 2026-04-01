"""Shared asymmetric signal policy derivation for promotion packages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SideSignalPolicy:
    """Side-specific inference policy derived from realized edge quality."""

    enabled: bool = True
    signal_scale: float = 1.0
    confidence_scale: float = 1.0
    threshold_adjustment: float = 0.0
    trade_count: int = 0
    active_rate: float = 0.0
    mean_signed_return: float = 0.0
    median_signed_return: float = 0.0
    win_rate: float = 0.5
    score: float = 0.0
    sample_confidence: float = 0.0


def _default_side_policy() -> SideSignalPolicy:
    return SideSignalPolicy()


def _derive_side_policy(
    *,
    side: str,
    probabilities: np.ndarray,
    realized_returns: np.ndarray,
    threshold: float,
    min_samples: int,
) -> SideSignalPolicy:
    side_name = str(side).strip().lower()
    if side_name not in {"long", "short"}:
        raise ValueError(f"Unsupported side policy requested: {side}")

    direction = 1.0 if side_name == "long" else -1.0
    if side_name == "long":
        mask = probabilities >= float(threshold)
        tighten_adjustment = 0.025
        relax_adjustment = -0.010
    else:
        mask = probabilities <= float(threshold)
        tighten_adjustment = -0.025
        relax_adjustment = 0.010

    trade_count = int(np.count_nonzero(mask))
    active_rate = float(trade_count / max(len(probabilities), 1))
    if trade_count == 0:
        return SideSignalPolicy(trade_count=0, active_rate=0.0, sample_confidence=0.0)

    signed_returns = np.asarray(realized_returns[mask], dtype=float) * direction
    signed_returns = signed_returns[np.isfinite(signed_returns)]
    if signed_returns.size == 0:
        return SideSignalPolicy(trade_count=trade_count, active_rate=active_rate, sample_confidence=0.0)

    mean_signed_return = float(np.mean(signed_returns))
    median_signed_return = float(np.median(signed_returns))
    win_rate = float(np.mean(signed_returns > 0.0))
    sample_confidence = float(np.clip(signed_returns.size / max(1, int(min_samples)), 0.0, 1.0))

    edge_bps = mean_signed_return * 10_000.0
    edge_score = float(np.clip(edge_bps / 12.0, -1.5, 1.5))
    hit_score = float(np.clip((win_rate - 0.5) / 0.12, -1.0, 1.0))
    score = float(sample_confidence * ((0.70 * edge_score) + (0.30 * hit_score)))

    if sample_confidence < 0.35:
        return SideSignalPolicy(
            trade_count=trade_count,
            active_rate=active_rate,
            mean_signed_return=mean_signed_return,
            median_signed_return=median_signed_return,
            win_rate=win_rate,
            score=score,
            sample_confidence=sample_confidence,
        )

    enabled = True
    signal_scale = 1.0
    confidence_scale = 1.0
    threshold_adjustment = 0.0
    if score <= -0.85:
        enabled = False
        signal_scale = 0.0
        confidence_scale = 0.0
        threshold_adjustment = tighten_adjustment * 1.6
    elif score <= -0.35:
        signal_scale = 0.45
        confidence_scale = 0.65
        threshold_adjustment = tighten_adjustment
    elif score < 0.10:
        signal_scale = 0.80
        confidence_scale = 0.90
        threshold_adjustment = tighten_adjustment * 0.4
    elif score >= 0.75:
        signal_scale = 1.15
        confidence_scale = 1.10
        threshold_adjustment = relax_adjustment
    elif score >= 0.35:
        signal_scale = 1.05
        confidence_scale = 1.05
        threshold_adjustment = relax_adjustment * 0.5

    return SideSignalPolicy(
        enabled=enabled,
        signal_scale=float(np.clip(signal_scale, 0.0, 1.25)),
        confidence_scale=float(np.clip(confidence_scale, 0.0, 1.20)),
        threshold_adjustment=float(np.clip(threshold_adjustment, -0.08, 0.08)),
        trade_count=trade_count,
        active_rate=active_rate,
        mean_signed_return=mean_signed_return,
        median_signed_return=median_signed_return,
        win_rate=win_rate,
        score=score,
        sample_confidence=sample_confidence,
    )


def derive_asymmetric_signal_policy(
    probabilities: np.ndarray | None,
    realized_returns: np.ndarray | None,
    *,
    long_threshold: float,
    short_threshold: float,
    min_samples: int | None = None,
) -> dict[str, Any]:
    """Derive long/short inference policy from realized side-specific edge."""
    if probabilities is None or realized_returns is None:
        return {
            "long": asdict(_default_side_policy()),
            "short": asdict(_default_side_policy()),
            "min_samples": int(min_samples or 0),
            "source": "unavailable",
        }

    prob = np.asarray(probabilities, dtype=float).reshape(-1)
    returns = np.asarray(realized_returns, dtype=float).reshape(-1)
    if prob.size == 0 or returns.size == 0:
        return {
            "long": asdict(_default_side_policy()),
            "short": asdict(_default_side_policy()),
            "min_samples": int(min_samples or 0),
            "source": "empty",
        }

    target_len = min(prob.size, returns.size)
    prob = np.clip(prob[-target_len:], 0.0, 1.0)
    returns = returns[-target_len:]
    finite_mask = np.isfinite(prob) & np.isfinite(returns)
    prob = prob[finite_mask]
    returns = returns[finite_mask]
    if prob.size == 0:
        return {
            "long": asdict(_default_side_policy()),
            "short": asdict(_default_side_policy()),
            "min_samples": int(min_samples or 0),
            "source": "no_finite_rows",
        }

    resolved_min_samples = max(8, int(min_samples or max(12, round(prob.size * 0.02))))
    long_policy = _derive_side_policy(
        side="long",
        probabilities=prob,
        realized_returns=returns,
        threshold=float(long_threshold),
        min_samples=resolved_min_samples,
    )
    short_policy = _derive_side_policy(
        side="short",
        probabilities=prob,
        realized_returns=returns,
        threshold=float(short_threshold),
        min_samples=resolved_min_samples,
    )
    return {
        "long": asdict(long_policy),
        "short": asdict(short_policy),
        "min_samples": int(resolved_min_samples),
        "source": "realized_returns",
    }
