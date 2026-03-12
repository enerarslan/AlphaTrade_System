"""Shared trading cost model for model training/evaluation pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TradingCostModel:
    """Canonical cost model in basis points."""

    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    impact_bps: float = 0.0
    turnover_penalty_bps: float = 0.0

    def __post_init__(self) -> None:
        for name in ("spread_bps", "slippage_bps", "impact_bps", "turnover_penalty_bps"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def execution_cost_bps(self) -> float:
        """Execution-layer cost component (spread + slippage + impact)."""
        return float(self.spread_bps + self.slippage_bps + self.impact_bps)

    @property
    def total_cost_bps(self) -> float:
        """Total cost in basis points including turnover penalty."""
        return float(self.execution_cost_bps + self.turnover_penalty_bps)

    @property
    def execution_cost_rate(self) -> float:
        """Execution-layer cost as decimal return."""
        return float(self.execution_cost_bps / 10000.0)

    @property
    def total_cost_rate(self) -> float:
        """Total cost as decimal return."""
        return float(self.total_cost_bps / 10000.0)

    @classmethod
    def from_assumed_costs(
        cls,
        assumed_cost_bps: float,
        turnover_penalty_bps: float = 0.0,
    ) -> "TradingCostModel":
        """Create model from aggregate assumed costs used in legacy APIs."""
        return cls(
            spread_bps=float(assumed_cost_bps),
            slippage_bps=0.0,
            impact_bps=0.0,
            turnover_penalty_bps=float(turnover_penalty_bps),
        )

    def apply_turnover_costs(
        self,
        gross_returns: np.ndarray,
        turnover: np.ndarray,
    ) -> np.ndarray:
        """Apply per-turnover costs to gross return stream."""
        gross_arr = np.asarray(gross_returns, dtype=float).reshape(-1)
        turnover_arr = np.asarray(turnover, dtype=float).reshape(-1)
        min_len = min(int(gross_arr.size), int(turnover_arr.size))
        if min_len <= 0:
            return np.array([], dtype=float)

        gross_arr = gross_arr[-min_len:]
        turnover_arr = turnover_arr[-min_len:]
        return gross_arr - (turnover_arr * self.total_cost_rate)

    def to_dict(self) -> dict[str, float]:
        """Serialize cost model."""
        return {
            "spread_bps": float(self.spread_bps),
            "slippage_bps": float(self.slippage_bps),
            "impact_bps": float(self.impact_bps),
            "turnover_penalty_bps": float(self.turnover_penalty_bps),
            "execution_cost_bps": float(self.execution_cost_bps),
            "total_cost_bps": float(self.total_cost_bps),
        }
