"""
Position sizing strategies for trade risk management.

Implements multiple position sizing algorithms:
- Fixed Fractional: Risk fixed % of equity per trade
- Kelly Criterion: Optimal growth rate sizing
- Volatility Based: Size inversely to volatility
- Optimal F: Maximize geometric growth
- RL-Based Sizing: Use reinforcement learning for sizing

All strategies respect hard and soft position limits.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from quant_trading_system.core.data_types import Direction, Portfolio, Position, TradeSignal
from quant_trading_system.core.exceptions import RiskError

logger = logging.getLogger(__name__)


class SizingMethod(str, Enum):
    """Position sizing method enumeration."""

    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_BASED = "volatility_based"
    OPTIMAL_F = "optimal_f"
    RL_BASED = "rl_based"


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""

    symbol: str
    recommended_size: Decimal
    method: SizingMethod
    raw_size: Decimal
    constraints_applied: list[str] = field(default_factory=list)
    risk_amount: Decimal = Decimal("0")
    entry_price: Decimal = Decimal("0")
    stop_loss_price: Decimal | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def was_constrained(self) -> bool:
        """Check if size was reduced by constraints."""
        return self.recommended_size < self.raw_size

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "recommended_size": float(self.recommended_size),
            "method": self.method.value,
            "raw_size": float(self.raw_size),
            "constraints_applied": self.constraints_applied,
            "risk_amount": float(self.risk_amount),
            "entry_price": float(self.entry_price),
            "stop_loss_price": float(self.stop_loss_price) if self.stop_loss_price else None,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class SizingConstraints(BaseModel):
    """Hard and soft limits for position sizing."""

    # Hard limits
    max_position_value: Decimal = Field(default=Decimal("1000000"), ge=0, description="Max $ value for single position")
    max_position_pct: float = Field(default=0.10, ge=0, le=1.0, description="Max % of equity for single position")
    max_portfolio_positions: int = Field(default=20, ge=1, description="Max number of positions")
    max_sector_exposure_pct: float = Field(default=0.25, ge=0, le=1.0, description="Max % exposure per sector")
    max_correlation_exposure: float = Field(default=0.30, ge=0, le=1.0, description="Max correlated positions exposure")
    max_daily_turnover_pct: float = Field(default=0.20, ge=0, le=1.0, description="Max daily turnover")
    min_position_value: Decimal = Field(default=Decimal("100"), ge=0, description="Min $ value for position")

    # Soft limits (gradual penalty)
    soft_limit_threshold: float = Field(default=0.8, ge=0, le=1.0, description="Start applying soft penalty at this % of hard limit")
    soft_limit_penalty_rate: float = Field(default=0.5, ge=0, le=1.0, description="Reduction rate per % over soft threshold")

    def apply_hard_limits(
        self,
        size: Decimal,
        equity: Decimal,
        entry_price: Decimal,
        current_positions: int,
    ) -> tuple[Decimal, list[str]]:
        """Apply hard position limits.

        Returns:
            Tuple of (adjusted size, list of constraints applied).
        """
        constraints_applied = []
        position_value = size * entry_price

        # Max position value limit
        if position_value > self.max_position_value:
            size = self.max_position_value / entry_price
            constraints_applied.append(f"max_position_value:{self.max_position_value}")

        # Max position % of equity
        max_value_by_pct = equity * Decimal(str(self.max_position_pct))
        if size * entry_price > max_value_by_pct:
            size = max_value_by_pct / entry_price
            constraints_applied.append(f"max_position_pct:{self.max_position_pct}")

        # Max number of positions
        if current_positions >= self.max_portfolio_positions:
            size = Decimal("0")
            constraints_applied.append(f"max_portfolio_positions:{self.max_portfolio_positions}")

        # Min position value
        if size * entry_price < self.min_position_value and size > 0:
            size = Decimal("0")
            constraints_applied.append(f"below_min_position_value:{self.min_position_value}")

        return size, constraints_applied


class BasePositionSizer(ABC):
    """Base class for position sizing strategies."""

    def __init__(
        self,
        constraints: SizingConstraints | None = None,
    ) -> None:
        """Initialize position sizer.

        Args:
            constraints: Position sizing constraints.
        """
        self.constraints = constraints or SizingConstraints()

    @abstractmethod
    def calculate_size(
        self,
        signal: TradeSignal,
        portfolio: Portfolio,
        entry_price: Decimal,
        stop_loss_price: Decimal | None = None,
        market_data: dict[str, Any] | None = None,
    ) -> PositionSizeResult:
        """Calculate recommended position size.

        Args:
            signal: Trade signal for sizing.
            portfolio: Current portfolio state.
            entry_price: Expected entry price.
            stop_loss_price: Stop loss price for risk calculation.
            market_data: Additional market data (volatility, volume, etc.).

        Returns:
            Position size result with recommended size.
        """
        pass

    def _apply_constraints(
        self,
        raw_size: Decimal,
        portfolio: Portfolio,
        entry_price: Decimal,
    ) -> tuple[Decimal, list[str]]:
        """Apply sizing constraints to raw size."""
        return self.constraints.apply_hard_limits(
            size=raw_size,
            equity=portfolio.equity,
            entry_price=entry_price,
            current_positions=portfolio.position_count,
        )


class FixedFractionalSizer(BasePositionSizer):
    """Fixed fractional position sizing.

    Risks a fixed percentage of equity per trade based on the distance
    to the stop loss.
    """

    def __init__(
        self,
        risk_fraction: float = 0.01,
        default_stop_pct: float = 0.02,
        constraints: SizingConstraints | None = None,
    ) -> None:
        """Initialize fixed fractional sizer.

        Args:
            risk_fraction: Fraction of equity to risk per trade (0.01 = 1%).
            default_stop_pct: Default stop loss % if not provided.
            constraints: Position sizing constraints.
        """
        super().__init__(constraints)
        if not 0 < risk_fraction <= 0.05:
            raise ValueError("risk_fraction should be between 0 and 0.05 (5%)")
        self.risk_fraction = risk_fraction
        self.default_stop_pct = default_stop_pct

    def calculate_size(
        self,
        signal: TradeSignal,
        portfolio: Portfolio,
        entry_price: Decimal,
        stop_loss_price: Decimal | None = None,
        market_data: dict[str, Any] | None = None,
    ) -> PositionSizeResult:
        """Calculate position size using fixed fractional method.

        Formula: position_size = (equity × risk_fraction) / (entry_price × stop_loss_pct)
        """
        if entry_price <= 0:
            raise RiskError("Entry price must be positive", details={"entry_price": float(entry_price)})

        # Calculate stop loss distance
        if stop_loss_price is not None and stop_loss_price > 0:
            stop_distance = abs(entry_price - stop_loss_price)
            stop_pct = float(stop_distance / entry_price)
        else:
            stop_pct = self.default_stop_pct
            # BUG FIX: Stop loss direction depends on position direction
            # LONG: stop is BELOW entry (1 - stop_pct)
            # SHORT: stop is ABOVE entry (1 + stop_pct)
            if signal.direction == Direction.SHORT:
                stop_loss_price = entry_price * Decimal(str(1 + stop_pct))
            else:
                stop_loss_price = entry_price * Decimal(str(1 - stop_pct))

        # Calculate risk amount
        risk_amount = portfolio.equity * Decimal(str(self.risk_fraction))

        # Calculate position size
        if stop_pct > 0:
            raw_size = risk_amount / (entry_price * Decimal(str(stop_pct)))
        else:
            raw_size = Decimal("0")

        # Apply constraints
        adjusted_size, constraints_applied = self._apply_constraints(
            raw_size, portfolio, entry_price
        )

        return PositionSizeResult(
            symbol=signal.symbol,
            recommended_size=adjusted_size,
            method=SizingMethod.FIXED_FRACTIONAL,
            raw_size=raw_size,
            constraints_applied=constraints_applied,
            risk_amount=risk_amount,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            confidence=signal.confidence,
            metadata={
                "risk_fraction": self.risk_fraction,
                "stop_pct": stop_pct,
            },
        )


class KellyCriterionSizer(BasePositionSizer):
    """Kelly Criterion position sizing for optimal growth.

    Uses historical win rate and win/loss ratio to determine optimal
    position size for geometric growth.
    """

    def __init__(
        self,
        win_rate: float = 0.5,
        win_loss_ratio: float = 1.5,
        kelly_fraction: float = 0.25,
        lookback_trades: int = 100,
        min_trades: int = 20,
        constraints: SizingConstraints | None = None,
    ) -> None:
        """Initialize Kelly Criterion sizer.

        Args:
            win_rate: Historical win rate (0.0 to 1.0).
            win_loss_ratio: Ratio of average win to average loss.
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly).
            lookback_trades: Number of trades to use for calculations.
            min_trades: Minimum trades required before using Kelly.
            constraints: Position sizing constraints.
        """
        super().__init__(constraints)
        self.win_rate = win_rate
        self.win_loss_ratio = win_loss_ratio
        self.kelly_fraction = kelly_fraction
        self.lookback_trades = lookback_trades
        self.min_trades = min_trades
        self._trade_history: list[float] = []

    def update_trade_history(self, pnl_pct: float) -> None:
        """Update trade history with new trade result.

        Args:
            pnl_pct: P&L percentage of the trade.
        """
        self._trade_history.append(pnl_pct)
        if len(self._trade_history) > self.lookback_trades:
            self._trade_history = self._trade_history[-self.lookback_trades:]
        self._recalculate_stats()

    def _recalculate_stats(self) -> None:
        """Recalculate win rate and win/loss ratio from history."""
        if len(self._trade_history) < self.min_trades:
            return

        wins = [t for t in self._trade_history if t > 0]
        losses = [t for t in self._trade_history if t < 0]

        if wins and losses:
            self.win_rate = len(wins) / len(self._trade_history)
            avg_win = np.mean(wins)
            avg_loss = abs(np.mean(losses))
            if avg_loss > 0:
                self.win_loss_ratio = avg_win / avg_loss

    def calculate_kelly_pct(self) -> float:
        """Calculate raw Kelly percentage.

        Formula: kelly_pct = (win_rate × win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        """
        kelly = (self.win_rate * self.win_loss_ratio - (1 - self.win_rate)) / self.win_loss_ratio
        # Apply Kelly fraction (e.g., quarter Kelly)
        adjusted_kelly = kelly * self.kelly_fraction
        # Floor at 0 (no negative sizing)
        return max(0.0, adjusted_kelly)

    def calculate_size(
        self,
        signal: TradeSignal,
        portfolio: Portfolio,
        entry_price: Decimal,
        stop_loss_price: Decimal | None = None,
        market_data: dict[str, Any] | None = None,
    ) -> PositionSizeResult:
        """Calculate position size using Kelly Criterion."""
        if entry_price <= 0:
            raise RiskError("Entry price must be positive", details={"entry_price": float(entry_price)})

        # Calculate Kelly percentage
        kelly_pct = self.calculate_kelly_pct()

        # Calculate position value
        position_value = portfolio.equity * Decimal(str(kelly_pct))

        # Calculate size in shares
        raw_size = position_value / entry_price if kelly_pct > 0 else Decimal("0")

        # Apply constraints
        adjusted_size, constraints_applied = self._apply_constraints(
            raw_size, portfolio, entry_price
        )

        return PositionSizeResult(
            symbol=signal.symbol,
            recommended_size=adjusted_size,
            method=SizingMethod.KELLY_CRITERION,
            raw_size=raw_size,
            constraints_applied=constraints_applied,
            risk_amount=position_value,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            confidence=signal.confidence,
            metadata={
                "kelly_pct": kelly_pct,
                "win_rate": self.win_rate,
                "win_loss_ratio": self.win_loss_ratio,
                "kelly_fraction": self.kelly_fraction,
                "trades_in_history": len(self._trade_history),
            },
        )


class VolatilityBasedSizer(BasePositionSizer):
    """Volatility-based position sizing.

    Sizes positions inversely to volatility to maintain consistent
    portfolio risk across different market conditions.
    """

    def __init__(
        self,
        target_volatility: float = 0.15,
        volatility_lookback: int = 20,
        volatility_method: str = "atr",
        max_leverage: float = 2.0,
        constraints: SizingConstraints | None = None,
    ) -> None:
        """Initialize volatility-based sizer.

        Args:
            target_volatility: Target annualized volatility (0.15 = 15%).
            volatility_lookback: Number of bars for volatility calculation.
            volatility_method: Method for volatility calculation ('atr' or 'std').
            max_leverage: Maximum leverage allowed.
            constraints: Position sizing constraints.
        """
        super().__init__(constraints)
        self.target_volatility = target_volatility
        self.volatility_lookback = volatility_lookback
        self.volatility_method = volatility_method
        self.max_leverage = max_leverage

    def calculate_size(
        self,
        signal: TradeSignal,
        portfolio: Portfolio,
        entry_price: Decimal,
        stop_loss_price: Decimal | None = None,
        market_data: dict[str, Any] | None = None,
    ) -> PositionSizeResult:
        """Calculate position size based on volatility.

        Formula: position_size = (target_vol / realized_vol) × base_size
        """
        if entry_price <= 0:
            raise RiskError("Entry price must be positive", details={"entry_price": float(entry_price)})

        market_data = market_data or {}

        # Get realized volatility from market data
        realized_vol = market_data.get("volatility", self.target_volatility)
        if realized_vol <= 0:
            realized_vol = self.target_volatility

        # Calculate volatility ratio
        vol_ratio = self.target_volatility / realized_vol

        # Apply max leverage constraint
        vol_ratio = min(vol_ratio, self.max_leverage)

        # Base size is 100% equity at target volatility
        base_value = portfolio.equity

        # Adjusted position value
        position_value = base_value * Decimal(str(vol_ratio))

        # Calculate size in shares
        raw_size = position_value / entry_price

        # Apply constraints
        adjusted_size, constraints_applied = self._apply_constraints(
            raw_size, portfolio, entry_price
        )

        # Check if leverage constraint was applied
        if vol_ratio == self.max_leverage and self.target_volatility / realized_vol > self.max_leverage:
            constraints_applied.append(f"max_leverage:{self.max_leverage}")

        return PositionSizeResult(
            symbol=signal.symbol,
            recommended_size=adjusted_size,
            method=SizingMethod.VOLATILITY_BASED,
            raw_size=raw_size,
            constraints_applied=constraints_applied,
            risk_amount=position_value,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            confidence=signal.confidence,
            metadata={
                "target_volatility": self.target_volatility,
                "realized_volatility": realized_vol,
                "volatility_ratio": vol_ratio,
                "max_leverage": self.max_leverage,
            },
        )


class OptimalFSizer(BasePositionSizer):
    """Optimal F position sizing for maximum geometric growth.

    Uses historical trade simulation and Monte Carlo optimization
    to find the optimal fraction of equity to risk. More aggressive
    than Kelly Criterion.
    """

    def __init__(
        self,
        trade_returns: list[float] | None = None,
        num_simulations: int = 1000,
        optimization_target: str = "geometric_mean",
        safety_margin: float = 0.8,
        constraints: SizingConstraints | None = None,
    ) -> None:
        """Initialize Optimal F sizer.

        Args:
            trade_returns: Historical trade returns for optimization.
            num_simulations: Number of Monte Carlo simulations.
            optimization_target: Target metric ('geometric_mean', 'twr').
            safety_margin: Fraction of optimal F to use (for safety).
            constraints: Position sizing constraints.
        """
        super().__init__(constraints)
        self.trade_returns = trade_returns or []
        self.num_simulations = num_simulations
        self.optimization_target = optimization_target
        self.safety_margin = safety_margin
        self._optimal_f: float | None = None

    def update_trade_returns(self, returns: list[float]) -> None:
        """Update trade returns and recalculate optimal F."""
        self.trade_returns = returns
        self._optimal_f = None  # Invalidate cached value

    def calculate_optimal_f(self) -> float:
        """Calculate optimal F using Monte Carlo simulation.

        Returns:
            Optimal fraction of equity to risk.
        """
        if self._optimal_f is not None:
            return self._optimal_f

        if len(self.trade_returns) < 10:
            return 0.0

        returns = np.array(self.trade_returns)
        best_f = 0.0
        best_twr = 0.0

        # Grid search for optimal F
        for f in np.arange(0.01, 1.0, 0.01):
            twr = self._calculate_twr(returns, f)
            if twr > best_twr:
                best_twr = twr
                best_f = f

        self._optimal_f = best_f * self.safety_margin
        return self._optimal_f

    def _calculate_twr(self, returns: np.ndarray, f: float) -> float:
        """Calculate Terminal Wealth Relative for given f.

        Args:
            returns: Array of trade returns.
            f: Fraction to test.

        Returns:
            Terminal Wealth Relative.
        """
        # Find the largest loss
        min_return = np.min(returns)
        if min_return == 0:
            return 0.0

        # Calculate HPR for each trade
        hpr = 1 + f * (returns / abs(min_return))

        # TWR is product of all HPRs
        twr = np.prod(hpr)
        return twr if twr > 0 else 0.0

    def calculate_size(
        self,
        signal: TradeSignal,
        portfolio: Portfolio,
        entry_price: Decimal,
        stop_loss_price: Decimal | None = None,
        market_data: dict[str, Any] | None = None,
    ) -> PositionSizeResult:
        """Calculate position size using Optimal F."""
        if entry_price <= 0:
            raise RiskError("Entry price must be positive", details={"entry_price": float(entry_price)})

        # Calculate optimal F
        optimal_f = self.calculate_optimal_f()

        # Calculate position value
        position_value = portfolio.equity * Decimal(str(optimal_f))

        # Calculate size in shares
        raw_size = position_value / entry_price if optimal_f > 0 else Decimal("0")

        # Apply constraints
        adjusted_size, constraints_applied = self._apply_constraints(
            raw_size, portfolio, entry_price
        )

        return PositionSizeResult(
            symbol=signal.symbol,
            recommended_size=adjusted_size,
            method=SizingMethod.OPTIMAL_F,
            raw_size=raw_size,
            constraints_applied=constraints_applied,
            risk_amount=position_value,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            confidence=signal.confidence,
            metadata={
                "optimal_f": optimal_f,
                "safety_margin": self.safety_margin,
                "trades_analyzed": len(self.trade_returns),
            },
        )


class PositionSizerFactory:
    """Factory for creating position sizers."""

    _sizers: dict[SizingMethod, type[BasePositionSizer]] = {
        SizingMethod.FIXED_FRACTIONAL: FixedFractionalSizer,
        SizingMethod.KELLY_CRITERION: KellyCriterionSizer,
        SizingMethod.VOLATILITY_BASED: VolatilityBasedSizer,
        SizingMethod.OPTIMAL_F: OptimalFSizer,
    }

    @classmethod
    def create(
        cls,
        method: SizingMethod,
        constraints: SizingConstraints | None = None,
        **kwargs: Any,
    ) -> BasePositionSizer:
        """Create a position sizer of the specified type.

        Args:
            method: Sizing method to use.
            constraints: Position sizing constraints.
            **kwargs: Method-specific parameters.

        Returns:
            Configured position sizer.

        Raises:
            ValueError: If method is not recognized.
        """
        sizer_class = cls._sizers.get(method)
        if sizer_class is None:
            raise ValueError(f"Unknown sizing method: {method}")
        return sizer_class(constraints=constraints, **kwargs)

    @classmethod
    def register(cls, method: SizingMethod, sizer_class: type[BasePositionSizer]) -> None:
        """Register a custom position sizer.

        Args:
            method: Sizing method identifier.
            sizer_class: Position sizer class to register.
        """
        cls._sizers[method] = sizer_class


class CompositePositionSizer(BasePositionSizer):
    """Combines multiple sizing strategies with weights."""

    def __init__(
        self,
        sizers: list[tuple[BasePositionSizer, float]],
        aggregation: str = "min",
        constraints: SizingConstraints | None = None,
    ) -> None:
        """Initialize composite sizer.

        Args:
            sizers: List of (sizer, weight) tuples.
            aggregation: How to combine sizes ('min', 'max', 'mean', 'weighted').
            constraints: Position sizing constraints.
        """
        super().__init__(constraints)
        self.sizers = sizers
        self.aggregation = aggregation

    def calculate_size(
        self,
        signal: TradeSignal,
        portfolio: Portfolio,
        entry_price: Decimal,
        stop_loss_price: Decimal | None = None,
        market_data: dict[str, Any] | None = None,
    ) -> PositionSizeResult:
        """Calculate position size from multiple strategies."""
        if entry_price <= 0:
            raise RiskError("Entry price must be positive", details={"entry_price": float(entry_price)})

        # Get sizes from all sizers
        results = []
        for sizer, weight in self.sizers:
            result = sizer.calculate_size(
                signal, portfolio, entry_price, stop_loss_price, market_data
            )
            results.append((result, weight))

        # Aggregate sizes
        sizes = [float(r.recommended_size) for r, _ in results]
        weights = [w for _, w in results]

        if self.aggregation == "min":
            final_size = Decimal(str(min(sizes)))
        elif self.aggregation == "max":
            final_size = Decimal(str(max(sizes)))
        elif self.aggregation == "mean":
            final_size = Decimal(str(np.mean(sizes)))
        elif self.aggregation == "weighted":
            final_size = Decimal(str(np.average(sizes, weights=weights)))
        else:
            final_size = Decimal(str(min(sizes)))

        # Collect all constraints applied
        all_constraints = []
        for result, _ in results:
            all_constraints.extend(result.constraints_applied)

        return PositionSizeResult(
            symbol=signal.symbol,
            recommended_size=final_size,
            method=SizingMethod.FIXED_FRACTIONAL,  # Use primary method
            raw_size=final_size,
            constraints_applied=list(set(all_constraints)),
            risk_amount=Decimal("0"),
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            confidence=signal.confidence,
            metadata={
                "aggregation": self.aggregation,
                "individual_sizes": sizes,
                "weights": weights,
            },
        )
