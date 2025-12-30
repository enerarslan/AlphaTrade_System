"""
Market simulation module for backtesting.

Provides realistic execution simulation including:
- Multiple slippage models (fixed, volume-based, volatility-based)
- Market impact modeling (square-root, Almgren-Chriss)
- Bid-ask spread simulation
- Fill probability and partial fills
- Latency simulation
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np

from quant_trading_system.core.data_types import Order, OrderSide, OrderType

logger = logging.getLogger(__name__)


class SlippageModel(str, Enum):
    """Slippage model types."""

    FIXED = "fixed"
    VOLUME_BASED = "volume_based"
    VOLATILITY_BASED = "volatility_based"
    MARKET_IMPACT = "market_impact"


class FillType(str, Enum):
    """Order fill types."""

    FULL = "full"
    PARTIAL = "partial"
    REJECTED = "rejected"


@dataclass
class MarketConditions:
    """Current market conditions for simulation."""

    price: Decimal
    bid: Decimal | None = None
    ask: Decimal | None = None
    volume: int = 0
    avg_daily_volume: int = 1000000
    volatility: float = 0.02  # Daily volatility
    spread_bps: float = 5.0  # Bid-ask spread in bps
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def spread(self) -> Decimal:
        """Get bid-ask spread."""
        if self.bid and self.ask:
            return self.ask - self.bid
        return self.price * Decimal(str(self.spread_bps / 10000))

    @property
    def mid_price(self) -> Decimal:
        """Get mid price."""
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.price


@dataclass
class FillResult:
    """Result of a fill simulation."""

    fill_type: FillType
    fill_price: Decimal
    fill_quantity: Decimal
    slippage: Decimal
    market_impact: Decimal
    commission: Decimal
    latency_ms: float
    timestamp: datetime
    partial_remaining: Decimal = Decimal("0")
    rejection_reason: str | None = None

    @property
    def total_cost(self) -> Decimal:
        """Total execution cost."""
        return self.slippage + self.market_impact + self.commission

    @property
    def effective_price(self) -> Decimal:
        """Effective price after all costs."""
        return self.fill_price + self.total_cost / self.fill_quantity if self.fill_quantity > 0 else self.fill_price

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fill_type": self.fill_type.value,
            "fill_price": float(self.fill_price),
            "fill_quantity": float(self.fill_quantity),
            "slippage": float(self.slippage),
            "market_impact": float(self.market_impact),
            "commission": float(self.commission),
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "partial_remaining": float(self.partial_remaining),
            "rejection_reason": self.rejection_reason,
        }


class BaseSlippageModel(ABC):
    """Base class for slippage models."""

    @abstractmethod
    def calculate_slippage(
        self,
        order: Order,
        conditions: MarketConditions,
    ) -> Decimal:
        """Calculate slippage for an order.

        Args:
            order: Order to calculate slippage for.
            conditions: Current market conditions.

        Returns:
            Slippage amount (positive = worse execution).
        """
        pass


class FixedSlippageModel(BaseSlippageModel):
    """Fixed basis points slippage model."""

    def __init__(self, slippage_bps: float = 5.0) -> None:
        """Initialize fixed slippage model.

        Args:
            slippage_bps: Fixed slippage in basis points.
        """
        self.slippage_bps = slippage_bps

    def calculate_slippage(
        self,
        order: Order,
        conditions: MarketConditions,
    ) -> Decimal:
        """Calculate fixed slippage."""
        return conditions.price * Decimal(str(self.slippage_bps / 10000))


class VolumeBasedSlippageModel(BaseSlippageModel):
    """Volume-based slippage model.

    Larger orders relative to volume experience more slippage.
    """

    def __init__(
        self,
        base_slippage_bps: float = 5.0,
        volume_factor: float = 10.0,
        max_slippage_bps: float = 50.0,
    ) -> None:
        """Initialize volume-based slippage model.

        Args:
            base_slippage_bps: Base slippage in basis points.
            volume_factor: Multiplier for volume impact.
            max_slippage_bps: Maximum slippage cap.
        """
        self.base_slippage_bps = base_slippage_bps
        self.volume_factor = volume_factor
        self.max_slippage_bps = max_slippage_bps

    def calculate_slippage(
        self,
        order: Order,
        conditions: MarketConditions,
    ) -> Decimal:
        """Calculate volume-based slippage.

        Formula: slippage_bps = base + (order_size / avg_volume) × factor

        Example: For volume_factor=10 and 1% of daily volume:
        slippage_bps = 5 + 0.01 × 10 = 5.1 bps

        BUG FIX: Removed erroneous *10000 multiplier which caused
        slippage to be 10000x too large (e.g., 1005 bps instead of 5.1 bps).
        """
        if conditions.avg_daily_volume <= 0:
            return conditions.price * Decimal(str(self.base_slippage_bps / 10000))

        volume_ratio = float(order.quantity) / conditions.avg_daily_volume
        # BUG FIX: Removed *10000 multiplier - slippage_bps already in basis points
        slippage_bps = self.base_slippage_bps + volume_ratio * self.volume_factor
        slippage_bps = min(slippage_bps, self.max_slippage_bps)

        return conditions.price * Decimal(str(slippage_bps / 10000))


class VolatilitySlippageModel(BaseSlippageModel):
    """Volatility-based slippage model.

    Higher volatility leads to higher slippage.
    """

    def __init__(
        self,
        base_slippage_bps: float = 5.0,
        avg_volatility: float = 0.02,
        max_slippage_bps: float = 50.0,
    ) -> None:
        """Initialize volatility-based slippage model.

        Args:
            base_slippage_bps: Base slippage at average volatility.
            avg_volatility: Average daily volatility for scaling.
            max_slippage_bps: Maximum slippage cap.
        """
        self.base_slippage_bps = base_slippage_bps
        self.avg_volatility = avg_volatility
        self.max_slippage_bps = max_slippage_bps

    def calculate_slippage(
        self,
        order: Order,
        conditions: MarketConditions,
    ) -> Decimal:
        """Calculate volatility-based slippage.

        Formula: slippage = base × (current_vol / avg_vol)
        """
        vol_ratio = conditions.volatility / self.avg_volatility if self.avg_volatility > 0 else 1.0
        slippage_bps = self.base_slippage_bps * vol_ratio
        slippage_bps = min(slippage_bps, self.max_slippage_bps)

        return conditions.price * Decimal(str(slippage_bps / 10000))


class MarketImpactModel(BaseSlippageModel):
    """Square-root market impact model.

    Implements I = σ × √(Q/V) × γ where:
    - σ: volatility
    - Q: order quantity
    - V: average volume
    - γ: permanent impact factor
    """

    def __init__(
        self,
        permanent_factor: float = 0.1,
        temporary_factor: float = 0.5,
        max_impact_pct: float = 0.02,
    ) -> None:
        """Initialize market impact model.

        Args:
            permanent_factor: Permanent impact coefficient.
            temporary_factor: Temporary impact coefficient.
            max_impact_pct: Maximum impact as percentage of price.
        """
        self.permanent_factor = permanent_factor
        self.temporary_factor = temporary_factor
        self.max_impact_pct = max_impact_pct

    def calculate_slippage(
        self,
        order: Order,
        conditions: MarketConditions,
    ) -> Decimal:
        """Calculate market impact using square-root model."""
        if conditions.avg_daily_volume <= 0:
            return Decimal("0")

        # Participation rate
        participation = float(order.quantity) / conditions.avg_daily_volume

        # Square-root impact
        impact_pct = conditions.volatility * np.sqrt(participation) * self.temporary_factor
        impact_pct = min(impact_pct, self.max_impact_pct)

        return conditions.price * Decimal(str(impact_pct))

    def calculate_permanent_impact(
        self,
        order: Order,
        conditions: MarketConditions,
    ) -> Decimal:
        """Calculate permanent price impact."""
        if conditions.avg_daily_volume <= 0:
            return Decimal("0")

        participation = float(order.quantity) / conditions.avg_daily_volume
        impact_pct = conditions.volatility * np.sqrt(participation) * self.permanent_factor
        impact_pct = min(impact_pct, self.max_impact_pct / 2)

        return conditions.price * Decimal(str(impact_pct))


class BidAskSimulator:
    """Simulates bid-ask spread and execution at bid/ask."""

    def __init__(
        self,
        base_spread_bps: float = 5.0,
        volatility_spread_factor: float = 2.0,
        time_of_day_factor: bool = True,
    ) -> None:
        """Initialize bid-ask simulator.

        Args:
            base_spread_bps: Base spread in basis points.
            volatility_spread_factor: Spread multiplier for volatility.
            time_of_day_factor: Apply time-of-day spread widening.
        """
        self.base_spread_bps = base_spread_bps
        self.volatility_spread_factor = volatility_spread_factor
        self.time_of_day_factor = time_of_day_factor

    def get_bid_ask(
        self,
        mid_price: Decimal,
        conditions: MarketConditions,
    ) -> tuple[Decimal, Decimal]:
        """Get simulated bid and ask prices.

        Args:
            mid_price: Mid/reference price.
            conditions: Current market conditions.

        Returns:
            Tuple of (bid, ask).
        """
        # Base spread
        spread_bps = self.base_spread_bps

        # Volatility adjustment
        vol_factor = 1 + (conditions.volatility / 0.02 - 1) * self.volatility_spread_factor
        spread_bps *= max(0.5, vol_factor)

        # Time of day adjustment (wider at open/close)
        if self.time_of_day_factor:
            hour = conditions.timestamp.hour
            if hour < 10 or hour >= 15:  # First hour or last hour
                spread_bps *= 1.5

        half_spread = mid_price * Decimal(str(spread_bps / 20000))
        bid = mid_price - half_spread
        ask = mid_price + half_spread

        return bid, ask

    def get_execution_price(
        self,
        order: Order,
        conditions: MarketConditions,
    ) -> Decimal:
        """Get execution price based on order side.

        Args:
            order: Order to execute.
            conditions: Market conditions.

        Returns:
            Execution price (ask for buys, bid for sells).
        """
        if conditions.bid and conditions.ask:
            bid, ask = conditions.bid, conditions.ask
        else:
            bid, ask = self.get_bid_ask(conditions.mid_price, conditions)

        if order.side == OrderSide.BUY:
            return ask
        return bid


class FillSimulator:
    """Simulates order fills with partial fills and rejections."""

    def __init__(
        self,
        fill_probability: float = 0.95,
        partial_fill_probability: float = 0.10,
        min_partial_pct: float = 0.25,
        max_partial_pct: float = 0.75,
    ) -> None:
        """Initialize fill simulator.

        Args:
            fill_probability: Base probability of fill.
            partial_fill_probability: Probability of partial fill.
            min_partial_pct: Minimum partial fill percentage.
            max_partial_pct: Maximum partial fill percentage.
        """
        self.fill_probability = fill_probability
        self.partial_fill_probability = partial_fill_probability
        self.min_partial_pct = min_partial_pct
        self.max_partial_pct = max_partial_pct

    def simulate_fill(
        self,
        order: Order,
        conditions: MarketConditions,
    ) -> tuple[FillType, Decimal]:
        """Simulate fill type and quantity.

        CRITICAL FIX: Use local variables for probability adjustments to avoid
        mutable state bug where probabilities would permanently change between calls.

        Args:
            order: Order to simulate.
            conditions: Market conditions.

        Returns:
            Tuple of (fill_type, fill_quantity).
        """
        # CRITICAL FIX: Use local variables, not instance attributes
        # This prevents the mutable state bug where probabilities accumulate
        fill_prob = self.fill_probability
        partial_prob = self.partial_fill_probability

        # Check volume constraint - adjust LOCAL probabilities only
        if conditions.avg_daily_volume > 0:
            participation_rate = float(order.quantity) / conditions.avg_daily_volume
            if participation_rate > 0.10:  # >10% ADV
                # High likelihood of partial fill or rejection
                fill_prob *= (1 - participation_rate)
                partial_prob = min(1.0, partial_prob * 2)

        # Determine fill type
        rand = np.random.random()

        if rand > fill_prob:
            return FillType.REJECTED, Decimal("0")

        if np.random.random() < partial_prob:
            fill_pct = np.random.uniform(self.min_partial_pct, self.max_partial_pct)
            return FillType.PARTIAL, order.quantity * Decimal(str(fill_pct))

        return FillType.FULL, order.quantity


class LatencySimulator:
    """Simulates order processing and execution latency."""

    def __init__(
        self,
        signal_latency_ms: tuple[float, float] = (50, 200),
        transmission_latency_ms: tuple[float, float] = (10, 50),
        exchange_latency_ms: tuple[float, float] = (1, 10),
        fill_latency_ms: tuple[float, float] = (10, 50),
    ) -> None:
        """Initialize latency simulator.

        Args:
            signal_latency_ms: Signal generation latency range.
            transmission_latency_ms: Order transmission latency range.
            exchange_latency_ms: Exchange processing latency range.
            fill_latency_ms: Fill confirmation latency range.
        """
        self.signal_latency_ms = signal_latency_ms
        self.transmission_latency_ms = transmission_latency_ms
        self.exchange_latency_ms = exchange_latency_ms
        self.fill_latency_ms = fill_latency_ms

    def get_total_latency(self) -> float:
        """Get total round-trip latency in milliseconds."""
        signal = np.random.uniform(*self.signal_latency_ms)
        transmission = np.random.uniform(*self.transmission_latency_ms)
        exchange = np.random.uniform(*self.exchange_latency_ms)
        fill = np.random.uniform(*self.fill_latency_ms)
        return signal + transmission + exchange + fill

    def get_execution_delay(self) -> timedelta:
        """Get execution delay as timedelta."""
        latency_ms = self.get_total_latency()
        return timedelta(milliseconds=latency_ms)


class MarketSimulator:
    """Complete market simulation combining all components."""

    def __init__(
        self,
        slippage_model: BaseSlippageModel | None = None,
        bid_ask_simulator: BidAskSimulator | None = None,
        fill_simulator: FillSimulator | None = None,
        latency_simulator: LatencySimulator | None = None,
        commission_bps: float = 5.0,
    ) -> None:
        """Initialize market simulator.

        Args:
            slippage_model: Slippage model to use.
            bid_ask_simulator: Bid-ask spread simulator.
            fill_simulator: Fill simulator.
            latency_simulator: Latency simulator.
            commission_bps: Commission in basis points.
        """
        self.slippage_model = slippage_model or VolumeBasedSlippageModel()
        self.bid_ask_simulator = bid_ask_simulator or BidAskSimulator()
        self.fill_simulator = fill_simulator or FillSimulator()
        self.latency_simulator = latency_simulator or LatencySimulator()
        self.commission_bps = commission_bps

    def simulate_execution(
        self,
        order: Order,
        conditions: MarketConditions,
    ) -> FillResult:
        """Simulate complete order execution.

        Args:
            order: Order to execute.
            conditions: Market conditions.

        Returns:
            Fill result with all execution details.
        """
        # Get latency
        latency_ms = self.latency_simulator.get_total_latency()
        execution_time = conditions.timestamp + timedelta(milliseconds=latency_ms)

        # Simulate fill
        fill_type, fill_quantity = self.fill_simulator.simulate_fill(order, conditions)

        if fill_type == FillType.REJECTED:
            return FillResult(
                fill_type=FillType.REJECTED,
                fill_price=Decimal("0"),
                fill_quantity=Decimal("0"),
                slippage=Decimal("0"),
                market_impact=Decimal("0"),
                commission=Decimal("0"),
                latency_ms=latency_ms,
                timestamp=execution_time,
                partial_remaining=order.quantity,
                rejection_reason="Insufficient liquidity",
            )

        # Get base execution price (at bid/ask)
        base_price = self.bid_ask_simulator.get_execution_price(order, conditions)

        # Calculate slippage
        slippage = self.slippage_model.calculate_slippage(order, conditions)

        # Calculate market impact if using impact model
        market_impact = Decimal("0")
        if isinstance(self.slippage_model, MarketImpactModel):
            market_impact = self.slippage_model.calculate_permanent_impact(order, conditions)

        # Apply slippage to price
        if order.side == OrderSide.BUY:
            fill_price = base_price + slippage
        else:
            fill_price = base_price - slippage

        # Calculate commission
        trade_value = fill_quantity * fill_price
        commission = trade_value * Decimal(str(self.commission_bps / 10000))

        # Calculate remaining for partial fills
        partial_remaining = Decimal("0")
        if fill_type == FillType.PARTIAL:
            partial_remaining = order.quantity - fill_quantity

        return FillResult(
            fill_type=fill_type,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            slippage=slippage * fill_quantity,
            market_impact=market_impact * fill_quantity,
            commission=commission,
            latency_ms=latency_ms,
            timestamp=execution_time,
            partial_remaining=partial_remaining,
        )


class SlippageModelFactory:
    """Factory for creating slippage models."""

    _models: dict[SlippageModel, type[BaseSlippageModel]] = {
        SlippageModel.FIXED: FixedSlippageModel,
        SlippageModel.VOLUME_BASED: VolumeBasedSlippageModel,
        SlippageModel.VOLATILITY_BASED: VolatilitySlippageModel,
        SlippageModel.MARKET_IMPACT: MarketImpactModel,
    }

    @classmethod
    def create(
        cls,
        model_type: SlippageModel,
        **kwargs: Any,
    ) -> BaseSlippageModel:
        """Create a slippage model.

        Args:
            model_type: Type of slippage model.
            **kwargs: Model-specific parameters.

        Returns:
            Configured slippage model.

        Raises:
            ValueError: If model type is not recognized.
        """
        model_class = cls._models.get(model_type)
        if model_class is None:
            raise ValueError(f"Unknown slippage model: {model_type}")
        return model_class(**kwargs)


def create_realistic_simulator(
    commission_bps: float = 5.0,
    base_slippage_bps: float = 5.0,
    base_spread_bps: float = 5.0,
) -> MarketSimulator:
    """Create a simulator with realistic defaults.

    Args:
        commission_bps: Commission in basis points.
        base_slippage_bps: Base slippage in basis points.
        base_spread_bps: Base spread in basis points.

    Returns:
        Configured market simulator.
    """
    return MarketSimulator(
        slippage_model=VolumeBasedSlippageModel(
            base_slippage_bps=base_slippage_bps,
            volume_factor=10.0,
            max_slippage_bps=50.0,
        ),
        bid_ask_simulator=BidAskSimulator(
            base_spread_bps=base_spread_bps,
            volatility_spread_factor=2.0,
        ),
        fill_simulator=FillSimulator(
            fill_probability=0.98,
            partial_fill_probability=0.05,
        ),
        latency_simulator=LatencySimulator(),
        commission_bps=commission_bps,
    )


def create_optimistic_simulator() -> MarketSimulator:
    """Create a simulator with optimistic assumptions (no costs)."""
    return MarketSimulator(
        slippage_model=FixedSlippageModel(slippage_bps=0.0),
        bid_ask_simulator=BidAskSimulator(base_spread_bps=0.0),
        fill_simulator=FillSimulator(
            fill_probability=1.0,
            partial_fill_probability=0.0,
        ),
        latency_simulator=LatencySimulator(
            signal_latency_ms=(0, 0),
            transmission_latency_ms=(0, 0),
            exchange_latency_ms=(0, 0),
            fill_latency_ms=(0, 0),
        ),
        commission_bps=0.0,
    )


def create_pessimistic_simulator(
    commission_bps: float = 20.0,
    base_slippage_bps: float = 20.0,
    base_spread_bps: float = 20.0,
) -> MarketSimulator:
    """Create a simulator with pessimistic assumptions (high costs)."""
    return MarketSimulator(
        slippage_model=VolumeBasedSlippageModel(
            base_slippage_bps=base_slippage_bps,
            volume_factor=20.0,
            max_slippage_bps=100.0,
        ),
        bid_ask_simulator=BidAskSimulator(
            base_spread_bps=base_spread_bps,
            volatility_spread_factor=3.0,
        ),
        fill_simulator=FillSimulator(
            fill_probability=0.90,
            partial_fill_probability=0.20,
        ),
        latency_simulator=LatencySimulator(
            signal_latency_ms=(100, 500),
            transmission_latency_ms=(50, 200),
            exchange_latency_ms=(5, 50),
            fill_latency_ms=(50, 200),
        ),
        commission_bps=commission_bps,
    )
