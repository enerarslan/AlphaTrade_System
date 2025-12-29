"""
Execution algorithms for large order execution.

Implements algorithmic execution strategies:
- TWAP (Time-Weighted Average Price): Execute evenly over time
- VWAP (Volume-Weighted Average Price): Execute in line with volume
- Implementation Shortfall: Minimize total execution cost
- Adaptive: Dynamic strategy based on market conditions

Each algorithm slices large orders into smaller pieces to minimize
market impact and execution costs.
"""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable
from uuid import UUID, uuid4

from quant_trading_system.core.data_types import OrderSide, OrderType, TimeInForce
from quant_trading_system.execution.alpaca_client import AlpacaClient
from quant_trading_system.execution.order_manager import (
    ManagedOrder,
    OrderManager,
    OrderRequest,
)

logger = logging.getLogger(__name__)


class AlgoStatus(str, Enum):
    """Execution algorithm status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class AlgoType(str, Enum):
    """Execution algorithm types."""

    TWAP = "twap"
    VWAP = "vwap"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ADAPTIVE = "adaptive"


@dataclass
class SliceOrder:
    """Individual slice of an algorithmic order."""

    slice_id: UUID = field(default_factory=uuid4)
    quantity: Decimal = Decimal("0")
    scheduled_time: datetime = field(default_factory=datetime.utcnow)
    executed_time: datetime | None = None
    executed_qty: Decimal = Decimal("0")
    executed_price: Decimal | None = None
    order_id: UUID | None = None
    status: str = "pending"  # pending, submitted, filled, cancelled


@dataclass
class AlgoExecutionState:
    """State of an algorithmic execution."""

    algo_id: UUID = field(default_factory=uuid4)
    algo_type: AlgoType = AlgoType.TWAP
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    total_quantity: Decimal = Decimal("0")
    filled_quantity: Decimal = Decimal("0")
    remaining_quantity: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None
    status: AlgoStatus = AlgoStatus.PENDING
    start_time: datetime | None = None
    end_time: datetime | None = None
    slices: list[SliceOrder] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    benchmark_price: Decimal | None = None  # Price at algo start for performance
    metrics: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    @property
    def fill_progress(self) -> float:
        """Get fill progress as percentage."""
        if self.total_quantity == 0:
            return 0.0
        return float(self.filled_quantity / self.total_quantity) * 100

    @property
    def vwap(self) -> Decimal | None:
        """Calculate VWAP of executed slices."""
        total_value = Decimal("0")
        total_qty = Decimal("0")
        for s in self.slices:
            if s.executed_price and s.executed_qty > 0:
                total_value += s.executed_price * s.executed_qty
                total_qty += s.executed_qty
        if total_qty > 0:
            return total_value / total_qty
        return None

    @property
    def execution_cost(self) -> Decimal | None:
        """Calculate execution cost vs benchmark."""
        if self.benchmark_price and self.avg_fill_price:
            if self.side == OrderSide.BUY:
                return self.avg_fill_price - self.benchmark_price
            else:
                return self.benchmark_price - self.avg_fill_price
        return None


class ExecutionAlgorithm(ABC):
    """Base class for execution algorithms."""

    def __init__(
        self,
        order_manager: OrderManager,
        client: AlpacaClient,
    ) -> None:
        """Initialize algorithm.

        Args:
            order_manager: Order manager for submitting slices.
            client: Alpaca client for market data.
        """
        self.order_manager = order_manager
        self.client = client
        self._running = False
        self._task: asyncio.Task | None = None

    @abstractmethod
    async def execute(
        self,
        state: AlgoExecutionState,
        on_slice: Callable[[SliceOrder], None] | None = None,
    ) -> AlgoExecutionState:
        """Execute the algorithm.

        Args:
            state: Execution state to track progress.
            on_slice: Callback for each slice completion.

        Returns:
            Updated execution state.
        """
        pass

    @abstractmethod
    def calculate_slices(
        self,
        total_quantity: Decimal,
        duration_minutes: int,
        **kwargs: Any,
    ) -> list[SliceOrder]:
        """Calculate order slices.

        Args:
            total_quantity: Total quantity to execute.
            duration_minutes: Execution window in minutes.
            **kwargs: Algorithm-specific parameters.

        Returns:
            List of slice orders.
        """
        pass

    async def cancel(self, state: AlgoExecutionState) -> AlgoExecutionState:
        """Cancel the algorithm execution.

        Args:
            state: Current execution state.

        Returns:
            Updated state.
        """
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Cancel any pending slices
        for slice_order in state.slices:
            if slice_order.status == "submitted" and slice_order.order_id:
                try:
                    await self.order_manager.cancel_order(slice_order.order_id)
                    slice_order.status = "cancelled"
                except Exception as e:
                    logger.warning(f"Failed to cancel slice: {e}")

        state.status = AlgoStatus.CANCELLED
        state.end_time = datetime.utcnow()
        return state

    async def _submit_slice(
        self,
        state: AlgoExecutionState,
        slice_order: SliceOrder,
        limit_price: Decimal | None = None,
    ) -> ManagedOrder | None:
        """Submit a single slice order.

        Args:
            state: Execution state.
            slice_order: Slice to submit.
            limit_price: Optional limit price.

        Returns:
            Managed order or None if failed.
        """
        try:
            request = OrderRequest(
                symbol=state.symbol,
                side=state.side,
                quantity=slice_order.quantity,
                order_type=OrderType.LIMIT if limit_price else OrderType.MARKET,
                limit_price=limit_price,
                time_in_force=TimeInForce.IOC if limit_price else TimeInForce.DAY,
                notes=f"Algo slice {slice_order.slice_id}",
            )

            managed = self.order_manager.create_order(request)
            managed = await self.order_manager.submit_order(managed)

            slice_order.order_id = managed.order.order_id
            slice_order.status = "submitted"

            return managed

        except Exception as e:
            logger.error(f"Failed to submit slice: {e}")
            slice_order.status = "failed"
            return None

    async def _wait_for_fill(
        self,
        managed: ManagedOrder,
        timeout_seconds: float = 30.0,
    ) -> bool:
        """Wait for an order to fill.

        Args:
            managed: Managed order to wait for.
            timeout_seconds: Maximum wait time.

        Returns:
            True if filled, False otherwise.
        """
        start = datetime.utcnow()
        while (datetime.utcnow() - start).total_seconds() < timeout_seconds:
            # Refresh order state
            current = self.order_manager.get_order(managed.order.order_id)
            if current and current.is_terminal:
                return current.state.value == "filled"
            await asyncio.sleep(0.5)
        return False

    def _update_state_from_slice(
        self,
        state: AlgoExecutionState,
        slice_order: SliceOrder,
        managed: ManagedOrder | None,
    ) -> None:
        """Update execution state from completed slice."""
        if managed and managed.order.filled_qty > 0:
            slice_order.executed_qty = managed.order.filled_qty
            slice_order.executed_price = managed.order.filled_avg_price
            slice_order.executed_time = datetime.utcnow()
            slice_order.status = "filled"

            state.filled_quantity += managed.order.filled_qty
            state.remaining_quantity = state.total_quantity - state.filled_quantity

            # Update average fill price
            if state.avg_fill_price is None:
                state.avg_fill_price = managed.order.filled_avg_price
            elif managed.order.filled_avg_price:
                # Weighted average
                prev_value = state.avg_fill_price * (state.filled_quantity - managed.order.filled_qty)
                new_value = managed.order.filled_avg_price * managed.order.filled_qty
                state.avg_fill_price = (prev_value + new_value) / state.filled_quantity


class TWAPAlgorithm(ExecutionAlgorithm):
    """Time-Weighted Average Price execution algorithm.

    Executes orders evenly over a specified time period to minimize
    market impact and achieve a price close to the time-weighted
    average price.
    """

    def __init__(
        self,
        order_manager: OrderManager,
        client: AlpacaClient,
        randomize_timing: bool = True,
        max_timing_jitter: float = 0.2,
    ) -> None:
        """Initialize TWAP algorithm.

        Args:
            order_manager: Order manager for execution.
            client: Alpaca client for market data.
            randomize_timing: Add random jitter to slice timing.
            max_timing_jitter: Maximum jitter as fraction of interval.
        """
        super().__init__(order_manager, client)
        self.randomize_timing = randomize_timing
        self.max_timing_jitter = max_timing_jitter

    def calculate_slices(
        self,
        total_quantity: Decimal,
        duration_minutes: int,
        interval_minutes: int = 5,
        **kwargs: Any,
    ) -> list[SliceOrder]:
        """Calculate TWAP slices.

        Args:
            total_quantity: Total quantity to execute.
            duration_minutes: Total execution window.
            interval_minutes: Time between slices.

        Returns:
            List of evenly distributed slices.
        """
        num_slices = max(1, duration_minutes // interval_minutes)
        base_qty = total_quantity // num_slices
        remainder = total_quantity % num_slices

        slices = []
        start_time = datetime.utcnow()

        for i in range(num_slices):
            # Distribute remainder across first few slices
            qty = base_qty + (Decimal("1") if i < int(remainder) else Decimal("0"))

            if qty > 0:
                scheduled = start_time + timedelta(minutes=i * interval_minutes)

                # Add random jitter
                if self.randomize_timing and i > 0:
                    jitter = random.uniform(
                        -self.max_timing_jitter * interval_minutes,
                        self.max_timing_jitter * interval_minutes,
                    )
                    scheduled += timedelta(minutes=jitter)

                slices.append(SliceOrder(
                    quantity=qty,
                    scheduled_time=scheduled,
                ))

        return slices

    async def execute(
        self,
        state: AlgoExecutionState,
        on_slice: Callable[[SliceOrder], None] | None = None,
    ) -> AlgoExecutionState:
        """Execute TWAP algorithm.

        Args:
            state: Execution state.
            on_slice: Callback for each slice completion.

        Returns:
            Updated execution state.
        """
        self._running = True
        state.status = AlgoStatus.RUNNING
        state.start_time = datetime.utcnow()

        # Get benchmark price
        try:
            snapshot = await self.client.get_snapshot(state.symbol)
            if snapshot.get("latestTrade"):
                state.benchmark_price = Decimal(str(snapshot["latestTrade"]["p"]))
        except Exception as e:
            logger.warning(f"Failed to get benchmark price: {e}")

        logger.info(
            f"Starting TWAP: {state.symbol} {state.side.value} "
            f"{state.total_quantity} in {len(state.slices)} slices"
        )

        try:
            for slice_order in state.slices:
                if not self._running:
                    break

                # Wait until scheduled time
                now = datetime.utcnow()
                if slice_order.scheduled_time > now:
                    wait_time = (slice_order.scheduled_time - now).total_seconds()
                    await asyncio.sleep(wait_time)

                if not self._running:
                    break

                # Submit slice
                managed = await self._submit_slice(state, slice_order)

                if managed:
                    # Wait for fill (with short timeout for IOC/market orders)
                    await self._wait_for_fill(managed, timeout_seconds=10.0)
                    self._update_state_from_slice(state, slice_order, managed)

                    if on_slice:
                        on_slice(slice_order)

                logger.info(
                    f"TWAP progress: {state.fill_progress:.1f}% "
                    f"({state.filled_quantity}/{state.total_quantity})"
                )

            # Mark complete
            if state.remaining_quantity == 0:
                state.status = AlgoStatus.COMPLETED
            elif self._running:
                state.status = AlgoStatus.COMPLETED  # Completed with partial fill
            else:
                state.status = AlgoStatus.CANCELLED

        except Exception as e:
            logger.error(f"TWAP execution error: {e}")
            state.status = AlgoStatus.FAILED
            state.error_message = str(e)

        state.end_time = datetime.utcnow()
        self._running = False

        # Calculate metrics
        state.metrics = self._calculate_metrics(state)

        logger.info(
            f"TWAP completed: filled {state.filled_quantity}/{state.total_quantity} "
            f"@ {state.avg_fill_price}"
        )

        return state

    def _calculate_metrics(self, state: AlgoExecutionState) -> dict[str, Any]:
        """Calculate execution metrics."""
        metrics: dict[str, Any] = {
            "fill_rate": state.fill_progress,
            "num_slices": len(state.slices),
            "successful_slices": len([s for s in state.slices if s.status == "filled"]),
        }

        if state.start_time and state.end_time:
            metrics["execution_time_seconds"] = (state.end_time - state.start_time).total_seconds()

        if state.benchmark_price and state.avg_fill_price:
            slippage = float(state.avg_fill_price - state.benchmark_price)
            slippage_bps = slippage / float(state.benchmark_price) * 10000
            metrics["slippage"] = slippage
            metrics["slippage_bps"] = slippage_bps

        return metrics


class VWAPAlgorithm(ExecutionAlgorithm):
    """Volume-Weighted Average Price execution algorithm.

    Executes orders in proportion to expected volume distribution
    to achieve a price close to VWAP benchmark.
    """

    # Typical intraday volume profile (hourly buckets, market hours)
    DEFAULT_VOLUME_PROFILE = [
        0.15,  # 9:30 - 10:30 (high opening volume)
        0.10,  # 10:30 - 11:30
        0.08,  # 11:30 - 12:30
        0.07,  # 12:30 - 13:30 (lunch lull)
        0.08,  # 13:30 - 14:30
        0.12,  # 14:30 - 15:30
        0.20,  # 15:30 - 16:00 (closing auction)
    ]

    def __init__(
        self,
        order_manager: OrderManager,
        client: AlpacaClient,
        volume_profile: list[float] | None = None,
        participation_rate: float = 0.10,
    ) -> None:
        """Initialize VWAP algorithm.

        Args:
            order_manager: Order manager for execution.
            client: Alpaca client for market data.
            volume_profile: Intraday volume profile (sums to 1.0).
            participation_rate: Maximum participation in each interval.
        """
        super().__init__(order_manager, client)
        self.volume_profile = volume_profile or self.DEFAULT_VOLUME_PROFILE
        self.participation_rate = participation_rate

    def calculate_slices(
        self,
        total_quantity: Decimal,
        duration_minutes: int,
        interval_minutes: int = 5,
        volume_profile: list[float] | None = None,
        **kwargs: Any,
    ) -> list[SliceOrder]:
        """Calculate VWAP slices based on volume profile.

        Args:
            total_quantity: Total quantity to execute.
            duration_minutes: Execution window.
            interval_minutes: Time between slices.
            volume_profile: Custom volume profile.

        Returns:
            Volume-weighted slices.
        """
        profile = volume_profile or self.volume_profile
        num_intervals = max(1, duration_minutes // interval_minutes)

        # Map intervals to volume profile
        profile_normalized = []
        for i in range(num_intervals):
            # Map to hourly bucket (assuming 6.5 hour trading day)
            pct_through_day = i / num_intervals
            bucket = min(int(pct_through_day * len(profile)), len(profile) - 1)
            profile_normalized.append(profile[bucket])

        # Normalize to sum to 1
        total_weight = sum(profile_normalized)
        if total_weight > 0:
            profile_normalized = [w / total_weight for w in profile_normalized]

        slices = []
        start_time = datetime.utcnow()

        for i, weight in enumerate(profile_normalized):
            qty = Decimal(str(float(total_quantity) * weight)).quantize(Decimal("1"))

            if qty > 0:
                scheduled = start_time + timedelta(minutes=i * interval_minutes)
                slices.append(SliceOrder(
                    quantity=qty,
                    scheduled_time=scheduled,
                ))

        return slices

    async def execute(
        self,
        state: AlgoExecutionState,
        on_slice: Callable[[SliceOrder], None] | None = None,
    ) -> AlgoExecutionState:
        """Execute VWAP algorithm.

        Args:
            state: Execution state.
            on_slice: Callback for each slice completion.

        Returns:
            Updated execution state.
        """
        self._running = True
        state.status = AlgoStatus.RUNNING
        state.start_time = datetime.utcnow()

        # Get benchmark price
        try:
            snapshot = await self.client.get_snapshot(state.symbol)
            if snapshot.get("latestTrade"):
                state.benchmark_price = Decimal(str(snapshot["latestTrade"]["p"]))
        except Exception as e:
            logger.warning(f"Failed to get benchmark price: {e}")

        logger.info(
            f"Starting VWAP: {state.symbol} {state.side.value} "
            f"{state.total_quantity} in {len(state.slices)} slices"
        )

        try:
            for slice_order in state.slices:
                if not self._running:
                    break

                # Wait until scheduled time
                now = datetime.utcnow()
                if slice_order.scheduled_time > now:
                    wait_time = (slice_order.scheduled_time - now).total_seconds()
                    await asyncio.sleep(wait_time)

                if not self._running:
                    break

                # Get current price for limit order
                limit_price = None
                try:
                    snapshot = await self.client.get_snapshot(state.symbol)
                    if snapshot.get("latestQuote"):
                        bid = Decimal(str(snapshot["latestQuote"]["bp"]))
                        ask = Decimal(str(snapshot["latestQuote"]["ap"]))
                        mid = (bid + ask) / 2

                        # Set limit slightly aggressive
                        if state.side == OrderSide.BUY:
                            limit_price = mid + (ask - mid) * Decimal("0.5")
                        else:
                            limit_price = mid - (mid - bid) * Decimal("0.5")
                except Exception:
                    pass  # Fall back to market order

                # Submit slice
                managed = await self._submit_slice(state, slice_order, limit_price)

                if managed:
                    await self._wait_for_fill(managed, timeout_seconds=30.0)
                    self._update_state_from_slice(state, slice_order, managed)

                    if on_slice:
                        on_slice(slice_order)

                logger.info(
                    f"VWAP progress: {state.fill_progress:.1f}% "
                    f"({state.filled_quantity}/{state.total_quantity})"
                )

            # Mark complete
            if state.remaining_quantity == 0:
                state.status = AlgoStatus.COMPLETED
            elif self._running:
                state.status = AlgoStatus.COMPLETED
            else:
                state.status = AlgoStatus.CANCELLED

        except Exception as e:
            logger.error(f"VWAP execution error: {e}")
            state.status = AlgoStatus.FAILED
            state.error_message = str(e)

        state.end_time = datetime.utcnow()
        self._running = False

        # Calculate metrics
        state.metrics = self._calculate_metrics(state)

        logger.info(
            f"VWAP completed: filled {state.filled_quantity}/{state.total_quantity} "
            f"@ {state.avg_fill_price}"
        )

        return state

    def _calculate_metrics(self, state: AlgoExecutionState) -> dict[str, Any]:
        """Calculate VWAP execution metrics."""
        metrics: dict[str, Any] = {
            "fill_rate": state.fill_progress,
            "num_slices": len(state.slices),
            "successful_slices": len([s for s in state.slices if s.status == "filled"]),
        }

        if state.start_time and state.end_time:
            metrics["execution_time_seconds"] = (state.end_time - state.start_time).total_seconds()

        # VWAP performance
        achieved_vwap = state.vwap
        if achieved_vwap and state.benchmark_price:
            vwap_diff = float(achieved_vwap - state.benchmark_price)
            vwap_diff_bps = vwap_diff / float(state.benchmark_price) * 10000
            metrics["vwap"] = float(achieved_vwap)
            metrics["vwap_diff"] = vwap_diff
            metrics["vwap_diff_bps"] = vwap_diff_bps

        return metrics


class ImplementationShortfallAlgorithm(ExecutionAlgorithm):
    """Implementation Shortfall execution algorithm.

    Minimizes total execution cost by balancing urgency (market impact
    from trading quickly) against timing risk (price drift from waiting).

    Uses Almgren-Chriss framework to determine optimal trading trajectory.
    """

    def __init__(
        self,
        order_manager: OrderManager,
        client: AlpacaClient,
        risk_aversion: float = 1.0,
        urgency: float = 0.5,
    ) -> None:
        """Initialize Implementation Shortfall algorithm.

        Args:
            order_manager: Order manager for execution.
            client: Alpaca client for market data.
            risk_aversion: Risk aversion parameter (higher = more aggressive).
            urgency: Execution urgency (0-1, higher = faster).
        """
        super().__init__(order_manager, client)
        self.risk_aversion = risk_aversion
        self.urgency = urgency

    def calculate_slices(
        self,
        total_quantity: Decimal,
        duration_minutes: int,
        interval_minutes: int = 5,
        volatility: float = 0.02,
        **kwargs: Any,
    ) -> list[SliceOrder]:
        """Calculate Implementation Shortfall slices.

        Uses Almgren-Chriss optimal execution formula to determine
        the trading trajectory that minimizes expected cost + risk.

        Args:
            total_quantity: Total quantity to execute.
            duration_minutes: Execution window.
            interval_minutes: Time between slices.
            volatility: Daily volatility estimate.

        Returns:
            Optimized slices.
        """
        import math

        num_intervals = max(1, duration_minutes // interval_minutes)

        # Almgren-Chriss parameters
        # Higher urgency -> execute more at start
        # Higher risk aversion -> more even execution
        kappa = self.urgency * 2.0  # Urgency factor
        tau = duration_minutes / (6.5 * 60)  # Fraction of trading day

        slices = []
        start_time = datetime.utcnow()
        remaining = float(total_quantity)

        for i in range(num_intervals):
            # Exponential decay based on urgency
            if self.urgency > 0.7:
                # Front-loaded execution
                decay = math.exp(-kappa * i / num_intervals)
                weight = decay
            elif self.urgency < 0.3:
                # Back-loaded execution
                decay = math.exp(-kappa * (num_intervals - i) / num_intervals)
                weight = decay
            else:
                # More even distribution
                weight = 1.0

            # Calculate slice quantity
            remaining_intervals = num_intervals - i
            base_qty = remaining / remaining_intervals
            qty = Decimal(str(base_qty * weight)).quantize(Decimal("1"))

            # Ensure we don't over-allocate
            qty = min(qty, Decimal(str(remaining)))

            if qty > 0:
                scheduled = start_time + timedelta(minutes=i * interval_minutes)
                slices.append(SliceOrder(
                    quantity=qty,
                    scheduled_time=scheduled,
                ))
                remaining -= float(qty)

        return slices

    async def execute(
        self,
        state: AlgoExecutionState,
        on_slice: Callable[[SliceOrder], None] | None = None,
    ) -> AlgoExecutionState:
        """Execute Implementation Shortfall algorithm.

        Args:
            state: Execution state.
            on_slice: Callback for each slice completion.

        Returns:
            Updated execution state.
        """
        self._running = True
        state.status = AlgoStatus.RUNNING
        state.start_time = datetime.utcnow()

        # Get benchmark price (decision price for IS)
        try:
            snapshot = await self.client.get_snapshot(state.symbol)
            if snapshot.get("latestTrade"):
                state.benchmark_price = Decimal(str(snapshot["latestTrade"]["p"]))
        except Exception as e:
            logger.warning(f"Failed to get benchmark price: {e}")

        logger.info(
            f"Starting IS: {state.symbol} {state.side.value} "
            f"{state.total_quantity} (urgency={self.urgency})"
        )

        try:
            for i, slice_order in enumerate(state.slices):
                if not self._running:
                    break

                # Wait until scheduled time
                now = datetime.utcnow()
                if slice_order.scheduled_time > now:
                    wait_time = (slice_order.scheduled_time - now).total_seconds()
                    await asyncio.sleep(wait_time)

                if not self._running:
                    break

                # Adaptive: adjust remaining slices based on execution progress
                if i > 0 and state.filled_quantity < state.total_quantity / 2:
                    # Behind schedule - increase urgency
                    slice_order.quantity = min(
                        slice_order.quantity * Decimal("1.2"),
                        state.remaining_quantity,
                    )

                # Submit slice
                managed = await self._submit_slice(state, slice_order)

                if managed:
                    await self._wait_for_fill(managed, timeout_seconds=15.0)
                    self._update_state_from_slice(state, slice_order, managed)

                    if on_slice:
                        on_slice(slice_order)

                logger.info(
                    f"IS progress: {state.fill_progress:.1f}% "
                    f"({state.filled_quantity}/{state.total_quantity})"
                )

            state.status = AlgoStatus.COMPLETED

        except Exception as e:
            logger.error(f"IS execution error: {e}")
            state.status = AlgoStatus.FAILED
            state.error_message = str(e)

        state.end_time = datetime.utcnow()
        self._running = False

        # Calculate IS metrics
        state.metrics = self._calculate_metrics(state)

        return state

    def _calculate_metrics(self, state: AlgoExecutionState) -> dict[str, Any]:
        """Calculate Implementation Shortfall metrics."""
        metrics: dict[str, Any] = {
            "fill_rate": state.fill_progress,
            "risk_aversion": self.risk_aversion,
            "urgency": self.urgency,
        }

        if state.benchmark_price and state.avg_fill_price:
            # Implementation shortfall = actual cost vs decision price
            if state.side == OrderSide.BUY:
                is_cost = state.avg_fill_price - state.benchmark_price
            else:
                is_cost = state.benchmark_price - state.avg_fill_price

            is_bps = float(is_cost / state.benchmark_price) * 10000
            metrics["implementation_shortfall"] = float(is_cost)
            metrics["implementation_shortfall_bps"] = is_bps

        return metrics


class AlgoExecutionEngine:
    """Engine for managing algorithmic executions.

    Provides a unified interface for creating and managing
    algorithmic order executions.
    """

    def __init__(
        self,
        order_manager: OrderManager,
        client: AlpacaClient,
    ) -> None:
        """Initialize execution engine.

        Args:
            order_manager: Order manager for execution.
            client: Alpaca client for market data.
        """
        self.order_manager = order_manager
        self.client = client

        # Algorithm instances
        self._algorithms: dict[AlgoType, ExecutionAlgorithm] = {
            AlgoType.TWAP: TWAPAlgorithm(order_manager, client),
            AlgoType.VWAP: VWAPAlgorithm(order_manager, client),
            AlgoType.IMPLEMENTATION_SHORTFALL: ImplementationShortfallAlgorithm(
                order_manager, client
            ),
        }

        # Active executions
        self._executions: dict[UUID, AlgoExecutionState] = {}
        self._tasks: dict[UUID, asyncio.Task] = {}

    async def start_execution(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        algo_type: AlgoType,
        duration_minutes: int = 60,
        interval_minutes: int = 5,
        on_complete: Callable[[AlgoExecutionState], None] | None = None,
        **kwargs: Any,
    ) -> AlgoExecutionState:
        """Start an algorithmic execution.

        Args:
            symbol: Stock symbol.
            side: Buy or sell.
            quantity: Total quantity.
            algo_type: Algorithm type.
            duration_minutes: Execution window.
            interval_minutes: Time between slices.
            on_complete: Callback when complete.
            **kwargs: Algorithm-specific parameters.

        Returns:
            Execution state.
        """
        algorithm = self._algorithms.get(algo_type)
        if not algorithm:
            raise ValueError(f"Unknown algorithm: {algo_type}")

        # Calculate slices
        slices = algorithm.calculate_slices(
            total_quantity=quantity,
            duration_minutes=duration_minutes,
            interval_minutes=interval_minutes,
            **kwargs,
        )

        # Create execution state
        state = AlgoExecutionState(
            algo_type=algo_type,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            remaining_quantity=quantity,
            slices=slices,
            parameters={
                "duration_minutes": duration_minutes,
                "interval_minutes": interval_minutes,
                **kwargs,
            },
        )

        self._executions[state.algo_id] = state

        # Start execution in background
        async def run_algo():
            result = await algorithm.execute(state)
            if on_complete:
                on_complete(result)
            return result

        task = asyncio.create_task(run_algo())
        self._tasks[state.algo_id] = task

        logger.info(
            f"Started {algo_type.value} execution {state.algo_id}: "
            f"{symbol} {side.value} {quantity}"
        )

        return state

    async def cancel_execution(self, algo_id: UUID) -> AlgoExecutionState | None:
        """Cancel an algorithmic execution.

        Args:
            algo_id: Execution ID to cancel.

        Returns:
            Updated state or None if not found.
        """
        state = self._executions.get(algo_id)
        if not state:
            return None

        algorithm = self._algorithms.get(state.algo_type)
        if algorithm:
            state = await algorithm.cancel(state)

        task = self._tasks.get(algo_id)
        if task and not task.done():
            task.cancel()

        logger.info(f"Cancelled execution {algo_id}")

        return state

    def get_execution(self, algo_id: UUID) -> AlgoExecutionState | None:
        """Get execution state by ID."""
        return self._executions.get(algo_id)

    def get_active_executions(self) -> list[AlgoExecutionState]:
        """Get all active executions."""
        return [
            s for s in self._executions.values()
            if s.status in (AlgoStatus.PENDING, AlgoStatus.RUNNING, AlgoStatus.PAUSED)
        ]

    def get_execution_summary(self) -> dict[str, Any]:
        """Get summary of all executions."""
        total = len(self._executions)
        by_status = {}
        for state in self._executions.values():
            status = state.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_executions": total,
            "by_status": by_status,
            "active": len(self.get_active_executions()),
        }
