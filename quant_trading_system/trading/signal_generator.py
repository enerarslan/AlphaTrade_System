"""
Signal generation and management module.

Orchestrates signal generation from multiple sources:
- Trading strategies
- ML model predictions
- Technical indicators
- External signals

Handles signal aggregation, filtering, and conflict resolution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable
from uuid import UUID

import numpy as np

from quant_trading_system.core.data_types import (
    Direction,
    FeatureVector,
    ModelPrediction,
    OHLCVBar,
    Portfolio,
    TradeSignal,
)
from quant_trading_system.core.events import (
    Event,
    EventBus,
    EventType,
    create_signal_event,
)
from quant_trading_system.trading.strategy import Strategy, StrategyConfig

logger = logging.getLogger(__name__)


class SignalPriority(int, Enum):
    """Signal priority levels."""

    CRITICAL = 0  # Risk-reducing signals
    HIGH = 1  # Strong model signals
    NORMAL = 2  # Regular signals
    LOW = 3  # Weak signals


class ConflictResolution(str, Enum):
    """How to resolve conflicting signals."""

    STRONGEST = "strongest"  # Use highest strength
    MOST_CONFIDENT = "most_confident"  # Use highest confidence
    MOST_RECENT = "most_recent"  # Use most recent
    AVERAGE = "average"  # Average conflicting signals
    CANCEL = "cancel"  # Cancel out conflicts


@dataclass
class SignalFilter:
    """Filter criteria for signals."""

    min_confidence: float = 0.5
    min_strength: float = 0.3
    max_age_seconds: int = 300
    symbols: set[str] | None = None
    directions: set[Direction] | None = None
    sources: set[str] | None = None


@dataclass
class SignalMetadata:
    """Additional metadata for a signal."""

    priority: SignalPriority = SignalPriority.NORMAL
    expires_at: datetime | None = None
    parent_signal_id: UUID | None = None
    correlated_symbols: list[str] = field(default_factory=list)
    risk_score: float = 0.0
    expected_pnl: Decimal | None = None
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    notes: str = ""


@dataclass
class EnrichedSignal:
    """Signal with additional context and metadata."""

    signal: TradeSignal
    metadata: SignalMetadata = field(default_factory=SignalMetadata)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processed: bool = False
    executed: bool = False

    @property
    def is_expired(self) -> bool:
        """Check if signal has expired."""
        if self.metadata.expires_at:
            return datetime.now(timezone.utc) > self.metadata.expires_at
        return False

    @property
    def is_actionable(self) -> bool:
        """Check if signal is still actionable."""
        return not self.is_expired and not self.processed


@dataclass
class SignalGeneratorConfig:
    """Signal generator configuration."""

    conflict_resolution: ConflictResolution = ConflictResolution.STRONGEST
    default_signal_ttl: int = 300  # seconds
    max_signals_per_symbol: int = 5
    enable_deduplication: bool = True
    dedup_window_seconds: int = 60
    min_signal_interval: int = 5  # seconds between signals per symbol
    aggregate_sources: bool = True
    max_history_size: int = 10000  # Bound signal history to prevent memory leak


class SignalAggregator:
    """Aggregates signals from multiple sources."""

    def __init__(
        self,
        resolution: ConflictResolution = ConflictResolution.STRONGEST,
    ) -> None:
        """Initialize aggregator.

        Args:
            resolution: Conflict resolution method.
        """
        self.resolution = resolution

    def aggregate(
        self,
        signals: list[TradeSignal],
        symbol: str,
    ) -> TradeSignal | None:
        """Aggregate signals for a symbol.

        Args:
            signals: List of signals to aggregate.
            symbol: Target symbol.

        Returns:
            Aggregated signal or None.
        """
        if not signals:
            return None

        # Filter to target symbol
        symbol_signals = [s for s in signals if s.symbol == symbol]
        if not symbol_signals:
            return None

        if len(symbol_signals) == 1:
            return symbol_signals[0]

        # Separate by direction
        long_signals = [s for s in symbol_signals if s.direction == Direction.LONG]
        short_signals = [s for s in symbol_signals if s.direction == Direction.SHORT]

        # Check for conflict
        if long_signals and short_signals:
            return self._resolve_conflict(long_signals, short_signals)

        # No conflict - aggregate same-direction signals
        if long_signals:
            return self._combine_signals(long_signals, Direction.LONG)
        else:
            return self._combine_signals(short_signals, Direction.SHORT)

    def _resolve_conflict(
        self,
        long_signals: list[TradeSignal],
        short_signals: list[TradeSignal],
    ) -> TradeSignal | None:
        """Resolve conflicting long/short signals."""
        if self.resolution == ConflictResolution.CANCEL:
            return None

        best_long = max(long_signals, key=self._signal_key)
        best_short = max(short_signals, key=self._signal_key)

        if self.resolution == ConflictResolution.STRONGEST:
            if abs(best_long.strength) >= abs(best_short.strength):
                return best_long
            return best_short
        elif self.resolution == ConflictResolution.MOST_CONFIDENT:
            if best_long.confidence >= best_short.confidence:
                return best_long
            return best_short
        elif self.resolution == ConflictResolution.MOST_RECENT:
            if best_long.timestamp >= best_short.timestamp:
                return best_long
            return best_short
        elif self.resolution == ConflictResolution.AVERAGE:
            # Net out the signals
            net_strength = np.mean([s.strength for s in long_signals]) - np.mean(
                [abs(s.strength) for s in short_signals]
            )
            if abs(net_strength) < 0.1:
                return None

            direction = Direction.LONG if net_strength > 0 else Direction.SHORT
            avg_confidence = np.mean([s.confidence for s in long_signals + short_signals])

            return TradeSignal(
                symbol=best_long.symbol,
                direction=direction,
                strength=net_strength,
                confidence=avg_confidence,
                horizon=best_long.horizon,
                model_source="SignalAggregator",
                metadata={"aggregation": "conflict_averaged"},
            )

        return None

    def _combine_signals(
        self,
        signals: list[TradeSignal],
        direction: Direction,
    ) -> TradeSignal:
        """Combine same-direction signals."""
        avg_strength = np.mean([abs(s.strength) for s in signals])
        avg_confidence = np.mean([s.confidence for s in signals])
        max_horizon = max(s.horizon for s in signals)

        # Boost confidence if multiple sources agree
        confidence_boost = min(0.1 * len(signals), 0.2)
        boosted_confidence = min(1.0, avg_confidence + confidence_boost)

        return TradeSignal(
            symbol=signals[0].symbol,
            direction=direction,
            strength=avg_strength if direction == Direction.LONG else -avg_strength,
            confidence=boosted_confidence,
            horizon=max_horizon,
            model_source="SignalAggregator",
            features_snapshot={
                "num_sources": len(signals),
                "sources": [s.model_source for s in signals],
            },
        )

    def _signal_key(self, signal: TradeSignal) -> tuple:
        """Key for sorting/comparing signals."""
        return (signal.confidence, abs(signal.strength), signal.timestamp)


class SignalGenerator:
    """Orchestrates signal generation from multiple sources.

    Manages:
    - Multiple trading strategies
    - Signal aggregation and filtering
    - Conflict resolution
    - Signal lifecycle tracking
    """

    def __init__(
        self,
        config: SignalGeneratorConfig | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize signal generator.

        Args:
            config: Generator configuration.
            event_bus: Event bus for signal events.
        """
        self.config = config or SignalGeneratorConfig()
        self.event_bus = event_bus

        # Signal sources
        self._strategies: dict[str, Strategy] = {}
        self._external_sources: dict[str, Callable[..., list[TradeSignal]]] = {}

        # Signal state
        self._pending_signals: dict[str, list[EnrichedSignal]] = {}
        self._signal_history: list[EnrichedSignal] = []
        self._last_signal_time: dict[str, datetime] = {}

        # Aggregator
        self._aggregator = SignalAggregator(self.config.conflict_resolution)

        # Callbacks
        self._signal_callbacks: list[Callable[[EnrichedSignal], None]] = []

    def add_strategy(self, strategy: Strategy) -> None:
        """Add a trading strategy.

        Args:
            strategy: Strategy to add.
        """
        self._strategies[strategy.strategy_id] = strategy
        strategy.start()
        logger.info(f"Added strategy: {strategy.strategy_id}")

    def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a trading strategy.

        Args:
            strategy_id: Strategy ID to remove.

        Returns:
            True if removed.
        """
        if strategy_id in self._strategies:
            self._strategies[strategy_id].stop()
            del self._strategies[strategy_id]
            return True
        return False

    def add_external_source(
        self,
        source_id: str,
        generator: Callable[..., list[TradeSignal]],
    ) -> None:
        """Add an external signal source.

        Args:
            source_id: Source identifier.
            generator: Function that generates signals.
        """
        self._external_sources[source_id] = generator
        logger.info(f"Added external signal source: {source_id}")

    def generate_signals(
        self,
        bars: dict[str, OHLCVBar],
        features: dict[str, FeatureVector] | None = None,
        predictions: dict[str, ModelPrediction] | None = None,
        portfolio: Portfolio | None = None,
        signal_filter: SignalFilter | None = None,
    ) -> list[EnrichedSignal]:
        """Generate signals from all sources.

        Args:
            bars: Current bars by symbol.
            features: Feature vectors by symbol.
            predictions: Model predictions by symbol.
            portfolio: Current portfolio state.
            signal_filter: Optional filter criteria.

        Returns:
            List of generated signals.
        """
        all_signals: list[TradeSignal] = []

        # Update and generate from strategies
        for strategy in self._strategies.values():
            strategy.update(bars)
            try:
                signals = strategy.generate_signals(bars, features, predictions, portfolio)
                all_signals.extend(signals)
            except Exception as e:
                logger.error(f"Strategy {strategy.strategy_id} error: {e}")

        # Generate from external sources
        for source_id, generator in self._external_sources.items():
            try:
                signals = generator(bars, features, predictions, portfolio)
                all_signals.extend(signals)
            except Exception as e:
                logger.error(f"External source {source_id} error: {e}")

        # Aggregate signals by symbol
        enriched_signals: list[EnrichedSignal] = []
        symbols = set(s.symbol for s in all_signals)

        for symbol in symbols:
            symbol_signals = [s for s in all_signals if s.symbol == symbol]

            # Apply deduplication
            if self.config.enable_deduplication:
                symbol_signals = self._deduplicate(symbol_signals)

            # Aggregate if configured
            if self.config.aggregate_sources and len(symbol_signals) > 1:
                aggregated = self._aggregator.aggregate(symbol_signals, symbol)
                if aggregated:
                    symbol_signals = [aggregated]

            # Rate limit
            if not self._check_rate_limit(symbol):
                continue

            # Enrich signals
            for signal in symbol_signals:
                if self._filter_signal(signal, signal_filter):
                    enriched = self._enrich_signal(signal)
                    enriched_signals.append(enriched)

                    # Track pending signals
                    if symbol not in self._pending_signals:
                        self._pending_signals[symbol] = []
                    self._pending_signals[symbol].append(enriched)

                    # Update rate limit
                    self._last_signal_time[symbol] = datetime.now(timezone.utc)

                    # Invoke callbacks
                    self._dispatch_signal(enriched)

        # Cleanup old signals
        self._cleanup_expired()

        return enriched_signals

    def get_pending_signals(
        self,
        symbol: str | None = None,
        direction: Direction | None = None,
    ) -> list[EnrichedSignal]:
        """Get pending signals.

        Args:
            symbol: Filter by symbol.
            direction: Filter by direction.

        Returns:
            List of pending signals.
        """
        signals = []
        for sym, sym_signals in self._pending_signals.items():
            if symbol and sym != symbol:
                continue
            for sig in sym_signals:
                if sig.is_actionable:
                    if direction is None or sig.signal.direction == direction:
                        signals.append(sig)
        return signals

    def get_signal_for_symbol(self, symbol: str) -> EnrichedSignal | None:
        """Get the best pending signal for a symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Best signal or None.
        """
        signals = self.get_pending_signals(symbol)
        if not signals:
            return None

        # Return highest priority, then highest confidence
        return max(signals, key=lambda s: (
            -s.metadata.priority.value,
            s.signal.confidence,
            abs(s.signal.strength),
        ))

    def mark_signal_processed(
        self,
        signal: EnrichedSignal,
        executed: bool = False,
    ) -> None:
        """Mark a signal as processed.

        Args:
            signal: Signal to mark.
            executed: Whether the signal was executed.
        """
        signal.processed = True
        signal.executed = executed
        self._signal_history.append(signal)

        # Bound history size to prevent memory leak
        if len(self._signal_history) > self.config.max_history_size:
            # Remove oldest 10% when limit exceeded
            trim_count = self.config.max_history_size // 10
            self._signal_history = self._signal_history[trim_count:]

        # Remove from pending
        symbol = signal.signal.symbol
        if symbol in self._pending_signals:
            self._pending_signals[symbol] = [
                s for s in self._pending_signals[symbol]
                if s.signal.signal_id != signal.signal.signal_id
            ]

    def cancel_signal(self, signal_id: UUID) -> bool:
        """Cancel a pending signal.

        Args:
            signal_id: Signal ID to cancel.

        Returns:
            True if cancelled.
        """
        for signals in self._pending_signals.values():
            for signal in signals:
                if signal.signal.signal_id == signal_id:
                    signal.processed = True
                    self._publish_signal_event(
                        signal.signal,
                        EventType.SIGNAL_CANCELLED,
                    )
                    return True
        return False

    def on_signal(self, callback: Callable[[EnrichedSignal], None]) -> None:
        """Register callback for new signals.

        Args:
            callback: Function to call on new signal.
        """
        self._signal_callbacks.append(callback)

    def _enrich_signal(self, signal: TradeSignal) -> EnrichedSignal:
        """Enrich a signal with metadata.

        CRITICAL FIX: Validates and clamps signal bounds to prevent
        invalid signals from corrupting position sizing.
        """
        # CRITICAL FIX: Validate and clamp signal bounds
        # Confidence must be in [0, 1]
        if not (0.0 <= signal.confidence <= 1.0):
            logger.warning(
                f"Signal {signal.symbol} has invalid confidence {signal.confidence}, "
                f"clamping to [0, 1]"
            )
            signal.confidence = max(0.0, min(1.0, signal.confidence))

        # Strength should be in [-1, 1] (allows negative for shorts)
        if not (-1.0 <= signal.strength <= 1.0):
            logger.warning(
                f"Signal {signal.symbol} has invalid strength {signal.strength}, "
                f"clamping to [-1, 1]"
            )
            signal.strength = max(-1.0, min(1.0, signal.strength))

        # Determine priority
        if signal.confidence > 0.8 and abs(signal.strength) > 0.7:
            priority = SignalPriority.HIGH
        elif signal.confidence < 0.5 or abs(signal.strength) < 0.3:
            priority = SignalPriority.LOW
        else:
            priority = SignalPriority.NORMAL

        # Set expiration
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.config.default_signal_ttl)

        metadata = SignalMetadata(
            priority=priority,
            expires_at=expires_at,
        )

        return EnrichedSignal(signal=signal, metadata=metadata)

    def _filter_signal(
        self,
        signal: TradeSignal,
        filter_criteria: SignalFilter | None,
    ) -> bool:
        """Check if signal passes filter.

        Args:
            signal: Signal to check.
            filter_criteria: Filter criteria.

        Returns:
            True if signal passes.
        """
        if filter_criteria is None:
            return True

        if signal.confidence < filter_criteria.min_confidence:
            return False
        if abs(signal.strength) < filter_criteria.min_strength:
            return False
        if filter_criteria.symbols and signal.symbol not in filter_criteria.symbols:
            return False
        if filter_criteria.directions and signal.direction not in filter_criteria.directions:
            return False
        if filter_criteria.sources and signal.model_source not in filter_criteria.sources:
            return False

        return True

    def _deduplicate(self, signals: list[TradeSignal]) -> list[TradeSignal]:
        """Remove duplicate signals.

        Args:
            signals: Signals to deduplicate.

        Returns:
            Deduplicated signals.
        """
        seen: dict[tuple, TradeSignal] = {}

        for signal in signals:
            # Key by symbol + direction + source
            key = (signal.symbol, signal.direction, signal.model_source)

            if key in seen:
                # Keep the stronger/more confident one
                existing = seen[key]
                if signal.confidence > existing.confidence:
                    seen[key] = signal
            else:
                seen[key] = signal

        return list(seen.values())

    def _check_rate_limit(self, symbol: str) -> bool:
        """Check if signal generation is rate limited for symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            True if can generate signal.
        """
        last_time = self._last_signal_time.get(symbol)
        if last_time is None:
            return True

        elapsed = (datetime.now(timezone.utc) - last_time).total_seconds()
        return elapsed >= self.config.min_signal_interval

    def _cleanup_expired(self) -> None:
        """Remove expired pending signals."""
        for symbol in list(self._pending_signals.keys()):
            active = []
            for signal in self._pending_signals[symbol]:
                if signal.is_expired:
                    self._publish_signal_event(
                        signal.signal,
                        EventType.SIGNAL_EXPIRED,
                    )
                else:
                    active.append(signal)

            if active:
                self._pending_signals[symbol] = active
            else:
                del self._pending_signals[symbol]

    def _dispatch_signal(self, signal: EnrichedSignal) -> None:
        """Dispatch signal to callbacks and event bus."""
        # Invoke callbacks
        for callback in self._signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

        # Publish event
        self._publish_signal_event(signal.signal, EventType.SIGNAL_GENERATED)

    def _publish_signal_event(
        self,
        signal: TradeSignal,
        event_type: EventType,
    ) -> None:
        """Publish signal event to event bus."""
        if not self.event_bus:
            return

        event = create_signal_event(
            symbol=signal.symbol,
            signal_data={
                "signal_id": str(signal.signal_id),
                "direction": signal.direction.value,
                "strength": signal.strength,
                "confidence": signal.confidence,
                "horizon": signal.horizon,
                "source": signal.model_source,
            },
            source="SignalGenerator",
        )
        self.event_bus.publish(event)

    def get_statistics(self) -> dict[str, Any]:
        """Get signal generator statistics.

        Returns:
            Dictionary with statistics.
        """
        total_pending = sum(len(s) for s in self._pending_signals.values())
        total_history = len(self._signal_history)
        executed = len([s for s in self._signal_history if s.executed])

        return {
            "num_strategies": len(self._strategies),
            "num_external_sources": len(self._external_sources),
            "pending_signals": total_pending,
            "pending_by_symbol": {s: len(sigs) for s, sigs in self._pending_signals.items()},
            "total_generated": total_history,
            "total_executed": executed,
            "execution_rate": executed / total_history if total_history > 0 else 0.0,
            "strategy_states": {
                sid: s.state.value for sid, s in self._strategies.items()
            },
        }

    def get_strategy(self, strategy_id: str) -> Strategy | None:
        """Get strategy by ID."""
        return self._strategies.get(strategy_id)

    def get_all_strategies(self) -> list[Strategy]:
        """Get all strategies."""
        return list(self._strategies.values())


class SignalQueue:
    """Priority queue for signal processing."""

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize signal queue.

        Args:
            max_size: Maximum queue size.
        """
        self.max_size = max_size
        self._queue: list[EnrichedSignal] = []

    def push(self, signal: EnrichedSignal) -> bool:
        """Add signal to queue.

        Args:
            signal: Signal to add.

        Returns:
            True if added.
        """
        if len(self._queue) >= self.max_size:
            # Remove lowest priority signal
            self._queue.sort(key=self._priority_key)
            if self._priority_key(signal) > self._priority_key(self._queue[0]):
                self._queue.pop(0)
            else:
                return False

        self._queue.append(signal)
        return True

    def pop(self) -> EnrichedSignal | None:
        """Get highest priority signal.

        Returns:
            Signal or None if empty.
        """
        if not self._queue:
            return None

        self._queue.sort(key=self._priority_key, reverse=True)
        return self._queue.pop(0)

    def peek(self) -> EnrichedSignal | None:
        """View highest priority signal without removing.

        Returns:
            Signal or None if empty.
        """
        if not self._queue:
            return None

        self._queue.sort(key=self._priority_key, reverse=True)
        return self._queue[0]

    def __len__(self) -> int:
        return len(self._queue)

    def _priority_key(self, signal: EnrichedSignal) -> tuple:
        """Priority ordering key."""
        return (
            -signal.metadata.priority.value,
            signal.signal.confidence,
            abs(signal.signal.strength),
        )
