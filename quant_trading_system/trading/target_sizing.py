"""
Shared target-sizing helpers used by live portfolio construction and backtests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Callable, Sequence

from quant_trading_system.core.data_types import Direction, Portfolio
from quant_trading_system.trading.signal_generator import EnrichedSignal

PositionWeightCalculator = Callable[
    [list[EnrichedSignal], Portfolio, dict[str, Decimal], dict[str, float] | None],
    dict[str, float],
]
ShareRounder = Callable[[Decimal], Decimal]


@dataclass(frozen=True, slots=True)
class SizedSignalTarget:
    """Shared target allocation result for a single symbol."""

    signal: EnrichedSignal
    price: Decimal
    weight: float
    target_value: Decimal
    target_shares: Decimal
    current_shares: Decimal
    delta_shares: Decimal


def _signal_selection_key(signal: EnrichedSignal) -> tuple[int, float, float, datetime]:
    """Rank signals so higher-priority, higher-confidence entries win."""
    return (
        -int(signal.metadata.priority.value),
        float(signal.signal.confidence),
        abs(float(signal.signal.strength)),
        signal.signal.timestamp,
    )


def select_target_signals(
    signals: Sequence[EnrichedSignal],
    *,
    max_total_positions: int,
) -> list[EnrichedSignal]:
    """Select the best directional signal per symbol and cap total positions."""
    best_by_symbol: dict[str, EnrichedSignal] = {}

    for signal in signals:
        if not signal.is_actionable:
            continue
        if signal.signal.direction not in (Direction.LONG, Direction.SHORT):
            continue

        existing = best_by_symbol.get(signal.signal.symbol)
        if existing is None or _signal_selection_key(signal) > _signal_selection_key(existing):
            best_by_symbol[signal.signal.symbol] = signal

    ranked = sorted(best_by_symbol.values(), key=_signal_selection_key, reverse=True)
    return ranked[: max(1, int(max_total_positions))]


def build_sized_signal_targets(
    signals: Sequence[EnrichedSignal],
    portfolio: Portfolio,
    prices: dict[str, Decimal],
    calculate_weights: PositionWeightCalculator,
    *,
    volatilities: dict[str, float] | None = None,
    allow_fractional: bool = True,
    max_total_positions: int,
    share_rounder: ShareRounder | None = None,
) -> list[SizedSignalTarget]:
    """Build ranked target allocations from directional signals."""
    if portfolio.equity <= 0:
        return []

    selected = select_target_signals(signals, max_total_positions=max_total_positions)
    if not selected:
        return []

    eligible_signals: list[EnrichedSignal] = []
    eligible_prices: dict[str, Decimal] = {}
    eligible_volatilities: dict[str, float] = {}
    volatilities = volatilities or {}

    for signal in selected:
        price = prices.get(signal.signal.symbol, Decimal("0"))
        if price <= 0:
            continue
        eligible_signals.append(signal)
        eligible_prices[signal.signal.symbol] = price
        if signal.signal.symbol in volatilities:
            eligible_volatilities[signal.signal.symbol] = volatilities[signal.signal.symbol]

    if not eligible_signals:
        return []

    weights = calculate_weights(
        eligible_signals,
        portfolio,
        eligible_prices,
        eligible_volatilities or None,
    )

    if share_rounder is None and not allow_fractional:
        share_rounder = lambda shares: shares.quantize(Decimal("1"))

    targets: list[SizedSignalTarget] = []
    for signal in eligible_signals:
        symbol = signal.signal.symbol
        weight = float(weights.get(symbol, 0.0))
        if abs(weight) <= 0.0:
            continue

        price = eligible_prices[symbol]
        target_value = portfolio.equity * Decimal(str(abs(weight)))
        target_shares = target_value / price
        if share_rounder is not None:
            target_shares = share_rounder(target_shares)
        if target_shares <= 0:
            continue
        if weight < 0:
            target_shares = -target_shares

        current_position = portfolio.positions.get(symbol)
        current_shares = current_position.quantity if current_position is not None else Decimal("0")
        delta_shares = target_shares - current_shares

        targets.append(
            SizedSignalTarget(
                signal=signal,
                price=price,
                weight=weight,
                target_value=target_value,
                target_shares=target_shares,
                current_shares=current_shares,
                delta_shares=delta_shares,
            )
        )

    return targets
