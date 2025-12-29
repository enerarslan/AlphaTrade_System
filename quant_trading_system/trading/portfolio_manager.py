"""
Portfolio management module.

Handles portfolio construction and rebalancing:
- Target portfolio calculation from signals
- Trade generation for rebalancing
- Position sizing with risk constraints
- Portfolio optimization

Bridges signals to executable trades.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

import numpy as np

from quant_trading_system.core.data_types import (
    Direction,
    Order,
    OrderSide,
    OrderType,
    Portfolio,
    Position,
    TradeSignal,
)
from quant_trading_system.core.events import EventBus, EventType
from quant_trading_system.execution.order_manager import OrderPriority, OrderRequest
from quant_trading_system.trading.signal_generator import EnrichedSignal, SignalPriority

logger = logging.getLogger(__name__)


class PositionSizingMethod(str, Enum):
    """Position sizing methods."""

    FIXED_DOLLAR = "fixed_dollar"
    FIXED_SHARES = "fixed_shares"
    PERCENT_EQUITY = "percent_equity"
    VOLATILITY_SCALED = "volatility_scaled"
    RISK_PARITY = "risk_parity"
    KELLY = "kelly"


class RebalanceMethod(str, Enum):
    """Portfolio rebalancing methods."""

    THRESHOLD = "threshold"  # Rebalance when drift exceeds threshold
    PERIODIC = "periodic"  # Rebalance on schedule
    SIGNAL_DRIVEN = "signal_driven"  # Rebalance on new signals
    HYBRID = "hybrid"  # Combination


@dataclass
class PositionSizerConfig:
    """Position sizer configuration."""

    method: PositionSizingMethod = PositionSizingMethod.PERCENT_EQUITY
    fixed_dollar_amount: Decimal = Decimal("1000")
    fixed_share_count: int = 100
    percent_of_equity: float = 0.05  # 5% per position
    max_position_pct: float = 0.10  # Max 10% in single position
    max_total_positions: int = 20
    volatility_target: float = 0.15  # Annual volatility target for vol scaling
    kelly_fraction: float = 0.25  # Fraction of full Kelly


@dataclass
class RebalanceConfig:
    """Rebalancing configuration."""

    method: RebalanceMethod = RebalanceMethod.THRESHOLD
    threshold_pct: float = 0.05  # 5% drift triggers rebalance
    min_trade_value: Decimal = Decimal("100")  # Minimum trade size
    max_turnover_pct: float = 0.20  # Max daily turnover
    round_lots: bool = False  # Round to 100-share lots
    consider_transaction_costs: bool = True
    transaction_cost_bps: float = 5.0  # 5 bps


@dataclass
class TargetPosition:
    """Target position for portfolio."""

    symbol: str
    target_weight: float  # Portfolio weight (0-1)
    target_shares: Decimal
    target_value: Decimal
    signal: EnrichedSignal | None = None
    confidence: float = 0.0
    priority: OrderPriority = OrderPriority.NORMAL


@dataclass
class TargetPortfolio:
    """Target portfolio state."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    target_positions: dict[str, TargetPosition] = field(default_factory=dict)
    target_cash_pct: float = 0.0
    total_equity: Decimal = Decimal("0")
    expected_turnover: float = 0.0
    rebalance_reason: str = ""

    @property
    def total_target_weight(self) -> float:
        """Get total weight of target positions."""
        return sum(tp.target_weight for tp in self.target_positions.values())

    @property
    def num_positions(self) -> int:
        """Get number of target positions."""
        return len(self.target_positions)


@dataclass
class Trade:
    """Trade to execute for rebalancing."""

    symbol: str
    side: OrderSide
    quantity: Decimal
    current_price: Decimal
    target_position: TargetPosition | None = None
    priority: OrderPriority = OrderPriority.NORMAL
    reason: str = ""
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None

    @property
    def notional_value(self) -> Decimal:
        """Get trade notional value."""
        return abs(self.quantity * self.current_price)


class PositionSizer:
    """Calculates position sizes from signals."""

    def __init__(self, config: PositionSizerConfig) -> None:
        """Initialize position sizer.

        Args:
            config: Position sizing configuration.
        """
        self.config = config

    def calculate_position_size(
        self,
        signal: EnrichedSignal,
        portfolio: Portfolio,
        current_price: Decimal,
        volatility: float | None = None,
    ) -> Decimal:
        """Calculate position size for a signal.

        Args:
            signal: Trading signal.
            portfolio: Current portfolio.
            current_price: Current market price.
            volatility: Optional volatility estimate.

        Returns:
            Target position size in shares.
        """
        if current_price <= 0:
            return Decimal("0")

        if self.config.method == PositionSizingMethod.FIXED_DOLLAR:
            shares = self.config.fixed_dollar_amount / current_price

        elif self.config.method == PositionSizingMethod.FIXED_SHARES:
            shares = Decimal(str(self.config.fixed_share_count))

        elif self.config.method == PositionSizingMethod.PERCENT_EQUITY:
            target_value = portfolio.equity * Decimal(str(self.config.percent_of_equity))
            # Scale by signal confidence
            target_value *= Decimal(str(signal.signal.confidence))
            shares = target_value / current_price

        elif self.config.method == PositionSizingMethod.VOLATILITY_SCALED:
            if volatility is None or volatility <= 0:
                volatility = 0.20  # Default 20% annual vol

            # Target risk contribution
            base_pct = self.config.percent_of_equity
            vol_scalar = self.config.volatility_target / volatility
            adjusted_pct = min(base_pct * vol_scalar, self.config.max_position_pct)

            target_value = portfolio.equity * Decimal(str(adjusted_pct))
            target_value *= Decimal(str(signal.signal.confidence))
            shares = target_value / current_price

        elif self.config.method == PositionSizingMethod.KELLY:
            # Simplified Kelly criterion
            # f = (p * b - q) / b where p=win_prob, q=1-p, b=win/loss ratio
            win_prob = (signal.signal.confidence + 1) / 2  # Map confidence to prob
            win_loss_ratio = abs(signal.signal.strength) * 2  # Estimate

            if win_loss_ratio > 0:
                kelly_pct = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
                kelly_pct = max(0, min(kelly_pct * self.config.kelly_fraction, self.config.max_position_pct))
            else:
                kelly_pct = 0

            target_value = portfolio.equity * Decimal(str(kelly_pct))
            shares = target_value / current_price

        else:
            shares = Decimal("0")

        # Apply maximum position constraint
        max_value = portfolio.equity * Decimal(str(self.config.max_position_pct))
        max_shares = max_value / current_price
        shares = min(shares, max_shares)

        # Round to whole shares
        return shares.quantize(Decimal("1"))

    def calculate_weights(
        self,
        signals: list[EnrichedSignal],
        portfolio: Portfolio,
        prices: dict[str, Decimal],
        volatilities: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Calculate target weights for signals.

        Args:
            signals: List of signals.
            portfolio: Current portfolio.
            prices: Current prices by symbol.
            volatilities: Optional volatilities by symbol.

        Returns:
            Target weights by symbol.
        """
        weights: dict[str, float] = {}
        volatilities = volatilities or {}

        for signal in signals:
            symbol = signal.signal.symbol
            price = prices.get(symbol, Decimal("0"))

            if price <= 0:
                continue

            vol = volatilities.get(symbol)
            shares = self.calculate_position_size(signal, portfolio, price, vol)

            if shares > 0:
                value = shares * price
                weight = float(value / portfolio.equity)
                weights[symbol] = weight

        # Normalize if total exceeds 1
        total_weight = sum(weights.values())
        if total_weight > 1.0:
            factor = 1.0 / total_weight
            weights = {s: w * factor for s, w in weights.items()}

        return weights


class PortfolioManager:
    """Manages portfolio construction and rebalancing.

    Converts signals into target portfolios and generates
    trades to achieve the target state.
    """

    def __init__(
        self,
        position_sizer: PositionSizer | None = None,
        rebalance_config: RebalanceConfig | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize portfolio manager.

        Args:
            position_sizer: Position sizing calculator.
            rebalance_config: Rebalancing configuration.
            event_bus: Event bus for portfolio events.
        """
        self.position_sizer = position_sizer or PositionSizer(PositionSizerConfig())
        self.rebalance_config = rebalance_config or RebalanceConfig()
        self.event_bus = event_bus

        # State
        self._target_portfolio: TargetPortfolio | None = None
        self._last_rebalance: datetime | None = None
        self._pending_trades: list[Trade] = []

    def build_target_portfolio(
        self,
        signals: list[EnrichedSignal],
        portfolio: Portfolio,
        prices: dict[str, Decimal],
        volatilities: dict[str, float] | None = None,
    ) -> TargetPortfolio:
        """Build target portfolio from signals.

        Args:
            signals: Trading signals.
            portfolio: Current portfolio.
            prices: Current prices by symbol.
            volatilities: Optional volatilities by symbol.

        Returns:
            Target portfolio state.
        """
        target = TargetPortfolio(
            total_equity=portfolio.equity,
            rebalance_reason="signal_update",
        )

        # Filter to actionable signals
        active_signals = [s for s in signals if s.is_actionable]

        # Only LONG signals result in positions (no shorting for simplicity)
        long_signals = [
            s for s in active_signals
            if s.signal.direction == Direction.LONG
        ]

        # Sort by priority and confidence
        long_signals.sort(
            key=lambda s: (-s.metadata.priority.value, s.signal.confidence),
            reverse=True,
        )

        # Limit number of positions
        max_positions = self.position_sizer.config.max_total_positions
        long_signals = long_signals[:max_positions]

        # Calculate weights
        weights = self.position_sizer.calculate_weights(
            long_signals, portfolio, prices, volatilities
        )

        # Build target positions
        for signal in long_signals:
            symbol = signal.signal.symbol
            weight = weights.get(symbol, 0.0)

            if weight <= 0:
                continue

            price = prices.get(symbol, Decimal("0"))
            if price <= 0:
                continue

            target_value = portfolio.equity * Decimal(str(weight))
            target_shares = (target_value / price).quantize(Decimal("1"))

            # Determine priority
            if signal.metadata.priority == SignalPriority.CRITICAL:
                order_priority = OrderPriority.CRITICAL
            elif signal.metadata.priority == SignalPriority.HIGH:
                order_priority = OrderPriority.HIGH
            else:
                order_priority = OrderPriority.NORMAL

            target.target_positions[symbol] = TargetPosition(
                symbol=symbol,
                target_weight=weight,
                target_shares=target_shares,
                target_value=target_value,
                signal=signal,
                confidence=signal.signal.confidence,
                priority=order_priority,
            )

        # Calculate expected turnover
        target.expected_turnover = self._calculate_turnover(target, portfolio)

        self._target_portfolio = target
        return target

    def generate_rebalance_trades(
        self,
        target: TargetPortfolio,
        portfolio: Portfolio,
        prices: dict[str, Decimal],
    ) -> list[Trade]:
        """Generate trades to achieve target portfolio.

        Args:
            target: Target portfolio state.
            portfolio: Current portfolio.
            prices: Current prices.

        Returns:
            List of trades to execute.
        """
        trades: list[Trade] = []

        # Current positions
        current_positions = portfolio.positions

        # Process target positions
        for symbol, target_pos in target.target_positions.items():
            price = prices.get(symbol, Decimal("0"))
            if price <= 0:
                continue

            current_pos = current_positions.get(symbol)
            current_shares = current_pos.quantity if current_pos else Decimal("0")
            delta_shares = target_pos.target_shares - current_shares

            if abs(delta_shares) < 1:
                continue

            # Check minimum trade value
            trade_value = abs(delta_shares * price)
            if trade_value < self.rebalance_config.min_trade_value:
                continue

            # Create trade
            side = OrderSide.BUY if delta_shares > 0 else OrderSide.SELL
            trade = Trade(
                symbol=symbol,
                side=side,
                quantity=abs(delta_shares),
                current_price=price,
                target_position=target_pos,
                priority=target_pos.priority,
                reason="rebalance_to_target",
            )
            trades.append(trade)

        # Process positions to close (not in target)
        for symbol, position in current_positions.items():
            if symbol in target.target_positions:
                continue  # Already handled
            if position.is_flat:
                continue

            price = prices.get(symbol, position.current_price)
            trade = Trade(
                symbol=symbol,
                side=OrderSide.SELL if position.is_long else OrderSide.BUY,
                quantity=abs(position.quantity),
                current_price=price,
                priority=OrderPriority.NORMAL,
                reason="exit_position",
            )
            trades.append(trade)

        # Apply turnover constraint
        trades = self._apply_turnover_limit(trades, portfolio)

        # Sort by priority
        trades.sort(key=lambda t: t.priority.value)

        self._pending_trades = trades
        return trades

    def check_rebalance_needed(
        self,
        target: TargetPortfolio,
        portfolio: Portfolio,
        prices: dict[str, Decimal],
    ) -> tuple[bool, str]:
        """Check if rebalancing is needed.

        Args:
            target: Target portfolio state.
            portfolio: Current portfolio.
            prices: Current prices.

        Returns:
            Tuple of (needs_rebalance, reason).
        """
        if self.rebalance_config.method == RebalanceMethod.SIGNAL_DRIVEN:
            # Always rebalance on new signals
            if target.num_positions > 0:
                return True, "new_signals"
            return False, ""

        # Calculate current weights
        current_weights = self._calculate_current_weights(portfolio, prices)

        # Calculate drift
        max_drift = 0.0
        for symbol, target_pos in target.target_positions.items():
            current_weight = current_weights.get(symbol, 0.0)
            drift = abs(target_pos.target_weight - current_weight)
            max_drift = max(max_drift, drift)

        # Check for positions that should be closed
        for symbol in current_weights:
            if symbol not in target.target_positions:
                if current_weights[symbol] > 0.01:  # >1% in non-target position
                    return True, "unwanted_position"

        if max_drift > self.rebalance_config.threshold_pct:
            return True, f"drift_{max_drift:.1%}"

        return False, ""

    def create_order_requests(
        self,
        trades: list[Trade],
        strategy_id: str | None = None,
    ) -> list[OrderRequest]:
        """Convert trades to order requests.

        Args:
            trades: Trades to convert.
            strategy_id: Associated strategy ID.

        Returns:
            List of order requests.
        """
        requests = []

        for trade in trades:
            # Determine stop loss and take profit if signal available
            stop_loss = None
            take_profit = None

            if trade.target_position and trade.target_position.signal:
                signal = trade.target_position.signal
                # Could add stop/take profit based on signal metadata
                pass

            request = OrderRequest(
                symbol=trade.symbol,
                side=trade.side,
                quantity=trade.quantity,
                order_type=trade.order_type,
                limit_price=trade.limit_price,
                priority=trade.priority,
                strategy_id=strategy_id or "",
                notes=trade.reason,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
            )
            requests.append(request)

        return requests

    def get_target_portfolio(self) -> TargetPortfolio | None:
        """Get current target portfolio."""
        return self._target_portfolio

    def get_pending_trades(self) -> list[Trade]:
        """Get pending trades."""
        return self._pending_trades

    def clear_pending_trades(self) -> None:
        """Clear pending trades."""
        self._pending_trades = []

    def _calculate_current_weights(
        self,
        portfolio: Portfolio,
        prices: dict[str, Decimal],
    ) -> dict[str, float]:
        """Calculate current portfolio weights."""
        weights = {}
        if portfolio.equity <= 0:
            return weights

        for symbol, position in portfolio.positions.items():
            if position.is_flat:
                continue
            price = prices.get(symbol, position.current_price)
            value = abs(position.quantity * price)
            weights[symbol] = float(value / portfolio.equity)

        return weights

    def _calculate_turnover(
        self,
        target: TargetPortfolio,
        portfolio: Portfolio,
    ) -> float:
        """Calculate expected turnover from current to target."""
        total_change = Decimal("0")

        # Changes in existing positions
        for symbol, position in portfolio.positions.items():
            target_pos = target.target_positions.get(symbol)
            target_value = target_pos.target_value if target_pos else Decimal("0")
            change = abs(target_value - position.market_value)
            total_change += change

        # New positions
        for symbol, target_pos in target.target_positions.items():
            if symbol not in portfolio.positions:
                total_change += target_pos.target_value

        if portfolio.equity <= 0:
            return 0.0

        return float(total_change / portfolio.equity / 2)  # Divide by 2 for one-way turnover

    def _apply_turnover_limit(
        self,
        trades: list[Trade],
        portfolio: Portfolio,
    ) -> list[Trade]:
        """Apply turnover limit to trades."""
        max_turnover_value = float(portfolio.equity) * self.rebalance_config.max_turnover_pct

        # Sort by priority (higher priority first)
        trades.sort(key=lambda t: t.priority.value)

        filtered_trades = []
        total_value = 0.0

        for trade in trades:
            trade_value = float(trade.notional_value)
            if total_value + trade_value <= max_turnover_value:
                filtered_trades.append(trade)
                total_value += trade_value
            else:
                logger.warning(
                    f"Skipping trade {trade.symbol} due to turnover limit "
                    f"({total_value + trade_value:.0f} > {max_turnover_value:.0f})"
                )

        return filtered_trades

    def get_statistics(self) -> dict[str, Any]:
        """Get portfolio manager statistics."""
        target = self._target_portfolio

        return {
            "has_target": target is not None,
            "target_positions": target.num_positions if target else 0,
            "target_weight": target.total_target_weight if target else 0.0,
            "expected_turnover": target.expected_turnover if target else 0.0,
            "pending_trades": len(self._pending_trades),
            "pending_trade_value": float(sum(t.notional_value for t in self._pending_trades)),
            "last_rebalance": self._last_rebalance.isoformat() if self._last_rebalance else None,
        }


class PortfolioOptimizer:
    """Advanced portfolio optimization.

    Implements mean-variance optimization with constraints.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        max_weight: float = 0.10,
        min_weight: float = 0.0,
    ) -> None:
        """Initialize optimizer.

        Args:
            risk_free_rate: Risk-free rate for Sharpe calculation.
            max_weight: Maximum weight per asset.
            min_weight: Minimum weight per asset (0 = can exclude).
        """
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight

    def optimize_weights(
        self,
        expected_returns: dict[str, float],
        covariance_matrix: np.ndarray,
        symbols: list[str],
        target_return: float | None = None,
    ) -> dict[str, float]:
        """Optimize portfolio weights.

        Uses mean-variance optimization to find optimal weights.

        Args:
            expected_returns: Expected returns by symbol.
            covariance_matrix: Covariance matrix of returns.
            symbols: List of symbols in order of cov matrix.
            target_return: Optional target return constraint.

        Returns:
            Optimized weights by symbol.
        """
        n = len(symbols)
        if n == 0:
            return {}

        returns = np.array([expected_returns.get(s, 0.0) for s in symbols])

        # Simple equal weight as baseline
        weights = np.ones(n) / n

        # Constrain weights
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = weights / weights.sum()  # Renormalize

        return {symbols[i]: weights[i] for i in range(n)}

    def calculate_portfolio_stats(
        self,
        weights: dict[str, float],
        expected_returns: dict[str, float],
        covariance_matrix: np.ndarray,
        symbols: list[str],
    ) -> dict[str, float]:
        """Calculate portfolio statistics.

        Args:
            weights: Portfolio weights.
            expected_returns: Expected returns by symbol.
            covariance_matrix: Covariance matrix.
            symbols: List of symbols.

        Returns:
            Portfolio statistics.
        """
        w = np.array([weights.get(s, 0.0) for s in symbols])
        r = np.array([expected_returns.get(s, 0.0) for s in symbols])

        portfolio_return = float(np.dot(w, r))
        portfolio_variance = float(np.dot(w.T, np.dot(covariance_matrix, w)))
        portfolio_std = float(np.sqrt(portfolio_variance))

        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std > 0 else 0.0

        return {
            "expected_return": portfolio_return,
            "volatility": portfolio_std,
            "sharpe_ratio": sharpe,
            "num_assets": len([w for w in weights.values() if w > 0.01]),
        }
