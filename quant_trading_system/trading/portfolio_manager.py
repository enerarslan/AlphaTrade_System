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
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    """Target position for portfolio.

    Supports both long and short positions:
    - Positive target_weight = long position
    - Negative target_weight = short position
    """

    symbol: str
    target_weight: float  # Portfolio weight (-1 to 1), negative for shorts
    target_shares: Decimal  # Positive for long, negative for short
    target_value: Decimal  # Absolute value of position
    signal: EnrichedSignal | None = None
    confidence: float = 0.0
    priority: OrderPriority = OrderPriority.NORMAL
    is_short: bool = False  # True if this is a short position


@dataclass
class TargetPortfolio:
    """Target portfolio state.

    Supports both long and short positions with separate exposure tracking.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    target_positions: dict[str, TargetPosition] = field(default_factory=dict)
    target_cash_pct: float = 0.0
    total_equity: Decimal = Decimal("0")
    expected_turnover: float = 0.0
    rebalance_reason: str = ""

    @property
    def total_target_weight(self) -> float:
        """Get net weight of target positions (long - short)."""
        return sum(tp.target_weight for tp in self.target_positions.values())

    @property
    def gross_exposure(self) -> float:
        """Get gross exposure (|long| + |short|)."""
        return sum(abs(tp.target_weight) for tp in self.target_positions.values())

    @property
    def long_exposure(self) -> float:
        """Get total long exposure."""
        return sum(tp.target_weight for tp in self.target_positions.values() if tp.target_weight > 0)

    @property
    def short_exposure(self) -> float:
        """Get total short exposure (as positive number)."""
        return sum(abs(tp.target_weight) for tp in self.target_positions.values() if tp.target_weight < 0)

    @property
    def num_positions(self) -> int:
        """Get total number of target positions."""
        return len(self.target_positions)

    @property
    def num_long_positions(self) -> int:
        """Get number of long positions."""
        return sum(1 for tp in self.target_positions.values() if not tp.is_short)

    @property
    def num_short_positions(self) -> int:
        """Get number of short positions."""
        return sum(1 for tp in self.target_positions.values() if tp.is_short)


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
            win_prob = max(0.0, min(1.0, signal.signal.confidence))
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

        Supports both LONG and SHORT positions:
        - LONG signals return positive weights
        - SHORT signals return negative weights

        Args:
            signals: List of signals.
            portfolio: Current portfolio.
            prices: Current prices by symbol.
            volatilities: Optional volatilities by symbol.

        Returns:
            Target weights by symbol (positive for long, negative for short).
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

                # Apply sign based on direction - SHORT signals get negative weight
                if signal.signal.direction == Direction.SHORT:
                    weight = -weight

                weights[symbol] = weight

        # Normalize if total absolute weight exceeds 1
        # (allows for 100% long + 100% short = 200% gross exposure with 1.0 net)
        total_long = sum(w for w in weights.values() if w > 0)
        total_short = sum(abs(w) for w in weights.values() if w < 0)

        # Cap gross exposure at 2x (100% long + 100% short max)
        max_gross = 2.0
        gross_exposure = total_long + total_short
        if gross_exposure > max_gross:
            factor = max_gross / gross_exposure
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

        # CRITICAL FIX: Add thread safety for shared state
        # Multiple methods can modify _target_portfolio and _pending_trades
        # concurrently (e.g., signal handler + main trading loop)
        self._state_lock = threading.RLock()

        # State (protected by _state_lock)
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

        Supports both LONG and SHORT positions:
        - LONG signals create positive weight positions
        - SHORT signals create negative weight positions

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

        # Process both LONG and SHORT signals (full short selling support)
        directional_signals = [
            s for s in active_signals
            if s.signal.direction in (Direction.LONG, Direction.SHORT)
        ]

        # Sort by priority and confidence
        directional_signals.sort(
            key=lambda s: (-s.metadata.priority.value, s.signal.confidence),
            reverse=True,
        )

        # Limit number of positions
        max_positions = self.position_sizer.config.max_total_positions
        directional_signals = directional_signals[:max_positions]

        # Calculate weights (positive for longs, negative for shorts)
        weights = self.position_sizer.calculate_weights(
            directional_signals, portfolio, prices, volatilities
        )

        # Build target positions
        for signal in directional_signals:
            symbol = signal.signal.symbol
            weight = weights.get(symbol, 0.0)

            # Skip zero weights
            if weight == 0.0:
                continue

            price = prices.get(symbol, Decimal("0"))
            if price <= 0:
                continue

            if self.rebalance_config.consider_transaction_costs:
                cost_drag = self.rebalance_config.transaction_cost_bps / 10000
                sign = -1.0 if weight < 0 else 1.0
                weight = sign * max(0.0, abs(weight) - cost_drag)
                if weight == 0.0:
                    continue

            # For shorts, weight is negative - use absolute value for target_value
            is_short = weight < 0
            abs_weight = abs(weight)
            target_value = portfolio.equity * Decimal(str(abs_weight))
            target_shares = (target_value / price).quantize(Decimal("1"))

            # For short positions, shares are negative
            if is_short:
                target_shares = -target_shares

            # Determine priority
            if signal.metadata.priority == SignalPriority.CRITICAL:
                order_priority = OrderPriority.CRITICAL
            elif signal.metadata.priority == SignalPriority.HIGH:
                order_priority = OrderPriority.HIGH
            else:
                order_priority = OrderPriority.NORMAL

            target.target_positions[symbol] = TargetPosition(
                symbol=symbol,
                target_weight=weight,  # Negative for shorts
                target_shares=target_shares,  # Negative for shorts
                target_value=target_value,  # Always positive (absolute value)
                signal=signal,
                confidence=signal.signal.confidence,
                priority=order_priority,
                is_short=is_short,
            )

        # Calculate expected turnover
        target.expected_turnover = self._calculate_turnover(target, portfolio)

        # THREAD SAFETY: Use lock when modifying shared state
        with self._state_lock:
            self._target_portfolio = target

        return target

    def generate_rebalance_trades(
        self,
        target: TargetPortfolio,
        portfolio: Portfolio,
        prices: dict[str, Decimal],
    ) -> list[Trade]:
        """Generate trades to achieve target portfolio.

        Handles both long and short positions:
        - Long targets (positive shares): BUY to increase, SELL to decrease
        - Short targets (negative shares): SELL to short, BUY to cover

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

            # Create trade - handles both long and short scenarios
            # delta_shares > 0: BUY (increase long or cover short)
            # delta_shares < 0: SELL (decrease long or increase short)
            side = OrderSide.BUY if delta_shares > 0 else OrderSide.SELL

            # Determine trade reason based on position type
            if target_pos.is_short:
                if side == OrderSide.SELL:
                    reason = "short_sell"
                else:
                    reason = "cover_short"
            else:
                if side == OrderSide.BUY:
                    reason = "buy_long"
                else:
                    reason = "reduce_long"

            trade = Trade(
                symbol=symbol,
                side=side,
                quantity=abs(delta_shares),
                current_price=price,
                target_position=target_pos,
                priority=target_pos.priority,
                reason=reason,
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

        # THREAD SAFETY: Use lock when modifying shared state
        with self._state_lock:
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

        # Check for positions that should be closed (both long and short)
        for symbol in current_weights:
            if symbol not in target.target_positions:
                if abs(current_weights[symbol]) > 0.01:  # >1% in non-target position
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
            signal_id = None

            if trade.target_position and trade.target_position.signal:
                signal = trade.target_position.signal
                # Could add stop/take profit based on signal metadata
                signal_id = signal.signal.signal_id

            request = OrderRequest(
                symbol=trade.symbol,
                side=trade.side,
                quantity=trade.quantity,
                order_type=trade.order_type,
                limit_price=trade.limit_price,
                priority=trade.priority,
                strategy_id=strategy_id or "",
                signal_id=signal_id,
                notes=trade.reason,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
            )
            requests.append(request)

        return requests

    def get_target_portfolio(self) -> TargetPortfolio | None:
        """Get current target portfolio. Thread-safe."""
        with self._state_lock:
            return self._target_portfolio

    def get_pending_trades(self) -> list[Trade]:
        """Get pending trades. Thread-safe - returns a copy."""
        with self._state_lock:
            return self._pending_trades.copy()

    def clear_pending_trades(self) -> None:
        """Clear pending trades. Thread-safe."""
        with self._state_lock:
            self._pending_trades = []

    def _calculate_current_weights(
        self,
        portfolio: Portfolio,
        prices: dict[str, Decimal],
    ) -> dict[str, float]:
        """Calculate current portfolio weights.

        Returns signed weights:
        - Positive weight = long position
        - Negative weight = short position
        """
        weights = {}
        if portfolio.equity <= 0:
            return weights

        for symbol, position in portfolio.positions.items():
            if position.is_flat:
                continue
            price = prices.get(symbol, position.current_price)
            # Use signed value - negative for short positions
            value = position.quantity * price
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
            current_abs_value = abs(position.market_value)
            change = abs(target_value - current_abs_value)
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
            if self.rebalance_config.consider_transaction_costs:
                estimated_cost = trade_value * (
                    self.rebalance_config.transaction_cost_bps / 10000
                )
            else:
                estimated_cost = 0.0
            effective_trade_value = trade_value + estimated_cost

            if total_value + effective_trade_value <= max_turnover_value:
                filtered_trades.append(trade)
                total_value += effective_trade_value
            else:
                logger.warning(
                    f"Skipping trade {trade.symbol} due to turnover limit "
                    f"({total_value + effective_trade_value:.0f} > {max_turnover_value:.0f})"
                )

        return filtered_trades

    def get_statistics(self) -> dict[str, Any]:
        """Get portfolio manager statistics.

        Returns statistics including long/short exposure breakdown.
        """
        target = self._target_portfolio

        return {
            "has_target": target is not None,
            "target_positions": target.num_positions if target else 0,
            "num_long_positions": target.num_long_positions if target else 0,
            "num_short_positions": target.num_short_positions if target else 0,
            "net_weight": target.total_target_weight if target else 0.0,
            "gross_exposure": target.gross_exposure if target else 0.0,
            "long_exposure": target.long_exposure if target else 0.0,
            "short_exposure": target.short_exposure if target else 0.0,
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
        """Optimize portfolio weights using mean-variance optimization.

        Uses scipy to find optimal weights that maximize Sharpe ratio
        (or minimize variance for a target return).

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

        # Check if scipy is available for optimization
        try:
            from scipy.optimize import minimize

            # Objective: Negative Sharpe ratio (we minimize, so negate to maximize)
            def neg_sharpe(w: np.ndarray) -> float:
                port_return = np.dot(w, returns)
                port_var = np.dot(w.T, np.dot(covariance_matrix, w))
                port_std = np.sqrt(port_var) if port_var > 0 else 1e-10
                sharpe = (port_return - self.risk_free_rate) / port_std
                return -sharpe

            # Constraints: weights sum to 1
            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

            # Add target return constraint if specified
            if target_return is not None:
                constraints.append({
                    "type": "eq",
                    "fun": lambda w: np.dot(w, returns) - target_return
                })

            # Bounds: each weight between min and max
            bounds = [(self.min_weight, self.max_weight) for _ in range(n)]

            # Initial guess: equal weights
            w0 = np.ones(n) / n

            # Optimize
            result = minimize(
                neg_sharpe,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-10},
            )

            if result.success:
                weights = result.x
                # Ensure non-negative and normalized
                weights = np.clip(weights, 0, self.max_weight)
                weights = weights / weights.sum() if weights.sum() > 0 else w0
                return {symbols[i]: float(weights[i]) for i in range(n)}
            else:
                logger.warning(f"Optimization failed: {result.message}, using equal weights")

        except ImportError:
            logger.warning("scipy not available, using equal-weight fallback")
        except Exception as e:
            logger.warning(f"Optimization error: {e}, using equal weights")

        # Fallback: equal weights
        weights = np.ones(n) / n
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = weights / weights.sum()

        return {symbols[i]: float(weights[i]) for i in range(n)}

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
