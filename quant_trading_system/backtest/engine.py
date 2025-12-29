"""
Backtesting engine module.

Provides both event-driven and vectorized backtesting capabilities:
- Event-driven: More realistic simulation with discrete events
- Vectorized: Faster for quick research and strategy comparison

Supports multiple symbols, timeframes, and realistic execution simulation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

from quant_trading_system.core.data_types import (
    Direction,
    OHLCVBar,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    TradeSignal,
)
from quant_trading_system.core.events import Event, EventBus, EventPriority, EventType

logger = logging.getLogger(__name__)


class BacktestMode(str, Enum):
    """Backtesting mode."""

    EVENT_DRIVEN = "event_driven"
    VECTORIZED = "vectorized"


class ExecutionMode(str, Enum):
    """Execution simulation mode."""

    REALISTIC = "realistic"  # Includes costs, slippage
    OPTIMISTIC = "optimistic"  # No costs (upper bound)
    PESSIMISTIC = "pessimistic"  # High costs (stress test)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    initial_capital: Decimal = Decimal("100000")
    mode: BacktestMode = BacktestMode.EVENT_DRIVEN
    execution_mode: ExecutionMode = ExecutionMode.REALISTIC

    # Transaction costs (basis points)
    commission_bps: float = 5.0
    slippage_bps: float = 5.0

    # Execution parameters
    fill_at: str = "next_open"  # 'next_open', 'next_close', 'same_close'
    allow_short: bool = True
    allow_fractional: bool = True
    max_leverage: float = 1.0

    # Risk parameters
    max_position_pct: float = 0.10
    max_drawdown_halt: float = 0.20

    # Time filters
    start_date: datetime | None = None
    end_date: datetime | None = None


@dataclass
class BacktestEvent:
    """Event in the backtesting event queue."""

    event_type: str
    timestamp: datetime
    data: dict[str, Any]
    priority: int = 0

    def __lt__(self, other: "BacktestEvent") -> bool:
        """Compare events for priority queue ordering."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


@dataclass
class Trade:
    """Record of a completed trade."""

    symbol: str
    entry_time: datetime
    exit_time: datetime
    side: OrderSide
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    pnl: Decimal
    pnl_pct: float
    commission: Decimal
    slippage: Decimal
    holding_period_bars: int
    exit_reason: str = "signal"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def gross_pnl(self) -> Decimal:
        """Get P&L before costs."""
        return self.pnl + self.commission + self.slippage

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "side": self.side.value,
            "quantity": float(self.quantity),
            "entry_price": float(self.entry_price),
            "exit_price": float(self.exit_price),
            "pnl": float(self.pnl),
            "pnl_pct": self.pnl_pct,
            "commission": float(self.commission),
            "slippage": float(self.slippage),
            "holding_period_bars": self.holding_period_bars,
            "exit_reason": self.exit_reason,
        }


@dataclass
class BacktestState:
    """Current state of the backtest."""

    timestamp: datetime
    equity: Decimal
    cash: Decimal
    positions: dict[str, Position]
    pending_orders: list[Order]
    equity_curve: list[tuple[datetime, float]]
    trades: list[Trade]
    bars_processed: int = 0
    signals_generated: int = 0
    orders_filled: int = 0

    @property
    def position_count(self) -> int:
        """Get number of open positions."""
        return len([p for p in self.positions.values() if not p.is_flat])


class DataHandler(ABC):
    """Abstract base class for data handling in backtests."""

    @abstractmethod
    def get_symbols(self) -> list[str]:
        """Get list of symbols in the data."""
        pass

    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> list[OHLCVBar]:
        """Get the latest N bars for a symbol."""
        pass

    @abstractmethod
    def update_bars(self) -> bool:
        """Update bars for all symbols. Returns False when finished."""
        pass

    @abstractmethod
    def get_current_bar(self, symbol: str) -> OHLCVBar | None:
        """Get the current bar for a symbol."""
        pass


class PandasDataHandler(DataHandler):
    """Data handler using pandas DataFrames."""

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> None:
        """Initialize pandas data handler.

        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data.
            start_date: Optional start date filter.
            end_date: Optional end date filter.
        """
        self._data = data
        self._symbols = list(data.keys())
        self._current_index = 0
        self._latest_bars: dict[str, deque] = {s: deque(maxlen=1000) for s in self._symbols}

        # Filter and align data
        self._prepare_data(start_date, end_date)

    def _prepare_data(
        self,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> None:
        """Prepare and align data across symbols."""
        # Get common date range
        all_indices = [df.index for df in self._data.values()]
        common_index = all_indices[0]
        for idx in all_indices[1:]:
            common_index = common_index.intersection(idx)

        # Apply date filters
        if start_date:
            common_index = common_index[common_index >= pd.Timestamp(start_date)]
        if end_date:
            common_index = common_index[common_index <= pd.Timestamp(end_date)]

        self._dates = list(common_index)
        self._max_index = len(self._dates)

    def get_symbols(self) -> list[str]:
        """Get list of symbols."""
        return self._symbols.copy()

    def get_latest_bars(self, symbol: str, n: int = 1) -> list[OHLCVBar]:
        """Get the latest N bars for a symbol."""
        if symbol not in self._latest_bars:
            return []
        bars = list(self._latest_bars[symbol])
        return bars[-n:]

    def update_bars(self) -> bool:
        """Update bars for all symbols."""
        if self._current_index >= self._max_index:
            return False

        current_date = self._dates[self._current_index]

        for symbol in self._symbols:
            df = self._data[symbol]
            if current_date in df.index:
                row = df.loc[current_date]
                # Round to 8 decimal places to satisfy OHLCVBar validation
                bar = OHLCVBar(
                    symbol=symbol,
                    timestamp=current_date.to_pydatetime() if hasattr(current_date, 'to_pydatetime') else current_date,
                    open=Decimal(str(round(row.get("open", row.get("Open", 0)), 8))),
                    high=Decimal(str(round(row.get("high", row.get("High", 0)), 8))),
                    low=Decimal(str(round(row.get("low", row.get("Low", 0)), 8))),
                    close=Decimal(str(round(row.get("close", row.get("Close", 0)), 8))),
                    volume=int(row.get("volume", row.get("Volume", 0))),
                )
                self._latest_bars[symbol].append(bar)

        self._current_index += 1
        return True

    def get_current_bar(self, symbol: str) -> OHLCVBar | None:
        """Get the current bar for a symbol."""
        bars = self.get_latest_bars(symbol, 1)
        return bars[0] if bars else None

    def reset(self) -> None:
        """Reset to beginning of data."""
        self._current_index = 0
        self._latest_bars = {s: deque(maxlen=1000) for s in self._symbols}


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    def generate_signals(
        self,
        data_handler: DataHandler,
        portfolio: Portfolio,
    ) -> list[TradeSignal]:
        """Generate trading signals based on current data.

        Args:
            data_handler: Data handler with market data.
            portfolio: Current portfolio state.

        Returns:
            List of trading signals.
        """
        pass

    def on_bar(self, symbol: str, bar: OHLCVBar) -> None:
        """Called for each new bar. Override for bar-level processing."""
        pass

    def on_fill(self, order: Order) -> None:
        """Called when an order is filled. Override for fill handling."""
        pass


class BacktestEngine:
    """Event-driven backtesting engine."""

    def __init__(
        self,
        data_handler: DataHandler,
        strategy: Strategy,
        config: BacktestConfig | None = None,
    ) -> None:
        """Initialize backtesting engine.

        Args:
            data_handler: Data handler for market data.
            strategy: Trading strategy to test.
            config: Backtest configuration.
        """
        self.data_handler = data_handler
        self.strategy = strategy
        self.config = config or BacktestConfig()

        # Initialize state
        self._state: BacktestState | None = None
        self._event_queue: deque = deque()
        self._running = False

        # Position tracking for trade calculation
        self._open_positions: dict[str, dict[str, Any]] = {}

        # Callbacks
        self._on_bar_callbacks: list[Callable] = []
        self._on_signal_callbacks: list[Callable] = []
        self._on_fill_callbacks: list[Callable] = []

    def run(self) -> BacktestState:
        """Run the backtest.

        Returns:
            Final backtest state with results.
        """
        logger.info("Starting backtest...")
        self._initialize()
        self._running = True

        while self._running:
            # Update data
            if not self.data_handler.update_bars():
                self._running = False
                break

            # Process current bar
            self._process_bar()

            # Check halt conditions
            if self._check_halt_conditions():
                self._running = False
                break

        logger.info(f"Backtest completed. Bars processed: {self._state.bars_processed}")
        return self._state

    def _initialize(self) -> None:
        """Initialize backtest state."""
        self._state = BacktestState(
            timestamp=datetime.utcnow(),
            equity=self.config.initial_capital,
            cash=self.config.initial_capital,
            positions={},
            pending_orders=[],
            equity_curve=[],
            trades=[],
        )
        self._open_positions = {}

        # Reset data handler if supported
        if hasattr(self.data_handler, 'reset'):
            self.data_handler.reset()

    def _process_bar(self) -> None:
        """Process current bar for all symbols."""
        for symbol in self.data_handler.get_symbols():
            bar = self.data_handler.get_current_bar(symbol)
            if bar is None:
                continue

            self._state.timestamp = bar.timestamp
            self._state.bars_processed += 1

            # Update positions with new prices
            self._update_position_prices(symbol, bar)

            # Process pending orders
            self._process_pending_orders(symbol, bar)

            # Notify strategy
            self.strategy.on_bar(symbol, bar)

            # Invoke callbacks
            for callback in self._on_bar_callbacks:
                callback(symbol, bar)

        # Generate signals after processing all bars
        signals = self.strategy.generate_signals(
            self.data_handler,
            self._get_portfolio(),
        )

        # Process signals
        for signal in signals:
            self._state.signals_generated += 1
            self._process_signal(signal)
            for callback in self._on_signal_callbacks:
                callback(signal)

        # Update equity curve
        self._update_equity()

    def _update_position_prices(self, symbol: str, bar: OHLCVBar) -> None:
        """Update position with current price."""
        if symbol in self._state.positions:
            position = self._state.positions[symbol]
            self._state.positions[symbol] = position.update_price(bar.close)

    def _process_pending_orders(self, symbol: str, bar: OHLCVBar) -> None:
        """Process pending orders for a symbol."""
        orders_to_remove = []

        for i, order in enumerate(self._state.pending_orders):
            if order.symbol != symbol:
                continue

            # Determine fill price based on config
            fill_price = self._get_fill_price(bar)
            if fill_price is None:
                continue

            # Apply slippage
            slippage = self._calculate_slippage(order, fill_price)
            if order.side == OrderSide.BUY:
                fill_price += slippage
            else:
                fill_price -= slippage

            # Check if order can be filled
            if self._can_fill_order(order, fill_price):
                self._fill_order(order, fill_price, bar.timestamp)
                orders_to_remove.append(i)

        # Remove filled orders
        for i in sorted(orders_to_remove, reverse=True):
            self._state.pending_orders.pop(i)

    def _get_fill_price(self, bar: OHLCVBar) -> Decimal | None:
        """Get fill price based on execution config."""
        if self.config.fill_at == "next_open":
            return bar.open
        elif self.config.fill_at == "next_close":
            return bar.close
        elif self.config.fill_at == "same_close":
            return bar.close
        return bar.open

    def _calculate_slippage(self, order: Order, price: Decimal) -> Decimal:
        """Calculate slippage for an order."""
        if self.config.execution_mode == ExecutionMode.OPTIMISTIC:
            return Decimal("0")

        slippage_bps = self.config.slippage_bps
        if self.config.execution_mode == ExecutionMode.PESSIMISTIC:
            slippage_bps *= 2

        return price * Decimal(str(slippage_bps / 10000))

    def _calculate_commission(self, order: Order, fill_price: Decimal) -> Decimal:
        """Calculate commission for an order."""
        if self.config.execution_mode == ExecutionMode.OPTIMISTIC:
            return Decimal("0")

        commission_bps = self.config.commission_bps
        if self.config.execution_mode == ExecutionMode.PESSIMISTIC:
            commission_bps *= 2

        trade_value = order.quantity * fill_price
        return trade_value * Decimal(str(commission_bps / 10000))

    def _can_fill_order(self, order: Order, fill_price: Decimal) -> bool:
        """Check if order can be filled."""
        trade_value = order.quantity * fill_price
        commission = self._calculate_commission(order, fill_price)

        if order.side == OrderSide.BUY:
            return self._state.cash >= trade_value + commission

        # For sells, check position
        position = self._state.positions.get(order.symbol)
        if position is None:
            return self.config.allow_short
        return position.quantity >= order.quantity or self.config.allow_short

    def _fill_order(
        self,
        order: Order,
        fill_price: Decimal,
        fill_time: datetime,
    ) -> None:
        """Fill an order and update state."""
        commission = self._calculate_commission(order, fill_price)
        trade_value = order.quantity * fill_price

        # Update position
        position = self._state.positions.get(order.symbol)

        if order.side == OrderSide.BUY:
            self._state.cash -= trade_value + commission

            if position is None:
                # New position
                self._state.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_entry_price=fill_price,
                    current_price=fill_price,
                    cost_basis=trade_value,
                    market_value=trade_value,
                )
                self._open_positions[order.symbol] = {
                    "entry_time": fill_time,
                    "entry_price": fill_price,
                    "quantity": order.quantity,
                    "bars_held": 0,
                }
            else:
                # Add to position
                new_qty = position.quantity + order.quantity
                new_cost = position.cost_basis + trade_value
                new_avg = new_cost / new_qty
                self._state.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=new_qty,
                    avg_entry_price=new_avg,
                    current_price=fill_price,
                    cost_basis=new_cost,
                    market_value=new_qty * fill_price,
                )
        else:
            # Sell
            self._state.cash += trade_value - commission

            if position is not None:
                # Record trade
                if order.symbol in self._open_positions:
                    open_pos = self._open_positions[order.symbol]
                    pnl = (fill_price - open_pos["entry_price"]) * order.quantity - commission
                    pnl_pct = float(pnl / (open_pos["entry_price"] * order.quantity))

                    trade = Trade(
                        symbol=order.symbol,
                        entry_time=open_pos["entry_time"],
                        exit_time=fill_time,
                        side=OrderSide.BUY,  # Original entry side
                        quantity=order.quantity,
                        entry_price=open_pos["entry_price"],
                        exit_price=fill_price,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        commission=commission,
                        slippage=Decimal("0"),  # Already applied to fill_price
                        holding_period_bars=open_pos["bars_held"],
                    )
                    self._state.trades.append(trade)

                new_qty = position.quantity - order.quantity
                if new_qty <= 0:
                    del self._state.positions[order.symbol]
                    self._open_positions.pop(order.symbol, None)
                else:
                    self._state.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        quantity=new_qty,
                        avg_entry_price=position.avg_entry_price,
                        current_price=fill_price,
                        cost_basis=position.avg_entry_price * new_qty,
                        market_value=new_qty * fill_price,
                    )

        self._state.orders_filled += 1
        self.strategy.on_fill(order)

        for callback in self._on_fill_callbacks:
            callback(order, fill_price)

    def _process_signal(self, signal: TradeSignal) -> None:
        """Process a trading signal and create order."""
        bar = self.data_handler.get_current_bar(signal.symbol)
        if bar is None:
            return

        # Determine order side based on signal direction
        if signal.direction == Direction.LONG:
            side = OrderSide.BUY
        elif signal.direction == Direction.SHORT:
            side = OrderSide.SELL
        else:
            return

        # Calculate position size
        position_value = self._state.equity * Decimal(str(self.config.max_position_pct))
        quantity = position_value / bar.close

        if not self.config.allow_fractional:
            quantity = Decimal(int(quantity))

        if quantity <= 0:
            return

        # Create order
        order = Order(
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            signal_id=signal.signal_id,
        )

        self._state.pending_orders.append(order)

    def _update_equity(self) -> None:
        """Update equity and equity curve."""
        position_value = sum(
            p.market_value for p in self._state.positions.values()
        )
        self._state.equity = self._state.cash + position_value

        self._state.equity_curve.append(
            (self._state.timestamp, float(self._state.equity))
        )

        # Update holding periods
        for symbol in self._open_positions:
            self._open_positions[symbol]["bars_held"] += 1

    def _check_halt_conditions(self) -> bool:
        """Check if backtest should halt."""
        if len(self._state.equity_curve) < 2:
            return False

        peak_equity = max(e for _, e in self._state.equity_curve)
        current_equity = float(self._state.equity)
        drawdown = 1 - current_equity / peak_equity

        return drawdown >= self.config.max_drawdown_halt

    def _get_portfolio(self) -> Portfolio:
        """Get current portfolio state."""
        return Portfolio(
            timestamp=self._state.timestamp,
            equity=self._state.equity,
            cash=self._state.cash,
            buying_power=self._state.cash,
            positions=self._state.positions.copy(),
            pending_orders=self._state.pending_orders.copy(),
        )

    def add_bar_callback(self, callback: Callable) -> None:
        """Add callback for bar events."""
        self._on_bar_callbacks.append(callback)

    def add_signal_callback(self, callback: Callable) -> None:
        """Add callback for signal events."""
        self._on_signal_callbacks.append(callback)

    def add_fill_callback(self, callback: Callable) -> None:
        """Add callback for fill events."""
        self._on_fill_callbacks.append(callback)


class VectorizedBacktest:
    """Vectorized backtesting for fast research."""

    def __init__(
        self,
        data: pd.DataFrame,
        config: BacktestConfig | None = None,
    ) -> None:
        """Initialize vectorized backtest.

        Args:
            data: DataFrame with OHLCV data and signals.
            config: Backtest configuration.
        """
        self.data = data.copy()
        self.config = config or BacktestConfig()

    def run(
        self,
        signal_column: str = "signal",
        position_column: str | None = None,
    ) -> pd.DataFrame:
        """Run vectorized backtest.

        Args:
            signal_column: Column containing trading signals (-1, 0, 1).
            position_column: Column containing position sizes (if None, uses signal).

        Returns:
            DataFrame with backtest results.
        """
        df = self.data.copy()

        # Get positions from signals or position column
        if position_column and position_column in df.columns:
            df["position"] = df[position_column]
        else:
            df["position"] = df[signal_column].shift(1).fillna(0)

        # Calculate returns
        df["returns"] = df["close"].pct_change()
        df["strategy_returns"] = df["position"] * df["returns"]

        # Apply transaction costs
        df["trades"] = df["position"].diff().abs()
        cost_pct = (self.config.commission_bps + self.config.slippage_bps) / 10000
        df["costs"] = df["trades"] * cost_pct

        if self.config.execution_mode != ExecutionMode.OPTIMISTIC:
            df["strategy_returns"] -= df["costs"]

        # Calculate cumulative returns
        df["cumulative_returns"] = (1 + df["returns"]).cumprod()
        df["strategy_cumulative"] = (1 + df["strategy_returns"]).cumprod()

        # Calculate equity
        initial_capital = float(self.config.initial_capital)
        df["equity"] = initial_capital * df["strategy_cumulative"]
        df["benchmark_equity"] = initial_capital * df["cumulative_returns"]

        # Calculate drawdown
        df["peak"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"]

        return df

    def get_metrics(self, results: pd.DataFrame) -> dict[str, float]:
        """Calculate performance metrics from results.

        Args:
            results: DataFrame from run().

        Returns:
            Dictionary of performance metrics.
        """
        strategy_returns = results["strategy_returns"].dropna()

        total_return = results["strategy_cumulative"].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1

        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0

        max_drawdown = results["drawdown"].min()

        # Win rate
        winning_days = (strategy_returns > 0).sum()
        total_days = len(strategy_returns)
        win_rate = winning_days / total_days if total_days > 0 else 0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": results["trades"].sum(),
            "total_days": len(strategy_returns),
        }
