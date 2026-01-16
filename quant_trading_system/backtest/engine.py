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
from datetime import datetime, timezone
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
from quant_trading_system.backtest.simulator import (
    MarketSimulator,
    MarketConditions,
    FillType,
    FillResult,
    create_realistic_simulator,
    create_optimistic_simulator,
    create_pessimistic_simulator,
)

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

    # CRITICAL FIX: Strict mode blocks look-ahead bias configurations
    # When True (default), 'same_close' execution is blocked as it introduces look-ahead bias
    strict_mode: bool = True

    # Risk parameters
    max_position_pct: float = 0.10
    max_drawdown_halt: float = 0.20

    # Time filters
    start_date: datetime | None = None
    end_date: datetime | None = None

    # JPMORGAN FIX: Market simulation settings
    use_market_simulator: bool = True  # Enable realistic market simulation
    simulate_partial_fills: bool = True  # Allow partial order fills
    simulate_latency: bool = True  # Simulate execution latency
    avg_daily_volume: int = 1000000  # Default ADV for volume-based slippage


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

    # P1: Regime tracking for regime-specific analysis
    regime_history: list[tuple[datetime, Any]] = field(default_factory=list)

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


class PolarsDataHandler(DataHandler):
    """Data handler using Polars DataFrames for high-performance backtesting."""

    def __init__(
        self,
        data: dict[str, Any],  # Polars DataFrames
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> None:
        """Initialize Polars data handler.

        Args:
            data: Dictionary mapping symbols to Polars DataFrames with OHLCV data.
            start_date: Optional start date filter.
            end_date: Optional end date filter.
        """
        try:
            import polars as pl
            self._pl = pl
        except ImportError:
            raise ImportError("Polars is required for PolarsDataHandler")

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
        pl = self._pl

        # Get all timestamps from each symbol
        all_timestamps = []
        for symbol, df in self._data.items():
            if "timestamp" in df.columns:
                ts = df["timestamp"].to_list()
                all_timestamps.append(set(ts))

        if not all_timestamps:
            self._dates = []
            self._max_index = 0
            return

        # Find common timestamps across all symbols
        common_timestamps = all_timestamps[0]
        for ts_set in all_timestamps[1:]:
            common_timestamps = common_timestamps.intersection(ts_set)

        # Convert to sorted list
        dates = sorted(list(common_timestamps))

        # Apply date filters
        if start_date:
            start_naive = start_date.replace(tzinfo=None) if hasattr(start_date, 'tzinfo') and start_date.tzinfo else start_date
            dates = [d for d in dates if d >= start_naive]
        if end_date:
            end_naive = end_date.replace(tzinfo=None) if hasattr(end_date, 'tzinfo') and end_date.tzinfo else end_date
            dates = [d for d in dates if d <= end_naive]

        self._dates = dates
        self._max_index = len(self._dates)

        # Pre-index data for fast lookup
        self._data_indexed: dict[str, dict] = {}
        for symbol, df in self._data.items():
            if "timestamp" in df.columns:
                # Create a lookup dict for O(1) access
                self._data_indexed[symbol] = {}
                for i in range(len(df)):
                    row = df.row(i, named=True)
                    ts = row.get("timestamp")
                    if ts in common_timestamps:
                        self._data_indexed[symbol][ts] = row

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
            if symbol not in self._data_indexed:
                continue

            row = self._data_indexed[symbol].get(current_date)
            if row is not None:
                # Create OHLCVBar from Polars row
                bar = OHLCVBar(
                    symbol=symbol,
                    timestamp=current_date if isinstance(current_date, datetime) else datetime.now(timezone.utc),
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
        market_simulator: MarketSimulator | None = None,
        regime_detector: Any = None,
    ) -> None:
        """Initialize backtesting engine.

        Args:
            data_handler: Data handler for market data.
            strategy: Trading strategy to test.
            config: Backtest configuration.
            market_simulator: Optional custom market simulator.
            regime_detector: Optional regime detector for regime-specific analysis.
                            Should have a detect(data) method returning RegimeState.
        """
        self.data_handler = data_handler
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self._regime_detector = regime_detector

        # JPMORGAN FIX: Initialize market simulator based on execution mode
        if market_simulator is not None:
            self.market_simulator = market_simulator
        elif self.config.use_market_simulator:
            self.market_simulator = self._create_market_simulator()
        else:
            self.market_simulator = None

        # Initialize state
        self._state: BacktestState | None = None
        self._event_queue: deque = deque()
        self._running = False

        # Position tracking for trade calculation
        self._open_positions: dict[str, dict[str, Any]] = {}

        # JPMORGAN FIX: Track market conditions per symbol
        self._market_conditions: dict[str, MarketConditions] = {}
        self._volatility_window: int = 20  # Bars for volatility calculation
        self._price_history: dict[str, deque] = {}

        # Callbacks
        self._on_bar_callbacks: list[Callable] = []
        self._on_signal_callbacks: list[Callable] = []
        self._on_fill_callbacks: list[Callable] = []

    def _create_market_simulator(self) -> MarketSimulator:
        """Create market simulator based on execution mode."""
        if self.config.execution_mode == ExecutionMode.OPTIMISTIC:
            return create_optimistic_simulator()
        elif self.config.execution_mode == ExecutionMode.PESSIMISTIC:
            return create_pessimistic_simulator(
                commission_bps=self.config.commission_bps * 2,
                base_slippage_bps=self.config.slippage_bps * 2,
            )
        else:
            return create_realistic_simulator(
                commission_bps=self.config.commission_bps,
                base_slippage_bps=self.config.slippage_bps,
            )

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
            timestamp=datetime.now(timezone.utc),
            equity=self.config.initial_capital,
            cash=self.config.initial_capital,
            positions={},
            pending_orders=[],
            equity_curve=[],
            trades=[],
        )
        self._open_positions = {}

        # JPMORGAN FIX: Initialize price history for volatility calculation
        self._price_history = {
            s: deque(maxlen=self._volatility_window)
            for s in self.data_handler.get_symbols()
        }
        self._market_conditions = {}

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

        # P1: Track regime state if detector available
        if self._regime_detector is not None:
            try:
                # Try to detect regime from current data
                # The detector should accept price data and return a RegimeState
                regime_state = self._detect_current_regime()
                if regime_state is not None:
                    self._state.regime_history.append(
                        (self._state.timestamp, regime_state)
                    )
            except Exception as e:
                logger.debug(f"Regime detection failed: {e}")

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

        # JPMORGAN FIX: Update market conditions for realistic simulation
        self._update_market_conditions(symbol, bar)

    def _update_market_conditions(self, symbol: str, bar: OHLCVBar) -> None:
        """Update market conditions for a symbol based on current bar."""
        # Update price history for volatility calculation
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self._volatility_window)
        self._price_history[symbol].append(float(bar.close))

        # Calculate realized volatility
        volatility = 0.02  # Default
        if len(self._price_history[symbol]) >= 2:
            prices = list(self._price_history[symbol])
            returns = [
                (prices[i] - prices[i-1]) / prices[i-1]
                for i in range(1, len(prices))
            ]
            if returns:
                volatility = float(np.std(returns) * np.sqrt(252))  # Annualized

        # Estimate bid/ask from high/low
        spread = bar.high - bar.low
        half_spread = spread / Decimal("4")  # Conservative estimate
        bid = bar.close - half_spread
        ask = bar.close + half_spread

        self._market_conditions[symbol] = MarketConditions(
            price=bar.close,
            bid=bid,
            ask=ask,
            volume=bar.volume,
            avg_daily_volume=self.config.avg_daily_volume,
            volatility=volatility,
            spread_bps=float(spread / bar.close * 10000) if bar.close > 0 else 5.0,
            timestamp=bar.timestamp,
        )

    def _process_pending_orders(self, symbol: str, bar: OHLCVBar) -> None:
        """Process pending orders for a symbol."""
        orders_to_remove = []

        for i, order in enumerate(self._state.pending_orders):
            if order.symbol != symbol:
                continue

            # JPMORGAN FIX: Use MarketSimulator for realistic execution
            if self.market_simulator is not None and symbol in self._market_conditions:
                fill_result = self._simulate_order_execution(order, symbol, bar)
                if fill_result is not None:
                    self._fill_order_with_result(order, fill_result, bar.timestamp)
                    orders_to_remove.append(i)
            else:
                # Fallback to simple execution
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

    def _simulate_order_execution(
        self,
        order: Order,
        symbol: str,
        bar: OHLCVBar,
    ) -> FillResult | None:
        """Simulate order execution using MarketSimulator.

        Returns FillResult if order should be filled, None otherwise.
        """
        conditions = self._market_conditions.get(symbol)
        if conditions is None:
            return None

        # Update conditions with execution bar price
        execution_price = self._get_fill_price(bar)
        if execution_price is None:
            return None

        conditions = MarketConditions(
            price=execution_price,
            bid=conditions.bid,
            ask=conditions.ask,
            volume=bar.volume,
            avg_daily_volume=conditions.avg_daily_volume,
            volatility=conditions.volatility,
            spread_bps=conditions.spread_bps,
            timestamp=bar.timestamp,
        )

        # Simulate execution
        fill_result = self.market_simulator.simulate_execution(order, conditions)

        # Check rejection
        if fill_result.fill_type == FillType.REJECTED:
            logger.debug(
                f"Order rejected: {order.symbol} - {fill_result.rejection_reason}"
            )
            return None

        # Check if we can afford the fill
        if not self._can_fill_order_amount(
            order,
            fill_result.fill_price,
            fill_result.fill_quantity,
            fill_result.commission,
        ):
            return None

        return fill_result

    def _can_fill_order_amount(
        self,
        order: Order,
        fill_price: Decimal,
        fill_quantity: Decimal,
        commission: Decimal,
    ) -> bool:
        """Check if order can be filled with given amount."""
        trade_value = fill_quantity * fill_price

        if order.side == OrderSide.BUY:
            return self._state.cash >= trade_value + commission

        # For sells, check position
        position = self._state.positions.get(order.symbol)
        if position is None:
            return self.config.allow_short
        return position.quantity >= fill_quantity or self.config.allow_short

    def _fill_order_with_result(
        self,
        order: Order,
        fill_result: FillResult,
        fill_time: datetime,
    ) -> None:
        """Fill an order using MarketSimulator result."""
        fill_price = fill_result.fill_price
        fill_quantity = fill_result.fill_quantity
        commission = fill_result.commission
        trade_value = fill_quantity * fill_price

        # Update position
        position = self._state.positions.get(order.symbol)

        if order.side == OrderSide.BUY:
            self._state.cash -= trade_value + commission

            if position is None:
                # New position
                self._state.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=fill_quantity,
                    avg_entry_price=fill_price,
                    current_price=fill_price,
                    cost_basis=trade_value,
                    market_value=trade_value,
                )
                self._open_positions[order.symbol] = {
                    "entry_time": fill_time,
                    "entry_price": fill_price,
                    "quantity": fill_quantity,
                    "bars_held": 0,
                    "entry_side": order.side,
                    "total_slippage": fill_result.slippage,
                    "total_commission": commission,
                }
            else:
                # Add to position
                new_qty = position.quantity + fill_quantity
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
                # Update open position tracking
                if order.symbol in self._open_positions:
                    self._open_positions[order.symbol]["total_slippage"] += fill_result.slippage
                    self._open_positions[order.symbol]["total_commission"] += commission
        else:
            # Sell
            self._state.cash += trade_value - commission

            if position is not None:
                # Record trade
                if order.symbol in self._open_positions:
                    open_pos = self._open_positions[order.symbol]
                    entry_side = open_pos.get("entry_side", OrderSide.BUY)

                    if entry_side == OrderSide.BUY:
                        pnl = (fill_price - open_pos["entry_price"]) * fill_quantity - commission
                    else:
                        pnl = (open_pos["entry_price"] - fill_price) * fill_quantity - commission

                    pnl_pct = float(pnl / (open_pos["entry_price"] * fill_quantity))

                    trade = Trade(
                        symbol=order.symbol,
                        entry_time=open_pos["entry_time"],
                        exit_time=fill_time,
                        side=entry_side,
                        quantity=fill_quantity,
                        entry_price=open_pos["entry_price"],
                        exit_price=fill_price,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        commission=commission + open_pos.get("total_commission", Decimal("0")),
                        slippage=fill_result.slippage + open_pos.get("total_slippage", Decimal("0")),
                        holding_period_bars=open_pos["bars_held"],
                        metadata={
                            "fill_type": fill_result.fill_type.value,
                            "latency_ms": fill_result.latency_ms,
                            "market_impact": float(fill_result.market_impact),
                        },
                    )
                    self._state.trades.append(trade)

                new_qty = position.quantity - fill_quantity
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

        # Handle partial fills - re-queue remaining
        if fill_result.fill_type == FillType.PARTIAL and fill_result.partial_remaining > 0:
            remaining_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=fill_result.partial_remaining,
                signal_id=order.signal_id,
            )
            self._state.pending_orders.append(remaining_order)
            logger.debug(
                f"Partial fill for {order.symbol}: {fill_quantity} filled, "
                f"{fill_result.partial_remaining} remaining"
            )

        self._state.orders_filled += 1
        self.strategy.on_fill(order)

        for callback in self._on_fill_callbacks:
            callback(order, fill_price)

    def _get_fill_price(self, bar: OHLCVBar) -> Decimal | None:
        """Get fill price based on execution config.

        CRITICAL FIX: In strict_mode (default), 'same_close' is blocked as it
        introduces look-ahead bias - the signal uses close data but execution
        happens at the same close, which is not realistically possible.

        WARNING: 'same_close' execution has look-ahead bias and should only
        be used for theoretical analysis. Use 'next_open' for realistic results.

        Raises:
            ValueError: If same_close is used with strict_mode=True.
        """
        if self.config.fill_at == "next_open":
            return bar.open
        elif self.config.fill_at == "next_close":
            return bar.close
        elif self.config.fill_at == "same_close":
            # CRITICAL FIX: Block same_close in strict_mode (default)
            if self.config.strict_mode:
                raise ValueError(
                    "LOOK-AHEAD BIAS ERROR: 'same_close' execution is blocked in strict_mode. "
                    "This mode uses close price for signals but executes at the same close, "
                    "which introduces look-ahead bias and produces unrealistic results. "
                    "Use 'next_open' for realistic backtesting. "
                    "Set strict_mode=False only for theoretical/research analysis."
                )
            # WARNING: This introduces look-ahead bias since signal uses close data
            # but execution happens at the same close. Only use for theoretical analysis.
            logger.warning(
                "Using same_close execution mode - this introduces look-ahead bias. "
                "Use next_open for realistic backtesting results."
            )
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
                    "entry_side": order.side,  # Track entry side for correct P&L calculation
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

                    # CRITICAL FIX: Calculate P&L correctly based on entry side
                    # Long position (entry_side=BUY): pnl = (exit - entry) * qty
                    # Short position (entry_side=SELL): pnl = (entry - exit) * qty
                    entry_side = open_pos.get("entry_side", OrderSide.BUY)
                    if entry_side == OrderSide.BUY:
                        # Closing a long position
                        pnl = (fill_price - open_pos["entry_price"]) * order.quantity - commission
                    else:
                        # Closing a short position
                        pnl = (open_pos["entry_price"] - fill_price) * order.quantity - commission

                    pnl_pct = float(pnl / (open_pos["entry_price"] * order.quantity))

                    trade = Trade(
                        symbol=order.symbol,
                        entry_time=open_pos["entry_time"],
                        exit_time=fill_time,
                        side=entry_side,  # Original entry side (BUY for long, SELL for short)
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

    def _detect_current_regime(self) -> Any:
        """
        Detect current market regime using the regime detector.

        Returns:
            RegimeState if detection succeeds, None otherwise.
        """
        if self._regime_detector is None:
            return None

        # Build data for regime detection from price history
        # Use first symbol's price history as representative
        symbols = self.data_handler.get_symbols()
        if not symbols:
            return None

        symbol = symbols[0]
        if symbol not in self._price_history:
            return None

        prices = list(self._price_history[symbol])
        if len(prices) < 10:  # Need minimum data
            return None

        # Create DataFrame-like structure for regime detector
        import pandas as pd
        price_df = pd.DataFrame({
            'close': prices,
        })

        # Calculate returns for volatility estimation
        price_df['returns'] = price_df['close'].pct_change()

        # Try different detector interfaces
        if hasattr(self._regime_detector, 'detect'):
            return self._regime_detector.detect(price_df)
        elif hasattr(self._regime_detector, 'detect_regime'):
            return self._regime_detector.detect_regime(price_df)
        elif callable(self._regime_detector):
            return self._regime_detector(price_df)

        return None


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


@dataclass
class WalkForwardWindow:
    """Configuration for a single walk-forward validation window."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int
    gap_bars: int = 0


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward window."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    equity_curve: list[tuple[datetime, float]]
    trades: list[Trade]
    is_oos_ratio: float = 0.0  # In-sample vs Out-of-sample performance ratio


@dataclass
class WalkForwardReport:
    """Aggregate report from walk-forward validation."""
    windows: list[WalkForwardResult]
    aggregate_metrics: dict[str, float]
    is_valid: bool
    validation_messages: list[str]
    total_trades: int = 0
    combined_equity_curve: list[tuple[datetime, float]] = field(default_factory=list)


class WalkForwardValidator:
    """
    MAJOR FIX: Walk-forward validation for backtest robustness.

    Walk-forward analysis divides data into multiple train/test windows,
    training the model on each training period and testing on the subsequent
    out-of-sample period. This is the gold standard for validating trading
    strategies and detecting overfitting.

    Critical for institutional-grade backtesting where in-sample optimization
    can lead to significantly overestimated performance.
    """

    def __init__(
        self,
        n_windows: int = 5,
        train_pct: float = 0.7,
        gap_bars: int = 5,
        min_train_samples: int = 500,
        min_test_samples: int = 100,
    ):
        """
        Initialize walk-forward validator.

        Args:
            n_windows: Number of walk-forward windows
            train_pct: Percentage of each window for training (rest for testing)
            gap_bars: Gap between train and test to prevent leakage
            min_train_samples: Minimum training samples per window
            min_test_samples: Minimum testing samples per window
        """
        self.n_windows = n_windows
        self.train_pct = train_pct
        self.gap_bars = gap_bars
        self.min_train_samples = min_train_samples
        self.min_test_samples = min_test_samples

        self._logger = logging.getLogger(__name__)

    def create_windows(
        self,
        data: pd.DataFrame,
        date_column: str | None = None,
    ) -> list[WalkForwardWindow]:
        """
        Create walk-forward windows from data.

        Args:
            data: DataFrame with time index or date column
            date_column: Optional column name for dates (if not index)

        Returns:
            List of WalkForwardWindow configurations
        """
        if date_column is not None:
            dates = pd.to_datetime(data[date_column])
        else:
            dates = pd.to_datetime(data.index)

        n_samples = len(dates)
        window_size = n_samples // self.n_windows
        train_size = int(window_size * self.train_pct)
        test_size = window_size - train_size - self.gap_bars

        if train_size < self.min_train_samples:
            self._logger.warning(
                f"Train size {train_size} is below minimum {self.min_train_samples}. "
                f"Consider reducing n_windows or increasing data."
            )

        if test_size < self.min_test_samples:
            self._logger.warning(
                f"Test size {test_size} is below minimum {self.min_test_samples}. "
                f"Consider reducing n_windows or gap_bars."
            )

        windows = []
        for i in range(self.n_windows):
            start_idx = i * window_size
            train_end_idx = start_idx + train_size
            test_start_idx = train_end_idx + self.gap_bars
            test_end_idx = min((i + 1) * window_size, n_samples)

            if test_end_idx > n_samples or test_start_idx >= test_end_idx:
                break

            window = WalkForwardWindow(
                train_start=dates.iloc[start_idx].to_pydatetime() if hasattr(dates.iloc[start_idx], 'to_pydatetime') else dates.iloc[start_idx],
                train_end=dates.iloc[train_end_idx - 1].to_pydatetime() if hasattr(dates.iloc[train_end_idx - 1], 'to_pydatetime') else dates.iloc[train_end_idx - 1],
                test_start=dates.iloc[test_start_idx].to_pydatetime() if hasattr(dates.iloc[test_start_idx], 'to_pydatetime') else dates.iloc[test_start_idx],
                test_end=dates.iloc[test_end_idx - 1].to_pydatetime() if hasattr(dates.iloc[test_end_idx - 1], 'to_pydatetime') else dates.iloc[test_end_idx - 1],
                window_id=i,
                gap_bars=self.gap_bars,
            )
            windows.append(window)

        self._logger.info(f"Created {len(windows)} walk-forward windows")
        return windows

    def run_validation(
        self,
        data: dict[str, pd.DataFrame],
        strategy_factory: Callable[[pd.DataFrame], Strategy],
        config: BacktestConfig | None = None,
        train_callback: Callable[[pd.DataFrame, WalkForwardWindow], None] | None = None,
    ) -> WalkForwardReport:
        """
        Run walk-forward validation.

        Args:
            data: Dictionary mapping symbols to DataFrames
            strategy_factory: Factory function to create strategy from training data
            config: Backtest configuration
            train_callback: Optional callback called after training each window

        Returns:
            WalkForwardReport with all results
        """
        config = config or BacktestConfig()

        # Get common dates across all symbols
        first_symbol = list(data.keys())[0]
        df = data[first_symbol]
        windows = self.create_windows(df)

        if len(windows) == 0:
            return WalkForwardReport(
                windows=[],
                aggregate_metrics={},
                is_valid=False,
                validation_messages=["No valid windows could be created from data"],
            )

        results = []
        all_trades = []
        combined_equity = []

        for window in windows:
            self._logger.info(
                f"Running window {window.window_id + 1}/{len(windows)}: "
                f"Train {window.train_start.date()} to {window.train_end.date()}, "
                f"Test {window.test_start.date()} to {window.test_end.date()}"
            )

            # Filter data for window
            train_data = {}
            test_data = {}
            for symbol, symbol_df in data.items():
                # Handle both index and regular datetime index
                idx = symbol_df.index
                if hasattr(idx, 'tz') and idx.tz is not None:
                    # Timezone-aware, make window dates aware
                    from datetime import timezone as tz
                    train_mask = (idx >= pd.Timestamp(window.train_start, tz=idx.tz)) & \
                                 (idx <= pd.Timestamp(window.train_end, tz=idx.tz))
                    test_mask = (idx >= pd.Timestamp(window.test_start, tz=idx.tz)) & \
                                (idx <= pd.Timestamp(window.test_end, tz=idx.tz))
                else:
                    train_mask = (idx >= pd.Timestamp(window.train_start)) & \
                                 (idx <= pd.Timestamp(window.train_end))
                    test_mask = (idx >= pd.Timestamp(window.test_start)) & \
                                (idx <= pd.Timestamp(window.test_end))

                train_data[symbol] = symbol_df[train_mask].copy()
                test_data[symbol] = symbol_df[test_mask].copy()

            # Skip if insufficient data
            if any(len(df) < self.min_train_samples for df in train_data.values()):
                self._logger.warning(f"Skipping window {window.window_id}: insufficient training data")
                continue

            if any(len(df) < self.min_test_samples for df in test_data.values()):
                self._logger.warning(f"Skipping window {window.window_id}: insufficient test data")
                continue

            # Create strategy from training data
            try:
                # Merge training data for strategy factory
                merged_train = pd.concat([df for df in train_data.values()], axis=0)
                strategy = strategy_factory(merged_train)

                if train_callback:
                    train_callback(merged_train, window)
            except Exception as e:
                self._logger.error(f"Strategy creation failed for window {window.window_id}: {e}")
                continue

            # Run backtest on training data (in-sample metrics)
            train_handler = PandasDataHandler(train_data)
            train_engine = BacktestEngine(train_handler, strategy, config)
            train_state = train_engine.run()
            train_metrics = self._calculate_metrics(train_state)

            # Run backtest on test data (out-of-sample metrics)
            test_handler = PandasDataHandler(test_data)
            test_engine = BacktestEngine(test_handler, strategy, config)
            test_state = test_engine.run()
            test_metrics = self._calculate_metrics(test_state)

            # Calculate IS/OOS ratio
            train_sharpe = train_metrics.get("sharpe_ratio", 0)
            test_sharpe = test_metrics.get("sharpe_ratio", 0)
            is_oos_ratio = train_sharpe / test_sharpe if test_sharpe > 0 else float("inf")

            result = WalkForwardResult(
                window_id=window.window_id,
                train_start=window.train_start,
                train_end=window.train_end,
                test_start=window.test_start,
                test_end=window.test_end,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                equity_curve=test_state.equity_curve,
                trades=test_state.trades,
                is_oos_ratio=is_oos_ratio,
            )
            results.append(result)
            all_trades.extend(test_state.trades)
            combined_equity.extend(test_state.equity_curve)

        # Aggregate results
        aggregate_metrics = self._aggregate_metrics(results)
        is_valid, messages = self._validate_results(results, aggregate_metrics)

        return WalkForwardReport(
            windows=results,
            aggregate_metrics=aggregate_metrics,
            is_valid=is_valid,
            validation_messages=messages,
            total_trades=len(all_trades),
            combined_equity_curve=sorted(combined_equity, key=lambda x: x[0]),
        )

    def _calculate_metrics(self, state: BacktestState) -> dict[str, float]:
        """Calculate metrics from backtest state."""
        if not state.equity_curve:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "trade_count": 0,
            }

        equity_values = [e for _, e in state.equity_curve]
        returns = [
            (equity_values[i] - equity_values[i - 1]) / equity_values[i - 1]
            for i in range(1, len(equity_values))
            if equity_values[i - 1] > 0
        ]

        if not returns:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "trade_count": 0,
            }

        total_return = (equity_values[-1] / equity_values[0] - 1) if equity_values[0] > 0 else 0
        volatility = float(np.std(returns) * np.sqrt(252 * 26)) if returns else 0  # 15-min bars
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 26)) if returns and np.std(returns) > 0 else 0

        # Max drawdown
        peak = equity_values[0]
        max_dd = 0.0
        for val in equity_values:
            peak = max(peak, val)
            dd = (peak - val) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Win rate from trades
        winning_trades = sum(1 for t in state.trades if float(t.pnl) > 0)
        total_trades = len(state.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "volatility": volatility,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "trade_count": total_trades,
        }

    def _aggregate_metrics(self, results: list[WalkForwardResult]) -> dict[str, float]:
        """Aggregate metrics across all walk-forward windows."""
        if not results:
            return {}

        # Average OOS metrics across windows
        oos_metrics = {
            "avg_oos_return": np.mean([r.test_metrics.get("total_return", 0) for r in results]),
            "avg_oos_sharpe": np.mean([r.test_metrics.get("sharpe_ratio", 0) for r in results]),
            "avg_oos_max_dd": np.mean([r.test_metrics.get("max_drawdown", 0) for r in results]),
            "avg_oos_win_rate": np.mean([r.test_metrics.get("win_rate", 0) for r in results]),
            "std_oos_sharpe": np.std([r.test_metrics.get("sharpe_ratio", 0) for r in results]),
            "avg_is_oos_ratio": np.mean([r.is_oos_ratio for r in results if r.is_oos_ratio < float("inf")]),
            "max_is_oos_ratio": max(r.is_oos_ratio for r in results if r.is_oos_ratio < float("inf")) if any(r.is_oos_ratio < float("inf") for r in results) else float("inf"),
            "windows_positive": sum(1 for r in results if r.test_metrics.get("total_return", 0) > 0),
            "windows_total": len(results),
        }

        # IS (in-sample) metrics for comparison
        oos_metrics.update({
            "avg_is_return": np.mean([r.train_metrics.get("total_return", 0) for r in results]),
            "avg_is_sharpe": np.mean([r.train_metrics.get("sharpe_ratio", 0) for r in results]),
        })

        return oos_metrics

    def _validate_results(
        self,
        results: list[WalkForwardResult],
        aggregate: dict[str, float],
    ) -> tuple[bool, list[str]]:
        """Validate walk-forward results for overfitting and robustness."""
        messages = []
        is_valid = True

        if not results:
            return False, ["No walk-forward windows completed"]

        # Check 1: Average IS/OOS ratio (overfitting indicator)
        avg_ratio = aggregate.get("avg_is_oos_ratio", 0)
        if avg_ratio > 2.0:
            messages.append(
                f"OVERFITTING WARNING: Average IS/OOS ratio {avg_ratio:.2f} > 2.0. "
                f"Strategy performs much better on training data than test data."
            )
            is_valid = False

        # Check 2: OOS Sharpe consistency
        std_sharpe = aggregate.get("std_oos_sharpe", 0)
        avg_sharpe = aggregate.get("avg_oos_sharpe", 0)
        if avg_sharpe > 0 and std_sharpe / avg_sharpe > 1.0:
            messages.append(
                f"INSTABILITY WARNING: High OOS Sharpe variance. "
                f"Avg={avg_sharpe:.2f}, Std={std_sharpe:.2f}. "
                f"Strategy performance is inconsistent across windows."
            )

        # Check 3: Percentage of positive windows
        windows_positive = aggregate.get("windows_positive", 0)
        windows_total = aggregate.get("windows_total", 1)
        pct_positive = windows_positive / windows_total
        if pct_positive < 0.6:
            messages.append(
                f"ROBUSTNESS WARNING: Only {pct_positive:.1%} of windows are profitable. "
                f"Strategy may not be robust across market conditions."
            )

        # Check 4: Average OOS Sharpe is positive
        if avg_sharpe <= 0:
            messages.append(
                f"PERFORMANCE WARNING: Average OOS Sharpe ratio is {avg_sharpe:.2f}. "
                f"Strategy does not show positive risk-adjusted returns."
            )
            is_valid = False

        if is_valid:
            messages.append(
                f"WALK-FORWARD VALIDATION PASSED: {windows_total} windows, "
                f"{pct_positive:.1%} profitable, avg OOS Sharpe={avg_sharpe:.2f}, "
                f"avg IS/OOS ratio={avg_ratio:.2f}"
            )

        return is_valid, messages
