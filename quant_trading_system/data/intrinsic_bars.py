"""
Intrinsic Time Bars - Alternative Bar Sampling Methods

This module implements information-driven bar sampling that provides more
uniform information content compared to time-based bars. Based on Marcos
Lopez de Prado's "Advances in Financial Machine Learning".

Bar Types:
- Tick Bars: Sample every N transactions
- Volume Bars: Sample every V units of volume
- Dollar Bars: Sample every D dollars traded
- Imbalance Bars: Sample when order flow imbalance exceeds threshold
- Run Bars: Sample when price runs exceed threshold

Author: AlphaTrade System
Version: 1.0.0
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Iterator, Tuple
from collections import deque

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class BarType(str, Enum):
    """Types of intrinsic time bars."""
    TICK = "tick"
    VOLUME = "volume"
    DOLLAR = "dollar"
    TICK_IMBALANCE = "tick_imbalance"
    VOLUME_IMBALANCE = "volume_imbalance"
    TICK_RUN = "tick_run"
    VOLUME_RUN = "volume_run"


@dataclass
class BarConfig:
    """Configuration for intrinsic bar generation."""
    bar_type: BarType
    threshold: float  # Ticks, volume, or dollars depending on bar_type

    # For imbalance/run bars
    initial_expected_ticks: int = 100
    ewm_span: int = 100  # EWM span for expected values

    # Validation
    min_threshold: float = 1.0
    max_bars_per_day: int = 10000

    def __post_init__(self):
        if self.threshold < self.min_threshold:
            raise ValueError(
                f"Threshold {self.threshold} below minimum {self.min_threshold}"
            )


@dataclass
class Trade:
    """Single trade tick."""
    timestamp: datetime
    price: float
    volume: float
    side: Optional[str] = None  # 'buy', 'sell', or None if unknown

    @property
    def dollar_value(self) -> float:
        """Dollar value of the trade."""
        return self.price * self.volume


@dataclass
class IntrinsicBar:
    """A bar generated from intrinsic time sampling."""
    timestamp: datetime  # Bar close time
    open: float
    high: float
    low: float
    close: float
    volume: float
    dollar_volume: float
    tick_count: int
    vwap: float

    # Additional metrics
    buy_volume: float = 0.0
    sell_volume: float = 0.0

    # Bar metadata
    bar_type: BarType = BarType.TICK
    threshold_used: float = 0.0
    duration_seconds: float = 0.0

    @property
    def ofi(self) -> float:
        """Order flow imbalance."""
        total = self.buy_volume + self.sell_volume
        if total == 0:
            return 0.0
        return (self.buy_volume - self.sell_volume) / total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'dollar_volume': self.dollar_volume,
            'tick_count': self.tick_count,
            'vwap': self.vwap,
            'buy_volume': self.buy_volume,
            'sell_volume': self.sell_volume,
            'ofi': self.ofi,
            'bar_type': self.bar_type.value,
            'threshold_used': self.threshold_used,
            'duration_seconds': self.duration_seconds,
        }


# =============================================================================
# TICK CLASSIFICATION
# =============================================================================

class TickClassifier:
    """
    Classify trades as buy or sell using tick rule.

    Uses the Lee-Ready algorithm:
    - If price > midpoint: buy
    - If price < midpoint: sell
    - If price = midpoint: use tick rule (compare to previous price)
    """

    def __init__(self):
        self._last_price: Optional[float] = None
        self._last_direction: int = 1  # 1 for buy, -1 for sell

    def classify(
        self,
        price: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None
    ) -> int:
        """
        Classify a trade as buy (1) or sell (-1).

        Args:
            price: Trade price
            bid: Current bid price (optional)
            ask: Current ask price (optional)

        Returns:
            1 for buy, -1 for sell
        """
        direction = self._last_direction

        # Lee-Ready: Compare to midpoint if available
        if bid is not None and ask is not None and bid < ask:
            midpoint = (bid + ask) / 2
            if price > midpoint:
                direction = 1
            elif price < midpoint:
                direction = -1
            # If at midpoint, fall through to tick rule
            else:
                direction = self._tick_rule(price)
        else:
            # Pure tick rule
            direction = self._tick_rule(price)

        self._last_price = price
        self._last_direction = direction

        return direction

    def _tick_rule(self, price: float) -> int:
        """Apply tick rule based on price change."""
        if self._last_price is None:
            return 1  # Default to buy for first tick

        if price > self._last_price:
            return 1
        elif price < self._last_price:
            return -1
        else:
            return self._last_direction  # No change, keep direction

    def reset(self):
        """Reset classifier state."""
        self._last_price = None
        self._last_direction = 1


# =============================================================================
# BASE BAR GENERATOR
# =============================================================================

class BaseBarGenerator(ABC):
    """Abstract base class for bar generators."""

    def __init__(self, config: BarConfig):
        self.config = config
        self._classifier = TickClassifier()
        self._reset_accumulator()

        # Statistics
        self._bars_generated: int = 0
        self._ticks_processed: int = 0

    def _reset_accumulator(self):
        """Reset trade accumulator for new bar."""
        self._trades: List[Trade] = []
        self._cumulative_volume: float = 0.0
        self._cumulative_dollars: float = 0.0
        self._buy_volume: float = 0.0
        self._sell_volume: float = 0.0
        self._bar_start_time: Optional[datetime] = None

    @abstractmethod
    def _should_emit_bar(self) -> bool:
        """Check if threshold is reached to emit a bar."""
        pass

    def process_trade(self, trade: Trade) -> Optional[IntrinsicBar]:
        """
        Process a single trade and potentially emit a bar.

        Args:
            trade: The trade to process

        Returns:
            IntrinsicBar if threshold reached, None otherwise
        """
        self._ticks_processed += 1

        if self._bar_start_time is None:
            self._bar_start_time = trade.timestamp

        # Classify trade direction
        direction = self._classifier.classify(trade.price)

        # Accumulate
        self._trades.append(trade)
        self._cumulative_volume += trade.volume
        self._cumulative_dollars += trade.dollar_value

        if direction == 1:
            self._buy_volume += trade.volume
        else:
            self._sell_volume += trade.volume

        # Check if we should emit a bar
        if self._should_emit_bar():
            bar = self._create_bar()
            self._reset_accumulator()
            return bar

        return None

    def process_trades(self, trades: List[Trade]) -> List[IntrinsicBar]:
        """Process multiple trades and return any generated bars."""
        bars = []
        for trade in trades:
            bar = self.process_trade(trade)
            if bar is not None:
                bars.append(bar)
        return bars

    def _create_bar(self) -> IntrinsicBar:
        """Create a bar from accumulated trades."""
        if not self._trades:
            raise ValueError("Cannot create bar with no trades")

        prices = [t.price for t in self._trades]
        volumes = [t.volume for t in self._trades]

        # Calculate VWAP
        total_dv = sum(t.dollar_value for t in self._trades)
        total_vol = sum(volumes)
        vwap = total_dv / total_vol if total_vol > 0 else prices[-1]

        # Calculate duration
        duration = 0.0
        if self._bar_start_time and self._trades:
            delta = self._trades[-1].timestamp - self._bar_start_time
            duration = delta.total_seconds()

        bar = IntrinsicBar(
            timestamp=self._trades[-1].timestamp,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=total_vol,
            dollar_volume=total_dv,
            tick_count=len(self._trades),
            vwap=vwap,
            buy_volume=self._buy_volume,
            sell_volume=self._sell_volume,
            bar_type=self.config.bar_type,
            threshold_used=self.config.threshold,
            duration_seconds=duration,
        )

        self._bars_generated += 1
        return bar

    def flush(self) -> Optional[IntrinsicBar]:
        """Flush any remaining trades as a partial bar."""
        if self._trades:
            bar = self._create_bar()
            self._reset_accumulator()
            return bar
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            'bar_type': self.config.bar_type.value,
            'threshold': self.config.threshold,
            'bars_generated': self._bars_generated,
            'ticks_processed': self._ticks_processed,
            'avg_ticks_per_bar': (
                self._ticks_processed / self._bars_generated
                if self._bars_generated > 0 else 0
            ),
        }


# =============================================================================
# STANDARD BAR GENERATORS
# =============================================================================

class TickBarGenerator(BaseBarGenerator):
    """Generate bars every N ticks."""

    def __init__(self, tick_threshold: int):
        config = BarConfig(
            bar_type=BarType.TICK,
            threshold=float(tick_threshold),
        )
        super().__init__(config)

    def _should_emit_bar(self) -> bool:
        return len(self._trades) >= int(self.config.threshold)


class VolumeBarGenerator(BaseBarGenerator):
    """Generate bars every V units of volume."""

    def __init__(self, volume_threshold: float):
        config = BarConfig(
            bar_type=BarType.VOLUME,
            threshold=volume_threshold,
        )
        super().__init__(config)

    def _should_emit_bar(self) -> bool:
        return self._cumulative_volume >= self.config.threshold


class DollarBarGenerator(BaseBarGenerator):
    """Generate bars every D dollars traded."""

    def __init__(self, dollar_threshold: float):
        config = BarConfig(
            bar_type=BarType.DOLLAR,
            threshold=dollar_threshold,
        )
        super().__init__(config)

    def _should_emit_bar(self) -> bool:
        return self._cumulative_dollars >= self.config.threshold


# =============================================================================
# IMBALANCE BAR GENERATORS
# =============================================================================

class ImbalanceBarGenerator(BaseBarGenerator):
    """
    Generate bars when order flow imbalance exceeds expected threshold.

    Based on the idea that bars should be emitted when there's a significant
    directional flow, indicating informed trading.

    θ_t = |Σb_t| >= E[T] * E[|2P[b_t=1] - 1|]

    Where:
    - b_t is the tick direction (+1 for buy, -1 for sell)
    - E[T] is expected bar size in ticks
    - P[b_t=1] is probability of uptick
    """

    def __init__(
        self,
        initial_expected_ticks: int = 100,
        ewm_span: int = 100,
        is_volume_imbalance: bool = False,
    ):
        bar_type = (
            BarType.VOLUME_IMBALANCE if is_volume_imbalance
            else BarType.TICK_IMBALANCE
        )
        config = BarConfig(
            bar_type=bar_type,
            threshold=float(initial_expected_ticks),
            initial_expected_ticks=initial_expected_ticks,
            ewm_span=ewm_span,
        )
        super().__init__(config)

        self._is_volume = is_volume_imbalance

        # EWM state
        self._expected_ticks: float = float(initial_expected_ticks)
        self._expected_imbalance: float = 0.5  # E[2P-1], starts neutral
        self._bar_sizes: deque = deque(maxlen=ewm_span)
        self._bar_imbalances: deque = deque(maxlen=ewm_span)

        # Current bar imbalance tracking
        self._tick_imbalance: float = 0.0
        self._volume_imbalance: float = 0.0

    def _reset_accumulator(self):
        super()._reset_accumulator()
        self._tick_imbalance = 0.0
        self._volume_imbalance = 0.0

    def process_trade(self, trade: Trade) -> Optional[IntrinsicBar]:
        """Process trade with imbalance tracking."""
        if self._bar_start_time is None:
            self._bar_start_time = trade.timestamp

        # Classify and track imbalance
        direction = self._classifier.classify(trade.price)

        self._tick_imbalance += direction
        self._volume_imbalance += direction * trade.volume

        # Standard accumulation
        self._trades.append(trade)
        self._cumulative_volume += trade.volume
        self._cumulative_dollars += trade.dollar_value
        self._ticks_processed += 1

        if direction == 1:
            self._buy_volume += trade.volume
        else:
            self._sell_volume += trade.volume

        if self._should_emit_bar():
            bar = self._create_bar()
            self._update_expectations()
            self._reset_accumulator()
            return bar

        return None

    def _should_emit_bar(self) -> bool:
        """Emit when imbalance exceeds expected threshold."""
        threshold = self._expected_ticks * self._expected_imbalance

        if self._is_volume:
            return abs(self._volume_imbalance) >= threshold
        else:
            return abs(self._tick_imbalance) >= threshold

    def _update_expectations(self):
        """Update EWM expectations after bar emission."""
        bar_size = len(self._trades)

        if self._is_volume:
            imbalance = abs(self._volume_imbalance) / self._cumulative_volume \
                if self._cumulative_volume > 0 else 0
        else:
            imbalance = abs(self._tick_imbalance) / bar_size if bar_size > 0 else 0

        self._bar_sizes.append(bar_size)
        self._bar_imbalances.append(imbalance)

        # Update EWM
        alpha = 2.0 / (self.config.ewm_span + 1)
        self._expected_ticks = (
            alpha * bar_size + (1 - alpha) * self._expected_ticks
        )
        self._expected_imbalance = (
            alpha * imbalance + (1 - alpha) * self._expected_imbalance
        )

        # Ensure minimum expected imbalance
        self._expected_imbalance = max(0.1, self._expected_imbalance)


class TickImbalanceBarGenerator(ImbalanceBarGenerator):
    """Convenience class for tick imbalance bars."""

    def __init__(
        self,
        initial_expected_ticks: int = 100,
        ewm_span: int = 100,
    ):
        super().__init__(
            initial_expected_ticks=initial_expected_ticks,
            ewm_span=ewm_span,
            is_volume_imbalance=False,
        )


class VolumeImbalanceBarGenerator(ImbalanceBarGenerator):
    """Convenience class for volume imbalance bars."""

    def __init__(
        self,
        initial_expected_ticks: int = 100,
        ewm_span: int = 100,
    ):
        super().__init__(
            initial_expected_ticks=initial_expected_ticks,
            ewm_span=ewm_span,
            is_volume_imbalance=True,
        )


# =============================================================================
# RUN BAR GENERATORS
# =============================================================================

class RunBarGenerator(BaseBarGenerator):
    """
    Generate bars when price runs exceed expected threshold.

    Captures momentum moves by measuring consecutive same-direction ticks.
    """

    def __init__(
        self,
        initial_expected_ticks: int = 100,
        ewm_span: int = 100,
        is_volume_run: bool = False,
    ):
        bar_type = BarType.VOLUME_RUN if is_volume_run else BarType.TICK_RUN
        config = BarConfig(
            bar_type=bar_type,
            threshold=float(initial_expected_ticks),
            initial_expected_ticks=initial_expected_ticks,
            ewm_span=ewm_span,
        )
        super().__init__(config)

        self._is_volume = is_volume_run

        # EWM state
        self._expected_ticks: float = float(initial_expected_ticks)
        self._expected_run: float = 0.5
        self._bar_sizes: deque = deque(maxlen=ewm_span)
        self._bar_runs: deque = deque(maxlen=ewm_span)

        # Run tracking
        self._buy_run: float = 0.0
        self._sell_run: float = 0.0

    def _reset_accumulator(self):
        super()._reset_accumulator()
        self._buy_run = 0.0
        self._sell_run = 0.0

    def process_trade(self, trade: Trade) -> Optional[IntrinsicBar]:
        """Process trade with run tracking."""
        if self._bar_start_time is None:
            self._bar_start_time = trade.timestamp

        direction = self._classifier.classify(trade.price)

        # Track runs
        increment = trade.volume if self._is_volume else 1.0
        if direction == 1:
            self._buy_run += increment
        else:
            self._sell_run += increment

        # Standard accumulation
        self._trades.append(trade)
        self._cumulative_volume += trade.volume
        self._cumulative_dollars += trade.dollar_value
        self._ticks_processed += 1

        if direction == 1:
            self._buy_volume += trade.volume
        else:
            self._sell_volume += trade.volume

        if self._should_emit_bar():
            bar = self._create_bar()
            self._update_expectations()
            self._reset_accumulator()
            return bar

        return None

    def _should_emit_bar(self) -> bool:
        """Emit when run exceeds expected threshold."""
        threshold = self._expected_ticks * self._expected_run
        max_run = max(self._buy_run, self._sell_run)
        return max_run >= threshold

    def _update_expectations(self):
        """Update EWM expectations after bar emission."""
        bar_size = len(self._trades)
        max_run = max(self._buy_run, self._sell_run)

        if self._is_volume:
            run_ratio = max_run / self._cumulative_volume \
                if self._cumulative_volume > 0 else 0
        else:
            run_ratio = max_run / bar_size if bar_size > 0 else 0

        self._bar_sizes.append(bar_size)
        self._bar_runs.append(run_ratio)

        alpha = 2.0 / (self.config.ewm_span + 1)
        self._expected_ticks = alpha * bar_size + (1 - alpha) * self._expected_ticks
        self._expected_run = alpha * run_ratio + (1 - alpha) * self._expected_run
        self._expected_run = max(0.1, self._expected_run)


class TickRunBarGenerator(RunBarGenerator):
    """Convenience class for tick run bars."""

    def __init__(
        self,
        initial_expected_ticks: int = 100,
        ewm_span: int = 100,
    ):
        super().__init__(
            initial_expected_ticks=initial_expected_ticks,
            ewm_span=ewm_span,
            is_volume_run=False,
        )


class VolumeRunBarGenerator(RunBarGenerator):
    """Convenience class for volume run bars."""

    def __init__(
        self,
        initial_expected_ticks: int = 100,
        ewm_span: int = 100,
    ):
        super().__init__(
            initial_expected_ticks=initial_expected_ticks,
            ewm_span=ewm_span,
            is_volume_run=True,
        )


# =============================================================================
# DATAFRAME CONVERSION
# =============================================================================

class DataFrameBarConverter:
    """
    Convert time bars DataFrame to intrinsic bars.

    When tick-level data isn't available, this class synthesizes approximate
    tick data from OHLCV bars and generates intrinsic bars.
    """

    def __init__(self, generator: BaseBarGenerator):
        self.generator = generator

    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert time-based OHLCV DataFrame to intrinsic bars.

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]

        Returns:
            DataFrame of intrinsic bars
        """
        if df.empty:
            return pd.DataFrame()

        bars = []

        for _, row in df.iterrows():
            # Synthesize trades from OHLCV
            trades = self._synthesize_trades(row)

            # Process through generator
            for trade in trades:
                bar = self.generator.process_trade(trade)
                if bar is not None:
                    bars.append(bar.to_dict())

        # Flush remaining
        final_bar = self.generator.flush()
        if final_bar is not None:
            bars.append(final_bar.to_dict())

        if not bars:
            return pd.DataFrame()

        return pd.DataFrame(bars)

    def _synthesize_trades(self, row: pd.Series) -> List[Trade]:
        """
        Synthesize trade sequence from OHLCV bar.

        Creates a plausible sequence: open -> high/low -> close
        """
        trades = []
        timestamp = row.get('timestamp', row.name)

        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        volume = row.get('volume', 100)

        # Determine price sequence
        if c >= o:  # Bullish: O -> L -> H -> C
            prices = [o, l, h, c]
        else:  # Bearish: O -> H -> L -> C
            prices = [o, h, l, c]

        # Remove duplicates while preserving order
        unique_prices = []
        for p in prices:
            if not unique_prices or p != unique_prices[-1]:
                unique_prices.append(p)

        # Split volume across trades
        n_trades = len(unique_prices)
        vol_per_trade = volume / n_trades

        for price in unique_prices:
            trades.append(Trade(
                timestamp=timestamp,
                price=price,
                volume=vol_per_trade,
            ))

        return trades


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_bar_generator(
    bar_type: BarType,
    threshold: float = 100,
    **kwargs,
) -> BaseBarGenerator:
    """
    Factory function to create appropriate bar generator.

    Args:
        bar_type: Type of bar to generate
        threshold: Threshold for bar emission (meaning depends on bar_type)
        **kwargs: Additional parameters for specific bar types

    Returns:
        Configured bar generator
    """
    generators = {
        BarType.TICK: lambda: TickBarGenerator(int(threshold)),
        BarType.VOLUME: lambda: VolumeBarGenerator(threshold),
        BarType.DOLLAR: lambda: DollarBarGenerator(threshold),
        BarType.TICK_IMBALANCE: lambda: TickImbalanceBarGenerator(
            initial_expected_ticks=int(threshold),
            ewm_span=kwargs.get('ewm_span', 100),
        ),
        BarType.VOLUME_IMBALANCE: lambda: VolumeImbalanceBarGenerator(
            initial_expected_ticks=int(threshold),
            ewm_span=kwargs.get('ewm_span', 100),
        ),
        BarType.TICK_RUN: lambda: TickRunBarGenerator(
            initial_expected_ticks=int(threshold),
            ewm_span=kwargs.get('ewm_span', 100),
        ),
        BarType.VOLUME_RUN: lambda: VolumeRunBarGenerator(
            initial_expected_ticks=int(threshold),
            ewm_span=kwargs.get('ewm_span', 100),
        ),
    }

    if bar_type not in generators:
        raise ValueError(f"Unknown bar type: {bar_type}")

    return generators[bar_type]()


def convert_to_intrinsic_bars(
    df: pd.DataFrame,
    bar_type: BarType = BarType.VOLUME,
    threshold: float = 10000,
    **kwargs,
) -> pd.DataFrame:
    """
    Convert OHLCV DataFrame to intrinsic bars.

    Args:
        df: Source OHLCV DataFrame
        bar_type: Type of intrinsic bar
        threshold: Threshold for bar emission
        **kwargs: Additional parameters

    Returns:
        DataFrame of intrinsic bars
    """
    generator = create_bar_generator(bar_type, threshold, **kwargs)
    converter = DataFrameBarConverter(generator)
    return converter.convert(df)


# =============================================================================
# ADAPTIVE THRESHOLD CALCULATOR
# =============================================================================

class AdaptiveThresholdCalculator:
    """
    Calculate optimal thresholds for intrinsic bars based on market activity.

    Targets a specific number of bars per day for consistent analysis.
    """

    DEFAULT_BARS_PER_DAY: int = 100
    TRADING_HOURS_PER_DAY: float = 6.5  # Regular market hours

    def __init__(self, target_bars_per_day: int = DEFAULT_BARS_PER_DAY):
        self.target_bars_per_day = target_bars_per_day

    def calculate_thresholds(
        self,
        df: pd.DataFrame,
        lookback_days: int = 20,
    ) -> Dict[BarType, float]:
        """
        Calculate thresholds for each bar type.

        Args:
            df: Historical OHLCV data
            lookback_days: Days to use for calculation

        Returns:
            Dictionary of bar types to recommended thresholds
        """
        if df.empty:
            return self._default_thresholds()

        # Get recent data
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        df_recent = df.tail(lookback_days * 26)  # ~26 bars per day at 15min

        n_days = max(1, len(df_recent) / 26)

        # Calculate daily averages
        total_volume = df_recent['volume'].sum()
        avg_daily_volume = total_volume / n_days

        avg_price = df_recent['close'].mean()
        total_dollar_volume = (df_recent['volume'] * df_recent['close']).sum()
        avg_daily_dollars = total_dollar_volume / n_days

        # Estimate ticks (approximate from bars)
        avg_daily_ticks = len(df_recent) * 4 / n_days  # ~4 price points per bar

        thresholds = {
            BarType.TICK: max(10, avg_daily_ticks / self.target_bars_per_day),
            BarType.VOLUME: max(100, avg_daily_volume / self.target_bars_per_day),
            BarType.DOLLAR: max(10000, avg_daily_dollars / self.target_bars_per_day),
            BarType.TICK_IMBALANCE: max(10, avg_daily_ticks / self.target_bars_per_day),
            BarType.VOLUME_IMBALANCE: max(10, avg_daily_ticks / self.target_bars_per_day),
            BarType.TICK_RUN: max(10, avg_daily_ticks / self.target_bars_per_day),
            BarType.VOLUME_RUN: max(10, avg_daily_ticks / self.target_bars_per_day),
        }

        logger.info(
            f"Calculated thresholds for {self.target_bars_per_day} bars/day: "
            f"tick={thresholds[BarType.TICK]:.0f}, "
            f"volume={thresholds[BarType.VOLUME]:.0f}, "
            f"dollar=${thresholds[BarType.DOLLAR]:,.0f}"
        )

        return thresholds

    def _default_thresholds(self) -> Dict[BarType, float]:
        """Default thresholds when no data available."""
        return {
            BarType.TICK: 100,
            BarType.VOLUME: 10000,
            BarType.DOLLAR: 1000000,
            BarType.TICK_IMBALANCE: 100,
            BarType.VOLUME_IMBALANCE: 100,
            BarType.TICK_RUN: 100,
            BarType.VOLUME_RUN: 100,
        }


# =============================================================================
# MULTI-SYMBOL BAR MANAGER
# =============================================================================

class IntrinsicBarManager:
    """
    Manage intrinsic bar generation for multiple symbols.

    Provides per-symbol generators with adaptive thresholds.
    """

    def __init__(
        self,
        bar_type: BarType = BarType.VOLUME,
        target_bars_per_day: int = 100,
    ):
        self.bar_type = bar_type
        self.target_bars_per_day = target_bars_per_day
        self._generators: Dict[str, BaseBarGenerator] = {}
        self._thresholds: Dict[str, float] = {}
        self._threshold_calculator = AdaptiveThresholdCalculator(target_bars_per_day)

    def initialize_symbol(
        self,
        symbol: str,
        historical_data: Optional[pd.DataFrame] = None,
        threshold: Optional[float] = None,
    ) -> BaseBarGenerator:
        """
        Initialize generator for a symbol.

        Args:
            symbol: Trading symbol
            historical_data: Historical OHLCV for threshold calculation
            threshold: Manual threshold override

        Returns:
            Configured bar generator
        """
        if threshold is None and historical_data is not None:
            thresholds = self._threshold_calculator.calculate_thresholds(
                historical_data
            )
            threshold = thresholds.get(self.bar_type, 100)
        elif threshold is None:
            threshold = 100

        self._thresholds[symbol] = threshold
        self._generators[symbol] = create_bar_generator(self.bar_type, threshold)

        logger.info(
            f"Initialized {self.bar_type.value} bar generator for {symbol} "
            f"with threshold={threshold:.2f}"
        )

        return self._generators[symbol]

    def process_trade(
        self,
        symbol: str,
        trade: Trade,
    ) -> Optional[IntrinsicBar]:
        """Process a trade for a symbol."""
        if symbol not in self._generators:
            self.initialize_symbol(symbol)
        return self._generators[symbol].process_trade(trade)

    def get_generator(self, symbol: str) -> Optional[BaseBarGenerator]:
        """Get generator for a symbol."""
        return self._generators.get(symbol)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all symbols."""
        return {
            symbol: gen.get_stats()
            for symbol, gen in self._generators.items()
        }

    def flush_all(self) -> Dict[str, Optional[IntrinsicBar]]:
        """Flush all generators."""
        return {
            symbol: gen.flush()
            for symbol, gen in self._generators.items()
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums and Config
    'BarType',
    'BarConfig',
    'Trade',
    'IntrinsicBar',

    # Classifiers
    'TickClassifier',

    # Generators
    'BaseBarGenerator',
    'TickBarGenerator',
    'VolumeBarGenerator',
    'DollarBarGenerator',
    'TickImbalanceBarGenerator',
    'VolumeImbalanceBarGenerator',
    'TickRunBarGenerator',
    'VolumeRunBarGenerator',

    # Converters and Utilities
    'DataFrameBarConverter',
    'AdaptiveThresholdCalculator',
    'IntrinsicBarManager',

    # Factory Functions
    'create_bar_generator',
    'convert_to_intrinsic_bars',
]
