"""
Trading strategy base classes and implementations.

Defines the Strategy interface and common strategy implementations:
- MomentumStrategy: Trend-following based on price momentum
- MeanReversionStrategy: Revert-to-mean based on deviation
- StatisticalArbitrageStrategy: Pairs/basket trading
- MLStrategy: Machine learning model-based strategy

Strategies generate signals based on market data and model predictions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from quant_trading_system.core.data_types import (
    Direction,
    FeatureVector,
    ModelPrediction,
    OHLCVBar,
    Portfolio,
    Position,
    TradeSignal,
)

logger = logging.getLogger(__name__)


class StrategyType(str, Enum):
    """Strategy type classification."""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    STATISTICAL_ARBITRAGE = "stat_arb"
    ML_BASED = "ml_based"
    HYBRID = "hybrid"


class StrategyState(str, Enum):
    """Strategy state."""

    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class StrategyConfig:
    """Strategy configuration parameters."""

    strategy_id: str = ""
    strategy_type: StrategyType = StrategyType.MOMENTUM
    symbols: list[str] = field(default_factory=list)
    lookback_periods: int = 20
    signal_threshold: float = 0.5
    confidence_threshold: float = 0.6
    max_position_size: Decimal = Decimal("10000")
    max_positions: int = 10
    rebalance_threshold: float = 0.05
    cooldown_bars: int = 5
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.02
    use_take_profit: bool = True
    take_profit_pct: float = 0.04
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyMetrics:
    """Strategy performance metrics."""

    signals_generated: int = 0
    signals_executed: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    profit_factor: float = 0.0
    last_signal_time: datetime | None = None


class Strategy(ABC):
    """Base class for trading strategies.

    Strategies process market data and generate trading signals.
    Each strategy encapsulates a specific trading logic.
    """

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize strategy.

        Args:
            config: Strategy configuration.
        """
        self.config = config
        self.strategy_id = config.strategy_id or str(uuid4())[:8]
        self.state = StrategyState.IDLE
        self.metrics = StrategyMetrics()

        # Internal state
        self._bar_history: dict[str, list[OHLCVBar]] = {}
        self._signal_cooldowns: dict[str, int] = {}
        self._last_signals: dict[str, TradeSignal] = {}

    @abstractmethod
    def generate_signals(
        self,
        bars: dict[str, OHLCVBar],
        features: dict[str, FeatureVector] | None = None,
        predictions: dict[str, ModelPrediction] | None = None,
        portfolio: Portfolio | None = None,
    ) -> list[TradeSignal]:
        """Generate trading signals from market data.

        Args:
            bars: Current bars by symbol.
            features: Feature vectors by symbol.
            predictions: Model predictions by symbol.
            portfolio: Current portfolio state.

        Returns:
            List of generated signals.
        """
        pass

    def update(self, bars: dict[str, OHLCVBar]) -> None:
        """Update internal state with new bars.

        Args:
            bars: New bars by symbol.
        """
        for symbol, bar in bars.items():
            if symbol not in self._bar_history:
                self._bar_history[symbol] = []
            self._bar_history[symbol].append(bar)

            # Maintain lookback window
            max_history = self.config.lookback_periods * 2
            if len(self._bar_history[symbol]) > max_history:
                self._bar_history[symbol] = self._bar_history[symbol][-max_history:]

        # Decrement cooldowns
        for symbol in list(self._signal_cooldowns.keys()):
            self._signal_cooldowns[symbol] -= 1
            if self._signal_cooldowns[symbol] <= 0:
                del self._signal_cooldowns[symbol]

    def can_generate_signal(self, symbol: str) -> bool:
        """Check if strategy can generate signal for symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            True if signal generation allowed.
        """
        if self.state != StrategyState.ACTIVE:
            return False
        if symbol in self._signal_cooldowns:
            return False
        if symbol not in self._bar_history:
            return False
        if len(self._bar_history[symbol]) < self.config.lookback_periods:
            return False
        return True

    def record_signal(self, signal: TradeSignal) -> None:
        """Record a generated signal.

        Args:
            signal: Generated signal.
        """
        self._last_signals[signal.symbol] = signal
        self._signal_cooldowns[signal.symbol] = self.config.cooldown_bars
        self.metrics.signals_generated += 1
        self.metrics.last_signal_time = datetime.utcnow()

    def start(self) -> None:
        """Start the strategy."""
        self.state = StrategyState.ACTIVE
        logger.info(f"Strategy {self.strategy_id} started")

    def stop(self) -> None:
        """Stop the strategy."""
        self.state = StrategyState.STOPPED
        logger.info(f"Strategy {self.strategy_id} stopped")

    def pause(self) -> None:
        """Pause the strategy."""
        self.state = StrategyState.PAUSED
        logger.info(f"Strategy {self.strategy_id} paused")

    def resume(self) -> None:
        """Resume the strategy."""
        self.state = StrategyState.ACTIVE
        logger.info(f"Strategy {self.strategy_id} resumed")

    def get_history(self, symbol: str, periods: int | None = None) -> list[OHLCVBar]:
        """Get bar history for symbol.

        Args:
            symbol: Stock symbol.
            periods: Number of periods (None = all).

        Returns:
            List of historical bars.
        """
        history = self._bar_history.get(symbol, [])
        if periods:
            return history[-periods:]
        return history

    def get_closes(self, symbol: str, periods: int | None = None) -> np.ndarray:
        """Get closing prices as numpy array.

        Args:
            symbol: Stock symbol.
            periods: Number of periods.

        Returns:
            Array of closing prices.
        """
        history = self.get_history(symbol, periods)
        return np.array([float(bar.close) for bar in history])

    def get_returns(self, symbol: str, periods: int | None = None) -> np.ndarray:
        """Get returns as numpy array.

        Args:
            symbol: Stock symbol.
            periods: Number of periods.

        Returns:
            Array of returns.
        """
        closes = self.get_closes(symbol, periods)
        if len(closes) < 2:
            return np.array([])
        return np.diff(closes) / closes[:-1]

    def to_dict(self) -> dict[str, Any]:
        """Convert strategy state to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "strategy_type": self.config.strategy_type.value,
            "state": self.state.value,
            "symbols": self.config.symbols,
            "metrics": {
                "signals_generated": self.metrics.signals_generated,
                "signals_executed": self.metrics.signals_executed,
                "winning_trades": self.metrics.winning_trades,
                "losing_trades": self.metrics.losing_trades,
                "total_pnl": str(self.metrics.total_pnl),
                "win_rate": self.metrics.win_rate,
            },
        }


class MomentumStrategy(Strategy):
    """Momentum/trend-following strategy.

    Generates signals based on price momentum indicators:
    - Moving average crossovers
    - Relative strength
    - Breakout detection
    """

    def __init__(
        self,
        config: StrategyConfig,
        fast_period: int = 10,
        slow_period: int = 30,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
    ) -> None:
        """Initialize momentum strategy.

        Args:
            config: Strategy configuration.
            fast_period: Fast MA period.
            slow_period: Slow MA period.
            rsi_period: RSI calculation period.
            rsi_overbought: RSI overbought threshold.
            rsi_oversold: RSI oversold threshold.
        """
        super().__init__(config)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def generate_signals(
        self,
        bars: dict[str, OHLCVBar],
        features: dict[str, FeatureVector] | None = None,
        predictions: dict[str, ModelPrediction] | None = None,
        portfolio: Portfolio | None = None,
    ) -> list[TradeSignal]:
        """Generate momentum-based signals."""
        signals = []

        for symbol in self.config.symbols:
            if not self.can_generate_signal(symbol):
                continue

            closes = self.get_closes(symbol)
            if len(closes) < self.slow_period:
                continue

            # Calculate indicators
            fast_ma = np.mean(closes[-self.fast_period:])
            slow_ma = np.mean(closes[-self.slow_period:])
            rsi = self._calculate_rsi(closes)

            current_price = float(closes[-1])

            # Determine signal
            direction = Direction.FLAT
            strength = 0.0
            confidence = 0.0

            # MA crossover logic
            ma_diff_pct = (fast_ma - slow_ma) / slow_ma

            if fast_ma > slow_ma:
                # Bullish
                if rsi < self.rsi_oversold:
                    # Strong bullish (oversold + uptrend)
                    direction = Direction.LONG
                    strength = min(1.0, abs(ma_diff_pct) * 10 + (self.rsi_oversold - rsi) / 50)
                    confidence = 0.8
                elif rsi < self.rsi_overbought:
                    # Moderate bullish
                    direction = Direction.LONG
                    strength = min(0.8, abs(ma_diff_pct) * 10)
                    confidence = 0.6
            else:
                # Bearish
                if rsi > self.rsi_overbought:
                    # Strong bearish (overbought + downtrend)
                    direction = Direction.SHORT
                    strength = min(1.0, abs(ma_diff_pct) * 10 + (rsi - self.rsi_overbought) / 50)
                    confidence = 0.8
                elif rsi > self.rsi_oversold:
                    # Moderate bearish
                    direction = Direction.SHORT
                    strength = min(0.8, abs(ma_diff_pct) * 10)
                    confidence = 0.6

            if direction != Direction.FLAT and confidence >= self.config.confidence_threshold:
                signal = TradeSignal(
                    symbol=symbol,
                    direction=direction,
                    strength=strength if direction == Direction.LONG else -strength,
                    confidence=confidence,
                    horizon=self.config.lookback_periods,
                    model_source=f"MomentumStrategy:{self.strategy_id}",
                    features_snapshot={
                        "fast_ma": fast_ma,
                        "slow_ma": slow_ma,
                        "rsi": rsi,
                        "ma_diff_pct": ma_diff_pct,
                    },
                )
                signals.append(signal)
                self.record_signal(signal)

        return signals

    def _calculate_rsi(self, closes: np.ndarray) -> float:
        """Calculate RSI indicator."""
        if len(closes) < self.rsi_period + 1:
            return 50.0

        deltas = np.diff(closes[-self.rsi_period - 1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


class MeanReversionStrategy(Strategy):
    """Mean reversion strategy.

    Generates signals when prices deviate significantly from mean,
    betting on reversion to the mean.
    """

    def __init__(
        self,
        config: StrategyConfig,
        z_entry_threshold: float = 2.0,
        z_exit_threshold: float = 0.5,
        bollinger_period: int = 20,
        bollinger_std: float = 2.0,
    ) -> None:
        """Initialize mean reversion strategy.

        Args:
            config: Strategy configuration.
            z_entry_threshold: Z-score threshold for entry.
            z_exit_threshold: Z-score threshold for exit.
            bollinger_period: Bollinger band period.
            bollinger_std: Bollinger band standard deviations.
        """
        super().__init__(config)
        self.z_entry_threshold = z_entry_threshold
        self.z_exit_threshold = z_exit_threshold
        self.bollinger_period = bollinger_period
        self.bollinger_std = bollinger_std

    def generate_signals(
        self,
        bars: dict[str, OHLCVBar],
        features: dict[str, FeatureVector] | None = None,
        predictions: dict[str, ModelPrediction] | None = None,
        portfolio: Portfolio | None = None,
    ) -> list[TradeSignal]:
        """Generate mean reversion signals."""
        signals = []

        for symbol in self.config.symbols:
            if not self.can_generate_signal(symbol):
                continue

            closes = self.get_closes(symbol)
            if len(closes) < self.bollinger_period:
                continue

            # Calculate Bollinger Bands and Z-score
            recent = closes[-self.bollinger_period:]
            mean = np.mean(recent)
            std = np.std(recent)

            if std == 0:
                continue

            current_price = float(closes[-1])
            z_score = (current_price - mean) / std

            # Calculate bands
            upper_band = mean + self.bollinger_std * std
            lower_band = mean - self.bollinger_std * std

            direction = Direction.FLAT
            strength = 0.0
            confidence = 0.0

            # Mean reversion logic
            if z_score < -self.z_entry_threshold:
                # Price below lower band - buy expecting reversion up
                direction = Direction.LONG
                strength = min(1.0, abs(z_score) / 3)
                confidence = min(0.9, 0.5 + abs(z_score) / 10)
            elif z_score > self.z_entry_threshold:
                # Price above upper band - sell expecting reversion down
                direction = Direction.SHORT
                strength = min(1.0, abs(z_score) / 3)
                confidence = min(0.9, 0.5 + abs(z_score) / 10)

            if direction != Direction.FLAT and confidence >= self.config.confidence_threshold:
                signal = TradeSignal(
                    symbol=symbol,
                    direction=direction,
                    strength=strength if direction == Direction.LONG else -strength,
                    confidence=confidence,
                    horizon=self.config.lookback_periods,
                    model_source=f"MeanReversionStrategy:{self.strategy_id}",
                    features_snapshot={
                        "z_score": z_score,
                        "mean": mean,
                        "std": std,
                        "upper_band": upper_band,
                        "lower_band": lower_band,
                    },
                )
                signals.append(signal)
                self.record_signal(signal)

        return signals


class MLStrategy(Strategy):
    """Machine learning model-based strategy.

    Generates signals based on ML model predictions, with
    proper threshold and confidence filtering.
    """

    def __init__(
        self,
        config: StrategyConfig,
        model_names: list[str] | None = None,
        prediction_threshold: float = 0.1,
        ensemble_method: str = "mean",
    ) -> None:
        """Initialize ML strategy.

        Args:
            config: Strategy configuration.
            model_names: Models to use for predictions.
            prediction_threshold: Minimum prediction magnitude.
            ensemble_method: How to combine predictions (mean, vote, weighted).
        """
        super().__init__(config)
        self.model_names = model_names or []
        self.prediction_threshold = prediction_threshold
        self.ensemble_method = ensemble_method

    def generate_signals(
        self,
        bars: dict[str, OHLCVBar],
        features: dict[str, FeatureVector] | None = None,
        predictions: dict[str, ModelPrediction] | None = None,
        portfolio: Portfolio | None = None,
    ) -> list[TradeSignal]:
        """Generate signals from ML predictions."""
        signals = []

        if not predictions:
            return signals

        for symbol in self.config.symbols:
            if not self.can_generate_signal(symbol):
                continue

            # Get predictions for this symbol
            symbol_preds = [p for p in predictions.values() if p.symbol == symbol]

            # Filter by model names if specified
            if self.model_names:
                symbol_preds = [p for p in symbol_preds if p.model_name in self.model_names]

            if not symbol_preds:
                continue

            # Ensemble predictions
            if self.ensemble_method == "mean":
                avg_pred = np.mean([p.prediction for p in symbol_preds])
                avg_conf = np.mean([p.confidence for p in symbol_preds])
            elif self.ensemble_method == "vote":
                # Majority voting
                votes = [1 if p.prediction > 0 else -1 for p in symbol_preds]
                avg_pred = np.mean(votes)
                avg_conf = np.mean([p.confidence for p in symbol_preds])
            else:
                # Weighted by confidence
                weights = np.array([p.confidence for p in symbol_preds])
                weights = weights / weights.sum() if weights.sum() > 0 else weights
                avg_pred = np.sum([p.prediction * w for p, w in zip(symbol_preds, weights)])
                avg_conf = np.mean([p.confidence for p in symbol_preds])

            # Determine direction
            if abs(avg_pred) < self.prediction_threshold:
                continue

            direction = Direction.LONG if avg_pred > 0 else Direction.SHORT
            strength = min(1.0, abs(avg_pred))

            if avg_conf >= self.config.confidence_threshold:
                signal = TradeSignal(
                    symbol=symbol,
                    direction=direction,
                    strength=strength if direction == Direction.LONG else -strength,
                    confidence=avg_conf,
                    horizon=symbol_preds[0].horizon if symbol_preds else 1,
                    model_source=f"MLStrategy:{self.strategy_id}",
                    features_snapshot={
                        "avg_prediction": avg_pred,
                        "num_models": len(symbol_preds),
                        "ensemble_method": self.ensemble_method,
                        "model_predictions": {p.model_name: p.prediction for p in symbol_preds},
                    },
                )
                signals.append(signal)
                self.record_signal(signal)

        return signals


class CompositeStrategy(Strategy):
    """Composite strategy combining multiple sub-strategies.

    Aggregates signals from multiple strategies using configurable
    combination logic.
    """

    def __init__(
        self,
        config: StrategyConfig,
        strategies: list[Strategy],
        combination_method: str = "vote",
        min_agreement: float = 0.5,
    ) -> None:
        """Initialize composite strategy.

        Args:
            config: Strategy configuration.
            strategies: List of sub-strategies.
            combination_method: How to combine (vote, average, unanimous).
            min_agreement: Minimum agreement threshold for vote method.
        """
        super().__init__(config)
        self.strategies = strategies
        self.combination_method = combination_method
        self.min_agreement = min_agreement

    def generate_signals(
        self,
        bars: dict[str, OHLCVBar],
        features: dict[str, FeatureVector] | None = None,
        predictions: dict[str, ModelPrediction] | None = None,
        portfolio: Portfolio | None = None,
    ) -> list[TradeSignal]:
        """Generate combined signals from sub-strategies."""
        # Update all sub-strategies
        for strategy in self.strategies:
            strategy.update(bars)

        # Collect signals from all strategies
        all_signals: dict[str, list[TradeSignal]] = {}

        for strategy in self.strategies:
            strategy_signals = strategy.generate_signals(bars, features, predictions, portfolio)
            for signal in strategy_signals:
                if signal.symbol not in all_signals:
                    all_signals[signal.symbol] = []
                all_signals[signal.symbol].append(signal)

        # Combine signals per symbol
        combined_signals = []

        for symbol, symbol_signals in all_signals.items():
            if not self.can_generate_signal(symbol):
                continue

            if self.combination_method == "vote":
                combined = self._vote_combine(symbol_signals)
            elif self.combination_method == "average":
                combined = self._average_combine(symbol_signals)
            elif self.combination_method == "unanimous":
                combined = self._unanimous_combine(symbol_signals)
            else:
                combined = None

            if combined:
                combined_signals.append(combined)
                self.record_signal(combined)

        return combined_signals

    def _vote_combine(self, signals: list[TradeSignal]) -> TradeSignal | None:
        """Combine signals by voting."""
        if not signals:
            return None

        long_votes = sum(1 for s in signals if s.direction == Direction.LONG)
        short_votes = sum(1 for s in signals if s.direction == Direction.SHORT)
        total = len(signals)

        if long_votes / total >= self.min_agreement:
            direction = Direction.LONG
            relevant = [s for s in signals if s.direction == Direction.LONG]
        elif short_votes / total >= self.min_agreement:
            direction = Direction.SHORT
            relevant = [s for s in signals if s.direction == Direction.SHORT]
        else:
            return None

        avg_strength = np.mean([abs(s.strength) for s in relevant])
        avg_confidence = np.mean([s.confidence for s in relevant])

        return TradeSignal(
            symbol=signals[0].symbol,
            direction=direction,
            strength=avg_strength if direction == Direction.LONG else -avg_strength,
            confidence=avg_confidence,
            horizon=signals[0].horizon,
            model_source=f"CompositeStrategy:{self.strategy_id}",
            features_snapshot={
                "combination_method": "vote",
                "long_votes": long_votes,
                "short_votes": short_votes,
                "total_strategies": total,
            },
        )

    def _average_combine(self, signals: list[TradeSignal]) -> TradeSignal | None:
        """Combine signals by averaging."""
        if not signals:
            return None

        avg_strength = np.mean([s.strength for s in signals])
        avg_confidence = np.mean([s.confidence for s in signals])

        if abs(avg_strength) < self.config.signal_threshold:
            return None

        direction = Direction.LONG if avg_strength > 0 else Direction.SHORT

        return TradeSignal(
            symbol=signals[0].symbol,
            direction=direction,
            strength=avg_strength,
            confidence=avg_confidence,
            horizon=signals[0].horizon,
            model_source=f"CompositeStrategy:{self.strategy_id}",
            features_snapshot={
                "combination_method": "average",
                "num_signals": len(signals),
            },
        )

    def _unanimous_combine(self, signals: list[TradeSignal]) -> TradeSignal | None:
        """Combine signals requiring unanimous agreement."""
        if not signals:
            return None

        directions = set(s.direction for s in signals)
        if len(directions) != 1 or Direction.FLAT in directions:
            return None

        direction = signals[0].direction
        avg_strength = np.mean([abs(s.strength) for s in signals])
        avg_confidence = np.mean([s.confidence for s in signals])

        return TradeSignal(
            symbol=signals[0].symbol,
            direction=direction,
            strength=avg_strength if direction == Direction.LONG else -avg_strength,
            confidence=avg_confidence,
            horizon=signals[0].horizon,
            model_source=f"CompositeStrategy:{self.strategy_id}",
            features_snapshot={
                "combination_method": "unanimous",
                "num_signals": len(signals),
            },
        )

    def update(self, bars: dict[str, OHLCVBar]) -> None:
        """Update all sub-strategies."""
        super().update(bars)
        for strategy in self.strategies:
            strategy.update(bars)

    def start(self) -> None:
        """Start all sub-strategies."""
        super().start()
        for strategy in self.strategies:
            strategy.start()

    def stop(self) -> None:
        """Stop all sub-strategies."""
        super().stop()
        for strategy in self.strategies:
            strategy.stop()


# Factory functions

def create_strategy(
    strategy_type: StrategyType,
    config: StrategyConfig,
    **kwargs: Any,
) -> Strategy:
    """Create a strategy instance.

    Args:
        strategy_type: Type of strategy to create.
        config: Strategy configuration.
        **kwargs: Strategy-specific parameters.

    Returns:
        Strategy instance.
    """
    if strategy_type == StrategyType.MOMENTUM:
        return MomentumStrategy(config, **kwargs)
    elif strategy_type == StrategyType.MEAN_REVERSION:
        return MeanReversionStrategy(config, **kwargs)
    elif strategy_type == StrategyType.ML_BASED:
        return MLStrategy(config, **kwargs)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
