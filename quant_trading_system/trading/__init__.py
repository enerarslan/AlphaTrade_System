"""
Trading module.

Orchestrates strategy execution, signal generation, portfolio management,
and the main trading loop.
"""

from quant_trading_system.trading.strategy import (
    CompositeStrategy,
    MeanReversionStrategy,
    MLStrategy,
    MomentumStrategy,
    Strategy,
    StrategyConfig,
    StrategyMetrics,
    StrategyState,
    StrategyType,
    create_strategy,
)
from quant_trading_system.trading.signal_generator import (
    ConflictResolution,
    EnrichedSignal,
    SignalAggregator,
    SignalFilter,
    SignalGenerator,
    SignalGeneratorConfig,
    SignalMetadata,
    SignalPriority,
    SignalQueue,
)
from quant_trading_system.trading.portfolio_manager import (
    PortfolioManager,
    PortfolioOptimizer,
    PositionSizer,
    PositionSizerConfig,
    PositionSizingMethod,
    RebalanceConfig,
    RebalanceMethod,
    TargetPortfolio,
    TargetPosition,
    Trade,
)
from quant_trading_system.trading.trading_engine import (
    EngineMetrics,
    EngineState,
    TradingEngine,
    TradingEngineConfig,
    TradingMode,
    TradingSession,
    create_trading_engine,
)

__all__ = [
    # strategy
    "CompositeStrategy",
    "MeanReversionStrategy",
    "MLStrategy",
    "MomentumStrategy",
    "Strategy",
    "StrategyConfig",
    "StrategyMetrics",
    "StrategyState",
    "StrategyType",
    "create_strategy",
    # signal_generator
    "ConflictResolution",
    "EnrichedSignal",
    "SignalAggregator",
    "SignalFilter",
    "SignalGenerator",
    "SignalGeneratorConfig",
    "SignalMetadata",
    "SignalPriority",
    "SignalQueue",
    # portfolio_manager
    "PortfolioManager",
    "PortfolioOptimizer",
    "PositionSizer",
    "PositionSizerConfig",
    "PositionSizingMethod",
    "RebalanceConfig",
    "RebalanceMethod",
    "TargetPortfolio",
    "TargetPosition",
    "Trade",
    # trading_engine
    "EngineMetrics",
    "EngineState",
    "TradingEngine",
    "TradingEngineConfig",
    "TradingMode",
    "TradingSession",
    "create_trading_engine",
]
