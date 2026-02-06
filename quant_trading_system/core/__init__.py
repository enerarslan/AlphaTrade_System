"""
Core infrastructure layer for the trading system.

Contains type definitions, exceptions, event system, component registry,
and shared utilities used across all modules.
"""

from .data_types import (
    OHLCVBar,
    TradeSignal,
    Order,
    Position,
    Portfolio,
    RiskMetrics,
    Direction,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
)
from .exceptions import (
    TradingSystemError,
    DataError,
    DataNotFoundError,
    DataValidationError,
    DataCorruptionError,
    DataConnectionError,
    ModelError,
    ModelNotFoundError,
    ModelLoadError,
    PredictionError,
    TrainingError,
    ExecutionError,
    OrderSubmissionError,
    OrderCancellationError,
    InsufficientFundsError,
    BrokerConnectionError,
    RiskError,
    PositionLimitError,
    DrawdownLimitError,
    ExposureLimitError,
    MarginCallError,
    ConfigurationError,
    InvalidConfigError,
    MissingConfigError,
    ConfigParseError,
)
from .events import EventBus, Event, EventType
from .registry import ComponentRegistry, registry
from .reproducibility import child_seed, set_global_seed
from .system_integrator import (
    IntegratorState,
    SystemIntegrator,
    SystemIntegratorConfig,
    create_system_integrator,
    get_system_integrator,
)

__all__ = [
    # Data types
    "OHLCVBar",
    "TradeSignal",
    "Order",
    "Position",
    "Portfolio",
    "RiskMetrics",
    "Direction",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    # Exceptions
    "TradingSystemError",
    "DataError",
    "DataNotFoundError",
    "DataValidationError",
    "DataCorruptionError",
    "DataConnectionError",
    "ModelError",
    "ModelNotFoundError",
    "ModelLoadError",
    "PredictionError",
    "TrainingError",
    "ExecutionError",
    "OrderSubmissionError",
    "OrderCancellationError",
    "InsufficientFundsError",
    "BrokerConnectionError",
    "RiskError",
    "PositionLimitError",
    "DrawdownLimitError",
    "ExposureLimitError",
    "MarginCallError",
    "ConfigurationError",
    "InvalidConfigError",
    "MissingConfigError",
    "ConfigParseError",
    # Events
    "EventBus",
    "Event",
    "EventType",
    # Registry
    "ComponentRegistry",
    "registry",
    # Reproducibility
    "set_global_seed",
    "child_seed",
    # System Integrator
    "IntegratorState",
    "SystemIntegrator",
    "SystemIntegratorConfig",
    "create_system_integrator",
    "get_system_integrator",
]
