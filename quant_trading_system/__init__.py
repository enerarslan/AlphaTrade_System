"""
Quant Trading System - Institutional-Grade Algorithmic Trading Platform

A production-grade algorithmic trading system with ML/DL integration,
designed for processing 15-minute bar data for 45-50 equities.

This package uses lazy loading to minimize startup time. Heavy modules
like deep learning models and backtesting engines are only imported
when actually accessed.

Usage:
    # Core types are always available
    from quant_trading_system.core.data_types import Order, Position

    # ML models are lazy-loaded on first access
    from quant_trading_system.models.deep_learning import LSTMModel

    # Or preload specific module groups
    from quant_trading_system.core.lazy_loader import preload_modules
    preload_modules("core", "trading")
"""

__version__ = "1.0.0"
__author__ = "AlphaTrade"

# Lazy loading support - modules are imported on demand
from quant_trading_system.core.lazy_loader import (
    lazy_import,
    preload_modules,
    get_import_metrics,
    LazyModuleLoader,
)

# Create lazy loader for this package
_loader = LazyModuleLoader(__name__)

# Register lazy imports for heavy modules
_loader.register_lazy(".models.deep_learning", [
    "LSTMModel",
    "TransformerModel",
    "TCNModel",
    "DeviceManager",
    "get_device_manager",
])

_loader.register_lazy(".models.classical_ml", [
    "XGBoostModel",
    "LightGBMModel",
    "RandomForestModel",
    "ElasticNetModel",
])

_loader.register_lazy(".models.ensemble", [
    "VotingEnsemble",
    "StackingEnsemble",
    "AdaptiveEnsemble",
])

_loader.register_lazy(".backtest.engine", [
    "BacktestEngine",
    "BacktestConfig",
])

_loader.register_lazy(".backtest.analyzer", [
    "BacktestAnalyzer",
])

_loader.register_lazy(".alpha.regime_detection", [
    "CompositeRegimeDetector",
    "MarketRegime",
])

# Exported names (all are lazy-loaded)
__all__ = [
    # Package info
    "__version__",
    "__author__",
    # Lazy loading utilities
    "lazy_import",
    "preload_modules",
    "get_import_metrics",
    # Note: Individual model classes are imported directly from submodules
]


def __getattr__(name: str):
    """
    Module-level __getattr__ for lazy attribute access.

    This enables lazy loading of registered attributes when accessed
    as package attributes (e.g., quant_trading_system.LSTMModel).
    """
    try:
        return _loader.get_lazy_attr(name)._get_attr()
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
