"""
Execution module.

Contains Alpaca API client, order management, execution algorithms
(TWAP, VWAP), and position tracking.
"""

from quant_trading_system.execution.alpaca_client import (
    AccountInfo,
    AlpacaClient,
    AlpacaOrder,
    AlpacaPosition,
    OrderClass,
    RateLimiter,
    TradingEnvironment,
    create_bracket_order,
    create_limit_order,
    create_market_order,
    create_stop_order,
)
from quant_trading_system.execution.order_manager import (
    ManagedOrder,
    OrderManager,
    OrderPriority,
    OrderRequest,
    OrderState,
    OrderValidationResult,
    OrderValidator,
    SmartOrderRouter,
)
from quant_trading_system.execution.execution_algo import (
    AlgoExecutionEngine,
    AlgoExecutionState,
    AlgoStatus,
    AlgoType,
    ExecutionAlgorithm,
    ImplementationShortfallAlgorithm,
    SliceOrder,
    TWAPAlgorithm,
    VWAPAlgorithm,
)
from quant_trading_system.execution.position_tracker import (
    CashState,
    PositionState,
    PositionTracker,
    PnLRecord,
    ReconciliationResult,
    SettlementStatus,
    TradeRecord,
)
from quant_trading_system.execution.tca import (
    BenchmarkType,
    CostBreakdown,
    CostComponent,
    ExecutionMetrics,
    ExecutionQuality,
    PostTradeAnalyzer,
    PreTradeCostEstimator,
    TCAConfig,
    TCAManager,
    TCAReport,
    create_tca_manager,
)
from quant_trading_system.execution.market_impact import (
    AdaptiveMarketImpactModel,
    AlmgrenChrissModel,
    ExecutionRecord,
    ImpactEstimate,
    ImpactModelType,
    MarketCondition,
    MarketConditionClassifier,
    MarketImpactConfig,
    TimeOfDayAdjuster,
    create_market_impact_model,
)

__all__ = [
    # alpaca_client
    "AccountInfo",
    "AlpacaClient",
    "AlpacaOrder",
    "AlpacaPosition",
    "OrderClass",
    "RateLimiter",
    "TradingEnvironment",
    "create_bracket_order",
    "create_limit_order",
    "create_market_order",
    "create_stop_order",
    # order_manager
    "ManagedOrder",
    "OrderManager",
    "OrderPriority",
    "OrderRequest",
    "OrderState",
    "OrderValidationResult",
    "OrderValidator",
    "SmartOrderRouter",
    # execution_algo
    "AlgoExecutionEngine",
    "AlgoExecutionState",
    "AlgoStatus",
    "AlgoType",
    "ExecutionAlgorithm",
    "ImplementationShortfallAlgorithm",
    "SliceOrder",
    "TWAPAlgorithm",
    "VWAPAlgorithm",
    # position_tracker
    "CashState",
    "PositionState",
    "PositionTracker",
    "PnLRecord",
    "ReconciliationResult",
    "SettlementStatus",
    "TradeRecord",
    # TCA (P1-D Enhancement)
    "BenchmarkType",
    "CostBreakdown",
    "CostComponent",
    "ExecutionMetrics",
    "ExecutionQuality",
    "PostTradeAnalyzer",
    "PreTradeCostEstimator",
    "TCAConfig",
    "TCAManager",
    "TCAReport",
    "create_tca_manager",
    # Market Impact (P3-B Enhancement)
    "ImpactModelType",
    "MarketCondition",
    "ImpactEstimate",
    "ExecutionRecord",
    "MarketImpactConfig",
    "MarketConditionClassifier",
    "AlmgrenChrissModel",
    "TimeOfDayAdjuster",
    "AdaptiveMarketImpactModel",
    "create_market_impact_model",
]
