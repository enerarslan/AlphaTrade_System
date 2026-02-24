"""
Data engineering pipeline module.

Handles historical data loading, preprocessing, feature storage,
and real-time data streaming.
"""

from quant_trading_system.data.feature_store import (
    FeatureDefinition,
    FeatureRegistry,
    FeatureStore,
    MemoryCache,
    StreamingFeatureUpdater,
)
from quant_trading_system.data.live_feed import (
    AlpacaLiveFeed,
    AlpacaWebSocketFeed,
    BarAggregator,
    BaseLiveFeed,
    CircuitBreaker,
    ConnectionState,
    MockLiveFeed,
    StreamType,
    WebSocketConfig,
    create_live_feed,
    start_live_feed,
)
from quant_trading_system.data.loader import (
    DataLoader,
    DataValidationRules,
    ParquetDataStore,
    create_sample_data,
)
from quant_trading_system.data.preprocessor import (
    DataAligner,
    DataPreprocessor,
    NormalizationMethod,
    preprocess_market_data,
)
from quant_trading_system.data.vix_feed import (
    AlpacaVIXFeed,
    BaseVIXFeed,
    MockVIXFeed,
    VIXData,
    VIXFeedConfig,
    VIXRegime,
    VIXRiskAdjustor,
    VIXThresholds,
    create_vix_feed,
    start_vix_feed,
)
from quant_trading_system.data.alternative_data import (
    AltDataAggregator,
    AltDataConfig,
    AltDataFeatureGenerator,
    AltDataPoint,
    AltDataProvider,
    AltDataType,
    CreditCardData,
    CreditCardProvider,
    SatelliteData,
    SatelliteProvider,
    SentimentData,
    SentimentProvider,
    SentimentScore,
    SupplyChainData,
    SupplyChainProvider,
    WebTrafficData,
    WebTrafficProvider,
    create_alt_data_aggregator,
    start_alt_data_aggregator,
)

__all__ = [
    # Loader
    "DataLoader",
    "DataValidationRules",
    "ParquetDataStore",
    "create_sample_data",
    # Preprocessor
    "DataPreprocessor",
    "DataAligner",
    "NormalizationMethod",
    "preprocess_market_data",
    # Feature Store
    "FeatureStore",
    "FeatureDefinition",
    "FeatureRegistry",
    "MemoryCache",
    "StreamingFeatureUpdater",
    # Live Feed
    "BaseLiveFeed",
    "AlpacaLiveFeed",
    "AlpacaWebSocketFeed",
    "MockLiveFeed",
    "BarAggregator",
    "ConnectionState",
    "StreamType",
    "WebSocketConfig",
    "CircuitBreaker",
    "create_live_feed",
    "start_live_feed",
    # VIX Feed (P1-A Enhancement)
    "BaseVIXFeed",
    "AlpacaVIXFeed",
    "MockVIXFeed",
    "VIXData",
    "VIXFeedConfig",
    "VIXRegime",
    "VIXThresholds",
    "VIXRiskAdjustor",
    "create_vix_feed",
    "start_vix_feed",
    # Alternative Data (P2-A Enhancement)
    "AltDataType",
    "SentimentScore",
    "AltDataPoint",
    "SentimentData",
    "WebTrafficData",
    "SatelliteData",
    "CreditCardData",
    "SupplyChainData",
    "AltDataConfig",
    "AltDataProvider",
    "SentimentProvider",
    "WebTrafficProvider",
    "SatelliteProvider",
    "CreditCardProvider",
    "SupplyChainProvider",
    "AltDataAggregator",
    "AltDataFeatureGenerator",
    "create_alt_data_aggregator",
    "start_alt_data_aggregator",
]
