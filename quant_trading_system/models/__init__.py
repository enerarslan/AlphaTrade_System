"""
Machine learning and deep learning models module.

Contains classical ML models (XGBoost, LightGBM, CatBoost, RF),
deep learning models (LSTM, Transformer, TCN), RL agents, and ensemble methods.
"""

from quant_trading_system.models.base import (
    EnsembleModel,
    ModelType,
    TimeSeriesModel,
    TradingModel,
)
from quant_trading_system.models.classical_ml import (
    CatBoostModel,
    ElasticNetModel,
    LightGBMModel,
    RandomForestModel,
    XGBoostModel,
    create_sample_weights,
)
from quant_trading_system.models.deep_learning import (
    LSTMModel,
    TCNModel,
    TransformerModel,
)
from quant_trading_system.models.ensemble import (
    AdaptiveEnsemble,
    AggregationMethod,
    AveragingEnsemble,
    ICBasedEnsemble,
    RegimeAwareEnsemble,
    StackingEnsemble,
    VotingEnsemble,
    create_model_ensemble,
)
from quant_trading_system.models.model_manager import (
    HyperparameterOptimizer,
    ModelManager,
    ModelRegistry,
    SplitMethod,
    TimeSeriesSplitter,
)
from quant_trading_system.models.reinforcement import (
    A2CAgent,
    PPOAgent,
    TradingEnvironment,
)
from quant_trading_system.models.purged_cv import (
    CombinatorialPurgedKFold,
    CVFoldResult,
    CVSummary,
    EventPurgedKFold,
    PurgedKFold,
    WalkForwardCV,
    create_purged_cv,
    validate_no_leakage,
)
from quant_trading_system.models.rl_meta_learning import (
    EpisodeStats,
    Experience,
    HierarchicalOption,
    HierarchicalRLController,
    IntrinsicCuriosityModule,
    MarketRegimeRL,
    MetaLearningAgent,
    RegimeAdaptiveRewardShaper,
    RewardType,
    RLMetaConfig,
    create_meta_learning_agent,
)

__all__ = [
    # Base classes
    "TradingModel",
    "TimeSeriesModel",
    "EnsembleModel",
    "ModelType",
    # Classical ML
    "XGBoostModel",
    "LightGBMModel",
    "CatBoostModel",
    "RandomForestModel",
    "ElasticNetModel",
    "create_sample_weights",
    # Deep Learning
    "LSTMModel",
    "TransformerModel",
    "TCNModel",
    # Reinforcement Learning
    "PPOAgent",
    "A2CAgent",
    "TradingEnvironment",
    # Ensembles
    "VotingEnsemble",
    "AveragingEnsemble",
    "StackingEnsemble",
    "AdaptiveEnsemble",
    "RegimeAwareEnsemble",
    "ICBasedEnsemble",  # P2-C Enhancement
    "AggregationMethod",
    "create_model_ensemble",
    # Model Management
    "ModelManager",
    "ModelRegistry",
    "HyperparameterOptimizer",
    "TimeSeriesSplitter",
    "SplitMethod",
    # Purged Cross-Validation (P2-B Enhancement)
    "PurgedKFold",
    "CombinatorialPurgedKFold",
    "WalkForwardCV",
    "EventPurgedKFold",
    "CVFoldResult",
    "CVSummary",
    "create_purged_cv",
    "validate_no_leakage",
    # RL Meta-Learning (P2-D Enhancement)
    "RewardType",
    "MarketRegimeRL",
    "Experience",
    "EpisodeStats",
    "RLMetaConfig",
    "RegimeAdaptiveRewardShaper",
    "IntrinsicCuriosityModule",
    "HierarchicalOption",
    "HierarchicalRLController",
    "MetaLearningAgent",
    "create_meta_learning_agent",
]
