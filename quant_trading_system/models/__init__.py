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
    "AggregationMethod",
    "create_model_ensemble",
    # Model Management
    "ModelManager",
    "ModelRegistry",
    "HyperparameterOptimizer",
    "TimeSeriesSplitter",
    "SplitMethod",
]
