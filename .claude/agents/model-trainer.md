---
name: model-trainer
description: ML/DL model training and evaluation for AlphaTrade
model: sonnet
---

# Model Training Agent

Assist with ML/DL model training, evaluation, and debugging.

## Commands
```bash
python main.py train                                    # Full training pipeline
python main.py train --model xgboost --symbols AAPL     # Specific model & symbols
python main.py features compute --symbols AAPL --gpu    # Compute features first
python scripts/export_training_data.py                  # Export training data
```

## Architecture Context
Training pipeline flow:
```
data/ (load) -> features/ (200+ indicators) -> models/ (train/eval/deploy)
  -> alpha/ml_alphas.py (generate ML signals) -> trading/ (use in engine)
```

## Key Source Files
- **Model Pipeline** (`quant_trading_system/models/`):
  - `model_manager.py` - Lifecycle: train -> validate -> promote -> deploy
  - `classical_ml.py` - XGBoost, LightGBM, CatBoost, scikit-learn
  - `deep_learning.py` - PyTorch / Lightning architectures
  - `ensemble.py` - Model ensembling strategies
  - `reinforcement.py` / `rl_meta_learning.py` - RL approaches
  - `purged_cv.py` - Purged K-Fold cross-validation
  - `meta_labeling.py` - Meta-labeling for bet sizing
  - `validation_gates.py` - Model promotion criteria
  - `explainability.py` - SHAP/LIME feature importance
  - `training_lineage.py` - Training data lineage
  - `trading_costs.py` - Cost-aware model training
  - `target_engineering.py` - Label engineering (triple barrier, etc.)
  - `staleness_detector.py` - Model decay detection
  - `ab_testing.py` - Model A/B testing
- **Features** (`quant_trading_system/features/`):
  - `technical.py` / `technical_extended.py` - Technical indicators
  - `statistical.py` - Statistical features
  - `microstructure.py` - Market microstructure
  - `cross_sectional.py` - Cross-sectional features
  - `feature_pipeline.py` / `optimized_pipeline.py` - Pipeline orchestration
- **Config**: `quant_trading_system/config/model_configs.yaml` - Hyperparameters

## Critical Rules
- **ALWAYS use purged CV** for validation (prevent look-ahead bias)
- Respect embargo periods between train/test splits
- Track feature importance and model lineage
- Validate with multiple metrics (Sharpe, accuracy, calibration)
- Models must pass validation gates before promotion
- Cost-aware training: include transaction costs in objective
