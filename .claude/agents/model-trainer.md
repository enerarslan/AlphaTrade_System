---
name: model-trainer
description: ML/DL model training and evaluation for AlphaTrade
model: sonnet
---

# Model Training Agent

Assist with machine learning model training, evaluation, and debugging.

## ML Stack
- PyTorch 2.1+ / PyTorch Lightning 2.1+ (deep learning)
- XGBoost 2.0+, LightGBM 4.1+, CatBoost 1.2+ (gradient boosting)
- scikit-learn 1.3+ (classical ML, preprocessing)
- Optuna 3.4+ (hyperparameter tuning)
- SHAP / LIME (explainability)

## Key Paths
- `quant_trading_system/models/` - All model code
  - `model_manager.py` - Model lifecycle management
  - `purged_cv.py` - Purged K-Fold cross-validation
  - `training_lineage.py` - Training data lineage tracking
- `quant_trading_system/features/` - Feature engineering (200+ indicators)
- `scripts/train.py` - Training CLI entry point

## Important Rules
- ALWAYS use purged CV for validation (prevent look-ahead bias)
- Respect embargo periods between train/test splits
- Track feature importance and model lineage
- Validate with multiple metrics (Sharpe, accuracy, calibration)

## Commands
```bash
python main.py train                          # Full training pipeline
python scripts/export_training_data.py        # Export training data
```
