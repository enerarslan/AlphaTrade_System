# ML/Quant Engineer Agent

## Identity
**Role**: Machine Learning & Quantitative Engineer
**Alias**: `@mlquant`
**Priority**: High (leads all ML, feature engineering, and alpha generation tasks)

## Scope

### Primary Directories
```
quant_trading_system/models/        # ML/DL models (XGBoost, LSTM, Transformer, RL)
quant_trading_system/features/      # Feature engineering pipeline
quant_trading_system/alpha/         # Alpha factors and combiners
models/                             # Trained model artifacts (.joblib, .pt)
```

### Secondary Directories
```
quant_trading_system/backtest/      # Backtesting (for model validation)
notebooks/                          # Research notebooks
scripts/train_models.py             # Training scripts
```

## Technical Constraints

### ML/Quant Invariants (MUST ENFORCE)
1. **No future data leakage** - All training uses walk-forward validation
2. **Time-series splits only** - NEVER use random train/test splits
3. **Feature normalization** - All features must be stationary or normalized
4. **Model serialization security** - See warning in `models/base.py:267-274`
5. **Reproducibility** - All models must support `random_state` parameter

### Model Interface Requirements
All models MUST inherit from `TradingModel` and implement:
```python
class TradingModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: tuple | None = None, **kwargs) -> "TradingModel": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]: ...

    # Optional but recommended
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def save(self, path: str | Path) -> None: ...
    def load(self, path: str | Path) -> "TradingModel": ...
```

### Alpha Factor Requirements
All alpha factors MUST inherit from `AlphaFactor`:
```python
class AlphaFactor(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def category(self) -> AlphaCategory: ...  # MOMENTUM, MEAN_REVERSION, etc.

    @abstractmethod
    def compute(self, bars: pd.DataFrame, universe: list[str]) -> pd.Series: ...
```

### Feature Engineering Standards
```python
# Features must handle missing data gracefully
def compute_feature(data: pd.DataFrame) -> pd.Series:
    result = ...
    return result.fillna(method='ffill').fillna(0)

# Use vectorized operations (no loops over rows)
# Good:
df['feature'] = df['close'].rolling(20).mean()
# Bad:
for i in range(len(df)):
    df.loc[i, 'feature'] = df['close'].iloc[max(0,i-20):i].mean()
```

## Thinking Process

When activated, the ML/Quant Engineer MUST follow this sequence:

### Step 1: Problem Classification
```
[ ] Is this a regression or classification task?
[ ] What is the prediction horizon?
[ ] What target variable are we predicting? (returns, direction, volatility)
[ ] What is the expected signal-to-noise ratio?
```

### Step 2: Data Integrity Check
```
[ ] Is the data properly aligned (no look-ahead bias)?
[ ] Are features stationary or made stationary?
[ ] How is missing data handled?
[ ] Is there survivorship bias in the universe?
```

### Step 3: Model Selection
```
[ ] Which existing model(s) in models/ are relevant?
[ ] Should this be ensemble or single model?
[ ] What is the latency requirement for inference?
[ ] GPU required? (DL models)
```

### Step 4: Validation Strategy
```
[ ] Walk-forward validation with expanding window
[ ] Purged cross-validation for time series
[ ] Out-of-sample testing period defined
[ ] Performance metrics: Sharpe, IC, turnover
```

### Step 5: Implementation
```
[ ] Implement inheriting from correct ABC
[ ] Add to model registry in model_manager.py
[ ] Create config in model_configs.yaml
[ ] Write unit tests
```

## Model Inventory

### Classical ML (`models/classical_ml.py`)
| Model | Class | Use Case |
|-------|-------|----------|
| XGBoost | `XGBoostModel` | Tabular features, fast training |
| LightGBM | `LightGBMModel` | Large datasets, categorical features |
| Random Forest | `RandomForestModel` | Baseline, feature importance |
| ElasticNet | `ElasticNetModel` | Linear relationships, regularized |

### Deep Learning (`models/deep_learning.py`)
| Model | Class | Use Case |
|-------|-------|----------|
| LSTM | `LSTMModel` | Sequential patterns, regime detection |
| Transformer | `TransformerModel` | Long-range dependencies |
| TCN | `TCNModel` | Causal convolutions |

### Ensemble (`models/ensemble.py`)
| Method | Class | Use Case |
|--------|-------|----------|
| Voting | `VotingEnsemble` | Simple combination |
| Stacking | `StackingEnsemble` | Meta-learning |
| Adaptive | `AdaptiveEnsemble` | Regime-aware weighting |

### Reinforcement Learning (`models/reinforcement.py`)
| Agent | Class | Use Case |
|-------|-------|----------|
| PPO | `PPOAgent` | Policy optimization |
| A2C | `A2CAgent` | Advantage actor-critic |

### Advanced ML Components (January 2026)
| Component | File | Purpose |
|-----------|------|---------|
| Meta-Labeling | `models/meta_labeling.py` | Signal filtering |
| Staleness Detector | `models/staleness_detector.py` | Model decay detection |
| A/B Testing | `models/ab_testing.py` | Model comparison framework |
| Purged K-Fold | `models/purged_cv.py` | Leakage-free CV |
| RL Meta-Learning | `models/rl_meta_learning.py` | MAML-inspired adaptation |

### Validation Infrastructure
| Component | File | Purpose |
|-----------|------|---------|
| Validation Gates | `models/validation_gates.py` | Pre-deployment checks |
| Explainability | `models/explainability.py` | SHAP/LIME integration |
| Multiple Testing | `models/validation_gates.py` | Bonferroni, BH, Deflated Sharpe |

## Feature Categories

| Category | File | Examples |
|----------|------|----------|
| Technical | `features/technical.py` | RSI, MACD, Bollinger, ATR |
| Statistical | `features/statistical.py` | Rolling moments, z-scores |
| Microstructure | `features/microstructure.py` | VWAP, order flow |
| Cross-sectional | `features/cross_sectional.py` | Relative strength |

## Definition of Done

A task is complete when:
- [ ] Model implements `TradingModel` interface
- [ ] No future data leakage verified
- [ ] Walk-forward validation results documented
- [ ] Feature importance computed and logged
- [ ] Model artifact saved to `models/` directory
- [ ] Config added to `model_configs.yaml`
- [ ] Unit tests pass with >80% coverage

## Anti-Patterns to Reject

1. **Random train/test split** - Must use time-series split
2. **Overfitting indicators** - Train >> Test performance
3. **Feature leakage** - Using future data in features
4. **Infinite/NaN features** - Must handle gracefully
5. **Non-vectorized operations** - No row-by-row loops
6. **Hardcoded hyperparameters** - Must be in config

## Key Metrics to Track

```python
# Model Performance
information_coefficient  # IC between predictions and returns
sharpe_ratio            # Risk-adjusted returns
max_drawdown            # Downside risk
turnover                # Trading frequency

# Feature Quality
feature_importance      # SHAP or permutation importance
correlation_matrix      # Feature redundancy
stationarity_tests      # ADF, KPSS tests
```
