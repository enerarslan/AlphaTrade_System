# AlphaTrade System - Architecture Documentation

## 1. System Overview

**AlphaTrade** is an institutional-grade quantitative trading platform for automated US equity trading. Python 3.11+, complete pipeline from data ingestion to live execution.

### Core Capabilities
- Real-time and historical market data (15-min bars via Alpaca)
- Multi-model ML pipeline (XGBoost, LightGBM, LSTM, Transformer, PPO)
- Alpha factor generation with IC/IR tracking
- Market regime detection for adaptive strategies
- Risk-aware position sizing and portfolio optimization
- Event-driven backtesting with realistic market simulation
- Paper and live trading with kill switch safety
- Full regulatory compliance (audit logging, data lineage)

### Technology Stack
| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| ML | PyTorch 2.1+, scikit-learn, XGBoost, LightGBM |
| Data | Pandas, Polars, NumPy |
| Database | PostgreSQL + TimescaleDB |
| Cache | Redis 7+ |
| Broker | Alpaca Markets |
| Monitoring | Prometheus, Grafana |
| Config | Pydantic + YAML |

---

## 2. Directory Structure

```
AlphaTrade_System/
├── main.py                         # Entry point (CLI)
├── pyproject.toml                  # Dependencies
├── .env                            # Environment variables (NOT in git)
│
├── quant_trading_system/           # Main package
│   ├── config/                     # Configuration (settings.py, *.yaml)
│   ├── core/                       # Foundation (data_types.py, events.py, exceptions.py)
│   ├── data/                       # Data ingestion (loader.py, preprocessor.py, live_feed.py)
│   ├── alpha/                      # Alpha factors (alpha_base.py, *_alphas.py, alpha_combiner.py)
│   ├── features/                   # Feature engineering (technical.py, statistical.py)
│   ├── models/                     # ML/DL models (base.py, classical_ml.py, deep_learning.py)
│   ├── execution/                  # Order execution (order_manager.py, alpaca_client.py)
│   ├── trading/                    # Trading orchestration (trading_engine.py, strategy.py)
│   ├── risk/                       # Risk management (limits.py, position_sizer.py)
│   ├── backtest/                   # Backtesting (engine.py, simulator.py, analyzer.py)
│   ├── database/                   # Database layer (connection.py, models.py, repository.py)
│   └── monitoring/                 # Observability (metrics.py, alerting.py, audit.py)
│
├── scripts/                        # Operational scripts
├── tests/                          # Test suite (700+ tests)
├── docker/                         # Docker infrastructure
├── data/raw/                       # Historical data (47 symbols)
├── models/                         # Trained model artifacts
└── .claude/                        # Claude agent configuration
```

---

## 3. Core Data Types

Located in `quant_trading_system/core/data_types.py`. All models use Pydantic with validation.

### Key Enums
- `Direction`: LONG, SHORT, FLAT
- `OrderSide`: BUY, SELL
- `OrderType`: MARKET, LIMIT, STOP, STOP_LIMIT, TRAILING_STOP
- `OrderStatus`: PENDING, SUBMITTED, ACCEPTED, FILLED, PARTIAL_FILLED, CANCELLED, REJECTED, EXPIRED
- `TimeInForce`: DAY, GTC, IOC, FOK, OPG, CLS

### Core Models
- **OHLCVBar**: OHLCV bar data with validation (high >= low, etc.)
- **TradeSignal**: Trading signal with direction, strength [-1,1], confidence [0,1], horizon
- **Order**: Full order lifecycle (order_id, broker_order_id, symbol, side, quantity, status, fills)
- **Position**: Current position (quantity, avg_entry_price, unrealized_pnl, realized_pnl)
- **Portfolio**: Complete state (equity, cash, buying_power, positions, exposures)
- **RiskMetrics**: VaR, Sharpe, Sortino, drawdown, volatility, sector exposures
- **FeatureVector**: Feature vector for model input
- **ModelPrediction**: Model output with prediction, probability, direction, confidence

---

## 4. Event System

Located in `quant_trading_system/core/events.py`. Thread-safe, priority-based EventBus singleton.

### Event Types
- **Market Data**: BAR_UPDATE, QUOTE_UPDATE, TRADE_UPDATE
- **Signals**: SIGNAL_GENERATED, SIGNAL_EXPIRED, SIGNAL_CANCELLED
- **Orders**: ORDER_SUBMITTED, ORDER_ACCEPTED, ORDER_FILLED, ORDER_PARTIAL, ORDER_CANCELLED, ORDER_REJECTED
- **Risk**: LIMIT_BREACH, DRAWDOWN_WARNING, KILL_SWITCH_TRIGGERED, KILL_SWITCH_RESET
- **Portfolio**: POSITION_OPENED, POSITION_CLOSED, POSITION_UPDATED
- **System**: SYSTEM_START, SYSTEM_STOP, SYSTEM_ERROR, MODEL_RELOAD

### EventPriority
- CRITICAL (0): Kill switch, system failures
- HIGH (1): Order events, risk alerts
- NORMAL (2): Data updates, signals
- LOW (3): Logging, metrics

### EventBus Methods
- `subscribe(event_type, handler, handler_name, priority)`
- `publish(event)` / `publish_async(event)`
- `get_event_history(event_type, limit)`

---

## 5. Configuration

Located in `quant_trading_system/config/settings.py`. Pydantic BaseSettings with env support.

### Settings Classes
- **DatabaseSettings**: host, port, name, user, password, pool_size
- **RedisSettings**: host, port, db, password
- **AlpacaSettings**: api_key, api_secret, base_url, paper_trading
- **RiskSettings**: max_position_pct (0.10), max_positions (20), max_daily_loss_pct (0.05), max_drawdown_pct (0.15)
- **TradingSettings**: bar_timeframe, signal_threshold, min_confidence, slippage_bps
- **ModelSettings**: models_dir, default_model, ensemble_models, retrain_frequency
- **AlertSettings**: slack_webhook, email config, pagerduty_key

### Environment Variables
```bash
ALPACA_API_KEY=your_api_key
ALPACA_API_SECRET=your_api_secret
DATABASE__HOST=localhost
REDIS__HOST=localhost
```

---

## 6. Models & Machine Learning

Located in `quant_trading_system/models/`.

### TradingModel ABC (`base.py`)
Abstract base with: `name`, `version`, `model_type`, `is_fitted`, `feature_names`, `fit()`, `predict()`, `get_feature_importance()`, `save()`, `load()`

### Model Types
| Type | Classes | Use Case |
|------|---------|----------|
| Classical ML | XGBoostModel, LightGBMModel, RandomForestModel, ElasticNetModel | Tabular data |
| Deep Learning | LSTMModel, TransformerModel, GRUModel, TCNModel | Sequential patterns |
| Ensemble | VotingEnsemble, StackingEnsemble, AdaptiveEnsemble | Model combination |
| RL | PPOAgent | Reinforcement learning |

### Model Validation Gates (`validation_gates.py`)
Pre-deployment validation: min_sharpe_ratio, max_drawdown, max_is_oos_ratio, min_trades, min_win_rate, min_ic

### Explainability (`explainability.py`)
SHAP/LIME integration for regulatory compliance. `SHAPExplainer`, `ModelExplainabilityService`

---

## 7. Alpha Factors

Located in `quant_trading_system/alpha/`.

### AlphaFactor ABC (`alpha_base.py`)
- Properties: `name`, `alpha_type`, `horizon`
- Methods: `compute(df, features) -> np.ndarray`, `get_params()`
- AlphaSignal: value [-1,1], confidence [0,1], horizon, direction

### Alpha Types
- **Momentum**: PriceMomentumAlpha, RSIMomentumAlpha, MACDMomentumAlpha, RelativeStrengthAlpha
- **Mean Reversion**: BollingerReversionAlpha, ZScoreReversionAlpha
- **ML-Based**: MLAlpha (wraps TradingModel)

### Alpha Combiner (`alpha_combiner.py`)
Combination methods: EQUAL, IC_WEIGHTED, SHARPE_WEIGHTED, INVERSE_VOLATILITY, OPTIMIZED
Neutralization: MARKET, SECTOR, FACTOR, BETA
Orthogonalization: PCA, GRAM_SCHMIDT, DECORRELATE

### Alpha Metrics (`alpha_metrics.py`)
`compute_ic()`, `compute_ir()`, `compute_ic_decay()`, `compute_turnover()`

### Regime Detection (`regime_detection.py`)
MarketRegime: BULL_LOW_VOL, BULL_HIGH_VOL, BEAR_LOW_VOL, BEAR_HIGH_VOL, RANGE_BOUND, CRISIS
CompositeRegimeDetector, RegimeAdaptiveController

---

## 8. Feature Engineering

Located in `quant_trading_system/features/`.

### Technical Features (`technical.py`)
- **Trend**: SMA, EMA, DEMA, TEMA, Keltner, Ichimoku
- **Momentum**: RSI, MACD, Stochastic, CCI, Williams %R
- **Volatility**: Bollinger Bands, ATR, NATR, Parabolic SAR
- **Volume**: OBV, MFI, CMF, AD Line, VWAP
- **Patterns**: 40+ candlestick patterns

### Statistical Features (`statistical.py`)
Rolling stats, autocorrelation, Hurst exponent, ADF/KPSS tests, rolling beta/correlation

### Microstructure Features (`microstructure.py`)
VPIN, order flow imbalance, Kyle's Lambda, Amihud illiquidity, realized spread

### Feature Pipeline (`feature_pipeline.py`)
FeatureConfig with groups, normalization (ZSCORE, MINMAX, ROBUST), NaN handling, variance/correlation thresholds

---

## 9. Trading Engine

Located in `quant_trading_system/trading/trading_engine.py`.

### Engine States
STOPPED, STARTING, PRE_MARKET, MARKET_HOURS, POST_MARKET, PAUSED, ERROR, SHUTTING_DOWN

### TradingEngineConfig
mode (LIVE/PAPER/DRY_RUN), symbols, bar_interval, trading hours, max_daily_trades, max_daily_loss_pct, kill_switch_drawdown

### TradingEngine Methods
- `start()`, `stop(reason)`, `pause(reason)`, `resume()`
- `process_bar(bar)`, `process_signal(signal)`
- `reconcile()`, `get_metrics()`, `get_session()`

---

## 10. Execution System

Located in `quant_trading_system/execution/`.

### OrderManager (`order_manager.py`)
- `submit_order(request)`, `cancel_order(order_id)`, `modify_order(order_id, **updates)`
- `get_order(order_id)`, `get_active_orders()`, `get_order_history()`

### AlpacaClient (`alpaca_client.py`)
Account, orders, positions, market data, streaming. Rate limiting and retry logic.

### Execution Algorithms (`execution_algo.py`)
- **TWAPAlgorithm**: Time-weighted slices
- **VWAPAlgorithm**: Volume-weighted slices
- **ImplementationShortfallAlgorithm**: Minimize execution cost vs arrival

---

## 11. Risk Management

Located in `quant_trading_system/risk/`.

### Kill Switch (`limits.py`)
**CRITICAL COMPONENT** - Emergency trading halt.
- Reasons: MAX_DRAWDOWN, DAILY_LOSS, RAPID_PNL_DECLINE, SYSTEM_ERROR, DATA_FEED_FAILURE, BROKER_CONNECTION_LOSS
- Methods: `activate(reason)`, `is_active()`, `reset(authorized_by)`

### RiskLimitsConfig
- Loss: max_daily_loss_pct (0.05), max_weekly_loss_pct (0.10), max_drawdown_pct (0.15)
- Position: max_position_pct (0.10), max_total_positions (20)
- Exposure: max_sector_exposure_pct (0.30), max_correlation (0.80)
- Volatility: max_portfolio_volatility (0.25), vix_halt_threshold (35.0)

### PreTradeRiskChecker
Validates ALL orders: buying power, position limits, concentration, sector exposure, correlation, blacklist

### Position Sizing (`position_sizer.py`)
Methods: FIXED_DOLLAR, PERCENT_EQUITY, VOLATILITY_SCALED, RISK_PARITY, KELLY

---

## 12. Backtesting

Located in `quant_trading_system/backtest/`.

### BacktestConfig
initial_capital, mode (EVENT_DRIVEN/VECTORIZED), execution_mode (REALISTIC/OPTIMISTIC/PESSIMISTIC), commission_bps, slippage_bps, max_leverage

### Market Simulator (`simulator.py`)
Slippage models: FIXED, VOLUME_BASED, VOLATILITY_BASED, MARKET_IMPACT (Almgren-Chriss)
Simulates partial fills, latency, market impact

### Performance Analyzer (`analyzer.py`)
- ReturnMetrics: total/annualized return, best/worst day/month
- RiskAdjustedMetrics: Sharpe, Sortino, Calmar, Information ratio
- DrawdownMetrics: max_drawdown, duration, recovery, ulcer_index
- TradeMetrics: win_rate, profit_factor, expectancy, avg_win/loss

### Performance Attribution (`performance_attribution.py`)
Brinson-Fachler attribution, factor attribution, risk attribution

---

## 13. Data Pipeline

Located in `quant_trading_system/data/`.

### DataLoader (`loader.py`)
Formats: CSV, Parquet, HDF5. Methods: `load_symbol()`, `load_symbols()`, `validate_data()`, `get_available_symbols()`

### Preprocessor (`preprocessor.py`)
`handle_missing()`, `handle_outliers()`, `validate_ohlcv()`, `ensure_timezone()`

### LiveFeed (`live_feed.py`)
WebSocket real-time data. `connect()`, `disconnect()`, `subscribe_bars()`, `subscribe_trades()`

### Data Lineage (`lineage.py`)
Track data provenance for regulatory compliance. `register_source()`, `add_transformation()`, `get_full_lineage()`

---

## 14. Monitoring & Observability

Located in `quant_trading_system/monitoring/`.

### Prometheus Metrics (`metrics.py`)
System (CPU, memory, uptime), Trading (orders, fills, positions, equity, PnL), Risk (drawdown, VaR, kill_switch), Model (predictions, latency)

### Audit Logging (`audit.py`)
Immutable trail with SHA-256 chain. Events: ORDER_*, POSITION_*, RISK_*, MODEL_*, SIGNAL_*, SYSTEM_*
Methods: `log_order_created()`, `log_risk_check()`, `log_kill_switch_activated()`, `verify_chain()`, `export_range()`

### Alerting (`alerting.py`)
Channels: SLACK, EMAIL, PAGERDUTY, SMS. Severity: INFO, WARNING, ERROR, CRITICAL

---

## 15. Infrastructure

### Docker Stack
| Service | Port | Purpose |
|---------|------|---------|
| trading_app | 8000 | Main application |
| postgres (TimescaleDB) | 5432 | Time-series DB |
| redis | 6379 | Cache |
| prometheus | 9090 | Metrics |
| grafana | 3000 | Dashboards |

### Commands
```bash
docker-compose -f docker/docker-compose.yml up -d
docker-compose -f docker/docker-compose.yml logs -f trading_app
```

---

## 16. Testing

```bash
pytest tests/                                    # All tests
pytest tests/ --cov=quant_trading_system         # With coverage
pytest tests/unit/                               # Unit only
pytest -k "risk_checker" -v                      # Pattern match
pytest tests/ -n auto                            # Parallel
```

Key fixtures in `conftest.py`: sample_ohlcv_data, sample_order, sample_portfolio, event_bus, mock_alpaca_client

---

## 17. Architecture Invariants

### CRITICAL SAFETY RULES
1. **Kill Switch is Sacred** - NEVER bypass `KillSwitch.is_active()` check
2. **Pre-Trade Risk Checks** - ALL orders pass through `PreTradeRiskChecker`
3. **Decimal for Money** - NEVER use `float` for monetary values
4. **No Future Data Leakage** - All features must be strictly backward-looking
5. **Credentials Never in Code** - Use environment variables or secrets manager

### Architecture Patterns
1. **Event-Driven** - All inter-component communication via `EventBus`
2. **Singleton Pattern** - `EventBus`, `Settings`, `MetricsCollector`
3. **Abstract Base Classes** - New components MUST implement existing ABCs
4. **Repository Pattern** - Data access through `repository.py`
5. **Thread Safety** - Use `RLock` for shared state

### Data Integrity
1. **Timezone Awareness** - All timestamps in UTC with tzinfo
2. **OHLCV Validation** - High >= Low, High >= Open/Close
3. **Time-Series Splits** - NEVER use random train/test splits
4. **Walk-Forward Validation** - Use expanding window for ML

### Error Handling
1. **Custom Exception Hierarchy** - Use `TradingSystemError` subclasses
2. **Fail-Safe Defaults** - On error, REJECT order (not accept)
3. **Structured Logging** - JSON format for production
4. **Audit Trail** - All significant events logged immutably

---

## 18. Agent System

### Agent Registry
| Agent | Alias | Priority | Scope |
|-------|-------|----------|-------|
| System Architect | `@architect` | Highest | Core, config, interfaces |
| ML/Quant Engineer | `@mlquant` | High | Models, features, alphas |
| Trading & Execution | `@trader` | **Critical** | Execution, risk, orders |
| Data Engineer | `@data` | High | Pipelines, DB, storage |
| Infrastructure & QA | `@infra` | High | Docker, tests, monitoring |

### Priority Resolution
1. **@trader** (Critical) - Safety/risk concerns override everything
2. **@architect** (Highest) - Architecture decisions next
3. **Domain agents** (High) - Specific implementation details last

Agent files in `.claude/agents/`

---

## 19. Quick Reference

### Commands
```bash
python main.py trade --mode paper               # Paper trading
python main.py trade --mode live                # Live trading
python scripts/run_backtest.py --start-date 2024-01-01 --end-date 2024-06-30
python scripts/train_models.py --model xgboost --symbols AAPL MSFT
python main.py dashboard --port 8000
```

### Key Files
| Purpose | Location |
|---------|----------|
| Entry point | `main.py` |
| Settings | `quant_trading_system/config/settings.py` |
| Data types | `quant_trading_system/core/data_types.py` |
| Events | `quant_trading_system/core/events.py` |
| Model base | `quant_trading_system/models/base.py` |
| Risk/Kill switch | `quant_trading_system/risk/limits.py` |
| Trading engine | `quant_trading_system/trading/trading_engine.py` |
| Order manager | `quant_trading_system/execution/order_manager.py` |
| Backtest engine | `quant_trading_system/backtest/engine.py` |
| Feature pipeline | `quant_trading_system/features/feature_pipeline.py` |
| Alpha base | `quant_trading_system/alpha/alpha_base.py` |

### Implementation Status: Complete
All components implemented: Core types, events, data pipeline, features, ML models, validation gates, explainability, alpha factors, regime detection, backtesting, market simulation, risk management, position sizing, Alpaca integration, execution algos, monitoring, audit logging, Docker, 700+ tests.

---

*AlphaTrade System v1.0*
