# AlphaTrade System - Architecture Documentation

## 1. Executive Summary

**AlphaTrade** is an institutional-grade quantitative trading platform designed for automated equity trading. The system integrates machine learning models, real-time market data processing, risk management, and execution through Alpaca Markets API.

### System Purpose
Automated trading system targeting US equities market with support for:
- Real-time and historical data ingestion (15-minute bars)
- Multi-model ML prediction pipeline (XGBoost, LightGBM, LSTM, Transformer, PPO)
- Alpha factor generation and combination
- Risk-aware position sizing and portfolio management
- Paper and live trading execution through Alpaca

### Technology Stack
| Component | Technology | Version |
|-----------|------------|---------|
| Language | Python | 3.11+ |
| ML Framework | PyTorch, scikit-learn | 2.1+, 1.4+ |
| Gradient Boosting | XGBoost, LightGBM, CatBoost | 2.0+, 4.0+, 1.2+ |
| Data Processing | Pandas, NumPy | 2.1+, 1.26+ |
| Database | PostgreSQL + SQLAlchemy | 2.0+ |
| Broker API | Alpaca Markets | alpaca-py 0.13+ |
| Monitoring | Prometheus metrics | prometheus_client |
| Task Queue | Redis + Celery | (planned) |
| Configuration | Pydantic + YAML | pydantic 2.5+ |

```

### 2.2 Event-Driven Architecture

The system uses a publish-subscribe EventBus pattern for loose coupling:

```
┌─────────────┐     ┌─────────────────────────────────────────────────────┐
│  EventBus   │     │                    Event Types                       │
│  (Singleton)│     ├─────────────────────────────────────────────────────┤
│             │     │ MarketEvent   - New bar data, price updates          │
│  publish()  │────▶│ SignalEvent   - Trading signals from models          │
│  subscribe()│     │ OrderEvent    - Order lifecycle (created/filled/etc) │
│  emit()     │     │ RiskEvent     - Risk alerts, limit breaches          │
│             │     │ SystemEvent   - Startup, shutdown, errors            │
└─────────────┘     └─────────────────────────────────────────────────────┘
```

**Location**: `quant_trading_system/core/events.py:1-200`

### 2.3 Threading Model

```
Main Thread (asyncio)
├── Data Feed Handler (WebSocket)
├── Trading Engine Loop
├── Order Management
└── Risk Monitor (periodic checks)

Background Workers
├── Model Inference (thread pool)
├── Database Operations (async)
└── Metrics Collection (periodic)
```

Critical sections use `threading.RLock` for thread safety (see `risk/limits.py:40`).

---

## 3. Directory Structure

```
AlphaTrade_System/
├── main.py                          # Application entry point (407 lines)
├── pyproject.toml                   # Project dependencies (Poetry)
├── requirements.txt                 # Pip requirements
├── alembic.ini                      # Database migrations config
├── .env                             # Environment variables (API keys, DB config)
├── .gitignore                       # Git ignore patterns
│
├── docker/                          # Docker infrastructure
│   ├── Dockerfile                   # Multi-stage build (93 lines)
│   ├── docker-compose.yml           # Full stack deployment (225 lines)
│   ├── prometheus/
│   │   └── prometheus.yml           # Metrics scraping config (43 lines)
│   ├── alertmanager/
│   │   └── alertmanager.yml         # Alert routing rules (85 lines)
│   └── grafana/
│       └── provisioning/
│           └── datasources/
│               └── datasources.yml  # Grafana data sources (30 lines)
│
├── redis/                           # Windows Redis installation
│   ├── redis-server.exe             # Redis server binary
│   ├── redis-cli.exe                # Redis CLI
│   ├── redis.windows.conf           # Windows configuration
│   └── redis.windows-service.conf   # Windows service config
│
├── models/                          # Trained model artifacts
│   ├── xgboost_model.joblib         # XGBoost model (~250KB)
│   ├── lightgbm_model.joblib        # LightGBM model (~241KB)
│   └── lstm_model.pt                # LSTM PyTorch model (~403KB)
│
├── backtest_results/                # Backtest output
│   └── backtest_results.json        # Latest backtest metrics
│
├── notebooks/                       # Jupyter notebooks (empty - for research)
│
├── quant_trading_system/            # Main package
│   ├── __init__.py
│   │
│   ├── config/                      # Configuration management
│   │   ├── settings.py              # Pydantic settings (455 lines)
│   │   ├── alpaca_config.yaml       # Broker credentials
│   │   ├── risk_params.yaml         # Risk limits configuration
│   │   ├── model_configs.yaml       # ML model hyperparameters
│   │   └── symbols.yaml             # Trading universe
│   │
│   ├── core/                        # Foundation components
│   │   ├── data_types.py            # Dataclasses: Order, Position, Portfolio (400 lines)
│   │   ├── events.py                # EventBus and event types (200 lines)
│   │   ├── exceptions.py            # Custom exception hierarchy (150 lines)
│   │   ├── registry.py              # Component registry pattern (130 lines)
│   │   └── utils.py                 # Utility functions (250 lines)
│   │
│   ├── data/                        # Data ingestion & processing
│   │   ├── loader.py                # Historical data loading (300 lines)
│   │   ├── preprocessor.py          # Data cleaning & normalization (400 lines)
│   │   ├── live_feed.py             # Real-time WebSocket feeds (350 lines)
│   │   └── feature_store.py         # Feature caching layer (200 lines)
│   │
│   ├── database/                    # Persistence layer
│   │   ├── models.py                # SQLAlchemy ORM models (600 lines)
│   │   ├── connection.py            # Connection management (150 lines)
│   │   ├── repository.py            # Data access patterns (450 lines)
│   │   └── migrations/              # Alembic migrations
│   │       └── versions/
│   │           └── 001_initial_schema.py  # Tables: orders, positions, bars
│   │
│   ├── alpha/                       # Alpha factor generation
│   │   ├── alpha_base.py            # AlphaFactor ABC, AlphaSignal (300 lines)
│   │   ├── momentum_alphas.py       # Momentum-based alphas
│   │   ├── mean_reversion_alphas.py # Mean reversion alphas
│   │   ├── ml_alphas.py             # ML-based alpha factors
│   │   └── alpha_combiner.py        # Alpha combination methods (400 lines)
│   │
│   ├── features/                    # Feature engineering
│   │   ├── technical.py             # RSI, MACD, Bollinger (600 lines)
│   │   ├── technical_extended.py    # Advanced indicators
│   │   ├── statistical.py           # Statistical features (400 lines)
│   │   ├── microstructure.py        # Order flow features (300 lines)
│   │   ├── cross_sectional.py       # Cross-asset features (250 lines)
│   │   └── feature_pipeline.py      # Feature orchestration (350 lines)
│   │
│   ├── models/                      # ML/DL models
│   │   ├── base.py                  # TradingModel ABC (505 lines)
│   │   ├── classical_ml.py          # XGBoost, LightGBM, RF (821 lines)
│   │   ├── deep_learning.py         # LSTM, Transformer, TCN (1486 lines)
│   │   ├── ensemble.py              # Voting, Stacking, Adaptive (646 lines)
│   │   ├── reinforcement.py         # PPO, A2C agents (1119 lines)
│   │   └── model_manager.py         # Training, Registry, HPO (836 lines)
│   │
│   ├── execution/                   # Order execution
│   │   ├── order_manager.py         # Order lifecycle (500 lines)
│   │   ├── position_tracker.py      # Position state (350 lines)
│   │   ├── alpaca_client.py         # Alpaca API wrapper (600 lines)
│   │   └── execution_algo.py        # TWAP, VWAP, Implementation Shortfall (500 lines)
│   │
│   ├── trading/                     # Trading logic
│   │   ├── trading_engine.py        # Main orchestrator (700 lines)
│   │   ├── strategy.py              # Strategy interface (400 lines)
│   │   ├── portfolio_manager.py     # Portfolio optimization (500 lines)
│   │   └── signal_generator.py      # Signal processing (300 lines)
│   │
│   ├── risk/                        # Risk management
│   │   ├── limits.py                # Risk limits & kill switch (1084 lines)
│   │   ├── position_sizer.py        # Position sizing algos (400 lines)
│   │   ├── risk_monitor.py          # Real-time monitoring (350 lines)
│   │   └── portfolio_optimizer.py   # Mean-Variance, Risk Parity, HRP (600 lines)
│   │
│   ├── backtest/                    # Backtesting framework
│   │   ├── engine.py                # Event-driven & vectorized (806 lines)
│   │   ├── simulator.py             # Market simulation
│   │   ├── analyzer.py              # Performance metrics
│   │   └── optimizer.py             # Parameter optimization
│   │
│   └── monitoring/                  # Observability
│       ├── metrics.py               # Prometheus metrics (977 lines)
│       ├── alerting.py              # Alert management (500 lines)
│       ├── dashboard.py             # Dashboard state (400 lines)
│       └── logger.py                # Structured logging (300 lines)
│
├── scripts/                         # Operational scripts
│   ├── run_trading.py               # Live/paper trading (125 lines)
│   ├── run_backtest.py              # Backtest runner (200 lines)
│   ├── train_models.py              # Model training pipeline
│   ├── load_data.py                 # Data loading utilities
│   ├── migrate_db.py                # Database migrations
│   ├── run_dashboard.py             # Dashboard launcher
│   ├── run_backtest_validation.py   # Validation backtests
│   └── run_momentum_backtest.py     # Momentum strategy backtest
│
├── tests/                           # Test suite
│   ├── conftest.py                  # Pytest fixtures (258 lines)
│   ├── unit/                        # Unit tests (17 files)
│   │   ├── test_data_types.py
│   │   ├── test_events.py
│   │   ├── test_execution.py
│   │   ├── test_models.py
│   │   ├── test_risk.py
│   │   └── ...
│   └── integration/                 # Integration tests (4 files)
│       ├── test_data_pipeline.py
│       ├── test_monitoring_integration.py
│       └── test_websocket_reconnection.py
│
└── data/                            # Data directory
    └── raw/                         # Raw OHLCV data (47 CSV files, ~130MB)
        ├── AAPL_15min_2024-08-01.csv
        ├── MSFT_15min_2024-08-01.csv
        ├── GOOGL_15min_2024-08-01.csv
        └── ... (47 symbols total)
```

---

## 4. Core Components

### 4.1 Data Types (`core/data_types.py`)

**Implemented dataclasses:**

```python
@dataclass
class OHLCVBar:
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    vwap: Decimal | None = None
    trade_count: int | None = None

@dataclass
class Order:
    id: str
    symbol: str
    side: OrderSide  # BUY, SELL
    quantity: Decimal
    order_type: OrderType  # MARKET, LIMIT, STOP, STOP_LIMIT
    status: OrderStatus  # PENDING, SUBMITTED, FILLED, CANCELLED, REJECTED
    time_in_force: TimeInForce  # DAY, GTC, IOC, FOK
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    filled_quantity: Decimal = Decimal("0")
    filled_avg_price: Decimal | None = None

@dataclass
class Position:
    symbol: str
    quantity: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    cost_basis: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Decimal("0")
    side: PositionSide = PositionSide.LONG

@dataclass
class Portfolio:
    equity: Decimal
    cash: Decimal
    buying_power: Decimal
    positions: dict[str, Position]
    daily_pnl: Decimal
    total_pnl: Decimal

@dataclass
class TradeSignal:
    symbol: str
    direction: Direction  # LONG, SHORT, FLAT
    strength: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    horizon: int  # Bars ahead
    model_source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
```

### 4.2 Event System (`core/events.py`)

```python
class EventBus:
    """Singleton publish-subscribe event bus."""

    def subscribe(self, event_type: Type[Event], handler: Callable) -> None
    def unsubscribe(self, event_type: Type[Event], handler: Callable) -> None
    def publish(self, event: Event) -> None
    async def publish_async(self, event: Event) -> None

# Event types
class MarketDataEvent(Event):    # New bar data
class SignalEvent(Event):        # Trading signals
class OrderEvent(Event):         # Order lifecycle
class FillEvent(Event):          # Trade executions
class RiskEvent(Event):          # Risk alerts
class SystemEvent(Event):        # System status
```

### 4.3 Exception Hierarchy (`core/exceptions.py`)

```python
class TradingSystemError(Exception):        # Base exception
    ├── DataError                           # Data loading/processing
    │   ├── DataLoadError
    │   ├── DataValidationError
    │   └── DataNotFoundError
    ├── ExecutionError                      # Order execution
    │   ├── OrderRejectedError
    │   ├── InsufficientFundsError
    │   └── BrokerConnectionError
    ├── RiskError                           # Risk violations
    │   ├── PositionLimitError
    │   ├── DrawdownLimitError
    │   └── KillSwitchActiveError
    ├── ModelError                          # ML models
    │   ├── ModelNotFittedError
    │   └── PredictionError
    └── ConfigurationError                  # Config issues
```

---

## 5. Data Architecture

### 5.1 Data Flow

```
                    ┌─────────────────┐
                    │  Alpaca API     │
                    │  (Historical +  │
                    │   Real-time)    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐          ┌─────────────────┐
    │  DataLoader     │          │   LiveFeed      │
    │  (Historical)   │          │   (WebSocket)   │
    │                 │          │                 │
    │  • CSV files    │          │  • Bars         │
    │  • Parquet      │          │  • Quotes       │
    │  • API fetch    │          │  • Trades       │
    └────────┬────────┘          └────────┬────────┘
             │                            │
             └──────────┬─────────────────┘
                        ▼
              ┌─────────────────┐
              │  DataPreprocessor│
              │                 │
              │  • Clean data   │
              │  • Handle gaps  │
              │  • Normalize    │
              │  • Validate     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ FeaturePipeline │
              │                 │
              │  • Technical    │
              │  • Statistical  │
              │  • Microstructure│
              │  • Cross-section│
              └────────┬────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌─────────────────┐        ┌─────────────────┐
│  FeatureStore   │        │    Database     │
│  (Redis/Memory) │        │  (PostgreSQL)   │
│                 │        │                 │
│  • TTL cache    │        │  • OHLCV bars   │
│  • Fast lookups │        │  • Orders       │
│  • Computed feat│        │  • Positions    │
└─────────────────┘        └─────────────────┘
```

### 5.2 Database Schema

**Tables defined in `database/models.py` and `migrations/versions/001_initial_schema.py`:**

```sql
-- OHLCV Bar Data
CREATE TABLE bars (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume BIGINT NOT NULL,
    vwap DECIMAL(20, 8),
    trade_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (symbol, timestamp)
);

-- Orders
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    broker_id VARCHAR(100),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    limit_price DECIMAL(20, 8),
    stop_price DECIMAL(20, 8),
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    filled_avg_price DECIMAL(20, 8),
    time_in_force VARCHAR(10) NOT NULL,
    submitted_at TIMESTAMP WITH TIME ZONE,
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Positions
CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    quantity DECIMAL(20, 8) NOT NULL,
    avg_entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    cost_basis DECIMAL(20, 8) NOT NULL,
    market_value DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    side VARCHAR(10) NOT NULL,
    opened_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trades (Executions)
CREATE TABLE trades (
    id UUID PRIMARY KEY,
    order_id UUID REFERENCES orders(id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    commission DECIMAL(20, 8) DEFAULT 0,
    executed_at TIMESTAMP WITH TIME ZONE NOT NULL
);

-- Trade Signals
CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    strength DECIMAL(10, 6) NOT NULL,
    confidence DECIMAL(10, 6) NOT NULL,
    model_source VARCHAR(100) NOT NULL,
    horizon INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Risk Events
CREATE TABLE risk_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    symbol VARCHAR(20),
    description TEXT NOT NULL,
    metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_bars_symbol_timestamp ON bars(symbol, timestamp DESC);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_symbol ON orders(symbol);
CREATE INDEX idx_signals_symbol_created ON signals(symbol, created_at DESC);
```

### 5.3 Raw Data Inventory

Located in `data/raw/` (47 CSV files, ~130MB total):

| Symbol | File | Size | Timeframe |
|--------|------|------|-----------|
| AAPL | AAPL_15min_2024-08-01.csv | ~3MB | 15-min bars |
| MSFT | MSFT_15min_2024-08-01.csv | ~3MB | 15-min bars |
| GOOGL | GOOGL_15min_2024-08-01.csv | ~3MB | 15-min bars |
| AMZN | AMZN_15min_2024-08-01.csv | ~3MB | 15-min bars |
| NVDA | NVDA_15min_2024-08-01.csv | ~3MB | 15-min bars |
| TSLA | TSLA_15min_2024-08-01.csv | ~3MB | 15-min bars |
| META | META_15min_2024-08-01.csv | ~3MB | 15-min bars |
| ... | ... (40 more symbols) | ... | 15-min bars |

**Data Format:**
```csv
timestamp,open,high,low,close,volume,vwap,trade_count
2024-08-01T09:30:00Z,223.45,224.10,223.20,223.85,1250000,223.67,8500
```

---

## 6. API Contracts

### 6.1 TradingModel Interface (`models/base.py:29-363`)

All ML models must implement this interface:

```python
class TradingModel(ABC):
    """Abstract base class for trading models."""

    # Properties
    @property
    def name(self) -> str: ...
    @property
    def version(self) -> str: ...
    @property
    def model_type(self) -> ModelType: ...  # CLASSIFIER or REGRESSOR
    @property
    def is_fitted(self) -> bool: ...
    @property
    def feature_names(self) -> list[str]: ...
    @property
    def training_metrics(self) -> dict[str, float]: ...

    # Required methods
    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "TradingModel": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]: ...

    # Optional methods
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def save(self, path: str | Path) -> None: ...
    def load(self, path: str | Path) -> "TradingModel": ...
```

### 6.2 AlphaFactor Interface (`alpha/factor_base.py`)

```python
class AlphaFactor(ABC):
    """Base class for alpha factors."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def category(self) -> AlphaCategory: ...  # MOMENTUM, MEAN_REVERSION, etc.

    @abstractmethod
    def compute(
        self,
        bars: pd.DataFrame,
        universe: list[str],
    ) -> pd.Series: ...  # Returns alpha scores by symbol

    def normalize(self, alphas: pd.Series) -> pd.Series: ...
    def winsorize(self, alphas: pd.Series, limits: tuple) -> pd.Series: ...
```

### 6.3 Strategy Interface (`trading/strategy.py`)

```python
class Strategy(ABC):
    """Trading strategy interface."""

    @abstractmethod
    def generate_signals(
        self,
        market_data: dict[str, pd.DataFrame],
        portfolio: Portfolio,
    ) -> list[TradeSignal]: ...

    @abstractmethod
    def get_position_sizes(
        self,
        signals: list[TradeSignal],
        portfolio: Portfolio,
        risk_limits: RiskLimits,
    ) -> dict[str, Decimal]: ...

    def on_bar(self, bar: OHLCVBar) -> None: ...
    def on_fill(self, fill: FillEvent) -> None: ...
```

### 6.4 Risk Checker Interface (`risk/limits.py`)

```python
class PreTradeRiskChecker:
    """Pre-trade risk validation."""

    def check_order(
        self,
        order: Order,
        portfolio: Portfolio,
        config: RiskLimitsConfig,
    ) -> tuple[bool, str]: ...

    def check_position_limit(self, symbol: str, new_qty: Decimal) -> bool: ...
    def check_concentration(self, symbol: str, value: Decimal) -> bool: ...
    def check_sector_exposure(self, sector: str, value: Decimal) -> bool: ...

class IntraTradeMonitor:
    """Real-time risk monitoring."""

    def update(self, portfolio: Portfolio, market_data: dict) -> list[RiskAlert]: ...
    def check_drawdown(self, portfolio: Portfolio) -> bool: ...
    def check_volatility(self, returns: np.ndarray) -> bool: ...
```

---

## 7. Trading Engine Specifics

### 7.1 Order Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Signal     │────▶│  Risk Check  │────▶│  Order Mgr   │
│  Generator   │     │  (Pre-Trade) │     │              │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                     ┌──────────────┐             │
                     │  Kill Switch │◀────────────┤
                     │  (if active) │             │
                     └──────────────┘             │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Position    │◀────│  Fill Event  │◀────│ Alpaca API   │
│  Tracker     │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

### 7.2 Position Sizing Algorithms (`risk/position_sizer.py`)

Implemented methods:
1. **Fixed Fractional**: `position_size = equity * risk_fraction / stop_distance`
2. **Kelly Criterion**: `f* = (p * b - q) / b` where p=win_rate, b=win/loss ratio
3. **Volatility Targeting**: `position_size = target_vol / realized_vol * equity`
4. **Risk Parity**: Equal risk contribution across positions
5. **Maximum Diversification**: Optimize for diversification ratio

### 7.3 Signal Generation Pipeline

```python
# Location: trading/signal_generator.py
class SignalGenerator:
    def generate(
        self,
        features: pd.DataFrame,
        models: list[TradingModel],
        alpha_factors: list[AlphaFactor],
    ) -> list[TradeSignal]:

        # 1. Get model predictions
        predictions = [model.predict(features) for model in models]

        # 2. Compute alpha scores
        alphas = [factor.compute(features) for factor in alpha_factors]

        # 3. Combine signals
        combined = self.combiner.combine(predictions, alphas)

        # 4. Apply thresholds
        signals = self._filter_signals(combined, threshold=0.5)

        return signals
```

### 7.4 Kill Switch Implementation (`risk/limits.py:800-900`)

```python
class KillSwitch:
    """Emergency trading halt mechanism."""

    def __init__(self):
        self._active = False
        self._lock = threading.RLock()
        self._reason: str | None = None
        self._activated_at: datetime | None = None

    def activate(self, reason: str) -> None:
        """Activate kill switch - halts ALL trading."""
        with self._lock:
            self._active = True
            self._reason = reason
            self._activated_at = datetime.now(timezone.utc)
            logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

    def is_active(self) -> bool:
        """Thread-safe check if kill switch is active."""
        with self._lock:
            return self._active

    def reset(self, authorized_by: str) -> None:
        """Reset kill switch - requires authorization."""
        with self._lock:
            if self._active:
                logger.warning(f"Kill switch reset by: {authorized_by}")
                self._active = False
                self._reason = None
```

Trigger conditions:
- Daily loss exceeds 5% (`max_daily_loss_pct`)
- Portfolio drawdown exceeds 15% (`max_drawdown_pct`)
- Consecutive losing trades exceeds threshold
- Manual activation via API

---

## 8. Infrastructure

### 8.1 Configuration Management

**Pydantic Settings (`config/settings.py`):**

```python
class Settings(BaseSettings):
    # Application
    app_name: str = "Quant Trading System"
    app_version: str = "1.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False

    # Database
    database: DatabaseSettings  # host, port, name, pool_size

    # Redis
    redis: RedisSettings  # host, port, db

    # Trading
    trading: TradingSettings
        bar_timeframe: str = "15Min"
        signal_threshold: float = 0.5
        max_open_orders: int = 10

    # Risk
    risk: RiskSettings
        max_position_pct: float = 0.10
        max_portfolio_positions: int = 20
        max_daily_loss_pct: float = 0.05
        max_drawdown_pct: float = 0.15

    # Alpaca
    alpaca: AlpacaSettings  # from alpaca_config.yaml

    # Model configs loaded from YAML
    model_configs: dict[str, Any]  # from model_configs.yaml
```

### 8.2 Docker Infrastructure

**Status: FULLY IMPLEMENTED** (`docker/`)

**Dockerfile** (`docker/Dockerfile`) - Multi-stage build:
```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder
# Install build dependencies, create venv, install requirements

# Stage 2: Runtime
FROM python:3.11-slim as runtime
# Non-root user for security
RUN useradd --uid 1000 --gid trading trading
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "main.py", "trade", "--mode", "paper"]
```

**docker-compose.yml** (`docker/docker-compose.yml`) - Full stack:
```yaml
services:
  trading_app:           # Main application
    build: .
    depends_on: [postgres, redis]
    ports: ["8000:8000"]
    deploy:
      resources:
        limits: { cpus: '2', memory: 4G }

  postgres:              # TimescaleDB for time-series
    image: timescale/timescaledb:latest-pg15
    ports: ["5432:5432"]

  redis:                 # Cache & message queue
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 512mb

  prometheus:            # Metrics collection
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:               # Visualization
    image: grafana/grafana:latest
    ports: ["3000:3000"]

  alertmanager:          # Alert routing
    image: prom/alertmanager:latest
    ports: ["9093:9093"]

networks:
  trading_network:
  monitoring_network:

volumes:
  postgres_data, redis_data, prometheus_data, grafana_data
```

**Start the stack:**
```bash
docker-compose -f docker/docker-compose.yml up -d
```

### 8.3 Windows Redis Installation

**Location:** `redis/`

The project includes a local Windows Redis installation for development:
- `redis-server.exe` - Redis server
- `redis-cli.exe` - Command-line interface
- `redis.windows.conf` - Configuration file (port 6379)

**Start Redis locally:**
```bash
cd redis && redis-server.exe redis.windows.conf
```

### 8.4 Monitoring Stack

**Prometheus Metrics (`monitoring/metrics.py`):**

```python
# System metrics
system_cpu_usage = Gauge("trading_system_cpu_percent", "CPU usage")
system_memory_usage = Gauge("trading_system_memory_mb", "Memory usage")

# Trading metrics
orders_total = Counter("trading_orders_total", "Total orders", ["side", "status"])
fills_total = Counter("trading_fills_total", "Total fills", ["side"])
position_value = Gauge("trading_position_value", "Position value", ["symbol"])
portfolio_equity = Gauge("trading_portfolio_equity", "Total equity")
portfolio_pnl_daily = Gauge("trading_pnl_daily", "Daily P&L")

# Risk metrics
drawdown_current = Gauge("risk_drawdown_current", "Current drawdown")
var_95 = Gauge("risk_var_95", "95% VaR")
sharpe_ratio = Gauge("risk_sharpe_ratio", "Sharpe ratio")

# Model metrics
model_predictions = Counter("model_predictions_total", "Predictions", ["model"])
model_latency = Histogram("model_prediction_latency_seconds", "Prediction latency")
```

### 8.5 Logging (`monitoring/logger.py`)

```python
class LogCategory(str, Enum):
    SYSTEM = "system"
    TRADING = "trading"
    DATA = "data"
    MODEL = "model"
    RISK = "risk"
    EXECUTION = "execution"

def setup_logging(
    level: str = "INFO",
    log_format: LogFormat = LogFormat.JSON,
    log_dir: Path | None = None,
) -> None:
    """Configure structured logging with rotation."""
```

Supports JSON and text formats, file rotation, and category-based filtering.

---

## 9. Security Considerations

### 9.1 Credential Management

**Environment File (`.env`):**
```bash
# Alpaca API (Paper Trading)
ALPACA_API_KEY=PKHG...      # Your API key
ALPACA_API_SECRET=Ej9p...   # Your API secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Database
DATABASE__HOST=localhost
DATABASE__PORT=5432
DATABASE__NAME=market_data
DATABASE__USER=alphatrade
DATABASE__PASSWORD=...      # Your DB password

# Redis
REDIS__HOST=localhost
REDIS__PORT=6379

# Application
APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO
```

**YAML Config (`config/alpaca_config.yaml`):**
```yaml
alpaca:
  api_key: "${ALPACA_API_KEY}"      # References .env
  api_secret: "${ALPACA_API_SECRET}"
  base_url: "https://paper-api.alpaca.markets"
```

**Security Recommendations:**
1. **NEVER commit `.env` to version control** - add to `.gitignore`
2. Use AWS Secrets Manager or HashiCorp Vault for production
3. Implement API key rotation policy
4. Current `.env` contains real API keys - rotate immediately if exposed

### 9.2 Model Serialization Warning

**Location: `models/base.py:267-274`**

```python
# SECURITY WARNING: pickle.load() can execute arbitrary code.
# Only load model files from trusted sources.
# The checksum verification above provides some integrity check,
# but does NOT protect against malicious files created with valid checksums.
# For production use, consider using safer serialization formats like
# safetensors for neural networks or joblib with mmap_mode for sklearn models.
```

### 9.3 API Rate Limiting

The Alpaca client (`execution/alpaca_client.py`) implements:
- Automatic retry with exponential backoff
- Rate limit awareness (200 requests/minute)
- Connection pooling

### 9.4 Input Validation

- All external data passes through `DataPreprocessor` validation
- Order parameters validated against `RiskLimitsConfig`
- SQL injection prevented via SQLAlchemy parameterized queries

---

## 10. Current State Assessment

### 10.1 Implemented Features

| Component | Status | Notes |
|-----------|--------|-------|
| Core data types | **Complete** | All dataclasses implemented |
| Event system | **Complete** | EventBus with async support |
| Exception hierarchy | **Complete** | Comprehensive error types |
| Data loader | **Complete** | CSV, Parquet, API support |
| Data preprocessor | **Complete** | Cleaning, normalization |
| Live feed | **Partial** | WebSocket structure, needs testing |
| Feature store | **Complete** | Redis/memory caching |
| Technical features | **Complete** | 50+ indicators |
| Statistical features | **Complete** | Rolling statistics |
| ML models | **Complete** | XGBoost, LightGBM, RF, ElasticNet |
| DL models | **Complete** | LSTM, Transformer, TCN |
| RL agents | **Complete** | PPO, A2C |
| Ensemble methods | **Complete** | Voting, Stacking, Adaptive |
| Model manager | **Complete** | Training, registry, HPO |
| Alpha factors | **Complete** | Momentum, MeanRev, Quality |
| Order manager | **Complete** | Full lifecycle |
| Position tracker | **Complete** | Real-time tracking |
| Alpaca client | **Complete** | REST + WebSocket |
| Risk limits | **Complete** | Pre-trade, intra-trade, kill switch |
| Position sizer | **Complete** | 5 algorithms |
| Portfolio optimizer | **Complete** | Mean-Variance, Risk Parity, HRP, Black-Litterman |
| Execution algos | **Complete** | TWAP, VWAP, Implementation Shortfall, Adaptive |
| Backtest analyzer | **Complete** | Return metrics, drawdowns, trade analysis |
| Backtest engine | **Complete** | Event-driven + vectorized |
| Prometheus metrics | **Complete** | 40+ metrics |
| Logging | **Complete** | Structured JSON logging |
| Database schema | **Complete** | All tables defined |
| Docker infrastructure | **Complete** | Dockerfile, docker-compose, Prometheus, Grafana, Alertmanager |
| Redis (Windows) | **Complete** | Local Redis installation included |
| Environment config | **Complete** | .env file with all settings |

### 10.2 Trained Model Artifacts

**Location:** `models/`

| Model | File | Size | Format |
|-------|------|------|--------|
| XGBoost | xgboost_model.joblib | 250KB | Joblib |
| LightGBM | lightgbm_model.joblib | 241KB | Joblib |
| LSTM | lstm_model.pt | 403KB | PyTorch |

### 10.3 Backtest Results

**Location:** `backtest_results/backtest_results.json`

Latest backtest (Trend Following Momentum strategy):
```json
{
  "initial_capital": 100000,
  "final_equity": 49380.64,
  "total_return_pct": -50.62,
  "sharpe_ratio": -3.20,
  "max_drawdown_pct": 52.58,
  "total_trades": 3080,
  "win_rate_pct": 39.87,
  "profit_factor": 1.14
}
```
**Note:** This backtest shows the strategy needs refinement - negative Sharpe ratio indicates poor risk-adjusted returns.

### 10.4 Stub/Placeholder Code

| Location | Description |
|----------|-------------|
| `main.py:121-138` | Component initialization is placeholder (TODO: connect DB, broker, load models) |
| `scripts/run_backtest.py:133-152` | Backtest results are hardcoded in script (use backtest engine instead) |
| `trading/trading_engine.py` | Full integration between components needs completion |

### 10.5 Known Issues & Gaps

1. **Celery task queue** - Task queue mentioned but not fully implemented
2. **Dashboard UI** - State management exists but no frontend (use Grafana instead)
3. **Alerting channels** - Alert manager configured but notification backends commented out (Slack, email, PagerDuty templates exist)
4. **Model versioning** - Registry filesystem-based, consider MLflow for production

### 10.6 Technical Debt

1. Some duplicate code between `classical_ml.py` models (`_compute_metrics`)
2. RL agents have unused imports (`from datetime import datetime, timezone`)
3. Test coverage varies - core modules well tested, trading engine less so
4. Backtest strategy parameters need optimization (current results show -50% return)

---

## 11. Development Guidelines

### 11.1 Code Style

- **Python**: Follow PEP 8, enforced via Black formatter
- **Type hints**: Required for all public functions
- **Docstrings**: Google style docstrings
- **Line length**: 100 characters
- **Imports**: isort for organization

### 11.2 Testing Strategy

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=quant_trading_system --cov-report=html

# Run specific category
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
```

Test fixtures in `tests/conftest.py` provide:
- `sample_ohlcv_data`, `sample_order_data`, `sample_position_data`
- `sample_portfolio`, `sample_risk_metrics`
- `event_bus`, `component_registry`, `metrics_collector`
- `mock_settings` for configuration testing

### 11.3 Adding New Models

1. Inherit from `TradingModel` or `TimeSeriesModel`
2. Implement required abstract methods: `fit()`, `predict()`, `get_feature_importance()`
3. Override `save()`/`load()` if custom serialization needed
4. Register in `model_manager.py:class_map` for auto-discovery
5. Add hyperparameters to `config/model_configs.yaml`
6. Write unit tests in `tests/unit/test_models.py`

### 11.4 Adding New Alpha Factors

1. Create class inheriting from `AlphaFactor`
2. Implement `compute()` returning `pd.Series` of alpha scores
3. Register in `alpha/alpha_strategies.py`
4. Add to combiner configuration

---

## 12. Integration Points

### 12.1 Alpaca Markets API

**Configuration:** `config/alpaca_config.yaml`

```yaml
alpaca:
  api_key: "${ALPACA_API_KEY}"
  api_secret: "${ALPACA_API_SECRET}"
  base_url: "https://paper-api.alpaca.markets"  # Paper
  # base_url: "https://api.alpaca.markets"      # Live
  data_url: "https://data.alpaca.markets"

  # WebSocket endpoints
  stream_url: "wss://stream.data.alpaca.markets/v2/iex"
  trading_stream_url: "wss://paper-api.alpaca.markets/stream"
```

**Capabilities:**
- Historical bars (REST): `/v2/stocks/{symbol}/bars`
- Real-time bars (WebSocket): Subscribe to bar updates
- Order management: Submit, cancel, get status
- Account info: Balance, positions, orders

### 12.2 Database (PostgreSQL)

**Connection:** `database/connection.py`

```python
# Async connection pool
engine = create_async_engine(
    settings.database.url,
    pool_size=settings.database.pool_size,
    max_overflow=10,
    pool_timeout=30,
)

async_session = async_sessionmaker(engine, class_=AsyncSession)
```

### 12.3 Redis (Feature Store)

**Configuration:** `config/settings.py`

```python
class RedisSettings(BaseSettings):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"
```

### 12.4 Prometheus

**Endpoint:** `/metrics` (when dashboard running)

Scrape config:
```yaml
scrape_configs:
  - job_name: 'trading-system'
    static_configs:
      - targets: ['localhost:8000']
```

---

## 13. Future AI Assistant Context

### 13.1 Key Design Decisions

1. **Event-driven architecture** - Chosen for loose coupling and testability
2. **Abstract base classes** - `TradingModel`, `AlphaFactor`, `Strategy` for extensibility
3. **Decimal for money** - Avoid floating-point precision issues
4. **Singleton patterns** - EventBus, MetricsCollector, Settings
5. **Time-series aware ML** - Walk-forward validation, no future leakage

### 13.2 Common Tasks

**Add a new trading strategy:**
```python
# 1. Create in trading/strategies/my_strategy.py
class MyStrategy(Strategy):
    def generate_signals(self, market_data, portfolio):
        # Your logic here
        return signals

# 2. Register in trading_engine.py
# 3. Add config in config/settings.py
```

**Train a new model:**
```python
from quant_trading_system.models import XGBoostModel
from quant_trading_system.models.model_manager import ModelManager

manager = ModelManager()
model = XGBoostModel(name="my_xgboost", n_estimators=500)
model = manager.train_model(model, X, y, validation_split=0.2)
manager.registry.register(model, tags=["production"])
```

**Run backtest:**
```bash
python scripts/run_backtest.py \
    --start-date 2024-01-01 \
    --end-date 2024-06-30 \
    --symbols AAPL MSFT GOOGL \
    --initial-capital 100000 \
    --model ensemble
```

**Start with Docker:**
```bash
# Start full stack (trading app, PostgreSQL, Redis, Prometheus, Grafana)
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f trading_app

# Access services:
# - Trading App: http://localhost:8000
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
# - Alertmanager: http://localhost:9093

# Stop all services
docker-compose -f docker/docker-compose.yml down
```

**Start Redis locally (Windows):**
```bash
cd redis && redis-server.exe redis.windows.conf
```

### 13.3 Architecture Invariants

1. **Never** use future data in training/backtesting (time-series split enforced)
2. **Always** validate orders through risk checks before submission
3. **Never** bypass the kill switch once activated
4. **Always** use Decimal for monetary values
5. **Never** store credentials in code - use environment variables

### 13.4 Important File Locations

| Purpose | Location |
|---------|----------|
| Entry point | `main.py` |
| Configuration | `quant_trading_system/config/settings.py` |
| Environment variables | `.env` |
| Data types | `quant_trading_system/core/data_types.py` |
| Event bus | `quant_trading_system/core/events.py` |
| Model base class | `quant_trading_system/models/base.py` |
| Risk limits | `quant_trading_system/risk/limits.py` |
| Order execution | `quant_trading_system/execution/order_manager.py` |
| Alpaca client | `quant_trading_system/execution/alpaca_client.py` |
| Backtest engine | `quant_trading_system/backtest/engine.py` |
| Test fixtures | `tests/conftest.py` |
| **Docker** | |
| Dockerfile | `docker/Dockerfile` |
| Docker Compose | `docker/docker-compose.yml` |
| Prometheus config | `docker/prometheus/prometheus.yml` |
| Alertmanager config | `docker/alertmanager/alertmanager.yml` |
| Grafana datasources | `docker/grafana/provisioning/datasources/datasources.yml` |
| **Local Redis** | |
| Redis server | `redis/redis-server.exe` |
| Redis config | `redis/redis.windows.conf` |
| **Artifacts** | |
| Trained models | `models/*.joblib`, `models/*.pt` |
| Backtest results | `backtest_results/backtest_results.json` |
| Raw market data | `data/raw/*.csv` (47 symbols) |

### 13.5 Debugging Tips

1. Enable debug logging: `settings.debug = True`
2. Check risk events in database: `SELECT * FROM risk_events ORDER BY created_at DESC`
3. Monitor metrics: `curl localhost:8000/metrics | grep trading_`
4. View kill switch status: `kill_switch.is_active()` with reason

### 13.6 Performance Considerations

- Feature computation is CPU-bound - consider multiprocessing for large universes
- Deep learning models benefit from GPU (`device="cuda"`)
- Database queries should use async for non-blocking I/O
- Redis caching reduces redundant feature computation

---