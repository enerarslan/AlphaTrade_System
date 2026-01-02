# AlphaTrade System - Comprehensive Architecture Documentation

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Directory Structure](#2-directory-structure)
3. [Core Data Types](#3-core-data-types)
4. [Event System](#4-event-system)
5. [Configuration](#5-configuration)
6. [Models & Machine Learning](#6-models--machine-learning)
7. [Alpha Factors](#7-alpha-factors)
8. [Feature Engineering](#8-feature-engineering)
9. [Trading Engine](#9-trading-engine)
10. [Execution System](#10-execution-system)
11. [Risk Management](#11-risk-management)
12. [Backtesting](#12-backtesting)
13. [Data Pipeline](#13-data-pipeline)
14. [Monitoring & Observability](#14-monitoring--observability)
15. [Infrastructure](#15-infrastructure)
16. [Testing](#16-testing)
17. [Architecture Invariants](#17-architecture-invariants)
18. [Agent System](#18-agent-system)
19. [Quick Reference](#19-quick-reference)

---

## 1. System Overview

**AlphaTrade** is an institutional-grade quantitative trading platform for automated US equity trading. Built with Python 3.11+, it implements a complete trading pipeline from data ingestion to live execution.

### Core Capabilities
- Real-time and historical market data (15-min bars via Alpaca)
- Multi-model ML pipeline (XGBoost, LightGBM, LSTM, Transformer, PPO)
- Alpha factor generation and combination with IC/IR tracking
- Market regime detection for adaptive strategies
- Risk-aware position sizing and portfolio optimization
- Event-driven backtesting with realistic market simulation
- Paper and live trading execution with kill switch safety
- Full regulatory compliance (audit logging, data lineage)

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Language | Python 3.11+ | Core runtime |
| ML Framework | PyTorch 2.1+, scikit-learn | Deep learning, classical ML |
| Gradient Boosting | XGBoost, LightGBM, CatBoost | Tabular models |
| Data Processing | Pandas, Polars, NumPy | Time-series manipulation |
| Database | PostgreSQL + TimescaleDB | Time-series storage |
| ORM | SQLAlchemy 2.0 | Database abstraction |
| Cache | Redis 7+ | Feature store, session cache |
| Broker | Alpaca Markets (alpaca-py) | Paper/live trading |
| Monitoring | Prometheus, Grafana | Metrics and dashboards |
| Alerting | Slack, Email, PagerDuty | Incident response |
| Config | Pydantic + YAML | Type-safe configuration |
| Containerization | Docker, docker-compose | Deployment |

---

## 2. Directory Structure

```
AlphaTrade_System/
├── main.py                         # Application entry point (CLI)
├── pyproject.toml                  # Project dependencies and metadata
├── CLAUDE.md                       # This documentation file
├── .env                            # Environment variables (NOT in git)
│
├── quant_trading_system/           # Main Python package
│   ├── __init__.py
│   │
│   ├── config/                     # Configuration management
│   │   ├── settings.py             # Pydantic settings models
│   │   ├── secure_config.py        # Secrets manager (AWS/Vault/env)
│   │   ├── trading_config.yaml     # Trading parameters
│   │   ├── risk_params.yaml        # Risk limits configuration
│   │   ├── model_configs.yaml      # Model hyperparameters
│   │   └── symbols.yaml            # Trading universe
│   │
│   ├── core/                       # Foundation layer
│   │   ├── data_types.py           # Core dataclasses (Order, Position, etc.)
│   │   ├── events.py               # EventBus implementation
│   │   ├── exceptions.py           # Custom exception hierarchy
│   │   ├── registry.py             # Component registry pattern
│   │   └── utils.py                # Shared utilities
│   │
│   ├── data/                       # Data ingestion layer
│   │   ├── loader.py               # Historical data loading
│   │   ├── preprocessor.py         # Data cleaning, normalization
│   │   ├── live_feed.py            # WebSocket real-time feeds
│   │   ├── feature_store.py        # Redis/memory feature cache
│   │   ├── lineage.py              # Data lineage tracking (regulatory)
│   │   └── data_lineage.py         # Extended lineage features
│   │
│   ├── alpha/                      # Alpha factor system
│   │   ├── alpha_base.py           # AlphaFactor ABC
│   │   ├── momentum_alphas.py      # Momentum factor implementations
│   │   ├── mean_reversion_alphas.py # Mean reversion factors
│   │   ├── ml_alphas.py            # ML-based alpha factors
│   │   ├── alpha_combiner.py       # Alpha combination methods
│   │   ├── alpha_metrics.py        # IC/IR metrics calculation
│   │   └── regime_detection.py     # Market regime classification
│   │
│   ├── features/                   # Feature engineering
│   │   ├── technical.py            # Technical indicators (RSI, MACD, etc.)
│   │   ├── technical_extended.py   # Extended technical features
│   │   ├── statistical.py          # Rolling stats, ADF test
│   │   ├── microstructure.py       # VPIN, order flow imbalance
│   │   ├── cross_sectional.py      # Cross-asset features
│   │   └── feature_pipeline.py     # Pipeline orchestration
│   │
│   ├── models/                     # ML/DL models
│   │   ├── base.py                 # TradingModel ABC
│   │   ├── classical_ml.py         # XGBoost, LightGBM, RF, ElasticNet
│   │   ├── deep_learning.py        # LSTM, Transformer, GRU, TCN
│   │   ├── ensemble.py             # Voting, Stacking, Adaptive ensemble
│   │   ├── reinforcement.py        # PPO, A2C, DQN agents
│   │   ├── model_manager.py        # Training, registry, HPO
│   │   ├── validation_gates.py     # Pre-deployment model validation
│   │   └── explainability.py       # SHAP/LIME interpretability
│   │
│   ├── execution/                  # Order execution
│   │   ├── order_manager.py        # Order lifecycle management
│   │   ├── position_tracker.py     # Position state tracking
│   │   ├── alpaca_client.py        # Alpaca API wrapper
│   │   └── execution_algo.py       # TWAP, VWAP, Implementation Shortfall
│   │
│   ├── trading/                    # Trading orchestration
│   │   ├── trading_engine.py       # Main trading orchestrator
│   │   ├── strategy.py             # Strategy interface and implementations
│   │   ├── signal_generator.py     # Signal generation logic
│   │   └── portfolio_manager.py    # Portfolio state management
│   │
│   ├── risk/                       # Risk management
│   │   ├── limits.py               # Risk limits, kill switch
│   │   ├── position_sizer.py       # Position sizing algorithms
│   │   ├── portfolio_optimizer.py  # Mean-variance, risk parity
│   │   └── risk_monitor.py         # Real-time risk monitoring
│   │
│   ├── backtest/                   # Backtesting engine
│   │   ├── engine.py               # Event-driven backtest engine
│   │   ├── simulator.py            # Market simulation (slippage, fills)
│   │   ├── analyzer.py             # Performance metrics calculation
│   │   ├── optimizer.py            # Parameter optimization
│   │   └── performance_attribution.py # Brinson, factor attribution
│   │
│   ├── database/                   # Database layer
│   │   ├── connection.py           # Connection management
│   │   ├── models.py               # SQLAlchemy ORM models
│   │   ├── repository.py           # Data access patterns
│   │   └── migrations/             # Alembic migrations
│   │
│   └── monitoring/                 # Observability
│       ├── metrics.py              # Prometheus metrics definitions
│       ├── alerting.py             # Alert management
│       ├── logger.py               # Structured logging
│       ├── dashboard.py            # Web dashboard
│       └── audit.py                # Regulatory audit logging
│
├── scripts/                        # Operational scripts
│   ├── run_trading.py              # Live/paper trading launcher
│   ├── run_backtest.py             # Backtest runner
│   └── train_models.py             # Model training script
│
├── tests/                          # Test suite (700+ tests)
│   ├── conftest.py                 # Pytest fixtures
│   ├── unit/                       # Unit tests (17 files)
│   └── integration/                # Integration tests
│
├── docker/                         # Docker infrastructure
│   ├── Dockerfile                  # Multi-stage build
│   ├── docker-compose.yml          # Full stack deployment
│   ├── prometheus/                 # Prometheus config
│   ├── grafana/                    # Grafana dashboards
│   └── alertmanager/               # Alert routing
│
├── data/raw/                       # Historical data (47 symbols)
│
├── models/                         # Trained model artifacts
│
└── .claude/                        # Claude agent configuration
    ├── agents/                     # Agent definitions
    └── skills/                     # Skill definitions
```

---

## 3. Core Data Types

Located in `quant_trading_system/core/data_types.py`. All models use Pydantic for validation and are frozen (immutable) where appropriate.

### Enumerations

```python
class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    FILLED = "filled"
    PARTIAL_FILLED = "partial_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class TimeInForce(str, Enum):
    DAY = "day"           # Valid for trading day
    GTC = "gtc"           # Good till cancelled
    IOC = "ioc"           # Immediate or cancel
    FOK = "fok"           # Fill or kill
    OPG = "opg"           # Market on open
    CLS = "cls"           # Market on close
```

### Core Models

#### OHLCVBar
```python
class OHLCVBar(BaseModel):
    """OHLCV bar data with validation."""
    symbol: str
    timestamp: datetime  # Must be timezone-aware (UTC)
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    vwap: Decimal | None = None
    trade_count: int | None = None

    # Validation: high >= low, high >= open/close, low <= open/close
```

#### TradeSignal
```python
class TradeSignal(BaseModel):
    """Trading signal from model/alpha."""
    signal_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime
    symbol: str
    direction: Direction
    strength: float = Field(ge=-1.0, le=1.0)    # [-1, 1] signal strength
    confidence: float = Field(ge=0.0, le=1.0)    # [0, 1] model confidence
    horizon: int = Field(ge=1)                   # Bars until expiry
    model_source: str                            # Which model generated
    features_snapshot: dict[str, float] = {}     # Input features

    def is_actionable(self, min_strength: float = 0.5,
                      min_confidence: float = 0.6) -> bool:
        """Check if signal meets thresholds for trading."""
```

#### Order
```python
class Order(BaseModel):
    """Order representation throughout lifecycle."""
    order_id: UUID = Field(default_factory=uuid4)
    broker_order_id: str | None = None
    symbol: str
    side: OrderSide
    quantity: Decimal = Field(gt=0)
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: Decimal = Decimal("0")
    filled_avg_price: Decimal | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    submitted_at: datetime | None = None
    filled_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        """Order is still working (not terminal)."""

    @property
    def is_terminal(self) -> bool:
        """Order has reached final state."""

    @property
    def remaining_qty(self) -> Decimal:
        """Quantity still to be filled."""
```

#### Position
```python
class Position(BaseModel):
    """Current position in a symbol."""
    symbol: str
    quantity: Decimal          # Positive=long, negative=short
    avg_entry_price: Decimal
    current_price: Decimal
    cost_basis: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Decimal("0")
    opened_at: datetime
    updated_at: datetime

    @property
    def is_long(self) -> bool

    @property
    def is_short(self) -> bool

    @property
    def is_flat(self) -> bool

    @property
    def unrealized_pnl_pct(self) -> Decimal

    def update_price(self, new_price: Decimal) -> "Position":
        """Return new Position with updated price."""
```

#### Portfolio
```python
class Portfolio(BaseModel):
    """Complete portfolio state snapshot."""
    timestamp: datetime
    equity: Decimal              # Total portfolio value
    cash: Decimal                # Available cash
    buying_power: Decimal        # Available for new positions
    positions: dict[str, Position] = {}
    pending_orders: list[UUID] = []
    daily_pnl: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")

    @property
    def position_count(self) -> int

    @property
    def total_market_value(self) -> Decimal

    @property
    def long_exposure(self) -> Decimal

    @property
    def short_exposure(self) -> Decimal

    @property
    def net_exposure(self) -> Decimal

    @property
    def gross_exposure(self) -> Decimal

    def get_position(self, symbol: str) -> Position | None
```

#### RiskMetrics
```python
class RiskMetrics(BaseModel):
    """Portfolio risk metrics snapshot."""
    timestamp: datetime
    var_95: Decimal              # 95% Value at Risk
    var_99: Decimal              # 99% Value at Risk
    cvar_95: Decimal             # Conditional VaR (Expected Shortfall)
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: Decimal
    current_drawdown: Decimal
    volatility_annual: float
    beta: float | None = None
    sector_exposures: dict[str, Decimal] = {}

    def is_within_limits(self, config: RiskSettings) -> bool:
        """Check if metrics are within configured limits."""
```

#### FeatureVector
```python
class FeatureVector(BaseModel):
    """Feature vector for model input."""
    symbol: str
    timestamp: datetime
    features: dict[str, float]
    feature_names: list[str]

    @property
    def values(self) -> list[float]

    def to_numpy(self) -> np.ndarray
```

#### ModelPrediction
```python
class ModelPrediction(BaseModel):
    """Model prediction output."""
    model_name: str
    symbol: str
    timestamp: datetime
    prediction: float           # Raw prediction value
    probability: float | None   # Classification probability
    direction: Direction
    confidence: float
    horizon: int
    feature_importance: dict[str, float] = {}
```

---

## 4. Event System

Located in `quant_trading_system/core/events.py`. Implements a thread-safe, priority-based event bus for inter-component communication.

### Event Types

```python
class EventType(str, Enum):
    # Market Data Events
    BAR_UPDATE = "bar_update"
    QUOTE_UPDATE = "quote_update"
    TRADE_UPDATE = "trade_update"

    # Signal Events
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_EXPIRED = "signal_expired"
    SIGNAL_CANCELLED = "signal_cancelled"

    # Order Events
    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACCEPTED = "order_accepted"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIAL = "order_partial"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_EXPIRED = "order_expired"

    # Risk Events
    LIMIT_BREACH = "limit_breach"
    DRAWDOWN_WARNING = "drawdown_warning"
    EXPOSURE_WARNING = "exposure_warning"
    KILL_SWITCH_TRIGGERED = "kill_switch_triggered"
    KILL_SWITCH_RESET = "kill_switch_reset"

    # Portfolio Events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"
    PORTFOLIO_REBALANCED = "portfolio_rebalanced"

    # System Events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SYSTEM_ERROR = "system_error"
    MODEL_RELOAD = "model_reload"
    CONFIG_RELOAD = "config_reload"
    HEARTBEAT = "heartbeat"
```

### EventPriority

```python
class EventPriority(IntEnum):
    CRITICAL = 0    # Kill switch, system failures
    HIGH = 1        # Order events, risk alerts
    NORMAL = 2      # Data updates, signals
    LOW = 3         # Logging, metrics
```

### Event Class

```python
class Event(BaseModel):
    """Base event for all system events."""
    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    priority: EventPriority = EventPriority.NORMAL
    source: str                    # Component that emitted event
    data: dict = {}                # Event payload
    correlation_id: UUID | None    # For request tracing
```

### EventBus (Singleton)

```python
class EventBus:
    """Thread-safe singleton event bus with priority processing."""
    _instance: Optional["EventBus"] = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "EventBus":
        """Double-check locking for thread-safe singleton."""

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], None | Coroutine],
        handler_name: str,
        priority: EventPriority = EventPriority.NORMAL
    ) -> None:
        """Subscribe handler to specific event type."""

    def subscribe_all(
        self,
        handler: Callable[[Event], None | Coroutine],
        handler_name: str,
        priority: EventPriority = EventPriority.LOW
    ) -> None:
        """Subscribe to all events (logging, audit)."""

    def unsubscribe(self, event_type: EventType, handler_name: str) -> bool:
        """Remove handler subscription."""

    def publish(self, event: Event) -> None:
        """Synchronous event publication."""

    async def publish_async(self, event: Event) -> None:
        """Asynchronous event publication with coroutine support."""

    def enqueue(self, event: Event) -> None:
        """Add event to priority queue for background processing."""

    def get_metrics(self) -> dict:
        """Get event bus performance metrics."""

    def get_event_history(
        self,
        event_type: EventType | None = None,
        limit: int = 100
    ) -> list[Event]:
        """Query event history."""

    def get_dead_letter_queue(self) -> list[tuple[Event, Exception]]:
        """Get failed events with their exceptions."""
```

### Usage Example

```python
from quant_trading_system.core.events import EventBus, Event, EventType, EventPriority

# Get singleton instance
event_bus = EventBus.get_instance()

# Subscribe to order events
def on_order_filled(event: Event):
    order = event.data["order"]
    logger.info(f"Order filled: {order.symbol} @ {order.filled_avg_price}")

event_bus.subscribe(
    EventType.ORDER_FILLED,
    on_order_filled,
    handler_name="portfolio_updater",
    priority=EventPriority.HIGH
)

# Publish an event
event_bus.publish(Event(
    event_type=EventType.ORDER_FILLED,
    source="order_manager",
    priority=EventPriority.HIGH,
    data={"order": filled_order}
))
```

---

## 5. Configuration

Located in `quant_trading_system/config/settings.py`. Uses Pydantic BaseSettings for type-safe configuration with environment variable support.

### Settings Classes

```python
class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    name: str = "alphatrade"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20

    @property
    def url(self) -> str:
        """PostgreSQL connection URL."""

class RedisSettings(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None

class AlpacaSettings(BaseModel):
    api_key: str
    api_secret: str
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"
    paper_trading: bool = True

    @classmethod
    def from_env(cls) -> "AlpacaSettings":
        """Load from environment variables."""

class RiskSettings(BaseModel):
    # Position Limits
    max_position_pct: float = 0.10        # Max 10% in single position
    max_positions: int = 20                # Max open positions
    max_sector_exposure: float = 0.30      # Max 30% in any sector
    max_correlation: float = 0.80          # Reject highly correlated adds

    # Loss Limits (trigger kill switch)
    max_daily_loss_pct: float = 0.05       # 5% daily loss
    max_weekly_loss_pct: float = 0.10      # 10% weekly loss
    max_monthly_loss_pct: float = 0.15     # 15% monthly loss
    max_drawdown_pct: float = 0.15         # 15% from peak

    # Volatility Limits
    max_portfolio_volatility: float = 0.25 # Annualized
    max_vix_threshold: float = 35.0        # Reduce exposure above this

    # Position Sizing
    kelly_fraction: float = 0.25           # Fraction of full Kelly
    volatility_target: float = 0.15        # Target portfolio vol

class TradingSettings(BaseModel):
    bar_timeframe: str = "15Min"           # Data frequency
    lookback_bars: int = 100               # Historical bars for features
    signal_threshold: float = 0.5          # Min signal strength
    min_confidence: float = 0.6            # Min model confidence
    slippage_bps: float = 5.0              # Slippage assumption
    commission_bps: float = 0.0            # Alpaca is zero commission
    market_open: time = time(9, 30)        # ET
    market_close: time = time(16, 0)       # ET
    extended_hours: bool = False

class ModelSettings(BaseModel):
    models_dir: Path = Path("models")
    default_model: str = "xgboost_v1"
    ensemble_models: list[str] = ["xgboost_v1", "lightgbm_v1", "lstm_v1"]
    retrain_frequency: str = "weekly"
    min_training_samples: int = 10000
    validation_split: float = 0.2
    walk_forward_windows: int = 5

class LoggingSettings(BaseModel):
    level: str = "INFO"
    format: str = "json"                   # json or text
    file_path: Path | None = None
    rotation: str = "1 day"
    retention: str = "30 days"

class AlertSettings(BaseModel):
    slack_webhook: str | None = None
    slack_channels: dict[str, str] = {}    # severity -> channel
    email_smtp_host: str | None = None
    email_smtp_port: int = 587
    email_from: str | None = None
    email_to: list[str] = []
    pagerduty_key: str | None = None
    twilio_sid: str | None = None
    twilio_token: str | None = None
    twilio_from: str | None = None
    twilio_to: list[str] = []

    @property
    def is_slack_configured(self) -> bool

    @property
    def is_email_configured(self) -> bool

    @property
    def is_pagerduty_configured(self) -> bool
```

### Main Settings

```python
class Settings(BaseSettings):
    """Root settings with nested configuration."""
    trading: TradingSettings = TradingSettings()
    risk: RiskSettings = RiskSettings()
    alpaca: AlpacaSettings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    models: ModelSettings = ModelSettings()
    logging: LoggingSettings = LoggingSettings()
    alerts: AlertSettings = AlertSettings()

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings singleton."""
```

### Environment Variables

```bash
# .env file
ALPACA_API_KEY=your_api_key
ALPACA_API_SECRET=your_api_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

DATABASE__HOST=localhost
DATABASE__PORT=5432
DATABASE__NAME=alphatrade
DATABASE__USER=postgres
DATABASE__PASSWORD=secret

REDIS__HOST=localhost
REDIS__PORT=6379

LOGGING__LEVEL=INFO
```

---

## 6. Models & Machine Learning

Located in `quant_trading_system/models/`.

### TradingModel ABC (`base.py`)

```python
class ModelType(str, Enum):
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"

class TradingModel(ABC):
    """Abstract base class for all trading models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for registry."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Model version string."""

    @property
    def model_type(self) -> ModelType:
        """Classification or regression."""

    @property
    def is_fitted(self) -> bool:
        """Whether model has been trained."""

    @property
    def feature_names(self) -> list[str]:
        """Feature names used in training."""

    @property
    def training_timestamp(self) -> datetime | None:
        """When model was last trained."""

    @property
    def training_metrics(self) -> dict:
        """Training/validation metrics."""

    @property
    def device(self) -> str:
        """Compute device (cuda, mps, cpu)."""

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        **kwargs
    ) -> "TradingModel":
        """Train the model. Returns self for chaining."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability predictions for classifiers."""

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Feature importance scores."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "TradingModel":
        """Load model from disk."""
```

### Classical ML (`classical_ml.py`)

| Model | Class | Key Features |
|-------|-------|--------------|
| XGBoost | `XGBoostModel` | GPU support, early stopping, sample weighting |
| LightGBM | `LightGBMModel` | Large datasets, categorical features |
| CatBoost | `CatBoostModel` | Built-in categorical handling |
| Random Forest | `RandomForestModel` | Baseline, robust feature importance |
| ElasticNet | `ElasticNetModel` | Linear, L1/L2 regularization |

```python
class XGBoostModel(TradingModel):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        use_gpu: bool = True,
        early_stopping_rounds: int = 50,
        **kwargs
    ): ...
```

### Deep Learning (`deep_learning.py`)

| Model | Class | Use Case |
|-------|-------|----------|
| LSTM | `LSTMModel` | Sequential patterns, regime detection |
| Transformer | `TransformerModel` | Long-range dependencies |
| GRU | `GRUModel` | Simpler sequential model |
| TCN | `TCNModel` | Causal convolutions |

```python
class LSTMModel(TradingModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        sequence_length: int = 20,
        use_mixed_precision: bool = True,
        **kwargs
    ): ...

class TransformerModel(TradingModel):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        sequence_length: int = 20,
        **kwargs
    ): ...
```

### Ensemble (`ensemble.py`)

```python
class VotingEnsemble(TradingModel):
    """Weighted voting across multiple models."""

    def __init__(
        self,
        models: list[TradingModel],
        weights: list[float] | None = None,  # Equal if None
        voting: str = "soft"                  # soft or hard
    ): ...

class StackingEnsemble(TradingModel):
    """Meta-learner on top of base models."""

    def __init__(
        self,
        base_models: list[TradingModel],
        meta_learner: TradingModel,
        use_probas: bool = True
    ): ...

class AdaptiveEnsemble(TradingModel):
    """Regime-aware model weighting."""

    def __init__(
        self,
        models: list[TradingModel],
        regime_detector: RegimeDetector,
        regime_weights: dict[MarketRegime, list[float]]
    ): ...
```

### Reinforcement Learning (`reinforcement.py`)

```python
class PPOAgent(TradingModel):
    """Proximal Policy Optimization for trading."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,  # Buy, sell, hold
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        **kwargs
    ): ...
```

### Model Validation Gates (`validation_gates.py`)

Pre-deployment validation to prevent overfitting and ensure model quality:

```python
class ValidationGateType(str, Enum):
    PERFORMANCE = "performance"
    OVERFITTING = "overfitting"
    STABILITY = "stability"
    STATISTICAL = "statistical"
    DATA_QUALITY = "data_quality"
    ECONOMIC = "economic"

class ModelValidationGates:
    """JPMorgan-level model validation before deployment."""

    def __init__(
        self,
        min_sharpe_ratio: float = 0.5,
        max_drawdown: float = 0.25,
        max_is_oos_ratio: float = 2.0,     # In-sample/Out-of-sample performance ratio
        min_trades: int = 100,
        min_win_rate: float = 0.4,
        max_correlation_to_benchmark: float = 0.95,
        min_ic: float = 0.02,              # Information coefficient
        significance_level: float = 0.05
    ): ...

    def validate(
        self,
        model_name: str,
        holdout_metrics: dict,
        in_sample_metrics: dict,
        predictions: np.ndarray | None = None,
        returns: np.ndarray | None = None
    ) -> ValidationReport:
        """Run all validation gates and return report."""

class ValidationReport(BaseModel):
    model_name: str
    timestamp: datetime
    passed: bool
    results: list[GateResult]
    summary: str
```

### Model Explainability (`explainability.py`)

SHAP and LIME integration for regulatory compliance:

```python
class SHAPExplainer:
    """SHAP-based model explanations."""

    def explain_local(
        self,
        model: TradingModel,
        X: np.ndarray,
        feature_names: list[str],
        index: int = 0
    ) -> LocalExplanation: ...

    def explain_global(
        self,
        model: TradingModel,
        X: np.ndarray,
        feature_names: list[str]
    ) -> GlobalExplanation: ...

class ModelExplainabilityService:
    """Service for generating compliance-ready explanations."""

    def explain_prediction(
        self,
        model: TradingModel,
        X: np.ndarray,
        feature_names: list[str],
        index: int = 0
    ) -> LocalExplanation: ...

    def explain_model(
        self,
        model: TradingModel,
        X: np.ndarray,
        feature_names: list[str]
    ) -> GlobalExplanation: ...

    def generate_compliance_report(
        self,
        model: TradingModel,
        model_name: str,
        X: np.ndarray,
        feature_names: list[str]
    ) -> ComplianceReport:
        """Generate full model explanation report for regulators."""
```

---

## 7. Alpha Factors

Located in `quant_trading_system/alpha/`.

### AlphaFactor ABC (`alpha_base.py`)

```python
class AlphaType(str, Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VALUE = "value"
    QUALITY = "quality"
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"
    ML_BASED = "ml_based"
    COMPOSITE = "composite"

class AlphaHorizon(str, Enum):
    SHORT = "short"      # 1-4 bars
    MEDIUM = "medium"    # 4-20 bars
    LONG = "long"        # 20+ bars

class AlphaSignal(BaseModel):
    """Alpha factor output signal."""
    alpha_id: UUID = Field(default_factory=uuid4)
    alpha_name: str
    symbol: str
    timestamp: datetime
    value: float = Field(ge=-1.0, le=1.0)     # Normalized [-1, 1]
    confidence: float = Field(ge=0.0, le=1.0)
    horizon: int
    metadata: dict = {}

    @property
    def direction(self) -> Direction:
        """LONG if value > 0.1, SHORT if < -0.1, else FLAT."""

    @property
    def strength(self) -> str:
        """STRONG/MODERATE/WEAK based on absolute value."""

    def is_actionable(
        self,
        min_value: float = 0.3,
        min_confidence: float = 0.5
    ) -> bool: ...

class AlphaFactor(ABC):
    """Abstract base class for all alpha factors."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def alpha_type(self) -> AlphaType: ...

    @property
    def horizon(self) -> AlphaHorizon:
        return AlphaHorizon.MEDIUM

    @abstractmethod
    def compute(
        self,
        df: pl.DataFrame | pd.DataFrame,
        features: dict[str, Any] | None = None
    ) -> np.ndarray:
        """Compute alpha values. Returns array of shape (n_symbols,)."""

    @abstractmethod
    def get_params(self) -> dict:
        """Get factor parameters."""
```

### Momentum Alphas (`momentum_alphas.py`)

```python
class PriceMomentumAlpha(AlphaFactor):
    """Rate of change momentum."""

    def __init__(self, lookback: int = 20, normalize: bool = True): ...

class RSIMomentumAlpha(AlphaFactor):
    """RSI-based momentum (overbought/oversold)."""

    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30): ...

class MACDMomentumAlpha(AlphaFactor):
    """MACD histogram momentum."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9): ...

class RelativeStrengthAlpha(AlphaFactor):
    """Cross-sectional relative strength."""

    def __init__(self, lookback: int = 20, top_pct: float = 0.2): ...
```

### Mean Reversion Alphas (`mean_reversion_alphas.py`)

```python
class BollingerReversionAlpha(AlphaFactor):
    """Bollinger band mean reversion."""

    def __init__(self, period: int = 20, num_std: float = 2.0): ...

class ZScoreReversionAlpha(AlphaFactor):
    """Z-score based mean reversion."""

    def __init__(self, lookback: int = 20, entry_zscore: float = 2.0): ...
```

### Alpha Combiner (`alpha_combiner.py`)

```python
class CombinationMethod(str, Enum):
    EQUAL = "equal"
    IC_WEIGHTED = "ic_weighted"
    SHARPE_WEIGHTED = "sharpe_weighted"
    INVERSE_VOLATILITY = "inverse_volatility"
    OPTIMIZED = "optimized"        # scipy optimization
    RANK_WEIGHTED = "rank_weighted"
    DECAY_WEIGHTED = "decay_weighted"

class NeutralizationMethod(str, Enum):
    MARKET = "market"
    SECTOR = "sector"
    FACTOR = "factor"
    BETA = "beta"
    NONE = "none"

class OrthogonalizationMethod(str, Enum):
    PCA = "pca"
    GRAM_SCHMIDT = "gram_schmidt"
    DECORRELATE = "decorrelate"
    NONE = "none"

class AlphaCombiner:
    """Combine multiple alpha factors into final signal."""

    def __init__(
        self,
        alphas: list[AlphaFactor],
        method: CombinationMethod = CombinationMethod.IC_WEIGHTED,
        neutralization: NeutralizationMethod = NeutralizationMethod.MARKET,
        orthogonalization: OrthogonalizationMethod = OrthogonalizationMethod.NONE,
        lookback_for_weights: int = 60
    ): ...

    def combine(
        self,
        df: pl.DataFrame,
        forward_returns: np.ndarray | None = None
    ) -> np.ndarray:
        """Combine alphas into single signal array."""
```

### Alpha Metrics (`alpha_metrics.py`)

```python
class AlphaMetricsCalculator:
    """Calculate IC, IR, and other alpha quality metrics."""

    def compute_ic(
        self,
        alpha: pd.Series,
        forward_returns: pd.Series,
        method: str = "spearman"  # or "pearson"
    ) -> ICResult:
        """Information Coefficient: rank correlation with future returns."""

    def compute_ir(
        self,
        alpha: pd.Series,
        forward_returns: pd.Series
    ) -> IRResult:
        """Information Ratio: IC / std(IC)."""

    def compute_ic_decay(
        self,
        alpha: pd.Series,
        returns_df: pd.DataFrame,
        max_horizon: int = 20
    ) -> pd.DataFrame:
        """IC decay across multiple horizons."""

    def compute_turnover(
        self,
        alpha_t: pd.Series,
        alpha_t_minus_1: pd.Series
    ) -> float:
        """Alpha turnover (trading cost proxy)."""

    def generate_report(
        self,
        alpha: pd.Series,
        forward_returns: pd.Series
    ) -> AlphaReport:
        """Comprehensive alpha quality report."""
```

### Regime Detection (`regime_detection.py`)

```python
class MarketRegime(str, Enum):
    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    RANGE_BOUND = "range_bound"
    CRISIS = "crisis"

class RegimeState(BaseModel):
    regime: MarketRegime
    confidence: float
    trend_strength: float
    volatility_percentile: float
    detected_at: datetime

class CompositeRegimeDetector:
    """Multi-factor regime detection."""

    def __init__(
        self,
        use_volatility: bool = True,
        use_trend: bool = True,
        use_correlation: bool = True,
        vol_lookback: int = 20,
        trend_lookback: int = 50,
        vol_threshold_low: float = 0.3,
        vol_threshold_high: float = 0.7
    ): ...

    def detect(self, price_data: pd.DataFrame) -> RegimeState:
        """Detect current market regime."""

class RegimeAdaptiveController:
    """Adapt trading parameters based on regime."""

    def __init__(self, detector: CompositeRegimeDetector): ...

    def adapt_position_size(
        self,
        base_size: Decimal,
        regime: MarketRegime
    ) -> Decimal:
        """Scale position size by regime."""

    def adapt_model_weights(
        self,
        base_weights: dict[str, float],
        regime: MarketRegime
    ) -> dict[str, float]:
        """Adjust model ensemble weights."""
```

---

## 8. Feature Engineering

Located in `quant_trading_system/features/`.

### Technical Features (`technical.py`)

```python
class TechnicalFeatureCalculator:
    """Calculate technical indicators."""

    # Trend Indicators
    def sma(self, close: pd.Series, period: int) -> pd.Series
    def ema(self, close: pd.Series, period: int) -> pd.Series
    def dema(self, close: pd.Series, period: int) -> pd.Series
    def tema(self, close: pd.Series, period: int) -> pd.Series
    def wma(self, close: pd.Series, period: int) -> pd.Series
    def keltner_channel(self, high, low, close, period: int, atr_mult: float)
    def ichimoku(self, high, low, close)

    # Momentum Indicators
    def rsi(self, close: pd.Series, period: int = 14) -> pd.Series
    def macd(self, close, fast: int = 12, slow: int = 26, signal: int = 9)
    def stochastic(self, high, low, close, k_period: int = 14, d_period: int = 3)
    def cci(self, high, low, close, period: int = 20) -> pd.Series
    def williams_r(self, high, low, close, period: int = 14) -> pd.Series
    def awesome_oscillator(self, high, low) -> pd.Series
    def kdj(self, high, low, close, n: int = 9, m1: int = 3, m2: int = 3)

    # Volatility Indicators
    def bollinger_bands(self, close, period: int = 20, num_std: float = 2.0)
    def atr(self, high, low, close, period: int = 14) -> pd.Series
    def natr(self, high, low, close, period: int = 14) -> pd.Series
    def historical_volatility(self, close, period: int = 20) -> pd.Series
    def parabolic_sar(self, high, low, af_start: float = 0.02, af_max: float = 0.2)

    # Volume Indicators
    def obv(self, close, volume) -> pd.Series
    def mfi(self, high, low, close, volume, period: int = 14) -> pd.Series
    def cmf(self, high, low, close, volume, period: int = 20) -> pd.Series
    def ad_line(self, high, low, close, volume) -> pd.Series
    def vwap(self, high, low, close, volume) -> pd.Series

    # Pattern Recognition (40+ patterns)
    def doji(self, open, high, low, close) -> pd.Series
    def hammer(self, open, high, low, close) -> pd.Series
    def engulfing(self, open, high, low, close) -> pd.Series
    # ... etc
```

### Statistical Features (`statistical.py`)

```python
class StatisticalFeatureCalculator:
    """Statistical and time-series features."""

    def rolling_mean(self, series, window: int) -> pd.Series
    def rolling_std(self, series, window: int) -> pd.Series
    def rolling_skew(self, series, window: int) -> pd.Series
    def rolling_kurtosis(self, series, window: int) -> pd.Series
    def rolling_zscore(self, series, window: int) -> pd.Series

    def autocorrelation(self, series, lag: int) -> pd.Series
    def partial_autocorrelation(self, series, lag: int) -> pd.Series

    def hurst_exponent(self, series, max_lag: int = 20) -> float
    def adf_test(self, series) -> ADFResult
    def kpss_test(self, series) -> KPSSResult

    def rolling_beta(self, returns, market_returns, window: int) -> pd.Series
    def rolling_correlation(self, series1, series2, window: int) -> pd.Series
```

### Microstructure Features (`microstructure.py`)

```python
class MicrostructureFeatureCalculator:
    """Market microstructure features."""

    def vpin(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        bucket_size: int = 50
    ) -> pd.Series:
        """Volume-synchronized Probability of Informed Trading."""

    def order_flow_imbalance(
        self,
        buy_volume: pd.Series,
        sell_volume: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """Buy vs sell volume imbalance."""

    def kyle_lambda(
        self,
        returns: pd.Series,
        volumes: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """Kyle's Lambda (price impact coefficient)."""

    def amihud_illiquidity(
        self,
        returns: pd.Series,
        volumes: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """Amihud illiquidity measure."""

    def realized_spread(
        self,
        prices: pd.Series,
        midpoints: pd.Series
    ) -> pd.Series:
        """Realized bid-ask spread."""
```

### Feature Pipeline (`feature_pipeline.py`)

```python
class FeatureGroup(str, Enum):
    TECHNICAL = "technical"
    STATISTICAL = "statistical"
    MICROSTRUCTURE = "microstructure"
    CROSS_SECTIONAL = "cross_sectional"
    ALL = "all"

class NormalizationMethod(str, Enum):
    NONE = "none"
    ZSCORE = "zscore"
    MINMAX = "minmax"
    ROBUST = "robust"
    QUANTILE = "quantile"

class FeatureConfig(BaseModel):
    groups: list[FeatureGroup] = [FeatureGroup.ALL]
    normalization: NormalizationMethod = NormalizationMethod.ZSCORE
    fill_nan: bool = True
    fill_method: str = "ffill"
    max_nan_ratio: float = 0.3
    variance_threshold: float = 0.0
    correlation_threshold: float = 0.95
    include_targets: bool = False
    target_horizons: list[int] = [1, 5, 10, 20]

class FeatureMetadata(BaseModel):
    name: str
    group: FeatureGroup
    computed_at: datetime
    num_samples: int
    nan_ratio: float
    mean: float
    std: float
    min_val: float
    max_val: float
    params: dict = {}

class FeaturePipeline:
    """Orchestrate feature computation."""

    def __init__(self, config: FeatureConfig): ...

    def compute(
        self,
        df: pd.DataFrame,
        symbols: list[str] | None = None
    ) -> tuple[pd.DataFrame, list[FeatureMetadata]]:
        """Compute all features for dataframe."""

    def fit_transform(
        self,
        df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Fit normalizers and transform."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform with fitted normalizers."""
```

---

## 9. Trading Engine

Located in `quant_trading_system/trading/trading_engine.py`.

### Engine States

```python
class EngineState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    PRE_MARKET = "pre_market"
    MARKET_HOURS = "market_hours"
    POST_MARKET = "post_market"
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
```

### Configuration

```python
class TradingEngineConfig(BaseModel):
    mode: TradingMode           # LIVE, PAPER, DRY_RUN
    symbols: list[str]          # Trading universe
    bar_interval: str = "15Min" # 1Min, 5Min, 15Min, 1Hour
    trading_start: time = time(9, 30)
    trading_end: time = time(16, 0)
    pre_market_minutes: int = 30
    post_market_minutes: int = 0
    max_daily_trades: int = 100
    max_daily_loss_pct: float = 0.05
    kill_switch_drawdown: float = 0.15
    watchdog_timeout: int = 60
    enable_reconciliation: bool = True
    heartbeat_interval: int = 30
    # Risk checks are MANDATORY (cannot be disabled)
```

### Session Tracking

```python
class TradingSession(BaseModel):
    """Daily trading session metrics."""
    date: date
    start_time: datetime | None = None
    end_time: datetime | None = None
    starting_equity: Decimal
    ending_equity: Decimal | None = None
    daily_pnl: Decimal = Decimal("0")
    daily_pnl_pct: float = 0.0
    trade_count: int = 0
    signal_count: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    errors: int = 0

class EngineMetrics(BaseModel):
    """Engine performance metrics."""
    uptime_seconds: float
    bars_processed: int
    signals_generated: int
    orders_submitted: int
    orders_filled: int
    orders_rejected: int
    total_pnl: Decimal
    max_drawdown: Decimal
    current_drawdown: Decimal
    peak_equity: Decimal
    avg_latency_ms: float
    error_count: int
    last_heartbeat: datetime
```

### TradingEngine

```python
class TradingEngine:
    """Main trading orchestrator."""

    def __init__(
        self,
        config: TradingEngineConfig,
        alpaca_client: AlpacaClient,
        model_manager: ModelManager,
        risk_limits: RiskLimits,
        event_bus: EventBus | None = None
    ): ...

    @property
    def state(self) -> EngineState: ...

    @property
    def is_trading(self) -> bool: ...

    async def start(self) -> None:
        """Start trading engine."""

    async def stop(self, reason: str = "shutdown") -> None:
        """Stop trading gracefully."""

    async def pause(self, reason: str) -> None:
        """Pause trading (orders rejected, positions held)."""

    async def resume(self) -> None:
        """Resume from paused state."""

    async def process_bar(self, bar: OHLCVBar) -> None:
        """Process incoming bar data."""

    async def process_signal(self, signal: TradeSignal) -> None:
        """Process trading signal, apply risk checks, submit order."""

    async def reconcile(self) -> ReconciliationReport:
        """Reconcile local state with broker."""

    def get_metrics(self) -> EngineMetrics: ...

    def get_session(self) -> TradingSession: ...
```

---

## 10. Execution System

Located in `quant_trading_system/execution/`.

### Order Manager (`order_manager.py`)

```python
class OrderRequest(BaseModel):
    """Pre-submission order request."""
    request_id: UUID = Field(default_factory=uuid4)
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    signal_id: UUID | None = None
    strategy_id: str | None = None
    priority: int = 0
    extended_hours: bool = False
    take_profit_price: Decimal | None = None
    stop_loss_price: Decimal | None = None
    notes: str | None = None
    metadata: dict = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)

class OrderManager:
    """Manage order lifecycle."""

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        risk_checker: PreTradeRiskChecker,
        kill_switch: KillSwitch,
        event_bus: EventBus
    ): ...

    async def submit_order(self, request: OrderRequest) -> Order:
        """Validate and submit order to broker."""

    async def cancel_order(self, order_id: UUID) -> bool:
        """Cancel pending order."""

    async def modify_order(
        self,
        order_id: UUID,
        **updates
    ) -> Order:
        """Modify pending order (quantity, price)."""

    def get_order(self, order_id: UUID) -> Order | None:
        """Get order by ID."""

    def get_order_status(self, order_id: UUID) -> OrderStatus | None:
        """Get order status."""

    def get_orders_by_symbol(self, symbol: str) -> list[Order]:
        """Get all orders for symbol."""

    def get_active_orders(self) -> list[Order]:
        """Get all non-terminal orders."""

    def get_order_history(
        self,
        symbol: str | None = None,
        status: OrderStatus | None = None,
        since: datetime | None = None
    ) -> list[Order]:
        """Query order history."""
```

### Alpaca Client (`alpaca_client.py`)

```python
class AlpacaClient:
    """Alpaca API wrapper with rate limiting and retry logic."""

    def __init__(
        self,
        settings: AlpacaSettings,
        rate_limit: int = 200  # requests per minute
    ): ...

    # Account
    async def get_account(self) -> AccountInfo: ...
    async def get_buying_power(self) -> Decimal: ...

    # Orders
    async def submit_order(self, order: Order) -> Order: ...
    async def cancel_order(self, order_id: str) -> bool: ...
    async def get_order(self, order_id: str) -> Order: ...
    async def list_orders(self, status: str | None = None) -> list[Order]: ...

    # Positions
    async def get_positions(self) -> list[Position]: ...
    async def get_position(self, symbol: str) -> Position | None: ...
    async def close_position(self, symbol: str) -> Order: ...
    async def close_all_positions(self) -> list[Order]: ...

    # Market Data
    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> list[OHLCVBar]: ...

    async def get_latest_bar(self, symbol: str) -> OHLCVBar: ...
    async def get_latest_quote(self, symbol: str) -> Quote: ...

    # Streaming
    async def subscribe_trades(
        self,
        symbols: list[str],
        handler: Callable[[TradeUpdate], None]
    ) -> None: ...

    async def subscribe_bars(
        self,
        symbols: list[str],
        handler: Callable[[OHLCVBar], None]
    ) -> None: ...
```

### Execution Algorithms (`execution_algo.py`)

```python
class ExecutionAlgorithm(ABC):
    """Base class for execution algorithms."""

    @abstractmethod
    async def execute(
        self,
        symbol: str,
        side: OrderSide,
        total_quantity: Decimal,
        **params
    ) -> list[Order]: ...

class TWAPAlgorithm(ExecutionAlgorithm):
    """Time-Weighted Average Price."""

    def __init__(
        self,
        num_slices: int = 10,
        interval_seconds: int = 60,
        randomize: bool = True
    ): ...

class VWAPAlgorithm(ExecutionAlgorithm):
    """Volume-Weighted Average Price."""

    def __init__(
        self,
        volume_profile: dict[int, float],  # hour -> weight
        num_slices: int = 10
    ): ...

class ImplementationShortfallAlgorithm(ExecutionAlgorithm):
    """Minimize execution cost vs arrival price."""

    def __init__(
        self,
        urgency: float = 0.5,      # 0=passive, 1=aggressive
        risk_aversion: float = 1.0,
        alpha_decay: float = 0.1   # Signal decay rate
    ): ...
```

---

## 11. Risk Management

Located in `quant_trading_system/risk/`.

### Kill Switch (`limits.py`)

```python
class KillSwitchReason(str, Enum):
    MAX_DRAWDOWN = "max_drawdown"
    DAILY_LOSS = "daily_loss"
    RAPID_PNL_DECLINE = "rapid_pnl_decline"
    SYSTEM_ERROR = "system_error"
    DATA_FEED_FAILURE = "data_feed_failure"
    BROKER_CONNECTION_LOSS = "broker_connection_loss"
    MANUAL_ACTIVATION = "manual_activation"
    LIMIT_BREACH = "limit_breach"

class KillSwitchState(BaseModel):
    is_active: bool = False
    activated_at: datetime | None = None
    reason: KillSwitchReason | None = None
    trigger_value: float | None = None
    orders_cancelled: int = 0
    positions_closed: int = 0
    activated_by: str | None = None

class KillSwitch:
    """Emergency trading halt mechanism."""

    def __init__(self, event_bus: EventBus | None = None): ...

    def activate(
        self,
        reason: KillSwitchReason | str,
        trigger_value: float | None = None,
        activated_by: str = "system"
    ) -> None:
        """Activate kill switch - HALTS ALL TRADING."""

    def is_active(self) -> bool:
        """Check if kill switch is active."""

    def get_state(self) -> KillSwitchState:
        """Get current kill switch state."""

    def reset(self, authorized_by: str) -> None:
        """Reset kill switch (requires authorization)."""
```

### Risk Limits (`limits.py`)

```python
class CheckResult(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"

class RiskCheckResult(BaseModel):
    check_name: str
    result: CheckResult
    message: str
    details: dict = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class RiskLimitsConfig(BaseModel):
    # Loss Limits
    max_daily_loss_pct: float = 0.05
    max_weekly_loss_pct: float = 0.10
    max_monthly_loss_pct: float = 0.15
    max_per_trade_loss_pct: float = 0.02

    # Position Limits
    max_position_value: Decimal = Decimal("100000")
    max_position_pct: float = 0.10
    max_total_positions: int = 20

    # Exposure Limits
    max_sector_exposure_pct: float = 0.30
    max_correlation: float = 0.80

    # Volatility Limits
    max_portfolio_volatility: float = 0.25
    vix_warning_threshold: float = 25.0
    vix_halt_threshold: float = 35.0

    # Drawdown Thresholds
    drawdown_warning_pct: float = 0.05
    drawdown_reduce_pct: float = 0.10
    drawdown_halt_pct: float = 0.15

    # Trading Controls
    max_daily_trades: int = 100
    max_turnover_pct: float = 0.50

class PreTradeRiskChecker:
    """Pre-trade risk validation."""

    def __init__(
        self,
        config: RiskLimitsConfig,
        portfolio: Portfolio,
        kill_switch: KillSwitch
    ): ...

    def check_order(self, order: OrderRequest) -> RiskCheckResult:
        """Run all pre-trade risk checks."""

    def check_buying_power(self, order: OrderRequest) -> RiskCheckResult: ...
    def check_position_limit(self, order: OrderRequest) -> RiskCheckResult: ...
    def check_concentration(self, order: OrderRequest) -> RiskCheckResult: ...
    def check_sector_exposure(self, order: OrderRequest) -> RiskCheckResult: ...
    def check_correlation(self, order: OrderRequest) -> RiskCheckResult: ...
    def check_blacklist(self, order: OrderRequest) -> RiskCheckResult: ...
```

### Position Sizing (`position_sizer.py`)

```python
class PositionSizingMethod(str, Enum):
    FIXED_DOLLAR = "fixed_dollar"
    FIXED_SHARES = "fixed_shares"
    PERCENT_EQUITY = "percent_equity"
    VOLATILITY_SCALED = "volatility_scaled"
    RISK_PARITY = "risk_parity"
    KELLY = "kelly"

class PositionSizerConfig(BaseModel):
    method: PositionSizingMethod = PositionSizingMethod.VOLATILITY_SCALED
    fixed_dollar_amount: Decimal = Decimal("10000")
    fixed_share_count: int = 100
    percent_of_equity: float = 0.05
    max_position_pct: float = 0.10
    max_total_positions: int = 20
    volatility_target: float = 0.15
    kelly_fraction: float = 0.25
    min_position_value: Decimal = Decimal("1000")

class PositionSizer:
    """Calculate position sizes."""

    def __init__(self, config: PositionSizerConfig): ...

    def calculate_size(
        self,
        signal: TradeSignal,
        portfolio: Portfolio,
        volatility: float,
        expected_return: float | None = None,
        win_rate: float | None = None
    ) -> Decimal:
        """Calculate position size based on method."""

    def fixed_dollar(self, price: Decimal) -> Decimal: ...
    def percent_equity(self, equity: Decimal, price: Decimal) -> Decimal: ...
    def volatility_scaled(self, equity: Decimal, vol: float, price: Decimal) -> Decimal: ...
    def kelly_criterion(self, equity: Decimal, win_rate: float, win_loss_ratio: float) -> Decimal: ...
```

---

## 12. Backtesting

Located in `quant_trading_system/backtest/`.

### Backtest Configuration (`engine.py`)

```python
class BacktestMode(str, Enum):
    EVENT_DRIVEN = "event_driven"  # Realistic simulation
    VECTORIZED = "vectorized"      # Fast research mode

class ExecutionMode(str, Enum):
    REALISTIC = "realistic"        # Includes slippage, costs
    OPTIMISTIC = "optimistic"      # No costs (upper bound)
    PESSIMISTIC = "pessimistic"    # High costs (stress test)

class BacktestConfig(BaseModel):
    initial_capital: Decimal = Decimal("100000")
    mode: BacktestMode = BacktestMode.EVENT_DRIVEN
    execution_mode: ExecutionMode = ExecutionMode.REALISTIC
    commission_bps: float = 5.0           # Basis points
    slippage_bps: float = 5.0             # Basis points
    fill_at: str = "next_open"            # next_open, next_close, same_close
    allow_short: bool = True
    allow_fractional: bool = True
    max_leverage: float = 1.0
    max_position_pct: float = 0.10
    max_drawdown_halt: float = 0.20       # Stop if breached

    # Market Simulation (JPMorgan fix)
    use_market_simulator: bool = True
    simulate_partial_fills: bool = True
    simulate_latency: bool = True
    avg_daily_volume: int = 1_000_000

    # Data Settings
    data_frequency: str = "15Min"
    warmup_bars: int = 100
```

### Trade Record

```python
class TradeRecord(BaseModel):
    """Completed trade for analysis."""
    trade_id: UUID
    symbol: str
    entry_time: datetime
    exit_time: datetime
    side: OrderSide
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    pnl: Decimal
    pnl_pct: float
    commission: Decimal
    slippage: Decimal
    holding_period_bars: int
    exit_reason: str
    metadata: dict = {}
```

### BacktestEngine

```python
class BacktestEngine:
    """Event-driven backtesting engine."""

    def __init__(
        self,
        data_handler: DataHandler,
        strategy: Strategy,
        config: BacktestConfig
    ): ...

    def run(self) -> BacktestState:
        """Execute backtest and return final state."""

    def get_trades(self) -> list[TradeRecord]:
        """Get all completed trades."""

    def get_equity_curve(self) -> pd.Series:
        """Get equity over time."""

    def get_positions_history(self) -> pd.DataFrame:
        """Get position history."""
```

### Market Simulator (`simulator.py`)

```python
class SlippageModel(str, Enum):
    FIXED = "fixed"
    VOLUME_BASED = "volume_based"
    VOLATILITY_BASED = "volatility_based"
    MARKET_IMPACT = "market_impact"  # Almgren-Chriss

class FillResult(BaseModel):
    fill_type: FillType  # FULL, PARTIAL, REJECTED
    fill_price: Decimal
    fill_quantity: Decimal
    slippage: Decimal
    market_impact: Decimal
    commission: Decimal
    latency_ms: float
    timestamp: datetime
    partial_remaining: Decimal | None = None
    rejection_reason: str | None = None

    @property
    def total_cost(self) -> Decimal

    @property
    def effective_price(self) -> Decimal

class MarketSimulator:
    """Simulate realistic market execution."""

    def __init__(
        self,
        slippage_model: SlippageModel = SlippageModel.VOLUME_BASED,
        commission_bps: float = 5.0,
        spread_bps: float = 2.0,
        avg_daily_volume: int = 1_000_000,
        simulate_partial_fills: bool = True
    ): ...

    def simulate_fill(
        self,
        order: Order,
        market_conditions: MarketConditions
    ) -> FillResult:
        """Simulate order fill with realistic conditions."""
```

### Performance Analyzer (`analyzer.py`)

```python
class ReturnMetrics(BaseModel):
    total_return: float
    annualized_return: float
    monthly_returns: list[float]
    daily_returns: list[float]
    best_day: float
    worst_day: float
    best_month: float
    worst_month: float
    positive_days_pct: float

class RiskAdjustedMetrics(BaseModel):
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float | None
    treynor_ratio: float | None
    omega_ratio: float
    volatility_annual: float
    downside_volatility: float

class DrawdownMetrics(BaseModel):
    max_drawdown: float
    max_drawdown_duration_days: int
    avg_drawdown: float
    avg_drawdown_duration_days: float
    recovery_time_days: int | None
    current_drawdown: float
    drawdown_periods: list[DrawdownPeriod]
    ulcer_index: float
    pain_index: float

class TradeMetrics(BaseModel):
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_holding_period: float
    max_consecutive_wins: int
    max_consecutive_losses: int

class BacktestAnalyzer:
    """Calculate comprehensive performance metrics."""

    def analyze(
        self,
        equity_curve: pd.Series,
        trades: list[TradeRecord],
        benchmark: pd.Series | None = None
    ) -> AnalysisReport: ...
```

### Performance Attribution (`performance_attribution.py`)

```python
class PerformanceAttributionService:
    """Brinson-Fachler and factor-based attribution."""

    def brinson_attribution(
        self,
        portfolio_returns: pd.Series,
        portfolio_weights: pd.DataFrame,
        benchmark_returns: pd.Series,
        benchmark_weights: pd.DataFrame
    ) -> BrinsonAttribution:
        """Allocation, selection, interaction effects."""

    def factor_attribution(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> FactorAttribution:
        """Factor exposure and contribution."""

    def risk_attribution(
        self,
        returns: pd.Series,
        weights: pd.DataFrame,
        covariance: pd.DataFrame
    ) -> RiskAttribution:
        """Risk contribution by position."""

    def generate_full_report(
        self,
        portfolio_returns: pd.Series,
        portfolio_weights: pd.DataFrame,
        benchmark_returns: pd.Series,
        benchmark_weights: pd.DataFrame,
        factor_returns: pd.DataFrame,
        position_returns: pd.DataFrame,
        trades: list[TradeRecord]
    ) -> AttributionReport: ...
```

---

## 13. Data Pipeline

Located in `quant_trading_system/data/`.

### Data Loader (`loader.py`)

```python
class DataLoader:
    """Load historical market data."""

    SUPPORTED_FORMATS = {".csv", ".parquet", ".hdf5", ".h5"}

    def __init__(
        self,
        data_dir: Path = Path("data/raw"),
        cache_enabled: bool = True
    ): ...

    def load_symbol(
        self,
        symbol: str,
        start_date: date | None = None,
        end_date: date | None = None,
        columns: list[str] | None = None
    ) -> pl.DataFrame:
        """Load data for single symbol."""

    def load_symbols(
        self,
        symbols: list[str],
        start_date: date | None = None,
        end_date: date | None = None
    ) -> dict[str, pl.DataFrame]:
        """Load data for multiple symbols."""

    def validate_data(self, df: pl.DataFrame) -> list[str]:
        """Validate data quality. Returns list of errors."""

    def get_available_symbols(self) -> list[str]:
        """List symbols with available data."""

    def get_date_range(self, symbol: str) -> tuple[date, date]:
        """Get available date range for symbol."""
```

### Preprocessor (`preprocessor.py`)

```python
class Preprocessor:
    """Clean and normalize market data."""

    def __init__(
        self,
        fill_method: str = "ffill",
        outlier_method: str = "clip",
        outlier_threshold: float = 5.0
    ): ...

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full preprocessing pipeline."""

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def ensure_timezone(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

### Live Feed (`live_feed.py`)

```python
class LiveFeed:
    """Real-time market data via WebSocket."""

    def __init__(
        self,
        alpaca_settings: AlpacaSettings,
        symbols: list[str],
        event_bus: EventBus
    ): ...

    async def connect(self) -> None:
        """Establish WebSocket connection."""

    async def disconnect(self) -> None:
        """Close connection gracefully."""

    async def subscribe_bars(self, symbols: list[str]) -> None:
        """Subscribe to bar updates."""

    async def subscribe_trades(self, symbols: list[str]) -> None:
        """Subscribe to trade updates."""

    async def subscribe_quotes(self, symbols: list[str]) -> None:
        """Subscribe to quote updates."""
```

### Data Lineage (`lineage.py`)

```python
class DataSource(str, Enum):
    EXTERNAL = "external"    # Alpaca, third-party
    COMPUTED = "computed"    # Derived features
    USER = "user"            # User uploads
    SYSTEM = "system"        # System-generated

class DataLineageNode(BaseModel):
    node_id: UUID
    name: str
    source: DataSource
    created_at: datetime
    parent_ids: list[UUID] = []
    transformation: str | None = None
    metadata: dict = {}

class DataLineageTracker:
    """Track data provenance for regulatory compliance."""

    def register_source(
        self,
        name: str,
        source: DataSource,
        metadata: dict = {}
    ) -> DataLineageNode:
        """Register new data source."""

    def add_transformation(
        self,
        name: str,
        parent_ids: list[UUID],
        transformation: str,
        metadata: dict = {}
    ) -> DataLineageNode:
        """Record data transformation."""

    def get_full_lineage(self, node_id: UUID) -> list[DataLineageNode]:
        """Get complete lineage chain."""

    def export_lineage(self, node_id: UUID) -> dict:
        """Export lineage for compliance reporting."""
```

---

## 14. Monitoring & Observability

Located in `quant_trading_system/monitoring/`.

### Prometheus Metrics (`metrics.py`)

```python
# System Metrics
system_cpu_usage = Gauge("trading_system_cpu_percent", "CPU usage")
system_memory_usage = Gauge("trading_system_memory_mb", "Memory usage")
system_uptime = Gauge("trading_system_uptime_seconds", "Uptime")

# Trading Metrics
orders_total = Counter(
    "trading_orders_total",
    "Total orders",
    ["symbol", "side", "status"]
)
fills_total = Counter(
    "trading_fills_total",
    "Total fills",
    ["symbol", "side"]
)
position_value = Gauge(
    "trading_position_value",
    "Position value",
    ["symbol"]
)
portfolio_equity = Gauge("trading_portfolio_equity", "Total equity")
daily_pnl = Gauge("trading_daily_pnl", "Daily P&L")

# Risk Metrics
current_drawdown = Gauge("risk_drawdown_current", "Current drawdown")
max_drawdown = Gauge("risk_drawdown_max", "Maximum drawdown")
var_95 = Gauge("risk_var_95", "95% VaR")
kill_switch_active = Gauge("risk_kill_switch_active", "Kill switch status")

# Model Metrics
model_predictions = Counter(
    "model_predictions_total",
    "Predictions",
    ["model_name"]
)
model_latency = Histogram(
    "model_prediction_latency_seconds",
    "Prediction latency",
    ["model_name"]
)

# Event Metrics
events_published = Counter(
    "events_published_total",
    "Events published",
    ["event_type"]
)
event_processing_latency = Histogram(
    "event_processing_latency_seconds",
    "Event processing time"
)
```

### Audit Logging (`audit.py`)

```python
class AuditEventType(str, Enum):
    # Trading
    ORDER_CREATED = "order_created"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_MODIFIED = "order_modified"

    # Positions
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_MODIFIED = "position_modified"

    # Risk
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    KILL_SWITCH_RESET = "kill_switch_reset"

    # Model
    MODEL_PREDICTION = "model_prediction"
    MODEL_TRAINING_STARTED = "model_training_started"
    MODEL_TRAINING_COMPLETED = "model_training_completed"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_RETIRED = "model_retired"

    # Signals
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_REJECTED = "signal_rejected"
    SIGNAL_EXECUTED = "signal_executed"

    # System
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIGURATION_CHANGED = "configuration_changed"
    ERROR_OCCURRED = "error_occurred"

    # User
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_ACTION = "user_action"
    AUTHORIZATION_GRANTED = "authorization_granted"
    AUTHORIZATION_DENIED = "authorization_denied"

class AuditSeverity(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AuditLogger:
    """Immutable audit trail for regulatory compliance."""

    def __init__(self, storage_dir: Path = Path("audit_logs")): ...

    def log_order_created(
        self,
        order_id: UUID,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str,
        **kwargs
    ) -> None: ...

    def log_risk_check(
        self,
        check_name: str,
        passed: bool,
        details: dict
    ) -> None: ...

    def log_kill_switch_activated(
        self,
        reason: str,
        triggered_by: str,
        **kwargs
    ) -> None: ...

    def verify_chain(self) -> bool:
        """Verify audit log integrity (SHA-256 chain)."""

    def export_range(
        self,
        start: datetime,
        end: datetime,
        event_types: list[AuditEventType] | None = None
    ) -> list[AuditRecord]:
        """Export records for compliance reporting."""
```

### Alerting (`alerting.py`)

```python
class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertChannel(str, Enum):
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    SMS = "sms"

class AlertManager:
    """Multi-channel alert management."""

    def __init__(self, settings: AlertSettings): ...

    async def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        channels: list[AlertChannel] | None = None,
        metadata: dict = {}
    ) -> None:
        """Send alert to configured channels."""

    async def send_slack(self, title: str, message: str, channel: str) -> None: ...
    async def send_email(self, title: str, message: str, recipients: list[str]) -> None: ...
    async def send_pagerduty(self, title: str, message: str, severity: str) -> None: ...
    async def send_sms(self, message: str, recipients: list[str]) -> None: ...
```

---

## 15. Infrastructure

### Docker Stack (`docker/docker-compose.yml`)

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| trading_app | custom | 8000 | Main application |
| postgres | timescaledb:pg15 | 5432 | Time-series database |
| redis | redis:7-alpine | 6379 | Cache & message queue |
| prometheus | prom/prometheus | 9090 | Metrics collection |
| grafana | grafana/grafana | 3000 | Visualization |
| alertmanager | prom/alertmanager | 9093 | Alert routing |

### Docker Commands

```bash
# Start full stack
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f trading_app

# Stop all
docker-compose -f docker/docker-compose.yml down

# Rebuild
docker-compose -f docker/docker-compose.yml build --no-cache
```

### Local Development (Windows)

```bash
# Start Redis
cd redis && redis-server.exe redis.windows.conf

# Start PostgreSQL (if not using Docker)
pg_ctl start -D /path/to/data

# Run application
python main.py trade --mode paper
```

---

## 16. Testing

### Test Structure

```
tests/
├── conftest.py                 # Shared fixtures
├── unit/                       # Unit tests (fast, isolated)
│   ├── test_data_types.py
│   ├── test_events.py
│   ├── test_exceptions.py
│   ├── test_execution.py
│   ├── test_models.py
│   ├── test_risk.py
│   ├── test_features.py
│   ├── test_alpha.py
│   └── ... (17 files total)
└── integration/                # Integration tests
    ├── test_data_pipeline.py
    ├── test_monitoring_integration.py
    └── test_websocket_reconnection.py
```

### Test Commands

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=quant_trading_system --cov-report=html

# Run unit tests only
pytest tests/unit/

# Run specific file
pytest tests/unit/test_risk.py -v

# Run tests matching pattern
pytest -k "risk_checker" -v

# Run with parallel execution
pytest tests/ -n auto
```

### Key Fixtures (`conftest.py`)

```python
@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame: ...

@pytest.fixture
def sample_order() -> Order: ...

@pytest.fixture
def sample_portfolio() -> Portfolio: ...

@pytest.fixture
def event_bus() -> EventBus: ...

@pytest.fixture
def metrics_collector() -> MetricsCollector: ...

@pytest.fixture
def mock_settings() -> Settings: ...

@pytest.fixture
def mock_alpaca_client() -> AsyncMock: ...
```

---

## 17. Architecture Invariants

These rules MUST be followed at all times:

### Critical Safety Rules
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
2. **OHLCV Validation** - High >= Low, High >= Open/Close, etc.
3. **Time-Series Splits** - NEVER use random train/test splits
4. **Walk-Forward Validation** - Use expanding window for ML

### Error Handling
1. **Custom Exception Hierarchy** - Use `TradingSystemError` subclasses
2. **Fail-Safe Defaults** - On error, REJECT order (not accept)
3. **Structured Logging** - JSON format for production
4. **Audit Trail** - All significant events logged immutably

---

## 18. Agent System

AlphaTrade uses a multi-agent system for development guidance. Each agent has specific expertise:

### Agent Registry

| ID | Agent | Alias | Priority | Scope |
|----|-------|-------|----------|-------|
| 01 | System Architect | `@architect` | Highest | Core, config, interfaces |
| 02 | ML/Quant Engineer | `@mlquant` | High | Models, features, alphas |
| 03 | Trading & Execution | `@trader` | **Critical** | Execution, risk, orders |
| 04 | Data Engineer | `@data` | High | Pipelines, DB, storage |
| 05 | Infrastructure & QA | `@infra` | High | Docker, tests, monitoring |

### Priority Resolution

When agents have conflicting guidance:
1. **@trader** (Critical) - Safety/risk concerns override everything
2. **@architect** (Highest) - Architecture decisions next
3. **Domain agents** (High) - Specific implementation details last

### Agent Files

Located in `.claude/agents/`:
- `01-system-architect.md` - Architecture patterns and invariants
- `02-ml-quant-engineer.md` - ML best practices and model interfaces
- `03-trading-execution.md` - Order flow and risk protocols
- `04-data-engineer.md` - Data pipeline and quality rules
- `05-infrastructure-qa.md` - Testing and deployment standards

---

## 19. Quick Reference

### Running the System

```bash
# Paper trading
python main.py trade --mode paper

# Live trading (requires Alpaca credentials)
python main.py trade --mode live

# Backtest
python scripts/run_backtest.py \
    --start-date 2024-01-01 --end-date 2024-06-30 \
    --symbols AAPL MSFT GOOGL \
    --strategy momentum \
    --execution-mode realistic

# Train models
python scripts/train_models.py \
    --model xgboost \
    --symbols AAPL MSFT \
    --start-date 2023-01-01

# Dashboard
python main.py dashboard --port 8000

# Docker deployment
docker-compose -f docker/docker-compose.yml up -d
```

### Key File Locations

| Purpose | Location |
|---------|----------|
| Entry point | `main.py` |
| Settings | `quant_trading_system/config/settings.py` |
| Core data types | `quant_trading_system/core/data_types.py` |
| Event system | `quant_trading_system/core/events.py` |
| Exceptions | `quant_trading_system/core/exceptions.py` |
| Model base | `quant_trading_system/models/base.py` |
| Risk limits | `quant_trading_system/risk/limits.py` |
| Kill switch | `quant_trading_system/risk/limits.py` |
| Trading engine | `quant_trading_system/trading/trading_engine.py` |
| Order manager | `quant_trading_system/execution/order_manager.py` |
| Alpaca client | `quant_trading_system/execution/alpaca_client.py` |
| Backtest engine | `quant_trading_system/backtest/engine.py` |
| Market simulator | `quant_trading_system/backtest/simulator.py` |
| Data loader | `quant_trading_system/data/loader.py` |
| Feature pipeline | `quant_trading_system/features/feature_pipeline.py` |
| Alpha base | `quant_trading_system/alpha/alpha_base.py` |
| Alpha metrics | `quant_trading_system/alpha/alpha_metrics.py` |
| Regime detection | `quant_trading_system/alpha/regime_detection.py` |
| Model validation | `quant_trading_system/models/validation_gates.py` |
| Explainability | `quant_trading_system/models/explainability.py` |
| Performance attribution | `quant_trading_system/backtest/performance_attribution.py` |
| Audit logging | `quant_trading_system/monitoring/audit.py` |
| Prometheus metrics | `quant_trading_system/monitoring/metrics.py` |
| Data lineage | `quant_trading_system/data/lineage.py` |

### Implementation Status

| Category | Status |
|----------|--------|
| Core data types, events | Complete |
| Data loading, preprocessing | Complete |
| Technical/statistical features | Complete |
| Microstructure features (VPIN) | Complete |
| ML models (XGBoost, LSTM, etc.) | Complete |
| Model validation gates | Complete |
| SHAP/LIME explainability | Complete |
| Alpha factors + IC/IR metrics | Complete |
| Regime detection | Complete |
| Backtest engine + MarketSimulator | Complete |
| Performance attribution | Complete |
| Risk limits, kill switch | Complete |
| Position sizing algorithms | Complete |
| Alpaca integration | Complete |
| Execution algorithms (TWAP/VWAP) | Complete |
| Prometheus metrics | Complete |
| Audit logging | Complete |
| Docker infrastructure | Complete |
| Test suite (700+ tests) | Complete |

---

*Last updated: 2026-01-02*
*AlphaTrade System v1.0*
