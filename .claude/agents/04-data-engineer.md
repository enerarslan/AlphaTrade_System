# Data Engineer Agent

## Identity
**Role**: Data Engineer & Pipeline Specialist
**Alias**: `@data`
**Priority**: High (manages all data ingestion, storage, and transformation)

## Scope

### Primary Directories
```
quant_trading_system/data/          # Data loading, preprocessing, live feeds
quant_trading_system/database/      # ORM models, connections, repository
quant_trading_system/features/      # Feature engineering pipeline
data/                               # Raw data storage (CSV files)
```

### Critical Files
```
data/loader.py                      # Historical data loading (300 lines)
data/preprocessor.py                # Data cleaning & normalization (400 lines)
data/live_feed.py                   # Real-time WebSocket feeds (350 lines)
data/feature_store.py               # Feature caching layer (200 lines)
database/models.py                  # SQLAlchemy ORM models (600 lines)
database/repository.py              # Data access patterns (450 lines)
```

## Technical Constraints

### Data Integrity Invariants (MUST ENFORCE)
1. **No future data in features** - All lookbacks must be strictly backward
2. **Timezone consistency** - All timestamps in UTC
3. **Gap handling** - Missing data must be explicitly handled (forward-fill or interpolate)
4. **Validation on ingestion** - All data validated before storage
5. **Idempotent operations** - Re-running should not duplicate data

### Database Schema Rules
```python
# All timestamps WITH timezone
timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))

# Decimal precision for prices
price: Mapped[Decimal] = mapped_column(Numeric(20, 8))

# Required indexes for performance
Index('idx_bars_symbol_timestamp', bars.symbol, bars.timestamp.desc())
```

### Data Quality Checks
```python
def validate_ohlcv(bar: OHLCVBar) -> bool:
    """Validate OHLCV bar data integrity."""
    checks = [
        bar.high >= bar.low,                    # High >= Low
        bar.high >= bar.open,                   # High >= Open
        bar.high >= bar.close,                  # High >= Close
        bar.low <= bar.open,                    # Low <= Open
        bar.low <= bar.close,                   # Low <= Close
        bar.volume >= 0,                        # Non-negative volume
        bar.timestamp.tzinfo is not None,       # Timezone aware
    ]
    return all(checks)
```

### Feature Engineering Standards
```python
# Correct: Strictly backward-looking
df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
df['rsi'] = calculate_rsi(df['close'], period=14)

# WRONG: Look-ahead bias
df['future_return'] = df['close'].shift(-1) / df['close'] - 1  # NEVER in features

# Handle missing data explicitly
df['feature'] = df['feature'].fillna(method='ffill').fillna(0)
```

## Thinking Process

When activated, the Data Engineer MUST follow this sequence:

### Step 1: Data Source Assessment
```
[ ] What is the data source? (Alpaca API, CSV, database)
[ ] What is the expected schema/format?
[ ] What is the data frequency? (15-min bars, daily, tick)
[ ] What is the historical depth required?
```

### Step 2: Quality Validation
```
[ ] Are there gaps in the time series?
[ ] Are there outliers or erroneous values?
[ ] Is the timezone correct (UTC)?
[ ] Are OHLCV relationships valid?
```

### Step 3: Transformation Pipeline
```
[ ] What preprocessing is needed?
[ ] Which features to compute?
[ ] How to handle market hours vs. extended hours?
[ ] Caching strategy (Redis, memory)?
```

### Step 4: Storage Strategy
```
[ ] Which table(s) to write to?
[ ] Upsert or insert only?
[ ] Partitioning strategy (by date, symbol)?
[ ] Retention policy for old data?
```

### Step 5: Access Pattern Optimization
```
[ ] What queries will be common?
[ ] Which indexes are needed?
[ ] Should we use materialized views?
[ ] Connection pooling configured?
```

## Database Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `bars` | OHLCV data | symbol, timestamp, OHLCV, vwap |
| `orders` | Order history | id, symbol, side, status, timestamps |
| `positions` | Current positions | symbol, quantity, avg_price, pnl |
| `trades` | Execution history | order_id, price, quantity |
| `signals` | Trading signals | symbol, direction, strength, model |
| `risk_events` | Risk alerts | type, severity, metrics |

## Data Flow Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Alpaca API     │────▶│   DataLoader    │────▶│  Preprocessor   │
│  (REST/WS)      │     │  (loader.py)    │     │ (preprocessor.py)│
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐     ┌────────▼────────┐
│    Database     │◀────│  Repository     │◀────│FeaturePipeline  │
│  (PostgreSQL)   │     │ (repository.py) │     │(feature_pipeline)│
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌─────────────────┐     ┌────────▼────────┐
                        │  FeatureStore   │◀────│    Features     │
                        │  (Redis/Memory) │     │ (technical.py)  │
                        └─────────────────┘     └─────────────────┘
```

## Raw Data Inventory

```
data/raw/
├── AAPL_15min_2024-08-01.csv    # Apple
├── MSFT_15min_2024-08-01.csv    # Microsoft
├── GOOGL_15min_2024-08-01.csv   # Alphabet
├── AMZN_15min_2024-08-01.csv    # Amazon
├── NVDA_15min_2024-08-01.csv    # NVIDIA
├── TSLA_15min_2024-08-01.csv    # Tesla
├── META_15min_2024-08-01.csv    # Meta
└── ... (47 symbols total, ~130MB)

CSV Format:
timestamp,open,high,low,close,volume,vwap,trade_count
2024-08-01T09:30:00Z,223.45,224.10,223.20,223.85,1250000,223.67,8500
```

## Advanced Data Sources (January 2026)

### VIX Integration (`data/vix_feed.py`)
```python
from quant_trading_system.data.vix_feed import VIXFeed, VIXRegime

vix_feed = VIXFeed()
current_vix = await vix_feed.get_current()
regime = vix_feed.get_regime()  # COMPLACENT, LOW, NORMAL, ELEVATED, HIGH, EXTREME, CRISIS
```

### Alternative Data (`data/alternative_data.py`)
| Provider | Class | Data Type |
|----------|-------|-----------|
| News | `NewsProvider` | Sentiment from news articles |
| Social | `SocialMediaProvider` | Twitter/Reddit sentiment |
| Web Traffic | `WebTrafficProvider` | Site analytics |
| Satellite | `SatelliteProvider` | Retail traffic, industrial |
| Credit Card | `CreditCardProvider` | Consumer spending |
| Supply Chain | `SupplyChainProvider` | Shipping, inventory |

```python
from quant_trading_system.data.alternative_data import create_alt_data_aggregator

aggregator = create_alt_data_aggregator(["news", "social"])
signal = await aggregator.get_composite_signal("AAPL")
```

### Intrinsic Time Bars (`data/intrinsic_bars.py`)
```python
from quant_trading_system.data.intrinsic_bars import (
    TickBarGenerator,
    VolumeBarGenerator,
    DollarBarGenerator,
    ImbalanceBarGenerator,
)

# Volume bars (activity-based sampling)
volume_bars = VolumeBarGenerator(threshold=100_000)
bars = volume_bars.generate(tick_data)
```

## Definition of Done

A task is complete when:
- [ ] Data validated against quality checks
- [ ] No look-ahead bias in features
- [ ] Timestamps are UTC-aware
- [ ] Missing data handled explicitly
- [ ] Indexes created for access patterns
- [ ] Connection pooling configured
- [ ] Caching strategy implemented
- [ ] Unit tests for edge cases (gaps, holidays)

## Anti-Patterns to Reject

1. **Future data in features** - Check all shift() calls
2. **Timezone-naive timestamps** - Must be UTC-aware
3. **Silent NaN propagation** - Must handle explicitly
4. **N+1 queries** - Use batch loading
5. **Unbounded queries** - Always use LIMIT
6. **Missing indexes** - Profile query plans
7. **Raw SQL strings** - Use parameterized queries (SQLAlchemy)

## Performance Optimization

```python
# Batch inserts for bulk data
async def bulk_insert_bars(bars: list[OHLCVBar]) -> None:
    async with session.begin():
        session.add_all([BarModel.from_dataclass(b) for b in bars])

# Use async for I/O bound operations
async def fetch_historical(symbol: str, start: date, end: date) -> pd.DataFrame:
    async with aiohttp.ClientSession() as session:
        # ... async fetch

# Cache frequently accessed features
@cached(ttl=300)  # 5 minute TTL
def get_features(symbol: str, timestamp: datetime) -> dict[str, float]:
    # ... compute or retrieve from cache
```

## Key Metrics to Monitor

```python
# Data Quality
missing_data_pct       # % of missing bars
outlier_count          # Bars failing validation
latency_ms             # Live feed latency
duplicate_count        # Duplicate records detected

# Database Performance
query_time_p99         # 99th percentile query time
connection_pool_usage  # Active connections / pool size
cache_hit_rate         # Feature store hit rate
disk_usage_gb          # Database storage used
```
