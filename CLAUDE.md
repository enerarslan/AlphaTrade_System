# AlphaTrade System - Claude Code Project Guide

## Project Overview
Institutional-grade quantitative trading platform. Python 3.11+ monolith with event-driven architecture, ML/DL signal generation, and full risk management pipeline.

**Entry Point**: `main.py` - Unified CLI for all operations (trade, backtest, replay, train, data, features, health, deploy).

## CLI Commands (main.py)
```bash
# Trading
python main.py trade --mode paper --symbols AAPL MSFT GOOGL
python main.py trade --mode live --symbols AAPL

# Backtesting & Replay
python main.py backtest --start 2024-01-01 --end 2024-06-30
python main.py replay --start 2024-01-01 --end 2024-01-15 --symbols AAPL MSFT

# ML/DL Training
python main.py train --model xgboost --symbols AAPL MSFT

# Data Pipeline
python main.py data load --source alpaca --symbols AAPL
python main.py features compute --symbols AAPL --gpu

# Operations
python main.py health check --full
python main.py deploy setup --env production
python main.py deploy migrate
python main.py dashboard
```

## Scripts (`scripts/`)
| Script | Purpose |
|--------|---------|
| `trade.py` | Trading execution entry point |
| `backtest.py` | Backtest CLI entry point |
| `replay.py` | Deterministic replay runner |
| `train.py` | ML training pipeline |
| `data.py` | Data ingestion/management |
| `features.py` | Feature computation |
| `health.py` | System diagnostics |
| `deploy.py` | Deployment & migrations |
| `export_training_data.py` | Training data export |
| `migrate_to_postgres.py` | PostgreSQL migration |
| `security_scan.py` | Security scanning |
| `repo_audit.py` | Repository audit |

## Architecture (`quant_trading_system/`)

### Core Layer
- **`core/`** - System foundations
  - `events.py` - Event-driven pub/sub bus (singleton EventBus)
  - `circuit_breaker.py` - External API circuit breaker
  - `cqrs.py` - Command/Query separation
  - `data_types.py` - Shared domain types
  - `registry.py` - Service registry
  - `system_integrator.py` - Component wiring
  - `redis_bridge.py` - Redis pub/sub integration
  - `reproducibility.py` - Deterministic execution support

### Signal & Alpha Layer
- **`alpha/`** - Alpha signal generation
  - `alpha_base.py` - Abstract base for all alphas
  - `momentum_alphas.py` - Momentum strategies
  - `mean_reversion_alphas.py` - Mean-reversion strategies
  - `ml_alphas.py` - ML-based signal generation
  - `regime_detection.py` - Market regime classification
  - `alpha_combiner.py` - Multi-alpha combination
  - `alpha_metrics.py` - Signal quality metrics

### Feature Engineering Layer
- **`features/`** - 200+ indicators
  - `technical.py` / `technical_extended.py` - Technical indicators
  - `statistical.py` - Statistical features
  - `microstructure.py` - Market microstructure
  - `cross_sectional.py` - Cross-sectional features
  - `feature_pipeline.py` / `optimized_pipeline.py` - Pipeline orchestration
  - `reference.py` - Reference data features

### ML/DL Layer
- **`models/`** - ML/DL model pipeline
  - `model_manager.py` - Model lifecycle (train/eval/deploy)
  - `classical_ml.py` - XGBoost, LightGBM, CatBoost, scikit-learn
  - `deep_learning.py` - PyTorch / Lightning models
  - `ensemble.py` - Model ensembling
  - `reinforcement.py` / `rl_meta_learning.py` - RL approaches
  - `purged_cv.py` - Purged K-Fold CV (anti look-ahead bias)
  - `meta_labeling.py` - Meta-labeling for bet sizing
  - `validation_gates.py` - Model promotion gates
  - `explainability.py` - SHAP/LIME
  - `training_lineage.py` - Lineage tracking
  - `trading_costs.py` - Cost-aware training
  - `target_engineering.py` - Label engineering

### Trading & Execution Layer
- **`trading/`** - Trading orchestration
  - `trading_engine.py` - Main engine (event loop, lifecycle)
  - `signal_generator.py` - Signal aggregation & dispatch
  - `portfolio_manager.py` - Portfolio state management
  - `strategy.py` - Strategy abstractions
- **`execution/`** - Order execution
  - `order_manager.py` - OMS, order routing
  - `alpaca_client.py` - Alpaca Markets broker API
  - `execution_algo.py` - TWAP/VWAP algorithms
  - `position_tracker.py` - Position reconciliation
  - `tca.py` - Transaction cost analysis
  - `market_impact.py` - Market impact models
  - `failover.py` - Execution failover

### Risk Layer
- **`risk/`** - Risk management (mandatory, never bypass)
  - `risk_monitor.py` - Real-time risk monitoring
  - `limits.py` - Position/exposure limits + kill switch
  - `drawdown_monitor.py` - Drawdown tracking
  - `var_stress_testing.py` - VaR & stress tests
  - `position_sizer.py` / `regime_position_sizer.py` - Position sizing
  - `portfolio_optimizer.py` - Mean-variance optimization
  - `correlation_monitor.py` - Correlation regime monitoring
  - `sector_rebalancer.py` - Sector allocation

### Backtest Layer
- **`backtest/`** - Historical simulation
  - `engine.py` - Event-driven backtest engine
  - `simulator.py` - Market simulation
  - `analyzer.py` - Performance analytics
  - `optimizer.py` - Strategy parameter optimization
  - `performance_attribution.py` - Return attribution
  - `replay.py` - Deterministic replay with SLO gates

### Data Layer
- **`data/`** - Data pipeline
  - `data_access.py` - Unified data access (PostgreSQL primary)
  - `db_loader.py` - Database loader
  - `db_feature_store.py` - Feature store (DB-backed)
  - `live_feed.py` / `live_feed_persister.py` - Real-time data
  - `preprocessor.py` - Data cleaning & normalization
  - `intrinsic_bars.py` - Volume/dollar/tick bars
  - `vix_feed.py` - VIX data feed
  - `data_lineage.py` / `lineage.py` - Data provenance

### Infrastructure Layer
- **`database/`** - Persistence
  - `models.py` - SQLAlchemy ORM models
  - `repository.py` - Repository pattern
  - `connection.py` - Connection pool management
  - `schema_sync.py` - Schema synchronization
  - `migrations/` - Alembic migrations
- **`config/`** - Configuration
  - `settings.py` - Pydantic Settings (env vars)
  - `feature_flags.py` - Feature toggles
  - `risk_params.yaml` - Risk limits
  - `model_configs.yaml` - ML hyperparameters
  - `symbols.yaml` - Trading universe (46 equities)
  - `secure_config.py` - Secrets management
- **`monitoring/`** - Observability
  - `dashboard.py` - FastAPI dashboard API
  - `metrics.py` - Prometheus metrics
  - `alerting.py` - Alert rules & dispatch
  - `audit.py` - Regulatory audit trail
  - `health.py` - Health checks
  - `logger.py` - Structured logging (structlog)
  - `tracing.py` - Distributed tracing
- **`security/`** - Security
  - `secret_scanner.py` - Credential scanning

## Data Flow
```
Data Sources → data/ (ingestion) → features/ (engineering) → models/ (prediction)
  → alpha/ (signals) → trading/ (engine) → risk/ (checks) → execution/ (orders)
  → monitoring/ (observe) → database/ (persist)
```

## Tests
```bash
pytest tests/ -v                              # All tests
pytest tests/unit -m unit                     # Unit only
pytest tests/integration -m integration       # Integration only
pytest tests/ --cov=quant_trading_system      # With coverage
```

## Code Style
- **Line length**: 100 (Black + Ruff)
- **Python**: 3.11+, strict mypy, type hints required
- **Formatter**: Black with isort (profile=black)
- **Linter**: Ruff (E, W, F, I, B, C4, UP)
- **Imports**: isort with `known_first_party = ["quant_trading_system"]`

## Database
- **TimescaleDB** (PostgreSQL 15) at `127.0.0.1:5433`
- DB name: `quant_trading`
- Always use Alembic for schema changes, never raw DDL
- PostgreSQL is the primary data source (no file fallback)

## Docker
```bash
docker-compose -f docker/docker-compose.yml up -d          # Dev stack
docker-compose -f docker/docker-compose.production.yml up   # Production
```
Services: trading_app (8000), postgres (5433), redis (6379), prometheus (9090), grafana (3001), alertmanager (9093)

## Critical Rules
- **Risk controls are mandatory** - never bypass kill switch or pre-trade checks
- **Purged CV** for all ML validation (prevent data leakage)
- **PostgreSQL primary** - no file fallback for data access
- **Structured logging** via structlog (never print())
- **Edge case handling** - zero division, NaN, missing data in all financial calculations
- **Event-driven** - use EventBus for cross-component communication
