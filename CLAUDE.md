# AlphaTrade System - Claude Code Project Guide

## Project Overview
Institutional-grade quantitative trading platform. Python 3.11+ backend (FastAPI, SQLAlchemy, PyTorch, XGBoost), React 19 frontend (TypeScript, Vite, Zustand, TailwindCSS).

## Quick Commands
```bash
# Tests
pytest tests/ -v                              # All tests
pytest tests/unit -m unit                     # Unit only
pytest tests/integration -m integration       # Integration only
pytest tests/ --cov=quant_trading_system      # With coverage

# Lint & Format
ruff check .                                  # Lint
ruff check . --fix                            # Auto-fix
black --check .                               # Format check
mypy quant_trading_system/                    # Type check

# Application
python main.py trade --mode paper             # Paper trading
python main.py backtest --start 2024-01-01    # Backtest
python main.py train                          # Model training
python main.py data load                      # Data ingestion
python main.py deploy migrate                 # DB migrations
python main.py health check --full            # Health check
python main.py dashboard                      # Launch dashboard

# Frontend
cd dashboard_ui && npm run dev                # Dev server
cd dashboard_ui && npm run build              # Production build
```

## Architecture
- **12 domain subpackages** under `quant_trading_system/`:
  - `alpha/` - Signal generation (momentum, mean-reversion, ML, regime detection)
  - `backtest/` - Event-driven historical simulation
  - `config/` - Pydantic Settings, feature flags, YAML configs
  - `core/` - Events, circuit breaker, CQRS, data types, registry
  - `data/` - Ingestion, feature store, loaders, lineage
  - `database/` - SQLAlchemy models, repository pattern, Alembic migrations
  - `execution/` - OMS, Alpaca client, TWAP/VWAP, TCA
  - `features/` - 200+ technical/statistical/microstructure indicators
  - `models/` - ML/DL pipeline, purged CV, meta-labeling, validation gates
  - `monitoring/` - Dashboard API, alerting, audit, Prometheus metrics
  - `risk/` - Limits, drawdown, VaR, position sizing, portfolio optimizer
  - `trading/` - Trading engine, signal generator, portfolio manager

## Code Style
- **Line length**: 100 (Black + Ruff)
- **Python**: 3.11+, strict mypy, type hints required
- **Formatter**: Black with isort (profile=black)
- **Linter**: Ruff (E, W, F, I, B, C4, UP)
- **Imports**: isort with `known_first_party = ["quant_trading_system"]`
- **Tests**: pytest with markers: `unit`, `integration`, `slow`

## Database
- **TimescaleDB** (PostgreSQL 15) at `127.0.0.1:5433`
- Migrations: `quant_trading_system/database/migrations/versions/`
- Always use Alembic for schema changes

## Key Config Files
- `quant_trading_system/config/settings.py` - All env var bindings
- `quant_trading_system/config/risk_params.yaml` - Risk limits
- `quant_trading_system/config/model_configs.yaml` - ML hyperparameters
- `quant_trading_system/config/symbols.yaml` - Trading universe (46 equities)

## Docker
```bash
docker-compose -f docker/docker-compose.yml up -d          # Dev stack
docker-compose -f docker/docker-compose.production.yml up   # Production
```
Services: trading_app, postgres, redis, prometheus, grafana, alertmanager

## Important Conventions
- PostgreSQL is the primary data source (no file fallback)
- Risk controls are mandatory - never bypass kill switch or pre-trade checks
- Purged CV for all ML validation (prevent data leakage)
- Structured logging via structlog
- All financial calculations must handle edge cases (zero division, NaN, missing data)
