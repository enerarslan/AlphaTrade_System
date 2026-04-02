# AlphaTrade

AlphaTrade is a production-oriented quantitative trading platform designed to take a model from research to deployment without changing the decision contract on the way.

At a high level, the system is built around:

`data -> features -> models -> alpha -> signal -> portfolio -> risk -> execution -> audit`

The repository currently contains:

- `135` Python modules under [`quant_trading_system/`](quant_trading_system/)
- `49` test files under [`tests/`](tests/)
- a unified operator CLI in [`main.py`](main.py)
- standalone scripts in [`scripts/`](scripts/)
- an optional frontend in [`dashboard_ui/`](dashboard_ui/)

This is not a notebook-only research project. It is structured as a real trading system with explicit runtime boundaries, artifact contracts, risk controls, replay validation, and paper/live deployment paths.

## Why This Project Matters

AlphaTrade is meant to demonstrate end-to-end quantitative trading engineering, not just model training. It combines:

- database-first market data infrastructure
- feature engineering across technical, statistical, cross-sectional, reference, and flow layers
- ML and DL model training with leakage-aware validation
- artifact-backed backtesting, replay, and paper trading
- explicit risk, execution, audit, and monitoring layers

From a CV and portfolio perspective, the interesting part is not just that it trains models. The interesting part is that it tries to solve the full production problem:

- how data is versioned
- how models are validated
- how predictions are converted into economically sensible trades
- how risk controls remain intact in live execution paths
- how the same model contract survives the jump from training to backtest to paper trading

## What It Demonstrates

| Area | What AlphaTrade Shows |
|---|---|
| System design | Modular package architecture with clear runtime layers and operator entrypoints |
| Quant research | Purged CV, embargo-aware evaluation, snapshot-based experiments, artifact lineage |
| Financial ML | LightGBM, XGBoost, deep learning models, calibration, meta-labeling, policy layers |
| Runtime discipline | Promotion package contract reused by backtest, replay, and paper/live trading |
| Risk engineering | Pre-trade checks, kill switch logic, drawdown monitoring, exposure controls |
| Execution | OMS lifecycle, broker boundary, execution-mode simulation, slippage and commission handling |
| Observability | Structured logging, health checks, monitoring hooks, audit-oriented design |
| Data engineering | PostgreSQL / TimescaleDB-first ingestion, export, validation, and training workflows |

## Core Idea: Promotion Packages

The central runtime contract in the current project is the promotion package.

A serious training run can produce:

- a model artifact
- an artifacts JSON file
- a replay manifest
- a promotion package JSON file

That promotion package is the object reused by:

- historical backtests
- deterministic replay
- paper trading
- live trading entrypoints

This matters because it keeps the same settings and policy layer across environments, including:

- thresholds
- sizing rules
- probability calibration
- meta-labeling
- expected-edge gating
- regime-conditioned policy
- asymmetric side controls
- universe quality gating

In other words, the project is not trying to compare a training-time model to a hand-waved runtime implementation. It tries to carry the actual decision policy forward.

## Current Modeling Stack

The current Wave 1 stack is post-hardening and goes beyond raw probability prediction.

Key pieces now in the model-to-trade path:

- purged CV and leakage-aware training
- frozen dataset snapshots and replayable bundles
- out-of-fold probability calibration
- meta-labeling
- OOF expected-edge policy layer
- regime-conditioned policy adjustments
- asymmetric long/short side policy
- universe quality gate
- minimum confidence position scaling
- replay and early paper-trading validation before deployment

The active work plan is tracked in [`docs/LIGHTGBM_TCN_WORK_PLAN.md`](docs/LIGHTGBM_TCN_WORK_PLAN.md).

## Architecture Overview

### 1. Data Layer

The platform is database-first. Historical and runtime paths are designed around PostgreSQL / TimescaleDB, not loose CSV workflows.

Relevant areas:

- [`quant_trading_system/data/`](quant_trading_system/data/)
- [`quant_trading_system/database/`](quant_trading_system/database/)
- [`quant_trading_system/models/training_lineage.py`](quant_trading_system/models/training_lineage.py)

Capabilities include:

- historical data loading
- multi-source bootstrap flows
- DB migration and export paths
- data quality reporting
- snapshot manifest generation
- deterministic dataset bundle hashing and reuse

### 2. Feature Layer

Feature engineering spans multiple families and is intentionally broader than classic indicator-only pipelines.

Current feature families include:

- technical features
- statistical features
- cross-sectional features
- reference features from multi-layer data
- quote / trade / flow-derived features where coverage exists

Relevant areas:

- [`quant_trading_system/features/`](quant_trading_system/features/)
- [`quant_trading_system/features/reference.py`](quant_trading_system/features/reference.py)
- [`quant_trading_system/features/tick_microstructure.py`](quant_trading_system/features/tick_microstructure.py)
- [`quant_trading_system/features/multi_timeframe.py`](quant_trading_system/features/multi_timeframe.py)

The current stack explicitly supports multi-timeframe feature fusion and runtime-compatible feature regeneration for promotion-package inference.

### 3. Model Layer

The model layer is not limited to fitting a base learner. It also covers validation, policy, lineage, and deployment-oriented logic.

Relevant areas:

- [`quant_trading_system/models/classical_ml.py`](quant_trading_system/models/classical_ml.py)
- [`quant_trading_system/models/deep_learning.py`](quant_trading_system/models/deep_learning.py)
- [`quant_trading_system/models/purged_cv.py`](quant_trading_system/models/purged_cv.py)
- [`quant_trading_system/models/meta_labeling.py`](quant_trading_system/models/meta_labeling.py)
- [`quant_trading_system/models/expected_edge_policy.py`](quant_trading_system/models/expected_edge_policy.py)
- [`quant_trading_system/models/target_engineering.py`](quant_trading_system/models/target_engineering.py)
- [`quant_trading_system/models/training_lineage.py`](quant_trading_system/models/training_lineage.py)

Supported training families from the CLI include:

- `xgboost`
- `lightgbm`
- `lightgbm_ranker`
- `xgboost_regressor`
- `lightgbm_regressor`
- `random_forest`
- `elastic_net`
- `lstm`
- `transformer`
- `tcn`
- `ensemble`

### 4. Trading Runtime

The production-critical runtime path is built inside [`quant_trading_system/trading/`](quant_trading_system/trading/).

Key components:

- [`quant_trading_system/trading/trading_engine.py`](quant_trading_system/trading/trading_engine.py)
- [`quant_trading_system/trading/signal_generator.py`](quant_trading_system/trading/signal_generator.py)
- [`quant_trading_system/trading/portfolio_manager.py`](quant_trading_system/trading/portfolio_manager.py)

This layer is responsible for:

- market-session lifecycle
- signal routing
- portfolio targeting
- integrating external signal sources
- coordinating risk and execution boundaries

### 5. Risk Layer

Risk controls are not optional wrappers. They are part of the runtime contract.

Key files:

- [`quant_trading_system/risk/limits.py`](quant_trading_system/risk/limits.py)
- [`quant_trading_system/risk/drawdown_monitor.py`](quant_trading_system/risk/drawdown_monitor.py)

Core behavior includes:

- pre-trade checks
- buying power and exposure controls
- concentration limits
- kill switch logic
- drawdown-based trading halts
- explicit reason-coded failures

### 6. Execution Layer

Execution is modeled as its own boundary rather than a direct broker API call from strategy code.

Key files:

- [`quant_trading_system/execution/order_manager.py`](quant_trading_system/execution/order_manager.py)
- [`quant_trading_system/execution/alpaca_client.py`](quant_trading_system/execution/alpaca_client.py)
- [`quant_trading_system/execution/position_tracker.py`](quant_trading_system/execution/position_tracker.py)

Responsibilities include:

- order lifecycle management
- submission and cancellation
- partial fill handling
- broker integration
- position reconciliation
- execution-quality-aware simulation settings

### 7. Backtest, Replay, And Promotion

Backtest and replay are first-class parts of the system, not afterthoughts.

Key areas:

- [`quant_trading_system/backtest/`](quant_trading_system/backtest/)
- [`quant_trading_system/backtest/promotion.py`](quant_trading_system/backtest/promotion.py)
- [`scripts/backtest.py`](scripts/backtest.py)
- [`scripts/replay.py`](scripts/replay.py)

The important design choice here is that replay and paper trading can consume the same promotion package that came out of training.

That means:

- the same model family
- the same thresholds
- the same policy layer
- the same runtime assumptions

are reused across environments instead of being reimplemented manually.

## Repository Layout

```text
quant_trading_system/
  alpha/        Alpha generation and regime-aware combination
  backtest/     Historical simulation, replay, promotion adapters
  config/       Runtime settings and environment loading
  core/         Events, system integration, shared contracts
  data/         Market data access, live feeds, timeframe utilities
  database/     PostgreSQL and TimescaleDB integration
  execution/    Order manager, broker boundary, execution analytics
  features/     Technical, statistical, cross-sectional, reference, flow features
  models/       Training, validation, calibration, policy, lineage
  monitoring/   Structured logging, audit trail, health telemetry
  risk/         Limits, drawdown monitoring, sizing, portfolio controls
  trading/      Trading engine, signal generation, portfolio manager

scripts/
  train.py      Standalone model training CLI
  backtest.py   Historical backtest CLI
  replay.py     Deterministic replay CLI
  trade.py      Paper/live trading CLI

docs/
  LIGHTGBM_TCN_WORK_PLAN.md

dashboard_ui/
  Optional dashboard frontend
```

## Unified CLI

The main operator surface is [`main.py`](main.py).

```bash
python main.py --help
```

Top-level commands:

- `trade`: execute paper, live, or dry-run trading
- `backtest`: historical simulation
- `replay`: deterministic replay with execution/risk SLO checks
- `train`: model training and promotion artifact generation
- `data`: load, bootstrap, validate, migrate, export data
- `features`: compute or validate feature sets
- `health`: diagnostics and health checks
- `deploy`: deployment and setup helpers
- `dashboard`: run the monitoring dashboard

## Common Workflows

### Health Check

```bash
python main.py health check --full
```

### Check Database Status

```bash
python main.py data db-status
```

### Train A Serious Candidate

Example promotion-style LightGBM run:

```bash
python main.py train \
  --model lightgbm_ranker \
  --training-profile promotion \
  --dataset-snapshot-bundle <PATH_TO_DATASET_BUNDLE_MANIFEST> \
  --strict-snapshot-replay \
  --timeframe 15Min \
  --timeframes 15Min 1Hour \
  --n-splits 5 \
  --n-trials 120 \
  --min-confidence-position-scale 0.20
```

### Backtest From A Promotion Package

```bash
python main.py backtest \
  --start 2025-01-01 \
  --end 2025-03-31 \
  --symbols AAPL MSFT NVDA \
  --promotion-package models/<model>.promotion_package.json \
  --timeframe 15Min
```

### Replay The Same Artifact

```bash
python main.py replay \
  --start 2025-01-01 \
  --end 2025-01-10 \
  --symbols AAPL MSFT NVDA \
  --promotion-package models/<model>.promotion_package.json \
  --timeframe 15Min
```

### Paper Trade The Same Artifact

```bash
python main.py trade \
  --mode paper \
  --promotion-package models/<model>.promotion_package.json \
  --max-positions 6 \
  --kill-switch-drawdown 0.08
```

The intended operator contract is:

1. Train on a frozen snapshot.
2. Inspect artifacts and validation gates.
3. Replay the promoted artifact.
4. Paper trade the exact same artifact.
5. Only then consider live rollout.

## Environment And Setup

Baseline requirements:

- Python `3.11+`
- PostgreSQL / TimescaleDB
- Redis
- Alpaca credentials for broker integration

Local install:

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e ".[dev]"
```

Configuration:

- copy [`.env.example`](.env.example) to `.env`
- review [`quant_trading_system/config/settings.py`](quant_trading_system/config/settings.py)
- keep PostgreSQL as the primary runtime source of truth

Operational note:

- serious training runs should be executed from the Linux filesystem inside WSL
- use the `~/AlphaTrade` copy for heavy training instead of `/mnt/c/...`

## Technology Stack

Main technologies currently in the repo:

- Python `3.11+`
- PostgreSQL / TimescaleDB
- Redis
- pandas, Polars, NumPy, PyArrow
- scikit-learn
- XGBoost
- LightGBM
- PyTorch and PyTorch Lightning
- FastAPI and WebSocket-related runtime components
- Alpaca broker integration
- pytest, mypy, black, isort, ruff

## Validation And Quality

The project is designed around targeted validation, not just "it runs once."

Typical validation surfaces include:

- unit tests
- integration tests
- deterministic replay
- runtime health checks
- training validation gates

Examples:

```bash
pytest tests/unit/test_settings.py -v
pytest tests/unit/test_trading.py tests/unit/test_risk.py tests/unit/test_execution.py -v
pytest tests/unit/test_backtest.py tests/unit/test_backtest_replay.py -v
pytest tests/unit/test_features.py tests/unit/test_models.py tests/unit/test_purged_cv.py -v
python main.py health check --full
```

When working on the current model stack, the most common targeted suites are:

```bash
pytest tests/unit/test_train_script.py tests/unit/test_backtest_script.py tests/unit/test_trade_script.py -v
pytest tests/unit/test_backtest_replay.py tests/unit/test_target_engineering.py -v
```

## Key Design Decisions

There are a few choices that define the project:

### Snapshot-First Training

The system prefers frozen dataset bundles and replayable manifests instead of loosely defined experiments.

### Artifact-Backed Deployment

Training output is expected to survive into backtest, replay, and paper/live trading through the promotion package contract.

### Policy Layer Above Base Predictions

The runtime decision is not just a raw class probability. It can also include:

- calibration
- meta-labeling
- expected-edge gating
- regime-aware policy adjustments
- side-specific controls
- universe eligibility rules

### Explicit Live Guardrails

Live behavior is expected to preserve:

- pre-trade checks
- drawdown halts
- kill switch logic
- structured logging
- auditability

## Files To Read First

If you are trying to understand the production-critical path, start with:

- [`quant_trading_system/config/settings.py`](quant_trading_system/config/settings.py)
- [`quant_trading_system/core/events.py`](quant_trading_system/core/events.py)
- [`quant_trading_system/core/system_integrator.py`](quant_trading_system/core/system_integrator.py)
- [`quant_trading_system/trading/trading_engine.py`](quant_trading_system/trading/trading_engine.py)
- [`quant_trading_system/trading/signal_generator.py`](quant_trading_system/trading/signal_generator.py)
- [`quant_trading_system/risk/limits.py`](quant_trading_system/risk/limits.py)
- [`quant_trading_system/risk/drawdown_monitor.py`](quant_trading_system/risk/drawdown_monitor.py)
- [`quant_trading_system/execution/order_manager.py`](quant_trading_system/execution/order_manager.py)
- [`quant_trading_system/execution/alpaca_client.py`](quant_trading_system/execution/alpaca_client.py)
- [`quant_trading_system/data/live_feed.py`](quant_trading_system/data/live_feed.py)
- [`quant_trading_system/data/data_access.py`](quant_trading_system/data/data_access.py)
- [`quant_trading_system/monitoring/audit.py`](quant_trading_system/monitoring/audit.py)
- [`quant_trading_system/monitoring/logger.py`](quant_trading_system/monitoring/logger.py)

## Related Documentation

- Work plan: [`docs/LIGHTGBM_TCN_WORK_PLAN.md`](docs/LIGHTGBM_TCN_WORK_PLAN.md)
- Production task list: [`docs/LIGHTGBM_TCN_PRODUCTION_TASK_LIST.md`](docs/LIGHTGBM_TCN_PRODUCTION_TASK_LIST.md)
- Agent instructions: [`AGENTS.md`](AGENTS.md)
- Project configuration and toolchain: [`pyproject.toml`](pyproject.toml)

## Current Status

The current focus of the repository is:

- repair dataset quality and training diagnostics around the frozen snapshot workflow
- produce a promotion-ready LightGBM baseline that clears institutional gates
- validate it through replay and early paper trading before reopening any TCN branch

If you want the fastest way to understand the project:

1. read the work plan
2. inspect [`quant_trading_system/`](quant_trading_system/)
3. run `python main.py --help`
