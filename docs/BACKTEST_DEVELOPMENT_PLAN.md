# AlphaTrade Backtest Development Plan

**Date:** 2026-04-01
**Status:** Ready to execute
**Scope:** Turn AlphaTrade backtesting into an artifact-driven, execution-realistic, promotion-grade validation system

## 0. Executive Summary

AlphaTrade does not currently have a true `train -> backtest -> replay -> paper/live` validation chain.

The biggest gap is structural:

- training produces models, meta-label artifacts, thresholds, and promotion packages
- backtest mostly runs alpha-factor strategies instead of promoted model artifacts
- replay still uses a deterministic return-threshold strategy instead of the promoted model package
- live portfolio sizing and backtest sizing do not share the same decision semantics

This means the system can train strong models without ever proving that the exact trained decision policy survives a realistic event-driven backtest and replay.

The plan below fixes that by making one artifact package the single source of truth for:

- feature schema
- selected features
- threshold policy
- meta-label policy
- horizon and exit semantics
- cost assumptions
- execution and sizing rules

The target is not "guaranteed daily profit." That is not a credible institutional objective.

The target is:

- positive expectancy after costs
- stable behavior across regimes
- deterministic replayability
- promotion gates based on the same decision logic that paper/live trading will actually use

## 1. Objective

Build a production-grade backtest and replay stack that:

- consumes the exact promoted training artifacts
- reproduces the same feature, threshold, and filtering logic used during model evaluation
- simulates realistic execution, costs, and holding constraints
- exposes the same portfolio construction and risk semantics used in live trading
- produces promotion decisions that remain valid in paper trading and shadow/live monitoring

## 2. Current State Assessment

### 2.1 Structural Gaps

The current backtest entrypoint in `scripts/backtest.py` still behaves like an alpha-strategy runner, not a model-artifact runner.

Observed gaps:

- "Model preparation" is documented in the workflow, but the runnable path still computes features and combines alpha factors.
- The CLI does not require or validate a trained model artifact, meta-model artifact, or promotion package.
- `SignalBasedStrategy` uses a fixed `signal_threshold=0.1` and a generic `horizon=1`.
- Replay uses `DeterministicReplayStrategy` rather than a promoted model package.

### 2.2 Train vs Backtest Parity Gaps

Training already contains materially more advanced logic than the backtest path:

- dynamic long and short threshold derivation from leakage-safe OOF probabilities
- meta-label training and confidence filtering
- cost-aware triple-barrier labels
- synthetic execution scoring for fast research loops
- promotion package generation with threshold metadata

Backtest currently loses most of that structure:

- thresholds revert to fixed values
- meta-label filter is not the decision gate
- triple-barrier semantics are not faithfully enforced as exit rules
- feature policy is narrower than the training pipeline

### 2.3 Execution and Sizing Gaps

Training, backtest, and live do not currently share one portfolio decision policy.

Current mismatch:

- training uses synthetic net-return estimation for fast scoring
- live uses `PortfolioManager` and confidence-aware weight construction
- event-driven backtest still opens positions using a mostly fixed `max_position_pct` sizing rule

This creates a severe promotion risk:

- a model may look good in training metrics
- the backtest may still not reflect the same position sizing
- live may then behave differently from both

### 2.4 Metrics and Measurement Gaps

The current annualization assumptions are not aligned with intraday strategies.

Key issue:

- backtest analytics default to `periods_per_year=252`
- training holdout Sharpe uses `sqrt(252)`
- this is not correct for `15Min` or other intraday bars

Result:

- Sharpe and related annualized metrics are mis-scaled
- research ranking and promotion thresholds can be distorted

### 2.5 Cost and Liquidity Gaps

Training and backtest do not yet share one canonical cost world.

Current issues:

- training labels and synthetic execution use spread, slippage, and impact assumptions
- backtest simulator separately models commission, spread, and slippage
- symbol-specific liquidity, ADV, and session-aware spread behavior are not yet the default promotion path

### 2.6 Operational Risk

Without parity, the system can drift into a dangerous state where:

- research optimizes one decision policy
- backtest validates another
- replay exercises a third
- live trading executes a fourth

That is not an institutional workflow.

## 3. Target End State

AlphaTrade should converge on the following validation chain:

1. `train`
2. `promotion_package`
3. artifact-driven event backtest
4. artifact-driven replay
5. shadow or paper trading
6. controlled production rollout

Each stage should consume the same package-level contract.

The end-state contract should include:

- primary model artifact
- meta-label model artifact if enabled
- feature schema hash
- selected feature list
- timeframe and multi-timeframe configuration
- threshold policy and realized thresholds
- cost model assumptions
- position sizing policy
- risk and holding constraints
- promotion metrics and validation gates

## 4. Non-Negotiable Design Principles

1. One promoted artifact package must be the source of truth for evaluation.
2. Fast synthetic execution scoring in training is allowed for research, but promotion decisions must come from event-driven backtest and replay.
3. Feature engineering parity is mandatory for promoted models.
4. Thresholding, meta-labeling, and exits must be package-driven, not hardcoded in scripts.
5. Backtest must move closer to live semantics, not farther from them.
6. Intraday metrics must be timeframe-aware and session-aware.
7. Capacity, liquidity, and turnover discipline must be first-class promotion gates.
8. Replay must explain exactly why a model was promotable or rejectable on a concrete market window.

## 5. Target Architecture

### 5.1 Promotion Package as the Core Contract

Extend the current promotion package so it becomes the standard input for backtest and replay.

The package should explicitly carry:

- artifact paths and model family metadata
- feature selection outputs
- selected timeframes
- label horizon and max holding rules
- long and short threshold policy
- meta-label configuration
- calibration metadata
- canonical cost model
- execution profile and position sizing policy
- promotion gate results
- lineage metadata such as dataset snapshot, feature schema hash, and code version

### 5.2 Artifact-Driven Model Prediction Adapter

Create one adapter layer that can:

- load the promoted model package
- validate feature schema compatibility
- materialize the same feature frame used in training
- generate primary probabilities or scores
- apply calibration and meta-label filtering
- emit trade intents in the same format expected by backtest, replay, and live orchestration

This adapter should be reusable across:

- `scripts/backtest.py`
- `quant_trading_system/backtest/replay.py`
- live or shadow prediction ingestion paths

### 5.3 Portfolio and Risk Parity Layer

Backtest should stop pretending that execution begins at fixed `max_position_pct`.

Instead, the same portfolio construction semantics should be shared across backtest and live:

- confidence-aware allocation
- volatility scaling
- concentration controls
- max exposure caps
- turnover discipline
- no-trade and risk-halt behavior

The cleanest target is to reuse or extract common logic from `quant_trading_system/trading/portfolio_manager.py`.

### 5.4 Exit and Holding Semantics

The backtest engine should enforce the same economics implied by labels and promotion logic.

Required rules:

- max holding bars
- stop loss
- take profit
- opposite-signal exits
- optional end-of-session flattening where strategy profile requires it
- explicit gap behavior and stale-data handling

Triple-barrier concepts should not remain only in labeling. They should inform the actual evaluation engine.

### 5.5 Cost and Liquidity Layer

Promotion-grade backtest should include:

- per-symbol spread models
- per-symbol ADV participation limits
- session-bucket slippage assumptions
- volatility-aware impact
- trade rejection or haircut when expected edge is smaller than estimated execution cost

### 5.6 Replay as a Governance Tool

Replay should become a deterministic audit layer for promoted models, not a separate toy strategy.

Replay goals:

- reproduce the backtest decision path on a chosen historical window
- surface model score, threshold, meta-label decision, risk decision, and final execution intent bar by bar
- provide audit-friendly explanations for every accepted and rejected trade

## 6. Delivery Plan

### Phase 1: Artifact Contract Hardening

**Goal:** Make promotion packages sufficient to drive backtest and replay without hidden defaults.

Work items:

- expand promotion package schema in `scripts/train.py`
- store selected features, schema hash, timeframes, cost model, execution profile, and thresholds explicitly
- persist the effective meta-label policy and calibration metadata
- add package versioning so old artifacts fail clearly when incompatible

Primary touchpoints:

- `scripts/train.py`
- `quant_trading_system/models/target_engineering.py`

Acceptance criteria:

- every promoted model emits a self-sufficient promotion package
- package validation fails loudly if required fields are missing
- package contains enough information to rerun holdout evaluation without training code inference

### Phase 2: Artifact-Driven Backtest Runner

**Goal:** Turn `scripts/backtest.py` into a model backtest entrypoint instead of an alpha-only runner.

Work items:

- add CLI support for promotion package input
- load model and meta-model artifacts from the package
- rebuild features using the package feature schema
- generate model predictions instead of only factor alphas
- map model outputs into standardized trade intents
- keep alpha-only strategies as an explicit research mode, not the default promotion path

Primary touchpoints:

- `scripts/backtest.py`
- `quant_trading_system/alpha/ml_alphas.py`
- shared model-loading utility under `quant_trading_system/`

Acceptance criteria:

- backtest can run from a single promotion package path
- package-driven backtest reproduces training holdout thresholds and filtering policy
- operator can still run alpha-factor backtests explicitly, but they are separated from promoted-model validation

### Phase 3: Exit and Horizon Parity

**Goal:** Enforce the same hold and exit assumptions used by labels and promotion logic.

Work items:

- carry horizon and max-holding metadata into trade intents
- add explicit stop-loss and take-profit exit rules
- add configurable end-of-session flattening behavior
- add deterministic handling for opposite-signal exits and stale signals
- ensure open positions track bars held and exit reason codes

Primary touchpoints:

- `quant_trading_system/backtest/engine.py`
- `quant_trading_system/backtest/simulator.py`
- `quant_trading_system/models/target_engineering.py`

Acceptance criteria:

- position lifecycle explains why each trade was closed
- backtest exits are no longer governed only by opposite signals or generic engine behavior
- holdout event economics align with label semantics within defined tolerance

### Phase 4: Portfolio Construction and Risk Parity

**Goal:** Remove the execution mismatch between backtest and live.

Work items:

- extract or reuse common sizing logic from `PortfolioManager`
- support confidence-aware and volatility-aware sizing in backtest
- add capacity-aware and concentration-aware portfolio construction
- ensure risk rejections and drawdown protection are visible in backtest analytics
- add reason-coded no-trade and size-reduction events

Primary touchpoints:

- `quant_trading_system/trading/portfolio_manager.py`
- `quant_trading_system/backtest/engine.py`
- `quant_trading_system/risk/limits.py`
- `quant_trading_system/risk/drawdown_monitor.py`

Acceptance criteria:

- backtest position sizing is materially aligned with live portfolio behavior
- risk-driven trade rejections appear in backtest outputs
- allocation logic is no longer equivalent to "always deploy max position size"

### Phase 5: Cost, Liquidity, and Capacity Realism

**Goal:** Stop promoting strategies that only work under underpriced execution assumptions.

Work items:

- create a canonical cost model object used by training labels and backtest simulator
- calibrate spread and slippage by symbol, timeframe, and session bucket
- enforce ADV and participation limits
- add expected-edge-minus-cost gating before trade entry
- report capacity-aware net returns and liquidity haircut metrics

Primary touchpoints:

- `scripts/train.py`
- `quant_trading_system/models/target_engineering.py`
- `quant_trading_system/backtest/simulator.py`
- market-data and analytics helpers under `quant_trading_system/data/`

Acceptance criteria:

- training and backtest reference the same named cost model family
- symbol-level execution assumptions are inspectable in reports
- promotion gates can reject models whose gross alpha disappears after realistic cost haircuts

### Phase 6: Intraday Metrics and Analytics Repair

**Goal:** Make performance measurement correct for intraday systems.

Work items:

- derive `periods_per_year` from timeframe and session structure
- fix annualized Sharpe, Sortino, and related ratios in backtest analytics
- fix training holdout annualization so ranking is comparable to backtest
- separate bar-level, trade-level, daily, and session-level metrics clearly
- add calibration and abstention metrics for model-driven strategies

Primary touchpoints:

- `quant_trading_system/backtest/analyzer.py`
- `scripts/train.py`

Acceptance criteria:

- intraday annualization is no longer hardcoded to daily assumptions
- holdout and backtest reports reconcile on metric definitions
- reports clearly show trade count, activity rate, turnover, cost ratio, and calibration quality

### Phase 7: Replay Overhaul

**Goal:** Make replay a real promotion and debugging instrument.

Work items:

- replace deterministic replay strategy with artifact-driven replay
- emit per-bar decision traces for prediction, threshold, meta-label, risk, and execution
- add replay summaries that compare realized and expected edge
- add operator-friendly explanations for rejected trades

Primary touchpoints:

- `quant_trading_system/backtest/replay.py`
- `quant_trading_system/trading/trading_engine.py`
- `quant_trading_system/trading/signal_generator.py`
- `quant_trading_system/monitoring/audit.py`

Acceptance criteria:

- replay can consume the same promotion package as backtest
- operator can inspect why a trade was accepted, resized, delayed, or rejected
- replay becomes the preferred validation step before paper or shadow rollout

### Phase 8: Governance, Testing, and Promotion Gates

**Goal:** Prevent future parity drift.

Work items:

- add package parity tests for train vs backtest vs replay
- add schema compatibility tests for promotion packages
- add regression tests for intraday annualization
- add golden-window replay snapshots for deterministic verification
- add promotion gates for regime robustness, turnover discipline, and cost realism

Primary touchpoints:

- `tests/unit/test_backtest.py`
- `tests/unit/test_backtest_replay.py`
- `tests/unit/test_train_script.py`
- new parity and contract tests under `tests/`

Acceptance criteria:

- CI fails when promotion packages lose required fields
- CI fails when holdout and package-driven backtest materially diverge beyond agreed tolerances
- replay regression windows remain deterministic for promoted artifacts

## 7. Recommended Validation Matrix

Every material backtest change should be validated across four layers:

### 7.1 Contract Validation

- promotion package schema validation
- artifact existence and version compatibility
- feature schema hash validation

### 7.2 Behavioral Parity

- training holdout vs package-driven backtest comparison
- backtest vs replay trade-count comparison
- backtest vs replay reason-code comparison

### 7.3 Risk and Execution Validation

- turnover sanity checks
- expected-edge vs cost sanity checks
- exposure and concentration guardrails
- drawdown and kill-switch style conditions in replay

### 7.4 Regime Validation

- high-volatility window
- low-volatility window
- trend regime
- mean-reversion regime
- earnings or macro-event stressed windows for affected symbols

## 8. Promotion Metrics That Should Matter

Do not promote on raw Sharpe alone.

Promotion reports should emphasize:

- net Sharpe and net Sortino after realistic costs
- worst-regime Sharpe
- trade count and active-rate sufficiency
- turnover and average holding period
- expected edge vs realized execution cost ratio
- calibration quality such as Brier score or probability buckets
- concentration and symbol dependency
- liquidity haircut sensitivity
- replay stability on untouched windows

## 9. High-Impact Ideas

These are not cosmetic improvements. They can materially change the quality of the platform if executed well.

### 9.1 Portfolio-Level Ranking and Allocation

Move from independent symbol thresholding toward a portfolio allocator that ranks all available opportunities at each decision step.

Benefits:

- capital flows to the best opportunities instead of every symbol crossing a local threshold
- portfolio concentration becomes explicit instead of accidental
- turnover can be optimized at the portfolio level

This is one of the highest-leverage changes for a multi-symbol intraday system.

### 9.2 Regime-Aware Routing

Introduce an explicit regime router that decides which model, threshold band, or trading policy should be active.

Possible regimes:

- trend
- mean reversion
- volatility shock
- low-liquidity or event-driven market

Benefits:

- fewer bad trades during regime mismatch
- better calibration stability
- cleaner explanation of when the strategy should abstain

### 9.3 Shadow Book and Decision Ledger

Create a persistent "decision ledger" that records, for every bar or decision event:

- model score
- calibrated probability
- threshold state
- meta-label decision
- risk decision
- intended size
- simulated fill assumptions
- realized outcome

Benefits:

- institutional-grade replay and auditability
- faster debugging of false positives and missed trades
- direct measurement of model alpha vs execution drag

This can become one of the most valuable internal assets in the project.

### 9.4 Execution Feedback Loop

Build a transaction-cost-analysis loop that feeds execution outcomes back into the model and promotion process.

Examples:

- slippage by symbol and session bucket
- fill quality by order type or volatility state
- realized impact vs expected impact

Benefits:

- promotion gates stop relying on stale static assumptions
- the system learns where it actually has capacity
- edge destruction from execution becomes measurable early

### 9.5 Abstention as a First-Class Alpha

Treat "do not trade" as a deliberate model capability, not a failure to find trades.

Implementation ideas:

- confidence-aware abstention
- cost-aware abstention
- regime-aware abstention
- low-liquidity abstention

Benefits:

- better net returns than forcing daily activity
- lower turnover
- less drawdown during hostile conditions

For an institutional intraday system, disciplined abstention is often more valuable than increasing raw trade count.

## 10. Practical Execution Order

### Week 1

- harden promotion package schema
- add package validation helpers
- add package input to `scripts/backtest.py`
- run first package-driven holdout backtest on one trained LightGBM artifact

### Week 2

- implement exit and horizon parity
- align backtest sizing with live portfolio construction
- fix intraday annualization
- add parity regression tests

### Week 3

- replace deterministic replay with package-driven replay
- add decision trace outputs and reason codes
- add cost and liquidity calibration reports
- define updated promotion gate thresholds

### Week 4

- add portfolio-level ranking prototype
- add regime router research branch
- stand up shadow book or decision ledger prototype
- run controlled paper-trading validation on one promoted model family

## 11. Definition of Done

This plan is complete only when all of the following are true:

- a promoted model can be backtested from its package without hidden training-time assumptions
- replay consumes the same package and produces explainable bar-level decision traces
- backtest position sizing is aligned with live portfolio behavior
- exit and holding semantics match the economics implied by the training labels
- intraday metrics are correctly annualized and consistently defined
- promotion gates reject models that fail after realistic cost and capacity assumptions
- parity tests prevent future drift between train, backtest, replay, and live semantics

## 12. Bottom Line

The next major step for AlphaTrade is not "more models."

The next major step is:

- one package
- one decision policy
- one realistic evaluation path
- one auditable replay path

Once that foundation is in place, model innovation will compound instead of fragmenting across incompatible evaluation layers.
