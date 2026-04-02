# AlphaTrade Production-Ready Training Task List

**Date:** 2026-04-02
**Purpose:** Convert the current diagnostic-grade LightGBM workflow into a production-ready training pipeline that can produce a promotable artifact, survive replay, and enter paper trading with traceable evidence.

## 1. Exit Criteria

This task list is complete only when all of the following are true:

- the active serious snapshot quality report passes
- the leading research candidate trains `expected-edge` successfully
- the leading research candidate clears the pre-promotion checklist in the work plan
- the promotion run produces `deployment_plan.ready_for_production=true`
- replay passes on the promoted artifact
- paper trading remains coherent for at least `2` weeks

## 2. P0 Blockers

### Task 1: Repair Snapshot Data Quality

Goal:

- remove the missing-bar failure that currently keeps the serious snapshot at diagnostic grade

Primary files:

- `quant_trading_system/data/data_access.py`
- `scripts/data.py`
- `scripts/train.py`
- `quant_trading_system/models/training_lineage.py`
- `quant_trading_system/models/symbol_quality.py`

Deliverables:

- root-cause summary for missing 15-minute bars by symbol and date range
- repaired dataset or documented universe adjustment
- new snapshot bundle with `quality_report_passed=true`

Done when:

- `models/snapshots/<new_snap>.quality.json` has no threshold breach
- the dropped-symbol list is empty or explicitly approved in the plan

### Task 2: Make Long-Running Training Durable

Goal:

- stop losing promotion runs to shell/session interruption

Primary files:

- `scripts/launch_wave1_wsl_run*.sh`
- `scripts/train.py`

Deliverables:

- durable WSL launcher guidance using `tmux`, `screen`, or equivalent
- persistent Optuna study or resumable training state for research and promotion profiles
- explicit run-resume procedure in docs or scripts

Done when:

- a killed session can be resumed or safely restarted from auditable intermediate state
- operators no longer rely on an attached foreground shell for multi-hour runs

### Task 3: Add Run Manifest Indexing

Goal:

- make every serious run discoverable without digging through raw stdout logs

Primary files:

- `scripts/train.py`
- `quant_trading_system/models/training_lineage.py`

Deliverables:

- append-only run index containing command, snapshot, timestamps, output paths, final status, and top metrics
- explicit interrupted/failed/success states

Done when:

- the answer to "which run produced which artifact and why did it fail" is available from one machine-readable file

### Task 4: Add Signal-Funnel Diagnostics

Goal:

- find where candidate trades are being suppressed

Primary files:

- `scripts/train.py`
- `quant_trading_system/models/signal_policy.py`
- `quant_trading_system/models/expected_edge_policy.py`

Deliverables:

- per-run artifact with counts at each stage:
  - raw model predictions
  - calibrated predictions
  - side-policy admissions
  - regime-policy admissions
  - expected-edge candidates
  - final selected trades
- counts split by symbol, side, and regime

Done when:

- the team can point to the exact stage causing `mean_trade_count` collapse and `expected-edge` starvation

### Task 5: Add Per-Symbol And Per-Regime Tail Diagnostics

Goal:

- identify why `holdout_symbol_sharpe_p25` is deeply negative while aggregate holdout Sharpe stays positive

Primary files:

- `scripts/train.py`
- `quant_trading_system/models/training_lineage.py`

Deliverables:

- holdout table by symbol with Sharpe, drawdown, trade count, hit rate, PnL, and underwater ratio
- holdout table by regime with the same metrics
- top contributors to tail losses

Done when:

- the worst symbols and regimes are explicit and can be tied to concrete challenger ideas

### Task 6: Make Feature Selection Auditable And Binding

Goal:

- stop treating feature selection as enabled when it effectively does nothing

Primary files:

- `scripts/train.py`

Deliverables:

- artifact fields for input feature count, retained feature count, rejected feature count, and rejection reasons
- family-level breakdown for retained vs rejected features
- warning when configured feature selection leaves the matrix unchanged

Done when:

- every run makes it obvious whether feature selection constrained the model or not

## 3. P1 Training Improvements

### Task 7: Establish The Clean-Snapshot Ranker Baseline

Goal:

- rerun `lightgbm_ranker h12` on the repaired snapshot with no hidden policy changes

Primary files:

- `scripts/launch_wave1_wsl_run1.sh`
- `scripts/train.py`

Deliverables:

- baseline artifact package on the clean snapshot
- updated training matrix and champion snapshot

Done when:

- the new baseline can be compared apples-to-apples against the old `h12` result

### Task 8: Build A Trade-Flow Challenger

Goal:

- increase candidate trade flow enough for expected-edge training and `min_trades` without wrecking holdout quality

Primary files:

- `scripts/train.py`
- `quant_trading_system/models/signal_policy.py`
- `quant_trading_system/models/expected_edge_policy.py`

Allowed changes:

- one admission-policy change at a time
- one threshold-policy change at a time
- no simultaneous feature-policy and label-policy rewrite in the same challenger

Done when:

- `expected-edge` trains successfully and trade-count improvement is measured, not guessed

### Task 9: Build A Drawdown-Control Challenger

Goal:

- reduce drawdown, PBO, and weak statistical validity on the same snapshot

Primary files:

- `scripts/train.py`

Likely levers:

- stronger feature pruning
- stronger tail-risk and CVaR weights
- stronger symbol-concentration penalties
- stronger symbol-tail exclusions if diagnostics justify them

Done when:

- at least one failed gate from `max_drawdown`, `risk_adjusted_positive`, `max_pbo`, or `max_white_reality_pvalue` materially improves without a hidden snapshot change

### Task 10: Add Candidate-Trade Floor Reporting For Expected-Edge

Goal:

- make the expected-edge skip reason actionable instead of a late warning

Primary files:

- `quant_trading_system/models/expected_edge_policy.py`
- `scripts/train.py`

Deliverables:

- explicit candidate-trade floor metric before expected-edge fitting
- stage-by-stage reason codes for why candidates were filtered out

Done when:

- no one has to infer why expected-edge was skipped from downstream symptoms

### Task 11: Tighten Research Run Comparison Discipline

Goal:

- make the next comparisons scientifically useful

Primary files:

- `scripts/train.py`
- `docs/LIGHTGBM_TCN_WORK_PLAN.md`

Deliverables:

- one-variable-at-a-time challenger policy
- snapshot hash recorded in every comparison summary
- explicit baseline vs challenger diff report

Done when:

- each serious run has a single clear thesis and a measurable pass/fail result

## 4. P1 Promotion Readiness Tasks

### Task 12: Add A Pre-Promotion Checklist Gate

Goal:

- stop wasting promotion compute on obviously unready research candidates

Primary files:

- `scripts/train.py`
- `scripts/launch_wave1_wsl_run3.sh`

Deliverables:

- lightweight pre-promotion checklist based on the work plan thresholds
- operator-visible message explaining why promotion is blocked

Done when:

- promotion is only launched for a research winner that is plausibly promotable

### Task 13: Rerun Promotion On The Winner

Goal:

- produce a real promotion artifact from the repaired pipeline

Primary files:

- `scripts/launch_wave1_wsl_run3.sh`
- `scripts/train.py`

Deliverables:

- complete promotion run
- model artifact
- promotion package
- replay manifest
- updated champion snapshot with an eligible candidate

Done when:

- `deployment_plan.ready_for_production=true`
- the run finishes cleanly and is indexed as success

## 5. P2 Deployment Validation Tasks

### Task 14: Replay Validation Across Three Windows

Goal:

- prove that the promoted artifact survives multiple market regimes

Primary files:

- `scripts/replay.py`
- `quant_trading_system/backtest/replay.py`

Deliverables:

- replay reports for the three required windows
- execution SLO review including fills, slippage, turnover, concentration, and risk warnings

Done when:

- replay outcomes are coherent across all required windows and no regime-specific failure invalidates promotion

### Task 15: Paper Trading Validation

Goal:

- validate the promoted artifact under deployment-like behavior before any live canary expansion

Primary files:

- `scripts/trade.py`
- `quant_trading_system/trading/trading_engine.py`
- `quant_trading_system/execution/order_manager.py`

Deliverables:

- at least `2` weeks of artifact-driven paper trading
- daily monitoring pack with PnL, turnover, slippage, symbol concentration, risk warnings, and expected-edge pass behavior

Done when:

- paper performance is operationally coherent and does not contradict replay or promotion assumptions

### Task 16: Canary Rollout Readiness Review

Goal:

- confirm that the promoted model is fit for controlled capital exposure

Primary files:

- promotion package artifacts
- replay outputs
- paper-trading logs

Deliverables:

- explicit decision memo for `5% -> 15% -> 35% -> 100%` rollout phases
- kill-switch and TCA guardrail review against the generated deployment plan

Done when:

- the team can justify why the model deserves canary capital instead of more research work

## 6. Deferred Tasks

### Task 17: Reopen The TCN Branch Only After LightGBM Stabilizes

Goal:

- avoid turning model-family experimentation into a substitute for unresolved baseline issues

Primary files:

- `docs/LIGHTGBM_TCN_WORK_PLAN.md`
- future TCN launcher scripts

Done when:

- LightGBM has already passed promotion, replay, and early paper validation and the team still wants an incremental challenger

## 7. Recommended Implementation Order

1. Tasks 1-6
2. Task 7
3. Tasks 8-11
4. Tasks 12-13
5. Tasks 14-16
6. Task 17 only after the baseline is stable
