# AlphaTrade Production-Ready Training Task List

**Date:** 2026-04-02
**Last Updated:** 2026-04-02
**Purpose:** Convert the current diagnostic-grade LightGBM workflow into a production-ready training pipeline that can produce a promotable artifact, survive replay, and enter paper trading with traceable evidence.

## 1. Exit Criteria

This task list is complete only when all of the following are true:

- the active serious snapshot quality report passes
- the leading research candidate trains `expected-edge` successfully
- the leading research candidate clears the pre-promotion checklist in the work plan
- the promotion run produces `deployment_plan.ready_for_production=true`
- replay passes on the promoted artifact
- paper trading remains coherent for at least `2` weeks

## 2. Current Status

### Completed

- Task 2: Make Long-Running Training Durable
- Task 3: Add Run Manifest Indexing
- Task 4: Add Signal-Funnel Diagnostics
- Task 5: Add Per-Symbol And Per-Regime Tail Diagnostics
- Task 6: Make Feature Selection Auditable And Binding
- Task 10: Add Candidate-Trade Floor Reporting For Expected-Edge
- Task 11: Tighten Research Run Comparison Discipline
- Task 12: Add A Pre-Promotion Checklist Gate

### In Progress

- Task 1: Repair Snapshot Data Quality

### Pending

- Task 7: Establish The Clean-Snapshot Ranker Baseline
- Task 8: Build A Trade-Flow Challenger
- Task 9: Build A Drawdown-Control Challenger
- Task 13: Rerun Promotion On The Winner
- Task 14: Replay Validation Across Three Windows
- Task 15: Paper Trading Validation
- Task 16: Canary Rollout Readiness Review

### Deferred

- Task 17: Reopen The TCN Branch Only After LightGBM Stabilizes

## 3. Completed Work

### Task 2: Make Long-Running Training Durable

Status:

- `completed`

What was implemented:

- WSL launcher flow now uses `tmux` through `scripts/wsl_tmux_launcher.sh`
- durable WSL launchers now require real `tmux`; there is no automatic fallback path
- `scripts/launch_wave1_wsl_run1.sh`, `scripts/launch_wave1_wsl_run2.sh`, and `scripts/launch_wave1_wsl_run3.sh` now launch through the durable wrapper
- Optuna studies are persisted to SQLite and can be resumed with the same `--name`
- work plan now documents the resume procedure

Primary files touched:

- `scripts/wsl_tmux_launcher.sh`
- `scripts/launch_wave1_wsl_run1.sh`
- `scripts/launch_wave1_wsl_run2.sh`
- `scripts/launch_wave1_wsl_run3.sh`
- `scripts/train.py`
- `docs/LIGHTGBM_TCN_WORK_PLAN.md`

Residual risk:

- durable compute is fixed, but promotion still should not run until a valid research winner exists

### Task 3: Add Run Manifest Indexing

Status:

- `completed`

What was implemented:

- append-only run event index added through `append_training_run_event(...)`
- training now records `started`, `completed`, `failed`, and `interrupted` states
- indexed payload includes run identity, snapshot, output paths, and top metrics

Primary files touched:

- `quant_trading_system/models/training_lineage.py`
- `scripts/train.py`

Residual risk:

- run index is now durable, but the underlying candidate quality is still insufficient

### Task 4: Add Signal-Funnel Diagnostics

Status:

- `completed`

What was implemented:

- expected-edge precheck funnel and full training funnel
- holdout funnel metrics
- symbol-level and regime-level funnel splits
- candidate floor metrics before expected-edge fitting

Primary files touched:

- `scripts/train.py`

Residual risk:

- diagnostics exist, but we have not yet used them to build and validate the next challenger

### Task 5: Add Per-Symbol And Per-Regime Tail Diagnostics

Status:

- `completed`

What was implemented:

- holdout symbol/regime payloads now include Sharpe, drawdown, trade count, hit rate, PnL, loss PnL, tail-loss PnL, underwater ratio, turnover, Calmar, and CVaR
- tail-loss contributor leaderboards are written for both symbol and regime

Primary files touched:

- `scripts/train.py`

Residual risk:

- tail diagnostics identify bad contributors, but we still need to turn that evidence into a challenger policy

### Task 6: Make Feature Selection Auditable And Binding

Status:

- `completed`

What was implemented:

- feature-selection audit now records input count, retained count, rejected count, rejected features, rejection reasons, family breakdown, and no-op detection
- training warns when configured feature selection leaves the matrix effectively unchanged

Primary files touched:

- `scripts/train.py`

Residual risk:

- feature selection is now auditable, but it still needs to be made more aggressively useful in the drawdown-control challenger

### Task 10: Add Candidate-Trade Floor Reporting For Expected-Edge

Status:

- `completed`

What was implemented:

- explicit candidate floor metrics are emitted before expected-edge fit
- training records whether the candidate floor passes `min_samples` and `min_coverage`
- skip reasons are visible without inferring them from downstream artifacts

Primary files touched:

- `scripts/train.py`

Residual risk:

- the pipeline now explains expected-edge starvation, but the research candidate still needs a real trade-flow fix

### Task 11: Tighten Research Run Comparison Discipline

Status:

- `completed`

What was implemented:

- training matrix rows now carry snapshot identity fields including snapshot hash lineage
- comparison summary now records baseline-vs-challenger diffs on the same snapshot identity
- work plan already enforces one-thesis-at-a-time challenger policy

Primary files touched:

- `scripts/train.py`
- `docs/LIGHTGBM_TCN_WORK_PLAN.md`

Residual risk:

- the comparison framework is ready, but it still needs new clean-snapshot runs to compare

### Task 12: Add A Pre-Promotion Checklist Gate

Status:

- `completed`

What was implemented:

- research outputs now produce a machine-readable pre-promotion checklist
- strict-snapshot promotion runs now preflight against the latest matching research matrix before training starts
- blocked promotion attempts now explain why they are blocked
- an explicit operator bypass exists for emergency/manual override

Primary files touched:

- `scripts/train.py`
- `scripts/launch_wave1_wsl_run3.sh`
- `docs/LIGHTGBM_TCN_WORK_PLAN.md`

Residual risk:

- the gate prevents wasted promotion compute, but it does not solve the underlying model-quality issues

## 4. In-Progress Work

### Task 1: Repair Snapshot Data Quality

Status:

- `in_progress`

What was implemented so far:

- PostgreSQL root-cause analysis showed most severe 15-minute "gaps" were sparse premarket-to-open / close-boundary rows being treated as missing regular-session bars
- training now filters intraday OHLCV to regular session by default before symbol-quality scoring and snapshot quality reporting
- snapshot auto-reuse scope now includes the market-session policy so old extended-hours snapshots cannot be silently reused
- `scripts/data.py` gap diagnostics now use the same session-filtered missing-bar logic as training
- data-quality reporting now includes root-cause style missing-bar summaries by symbol and date range
- training logs now surface the top missing-bar symbols and the largest missing-bar window
- quality-report lineage is already wired into training, snapshots, and benchmark comparisons
- training now supports `--snapshot-only` so operators can materialize the dataset snapshot bundle, quality report, and dropped-symbol review artifact before any Optuna/model run
- snapshot-only runs now emit `<model_name>.snapshot_review.json` with explicit `data_quality_passed` and `no_silent_symbol_drop` checks
- snapshot-only preflight now fast-fails immediately after Phase 1 when the quality report fails or symbols drop from the effective universe, so Task 1 can be checked without paying feature-compute cost
- market-session preflight filtering was vectorized in the package layer, so the same serious snapshot review now completes in seconds instead of stalling inside Phase 1

Primary files touched so far:

- `quant_trading_system/data/data_access.py`
- `quant_trading_system/models/training_lineage.py`
- `scripts/data.py`
- `scripts/train.py`

What is still required:

- rerun serious snapshot generation and confirm the regular-session quality report now passes
- use `python main.py train ... --snapshot-only --disable-auto-snapshot-reuse` for the first post-fix rebuild so the snapshot review package is produced before any research rerun
- verify whether any symbols still show true intraday gaps after the session-policy fix
- latest preflight evidence: `wave1_snapshotonly_20260402_h12_preflight3` finished in `18.6s`, `data_quality_passed=true`, but snapshot review still failed because `GLD` was dropped by the symbol-quality gate for `median_dollar_volume`
- decide whether remaining real gaps require:
  - repairing data in-place
  - backfilling missing windows
  - explicitly retiring a broken symbol from the universe
- decide whether `GLD` should be explicitly retired from the Wave 1 universe or whether the liquidity threshold should be changed with written justification
- regenerate a serious snapshot with `quality_report_passed=true`
- make sure the dropped-symbol list is empty or explicitly approved

Done when:

- `models/snapshots/<new_snap>.quality.json` has no threshold breach
- the next research run can use a clean snapshot without silent universe degradation

## 5. Remaining Work

### Task 7: Establish The Clean-Snapshot Ranker Baseline

Status:

- `pending`

Blocked by:

- Task 1

Next action:

- rerun `lightgbm_ranker h12` on the repaired snapshot with no hidden policy changes

Done when:

- the clean-snapshot baseline can be compared directly against the old `h12` result

### Task 8: Build A Trade-Flow Challenger

Status:

- `pending`

Blocked by:

- Task 1
- Task 7

Next action:

- use the new funnel diagnostics to change one admission/threshold lever at a time and recover enough candidate trade flow for expected-edge training

Done when:

- `expected-edge` trains successfully and trade-count improvement is measurable on the clean snapshot

### Task 9: Build A Drawdown-Control Challenger

Status:

- `pending`

Blocked by:

- Task 1
- Task 7

Next action:

- use the new tail diagnostics and feature-selection audit to tighten pruning, downside penalties, and symbol-tail protection without changing snapshots

Done when:

- at least one failed gate from drawdown/PBO/White-Reality materially improves on the same snapshot

### Task 13: Rerun Promotion On The Winner

Status:

- `pending`

Blocked by:

- Tasks 1, 7, 8, 9

Next action:

- promote only the clean-snapshot research winner that clears the pre-promotion checklist

Done when:

- the promotion run finishes cleanly and produces model artifact, promotion package, replay manifest, and `deployment_plan.ready_for_production=true`

### Task 14: Replay Validation Across Three Windows

Status:

- `pending`

Blocked by:

- Task 13

Next action:

- replay the promoted artifact across the three required windows and compare execution realism against promotion assumptions

### Task 15: Paper Trading Validation

Status:

- `pending`

Blocked by:

- Task 14

Next action:

- run at least `2` weeks of artifact-driven paper trading with daily monitoring packs

### Task 16: Canary Rollout Readiness Review

Status:

- `pending`

Blocked by:

- Tasks 14 and 15

Next action:

- write the capital rollout memo and reconcile deployment guardrails against replay and paper evidence

## 6. Deferred Work

### Task 17: Reopen The TCN Branch Only After LightGBM Stabilizes

Status:

- `deferred`

Rule:

- TCN stays closed until LightGBM passes promotion, replay, and early paper validation

## 7. Recommended Next Implementation Order

1. Finish Task 1 by fixing the actual 15-minute gap source and generating a clean snapshot
2. Run Task 7 to create the clean-snapshot `lightgbm_ranker h12` baseline
3. Run Task 8 using the existing signal-funnel diagnostics
4. Run Task 9 using the existing tail diagnostics and feature-selection audit
5. Run Task 13 only after the research winner clears the checklist
6. Run Tasks 14-16 in order
7. Keep Task 17 deferred unless the LightGBM baseline is already operationally stable
