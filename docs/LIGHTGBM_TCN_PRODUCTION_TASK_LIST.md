# AlphaTrade Production-Ready Training Task List

**Date:** 2026-04-02
**Last Updated:** 2026-04-02
**Purpose:** Convert the clean-snapshot LightGBM workflow into a promotable research winner that can survive promotion, replay, paper trading, and canary rollout without hidden failure modes.

## 1. Exit Criteria

This task list is complete only when all of the following are true:

- the active frozen snapshot quality review passes with no silent symbol drop
- the clean-snapshot baseline is archived and reproducible on the approved bundle
- the leading research candidate no longer collapses to a no-trade holdout
- the leading research candidate trains `expected-edge` successfully on the approved bundle
- the leading research candidate clears the pre-promotion checklist with no bypass
- the promotion run produces `deployment_plan.ready_for_production=true`
- replay passes on the promoted artifact across the required windows
- paper trading remains coherent for at least `2` weeks
- the canary rollout readiness review is completed with explicit guardrails

## 2. Latest Evidence

### Clean Snapshot Evidence

- `run_id=wave1_ranker_research_20260402_wsl_run1_clean_core11_h12`
- `snapshot_id=snap_8803ebe20127c4fd`
- `dataset_bundle_hash=d6522c7784a43844004b17d910bf51e1ff3754888d349faaa9e491bdf254e880`
- `snapshot_review.ready=true`
- `data_quality_passed=true`
- `dropped_symbols=0`
- regular-session missing-bars ratio is now `0.00082575`, below the configured `0.0100` threshold

Implication:

- Task 1 is no longer the main blocker
- the old failed-quality snapshot `snap_76d371975b6e817d` is now diagnostic-only evidence and must not be reused for promotion

### Clean-Snapshot Baseline Evidence

- `run_id=wave1_ranker_research_20260402_runB_clean_core11_h12`
- `snapshot_id=snap_8803ebe20127c4fd`
- finished on 2026-04-02 with `exit code 1`
- artifact path: `models/wave1_ranker_research_20260402_runB_clean_core11_h12_artifacts.json`
- training matrix path: `models/benchmarks/training_matrix_20260402T135806Z.json`

Key metrics:

- `mean_sharpe=0.9197`
- `mean_trade_count=11.0`
- `mean_risk_adjusted_score=-2.2538`
- `pbo=0.5535`
- `white_reality_pvalue=0.1967`
- `holdout_sharpe=0.0`
- `holdout_trade_count=0.0`
- `holdout_active_signal_rate=0.0`
- `expected_edge candidate_count=0`

Failed gates:

- `max_drawdown`
- `min_trades`
- `risk_adjusted_positive`
- `max_pbo`
- `max_white_reality_pvalue`
- `min_holdout_sharpe`
- `holdout_sharpe_consistency`

Pre-promotion checklist blockers:

- `validation_layers_ready`
- `expected_edge_trained`
- `mean_trade_count`
- `mean_risk_adjusted_score`
- `pbo`
- `white_reality_pvalue`

Most important root-cause observation:

- the clean-snapshot baseline did not merely underperform; it produced a no-trade holdout
- `expected_edge_training_precheck_signal_funnel.zero_signal_count=200780`
- `expected_edge_training_precheck_signal_funnel.candidate_count=0`
- `holdout_trade_count=0`
- therefore `holdout_max_drawdown=0.0` and `holdout_symbol_sharpe_p25=0.0` are not genuine robustness wins; they are no-trade artifacts

### Comparison Against The Old Diagnostic Ranker

Old failed-quality ranker reference:

- `run_id=wave1_ranker_research_20260402_wsl_run1_gpucuda`
- `snapshot_id=snap_76d371975b6e817d`

Comparison summary:

- clean baseline fixed the snapshot problem
- clean baseline did **not** improve research quality enough to become promotable
- compared with the old diagnostic ranker:
  - `mean_sharpe` worsened from `1.3114` to `0.9197`
  - `mean_trade_count` stayed stuck at `11`
  - `mean_risk_adjusted_score` worsened from `-2.0554` to `-2.2538`
  - `white_reality_pvalue` worsened from `0.1156` to `0.1967`
  - holdout metrics that appear cleaner are contaminated by the no-trade collapse

## 3. Current Status

### Completed

- Task 1: Repair Snapshot Data Quality
- Task 2: Make Long-Running Training Durable
- Task 3: Add Run Manifest Indexing
- Task 4: Add Signal-Funnel Diagnostics
- Task 5: Add Per-Symbol And Per-Regime Tail Diagnostics
- Task 6: Make Feature Selection Auditable And Binding
- Task 7: Establish The Clean-Snapshot Ranker Baseline
- Foundation: Add Candidate-Trade Floor Reporting For Expected-Edge
- Foundation: Tighten Research Run Comparison Discipline
- Foundation: Add A Pre-Promotion Checklist Gate

### In Progress

- Task 8: Eliminate The No-Trade Holdout Failure Mode

### Pending

- Task 9: Freeze The Clean-Snapshot Operator Surface
- Task 10: Restore Candidate Trade Flow On The Clean Snapshot
- Task 11: Reduce Drawdown And Improve Risk-Adjusted Score
- Task 12: Lower PBO And White-Reality P-Value
- Task 13: Confirm A Research Winner On The Approved Bundle
- Task 14: Promotion On The Winner
- Task 15: Replay Validation Across Three Windows
- Task 16: Paper Trading Validation
- Task 17: Canary Rollout Readiness Review

### Deferred

- Task 18: Reopen The TCN Branch Only After LightGBM Stabilizes

## 4. Completed Foundation

The following foundation work is already in place and should be preserved:

- durable WSL execution through `tmux`
- Optuna state persisted under `models/optuna_state`
- append-only run event indexing
- snapshot-only review workflow
- signal funnel diagnostics
- candidate-floor reporting before expected-edge training
- symbol and regime tail diagnostics
- auditable feature selection
- same-bundle research comparison discipline
- machine-readable pre-promotion checklist
- clean approved universe in `data/training/universes/wave1_clean_core11_20260402.json`

## 5. Active And Remaining Tasks

### Task 8: Eliminate The No-Trade Holdout Failure Mode

Status:

- `in_progress`

Why this is next:

- the current clean-snapshot baseline can still look superficially stable while producing zero holdout trades
- production research cannot continue while zero-trade holdout behavior is allowed to masquerade as low drawdown or acceptable cross-symbol metrics

Required actions:

- make `holdout_trade_count=0` and `holdout_active_signal_rate=0` explicit hard-fail conditions in research validation and checklist reporting
- make `expected_edge candidate_count=0` an explicit blocker separate from downstream policy training failure
- emit raw-probability and calibrated-probability distribution diagnostics for both OOF and holdout paths
- emit threshold hit counts before calibration, after calibration, before dynamic no-trade band, and after dynamic no-trade band
- emit per-symbol and per-regime threshold hit counts on holdout
- add tests covering zero-trade holdout, zero-candidate expected-edge floor, and no-trade metric masking

Done when:

- any future no-trade holdout run fails with explicit named reasons
- artifact diagnostics isolate whether the collapse is caused by model score compression, calibration, threshold policy, or execution-band logic

### Task 9: Freeze The Clean-Snapshot Operator Surface

Status:

- `pending`

Why this matters:

- current operator launchers still contain retired snapshot and `GLD` assumptions outside the clean approved universe
- once data quality is fixed, accidentally launching against a retired snapshot becomes an avoidable operator error

Required actions:

- remove `snap_76d371975b6e817d` from launcher defaults
- remove `GLD` from Wave 1 launcher defaults and use the clean universe file consistently
- parameterize research and promotion launchers so bundle path and run identity are explicit
- fail fast if a launcher targets a retired snapshot or a symbol set that disagrees with the approved universe
- update docs and command examples to reference the clean bundle only

Done when:

- no active launcher or operator doc points at the failed-quality snapshot
- no launcher silently reintroduces `GLD` or an unapproved symbol set

### Task 10: Restore Candidate Trade Flow On The Clean Snapshot

Status:

- `pending`

Blocked by:

- Task 8
- Task 9

Why this is the main model blocker:

- the current clean baseline has `mean_trade_count=11`
- the holdout has `0` trades
- the expected-edge candidate floor has `0` candidates

Required actions:

- run one-lever-at-a-time challengers on the same clean bundle to recover score spread and trade flow
- compare `isotonic`, `sigmoid`, and disabled probability calibration on the same snapshot
- compare dynamic no-trade band enabled versus disabled on the same snapshot
- inspect whether threshold derivation is overshooting holdout score dispersion
- add threshold-search penalties for zero holdout activity and zero expected-edge candidate floors
- keep the snapshot, universe, and training profile fixed while testing these levers

Done when:

- `expected-edge` trains successfully
- `mean_trade_count >= 100`
- holdout no longer collapses to zero trades
- the selected challenger materially improves candidate flow without silently degrading the clean snapshot contract

### Task 11: Reduce Drawdown And Improve Risk-Adjusted Score

Status:

- `pending`

Blocked by:

- Task 10

Why this remains necessary:

- even before the no-trade holdout collapse, CV drawdown and risk-adjusted score were still failing

Required actions:

- use the existing tail diagnostics to identify the symbols and regimes driving downside
- tighten feature pruning once trade flow is restored
- raise downside and tail-risk penalties only after trade activity is back in a valid range
- add symbol-tail protection if a small subset of names dominates losses
- verify that any trade-flow recovery does not simply buy more bad trades

Done when:

- `mean_risk_adjusted_score > 0`
- `max_drawdown <= 0.20`
- `holdout_max_drawdown <= 0.35`

### Task 12: Lower PBO And White-Reality P-Value

Status:

- `pending`

Blocked by:

- Task 10

Why this is distinct:

- the clean baseline still shows unacceptable statistical fragility even before promotion

Required actions:

- keep comparing challengers only on the same clean bundle
- reject unstable nested-outer candidates more aggressively when stability ratios drift outside tolerance
- add explicit selection pressure against models that win CV but collapse on holdout activity or holdout Sharpe consistency
- tighten any objective components that reward noisy CV Sharpe without robust out-of-sample support

Done when:

- `pbo <= 0.45`
- `white_reality_pvalue <= 0.10`
- holdout Sharpe and holdout Sharpe consistency both pass on a trading holdout, not on a zero-trade artifact

### Task 13: Confirm A Research Winner On The Approved Bundle

Status:

- `pending`

Blocked by:

- Tasks 10, 11, 12

Required actions:

- rerun the baseline and challengers on the same clean bundle
- use the updated training matrix and champion snapshot to confirm the leader
- require `pre_promotion_checklist.ready=true` before any promotion attempt

Done when:

- at least one research candidate is both the clean-bundle leader and checklist-ready

### Task 14: Promotion On The Winner

Status:

- `pending`

Blocked by:

- Task 13

Required actions:

- promote only the clean-bundle research winner
- use strict snapshot replay on the exact approved bundle
- treat interrupted runs as non-evidence

Done when:

- promotion produces model artifact, promotion package, replay manifest, and `deployment_plan.ready_for_production=true`

### Task 15: Replay Validation Across Three Windows

Status:

- `pending`

Blocked by:

- Task 14

Required actions:

- replay the promoted artifact on the required windows
- compare execution realism against promotion assumptions
- investigate slippage, turnover, regime behavior, and symbol concentration drift

Done when:

- all required replay windows pass with coherent execution and risk behavior

### Task 16: Paper Trading Validation

Status:

- `pending`

Blocked by:

- Task 15

Required actions:

- run artifact-driven paper trading for at least `2` weeks
- produce daily monitoring packs for trade flow, expected-edge pass rate, slippage, and risk warnings

Done when:

- paper behavior remains coherent with promotion and replay evidence

### Task 17: Canary Rollout Readiness Review

Status:

- `pending`

Blocked by:

- Tasks 15 and 16

Required actions:

- write the rollout memo
- reconcile deployment guardrails against replay and paper evidence
- confirm the canary phases, kill-switch thresholds, and operator response procedures

Done when:

- capital rollout can proceed under explicit written approval and monitored guardrails

## 6. Recommended Next Implementation Order

1. Finish Task 8 so no-trade holdout behavior cannot be misread as robustness.
2. Finish Task 9 so all future runs stay on the approved clean snapshot and universe.
3. Run Task 10 challengers to recover candidate trade flow and expected-edge training.
4. Run Task 11 and Task 12 on the same bundle to improve drawdown and statistical robustness.
5. Run Task 13 only after one clean-bundle research candidate clears the checklist.
6. Run Tasks 14, 15, 16, and 17 in order.
7. Keep Task 18 deferred unless LightGBM is already promotion, replay, and paper stable.

## 7. Deferred Work

### Task 18: Reopen The TCN Branch Only After LightGBM Stabilizes

Status:

- `deferred`

Rule:

- TCN stays closed until LightGBM passes promotion, replay, and early paper validation on the clean bundle
