# AlphaTrade LightGBM + TCN Work Plan

**Date:** 2026-04-02
**Status:** Revised after the clean-snapshot rebuild (`Run A`) and the failed clean-snapshot ranker baseline (`Run B`)
**Scope:** Reach a production-ready LightGBM baseline on the approved clean snapshot before any promotion, replay, paper trading, or TCN work

## 1. Objective

Wave 1 is now about producing a LightGBM research winner that both trades and survives institutional validation on the approved clean snapshot.

The program must:

- use PostgreSQL and the frozen snapshot bundle as the source of truth
- keep all research, promotion, replay, and paper evidence tied to one approved bundle at a time
- eliminate hidden no-trade failure modes before treating holdout metrics as real evidence
- restore enough candidate trade flow for expected-edge training without buying low-quality trades
- reduce drawdown, overfitting risk, and weak statistical evidence before promotion
- keep TCN deferred until LightGBM is truly promotion-ready

Wave 1 working stack:

- **Primary family:** `lightgbm_ranker`
- **Fallback comparator:** `lightgbm`
- **Deferred challenger:** `tcn`
- **Policy stack:** probability calibration + asymmetric side policy + regime-conditioned policy + expected-edge policy + universe quality gate

## 2. Current Evidence

### 2.1 Approved Clean Snapshot

The active approved bundle is now:

- `snapshot_id=snap_8803ebe20127c4fd`
- `dataset_bundle_hash=d6522c7784a43844004b17d910bf51e1ff3754888d349faaa9e491bdf254e880`
- `symbol_count=11`
- `quality_report_passed=true`
- `silent_symbol_drop_count=0`

Snapshot review evidence:

- `models/wave1_ranker_research_20260402_wsl_run1_clean_core11_h12.snapshot_review.json`
- `ready=true`
- `data_quality_passed=true`
- `no_silent_symbol_drop=true`

Implication:

- the old failed-quality snapshot `snap_76d371975b6e817d` is retired for promotion use
- data quality is no longer the immediate blocker

### 2.2 Clean-Snapshot Baseline Result

Baseline run:

- `run_id=wave1_ranker_research_20260402_runB_clean_core11_h12`
- `snapshot_id=snap_8803ebe20127c4fd`
- finished on 2026-04-02
- `validation_passed=false`
- `pre_promotion_checklist.ready=false`

Key metrics:

- `mean_sharpe=0.9197`
- `mean_trade_count=11.0`
- `mean_risk_adjusted_score=-2.2538`
- `pbo=0.5535`
- `white_reality_pvalue=0.1967`
- `holdout_sharpe=0.0`
- `holdout_trade_count=0.0`
- `holdout_active_signal_rate=0.0`

Most important diagnostics:

- `expected_edge_training_precheck_signal_funnel.zero_signal_count=200780`
- `expected_edge_training_precheck_signal_funnel.candidate_count=0`
- `expected_edge_candidate_floor.candidate_count=0`
- `expected_edge` skipped because it needed at least `120` candidate trades and received `0`

Interpretation:

- the baseline on the clean bundle is not just under-trading; it fully collapses to a no-trade holdout
- holdout metrics such as `drawdown=0.0` and `symbol_p25=0.0` are contaminated by that collapse
- the next work is not promotion and not TCN; it is to eliminate no-trade masking and restore valid trade flow on the same bundle

### 2.3 Comparison With Prior Diagnostic Evidence

The old diagnostic ranker on `snap_76d371975b6e817d` still had better headline Sharpe and a trading holdout, but it was not clean-snapshot evidence and still failed key gates.

Implication:

- the clean snapshot solved the data-quality blocker
- it did not solve the model-selection blocker
- the next challenger sequence must explain why the clean bundle now produces zero holdout trades

## 3. Strategic Decisions

1. `lightgbm_ranker` with primary horizon `12` remains the Wave 1 anchor until a challenger beats it on the same approved bundle.
2. The immediate blocker is now `no-trade holdout / score compression`, not snapshot quality.
3. Promotion is forbidden until a clean-bundle research candidate clears the checklist with no bypass.
4. Any run that produces zero holdout trades must be treated as explicit negative evidence, not as a benign low-drawdown result.
5. All challenger work must stay on `snap_8803ebe20127c4fd` until that bundle is formally retired.
6. Operator launchers and docs must be cleaned up so they cannot silently target the retired snapshot or reintroduce `GLD`.
7. TCN remains closed until LightGBM passes promotion, replay, and early paper validation.

## 4. Production Readiness Requirements

### 4.1 Pre-Promotion Research Requirements

Before another promotion run is authorized, the leading research candidate must satisfy all of the following on one frozen bundle:

- snapshot review passes with no silent symbol drop
- holdout produces real trades; zero-trade holdout is not allowed as evidence
- `expected-edge` trains successfully
- `mean_trade_count >= 100`
- `mean_risk_adjusted_score > 0`
- `holdout_max_drawdown <= 0.35`
- `holdout_symbol_sharpe_p25 >= -0.10`
- `pbo <= 0.45`
- `white_reality_pvalue <= 0.10`
- all research validation layers pass

### 4.2 Promotion Requirements

Promotion is only valid when:

- `training_profile=promotion`
- the exact same approved bundle is reused with `--strict-snapshot-replay`
- the latest matching research row has `pre_promotion_checklist.ready=true`
- the artifact reports `deployment_plan.ready_for_production=true`

### 4.3 Pre-Live Requirements

Before any live capital increase:

- replay passes across the three required windows
- paper trading runs on the exact promoted artifact for at least `2` weeks
- execution costs, slippage, risk warnings, symbol concentration, and expected-edge pass rates remain consistent with the promoted contract

## 5. Workstreams

### 5.1 Snapshot Integrity

Current state:

- complete for the active bundle

What must remain true:

- the approved bundle stays frozen
- the retired snapshot is not reused by launcher default, documentation drift, or operator habit

### 5.2 No-Trade Holdout And Score-Compression Diagnostics

Current blocker:

- the clean baseline produces zero holdout trades and zero expected-edge candidates

Required actions:

- make zero-trade holdout an explicit validation failure
- expose raw-score versus calibrated-score spread on OOF and holdout
- expose threshold hit counts before and after the dynamic no-trade band
- expose per-symbol and per-regime holdout activity diagnostics
- determine whether the collapse comes from model score compression, isotonic calibration, threshold policy, or execution-band widening

### 5.3 Trade-Flow Recovery

Current blocker:

- the baseline cannot train expected-edge because the candidate floor is zero

Required actions:

- run one-lever-at-a-time clean-bundle challengers
- compare calibration choices: `isotonic`, `sigmoid`, and disabled calibration
- compare dynamic no-trade band enabled versus disabled
- revise threshold selection so it does not choose a configuration that looks acceptable in CV but produces zero holdout activity
- accept only challengers that restore trade flow without degrading holdout quality

### 5.4 Robustness Recovery

Current blocker:

- even before expected-edge training, drawdown, risk-adjusted score, PBO, and White Reality still fail

Required actions:

- use tail diagnostics to isolate bad symbols and regimes
- tighten feature pruning only after trade flow returns
- improve downside and concentration penalties on the same bundle
- reject unstable nested-outer winners more aggressively

### 5.5 Operator Surface And Run Hygiene

Current blocker:

- launcher scripts still encode retired snapshot and `GLD` assumptions outside the approved clean universe

Required actions:

- parameterize bundle path and run naming for clean-snapshot research and promotion launchers
- remove retired snapshot defaults
- use the clean universe file consistently
- keep all operator surfaces aligned with the active plan

### 5.6 Promotion, Replay, And Paper Trading

Current blocker:

- there is still no eligible research champion

Required actions:

- promote only after the clean-bundle research winner clears the checklist
- replay only the promoted artifact
- use paper trading to validate execution realism and policy behavior, not to rescue a weak research candidate

## 6. Recommended Next Runs

### Run C0: No-Trade Diagnostic Rerun

Goal:

- rerun the clean-bundle baseline after Task 8 instrumentation so no-trade failure is reported explicitly and with the right diagnostics

Rules:

- same clean bundle
- no hidden model-policy change
- this run exists to localize the collapse, not to claim improvement

### Run C1: Calibration Challenger

Goal:

- test whether isotonic calibration is compressing score spread too aggressively

Allowed levers:

- `--probability-calibration-method sigmoid`
- or `--disable-probability-calibration`

Rules:

- same clean bundle
- one calibration lever per run

### Run C2: Dynamic No-Trade Band Challenger

Goal:

- determine whether the execution-band logic is zeroing holdout activity after threshold selection

Allowed lever:

- `--disable-dynamic-no-trade-band`

Rules:

- same clean bundle
- keep the best calibration decision from Run C1 fixed

### Run C3: Trade-Flow Threshold Challenger

Goal:

- recover enough candidate flow for expected-edge training and the `min_trades` gate

Allowed levers:

- threshold selection logic
- explicit activity penalties
- holdout candidate-floor preservation

Rules:

- same clean bundle
- no simultaneous robustness-policy changes

### Run D1: Robustness Challenger

Goal:

- reduce drawdown and improve statistical robustness after trade flow is restored

Allowed levers:

- stronger feature pruning
- downside-risk penalties
- symbol-tail protection
- nested-stability selection pressure

Rules:

- same clean bundle
- start from the trade-flow winner, not from the broken baseline

### Run E: Promotion On The Winner

Goal:

- run the best research configuration under promotion settings on the exact approved bundle

Rules:

- checklist-ready winner only
- strict snapshot replay only
- interrupted runs do not count

### Run F: Replay And Early Paper Validation

Goal:

- validate the promoted artifact in replay and then in paper trading

Replay windows:

- `2024-01-01` to `2024-01-05`
- `2024-08-01` to `2024-08-09`
- `2025-03-03` to `2025-03-14`

Paper minimum:

- `2` weeks minimum
- `4` weeks preferred

### Deferred Run G: TCN On The Accepted Bundle

TCN stays deferred until all of the following are true:

- a LightGBM promotion candidate exists
- replay is coherent
- early paper trading is coherent
- the team still believes incremental sequence edge is worth the extra complexity

## 7. Operating Rules

Allowed as evidence:

- research runs on the approved clean bundle
- promotion runs on that same bundle
- replay on the exact promoted artifact
- paper trading on the exact promoted artifact

Not allowed as evidence:

- no-trade holdout runs presented as low-risk wins
- interrupted runs
- retired-snapshot reruns
- AAPL-only shortcuts
- cross-snapshot comparisons without explicit retirement and replacement
- TCN work before LightGBM stabilizes

## 8. Short Summary

The plan changed again because the clean snapshot is now real, but the first clean-bundle baseline exposed a stricter blocker: the model stack collapses to zero holdout trades and zero expected-edge candidates. That means the next production path is no longer data repair. It is: eliminate no-trade masking, recover trade flow on the approved bundle, improve robustness on that same bundle, then promote, replay, paper trade, and only then reconsider TCN.
