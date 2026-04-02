# Wave 1 RunD Analysis And Training Plan

**Date:** 2026-04-02  
**Run:** `wave1_ranker_research_20260402_runD_structfix_h12`  
**Snapshot:** `snap_8803ebe20127c4fd`  
**Universe:** `wave1_clean_core11_20260402`  
**Timeframe:** `15Min`, regular session only

## Executive Summary

RunD did not fail because of dirty data, missing symbols, or label imbalance. It failed because the model produced a fully collapsed score surface:

- out-of-fold raw probabilities were exactly `0.5`
- holdout raw probabilities were exactly `0.5`
- raw threshold hit rate was `0.0`
- calibrated threshold hit rate was `0.0`
- execution threshold hit rate was `0.0`

This is materially worse than the earlier `runC3` ranker result. `runC3` was degenerate because it admitted almost every holdout row on the long side; `runD` is degenerate because it admits nothing at all.

The immediate conclusion is:

1. do **not** start TCN training yet
2. keep LightGBM as the active family
3. treat the next step as a **score-dispersion recovery campaign**, not a broad model sweep

## What RunD Proved

### 1. The dataset is still clean

The snapshot review remains healthy:

- `snapshot_id = snap_8803ebe20127c4fd`
- `dataset_bundle_hash = d6522c7784a43844004b17d910bf51e1ff3754888d349faaa9e491bdf254e880`
- `rows_total = 359,376`
- `development_rows = 200,780`
- `holdout_rows = 33,125`
- `symbol_count = 11`
- `missing_bars_ratio = 0.00082575`
- `data_quality_passed = true`
- `no_silent_symbol_drop = true`

So the training failure is not explained by snapshot corruption or a broken universe build.

### 2. Labels are not collapsed

The label distribution is still approximately balanced:

- `label_positive_rate = 0.49356`
- first-half positive rate `= 0.50596`
- second-half positive rate `= 0.48116`

So this is not a trivial class-imbalance failure.

### 3. The model output collapsed before execution shaping

Key RunD metrics:

- `mean_accuracy = 0.4966`
- `mean_sharpe = 0.0000`
- `mean_trade_count = 0.0`
- `mean_risk_adjusted_score = -1.0000`
- `holdout_trade_count = 0.0`
- `holdout_active_signal_rate = 0.0`
- `holdout_sharpe = 0.0`
- `white_reality_pvalue = 0.5000`
- `pbo = 0.5000`
- `deflated_sharpe = NaN`
- `expected_edge_candidate_floor_candidate_count = 0`

The decisive diagnostics are these:

- `oof_raw_probability_distribution_min = 0.5`
- `oof_raw_probability_distribution_max = 0.5`
- `oof_raw_probability_distribution_std = 0.0`
- `holdout_raw_probability_distribution_min = 0.5`
- `holdout_raw_probability_distribution_max = 0.5`
- `holdout_raw_probability_distribution_std = 0.0`
- `oof_raw_threshold_hits_active_rate = 0.0`
- `holdout_raw_threshold_hits_active_rate = 0.0`

That means the failure happens before expected-edge policy and before execution shaping. The model is not producing usable ranking dispersion.

### 4. Calibration was not the blocker in RunD

In RunD:

- `probability_calibration_enabled = 0`
- `probability_calibration_reason = disabled_or_unsupported`

This matters because the earlier zero-trade story was partly caused by calibration collapse. In RunD, even the **raw** signal surface is dead. Calibration is no longer the primary explanation.

## Comparison To Prior Runs

| Run | Core behavior | Mean Sharpe | Holdout Sharpe | Holdout active rate | Long trades | Short trades | Main issue |
|---|---|---:|---:|---:|---:|---:|---|
| `wave1_lightgbm_research_20260402_wsl_run2_gpucuda_h12` | classifier baseline | `-0.6456` | `0.7742` | `0.9988` | not useful | not useful | terrible CV utility, weak statistics |
| `runC3_clean_ranker_guarded_cal_core11_h12` | ranker, full long-only admission | `0.9197` | `0.8296` | `1.0` | `33125` | `0` | one-sided over-admission |
| `runD_structfix_h12` | ranker, no admission | `0.0000` | `0.0000` | `0.0` | `0` | `0` | raw score collapse to `0.5` |

Interpretation:

- plain `lightgbm` is not good enough to promote, but it still acts as a useful control model
- `runC3` showed the ranker could at least create separable scores, even though they were directionally broken
- `runD` removed the long-only pathology but overshot into a no-signal regime

So the system moved from:

- `bad because everything is admitted long`

to:

- `bad because nothing is admitted at all`

That is not progress toward production, but it is a clearer diagnosis.

## Assessment Of Database / Snapshot / Train Usage

The current architecture is mostly correct and should be preserved:

- PostgreSQL + TimescaleDB is the real source of truth
- Redis is being used as a cache layer, not as a training authority
- immutable snapshot bundles are the correct unit for research replay
- strict snapshot replay is the correct way to compare challengers
- feature persistence back to PostgreSQL is the right default for institutional reuse

What should improve:

1. `feature_set_id = default` is too generic  
   Use campaign-scoped names such as `wave1_core11_h12_ranker_v1`.

2. file fallback should not be part of the production training mindset  
   PostgreSQL should stay mandatory for official runs, snapshots, and promotions.

3. Parquet export should be treated as a derived training artifact  
   It is useful for offline sequence work and diagnostics, but not as the primary authority.

4. no more duplicate WSL project roots  
   WSL should keep a single canonical repo path. This is now `/root/AlphaTrade_wsl`.

## Training Decision

### What we should train now

Train **only LightGBM-family models** right now:

1. `lightgbm_ranker` as the primary recovery path
2. `lightgbm` as a control baseline

Do **not** train `tcn` yet.

### Why TCN should wait

TCN is more expensive and slower, but RunD shows the current blocker is upstream:

- the current snapshot is clean
- the labels are not collapsed
- the ranker output surface itself is collapsed

If we train TCN now, we risk paying several hours to learn on a broken candidate-generation surface and getting a more expensive version of the same failure.

## Concrete Training Plan

## Phase 0: Freeze Data Standard

Use this exact data contract for the next campaign:

- snapshot: `snap_8803ebe20127c4fd`
- universe: core 11 symbols
- timeframe: `15Min`
- session: regular only
- date span: `2021-03-15` to `2026-03-30`
- holdout: `15%` for research, `20%` for promotion
- source of truth: PostgreSQL + TimescaleDB only
- replay mode: strict snapshot replay

Do not enlarge the universe yet. The current problem is not lack of rows.

## Phase 1: Ranker Recovery Campaign

Goal: restore non-flat raw score dispersion and non-zero two-sided candidate flow.

### Run R0: short smoke test after ranker fix

Purpose:

- verify that raw score variance is no longer exactly zero before spending a full run

Settings:

- model: `lightgbm_ranker`
- profile: `research`
- trials: `10`
- splits: `3`
- snapshot: same frozen bundle

Expected runtime:

- `10-15 minutes`

Hard pass criteria:

- `oof_raw_probability_distribution_std > 0`
- `holdout_raw_probability_distribution_std > 0`
- `oof_raw_threshold_hits_active_rate > 0.01`

If this fails, stop training and fix the ranker path before any more experiments.

### Run R1: full ranker research run

Purpose:

- confirm that the recovered score surface survives a normal research run

Settings:

- model: `lightgbm_ranker`
- profile: `research`
- trials: `30-40`
- splits: `3`
- same snapshot and universe

Expected runtime:

- `25-40 minutes`

Required outputs:

- non-zero holdout trade count
- non-zero short candidate flow
- expected-edge candidate floor greater than zero
- no raw-score collapse

### Run C1: classifier control run

Purpose:

- determine whether any remaining failure is ranker-specific or general to the feature/label contract

Settings:

- model: `lightgbm`
- profile: `research`
- trials: `30`
- splits: `3`
- same snapshot and universe

Expected runtime:

- `20-30 minutes`

Decision rule:

- if classifier is alive and ranker is still dead, the problem is likely ranker-path specific
- if both are dead, the next work should focus on label/feature/threshold contract, not model family

## Phase 2: Promotion-Grade LightGBM Run

Only do this if one research candidate is clearly alive.

Promotion prerequisites:

- `mean_trade_count > 0`
- `holdout_trade_count > 0`
- `holdout_active_signal_rate > 0`
- `expected_edge_candidate_floor_candidate_count > 0`
- `pbo <= 0.45`
- `white_reality_pvalue <= 0.10`
- `holdout_symbol_sharpe_p25 >= -0.10`
- short-side activity is non-zero

Promotion settings:

- model: best LightGBM candidate from Phase 1
- profile: `promotion`
- trials: `100-180`
- outer splits: `5`
- inner splits: `3-4`
- holdout: `20%`
- SHAP: enabled
- meta-labeling: enabled

Expected runtime:

- `90-180 minutes`

## Phase 3: TCN Only After LightGBM Passes

TCN should start only after Phase 2 yields a live candidate.

### Why this ordering is still correct

The repo already supports TCN and GPU training, but TCN should be used as a second-stage sequence challenger, not as the first rescue attempt.

### TCN research data contract

Use:

- the same frozen snapshot family
- the same label contract
- the same feature schema
- sequence lookback from the selected LightGBM candidate

Prefer a shorter training window for the first TCN pass:

- last `2.5-3.0 years` of the same `15Min` data

Reason:

- enough samples for a sequence model
- lower runtime
- less risk of learning stale early-regime artifacts first

### TCN research settings

- model: `tcn`
- profile: research-equivalent candidate run
- epochs: `30-40`
- batch size: `128`
- patience: `8-10`
- same holdout methodology

Expected runtime:

- `2-4 hours`

### TCN promotion settings

Only after TCN research beats the best LightGBM candidate on both utility and robustness:

- epochs: `60-80`
- patience: `12-15`
- runtime: `4-8 hours`

## Stop Rules

Stop the campaign immediately if any of the following repeats:

1. raw OOF scores are flat again
2. raw holdout scores are flat again
3. holdout trade count is zero again
4. expected-edge candidate floor remains zero again

If any of those occur, the next step is not “train more.” The next step is a code-path fix.

## Recommended Immediate Sequence

This is the practical order:

1. fix the ranker score-dispersion collapse
2. run `R0` smoke test
3. run one full `lightgbm_ranker` research retry
4. run one `lightgbm` control retry
5. choose the best surviving LightGBM candidate
6. only then launch a promotion-grade run
7. only after that consider TCN

## Bottom Line

RunD is useful because it removed ambiguity:

- the snapshot is clean
- the labels are usable
- the current blocker is model-output collapse, not data quality

So the right plan is not a broad training sweep and not a jump to TCN. The right plan is:

- recover LightGBM score dispersion first
- validate with one ranker run and one classifier control
- require a promotion-grade LightGBM pass
- then let TCN challenge that stabilized baseline
