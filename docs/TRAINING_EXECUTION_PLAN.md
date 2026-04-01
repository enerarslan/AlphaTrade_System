# AlphaTrade Training Execution Plan

**Date:** 2026-04-01
**Status:** Completed for initial optimization scope
**Scope:** Training pipeline optimization without degrading final model quality

## 0. Execution Tracker

Completed on 2026-04-01:

- [x] Added explicit `research` and `promotion` training profiles in the training pipeline.
- [x] Wired profile-aware defaults so `research` now disables SHAP and meta-labeling and applies lighter CV/Optuna budgets by default.
- [x] Preserved CLI override priority so explicit operator budgets still win over profile presets.
- [x] Exposed snapshot replay, feature-selection, warm-start, reference-feature, and tick-microstructure controls through `main.py train`.
- [x] Added runtime guidance for Windows and WSL-mounted filesystem training so the operator sees when WSL/Linux storage is the preferred path.
- [x] Added `training_profile` to training lineage summaries for auditability.
- [x] Switched the default train model selection to `lightgbm`.
- [x] Added automatic local snapshot-bundle reuse for matching training scopes, with an explicit CLI escape hatch.
- [x] Added `primary_challenger` orchestration so one command can run `lightgbm` primary plus `tcn` challenger.
- [x] Stored snapshot training scope metadata in dataset bundle manifests so automatic reuse is deterministic and auditable.
- [x] Exposed multi-timeframe fusion and database-first universe controls through `main.py train` (`--timeframes`, `--target-universe-size`, `--universe-selection-buffer-size`).
- [x] Reworked PostgreSQL training loads to bulk-load one OHLCV panel per timeframe instead of looping symbol by symbol.
- [x] Added liquidity-ranked universe selection that prioritizes symbols with full requested-timeframe coverage before loading data.
- [x] Updated multi-timeframe feature materialization to prefer database-native higher-timeframe OHLCV panels before resampling fallback.
- [x] Fixed PostgreSQL feature persistence for post-label training matrices by sourcing identifiers from the development frame after multi-timeframe materialization.
- [x] Validated the database-first multi-timeframe path in WSL on a real `15Min + 1Hour + 1Day` LightGBM smoke run through artifact save.
- [x] Fixed Docker Desktop WSL integration for `Ubuntu-24.04` and verified Docker CLI access from inside the distro.
- [x] Restored non-interactive `wsl.exe -d Ubuntu-24.04 -- ...` command execution.
- [x] Cloned the active repo into the WSL ext4 filesystem at `~/AlphaTrade` and created a Linux-side virtual environment.
- [x] Installed the training Python stack in WSL and validated PostgreSQL, Redis, Docker, and GPU reachability there.
- [x] Ran a WSL end-to-end LightGBM research smoke training pass through model save and artifact generation.
- [x] Fixed a LightGBM monotone-constraint mismatch after feature selection by reapplying model priors and adding a defensive fallback in the model wrapper.
- [x] Added regression coverage for the monotone-constraint mismatch in both trainer-level and model-level unit tests.

No blocking gaps remain in the initial optimization plan:

- [x] Run the first benchmark matrix on one fixed snapshot and compare runtime and quality.

## 1. Objective

Train paper-trading and backtest-ready models faster without throwing away the multi-layer dataset.

The core principle is:

- keep the full data stack for final models
- make iteration fast by splitting research and promotion workflows
- optimize the pipeline before reducing model or feature quality

## 2. Target Outcome

AlphaTrade should produce:

- one primary model for paper trading: `lightgbm`
- one challenger model for paper trading: `tcn`
- repeatable, snapshot-based training runs
- fast research loops for parameter and feature experiments
- full institutional validation only for shortlisted candidates

The system should stop doing full all-model sweeps by default. Training every model on every run is not the right default for this codebase or this hardware budget.

## 3. Key Decisions

### 3.1 Primary Training Environment

Use **WSL Ubuntu as the default training environment**.

Reason:

- the current training code takes a slower path on Windows during feature computation
- Windows explicitly disables the optimized technical feature path in the full pipeline
- Linux/WSL is the better target for Polars, PyTorch, LightGBM, and filesystem-heavy snapshot workflows

Important operational rule:

- keep the repo and training artifacts on the **Linux filesystem** such as `~/AlphaTrade`
- do not run heavy training from `/mnt/c/...` unless necessary

Running inside WSL while reading and writing on the Windows-mounted filesystem will erase a meaningful part of the expected performance gain.

### 3.2 Model Scope

Default training scope:

- primary: `lightgbm`
- challenger: `tcn`

Do not train:

- `xgboost`
- `random_forest`
- `elastic_net`
- `ensemble`
- other deep models

unless a scheduled benchmark window or explicit comparison run requires them.

### 3.3 Two-Lane Training Architecture

Introduce two explicit training profiles:

- `research`
- `promotion`

`research` exists to search, iterate, and rank candidates.

`promotion` exists to produce the final paper-trading candidate with the full governance stack.

This separation is the main mechanism for getting speed without giving up final quality.

## 4. Pipeline Design

### 4.1 Research Profile

Use for day-to-day iteration.

Behavior:

- model scope limited to `lightgbm` and optionally `tcn`
- load from immutable dataset snapshot bundle when available
- skip SHAP
- skip meta-labeling
- reduce nested Optuna budget
- keep holdout split and core CV logic
- keep feature selection, but make the stability budget configurable and lighter

Initial research budget:

- `n_trials`: 20 to 30
- `n_splits`: 3
- `nested_outer_splits`: 2
- `nested_inner_splits`: 2

Expected result:

- much faster candidate search
- little or no measurable degradation in final promoted model quality because the final model is still retrained in `promotion`

### 4.2 Promotion Profile

Use only for shortlisted candidates.

Behavior:

- full feature stack
- full nested Optuna trace
- multiple-testing correction
- holdout evaluation
- meta-labeling
- SHAP
- full artifact and lineage bundle

Suggested initial promotion budget:

- `n_trials`: 50 for `lightgbm`
- `n_splits`: 5
- `nested_outer_splits`: 4
- `nested_inner_splits`: 3

This is still cheaper than the current default broad-sweep behavior while preserving institutional quality.

### 4.3 Snapshot-First Training

Make the dataset snapshot bundle the default source for repeated experiments.

Required behavior:

- first run materializes OHLCV, features, labels, holdout, and CV splits
- later runs train from the snapshot bundle instead of rebuilding the dataset
- promotion runs should consume the same snapshot used in research unless the data window changed

This prevents the feature pipeline from dominating every repeat run.

### 4.4 Database-First Universe and Multi-Timeframe Loading

Use PostgreSQL as the source of truth for both:

- symbol-universe selection
- native higher-timeframe OHLCV loading

Required behavior:

- if the operator does not pin `--symbols`, the training pipeline should rank the candidate universe in PostgreSQL before loading full panels
- ranking should favor symbols with strong liquidity and full requested-timeframe coverage
- requested higher timeframes should be loaded directly from PostgreSQL when available
- resampling from the base timeframe should remain only as a deterministic fallback

This is the right default for the current dataset because the platform already stores multi-layer, multi-timeframe market data in PostgreSQL.

### 4.5 Feature Strategy

Do **not** solve the performance problem by throwing away the multi-layer dataset.

Default feature policy:

- keep technical features
- keep statistical features
- keep microstructure features
- keep cross-sectional features if they pass scale guardrails
- keep reference and tick microstructure enrichment when data coverage is good

Optimization policy:

- prefer cache reuse and snapshot reuse over feature removal
- prefer feature selection over manual feature-group deletion
- only disable a feature layer after an ablation proves that it is low value

## 5. Implementation Plan

### Phase 1: Environment and Runtime

1. Move the active training workspace to WSL Ubuntu under `~/AlphaTrade`.
2. Validate that PostgreSQL and Redis are reachable from WSL.
3. Validate GPU access in WSL for PyTorch and LightGBM if available.
4. Keep model artifacts and snapshot bundles on the Linux filesystem.

Acceptance criteria:

- WSL training can read data, write models, and run one end-to-end training pass
- no heavy training runs remain on the Windows filesystem path

Status:

- Implemented. Docker Desktop integration and non-interactive `wsl.exe -d Ubuntu-24.04 -- ...` execution both work normally.
- Implemented. The active repo has been cloned into `~/AlphaTrade`, a WSL virtual environment is in place, and PostgreSQL, Redis, Docker, and GPU access have all been validated from inside Ubuntu.
- Implemented. A WSL LightGBM smoke training run now completes the full training path through model save and artifact generation.

### Phase 2: Training Profiles

1. Add a training profile setting: `research` or `promotion`.
2. In `research`, allow `compute_shap=False` and `use_meta_labeling=False`.
3. In `research`, reduce nested CV and Optuna budgets.
4. In `promotion`, keep the current institutional safeguards.

Acceptance criteria:

- one command can run a fast research pass
- one command can run a full promotion pass
- final promotion behavior remains auditable and deterministic

Status:

- Implemented. `research` and `promotion` are now explicit CLI profiles.
- Implemented. `research` defaults now lower `n_trials`, `n_splits`, `nested_outer_splits`, `nested_inner_splits`, and `feature_selection_stability_iterations`.
- Implemented. `research` disables SHAP and meta-labeling by default, while `promotion` keeps those governance layers enforced.

### Phase 3: Snapshot-First Workflow

1. Make dataset snapshot reuse a first-class workflow.
2. Add an operator-friendly path to:
   - build snapshot once
   - train many times from snapshot
3. Standardize artifact naming so research and promotion runs can be tied to the same snapshot ID.

Acceptance criteria:

- repeated training runs do not recompute features unless the data scope or feature schema changed
- snapshot reuse becomes the normal path for experiments

Status:

- Implemented. Matching local dataset snapshot bundles are now auto-reused by default for repeated runs.
- Implemented. Operators can force a live rebuild with `--disable-auto-snapshot-reuse`.
- Implemented. Snapshot training scopes now include universe-size controls and requested multi-timeframe scope so reuse remains deterministic when the data footprint changes.

### Phase 4: LightGBM Primary Path

1. Build the default paper-trading path around `lightgbm`.
2. Keep full feature stack unless ablation data proves otherwise.
3. Add warm-start usage for final refits where beneficial.
4. Tune the Optuna search space and trial budgets for `lightgbm` specifically.

Acceptance criteria:

- `lightgbm` can produce a paper-ready candidate from the promotion profile
- training time is materially lower than the current full-sweep path

Status:

- Partial. `lightgbm` is the default training model and now participates in the explicit `primary_challenger` orchestration flow.
- Implemented. A WSL smoke training run exposed and confirmed a real LightGBM training blocker around monotone constraints after feature selection.
- Implemented. The blocker is fixed by reapplying model priors after feature selection and dropping stale monotone constraints defensively in the wrapper when dimensions do not match.
- Implemented. Runtime/quality benchmarking on a fixed snapshot is complete, and `lightgbm` remains the strongest primary path on that benchmark.

### Phase 5: TCN Challenger Path

1. Keep `tcn` as the only deep challenger by default.
2. Run `tcn` in research only after the tabular baseline is stable.
3. Promotion runs for `tcn` should be less frequent and snapshot-based.

Acceptance criteria:

- `tcn` runs are controlled and comparable against `lightgbm`
- deep learning no longer blocks the core paper-trading workflow

Status:

- Implemented in orchestration. `tcn` now runs as the explicit challenger when `--model primary_challenger` is used.
- Implemented. Runtime and quality comparison versus `lightgbm` has now been measured on the fixed snapshot, and `tcn` remains a slower challenger rather than the default primary path.

### Phase 6: Benchmark and Gating

Run controlled A/B comparisons on the same snapshot:

- current baseline pipeline
- research profile
- promotion profile
- Windows vs WSL
- `lightgbm` vs `tcn`

Track:

- wall-clock runtime
- feature materialization time
- Optuna time
- CV training time
- holdout Sharpe
- DSR
- PBO
- holdout drawdown
- symbol coverage

Acceptance rule:

- research profile may lose some search fidelity
- promotion profile must not show a statistically meaningful degradation versus the current best full-quality run

Status:

- Partial. The code path is ready: `primary_challenger` plus snapshot reuse now drives the existing benchmark matrix and champion/challenger report generation.
- Implemented. WSL runtime validation is complete and the training path is now proven end to end on the Linux filesystem.
- Implemented. The PostgreSQL data path is now multi-timeframe aware: `main.py train` accepts explicit timeframe scopes, symbol-universe selection ranks candidates by timeframe coverage plus liquidity, and higher-timeframe OHLCV is bulk-loaded natively from PostgreSQL before any resampling fallback.
- Implemented. A WSL research smoke run on `2024-01-01` to `2024-02-29` with `15Min + 1Hour + 1Day` and a ranked universe target of `4` completed end to end: `3` symbols loaded from PostgreSQL, `766,299` selected feature rows persisted back to PostgreSQL, model/replay/promotion artifacts were saved, and the run failed only on research validation gates rather than pipeline execution.
- Implemented. The first fixed-snapshot benchmark run is complete on snapshot `snap_d7306f388a86906d` (`AAPL`, `2024-01-01` to `2024-12-31`, 3057 development rows, 539 holdout rows, 59 selected technical/statistical features).
- Implemented. Research benchmark artifacts were written to `/root/AlphaTrade/models/benchmark_fixed_snapshot/research_v2/benchmarks/training_matrix_20260401T104818Z.json` and `/root/AlphaTrade/models/benchmark_fixed_snapshot/research_v2/champion_challenger_snapshot.json`.
- Implemented. Promotion benchmark artifacts were written to `/root/AlphaTrade/models/benchmark_fixed_snapshot/promotion_v1/benchmarks/training_matrix_20260401T105020Z.json`.
- Implemented. Benchmark outcome on the fixed snapshot:
  - `research/lightgbm`: `67.3s`, mean Sharpe `0.3414`, holdout Sharpe `-0.1571`, deflated Sharpe `2.3132`, PBO `0.5108`.
  - `research/tcn`: `214.2s`, mean Sharpe `0.1550`, holdout Sharpe `-0.6958`, deflated Sharpe `1.1424`, PBO `0.5936`.
  - `promotion/lightgbm`: `99.6s`, mean Sharpe `0.3439`, holdout Sharpe `-0.2390`, deflated Sharpe `2.0799`, PBO `0.6249`.
- Implemented. Benchmark conclusion: on this first fixed snapshot, `lightgbm` remains the primary path; `tcn` is materially slower and lower quality, while `promotion` adds governance/runtime cost and should remain reserved for shortlisted final candidates rather than normal iteration.

## 6. Expected Performance Impact

These are engineering targets, not guarantees.

### WSL Migration

Expected benefit:

- moderate to high speedup in feature-heavy runs
- better filesystem throughput when repo and artifacts live inside the Linux filesystem
- fewer Windows-specific slow paths

### Research vs Promotion Split

Expected benefit:

- largest single reduction in total training time
- faster iteration by reducing unnecessary full-governance runs

### Snapshot Reuse

Expected benefit:

- strongest improvement for repeated experiments on the same data window
- major reduction in feature recomputation cost

### Reduced Model Scope

Expected benefit:

- eliminates waste from broad sweeps
- focuses compute budget on the two most relevant candidates

## 7. Quality Protection Rules

The following rules are mandatory:

- do not disable holdout evaluation for promotion runs
- do not remove the full feature stack without ablation evidence
- do not rely on Windows fallback feature mode for production candidate training
- do not use research-profile results directly as the final paper-trading model
- every promoted candidate must come from the `promotion` profile

## 8. Recommended Command Pattern

Research:

```bash
python main.py train \
  --training-profile research \
  --model primary_challenger \
  --timeframes 15Min 1Hour 1Day \
  --target-universe-size 48 \
  --universe-selection-buffer-size 24 \
  --feature-set-id lgbm_tcn_research
```

Promotion:

```bash
python main.py train \
  --training-profile promotion \
  --model lightgbm \
  --timeframes 15Min 1Hour 1Day \
  --target-universe-size 64 \
  --universe-selection-buffer-size 24 \
  --n-trials 50 \
  --n-splits 5 \
  --nested-outer-splits 4 \
  --nested-inner-splits 3 \
  --feature-set-id lgbm_promotion
```

WSL operator wrapper:

```bash
wsl.exe -d Ubuntu-24.04 -u root -e sh -lc '
  cd /root/AlphaTrade &&
  . .venv/bin/activate &&
  python main.py train --training-profile research --model primary_challenger
'
```

## 9. Immediate Next Steps

1. Reuse the fixed-snapshot benchmark pattern for a broader multi-symbol, full-layer benchmark when moving from smoke-scale validation to final paper-candidate selection.
2. Keep `research` as the default search lane and reserve `promotion` for shortlisted `lightgbm` candidates.
3. Keep `tcn` as an explicit challenger, not a default equal-budget peer to `lightgbm`.
4. Run the next benchmark on a multi-symbol database-native timeframe stack such as `15Min + 1Hour + 1Day` with a ranked `48-64` symbol universe.
5. Tune any remaining Redis health-check timeout issue seen during `python main.py health check --full` in WSL if it persists under steady-state load.

## 10. Success Definition

This plan is successful when AlphaTrade can:

- train a strong `lightgbm` paper candidate quickly
- train a `tcn` challenger on demand
- preserve multi-layer data usage in final models
- avoid full all-model sweeps for normal operation
- complete most experiment loops in hours instead of days
- keep final promoted model quality at or near the current institutional baseline
