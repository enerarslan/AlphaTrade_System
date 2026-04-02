# LightGBM Full Audit

**Date:** 2026-04-02  
**Repo state audited:** current working tree on `C:\Users\ener\Desktop\AlphaTrade`  
**Scope:** `lightgbm`, `lightgbm_ranker`, training pipeline, promotion/runtime parity, validation gates, artifact evidence, and test coverage.

## Executive Summary

LightGBM is not currently trustworthy in this repository.

The top blocker is not the data. It is the training stack itself.

The strongest evidence is direct:

- the saved `lightgbm_ranker` and plain `lightgbm` artifacts I inspected are **single-leaf / zero-split boosters**
- some of those dead boosters still produce **non-zero or near-full trade activity** because thresholding and dynamic-band logic can manufacture trades from flat scores
- `lightgbm_ranker` promotion/runtime scoring is **structurally incompatible** with training because training is cross-sectional by timestamp, while runtime scores **symbol-by-symbol**

Until the P0 items below are fixed, running more LightGBM experiments is mostly noise. TCN should stay blocked.

## Evidence Base

I used four evidence sources:

1. Code audit of:
   - `scripts/train.py`
   - `quant_trading_system/backtest/promotion.py`
   - `scripts/backtest.py`
   - `quant_trading_system/config/model_configs.yaml`
   - `quant_trading_system/models/classical_ml.py`
   - `tests/unit/test_train_script.py`
   - `tests/unit/test_backtest_script.py`

2. Artifact inspection of:
   - `\\wsl.localhost\Ubuntu-24.04\root\AlphaTrade_wsl\models\wave1_ranker_research_20260402_runD_structfix_h12_artifacts.json`
   - `\\wsl.localhost\Ubuntu-24.04\root\AlphaTrade_wsl\models\wave1_ranker_research_20260402_runC3_clean_ranker_guarded_cal_core11_h12_artifacts.json`
   - `\\wsl.localhost\Ubuntu-24.04\root\AlphaTrade_wsl\models\wave1_lightgbm_research_20260402_wsl_run2_gpucuda_h12_artifacts.json`

3. Direct model introspection of saved `.pkl` boosters

4. Targeted test execution:
   - `python -m pytest tests/unit/test_train_script.py -q`
   - `python -m pytest tests/unit/test_backtest_script.py -q`

Both test files pass. That is part of the problem: the current tests do not protect the critical failures below.

## What The Artifacts Prove

### Saved LightGBM boosters are dead

I opened the saved models and inspected their boosters.

For:

- `wave1_ranker_research_20260402_runD_structfix_h12`
- `wave1_ranker_research_20260402_runC3_clean_ranker_guarded_cal_core11_h12`
- `wave1_lightgbm_research_20260402_wsl_run2_gpucuda_h12`

the booster-level result is the same:

- `num_trees = 1`
- `feature_importance(split).sum() = 0`
- `nonzero_features = 0`
- first tree is a single `leaf_value`

This means the model is not learning splits. It is effectively a constant scorer.

### Flat models are still generating trade flow in some runs

Artifact evidence:

| Run | Model | Raw score spread | Holdout active rate | Holdout trades | Interpretation |
| --- | --- | --- | --- | --- | --- |
| `runD_structfix_h12` | `lightgbm_ranker` | `std=0.0`, `span=0.0` | `0.0` | `0` | dead model, dead signal |
| `runC3_clean_ranker_guarded_cal_core11_h12` | `lightgbm_ranker` | `std approx 0`, `span=0.0` | `1.0` | `11` | dead model, fabricated activity |
| `wsl_run2_gpucuda_h12` | `lightgbm` | booster has zero splits | `0.998844` | `11` | dead model, fabricated activity |

So the system can currently take a constant-output LightGBM and convert it into an apparently active strategy. That makes earlier positive holdout readings unsafe to trust.

### Label diversity is not the main blocker

Using the frozen snapshot `snap_8803ebe20127c4fd`:

- development rows: `200,780`
- unique development timestamps: `27,785`
- average symbols per timestamp: `7.23`
- development timestamps with both label classes present: `81.27%`
- holdout timestamps with both label classes present: `85.74%`

That does not prove the label design is optimal for ranking, but it does prove the current collapse is not explained by a trivial "all queries are single-class" failure.

## Critical Findings

### 1. LightGBM regularization math is catastrophically wrong on realistic fold sizes

**Severity:** Critical  
**Files:** `scripts/train.py`, `tests/unit/test_train_script.py`

The LightGBM search space and train-size adjustment logic scale `min_data_in_leaf` as a large percentage of the training set.

Relevant code:

- `scripts/train.py:6526-6543`
- `scripts/train.py:6634-6649`

Current formulas:

- search space low bound: `ceil(train_size * 0.06)`
- train-size floor at fit time: `int(train_size * 0.05)`

This is fatal at real training sizes.

Concrete values:

| Train size | Search `min_data_in_leaf` low/high | Fit-time floor |
| --- | --- | --- |
| `180` | `30..44` | `20` |
| `1,000` | `60..220` | `50` |
| `10,000` | `600..600` | `500` |
| `132,092` | `7,926..7,926` | `6,604` |
| `147,800` | `8,868..8,868` | `7,390` |

The search-space bug is worse than it looks:

```python
leaf_floor = max(30, ceil(train * 0.06))
leaf_ceiling = max(leaf_floor, min(220, ceil(train * 0.24)))
```

Once `leaf_floor > 220`, the upper bound collapses to the lower bound.  
At realistic fold sizes, `min_data_in_leaf` becomes a giant fixed constant.

Artifact proof:

- `runD` nested params included `min_data_in_leaf = 7418`
- plain classifier `wsl_run2` nested params included `min_data_in_leaf = 7790`

That is consistent with the saved boosters learning **zero splits**.

Why tests missed it:

- `tests/unit/test_train_script.py:1580-1600`
- `tests/unit/test_train_script.py:1603-1624`

Current tests only validate small folds like `train_size=180` or `200`. They never hit the realistic regime where the formula collapses.

**Required fix**

Replace percentage-based leaf floors with a capped, realistic schedule. Example direction:

- search space: `min_data_in_leaf` in a bounded range such as `20..200` or a mild train-size function capped aggressively
- fit-time adjustment: never force `min_data_in_leaf` above a modest upper cap
- add regression tests for `train_size` in the `10k`, `50k`, and `150k` range

### 2. Dead LightGBM boosters can still create apparent strategy activity

**Severity:** Critical  
**Files:** `scripts/train.py`

Relevant code:

- threshold derivation: `scripts/train.py:7029-7404`
- execution-band forcing: `scripts/train.py:10498-10582`
- validation gates: `scripts/train.py:12052-12195`

Current behavior:

- score dispersion is summarized
- but dead-score conditions are **not** used as hard failure gates before thresholding
- dynamic no-trade band logic can relax thresholds, force tails, and apply emergency relaxation

Result:

- constant-score LightGBM models can still show `holdout_active_signal_rate > 0`
- some artifacts show near-full admission despite zero split learning

This is institutionally unacceptable. A dead model must fail closed, not be "rescued" into trade activity.

There is currently no explicit gate for:

- LightGBM booster with zero split count
- raw OOF score `std/span` below epsilon
- holdout raw score `std/span` below epsilon
- ranker query-level zero-span across nearly all queries

**Required fix**

Add a hard fail-closed dead-model gate before any threshold shaping:

- if LightGBM booster has zero non-leaf splits, fail the trial/fold/model
- if `oof_raw_probability_distribution_std/span` is below epsilon, fail the trial/fold/model
- if holdout raw score dispersion is below epsilon, fail validation regardless of downstream trade count
- disable `forced_tail` / emergency relaxation when raw score dispersion is below epsilon

### 3. `lightgbm_ranker` runtime/promotion/backtest parity is structurally broken

**Severity:** Critical  
**Files:** `quant_trading_system/backtest/promotion.py`, `scripts/train.py`, `tests/unit/test_backtest_script.py`

Training behavior:

- `scripts/train.py:3592-3627` normalizes ranker scores by **timestamp query**
- `scripts/train.py:8187-8203` trains ranker using **timestamp group sizes**

Runtime/promotion behavior:

- `quant_trading_system/backtest/promotion.py:917-929` converts ranker scores with a **sigmoid**
- `quant_trading_system/backtest/promotion.py:1079-1083` predicts **one symbol frame at a time**
- `quant_trading_system/backtest/promotion.py:1128-1142` thresholds those per-symbol scores directly

This is a structural mismatch.

`lightgbm_ranker` is trained on a cross-sectional problem:

- query key = timestamp
- items inside query = symbols at that timestamp

But runtime currently scores:

- one symbol independently
- across time
- without the full timestamp cross-section

So even if the sigmoid bug were removed, runtime still would not reproduce training semantics. Query-percentile rank normalization requires the full cross-section for each timestamp.

The promotion package contract also lacks ranker-specific scoring metadata:

- `quant_trading_system/backtest/promotion.py:104-143`
- `scripts/train.py:13316-13349`

There is no persisted contract for:

- scoring mode
- query key
- cross-sectional scoring requirement
- rank normalization policy

The current test suite explicitly codifies the wrong behavior:

- `tests/unit/test_backtest_script.py:114-123`

That test asserts sigmoid parity for ranker outputs.

**Required fix**

Redesign ranker runtime scoring end to end:

1. Persist ranker scoring contract in the promotion package:
   - `scoring_mode = query_percentile`
   - `query_key = timestamp`
   - `requires_cross_sectional_panel = true`

2. In promotion/backtest runtime:
   - score all symbols for the same timestamp together
   - normalize raw ranker outputs within timestamp query
   - then split results back to symbol frames

3. Delete the ranker sigmoid path entirely

4. Replace the backtest test with a cross-symbol, same-timestamp parity test

### 4. Ranker inherits classifier config and classifier search knobs

**Severity:** High  
**Files:** `quant_trading_system/config/model_configs.yaml`, `scripts/train.py`

Relevant code:

- `quant_trading_system/config/model_configs.yaml:27-54`
- `scripts/train.py:639-654`
- `scripts/train.py:6547-6551`
- `scripts/train.py:6665-6668`

Problems:

1. There is no `lightgbm_ranker` section in YAML.
2. `_load_model_defaults()` falls back from `lightgbm_ranker` to `lightgbm`.
3. The ranker Optuna space includes `scale_pos_weight`.
4. Fit-time param regularization also clamps `scale_pos_weight` for ranker.

`scale_pos_weight` is classifier class-imbalance logic. It does not belong in a timestamp-group ranking model search surface.

This increases noise, hides intent, and makes the ranker configuration surface ambiguous.

**Required fix**

- add a dedicated `lightgbm_ranker` config section
- stop inheriting classifier defaults
- remove classifier-only knobs from the ranker search/fit path
- explicitly define ranker-only defaults, including evaluation settings and any future label-gain policy

### 5. Validation still lacks ranker-safe structural gates and still hard-codes classifier proxies

**Severity:** High  
**Files:** `scripts/train.py`

Relevant code:

- `scripts/train.py:12156-12175`

Current hard gates include:

- `min_accuracy`
- `min_win_rate`

Those can be useful diagnostics, but they are weak primary promotion gates for `lightgbm_ranker`, which is trained for ordering and then transformed into a downstream execution process.

More importantly, validation does **not** gate on:

- zero-split LightGBM structure
- zero raw-score dispersion
- zero ranker query dispersion
- impossible runtime parity for ranker promotion packages

So the gate stack can reject or accept the wrong thing.

**Required fix**

For LightGBM family, add structural gates first. For ranker specifically:

- make dead-model checks hard blockers
- make runtime-parity contract checks hard blockers
- demote `min_accuracy` and `min_win_rate` to advisory diagnostics or make them model-family specific

## High / Medium Findings

### 6. Ranker path bypasses the first-party LightGBM wrapper and loses model-level safeguards

**Severity:** Medium-High  
**Files:** `scripts/train.py`, `quant_trading_system/models/classical_ml.py`

Relevant code:

- classifier/regressor use wrapper: `scripts/train.py:7990-8005`
- ranker uses raw model: `scripts/train.py:8007-8015`

This creates behavior drift:

- ranker bypasses the wrapper used by classifier/regressor
- saved ranker boosters lose real feature names and store `Column_0`, `Column_1`, ... instead
- wrapper-specific guardrails and consistency patterns do not automatically apply to ranker

I verified this directly from the saved ranker artifacts: the booster feature names are generic `Column_i`.

**Required fix**

Introduce a first-party `LightGBMRanker` wrapper or train ranker with DataFrames carrying real feature names.

### 7. Grid-search / random-search scoring is invalid for parts of the LightGBM family

**Severity:** Medium  
**Files:** `scripts/train.py`

Relevant code:

- `scripts/train.py:6442-6459`
- `scripts/train.py:6482-6494`

`GridSearchCV` and `RandomizedSearchCV` use `scoring="accuracy"`.

That is invalid or misleading for:

- `lightgbm_ranker`
- `lightgbm_regressor`

This may not affect the current WSL research runs if Optuna is always used, but the code path is still wrong and unsafe.

**Required fix**

Make optimization metric model-family aware, or block unsupported search modes for ranker/regressor.

### 8. Current tests actively preserve wrong ranker runtime behavior

**Severity:** Medium  
**Files:** `tests/unit/test_backtest_script.py`, `tests/unit/test_train_script.py`

Problems:

- no large-fold LightGBM regression test for realistic `min_data_in_leaf`
- no test that loads a saved LightGBM artifact and asserts `num_trees > 1`
- no test that requires non-zero raw-score dispersion
- no end-to-end ranker test for cross-symbol same-timestamp runtime scoring
- promotion test explicitly expects sigmoid behavior for ranker

Passing tests therefore do not mean LightGBM is healthy.

### 9. LightGBM monotonic constraints can silently disappear

**Severity:** Medium  
**Files:** `quant_trading_system/models/classical_ml.py`

Relevant code:

- `quant_trading_system/models/classical_ml.py:349-360`

If monotone constraint length mismatches the feature count, the code logs a warning and drops the constraints.

For an institutional training stack, this should usually fail closed, not degrade silently.

**Required fix**

- raise in strict training mode
- or persist an explicit hard metric that blocks promotion when configured priors were not actually applied

### 10. Snapshot review has an observability gap around symbol-quality reporting

**Severity:** Medium  
**Evidence:** `wave1_ranker_research_20260402_runD_structfix_h12.snapshot_review.json`

The snapshot review shows:

- `symbol_quality.input_symbols = 0`
- `selected_symbols = 0`
- `selected_universe = []`
- `report = {}`

even though the same run clearly trained on an 11-symbol universe and the dataset bundle manifest contains the symbol universe.

This is not the root cause of model collapse, but it weakens auditability and makes postmortems harder.

## Prioritized Fix Plan

### P0. Stop model degeneration and stop fake activity

1. Fix `min_data_in_leaf` search-space math in `scripts/train.py`.
2. Fix fit-time LightGBM parameter regularization in `scripts/train.py`.
3. Add hard LightGBM booster sanity checks:
   - `num_trees > 1`
   - non-zero split count
   - non-zero feature importance split sum
4. Add hard raw-score dispersion gates:
   - OOF raw score `std/span`
   - holdout raw score `std/span`
   - ranker query-level span coverage
5. Disable forced-tail and emergency-relaxation recovery when score dispersion is dead.

### P0. Fix ranker runtime parity before any new promotion attempt

1. Remove sigmoid ranker scoring from `quant_trading_system/backtest/promotion.py`.
2. Introduce cross-sectional timestamp scoring for runtime/backtest.
3. Persist ranker scoring contract into promotion package / replay manifest.
4. Add an end-to-end ranker parity test covering:
   - multiple symbols
   - same timestamp query
   - query-percentile normalization
   - split-back to per-symbol frames

### P1. Clean the ranker configuration surface

1. Add `lightgbm_ranker` YAML defaults.
2. Remove `scale_pos_weight` from ranker search and fit-time tuning.
3. Add ranker-specific promotion gates.
4. Wrap `LGBMRanker` in a first-party model wrapper or preserve DataFrame feature names end to end.

### P2. Tighten guardrails and observability

1. Make monotonic-constraint loss fail closed.
2. Make grid/random search metric routing model-family aware.
3. Fix snapshot review symbol-quality population.
4. Add artifact-level validation checks to CI:
   - no zero-split LightGBM artifacts
   - no flat raw-score artifacts
   - no ranker promotion package without ranker scoring contract

## Implementation Order I Recommend

Do this in order:

1. Fix LightGBM leaf-size math
2. Add dead-model gates
3. Re-run one plain `lightgbm` smoke train
4. Re-run one `lightgbm_ranker` smoke train
5. Only after both produce non-flat boosters, fix promotion/runtime ranker parity
6. Then run full research again

If step 3 or 4 still produces a one-leaf booster after the leaf-size fix, the next place to inspect is:

- threshold-neutralized raw booster learning on the development frame
- label relevance design for ranker
- any hidden feature-matrix collapse before fit

## Bottom Line

The current LightGBM problem is not "just tune hyperparameters better."

There are at least three real structural defects:

1. LightGBM leaf regularization is mathematically broken at real sample sizes.
2. Dead models can still be converted into trades by downstream threshold logic.
3. Ranker runtime scoring does not match ranker training semantics.

Until those are fixed, LightGBM metrics in this repo should be treated as non-authoritative.
