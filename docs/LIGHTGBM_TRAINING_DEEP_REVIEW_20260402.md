# LightGBM Training Deep Review

**Date:** 2026-04-02
**Scope:** Second-pass structural review of the LightGBM ranker training pipeline after the initial rank-score normalization fix.

## Executive Read

The first fix removed the most obvious blocker: ranker outputs were being treated like globally calibrated probabilities. That was necessary, but it was not enough.

This review found four additional structural problems that were still capable of producing misleading CV scores, degenerate holdout behavior, or invalid promotion decisions:

1. the ranker objective path still scored and penalized the model like a classifier
2. execution shaping still contained directional drift logic that could bias ranker output into one-sided exposure
3. expected-edge policy metrics were trained too late to influence multiple-testing diagnostics or validation gates
4. training-time secondary-policy sequencing still diverged from runtime sequencing

The codebase was updated so retraining now evaluates the ranker on a more internally consistent stack.

## Root Causes And Fixes

### 1. Ranker Metrics Were Still Partly Scored Like Classifier Outputs

Even after query-wise score normalization, the fold metric path still used raw `model.predict(...)` values for:

- binary prediction conversion
- `mse`
- `r2`
- objective calibration penalty inputs

That was methodologically wrong for `lightgbm_ranker` because raw rank scores are ordering scores, not calibrated probabilities.

**Fixes applied**

- ranker fold classification decisions now use normalized `y_proba`, not raw rank scores
- ranker continuous error proxies now use normalized `y_proba`
- ranker `brier_score` is zeroed out so classifier-style calibration does not contaminate optimization
- objective calibration penalty is disabled for ranker mode
- threshold-pair search no longer subtracts a calibration penalty in ranker mode

**Impact expected on retrain**

- CV metrics should become more honest for ranker runs
- Optuna / fold selection should stop favoring classifier-like probability behavior that a ranker cannot satisfy correctly
- threshold search should stop over-rewarding midpoint hugging

### 2. Ranker Execution Shaping Still Had One-Sided Drift Pressure

The execution profile builder still had dynamic activity forcing and confidence-trend drift logic designed for classifier probabilities. In practice, that logic could:

- push short thresholds too low in bullish drifts
- widen long-side admission asymmetrically
- recover activity by admitting too much exposure

That was consistent with the earlier full-coverage / long-only holdout failures.

**Fixes applied**

- ranker mode now uses a more conservative dynamic activity target
- forced-tail relaxation is capped more tightly for ranker mode
- emergency relaxation is tightened for ranker mode
- directional drift biasing is skipped entirely for ranker mode

**Impact expected on retrain**

- ranker holdout should be less likely to snap into one-sided full admission
- activity recovery should be more selective instead of simply forcing exposure

### 3. Expected-Edge Policy Was Invisible To The Main Gating Stack

The pipeline already trained an expected-edge policy, but it happened after multiple-testing correction and after validation. That meant:

- `PBO` holdout-gap logic still used base holdout Sharpe
- validation gates still read base holdout trade-flow metrics
- promotion decisions could ignore the policy layer that runtime actually uses

**Fixes applied**

- run order changed to:
  - CV
  - meta-labeling
  - expected-edge policy
  - multiple-testing correction
  - validation gates
- expected-edge holdout evaluation now computes policy-adjusted execution metrics
- validation and PBO holdout-gap logic now resolve an `effective_holdout_metric_source`
- when expected-edge holdout metrics are present, holdout Sharpe / trade-count / activity / drawdown gates use the policy-adjusted stream instead of the raw base stream

**Impact expected on retrain**

- promotion gates should better reflect the signal stack actually intended for downstream use
- models should be less likely to be promoted or rejected on the wrong holdout surface

### 4. Training-Time Secondary Policy Sequencing Diverged From Runtime

Runtime promotion flow applies:

- primary model
- meta-label filter
- expected-edge admission / sizing

Training did not consistently reflect that sequencing.

**Fixes applied**

- holdout close prices are now persisted during dataset preparation
- expected-edge training path now applies the fitted meta filter when the required inputs exist
- expected-edge holdout evaluation also applies the meta filter when holdout prices are available
- meta-filter diagnostics are recorded under `expected_edge_meta_filter_*`

**Impact expected on retrain**

- expected-edge candidate flow should better match the runtime trade surface
- diagnostics should show whether meta filtering is starving the edge policy or improving selectivity

## Files Changed

- `scripts/train.py`
- `tests/unit/test_train_script.py`

## Validation Completed

Executed:

```bash
python -m pytest tests/unit/test_train_script.py -q
```

Result:

- `172 passed`

## Retrain Checklist

For the next WSL retrain, the main things to inspect are:

1. `effective_holdout_metric_source`
   It should say `expected_edge_policy` when the policy trains successfully.

2. holdout trade flow
   `holdout_active_signal_rate` and the expected-edge policy holdout metrics should no longer show the previous zero-trade versus full-coverage whipsaw.

3. directional balance
   check long/short candidate counts and selected counts; short-side activity should no longer collapse purely because of execution shaping.

4. statistical diagnostics
   `pbo_holdout_metric_source`, `pbo_holdout_gap_ratio`, `deflated_sharpe`, and `white_reality_pvalue` now matter more because the holdout reference surface is less distorted.

5. policy funnel health
   inspect:
   - `expected_edge_candidate_floor`
   - `expected_edge_training_signal_funnel`
   - `expected_edge_holdout_signal_funnel`
   - `expected_edge_meta_filter_*`

## Bottom Line

This does not guarantee an institutional-grade Sharpe by itself. It does remove several structural distortions that were making the previous LightGBM ranker runs unreliable to interpret.

The next retrain should be treated as the first fully coherent read of this ranker stack, not just another incremental retry of the old pipeline.
