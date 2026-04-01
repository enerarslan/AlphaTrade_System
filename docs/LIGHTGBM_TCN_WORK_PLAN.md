# AlphaTrade LightGBM + TCN Work Plan

**Date:** 2026-04-01
**Status:** Updated after pre-run performance hardening
**Scope:** Produce one serious LightGBM primary candidate and one TCN challenger without contaminating comparisons

## 1. Objective

Wave 1 should maximize real trading edge, not just backtest score. The program must:

- use the PostgreSQL multi-layer dataset as the source of truth
- freeze one dataset snapshot and compare models only on that snapshot
- prefer cleaner trade selection over raw trade count
- promote only after validation gates, replay, and paper-trading evidence

Wave 1 working stack:

- **Primary candidate:** `lightgbm_ranker`
- **Fallback primary:** `lightgbm`
- **Challenger:** `tcn`
- **Filter layer:** existing meta-labeling flow in `scripts/train.py`

## 2. What Changed In Code Before The Next Run

The repo now hardens the pre-run edge stack before any new LightGBM training:

- richer reference/news features in `quant_trading_system/features/reference.py`
- richer quote/trade flow regime features in `quant_trading_system/features/tick_microstructure.py`
- tighter default label thresholds, feature-selection thresholds, and turnover discipline in `scripts/train.py`
- out-of-fold probability calibration for probability-producing models
- non-zero minimum confidence position sizing floor for execution-aware evaluation and promotion packages

Implication:

- earlier smoke runs and invalid runs are not decision inputs
- the next LightGBM run should be treated as the first serious post-hardening candidate

## 3. Core Principles

1. Do not compare runs across different snapshots.
2. Do not use smoke tests as evidence.
3. Research profile is diagnostic only; promotion profile determines serious candidates.
4. Optimize trade quality, calibration, and turnover before expanding the universe.
5. TCN does not get tuned against a moving target; it must reuse the best LightGBM snapshot.
6. Run heavy training from the Linux filesystem inside WSL, not from `/mnt/c/...`.

## 4. Wave 1 Training Policy

### 4.1 Base Scope

- `timeframe=15Min`
- core universe: `SPY QQQ AAPL MSFT NVDA AMD AMZN META GOOGL JPM XOM GLD`
- feature groups: `technical statistical microstructure cross_sectional`
- reference features enabled
- tick microstructure features enabled
- feature selection enabled

### 4.2 Label Policy

Default post-hardening target thresholds:

- `label_horizons=[1, 5, 20]`
- `primary_label_horizon=5`
- `profit_taking=0.015`
- `stop_loss=0.010`
- `max_holding=20`
- `spread_bps=1.0`
- `slippage_bps=3.0`
- `impact_bps=2.0`
- `label_min_signal_abs_return_bps=10.0`
- `label_neutral_buffer_bps=5.0`
- `label_edge_cost_buffer_bps=3.0`
- `label_volatility_lookback=20`
- `label_regime_lookback=30`
- `label_temporal_weight_decay=0.999`
- `label_apply_uniqueness_weighting=true`
- `label_apply_volatility_inverse_weighting=true`

These defaults are the floor, not the ceiling. If turnover or weak-edge churn remains high, the next adjustment band is:

- `label_min_signal_abs_return_bps=12-16`
- `label_neutral_buffer_bps=6-8`
- `label_edge_cost_buffer_bps=4-6`

### 4.3 Feature-Selection Policy

Wave 1 default feature-selection settings:

- `feature_selection_min_ic=0.015`
- `feature_selection_max_corr=0.90`
- `feature_selection_max_features=180`
- `feature_selection_stability_iterations=16`
- `feature_selection_min_stability_support=0.60`

If the first serious candidate still overfits, the first tightening move is:

- `feature_selection_min_ic=0.02`
- `feature_selection_max_features=120-140`
- `feature_selection_min_stability_support=0.65-0.75`

### 4.4 Execution-Aware Discipline

Wave 1 execution-aware defaults now assume that overtrading is a bug:

- `objective_weight_turnover=0.20`
- `objective_weight_calibration=0.35`
- `execution_turnover_cap=0.60`
- `min_confidence_position_scale=0.20`
- probability calibration enabled
- `probability_calibration_method=isotonic`

Implementation note:

- calibration is fit only from out-of-fold predictions
- if OOF sample size is too small for stable isotonic calibration, the pipeline falls back to sigmoid calibration automatically

## 5. Horizon Policy

Do not assume `h=5` is optimal just because it is the current default. The next serious LightGBM diagnostic pass should benchmark:

- `primary_horizon_sweep=3 5 8 12`

Decision rule:

- keep the horizon that best balances holdout Sharpe, turnover, and holdout symbol breadth
- do not select a horizon only because CV improves if holdout quality degrades

## 6. Phase Plan

### Phase 0: WSL Pre-Flight

Goal:

- confirm the training path is `~/AlphaTrade`
- confirm PostgreSQL, Redis, artifact writes, and snapshot reuse all work from Linux-side storage
- do not touch any still-running training process until it completes

Exit criteria:

- one clean snapshot bundle is created from WSL
- the same bundle can be replayed with `--strict-snapshot-replay`

### Phase 1: LightGBM Diagnostic Sweep

Goal:

- produce one clean frozen snapshot and decide the best horizon / model family before full promotion

Allowed models:

- `lightgbm_ranker`
- `lightgbm`

Rules:

- same symbols
- same timeframe
- same snapshot
- one variable changed at a time
- research profile outputs are diagnostic only

Recommended diagnostic command template:

```bash
python main.py train \
  --model lightgbm_ranker \
  --training-profile research \
  --symbols SPY QQQ AAPL MSFT NVDA AMD AMZN META GOOGL JPM XOM GLD \
  --timeframe 15Min \
  --cv-method purged_kfold \
  --n-splits 3 \
  --n-trials 30 \
  --holdout-pct 0.15 \
  --label-horizons 1 5 20 \
  --primary-horizon-sweep 3 5 8 12 \
  --profit-taking 0.015 \
  --stop-loss 0.010 \
  --max-holding 20 \
  --spread-bps 1 \
  --slippage-bps 3 \
  --impact-bps 2 \
  --label-min-signal-abs-return-bps 10 \
  --label-neutral-buffer-bps 5 \
  --label-edge-cost-buffer-bps 3 \
  --feature-groups technical statistical microstructure cross_sectional \
  --feature-selection-min-ic 0.015 \
  --feature-selection-max-corr 0.90 \
  --feature-selection-max-features 180 \
  --feature-selection-min-stability-support 0.60 \
  --objective-weight-turnover 0.20 \
  --objective-weight-calibration 0.35 \
  --execution-turnover-cap 0.60 \
  --min-confidence-position-scale 0.20 \
  --probability-calibration-method isotonic
```

Review after Phase 1:

- `holdout_sharpe`
- `effective_holdout_sharpe_gate_metric`
- `holdout_symbol_coverage_ratio`
- `holdout_symbol_underwater_ratio`
- `mean_trade_count`
- `execution_turnover_cap` utilization
- `probability_calibration_brier_improvement`
- `label_drift_abs`

### Phase 2: LightGBM Serious Candidate

Goal:

- run the best Phase 1 setup under full institutional promotion settings

Promotion command template:

```bash
python main.py train \
  --model lightgbm_ranker \
  --training-profile promotion \
  --dataset-snapshot-bundle <PATH_TO_DATASET_BUNDLE_MANIFEST> \
  --strict-snapshot-replay \
  --n-splits 5 \
  --n-trials 120 \
  --label-horizons 1 5 20 \
  --primary-horizon <WINNING_HORIZON> \
  --label-min-signal-abs-return-bps 10 \
  --label-neutral-buffer-bps 5 \
  --label-edge-cost-buffer-bps 3 \
  --feature-selection-min-ic 0.015 \
  --feature-selection-max-corr 0.90 \
  --feature-selection-max-features 180 \
  --feature-selection-min-stability-support 0.60 \
  --objective-weight-turnover 0.20 \
  --objective-weight-calibration 0.35 \
  --execution-turnover-cap 0.60 \
  --min-confidence-position-scale 0.20 \
  --probability-calibration-method isotonic \
  --meta-label-min-confidence 0.55
```

If Phase 1 still shows excessive churn or holdout fragility, do one tightened challenger before moving on:

- `label_min_signal_abs_return_bps=12`
- `label_neutral_buffer_bps=6`
- `label_edge_cost_buffer_bps=4`
- `feature_selection_min_ic=0.02`
- `feature_selection_max_features=140`
- `feature_selection_min_stability_support=0.70`

Promotion pass conditions:

- built-in validation gates pass
- holdout quality does not collapse relative to CV
- probability calibration does not worsen Brier score
- turnover remains within cap without starving trade count
- symbol breadth remains sufficient for ranking

### Phase 3: TCN Challenger On The Same Snapshot

Goal:

- test whether sequence modeling adds incremental edge beyond the hardened LightGBM candidate

Rules:

- exact same dataset snapshot bundle as the LightGBM serious candidate
- no silent universe or date-range changes
- no new feature-policy change between LightGBM and TCN comparison

TCN research template:

```bash
python main.py train \
  --model tcn \
  --training-profile research \
  --dataset-snapshot-bundle <PATH_TO_DATASET_BUNDLE_MANIFEST> \
  --strict-snapshot-replay \
  --n-splits 3 \
  --n-trials 40 \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.001
```

TCN promotion template:

```bash
python main.py train \
  --model tcn \
  --training-profile promotion \
  --dataset-snapshot-bundle <PATH_TO_DATASET_BUNDLE_MANIFEST> \
  --strict-snapshot-replay \
  --n-splits 5 \
  --n-trials 50 \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.001
```

TCN only replaces LightGBM if it improves holdout and replay quality without materially worsening turnover, drawdown, or symbol breadth.

### Phase 4: Replay And Paper Trading

Replay is mandatory before paper promotion. Minimum replay requirements:

- three windows
- one trending window
- one high-volatility window
- one mixed/range window

Suggested first replay windows:

- `2024-01-01` to `2024-01-05`
- `2024-08-01` to `2024-08-09`
- `2025-03-03` to `2025-03-14`

Paper-trading minimum:

- `2` weeks minimum
- `4` weeks preferred before declaring the model stable

Track daily:

- net PnL
- turnover
- slippage
- fill / rejection rate
- symbol concentration
- risk warnings
- no-trade band behavior

## 7. Operating Rules

Allowed as evidence:

- promotion run on a frozen snapshot
- replay on the same promoted artifact
- paper-trading observations on the promoted artifact

Not allowed as evidence:

- smoke tests
- AAPL-only shortcuts
- repeated runs on silently changed snapshots
- CV-only wins with holdout collapse

## 8. Recommended Next 4 Runs

Run 1:

- `lightgbm_ranker`
- `research`
- frozen snapshot creation
- `primary_horizon_sweep=3 5 8 12`

Run 2:

- `lightgbm`
- `research`
- exact same snapshot as Run 1
- same horizon sweep for fallback comparison

Run 3:

- best LightGBM family from Runs 1-2
- `promotion`
- same frozen snapshot

Run 4:

- `tcn`
- `promotion`
- exact same snapshot as Run 3

After Run 4:

- keep the strongest base model
- use replay and paper trading to validate the winner

## 9. Short Summary

The next LightGBM run should not be another loose experiment. It should be the first serious post-hardening candidate on a frozen snapshot, with tighter label discipline, tighter turnover control, out-of-fold probability calibration, and a non-zero confidence sizing floor. Only after that baseline is credible should TCN be judged against it.
