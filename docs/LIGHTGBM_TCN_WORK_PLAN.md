# AlphaTrade LightGBM + TCN Work Plan

**Date:** 2026-04-01
**Status:** Ready to execute
**Scope:** Backtest and paper-trading model program after Ubuntu / WSL migration

## 1. Objective

Build a disciplined training program for AlphaTrade that:

- uses the multi-layer PostgreSQL dataset as the source of truth
- produces one strong primary paper-trading model
- trains one serious challenger on the exact same dataset snapshot
- promotes models only after institutional validation, replay, and paper-trading evidence

The working assumption for Wave 1 is:

- **Primary:** `lightgbm_ranker` for multi-symbol training
- **Fallback primary:** `lightgbm` when the universe is too small for ranking
- **Challenger:** `tcn`
- **Filter layer:** existing meta-labeling flow in `scripts/train.py`

## 2. Principles

1. Do not compare models trained on different datasets. Freeze one snapshot and compare on that snapshot.
2. Do not start with multi-timeframe fusion. First stabilize `15Min` only.
3. Do not use `--model all` for normal work. Research should be narrow and repeatable.
4. Do not optimize TCN before LightGBM has a stable baseline.
5. Do not promote from backtest alone. Promotion requires validation gates, replay, and paper trading.
6. Run heavy training on the Linux filesystem inside WSL / Ubuntu, not on `/mnt/c/...`.

## 3. Wave 1 Target Design

Use the existing cost-aware institutional target path already implemented in:

- `quant_trading_system/models/target_engineering.py`
- `scripts/train.py`

Wave 1 target config:

- `timeframe=15Min`
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

Rationale:

- `h=5` on `15Min` is a practical first paper-trading horizon
- the target already includes costs, neutral filtering, and volatility-aware signal floors
- we avoid inventing a new label before the current institutional path is exhausted

## 4. Universe Plan

### 4.1 Research Universe

Start with a core 12-symbol intraday universe:

- `SPY`
- `QQQ`
- `AAPL`
- `MSFT`
- `NVDA`
- `AMD`
- `AMZN`
- `META`
- `GOOGL`
- `JPM`
- `XOM`
- `GLD`

Why:

- liquid names
- broad sector spread
- strong intraday data coverage
- enough cross-sectional signal to justify ranking

### 4.2 Promotion Universe

After the research baseline is stable:

- expand to top 24 names
- then expand to the full 48-symbol `1Min/15Min` universe only if symbol-quality gates remain healthy

Do not jump directly to 48 symbols before the first baseline is understood.

## 5. Feature Policy

Wave 1 feature policy:

- keep `technical`
- keep `statistical`
- keep `microstructure`
- keep `cross_sectional`
- keep reference features enabled
- keep feature selection enabled

Wave 1 feature-selection settings:

- `feature_selection_min_ic=0.015`
- `feature_selection_max_corr=0.90`
- `feature_selection_max_features=180`
- `feature_selection_stability_iterations=16`
- `feature_selection_min_stability_support=0.60`

Do not disable entire feature groups unless an ablation run proves they are low-value.

## 6. Phase Plan

### Phase 0: Pre-flight

Goal:

- confirm the Ubuntu / WSL training path is the default operator path
- confirm PostgreSQL, Redis, artifacts, and snapshot reuse all work from Linux-side storage

Checklist:

1. Work from `~/AlphaTrade`.
2. Confirm the Python environment is the Linux-side virtualenv.
3. Confirm PostgreSQL and Redis health from WSL.
4. Confirm the training run writes under the Linux filesystem.
5. Confirm a dataset snapshot bundle is created and reusable.

Exit criteria:

- one `research` training run completes end-to-end
- one snapshot bundle is created
- the run is repeatable from the same snapshot bundle

### Phase 1: LightGBM Research Baseline

Goal:

- build the first real primary-model baseline on the core 12-symbol universe

Model choice:

- use `lightgbm_ranker` because the objective is cross-sectional symbol selection per timestamp

Research budget:

- `training-profile=research`
- `n_splits=3`
- `n_trials=30`
- `holdout_pct=0.15`
- SHAP off via profile
- meta-labeling off via profile

Research command template:

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
  --primary-horizon 5 \
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
  --feature-selection-min-stability-support 0.60
```

What to review after Phase 1:

- `mean_sharpe`
- `mean_max_drawdown`
- `mean_trade_count`
- `label_positive_rate`
- `label_drift_abs`
- holdout symbol coverage
- feature count after selection

Exit criteria:

- no leakage or snapshot errors
- no feature-materialization instability
- a sensible trade count
- no obvious overfitting blow-up between CV and holdout

### Phase 2: LightGBM Promotion Candidate

Goal:

- retrain the same primary idea under full institutional promotion settings

Promotion budget:

- `training-profile=promotion`
- `n_splits=5`
- `n_trials=120`
- nested walk-forward on
- SHAP on
- meta-labeling on
- snapshot reuse on

Promotion command template:

```bash
python main.py train \
  --model lightgbm_ranker \
  --training-profile promotion \
  --symbols SPY QQQ AAPL MSFT NVDA AMD AMZN META GOOGL JPM XOM GLD \
  --timeframe 15Min \
  --cv-method purged_kfold \
  --n-splits 5 \
  --n-trials 120 \
  --holdout-pct 0.15 \
  --label-horizons 1 5 20 \
  --primary-horizon 5 \
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
  --meta-label-min-confidence 0.55
```

If the previous research run already produced a clean dataset bundle, prefer replaying the exact bundle:

```bash
python main.py train \
  --model lightgbm_ranker \
  --training-profile promotion \
  --dataset-snapshot-bundle <PATH_TO_DATASET_BUNDLE_MANIFEST> \
  --strict-snapshot-replay \
  --n-splits 5 \
  --n-trials 120
```

Promotion gate expectation:

- built-in validation gates must pass
- holdout metrics must not collapse relative to CV
- symbol coverage must remain broad enough to justify ranking

### Phase 3: TCN Challenger On The Same Snapshot

Goal:

- evaluate whether sequence modeling adds real incremental edge over the LightGBM primary

Rule:

- TCN must use the exact same dataset snapshot bundle as the promoted LightGBM candidate

Initial TCN shape:

- `lookback_window=60`
- `num_channels=[64, 128, 128, 64]`
- `kernel_size=3`
- `dropout=0.20`
- `learning_rate=0.001`
- `batch_size=64`
- `epochs=100`
- `patience=15`

TCN research command template:

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

TCN promotion command template:

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

TCN replacement criteria:

- holdout Sharpe at least `+5%` vs primary LightGBM
- drawdown no worse than primary by more than `0.02`
- turnover no worse than primary by more than `10%`
- symbol coverage does not shrink materially
- replay and paper metrics are at least as robust as primary

If these are not met, TCN stays a challenger only.

### Phase 4: Final Primary + Meta-Label Filter

Goal:

- turn the best base model into a paper-trading candidate with signal filtering

Preferred order:

1. choose base winner: `lightgbm_ranker` or `tcn`
2. rerun promotion with meta-labeling enabled
3. preserve the same target horizon and same snapshot scope

Meta-label policy:

- keep `meta_label_dynamic_threshold=true`
- start from `meta_label_min_confidence=0.55`
- do not increase confidence aggressively before paper-trading evidence exists

Why:

- early over-filtering can make backtests look clean while starving live trade count

### Phase 5: Replay

Goal:

- test whether the promoted candidate survives deterministic replay before paper trading

Required replay window:

- at least 3 separate windows
- include one trend-up window
- include one high-volatility window
- include one mixed / range window

Suggested first replay windows:

- `2024-01-01` to `2024-01-05`
- `2024-08-01` to `2024-08-09`
- `2025-03-03` to `2025-03-14`

Baseline command:

```bash
python main.py replay --start 2024-01-01 --end 2024-01-05 --symbols SPY QQQ AAPL MSFT NVDA AMD AMZN META GOOGL JPM XOM GLD
```

Replay pass conditions:

- no replay SLO failure
- no risk-escalation instability
- no kill-switch surprise
- no unrealistic concentration in one or two symbols

### Phase 6: Paper Trading

Goal:

- verify the candidate under broker semantics before any further escalation

Minimum paper-trading period:

- 2 weeks minimum
- 4 weeks preferred before declaring the model stable

Track daily:

- net PnL
- turnover
- fill / rejection rate
- slippage
- symbol concentration
- no-trade band behavior
- kill-switch or risk warnings

Paper pass conditions:

- stable trade count
- stable latency and execution quality
- no repeated risk or order-management anomalies
- realized turnover within expected cap

## 7. Operating Cadence

### Daily / Frequent Research

- run `research` profile on the frozen or auto-reused snapshot
- test feature, target-threshold, and universe variants
- compare only 1 variable at a time

### Weekly Promotion

- promote only the best research candidate
- run full promotion once per week or after a clearly justified change

### Monthly Expansion

- expand from 12 symbols to 24 symbols
- then consider 48-symbol intraday scope
- only after the smaller universe is stable

## 8. Go / No-Go Rules

Go to the next phase only if the previous phase is stable.

No-Go conditions:

- unstable snapshot reuse
- feature-materialization drift between runs
- holdout collapse after optimization
- paper trade starvation due to over-filtering
- concentration risk dominated by a small subset of symbols
- TCN improvement visible only in CV but not in holdout / replay / paper

## 9. Deliverables

Each serious candidate should produce:

- model artifact
- dataset snapshot bundle manifest
- snapshot ID
- validation report
- replay evidence
- model card / deployment plan
- paper-trading observation notes

## 10. Recommended First 4 Runs

Run 1:

- `lightgbm_ranker`
- `research`
- core 12-symbol universe

Run 2:

- `lightgbm_ranker`
- `promotion`
- same universe or same snapshot

Run 3:

- `tcn`
- `research`
- exact same snapshot as Run 2

Run 4:

- `tcn`
- `promotion`
- exact same snapshot as Run 2

After Run 4:

- choose the base winner
- rerun with meta-labeling as the paper-trading candidate

## 11. Short Summary

Wave 1 should not try to do everything at once.

The correct first program is:

1. stabilize `15Min` target engineering
2. train `lightgbm_ranker` as the primary baseline
3. freeze and reuse the dataset snapshot
4. train `tcn` on the exact same snapshot
5. keep the winner, add meta-labeling, then run replay
6. move to paper trading

That path is the fastest way to get a strong model without breaking institutional discipline.
