# Wave 1 Clean Snapshot Results

**Date:** 2026-04-02
**Scope:** Summary of the clean-snapshot Wave 1 LightGBM investigation, the runs that were executed, the results that were obtained, and the most likely source areas behind those results.

## 1. What Was Done

The work completed in this cycle was:

- the Wave 1 dataset was rebuilt around the approved clean snapshot `snap_8803ebe20127c4fd`
- snapshot quality and silent symbol-drop behavior were checked and recorded
- the training pipeline was instrumented to expose:
  - out-of-fold and holdout raw-vs-calibrated probability distributions
  - threshold-hit activity before and after calibration
  - threshold-hit activity before and after execution shaping
  - holdout signal funnels
  - expected-edge candidate floors
  - symbol and regime tail-loss diagnostics
- validation was tightened so zero-trade holdout behavior became an explicit failure
- launcher surface was cleaned so the retired snapshot and symbol drift were no longer part of the default research path
- challenger runs were executed on the same clean bundle to isolate where the holdout failure came from
- an OOF calibration guardrail was added so calibration collapse could be rejected automatically before holdout evaluation

## 2. Clean Snapshot Outcome

The clean snapshot itself passed the quality review:

- `snapshot_id=snap_8803ebe20127c4fd`
- `dataset_bundle_hash=d6522c7784a43844004b17d910bf51e1ff3754888d349faaa9e491bdf254e880`
- `snapshot_review.ready=true`
- `data_quality_passed=true`
- `dropped_symbols=0`
- regular-session missing-bars ratio was `0.00082575`, below the configured threshold

This established that the primary data-quality problem from the earlier retired snapshot was no longer the active blocker.

## 3. Run Results

### 3.1 Run B: Clean Baseline

Run:

- `wave1_ranker_research_20260402_runB_clean_core11_h12`

Main results:

- `mean_sharpe=0.9197`
- `mean_trade_count=11.0`
- `mean_risk_adjusted_score=-2.2538`
- `pbo=0.5535`
- `white_reality_pvalue=0.1967`
- `holdout_trade_count=0.0`
- `holdout_active_signal_rate=0.0`
- `holdout_sharpe=0.0`
- `expected_edge_candidate_floor_candidate_count=0`

Observed behavior:

- the clean baseline produced a zero-trade holdout
- the expected-edge policy could not train because candidate flow was zero
- holdout drawdown and holdout symbol statistics were not meaningful because the holdout never entered the market

### 3.2 Run C0: Clean Diagnostic Rerun

Run:

- `wave1_ranker_research_20260402_runC0_clean_ranker_diag_core11_h12`

Main results:

- `holdout_raw_probability_distribution_mean=0.6190485`
- `holdout_raw_probability_distribution_std ~ 0`
- `holdout_raw_probability_distribution_span=0`
- `holdout_calibrated_probability_distribution_mean=0.4855116`
- `holdout_calibrated_probability_distribution_std ~ 0`
- `holdout_calibrated_probability_distribution_span=0`
- `holdout_raw_threshold_hits_active_rate=1.0`
- `holdout_calibrated_threshold_hits_active_rate=0.0`
- `holdout_execution_threshold_hits_active_rate=0.0`

Observed behavior:

- raw holdout scores were already collapsed to a single point
- isotonic calibration shifted that single-point score into the no-trade region
- after calibration, neither the threshold layer nor the execution layer allowed activity

### 3.3 Run C1: Sigmoid Calibration Challenger

Run:

- `wave1_ranker_research_20260402_runC1_clean_ranker_sigmoid_core11_h12`

Main results:

- `probability_calibration_method=sigmoid`
- `holdout_trade_count=0.0`
- `holdout_active_signal_rate=0.0`
- `holdout_calibrated_threshold_hits_active_rate=0.0`

Observed behavior:

- sigmoid was less destructive than isotonic on OOF diagnostics
- the holdout still remained zero-trade
- changing only the calibration method did not produce usable out-of-sample activity

### 3.4 Run C1b: No-Calibration Challenger

Run:

- `wave1_ranker_research_20260402_runC1b_clean_ranker_nocal_core11_h12`

Main results:

- `probability_calibration_enabled=0`
- `holdout_trade_count=11.0`
- `holdout_active_signal_rate=1.0`
- `holdout_sharpe=0.8296`
- `holdout_worst_regime_sharpe=-0.5554`
- `holdout_symbol_sharpe_p25=-1.3046`
- `expected_edge_candidate_floor_candidate_count=64686`
- `expected_edge_policy_reason=Expected-edge policy coverage too low (32.22% < 55.00%).`
- `holdout_long_trade_count=33125`
- `holdout_short_trade_count=0`

Observed behavior:

- removing calibration restored holdout activity
- the restored activity was one-sided and degenerate:
  - long activity dominated everything
  - short activity disappeared entirely
  - holdout threshold activity became effectively full coverage

### 3.5 Run C3: Guarded-Calibration Confirmation

Run:

- `wave1_ranker_research_20260402_runC3_clean_ranker_guarded_cal_core11_h12`

Main results:

- `probability_calibration_enabled=0`
- `probability_calibration_reason=guardrail_activity_collapse,dispersion_collapse`
- `mean_sharpe=0.9197`
- `mean_trade_count=11.0`
- `mean_risk_adjusted_score=-2.2538`
- `pbo=0.5535`
- `white_reality_pvalue=0.1967`
- `holdout_trade_count=11.0`
- `holdout_active_signal_rate=1.0`
- `holdout_sharpe=0.8296`
- `holdout_worst_regime_sharpe=-0.5554`
- `holdout_symbol_sharpe_p25=-1.3046`
- `expected_edge_policy_reason=Expected-edge policy coverage too low (32.22% < 55.00%).`

Guardrail diagnostics:

- `raw_active_rate=0.6612`
- `calibrated_active_rate=0.0`
- `raw_std=0.0008534`
- `raw_span=0.0019529`
- `calibrated_std ~ 0`
- `calibrated_span=0`
- `calibrated_near_mid_rate=1.0`

Observed behavior:

- the new guardrail correctly detected calibration collapse and rejected it
- after rejection, the run behaved the same way as no-calibration
- this confirmed that calibration was a direct cause of the zero-trade holdout, but not the only source of model weakness

## 4. Cross-Run Summary

Across `Run B`, `Run C0`, `Run C1`, `Run C1b`, and `Run C3`, the following remained stable:

- `mean_sharpe` stayed around `0.9197`
- `mean_trade_count` stayed at `11`
- `mean_risk_adjusted_score` stayed around `-2.2538`
- `pbo` stayed around `0.5535`
- `white_reality_pvalue` stayed around `0.1967`

This means the challenger sequence changed holdout activity behavior, but did not change the underlying model-quality profile in cross-validation or the statistical fragility measures.

## 5. What The Holdout Actually Looked Like

Once calibration was removed or rejected, the holdout stopped being zero-trade, but it did not become selective.

Important characteristics:

- `holdout_active_signal_rate=1.0`
- `holdout_long_trade_count=33125`
- `holdout_short_trade_count=0`
- every holdout symbol had:
  - `candidate_rate=1.0`
  - `selected_rate=1.0`
  - `selected_vs_candidate_rate=1.0`

This means the model was not picking a subset of strong opportunities. It was effectively treating the entire holdout as admissible long exposure.

## 6. Regime And Symbol Outcomes

### Regime Results

Holdout regime metrics from the guarded-calibration run showed:

- `high_vol sharpe=-0.5554`
- `trend_down sharpe=0.8297`
- `trend_up sharpe=0.3975`

The weakest regime was clearly `high_vol`, with:

- negative Sharpe
- high underwater ratio
- negative risk-adjusted profile

### Symbol Results

Holdout symbol distribution was broad, but weak in the lower tail:

- `holdout_symbol_coverage_ratio=1.0`
- `holdout_symbol_sharpe_p25=-1.3046`
- `holdout_symbol_underwater_ratio=0.5455`

Largest tail-loss contributors included:

- `AMD`
- `NVDA`
- `META`
- `AMZN`
- `GOOGL`
- `JPM`
- `MSFT`
- `XOM`

This means symbol coverage itself was not the issue. The problem was that too many covered symbols still had weak or negative risk-adjusted behavior.

## 7. Most Likely Source Areas

The results point to the following source areas.

### 7.1 Raw Score Compression

The model’s raw out-of-sample scores were extremely compressed:

- OOF spread was already narrow
- holdout raw scores collapsed to a single point

This is the strongest indication that the model was not generating meaningful score separation out of sample.

### 7.2 Calibration Collapse

Both isotonic and sigmoid calibration pushed already-weak scores into an even narrower region:

- calibrated holdout activity went from full raw threshold-hit activity to zero
- calibrated distributions collapsed around the midpoint/no-trade region

This made calibration a direct trigger for the zero-trade holdout.

### 7.3 One-Sided Directional Behavior

When calibration was removed, the model did not become balanced. Instead:

- long activity became effectively full coverage
- short activity vanished

This indicates that the model’s score surface was not separating long and short opportunities in a stable way.

### 7.4 Weak Selectivity

Once the no-trade issue was removed, the model still selected almost everything in the holdout:

- signal funnels showed full candidate and selected rates across symbols
- expected-edge candidate flow existed, but expected-edge coverage remained too low

This indicates that the model was recovering activity by losing selectivity rather than by identifying a smaller set of strong opportunities.

### 7.5 Robustness Failure In High-Vol And Weak Symbols

The holdout’s weakest performance concentrated in:

- the `high_vol` regime
- the lower tail of symbol Sharpe outcomes

This indicates that even with restored activity, the model remained fragile where regime stress and symbol-specific downside were strongest.

### 7.6 Statistical Fragility

`PBO` and `White Reality` remained poor across all challengers:

- `pbo` stayed above the gate
- `white_reality_pvalue` stayed above the gate

This indicates that the model’s apparent cross-validation usefulness was not backed by strong out-of-sample statistical evidence.

## 8. Final Reading Of The Investigation

The clean snapshot fixed the data problem. The challenger sequence then separated two different failure modes:

- calibration collapse was the direct reason the first clean-snapshot runs produced zero holdout trades
- after that collapse was removed, the model still did not produce production-grade behavior

The remaining observed pattern was:

- narrow or collapsed raw out-of-sample score dispersion
- calibration making that collapse worse
- no-calibration restoring activity only by turning the holdout into a one-sided, almost fully admitted long stream
- weak regime robustness
- weak lower-tail symbol robustness
- persistent statistical fragility

This means the active problem at the end of this cycle was no longer snapshot quality and no longer purely calibration. The dominant pattern was weak out-of-sample score discrimination with degenerate directional behavior and weak robustness.
