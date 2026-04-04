# Wave1 D1 Symbol Diagnosis

## Scope

This note diagnoses why the clean `D1` champion
(`sec_filings + corporate_actions` LightGBM ranker on `snap_86ac4fbc2826d9cd`)
still fails promotion on:

- `max_pbo`
- `min_holdout_symbol_p25_sharpe`

Primary reference artifact:

- `/root/AlphaTrade_wsl/models/wave1_ranker_research_20260402_runD1_clean_ranker_seccorp_core11_h12_artifacts.json`

Comparison artifacts:

- `/root/AlphaTrade_wsl/models/wave1_xgboost_research_20260404_runE0_clean_xgboost_seccorp_core11_h12_artifacts.json`
- `/root/AlphaTrade_wsl/models/wave1_ranker_research_20260402_runD3_clean_ranker_seccorp_objectivepatch_core11_h12_artifacts.json`

## Executive Diagnosis

The current problem is not a generic data-quality failure and not primarily a
"need more model families" problem.

The failure comes from three stacked issues:

1. `META` and `MSFT` are weak already at the base signal layer.
2. `JPM` is not obviously edge-free, but it is tail-unstable: small positive
   average returns are overwhelmed by bad windows and poor symbol-level Sharpe.
3. The expected-edge selector is only partially aligned with the ranker:
   it improves some weak symbols, but it also degrades some strong ones
   (`SPY`, `NVDA` in D1 holdout).

This means blind retraining on the same universe will likely reproduce the same
failure shape.

## Key Facts

### 1. D1 is still the best overall candidate

- `mean_sharpe = 1.5240`
- `mean_risk_adjusted_score = 0.8678`
- `holdout_sharpe = 2.7308`
- `expected_edge_holdout_policy_sharpe = 1.7665`
- all three holdout regimes are positive

But it fails:

- `max_pbo`: actual `0.5219`, threshold `0.45`
- `min_holdout_symbol_p25_sharpe`: actual `-0.6197`, threshold `-0.10`

### 2. No clear data-quality smoking gun for the worst symbols

All core symbols pass symbol-quality checks in the replayed snapshot.

The largest missing-bar issue is actually `GOOGL`, not `META`/`MSFT`:

- `GOOGL`: `233` missing bars
- `JPM`: `46`
- `META`: `3`
- `MSFT`: `1`

`GOOGL` is not one of the worst holdout symbols in D1.

Interpretation:

- the main failure is not explained by obvious data corruption or missing-bar load
- expanding the universe before fixing symbol-tail behavior is likely to worsen
  the p25 problem

### 3. The worst D1 symbols are genuinely bad on holdout

Worst holdout symbol Sharpe in D1:

- `META = -1.8820`
- `JPM = -1.5476`
- `MSFT = -1.5175`

These names also have near-maximum underwater ratios:

- `META = 0.9997`
- `JPM = 0.9997`
- `MSFT = 0.9997`

Interpretation:

- these are not mild underperformers
- they spend almost the entire holdout underwater once active

## Layer-by-Layer Diagnosis

### A. Base signal layer

Using D1 holdout base signal funnel and holdout execution symbol metrics:

#### META

- base selected mean trade return: `-0.0001283`
- execution selected mean trade return: `-0.0001626`
- expected-edge selected mean trade return: `+0.0004011`
- final symbol Sharpe: `-1.8820`

Interpretation:

- the raw model is already wrong on `META`
- the selector improves trade quality, but not enough to rescue the symbol

#### MSFT

- base selected mean trade return: `-0.0001522`
- execution selected mean trade return: `-0.0000978`
- expected-edge selected mean trade return: `+0.0004176`
- final symbol Sharpe: `-1.5175`

Interpretation:

- same pattern as `META`
- the selector helps, but the base symbol edge is weak enough that the symbol
  still fails after filtering

#### JPM

- base selected mean trade return: `+0.0001797`
- execution selected mean trade return: `+0.0002287`
- expected-edge selected mean trade return: `+0.0003867`
- final symbol Sharpe: `-1.5476`

Interpretation:

- `JPM` is different
- average trade quality is not obviously broken
- the failure is more about volatility/tail concentration than raw mean edge

Conclusion:

- `META` and `MSFT` look like symbol-model mismatch
- `JPM` looks like a tail-risk / path-shape problem

### B. Expected-edge selector layer

The selector is not uniformly helpful.

In D1 holdout expected-edge funnel:

- `META` selected edge lift: `+0.0005294`
- `MSFT` selected edge lift: `+0.0005697`
- `JPM` selected edge lift: `+0.0002070`
- `GOOGL` selected edge lift: `+0.0023879`

But for strong names:

- `SPY` selected edge lift: `-0.0001637`
- `NVDA` selected edge lift: `-0.0000956`

Interpretation:

- the selector is directionally useful on weak names
- but it is also over-filtering or mis-ranking some strong names
- this is why D3 could improve p25/stability while breaking downstream policy utility

Conclusion:

- the ranker and expected-edge handoff is part of the failure
- the selector should not be treated as fully solved

### C. Overtrading / signal density problem on weak names

Compare D1 vs E0 on the same holdout symbols:

- `JPM` active signal rate: `21.34%` -> `2.33%`
- `MSFT` active signal rate: `28.80%` -> `3.55%`
- `GOOGL` active signal rate: `18.76%` -> `1.69%`
- `NVDA` active signal rate: `34.61%` -> `3.02%`

E0 is weaker overall, but these weak symbols improve materially because it
trades them far less.

Interpretation:

- D1 is likely too willing to fire on low-quality symbol-specific setups
- the issue is not only edge estimation; it is also symbol-level activity density

This is the strongest evidence that a targeted fix should first reduce exposure
to the bad symbols rather than broaden the universe.

### D. Regime problem is not global

D1 holdout regime Sharpe:

- `high_vol = 2.1907`
- `trend_down = 1.8815`
- `trend_up = 2.0556`

Interpretation:

- D1 is not failing because one whole regime is broken
- the failure is cross-symbol heterogeneity inside otherwise healthy regimes

## What The Evidence Rules Out

These are weak explanations given current evidence:

- "Data snapshot is bad"
- "Need more model families first"
- "Need more symbols right now"
- "Need a generic regime fix before anything else"

These are stronger explanations:

- certain symbols do not share the same usable alpha structure
- the current ranker is overactive on some symbols
- the expected-edge selector is not fully aligned with symbol-level trade quality

## Recommended Next Steps

### Immediate

1. Keep `D1` as the backtest and paper-trading reference model.
2. Do not promote the full clean-core `11` symbol universe yet.
3. Treat `META`, `JPM`, and `MSFT` as the first exclusion candidates.

### Next retrain

Run `D1b`:

- same clean `sec_filings + corporate_actions` stack
- same snapshot discipline
- reduced universe:
  - remove `META`
  - remove `JPM`
  - remove `MSFT`

Expected effect:

- biggest chance of lifting `holdout_symbol_sharpe_p25`
- likely lower PBO through lower cross-symbol instability
- minimal risk of losing the core D1 edge, because the strongest symbols stay

### Paper-trading rollout

Use a stable subset first, not the full core universe.

Best current starting candidates by cross-run consistency:

- `SPY`
- `QQQ`
- `AMD`
- `XOM`
- `GOOGL` or `NVDA`

Avoid in first paper wave:

- `META`
- `JPM`
- `MSFT`

### Diagnostics before any broader expansion

Before adding more history or more symbols, add one targeted diagnostic pass:

1. worst-loss windows by symbol
2. symbol x regime loss concentration
3. symbol-specific activity density and turnover concentration
4. selected-vs-base trade return delta by symbol

If these diagnostics still show the same pattern after `D1b`, then the next
step is not more symbols. The next step is:

- per-symbol activity throttling, or
- cohort-specific modeling, or
- selector retraining with symbol-aware candidate gating

## Recommendation On More Data / More Symbols

### Longer history

Potentially useful, but second-order right now.

Current clean snapshot already spans:

- `2021-03-15` to `2026-03-30`

That is enough to first stabilize symbol-tail behavior on the approved core
universe. Add more history only after the reduced-universe champion is stable.

### More symbols

Not recommended yet.

If p25 fails on `11` curated symbols, broadening the universe now will most
likely dilute the signal and deepen the left tail before it improves it.

## Final Recommendation

Do not do a blind retrain on the same full universe.

Do this instead:

1. accept the diagnosis that the current failure is symbol-specific
2. retrain `D1b` on the reduced universe
3. start paper on the stable subset only
4. postpone universe expansion until p25 and PBO are stable on the reduced set
