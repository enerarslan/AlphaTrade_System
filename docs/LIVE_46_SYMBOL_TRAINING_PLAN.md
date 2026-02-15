# Live Training Plan (5Y / 46 Symbols)

## Objective
Train a production candidate that is robust across time, regimes, and symbols, then only promote models that pass strict statistical and holdout gates.

## What Is Enforced
1. Nested walk-forward optimization remains mandatory.
2. Multiple-testing correction remains mandatory (`deflated_sharpe` + `PBO`).
3. Automatic multi-symbol live profile activates when dataset scale is large:
   - symbol count >= 40
   - history >= 4.0 years
4. Holdout validation is expanded from aggregate-only checks to symbol-level checks.

## New Symbol-Level Holdout Gates
1. `min_holdout_symbol_coverage`: minimum ratio of holdout symbols with enough observations.
2. `min_holdout_symbol_p25_sharpe`: lower quartile Sharpe across evaluated holdout symbols.
3. `max_holdout_symbol_underwater_ratio`: maximum fraction of holdout symbols with negative Sharpe.

These are in addition to:
1. `min_holdout_sharpe`
2. `min_holdout_regime_sharpe`
3. `max_holdout_drawdown`
4. `max_pbo`
5. `min_deflated_sharpe` / `max_deflated_sharpe_pvalue`

## Operational Runbook
1. Ensure data coverage and quality:
   - `python main.py health check --full`
2. Run training with institutional defaults:
   - `python main.py train --model xgboost --symbols <46-symbol-list> --start-date 2021-01-01 --end-date 2025-12-31`
3. Run model-family sweep for champion/challenger:
   - `python main.py train --model all --symbols <46-symbol-list> --start-date 2021-01-01 --end-date 2025-12-31`
4. Review generated promotion package and gates before activation.

## Tuning Knobs (If Needed)
1. Disable auto profile explicitly:
   - `--disable-auto-live-profile`
2. Change activation criteria:
   - `--auto-live-profile-symbol-threshold`
   - `--auto-live-profile-min-years`
3. Adjust symbol-level gates:
   - `--min-holdout-symbol-coverage`
   - `--min-holdout-symbol-p25-sharpe`
   - `--max-holdout-symbol-underwater-ratio`

## Expected Outcome
Models are no longer accepted based only on aggregate holdout Sharpe. Promotion requires cross-sectional robustness across the symbol universe as well.
