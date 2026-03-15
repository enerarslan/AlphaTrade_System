# AlphaTrade Training Pipeline Audit - Verified Update

> Date: 2026-03-15
> Scope: verification of `docs/TRAINING_PIPELINE_AUDIT.md` against the current codebase
> Status legend:
> - `CONFIRMED + FIXED`
> - `INACCURATE / STALE`
> - `PARTIALLY TRUE`
> - `POLICY / TUNING`, not a correctness bug

## Executive Summary

The previous audit mixed together:

- real correctness defects,
- observability gaps,
- policy/tuning opinions,
- and a few claims that no longer matched the current code.

After verification against the package code, the outcome is:

- 5 real defects were confirmed and fixed.
- 3 findings were partially true and have been reclassified.
- multiple original claims were inaccurate or stale.

This document replaces the earlier version so operators do not act on false positives.

## Confirmed And Fixed

| Finding | Verdict | Resolution |
|---|---|---|
| `meta_labeling.py` used training data for metrics when the temporal split was too small | `CONFIRMED + FIXED` | The meta-labeler now fails fast when an unbiased temporal holdout cannot be formed, instead of reporting optimistic in-sample precision/AUC. |
| `db_loader.py` Parquet load path had broken success/error handling | `CONFIRMED + FIXED` | The loader now infers/normalizes timeframe for Parquet inputs and raises `Failed to load Parquet ...` with the correct path. |
| Feature persistence had no upsert behavior for duplicate feature keys | `CONFIRMED + FIXED` | `FeatureRepository.bulk_insert()` now performs upsert semantics instead of relying on plain inserts. |
| `live_feed.py` circuit breaker stored a naive failure timestamp | `CONFIRMED + FIXED` | Failure timestamps are now recorded as UTC-aware datetimes. |
| `scripts/train.py` only checked DB/Redis liveness, not schema/data readiness | `CONFIRMED + FIXED` | Training startup now verifies required PostgreSQL tables and rejects an empty `ohlcv_bars` table. |

## Partially True Findings

| Original Claim | Verified Status | Notes |
|---|---|---|
| Forward-return outlier filtering was "silent" | `PARTIALLY TRUE` | It was not fully silent: `forward_outlier_filtered_count` was already present in target diagnostics. The observability was still weak, so outlier rates are now also propagated into training metrics. |
| ElasticNet "Platt scaling" was wrong | `PARTIALLY TRUE` | `predict_proba(..., calibration_method="platt")` is heuristic sigmoid scaling, not fitted logistic Platt scaling. However, proper calibration already exists through `calibrate(method="sigmoid")` plus `predict_proba_calibrated()`. |
| Validation gates were too loose | `POLICY / TUNING` | Default thresholds are permissive, but that is a governance choice, not a correctness bug. The code already exposes a stricter deployment helper preset. |

## Inaccurate Or Stale Findings

| Original Claim | Verified Status | Reason |
|---|---|---|
| Sample-weight NaN propagation in `target_engineering.py` | `INACCURATE / STALE` | The class-weight mapping is only applied to labels in `{0, 1}` and final sample weights are sanitized. |
| Cost model was incorrectly scaled by signal magnitude in `target_engineering.py` | `INACCURATE / STALE` | `primary_signals` are already discrete `{-1, 0, 1}`, so `abs(signal)` only gates cost on active trades. |
| Parquet loader was missing timeframe and would hit a DB primary-key violation | `INACCURATE / STALE` | Missing timeframe values were defaulted during row materialization. The real bug was undefined `timeframe`/`csv_path` handling plus weak normalization. |
| Quantile calibration in `classical_ml.py` was dead code | `INACCURATE / STALE` | The `quantile` branch is reachable through `predict_proba(calibration_method="quantile")`. |

## Changes Applied

- [`quant_trading_system/models/meta_labeling.py`](/C:/Users/ener/Desktop/AlphaTrade_System-master/quant_trading_system/models/meta_labeling.py)
- [`quant_trading_system/models/classical_ml.py`](/C:/Users/ener/Desktop/AlphaTrade_System-master/quant_trading_system/models/classical_ml.py)
- [`quant_trading_system/models/validation_gates.py`](/C:/Users/ener/Desktop/AlphaTrade_System-master/quant_trading_system/models/validation_gates.py)
- [`quant_trading_system/data/db_loader.py`](/C:/Users/ener/Desktop/AlphaTrade_System-master/quant_trading_system/data/db_loader.py)
- [`quant_trading_system/data/data_access.py`](/C:/Users/ener/Desktop/AlphaTrade_System-master/quant_trading_system/data/data_access.py)
- [`quant_trading_system/database/repository.py`](/C:/Users/ener/Desktop/AlphaTrade_System-master/quant_trading_system/database/repository.py)
- [`quant_trading_system/data/live_feed.py`](/C:/Users/ener/Desktop/AlphaTrade_System-master/quant_trading_system/data/live_feed.py)
- [`scripts/train.py`](/C:/Users/ener/Desktop/AlphaTrade_System-master/scripts/train.py)
- [`quant_trading_system/models/target_engineering.py`](/C:/Users/ener/Desktop/AlphaTrade_System-master/quant_trading_system/models/target_engineering.py)

## Test Coverage Added Or Updated

- meta-labeling rejects biased small-sample evaluation
- ElasticNet probability calibration semantics are explicit and backward-compatible
- default validation gates are now institutional-safe by default
- Parquet loader defaults/infers timeframe correctly
- Parquet loader surfaces the correct file path on errors
- multi-symbol data access now fails fast unless missing-symbol tolerance is explicitly configured
- feature writes upsert duplicate keys
- live-feed circuit-breaker timestamps stay UTC-aware
- training infrastructure checks reject missing or empty `ohlcv_bars`
- target-engineering diagnostics now include outlier/neutral rates and emit summary/warning logs

## Validation Run

Executed:

```bash
pytest tests/unit/test_target_engineering.py tests/unit/test_models.py tests/unit/test_data.py tests/unit/test_database.py tests/unit/test_train_script.py -q
```

Result:

- `235 passed`
- `1 skipped` (GPU-only test)

## Remaining Follow-Ups

- No remaining code changes from the verified audit are outstanding.
- Any further changes are optional governance tuning rather than unresolved correctness issues.
