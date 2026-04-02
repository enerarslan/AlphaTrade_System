# AlphaTrade LightGBM + TCN Work Plan

**Date:** 2026-04-02
**Status:** Revised after the completed WSL research sweep, fallback comparison, and an interrupted promotion run
**Scope:** Reach a production-ready LightGBM baseline before replay, paper trading, or any TCN comparison

## 1. Objective

Wave 1 is no longer about proving that the stack can train. It is about producing a LightGBM
candidate that can survive institutional promotion gates and then hold up in replay and paper
trading.

The program must:

- use PostgreSQL and the frozen snapshot bundle as the source of truth
- treat research runs as diagnostics, not promotion evidence
- improve trade quality without collapsing candidate trade flow
- keep validation, replay, and paper-trading contracts aligned through the promotion package
- defer TCN until a LightGBM baseline is demonstrably stable

Wave 1 working stack:

- **Primary family:** `lightgbm_ranker`
- **Fallback comparator:** `lightgbm`
- **Deferred challenger:** `tcn`
- **Policy stack:** probability calibration + asymmetric side policy + regime-conditioned policy + expected-edge policy + universe quality gate

## 2. Current Evidence

Completed runs on the serious WSL-native snapshot workflow changed the state of the plan.

### 2.1 Completed Research Evidence

Frozen snapshot used by the serious runs:

- `snapshot_id=snap_76d371975b6e817d`
- `row_count=379635`
- `symbol_count=11`
- `quality_report_passed=false`

Why the quality report failed:

- missing-bars ratio was `0.0188`, above the configured `0.0100` threshold
- `GLD` dropped from the effective universe during symbol-quality filtering

Serious research sweep result summary:

- `lightgbm_ranker h12` was the best research candidate by governance score (`29.24`)
- `lightgbm_ranker h12` still failed validation on `max_drawdown`, `min_trades`, `risk_adjusted_positive`, `max_pbo`, `max_white_reality_pvalue`, `max_holdout_drawdown`, and `min_holdout_symbol_p25_sharpe`
- `lightgbm_ranker h12` metrics worth remembering:
  - `mean_sharpe=1.3114`
  - `mean_trade_count=11.0`
  - `mean_risk_adjusted_score=-2.0554`
  - `holdout_sharpe=0.7742`
  - `holdout_max_drawdown=0.4331`
  - `holdout_symbol_sharpe_p25=-1.1827`
  - `probability_calibration_brier_improvement=0.0146`
- `expected-edge` did not train because candidate-trade volume was too low

Fallback comparison result summary:

- `lightgbm h12` was materially worse than `lightgbm_ranker h12`
- `lightgbm h12` had `mean_sharpe=-0.6456`, `mean_trade_count=7.33`, `risk_adjusted_score=-3.3098`, `pbo=0.875`, and `white_reality_pvalue=0.8144`
- `lightgbm` remains a fallback comparator, not the current promotion path

Promotion evidence summary:

- the first promotion attempt for `lightgbm_ranker h12` started from the same frozen snapshot
- that run did not complete cleanly and produced no model, promotion package, or replay manifest
- treat that interrupted promotion attempt as non-evidence

Implication:

- there is currently no eligible champion
- the best path is not to push promotion harder; it is to repair data quality, improve training diagnostics, and produce a materially better research candidate first

## 3. Strategic Decisions

1. `lightgbm_ranker` with primary horizon `12` remains the Wave 1 anchor until another LightGBM variant beats it on the same snapshot with better robustness.
2. `lightgbm` stays in the plan only as a fallback comparator and sanity check.
3. Do not rerun promotion on the current configuration until the next research candidate clears a pre-promotion checklist.
4. Do not open the TCN branch until LightGBM passes promotion, replay, and early paper validation.
5. Do not compare runs across snapshots unless the previous snapshot is formally retired for a documented quality reason.
6. Do not tighten labels further by default. The current failure mode is under-trading with weak robustness, not uncontrolled churn.
7. All heavy training must run from Linux-native WSL storage in a durable session, not from `/mnt/c/...` and not from a fragile foreground shell.

## 4. Production Readiness Requirements

### 4.1 Pre-Promotion Research Requirements

Before another promotion run is authorized, the leading research candidate must satisfy all of the
following on one frozen snapshot:

- snapshot quality report passes with no unresolved threshold breach
- candidate universe is explicit and stable; no silent symbol drop is ignored
- `expected-edge` policy trains successfully; no `received 0 candidate trades` condition
- `mean_trade_count >= 100`
- `mean_risk_adjusted_score > 0`
- `holdout_max_drawdown <= 0.35`
- `holdout_symbol_sharpe_p25 >= -0.10`
- `pbo <= 0.45`
- `white_reality_pvalue <= 0.10`
- all research validation layers are either passing or one documented shortfall remains with explicit sign-off

### 4.2 Promotion Requirements

Promotion is only valid when:

- `training_profile=promotion`
- the exact same snapshot bundle is reused with `--strict-snapshot-replay`
- the auto-tightened institutional live profile in `scripts/train.py` is accepted as the gatekeeper
- the artifact reports `deployment_plan.ready_for_production=true`
- the champion snapshot shows an eligible promoted candidate

### 4.3 Pre-Live Requirements

Before any live capital increase:

- replay passes across the three required windows
- paper trading runs on the exact promoted artifact for at least `2` weeks, preferably `4`
- execution costs, slippage, risk warnings, symbol concentration, and expected-edge pass rates remain consistent with the promoted contract

## 5. Workstreams

### 5.1 Data Quality And Snapshot Integrity

Current blocker:

- the active serious snapshot is diagnostic-grade, not promotion-grade, because the quality report failed on missing bars
- latest Task 1 preflight (`wave1_snapshotonly_20260402_h12_preflight3`) now shows the regular-session quality report passes, but snapshot review still fails because `GLD` is silently dropped by the symbol-quality gate

Required actions:

- root-cause the 15-minute bar gaps in PostgreSQL and patch the ingestion/backfill path
- regenerate the serious snapshot only after the quality report passes
- make the quality report and dropped-symbol list mandatory review items before approving any promotion run
- use `python main.py train ... --snapshot-only` to generate the snapshot bundle and a review artifact before spending research or promotion compute; if Phase 1 already fails quality or drops symbols, the run now stops there instead of paying feature-compute cost
- formally retire the current snapshot for promotion use if the gaps cannot be repaired cleanly

### 5.2 Training Diagnostics And Traceability

Current blocker:

- the system tells us that trade volume is too low and robustness is weak, but it does not yet expose enough funnel detail to show where candidate trades die

Required actions:

- add an auditable signal-funnel artifact for every training run
- add per-symbol and per-regime trade diagnostics to the artifacts package
- make feature-selection deltas explicit so that `116 -> 116` is visible as a non-binding selection outcome
- add a run-manifest index so operators can recover command, snapshot, metrics, and artifact paths without log archaeology

### 5.3 Model Improvement Policy

Current blocker:

- the best ranker candidate has decent headline Sharpe but fails on drawdown, trade count, cross-symbol robustness, and statistical validity

Required actions:

- keep `lightgbm_ranker h12` as the baseline comparator
- investigate trade starvation before tightening labels again
- make one change at a time on the same snapshot: trade-admission policy, feature pruning, downside-risk penalties, or symbol-tail protection
- require every challenger to explain why it should improve one failed gate without silently harming another

### 5.4 Promotion, Replay, And Paper Trading

Current blocker:

- promotion was attempted before the research evidence was strong enough and the run was interrupted

Required actions:

- rerun promotion only after the next clean research winner is chosen
- replay only the promoted artifact
- use paper trading to validate execution realism and policy behavior, not to compensate for a weak training candidate

Operational procedure:

- launch serious WSL runs through `scripts/launch_wave1_wsl_run*.sh`; `tmux` is now a hard requirement for durable WSL execution
- the Wave 1 WSL launchers now pass through extra CLI flags, so operators can append `--snapshot-only` for preflight snapshot review without cloning a script
- resume an interrupted search by rerunning the same launch script with the same `--name`; Optuna state is persisted under `models/optuna_state`
- if the session is still alive, reattach with `tmux attach -t <session_name>` instead of starting a second foreground process
- strict-snapshot promotion runs now enforce a pre-promotion checklist against the latest matching research matrix before training starts
- if an operator must bypass that guard intentionally, set `ALPHATRADE_FORCE_PROMOTION_PRECHECK_BYPASS=1` before launching `launch_wave1_wsl_run3.sh`

## 6. Recommended Next Runs

### Run A: Data-Quality Rebuild

Goal:

- repair the missing-bar issue and produce a fresh serious snapshot with `quality_report_passed=true`

Rules:

- same 15-minute scope unless the data audit proves that the universe must change
- if the universe changes, document the exact reason and retire the old snapshot
- the current preflight evidence points to `GLD` as the only dropped symbol, driven by `median_dollar_volume`, so the next Run A decision is whether to retire `GLD` explicitly or adjust the liquidity gate with written justification
- Wave 1 now uses the explicit universe file `data/training/universes/wave1_clean_core11_20260402.json`, which retires `GLD` with a recorded reason instead of allowing silent symbol drop
- first rebuild should run in `--snapshot-only` mode so `*.snapshot_review.json` captures the quality report and dropped-symbol list before the next research candidate is launched
- snapshot-only runs now skip post-selection PostgreSQL feature persistence entirely; Run A only needs the dataset bundle and review package, and removing that write path avoids wasting time in non-essential cache materialization

### Run B: Clean-Snapshot Ranker Baseline

Goal:

- rerun `lightgbm_ranker` on the clean snapshot
- keep `primary_horizon=12` as the baseline starting point

Rules:

- no hidden policy changes
- generate the new diagnostics artifacts described in this plan
- launch the baseline with `scripts/launch_wave1_wsl_runB_clean_ranker_h12.sh <dataset_bundle.manifest.json>` so the exact Run A snapshot bundle is replayed under research settings

### Run C: Ranker Trade-Flow Challenger

Goal:

- increase candidate-trade flow enough for expected-edge training and the `min_trades` gate without destroying holdout quality

Allowed levers:

- one explicit trade-admission or threshold-policy change justified by signal-funnel evidence

### Run D: Ranker Robustness Challenger

Goal:

- reduce drawdown, PBO, and weak tail-symbol behavior on the same snapshot

Allowed levers:

- stronger feature pruning
- stronger downside-risk or concentration penalties
- explicit symbol-tail protection if diagnostics show a small set of names is poisoning the p25 tail

### Run E: Promotion On The Winner

Goal:

- run the best research configuration under promotion settings on the same clean snapshot

Rules:

- durable WSL session only
- same snapshot bundle with `--strict-snapshot-replay`
- interrupted runs do not count as evidence

### Run F: Replay And Early Paper Validation

Goal:

- validate the promoted artifact across replay windows and then in paper trading

Replay windows:

- `2024-01-01` to `2024-01-05`
- `2024-08-01` to `2024-08-09`
- `2025-03-03` to `2025-03-14`

Paper minimum:

- `2` weeks minimum
- `4` weeks preferred

### Deferred Run G: TCN On The Accepted Snapshot

TCN stays deferred until all of the following are true:

- a LightGBM promotion candidate exists
- replay is coherent
- early paper trading is coherent
- the team still believes incremental sequence edge is worth the additional compute and operational complexity

## 7. Operating Rules

Allowed as evidence:

- research runs on an approved frozen snapshot
- promotion runs on the same frozen snapshot
- replay on the exact promoted artifact
- paper trading on the exact promoted artifact

Not allowed as evidence:

- smoke tests
- interrupted runs
- AAPL-only shortcuts
- repeated runs on silently changed snapshots
- CV-only wins with holdout collapse
- promotion on a failed-quality snapshot unless a written exception retires the quality threshold

## 8. Short Summary

The plan changed because the serious runs taught us something concrete: the system can now train on a
frozen snapshot in WSL, but the best current LightGBM candidate is still not promotion-ready. The
next work is therefore not a blind promotion retry or an early TCN branch. It is a disciplined
sequence of data-quality repair, better diagnostics, ranker improvement on one frozen snapshot, then
promotion, replay, and paper validation.
