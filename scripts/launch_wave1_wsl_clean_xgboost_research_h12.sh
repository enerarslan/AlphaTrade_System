#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run-suffix> [extra train args...]" >&2
  exit 1
fi

run_suffix="$1"
shift

approved_snapshot_id="snap_8803ebe20127c4fd"
retired_snapshot_id="snap_76d371975b6e817d"
snapshot_bundle="models/snapshots/${approved_snapshot_id}/dataset_bundle.manifest.json"
symbols_file="data/training/universes/wave1_clean_core11_20260402.json"

blocked_overrides=(
  "--dataset-snapshot-bundle"
  "--symbols"
  "--symbols-file"
  "--training-profile"
)

for arg in "$@"; do
  if [[ "$arg" == *"${retired_snapshot_id}"* ]]; then
    echo "Refusing retired snapshot override: ${arg}" >&2
    exit 1
  fi
  for blocked in "${blocked_overrides[@]}"; do
    if [[ "$arg" == "$blocked" ]]; then
      echo "Refusing clean-run override for ${blocked}; edit the launcher intentionally instead." >&2
      exit 1
    fi
  done
done

cd ~/AlphaTrade_wsl

if [[ ! -f "$snapshot_bundle" ]]; then
  echo "Approved clean snapshot bundle not found: $snapshot_bundle" >&2
  exit 1
fi

if [[ ! -f "$symbols_file" ]]; then
  echo "Approved clean universe file not found: $symbols_file" >&2
  exit 1
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONFAULTHANDLER=1

log_file="logs/train_wave1_xgboost_research_${run_suffix}.stdout.log"
model_name="wave1_xgboost_research_20260403_${run_suffix}"
session_name="alphatrade_${run_suffix}"
launcher_script="$(dirname "$0")/wsl_tmux_launcher.sh"

mkdir -p "$(dirname "$log_file")"

command=(
  .venv/bin/python
  main.py
  train
  --model xgboost
  --name "$model_name"
  --training-profile research
  --dataset-snapshot-bundle "$snapshot_bundle"
  --strict-snapshot-replay
  --symbols-file "$symbols_file"
  --timeframe 15Min
  --cv-method purged_kfold
  --n-splits 3
  --n-trials 30
  --optuna-storage-dir models/optuna_state
  --holdout-pct 0.15
  --label-horizons 1 5 20
  --primary-horizon 12
  --profit-taking 0.015
  --stop-loss 0.010
  --max-holding 20
  --spread-bps 1
  --slippage-bps 3
  --impact-bps 2
  --label-min-signal-abs-return-bps 10
  --label-neutral-buffer-bps 5
  --label-edge-cost-buffer-bps 3
  --feature-groups technical statistical microstructure cross_sectional
  --reference-feature-sources macro sec_filings news corporate_actions
  --feature-selection-min-ic 0.015
  --feature-selection-max-corr 0.90
  --feature-selection-max-features 180
  --feature-selection-stability-iterations 16
  --feature-selection-min-stability-support 0.60
  --objective-weight-turnover 0.20
  --objective-weight-calibration 0.35
  --execution-turnover-cap 0.60
  --min-confidence-position-scale 0.20
  --probability-calibration-method isotonic
  --require-model-gpu
  --max-cross-sectional-rows 500000
  "$@"
)

"$launcher_script" "$session_name" "$log_file" "${command[@]}"
