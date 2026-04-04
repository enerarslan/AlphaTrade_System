#!/usr/bin/env bash
set -euo pipefail

approved_snapshot_id="snap_86ac4fbc2826d9cd"
snapshot_bundle="models/snapshots/${approved_snapshot_id}/dataset_bundle.manifest.json"
symbols_file="data/training/universes/wave1_clean_core11_20260402.json"
run_suffix="runE1_clean_elastic_seccorp_core11_h12"
log_file="logs/train_wave1_elastic_research_${run_suffix}.stdout.log"
model_name="wave1_elastic_research_20260404_${run_suffix}"
session_name="alphatrade_${run_suffix}"
launcher_script="$(dirname "$0")/wsl_tmux_launcher.sh"

blocked_overrides=(
  "--dataset-snapshot-bundle"
  "--symbols"
  "--symbols-file"
  "--training-profile"
  "--snapshot-only"
  "--disable-auto-snapshot-reuse"
)

for arg in "$@"; do
  for blocked in "${blocked_overrides[@]}"; do
    if [[ "$arg" == "$blocked" ]]; then
      echo "Refusing elastic sec+corp replay override for ${blocked}; edit the launcher intentionally instead." >&2
      exit 1
    fi
  done
done

cd ~/AlphaTrade_wsl

if [[ ! -f "$snapshot_bundle" ]]; then
  echo "Approved sec+corp snapshot bundle not found: $snapshot_bundle" >&2
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

mkdir -p "$(dirname "$log_file")"

command=(
  .venv/bin/python
  main.py
  train
  --model elastic_net
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
  --reference-feature-sources sec_filings corporate_actions
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
  --max-cross-sectional-rows 500000
  "$@"
)

"$launcher_script" "$session_name" "$log_file" "${command[@]}"
