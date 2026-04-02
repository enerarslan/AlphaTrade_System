#!/usr/bin/env bash
set -euo pipefail

cd ~/AlphaTrade_wsl

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONFAULTHANDLER=1

log_file="logs/train_wave1_lightgbm_research_wsl_run2_gpucuda_h12.stdout.log"
model_name="wave1_lightgbm_research_20260402_wsl_run2_gpucuda_h12"
snapshot_bundle="models/snapshots/snap_76d371975b6e817d/dataset_bundle.manifest.json"
session_name="alphatrade_wave1_run2"
launcher_script="$(dirname "$0")/wsl_tmux_launcher.sh"

mkdir -p "$(dirname "$log_file")"

command=(
  .venv/bin/python
  main.py
  train
  --model lightgbm
  --name "$model_name"
  --training-profile research
  --dataset-snapshot-bundle "$snapshot_bundle"
  --strict-snapshot-replay
  --symbols SPY QQQ AAPL MSFT NVDA AMD AMZN META GOOGL JPM XOM GLD
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
  --require-gpu
  "$@"
)

"$launcher_script" "$session_name" "$log_file" "${command[@]}"
