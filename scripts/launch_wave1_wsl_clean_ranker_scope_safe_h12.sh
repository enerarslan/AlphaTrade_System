#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run-suffix> [extra train args...]" >&2
  exit 1
fi

run_suffix="$1"
shift

symbols_file="${ALPHATRADE_SYMBOLS_FILE:-data/training/universes/wave1_clean_core11_20260402.json}"
training_profile="${ALPHATRADE_TRAINING_PROFILE:-research}"
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
      echo "Refusing scope-safe clean-run override for ${blocked}; edit the launcher intentionally instead." >&2
      exit 1
    fi
  done
done

case "$training_profile" in
  research|promotion)
    ;;
  *)
    echo "Unsupported ALPHATRADE_TRAINING_PROFILE: ${training_profile}. Use 'research' or 'promotion'." >&2
    exit 1
    ;;
esac

workspace_dir="${ALPHATRADE_WSL_WORKSPACE:-$HOME/AlphaTrade_wsl}"
cd "$workspace_dir"

if [[ "$training_profile" == "promotion" ]]; then
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Promotion launcher requires a real Git checkout/worktree. Current workspace is not a Git repository: ${workspace_dir}" >&2
    exit 1
  fi
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

log_file="logs/train_wave1_ranker_${training_profile}_${run_suffix}.stdout.log"
model_name="wave1_ranker_${training_profile}_20260402_${run_suffix}"
snapshot_prep_name="${model_name}_snapshotprep"
session_name="alphatrade_${run_suffix}"
launcher_script="$(dirname "$0")/wsl_tmux_launcher.sh"

mkdir -p "$(dirname "$log_file")"

common_args=(
  main.py
  train
  --model lightgbm_ranker
  --training-profile "$training_profile"
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
  --max-cross-sectional-rows 500000
  "$@"
)

prep_command=(
  .venv/bin/python
  "${common_args[@]}"
  --name "$snapshot_prep_name"
  --snapshot-only
  --disable-auto-snapshot-reuse
)
train_command=(
  .venv/bin/python
  "${common_args[@]}"
  --name "$model_name"
)

printf -v prep_text '%q ' "${prep_command[@]}"
printf -v train_text '%q ' "${train_command[@]}"
run_script="$(mktemp "${TMPDIR:-/tmp}/alphatrade-scope-safe-${run_suffix}.XXXXXX.sh")"
cat > "$run_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
echo "[scope-prep] Building fresh scope-matched dataset snapshot bundle..."
${prep_text}
echo "[train] Launching training run with auto-reused scope-matched snapshot bundle..."
${train_text}
EOF
chmod +x "$run_script"

bash "$launcher_script" "$session_name" "$log_file" bash "$run_script"
