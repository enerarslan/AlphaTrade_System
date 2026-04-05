$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$logFile = "logs/train_wave1_ranker_research_wsl_run1_gpucuda.stdout.log"
$modelName = "wave1_ranker_research_20260401_wsl_run1_gpucuda"

$wslCommand = @"
cd ~/AlphaTrade_wsl &&
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONFAULTHANDLER=1 &&
.venv/bin/python main.py train \
  --model lightgbm_ranker \
  --name $modelName \
  --training-profile research \
  --symbols SPY QQQ AAPL MSFT NVDA AMD AMZN META GOOGL JPM XOM GLD \
  --timeframe 15Min \
  --cv-method purged_kfold \
  --n-splits 3 \
  --n-trials 30 \
  --holdout-pct 0.15 \
  --label-horizons 1 5 20 \
  --primary-horizon-sweep 3 5 8 12 \
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
  --feature-selection-stability-iterations 16 \
  --feature-selection-min-stability-support 0.60 \
  --objective-weight-turnover 0.20 \
  --objective-weight-calibration 0.35 \
  --execution-turnover-cap 0.60 \
  --min-confidence-position-scale 0.20 \
  --probability-calibration-method isotonic \
  --require-model-gpu \
  --max-cross-sectional-rows 500000 \
  --disable-auto-snapshot-reuse \
  2>&1 | tee $logFile
"@

Write-Host "Launching Wave 1 Run 1 in visible WSL console..."
Write-Host "Repo root: $repoRoot"
Write-Host "WSL log: ~/AlphaTrade_wsl/$logFile"
Write-Host ""

wsl.exe bash -lc $wslCommand
$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "WSL training process exited with code $exitCode"
exit $exitCode
