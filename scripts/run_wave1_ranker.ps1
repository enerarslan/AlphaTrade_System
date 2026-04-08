<#
.SYNOPSIS
Launch the recommended wave1 LightGBM ranker workflow inside WSL.

.DESCRIPTION
This replaces the ad-hoc `launch_wave1_wsl_*` family with one durable entrypoint.
Default candidate settings are based on the stronger D2 sec-filings + corporate-actions
baseline and intentionally exclude the noisier short-sale / FTD extensions because the
latest D2f full-fit failed all outer-fold stability gates.

Recommended usage:
  Candidate discovery:
    .\scripts\run_wave1_ranker.ps1 -Mode full -Profile candidate -Tail

  Promotion replay from an approved immutable snapshot:
    .\scripts\run_wave1_ranker.ps1 -Mode fit -Profile promotion `
      -SnapshotManifest models/snapshots/<snapshot_id>/dataset_bundle.manifest.json -Tail
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("full", "snapshot", "fit")]
    [string]$Mode = "full",

    [ValidateSet("candidate", "promotion")]
    [string]$Profile = "candidate",

    [string]$RunSuffix = "",
    [string]$WorkspaceDir = "/root/AlphaTrade_wsl",
    [string]$SymbolsFile = "data/training/universes/wave1_clean_core11_20260402.json",
    [string]$SnapshotManifest = "",
    [switch]$Tail,
    [switch]$Foreground,
    [switch]$SkipCodeSync,
    [switch]$SkipPreflight,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraTrainArgs = @()
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Quote-Bash {
    param([AllowEmptyString()][string]$Value)
    return "'" + $Value.Replace("'", "'`"`'`"`'") + "'"
}

function Join-BashArgs {
    param([Parameter(Mandatory = $true)][string[]]$Values)
    return (($Values | ForEach-Object { Quote-Bash $_ }) -join " ")
}

function Invoke-WSLBash {
    param([Parameter(Mandatory = $true)][string]$Command)
    & wsl.exe -e bash -lc $Command
}

$blockedExtraArgs = @(
    "--model",
    "--name",
    "--training-profile",
    "--symbols",
    "--symbols-file",
    "--snapshot-only",
    "--dataset-snapshot-bundle",
    "--strict-snapshot-replay",
    "--disable-auto-snapshot-reuse",
    "--allow-unstable-outer-fold-fallback",
    "--disable-unstable-outer-fold-fallback"
)
foreach ($arg in $ExtraTrainArgs) {
    if ($blockedExtraArgs -contains $arg) {
        throw "ExtraTrainArgs cannot override $arg. Edit the launcher intentionally instead."
    }
}

if ($Profile -eq "promotion" -and $Mode -eq "full") {
    throw "Promotion runs should replay an approved immutable snapshot. Use -Mode fit."
}
if (($Mode -eq "fit" -or $Profile -eq "promotion") -and [string]::IsNullOrWhiteSpace($SnapshotManifest)) {
    throw "SnapshotManifest is required for fit-only and promotion runs."
}
if ($SnapshotManifest -and -not $SnapshotManifest.EndsWith("dataset_bundle.manifest.json")) {
    throw "SnapshotManifest must point to a dataset_bundle.manifest.json file."
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
if ([string]::IsNullOrWhiteSpace($RunSuffix)) {
    $RunSuffix = if ($Profile -eq "candidate") {
        "stable_ranker_seccorp_hardening_core11_h12_$timestamp"
    } else {
        "promotion_ranker_seccorp_hardening_core11_h12_$timestamp"
    }
}

$trainingProfile = if ($Profile -eq "candidate") { "research" } else { "promotion" }
$modelName = "wave1_ranker_${trainingProfile}_20260402_${RunSuffix}"
$snapshotPrepName = "${modelName}_snapshotprep"
$sessionName = "alphatrade_${RunSuffix}"
$logFile = "logs/train_${RunSuffix}_${Mode}.stdout.log"

$baseArgs = @(
    ".venv/bin/python",
    "main.py",
    "train",
    "--model",
    "lightgbm_ranker",
    "--training-profile",
    $trainingProfile,
    "--symbols-file",
    $SymbolsFile,
    "--timeframe",
    "15Min",
    "--cv-method",
    "purged_kfold",
    "--n-splits",
    "4",
    "--n-trials",
    "100",
    "--nested-outer-splits",
    "3",
    "--nested-inner-splits",
    "3",
    "--nested-outer-stability-ratio-cap",
    "1.25",
    "--nested-outer-stability-min-trials",
    "8",
    "--optuna-storage-dir",
    "models/optuna_state",
    "--holdout-pct",
    "0.20",
    "--label-horizons",
    "1",
    "5",
    "20",
    "--primary-horizon",
    "12",
    "--profit-taking",
    "0.015",
    "--stop-loss",
    "0.010",
    "--max-holding",
    "20",
    "--spread-bps",
    "1",
    "--slippage-bps",
    "3",
    "--impact-bps",
    "2",
    "--label-min-signal-abs-return-bps",
    "10",
    "--label-neutral-buffer-bps",
    "5",
    "--label-edge-cost-buffer-bps",
    "3",
    "--feature-groups",
    "technical",
    "statistical",
    "microstructure",
    "cross_sectional",
    "--reference-feature-sources",
    "sec_filings",
    "corporate_actions",
    "--enable-expected-edge-symbol-priors",
    "--feature-selection-min-ic",
    "0.015",
    "--feature-selection-max-corr",
    "0.90",
    "--feature-selection-max-features",
    "160",
    "--feature-selection-stability-iterations",
    "16",
    "--feature-selection-min-stability-support",
    "0.65",
    "--objective-weight-turnover",
    "0.20",
    "--objective-weight-calibration",
    "0.35",
    "--objective-weight-tail-risk",
    "0.45",
    "--objective-weight-symbol-concentration",
    "0.30",
    "--execution-turnover-cap",
    "0.45",
    "--execution-max-symbol-entry-share",
    "0.55",
    "--min-confidence-position-scale",
    "0.30",
    "--probability-calibration-method",
    "isotonic",
    "--max-cross-sectional-rows",
    "500000",
    "--require-model-gpu"
)
if ($Profile -eq "candidate") {
    $baseArgs += "--disable-unstable-outer-fold-fallback"
}
if ($ExtraTrainArgs.Count -gt 0) {
    $baseArgs += $ExtraTrainArgs
}

$snapshotCommand = @($baseArgs + @(
        "--name",
        $snapshotPrepName,
        "--snapshot-only",
        "--disable-auto-snapshot-reuse"
    ))
$fitCommand = @($baseArgs + @("--name", $modelName))

$quotedWorkspaceDir = Quote-Bash $WorkspaceDir
$quotedSymbolsFile = Quote-Bash $SymbolsFile
$quotedSnapshotManifest = Quote-Bash $SnapshotManifest
$quotedLogFile = Quote-Bash $logFile
$quotedSessionName = Quote-Bash $sessionName
$snapshotCommandText = Join-BashArgs $snapshotCommand
$fitCommandText = Join-BashArgs $fitCommand
$forceForeground = if ($Foreground.IsPresent) { "1" } else { "0" }

if (-not $SkipCodeSync) {
    $repoRootWin = Split-Path $PSScriptRoot -Parent
    $repoRootWsl = (Invoke-WSLBash ("wslpath -a " + (Quote-Bash $repoRootWin))).Trim()
    $syncTemplate = @'
set -euo pipefail
mkdir -p __WORKSPACE__/scripts __WORKSPACE__/quant_trading_system/models
cp -f __REPO_ROOT__/scripts/train.py __WORKSPACE__/scripts/train.py
cp -f __REPO_ROOT__/quant_trading_system/models/feature_selection.py __WORKSPACE__/quant_trading_system/models/feature_selection.py
'@
    $syncScript = $syncTemplate.
        Replace("__WORKSPACE__", $quotedWorkspaceDir).
        Replace("__REPO_ROOT__", (Quote-Bash $repoRootWsl))
    Invoke-WSLBash $syncScript | Out-Host
}

if (-not $SkipPreflight) {
    $preflightTemplate = @'
set -euo pipefail
cd __WORKSPACE__
[[ -d .venv ]] || { echo "WSL workspace is missing .venv: __WORKSPACE_RAW__" >&2; exit 1; }
[[ -f __SYMBOLS_FILE__ ]] || { echo "Symbols file not found: __SYMBOLS_FILE_RAW__" >&2; exit 1; }
if [[ "__MODE__" == "fit" || "__PROFILE__" == "promotion" ]]; then
  [[ -f __SNAPSHOT_MANIFEST__ ]] || { echo "Snapshot manifest not found: __SNAPSHOT_MANIFEST_RAW__" >&2; exit 1; }
  [[ "__SNAPSHOT_MANIFEST_RAW__" == *dataset_bundle.manifest.json ]] || {
    echo "Snapshot manifest must point to dataset_bundle.manifest.json" >&2
    exit 1
  }
fi
if [[ "__FORCE_FOREGROUND__" != "1" ]] && ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for durable launches. Re-run with -Foreground only intentionally." >&2
  exit 1
fi
if command -v pg_isready >/dev/null 2>&1; then
  pg_isready -q || { echo "PostgreSQL is not ready in WSL." >&2; exit 1; }
fi
if command -v redis-cli >/dev/null 2>&1; then
  redis-cli ping >/dev/null || { echo "Redis ping failed in WSL." >&2; exit 1; }
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi >/dev/null 2>&1 || { echo "NVIDIA GPU stack is unavailable in WSL." >&2; exit 1; }
fi
'@
    $preflightScript = $preflightTemplate.
        Replace("__WORKSPACE__", $quotedWorkspaceDir).
        Replace("__WORKSPACE_RAW__", $WorkspaceDir).
        Replace("__SYMBOLS_FILE__", $quotedSymbolsFile).
        Replace("__SYMBOLS_FILE_RAW__", $SymbolsFile).
        Replace("__SNAPSHOT_MANIFEST__", $quotedSnapshotManifest).
        Replace("__SNAPSHOT_MANIFEST_RAW__", $SnapshotManifest).
        Replace("__MODE__", $Mode).
        Replace("__PROFILE__", $Profile).
        Replace("__FORCE_FOREGROUND__", $forceForeground)
    Invoke-WSLBash $preflightScript | Out-Host
}

$runnerTemplate = @'
#!/usr/bin/env bash
set -euo pipefail

cd __WORKSPACE__
mkdir -p "$(dirname __LOGFILE__)"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONFAULTHANDLER=1

run_pipeline() {
  local snapshot_manifest_path=__SNAPSHOT_MANIFEST__
  if [[ "__MODE__" == "snapshot" || "__MODE__" == "full" ]]; then
    echo "[snapshot] Building immutable dataset snapshot bundle..."
    snapshot_command=( __SNAPSHOT_COMMAND__ )
    "${snapshot_command[@]}"
  fi

  if [[ "__MODE__" == "fit" || "__MODE__" == "full" ]]; then
    if [[ "__MODE__" == "full" ]]; then
      snapshot_manifest_path="$(find models/snapshots -maxdepth 2 -type f -name dataset_bundle.manifest.json -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')"
    fi
    if [[ -z "$snapshot_manifest_path" || ! -f "$snapshot_manifest_path" ]]; then
      echo "No dataset snapshot bundle manifest is available for fit stage." >&2
      exit 1
    fi
    if [[ "$snapshot_manifest_path" != *dataset_bundle.manifest.json ]]; then
      echo "Resolved snapshot path is not a dataset bundle manifest: $snapshot_manifest_path" >&2
      exit 1
    fi
    echo "[fit] Replaying immutable dataset snapshot: $snapshot_manifest_path"
    fit_command=( __FIT_COMMAND__ )
    fit_command+=(--dataset-snapshot-bundle "$snapshot_manifest_path" --strict-snapshot-replay)
    "${fit_command[@]}"
  fi
}

run_logged_command() {
  run_pipeline 2>&1 | tee __LOGFILE__
  return "${PIPESTATUS[0]}"
}

if [[ "__FORCE_FOREGROUND__" == "1" ]] || [[ -n "${TMUX:-}" ]]; then
  run_logged_command
  exit $?
fi

if tmux has-session -t __SESSION_NAME__ 2>/dev/null; then
  echo "tmux session already exists: __SESSION_NAME_RAW__"
  echo "Attach with: tmux attach -t __SESSION_NAME_RAW__"
  exit 0
fi

printf -v cwd_text '%q' "$PWD"
printf -v log_text '%q' __LOGFILE__
launch_script="$(mktemp "${TMPDIR:-/tmp}/alphatrade-__SESSION_NAME_RAW__.XXXXXX.sh")"
cat > "$launch_script" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd __WORKSPACE__
run_pipeline() {
  local snapshot_manifest_path=__SNAPSHOT_MANIFEST__
  if [[ "__MODE__" == "snapshot" || "__MODE__" == "full" ]]; then
    echo "[snapshot] Building immutable dataset snapshot bundle..."
    snapshot_command=( __SNAPSHOT_COMMAND__ )
    "${snapshot_command[@]}"
  fi
  if [[ "__MODE__" == "fit" || "__MODE__" == "full" ]]; then
    if [[ "__MODE__" == "full" ]]; then
      snapshot_manifest_path="$(find models/snapshots -maxdepth 2 -type f -name dataset_bundle.manifest.json -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')"
    fi
    if [[ -z "$snapshot_manifest_path" || ! -f "$snapshot_manifest_path" ]]; then
      echo "No dataset snapshot bundle manifest is available for fit stage." >&2
      exit 1
    fi
    if [[ "$snapshot_manifest_path" != *dataset_bundle.manifest.json ]]; then
      echo "Resolved snapshot path is not a dataset bundle manifest: $snapshot_manifest_path" >&2
      exit 1
    fi
    echo "[fit] Replaying immutable dataset snapshot: $snapshot_manifest_path"
    fit_command=( __FIT_COMMAND__ )
    fit_command+=(--dataset-snapshot-bundle "$snapshot_manifest_path" --strict-snapshot-replay)
    "${fit_command[@]}"
  fi
}
run_pipeline 2>&1 | tee __LOGFILE__
exit ${PIPESTATUS[0]}
EOF
chmod +x "$launch_script"
tmux new-session -d -s __SESSION_NAME__ "bash $(printf '%q' "$launch_script")"
echo "Started tmux session: __SESSION_NAME_RAW__"
echo "Log file: __LOGFILE_RAW__"
echo "Attach with: tmux attach -t __SESSION_NAME_RAW__"
'@

$runnerScript = $runnerTemplate.
    Replace("__WORKSPACE__", $quotedWorkspaceDir).
    Replace("__LOGFILE__", $quotedLogFile).
    Replace("__LOGFILE_RAW__", $logFile).
    Replace("__SESSION_NAME__", $quotedSessionName).
    Replace("__SESSION_NAME_RAW__", $sessionName).
    Replace("__SNAPSHOT_MANIFEST__", $quotedSnapshotManifest).
    Replace("__MODE__", $Mode).
    Replace("__FORCE_FOREGROUND__", $forceForeground).
    Replace("__SNAPSHOT_COMMAND__", $snapshotCommandText).
    Replace("__FIT_COMMAND__", $fitCommandText)

$runnerPath = Join-Path $env:TEMP "alphatrade_${RunSuffix}_${Mode}.sh"
[System.IO.File]::WriteAllText($runnerPath, $runnerScript, [System.Text.Encoding]::ASCII)
$runnerWslPath = (Invoke-WSLBash ("wslpath -a " + (Quote-Bash $runnerPath))).Trim()

Write-Host "Launching wave1 ranker run..."
Write-Host "  Mode: $Mode"
Write-Host "  Profile: $Profile ($trainingProfile)"
Write-Host "  Run suffix: $RunSuffix"
Write-Host "  Session: $sessionName"
Write-Host "  Log: $WorkspaceDir/$logFile"
if ($SnapshotManifest) {
    Write-Host "  Snapshot manifest: $SnapshotManifest"
}

Invoke-WSLBash ("bash " + (Quote-Bash $runnerWslPath)) | Out-Host

if ($Tail) {
    $tailCommand = "wsl.exe -e bash -lc " + (Quote-Bash ("cd $WorkspaceDir && tail -f $logFile"))
    Start-Process powershell -ArgumentList @("-NoExit", "-Command", $tailCommand) | Out-Null
}
