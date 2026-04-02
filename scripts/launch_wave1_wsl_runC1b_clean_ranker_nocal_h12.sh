#!/usr/bin/env bash
set -euo pipefail

"$(dirname "$0")/launch_wave1_wsl_clean_ranker_research_h12.sh" \
  "runC1b_clean_ranker_nocal_core11_h12" \
  --disable-probability-calibration \
  "$@"
