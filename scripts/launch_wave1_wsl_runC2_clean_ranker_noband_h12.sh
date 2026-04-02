#!/usr/bin/env bash
set -euo pipefail

"$(dirname "$0")/launch_wave1_wsl_clean_ranker_research_h12.sh" \
  "runC2_clean_ranker_noband_core11_h12" \
  --disable-dynamic-no-trade-band \
  "$@"
