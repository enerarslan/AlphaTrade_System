#!/usr/bin/env bash
set -euo pipefail

"$(dirname "$0")/launch_wave1_wsl_clean_ranker_research_h12.sh" \
  "runD0_clean_ranker_priceonly_core11_h12" \
  --disable-reference-features \
  "$@"
