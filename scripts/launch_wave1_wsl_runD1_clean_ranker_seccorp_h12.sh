#!/usr/bin/env bash
set -euo pipefail

"$(dirname "$0")/launch_wave1_wsl_clean_ranker_research_h12.sh" \
  "runD1_clean_ranker_seccorp_core11_h12" \
  --reference-feature-sources sec_filings corporate_actions \
  "$@"
