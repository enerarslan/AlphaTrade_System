#!/usr/bin/env bash
set -euo pipefail

bash "$(dirname "$0")/launch_wave1_wsl_clean_ranker_scope_safe_h12.sh" \
  "runD1d_clean_ranker_seccorp_symbolaware_core11_h12" \
  --reference-feature-sources sec_filings corporate_actions \
  "$@"
