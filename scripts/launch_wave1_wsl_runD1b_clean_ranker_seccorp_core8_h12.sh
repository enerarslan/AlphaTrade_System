#!/usr/bin/env bash
set -euo pipefail

ALPHATRADE_SYMBOLS_FILE="data/training/universes/wave1_clean_core8_no_meta_jpm_msft_20260404.json" \
bash "$(dirname "$0")/launch_wave1_wsl_clean_ranker_scope_safe_h12.sh" \
  "runD1b_clean_ranker_seccorp_core8_h12" \
  --reference-feature-sources sec_filings corporate_actions \
  "$@"
