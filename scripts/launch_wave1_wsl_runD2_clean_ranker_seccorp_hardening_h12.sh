#!/usr/bin/env bash
set -euo pipefail

bash "$(dirname "$0")/launch_wave1_wsl_clean_ranker_scope_safe_h12.sh" \
  "runD2_clean_ranker_seccorp_hardening_core11_h12" \
  --reference-feature-sources sec_filings corporate_actions \
  --n-trials 100 \
  --n-splits 4 \
  --nested-outer-splits 3 \
  --nested-inner-splits 3 \
  --holdout-pct 0.20 \
  --feature-selection-max-features 160 \
  --feature-selection-min-stability-support 0.65 \
  --execution-turnover-cap 0.45 \
  --execution-max-symbol-entry-share 0.55 \
  --min-confidence-position-scale 0.30 \
  --objective-weight-tail-risk 0.45 \
  --objective-weight-symbol-concentration 0.30 \
  "$@"
