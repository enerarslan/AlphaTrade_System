#!/usr/bin/env bash
set -euo pipefail

bash "$(dirname "$0")/launch_wave1_wsl_clean_ranker_scope_safe_h12.sh" \
  "runD2f_clean_ranker_seccorpshortftd_hardening_symbolpriors_core11_h12" \
  --reference-feature-sources sec_filings corporate_actions short_sale ftd \
  --enable-expected-edge-symbol-priors \
  --n-trials 100 \
  --n-splits 4 \
  --nested-outer-splits 3 \
  --nested-inner-splits 3 \
  --holdout-pct 0.20 \
  --label-min-signal-abs-return-bps 14 \
  --label-neutral-buffer-bps 7 \
  --label-edge-cost-buffer-bps 5 \
  --feature-selection-min-ic 0.02 \
  --feature-selection-max-features 72 \
  --feature-selection-min-stability-support 0.65 \
  --execution-turnover-cap 0.45 \
  --execution-max-symbol-entry-share 0.55 \
  --min-confidence-position-scale 0.30 \
  --objective-weight-tail-risk 0.45 \
  --objective-weight-symbol-concentration 0.30 \
  "$@"
