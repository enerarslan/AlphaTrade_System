#!/usr/bin/env bash
set -euo pipefail

export ALPHATRADE_TRAINING_PROFILE=promotion

bash "$(dirname "$0")/launch_wave1_wsl_clean_ranker_scope_safe_h12.sh" \
  "runP0_promotion_ranker_seccorpshortftd_hardening_symbolpriors_core11_h12" \
  --reference-feature-sources sec_filings corporate_actions short_sale ftd \
  --enable-expected-edge-symbol-priors \
  --enable-ranker-probability-calibration \
  --expected-edge-min-coverage 0.25 \
  --n-trials 120 \
  --n-splits 5 \
  --nested-outer-splits 4 \
  --nested-inner-splits 3 \
  --holdout-pct 0.20 \
  --label-min-signal-abs-return-bps 14 \
  --label-neutral-buffer-bps 8 \
  --label-edge-cost-buffer-bps 5 \
  --feature-selection-min-ic 0.025 \
  --feature-selection-max-features 72 \
  --feature-selection-min-stability-support 0.70 \
  --execution-turnover-cap 0.40 \
  --execution-max-symbol-entry-share 0.50 \
  --min-confidence-position-scale 0.35 \
  --objective-weight-tail-risk 0.50 \
  --objective-weight-symbol-concentration 0.35 \
  --objective-weight-skew 0.10 \
  --objective-skew-penalty-cap 1.50 \
  "$@"
