#!/usr/bin/env bash
set -euo pipefail

bash "$(dirname "$0")/launch_wave1_wsl_clean_ranker_scope_safe_h12.sh" \
  "runD0_clean_ranker_priceonly_core11_h12" \
  --disable-reference-features \
  "$@"
