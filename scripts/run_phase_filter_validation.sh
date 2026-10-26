#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="$(gpah_output_dir "${1:-results/generated/revision_experiments/phase_filter}")"
gpah_setup_cache

"$PYTHON_BIN" experiment/map_gpah_paper1_phase_filter_validation.py \
  --output-dir "$OUTPUT_DIR" \
  --regimes oracle,bayes_known,bayes_estimated,stationary \
  --replications 30 \
  --warmup-steps 500 \
  --eval-steps 3000 \
  --calibration-steps 10000 \
  --em-iterations 20 \
  --seed 34260527

find "$OUTPUT_DIR" -type f -name '._*' -delete 2>/dev/null || true
