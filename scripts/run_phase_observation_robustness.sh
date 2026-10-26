#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="$(gpah_output_dir "${1:-results/generated/revision_experiments/phase_observation}")"
gpah_setup_cache

"$PYTHON_BIN" experiment/map_gpah_paper1_phase_observation_robustness.py \
  --output-dir "$OUTPUT_DIR" \
  --replications 30 \
  --warmup-steps 500 \
  --eval-steps 3000 \
  --seed 30260527

find "$OUTPUT_DIR" -type f -name '._*' -delete 2>/dev/null || true
