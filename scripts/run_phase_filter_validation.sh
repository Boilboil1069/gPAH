#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${1:-results/generated/revision_experiments/phase_filter}"
mkdir -p "$ROOT/aux/pycache" "$ROOT/aux/matplotlib" "$ROOT/aux/cache"
export PYTHONPYCACHEPREFIX="$ROOT/aux/pycache"
export MPLCONFIGDIR="$ROOT/aux/matplotlib"
export XDG_CACHE_HOME="$ROOT/aux/cache"
export MPLBACKEND="Agg"

"$PYTHON_BIN" experiment/map_gpah_paper1_phase_filter_validation.py \
  --output-dir "$OUTPUT_DIR" \
  --regimes oracle,bayes_known,bayes_estimated,stationary \
  --replications 30 \
  --warmup-steps 500 \
  --eval-steps 3000 \
  --calibration-steps 10000 \
  --em-iterations 20 \
  --seed 34260527

find "$OUTPUT_DIR" "$ROOT/aux" -type f -name '._*' -delete 2>/dev/null || true
