#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${1:-results/generated/revision_experiments/phase_observation}"
mkdir -p "$ROOT/aux/pycache" "$ROOT/aux/matplotlib" "$ROOT/aux/cache"
export PYTHONPYCACHEPREFIX="$ROOT/aux/pycache"
export MPLCONFIGDIR="$ROOT/aux/matplotlib"
export XDG_CACHE_HOME="$ROOT/aux/cache"
export MPLBACKEND="Agg"

"$PYTHON_BIN" experiment/map_gpah_paper1_phase_observation_robustness.py \
  --output-dir "$OUTPUT_DIR" \
  --replications 30 \
  --warmup-steps 500 \
  --eval-steps 3000 \
  --seed 30260527

find "$OUTPUT_DIR" "$ROOT/aux" -type f -name '._*' -delete 2>/dev/null || true
