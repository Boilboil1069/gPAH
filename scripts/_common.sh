#!/usr/bin/env bash
# Shared setup for the supported repository entry points.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHON_BIN
export MPLBACKEND="Agg"
export PYTHONDONTWRITEBYTECODE=1

gpah_output_dir() {
  "$PYTHON_BIN" "$ROOT/scripts/reproduction_paths.py" "$1"
}

gpah_setup_cache() {
  export MPLCONFIGDIR="$ROOT/results/.cache/matplotlib"
  export XDG_CACHE_HOME="$ROOT/results/.cache"
  mkdir -p "$MPLCONFIGDIR"
}
