#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
OUTPUT_DIR="$(gpah_output_dir "${1:-results/replotted}")"
gpah_setup_cache
"$PYTHON_BIN" scripts/regenerate_revision_figures.py --output-dir "$OUTPUT_DIR"
printf 'All 18 manuscript figure PDFs: %s\n' "$OUTPUT_DIR"
