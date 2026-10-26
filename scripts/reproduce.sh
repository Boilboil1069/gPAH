#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
MODE="${1:-verify}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 1 ]]; then
  printf 'Usage: bash scripts/reproduce.sh {verify|smoke|full|figures|reanalyze|revision} [output-directory]\n' >&2
  exit 2
fi
case "$MODE" in
  verify)
    if [[ $# -ne 0 ]]; then printf 'verify takes no output-directory argument\n' >&2; exit 2; fi
    "$PYTHON_BIN" "$ROOT/scripts/verify_repository.py"
    ;;
  smoke)
    RUN_ROOT="$(gpah_output_dir "${1:-results/generated/smoke}")"
    bash "$ROOT/scripts/run_smoke.sh" "$RUN_ROOT/main"
    bash "$ROOT/scripts/run_revision_experiments_smoke.sh" "$RUN_ROOT/revision_experiments"
    ;;
  full)
    RUN_ROOT="$(gpah_output_dir "${1:-results/generated/full}")"
    printf 'Running full paper budgets; RL training and uncapped diagnostics are expensive.\n'
    bash "$ROOT/scripts/run_full.sh" "$RUN_ROOT"
    bash "$ROOT/scripts/run_revision_experiments.sh" "$RUN_ROOT/revision_experiments"
    gpah_setup_cache
    "$PYTHON_BIN" experiment/reanalyze_gpah_paper1_family_blocked.py \
      --input "$RUN_ROOT/policy/paper1_policy_replications.csv" \
      --output-dir "$RUN_ROOT/family_blocked_reanalysis"
    ;;
  figures)
    bash "$ROOT/scripts/reproduce_figures.sh" "${1:-results/replotted}"
    ;;
  reanalyze)
    OUTPUT_DIR="$(gpah_output_dir "${1:-results/reanalysis/family_blocked}")"
    "$PYTHON_BIN" experiment/reanalyze_gpah_paper1_family_blocked.py \
      --input "$ROOT/data/raw/final_run/policy/paper1_policy_replications.csv" \
      --output-dir "$OUTPUT_DIR"
    ;;
  revision)
    bash "$ROOT/scripts/run_revision_experiments.sh" "${1:-results/generated/revision_experiments}"
    ;;
  -h|--help|help)
    cat <<'HELP'
Usage: bash scripts/reproduce.sh MODE [output-directory]
  verify     Check the distributed files and SHA-256 manifest (default; no simulations).
  reanalyze  Recompute family-blocked tables from included results (standard library).
  figures    Regenerate all 18 manuscript figure PDFs from included data.
  smoke      Run small main and revision software checks, not paper evidence.
  full       Rerun main, topology, buffer, ablation and revision paper budgets.
  revision   Rerun the five revision studies only.
Set PYTHON_BIN to select Python and RL_DEVICE to select cpu (default) or mps.
Relative output paths are relative to the repository root. Repository outputs
must be under results/. Existing generated outputs may be replaced on reruns.
HELP
    ;;
  *) printf 'Unknown mode: %s (use --help)\n' "$MODE" >&2; exit 2 ;;
esac
