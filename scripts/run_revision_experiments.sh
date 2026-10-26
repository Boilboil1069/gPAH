#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"
RL_DEVICE="${RL_DEVICE:-cpu}"
RUN_ROOT="$(gpah_output_dir "${1:-results/generated/revision_experiments}")"
REVISION_SKIP_COMPLETED="${REVISION_SKIP_COMPLETED:-0}"
cleanup_metadata() {
  find "$RUN_ROOT" -type f -name '._*' -delete 2>/dev/null || true
}
run_stage() {
  local stage="$1"
  shift
  local marker="$RUN_ROOT/$stage/run_manifest.json"
  if [[ "$REVISION_SKIP_COMPLETED" == "1" && -s "$marker" ]]; then
    printf 'Skipping completed revision stage: %s (%s)\n' "$stage" "$marker"
    return 0
  fi
  "$@"
}
trap cleanup_metadata EXIT
gpah_setup_cache

run_stage phase_observation env PYTHON_BIN="$PYTHON_BIN" bash scripts/run_phase_observation_robustness.sh \
  "$RUN_ROOT/phase_observation"

run_stage phase_filter env PYTHON_BIN="$PYTHON_BIN" bash scripts/run_phase_filter_validation.sh \
  "$RUN_ROOT/phase_filter"

run_stage infinite_buffer "$PYTHON_BIN" experiment/map_gpah_paper1_infinite_buffer.py \
  --output-dir "$RUN_ROOT/infinite_buffer" \
  --replications 20 \
  --warmup-steps 1000 \
  --eval-steps 100000 \
  --block-steps 1000 \
  --seed 31260527

run_stage runtime env \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  "$PYTHON_BIN" experiment/map_gpah_paper1_runtime_benchmark.py \
  --output-dir "$RUN_ROOT/runtime" \
  --sizes 20x10,50x20,100x40 \
  --replications 20 \
  --warmup-decisions 200 \
  --measured-decisions 2000 \
  --seed 32260527

run_stage rl_hyperparameters "$PYTHON_BIN" experiment/map_gpah_paper1_rl_hyperparameter_validation.py \
  --output-dir "$RUN_ROOT/rl_hyperparameters" \
  --candidate-config configs/revision_rl_hyperparameters.json \
  --algorithms dqn,ppo,sac \
  --search-train-steps 100000 \
  --final-train-steps 250000 \
  --search-train-seeds 3 \
  --final-train-seeds 5 \
  --validation-replications 3 \
  --test-replications 10 \
  --warmup-steps 500 \
  --eval-steps 3000 \
  --rollout-steps 256 \
  --device "$RL_DEVICE" \
  --train-progress \
  --seed 33260527

printf 'Revision experiment results: %s\n' "$RUN_ROOT"
