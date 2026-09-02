#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${1:-results/generated/revision_experiments_smoke}"
cleanup_metadata() {
  find "$RUN_ROOT" "$ROOT/aux" -type f -name '._*' -delete 2>/dev/null || true
}
trap cleanup_metadata EXIT
mkdir -p "$ROOT/aux/pycache" "$ROOT/aux/matplotlib" "$ROOT/aux/cache"
export PYTHONPYCACHEPREFIX="$ROOT/aux/pycache"
export MPLCONFIGDIR="$ROOT/aux/matplotlib"
export XDG_CACHE_HOME="$ROOT/aux/cache"
export MPLBACKEND="Agg"

"$PYTHON_BIN" experiment/map_gpah_paper1_phase_observation_robustness.py \
  --output-dir "$RUN_ROOT/phase_observation" \
  --case-ids predictive_burst \
  --regimes oracle,posterior_0.25,delay_1,stationary \
  --replications 1 \
  --warmup-steps 5 \
  --eval-steps 20

"$PYTHON_BIN" experiment/map_gpah_paper1_phase_filter_validation.py \
  --output-dir "$RUN_ROOT/phase_filter" \
  --case-ids predictive_burst \
  --regimes oracle,bayes_known,bayes_estimated,stationary \
  --replications 1 \
  --warmup-steps 5 \
  --eval-steps 20 \
  --calibration-steps 200 \
  --em-iterations 2

"$PYTHON_BIN" experiment/map_gpah_paper1_infinite_buffer.py \
  --output-dir "$RUN_ROOT/infinite_buffer" \
  --policies gpah \
  --loads 0.90 \
  --replications 1 \
  --warmup-steps 5 \
  --eval-steps 30 \
  --block-steps 10

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
"$PYTHON_BIN" experiment/map_gpah_paper1_runtime_benchmark.py \
  --output-dir "$RUN_ROOT/runtime" \
  --sizes 100x40 \
  --policies gpah \
  --replications 1 \
  --warmup-decisions 5 \
  --measured-decisions 20

"$PYTHON_BIN" experiment/map_gpah_paper1_rl_hyperparameter_validation.py \
  --output-dir "$RUN_ROOT/rl_hyperparameters" \
  --candidate-config configs/revision_rl_hyperparameters.json \
  --algorithms dqn,ppo,sac \
  --validation-cases A1_topology/chain \
  --test-cases A1_topology/bridge_bottleneck \
  --max-candidates 1 \
  --search-train-steps 32 \
  --final-train-steps 32 \
  --search-train-seeds 1 \
  --final-train-seeds 1 \
  --validation-replications 1 \
  --test-replications 1 \
  --warmup-steps 5 \
  --eval-steps 20 \
  --rollout-steps 16 \
  --device cpu

printf 'Revision smoke-test results: %s\n' "$RUN_ROOT"
