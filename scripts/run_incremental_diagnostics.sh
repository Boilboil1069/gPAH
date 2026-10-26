#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
RUN_ROOT="$(gpah_output_dir "${1:-results/generated/diagnostics}")"
gpah_setup_cache

"$PYTHON_BIN" experiment/run_gpah_paper1_all_experiments.py \
  --output_dir "$RUN_ROOT" \
  --no_timestamped_output \
  --skip_policy \
  --skip_rl \
  --skip_sensitivity \
  --run_rl_topology_generalization \
  --run_buffer_robustness \
  --rl_generalization_algorithms dqn,ppo,sac \
  --rl_generalization_budget_preset full \
  --rl_generalization_train_steps 250000 \
  --rl_generalization_train_seeds 10 \
  --rl_generalization_replications 10 \
  --rl_warmup_steps 500 \
  --rl_eval_steps 3000 \
  --rl_rollout_steps 256 \
  --rl_device "${RL_DEVICE:-cpu}" \
  --buffer_robustness_replications 20 \
  --buffer_robustness_warmup_steps 1000 \
  --buffer_robustness_base_eval_steps 3000 \
  --buffer_robustness_long_eval_steps 30000 \
  --buffer_robustness_base_cap 2000 \
  --buffer_robustness_large_cap 10000 \
  --seed 20260527

