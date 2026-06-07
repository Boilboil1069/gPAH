#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python experiment/run_gpah_paper1_all_experiments.py \
  --output_dir results/generated/full \
  --no_timestamped_output \
  --policy_families A1_topology,A2_map_burst,A3_density,A4_service,A5_switch,A6_scale,A7_load,A8_large_scale \
  --policy_replications 50 \
  --policy_warmup_steps 500 \
  --policy_eval_steps 3000 \
  --sensitivity_families A1_topology,A2_map_burst,A3_density,A4_service,A5_switch,A6_scale,A7_load,A8_large_scale \
  --sensitivity_parameters alpha_Q,alpha_A,alpha_mu,alpha_S,alpha_C,alpha_L \
  --sensitivity_multipliers 0,0.5,1,1.5,2 \
  --sensitivity_replications 30 \
  --sensitivity_warmup_steps 500 \
  --sensitivity_eval_steps 3000 \
  --rl_algorithms dqn,ppo,sac \
  --rl_families A1_topology,A2_map_burst,A3_density,A4_service,A5_switch,A6_scale,A7_load,A8_large_scale \
  --rl_budget_preset full \
  --rl_train_steps 250000 \
  --rl_train_seeds 10 \
  --rl_replications 10 \
  --rl_warmup_steps 500 \
  --rl_eval_steps 3000 \
  --rl_rollout_steps 256 \
  --rl_device cpu \
  --rl_jobs 2 \
  --rl_seed_offset 0 \
  --rl_reuse_fixed_policy_dir results/generated/full/policy \
  --seed 20260527

