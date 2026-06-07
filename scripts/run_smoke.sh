#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python experiment/run_gpah_paper1_all_experiments.py \
  --output_dir results/generated/smoke \
  --no_timestamped_output \
  --policy_families A1_topology,A7_load \
  --policy_replications 2 \
  --policy_warmup_steps 20 \
  --policy_eval_steps 50 \
  --sensitivity_families A7_load \
  --sensitivity_parameters alpha_Q,alpha_A \
  --sensitivity_multipliers 0,1 \
  --sensitivity_replications 2 \
  --sensitivity_warmup_steps 20 \
  --sensitivity_eval_steps 50 \
  --rl_families A1_topology \
  --rl_case_ids chain \
  --rl_algorithms dqn,ppo,sac \
  --rl_budget_preset smoke \
  --rl_train_steps 128 \
  --rl_train_seeds 1 \
  --rl_replications 1 \
  --rl_warmup_steps 20 \
  --rl_eval_steps 50 \
  --rl_rollout_steps 32 \
  --rl_device cpu \
  --rl_jobs 1 \
  --rl_seed_offset 0 \
  --rl_reuse_fixed_policy_dir results/generated/smoke/policy \
  --seed 20260527

