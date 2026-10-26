#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
RUN_ROOT="$(gpah_output_dir "${1:-results/generated/smoke}")"
gpah_setup_cache

"$PYTHON_BIN" experiment/run_gpah_paper1_all_experiments.py \
  --output_dir "$RUN_ROOT" \
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
  --rl_device "${RL_DEVICE:-cpu}" \
  --rl_jobs 1 \
  --rl_seed_offset 0 \
  --rl_reuse_fixed_policy_dir "$RUN_ROOT/policy" \
  --run_rl_topology_generalization \
  --rl_generalization_train_topologies chain \
  --rl_generalization_test_topologies bridge_bottleneck \
  --rl_generalization_algorithms dqn,ppo,sac \
  --rl_generalization_budget_preset smoke \
  --rl_generalization_train_steps 32 \
  --rl_generalization_train_seeds 1 \
  --rl_generalization_replications 1 \
  --run_buffer_robustness \
  --buffer_robustness_policies gpah \
  --buffer_robustness_loads 0.90 \
  --buffer_robustness_replications 1 \
  --buffer_robustness_warmup_steps 5 \
  --buffer_robustness_base_eval_steps 20 \
  --buffer_robustness_long_eval_steps 40 \
  --seed 20260527


"$PYTHON_BIN" experiment/map_gpah_paper1_ablation.py \
  --output_dir "$RUN_ROOT/ablation_r50_conditional" \
  --no_timestamped_output \
  --families B1_phase \
  --policies gpah_full,no_phase,no_service,no_scarcity,no_switch,no_queue,mean_map \
  --replications 1 \
  --warmup_steps 5 \
  --eval_steps 20 \
  --seed 20260508
