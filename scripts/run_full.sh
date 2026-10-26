#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
RUN_ROOT="$(gpah_output_dir "${1:-results/generated/full}")"
gpah_setup_cache

"$PYTHON_BIN" experiment/run_gpah_paper1_all_experiments.py \
  --output_dir "$RUN_ROOT" \
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
  --rl_device "${RL_DEVICE:-cpu}" \
  --rl_jobs 2 \
  --rl_seed_offset 0 \
  --rl_reuse_fixed_policy_dir "$RUN_ROOT/policy" \
  --run_rl_topology_generalization \
  --rl_generalization_algorithms dqn,ppo,sac \
  --rl_generalization_budget_preset full \
  --rl_generalization_train_steps 250000 \
  --rl_generalization_train_seeds 10 \
  --rl_generalization_replications 10 \
  --run_buffer_robustness \
  --buffer_robustness_replications 20 \
  --buffer_robustness_warmup_steps 1000 \
  --buffer_robustness_base_eval_steps 3000 \
  --buffer_robustness_long_eval_steps 30000 \
  --buffer_robustness_base_cap 2000 \
  --buffer_robustness_large_cap 10000 \
  --seed 20260527

"$PYTHON_BIN" experiment/map_gpah_paper1_ablation.py \
  --output_dir "$RUN_ROOT/ablation_r50_conditional" \
  --no_timestamped_output \
  --families B1_phase,B2_service,B3_scarcity,B4_switch,B5_mixed \
  --policies gpah_full,no_phase,no_service,no_scarcity,no_switch,no_queue,mean_map \
  --replications 50 \
  --warmup_steps 500 \
  --eval_steps 3000 \
  --seed 20260508
