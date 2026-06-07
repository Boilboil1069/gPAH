# Data dictionary

## Main raw-result files

- `paper1_policy_replications.csv`: one row per structured-policy evaluation
  replication and scenario instance.
- `paper1_sensitivity_replications.csv`: paired baseline and one-at-a-time
  parameter-variation evaluations.
- `paper1_rl_policy_replications.csv`: fixed-policy and trained-RL independent
  evaluation replications.
- `paper1_rl_training_history.csv`: learning-curve observations by algorithm,
  instance, and training seed.
- `paper1_ablation_replications.csv`: conditional component-ablation
  replications.
- `paper1_buffer_robustness_replications.csv`: high-load finite-buffer and
  long-horizon diagnostic replications.

## Common columns

- `family`, `case_id`: experiment-family and instance identifiers.
- `policy`, `policy_label`: machine-readable and display policy names.
- `replication`, `seed`: independent evaluation replication and realized seed.
- `mean_total_queue`, `mean_waiting_queue`, `p95_total_queue`: queue metrics.
- `mean_switch_cost`, `switch_rate`: switching-friction metrics.
- `mean_arrivals`, `mean_served`: observed flow metrics.
- `instability`: simulation instability diagnostic.
- `objective`, `objective_raw`: normalized and unnormalized composite objective.

Files ending in `_case_summary.csv`, `_family_summary.csv`, or
`_paired_*.csv` are derived summaries. Replication files are the primary
statistical-analysis inputs.
