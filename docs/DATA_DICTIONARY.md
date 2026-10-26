# Data dictionary

All paths below are relative to the repository root. The frozen result
locations are listed in the [README](../README.md).

## Main raw-result files

| Filename | Unit or purpose |
| --- | --- |
| `paper1_policy_replications.csv` | Structured-policy evaluation replication and scenario instance |
| `paper1_sensitivity_replications.csv` | Paired baseline and one-at-a-time parameter-variation evaluations |
| `paper1_rl_policy_replications.csv` | Fixed-policy and trained-RL independent evaluation replications |
| `paper1_rl_training_history.csv` | Learning-curve observations by algorithm, instance, and training seed |
| `paper1_ablation_replications.csv` | Conditional component-ablation replications |
| `paper1_buffer_robustness_replications.csv` | High-load finite-buffer and long-horizon diagnostic replications |
| `paper1_rl_unseen_topology_replications.csv` | No-fine-tuning evaluation on held-out topology instances |

The topology-transfer folder also includes summary/training-history CSVs,
`run_manifest.json`, and 30 `.pt` checkpoints. The checkpoint files belong to
that shared-policy experiment, not the main per-instance RL experiment.

## Revision result files

These files are under `data/raw/revision_experiments/` in the corresponding
experiment subfolder.

| Filename or prefix | Unit or purpose |
| --- | --- |
| `paper1_phase_observation_replications.csv` | Paired evaluation under controlled phase-belief and delay regimes |
| `paper1_phase_filter_replications.csv` | Paired oracle, stationary, and online-filter evaluations |
| `paper1_phase_filter_calibration_summary.csv` | Hidden-phase parameter-estimation calibration diagnostics |
| `paper1_infinite_buffer_replications.csv` | Uncapped high-load evaluation replication |
| `paper1_infinite_buffer_trace_blocks.csv` | Within-run block summaries for queue trajectories |
| `paper1_infinite_buffer_trace_summary.csv` | Aggregated trajectory summaries across replications |
| `paper1_runtime_replications.csv` | Policy-action timing measurements |
| `paper1_rl_hyperparameter_validation_replications.csv` | Candidate comparison on declared validation cases |
| `paper1_rl_hyperparameter_test_replications.csv` | Fresh-seed evaluation on disjoint test cases |
| `paper1_rl_hyperparameter_training_history.csv` | Candidate and selected-model training histories |
| `paper1_rl_hyperparameter_convergence_summary.csv` | Training-progress diagnostics |
| `selected_hyperparameters.json` | Configurations selected by validation, without test-set tuning |

Each study supplies its own run manifest and relevant summaries. File schemas
vary by experiment; column headers and the producing script define each file.

## Common columns

- `family`, `case_id`: experiment-family and instance identifiers.
- `policy`, `policy_label`: stable machine identifier and historical display
  label; see [POLICY_NAMING.md](POLICY_NAMING.md) for current display names.
- `replication`, `seed`: evaluation replication and realized seed. Pairing is
  defined within each experiment; identical integers across different studies
  do not imply shared random draws.
- `mean_total_queue`, `mean_waiting_queue`, `p95_total_queue`: queue metrics.
- `mean_switch_cost`, `switch_rate`: switching-friction metrics.
- `mean_arrivals`, `mean_served`: observed flow metrics.
- `instability`: simulation diagnostic, not a mathematical stability result.
- `objective`, `objective_raw`: standardized and unstandardized composite
  objective fields where provided; consult the producing script for the
  experiment's definition and normalization.

Files ending in `_case_summary.csv`, `_family_summary.csv`, or
`_paired_*.csv` are derived summaries. Replication files are the primary
statistical-analysis inputs. The family-blocked reanalysis additionally gives
equal weight to cases within each family and then to the eight families;
see [REVISION_EXPERIMENTS.md](REVISION_EXPERIMENTS.md).
