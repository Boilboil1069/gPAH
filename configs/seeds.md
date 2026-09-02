# Random seeds

The experiment suite uses deterministic seed arithmetic. The master seed for
the final unified run is `20260527`.

| Component | Base seed |
|---|---:|
| Fixed-policy comparison | `20260527` |
| Sensitivity analysis | `21060527` (`master + 700000`) |
| Trained RL comparison | `20260527` in the supplied final command |
| Held-out topology generalization | `26160527` (`master + 6000000`) |
| Buffer robustness | `27260527` (`master + 7000000`) |
| Phase-observation robustness | `30260527` |
| Infinite-buffer high-load diagnostic | `31260527` |
| Wall-clock benchmark | `32260527` |
| RL hyperparameter validation | `33260527` |
| Hidden-phase Bayesian-filter validation | `34260527` |

Within each experiment, problem-generation, case, replication, algorithm, and
training-seed offsets are generated deterministically in the corresponding
script. Every raw replication CSV includes the realized seed.

The final RL output records the complete training budget and hyperparameters in
`data/raw/final_run/rl/paper1_rl_training_budget.json`.
