# Policy names and historical identifiers

The final revised manuscript uses the display names below. The raw identifiers
in code, configurations, checkpoints, and CSV files are retained to preserve
compatibility and provenance. A display-name correction does not change a
policy, parameter, random seed, or experimental value.

| Raw policy identifier | Current display name | Interpretation |
| --- | --- | --- |
| `gpah_meanmap` | gPAH-StationaryMean | gPAH with the stationary arrival mean replacing the phase-aware forecast |
| `rl_queueing` | Queueing-Surrogate | Non-trained queueing-control diagnostic; not a trained RL baseline |
| `dqn` | DQN-Candidate | DQN over candidate joint actions |
| `ppo` | PPO-Factorized | PPO with factorized pool actions |
| `sac` | Discrete-SAC-Factorized | Discrete SAC with factorized pool actions |
| `affinity` | GB-PANDAS-style | Affinity-style structured comparator |
| `gpah` | gPAH | Main-comparison configuration |
| `gpah_full` | gPAH-Full | Separately frozen conditional-ablation reference |
| `mean_map` | Abl-StationaryMean | Stationary-mean substitution in the conditional-ablation series |

Historical `policy_label` fields and auxiliary raw-result plots may retain
older names such as `RL-Queueing-Surrogate` or `gPAH-MeanMAP`. Use the raw
`policy` identifier when joining data, and apply this mapping for new displays.
Do not interpret the old and current labels as separate policies.

The [parameter tables](REPRODUCIBILITY_PARAMETERS.md) use the current names.
The 18 active manuscript figure assets are in
[`paper_artifacts/figs/`](../paper_artifacts/figs/) and match the final revised
manuscript numbering. Those files are the reference for the final presentation;
raw-result artifacts are retained as historical computational evidence.
