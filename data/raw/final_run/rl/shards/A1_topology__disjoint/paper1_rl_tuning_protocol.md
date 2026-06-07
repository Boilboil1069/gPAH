# RL Training Budget and Tuning Protocol

## Budget

- Budget preset: `full`.
- Algorithms: `dqn,ppo,sac`.
- Training steps per algorithm, instance, and training seed: `250000`.
- Independent training seeds per algorithm and instance: `10`.
- Evaluation replications per trained policy: `10`.
- Warmup/evaluation steps per replication: `500` / `3000`.
- Learning-curve smoothing window: `500` environment steps.

## Hyperparameters

- Discount factor gamma: `0.995`.
- Learning rate: `0.0003`.
- Batch size: `128`.
- Hidden units per MLP layer: `128`.
- Replay buffer size: `50000`.
- Off-policy update_after/update_every: `500` / `4`.
- Target-network Polyak coefficient: `0.01`.
- PPO epochs/clip/entropy/value coefficients: `4` / `0.2` / `0.02` / `0.5`.
- SAC entropy temperature alpha: `0.08`.

## Tuning Protocol

- The default `review` budget is the primary paper-facing budget; `smoke` is only for code checks.
- Hyperparameters are shared across all case families to avoid per-instance test-set tuning.
- Candidate changes should be selected on aggregate training-curve stability and validation-style evaluation, then frozen before final policy-comparison runs.
- Report `paper1_rl_training_history.csv` and `fig_rl_training_curves.pdf` with the final tables so reviewers can inspect convergence behavior.
- When compute permits, rerun with `--budget-preset full` and report it as a robustness appendix check.
