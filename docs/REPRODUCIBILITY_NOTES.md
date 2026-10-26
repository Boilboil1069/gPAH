# Reproducibility notes

## Frozen package and generated outputs

The released data and manuscript figure assets correspond to the final revised
manuscript. This repository update assembles the completed experiments and
final presentation assets; it does not represent a new experiment run.
Machine-specific paths in historical manifests have been replaced with
repository-relative paths. Numerical CSV data and raw policy identifiers remain
unchanged. See [POLICY_NAMING.md](POLICY_NAMING.md) for current display labels.

Treat `data/` and `paper_artifacts/` as the frozen reference. The unified
`scripts/reproduce.sh` entry point writes reanalysis, regenerated figures,
smoke runs, and full reruns into separate directories under `results/`.
Do not use the frozen directories as output destinations for custom runs.

## Levels of verification

- `verify` checks file integrity and package completeness; it runs no simulation.
- `reanalyze` recomputes family-blocked summaries from included replications;
  it runs no simulation or training.
- `figures` regenerates figures from the included result tables and histories;
  the numbered PDFs in `paper_artifacts/figs/` remain the final manuscript assets.
- `smoke` tests execution with deliberately small budgets; its outputs cannot
  substantiate the paper's reported results.
- `full` performs new simulations and training with the documented protocols.

A passing integrity check or smoke run does not demonstrate that a full
experimental campaign has been rerun. Compare regenerated numerical summaries
with the frozen data using the pairing and aggregation rules in the scripts.

## Parameter-selection protocol

The six gPAH weights are fixed across the final A1–A8 comparisons, RL
comparisons, and sensitivity analysis. No family-specific or instance-specific
weights are used. Sensitivity results are a post-freeze robustness diagnostic
and are not substituted back into the main comparison. The conditional-ablation
reference has its own separately frozen mechanism-identification weights,
as documented in [REPRODUCIBILITY_PARAMETERS.md](REPRODUCIBILITY_PARAMETERS.md).

The main RL protocol uses shared hyperparameters, 250,000 training steps, and
10 independent training seeds per algorithm and instance. Evaluation uses
independent replications for trained agents and fixed policies. The separate
revision RL study selects candidates on declared validation cases, retrains
with fresh seeds, and evaluates on disjoint test cases. Its budgets are listed
in [REVISION_EXPERIMENTS.md](REVISION_EXPERIMENTS.md).

## Hardware and nondeterminism

NumPy and PyTorch seeds are explicitly set. Exact floating-point equality is
not guaranteed across CPU, CUDA, and Apple MPS backends. Assess reruns through
paired summaries, statistical agreement, and the reported experimental scope.
Wall-clock timing is hardware-dependent; the runtime manifest records the
measurement environment and protocol.

## Evidence boundaries

The finite-buffer snapshot contains replication-level data, summaries, and
congestion diagnostics for loads 0.90 and 0.95 under three buffer and evaluation
horizon settings. The uncapped high-load study records long-horizon drift; it
does not prove queue stability.

The held-out-topology snapshot contains all four CSV outputs, the run manifest,
and 30 shared-policy checkpoints (three algorithms and 10 training seeds).
Evaluation uses two fixed held-out graph instances with no fine-tuning.
This limited transfer test is not a many-random-graph generalization study.
RL comparisons describe the released action spaces, architectures, seeds, and
budgets and do not establish universal superiority over RL methods.
