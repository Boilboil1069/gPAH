# Reproducibility notes

## Parameter-selection protocol

The six gPAH weights are fixed across the final A1--A8 comparisons, RL
comparisons, and sensitivity analysis. No family-specific or instance-specific
weights are used. Sensitivity results are reported as a post-freeze robustness
diagnostic and are not substituted back into the main comparison.

RL hyperparameters are shared across all scenario families. The `full` budget
uses 250,000 training steps and 10 independent training seeds per algorithm and
instance. Evaluation uses common independent replications for trained agents
and fixed structural policies.

## Hardware and nondeterminism

NumPy and PyTorch seeds are explicitly set. Exact floating-point equality is
not guaranteed across CPU, CUDA, and Apple MPS backends. Reproduction should
focus on statistical agreement of paired summaries and qualitative conclusions,
not bitwise equality of RL trajectories.

## Result provenance

The included main result snapshot was produced in the parent research workspace
and copied into this standalone repository. Local absolute paths contained in
historical run manifests are provenance records only; all commands documented
in this repository use relative paths.

The buffer-robustness snapshot is complete and includes replication-level data,
summaries, and diagnostics for \(\rho\in\{0.90,0.95\}\) under three buffer and
evaluation-horizon settings. The held-out-topology result snapshot remains
incomplete; its script is included to make the missing incremental experiment
explicit and executable.
