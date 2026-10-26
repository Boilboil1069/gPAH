# gPAH Computational Reproducibility Repository

This repository is the supplementary computational reproducibility package for
the Generalized Phase-Aware Heuristic (gPAH) study. The frozen results and 18
numbered manuscript figure assets correspond to the final revised manuscript.
It includes the simulation and policy implementations, experiment configurations,
seed rules, replication-level outputs, RL training histories, and scripts for
checking, reanalyzing, plotting, and rerunning the experiments.

Download the repository using **Code → Download ZIP**, then extract it before
running the commands below. A Git clone is also supported. No files from a
separate development workspace are required. Record the Git commit identifier
when citing or comparing this package; no archival DOI is claimed here.

## Start here

From the extracted repository root, verify the included files:

```bash
python scripts/verify_repository.py
```

The unified entry point provides the same verification:

```bash
bash scripts/reproduce.sh verify
```

Verification checks the packaged files and checksums. It does not rerun an
experiment or establish statistical reproducibility.

For analysis and simulation, create the pinned environment:

```bash
conda env create -f environment.yml
conda activate gpah-reproduction
```

The reported environment used Python 3.13 on macOS. The main and
held-out-topology RL runs used Apple MPS; the separate revision RL
hyperparameter-validation study used CPU, as recorded in its manifest.
CPU is the portable default for new runs. Set `PYTHON_BIN` to select a Python executable;
set `RL_DEVICE=mps` or `RL_DEVICE=cuda` for full training when that device is
available in PyTorch.

## Choose a task

| Purpose | Command | Default output |
| --- | --- | --- |
| Check packaged files and SHA-256 checksums | `bash scripts/reproduce.sh verify` | Verification report in the terminal |
| Recompute the family-blocked tables from frozen replications | `bash scripts/reproduce.sh reanalyze` | `results/reanalysis/family_blocked/` |
| Regenerate manuscript figures from included numerical results | `bash scripts/reproduce.sh figures` | `results/replotted/` |
| Exercise main and revision entry points with small budgets | `bash scripts/reproduce.sh smoke` | `results/generated/smoke/` |
| Rerun the full main, diagnostic, ablation, and revision protocols | `bash scripts/reproduce.sh full` | `results/generated/full/` |

An optional output directory follows `reanalyze`, `figures`, `smoke`, or `full`.
Relative output paths are interpreted from the repository root; output paths
inside the repository must be under `results/`. Existing generated outputs may
be replaced when rerunning the same command. The script can also be invoked by its path from another working directory. For example:

```bash
PYTHON_BIN=python bash scripts/reproduce.sh reanalyze results/reanalysis/check
```

Reanalysis and plotting use existing results; the smoke command is only a
software check. A full run performs new simulation and training and is
computationally expensive: the main RL protocol alone uses 250,000 environment
steps, 10 training seeds per algorithm and instance, and 10 independent
evaluation replications per trained agent. Numerical agreement across hardware
is statistical, not guaranteed bitwise equality. All generated outputs are
kept separate from the frozen `data/` and `paper_artifacts/` files.

The individual scripts remain available, including `scripts/run_full.sh`,
`scripts/run_incremental_diagnostics.sh`, and
`scripts/run_revision_experiments.sh`. The unified `full` command combines the
paper-facing protocols; the individual main-run script does not run all five
revision experiments. See [revision protocols](docs/REVISION_EXPERIMENTS.md)
for the scope and standalone commands.

## Included results

| Material | Location |
| --- | --- |
| Main structured-policy, sensitivity, RL, and finite-buffer outputs | `data/raw/final_run/` |
| Complete held-out-topology transfer outputs, four CSVs, manifest, and 30 checkpoints | `data/raw/final_run/rl_topology_generalization/` |
| Conditional component-ablation replications and summaries | `data/raw/ablation_r50_conditional/` |
| Controlled phase-observation results | `data/raw/revision_experiments/phase_observation/` |
| Online hidden-phase filtering results | `data/raw/revision_experiments/phase_filter/` |
| Uncapped high-load diagnostic results | `data/raw/revision_experiments/infinite_buffer/` |
| Policy-action runtime results | `data/raw/revision_experiments/runtime/` |
| RL validation, selected hyperparameters, test outputs, and convergence histories | `data/raw/revision_experiments/rl_hyperparameters/` |
| Frozen family-blocked reanalysis tables and input manifest | `data/derived/family_blocked_reanalysis/` |
| Final manuscript figures (`fig1.pdf`–`fig17.pdf`, with `fig13a.pdf` and `fig13b.pdf`) | `paper_artifacts/figs/` |
| Additional analysis tables | `paper_artifacts/tables/` |

This package synchronization adds the completed results and final figure assets;
it is not a new experimental campaign. Scientific values in the frozen CSVs
are retained. Historical machine-specific manifest paths have been made
repository-relative. Old raw policy identifiers are preserved for compatibility;
use the [policy-name mapping](docs/POLICY_NAMING.md) for current display names.

## Parameters, provenance, and interpretation

- [Reproducibility parameters](docs/REPRODUCIBILITY_PARAMETERS.md) and
  [machine-readable configuration](configs/final_configuration.json).
- [Random seeds](configs/seeds.md), [data dictionary](docs/DATA_DICTIONARY.md),
  and [reproducibility notes](docs/REPRODUCIBILITY_NOTES.md).
- [Revision experiments](docs/REVISION_EXPERIMENTS.md) and
  [code and data availability](CODE_AND_DATA_AVAILABILITY.md).

The final main-comparison weights are
`alpha_Q=1.00, alpha_A=1.60, alpha_mu=1.00, alpha_S=0.55, alpha_C=0.90,
alpha_L=0.35`, with forecast horizon `H=6` and assigned-load coefficient `0.25`.
Inputs retain their operational units. Weights are shared across scenario
families and are not tuned per test instance.

The uncapped experiment is a congestion diagnostic, not a stability proof.
Topology transfer is tested on two fixed held-out graphs without fine-tuning;
it does not establish generalization over arbitrary graphs. The RL comparisons
are limited to the released architectures, action spaces, seeds, and budgets.
