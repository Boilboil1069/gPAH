# gPAH Computational Reproducibility Repository

This repository contains the code, fixed configurations, random-seed rules,
raw numerical results, and paper artifacts for the computational study of the
Generalized Phase-Aware Heuristic (gPAH).

The repository is self-contained for the Paper 1 experiments. No files from
the parent development workspace are required.

## Repository layout

- `experiment/`: simulation model, structured policies, RL baselines,
  sensitivity analysis, robustness diagnostics, and plotting code.
- `configs/`: frozen paper-facing configuration and seed documentation.
- `data/raw/final_run/`: raw and summarized outputs from the final main run.
- `data/raw/ablation_r50_conditional/`: conditional-ablation raw results.
- `paper_artifacts/`: figures and tables used by the manuscript.
- `scripts/`: one-command smoke test, full run, and figure regeneration.
- `docs/`: data dictionary, reproducibility notes, and availability statement.

## Environment

The reported run used Python 3.13 on macOS with Apple MPS for RL training.
Create the environment with:

```bash
conda env create -f environment.yml
conda activate gpah-reproduction
```

CPU execution is supported and is the most portable option. To use Apple MPS,
change `--rl_device cpu` to `--rl_device mps` in `scripts/run_full.sh`.

## Quick verification

The smoke test checks all experiment entry points with a deliberately small
budget:

```bash
bash scripts/run_smoke.sh
```

This is a software check only and must not be used to reproduce paper results.

## Full reproduction

Run the complete paper-facing experiment suite with:

```bash
bash scripts/run_full.sh
```

The full RL configuration trains DQN, PPO, and discrete SAC for 250,000
environment steps using 10 independent training seeds and 10 evaluation
replications. The complete run is computationally expensive.

The command writes to `results/generated/full/`. Fixed-policy results generated
earlier in the same run are reused during the RL comparison. The same entry
also runs the 50-replication conditional component-ablation experiment.

## Regenerate figures from included raw data

```bash
bash scripts/reproduce_figures.sh
```

Generated figures are written under `results/replotted/`; the included raw data
are not modified.

## Frozen parameters and seeds

The final gPAH weights are:

```text
alpha_Q=1.00, alpha_A=1.60, alpha_mu=1.00,
alpha_S=0.55, alpha_C=0.90, alpha_L=0.35
```

The forecast horizon is `H=6`, and the within-step assigned-load coefficient is
`0.25`. Inputs retain their operational units; no instance-wise min-max,
z-score, or test-result-based normalization is applied.

The master seed and all deterministic seed offsets are documented in
`configs/seeds.md`. Hyperparameters are shared across scenario families and
are not tuned per test instance.

## Included-result status

`data/raw/final_run/` contains the final policy, sensitivity, and trained-RL
comparison outputs used for the paper-facing analysis, including replication
CSV files and RL learning histories.

The held-out-topology folder contains a partial checkpoint snapshot rather than
a complete result table, and the buffer-robustness folder currently contains no
completed output files. Their executable experiment code is included, but these
two diagnostics must be run separately before claiming completed results.

## Integrity

Run:

```bash
shasum -a 256 -c CHECKSUMS.sha256
```

to verify the committed code, configuration, raw data, and paper artifacts.

## Code and data availability

See `CODE_AND_DATA_AVAILABILITY.md` for manuscript-ready wording. Replace the
repository URL placeholder after uploading this repository to an anonymous
review service or public archival repository.
