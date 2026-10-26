# Revision experiments and reanalysis

The five completed revision studies are included under
`data/raw/revision_experiments/`. These frozen data support the final revised
manuscript. Rerun commands create new outputs under `results/`; they do not
replace the packaged evidence.

## Completed family-blocked reanalysis (C8.4)

This deterministic reanalysis uses the completed 42-case, 50-replication
policy dataset; it runs no simulation. It verifies paired replication IDs,
averages replications within each case, averages case-level relative effects
within each family, and finally gives each of the eight families equal weight.
Run it from the repository root with:

```bash
bash scripts/reproduce.sh reanalyze
```

Frozen outputs and an input-checksum manifest are under
`data/derived/family_blocked_reanalysis/`. The command writes a new copy to
`results/reanalysis/family_blocked/` so that the reference tables remain unchanged.

The five supplemental experiments requested during review are implemented as
independent entry points with explicit seed settings. Generated results are
written under `results/generated/`, which is excluded from version control.

## 1. Phase-observation robustness (B3.4)

Frozen results: `data/raw/revision_experiments/phase_observation/`.

`experiment/map_gpah_paper1_phase_observation_robustness.py` changes only the
phase belief used by the gPAH forecast. It compares oracle phase information
with posterior contamination, delayed phase information, and the stationary
mean. Initial states, arrival streams, and phase streams are identical across
observation regimes and isolated from policy-dependent service draws. Run only
this experiment with:

```bash
PYTHON_BIN=python bash scripts/run_phase_observation_robustness.sh
```

## 2. Online hidden-phase identification (B3.4)

Frozen results: `data/raw/revision_experiments/phase_filter/`.

`experiment/map_gpah_paper1_phase_filter_validation.py` performs actual online
phase inference from per-queue arrival counts. `bayes_known` uses known MMPP
parameters as a filter benchmark, while `bayes_estimated` fits a Poisson HMM by expectation
maximization on an independent arrival-only calibration trace. Neither filter
reads the simulator's true phase when making a decision; true phases are used
only by evaluation diagnostics to compute MAP accuracy, Brier score, and
negative log likelihood, and those diagnostics do not feed back into the
policy. Arrival and phase random-number streams are paired across regimes
and isolated from policy-dependent service draws.

Its standalone paper-facing entry point is:

```bash
PYTHON_BIN=python bash scripts/run_phase_filter_validation.sh
```

Direct invocation with custom budgets is also supported:

```bash
python experiment/map_gpah_paper1_phase_filter_validation.py \
  --output-dir results/generated/revision_experiments/phase_filter \
  --replications 30 --calibration-steps 10000 --em-iterations 20
```

## 3. Uncapped high-load diagnostic (B5.5)

Frozen results: `data/raw/revision_experiments/infinite_buffer/`.

`experiment/map_gpah_paper1_infinite_buffer.py` disables queue clipping and
reports long-horizon trajectories, second-half queue slopes, terminal drift,
and arrival-minus-service rates. It is a congestion diagnostic, not a proof of
stability. Its trajectory figure uses a logarithmic queue axis. The protocol
uses 20 replications, 1,000 warm-up steps, and 100,000 evaluation steps. To regenerate
figures from the frozen results without simulation, run:

```bash
bash scripts/reproduce.sh figures
```

## 4. Wall-clock benchmark (C9.3)

Frozen results: `data/raw/revision_experiments/runtime/`.

`experiment/map_gpah_paper1_runtime_benchmark.py` times the `policy.action`
region separately from the simulator step. It records platform metadata,
timer overhead, fixed thread settings, and includes `d=100, K=40`.

## 5. RL validation and convergence (B4.5/C6.5)

Frozen results: `data/raw/revision_experiments/rl_hyperparameters/`.

`experiment/map_gpah_paper1_rl_hyperparameter_validation.py` selects each
algorithm's candidate only on declared validation cases. The winner is then
retrained from fresh seeds and evaluated on disjoint test cases. Candidate
settings are versioned in `configs/revision_rl_hyperparameters.json`, and the
selected values are preserved in the result folder. The search uses 100,000
training steps and three training seeds; final retraining uses 250,000 steps
and five fresh training seeds. Validation uses three evaluation replications;
testing uses 10, with 500 warm-up and 3,000 evaluation steps. These are the
revision-study budgets, separate from the main RL comparison.

## Commands

Verify all packaged evidence without running experiments:

```bash
bash scripts/reproduce.sh verify
```

Run a lightweight software check:

```bash
bash scripts/run_revision_experiments_smoke.sh
```

Run the five revision protocols with their paper-facing budgets:

```bash
bash scripts/run_revision_experiments.sh
```

Resume an interrupted paper-facing run without repeating stages that already
have a nonempty final `run_manifest.json`:

```bash
REVISION_SKIP_COMPLETED=1 bash scripts/run_revision_experiments.sh
```

An alternative output root can be passed as the first positional argument.
Set `PYTHON_BIN` to choose a Python executable and `RL_DEVICE=mps` or
`RL_DEVICE=cuda` when the corresponding PyTorch device is available. The
paper-facing runner enables per-training-run progress for the fifth, RL
validation stage and prints it directly to the active terminal.

To rerun the main experiments, held-out-topology and finite-buffer diagnostics,
conditional ablation, all five revision studies, and family-blocked reanalysis
through one entry point, use:

```bash
bash scripts/reproduce.sh full
```

This full command performs new simulation and training. A successful smoke run
or a checksum check is not evidence that the full protocols have been rerun.
