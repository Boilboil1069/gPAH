# Code and data availability

The supplementary computational reproducibility package is provided in this
repository. It includes the simulation and policy implementations, frozen
experiment configurations, random-seed rules, RL hyperparameters, raw
replication-level results, training histories, summary tables, plotting code,
and the final manuscript figure assets.

The release includes the completed held-out-topology transfer diagnostic and
all five revision studies: controlled phase observation, hidden-phase filtering,
uncapped high-load behavior, policy-action runtime, and validation-selected RL
hyperparameters and convergence. Dataset locations and supported commands are
listed in the [README](README.md).

Download and extract the complete repository archive to use it as the
supplementary package. Record the Git commit identifier alongside the download
for version identification. SHA-256 file checksums are supplied in
[CHECKSUMS.sha256](CHECKSUMS.sha256); run
`bash scripts/reproduce.sh verify` to check them and required package contents.
No persistent archive DOI is asserted by this repository.

The [pinned environment](environment.yml),
[parameter tables](docs/REPRODUCIBILITY_PARAMETERS.md),
[seed documentation](configs/seeds.md), and
[revision protocols](docs/REVISION_EXPERIMENTS.md) describe how to use the data
and rerun the experiments. Verification, deterministic reanalysis, figure
regeneration, small-budget smoke checks, and full experimental reruns are
separate operations. Generated outputs do not replace the frozen evidence.
