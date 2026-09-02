# Reproducibility parameters

This document is the version-controlled home of the reproducibility-parameter
tables for the gPAH computational study. These tables were moved out of the
manuscript appendix so that parameter corrections, executable configurations,
and experiment entry points remain synchronized in one repository.

The machine-readable primary configuration is
[`configs/final_configuration.json`](../configs/final_configuration.json).
Seed offsets are documented in [`configs/seeds.md`](../configs/seeds.md), and
the reviewer-requested incremental experiments are described in
[`REVISION_EXPERIMENTS.md`](REVISION_EXPERIMENTS.md).

## Construction of MMPP arrival profiles

For each queue $i$, arrivals are generated as a finite-state
Markov-modulated Poisson process (MMPP). The exogenous phase continuous-time
Markov chain has generator $Q_i^A$, conditional Poisson arrivals have
state-dependent rates $\lambda_i(a)$, and arrival events leave the phase
unchanged. The equivalent Markovian arrival process representation is

$$
D_{1,i}=\operatorname{diag}(\lambda_i),
\qquad
D_{0,i}=Q_i^A-D_{1,i}.
$$

The profiles below first define relative phase rates and phase-transition
matrices. A queue-specific multiplier $u_i$ creates mild cross-queue
heterogeneity. All arrival rates in an instance are then multiplied by a
common factor so that the stationary mean arrival vector matches the target
load $\rho$ under the conservative compatible-capacity proxy used by the
simulator. The gPAH phase term uses the same $Q_i^A$ and $\lambda_i(a)$ to
compute short-horizon forecast pressure.

| Profile | States | Relative rates before load scaling | Phase-transition construction and purpose |
|---|---:|---|---|
| `low_burst` | 2 | $[0.65,1.45]u_i$, $u_i\sim U[0.85,1.20]$ | Two-state MMPP with transition rates $q_{01}=0.18$, $q_{10}=0.75q_{01}$; weak burst contrast. |
| `medium_burst` | 2 | $[0.25,2.40]u_i$, $u_i\sim U[0.85,1.20]$ | Same two-state structure with $q_{01}=0.08$; default persistent burst profile used outside A2 unless stated otherwise. |
| `high_burst` | 2 | $[0.06,4.40]u_i$, $u_i\sim U[0.85,1.20]$ | Same two-state structure with $q_{01}=0.025$; rare but long high-arrival periods. |
| `low_contrast` | 2 | $[0.85,1.25]u_i$, $u_i\sim U[0.85,1.20]$ | Same two-state structure with $q_{01}=0.16$; phase is intentionally weakly informative. |
| `fast_mixing` | 2 | $[0.20,3.20]u_i$, $u_i\sim U[0.85,1.20]$ | Same two-state structure with $q_{01}=0.65$; high contrast but short persistence, reducing forecast value. |
| `predictive_burst` | 5 | even $i$: $[0.05,0.55,2.30,5.20,1.00]u_i$; odd $i$: $[0.08,0.75,3.00,6.00,1.20]u_i$, $u_i\sim U[0.85,1.20]$ | Cyclic generator with forward rate $0.045$ and backward rate $0.004$; ramp-up phases predict near-future bursts. |
| `multi_state_burst` | 6 | one of $[0.04,0.20,0.70,2.80,6.40,1.20]$, $[0.05,0.35,1.10,4.00,7.20,1.50]$, or $[0.08,0.45,1.60,5.00,8.00,1.80]$, scaled by $u_i\sim U[0.80,1.20]$ | Cyclic generator with forward $0.035$, backward $0.003$, and skip $0.002$; multi-stage build-up and decay. |
| `alternating_hotspots` | 4 | base $[0.18,0.35,0.65,1.00]u_i$, with states $i\bmod4$ and $(i+1)\bmod4$ replaced by $6.50u_i$ and $2.20u_i$, $u_i\sim U[0.85,1.15]$ | Four-state cyclic generator with forward $0.055$ and backward $0.006$; queue-specific hot spots rotate across phases. |
| `asymmetric` | 2-3 | if $i\bmod3=0$: $[0.04,1.60,4.80]u_i$; if $i\bmod3=1$: $[0.25,2.20]u_i$; otherwise $[0.55,1.55]u_i$, $u_i\sim U[0.75,1.35]$ | Queue-specific asymmetric generators with different persistence and contrast; tests heterogeneous phase informativeness. |

## Global reproducibility parameters

| Parameter | Value |
|---|---|
| Base random seed | `20260527` |
| Simulation step | $\Delta t=1$ |
| Queue capacity | $B=2000$; a step is marked as overflow if any queue reaches the cap. |
| Server-pool capacities | $s_k\in\{2,3,4\}$, generated per instance and fixed across policies. |
| Arrival process | MMPP per queue; scenario-specific `map_profile`; scaled by target load. |
| Service process | homogeneous, mild, or strong; Poisson or Gamma-Poisson completions. |
| Switching friction | none, moderate, high, or lag. |
| Warm-up and evaluation | 500 warm-up steps; 3000 evaluation steps. |
| Main replications | 50 independent replications per non-RL policy and instance. |
| RL evaluation replications | 10 independent evaluation replications per trained RL agent; 10 independent replications per structural comparison policy and instance in the RL comparison. |
| RL training budget | 250000 environment steps per algorithm, instance, and training seed; 10 training seeds per algorithm and instance. |
| Main policies | Equal-Split, BackPressure, MaxWeight-$\mu$, $c\mu$, GB-PANDAS-style, RL-Queueing-Surrogate, QAH+Switch, gPAH-StationaryMean, and gPAH. |
| gPAH main default | $(\alpha_Q,\alpha_A,\alpha_\mu,\alpha_S,\alpha_C,\alpha_L)=(1.00,1.60,1.00,0.55,0.90,0.35)$, $H=6$, load-balance coefficient $0.25$; fixed across A1-A8. |
| Conditional-ablation reference | `gpah_full` uses a separately frozen mechanism-identification configuration $(1.00,1.50,1.25,0.25,1.25,0.80)$; it is not used for the main comparison or sensitivity baseline. |
| RL policies | DQN-Candidate, PPO-Factorized, and Discrete-SAC-Factorized. |
| Ablation policies | `gpah_full`, `mean_map`, `no_phase`, `no_service`, `no_scarcity`, `no_switch`, and `no_queue`. |
| Confidence intervals | 95% intervals computed as 1.96 standard errors over independent replications. |
| Multiple comparisons | Paired tests are pre-specified within structured-policy and RL comparison families; Holm step-down checks are applied within each family, and median and interquartile-range effect-size summaries are reported in the text. |

## Main experiment instance parameters

| Family | Instances or values | Fixed parameters | Varied parameters |
|---|---|---|---|
| `A1_topology` | `disjoint`, `chain`, `ring`, `star`, `bridge_bottleneck`, `high_overlap`, `random_sparse` | $d=8,K=5$; MMPP=`medium_burst`; service=mild; switch=moderate; density $0.40$ | topology |
| `A2_map_burst` | `low_burst`, `medium_burst`, `high_burst`, `low_contrast`, `fast_mixing`, `predictive_burst`, `multi_state_burst`, `alternating_hotspots`, `asymmetric` | $d=8,K=5$; topology=chain; density $0.40$; service=mild; switch=moderate | MMPP profile |
| `A3_density` | very_sparse: $0.25$; sparse: $0.40$; medium: $0.60$ | $d=8,K=5$; topology=random_sparse; MMPP=medium_burst; service=mild; switch=moderate | density |
| `A4_service` | homogeneous, mild, strong | $d=8,K=5$; topology=chain; density $0.40$; MMPP=medium_burst; switch=moderate | service regime |
| `A5_switch` | none, moderate, high, lag | $d=8,K=5$; topology=chain; density $0.40$; MMPP=medium_burst; service=mild | switch regime |
| `A6_scale` | `d6_K4`, `d6_K6`, `d8_K4`, `d8_K6`, `d12_K4`, `d12_K6` | topology=random_sparse; density $0.42$; MMPP=medium_burst; service=mild; switch=moderate | $d,K$ |
| `A7_load` | `rho_0_6`, `rho_0_75`, `rho_0_9`, `rho_0_95` | $d=10,K=6$; topology=random_sparse; density $0.38$; MMPP=medium_burst; service=mild; switch=moderate | $\rho\in\{0.60,0.75,0.90,0.95\}$ |
| `A8_large_scale` | `d20_K10`, `d30_K12`, `d40_K16`, `d50_K20`, `d75_K30`, `d100_K40` | topology=random_sparse; density $0.30$; MMPP=medium_burst; service=mild; switch=moderate; load $0.75$ | $d,K$ |

## Deep RL baseline settings

| Parameter | Value |
|---|---|
| Algorithms | `dqn`, `ppo`, `sac`, corresponding to DQN-Candidate, PPO-Factorized, and Discrete-SAC-Factorized. |
| Covered scenarios | `A1_topology`, `A2_map_burst`, `A3_density`, `A4_service`, `A5_switch`, `A6_scale`, `A7_load`, and `A8_large_scale`; 38 RL evaluation instances, with A8 covering `d20_K10` and `d50_K20`. |
| Training budget | 250000 environment steps per algorithm, instance, and training seed; 10 training seeds per algorithm and instance. |
| Evaluation | Deterministic post-training policy; 10 independent evaluation replications per trained agent; 500 warm-up and 3000 evaluation steps. |
| Observation | Dimension $3d+K(d+1)+K$; normalized queues, MMPP rate ratios, forecast pressure, previous allocation one-hot, switching cooldown. |
| Action constraints | Compatibility masks; each pool selects idle or one compatible queue. |
| DQN action set | Seven candidate joint-action generators: Equal-Split, queue-only rule, MaxWeight-$\mu$, GB-PANDAS-style, QAH+Switch, gPAH-StationaryMean, and gPAH. |
| PPO/SAC action factorization | Masked categorical actor by server pool; greedy action at evaluation. |
| Action-space fairness | All RL policies obey the same compatibility masks and capacity feasibility; DQN is given direct access to strong structured joint-action generators, including gPAH, while PPO/SAC use a more expressive but harder factorized primitive action space. |
| Training reward | $-\left(Q_{\mathrm{tot}}/(Bd)+C_{\mathrm{sw}}/\sum_{k,i}c_{ki}^{\mathrm{sw}}+3I_{\mathrm{overflow}}\right)$ |
| Shared network | Two-layer rectified-linear-unit multilayer perceptron, width 128; Adam, learning rate $3\times10^{-4}$; discount $\gamma=0.995$; gradient clipping at 5.0; full runs use parallel jobs and available hardware acceleration, while algorithmic settings are device-independent. |
| DQN hyperparameters | Replay buffer 50000; batch size 128; update after 500; update every 4; target soft update $\tau=0.01$; Huber loss; $\varepsilon=\max(0.05,0.9(1-t/250000))$. |
| PPO hyperparameters | Rollout length 256; PPO epochs 4; clip ratio 0.2; entropy coefficient 0.02; value coefficient 0.5; advantage normalization. |
| SAC hyperparameters | Replay buffer 50000; batch size 128; update after 500; update every 4; target soft update $\tau=0.01$; entropy temperature $\alpha=0.08$; twin-Q critic. |

## Shared-policy held-out topology transfer protocol

| Parameter | Value |
|---|---|
| Purpose | Separate topology transfer from per-instance RL retraining; no parameter update is performed on held-out topologies. |
| Training instances | Five fixed A1 topology instances: `disjoint`, `chain`, `ring`, `star`, and `high_overlap`. |
| Held-out instances | Two fixed A1 topology instances: `bridge_bottleneck` and `random_sparse`. |
| Random graph count | One graph instance per topology; `random_sparse` is generated once from the documented seed rather than resampled across many graphs. |
| Shared dimensions | All train and held-out instances use $d=8$, $K=5$, density parameter $0.40$, medium-burst MMPP arrivals, mild service heterogeneity, moderate switching friction, and the same target load construction. |
| Training budget | 250000 total environment steps per algorithm and training seed across the five training instances; 10 training seeds per algorithm. |
| Training instance sampling | During training, the environment is periodically reset to a uniformly selected training instance; DQN/PPO use rollout block length 256, and SAC uses longer replay blocks as implemented in the released script. |
| Observation dimension | $3d+K(d+1)+K=74$; normalized queues, current MMPP rate ratios, forecast pressure, previous allocation one-hot vector, and switching cooldown. |
| Action adaptation | DQN keeps seven candidate joint-action generators; PPO/SAC keep $K(d+1)=45$ masked logits. Held-out evaluation replaces only the compatibility mask and instance parameters, so neural-network dimensions are unchanged. |
| Evaluation | Zero fine-tuning; deterministic post-training policy; 10 independent evaluation replications per trained agent and 10 independent gPAH replications per held-out topology; 500 warm-up and 3000 evaluation steps. |
| Output files | `paper1_rl_unseen_topology_replications.csv`, `paper1_rl_unseen_topology_case_summary.csv`, and `run_manifest.json` in the `rl_topology_generalization` result directory. |

## Conditional ablation instance parameters

| Family | Instance | Regime | Full configuration |
|---|---|---|---|
| `B1_phase` | `phase_off_low_burst` | off | component=phase; $d=10,K=6$; topology=random_sparse; density $0.35$; MMPP=low_burst; service=mild; switch=moderate |
| `B1_phase` | `phase_on_asymmetric_sparse` | on | component=phase; $d=10,K=6$; topology=random_sparse; density $0.35$; MMPP=asymmetric; service=mild; switch=moderate |
| `B1_phase` | `phase_on_high_burst` | on | component=phase; $d=10,K=6$; topology=random_sparse; density $0.35$; MMPP=high_burst; service=mild; switch=moderate |
| `B2_service` | `service_off_homogeneous` | off | component=service; $d=10,K=6$; topology=random_sparse; density $0.35$; MMPP=medium_burst; service=homogeneous; switch=moderate |
| `B2_service` | `service_on_chain_strong` | on | component=service; $d=8,K=5$; topology=chain; density $0.40$; MMPP=medium_burst; service=strong; switch=moderate |
| `B2_service` | `service_on_sparse_strong` | on | component=service; $d=10,K=6$; topology=random_sparse; density $0.35$; MMPP=medium_burst; service=strong; switch=moderate |
| `B3_scarcity` | `scarcity_off_overlap` | off | component=scarcity; $d=10,K=6$; topology=high_overlap; density $0.40$; MMPP=medium_burst; service=mild; switch=moderate |
| `B3_scarcity` | `scarcity_on_bridge` | on | component=scarcity; $d=8,K=5$; topology=bridge_bottleneck; density $0.40$; MMPP=medium_burst; service=mild; switch=moderate |
| `B3_scarcity` | `scarcity_on_very_sparse` | on | component=scarcity; $d=10,K=6$; topology=random_sparse; density $0.22$; MMPP=medium_burst; service=mild; switch=moderate |
| `B4_switch` | `switch_off_none` | off | component=switch; $d=10,K=6$; topology=random_sparse; density $0.35$; MMPP=medium_burst; service=mild; switch=none |
| `B4_switch` | `switch_on_high_cost_sparse` | on | component=switch; $d=10,K=6$; topology=random_sparse; density $0.35$; MMPP=medium_burst; service=mild; switch=high |
| `B4_switch` | `switch_on_lag_chain` | on | component=switch; $d=8,K=5$; topology=chain; density $0.40$; MMPP=medium_burst; service=mild; switch=lag |
| `B5_mixed` | `mixed_off_regular` | off | component=mixed; $d=10,K=6$; topology=high_overlap; density $0.40$; MMPP=low_burst; service=homogeneous; switch=none |
| `B5_mixed` | `mixed_on_bridge_asymmetric` | on | component=mixed; $d=10,K=6$; topology=bridge_bottleneck; density $0.40$; MMPP=asymmetric; service=strong; switch=high |
| `B5_mixed` | `mixed_on_sparse_high_burst` | on | component=mixed; $d=10,K=6$; topology=random_sparse; density $0.34$; MMPP=high_burst; service=strong; switch=high |

## Version-control policy

The tables above document paper-facing values. If a prose table and an
executable configuration diverge, treat the discrepancy as a reproducibility
issue and reconcile it before running or reporting new results. Do not tune
paper-facing parameters from test-set outcomes.
