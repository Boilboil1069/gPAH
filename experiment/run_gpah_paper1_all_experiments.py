#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


DEFAULT_POLICY_FAMILIES = "A1_topology,A2_map_burst,A3_density,A4_service,A5_switch,A6_scale,A7_load,A8_large_scale"
DEFAULT_POLICY_POLICIES = "equal_split,backpressure,maxweight_mu,cmu,affinity,rl_queueing,qah_switch,gpah_meanmap,gpah"
DEFAULT_RL_FAMILIES = DEFAULT_POLICY_FAMILIES
DEFAULT_RL_ALGORITHMS = "dqn,ppo,sac"
DEFAULT_RL_BUDGET_PRESET = "full"
DEFAULT_SENSITIVITY_FAMILIES = "A2_map_burst,A5_switch,A7_load,A8_large_scale"
DEFAULT_SENSITIVITY_PARAMETERS = "alpha_Q,alpha_A,alpha_mu,alpha_S,alpha_C,alpha_L"
DEFAULT_SENSITIVITY_MULTIPLIERS = "0,0.5,1,1.5,2"


def run_command(cmd: List[str], cwd: Path, dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified runner for Paper 1 policy-comparison and RL baseline experiments."
    )
    parser.add_argument(
        "--output_dir",
        default="gpah_papers/paper1_gpah_advantage/new_experiments/paper1_unified",
        help="Base output directory. A run_name subdirectory is created unless --no_timestamped_output is set.",
    )
    parser.add_argument("--run_name", default="", help="Run folder name. Defaults to run_YYYYmmdd_HHMMSS.")
    parser.add_argument("--no_timestamped_output", action="store_true", help="Write directly under output_dir.")
    parser.add_argument("--skip_policy", action="store_true", help="Skip the fixed-policy comparison experiment.")
    parser.add_argument("--skip_rl", action="store_true", help="Skip DQN/PPO/SAC RL baseline training.")
    parser.add_argument("--skip_sensitivity", action="store_true", help="Skip gPAH alpha-parameter sensitivity experiments.")
    parser.add_argument(
        "--run_buffer_robustness",
        action="store_true",
        help="Run only the incremental high-load buffer-cap and long-horizon robustness diagnostics.",
    )
    parser.add_argument(
        "--run_rl_topology_generalization",
        action="store_true",
        help="Run only the additional shared-policy zero-shot evaluation on held-out topology types.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them.")

    parser.add_argument("--policy_families", default=DEFAULT_POLICY_FAMILIES)
    parser.add_argument("--policy_policies", default=DEFAULT_POLICY_POLICIES)
    parser.add_argument("--policy_replications", type=int, default=50)
    parser.add_argument("--policy_warmup_steps", type=int, default=500)
    parser.add_argument("--policy_eval_steps", type=int, default=3000)

    parser.add_argument("--rl_families", default=DEFAULT_RL_FAMILIES)
    parser.add_argument("--rl_case_ids", default="", help="Optional comma-separated RL case_id subset within selected families.")
    parser.add_argument("--rl_algorithms", default=DEFAULT_RL_ALGORITHMS)
    parser.add_argument("--rl_budget_preset", choices=["smoke", "review", "full"], default=DEFAULT_RL_BUDGET_PRESET)
    parser.add_argument("--rl_train_steps", type=int, default=0, help="Override RL preset train steps when positive.")
    parser.add_argument("--rl_train_seeds", type=int, default=0, help="Override RL preset train seeds when positive.")
    parser.add_argument("--rl_replications", type=int, default=10)
    parser.add_argument("--rl_warmup_steps", type=int, default=500)
    parser.add_argument("--rl_eval_steps", type=int, default=3000)
    parser.add_argument("--rl_rollout_steps", type=int, default=256)
    parser.add_argument("--rl_device", default="cpu")
    parser.add_argument("--rl_jobs", type=int, default=1, help="Number of parallel RL case workers.")
    parser.add_argument("--rl_seed_offset", type=int, default=900000, help="Offset added to --seed for the RL baseline subexperiment.")
    parser.add_argument("--no_rl_train_progress", action="store_true", help="Disable per-agent RL training progress bars.")
    parser.add_argument(
        "--rl_reuse_fixed_policy_dir",
        default="",
        help="Optional directory containing existing fixed-policy replication CSVs for the RL comparison.",
    )
    parser.add_argument("--rl_generalization_train_topologies", default="disjoint,chain,ring,star,high_overlap")
    parser.add_argument("--rl_generalization_test_topologies", default="bridge_bottleneck,random_sparse")
    parser.add_argument("--rl_generalization_algorithms", default=DEFAULT_RL_ALGORITHMS)
    parser.add_argument("--rl_generalization_budget_preset", choices=["smoke", "review", "full"], default="full")
    parser.add_argument("--rl_generalization_train_steps", type=int, default=0)
    parser.add_argument("--rl_generalization_train_seeds", type=int, default=0)
    parser.add_argument("--rl_generalization_replications", type=int, default=10)

    parser.add_argument("--sensitivity_families", default=DEFAULT_SENSITIVITY_FAMILIES)
    parser.add_argument("--sensitivity_parameters", default=DEFAULT_SENSITIVITY_PARAMETERS)
    parser.add_argument("--sensitivity_multipliers", default=DEFAULT_SENSITIVITY_MULTIPLIERS)
    parser.add_argument("--sensitivity_replications", type=int, default=30)
    parser.add_argument("--sensitivity_warmup_steps", type=int, default=500)
    parser.add_argument("--sensitivity_eval_steps", type=int, default=3000)
    parser.add_argument("--buffer_robustness_policies", default=DEFAULT_POLICY_POLICIES)
    parser.add_argument("--buffer_robustness_loads", default="0.90,0.95")
    parser.add_argument("--buffer_robustness_replications", type=int, default=20)
    parser.add_argument("--buffer_robustness_warmup_steps", type=int, default=1000)
    parser.add_argument("--buffer_robustness_base_eval_steps", type=int, default=3000)
    parser.add_argument("--buffer_robustness_long_eval_steps", type=int, default=30000)
    parser.add_argument("--buffer_robustness_base_cap", type=int, default=2000)
    parser.add_argument("--buffer_robustness_large_cap", type=int, default=10000)

    parser.add_argument("--seed", type=int, default=20260527)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_root = repo_root / args.output_dir
    if not args.no_timestamped_output:
        run_name = args.run_name.strip() or "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = run_root / run_name
    policy_dir = run_root / "policy"
    rl_dir = run_root / "rl"
    sensitivity_dir = run_root / "sensitivity"
    rl_generalization_dir = run_root / "rl_topology_generalization"
    buffer_robustness_dir = run_root / "buffer_robustness"
    run_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_root": str(run_root),
        "policy_dir": str(policy_dir),
        "rl_dir": str(rl_dir),
        "seed": args.seed,
        "skip_policy": args.skip_policy,
        "skip_rl": args.skip_rl,
        "skip_sensitivity": args.skip_sensitivity,
        "policy_families": args.policy_families,
        "policy_policies": args.policy_policies,
        "rl_families": args.rl_families,
        "rl_case_ids": args.rl_case_ids,
        "rl_algorithms": args.rl_algorithms,
        "rl_budget_preset": args.rl_budget_preset,
        "rl_device": args.rl_device,
        "rl_jobs": args.rl_jobs,
        "rl_seed_offset": args.rl_seed_offset,
        "rl_train_progress": not args.no_rl_train_progress,
        "rl_reuse_fixed_policy_dir": args.rl_reuse_fixed_policy_dir,
        "run_rl_topology_generalization": args.run_rl_topology_generalization,
        "run_buffer_robustness": args.run_buffer_robustness,
        "rl_generalization_train_topologies": args.rl_generalization_train_topologies,
        "rl_generalization_test_topologies": args.rl_generalization_test_topologies,
        "rl_generalization_algorithms": args.rl_generalization_algorithms,
        "rl_generalization_budget_preset": args.rl_generalization_budget_preset,
        "rl_train_steps_override": args.rl_train_steps if args.rl_train_steps > 0 else None,
        "rl_train_seeds_override": args.rl_train_seeds if args.rl_train_seeds > 0 else None,
        "sensitivity_families": args.sensitivity_families,
        "sensitivity_parameters": args.sensitivity_parameters,
        "sensitivity_multipliers": args.sensitivity_multipliers,
        "buffer_robustness_policies": args.buffer_robustness_policies,
        "buffer_robustness_loads": args.buffer_robustness_loads,
        "buffer_robustness_replications": args.buffer_robustness_replications,
        "buffer_robustness_base_eval_steps": args.buffer_robustness_base_eval_steps,
        "buffer_robustness_long_eval_steps": args.buffer_robustness_long_eval_steps,
        "buffer_robustness_base_cap": args.buffer_robustness_base_cap,
        "buffer_robustness_large_cap": args.buffer_robustness_large_cap,
    }
    (run_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Unified output directory: {run_root}", flush=True)

    if not args.skip_policy:
        policy_cmd = [
            sys.executable,
            "experiment/map_gpah_paper1_experiments.py",
            "--output_dir",
            str(policy_dir),
            "--no_timestamped_output",
            "--families",
            args.policy_families,
            "--policies",
            args.policy_policies,
            "--replications",
            str(args.policy_replications),
            "--warmup_steps",
            str(args.policy_warmup_steps),
            "--eval_steps",
            str(args.policy_eval_steps),
            "--seed",
            str(args.seed),
        ]
        run_command(policy_cmd, repo_root, args.dry_run)

    if not args.skip_sensitivity:
        sensitivity_cmd = [
            sys.executable,
            "experiment/map_gpah_paper1_sensitivity.py",
            "--output_dir",
            str(sensitivity_dir),
            "--no_timestamped_output",
            "--families",
            args.sensitivity_families,
            "--parameters",
            args.sensitivity_parameters,
            "--multipliers",
            args.sensitivity_multipliers,
            "--replications",
            str(args.sensitivity_replications),
            "--warmup_steps",
            str(args.sensitivity_warmup_steps),
            "--eval_steps",
            str(args.sensitivity_eval_steps),
            "--seed",
            str(args.seed + 700000),
        ]
        run_command(sensitivity_cmd, repo_root, args.dry_run)

    if not args.skip_rl:
        rl_cmd = [
            sys.executable,
            "experiment/map_gpah_paper1_rl_baselines.py",
            "--output_dir",
            str(rl_dir),
            "--algorithms",
            args.rl_algorithms,
            "--families",
            args.rl_families,
            "--case-ids",
            args.rl_case_ids,
            "--budget-preset",
            args.rl_budget_preset,
            "--replications",
            str(args.rl_replications),
            "--warmup-steps",
            str(args.rl_warmup_steps),
            "--eval-steps",
            str(args.rl_eval_steps),
            "--rollout-steps",
            str(args.rl_rollout_steps),
            "--seed",
            str(args.seed + args.rl_seed_offset),
            "--device",
            args.rl_device,
            "--jobs",
            str(max(1, int(args.rl_jobs))),
        ]
        if args.no_rl_train_progress:
            rl_cmd.append("--no-train-progress")
        if args.rl_reuse_fixed_policy_dir:
            rl_cmd.extend(["--reuse-fixed-policy-dir", args.rl_reuse_fixed_policy_dir])
        if args.rl_train_steps > 0:
            rl_cmd.extend(["--train-steps", str(args.rl_train_steps)])
        if args.rl_train_seeds > 0:
            rl_cmd.extend(["--train-seeds", str(args.rl_train_seeds)])
        run_command(rl_cmd, repo_root, args.dry_run)
        rl_plot_cmd = [
            sys.executable,
            "experiment/replot_gpah_paper1_rl_figures.py",
            "--output_dir",
            str(rl_dir),
        ]
        run_command(rl_plot_cmd, repo_root, args.dry_run)

    if args.run_rl_topology_generalization:
        generalization_cmd = [
            sys.executable,
            "experiment/map_gpah_paper1_rl_topology_generalization.py",
            "--output_dir",
            str(rl_generalization_dir),
            "--algorithms",
            args.rl_generalization_algorithms,
            "--train-topologies",
            args.rl_generalization_train_topologies,
            "--test-topologies",
            args.rl_generalization_test_topologies,
            "--budget-preset",
            args.rl_generalization_budget_preset,
            "--replications",
            str(args.rl_generalization_replications),
            "--warmup-steps",
            str(args.rl_warmup_steps),
            "--eval-steps",
            str(args.rl_eval_steps),
            "--rollout-steps",
            str(args.rl_rollout_steps),
            "--seed",
            str(args.seed + args.rl_seed_offset + 5000000),
            "--device",
            args.rl_device,
        ]
        if args.no_rl_train_progress:
            generalization_cmd.append("--no-train-progress")
        if args.rl_generalization_train_steps > 0:
            generalization_cmd.extend(["--train-steps", str(args.rl_generalization_train_steps)])
        if args.rl_generalization_train_seeds > 0:
            generalization_cmd.extend(["--train-seeds", str(args.rl_generalization_train_seeds)])
        run_command(generalization_cmd, repo_root, args.dry_run)

    if args.run_buffer_robustness:
        buffer_cmd = [
            sys.executable,
            "experiment/map_gpah_paper1_buffer_robustness.py",
            "--output_dir",
            str(buffer_robustness_dir),
            "--policies",
            args.buffer_robustness_policies,
            "--loads",
            args.buffer_robustness_loads,
            "--replications",
            str(args.buffer_robustness_replications),
            "--warmup_steps",
            str(args.buffer_robustness_warmup_steps),
            "--base_eval_steps",
            str(args.buffer_robustness_base_eval_steps),
            "--long_eval_steps",
            str(args.buffer_robustness_long_eval_steps),
            "--base_buffer_cap",
            str(args.buffer_robustness_base_cap),
            "--large_buffer_cap",
            str(args.buffer_robustness_large_cap),
            "--seed",
            str(args.seed + 7000000),
        ]
        run_command(buffer_cmd, repo_root, args.dry_run)

    print("\nDone.", flush=True)
    print(f"Results root: {run_root}", flush=True)
    if not args.skip_policy:
        print(f"Policy results: {policy_dir}", flush=True)
    if not args.skip_sensitivity:
        print(f"Sensitivity results: {sensitivity_dir}", flush=True)
    if not args.skip_rl:
        print(f"RL results: {rl_dir}", flush=True)
    if args.run_rl_topology_generalization:
        print(f"RL unseen-topology results: {rl_generalization_dir}", flush=True)
    if args.run_buffer_robustness:
        print(f"Buffer robustness results: {buffer_robustness_dir}", flush=True)


if __name__ == "__main__":
    main()
