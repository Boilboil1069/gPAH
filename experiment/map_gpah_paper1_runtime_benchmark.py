#!/usr/bin/env python3
"""Wall-clock benchmark for policy decision time, including d=100, K=40."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from map_gpah_paper1_common import (
    PAPER_COLORS,
    POLICY_COLORS,
    POLICY_LABELS,
    CaseSpec,
    SparseMapPhEnv,
    make_problem,
    mean_ci,
    policy_from_name,
    render_progress,
    write_csv,
)


DEFAULT_POLICIES = "maxweight_mu,affinity,qah_switch,gpah_meanmap,gpah"


def parse_sizes(raw: str) -> List[Tuple[int, int]]:
    sizes: List[Tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        d_raw, k_raw = item.split("x", 1)
        d, K = int(d_raw), int(k_raw)
        if d < 1 or K < 1:
            raise ValueError(f"invalid size: {item}")
        sizes.append((d, K))
    if not sizes:
        raise ValueError("--sizes must contain at least one d x K pair")
    return sizes


def timer_overhead_ns(samples: int = 20000) -> float:
    durations = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        durations.append(time.perf_counter_ns() - start)
    return float(np.median(np.asarray(durations, dtype=float)))


def benchmark_replication(
    problem,
    policy_name: str,
    seed: int,
    warmup_decisions: int,
    measured_decisions: int,
) -> Dict[str, float]:
    env = SparseMapPhEnv(problem, seed)
    env.reset()
    policy = policy_from_name(policy_name)
    policy.reset(problem)
    for _ in range(warmup_decisions):
        env.step(policy.action(env))

    action_ns = np.empty(measured_decisions, dtype=float)
    step_ns = np.empty(measured_decisions, dtype=float)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for index in range(measured_decisions):
            start = time.perf_counter_ns()
            action = policy.action(env)
            action_ns[index] = time.perf_counter_ns() - start
            start = time.perf_counter_ns()
            env.step(action)
            step_ns[index] = time.perf_counter_ns() - start
    finally:
        if gc_was_enabled:
            gc.enable()
    return {
        "action_mean_us": float(action_ns.mean() / 1000.0),
        "action_median_us": float(np.median(action_ns) / 1000.0),
        "action_p95_us": float(np.quantile(action_ns, 0.95) / 1000.0),
        "action_p99_us": float(np.quantile(action_ns, 0.99) / 1000.0),
        "action_decisions_per_second": float(1e9 / max(1.0, action_ns.mean())),
        "simulator_step_mean_us": float(step_ns.mean() / 1000.0),
        "combined_mean_us": float((action_ns + step_ns).mean() / 1000.0),
    }


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    metrics = (
        "action_mean_us",
        "action_median_us",
        "action_p95_us",
        "action_p99_us",
        "action_decisions_per_second",
        "simulator_step_mean_us",
        "combined_mean_us",
    )
    groups: Dict[tuple[int, int, str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (int(row["d"]), int(row["K"]), str(row["policy"]), str(row["policy_label"]))
        groups.setdefault(key, []).append(row)
    output: List[Dict[str, object]] = []
    for (d, K, policy, label), group in sorted(groups.items()):
        result: Dict[str, object] = {"d": d, "K": K, "policy": policy, "policy_label": label, "replications": len(group)}
        for metric in metrics:
            stats = mean_ci(float(row[metric]) for row in group)
            result.update({f"{metric}_{name}": value for name, value in stats.items()})
        output.append(result)
    return output


def plot_results(rows: Sequence[Dict[str, object]], output_path: str) -> None:
    policies = list(dict.fromkeys(str(row["policy"]) for row in rows))
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    for policy in policies:
        subset = sorted((row for row in rows if row["policy"] == policy), key=lambda row: int(row["d"]))
        ax.plot(
            [int(row["d"]) * int(row["K"]) for row in subset],
            [float(row["action_median_us_mean"]) for row in subset],
            marker="o",
            linewidth=1.7,
            label=POLICY_LABELS.get(policy, policy),
            color=POLICY_COLORS.get(policy, PAPER_COLORS["soft_gray"]),
        )
    ax.set_xlabel(r"Dense scan size $Kd$")
    ax.set_ylabel("Median decision time (microseconds)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy wall-clock benchmark at large sparse-network sizes.")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--sizes", default="20x10,50x20,100x40")
    parser.add_argument("--policies", default=DEFAULT_POLICIES)
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--warmup-decisions", type=int, default=200)
    parser.add_argument("--measured-decisions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=32260527)
    args = parser.parse_args()
    if args.replications < 1 or args.measured_decisions < 1:
        raise ValueError("replications and measured-decisions must be positive")

    sizes = parse_sizes(args.sizes)
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    rows: List[Dict[str, object]] = []
    total = len(sizes) * len(policies) * args.replications
    done = 0
    overhead = timer_overhead_ns()
    for size_index, (d, K) in enumerate(sizes, 1):
        case = CaseSpec(
            "revision_runtime",
            f"d{d}_K{K}",
            f"runtime benchmark d={d}, K={K}",
            d,
            K,
            topology="random_sparse",
            density=0.30,
            map_profile="medium_burst",
            service_profile="mild",
            switch_profile="moderate",
            load=0.75,
        )
        problem = make_problem(case, args.seed + 1009 * size_index)
        edge_count = int(problem.compat.sum())
        for replication in range(args.replications):
            simulation_seed = args.seed + 100000 * size_index + replication
            for policy in policies:
                metrics = benchmark_replication(
                    problem,
                    policy,
                    simulation_seed,
                    args.warmup_decisions,
                    args.measured_decisions,
                )
                rows.append({
                    "experiment": "runtime_benchmark",
                    "d": d,
                    "K": K,
                    "Kd": d * K,
                    "edge_count": edge_count,
                    "policy": policy,
                    "policy_label": POLICY_LABELS.get(policy, policy),
                    "replication": replication,
                    "seed": simulation_seed,
                    "warmup_decisions": args.warmup_decisions,
                    "measured_decisions": args.measured_decisions,
                    "timer_overhead_median_ns": overhead,
                    **metrics,
                })
                done += 1
                render_progress(done, total, f"d={d},K={K} {policy}")

    summary = summarize(rows)
    write_csv(os.path.join(args.output_dir, "paper1_runtime_replications.csv"), rows)
    write_csv(os.path.join(args.output_dir, "paper1_runtime_summary.csv"), summary)
    plot_results(summary, os.path.join(args.output_dir, "fig_runtime_scaling.pdf"))
    metadata = {
        "experiment": "runtime_benchmark",
        "clock": "time.perf_counter_ns",
        "timed_region": "policy.action only; simulator step timed separately",
        "garbage_collection_during_timing": "disabled",
        "timer_overhead_median_ns": overhead,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "sizes": [{"d": d, "K": K} for d, K in sizes],
        "policies": policies,
        "replications": args.replications,
        "warmup_decisions": args.warmup_decisions,
        "measured_decisions": args.measured_decisions,
        "seed": args.seed,
        "thread_environment": {
            key: os.environ.get(key, "")
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
    }
    with open(os.path.join(args.output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Runtime benchmark results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
