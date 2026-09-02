#!/usr/bin/env python3
"""Uncapped long-horizon queue-growth diagnostic for high-load cases.

This experiment deliberately reports queue trajectories and drift diagnostics,
not a finite-buffer objective and not a proof of positive recurrence.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import os
from typing import Dict, List, Sequence

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


DEFAULT_POLICIES = "equal_split,backpressure,maxweight_mu,affinity,qah_switch,gpah_meanmap,gpah"


def simulate_uncapped(
    problem,
    policy_name: str,
    seed: int,
    warmup_steps: int,
    eval_steps: int,
    block_steps: int,
) -> tuple[Dict[str, float], List[Dict[str, float]]]:
    if problem.buffer_cap is not None:
        raise ValueError("infinite-buffer diagnostic requires problem.buffer_cap=None")
    env = SparseMapPhEnv(problem, seed)
    env.reset()
    policy = policy_from_name(policy_name)
    policy.reset(problem)
    for _ in range(warmup_steps):
        info = env.step(policy.action(env))
        if info["dropped_arrivals"] != 0.0 or info["overflow"] != 0.0:
            raise RuntimeError("uncapped environment unexpectedly reported censoring")

    queues: List[float] = []
    arrivals = 0.0
    served = 0.0
    switch_cost = 0.0
    traces: List[Dict[str, float]] = []
    current_block: List[float] = []
    for step in range(1, eval_steps + 1):
        info = env.step(policy.action(env))
        queue = float(info["total_queue"])
        queues.append(queue)
        current_block.append(queue)
        arrivals += float(info["arrivals"])
        served += float(info["served"])
        switch_cost += float(info["switch_penalty"])
        if step % block_steps == 0 or step == eval_steps:
            traces.append({
                "step_end": float(step),
                "block_mean_total_queue": float(np.mean(current_block)),
                "block_last_total_queue": queue,
            })
            current_block = []

    q = np.asarray(queues, dtype=float)
    half = max(1, len(q) // 2)
    second_x = np.arange(half, len(q), dtype=float)
    second_q = q[half:]
    slope = float(np.polyfit(second_x, second_q, 1)[0]) if len(second_q) > 1 else 0.0
    return {
        "mean_total_queue": float(q.mean()),
        "p95_total_queue": float(np.quantile(q, 0.95)),
        "maximum_total_queue": float(q.max()),
        "initial_total_queue": float(q[0]),
        "terminal_total_queue": float(q[-1]),
        "first_half_mean_queue": float(q[:half].mean()),
        "second_half_mean_queue": float(q[half:].mean()),
        "queue_growth": float(q[half:].mean() - q[:half].mean()),
        "queue_time_slope_second_half": slope,
        "terminal_drift_per_step": float((q[-1] - q[0]) / max(1, eval_steps - 1)),
        "net_input_rate": float((arrivals - served) / max(1, eval_steps)),
        "mean_arrivals": float(arrivals / max(1, eval_steps)),
        "mean_served": float(served / max(1, eval_steps)),
        "mean_switch_cost": float(switch_cost / max(1, eval_steps)),
        "drop_rate": 0.0,
        "cap_time_fraction": 0.0,
    }, traces


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    metrics = (
        "mean_total_queue",
        "p95_total_queue",
        "maximum_total_queue",
        "terminal_total_queue",
        "queue_growth",
        "queue_time_slope_second_half",
        "terminal_drift_per_step",
        "net_input_rate",
        "mean_arrivals",
        "mean_served",
        "mean_switch_cost",
    )
    groups: Dict[tuple[float, str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (float(row["load"]), str(row["policy"]), str(row["policy_label"]))
        groups.setdefault(key, []).append(row)
    summary: List[Dict[str, object]] = []
    for (load, policy, label), group in sorted(groups.items()):
        result: Dict[str, object] = {
            "load": load,
            "policy": policy,
            "policy_label": label,
            "replications": len(group),
        }
        for metric in metrics:
            stats = mean_ci(float(row[metric]) for row in group)
            result.update({f"{metric}_{name}": value for name, value in stats.items()})
        summary.append(result)
    return summary


def summarize_traces(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple[float, str, str, int], List[float]] = {}
    for row in rows:
        key = (
            float(row["load"]),
            str(row["policy"]),
            str(row["policy_label"]),
            int(row["step_end"]),
        )
        groups.setdefault(key, []).append(float(row["block_mean_total_queue"]))
    output: List[Dict[str, object]] = []
    for (load, policy, label, step_end), values in sorted(groups.items()):
        stats = mean_ci(values)
        output.append({
            "load": load,
            "policy": policy,
            "policy_label": label,
            "step_end": step_end,
            **{f"block_mean_total_queue_{name}": value for name, value in stats.items()},
        })
    return output


def plot_trajectories(rows: Sequence[Dict[str, object]], output_path: str) -> None:
    loads = sorted({float(row["load"]) for row in rows})
    policies = list(dict.fromkeys(str(row["policy"]) for row in rows))
    fig, axes = plt.subplots(1, len(loads), figsize=(6.0 * len(loads), 4.8), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, load in zip(axes, loads):
        for policy in policies:
            subset = sorted(
                (row for row in rows if float(row["load"]) == load and row["policy"] == policy),
                key=lambda row: int(row["step_end"]),
            )
            if not subset:
                continue
            ax.plot(
                [int(row["step_end"]) for row in subset],
                [float(row["block_mean_total_queue_mean"]) for row in subset],
                linewidth=1.7,
                label=POLICY_LABELS.get(policy, policy),
                color=POLICY_COLORS.get(policy, PAPER_COLORS["soft_gray"]),
            )
        ax.set_title(rf"Uncapped, $\rho={load:.2f}$")
        ax.set_xlabel("Evaluation step")
        ax.set_ylabel("Block mean total queue (log scale)")
        ax.set_yscale("log")
        ax.grid(which="both", alpha=0.22)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(4, len(labels)),
        frameon=False,
    )
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.075, right=0.985, wspace=0.20)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Uncapped high-load long-horizon diagnostic.")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--policies", default=DEFAULT_POLICIES)
    parser.add_argument("--loads", default="0.90,0.95")
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=100000)
    parser.add_argument("--block-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=31260527)
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Regenerate the trajectory PDF from the existing trace-summary CSV without simulation.",
    )
    args = parser.parse_args()
    if args.eval_steps < 2 or args.block_steps < 1:
        raise ValueError("eval-steps must be at least 2 and block-steps must be positive")

    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    loads = [float(item) for item in args.loads.split(",") if item.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    if args.replot_only:
        trace_summary_path = os.path.join(args.output_dir, "paper1_infinite_buffer_trace_summary.csv")
        with open(trace_summary_path, newline="", encoding="utf-8") as handle:
            trace_summary = list(csv.DictReader(handle))
        if not trace_summary:
            raise ValueError(f"trace summary is empty: {trace_summary_path}")
        plot_trajectories(trace_summary, os.path.join(args.output_dir, "fig_infinite_buffer_queue_trajectories.pdf"))
        print(f"Replotted infinite-buffer diagnostic: {args.output_dir}", flush=True)
        return
    rep_rows: List[Dict[str, object]] = []
    trace_rows: List[Dict[str, object]] = []
    total = len(loads) * len(policies) * args.replications
    done = 0
    for load_index, load in enumerate(loads, 1):
        case = CaseSpec(
            "revision_infinite_buffer",
            f"rho_{load:.2f}",
            f"uncapped high-load diagnostic rho={load:.2f}",
            10,
            6,
            topology="random_sparse",
            density=0.38,
            map_profile="medium_burst",
            service_profile="mild",
            switch_profile="moderate",
            load=load,
        )
        base_problem = make_problem(case, args.seed + 1009 * load_index)
        problem = copy.deepcopy(base_problem)
        problem.buffer_cap = None
        for replication in range(args.replications):
            simulation_seed = args.seed + 100000 * load_index + replication
            for policy in policies:
                metrics, traces = simulate_uncapped(
                    problem,
                    policy,
                    simulation_seed,
                    args.warmup_steps,
                    args.eval_steps,
                    args.block_steps,
                )
                rep_rows.append({
                    "experiment": "infinite_buffer_high_load",
                    "load": load,
                    "buffer_cap": "uncapped",
                    "policy": policy,
                    "policy_label": POLICY_LABELS.get(policy, policy),
                    "replication": replication,
                    "seed": simulation_seed,
                    "warmup_steps": args.warmup_steps,
                    "eval_steps": args.eval_steps,
                    **metrics,
                })
                for trace in traces:
                    trace_rows.append({
                        "load": load,
                        "policy": policy,
                        "policy_label": POLICY_LABELS.get(policy, policy),
                        "replication": replication,
                        "seed": simulation_seed,
                        **trace,
                    })
                done += 1
                render_progress(done, total, f"rho={load:.2f} {policy}")

    summary = summarize(rep_rows)
    trace_summary = summarize_traces(trace_rows)
    write_csv(os.path.join(args.output_dir, "paper1_infinite_buffer_replications.csv"), rep_rows)
    write_csv(os.path.join(args.output_dir, "paper1_infinite_buffer_summary.csv"), summary)
    write_csv(os.path.join(args.output_dir, "paper1_infinite_buffer_trace_blocks.csv"), trace_rows)
    write_csv(os.path.join(args.output_dir, "paper1_infinite_buffer_trace_summary.csv"), trace_summary)
    plot_trajectories(trace_summary, os.path.join(args.output_dir, "fig_infinite_buffer_queue_trajectories.pdf"))
    manifest = {
        "experiment": "infinite_buffer_high_load",
        "interpretation": "long-horizon uncapped drift diagnostic; not a stability proof",
        "loads": loads,
        "policies": policies,
        "replications": args.replications,
        "warmup_steps": args.warmup_steps,
        "eval_steps": args.eval_steps,
        "block_steps": args.block_steps,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Infinite-buffer diagnostic results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
