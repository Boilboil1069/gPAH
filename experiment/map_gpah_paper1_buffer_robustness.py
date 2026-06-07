#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

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


DEFAULT_POLICIES = "equal_split,backpressure,maxweight_mu,cmu,affinity,rl_queueing,qah_switch,gpah_meanmap,gpah"

BUFFER_POLICY_ORDER = [
    "gpah",
    "gpah_meanmap",
    "rl_queueing",
    "affinity",
    "maxweight_mu",
    "cmu",
    "qah_switch",
    "backpressure",
    "equal_split",
]


def simulate_diagnostic(problem, policy_name: str, seed: int, warmup_steps: int, eval_steps: int) -> Dict[str, float]:
    env = SparseMapPhEnv(problem, seed=seed)
    env.reset()
    policy = policy_from_name(policy_name)
    policy.reset(problem)
    for _ in range(warmup_steps):
        env.step(policy.action(env))

    queues: List[float] = []
    sums = {k: 0.0 for k in ["arrivals", "dropped_arrivals", "overflow", "cap_queue_fraction"]}
    for _ in range(eval_steps):
        info = env.step(policy.action(env))
        queues.append(float(info["total_queue"]))
        for key in sums:
            sums[key] += float(info[key])

    q = np.asarray(queues, dtype=float)
    split = max(1, len(q) // 2)
    first = float(q[:split].mean())
    second = float(q[split:].mean())
    x = np.arange(len(q), dtype=float)
    slope = float(np.polyfit(x, q, 1)[0]) if len(q) > 1 else 0.0
    return {
        "mean_total_queue": float(q.mean()),
        "p95_total_queue": float(np.quantile(q, 0.95)),
        "cap_time_fraction": sums["overflow"] / max(1, eval_steps),
        "cap_queue_fraction": sums["cap_queue_fraction"] / max(1, eval_steps),
        "mean_dropped_arrivals": sums["dropped_arrivals"] / max(1, eval_steps),
        "drop_rate": sums["dropped_arrivals"] / max(1.0, sums["arrivals"]),
        "first_half_mean_queue": first,
        "second_half_mean_queue": second,
        "queue_growth": second - first,
        "queue_growth_ratio": (second - first) / max(1.0, first),
        "queue_time_slope": slope,
    }


def plot_metric(rows: List[Dict[str, object]], metric: str, ylabel: str, output_path: str) -> None:
    policies = list(dict.fromkeys(str(r["policy"]) for r in rows))
    scenarios = list(dict.fromkeys(str(r["scenario"]) for r in rows))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.4 * len(scenarios), 3.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, scenario in zip(axes, scenarios):
        subset = [r for r in rows if r["scenario"] == scenario]
        for policy in policies:
            p = sorted((r for r in subset if r["policy"] == policy), key=lambda r: float(r["load"]))
            ax.plot(
                [float(r["load"]) for r in p],
                [float(r[f"{metric}_mean"]) for r in p],
                marker="o",
                linewidth=1.5,
                label=POLICY_LABELS.get(policy, policy),
                color=POLICY_COLORS.get(policy, PAPER_COLORS["soft_gray"]),
            )
        ax.set_title(scenario)
        ax.set_xlabel(r"Load $\rho$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostic_heatmap(rows: List[Dict[str, object]], output_path: str) -> None:
    policies = [p for p in BUFFER_POLICY_ORDER if any(str(r["policy"]) == p for r in rows)]
    conditions = [
        (0.90, "B2000_T3000", "$\\rho=.90$\n$B=2k,T=3k$"),
        (0.90, "B10000_T3000", "$\\rho=.90$\n$B=10k,T=3k$"),
        (0.90, "B10000_T30000", "$\\rho=.90$\n$B=10k,T=30k$"),
        (0.95, "B2000_T3000", "$\\rho=.95$\n$B=2k,T=3k$"),
        (0.95, "B10000_T3000", "$\\rho=.95$\n$B=10k,T=3k$"),
        (0.95, "B10000_T30000", "$\\rho=.95$\n$B=10k,T=30k$"),
    ]
    lookup = {(float(r["load"]), str(r["scenario"]), str(r["policy"])): r for r in rows}
    metrics = [
        ("drop_rate_mean", "Dropped arrivals (%)", 100.0, 45.0, "risk"),
        ("cap_time_fraction_mean", "Time with any capped queue (%)", 100.0, 100.0, "risk"),
        ("queue_time_slope_mean", "Queue time slope (jobs/step)", 1.0, 11.0, "trend"),
    ]
    risk_cmap = LinearSegmentedColormap.from_list(
        "buffer_risk",
        ["#edf4f8", "#f5d46f", "#ed8a52", "#c64b45"],
    )
    trend_cmap = LinearSegmentedColormap.from_list(
        "buffer_trend",
        ["#edf4f8", "#95c6c8", "#3f8e9b", "#31546f"],
    )
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0), constrained_layout=True)
    for panel, (ax, (metric, label, multiplier, vmax, cmap_kind)) in enumerate(zip(axes, metrics)):
        mat = np.full((len(policies), len(conditions)), np.nan)
        for i, policy in enumerate(policies):
            for j, (load, scenario, _) in enumerate(conditions):
                row = lookup.get((load, scenario, policy))
                if row is not None:
                    mat[i, j] = float(row[metric]) * multiplier
        cmap = risk_cmap if cmap_kind == "risk" else trend_cmap
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap=cmap,
            norm=PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax),
        )
        ax.set_xticks(np.arange(len(conditions)))
        ax.set_xticklabels([c[2] for c in conditions], fontsize=8)
        ax.set_yticks(np.arange(len(policies)))
        ax.set_yticklabels([POLICY_LABELS.get(p, p) for p in policies] if panel == 0 else [])
        ax.set_xlabel(label, labelpad=8)
        ax.set_xticks(np.arange(len(conditions) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(policies) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="#ffffff", linewidth=1.1)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                value = mat[i, j]
                if not np.isfinite(value):
                    continue
                text = f"{value:.1f}" if value >= 0.095 else "0"
                color = "white" if value > 0.52 * vmax else "#202020"
                ax.text(j, i, text, ha="center", va="center", fontsize=7.3, color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental buffer-cap and long-horizon robustness diagnostics.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--policies", default=DEFAULT_POLICIES)
    parser.add_argument("--loads", default="0.90,0.95")
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--base_eval_steps", type=int, default=3000)
    parser.add_argument("--long_eval_steps", type=int, default=30000)
    parser.add_argument("--base_buffer_cap", type=int, default=2000)
    parser.add_argument("--large_buffer_cap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=27260527)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    loads = [float(x) for x in args.loads.split(",") if x.strip()]
    scenarios = [
        (f"B{args.base_buffer_cap}_T{args.base_eval_steps}", args.base_buffer_cap, args.base_eval_steps),
        (f"B{args.large_buffer_cap}_T{args.base_eval_steps}", args.large_buffer_cap, args.base_eval_steps),
        (f"B{args.large_buffer_cap}_T{args.long_eval_steps}", args.large_buffer_cap, args.long_eval_steps),
    ]
    rep_rows: List[Dict[str, object]] = []
    total = len(loads) * len(scenarios) * len(policies) * args.replications
    done = 0
    for li, load in enumerate(loads):
        case = CaseSpec(
            "C1_buffer_robustness",
            f"rho_{load:.2f}",
            f"high-load robustness rho={load:.2f}",
            10,
            6,
            topology="random_sparse",
            density=0.38,
            map_profile="medium_burst",
            service_profile="mild",
            switch_profile="moderate",
            load=load,
        )
        base_problem = make_problem(case, args.seed + 1009 * (li + 1))
        for scenario, cap, eval_steps in scenarios:
            problem = copy.deepcopy(base_problem)
            problem.buffer_cap = cap
            for replication in range(args.replications):
                seed = args.seed + 100000 * (li + 1) + replication
                for policy in policies:
                    row: Dict[str, object] = {
                        "family": case.family,
                        "case_id": case.case_id,
                        "load": load,
                        "scenario": scenario,
                        "buffer_cap": cap,
                        "eval_steps": eval_steps,
                        "policy": policy,
                        "policy_label": POLICY_LABELS.get(policy, policy),
                        "replication": replication,
                        "seed": seed,
                    }
                    row.update(simulate_diagnostic(problem, policy, seed, args.warmup_steps, eval_steps))
                    rep_rows.append(row)
                    done += 1
                    render_progress(done, total, f"{case.case_id} {scenario} {policy}")

    metrics = [
        "mean_total_queue", "p95_total_queue", "cap_time_fraction", "cap_queue_fraction",
        "mean_dropped_arrivals", "drop_rate", "queue_growth", "queue_growth_ratio", "queue_time_slope",
    ]
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rep_rows:
        key = (row["load"], row["scenario"], row["buffer_cap"], row["eval_steps"], row["policy"], row["policy_label"])
        groups.setdefault(key, []).append(row)
    summary: List[Dict[str, object]] = []
    for key, group in sorted(groups.items()):
        row = dict(zip(["load", "scenario", "buffer_cap", "eval_steps", "policy", "policy_label"], key))
        for metric in metrics:
            stats = mean_ci(float(x[metric]) for x in group)
            row.update({f"{metric}_{k}": v for k, v in stats.items()})
        summary.append(row)

    write_csv(os.path.join(args.output_dir, "paper1_buffer_robustness_replications.csv"), rep_rows)
    write_csv(os.path.join(args.output_dir, "paper1_buffer_robustness_summary.csv"), summary)
    plot_metric(summary, "drop_rate", "Dropped-arrival rate", os.path.join(args.output_dir, "fig_buffer_robustness_drop_rate.pdf"))
    plot_metric(summary, "cap_time_fraction", "Fraction of time any queue is capped", os.path.join(args.output_dir, "fig_buffer_robustness_cap_time.pdf"))
    plot_metric(summary, "queue_growth_ratio", "Second-half queue growth ratio", os.path.join(args.output_dir, "fig_buffer_robustness_queue_growth.pdf"))
    plot_diagnostic_heatmap(summary, os.path.join(args.output_dir, "fig_buffer_robustness_diagnostics.pdf"))
    print(f"Buffer robustness results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
