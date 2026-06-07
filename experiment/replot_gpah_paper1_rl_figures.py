#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from map_gpah_paper1_common import PAPER_COLORS, POLICY_COLORS, order_policies_for_legend


RL_LABELS = {
    "affinity": "GB-PANDAS-style",
    "backpressure": "BackPressure",
    "cmu": "c-mu",
    "dqn": "DQN-Candidate",
    "gpah": "gPAH",
    "gpah_meanmap": "gPAH-MeanMAP",
    "maxweight_mu": "MaxWeight-mu",
    "ppo": "PPO-Factorized",
    "qah_switch": "QAH+Switch",
    "sac": "Discrete-SAC",
}

RL_COLORS = {
    **POLICY_COLORS,
    "dqn": "#C7C1DE",
    "ppo": "#EEB6D4",
    "sac": "#52AADC",
    "affinity": "#B5CE4E",
}


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def label(policy: str) -> str:
    return RL_LABELS.get(policy, policy)


def require_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_family_log_bars(rows: Sequence[Dict[str, str]], output_dir: str) -> None:
    plt = require_matplotlib()
    families = sorted({r["family"] for r in rows})
    policies = order_policies_for_legend(sorted({r["policy"] for r in rows}))
    data = {(r["family"], r["policy"]): float(r["mean_total_queue_mean"]) for r in rows}
    x = np.arange(len(families))
    width = 0.82 / max(1, len(policies))
    fig, ax = plt.subplots(figsize=(max(8, len(families) * 1.05), 4.8))
    for j, pol in enumerate(policies):
        vals = [max(1e-6, data.get((fam, pol), np.nan)) for fam in families]
        ax.bar(
            x + (j - (len(policies) - 1) / 2) * width,
            vals,
            width,
            label=label(pol),
            color=RL_COLORS.get(pol, PAPER_COLORS["soft_gray"]),
            edgecolor="none",
            linewidth=0,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Mean total queue (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=25, ha="right")
    ax.set_title("RL baselines and structural policies across experiment families")
    ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.7, which="both")
    ax.legend(fontsize=7, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    fig.tight_layout(rect=[0.0, 0.18, 1.0, 1.0])
    fig.savefig(os.path.join(output_dir, "fig_rl_family_log_bars.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_relative_objective_heatmap(rows: Sequence[Dict[str, str]], output_dir: str) -> None:
    plt = require_matplotlib()
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cases = sorted({(r["family"], r["case_id"]) for r in rows})
    policies = [p for p in sorted({r["policy"] for r in rows}) if p != "gpah"]
    values = {(r["family"], r["case_id"], r["policy"]): float(r["objective_mean"]) for r in rows}
    mat = np.full((len(cases), len(policies)), np.nan)
    for i, (fam, cid) in enumerate(cases):
        base = values.get((fam, cid, "gpah"), np.nan)
        for j, pol in enumerate(policies):
            val = values.get((fam, cid, pol), np.nan)
            if np.isfinite(base) and base > 0 and np.isfinite(val):
                mat[i, j] = val / base - 1.0
    fig, ax = plt.subplots(figsize=(max(8, len(policies) * 0.9), max(5.5, len(cases) * 0.23)))
    cmap = LinearSegmentedColormap.from_list("rl_rel", [PAPER_COLORS["c02"], "#ffffff", PAPER_COLORS["c16"]])
    finite = mat[np.isfinite(mat)]
    norm = None
    if finite.size:
        span = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))), 0.05)
        norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(policies)))
    ax.set_xticklabels([label(p) for p in policies], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(cases)))
    ax.set_yticklabels([f"{fam}/{cid}" for fam, cid in cases], fontsize=7)
    ax.set_xticks(np.arange(len(policies) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(cases) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#d9d9d9", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("Relative objective gap versus gPAH")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Case")
    fig.colorbar(im, ax=ax, label="policy/gPAH - 1")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_rl_relative_objective_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_family_radar(rows: Sequence[Dict[str, str]], output_dir: str) -> None:
    plt = require_matplotlib()
    families = sorted({r["family"] for r in rows})
    policies = order_policies_for_legend(sorted({r["policy"] for r in rows}))
    values = {(r["family"], r["policy"]): float(r["objective_mean"]) for r in rows}
    best = {fam: min(values[(fam, p)] for p in policies if (fam, p) in values) for fam in families}
    angles = np.linspace(0, 2 * np.pi, len(families), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.8, 6.2), subplot_kw={"projection": "polar"})
    for pol in policies:
        vals = [min(1.0, best[fam] / max(values.get((fam, pol), np.nan), 1e-9)) for fam in families]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.25, label=label(pol), color=RL_COLORS.get(pol, PAPER_COLORS["soft_gray"]))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(families, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "best"], fontsize=7)
    ax.set_title("Family-level normalized objective performance", pad=18)
    ax.grid(color=PAPER_COLORS["grid"], alpha=0.8)
    ax.legend(fontsize=7, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_rl_family_radar.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(rows: Sequence[Dict[str, str]], output_dir: str) -> None:
    plt = require_matplotlib()
    algorithms = sorted({r["algorithm"] for r in rows})
    if not algorithms:
        return
    metrics_by_algo = {
        "dqn": ["train_reward_mean_window", "loss"],
        "ppo": ["train_reward_mean_window", "loss"],
        "sac": ["train_reward_mean_window", "actor_loss", "q_loss"],
    }
    panels = [(algo, metric) for algo in algorithms for metric in metrics_by_algo.get(algo, ["train_reward_mean_window"])]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.2, max(3.2, 2.1 * len(panels))), sharex=False)
    if len(panels) == 1:
        axes = [axes]
    for ax, (algo, metric) in zip(axes, panels):
        grouped: Dict[int, List[float]] = {}
        for r in rows:
            if r["algorithm"] != algo or not r.get(metric):
                continue
            grouped.setdefault(int(float(r["step"])), []).append(float(r[metric]))
        if grouped:
            xs = sorted(grouped)
            ys = [float(np.mean(grouped[x])) for x in xs]
            ax.plot(xs, ys, color=RL_COLORS.get(algo, PAPER_COLORS["highlight"]), linewidth=1.4)
        ax.set_title(f"{label(algo)} training {metric}")
        ax.set_xlabel("Training step")
        ax.set_ylabel(metric)
        ax.grid(color=PAPER_COLORS["grid"], alpha=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_rl_training_curves.pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot Paper 1 RL figures from existing CSV files.")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    family_rows = read_csv_rows(os.path.join(args.output_dir, "paper1_rl_policy_family_summary.csv"))
    case_rows = read_csv_rows(os.path.join(args.output_dir, "paper1_rl_policy_case_summary.csv"))
    history_rows = read_csv_rows(os.path.join(args.output_dir, "paper1_rl_training_history.csv"))
    plot_family_log_bars(family_rows, args.output_dir)
    plot_relative_objective_heatmap(case_rows, args.output_dir)
    plot_family_radar(family_rows, args.output_dir)
    plot_training_curves(history_rows, args.output_dir)
    print(f"RL figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
