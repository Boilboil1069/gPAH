#!/usr/bin/env python3
"""Regenerate the revised manuscript figures with publication-scale typography."""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import re
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, TwoSlopeNorm


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
FIGS = ROOT / "results/replotted"
EXPERIMENT_CODE = ROOT / "experiment"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(EXPERIMENT_CODE))

RUN = ROOT / "data/raw/final_run"
POLICY_DIR = RUN / "policy"
RL_DIR = RUN / "rl"
BUFFER_DIR = RUN / "buffer_robustness"
SENSITIVITY_DIR = RUN / "sensitivity"
TOPOLOGY_DIR = RUN / "rl_topology_generalization"
ABLATION_DIR = ROOT / "data/raw/ablation_r50_conditional"

POLICY_LABELS = {
    "equal_split": "Equal-Split",
    "affinity": "GB-PANDAS-style",
    "backpressure": "BackPressure",
    "cmu": "c-mu",
    "maxweight_mu": "MaxWeight-mu",
    "rl_queueing": "Queueing-Surrogate",
    "dqn": "DQN-Candidate",
    "gpah_meanmap": "gPAH-StationaryMean",
    "ppo": "PPO-Factorized",
    "qah_switch": "QAH+Switch",
    "sac": "Discrete-SAC-Factorized",
    "gpah": "gPAH",
}

COLORS = {
    "equal_split": "#7c7979",
    "affinity": "#b2cf35",
    "backpressure": "#86add5",
    "cmu": "#49a7df",
    "maxweight_mu": "#78a5d1",
    "rl_queueing": "#eeb6d4",
    "dqn": "#c8c2df",
    "gpah_meanmap": "#f09b63",
    "ppo": "#e9acd1",
    "qah_switch": "#f2a444",
    "sac": "#4aa3d8",
    "gpah": "#e85c55",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.2,
            "lines.linewidth": 1.6,
            "lines.markersize": 5.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIGS / filename, bbox_inches="tight")
    plt.close(fig)


def regenerate_standard_policy_figures() -> None:
    paper_path = SCRIPTS / "plot_paper_results_no_titles.py"
    paper = types.ModuleType("revision_plot_paper_results")
    paper.__file__ = str(paper_path)
    paper_source = "from __future__ import annotations\n" + paper_path.read_text(encoding="utf-8")
    exec(compile(paper_source, str(paper_path), "exec"), paper.__dict__)
    paper.POLICY_LABELS.update(POLICY_LABELS)
    setup_style()
    policy_case = pd.read_csv(POLICY_DIR / "paper1_policy_case_summary.csv")
    rl_case = pd.read_csv(RL_DIR / "paper1_rl_policy_case_summary.csv")
    rl_family = pd.read_csv(RL_DIR / "paper1_rl_policy_family_summary.csv")
    policy_family = pd.read_csv(POLICY_DIR / "paper1_policy_family_summary.csv")

    for family, filename in (
        ("A1_topology", "fig4.pdf"),
        ("A2_map_burst", "fig5.pdf"),
        ("A6_scale", "fig6.pdf"),
        ("A5_switch", "fig9.pdf"),
        ("A7_load", "fig7.pdf"),
        ("A8_large_scale", "fig8.pdf"),
    ):
        paper.grouped_log_bars(policy_case, family, FIGS / filename)

    paper.rl_family_log_bars(rl_family, FIGS / "fig11.pdf")
    paper.relative_objective_heatmap(
        paper.combined_rl_case_comparison(policy_case, rl_case),
        FIGS / "fig15.pdf",
    )
    paper.policy_rl_complete_radar(
        policy_family,
        rl_family,
        FIGS / "fig16.pdf",
    )


def regenerate_ablation() -> None:
    df = pd.read_csv(ABLATION_DIR / "paper1_ablation_paired_gap_vs_full_gpah.csv")
    family_order = ["B1_phase", "B2_service", "B3_scarcity", "B4_switch", "B5_mixed"]
    policy_order = ["mean_map", "no_phase", "no_queue", "no_scarcity", "no_service", "no_switch"]
    labels = {
        "mean_map": "Abl-StationaryMean",
        "no_phase": "Abl-NoPhase",
        "no_queue": "Abl-NoQueue",
        "no_scarcity": "Abl-NoScarcity",
        "no_service": "Abl-NoService",
        "no_switch": "Abl-NoSwitch",
    }
    grouped = (
        df[df["policy"].isin(policy_order)]
        .groupby(["family", "policy"], as_index=False)["paired_objective_gap_mean"]
        .mean()
    )
    matrix = np.full((len(family_order), len(policy_order)), np.nan)
    for i, family in enumerate(family_order):
        for j, policy in enumerate(policy_order):
            hit = grouped[(grouped["family"] == family) & (grouped["policy"] == policy)]
            if not hit.empty:
                matrix[i, j] = hit["paired_objective_gap_mean"].iloc[0]
    limit = max(float(np.nanmax(np.abs(matrix))), 0.05)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    im = ax.imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    ax.set_xticks(np.arange(len(policy_order)))
    ax.set_xticklabels([labels[p] for p in policy_order], rotation=32, ha="right")
    ax.set_yticks(np.arange(len(family_order)))
    ax.set_yticklabels(family_order)
    ax.set_xlabel("Ablation / policy")
    ax.set_ylabel("Case family")
    ax.set_xticks(np.arange(-0.5, len(policy_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(family_order), 1), minor=True)
    ax.grid(which="minor", color="#d8d8d8", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Paired objective gap vs full gPAH")
    save(fig, "fig12.pdf")


def regenerate_buffer_diagnostics() -> None:
    rows = pd.read_csv(BUFFER_DIR / "paper1_buffer_robustness_summary.csv")
    policies = [
        p
        for p in [
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
        if p in set(rows["policy"])
    ]
    conditions = [
        (0.90, "B2000_T3000", "2/3"),
        (0.90, "B10000_T3000", "10/3"),
        (0.90, "B10000_T30000", "10/30"),
        (0.95, "B2000_T3000", "2/3"),
        (0.95, "B10000_T3000", "10/3"),
        (0.95, "B10000_T30000", "10/30"),
    ]
    metrics = [
        ("drop_rate_mean", "Dropped arrivals (%)", 100.0, 45.0, "risk"),
        ("cap_time_fraction_mean", "Time with any capped queue (%)", 100.0, 100.0, "risk"),
        ("queue_time_slope_mean", "Queue-time slope (jobs/step)", 1.0, 11.0, "trend"),
    ]
    lookup = {(float(r.load), str(r.scenario), str(r.policy)): r for r in rows.itertuples()}
    risk_cmap = LinearSegmentedColormap.from_list("buffer_risk", ["#edf4f8", "#f5d46f", "#ed8a52", "#c64b45"])
    trend_cmap = LinearSegmentedColormap.from_list("buffer_trend", ["#edf4f8", "#95c6c8", "#3f8e9b", "#31546f"])
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 5.2), constrained_layout=True)
    for panel, (ax, (metric, label, multiplier, vmax, kind)) in enumerate(zip(axes, metrics)):
        matrix = np.full((len(policies), len(conditions)), np.nan)
        for i, policy in enumerate(policies):
            for j, (load, scenario, _) in enumerate(conditions):
                row = lookup.get((load, scenario, policy))
                if row is not None:
                    matrix[i, j] = float(getattr(row, metric)) * multiplier
        cmap = risk_cmap if kind == "risk" else trend_cmap
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax))
        ax.set_xticks(np.arange(len(conditions)))
        ax.set_xticklabels([c[2] for c in conditions], fontsize=8.5)
        ax.set_yticks(np.arange(len(policies)))
        ax.set_yticklabels([POLICY_LABELS[p] for p in policies] if panel == 0 else [])
        ax.set_xlabel(label + "\n" + r"$B/T$ (thousands)", labelpad=8, linespacing=1.2)
        ax.text(1.0, 1.025, r"$\rho=.90$", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9.2, fontweight="semibold", clip_on=False)
        ax.text(4.0, 1.025, r"$\rho=.95$", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9.2, fontweight="semibold", clip_on=False)
        ax.axvline(2.5, color="#ffffff", linewidth=2.2)
        ax.set_xticks(np.arange(len(conditions) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(policies) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="#ffffff", linewidth=1.1)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if np.isfinite(value):
                    text = f"{value:.1f}" if value >= 0.095 else "0"
                    color = "white" if value > 0.52 * vmax else "#202020"
                    ax.text(j, i, text, ha="center", va="center", fontsize=10, color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.ax.tick_params(labelsize=9)
    save(fig, "fig1.pdf")


def regenerate_training_curves() -> None:
    rows = pd.read_csv(RL_DIR / "paper1_rl_training_history.csv")
    panels = [
        ("dqn", "train_reward_mean_window", "DQN-Candidate: reward"),
        ("dqn", "loss", "DQN-Candidate: loss"),
        ("ppo", "train_reward_mean_window", "PPO-Factorized: reward"),
        ("ppo", "loss", "PPO-Factorized: loss"),
        ("sac", "train_reward_mean_window", "Discrete-SAC-Factorized: reward"),
        ("sac", "actor_loss", "Discrete-SAC-Factorized: actor loss"),
        ("sac", "q_loss", "Discrete-SAC-Factorized: Q loss"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(10.6, 7.7), constrained_layout=True)
    axes = axes.ravel()
    for ax, (algorithm, metric, title) in zip(axes, panels):
        subset = rows[(rows["algorithm"] == algorithm) & rows[metric].notna()]
        grouped = subset.groupby("step", as_index=False)[metric].mean().sort_values("step")
        ax.plot(grouped["step"], grouped[metric], color=COLORS[algorithm], linewidth=1.9)
        ax.set_title(title, pad=4)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Reward" if "reward" in metric else "Loss")
        ax.grid(color="#c7c1de", alpha=0.72, linewidth=0.8)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[len(panels):]:
        ax.axis("off")
    save(fig, "fig10.pdf")


def regenerate_unseen_topology() -> None:
    case = pd.read_csv(TOPOLOGY_DIR / "paper1_rl_unseen_topology_case_summary.csv")
    topologies = list(case["case_id"].drop_duplicates())
    policies = ["gpah", "dqn", "ppo", "sac"]
    x = np.arange(len(topologies))
    width = 0.78 / len(policies)
    fig, ax = plt.subplots(figsize=(7.2, 3.3), constrained_layout=True)
    for j, policy in enumerate(policies):
        values, errors = [], []
        for topology in topologies:
            row = case[(case["case_id"] == topology) & (case["policy"] == policy)].iloc[0]
            values.append(row["objective_mean"])
            errors.append(row["objective_ci95"])
        ax.bar(
            x + (j - (len(policies) - 1) / 2) * width,
            values,
            width,
            yerr=errors,
            capsize=2.5,
            label=POLICY_LABELS[policy],
            color=COLORS[policy],
            edgecolor="none",
            error_kw={"elinewidth": 1.0, "ecolor": "#555555"},
        )
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", " ").title() for t in topologies])
    ax.set_ylabel("Standardized objective")
    ax.grid(axis="y", color="#c7c1de", alpha=0.65, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save(fig, "fig14.pdf")


def regenerate_half_violin() -> None:
    violin = importlib.import_module("plot_rl_evaluation_half_violin")
    violin.POLICY_LABELS.update(POLICY_LABELS)
    policy_df = pd.read_csv(ROOT / violin.POLICY_INPUT_CSV)
    policy_df = policy_df[
        policy_df["policy"].isin([p for p in violin.POLICY_ORDER if p not in violin.TRAINED_RL_POLICIES])
    ]
    rl_df = pd.read_csv(ROOT / violin.RL_INPUT_CSV)
    rl_df = rl_df[rl_df["policy"].isin(violin.TRAINED_RL_POLICIES)]
    data = pd.concat([policy_df, rl_df], ignore_index=True)
    data = data[data["policy"].isin(violin.POLICY_ORDER) & data["family"].isin(violin.FAMILY_ORDER)].copy()
    rng = np.random.default_rng(20260601)
    fig, axes = plt.subplots(4, 2, figsize=(10.8, 10.0), sharey=True, constrained_layout=True)
    axes = axes.ravel()
    for ax, family in zip(axes, violin.FAMILY_ORDER):
        violin.draw_panel(ax, data, family, rng)
        for label in ax.get_xticklabels():
            label.set_fontsize(8.6)
        for text in ax.texts:
            text.set_fontsize(10.5)
    for ax in axes[::2]:
        ax.set_ylabel("Standardized objective")
    for ax in axes[-2:]:
        ax.set_xlabel("Policy")
    save(fig, "fig17.pdf")


def regenerate_sensitivity() -> None:
    summary = pd.read_csv(SENSITIVITY_DIR / "paper1_sensitivity_parameter_summary.csv")
    parameter_order = ["alpha_Q", "alpha_A", "alpha_mu", "alpha_S", "alpha_C", "alpha_L"]
    labels = {
        "alpha_Q": "Q queue",
        "alpha_A": "A phase",
        "alpha_mu": "mu service",
        "alpha_S": "S scarcity",
        "alpha_C": "C switch cost",
        "alpha_L": "L switch lag",
    }
    colors = dict(zip(parameter_order, ["#e85c55", "#f09b63", "#49a7df", "#b2cf35", "#71669e", "#4aa3d8"]))

    deletion = summary[summary["alpha_multiplier"] == 0].set_index("alpha_parameter").loc[parameter_order]
    y = np.arange(len(parameter_order))
    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.barh(
        y,
        deletion["objective_gap_mean_across_cases"],
        xerr=deletion["objective_gap_ci95_across_cases"],
        color=[colors[p] for p in parameter_order],
        edgecolor="none",
        error_kw={"elinewidth": 1.1, "ecolor": "#555555", "capsize": 2.5},
    )
    ax.set_yticks(y)
    ax.set_yticklabels([labels[p] for p in parameter_order])
    ax.invert_yaxis()
    ax.set_xlabel("Objective gap")
    ax.axvline(0, color="#666666", linewidth=1.0)
    ax.grid(axis="x", color="#c7c1de", alpha=0.65, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig13a.pdf")

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    for parameter in parameter_order:
        sub = summary[(summary["alpha_parameter"] == parameter) & (summary["alpha_multiplier"] > 0)].sort_values("alpha_multiplier")
        ax.plot(
            sub["alpha_multiplier"],
            sub["objective_gap_mean_across_cases"],
            marker="o",
            linewidth=1.9,
            label=labels[parameter],
            color=colors[parameter],
        )
    ax.axhline(0, color="#666666", linewidth=1.0)
    ax.axvline(1, color="#999999", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Weight multiplier")
    ax.set_ylabel("Objective gap vs default gPAH")
    ax.set_xticks([0.5, 1.0, 1.5, 2.0])
    ax.grid(color="#c7c1de", alpha=0.65, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    save(fig, "fig13b.pdf")


def regenerate_compatibility_topologies() -> None:
    source = (SCRIPTS / "plot_compat_topologies.py").read_text(encoding="utf-8")
    source = source.replace("'font.size':10", "'font.size':12.5")
    source = source.replace("figsize=(14, 8.4)", "figsize=(10.2, 6.2)")
    source = source.replace("'fig_compat_topologies.pdf'", "'fig3.pdf'")
    namespace = {
        "__file__": str(FIGS / "_compatibility_generator.py"),
        "__name__": "revision_compatibility_generator",
        "OUTPUT_DIR": FIGS,
    }
    exec(compile(source, str(SCRIPTS / "plot_compat_topologies.py"), "exec"), namespace)
    for helper in (
        FIGS / "fig_compat_topologies_preview.png",
        FIGS / "fig_compat_topologies_editable.svg",
    ):
        helper.unlink(missing_ok=True)


def validate_outputs() -> None:
    expected = {
        "fig1.pdf",
        "fig2.pdf",
        "fig3.pdf",
        "fig4.pdf",
        "fig5.pdf",
        "fig6.pdf",
        "fig7.pdf",
        "fig8.pdf",
        "fig9.pdf",
        "fig10.pdf",
        "fig11.pdf",
        "fig12.pdf",
        "fig13a.pdf",
        "fig13b.pdf",
        "fig14.pdf",
        "fig15.pdf",
        "fig16.pdf",
        "fig17.pdf",
    }
    missing = sorted(name for name in expected if not (FIGS / name).exists())
    if missing:
        raise RuntimeError("Missing regenerated figures: " + ", ".join(missing))

    try:
        from pypdf import PdfReader
    except ImportError:
        PdfReader = None
    if PdfReader is not None:
        legacy = re.compile(r"gPAH-MeanMAP|Abl-MeanMAP|(?<!gPAH-)MeanMAP")
        bad = []
        for name in sorted(expected):
            text = "\n".join(page.extract_text() or "" for page in PdfReader(FIGS / name).pages)
            if legacy.search(text):
                bad.append(name)
        if bad:
            raise RuntimeError("Legacy policy labels remain in: " + ", ".join(bad))

    print(f"Regenerated and validated {len(expected)} result figures.")


def regenerate_infinite_buffer() -> None:
    from map_gpah_paper1_infinite_buffer import plot_trajectories
    source = ROOT / "data/raw/revision_experiments/infinite_buffer/paper1_infinite_buffer_trace_summary.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Infinite-buffer trace summary is empty")
    # This diagnostic was originally plotted in a standalone process with
    # Matplotlib defaults; isolate it from the other figures' typography.
    with plt.rc_context():
        plt.rcdefaults()
        plot_trajectories(rows, str(FIGS / "fig2.pdf"))


def main() -> None:
    global FIGS
    from reproduction_paths import output_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/replotted")
    args = parser.parse_args()
    FIGS = output_path(args.output_dir)
    FIGS.mkdir(parents=True, exist_ok=True)
    setup_style()
    regenerate_standard_policy_figures()
    regenerate_ablation()
    regenerate_buffer_diagnostics()
    regenerate_infinite_buffer()
    regenerate_training_curves()
    regenerate_unseen_topology()
    regenerate_half_violin()
    regenerate_sensitivity()
    regenerate_compatibility_topologies()
    validate_outputs()


if __name__ == "__main__":
    main()
