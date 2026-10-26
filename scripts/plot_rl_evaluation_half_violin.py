import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


POLICY_INPUT_CSV = (
    "data/raw/final_run/policy/"
    "paper1_policy_replications.csv"
)
RL_INPUT_CSV = (
    "data/raw/final_run/rl/"
    "paper1_rl_policy_replications.csv"
)
OUTPUT_BASE = "figs/fig_rl_evaluation_half_violin_box"

FAMILY_ORDER = [
    "A1_topology",
    "A2_map_burst",
    "A3_density",
    "A4_service",
    "A5_switch",
    "A6_scale",
    "A7_load",
    "A8_large_scale",
]

FAMILY_LABELS = {
    "A1_topology": "A1 topology",
    "A2_map_burst": "A2 MAP burst",
    "A3_density": "A3 density",
    "A4_service": "A4 service",
    "A5_switch": "A5 switch",
    "A6_scale": "A6 scale",
    "A7_load": "A7 load",
    "A8_large_scale": "A8 large scale",
}

POLICY_ORDER = [
    "gpah",
    "gpah_meanmap",
    "qah_switch",
    "rl_queueing",
    "backpressure",
    "maxweight_mu",
    "cmu",
    "affinity",
    "equal_split",
    "dqn",
    "ppo",
    "sac",
]

POLICY_LABELS = {
    "gpah": "gPAH",
    "gpah_meanmap": "gPAH-StationaryMean",
    "qah_switch": "QAH+Switch",
    "rl_queueing": "Queueing-Surrogate",
    "backpressure": "BackPressure",
    "maxweight_mu": "MaxWeight-mu",
    "cmu": "c-mu",
    "affinity": "GB-PANDAS",
    "equal_split": "Equal-Split",
    "dqn": "DQN",
    "ppo": "PPO",
    "sac": "Discrete-SAC",
}

COLORS = {
    "gpah": "#e85c55",
    "gpah_meanmap": "#f09b63",
    "qah_switch": "#f2a444",
    "rl_queueing": "#eeb6d4",
    "backpressure": "#86add5",
    "maxweight_mu": "#78a5d1",
    "cmu": "#49a7df",
    "affinity": "#b2cf35",
    "equal_split": "#7c7979",
    "dqn": "#c8c2df",
    "ppo": "#e9acd1",
    "sac": "#4aa3d8",
}

TRAINED_RL_POLICIES = {"dqn", "ppo", "sac"}


def draw_panel(ax, df: pd.DataFrame, family: str, rng: np.random.Generator) -> None:
    family_df = df[df["family"] == family]
    data = {
        policy: family_df.loc[family_df["policy"] == policy, "objective"].dropna().to_numpy()
        for policy in POLICY_ORDER
    }
    positions = np.arange(1, len(POLICY_ORDER) + 1)

    for center, policy in zip(positions, POLICY_ORDER):
        values = data[policy]
        if len(values) == 0:
            continue
        color = COLORS[policy]
        violin = ax.violinplot(
            [values],
            positions=[center],
            widths=0.58,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        body = violin["bodies"][0]
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_linewidth(0.8)
        body.set_alpha(0.34)
        for path in body.get_paths():
            vertices = path.vertices
            vertices[:, 0] = np.maximum(vertices[:, 0], center)

        x = rng.uniform(center - 0.31, center - 0.09, size=len(values))
        ax.scatter(
            x,
            values,
            s=4.6,
            color=color,
            alpha=0.54,
            linewidths=0,
            zorder=2,
        )

        box = ax.boxplot(
            [values],
            positions=[center + 0.02],
            widths=0.11,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            zorder=4,
        )
        for patch in box["boxes"]:
            patch.set(facecolor="white", edgecolor="black", linewidth=1.0)
        for key in ["whiskers", "caps", "medians"]:
            for artist in box[key]:
                artist.set(color="black", linewidth=1.0)

    gpah_median = family_df.loc[family_df["policy"] == "gpah", "objective"].median()
    ax.axhline(
        gpah_median,
        color="#333333",
        linestyle=(0, (4, 3)),
        linewidth=0.8,
        alpha=0.52,
    )

    ax.text(
        0.02,
        0.94,
        FAMILY_LABELS[family],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([POLICY_LABELS[p] for p in POLICY_ORDER], rotation=38, ha="right")
    ax.set_xlim(0.3, len(POLICY_ORDER) + 0.75)
    ax.set_ylim(0, 4.35)
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.6, alpha=0.75)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
