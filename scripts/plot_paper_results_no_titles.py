from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "data/raw/final_run/policy"
RL_DIR = ROOT / "data/raw/final_run/rl"
ABLATION_DIR = ROOT / "data/raw/ablation_r50_conditional"

POLICY_FAMILY_ORDER = [
    "A1_topology",
    "A2_map_burst",
    "A3_density",
    "A4_service",
    "A5_switch",
    "A6_scale",
    "A7_load",
    "A8_large_scale",
]

RL_FAMILY_ORDER = [
    "A1_topology",
    "A2_map_burst",
    "A3_density",
    "A4_service",
    "A5_switch",
    "A6_scale",
    "A7_load",
    "A8_large_scale",
]

CASE_ORDER = {
    "A1_topology": [
        "bridge_bottleneck",
        "chain",
        "disjoint",
        "high_overlap",
        "random_sparse",
        "ring",
        "star",
    ],
    "A2_map_burst": [
        "alternating_hotspots",
        "asymmetric",
        "fast_mixing",
        "high_burst",
        "low_burst",
        "low_contrast",
        "medium_burst",
        "multi_state_burst",
        "predictive_burst",
    ],
    "A3_density": ["very_sparse", "sparse", "medium"],
    "A4_service": ["homogeneous", "mild", "strong"],
    "A5_switch": ["high", "lag", "moderate", "none"],
    "A6_scale": ["d12_K4", "d12_K6", "d6_K4", "d6_K6", "d8_K4", "d8_K6"],
    "A7_load": ["rho_0_6", "rho_0_75", "rho_0_9", "rho_0_95"],
    "A8_large_scale": ["d20_K10", "d30_K12", "d40_K16", "d50_K20", "d75_K30", "d100_K40"],
}

STRUCTURED_POLICY_ORDER = [
    "equal_split",
    "affinity",
    "backpressure",
    "cmu",
    "maxweight_mu",
    "rl_queueing",
    "gpah_meanmap",
    "qah_switch",
    "gpah",
]

RL_POLICY_ORDER = [
    "affinity",
    "backpressure",
    "cmu",
    "maxweight_mu",
    "dqn",
    "gpah_meanmap",
    "ppo",
    "qah_switch",
    "sac",
    "gpah",
]

RL_FOCUSED_POLICY_ORDER = [
    "dqn",
    "ppo",
    "sac",
    "gpah",
]

RL_HEATMAP_POLICY_ORDER = [
    "equal_split",
    "affinity",
    "backpressure",
    "cmu",
    "maxweight_mu",
    "rl_queueing",
    "dqn",
    "gpah_meanmap",
    "ppo",
    "qah_switch",
    "sac",
    "gpah",
]

COMBINED_POLICY_RL_ORDER = [
    "equal_split",
    "affinity",
    "backpressure",
    "cmu",
    "maxweight_mu",
    "rl_queueing",
    "dqn",
    "gpah_meanmap",
    "ppo",
    "qah_switch",
    "sac",
    "gpah",
]

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
    "sac": "Discrete-SAC",
    "gpah": "gPAH",
}

POLICY_FAMILY_LABELS = {
    "A1_topology": "A1\nTopology",
    "A2_map_burst": "A2\nMAP",
    "A3_density": "A3\nDensity",
    "A4_service": "A4\nService",
    "A5_switch": "A5\nSwitch",
    "A6_scale": "A6\nScale",
    "A7_load": "A7\nLoad",
    "A8_large_scale": "A8\nLarge scale",
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

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def grouped_log_bars(
    df: pd.DataFrame,
    family: str,
    out_path: Path,
    policy_order: list[str] | None = None,
) -> None:
    cases = CASE_ORDER[family]
    policy_order = policy_order or STRUCTURED_POLICY_ORDER
    policy_order = [p for p in policy_order if p in set(df.loc[df["family"] == family, "policy"])]
    fig, ax = plt.subplots(figsize=(9.2, 3.85))
    x = np.arange(len(cases))
    width = min(0.085, 0.78 / max(1, len(policy_order)))
    offsets = (np.arange(len(policy_order)) - (len(policy_order) - 1) / 2) * width

    for offset, policy in zip(offsets, policy_order):
        sub = df[(df["family"] == family) & (df["policy"] == policy)]
        values = [
            sub.loc[sub["case_id"] == case, "mean_total_queue_mean"].iloc[0]
            if not sub.loc[sub["case_id"] == case].empty
            else np.nan
            for case in cases
        ]
        ax.bar(
            x + offset,
            values,
            width=width,
            label=POLICY_LABELS.get(policy, policy),
            color=COLORS.get(policy, "#c8c2df"),
            edgecolor="none",
            alpha=0.95,
        )

    ax.set_yscale("log")
    ax.set_ylabel("Mean total queue (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([case.replace("_", " ") for case in cases], rotation=28, ha="right")
    ax.tick_params(axis="x", pad=4)
    ax.grid(axis="y", which="both", color="#cfc8e8", linewidth=0.8, alpha=0.9)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.40), frameon=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.43)
    save(fig, out_path)


def rl_family_log_bars(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 3.75))
    x = np.arange(len(RL_FAMILY_ORDER))
    width = min(0.16, 0.78 / max(1, len(RL_FOCUSED_POLICY_ORDER)))
    offsets = (np.arange(len(RL_FOCUSED_POLICY_ORDER)) - (len(RL_FOCUSED_POLICY_ORDER) - 1) / 2) * width

    for offset, policy in zip(offsets, RL_FOCUSED_POLICY_ORDER):
        sub = df[df["policy"] == policy]
        values = [
            sub.loc[sub["family"] == family, "mean_total_queue_mean"].iloc[0]
            for family in RL_FAMILY_ORDER
        ]
        ax.bar(
            x + offset,
            values,
            width=width,
            label=POLICY_LABELS[policy],
            color=COLORS[policy],
            edgecolor="none",
            alpha=0.95,
        )

    ax.set_yscale("log")
    ax.set_ylabel("Mean total queue (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(RL_FAMILY_ORDER, rotation=28, ha="right")
    ax.tick_params(axis="x", pad=4)
    ax.grid(axis="y", which="both", color="#cfc8e8", linewidth=0.8, alpha=0.9)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.40), frameon=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.43)
    save(fig, out_path)


def relative_objective_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    keys = ["family", "case_id"]
    base = df[df["policy"] == "gpah"][keys + ["objective_mean"]].rename(
        columns={"objective_mean": "gpah_objective"}
    )
    rel = df.merge(base, on=keys)
    rel = rel[rel["policy"] != "gpah"].copy()
    rel["relative_gap"] = rel["objective_mean"] / rel["gpah_objective"] - 1.0

    rows = []
    for family in RL_FAMILY_ORDER:
        for case in CASE_ORDER.get(family, sorted(rel.loc[rel["family"] == family, "case_id"].unique())):
            if not rel[(rel["family"] == family) & (rel["case_id"] == case)].empty:
                rows.append((family, case, f"{family}/{case}"))

    cols = [p for p in RL_HEATMAP_POLICY_ORDER if p != "gpah" and p in set(rel["policy"])]
    matrix = np.full((len(rows), len(cols)), np.nan)
    for i, (family, case, _) in enumerate(rows):
        for j, policy in enumerate(cols):
            hit = rel[(rel["family"] == family) & (rel["case_id"] == case) & (rel["policy"] == policy)]
            if not hit.empty:
                matrix[i, j] = hit["relative_gap"].iloc[0]

    limit = np.nanpercentile(np.abs(matrix), 98)
    limit = max(float(limit), 0.25)
    # This figure is printed at roughly half-page width in the manuscript.
    # A narrower source canvas and explicit, larger type keep labels readable
    # after LaTeX scaling without allowing the dense case labels to collide.
    fig, ax = plt.subplots(figsize=(7.35, 9.4), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list(
        "gpah_heatmap_contrast",
        [
            (0.00, "#78a5d1"),
            (0.485, "#a9d4e8"),
            (0.500, "#ffffff"),
            (0.515, "#f2b0aa"),
            (1.00, "#d75b4e"),
        ],
    )
    im = ax.imshow(matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(
        [POLICY_LABELS[p] for p in cols],
        rotation=52,
        ha="right",
        rotation_mode="anchor",
        fontsize=11.5,
    )
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([label for _, _, label in rows], fontsize=9.5)
    ax.set_xlabel("Policy", fontsize=12)
    ax.set_ylabel("Case", fontsize=12)
    ax.tick_params(axis="x", pad=5)
    ax.tick_params(axis="y", pad=2)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="#d8d8d8", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, pad=0.035)
    cbar.set_label("policy/gPAH - 1", fontsize=11.5)
    cbar.ax.tick_params(labelsize=10)
    save(fig, out_path)


def combined_rl_case_comparison(policy_case: pd.DataFrame, rl_case: pd.DataFrame) -> pd.DataFrame:
    keys = ["family", "case_id"]
    rl_keys = rl_case[rl_case["policy"].isin(["dqn", "ppo", "sac"])][keys].drop_duplicates()
    structured = policy_case[policy_case["policy"].isin(STRUCTURED_POLICY_ORDER)].merge(rl_keys, on=keys)
    trained = rl_case[rl_case["policy"].isin(["dqn", "ppo", "sac"])]
    return pd.concat([structured, trained], ignore_index=True, sort=False)


def policy_objective_gap_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    keys = ["family", "case_id"]
    base = df[df["policy"] == "gpah"][keys + ["objective_mean"]].rename(
        columns={"objective_mean": "gpah_objective"}
    )
    rel = df.merge(base, on=keys)
    rel = rel[rel["policy"] != "gpah"].copy()
    rel["relative_gap"] = rel["objective_mean"] / rel["gpah_objective"] - 1.0

    rows = []
    for family in POLICY_FAMILY_ORDER:
        for case in CASE_ORDER.get(family, sorted(rel.loc[rel["family"] == family, "case_id"].unique())):
            rows.append((family, case, f"{family}/{case}"))

    cols = [p for p in STRUCTURED_POLICY_ORDER if p != "gpah" and p in set(rel["policy"])]
    matrix = np.full((len(rows), len(cols)), np.nan)
    for i, (family, case, _) in enumerate(rows):
        for j, policy in enumerate(cols):
            hit = rel[(rel["family"] == family) & (rel["case_id"] == case) & (rel["policy"] == policy)]
            if not hit.empty:
                matrix[i, j] = hit["relative_gap"].iloc[0]

    finite = matrix[np.isfinite(matrix)]
    limit = max(float(np.nanpercentile(np.abs(finite), 95)), 0.10) if finite.size else 0.10
    cmap = LinearSegmentedColormap.from_list(
        "gpah_policy_heatmap_contrast",
        [
            (0.00, "#78a5d1"),
            (0.485, "#a9d4e8"),
            (0.500, "#ffffff"),
            (0.515, "#f2b0aa"),
            (1.00, "#d75b4e"),
        ],
    )
    cmap.set_under("#78a5d1")
    cmap.set_over("#d75b4e")

    fig, ax = plt.subplots(figsize=(8.2, 9.4), constrained_layout=True)
    im = ax.imshow(
        matrix,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([POLICY_LABELS.get(p, p) for p in cols], rotation=42, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([label for _, _, label in rows], fontsize=6.8)
    ax.set_xlabel("Policy")
    ax.set_ylabel("Case")
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="#d8d8d8", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, extend="both")
    cbar.set_label("policy/gPAH - 1")
    save(fig, out_path)


def radar(df: pd.DataFrame, out_path: Path) -> None:
    pivot = df.pivot(index="policy", columns="family", values="objective_mean")
    normalized = pivot.copy()
    for family in RL_FAMILY_ORDER:
        best = pivot[family].min()
        normalized[family] = best / pivot[family]

    angles = np.linspace(0, 2 * np.pi, len(RL_FAMILY_ORDER), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(8.2, 8.8), constrained_layout=False)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.tick_params(axis="x", pad=18)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_rlabel_position(24)
    ax.grid(color="#cbc5e8", linewidth=0.8)

    label_radius = 0.55
    for angle, family in zip(angles[:-1], RL_FAMILY_ORDER):
        x = 0.5 + label_radius * np.sin(angle)
        y = 0.5 + label_radius * np.cos(angle)
        ha = "center"
        va = "center"
        if x < 0.43:
            ha = "right"
        elif x > 0.57:
            ha = "left"
        if y < 0.43:
            va = "top"
        elif y > 0.57:
            va = "bottom"
        ax.text(
            x,
            y,
            POLICY_FAMILY_LABELS.get(family, family.replace("_", " ")),
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=9,
            linespacing=0.95,
            clip_on=False,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.93, "pad": 1.4},
        )

    for policy in RL_POLICY_ORDER:
        values = normalized.loc[policy, RL_FAMILY_ORDER].to_numpy(dtype=float)
        values = np.concatenate([values, values[:1]])
        ax.plot(angles, values, linewidth=1.6, color=COLORS[policy], label=POLICY_LABELS[policy])

    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False)
    fig.subplots_adjust(left=0.16, right=0.84, top=0.88, bottom=0.26)
    save(fig, out_path)


def policy_complete_radar(df: pd.DataFrame, out_path: Path) -> None:
    value_col = "objective_mean_across_cases"
    if value_col not in df.columns:
        value_col = "objective_mean"
    families = [f for f in POLICY_FAMILY_ORDER if f in set(df["family"])]
    policies = [p for p in STRUCTURED_POLICY_ORDER if p in set(df["policy"])]
    pivot = df.pivot(index="policy", columns="family", values=value_col)
    pivot = pivot.reindex(index=policies, columns=families)
    pivot = pivot.dropna(axis=0, how="any")
    normalized = pivot.copy()
    for family in families:
        best = pivot[family].min()
        normalized[family] = best / pivot[family]

    angles = np.linspace(0, 2 * np.pi, len(families), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(9.8, 9.6), constrained_layout=False)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_rlabel_position(18)
    ax.grid(color="#cbc5e8", linewidth=0.8, alpha=0.95)
    ax.spines["polar"].set_color("#b8b1d8")
    ax.spines["polar"].set_linewidth(0.9)

    label_radius = 0.55
    for angle, family in zip(angles, families):
        x = 0.5 + label_radius * np.sin(angle)
        y = 0.5 + label_radius * np.cos(angle)
        ha = "center"
        va = "center"
        if x < 0.43:
            ha = "right"
        elif x > 0.57:
            ha = "left"
        if y < 0.43:
            va = "top"
        elif y > 0.57:
            va = "bottom"
        ax.text(
            x,
            y,
            POLICY_FAMILY_LABELS.get(family, family.replace("_", " ")),
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=9,
            linespacing=0.95,
            clip_on=False,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.93, "pad": 1.4},
        )

    for policy in policies:
        values = normalized.loc[policy, families].to_numpy(dtype=float)
        values = np.concatenate([values, values[:1]])
        ax.plot(
            closed_angles,
            values,
            linewidth=1.45 if policy != "gpah" else 2.0,
            alpha=0.88 if policy != "gpah" else 0.98,
            color=COLORS.get(policy, "#c8c2df"),
            label=POLICY_LABELS.get(policy, policy),
        )

    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=False,
        columnspacing=1.15,
        handlelength=2.2,
    )
    fig.subplots_adjust(left=0.18, right=0.82, top=0.82, bottom=0.25)
    save(fig, out_path)


def policy_rl_complete_radar(policy_df: pd.DataFrame, rl_df: pd.DataFrame, out_path: Path) -> None:
    policy_value_col = "objective_mean_across_cases"
    if policy_value_col not in policy_df.columns:
        policy_value_col = "objective_mean"

    policy_part = policy_df[
        policy_df["policy"].isin([p for p in STRUCTURED_POLICY_ORDER if p != "dqn"])
    ][["family", "policy", policy_value_col]].rename(columns={policy_value_col: "objective"})
    rl_part = rl_df[rl_df["policy"].isin(["dqn", "ppo", "sac"])][
        ["family", "policy", "objective_mean"]
    ].rename(columns={"objective_mean": "objective"})
    combined = pd.concat([policy_part, rl_part], ignore_index=True)

    families = [f for f in POLICY_FAMILY_ORDER if f in set(combined["family"])]
    policies = [p for p in COMBINED_POLICY_RL_ORDER if p in set(combined["policy"])]
    pivot = combined.pivot(index="policy", columns="family", values="objective")
    pivot = pivot.reindex(index=policies, columns=families).dropna(axis=0, how="any")

    normalized = pivot.copy()
    for family in families:
        best = pivot[family].min()
        normalized[family] = best / pivot[family]

    angles = np.linspace(0, 2 * np.pi, len(families), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])

    # Figure 16 is also placed in a half-page panel.  Reducing the source
    # canvas increases its effective print type size; the explicit spacing
    # below prevents the larger family labels and three-column legend from
    # overlapping.
    fig = plt.figure(figsize=(8.2, 8.0), constrained_layout=False)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=10.5)
    ax.set_rlabel_position(18)
    ax.grid(color="#cbc5e8", linewidth=0.8, alpha=0.95)
    ax.spines["polar"].set_color("#b8b1d8")
    ax.spines["polar"].set_linewidth(0.9)

    label_radius = 0.545
    for angle, family in zip(angles, families):
        x = 0.5 + label_radius * np.sin(angle)
        y = 0.5 + label_radius * np.cos(angle)
        ha = "center"
        va = "center"
        if x < 0.43:
            ha = "right"
        elif x > 0.57:
            ha = "left"
        if y < 0.43:
            va = "top"
        elif y > 0.57:
            va = "bottom"
        ax.text(
            x,
            y,
            POLICY_FAMILY_LABELS.get(family, family.replace("_", " ")),
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=12.5,
            linespacing=0.92,
            clip_on=False,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.93, "pad": 1.0},
        )

    rl_styles = {
        "dqn": {"marker": "o", "linestyle": "-", "linewidth": 1.85, "markersize": 4.0, "zorder": 6},
        "sac": {"marker": "^", "linestyle": (0, (4, 2)), "linewidth": 1.85, "markersize": 4.2, "zorder": 7},
        "ppo": {"marker": "s", "linestyle": (0, (1.2, 1.4)), "linewidth": 2.65, "markersize": 5.0, "zorder": 8},
    }
    plot_order = [p for p in policies if p not in {"dqn", "ppo", "sac"}] + ["dqn", "sac", "ppo"]
    for policy in plot_order:
        values = normalized.loc[policy, families].to_numpy(dtype=float)
        values = np.concatenate([values, values[:1]])
        is_rl = policy in rl_styles
        is_gpah = policy == "gpah"
        style = rl_styles.get(policy, {})
        ax.plot(
            closed_angles,
            values,
            linewidth=2.0 if is_gpah else style.get("linewidth", 1.35),
            alpha=0.98 if is_gpah else (0.94 if is_rl else 0.82),
            color=COLORS.get(policy, "#c8c2df"),
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker", None),
            markersize=style.get("markersize", 0),
            markerfacecolor="white" if is_rl else None,
            markeredgecolor=COLORS.get(policy, "#c8c2df") if is_rl else None,
            markeredgewidth=0.9 if is_rl else 0,
            label=POLICY_LABELS.get(policy, policy),
            zorder=style.get("zorder", 6 if is_gpah else 3),
        )

    ax.annotate(
        "PPO/SAC\nnearly overlap",
        xy=(np.deg2rad(235), 0.18),
        xycoords="data",
        xytext=(1.015, 0.36),
        textcoords=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10,
        color="#4b5563",
        arrowprops={"arrowstyle": "->", "color": "#6b7280", "lw": 0.8},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2},
        clip_on=False,
        zorder=10,
    )

    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        fontsize=10.5,
        columnspacing=0.75,
        handlelength=1.7,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.14, right=0.84, top=0.84, bottom=0.28)
    save(fig, out_path)


def ablation_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    family_order = ["B1_phase", "B2_service", "B3_scarcity", "B4_switch", "B5_mixed"]
    policy_order = ["mean_map", "no_phase", "no_queue", "no_scarcity", "no_service", "no_switch"]
    policy_labels = {
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
    fig, ax = plt.subplots(figsize=(7.2, 3.95), constrained_layout=True)
    im = ax.imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    ax.set_xticks(np.arange(len(policy_order)))
    ax.set_xticklabels([policy_labels[p] for p in policy_order], rotation=42, ha="right")
    ax.set_yticks(np.arange(len(family_order)))
    ax.set_yticklabels(family_order)
    ax.set_xlabel("Ablation / policy")
    ax.set_ylabel("Case family")
    ax.set_xticks(np.arange(-0.5, len(policy_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(family_order), 1), minor=True)
    ax.grid(which="minor", color="#d8d8d8", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Paired objective gap vs full gPAH")
    save(fig, out_path)
