#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

from map_gpah_paper1_common import plot_bar_by_family, plot_gap_heatmap, plot_log_bar_by_family


def read_csv(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def case_axis(case_summary: List[Dict[str, object]], family: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in case_summary:
        if row["family"] == family:
            relabeled = dict(row)
            relabeled["family"] = str(row["case_id"])
            rows.append(relabeled)
    return rows


def replot_policy(output_dir: Path) -> None:
    case_summary = read_csv(output_dir / "paper1_policy_case_summary.csv")
    gap_summary = read_csv(output_dir / "paper1_policy_paired_gap_vs_gpah.csv")
    preferred = [
        "equal_split",
        "backpressure",
        "maxweight_mu",
        "cmu",
        "affinity",
        "qah",
        "sah",
        "qah_switch",
        "gpah_meanmap",
        "gpah",
    ]
    present = {str(r["policy"]) for r in case_summary}
    policies = [p for p in preferred if p in present]

    for family, filename, title in [
        ("A1_topology", "fig_policy_topology.pdf", "Policy comparison across sparse compatibility topologies"),
        ("A2_map_burst", "fig_policy_map_burst.pdf", "Policy comparison across MAP burst profiles"),
        ("A5_switch", "fig_policy_switch_tradeoff.pdf", "Policy comparison across switching-friction regimes"),
        ("A6_scale", "fig_policy_scale.pdf", "Policy comparison across model scales"),
    ]:
        rows = case_axis(case_summary, family)
        families = sorted({str(r["family"]) for r in rows})
        plot_bar_by_family(rows, str(output_dir / filename), title, families, policies)
        log_filename = filename.replace(".pdf", "_log.pdf")
        plot_log_bar_by_family(rows, str(output_dir / log_filename), title, families, policies)

    plot_gap_heatmap(gap_summary, str(output_dir / "fig_policy_gap_heatmap.pdf"), "Paired gap relative to gPAH")


def replot_ablation(output_dir: Path) -> None:
    gap_summary = read_csv(output_dir / "paper1_ablation_paired_gap_vs_full_gpah.csv")
    plot_gap_heatmap(gap_summary, str(output_dir / "fig_ablation_gap.pdf"), "Ablation paired queue gap relative to full gPAH")
    plot_gap_heatmap(
        gap_summary,
        str(output_dir / "fig_ablation_objective_gap.pdf"),
        "Ablation paired objective gap relative to full gPAH",
        value_col="paired_objective_gap_mean",
        colorbar_label="Paired objective gap vs full gPAH",
    )
    plot_gap_heatmap(
        gap_summary,
        str(output_dir / "fig_ablation_component_heatmap.pdf"),
        "Component sensitivity by hard case family",
        value_col="paired_objective_gap_mean",
        colorbar_label="Paired objective gap vs full gPAH",
    )
    phase_rows = [r for r in gap_summary if r["family"] == "B1_phase"]
    plot_gap_heatmap(
        phase_rows,
        str(output_dir / "fig_ablation_map_sensitivity.pdf"),
        "MAP phase-information sensitivity",
        value_col="paired_objective_gap_mean",
        colorbar_label="Paired objective gap vs full gPAH",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot Paper 1 figures from existing CSV summaries.")
    parser.add_argument("--policy_dir", type=Path, required=True)
    parser.add_argument("--ablation_dir", type=Path, required=True)
    args = parser.parse_args()

    replot_policy(args.policy_dir)
    replot_ablation(args.ablation_dir)
    print(os.fspath(args.policy_dir))
    print(os.fspath(args.ablation_dir))


if __name__ == "__main__":
    main()
