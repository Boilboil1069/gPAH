#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Dict, List

from map_gpah_paper1_common import (
    POLICY_LABELS,
    add_common_args,
    aggregate_family,
    config_from_args,
    make_problem,
    paired_family_summary,
    paired_gap_summary,
    plot_bar_by_family,
    plot_gap_heatmap,
    plot_log_bar_by_family,
    plot_policy_radar,
    plot_relative_heatmap,
    policy_cases,
    render_progress,
    simulate_policy,
    summarize_replications,
    write_csv,
)


def run(args: argparse.Namespace) -> None:
    cfg = config_from_args(args)
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"Output directory: {cfg.output_dir}", flush=True)
    cases = policy_cases(cfg)
    rep_rows: List[Dict[str, object]] = []

    total = len(cases) * len(cfg.policies) * cfg.replications
    done = 0
    render_progress(done, total, "starting")
    for ci, case in enumerate(cases, start=1):
        problem = make_problem(case, cfg.seed + 1009 * ci)
        for r in range(cfg.replications):
            # Common random numbers: all policies in the same case/replication
            # see the same exogenous random stream, making paired gaps strict.
            seed = cfg.seed + 100000 * ci + r
            for policy in cfg.policies:
                metrics = simulate_policy(problem, policy, seed, cfg.warmup_steps, cfg.eval_steps)
                row: Dict[str, object] = {
                    "experiment": "policy_comparison",
                    "family": case.family,
                    "case_id": case.case_id,
                    "description": case.description,
                    "d": case.d,
                    "K": case.K,
                    "topology": case.topology,
                    "map_profile": case.map_profile,
                    "density": case.density,
                    "service_profile": case.service_profile,
                    "switch_profile": case.switch_profile,
                    "load": case.load,
                    "policy": policy,
                    "policy_label": POLICY_LABELS.get(policy, policy),
                    "replication": r,
                    "seed": seed,
                }
                row.update(metrics)
                rep_rows.append(row)
                done += 1
                render_progress(done, total, f"case {ci}/{len(cases)} {case.case_id} policy={policy}")

    case_summary = summarize_replications(rep_rows, ["family", "case_id", "description", "policy"], baseline_policy="gpah")
    gap_summary = paired_gap_summary(rep_rows, baseline_policy="gpah")
    family_paired_summary = paired_family_summary(rep_rows, baseline_policy="gpah")
    family_summary = aggregate_family(case_summary, paired_rows=family_paired_summary)

    write_csv(os.path.join(cfg.output_dir, "paper1_policy_replications.csv"), rep_rows)
    write_csv(os.path.join(cfg.output_dir, "paper1_policy_case_summary.csv"), case_summary)
    write_csv(os.path.join(cfg.output_dir, "paper1_policy_family_summary.csv"), family_summary)
    write_csv(os.path.join(cfg.output_dir, "paper1_policy_family_paired_summary.csv"), family_paired_summary)
    write_csv(os.path.join(cfg.output_dir, "paper1_policy_paired_gap_vs_gpah.csv"), gap_summary)

    def case_axis(family: str) -> List[Dict[str, object]]:
        rows = []
        for row in case_summary:
            if row["family"] == family:
                relabeled = dict(row)
                relabeled["family"] = str(row["case_id"])
                rows.append(relabeled)
        return rows

    topology_cases = case_axis("A1_topology")
    map_cases = case_axis("A2_map_burst")
    switch_cases = case_axis("A5_switch")
    scale_cases = case_axis("A6_scale")
    plot_bar_by_family(topology_cases, os.path.join(cfg.output_dir, "fig_policy_topology.pdf"), "Policy comparison across sparse compatibility topologies", sorted({str(r["family"]) for r in topology_cases}), list(cfg.policies))
    plot_bar_by_family(map_cases, os.path.join(cfg.output_dir, "fig_policy_map_burst.pdf"), "Policy comparison across MAP burst profiles", sorted({str(r["family"]) for r in map_cases}), list(cfg.policies))
    plot_bar_by_family(switch_cases, os.path.join(cfg.output_dir, "fig_policy_switch_tradeoff.pdf"), "Policy comparison across switching-friction regimes", sorted({str(r["family"]) for r in switch_cases}), list(cfg.policies))
    plot_bar_by_family(scale_cases, os.path.join(cfg.output_dir, "fig_policy_scale.pdf"), "Policy comparison across model scales", sorted({str(r["family"]) for r in scale_cases}), list(cfg.policies))
    plot_log_bar_by_family(topology_cases, os.path.join(cfg.output_dir, "fig_policy_topology_log.pdf"), "Log-scale policy comparison across sparse compatibility topologies", sorted({str(r["family"]) for r in topology_cases}), list(cfg.policies))
    plot_log_bar_by_family(map_cases, os.path.join(cfg.output_dir, "fig_policy_map_burst_log.pdf"), "Log-scale policy comparison across MAP burst profiles", sorted({str(r["family"]) for r in map_cases}), list(cfg.policies))
    plot_log_bar_by_family(switch_cases, os.path.join(cfg.output_dir, "fig_policy_switch_tradeoff_log.pdf"), "Log-scale policy comparison across switching-friction regimes", sorted({str(r["family"]) for r in switch_cases}), list(cfg.policies))
    plot_log_bar_by_family(scale_cases, os.path.join(cfg.output_dir, "fig_policy_scale_log.pdf"), "Log-scale policy comparison across model scales", sorted({str(r["family"]) for r in scale_cases}), list(cfg.policies))
    plot_gap_heatmap(gap_summary, os.path.join(cfg.output_dir, "fig_policy_gap_heatmap.pdf"), "Paired gap relative to gPAH")
    plot_gap_heatmap(gap_summary, os.path.join(cfg.output_dir, "fig_policy_objective_gap_heatmap.pdf"), "Paired objective gap relative to gPAH", value_col="paired_objective_gap_mean", colorbar_label="Objective gap vs gPAH")
    plot_relative_heatmap(case_summary, os.path.join(cfg.output_dir, "fig_policy_relative_queue_heatmap.pdf"), "Mean-total-queue ratio relative to gPAH")
    plot_policy_radar(family_summary, os.path.join(cfg.output_dir, "fig_policy_family_radar.pdf"), "Normalized family-level performance radar", list(cfg.policies))


def main() -> None:
    p = argparse.ArgumentParser(description="Paper 1 policy-comparison experiments for generalized gPAH.")
    add_common_args(
        p,
        default_families="A1_topology,A2_map_burst,A3_density,A4_service,A5_switch,A6_scale,A7_load,A8_large_scale",
        default_policies="equal_split,backpressure,maxweight_mu,cmu,affinity,rl_queueing,qah_switch,gpah_meanmap,gpah",
        seed=20260507,
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
