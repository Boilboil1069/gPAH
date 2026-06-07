#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Dict, List

from map_gpah_paper1_common import (
    POLICY_LABELS,
    add_common_args,
    aggregate_family,
    ablation_cases,
    config_from_args,
    make_problem,
    paired_family_summary,
    paired_gap_summary,
    plot_gap_heatmap,
    render_progress,
    simulate_policy,
    summarize_replications,
    write_csv,
)


ABLATION_TARGET_POLICY = {
    "phase": "no_phase",
    "service": "no_service",
    "scarcity": "no_scarcity",
    "switch": "no_switch",
    "mixed": "no_queue",
}


def component_regime_summary(gap_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[float]] = {}
    queue_groups: Dict[tuple, List[float]] = {}
    ns: Dict[tuple, int] = {}
    for row in gap_rows:
        component = str(row.get("component", ""))
        regime = str(row.get("regime", ""))
        policy = str(row.get("policy", ""))
        if not component or not regime:
            continue
        target = ABLATION_TARGET_POLICY.get(component)
        if policy != target:
            continue
        key = (component, regime, policy)
        groups.setdefault(key, []).append(float(row["paired_objective_gap_mean"]))
        queue_groups.setdefault(key, []).append(float(row["paired_gap_mean"]))
        ns[key] = ns.get(key, 0) + int(row.get("paired_objective_gap_n", 0))
    out: List[Dict[str, object]] = []
    for (component, regime, policy), vals in sorted(groups.items()):
        qvals = queue_groups[(component, regime, policy)]
        out.append({
            "component": component,
            "regime": regime,
            "ablated_policy": policy,
            "num_cases": len(vals),
            "num_replications": ns[(component, regime, policy)],
            "mean_paired_objective_gap": sum(vals) / max(1, len(vals)),
            "mean_paired_queue_gap": sum(qvals) / max(1, len(qvals)),
            "interpretation": "necessary_in_this_regime" if sum(vals) / max(1, len(vals)) > 0 else "not_necessary_or_overweighted",
        })
    by_component: Dict[str, Dict[str, float]] = {}
    for row in out:
        by_component.setdefault(str(row["component"]), {})[str(row["regime"])] = float(row["mean_paired_objective_gap"])
    for row in out:
        comp = str(row["component"])
        regimes = by_component.get(comp, {})
        if "on" in regimes and "off" in regimes:
            row["on_minus_off_objective_gap"] = regimes["on"] - regimes["off"]
        else:
            row["on_minus_off_objective_gap"] = ""
    return out


def run(args: argparse.Namespace) -> None:
    cfg = config_from_args(args)
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"Output directory: {cfg.output_dir}", flush=True)
    cases = ablation_cases(cfg)
    rep_rows: List[Dict[str, object]] = []
    policy_offsets = {policy: idx for idx, policy in enumerate(cfg.policies)}

    total = len(cases) * len(cfg.policies) * cfg.replications
    done = 0
    render_progress(done, total, "starting")
    for ci, case in enumerate(cases, start=1):
        problem = make_problem(case, cfg.seed + 2003 * ci)
        for policy in cfg.policies:
            for r in range(cfg.replications):
                seed = cfg.seed + 100000 * ci + 1000 * policy_offsets[policy] + r
                metrics = simulate_policy(problem, policy, seed, cfg.warmup_steps, cfg.eval_steps)
                row: Dict[str, object] = {
                    "experiment": "component_ablation",
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
                    "component": case.component,
                    "regime": case.regime,
                    "policy": policy,
                    "policy_label": POLICY_LABELS.get(policy, policy),
                    "replication": r,
                    "seed": seed,
                }
                row.update(metrics)
                rep_rows.append(row)
                done += 1
                render_progress(done, total, f"case {ci}/{len(cases)} {case.case_id} policy={policy}")

    case_summary = summarize_replications(rep_rows, ["family", "case_id", "description", "component", "regime", "policy"], baseline_policy="gpah_full")
    gap_summary = paired_gap_summary(rep_rows, baseline_policy="gpah_full")
    family_paired_summary = paired_family_summary(rep_rows, baseline_policy="gpah_full")
    family_summary = aggregate_family(case_summary, paired_rows=family_paired_summary)
    regime_summary = component_regime_summary(gap_summary)

    write_csv(os.path.join(cfg.output_dir, "paper1_ablation_replications.csv"), rep_rows)
    write_csv(os.path.join(cfg.output_dir, "paper1_ablation_case_summary.csv"), case_summary)
    write_csv(os.path.join(cfg.output_dir, "paper1_ablation_family_summary.csv"), family_summary)
    write_csv(os.path.join(cfg.output_dir, "paper1_ablation_family_paired_summary.csv"), family_paired_summary)
    write_csv(os.path.join(cfg.output_dir, "paper1_ablation_paired_gap_vs_full_gpah.csv"), gap_summary)
    write_csv(os.path.join(cfg.output_dir, "paper1_ablation_component_regime_summary.csv"), regime_summary)

    plot_gap_heatmap(gap_summary, os.path.join(cfg.output_dir, "fig_ablation_gap.pdf"), "Ablation paired queue gap relative to full gPAH")
    plot_gap_heatmap(
        gap_summary,
        os.path.join(cfg.output_dir, "fig_ablation_objective_gap.pdf"),
        "Ablation paired objective gap relative to full gPAH",
        value_col="paired_objective_gap_mean",
        colorbar_label="Paired objective gap vs full gPAH",
    )
    plot_gap_heatmap(
        gap_summary,
        os.path.join(cfg.output_dir, "fig_ablation_component_heatmap.pdf"),
        "Component sensitivity by hard case family",
        value_col="paired_objective_gap_mean",
        colorbar_label="Paired objective gap vs full gPAH",
    )
    phase_rows = [r for r in gap_summary if r["family"] == "B1_phase"]
    plot_gap_heatmap(
        phase_rows,
        os.path.join(cfg.output_dir, "fig_ablation_map_sensitivity.pdf"),
        "MAP phase-information sensitivity",
        value_col="paired_objective_gap_mean",
        colorbar_label="Paired objective gap vs full gPAH",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Paper 1 component-ablation experiments for generalized gPAH.")
    add_common_args(
        p,
        default_families="B1_phase,B2_service,B3_scarcity,B4_switch,B5_mixed",
        default_policies="gpah_full,no_phase,no_service,no_scarcity,no_switch,no_queue,mean_map",
        seed=20260508,
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
