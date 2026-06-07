#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from typing import Dict, List

from map_gpah_paper1_common import (
    POLICY_LABELS,
    SENSITIVITY_PARAMETERS,
    ScoreWeights,
    SparseMapPhEnv,
    add_common_args,
    aggregate_family,
    config_from_args,
    gpah_policy_with_weights,
    make_problem,
    mean_ci,
    paired_family_summary,
    paired_gap_summary,
    plot_gap_heatmap,
    policy_cases,
    render_progress,
    simulate_policy,
    simulate_policy_object,
    summarize_replications,
    write_csv,
)


def multiplier_token(value: float) -> str:
    return f"{value:.2f}".replace(".", "_").replace("-", "m")


def sensitivity_policy_id(parameter: str, multiplier: float) -> str:
    return f"sens_{parameter}_x{multiplier_token(multiplier)}"


def build_sensitivity_weights(parameter: str, multiplier: float, base: ScoreWeights) -> ScoreWeights:
    if parameter not in SENSITIVITY_PARAMETERS:
        raise ValueError(f"unknown sensitivity parameter: {parameter}")
    field, _ = SENSITIVITY_PARAMETERS[parameter]
    return replace(base, **{field: getattr(base, field) * multiplier})


def annotate_policy_metadata(rows: List[Dict[str, object]], metadata: Dict[str, Dict[str, object]]) -> None:
    for row in rows:
        meta = metadata.get(str(row.get("policy", "")))
        if not meta:
            continue
        for key, value in meta.items():
            row[key] = value


def parameter_summary(gap_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in gap_rows:
        parameter = str(row.get("alpha_parameter", ""))
        if not parameter:
            continue
        multiplier = float(row.get("alpha_multiplier", 1.0))
        groups.setdefault((parameter, multiplier), []).append(row)

    out: List[Dict[str, object]] = []
    for (parameter, multiplier), rows in sorted(groups.items()):
        queue_gaps = [float(r["paired_queue_gap_mean"]) for r in rows]
        objective_gaps = [float(r["paired_objective_gap_mean"]) for r in rows]
        queue_improvements = [float(r["paired_queue_relative_improvement_mean"]) for r in rows]
        objective_improvements = [float(r["paired_objective_relative_improvement_mean"]) for r in rows]
        total_n = sum(int(r.get("paired_queue_gap_n", 0)) for r in rows)
        queue_win = sum(float(r.get("paired_queue_baseline_win_rate", 0.0)) * int(r.get("paired_queue_gap_n", 0)) for r in rows)
        objective_win = sum(float(r.get("paired_objective_baseline_win_rate", 0.0)) * int(r.get("paired_objective_gap_n", 0)) for r in rows)
        qst = mean_ci(queue_gaps)
        ost = mean_ci(objective_gaps)
        qist = mean_ci(queue_improvements)
        oist = mean_ci(objective_improvements)
        out.append({
            "alpha_parameter": parameter,
            "alpha_field": SENSITIVITY_PARAMETERS[parameter][0],
            "alpha_description": SENSITIVITY_PARAMETERS[parameter][1],
            "alpha_multiplier": multiplier,
            "num_cases": len(rows),
            "num_paired_replications": total_n,
            "queue_gap_mean_across_cases": qst["mean"],
            "queue_gap_ci95_across_cases": qst["ci95"],
            "objective_gap_mean_across_cases": ost["mean"],
            "objective_gap_ci95_across_cases": ost["ci95"],
            "queue_relative_improvement_mean_across_cases": qist["mean"],
            "queue_relative_improvement_ci95_across_cases": qist["ci95"],
            "objective_relative_improvement_mean_across_cases": oist["mean"],
            "objective_relative_improvement_ci95_across_cases": oist["ci95"],
            "queue_baseline_win_rate": queue_win / max(1, total_n),
            "objective_baseline_win_rate": objective_win / max(1, total_n),
        })
    return out


def run(args: argparse.Namespace) -> None:
    cfg = config_from_args(args)
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"Output directory: {cfg.output_dir}", flush=True)

    parameters = [x.strip() for x in args.parameters.split(",") if x.strip()]
    multipliers = [float(x.strip()) for x in args.multipliers.split(",") if x.strip()]
    base_weights = ScoreWeights()
    cases = policy_cases(cfg)

    policy_metadata: Dict[str, Dict[str, object]] = {
        "gpah": {
            "policy_label": POLICY_LABELS["gpah"],
            "alpha_parameter": "",
            "alpha_field": "",
            "alpha_description": "baseline",
            "alpha_multiplier": 1.0,
        }
    }
    variants = []
    for parameter in parameters:
        if parameter not in SENSITIVITY_PARAMETERS:
            raise ValueError(f"unknown sensitivity parameter: {parameter}")
        field, description = SENSITIVITY_PARAMETERS[parameter]
        for multiplier in multipliers:
            policy = sensitivity_policy_id(parameter, multiplier)
            weights = build_sensitivity_weights(parameter, multiplier, base_weights)
            variants.append((policy, parameter, field, description, multiplier, weights))
            policy_metadata[policy] = {
                "policy_label": f"{parameter} x {multiplier:g}",
                "alpha_parameter": parameter,
                "alpha_field": field,
                "alpha_description": description,
                "alpha_multiplier": multiplier,
            }

    rep_rows: List[Dict[str, object]] = []
    total = len(cases) * cfg.replications * (1 + len(variants))
    done = 0
    render_progress(done, total, "starting")
    for ci, case in enumerate(cases, start=1):
        problem = make_problem(case, cfg.seed + 3001 * ci)
        for r in range(cfg.replications):
            seed = cfg.seed + 100000 * ci + r
            baseline_metrics = simulate_policy(problem, "gpah", seed, cfg.warmup_steps, cfg.eval_steps)
            baseline_row: Dict[str, object] = {
                "experiment": "parameter_sensitivity",
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
                "policy": "gpah",
                "replication": r,
                "seed": seed,
            }
            baseline_row.update(policy_metadata["gpah"])
            baseline_row.update(baseline_metrics)
            rep_rows.append(baseline_row)
            done += 1
            render_progress(done, total, f"case {ci}/{len(cases)} {case.case_id} baseline")

            for policy_name, parameter, field, description, multiplier, weights in variants:
                env = SparseMapPhEnv(problem, seed=seed)
                env.reset()
                policy = gpah_policy_with_weights(weights, use_current_map=True, label=policy_name)
                policy.reset(problem)
                metrics = simulate_policy_object(problem, policy, env, cfg.warmup_steps, cfg.eval_steps)
                row: Dict[str, object] = {
                    "experiment": "parameter_sensitivity",
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
                    "policy": policy_name,
                    "replication": r,
                    "seed": seed,
                    "policy_label": policy_metadata[policy_name]["policy_label"],
                    "alpha_parameter": parameter,
                    "alpha_field": field,
                    "alpha_description": description,
                    "alpha_multiplier": multiplier,
                }
                row.update(metrics)
                rep_rows.append(row)
                done += 1
                render_progress(done, total, f"case {ci}/{len(cases)} {case.case_id} {policy_name}")

    case_summary = summarize_replications(rep_rows, ["family", "case_id", "description", "policy"], baseline_policy="gpah")
    gap_summary = paired_gap_summary(rep_rows, baseline_policy="gpah")
    family_paired_summary = paired_family_summary(rep_rows, baseline_policy="gpah")
    family_summary = aggregate_family(case_summary, paired_rows=family_paired_summary)
    for rows in (case_summary, gap_summary, family_paired_summary, family_summary):
        annotate_policy_metadata(rows, policy_metadata)
    param_summary = parameter_summary(gap_summary)

    write_csv(os.path.join(cfg.output_dir, "paper1_sensitivity_replications.csv"), rep_rows)
    write_csv(os.path.join(cfg.output_dir, "paper1_sensitivity_case_summary.csv"), case_summary)
    write_csv(os.path.join(cfg.output_dir, "paper1_sensitivity_family_summary.csv"), family_summary)
    write_csv(os.path.join(cfg.output_dir, "paper1_sensitivity_family_paired_summary.csv"), family_paired_summary)
    write_csv(os.path.join(cfg.output_dir, "paper1_sensitivity_paired_gap_vs_gpah.csv"), gap_summary)
    write_csv(os.path.join(cfg.output_dir, "paper1_sensitivity_parameter_summary.csv"), param_summary)

    plot_gap_heatmap(
        gap_summary,
        os.path.join(cfg.output_dir, "fig_sensitivity_queue_gap.pdf"),
        "gPAH parameter sensitivity: paired queue gap vs baseline",
        value_col="paired_queue_gap_mean",
        colorbar_label="Paired queue gap vs baseline gPAH",
    )
    plot_gap_heatmap(
        gap_summary,
        os.path.join(cfg.output_dir, "fig_sensitivity_objective_gap.pdf"),
        "gPAH parameter sensitivity: paired objective gap vs baseline",
        value_col="paired_objective_gap_mean",
        colorbar_label="Paired objective gap vs baseline gPAH",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Paper 1 parameter-sensitivity experiments for generalized gPAH.")
    add_common_args(
        p,
        default_families="A2_map_burst,A5_switch,A7_load,A8_large_scale",
        default_policies="gpah",
        seed=20260509,
    )
    p.add_argument("--parameters", default="alpha_Q,alpha_A,alpha_mu,alpha_S,alpha_C,alpha_L")
    p.add_argument("--multipliers", default="0,0.5,1,1.5,2")
    run(p.parse_args())


if __name__ == "__main__":
    main()
