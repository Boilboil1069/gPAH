#!/usr/bin/env python3
"""Validation-only RL hyperparameter search with disjoint test cases.

Candidates are selected by mean standardized objective on the declared
validation cases. The selected candidate for each algorithm is then retrained
from fresh seeds and evaluated on disjoint test cases. Test results never enter
candidate selection.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from map_gpah_paper1_common import RunConfig, make_problem, mean_ci, policy_cases, simulate_policy, write_csv
from map_gpah_paper1_rl_baselines import (
    RLConfig,
    RL_LABELS,
    evaluate_agent,
    train_dqn,
    train_ppo,
    train_sac,
)


DEFAULT_VALIDATION_CASES = "A1_topology/chain,A2_map_burst/predictive_burst,A5_switch/high"
DEFAULT_TEST_CASES = "A1_topology/bridge_bottleneck,A2_map_burst/high_burst,A5_switch/lag,A7_load/rho_0_95"
ALL_FAMILIES = (
    "A1_topology",
    "A2_map_burst",
    "A3_density",
    "A4_service",
    "A5_switch",
    "A6_scale",
    "A7_load",
    "A8_large_scale",
)
ALGORITHM_OFFSETS = {"dqn": 101, "ppo": 211, "sac": 307}


def parse_csv(raw: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def resolve_cases(raw: str):
    requested = parse_csv(raw)
    available = policy_cases(RunConfig(output_dir="", families=ALL_FAMILIES))
    by_key = {(case.family, case.case_id): case for case in available}
    output = []
    for item in requested:
        if "/" not in item:
            raise ValueError(f"case must be FAMILY/CASE_ID: {item}")
        family, case_id = item.split("/", 1)
        key = (family, case_id)
        if key not in by_key:
            raise ValueError(f"unknown case: {item}")
        output.append(by_key[key])
    return output


def load_candidates(path: str, algorithms: Sequence[str], max_candidates: int) -> Dict[str, List[Dict[str, object]]]:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    output: Dict[str, List[Dict[str, object]]] = {}
    valid_fields = set(RLConfig.__dataclass_fields__)
    for algorithm in algorithms:
        if algorithm not in raw:
            raise ValueError(f"candidate configuration has no {algorithm} section")
        candidates = list(raw[algorithm])
        if max_candidates > 0:
            candidates = candidates[:max_candidates]
        identifiers = [str(candidate.get("candidate_id", "")) for candidate in candidates]
        if not candidates or any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{algorithm} candidate_id values must be nonempty and unique")
        for candidate in candidates:
            unknown = set(candidate) - valid_fields - {"candidate_id"}
            if unknown:
                raise ValueError(f"unknown RLConfig fields for {algorithm}/{candidate['candidate_id']}: {sorted(unknown)}")
        output[algorithm] = candidates
    return output


def config_with_candidate(base: RLConfig, candidate: Mapping[str, object], train_steps: int) -> RLConfig:
    valid_fields = set(RLConfig.__dataclass_fields__)
    overrides = {key: value for key, value in candidate.items() if key in valid_fields}
    return replace(base, train_steps=train_steps, **overrides)


def train_agent(algorithm: str, problem, cfg: RLConfig, seed: int, label: str):
    if algorithm == "dqn":
        return train_dqn(problem, cfg, seed, label)
    if algorithm == "ppo":
        return train_ppo(problem, cfg, seed, label)
    if algorithm == "sac":
        return train_sac(problem, cfg, seed, label)
    raise ValueError(f"unknown algorithm: {algorithm}")


def append_history(
    output: List[Dict[str, object]],
    history: Sequence[Dict[str, float]],
    stage: str,
    algorithm: str,
    candidate_id: str,
    case,
    train_seed: int,
    cfg: RLConfig,
) -> None:
    for row in history:
        output.append({
            "stage": stage,
            "algorithm": algorithm,
            "candidate_id": candidate_id,
            "family": case.family,
            "case_id": case.case_id,
            "train_seed": train_seed,
            "train_steps_budget": cfg.train_steps,
            "lr": cfg.lr,
            "gamma": cfg.gamma,
            "batch_size": cfg.batch_size,
            "hidden": cfg.hidden,
            "target_tau": cfg.target_tau,
            "ppo_clip": cfg.ppo_clip,
            "entropy_coef": cfg.entropy_coef,
            "alpha": cfg.alpha,
            **row,
        })


def validation_summary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["algorithm"]), str(row["candidate_id"])), []).append(row)
    output: List[Dict[str, object]] = []
    for (algorithm, candidate_id), group in sorted(groups.items()):
        # First average evaluation replications within each case/train seed;
        # then give each validation case and training seed equal weight.
        cells: Dict[tuple[str, str, int], List[float]] = {}
        for row in group:
            key = (str(row["family"]), str(row["case_id"]), int(row["train_seed_index"]))
            cells.setdefault(key, []).append(float(row["objective"]))
        values = [float(np.mean(cell)) for cell in cells.values()]
        stats = mean_ci(values)
        output.append({
            "algorithm": algorithm,
            "candidate_id": candidate_id,
            "validation_cases": len({(str(row["family"]), str(row["case_id"])) for row in group}),
            "training_seeds": len({int(row["train_seed_index"]) for row in group}),
            "evaluation_replications": len({int(row["eval_replication"]) for row in group}),
            **{f"validation_objective_{name}": value for name, value in stats.items()},
        })
    return output


def select_candidates(
    summary: Sequence[Dict[str, object]],
    candidates: Mapping[str, Sequence[Dict[str, object]]],
) -> Dict[str, Dict[str, object]]:
    selected: Dict[str, Dict[str, object]] = {}
    for algorithm, algorithm_candidates in candidates.items():
        rows = [row for row in summary if row["algorithm"] == algorithm]
        winner = min(rows, key=lambda row: (float(row["validation_objective_mean"]), str(row["candidate_id"])))
        candidate = next(item for item in algorithm_candidates if item["candidate_id"] == winner["candidate_id"])
        selected[algorithm] = {
            **candidate,
            "selection_metric": "equal-case-and-training-seed mean validation objective",
            "validation_objective_mean": winner["validation_objective_mean"],
            "validation_objective_ci95": winner["validation_objective_ci95"],
        }
    return selected


def test_summary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["algorithm"]), str(row["candidate_id"])), []).append(row)
    output: List[Dict[str, object]] = []
    for (algorithm, candidate_id), group in sorted(groups.items()):
        cells: Dict[tuple[str, str, int], List[float]] = {}
        gap_cells: Dict[tuple[str, str, int], List[float]] = {}
        for row in group:
            key = (str(row["family"]), str(row["case_id"]), int(row["train_seed_index"]))
            cells.setdefault(key, []).append(float(row["objective"]))
            gap_cells.setdefault(key, []).append(float(row["paired_objective_gap_vs_gpah"]))
        objectives = [float(np.mean(values)) for values in cells.values()]
        gaps = [float(np.mean(values)) for values in gap_cells.values()]
        objective_stats = mean_ci(objectives)
        gap_stats = mean_ci(gaps)
        output.append({
            "algorithm": algorithm,
            "candidate_id": candidate_id,
            "test_cases": len({(str(row["family"]), str(row["case_id"])) for row in group}),
            "training_seeds": len({int(row["train_seed_index"]) for row in group}),
            "evaluation_replications": len({int(row["eval_replication"]) for row in group}),
            **{f"test_objective_{name}": value for name, value in objective_stats.items()},
            **{f"paired_objective_gap_vs_gpah_{name}": value for name, value in gap_stats.items()},
        })
    return output


def convergence_summary(history_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple[str, str, str, str, int], List[Dict[str, object]]] = {}
    for row in history_rows:
        if "train_reward_mean_window" not in row or row["train_reward_mean_window"] in ("", None):
            continue
        key = (
            str(row["stage"]),
            str(row["algorithm"]),
            str(row["candidate_id"]),
            str(row["case_id"]),
            int(row["train_seed"]),
        )
        groups.setdefault(key, []).append(row)
    output: List[Dict[str, object]] = []
    for key, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: int(float(row["step"])))
        rewards = np.asarray([float(row["train_reward_mean_window"]) for row in ordered], dtype=float)
        steps = np.asarray([float(row["step"]) for row in ordered], dtype=float)
        quarter = max(1, len(rewards) // 4)
        late_reward = float(rewards[-quarter:].mean())
        early_reward = float(rewards[:quarter].mean())
        late_steps = steps[-quarter:]
        late_values = rewards[-quarter:]
        slope = float(np.polyfit(late_steps, late_values, 1)[0]) if len(late_values) > 1 else 0.0
        stage, algorithm, candidate_id, case_id, train_seed = key
        output.append({
            "stage": stage,
            "algorithm": algorithm,
            "candidate_id": candidate_id,
            "case_id": case_id,
            "train_seed": train_seed,
            "history_points": len(ordered),
            "early_quarter_reward_mean": early_reward,
            "late_quarter_reward_mean": late_reward,
            "late_minus_early_reward": late_reward - early_reward,
            "late_reward_slope_per_step": slope,
        })
    return output


def plot_validation(rows: Sequence[Dict[str, object]], output_path: str) -> None:
    algorithms = list(dict.fromkeys(str(row["algorithm"]) for row in rows))
    fig, axes = plt.subplots(1, len(algorithms), figsize=(4.5 * len(algorithms), 4.0), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, algorithm in zip(axes, algorithms):
        subset = [row for row in rows if row["algorithm"] == algorithm]
        x = np.arange(len(subset))
        ax.bar(
            x,
            [float(row["validation_objective_mean"]) for row in subset],
            yerr=[float(row["validation_objective_ci95"]) for row in subset],
            capsize=3,
            color="#7FABD1",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(row["candidate_id"]).replace("_", " ") for row in subset], rotation=30, ha="right")
        ax.set_title(RL_LABELS.get(algorithm, algorithm))
        ax.set_ylabel("Validation standardized objective")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_selected_history(rows: Sequence[Dict[str, object]], output_path: str) -> None:
    rows = [row for row in rows if row["stage"] == "test" and "train_reward_mean_window" in row]
    algorithms = list(dict.fromkeys(str(row["algorithm"]) for row in rows))
    if not algorithms:
        return
    fig, axes = plt.subplots(1, len(algorithms), figsize=(4.7 * len(algorithms), 4.0), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, algorithm in zip(axes, algorithms):
        subset = [row for row in rows if row["algorithm"] == algorithm]
        by_step: Dict[int, List[float]] = {}
        for row in subset:
            by_step.setdefault(int(float(row["step"])), []).append(float(row["train_reward_mean_window"]))
        steps = sorted(by_step)
        ax.plot(steps, [float(np.mean(by_step[step])) for step in steps], linewidth=1.8, color="#EC6E66")
        ax.set_title(RL_LABELS.get(algorithm, algorithm))
        ax.set_xlabel("Training step")
        ax.set_ylabel("Mean windowed training reward")
        ax.grid(alpha=0.25)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-based RL hyperparameter and convergence study.")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--candidate-config", default="configs/revision_rl_hyperparameters.json")
    parser.add_argument("--algorithms", default="dqn,ppo,sac")
    parser.add_argument("--validation-cases", default=DEFAULT_VALIDATION_CASES)
    parser.add_argument("--test-cases", default=DEFAULT_TEST_CASES)
    parser.add_argument("--max-candidates", type=int, default=0, help="Limit candidates per algorithm; 0 uses all.")
    parser.add_argument("--search-train-steps", type=int, default=100000)
    parser.add_argument("--final-train-steps", type=int, default=250000)
    parser.add_argument("--search-train-seeds", type=int, default=3)
    parser.add_argument("--final-train-seeds", type=int, default=5)
    parser.add_argument("--validation-replications", type=int, default=3)
    parser.add_argument("--test-replications", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=3000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=33260527)
    args = parser.parse_args()

    algorithms = parse_csv(args.algorithms)
    invalid = sorted(set(algorithms) - set(ALGORITHM_OFFSETS))
    if invalid:
        raise ValueError("unknown algorithms: " + ", ".join(invalid))
    validation_cases = resolve_cases(args.validation_cases)
    test_cases = resolve_cases(args.test_cases)
    validation_keys = {(case.family, case.case_id) for case in validation_cases}
    test_keys = {(case.family, case.case_id) for case in test_cases}
    if validation_keys & test_keys:
        raise ValueError(f"validation and test cases overlap: {sorted(validation_keys & test_keys)}")
    candidates = load_candidates(args.candidate_config, algorithms, args.max_candidates)
    os.makedirs(args.output_dir, exist_ok=True)

    base_cfg = RLConfig(
        output_dir=args.output_dir,
        algorithms=algorithms,
        families=tuple(),
        case_ids=tuple(),
        replications=args.validation_replications,
        train_seeds=args.search_train_seeds,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        train_steps=args.search_train_steps,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
        device=args.device,
        train_progress=args.train_progress,
        budget_preset="revision_validation",
    )

    validation_rows: List[Dict[str, object]] = []
    history_rows: List[Dict[str, object]] = []
    for algorithm in algorithms:
        for candidate in candidates[algorithm]:
            candidate_id = str(candidate["candidate_id"])
            cfg = config_with_candidate(base_cfg, candidate, args.search_train_steps)
            for case_index, case in enumerate(validation_cases, 1):
                problem = make_problem(
                    case,
                    args.seed + 1009 * case_index,
                )
                for train_seed_index in range(args.search_train_seeds):
                    train_seed = (
                        args.seed
                        + ALGORITHM_OFFSETS[algorithm]
                        + 1000000 * case_index
                        + train_seed_index
                    )
                    label = f"validation {algorithm}/{candidate_id} {case.case_id} seed={train_seed_index}"
                    agent, history = train_agent(algorithm, problem, cfg, train_seed, label)
                    append_history(history_rows, history, "validation", algorithm, candidate_id, case, train_seed, cfg)
                    for eval_replication in range(args.validation_replications):
                        eval_seed = args.seed + 50000000 + 100000 * case_index + eval_replication
                        metrics = evaluate_agent(problem, agent, eval_seed, args.warmup_steps, args.eval_steps)
                        validation_rows.append({
                            "stage": "validation",
                            "algorithm": algorithm,
                            "candidate_id": candidate_id,
                            "family": case.family,
                            "case_id": case.case_id,
                            "train_seed_index": train_seed_index,
                            "train_seed": train_seed,
                            "eval_replication": eval_replication,
                            "eval_seed": eval_seed,
                            "search_train_steps": args.search_train_steps,
                            **metrics,
                        })

    validation_summary_rows = validation_summary(validation_rows)
    selected = select_candidates(validation_summary_rows, candidates)
    with open(os.path.join(args.output_dir, "selected_hyperparameters.json"), "w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)

    test_rows: List[Dict[str, object]] = []
    baseline_cache: Dict[tuple[str, str, int], Dict[str, float]] = {}
    for algorithm in algorithms:
        candidate = selected[algorithm]
        candidate_id = str(candidate["candidate_id"])
        cfg = config_with_candidate(
            replace(base_cfg, replications=args.test_replications, train_seeds=args.final_train_seeds),
            candidate,
            args.final_train_steps,
        )
        for case_index, case in enumerate(test_cases, 1):
            problem = make_problem(
                case,
                args.seed + 70000000 + 1009 * case_index,
            )
            for train_seed_index in range(args.final_train_seeds):
                train_seed = (
                    args.seed
                    + 80000000
                    + ALGORITHM_OFFSETS[algorithm]
                    + 1000000 * case_index
                    + train_seed_index
                )
                label = f"test {algorithm}/{candidate_id} {case.case_id} seed={train_seed_index}"
                agent, history = train_agent(algorithm, problem, cfg, train_seed, label)
                append_history(history_rows, history, "test", algorithm, candidate_id, case, train_seed, cfg)
                for eval_replication in range(args.test_replications):
                    eval_seed = args.seed + 90000000 + 100000 * case_index + eval_replication
                    cache_key = (case.family, case.case_id, eval_seed)
                    if cache_key not in baseline_cache:
                        baseline_cache[cache_key] = simulate_policy(
                            problem,
                            "gpah",
                            eval_seed,
                            args.warmup_steps,
                            args.eval_steps,
                        )
                    baseline = baseline_cache[cache_key]
                    metrics = evaluate_agent(problem, agent, eval_seed, args.warmup_steps, args.eval_steps)
                    test_rows.append({
                        "stage": "test",
                        "algorithm": algorithm,
                        "algorithm_label": RL_LABELS[algorithm],
                        "candidate_id": candidate_id,
                        "family": case.family,
                        "case_id": case.case_id,
                        "train_seed_index": train_seed_index,
                        "train_seed": train_seed,
                        "eval_replication": eval_replication,
                        "eval_seed": eval_seed,
                        "final_train_steps": args.final_train_steps,
                        "gpah_objective": baseline["objective"],
                        "paired_objective_gap_vs_gpah": float(metrics["objective"] - baseline["objective"]),
                        **metrics,
                    })

    test_summary_rows = test_summary(test_rows)
    convergence_rows = convergence_summary(history_rows)
    write_csv(os.path.join(args.output_dir, "paper1_rl_hyperparameter_validation_replications.csv"), validation_rows)
    write_csv(os.path.join(args.output_dir, "paper1_rl_hyperparameter_validation_summary.csv"), validation_summary_rows)
    write_csv(os.path.join(args.output_dir, "paper1_rl_hyperparameter_test_replications.csv"), test_rows)
    write_csv(os.path.join(args.output_dir, "paper1_rl_hyperparameter_test_summary.csv"), test_summary_rows)
    write_csv(os.path.join(args.output_dir, "paper1_rl_hyperparameter_training_history.csv"), history_rows)
    write_csv(os.path.join(args.output_dir, "paper1_rl_hyperparameter_convergence_summary.csv"), convergence_rows)
    plot_validation(validation_summary_rows, os.path.join(args.output_dir, "fig_rl_hyperparameter_validation.pdf"))
    plot_selected_history(history_rows, os.path.join(args.output_dir, "fig_rl_selected_convergence.pdf"))

    manifest = {
        "experiment": "rl_hyperparameter_validation",
        "selection_rule": "minimum equal-case-and-training-seed mean standardized objective on validation cases",
        "paired_search_seeds": "all candidates of an algorithm use the same problem, training, and evaluation seeds",
        "test_isolation": "test cases and fresh training seeds are not used during candidate selection",
        "candidate_config": args.candidate_config,
        "algorithms": list(algorithms),
        "validation_cases": [f"{case.family}/{case.case_id}" for case in validation_cases],
        "test_cases": [f"{case.family}/{case.case_id}" for case in test_cases],
        "search_train_steps": args.search_train_steps,
        "final_train_steps": args.final_train_steps,
        "search_train_seeds": args.search_train_seeds,
        "final_train_seeds": args.final_train_seeds,
        "validation_replications": args.validation_replications,
        "test_replications": args.test_replications,
        "warmup_steps": args.warmup_steps,
        "eval_steps": args.eval_steps,
        "rollout_steps": args.rollout_steps,
        "device": args.device,
        "seed": args.seed,
        "selected": selected,
    }
    with open(os.path.join(args.output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"RL hyperparameter validation results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
