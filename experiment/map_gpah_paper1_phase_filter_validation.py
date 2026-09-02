#!/usr/bin/env python3
"""Online hidden-MMPP-phase identification and scheduling validation.

The Bayesian filters observe only per-queue arrival counts.  The true phase is
never used by their forecasts and is accessed only by evaluation diagnostics
that do not feed back into the policy.  The estimated-parameter filter is calibrated on
an independent arrival-only trace using Poisson-HMM expectation maximization.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from map_gpah_paper1_common import (
    PAPER_COLORS,
    CaseSpec,
    RunConfig,
    SparseMapPhEnv,
    make_problem,
    mean_ci,
    policy_cases,
    policy_from_name,
    render_progress,
    write_csv,
)


DEFAULT_CASE_IDS = (
    "low_contrast",
    "fast_mixing",
    "high_burst",
    "predictive_burst",
    "multi_state_burst",
    "alternating_hotspots",
)
DEFAULT_REGIMES = "oracle,bayes_known,bayes_estimated,stationary"
REGIME_LABELS = {
    "oracle": "Oracle phase",
    "bayes_known": "Bayes filter (known parameters)",
    "bayes_estimated": "Bayes filter (estimated parameters)",
    "stationary": "Stationary mean",
}


def exact_phase_transition(Q: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Transition matrix matching SparseMapPhEnv._advance_map_phases."""
    n = Q.shape[0]
    transition = np.zeros((n, n), dtype=float)
    for state in range(n):
        departure_rate = max(0.0, float(-Q[state, state]))
        if departure_rate <= 0.0:
            transition[state, state] = 1.0
            continue
        stay = math.exp(-departure_rate * dt)
        transition[state, state] = stay
        off_diagonal = np.maximum(Q[state].copy(), 0.0)
        off_diagonal[state] = 0.0
        total = float(off_diagonal.sum())
        if total <= 0.0:
            transition[state, state] = 1.0
        else:
            transition[state] += (1.0 - stay) * off_diagonal / total
    transition /= transition.sum(axis=1, keepdims=True)
    return transition


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    matrix = transition.T - np.eye(transition.shape[0])
    matrix[-1, :] = 1.0
    rhs = np.zeros(transition.shape[0], dtype=float)
    rhs[-1] = 1.0
    try:
        stationary = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        stationary = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    stationary = np.maximum(stationary, 0.0)
    total = float(stationary.sum())
    return stationary / total if total > 0.0 else np.full(transition.shape[0], 1.0 / transition.shape[0])


def poisson_emission_weights(observations: np.ndarray, rates: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    means = np.maximum(1e-8, rates * dt)
    log_factorial = np.asarray([math.lgamma(float(value) + 1.0) for value in observations])
    log_weights = observations[:, None] * np.log(means[None, :]) - means[None, :] - log_factorial[:, None]
    row_max = np.max(log_weights, axis=1)
    return np.exp(log_weights - row_max[:, None]), row_max


def poisson_hmm_em(
    observations: np.ndarray,
    states: int,
    iterations: int,
    dt: float,
    persistence: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit a discrete-time Poisson HMM with scaled forward-backward EM."""
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 1 or len(observations) < 2:
        raise ValueError("Poisson-HMM calibration requires at least two observations")
    quantiles = np.linspace(0.08, 0.92, states)
    rates = np.maximum(0.02, np.quantile(observations, quantiles) / dt)
    rates += np.linspace(0.0, 1e-3, states)
    if states == 1:
        transition = np.ones((1, 1), dtype=float)
    else:
        transition = np.full((states, states), (1.0 - persistence) / (states - 1), dtype=float)
        np.fill_diagonal(transition, persistence)
    initial = stationary_distribution(transition)
    log_likelihood = float("-inf")

    for _ in range(max(1, iterations)):
        emission, row_max = poisson_emission_weights(observations, rates, dt)
        length = len(observations)
        alpha = np.empty((length, states), dtype=float)
        scales = np.empty(length, dtype=float)
        alpha[0] = initial * emission[0]
        scales[0] = max(1e-300, float(alpha[0].sum()))
        alpha[0] /= scales[0]
        for index in range(1, length):
            alpha[index] = (alpha[index - 1] @ transition) * emission[index]
            scales[index] = max(1e-300, float(alpha[index].sum()))
            alpha[index] /= scales[index]

        beta = np.ones((length, states), dtype=float)
        for index in range(length - 2, -1, -1):
            beta[index] = transition @ (emission[index + 1] * beta[index + 1])
            beta[index] /= scales[index + 1]
        gamma = alpha * beta
        gamma /= np.maximum(1e-300, gamma.sum(axis=1, keepdims=True))

        transition_counts = np.zeros_like(transition)
        for index in range(length - 1):
            right = emission[index + 1] * beta[index + 1]
            xi = alpha[index, :, None] * transition * right[None, :]
            denominator = max(1e-300, float(xi.sum()))
            transition_counts += xi / denominator
        transition = transition_counts + 1e-8
        transition /= transition.sum(axis=1, keepdims=True)
        rates = (gamma.T @ observations) / np.maximum(1e-8, gamma.sum(axis=0) * dt)
        rates = np.maximum(0.01, rates)
        initial = stationary_distribution(transition)
        log_likelihood = float(np.sum(np.log(scales) + row_max))

    order = np.argsort(rates)
    rates = rates[order]
    transition = transition[np.ix_(order, order)]
    return rates, transition, log_likelihood


def fit_poisson_hmm(
    observations: np.ndarray,
    states: int,
    iterations: int,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    fits = [
        poisson_hmm_em(observations, states, iterations, dt, persistence)
        for persistence in (0.80, 0.95)
    ]
    return max(fits, key=lambda fit: fit[2])


def align_estimated_states(estimated_rates: np.ndarray, true_rates: np.ndarray) -> Tuple[int, ...]:
    """Return estimated-state -> true-state mapping for diagnostics only."""
    states = len(true_rates)
    if len(estimated_rates) != states:
        raise ValueError("estimated and true state counts differ")
    best_mapping: Tuple[int, ...] | None = None
    best_cost = float("inf")
    for mapping in itertools.permutations(range(states)):
        cost = sum(
            abs(math.log(max(1e-8, estimated_rates[estimated])) - math.log(max(1e-8, true_rates[true])))
            for estimated, true in enumerate(mapping)
        )
        if cost < best_cost:
            best_cost = cost
            best_mapping = mapping
    assert best_mapping is not None
    return best_mapping


def generate_calibration_counts(problem, steps: int, seed: int) -> List[np.ndarray]:
    """Generate an arrival-only trace independent of all evaluation seeds."""
    rng = np.random.default_rng(seed)
    observations: List[np.ndarray] = []
    for rates, Q in zip(problem.arrival_rates, problem.arrival_Q):
        transition = exact_phase_transition(Q)
        stationary = stationary_distribution(transition)
        state = int(rng.choice(len(rates), p=stationary))
        counts = np.empty(steps, dtype=int)
        for index in range(steps):
            counts[index] = int(rng.poisson(max(0.0, float(rates[state]))))
            state = int(rng.choice(len(rates), p=transition[state]))
        observations.append(counts)
    return observations


def estimate_case_model(problem, calibration_steps: int, em_iterations: int, seed: int):
    observations = generate_calibration_counts(problem, calibration_steps, seed)
    rates_out: List[np.ndarray] = []
    transitions_out: List[np.ndarray] = []
    mappings: List[Tuple[int, ...]] = []
    calibration_rows: List[Dict[str, object]] = []
    for queue, (counts, true_rates, Q) in enumerate(zip(observations, problem.arrival_rates, problem.arrival_Q)):
        estimated_rates, estimated_transition, log_likelihood = fit_poisson_hmm(
            counts,
            len(true_rates),
            em_iterations,
            1.0,
        )
        mapping = align_estimated_states(estimated_rates, true_rates)
        aligned_rates = np.zeros_like(true_rates, dtype=float)
        aligned_transition = np.zeros_like(estimated_transition, dtype=float)
        for estimated, true in enumerate(mapping):
            aligned_rates[true] = estimated_rates[estimated]
            for estimated_next, true_next in enumerate(mapping):
                aligned_transition[true, true_next] = estimated_transition[estimated, estimated_next]
        true_transition = exact_phase_transition(Q)
        calibration_rows.append({
            "queue": queue,
            "states": len(true_rates),
            "calibration_steps": calibration_steps,
            "em_iterations": em_iterations,
            "log_likelihood_per_step": log_likelihood / calibration_steps,
            "arrival_rate_rmse": float(np.sqrt(np.mean((aligned_rates - true_rates) ** 2))),
            "transition_mae": float(np.mean(np.abs(aligned_transition - true_transition))),
            "true_stationary_mean": float(stationary_distribution(true_transition) @ true_rates),
            "estimated_stationary_mean": float(stationary_distribution(estimated_transition) @ estimated_rates),
        })
        rates_out.append(estimated_rates)
        transitions_out.append(estimated_transition)
        mappings.append(mapping)
    return rates_out, transitions_out, mappings, calibration_rows


class PhaseFilterEnv(SparseMapPhEnv):
    """Scheduling environment whose forecast consumes an online phase belief."""

    def __init__(
        self,
        problem,
        seed: int,
        regime: str,
        model_rates: Sequence[np.ndarray] | None = None,
        model_transitions: Sequence[np.ndarray] | None = None,
        estimated_to_true: Sequence[Tuple[int, ...]] | None = None,
    ):
        super().__init__(problem, seed)
        self.base_seed = int(seed)
        self.arrival_rng = np.random.default_rng(self.base_seed + 100000000)
        self.phase_rng = np.random.default_rng(self.base_seed + 200000000)
        self.service_rng = np.random.default_rng(self.base_seed + 300000000)
        self.regime = regime
        if regime == "bayes_known":
            self.model_rates = [rates.copy() for rates in problem.arrival_rates]
            self.filter_transitions = [exact_phase_transition(Q) for Q in problem.arrival_Q]
            self.forecast_transitions = [self._transition_matrix(Q) for Q in problem.arrival_Q]
            self.estimated_to_true = [tuple(range(len(rates))) for rates in problem.arrival_rates]
        elif regime == "bayes_estimated":
            if model_rates is None or model_transitions is None or estimated_to_true is None:
                raise ValueError("estimated filter requires rates, transitions, and state mappings")
            self.model_rates = [np.asarray(rates, dtype=float).copy() for rates in model_rates]
            self.filter_transitions = [np.asarray(P, dtype=float).copy() for P in model_transitions]
            self.forecast_transitions = [np.asarray(P, dtype=float).copy() for P in model_transitions]
            self.estimated_to_true = list(estimated_to_true)
        else:
            self.model_rates = [rates.copy() for rates in problem.arrival_rates]
            self.filter_transitions = [exact_phase_transition(Q) for Q in problem.arrival_Q]
            self.forecast_transitions = [self._transition_matrix(Q) for Q in problem.arrival_Q]
            self.estimated_to_true = [tuple(range(len(rates))) for rates in problem.arrival_rates]
        self.stationary_beliefs = [stationary_distribution(P) for P in self.filter_transitions]
        self.beliefs = [belief.copy() for belief in self.stationary_beliefs]
        self.model_stationary_means = [
            float(stationary_distribution(P) @ rates)
            for rates, P in zip(self.model_rates, self.filter_transitions)
        ]

    def reset(self):
        self.rng = np.random.default_rng(self.base_seed)
        self.arrival_rng = np.random.default_rng(self.base_seed + 100000000)
        self.phase_rng = np.random.default_rng(self.base_seed + 200000000)
        self.service_rng = np.random.default_rng(self.base_seed + 300000000)
        state = super().reset()
        self.beliefs = [belief.copy() for belief in self.stationary_beliefs]
        return state

    def _advance_map_phases(self) -> None:
        """Advance a policy-independent phase stream shared across regimes."""
        assert self.state is not None
        for queue, Q in enumerate(self.problem.arrival_Q):
            current = int(self.state.phase[queue])
            departure_rate = float(-Q[current, current])
            if departure_rate <= 0.0:
                continue
            if self.phase_rng.random() >= 1.0 - math.exp(-departure_rate * self.dt):
                continue
            probabilities = np.maximum(Q[current].copy(), 0.0)
            probabilities[current] = 0.0
            probabilities /= probabilities.sum()
            self.state.phase[queue] = int(self.phase_rng.choice(len(probabilities), p=probabilities))

    def _physical_step(self, action: np.ndarray) -> Dict[str, float]:
        """Step with arrival/phase RNG streams independent of policy actions."""
        assert self.state is not None
        problem = self.problem
        state = self.state
        action = np.asarray(action, dtype=int)
        if action.shape != (problem.K,):
            raise ValueError(f"action shape {action.shape}, expected {(problem.K,)}")

        switch_penalty = 0.0
        switches = 0
        for server in range(problem.K):
            desired = int(action[server]) - 1
            if desired >= 0 and not problem.compat[server, desired]:
                desired = -1
            if desired != int(state.assigned[server]):
                if desired >= 0 or int(state.assigned[server]) >= 0:
                    switch_penalty += float(problem.switch_cost[server])
                    switches += 1
                state.assigned[server] = desired
                state.cooldown[server] = int(problem.switch_lag[server])

        previous_queue = state.q.copy()
        service = np.zeros(problem.d, dtype=int)
        for server in range(problem.K):
            if state.cooldown[server] > 0:
                state.cooldown[server] -= 1
                continue
            queue = int(state.assigned[server])
            if queue < 0:
                continue
            mean = max(0.0, float(problem.servers[server] * problem.service_mu[server, queue] * self.dt))
            if mean <= 0.0:
                continue
            coefficient_of_variation = max(0.05, float(problem.service_cv[server, queue]))
            if coefficient_of_variation <= 1.05:
                completion = self.service_rng.poisson(mean)
            else:
                shape = 1.0 / max(1e-6, coefficient_of_variation**2 - 1.0)
                scale = mean / shape
                completion = self.service_rng.poisson(self.service_rng.gamma(shape, scale))
            service[queue] += int(completion)

        rates = self.current_arrival_rates()
        arrivals = self.arrival_rng.poisson(np.maximum(0.0, rates * self.dt)).astype(int)
        self.last_arrivals_by_queue = arrivals.copy()
        self._advance_map_phases()
        served = np.minimum(previous_queue, service)
        uncapped_queue = previous_queue - served + arrivals
        assert problem.buffer_cap is not None
        dropped = np.maximum(0, uncapped_queue - problem.buffer_cap)
        state.q = np.clip(uncapped_queue, 0, problem.buffer_cap).astype(int)
        overflow = float(np.any(state.q >= problem.buffer_cap))
        cap_queue_fraction = float(np.mean(state.q >= problem.buffer_cap))
        in_service = np.zeros(problem.d, dtype=float)
        for server in range(problem.K):
            queue = int(state.assigned[server])
            if queue >= 0 and state.cooldown[server] == 0:
                in_service[queue] += problem.servers[server]
        waiting = np.maximum(0.0, state.q.astype(float) - in_service)
        return {
            "total_queue": float(state.q.sum()),
            "waiting_queue": float(waiting.sum()),
            "switch_penalty": float(switch_penalty),
            "switches": float(switches),
            "arrivals": float(arrivals.sum()),
            "admitted_arrivals": float(arrivals.sum() - dropped.sum()),
            "dropped_arrivals": float(dropped.sum()),
            "served": float(served.sum()),
            "overflow": overflow,
            "cap_queue_fraction": cap_queue_fraction,
        }

    def step(self, action: np.ndarray) -> Dict[str, float]:
        info = self._physical_step(action)
        if self.regime in {"bayes_known", "bayes_estimated"}:
            updated: List[np.ndarray] = []
            for belief, count, rates, transition in zip(
                self.beliefs,
                self.last_arrivals_by_queue,
                self.model_rates,
                self.filter_transitions,
            ):
                means = np.maximum(1e-8, rates * self.dt)
                log_likelihood = float(count) * np.log(means) - means
                likelihood = np.exp(log_likelihood - float(np.max(log_likelihood)))
                posterior = belief * likelihood
                total = float(posterior.sum())
                posterior = posterior / total if total > 0.0 else belief
                prediction = posterior @ transition
                prediction /= max(1e-300, float(prediction.sum()))
                updated.append(prediction)
            self.beliefs = updated
        return info

    def forecast_arrival_excess(self, horizon: int = 6) -> np.ndarray:
        if self.regime == "oracle":
            return super().forecast_arrival_excess(horizon)
        if self.regime == "stationary":
            return np.zeros(self.problem.d, dtype=float)
        output = np.zeros(self.problem.d, dtype=float)
        for queue, (belief, rates, transition, stationary_mean) in enumerate(zip(
            self.beliefs,
            self.model_rates,
            self.forecast_transitions,
            self.model_stationary_means,
        )):
            distribution = belief.copy()
            expected = 0.0
            for _ in range(max(1, horizon)):
                expected += float(distribution @ rates) * self.dt
                distribution = distribution @ transition
            baseline = stationary_mean * self.dt * max(1, horizon)
            output[queue] = max(0.0, expected - baseline)
        return output

    def true_aligned_beliefs(self) -> List[np.ndarray]:
        assert self.state is not None
        if self.regime == "oracle":
            output = []
            for queue, rates in enumerate(self.problem.arrival_rates):
                belief = np.zeros(len(rates), dtype=float)
                belief[int(self.state.phase[queue])] = 1.0
                output.append(belief)
            return output
        if self.regime == "stationary":
            return [
                stationary_distribution(exact_phase_transition(Q))
                for Q in self.problem.arrival_Q
            ]
        output = []
        for belief, mapping in zip(self.beliefs, self.estimated_to_true):
            aligned = np.zeros_like(belief)
            for estimated, true in enumerate(mapping):
                aligned[true] = belief[estimated]
            output.append(aligned)
        return output

    def inference_diagnostics(self) -> Dict[str, float]:
        assert self.state is not None
        correct = 0.0
        brier = 0.0
        negative_log_likelihood = 0.0
        for queue, belief in enumerate(self.true_aligned_beliefs()):
            true = int(self.state.phase[queue])
            target = np.zeros_like(belief)
            target[true] = 1.0
            correct += float(int(np.argmax(belief)) == true)
            brier += float(np.sum((belief - target) ** 2))
            negative_log_likelihood += -math.log(max(1e-12, float(belief[true])))
        queues = max(1, self.problem.d)
        return {
            "phase_map_accuracy": correct / queues,
            "phase_brier_score": brier / queues,
            "phase_negative_log_likelihood": negative_log_likelihood / queues,
        }


def simulate_filter_policy(problem, env: PhaseFilterEnv, warmup_steps: int, eval_steps: int) -> Dict[str, float]:
    policy = policy_from_name("gpah")
    policy.reset(problem)
    for _ in range(warmup_steps):
        env.step(policy.action(env))
    sums = {
        key: 0.0
        for key in (
            "total_queue",
            "waiting_queue",
            "switch_penalty",
            "switches",
            "arrivals",
            "admitted_arrivals",
            "dropped_arrivals",
            "served",
            "overflow",
            "cap_queue_fraction",
        )
    }
    inference_sums = {
        "phase_map_accuracy": 0.0,
        "phase_brier_score": 0.0,
        "phase_negative_log_likelihood": 0.0,
    }
    queue_samples: List[float] = []
    for _ in range(eval_steps):
        diagnostics = env.inference_diagnostics()
        info = env.step(policy.action(env))
        for key in sums:
            sums[key] += float(info[key])
        for key in inference_sums:
            inference_sums[key] += diagnostics[key]
        queue_samples.append(float(info["total_queue"]))
    count = max(1, eval_steps)
    mean_total_queue = sums["total_queue"] / count
    mean_switch_cost = sums["switch_penalty"] / count
    cap_time_fraction = sums["overflow"] / count
    assert problem.buffer_cap is not None
    queue_scale = max(1.0, float(problem.buffer_cap * problem.d))
    switch_scale = max(1.0, float(np.sum(problem.switch_cost)))
    objective = mean_total_queue / queue_scale + mean_switch_cost / switch_scale + 3.0 * cap_time_fraction
    return {
        "mean_total_queue": mean_total_queue,
        "mean_waiting_queue": sums["waiting_queue"] / count,
        "p95_total_queue": float(np.quantile(np.asarray(queue_samples), 0.95)),
        "mean_switch_cost": mean_switch_cost,
        "switch_rate": sums["switches"] / count,
        "mean_arrivals": sums["arrivals"] / count,
        "mean_served": sums["served"] / count,
        "drop_rate": sums["dropped_arrivals"] / max(1.0, sums["arrivals"]),
        "cap_time_fraction": cap_time_fraction,
        "objective": objective,
        **{key: value / count for key, value in inference_sums.items()},
    }


def select_cases(case_ids: Sequence[str]) -> List[CaseSpec]:
    cases = policy_cases(RunConfig(output_dir="", families=("A2_map_burst",)))
    by_id = {case.case_id: case for case in cases}
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise ValueError("unknown A2 case ids: " + ", ".join(missing))
    return [by_id[case_id] for case_id in case_ids]


def summarize(replications: Sequence[Dict[str, object]], regimes: Sequence[str]):
    oracle = {
        (str(row["case_id"]), int(row["replication"])): row
        for row in replications
        if row["inference_regime"] == "oracle"
    }
    metrics = (
        "objective",
        "mean_total_queue",
        "mean_switch_cost",
        "switch_rate",
        "phase_map_accuracy",
        "phase_brier_score",
        "phase_negative_log_likelihood",
    )
    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in replications:
        groups.setdefault((str(row["case_id"]), str(row["inference_regime"])), []).append(row)
    case_rows: List[Dict[str, object]] = []
    for (case_id, regime), rows in groups.items():
        result: Dict[str, object] = {
            "case_id": case_id,
            "inference_regime": regime,
            "regime_label": REGIME_LABELS[regime],
            "n": len(rows),
        }
        for metric in metrics:
            stats = mean_ci([float(row[metric]) for row in rows])
            result.update({f"{metric}_{name}": value for name, value in stats.items()})
        gaps = [
            float(row["objective"]) - float(oracle[(case_id, int(row["replication"]))]["objective"])
            for row in rows
        ]
        result.update({f"paired_objective_gap_{name}": value for name, value in mean_ci(gaps).items()})
        case_rows.append(result)

    overall_rows: List[Dict[str, object]] = []
    for regime in regimes:
        rows = [row for row in replications if row["inference_regime"] == regime]
        by_replication: Dict[int, List[Dict[str, object]]] = {}
        for row in rows:
            by_replication.setdefault(int(row["replication"]), []).append(row)
        aggregate: Dict[str, List[float]] = {
            "paired_objective_gap": [],
            "phase_map_accuracy": [],
            "phase_brier_score": [],
            "phase_negative_log_likelihood": [],
        }
        for cells in by_replication.values():
            aggregate["paired_objective_gap"].append(float(np.mean([
                float(row["objective"])
                - float(oracle[(str(row["case_id"]), int(row["replication"]))]["objective"])
                for row in cells
            ])))
            for metric in ("phase_map_accuracy", "phase_brier_score", "phase_negative_log_likelihood"):
                aggregate[metric].append(float(np.mean([float(row[metric]) for row in cells])))
        result = {
            "inference_regime": regime,
            "regime_label": REGIME_LABELS[regime],
            "case_count": len({str(row["case_id"]) for row in rows}),
            "replications": len(by_replication),
        }
        for metric, values in aggregate.items():
            result.update({f"equal_case_{metric}_{name}": value for name, value in mean_ci(values).items()})
        overall_rows.append(result)
    return case_rows, overall_rows


def plot_results(rows: Sequence[Mapping[str, object]], output_path: str) -> None:
    regimes = [str(row["inference_regime"]) for row in rows]
    labels = [str(row["regime_label"]) for row in rows]
    x = np.arange(len(regimes))
    colors = [PAPER_COLORS["c02"], PAPER_COLORS["c05"], PAPER_COLORS["c03"], PAPER_COLORS["c07"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].bar(
        x,
        [float(row["equal_case_paired_objective_gap_mean"]) for row in rows],
        yerr=[float(row["equal_case_paired_objective_gap_ci95"]) for row in rows],
        color=colors[: len(rows)],
        capsize=3,
    )
    axes[0].axhline(0.0, color=PAPER_COLORS["text"], linewidth=1.0)
    axes[0].set_ylabel("Paired standardized-objective gap vs oracle")
    axes[1].bar(
        x,
        [float(row["equal_case_phase_map_accuracy_mean"]) for row in rows],
        yerr=[float(row["equal_case_phase_map_accuracy_ci95"]) for row in rows],
        color=colors[: len(rows)],
        capsize=3,
    )
    axes[1].set_ylabel("Hidden-phase MAP accuracy")
    axes[1].set_ylim(0.0, 1.05)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=24, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--case-ids", default=",".join(DEFAULT_CASE_IDS))
    parser.add_argument("--regimes", default=DEFAULT_REGIMES)
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=3000)
    parser.add_argument("--calibration-steps", type=int, default=10000)
    parser.add_argument("--em-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=34260527)
    args = parser.parse_args()
    if min(args.replications, args.eval_steps, args.calibration_steps, args.em_iterations) < 1:
        raise ValueError("replications, eval-steps, calibration-steps, and em-iterations must be positive")
    regimes = [item.strip() for item in args.regimes.split(",") if item.strip()]
    invalid = sorted(set(regimes) - set(REGIME_LABELS))
    if invalid:
        raise ValueError("unknown inference regimes: " + ", ".join(invalid))
    if "oracle" not in regimes:
        raise ValueError("--regimes must include oracle for paired comparisons")
    case_ids = [item.strip() for item in args.case_ids.split(",") if item.strip()]
    cases = select_cases(case_ids)
    os.makedirs(args.output_dir, exist_ok=True)

    replication_rows: List[Dict[str, object]] = []
    calibration_rows: List[Dict[str, object]] = []
    total = len(cases) * len(regimes) * args.replications
    done = 0
    for case_index, case in enumerate(cases, 1):
        problem = make_problem(case, args.seed + 1009 * case_index)
        estimated_rates = None
        estimated_transitions = None
        estimated_mappings = None
        if "bayes_estimated" in regimes:
            estimated_rates, estimated_transitions, estimated_mappings, rows = estimate_case_model(
                problem,
                args.calibration_steps,
                args.em_iterations,
                args.seed + 50000000 + 100000 * case_index,
            )
            calibration_rows.extend({"case_id": case.case_id, **row} for row in rows)
        for replication in range(args.replications):
            simulation_seed = args.seed + 100000 * case_index + replication
            for regime in regimes:
                env = PhaseFilterEnv(
                    copy.deepcopy(problem),
                    simulation_seed,
                    regime,
                    estimated_rates,
                    estimated_transitions,
                    estimated_mappings,
                )
                env.reset()
                metrics = simulate_filter_policy(problem, env, args.warmup_steps, args.eval_steps)
                replication_rows.append({
                    "experiment": "hidden_phase_filter_validation",
                    "family": case.family,
                    "case_id": case.case_id,
                    "map_profile": case.map_profile,
                    "inference_regime": regime,
                    "regime_label": REGIME_LABELS[regime],
                    "replication": replication,
                    "seed": simulation_seed,
                    "calibration_seed": args.seed + 50000000 + 100000 * case_index,
                    "calibration_steps": args.calibration_steps,
                    "em_iterations": args.em_iterations,
                    **metrics,
                })
                done += 1
                render_progress(done, total, f"{case.case_id} {regime}")

    case_rows, overall_rows = summarize(replication_rows, regimes)
    write_csv(os.path.join(args.output_dir, "paper1_phase_filter_replications.csv"), replication_rows)
    write_csv(os.path.join(args.output_dir, "paper1_phase_filter_case_summary.csv"), case_rows)
    write_csv(os.path.join(args.output_dir, "paper1_phase_filter_overall_summary.csv"), overall_rows)
    if calibration_rows:
        write_csv(os.path.join(args.output_dir, "paper1_phase_filter_calibration_summary.csv"), calibration_rows)
    plot_results(overall_rows, os.path.join(args.output_dir, "fig_phase_filter_validation.pdf"))
    manifest = {
        "experiment": "hidden_phase_filter_validation",
        "decision_information": "per-queue arrival counts and queueing state; true phase excluded from Bayesian-filter forecasts",
        "true_phase_use": "evaluation-only identification diagnostics; never policy input",
        "calibration_isolation": "estimated parameters use an independent arrival-only trace and disjoint seed range",
        "paired_exogenous_streams": "arrival and phase RNG streams are identical across inference regimes and independent of policy-dependent service draws",
        "case_ids": case_ids,
        "regimes": regimes,
        "replications": args.replications,
        "warmup_steps": args.warmup_steps,
        "eval_steps": args.eval_steps,
        "calibration_steps": args.calibration_steps,
        "em_iterations": args.em_iterations,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Hidden-phase filter validation results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
