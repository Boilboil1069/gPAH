#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


POLICY_LABELS: Dict[str, str] = {
    "equal_split": "Equal-Split",
    "qah": "QAH",
    "backpressure": "BackPressure",
    "maxweight_mu": "MaxWeight-mu",
    "cmu": "c-mu",
    "affinity": "GB-PANDAS-style",
    "rl_queueing": "Queueing-Surrogate",
    "sah": "SAH",
    "qah_switch": "QAH+Switch",
    "gpah_meanmap": "gPAH-StationaryMean",
    "gpah": "gPAH",
    "gpah_full": "gPAH-Full",
    "no_phase": "Abl-NoPhase",
    "no_service": "Abl-NoService",
    "no_scarcity": "Abl-NoScarcity",
    "no_switch": "Abl-NoSwitch",
    "no_queue": "Abl-NoQueue",
    "mean_map": "Abl-StationaryMean",
    "dqn": "DQN-Candidate",
    "ppo": "PPO-Factorized",
    "sac": "Discrete-SAC-Factorized",
}

PAPER_COLORS: Dict[str, str] = {
    "server": "#7FABD1",
    "queue": "#91CCC0",
    "highlight": "#EC6E66",
    "edge": "#C7C1DE",
    "text": "#4f4f4f",
    "grid": "#C7C1DE",
    "background": "#ffffff",
    "c01": "#91CCC0",
    "c02": "#7FABD1",
    "c03": "#F7AC53",
    "c04": "#EC6E66",
    "c05": "#B5CE4E",
    "c06": "#BD7795",
    "c07": "#7C7979",
    "c08": "#963B79",
    "c09": "#97D0C5",
    "c10": "#F39865",
    "c11": "#52AADC",
    "c12": "#C7C1DE",
    "c13": "#EEB6D4",
    "c14": "#C89736",
    "c15": "#2D8875",
    "c16": "#D75B4E",
    "soft_gray": "#C7C1DE",
}

POLICY_COLORS: Dict[str, str] = {
    "gpah": PAPER_COLORS["highlight"],
    "gpah_full": PAPER_COLORS["highlight"],
    "gpah_meanmap": PAPER_COLORS["c10"],
    "mean_map": PAPER_COLORS["c10"],
    "qah": PAPER_COLORS["server"],
    "backpressure": PAPER_COLORS["server"],
    "maxweight_mu": PAPER_COLORS["c02"],
    "cmu": PAPER_COLORS["c11"],
    "affinity": PAPER_COLORS["c05"],
    "rl_queueing": PAPER_COLORS["c13"],
    "sah": PAPER_COLORS["c12"],
    "qah_switch": PAPER_COLORS["c03"],
    "equal_split": PAPER_COLORS["c07"],
    "no_phase": PAPER_COLORS["c11"],
    "no_service": PAPER_COLORS["c02"],
    "no_scarcity": PAPER_COLORS["server"],
    "no_switch": PAPER_COLORS["c03"],
    "no_queue": PAPER_COLORS["c16"],
    "dqn": PAPER_COLORS["c12"],
    "ppo": PAPER_COLORS["c13"],
    "sac": PAPER_COLORS["c11"],
}


def order_policies_for_legend(policies: Iterable[str]) -> List[str]:
    ordered = list(policies)
    for preferred_last in ("gpah", "gpah_full"):
        if preferred_last in ordered:
            ordered = [p for p in ordered if p != preferred_last] + [preferred_last]
    return ordered


@dataclass
class Problem:
    d: int
    K: int
    compat: np.ndarray
    servers: np.ndarray
    service_mu: np.ndarray
    service_cv: np.ndarray
    arrival_rates: List[np.ndarray]
    arrival_Q: List[np.ndarray]
    arrival_mean: np.ndarray
    switch_cost: np.ndarray
    switch_lag: np.ndarray
    # ``None`` is reserved for the uncapped high-load diagnostic. All main
    # paper experiments retain the finite default B=2000.
    buffer_cap: Optional[int] = 2000


@dataclass
class State:
    q: np.ndarray
    phase: np.ndarray
    assigned: np.ndarray
    cooldown: np.ndarray


@dataclass
class CaseSpec:
    family: str
    case_id: str
    description: str
    d: int
    K: int
    topology: str = "chain"
    map_profile: str = "medium_burst"
    density: float = 0.4
    service_profile: str = "mild"
    switch_profile: str = "moderate"
    load: float = 0.82
    component: str = ""
    regime: str = ""


@dataclass
class ScoreWeights:
    queue: float = 1.0
    phase: float = 1.6
    service: float = 1.0
    scarcity: float = 0.55
    switch_cost: float = 0.9
    switch_lag: float = 0.35


ABLATION_WEIGHTS = ScoreWeights(
    queue=1.0,
    phase=1.5,
    service=1.25,
    scarcity=0.25,
    switch_cost=1.25,
    switch_lag=0.8,
)


@dataclass
class RunConfig:
    output_dir: str
    replications: int = 50
    warmup_steps: int = 500
    eval_steps: int = 3000
    seed: int = 20260507
    d_values: Tuple[int, ...] = (6, 8, 12)
    K_values: Tuple[int, ...] = (4, 6)
    families: Tuple[str, ...] = ()
    policies: Tuple[str, ...] = ()


SENSITIVITY_PARAMETERS: Dict[str, Tuple[str, str]] = {
    "alpha_Q": ("queue", "queue backlog"),
    "alpha_A": ("phase", "MMPP arrival forecast"),
    "alpha_mu": ("service", "service-rate matching"),
    "alpha_S": ("scarcity", "compatibility scarcity"),
    "alpha_C": ("switch_cost", "switching cost"),
    "alpha_L": ("switch_lag", "switching lag"),
}


class SparseMapPhEnv:
    def __init__(self, problem: Problem, seed: int, dt: float = 1.0):
        self.problem = problem
        self.rng = np.random.default_rng(seed)
        self.dt = float(dt)
        self.state: Optional[State] = None
        # Per-queue arrivals are observable after each interval and support
        # hidden-phase filtering without exposing the simulator's true phase.
        self.last_arrivals_by_queue = np.zeros(problem.d, dtype=int)

    def reset(self) -> State:
        p = self.problem
        self.state = State(
            q=self.rng.integers(0, 6, size=p.d).astype(int),
            phase=np.array([self.rng.integers(0, len(r)) for r in p.arrival_rates], dtype=int),
            assigned=np.full(p.K, -1, dtype=int),
            cooldown=np.zeros(p.K, dtype=int),
        )
        self.last_arrivals_by_queue = np.zeros(p.d, dtype=int)
        return self.copy_state()

    def copy_state(self) -> State:
        assert self.state is not None
        s = self.state
        return State(s.q.copy(), s.phase.copy(), s.assigned.copy(), s.cooldown.copy())

    def current_arrival_rates(self) -> np.ndarray:
        assert self.state is not None
        return np.array([self.problem.arrival_rates[i][self.state.phase[i]] for i in range(self.problem.d)], dtype=float)

    def _transition_matrix(self, Q: np.ndarray) -> np.ndarray:
        P = np.eye(Q.shape[0]) + Q * self.dt
        P = np.maximum(P, 0.0)
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        return P / row_sums

    def forecast_arrival_excess(self, horizon: int = 6) -> np.ndarray:
        assert self.state is not None
        p = self.problem
        out = np.zeros(p.d, dtype=float)
        for i, (rates, Q) in enumerate(zip(p.arrival_rates, p.arrival_Q)):
            P = self._transition_matrix(Q)
            dist = np.zeros(len(rates), dtype=float)
            dist[int(self.state.phase[i])] = 1.0
            expected = 0.0
            for _ in range(max(1, horizon)):
                expected += float(dist @ rates) * self.dt
                dist = dist @ P
            baseline = float(p.arrival_mean[i]) * self.dt * max(1, horizon)
            out[i] = max(0.0, expected - baseline)
        return out

    def _advance_map_phases(self) -> None:
        assert self.state is not None
        for i, Q in enumerate(self.problem.arrival_Q):
            cur = int(self.state.phase[i])
            out = float(-Q[cur, cur])
            if out <= 0:
                continue
            if self.rng.random() >= 1.0 - math.exp(-out * self.dt):
                continue
            probs = np.maximum(Q[cur].copy(), 0.0)
            probs[cur] = 0.0
            probs = probs / probs.sum()
            self.state.phase[i] = int(self.rng.choice(len(probs), p=probs))

    def step(self, action: np.ndarray) -> Dict[str, float]:
        assert self.state is not None
        p = self.problem
        s = self.state
        action = np.asarray(action, dtype=int)
        if action.shape != (p.K,):
            raise ValueError(f"action shape {action.shape}, expected {(p.K,)}")

        switch_penalty = 0.0
        switches = 0
        for k in range(p.K):
            desired = int(action[k]) - 1
            if desired >= 0 and not p.compat[k, desired]:
                desired = -1
            if desired != int(s.assigned[k]):
                if desired >= 0 or int(s.assigned[k]) >= 0:
                    switch_penalty += float(p.switch_cost[k])
                    switches += 1
                s.assigned[k] = desired
                s.cooldown[k] = int(p.switch_lag[k])

        prev_q = s.q.copy()
        service = np.zeros(p.d, dtype=int)
        for k in range(p.K):
            if s.cooldown[k] > 0:
                s.cooldown[k] -= 1
                continue
            i = int(s.assigned[k])
            if i < 0:
                continue
            mean = max(0.0, float(p.servers[k] * p.service_mu[k, i] * self.dt))
            if mean <= 0:
                continue
            cv = max(0.05, float(p.service_cv[k, i]))
            if cv <= 1.05:
                comp = self.rng.poisson(mean)
            else:
                shape = 1.0 / max(1e-6, cv * cv - 1.0)
                scale = mean / shape
                comp = self.rng.poisson(self.rng.gamma(shape, scale))
            service[i] += int(comp)

        rates = self.current_arrival_rates()
        arrivals = self.rng.poisson(np.maximum(0.0, rates * self.dt)).astype(int)
        self.last_arrivals_by_queue = arrivals.copy()
        self._advance_map_phases()

        served = np.minimum(prev_q, service)
        uncapped_q = prev_q - served + arrivals
        if p.buffer_cap is None:
            dropped = np.zeros_like(uncapped_q)
            s.q = np.maximum(0, uncapped_q).astype(int)
            overflow = 0.0
            cap_queue_fraction = 0.0
        else:
            dropped = np.maximum(0, uncapped_q - p.buffer_cap)
            s.q = np.clip(uncapped_q, 0, p.buffer_cap).astype(int)
            overflow = float(np.any(s.q >= p.buffer_cap))
            cap_queue_fraction = float(np.mean(s.q >= p.buffer_cap))
        in_service = np.zeros(p.d, dtype=float)
        for k in range(p.K):
            i = int(s.assigned[k])
            if i >= 0 and s.cooldown[k] == 0:
                in_service[i] += p.servers[k]
        waiting = np.maximum(0.0, s.q.astype(float) - in_service)
        return {
            "total_queue": float(s.q.sum()),
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


class Policy:
    def reset(self, problem: Problem) -> None:
        return None

    def action(self, env: SparseMapPhEnv) -> np.ndarray:
        raise NotImplementedError


class EqualSplitPolicy(Policy):
    def reset(self, problem: Problem) -> None:
        self.assignment = np.zeros(problem.K, dtype=int)
        load = problem.arrival_mean.copy()
        for k in range(problem.K):
            candidates = [i for i in range(problem.d) if problem.compat[k, i]]
            if not candidates:
                self.assignment[k] = 0
                continue
            best = max(candidates, key=lambda i: load[i] / max(0.1, problem.service_mu[k, i]))
            self.assignment[k] = best + 1
            load[best] = max(0.0, load[best] - problem.servers[k] * problem.service_mu[k, best])

    def action(self, env: SparseMapPhEnv) -> np.ndarray:
        return self.assignment.copy()


class ScorePolicy(Policy):
    def __init__(self, weights: ScoreWeights, use_current_map: bool = True, label: str = "score"):
        self.weights = weights
        self.use_current_map = use_current_map
        self.label = label

    def action(self, env: SparseMapPhEnv) -> np.ndarray:
        assert env.state is not None
        p = env.problem
        s = env.state
        q = s.q.astype(float)
        phase_pressure = env.forecast_arrival_excess() if self.use_current_map else np.zeros(p.d, dtype=float)
        degrees = np.maximum(1.0, p.compat.sum(axis=0).astype(float))
        scarcity = 1.0 / degrees
        action = np.zeros(p.K, dtype=int)
        allocated = np.zeros(p.d, dtype=float)
        for k in np.argsort(-p.servers):
            best_score = 0.0
            best = 0
            for i in range(p.d):
                if not p.compat[k, i]:
                    continue
                switching = int(s.assigned[k]) != i
                pressure = self.weights.queue * q[i] + self.weights.phase * phase_pressure[i]
                score = pressure
                score += self.weights.service * pressure * p.service_mu[k, i]
                score += self.weights.scarcity * scarcity[i] * max(1.0, pressure)
                score -= 0.25 * allocated[i]
                if switching:
                    score -= self.weights.switch_cost * p.switch_cost[k]
                    score -= self.weights.switch_lag * p.switch_lag[k]
                if score > best_score:
                    best_score = score
                    best = i + 1
            action[k] = best
            if best > 0:
                i = best - 1
                allocated[i] += p.servers[k] * p.service_mu[k, i]
        return action


class MaxWeightMuPolicy(Policy):
    def action(self, env: SparseMapPhEnv) -> np.ndarray:
        assert env.state is not None
        p = env.problem
        q = env.state.q.astype(float)
        action = np.zeros(p.K, dtype=int)
        for k in np.argsort(-p.servers):
            best_score = 0.0
            best = 0
            for i in range(p.d):
                if not p.compat[k, i]:
                    continue
                score = q[i] * p.service_mu[k, i]
                if score > best_score:
                    best_score = score
                    best = i + 1
            action[k] = best
        return action


class AffinityPolicy(Policy):
    def action(self, env: SparseMapPhEnv) -> np.ndarray:
        assert env.state is not None
        p = env.problem
        q = env.state.q.astype(float)
        best_for_queue = np.maximum(1e-9, p.service_mu.max(axis=0))
        action = np.zeros(p.K, dtype=int)
        for k in np.argsort(-p.servers):
            best_score = 0.0
            best = 0
            for i in range(p.d):
                if not p.compat[k, i]:
                    continue
                affinity = p.service_mu[k, i] / best_for_queue[i]
                score = q[i] * (0.5 + affinity) * p.service_mu[k, i]
                if score > best_score:
                    best_score = score
                    best = i + 1
            action[k] = best
        return action


class RLQueueingSurrogatePolicy(Policy):
    """Lightweight RL-style baseline: a fixed residual score over queue features.

    This is not a trained deep RL controller. It mimics common queueing-RL
    baselines that learn a low-dimensional action score, while keeping the
    experiment reproducible without a separate training pipeline.
    """

    def action(self, env: SparseMapPhEnv) -> np.ndarray:
        assert env.state is not None
        p = env.problem
        s = env.state
        q = s.q.astype(float)
        phase_pressure = env.forecast_arrival_excess()
        degrees = np.maximum(1.0, p.compat.sum(axis=0).astype(float))
        scarcity = 1.0 / degrees
        action = np.zeros(p.K, dtype=int)
        for k in np.argsort(-p.servers):
            best_score = 0.0
            best = 0
            for i in range(p.d):
                if not p.compat[k, i]:
                    continue
                switching = int(s.assigned[k]) != i
                score = 1.10 * q[i]
                score += 0.65 * phase_pressure[i]
                score += 0.80 * q[i] * p.service_mu[k, i]
                score += 0.20 * scarcity[i] * max(1.0, q[i])
                if switching:
                    score -= 0.35 * p.switch_cost[k] + 0.15 * p.switch_lag[k]
                if score > best_score:
                    best_score = score
                    best = i + 1
            action[k] = best
        return action


def policy_from_name(name: str) -> Policy:
    if name == "equal_split":
        return EqualSplitPolicy()
    if name == "qah":
        return ScorePolicy(ScoreWeights(queue=1.0, phase=0.0, service=0.0, scarcity=0.0, switch_cost=0.0, switch_lag=0.0), False, name)
    if name == "backpressure":
        return ScorePolicy(ScoreWeights(queue=1.0, phase=0.0, service=0.0, scarcity=0.0, switch_cost=0.0, switch_lag=0.0), False, name)
    if name == "maxweight_mu":
        return MaxWeightMuPolicy()
    if name == "cmu":
        return MaxWeightMuPolicy()
    if name == "affinity":
        return AffinityPolicy()
    if name == "rl_queueing":
        return RLQueueingSurrogatePolicy()
    if name == "sah":
        return ScorePolicy(ScoreWeights(queue=0.0, phase=0.0, service=1.0, scarcity=0.0, switch_cost=0.0, switch_lag=0.0), False, name)
    if name == "qah_switch":
        return ScorePolicy(ScoreWeights(queue=1.0, phase=0.0, service=0.0, scarcity=0.0, switch_cost=0.9, switch_lag=0.35), False, name)
    if name == "gpah":
        return ScorePolicy(ScoreWeights(), True, name)
    if name == "gpah_full":
        return ScorePolicy(ABLATION_WEIGHTS, True, name)
    if name == "gpah_meanmap":
        return ScorePolicy(ScoreWeights(), False, name)
    if name == "mean_map":
        return ScorePolicy(ABLATION_WEIGHTS, False, name)
    if name == "no_phase":
        return ScorePolicy(replace(ABLATION_WEIGHTS, phase=0.0), False, name)
    if name == "no_service":
        return ScorePolicy(replace(ABLATION_WEIGHTS, service=0.0), True, name)
    if name == "no_scarcity":
        return ScorePolicy(replace(ABLATION_WEIGHTS, scarcity=0.0), True, name)
    if name == "no_switch":
        return ScorePolicy(replace(ABLATION_WEIGHTS, switch_cost=0.0, switch_lag=0.0), True, name)
    if name == "no_queue":
        return ScorePolicy(replace(ABLATION_WEIGHTS, queue=0.0), True, name)
    raise ValueError(f"unknown policy: {name}")


def gpah_policy_with_weights(weights: ScoreWeights, use_current_map: bool = True, label: str = "gpah_weighted") -> Policy:
    return ScorePolicy(weights, use_current_map, label)


def parse_csv_list(raw: str, cast=str) -> Tuple:
    if raw is None or raw == "":
        return tuple()
    return tuple(cast(x.strip()) for x in raw.split(",") if x.strip())


def repair_compat(compat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    K, d = compat.shape
    for k in range(K):
        if not compat[k].any():
            compat[k, int(rng.integers(0, d))] = True
    for i in range(d):
        if not compat[:, i].any():
            compat[int(rng.integers(0, K)), i] = True
    return compat


def topology_matrix(topology: str, d: int, K: int, density: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    compat = np.zeros((K, d), dtype=bool)
    if topology == "disjoint":
        for i in range(d):
            compat[i % K, i] = True
    elif topology == "chain":
        width = max(2, math.ceil(d / K))
        for k in range(K):
            start = int(round(k * (d - 1) / max(1, K - 1)))
            for off in range(width):
                compat[k, min(d - 1, start + off)] = True
                compat[k, max(0, start - off)] = True
    elif topology == "ring":
        width = max(2, math.ceil(d / K))
        for k in range(K):
            center = int(round(k * d / K)) % d
            for off in range(width):
                compat[k, (center + off) % d] = True
                compat[k, (center - off) % d] = True
    elif topology == "star":
        compat[0, :] = True
        for k in range(1, K):
            for i in range(d):
                compat[k, i] = (i % max(1, K - 1)) == (k - 1)
    elif topology == "bridge_bottleneck":
        mid = max(1, d // 2)
        for k in range(K):
            if k < K // 2:
                compat[k, : mid + 1] = True
            else:
                compat[k, max(0, mid - 1) :] = True
        compat[:, mid] = rng.random(K) < 0.5
        compat[K // 2, mid] = True
    elif topology == "high_overlap":
        compat = rng.random((K, d)) < max(0.65, density)
    elif topology == "random_sparse":
        compat = rng.random((K, d)) < density
    else:
        raise ValueError(f"unknown topology {topology}")
    return repair_compat(compat, rng)


def map_profile(profile: str, d: int, rng: np.random.Generator) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    def cyclic_q(n: int, forward: float, backward: float = 0.0, skip: float = 0.0) -> np.ndarray:
        q = np.zeros((n, n), dtype=float)
        for s in range(n):
            q[s, (s + 1) % n] += forward
            if backward > 0:
                q[s, (s - 1) % n] += backward
            if skip > 0:
                q[s, (s + 2) % n] += skip
        np.fill_diagonal(q, -q.sum(axis=1))
        return q

    if profile == "low_burst":
        low, high, flip = 0.65, 1.45, 0.18
    elif profile == "medium_burst":
        low, high, flip = 0.25, 2.4, 0.08
    elif profile == "high_burst":
        low, high, flip = 0.06, 4.4, 0.025
    elif profile == "low_contrast":
        low, high, flip = 0.85, 1.25, 0.16
    elif profile == "fast_mixing":
        low, high, flip = 0.20, 3.2, 0.65
    elif profile == "predictive_burst":
        rates: List[np.ndarray] = []
        Qs: List[np.ndarray] = []
        base_q = cyclic_q(5, forward=0.045, backward=0.004)
        for i in range(d):
            scale = rng.uniform(0.85, 1.20)
            if i % 2 == 0:
                r = np.array([0.05, 0.55, 2.3, 5.2, 1.0]) * scale
            else:
                r = np.array([0.08, 0.75, 3.0, 6.0, 1.2]) * scale
            rates.append(r)
            Qs.append(base_q.copy())
        return rates, Qs
    elif profile == "multi_state_burst":
        rates = []
        Qs = []
        base_q = cyclic_q(6, forward=0.035, backward=0.003, skip=0.002)
        templates = [
            np.array([0.04, 0.20, 0.70, 2.8, 6.4, 1.2]),
            np.array([0.05, 0.35, 1.1, 4.0, 7.2, 1.5]),
            np.array([0.08, 0.45, 1.6, 5.0, 8.0, 1.8]),
        ]
        for i in range(d):
            rates.append(templates[i % len(templates)] * rng.uniform(0.80, 1.20))
            Qs.append(base_q.copy())
        return rates, Qs
    elif profile == "alternating_hotspots":
        rates = []
        Qs = []
        base_q = cyclic_q(4, forward=0.055, backward=0.006)
        for i in range(d):
            scale = rng.uniform(0.85, 1.15)
            r = np.array([0.18, 0.35, 0.65, 1.0]) * scale
            r[i % 4] = 6.5 * scale
            r[(i + 1) % 4] = 2.2 * scale
            rates.append(r)
            Qs.append(base_q.copy())
        return rates, Qs
    elif profile == "asymmetric":
        rates: List[np.ndarray] = []
        Qs: List[np.ndarray] = []
        for i in range(d):
            scale = rng.uniform(0.75, 1.35)
            if i % 3 == 0:
                r = np.array([0.04, 1.6, 4.8]) * scale
                q = np.array([[-0.04, 0.035, 0.005], [0.03, -0.055, 0.025], [0.02, 0.035, -0.055]])
            elif i % 3 == 1:
                r = np.array([0.25, 2.2]) * scale
                q = np.array([[-0.07, 0.07], [0.05, -0.05]])
            else:
                r = np.array([0.55, 1.55]) * scale
                q = np.array([[-0.16, 0.16], [0.14, -0.14]])
            rates.append(r)
            Qs.append(q)
        return rates, Qs
    else:
        raise ValueError(f"unknown MAP profile {profile}")
    rates = []
    Qs = []
    for _ in range(d):
        scale = rng.uniform(0.85, 1.2)
        rates.append(np.array([low, high]) * scale)
        Qs.append(np.array([[-flip, flip], [0.75 * flip, -0.75 * flip]], dtype=float))
    return rates, Qs


def stationary_mean_arrival(rates: Sequence[np.ndarray], Qs: Sequence[np.ndarray]) -> np.ndarray:
    means = []
    for r, Q in zip(rates, Qs):
        n = len(r)
        A = Q.T.copy()
        A[-1, :] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0
        pi = np.linalg.solve(A, b)
        means.append(float(pi @ r))
    return np.asarray(means, dtype=float)


def make_problem(case: CaseSpec, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    compat = topology_matrix(case.topology, case.d, case.K, case.density, seed + 11)
    servers = rng.integers(2, 5, size=case.K)
    rates, Qs = map_profile(case.map_profile, case.d, rng)
    arrival_mean = stationary_mean_arrival(rates, Qs)

    service_mu = np.zeros((case.K, case.d), dtype=float)
    service_cv = np.ones((case.K, case.d), dtype=float)
    if case.service_profile == "homogeneous":
        base = np.ones((case.K, case.d))
        cv_base = np.full((case.K, case.d), 0.8)
    elif case.service_profile == "mild":
        base = rng.uniform(0.8, 1.4, size=(case.K, case.d))
        cv_base = rng.uniform(0.8, 1.3, size=(case.K, case.d))
    elif case.service_profile == "strong":
        base = rng.lognormal(mean=0.0, sigma=0.55, size=(case.K, case.d))
        cv_base = rng.uniform(0.9, 2.5, size=(case.K, case.d))
    else:
        raise ValueError(f"unknown service profile {case.service_profile}")
    service_mu[compat] = base[compat]
    service_cv[compat] = cv_base[compat]

    # Scale arrivals to a conservative compatible-capacity proxy. Each pool can
    # serve only one queue at a time, so summing the best pool for every queue
    # would count the same pool multiple times and create unstable instances.
    total_arrival = max(1e-6, float(arrival_mean.sum()))
    pool_cap = 0.0
    for k in range(case.K):
        pool_cap += max((servers[k] * service_mu[k, i] for i in range(case.d) if compat[k, i]), default=0.1)
    demand_mix = arrival_mean / total_arrival
    queue_caps = np.zeros(case.d, dtype=float)
    for i in range(case.d):
        queue_caps[i] = sum(servers[k] * service_mu[k, i] for k in range(case.K) if compat[k, i])
    bottleneck_cap = min(queue_caps[i] / max(1e-6, demand_mix[i]) for i in range(case.d))
    cap = min(pool_cap, bottleneck_cap)
    scale = case.load * cap / total_arrival
    rates = [r * scale for r in rates]
    arrival_mean = stationary_mean_arrival(rates, Qs)

    if case.switch_profile == "none":
        switch_cost = np.zeros(case.K)
        switch_lag = np.zeros(case.K, dtype=int)
    elif case.switch_profile == "moderate":
        switch_cost = rng.uniform(0.3, 0.8, size=case.K)
        switch_lag = rng.integers(0, 2, size=case.K)
    elif case.switch_profile == "high":
        switch_cost = rng.uniform(1.0, 2.0, size=case.K)
        switch_lag = rng.integers(1, 3, size=case.K)
    elif case.switch_profile == "lag":
        switch_cost = rng.uniform(0.5, 1.0, size=case.K)
        switch_lag = rng.integers(2, 5, size=case.K)
    else:
        raise ValueError(f"unknown switch profile {case.switch_profile}")

    return Problem(
        d=case.d,
        K=case.K,
        compat=compat,
        servers=servers,
        service_mu=service_mu,
        service_cv=service_cv,
        arrival_rates=rates,
        arrival_Q=Qs,
        arrival_mean=arrival_mean,
        switch_cost=switch_cost,
        switch_lag=switch_lag,
    )


def policy_cases(cfg: RunConfig) -> List[CaseSpec]:
    fams = set(cfg.families)
    cases: List[CaseSpec] = []
    if "A1_topology" in fams:
        for topo in ["disjoint", "chain", "ring", "star", "bridge_bottleneck", "high_overlap", "random_sparse"]:
            cases.append(CaseSpec("A1_topology", topo, f"topology={topo}", 8, 5, topology=topo, map_profile="medium_burst", service_profile="mild", switch_profile="moderate"))
    if "A2_map_burst" in fams:
        for prof in ["low_contrast", "low_burst", "medium_burst", "high_burst", "fast_mixing", "predictive_burst", "multi_state_burst", "alternating_hotspots", "asymmetric"]:
            cases.append(CaseSpec("A2_map_burst", prof, f"MAP={prof}", 8, 5, topology="chain", map_profile=prof, service_profile="mild", switch_profile="moderate"))
    if "A3_density" in fams:
        for cid, den in [("very_sparse", 0.25), ("sparse", 0.40), ("medium", 0.60)]:
            cases.append(CaseSpec("A3_density", cid, f"compat_density={den}", 8, 5, topology="random_sparse", density=den, map_profile="medium_burst", service_profile="mild", switch_profile="moderate"))
    if "A4_service" in fams:
        for prof in ["homogeneous", "mild", "strong"]:
            cases.append(CaseSpec("A4_service", prof, f"service={prof}", 8, 5, topology="chain", map_profile="medium_burst", service_profile=prof, switch_profile="moderate"))
    if "A5_switch" in fams:
        for prof in ["none", "moderate", "high", "lag"]:
            cases.append(CaseSpec("A5_switch", prof, f"switch={prof}", 8, 5, topology="chain", map_profile="medium_burst", service_profile="mild", switch_profile=prof))
    if "A6_scale" in fams:
        for d in cfg.d_values:
            for K in cfg.K_values:
                if K <= d:
                    cases.append(CaseSpec("A6_scale", f"d{d}_K{K}", f"scale d={d}, K={K}", d, K, topology="random_sparse", density=0.42, map_profile="medium_burst", service_profile="mild", switch_profile="moderate"))
    if "A7_load" in fams:
        for rho in [0.60, 0.75, 0.90, 0.95]:
            cid = f"rho_{str(rho).replace('.', '_')}"
            cases.append(CaseSpec("A7_load", cid, f"load rho={rho:.2f}", 10, 6, topology="random_sparse", density=0.38, map_profile="medium_burst", service_profile="mild", switch_profile="moderate", load=rho))
    if "A8_large_scale" in fams:
        for d, K in [(20, 10), (30, 12), (40, 16), (50, 20), (75, 30), (100, 40)]:
            cases.append(CaseSpec("A8_large_scale", f"d{d}_K{K}", f"large scale d={d}, K={K}", d, K, topology="random_sparse", density=0.30, map_profile="medium_burst", service_profile="mild", switch_profile="moderate", load=0.75))
    return cases


def ablation_cases(cfg: RunConfig) -> List[CaseSpec]:
    fams = set(cfg.families)
    cases: List[CaseSpec] = []
    if "B1_phase" in fams:
        cases.extend([
            CaseSpec("B1_phase", "phase_on_predictive_burst", "phase-on: five-state predictive burst MAP", 10, 6, topology="random_sparse", density=0.35, map_profile="predictive_burst", service_profile="mild", switch_profile="moderate", load=0.74, component="phase", regime="on"),
            CaseSpec("B1_phase", "phase_on_multistate_burst", "phase-on: six-state high-contrast burst MAP", 10, 6, topology="random_sparse", density=0.35, map_profile="multi_state_burst", service_profile="mild", switch_profile="moderate", load=0.72, component="phase", regime="on"),
            CaseSpec("B1_phase", "phase_on_alternating_hotspots", "phase-on: queue-specific alternating MAP hotspots", 10, 6, topology="ring", map_profile="alternating_hotspots", service_profile="mild", switch_profile="moderate", load=0.72, component="phase", regime="on"),
            CaseSpec("B1_phase", "phase_off_low_contrast", "phase-off: low-contrast MAP where phase information is weak", 10, 6, topology="random_sparse", density=0.35, map_profile="low_contrast", service_profile="mild", switch_profile="moderate", load=0.76, component="phase", regime="off"),
            CaseSpec("B1_phase", "phase_off_fast_mixing", "phase-off: fast-mixing MAP with weak persistence", 10, 6, topology="random_sparse", density=0.35, map_profile="fast_mixing", service_profile="mild", switch_profile="moderate", load=0.72, component="phase", regime="off"),
        ])
    if "B2_service" in fams:
        cases.extend([
            CaseSpec("B2_service", "service_on_chain_strong", "service-on: strong edge service heterogeneity on chain", 8, 5, topology="chain", map_profile="medium_burst", service_profile="strong", switch_profile="moderate", load=0.74, component="service", regime="on"),
            CaseSpec("B2_service", "service_on_sparse_strong", "service-on: strong service heterogeneity on sparse graph", 10, 6, topology="random_sparse", density=0.35, map_profile="medium_burst", service_profile="strong", switch_profile="moderate", load=0.74, component="service", regime="on"),
            CaseSpec("B2_service", "service_off_homogeneous", "service-off: homogeneous service rates", 10, 6, topology="random_sparse", density=0.35, map_profile="medium_burst", service_profile="homogeneous", switch_profile="moderate", load=0.74, component="service", regime="off"),
        ])
    if "B3_scarcity" in fams:
        cases.extend([
            CaseSpec("B3_scarcity", "scarcity_on_bridge", "scarcity-on: bridge bottleneck with low-degree queues", 8, 5, topology="bridge_bottleneck", map_profile="medium_burst", service_profile="mild", switch_profile="moderate", load=0.74, component="scarcity", regime="on"),
            CaseSpec("B3_scarcity", "scarcity_on_very_sparse", "scarcity-on: very sparse random compatibility graph", 10, 6, topology="random_sparse", density=0.22, map_profile="medium_burst", service_profile="mild", switch_profile="moderate", load=0.70, component="scarcity", regime="on"),
            CaseSpec("B3_scarcity", "scarcity_off_overlap", "scarcity-off: high-overlap compatibility graph", 10, 6, topology="high_overlap", map_profile="medium_burst", service_profile="mild", switch_profile="moderate", load=0.74, component="scarcity", regime="off"),
        ])
    if "B4_switch" in fams:
        cases.extend([
            CaseSpec("B4_switch", "switch_on_lag_chain", "switch-on: high lag under chain compatibility", 8, 5, topology="chain", map_profile="medium_burst", service_profile="mild", switch_profile="lag", load=0.66, component="switch", regime="on"),
            CaseSpec("B4_switch", "switch_on_high_cost_sparse", "switch-on: high switch cost under sparse compatibility", 10, 6, topology="random_sparse", density=0.35, map_profile="medium_burst", service_profile="mild", switch_profile="high", load=0.68, component="switch", regime="on"),
            CaseSpec("B4_switch", "switch_off_none", "switch-off: no switching cost or lag", 10, 6, topology="random_sparse", density=0.35, map_profile="medium_burst", service_profile="mild", switch_profile="none", load=0.74, component="switch", regime="off"),
        ])
    if "B5_mixed" in fams:
        cases.extend([
            CaseSpec("B5_mixed", "mixed_on_sparse_high_burst", "mixed-on: bursty sparse heterogeneous system with switch friction", 10, 6, topology="random_sparse", density=0.34, map_profile="high_burst", service_profile="strong", switch_profile="high", load=0.62, component="mixed", regime="on"),
            CaseSpec("B5_mixed", "mixed_on_bridge_asymmetric", "mixed-on: bridge bottleneck with asymmetric MAP and heterogeneity", 10, 6, topology="bridge_bottleneck", map_profile="asymmetric", service_profile="strong", switch_profile="high", load=0.62, component="mixed", regime="on"),
            CaseSpec("B5_mixed", "mixed_off_regular", "mixed-off: regular graph, low burst, homogeneous service, no switch friction", 10, 6, topology="high_overlap", map_profile="low_burst", service_profile="homogeneous", switch_profile="none", load=0.74, component="mixed", regime="off"),
        ])
    return cases


def simulate_policy(problem: Problem, policy_name: str, seed: int, warmup_steps: int, eval_steps: int) -> Dict[str, float]:
    env = SparseMapPhEnv(problem, seed=seed)
    env.reset()
    policy = policy_from_name(policy_name)
    policy.reset(problem)
    return simulate_policy_object(problem, policy, env, warmup_steps, eval_steps)


def simulate_policy_object(problem: Problem, policy: Policy, env: SparseMapPhEnv, warmup_steps: int, eval_steps: int) -> Dict[str, float]:
    for _ in range(warmup_steps):
        env.step(policy.action(env))
    sums = {"total_queue": 0.0, "waiting_queue": 0.0, "switch_penalty": 0.0, "switches": 0.0, "arrivals": 0.0, "admitted_arrivals": 0.0, "dropped_arrivals": 0.0, "served": 0.0, "overflow": 0.0, "cap_queue_fraction": 0.0}
    p95_samples: List[float] = []
    for _ in range(eval_steps):
        info = env.step(policy.action(env))
        for k in sums:
            sums[k] += float(info[k])
        p95_samples.append(float(info["total_queue"]))
    n = max(1, eval_steps)
    mean_total_queue = sums["total_queue"] / n
    mean_switch_cost = sums["switch_penalty"] / n
    instability = sums["overflow"] / n
    objective_raw = mean_total_queue + 25.0 * mean_switch_cost + 3.0 * problem.buffer_cap * problem.d * instability
    queue_scale = max(1.0, float(problem.buffer_cap * problem.d))
    switch_scale = max(1.0, float(np.sum(problem.switch_cost)))
    objective = mean_total_queue / queue_scale + mean_switch_cost / switch_scale + 3.0 * instability
    return {
        "mean_total_queue": mean_total_queue,
        "mean_waiting_queue": sums["waiting_queue"] / n,
        "p95_total_queue": float(np.quantile(np.asarray(p95_samples), 0.95)),
        "mean_switch_cost": mean_switch_cost,
        "switch_rate": sums["switches"] / n,
        "mean_arrivals": sums["arrivals"] / n,
        "mean_admitted_arrivals": sums["admitted_arrivals"] / n,
        "mean_dropped_arrivals": sums["dropped_arrivals"] / n,
        "drop_rate": sums["dropped_arrivals"] / max(1.0, sums["arrivals"]),
        "mean_served": sums["served"] / n,
        "instability": instability,
        "cap_queue_fraction": sums["cap_queue_fraction"] / n,
        "objective": objective,
        "objective_raw": objective_raw,
    }


def mean_ci(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci = float(1.96 * std / math.sqrt(arr.size))
    return {"mean": mean, "std": std, "ci95": ci, "low": mean - ci, "high": mean + ci, "n": int(arr.size)}


def _normal_two_sided_p(z: float) -> float:
    if not math.isfinite(z):
        return 1.0
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def paired_t_test(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return {"t_stat": 0.0, "t_pvalue": 1.0}
    try:
        from scipy import stats  # type: ignore

        res = stats.ttest_1samp(arr, 0.0, nan_policy="omit")
        stat = float(res.statistic) if math.isfinite(float(res.statistic)) else 0.0
        pvalue = float(res.pvalue) if math.isfinite(float(res.pvalue)) else 1.0
        return {"t_stat": stat, "t_pvalue": pvalue}
    except Exception:
        std = float(arr.std(ddof=1))
        if std <= 0.0:
            return {"t_stat": 0.0, "t_pvalue": 1.0 if abs(float(arr.mean())) <= 1e-12 else 0.0}
        stat = float(arr.mean()) / (std / math.sqrt(arr.size))
        return {"t_stat": stat, "t_pvalue": _normal_two_sided_p(stat)}


def wilcoxon_signed_rank_test(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[np.abs(arr) > 1e-12]
    if arr.size == 0:
        return {"wilcoxon_stat": 0.0, "wilcoxon_pvalue": 1.0}
    try:
        from scipy import stats  # type: ignore

        res = stats.wilcoxon(arr, zero_method="wilcox", alternative="two-sided")
        stat = float(res.statistic) if math.isfinite(float(res.statistic)) else 0.0
        pvalue = float(res.pvalue) if math.isfinite(float(res.pvalue)) else 1.0
        return {"wilcoxon_stat": stat, "wilcoxon_pvalue": pvalue}
    except Exception:
        order = np.argsort(np.abs(arr))
        ranks = np.empty(arr.size, dtype=float)
        sorted_abs = np.abs(arr)[order]
        start = 0
        while start < arr.size:
            end = start + 1
            while end < arr.size and abs(sorted_abs[end] - sorted_abs[start]) <= 1e-12:
                end += 1
            rank = 0.5 * (start + 1 + end)
            ranks[order[start:end]] = rank
            start = end
        w_plus = float(ranks[arr > 0].sum())
        w_minus = float(ranks[arr < 0].sum())
        stat = min(w_plus, w_minus)
        n = float(arr.size)
        mean = n * (n + 1.0) / 4.0
        var = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0
        if var <= 0.0:
            return {"wilcoxon_stat": stat, "wilcoxon_pvalue": 1.0}
        z = (stat - mean) / math.sqrt(var)
        return {"wilcoxon_stat": stat, "wilcoxon_pvalue": _normal_two_sided_p(z)}


def paired_metric_summary(policy_values: Sequence[float], baseline_values: Sequence[float], prefix: str) -> Dict[str, object]:
    pvals = np.asarray(list(policy_values), dtype=float)
    bvals = np.asarray(list(baseline_values), dtype=float)
    n = min(pvals.size, bvals.size)
    if n == 0:
        return {
            f"{prefix}_baseline_win_rate": 0.0,
            f"{prefix}_policy_win_rate": 0.0,
            f"{prefix}_tie_rate": 0.0,
        }
    pvals = pvals[:n]
    bvals = bvals[:n]
    gaps = pvals - bvals
    rel_improvements = gaps / np.maximum(np.abs(pvals), 1e-9)
    out: Dict[str, object] = {}
    for key, vals in [
        (f"{prefix}_gap", gaps),
        (f"{prefix}_relative_improvement", rel_improvements),
    ]:
        st = mean_ci(vals)
        out.update({f"{key}_{k}": v for k, v in st.items()})
    out.update({f"{prefix}_{k}": v for k, v in paired_t_test(gaps).items()})
    out.update({f"{prefix}_{k}": v for k, v in wilcoxon_signed_rank_test(gaps).items()})
    out[f"{prefix}_baseline_win_rate"] = float(np.mean(bvals < pvals))
    out[f"{prefix}_policy_win_rate"] = float(np.mean(pvals < bvals))
    out[f"{prefix}_tie_rate"] = float(np.mean(np.isclose(pvals, bvals)))
    return out


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize_replications(rows: Sequence[Dict[str, object]], group_cols: Sequence[str], baseline_policy: str) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for r in rows:
        key = tuple(r[c] for c in group_cols)
        groups.setdefault(key, []).append(r)
    out: List[Dict[str, object]] = []
    for key, g in sorted(groups.items()):
        row = {c: v for c, v in zip(group_cols, key)}
        for metric in ["mean_total_queue", "mean_waiting_queue", "p95_total_queue", "mean_switch_cost", "switch_rate", "mean_dropped_arrivals", "drop_rate", "instability", "cap_queue_fraction", "objective", "objective_raw"]:
            available = [float(x[metric]) for x in g if metric in x and str(x[metric]) != ""]
            if not available:
                continue
            st = mean_ci(available)
            for sk, sv in st.items():
                row[f"{metric}_{sk}"] = sv
        out.append(row)

    if "policy" in group_cols:
        case_cols = [c for c in group_cols if c != "policy"]
        base_by_case: Dict[Tuple[object, ...], Dict[str, object]] = {}
        for r in out:
            if r["policy"] == baseline_policy:
                base_by_case[tuple(r[c] for c in case_cols)] = r
        for r in out:
            b = base_by_case.get(tuple(r[c] for c in case_cols))
            if b:
                denom = max(1e-9, float(b["mean_total_queue_mean"]))
                r["relative_gap_vs_baseline"] = (float(r["mean_total_queue_mean"]) - denom) / denom
                objective_denom = max(1e-9, float(b["objective_mean"]))
                r["objective_relative_gap_vs_baseline"] = (float(r["objective_mean"]) - objective_denom) / objective_denom
    return out


def paired_gap_summary(rep_rows: Sequence[Dict[str, object]], baseline_policy: str) -> List[Dict[str, object]]:
    by_case_rep: Dict[Tuple[object, object, object], float] = {}
    objective_by_case_rep: Dict[Tuple[object, object, object], float] = {}
    for r in rep_rows:
        if r["policy"] == baseline_policy:
            key = (r["family"], r["case_id"], r.get("replication", r["seed"]))
            by_case_rep[key] = float(r["mean_total_queue"])
            objective_by_case_rep[key] = float(r.get("objective", r["mean_total_queue"]))
    gaps: Dict[Tuple[object, object, object, object, object, object], List[float]] = {}
    objective_gaps: Dict[Tuple[object, object, object, object, object, object], List[float]] = {}
    policy_values: Dict[Tuple[object, object, object, object, object, object], List[float]] = {}
    baseline_values: Dict[Tuple[object, object, object, object, object, object], List[float]] = {}
    objective_policy_values: Dict[Tuple[object, object, object, object, object, object], List[float]] = {}
    objective_baseline_values: Dict[Tuple[object, object, object, object, object, object], List[float]] = {}
    for r in rep_rows:
        rep_key = (r["family"], r["case_id"], r.get("replication", r["seed"]))
        b = by_case_rep.get(rep_key)
        if b is None:
            continue
        key = (r["family"], r["case_id"], r["description"], r.get("component", ""), r.get("regime", ""), r["policy"])
        policy_value = float(r["mean_total_queue"])
        objective_policy_value = float(r.get("objective", r["mean_total_queue"]))
        gaps.setdefault(key, []).append(policy_value - b)
        objective_gaps.setdefault(key, []).append(objective_policy_value - objective_by_case_rep[rep_key])
        policy_values.setdefault(key, []).append(policy_value)
        baseline_values.setdefault(key, []).append(b)
        objective_policy_values.setdefault(key, []).append(objective_policy_value)
        objective_baseline_values.setdefault(key, []).append(objective_by_case_rep[rep_key])
    out = []
    for key, vals in sorted(gaps.items()):
        st = mean_ci(vals)
        row = {"family": key[0], "case_id": key[1], "description": key[2], "component": key[3], "regime": key[4], "policy": key[5]}
        row.update({f"paired_gap_{k}": v for k, v in st.items()})
        queue_stats = paired_metric_summary(policy_values.get(key, []), baseline_values.get(key, []), "paired_queue")
        row.update(queue_stats)
        objective_st = mean_ci(objective_gaps.get(key, []))
        row.update({f"paired_objective_gap_{k}": v for k, v in objective_st.items()})
        objective_stats = paired_metric_summary(objective_policy_values.get(key, []), objective_baseline_values.get(key, []), "paired_objective")
        row.update(objective_stats)
        row["baseline_policy"] = baseline_policy
        out.append(row)
    return out


def paired_family_summary(rep_rows: Sequence[Dict[str, object]], baseline_policy: str) -> List[Dict[str, object]]:
    by_case_rep: Dict[Tuple[object, object, object], Dict[str, float]] = {}
    for r in rep_rows:
        if r["policy"] == baseline_policy:
            key = (r["family"], r["case_id"], r.get("replication", r["seed"]))
            by_case_rep[key] = {
                "queue": float(r["mean_total_queue"]),
                "objective": float(r.get("objective", r["mean_total_queue"])),
            }
    grouped_policy: Dict[Tuple[object, object], Dict[str, List[float]]] = {}
    grouped_base: Dict[Tuple[object, object], Dict[str, List[float]]] = {}
    for r in rep_rows:
        rep_key = (r["family"], r["case_id"], r.get("replication", r["seed"]))
        base = by_case_rep.get(rep_key)
        if base is None:
            continue
        key = (r["family"], r["policy"])
        grouped_policy.setdefault(key, {"queue": [], "objective": []})
        grouped_base.setdefault(key, {"queue": [], "objective": []})
        grouped_policy[key]["queue"].append(float(r["mean_total_queue"]))
        grouped_base[key]["queue"].append(base["queue"])
        grouped_policy[key]["objective"].append(float(r.get("objective", r["mean_total_queue"])))
        grouped_base[key]["objective"].append(base["objective"])
    out: List[Dict[str, object]] = []
    for (family, policy), vals in sorted(grouped_policy.items()):
        row: Dict[str, object] = {"family": family, "policy": policy, "baseline_policy": baseline_policy}
        row.update(paired_metric_summary(vals["queue"], grouped_base[(family, policy)]["queue"], "family_paired_queue"))
        row.update(paired_metric_summary(vals["objective"], grouped_base[(family, policy)]["objective"], "family_paired_objective"))
        out.append(row)
    return out


def aggregate_family(
    case_rows: Sequence[Dict[str, object]],
    policy_col: str = "policy",
    paired_rows: Sequence[Dict[str, object]] = (),
) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, object], List[Dict[str, object]]] = {}
    for r in case_rows:
        groups.setdefault((r["family"], r[policy_col]), []).append(r)
    paired_by_key = {(r["family"], r[policy_col]): r for r in paired_rows if policy_col in r}
    out = []
    for (fam, pol), g in sorted(groups.items()):
        row: Dict[str, object] = {"family": fam, policy_col: pol, "num_cases": len(g)}
        for metric in ["mean_total_queue_mean", "relative_gap_vs_baseline", "objective_mean", "objective_relative_gap_vs_baseline"]:
            vals = [float(x[metric]) for x in g if metric in x and x[metric] != ""]
            st = mean_ci(vals)
            row[f"{metric}_across_cases"] = st["mean"]
            row[f"{metric}_ci95_across_cases"] = st["ci95"]
        paired = paired_by_key.get((fam, pol))
        if paired:
            for k, v in paired.items():
                if k in {"family", policy_col, "baseline_policy"}:
                    continue
                row[k] = v
        out.append(row)
    return out


def render_progress(done: int, total: int, label: str = "", width: int = 36) -> None:
    if total <= 0:
        return
    done = min(max(done, 0), total)
    ratio = done / total
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    pct = 100.0 * ratio
    text = f"\r[{bar}] {done}/{total} ({pct:5.1f}%)"
    if label:
        text += f" {label}"
    sys.stdout.write(text)
    if done >= total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def plot_bar_by_family(case_summary: Sequence[Dict[str, object]], path: str, title: str, families: Sequence[str], policies: Sequence[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    data = {(r["family"], r["policy"]): float(r["mean_total_queue_mean"]) for r in case_summary}
    x = np.arange(len(families))
    width = 0.8 / max(1, len(policies))
    fig, ax = plt.subplots(figsize=(max(7, len(families) * 1.1), 4.4))
    fig.patch.set_facecolor(PAPER_COLORS["background"])
    ax.set_facecolor(PAPER_COLORS["background"])
    policies = order_policies_for_legend(policies)
    for j, pol in enumerate(policies):
        vals = [data.get((fam, pol), np.nan) for fam in families]
        ax.bar(
            x + (j - (len(policies) - 1) / 2) * width,
            vals,
            width,
            label=POLICY_LABELS.get(pol, pol),
            color=POLICY_COLORS.get(pol, PAPER_COLORS["soft_gray"]),
            edgecolor="none",
            linewidth=0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=25, ha="right")
    ax.set_ylabel("Mean total queue")
    ax.set_title(title)
    ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.7, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(PAPER_COLORS["grid"])
    ax.tick_params(colors=PAPER_COLORS["text"])
    ax.xaxis.label.set_color(PAPER_COLORS["text"])
    ax.yaxis.label.set_color(PAPER_COLORS["text"])
    ax.title.set_color(PAPER_COLORS["text"])
    ax.legend(
        fontsize=8,
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=[0.0, 0.18, 1.0, 1.0])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_log_bar_by_family(case_summary: Sequence[Dict[str, object]], path: str, title: str, families: Sequence[str], policies: Sequence[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    data = {(r["family"], r["policy"]): float(r["mean_total_queue_mean"]) for r in case_summary}
    x = np.arange(len(families))
    width = 0.8 / max(1, len(policies))
    fig, ax = plt.subplots(figsize=(max(7, len(families) * 1.1), 4.4))
    fig.patch.set_facecolor(PAPER_COLORS["background"])
    ax.set_facecolor(PAPER_COLORS["background"])
    policies = order_policies_for_legend(policies)
    for j, pol in enumerate(policies):
        vals = [max(1e-6, data.get((fam, pol), np.nan)) for fam in families]
        ax.bar(
            x + (j - (len(policies) - 1) / 2) * width,
            vals,
            width,
            label=POLICY_LABELS.get(pol, pol),
            color=POLICY_COLORS.get(pol, PAPER_COLORS["soft_gray"]),
            edgecolor="none",
            linewidth=0,
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=25, ha="right")
    ax.set_ylabel("Mean total queue (log scale)")
    ax.set_title(title)
    ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.7, linewidth=0.7, which="both")
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(PAPER_COLORS["grid"])
    ax.tick_params(colors=PAPER_COLORS["text"])
    ax.xaxis.label.set_color(PAPER_COLORS["text"])
    ax.yaxis.label.set_color(PAPER_COLORS["text"])
    ax.title.set_color(PAPER_COLORS["text"])
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30), borderaxespad=0.0)
    fig.tight_layout(rect=[0.0, 0.18, 1.0, 1.0])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_gap_heatmap(gap_rows: Sequence[Dict[str, object]], path: str, title: str, value_col: str = "paired_gap_mean", colorbar_label: str = "Paired gap vs full gPAH") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    except Exception:
        return
    families = sorted(set(str(r["family"]) for r in gap_rows))
    policies = [p for p in sorted(set(str(r["policy"]) for r in gap_rows)) if p not in {"gpah", "gpah_full"}]
    mat = np.full((len(families), len(policies)), np.nan)
    vals: Dict[Tuple[str, str], List[float]] = {}
    policy_labels: Dict[str, str] = {}
    for r in gap_rows:
        policy = str(r["policy"])
        vals.setdefault((str(r["family"]), policy), []).append(float(r[value_col]))
        label = str(r.get("policy_label", "") or "")
        if label and policy not in {"gpah", "gpah_full"}:
            policy_labels[policy] = label
    for i, fam in enumerate(families):
        for j, pol in enumerate(policies):
            if (fam, pol) in vals:
                mat[i, j] = float(np.mean(vals[(fam, pol)]))
    fig, ax = plt.subplots(figsize=(max(7, len(policies) * 1.0), max(3.8, len(families) * 0.55)))
    fig.patch.set_facecolor(PAPER_COLORS["background"])
    ax.set_facecolor(PAPER_COLORS["background"])
    cmap = LinearSegmentedColormap.from_list(
        "gpah_contrast_heat",
        [
            (0.00, PAPER_COLORS["c02"]),
            (0.485, "#A9D4E8"),
            (0.500, "#ffffff"),
            (0.515, "#F2B0AA"),
            (1.00, PAPER_COLORS["c16"]),
        ],
    )
    cmap.set_under(PAPER_COLORS["c02"])
    cmap.set_over(PAPER_COLORS["c16"])
    finite = mat[np.isfinite(mat)]
    norm = None
    colorbar_extend = "neither"
    if finite.size:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        if lo < 0 < hi:
            span = float(np.nanpercentile(np.abs(finite), 95))
            span = max(span, 1e-9)
            norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
            if lo < -span or hi > span:
                colorbar_extend = "both"
    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(policies)))
    ax.set_xticklabels([policy_labels.get(p, POLICY_LABELS.get(p, p)) for p in policies], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(families)))
    ax.set_yticklabels(families)
    ax.set_xticks(np.arange(len(policies) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(families) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#d9d9d9", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title)
    ax.set_xlabel("Variant / policy")
    ax.set_ylabel("Case family")
    ax.tick_params(colors=PAPER_COLORS["text"])
    ax.xaxis.label.set_color(PAPER_COLORS["text"])
    ax.yaxis.label.set_color(PAPER_COLORS["text"])
    ax.title.set_color(PAPER_COLORS["text"])
    for spine in ax.spines.values():
        spine.set_color(PAPER_COLORS["grid"])
    cbar = fig.colorbar(im, ax=ax, label=colorbar_label, extend=colorbar_extend)
    cbar.outline.set_edgecolor(PAPER_COLORS["grid"])
    cbar.ax.yaxis.label.set_color(PAPER_COLORS["text"])
    cbar.ax.tick_params(colors=PAPER_COLORS["text"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_relative_heatmap(case_summary: Sequence[Dict[str, object]], path: str, title: str, baseline_policy: str = "gpah") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    except Exception:
        return
    cases = sorted(set((str(r["family"]), str(r["case_id"])) for r in case_summary))
    policies = [p for p in sorted(set(str(r["policy"]) for r in case_summary)) if p != baseline_policy]
    values = {(str(r["family"]), str(r["case_id"]), str(r["policy"])): float(r["mean_total_queue_mean"]) for r in case_summary}
    mat = np.full((len(cases), len(policies)), np.nan)
    for i, (fam, cid) in enumerate(cases):
        base = values.get((fam, cid, baseline_policy), np.nan)
        for j, pol in enumerate(policies):
            val = values.get((fam, cid, pol), np.nan)
            if np.isfinite(base) and base > 0 and np.isfinite(val):
                mat[i, j] = val / base - 1.0
    fig, ax = plt.subplots(figsize=(max(8, len(policies) * 1.05), max(5, len(cases) * 0.23)))
    fig.patch.set_facecolor(PAPER_COLORS["background"])
    ax.set_facecolor(PAPER_COLORS["background"])
    cmap = LinearSegmentedColormap.from_list("gpah_rel", [PAPER_COLORS["c02"], "#ffffff", PAPER_COLORS["c16"]])
    finite = mat[np.isfinite(mat)]
    norm = None
    if finite.size:
        span = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))), 0.05)
        norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(policies)))
    ax.set_xticklabels([POLICY_LABELS.get(p, p) for p in policies], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(cases)))
    ax.set_yticklabels([f"{fam}/{cid}" for fam, cid in cases], fontsize=7)
    ax.set_xticks(np.arange(len(policies) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(cases) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#d9d9d9", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title)
    ax.set_xlabel("Policy")
    ax.set_ylabel("Case")
    for spine in ax.spines.values():
        spine.set_color(PAPER_COLORS["grid"])
    ax.tick_params(colors=PAPER_COLORS["text"])
    ax.xaxis.label.set_color(PAPER_COLORS["text"])
    ax.yaxis.label.set_color(PAPER_COLORS["text"])
    ax.title.set_color(PAPER_COLORS["text"])
    cbar = fig.colorbar(im, ax=ax, label="Relative queue ratio: policy/gPAH - 1")
    cbar.outline.set_edgecolor(PAPER_COLORS["grid"])
    cbar.ax.yaxis.label.set_color(PAPER_COLORS["text"])
    cbar.ax.tick_params(colors=PAPER_COLORS["text"])
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_policy_radar(family_summary: Sequence[Dict[str, object]], path: str, title: str, policies: Sequence[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    families = sorted(set(str(r["family"]) for r in family_summary))
    if len(families) < 3:
        return
    by_key = {(str(r["family"]), str(r["policy"])): float(r["objective_mean_across_cases"]) for r in family_summary}
    best = {}
    for fam in families:
        vals = [by_key[(fam, p)] for p in policies if (fam, p) in by_key and np.isfinite(by_key[(fam, p)])]
        if vals:
            best[fam] = max(1e-9, min(vals))
    angles = np.linspace(0, 2 * np.pi, len(families), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.6, 6.0), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(PAPER_COLORS["background"])
    ax.set_facecolor(PAPER_COLORS["background"])
    policies = order_policies_for_legend(policies)
    for pol in policies:
        vals = []
        for fam in families:
            raw = by_key.get((fam, pol), np.nan)
            if fam not in best or not np.isfinite(raw):
                vals.append(np.nan)
            else:
                # Larger is better; 1.0 is best family performance.
                vals.append(min(1.0, best[fam] / max(raw, 1e-9)))
        if not any(np.isfinite(vals)):
            continue
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.4, label=POLICY_LABELS.get(pol, pol), color=POLICY_COLORS.get(pol, PAPER_COLORS["soft_gray"]))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(families, fontsize=8)
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "best"], fontsize=7)
    ax.set_title(title, pad=18, color=PAPER_COLORS["text"])
    ax.grid(color=PAPER_COLORS["grid"], alpha=0.8)
    ax.spines["polar"].set_color(PAPER_COLORS["grid"])
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.20))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def add_common_args(p: argparse.ArgumentParser, default_families: str, default_policies: str, seed: int) -> None:
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", default="")
    p.add_argument("--no_timestamped_output", action="store_true")
    p.add_argument("--families", default=default_families)
    p.add_argument("--policies", default=default_policies)
    p.add_argument("--d_values", default="6,8,12")
    p.add_argument("--K_values", default="4,6")
    p.add_argument("--replications", type=int, default=50)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=seed)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    output_dir = args.output_dir
    if not args.no_timestamped_output:
        run_name = args.run_name.strip() if args.run_name else ""
        if not run_name:
            run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, run_name)
    return RunConfig(
        output_dir=output_dir,
        replications=args.replications,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
        d_values=parse_csv_list(args.d_values, int),
        K_values=parse_csv_list(args.K_values, int),
        families=parse_csv_list(args.families, str),
        policies=parse_csv_list(args.policies, str),
    )
