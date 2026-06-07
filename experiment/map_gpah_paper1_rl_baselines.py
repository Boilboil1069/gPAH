#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from map_gpah_paper1_common import (
    POLICY_LABELS,
    PAPER_COLORS,
    POLICY_COLORS,
    AffinityPolicy,
    EqualSplitPolicy,
    MaxWeightMuPolicy,
    Problem,
    ScorePolicy,
    ScoreWeights,
    SparseMapPhEnv,
    make_problem,
    mean_ci,
    policy_cases,
    render_progress,
    simulate_policy,
    write_csv,
    RunConfig,
    order_policies_for_legend,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for RL baselines. Install it in the experiment "
        "environment first, e.g. `python3 -m pip install -r experiment/requirements.txt`."
    ) from exc


RL_LABELS = {
    "dqn": "DQN-Candidate",
    "ppo": "PPO-Factorized",
    "sac": "Discrete-SAC-Factorized",
}

RL_COLORS = {
    **POLICY_COLORS,
    "dqn": "#C7C1DE",
    "ppo": "#EEB6D4",
    "sac": "#52AADC",
}

FIXED_POLICIES = ["backpressure", "maxweight_mu", "cmu", "affinity", "qah_switch", "gpah_meanmap", "gpah"]
METRIC_COLUMNS = [
    "mean_total_queue",
    "mean_waiting_queue",
    "p95_total_queue",
    "mean_switch_cost",
    "switch_rate",
    "mean_arrivals",
    "mean_served",
    "instability",
    "objective",
    "objective_raw",
]


@dataclass
class RLConfig:
    output_dir: str
    algorithms: Tuple[str, ...]
    families: Tuple[str, ...]
    case_ids: Tuple[str, ...]
    replications: int
    train_seeds: int
    warmup_steps: int
    eval_steps: int
    train_steps: int
    rollout_steps: int
    seed: int
    device: str
    jobs: int = 1
    reuse_fixed_policy_dir: str = ""
    train_progress: bool = True
    gamma: float = 0.995
    lr: float = 3e-4
    batch_size: int = 128
    hidden: int = 128
    buffer_size: int = 50000
    update_after: int = 500
    update_every: int = 4
    target_tau: float = 0.01
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    alpha: float = 0.08
    curve_window: int = 500
    budget_preset: str = "review"


BUDGET_PRESETS = {
    "smoke": {"train_steps": 12000, "train_seeds": 1},
    "review": {"train_steps": 100000, "train_seeds": 5},
    "full": {"train_steps": 250000, "train_seeds": 10},
}


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_csv(raw: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def observation(env: SparseMapPhEnv) -> np.ndarray:
    assert env.state is not None
    p = env.problem
    s = env.state
    q = s.q.astype(np.float32) / max(1.0, float(p.buffer_cap))
    cur_rates = env.current_arrival_rates().astype(np.float32)
    mean_rates = np.maximum(1e-6, p.arrival_mean.astype(np.float32))
    rate_ratio = np.clip(cur_rates / mean_rates, 0.0, 8.0) / 8.0
    forecast = env.forecast_arrival_excess().astype(np.float32)
    forecast = np.clip(forecast / np.maximum(1.0, mean_rates * 6.0), 0.0, 8.0) / 8.0
    assigned = np.zeros((p.K, p.d + 1), dtype=np.float32)
    for k in range(p.K):
        assigned[k, int(s.assigned[k]) + 1] = 1.0
    cooldown = s.cooldown.astype(np.float32) / max(1.0, float(max(1, int(np.max(p.switch_lag)))))
    return np.concatenate([q, rate_ratio, forecast, assigned.ravel(), cooldown], axis=0).astype(np.float32)


def obs_dim(problem: Problem) -> int:
    return 3 * problem.d + problem.K * (problem.d + 1) + problem.K


def action_mask(problem: Problem) -> np.ndarray:
    mask = np.zeros((problem.K, problem.d + 1), dtype=bool)
    mask[:, 0] = True
    mask[:, 1:] = problem.compat
    return mask


def reward_from_info(problem: Problem, info: Dict[str, float]) -> float:
    q_scale = max(1.0, float(problem.buffer_cap * problem.d))
    sw_scale = max(1.0, float(np.sum(problem.switch_cost)))
    return -(
        float(info["total_queue"]) / q_scale
        + float(info["switch_penalty"]) / sw_scale
        + 3.0 * float(info["overflow"])
    )


def fallback_action(env: SparseMapPhEnv) -> np.ndarray:
    return ScorePolicy(ScoreWeights(), True, "gpah").action(env)


def candidate_actions(env: SparseMapPhEnv) -> np.ndarray:
    policies = [
        EqualSplitPolicy(),
        ScorePolicy(ScoreWeights(queue=1.0, phase=0.0, service=0.0, scarcity=0.0, switch_cost=0.0, switch_lag=0.0), False),
        MaxWeightMuPolicy(),
        AffinityPolicy(),
        ScorePolicy(ScoreWeights(queue=1.0, phase=0.0, service=0.0, scarcity=0.0, switch_cost=0.9, switch_lag=0.35), False),
        ScorePolicy(ScoreWeights(), False),
        ScorePolicy(ScoreWeights(), True),
    ]
    out = []
    for pol in policies:
        pol.reset(env.problem)
        out.append(pol.action(env))
    return np.asarray(out, dtype=int)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data: List[Tuple[np.ndarray, np.ndarray | int, float, np.ndarray, bool]] = []
        self.pos = 0

    def add(self, obs: np.ndarray, act: np.ndarray | int, rew: float, nxt: np.ndarray, done: bool) -> None:
        item = (obs.copy(), np.array(act).copy() if not isinstance(act, int) else int(act), float(rew), nxt.copy(), bool(done))
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self.data), size=batch_size)
        batch = [self.data[int(i)] for i in idx]
        obs, act, rew, nxt, done = zip(*batch)
        return np.asarray(obs), np.asarray(act), np.asarray(rew, dtype=np.float32), np.asarray(nxt), np.asarray(done, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.data)


class DQNAgent:
    def __init__(self, problem: Problem, cfg: RLConfig, seed: int):
        torch.manual_seed(seed)
        self.problem = problem
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(cfg.device)
        self.n_candidates = 7
        self.q = MLP(obs_dim(problem), self.n_candidates, cfg.hidden).to(self.device)
        self.target = MLP(obs_dim(problem), self.n_candidates, cfg.hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)

    def action(self, env: SparseMapPhEnv, eps: float = 0.0) -> np.ndarray:
        cands = candidate_actions(env)
        if self.rng.random() < eps:
            return cands[int(self.rng.integers(0, len(cands)))]
        with torch.no_grad():
            o = torch.as_tensor(observation(env), dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q(o)[0, : len(cands)]
            return cands[int(torch.argmax(q).item())]

    def update(self, rb: ReplayBuffer) -> Dict[str, float]:
        obs, act, rew, nxt, done = rb.sample(self.cfg.batch_size, self.rng)
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(act, dtype=torch.long, device=self.device)
        r = torch.as_tensor(rew, dtype=torch.float32, device=self.device)
        no = torch.as_tensor(nxt, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        q = self.q(o).gather(1, a.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            y = r + self.cfg.gamma * (1.0 - d) * self.target(no).max(dim=1).values
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.target.parameters()):
                tp.data.mul_(1.0 - self.cfg.target_tau).add_(self.cfg.target_tau * p.data)
        return {"loss": float(loss.item())}


class FactorActorCritic(nn.Module):
    def __init__(self, in_dim: int, K: int, d: int, hidden: int):
        super().__init__()
        self.K = K
        self.d = d
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.actor = nn.Linear(hidden, K * (d + 1))
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(obs)
        return self.actor(h).view(-1, self.K, self.d + 1), self.value(h).squeeze(-1)


class PPOAgent:
    def __init__(self, problem: Problem, cfg: RLConfig, seed: int):
        torch.manual_seed(seed)
        self.problem = problem
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(cfg.device)
        self.net = FactorActorCritic(obs_dim(problem), problem.K, problem.d, cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.mask = torch.as_tensor(action_mask(problem), dtype=torch.bool, device=self.device)

    def _dist(self, obs_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, value = self.net(obs_t)
        logits = logits.masked_fill(~self.mask.unsqueeze(0), -1e9)
        return logits, value

    def action_and_logp(self, env: SparseMapPhEnv, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        with torch.no_grad():
            obs_t = torch.as_tensor(observation(env), dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self._dist(obs_t)
            acts = []
            logps = []
            for k in range(self.problem.K):
                dist = Categorical(logits=logits[0, k])
                a = torch.argmax(logits[0, k]) if deterministic else dist.sample()
                acts.append(int(a.item()))
                logps.append(dist.log_prob(a))
            return np.asarray(acts, dtype=int), float(torch.stack(logps).sum().item()), float(value.item())

    def action(self, env: SparseMapPhEnv) -> np.ndarray:
        return self.action_and_logp(env, deterministic=True)[0]


class TwinQ(nn.Module):
    def __init__(self, obs_in: int, act_in: int, hidden: int):
        super().__init__()
        self.q1 = MLP(obs_in + act_in, 1, hidden)
        self.q2 = MLP(obs_in + act_in, 1, hidden)

    def forward(self, obs: torch.Tensor, act_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act_onehot], dim=1)
        return self.q1(x).squeeze(1), self.q2(x).squeeze(1)


class SACAgent:
    def __init__(self, problem: Problem, cfg: RLConfig, seed: int):
        torch.manual_seed(seed)
        self.problem = problem
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(cfg.device)
        self.actor = FactorActorCritic(obs_dim(problem), problem.K, problem.d, cfg.hidden).to(self.device)
        act_dim = problem.K * (problem.d + 1)
        self.q = TwinQ(obs_dim(problem), act_dim, cfg.hidden).to(self.device)
        self.target = TwinQ(obs_dim(problem), act_dim, cfg.hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.mask = torch.as_tensor(action_mask(problem), dtype=torch.bool, device=self.device)

    def action(self, env: SparseMapPhEnv, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(observation(env), dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, _ = self.actor(obs_t)
            logits = logits.masked_fill(~self.mask.unsqueeze(0), -1e9)
            acts = []
            for k in range(self.problem.K):
                dist = Categorical(logits=logits[0, k])
                a = torch.argmax(logits[0, k]) if deterministic else dist.sample()
                acts.append(int(a.item()))
            return np.asarray(acts, dtype=int)

    def onehot(self, acts: torch.Tensor) -> torch.Tensor:
        parts = [F.one_hot(acts[:, k], num_classes=self.problem.d + 1).float() for k in range(self.problem.K)]
        return torch.cat(parts, dim=1)

    def sample_actor(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, _ = self.actor(obs)
        logits = logits.masked_fill(~self.mask.unsqueeze(0), -1e9)
        acts = []
        logps = []
        for k in range(self.problem.K):
            dist = Categorical(logits=logits[:, k])
            a = dist.sample()
            acts.append(a)
            logps.append(dist.log_prob(a))
        act_t = torch.stack(acts, dim=1)
        return act_t, torch.stack(logps, dim=1).sum(dim=1)

    def update(self, rb: ReplayBuffer) -> Dict[str, float]:
        obs, act, rew, nxt, done = rb.sample(self.cfg.batch_size, self.rng)
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(act, dtype=torch.long, device=self.device)
        r = torch.as_tensor(rew, dtype=torch.float32, device=self.device)
        no = torch.as_tensor(nxt, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            na, nlogp = self.sample_actor(no)
            nq1, nq2 = self.target(no, self.onehot(na))
            y = r + self.cfg.gamma * (1.0 - d) * (torch.minimum(nq1, nq2) - self.cfg.alpha * nlogp)
        q1, q2 = self.q(o, self.onehot(a))
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.q_opt.step()

        pa, logp = self.sample_actor(o)
        pq1, pq2 = self.q(o, self.onehot(pa))
        actor_loss = (self.cfg.alpha * logp - torch.minimum(pq1, pq2)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.target.parameters()):
                tp.data.mul_(1.0 - self.cfg.target_tau).add_(self.cfg.target_tau * p.data)
        return {"q_loss": float(q_loss.item()), "actor_loss": float(actor_loss.item())}


def show_train_progress(cfg: RLConfig, step: int, label: str, metrics: Dict[str, float] | None = None) -> None:
    if not cfg.train_progress or not label:
        return
    suffix = label
    if metrics:
        if "train_reward_mean_window" in metrics:
            suffix += f" reward={float(metrics['train_reward_mean_window']):.4g}"
        if "loss" in metrics:
            suffix += f" loss={float(metrics['loss']):.4g}"
        elif "actor_loss" in metrics:
            suffix += f" actor_loss={float(metrics['actor_loss']):.4g}"
        elif "q_loss" in metrics:
            suffix += f" q_loss={float(metrics['q_loss']):.4g}"
    render_progress(step, cfg.train_steps, suffix)


def train_dqn(problem: Problem, cfg: RLConfig, seed: int, progress_label: str = "") -> Tuple[DQNAgent, List[Dict[str, float]]]:
    env = SparseMapPhEnv(problem, seed)
    env.reset()
    agent = DQNAgent(problem, cfg, seed)
    rb = ReplayBuffer(cfg.buffer_size)
    hist: List[Dict[str, float]] = []
    rewards: List[float] = []
    obs = observation(env)
    report_every = max(1, cfg.train_steps // 50)
    show_train_progress(cfg, 0, progress_label)
    for step in range(1, cfg.train_steps + 1):
        eps = max(0.05, 0.9 * (1.0 - step / max(1, cfg.train_steps)))
        cands = candidate_actions(env)
        if agent.rng.random() < eps:
            idx = int(agent.rng.integers(0, len(cands)))
        else:
            with torch.no_grad():
                q = agent.q(torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0))[0, : len(cands)]
                idx = int(torch.argmax(q).item())
        info = env.step(cands[idx])
        reward = reward_from_info(problem, info)
        nxt = observation(env)
        rb.add(obs, idx, reward, nxt, False)
        rewards.append(reward)
        if len(rewards) > max(1, cfg.curve_window):
            rewards.pop(0)
        obs = nxt
        if len(rb) >= max(cfg.update_after, cfg.batch_size) and step % cfg.update_every == 0:
            metrics = agent.update(rb)
        else:
            metrics = {}
        if step % report_every == 0 or step == cfg.train_steps:
            row = {
                "step": step,
                "epsilon": eps,
                "train_reward_mean_window": float(np.mean(rewards)) if rewards else 0.0,
                **metrics,
            }
            hist.append(row)
            show_train_progress(cfg, step, progress_label, row)
    return agent, hist


def train_ppo(problem: Problem, cfg: RLConfig, seed: int, progress_label: str = "") -> Tuple[PPOAgent, List[Dict[str, float]]]:
    env = SparseMapPhEnv(problem, seed)
    env.reset()
    agent = PPOAgent(problem, cfg, seed)
    hist: List[Dict[str, float]] = []
    completed = 0
    report_every = max(1, cfg.train_steps // 50)
    last_report = 0
    show_train_progress(cfg, 0, progress_label)
    while completed < cfg.train_steps:
        obs_l: List[np.ndarray] = []
        act_l: List[np.ndarray] = []
        rew_l: List[float] = []
        logp_l: List[float] = []
        val_l: List[float] = []
        for _ in range(min(cfg.rollout_steps, cfg.train_steps - completed)):
            obs_l.append(observation(env))
            act, logp, val = agent.action_and_logp(env, deterministic=False)
            info = env.step(act)
            act_l.append(act)
            logp_l.append(logp)
            val_l.append(val)
            rew_l.append(reward_from_info(problem, info))
            completed += 1
        returns = []
        g = 0.0
        for r in reversed(rew_l):
            g = r + cfg.gamma * g
            returns.append(g)
        returns = list(reversed(returns))
        obs_t = torch.as_tensor(np.asarray(obs_l), dtype=torch.float32, device=agent.device)
        act_t = torch.as_tensor(np.asarray(act_l), dtype=torch.long, device=agent.device)
        old_logp_t = torch.as_tensor(np.asarray(logp_l), dtype=torch.float32, device=agent.device)
        ret_t = torch.as_tensor(np.asarray(returns), dtype=torch.float32, device=agent.device)
        val_t = torch.as_tensor(np.asarray(val_l), dtype=torch.float32, device=agent.device)
        adv_t = ret_t - val_t
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-6)
        last_loss = 0.0
        for _ in range(cfg.ppo_epochs):
            logits, values = agent._dist(obs_t)
            logps = []
            entropies = []
            for k in range(problem.K):
                dist = Categorical(logits=logits[:, k])
                logps.append(dist.log_prob(act_t[:, k]))
                entropies.append(dist.entropy())
            logp = torch.stack(logps, dim=1).sum(dim=1)
            entropy = torch.stack(entropies, dim=1).sum(dim=1).mean()
            ratio = torch.exp(logp - old_logp_t)
            pg = -torch.minimum(ratio * adv_t, torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * adv_t).mean()
            value_loss = F.mse_loss(values, ret_t)
            loss = pg + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            agent.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.net.parameters(), 5.0)
            agent.opt.step()
            last_loss = float(loss.item())
        if completed - last_report >= report_every or completed >= cfg.train_steps:
            row = {
                "step": completed,
                "loss": last_loss,
                "rollout_reward_mean": float(np.mean(rew_l)),
                "train_reward_mean_window": float(np.mean(rew_l)),
            }
            hist.append(row)
            show_train_progress(cfg, completed, progress_label, row)
            last_report = completed
    return agent, hist


def train_sac(problem: Problem, cfg: RLConfig, seed: int, progress_label: str = "") -> Tuple[SACAgent, List[Dict[str, float]]]:
    env = SparseMapPhEnv(problem, seed)
    env.reset()
    agent = SACAgent(problem, cfg, seed)
    rb = ReplayBuffer(cfg.buffer_size)
    hist: List[Dict[str, float]] = []
    rewards: List[float] = []
    obs = observation(env)
    report_every = max(1, cfg.train_steps // 50)
    show_train_progress(cfg, 0, progress_label)
    for step in range(1, cfg.train_steps + 1):
        if step < cfg.update_after:
            mask = action_mask(problem)
            act = np.asarray([agent.rng.choice(np.flatnonzero(mask[k])) for k in range(problem.K)], dtype=int)
        else:
            act = agent.action(env, deterministic=False)
        info = env.step(act)
        reward = reward_from_info(problem, info)
        nxt = observation(env)
        rb.add(obs, act, reward, nxt, False)
        rewards.append(reward)
        if len(rewards) > max(1, cfg.curve_window):
            rewards.pop(0)
        obs = nxt
        if len(rb) >= max(cfg.update_after, cfg.batch_size) and step % cfg.update_every == 0:
            metrics = agent.update(rb)
        else:
            metrics = {}
        if step % report_every == 0 or step == cfg.train_steps:
            row = {
                "step": step,
                "train_reward_mean_window": float(np.mean(rewards)) if rewards else 0.0,
                **metrics,
            }
            hist.append(row)
            show_train_progress(cfg, step, progress_label, row)
    return agent, hist


def evaluate_agent(problem: Problem, agent, seed: int, warmup_steps: int, eval_steps: int) -> Dict[str, float]:
    env = SparseMapPhEnv(problem, seed)
    env.reset()
    for _ in range(warmup_steps):
        env.step(agent.action(env))
    sums = {k: 0.0 for k in ["total_queue", "waiting_queue", "switch_penalty", "switches", "arrivals", "admitted_arrivals", "dropped_arrivals", "served", "overflow", "cap_queue_fraction"]}
    samples: List[float] = []
    for _ in range(eval_steps):
        info = env.step(agent.action(env))
        for k in sums:
            sums[k] += float(info[k])
        samples.append(float(info["total_queue"]))
    n = max(1, eval_steps)
    mean_total_queue = sums["total_queue"] / n
    mean_switch_cost = sums["switch_penalty"] / n
    instability = sums["overflow"] / n
    queue_scale = max(1.0, float(problem.buffer_cap * problem.d))
    switch_scale = max(1.0, float(np.sum(problem.switch_cost)))
    return {
        "mean_total_queue": mean_total_queue,
        "mean_waiting_queue": sums["waiting_queue"] / n,
        "p95_total_queue": float(np.quantile(np.asarray(samples), 0.95)),
        "mean_switch_cost": mean_switch_cost,
        "switch_rate": sums["switches"] / n,
        "mean_arrivals": sums["arrivals"] / n,
        "mean_admitted_arrivals": sums["admitted_arrivals"] / n,
        "mean_dropped_arrivals": sums["dropped_arrivals"] / n,
        "drop_rate": sums["dropped_arrivals"] / max(1.0, sums["arrivals"]),
        "mean_served": sums["served"] / n,
        "instability": instability,
        "cap_queue_fraction": sums["cap_queue_fraction"] / n,
        "objective": mean_total_queue / queue_scale + mean_switch_cost / switch_scale + 3.0 * instability,
    }


def summarize(rows: Sequence[Dict[str, object]], group_cols: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[c] for c in group_cols), []).append(row)
    out: List[Dict[str, object]] = []
    for key, group in sorted(groups.items()):
        r = {c: v for c, v in zip(group_cols, key)}
        for metric in ["mean_total_queue", "mean_waiting_queue", "p95_total_queue", "mean_switch_cost", "switch_rate", "mean_dropped_arrivals", "drop_rate", "instability", "cap_queue_fraction", "objective"]:
            available = [float(x[metric]) for x in group if metric in x and str(x[metric]) != ""]
            if not available:
                continue
            st = mean_ci(available)
            for sk, sv in st.items():
                r[f"{metric}_{sk}"] = sv
        out.append(r)
    return out


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def policy_label(policy: str, row: Dict[str, object] | None = None) -> str:
    if row and row.get("policy_label"):
        return str(row["policy_label"])
    return RL_LABELS.get(policy, POLICY_LABELS.get(policy, policy))


def plot_family_log_bars(family_rows: Sequence[Dict[str, str]], output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not family_rows:
        return
    families = sorted({r["family"] for r in family_rows})
    policies = order_policies_for_legend(sorted({r["policy"] for r in family_rows}))
    data = {(r["family"], r["policy"]): float(r["mean_total_queue_mean"]) for r in family_rows}
    x = np.arange(len(families))
    width = 0.82 / max(1, len(policies))
    fig, ax = plt.subplots(figsize=(max(8, len(families) * 1.05), 4.8))
    fig.patch.set_facecolor(PAPER_COLORS["background"])
    ax.set_facecolor(PAPER_COLORS["background"])
    for j, pol in enumerate(policies):
        vals = [max(1e-6, data.get((fam, pol), np.nan)) for fam in families]
        ax.bar(
            x + (j - (len(policies) - 1) / 2) * width,
            vals,
            width,
            label=policy_label(pol),
            color=RL_COLORS.get(pol, PAPER_COLORS["soft_gray"]),
            edgecolor="none",
            linewidth=0,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Mean total queue (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=25, ha="right")
    ax.set_title("RL baselines and structural policies across experiment families")
    ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.7, which="both")
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=7, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    fig.tight_layout(rect=[0.0, 0.18, 1.0, 1.0])
    fig.savefig(os.path.join(output_dir, "fig_rl_family_log_bars.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_rl_relative_heatmap(case_rows: Sequence[Dict[str, str]], output_dir: str, metric: str = "objective_mean") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    except Exception:
        return
    if not case_rows:
        return
    cases = sorted({(r["family"], r["case_id"]) for r in case_rows})
    policies = [p for p in sorted({r["policy"] for r in case_rows}) if p != "gpah"]
    values = {(r["family"], r["case_id"], r["policy"]): float(r[metric]) for r in case_rows}
    mat = np.full((len(cases), len(policies)), np.nan)
    for i, (fam, cid) in enumerate(cases):
        base = values.get((fam, cid, "gpah"), np.nan)
        for j, pol in enumerate(policies):
            val = values.get((fam, cid, pol), np.nan)
            if np.isfinite(base) and base > 0 and np.isfinite(val):
                mat[i, j] = val / base - 1.0
    fig, ax = plt.subplots(figsize=(max(8, len(policies) * 0.9), max(5.5, len(cases) * 0.23)))
    fig.patch.set_facecolor(PAPER_COLORS["background"])
    ax.set_facecolor(PAPER_COLORS["background"])
    cmap = LinearSegmentedColormap.from_list("rl_rel", [PAPER_COLORS["c02"], "#ffffff", PAPER_COLORS["c16"]])
    finite = mat[np.isfinite(mat)]
    norm = None
    if finite.size:
        span = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))), 0.05)
        norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(policies)))
    ax.set_xticklabels([policy_label(p) for p in policies], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(cases)))
    ax.set_yticklabels([f"{fam}/{cid}" for fam, cid in cases], fontsize=7)
    ax.set_xticks(np.arange(len(policies) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(cases) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#d9d9d9", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("Relative objective gap versus gPAH")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Case")
    cbar = fig.colorbar(im, ax=ax, label="policy/gPAH - 1")
    cbar.outline.set_edgecolor(PAPER_COLORS["grid"])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_rl_relative_objective_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_rl_radar(family_rows: Sequence[Dict[str, str]], output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not family_rows:
        return
    families = sorted({r["family"] for r in family_rows})
    policies = order_policies_for_legend(sorted({r["policy"] for r in family_rows}))
    values = {(r["family"], r["policy"]): float(r["objective_mean"]) for r in family_rows}
    best = {fam: min(values[(fam, p)] for p in policies if (fam, p) in values) for fam in families}
    angles = np.linspace(0, 2 * np.pi, len(families), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.8, 6.2), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(PAPER_COLORS["background"])
    ax.set_facecolor(PAPER_COLORS["background"])
    for pol in policies:
        vals = []
        for fam in families:
            raw = values.get((fam, pol), np.nan)
            vals.append(min(1.0, best[fam] / max(raw, 1e-9)) if np.isfinite(raw) else np.nan)
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.25, label=policy_label(pol), color=RL_COLORS.get(pol, PAPER_COLORS["soft_gray"]))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(families, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "best"], fontsize=7)
    ax.set_title("Family-level normalized objective performance", pad=18)
    ax.grid(color=PAPER_COLORS["grid"], alpha=0.8)
    ax.legend(fontsize=7, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_rl_family_radar.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history_rows: Sequence[Dict[str, str]], output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not history_rows:
        return
    algorithms = sorted({r["algorithm"] for r in history_rows})
    metrics_by_algo = {
        "dqn": ["train_reward_mean_window", "loss"],
        "ppo": ["train_reward_mean_window", "loss"],
        "sac": ["train_reward_mean_window", "actor_loss", "q_loss"],
    }
    panels = [(algo, metric) for algo in algorithms for metric in metrics_by_algo.get(algo, ["train_reward_mean_window"])]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.2, max(3.2, 2.1 * len(panels))), sharex=False)
    if len(panels) == 1:
        axes = [axes]
    fig.patch.set_facecolor(PAPER_COLORS["background"])
    for ax, (algo, metric) in zip(axes, panels):
        rows = [r for r in history_rows if r["algorithm"] == algo]
        grouped: Dict[int, List[float]] = {}
        for r in rows:
            raw = r.get(metric, "")
            if raw == "":
                continue
            grouped.setdefault(int(float(r["step"])), []).append(float(raw))
        if grouped:
            xs = sorted(grouped)
            ys = [float(np.mean(grouped[x])) for x in xs]
            ax.plot(xs, ys, color=RL_COLORS.get(algo, PAPER_COLORS["highlight"]), linewidth=1.4)
        ax.set_title(f"{policy_label(algo)} training: {metric}")
        ax.set_xlabel("Training step")
        ax.set_ylabel(metric)
        ax.grid(color=PAPER_COLORS["grid"], alpha=0.7)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_rl_training_curves.pdf"), bbox_inches="tight")
    plt.close(fig)


def write_training_budget_artifacts(cfg: RLConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    payload = asdict(cfg)
    payload["algorithms"] = list(cfg.algorithms)
    payload["families"] = list(cfg.families)
    payload["budget_presets"] = BUDGET_PRESETS
    payload["notes"] = [
        "Review preset increases DQN/PPO/SAC from the earlier smoke budget of 12k steps and one seed to 100k steps and five independent training seeds.",
        "Full preset is intended for a camera-ready or appendix-grade robustness run.",
        "Hyperparameters are fixed across case families unless explicitly overridden on the command line.",
        "Evaluation uses common evaluation replications across trained agents and fixed structural policies.",
    ]
    with open(os.path.join(cfg.output_dir, "paper1_rl_training_budget.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# RL Training Budget and Tuning Protocol",
        "",
        "## Budget",
        "",
        f"- Budget preset: `{cfg.budget_preset}`.",
        f"- Algorithms: `{','.join(cfg.algorithms)}`.",
        f"- Training steps per algorithm, instance, and training seed: `{cfg.train_steps}`.",
        f"- Independent training seeds per algorithm and instance: `{cfg.train_seeds}`.",
        f"- Evaluation replications per trained policy: `{cfg.replications}`.",
        f"- Warmup/evaluation steps per replication: `{cfg.warmup_steps}` / `{cfg.eval_steps}`.",
        f"- Learning-curve smoothing window: `{cfg.curve_window}` environment steps.",
        "",
        "## Hyperparameters",
        "",
        f"- Discount factor gamma: `{cfg.gamma}`.",
        f"- Learning rate: `{cfg.lr}`.",
        f"- Batch size: `{cfg.batch_size}`.",
        f"- Hidden units per MLP layer: `{cfg.hidden}`.",
        f"- Replay buffer size: `{cfg.buffer_size}`.",
        f"- Off-policy update_after/update_every: `{cfg.update_after}` / `{cfg.update_every}`.",
        f"- Target-network Polyak coefficient: `{cfg.target_tau}`.",
        f"- PPO epochs/clip/entropy/value coefficients: `{cfg.ppo_epochs}` / `{cfg.ppo_clip}` / `{cfg.entropy_coef}` / `{cfg.value_coef}`.",
        f"- SAC entropy temperature alpha: `{cfg.alpha}`.",
        "",
        "## Tuning Protocol",
        "",
        "- The default `review` budget is the primary paper-facing budget; `smoke` is only for code checks.",
        "- Hyperparameters are shared across all case families to avoid per-instance test-set tuning.",
        "- Candidate changes should be selected on aggregate training-curve stability and validation-style evaluation, then frozen before final policy-comparison runs.",
        "- Report `paper1_rl_training_history.csv` and `fig_rl_training_curves.pdf` with the final tables so reviewers can inspect convergence behavior.",
        "- When compute permits, rerun with `--budget-preset full` and report it as a robustness appendix check.",
        "",
    ]
    with open(os.path.join(cfg.output_dir, "paper1_rl_tuning_protocol.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_rl_outputs(output_dir: str) -> None:
    family_rows = read_csv_rows(os.path.join(output_dir, "paper1_rl_policy_family_summary.csv"))
    case_rows = read_csv_rows(os.path.join(output_dir, "paper1_rl_policy_case_summary.csv"))
    history_rows = read_csv_rows(os.path.join(output_dir, "paper1_rl_training_history.csv"))
    plot_family_log_bars(family_rows, output_dir)
    plot_rl_relative_heatmap(case_rows, output_dir)
    plot_rl_radar(family_rows, output_dir)
    plot_training_curves(history_rows, output_dir)


def fixed_policy_cache_paths(source_dir: str) -> List[str]:
    if not source_dir:
        return []
    return [
        os.path.join(source_dir, "paper1_rl_policy_replications.csv"),
        os.path.join(source_dir, "paper1_policy_replications.csv"),
    ]


def rl_replication_cache_paths(source_dir: str) -> List[str]:
    if not source_dir:
        return []
    return [os.path.join(source_dir, "paper1_rl_policy_replications.csv")]


def rl_history_cache_paths(source_dir: str) -> List[str]:
    if not source_dir:
        return []
    return [os.path.join(source_dir, "paper1_rl_training_history.csv")]


def reuse_source_dirs(cfg: RLConfig) -> List[str]:
    out: List[str] = []
    for source_dir in [cfg.reuse_fixed_policy_dir, cfg.output_dir]:
        if source_dir and source_dir not in out:
            out.append(source_dir)
    return out


def normalize_replication_row(raw: Dict[str, str], policy: str, replication: int, seed: int, train_seed: str = "") -> Dict[str, object]:
    label = RL_LABELS.get(policy, POLICY_LABELS.get(policy, policy))
    row: Dict[str, object] = {
        "family": raw.get("family", ""),
        "case_id": raw.get("case_id", ""),
        "policy": policy,
        "policy_label": raw.get("policy_label", label),
        "train_seed": train_seed,
        "replication": replication,
        "seed": seed,
    }
    for col in METRIC_COLUMNS:
        if col in raw:
            row[col] = raw[col]
    return row


def expected_fixed_keys(cases: Sequence[object], cfg: RLConfig) -> Dict[Tuple[str, str, str, int, int], Tuple[int, object]]:
    expected: Dict[Tuple[str, str, str, int, int], Tuple[int, object]] = {}
    for ci, case in enumerate(cases, start=1):
        for r in range(cfg.replications):
            seed = cfg.seed + 100000 * ci + r
            for policy in FIXED_POLICIES:
                expected[(str(case.family), str(case.case_id), policy, r, seed)] = (ci, case)
    return expected


def load_reusable_fixed_rows(cases: Sequence[object], cfg: RLConfig) -> Dict[Tuple[str, str], List[Dict[str, object]]]:
    expected = expected_fixed_keys(cases, cfg)
    rows_by_case: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for source_dir in reuse_source_dirs(cfg):
        for path in fixed_policy_cache_paths(source_dir):
            if not os.path.exists(path):
                continue
            for raw in read_csv_rows(path):
                try:
                    replication = int(float(raw.get("replication", "")))
                    seed = int(float(raw.get("seed", "")))
                except ValueError:
                    continue
                policy = str(raw.get("policy", ""))
                key = (str(raw.get("family", "")), str(raw.get("case_id", "")), policy, replication, seed)
                if key not in expected:
                    continue
                row = normalize_replication_row(raw, policy, replication, seed, "")
                case_key = (key[0], key[1])
                existing_keys = {
                    (str(x["policy"]), int(float(x["replication"])), int(float(x["seed"])))
                    for x in rows_by_case.get(case_key, [])
                }
                if (policy, replication, seed) not in existing_keys:
                    rows_by_case.setdefault(case_key, []).append(row)
    reused = sum(len(v) for v in rows_by_case.values())
    if reused:
        print(f"Reused {reused} fixed-policy replication rows from existing CSV files.", flush=True)
    return rows_by_case


def expected_rl_agents(cases: Sequence[object], cfg: RLConfig) -> Dict[Tuple[str, str, str, str], set[Tuple[int, int]]]:
    expected: Dict[Tuple[str, str, str, str], set[Tuple[int, int]]] = {}
    for ci, case in enumerate(cases, start=1):
        for algo in cfg.algorithms:
            for ts in range(cfg.train_seeds):
                algo_offset = {"dqn": 101, "ppo": 211, "sac": 307}[algo]
                train_seed = str(cfg.seed + 777000 * ci + 10000 * (ts + 1) + algo_offset)
                agent_key = (str(case.family), str(case.case_id), algo, train_seed)
                expected[agent_key] = {
                    (r, cfg.seed + 100000 * ci + r)
                    for r in range(cfg.replications)
                }
    return expected


def load_reusable_rl_artifacts(
    cases: Sequence[object],
    cfg: RLConfig,
) -> Tuple[Dict[Tuple[str, str], Dict[Tuple[str, str], List[Dict[str, object]]]], Dict[Tuple[str, str], Dict[Tuple[str, str], List[Dict[str, object]]]]]:
    expected = expected_rl_agents(cases, cfg)
    candidate_rows: Dict[Tuple[str, str, str, str], Dict[Tuple[int, int], Dict[str, object]]] = {}
    source_dirs = reuse_source_dirs(cfg)
    for source_dir in source_dirs:
        for path in rl_replication_cache_paths(source_dir):
            if not os.path.exists(path):
                continue
            for raw in read_csv_rows(path):
                policy = str(raw.get("policy", ""))
                if policy not in cfg.algorithms:
                    continue
                train_seed = str(raw.get("train_seed", ""))
                if not train_seed:
                    continue
                try:
                    replication = int(float(raw.get("replication", "")))
                    seed = int(float(raw.get("seed", "")))
                except ValueError:
                    continue
                agent_key = (str(raw.get("family", "")), str(raw.get("case_id", "")), policy, train_seed)
                if agent_key not in expected or (replication, seed) not in expected[agent_key]:
                    continue
                row = normalize_replication_row(raw, policy, replication, seed, train_seed)
                candidate_rows.setdefault(agent_key, {})[(replication, seed)] = row

    complete_agents = {
        agent_key
        for agent_key, required in expected.items()
        if agent_key in candidate_rows and required.issubset(set(candidate_rows[agent_key]))
    }

    rep_by_case_agent: Dict[Tuple[str, str], Dict[Tuple[str, str], List[Dict[str, object]]]] = {}
    for family, case_id, algo, train_seed in sorted(complete_agents):
        rows = [
            candidate_rows[(family, case_id, algo, train_seed)][key]
            for key in sorted(expected[(family, case_id, algo, train_seed)])
        ]
        rep_by_case_agent.setdefault((family, case_id), {})[(algo, train_seed)] = rows

    hist_by_case_agent: Dict[Tuple[str, str], Dict[Tuple[str, str], List[Dict[str, object]]]] = {}
    if complete_agents:
        for source_dir in source_dirs:
            for path in rl_history_cache_paths(source_dir):
                if not os.path.exists(path):
                    continue
                for raw in read_csv_rows(path):
                    agent_key = (
                        str(raw.get("family", "")),
                        str(raw.get("case_id", "")),
                        str(raw.get("algorithm", "")),
                        str(raw.get("train_seed", "")),
                    )
                    if agent_key not in complete_agents:
                        continue
                    family, case_id, algo, train_seed = agent_key
                    row: Dict[str, object] = dict(raw)
                    hist_by_case_agent.setdefault((family, case_id), {}).setdefault((algo, train_seed), []).append(row)

    reused_rows = sum(len(rows) for agents in rep_by_case_agent.values() for rows in agents.values())
    if reused_rows:
        print(f"Reused {reused_rows} complete RL replication rows from {len(complete_agents)} trained agents.", flush=True)
    return rep_by_case_agent, hist_by_case_agent


def _run_case(
    payload: Tuple[
        int,
        object,
        RLConfig,
        List[Dict[str, object]],
        Dict[Tuple[str, str], List[Dict[str, object]]],
        Dict[Tuple[str, str], List[Dict[str, object]]],
    ]
) -> Tuple[int, int, List[Dict[str, object]], List[Dict[str, object]]]:
    ci, case, cfg, reused_fixed_rows, reused_rl_rows_by_agent, reused_rl_hist_by_agent = payload
    rep_rows: List[Dict[str, object]] = []
    hist_rows: List[Dict[str, object]] = []
    problem = make_problem(case, cfg.seed + 1009 * ci)
    reused_count = len(reused_fixed_rows) + sum(len(rows) for rows in reused_rl_rows_by_agent.values())
    reused_fixed_keys = {
        (str(row["policy"]), int(float(row["replication"])), int(float(row["seed"])))
        for row in reused_fixed_rows
    }
    rep_rows.extend(reused_fixed_rows)
    for r in range(cfg.replications):
        eval_seed = cfg.seed + 100000 * ci + r
        for policy in FIXED_POLICIES:
            if (policy, r, eval_seed) in reused_fixed_keys:
                continue
            row = {
                "family": case.family,
                "case_id": case.case_id,
                "policy": policy,
                "policy_label": POLICY_LABELS.get(policy, policy),
                "train_seed": "",
                "replication": r,
                "seed": eval_seed,
            }
            row.update(simulate_policy(problem, policy, eval_seed, cfg.warmup_steps, cfg.eval_steps))
            rep_rows.append(row)
    for algo in cfg.algorithms:
        for ts in range(cfg.train_seeds):
            algo_offset = {"dqn": 101, "ppo": 211, "sac": 307}[algo]
            train_seed = cfg.seed + 777000 * ci + 10000 * (ts + 1) + algo_offset
            reused_agent_key = (algo, str(train_seed))
            if reused_agent_key in reused_rl_rows_by_agent:
                rep_rows.extend(reused_rl_rows_by_agent[reused_agent_key])
                hist_rows.extend(reused_rl_hist_by_agent.get(reused_agent_key, []))
                continue
            progress_label = f"RL train {case.family}/{case.case_id} {algo} train_seed={train_seed}"
            if algo == "dqn":
                agent, hist = train_dqn(problem, cfg, train_seed, progress_label)
            elif algo == "ppo":
                agent, hist = train_ppo(problem, cfg, train_seed, progress_label)
            elif algo == "sac":
                agent, hist = train_sac(problem, cfg, train_seed, progress_label)
            else:
                raise ValueError(f"unknown algorithm {algo}")
            for h in hist:
                hist_rows.append({
                    "family": case.family,
                    "case_id": case.case_id,
                    "algorithm": algo,
                    "train_seed": train_seed,
                    "budget_preset": cfg.budget_preset,
                    "train_steps": cfg.train_steps,
                    "gamma": cfg.gamma,
                    "lr": cfg.lr,
                    "batch_size": cfg.batch_size,
                    "hidden": cfg.hidden,
                    **h,
                })
            for r in range(cfg.replications):
                eval_seed = cfg.seed + 100000 * ci + r
                row = {
                    "family": case.family,
                    "case_id": case.case_id,
                    "policy": algo,
                    "policy_label": RL_LABELS[algo],
                    "train_seed": train_seed,
                    "replication": r,
                    "seed": eval_seed,
                }
                row.update(evaluate_agent(problem, agent, eval_seed, cfg.warmup_steps, cfg.eval_steps))
                rep_rows.append(row)
    return ci, reused_count, rep_rows, hist_rows


def run(cfg: RLConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    write_training_budget_artifacts(cfg)
    cases = policy_cases(RunConfig(output_dir=cfg.output_dir, families=cfg.families))
    if cfg.case_ids:
        wanted = set(cfg.case_ids)
        cases = [case for case in cases if case.case_id in wanted]
        missing = sorted(wanted - {case.case_id for case in cases})
        if missing:
            raise ValueError(f"unknown --case-ids for selected families: {','.join(missing)}")
    reused_fixed_rows_by_case = load_reusable_fixed_rows(cases, cfg)
    reused_rl_rows_by_case_agent, reused_rl_hist_by_case_agent = load_reusable_rl_artifacts(cases, cfg)
    if cfg.jobs > 1:
        rep_rows: List[Dict[str, object]] = []
        hist_rows: List[Dict[str, object]] = []
        tasks = [
            (
                ci,
                case,
                cfg,
                reused_fixed_rows_by_case.get((str(case.family), str(case.case_id)), []),
                reused_rl_rows_by_case_agent.get((str(case.family), str(case.case_id)), {}),
                reused_rl_hist_by_case_agent.get((str(case.family), str(case.case_id)), {}),
            )
            for ci, case in enumerate(cases, start=1)
        ]
        total = len(cases) * (len(FIXED_POLICIES) * cfg.replications + len(cfg.algorithms) * cfg.train_seeds * cfg.replications)
        done = (
            sum(len(rows) for rows in reused_fixed_rows_by_case.values())
            + sum(len(rows) for agents in reused_rl_rows_by_case_agent.values() for rows in agents.values())
        )
        render_progress(done, total, f"starting RL baselines with jobs={cfg.jobs}")
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=cfg.jobs, mp_context=ctx) as executor:
            futures = {executor.submit(_run_case, task): task[1].case_id for task in tasks}
            completed = []
            for future in as_completed(futures):
                ci, reused_count, case_rep_rows, case_hist_rows = future.result()
                completed.append((ci, case_rep_rows, case_hist_rows))
                done += len(case_rep_rows) - reused_count
                render_progress(done, total, f"finished {futures[future]}")
        for _, case_rep_rows, case_hist_rows in sorted(completed, key=lambda item: item[0]):
            rep_rows.extend(case_rep_rows)
            hist_rows.extend(case_hist_rows)
        case_summary = summarize(rep_rows, ["family", "case_id", "policy", "policy_label"])
        family_summary = summarize(rep_rows, ["family", "policy", "policy_label"])
        write_csv(os.path.join(cfg.output_dir, "paper1_rl_policy_replications.csv"), rep_rows)
        write_csv(os.path.join(cfg.output_dir, "paper1_rl_policy_case_summary.csv"), case_summary)
        write_csv(os.path.join(cfg.output_dir, "paper1_rl_policy_family_summary.csv"), family_summary)
        write_csv(os.path.join(cfg.output_dir, "paper1_rl_training_history.csv"), hist_rows)
        plot_rl_outputs(cfg.output_dir)
        return
    rep_rows: List[Dict[str, object]] = []
    hist_rows: List[Dict[str, object]] = []
    total = len(cases) * (len(FIXED_POLICIES) * cfg.replications + len(cfg.algorithms) * cfg.train_seeds * cfg.replications)
    done = (
        sum(len(rows) for rows in reused_fixed_rows_by_case.values())
        + sum(len(rows) for agents in reused_rl_rows_by_case_agent.values() for rows in agents.values())
    )
    render_progress(done, total, "starting RL baselines")
    for ci, case in enumerate(cases, start=1):
        problem = make_problem(case, cfg.seed + 1009 * ci)
        reused_fixed_rows = reused_fixed_rows_by_case.get((str(case.family), str(case.case_id)), [])
        reused_rl_rows_by_agent = reused_rl_rows_by_case_agent.get((str(case.family), str(case.case_id)), {})
        reused_rl_hist_by_agent = reused_rl_hist_by_case_agent.get((str(case.family), str(case.case_id)), {})
        reused_fixed_keys = {
            (str(row["policy"]), int(float(row["replication"])), int(float(row["seed"])))
            for row in reused_fixed_rows
        }
        rep_rows.extend(reused_fixed_rows)
        for r in range(cfg.replications):
            eval_seed = cfg.seed + 100000 * ci + r
            for policy in FIXED_POLICIES:
                if (policy, r, eval_seed) in reused_fixed_keys:
                    continue
                row = {
                    "family": case.family,
                    "case_id": case.case_id,
                    "policy": policy,
                    "policy_label": POLICY_LABELS.get(policy, policy),
                    "train_seed": "",
                    "replication": r,
                    "seed": eval_seed,
                }
                row.update(simulate_policy(problem, policy, eval_seed, cfg.warmup_steps, cfg.eval_steps))
                rep_rows.append(row)
                done += 1
                render_progress(done, total, f"{case.case_id} fixed {policy}")
        for algo in cfg.algorithms:
            for ts in range(cfg.train_seeds):
                algo_offset = {"dqn": 101, "ppo": 211, "sac": 307}[algo]
                train_seed = cfg.seed + 777000 * ci + 10000 * (ts + 1) + algo_offset
                reused_agent_key = (algo, str(train_seed))
                if reused_agent_key in reused_rl_rows_by_agent:
                    rep_rows.extend(reused_rl_rows_by_agent[reused_agent_key])
                    hist_rows.extend(reused_rl_hist_by_agent.get(reused_agent_key, []))
                    render_progress(done, total, f"{case.case_id} {algo} seed={train_seed} reused")
                    continue
                progress_label = f"RL train {case.family}/{case.case_id} {algo} train_seed={train_seed}"
                if algo == "dqn":
                    agent, hist = train_dqn(problem, cfg, train_seed, progress_label)
                elif algo == "ppo":
                    agent, hist = train_ppo(problem, cfg, train_seed, progress_label)
                elif algo == "sac":
                    agent, hist = train_sac(problem, cfg, train_seed, progress_label)
                else:
                    raise ValueError(f"unknown algorithm {algo}")
                for h in hist:
                    hist_rows.append({
                        "family": case.family,
                        "case_id": case.case_id,
                        "algorithm": algo,
                        "train_seed": train_seed,
                        "budget_preset": cfg.budget_preset,
                        "train_steps": cfg.train_steps,
                        "gamma": cfg.gamma,
                        "lr": cfg.lr,
                        "batch_size": cfg.batch_size,
                        "hidden": cfg.hidden,
                        **h,
                    })
                for r in range(cfg.replications):
                    eval_seed = cfg.seed + 100000 * ci + r
                    row = {
                        "family": case.family,
                        "case_id": case.case_id,
                        "policy": algo,
                        "policy_label": RL_LABELS[algo],
                        "train_seed": train_seed,
                        "replication": r,
                        "seed": eval_seed,
                    }
                    row.update(evaluate_agent(problem, agent, eval_seed, cfg.warmup_steps, cfg.eval_steps))
                    rep_rows.append(row)
                    done += 1
                    render_progress(done, total, f"{case.case_id} {algo} seed={train_seed}")
    case_summary = summarize(rep_rows, ["family", "case_id", "policy", "policy_label"])
    family_summary = summarize(rep_rows, ["family", "policy", "policy_label"])
    write_csv(os.path.join(cfg.output_dir, "paper1_rl_policy_replications.csv"), rep_rows)
    write_csv(os.path.join(cfg.output_dir, "paper1_rl_policy_case_summary.csv"), case_summary)
    write_csv(os.path.join(cfg.output_dir, "paper1_rl_policy_family_summary.csv"), family_summary)
    write_csv(os.path.join(cfg.output_dir, "paper1_rl_training_history.csv"), hist_rows)
    plot_rl_outputs(cfg.output_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="Train DQN/PPO/SAC RL baselines for Paper 1 MAP-gPAH experiments.")
    p.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    p.add_argument("--algorithms", default="dqn,ppo,sac")
    p.add_argument(
        "--families",
        default="A1_topology,A2_map_burst,A3_density,A4_service,A5_switch,A6_scale,A7_load,A8_large_scale",
    )
    p.add_argument("--case-ids", "--case_ids", dest="case_ids", default="", help="Optional comma-separated case_id subset within selected families.")
    p.add_argument("--replications", type=int, default=10)
    p.add_argument("--budget-preset", choices=sorted(BUDGET_PRESETS), default="review")
    p.add_argument("--train-seeds", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=3000)
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260526)
    p.add_argument("--device", default="cpu")
    p.add_argument("--jobs", type=int, default=1, help="Number of parallel case workers. Use 1 for serial execution.")
    p.add_argument("--no-train-progress", action="store_true", help="Disable per-agent RL training progress bars.")
    p.add_argument(
        "--reuse-fixed-policy-dir",
        "--reuse_fixed_policy_dir",
        dest="reuse_fixed_policy_dir",
        default="",
        help="Optional directory containing paper1_rl_policy_replications.csv or paper1_policy_replications.csv fixed-policy rows to reuse when seeds match.",
    )
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=50000)
    p.add_argument("--update-after", type=int, default=500)
    p.add_argument("--update-every", type=int, default=4)
    p.add_argument("--target-tau", type=float, default=0.01)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--ppo-clip", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.08)
    p.add_argument("--curve-window", type=int, default=500)
    p.add_argument("--plot-only", action="store_true", help="Only generate figures from existing RL CSV files.")
    args = p.parse_args()
    if args.plot_only:
        plot_rl_outputs(args.output_dir)
        return
    preset = BUDGET_PRESETS[args.budget_preset]
    train_steps = int(args.train_steps if args.train_steps is not None else preset["train_steps"])
    train_seeds = int(args.train_seeds if args.train_seeds is not None else preset["train_seeds"])
    cfg = RLConfig(
        output_dir=args.output_dir,
        algorithms=parse_csv(args.algorithms),
        families=parse_csv(args.families),
        case_ids=parse_csv(args.case_ids),
        replications=args.replications,
        train_seeds=train_seeds,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        train_steps=train_steps,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
        device=args.device,
        jobs=max(1, int(args.jobs)),
        reuse_fixed_policy_dir=args.reuse_fixed_policy_dir,
        train_progress=not args.no_train_progress,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden=args.hidden,
        buffer_size=args.buffer_size,
        update_after=args.update_after,
        update_every=args.update_every,
        target_tau=args.target_tau,
        ppo_epochs=args.ppo_epochs,
        ppo_clip=args.ppo_clip,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        alpha=args.alpha,
        curve_window=args.curve_window,
        budget_preset=args.budget_preset,
    )
    run(cfg)


if __name__ == "__main__":
    main()
