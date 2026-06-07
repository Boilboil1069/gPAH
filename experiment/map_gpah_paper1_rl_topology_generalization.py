#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from map_gpah_paper1_common import (
    PAPER_COLORS,
    RunConfig,
    SparseMapPhEnv,
    make_problem,
    policy_cases,
    render_progress,
    simulate_policy,
    write_csv,
)
from map_gpah_paper1_rl_baselines import (
    BUDGET_PRESETS,
    RL_COLORS,
    RL_LABELS,
    DQNAgent,
    PPOAgent,
    ReplayBuffer,
    RLConfig,
    SACAgent,
    action_mask,
    candidate_actions,
    evaluate_agent,
    observation,
    parse_csv,
    reward_from_info,
    summarize,
)


DEFAULT_TRAIN_TOPOLOGIES = ("disjoint", "chain", "ring", "star", "high_overlap")
DEFAULT_TEST_TOPOLOGIES = ("bridge_bottleneck", "random_sparse")
ALGO_OFFSETS = {"dqn": 101, "ppo": 211, "sac": 307}


def set_agent_problem(agent, problem) -> None:
    agent.problem = problem
    if hasattr(agent, "mask"):
        agent.mask = torch.as_tensor(action_mask(problem), dtype=torch.bool, device=agent.device)


def choose_problem(problems: Sequence[object], rng: np.random.Generator, agent=None):
    problem = problems[int(rng.integers(0, len(problems)))]
    if agent is not None:
        set_agent_problem(agent, problem)
    return problem


def train_multitopology_dqn(
    problems: Sequence[object], cfg: RLConfig, seed: int, label: str
) -> Tuple[DQNAgent, List[Dict[str, float]]]:
    agent = DQNAgent(problems[0], cfg, seed)
    rng = np.random.default_rng(seed + 17)
    rb = ReplayBuffer(cfg.buffer_size)
    hist: List[Dict[str, float]] = []
    rewards: List[float] = []
    report_every = max(1, cfg.train_steps // 50)
    block_steps = max(1, cfg.rollout_steps)
    env = None
    problem = None
    obs = None
    metrics: Dict[str, float] = {}
    render_progress(0, cfg.train_steps, label)
    for step in range(1, cfg.train_steps + 1):
        if env is None or (step - 1) % block_steps == 0:
            problem = choose_problem(problems, rng, agent)
            env = SparseMapPhEnv(problem, seed + 1000003 * step)
            env.reset()
            obs = observation(env)
        assert problem is not None and env is not None and obs is not None
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
        obs = nxt
        rewards.append(reward)
        if len(rewards) > max(1, cfg.curve_window):
            rewards.pop(0)
        if len(rb) >= max(cfg.update_after, cfg.batch_size) and step % cfg.update_every == 0:
            metrics = agent.update(rb)
        if step % report_every == 0 or step == cfg.train_steps:
            row = {
                "step": step,
                "epsilon": eps,
                "train_reward_mean_window": float(np.mean(rewards)) if rewards else 0.0,
                **metrics,
            }
            hist.append(row)
            render_progress(step, cfg.train_steps, f"{label} reward={row['train_reward_mean_window']:.4g}")
    return agent, hist


def train_multitopology_ppo(
    problems: Sequence[object], cfg: RLConfig, seed: int, label: str
) -> Tuple[PPOAgent, List[Dict[str, float]]]:
    agent = PPOAgent(problems[0], cfg, seed)
    rng = np.random.default_rng(seed + 29)
    hist: List[Dict[str, float]] = []
    completed = 0
    report_every = max(1, cfg.train_steps // 50)
    last_report = 0
    render_progress(0, cfg.train_steps, label)
    while completed < cfg.train_steps:
        problem = choose_problem(problems, rng, agent)
        env = SparseMapPhEnv(problem, seed + 1000003 * (completed + 1))
        env.reset()
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
        for reward in reversed(rew_l):
            g = reward + cfg.gamma * g
            returns.append(g)
        returns.reverse()
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
            pg = -torch.minimum(
                ratio * adv_t,
                torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * adv_t,
            ).mean()
            value_loss = F.mse_loss(values, ret_t)
            loss = pg + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            agent.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.net.parameters(), 5.0)
            agent.opt.step()
            last_loss = float(loss.item())
        if completed - last_report >= report_every or completed >= cfg.train_steps:
            row = {
                "step": completed,
                "loss": last_loss,
                "train_reward_mean_window": float(np.mean(rew_l)),
            }
            hist.append(row)
            render_progress(completed, cfg.train_steps, f"{label} reward={row['train_reward_mean_window']:.4g}")
            last_report = completed
    return agent, hist


def train_multitopology_sac(
    problems: Sequence[object], cfg: RLConfig, seed: int, label: str
) -> Tuple[SACAgent, List[Dict[str, float]]]:
    agent = SACAgent(problems[0], cfg, seed)
    rng = np.random.default_rng(seed + 43)
    hist: List[Dict[str, float]] = []
    rewards: List[float] = []
    report_every = max(1, cfg.train_steps // 50)
    block_steps = max(cfg.update_after * 2, cfg.rollout_steps * 8)
    env = None
    problem = None
    rb = ReplayBuffer(cfg.buffer_size)
    metrics: Dict[str, float] = {}
    render_progress(0, cfg.train_steps, label)
    for step in range(1, cfg.train_steps + 1):
        block_pos = (step - 1) % block_steps
        if env is None or block_pos == 0:
            problem = choose_problem(problems, rng, agent)
            env = SparseMapPhEnv(problem, seed + 1000003 * step)
            env.reset()
            rb = ReplayBuffer(cfg.buffer_size)
        assert problem is not None and env is not None
        if block_pos < cfg.update_after:
            mask = action_mask(problem)
            act = np.asarray([agent.rng.choice(np.flatnonzero(mask[k])) for k in range(problem.K)], dtype=int)
        else:
            act = agent.action(env, deterministic=False)
        obs = observation(env)
        info = env.step(act)
        reward = reward_from_info(problem, info)
        rb.add(obs, act, reward, observation(env), False)
        rewards.append(reward)
        if len(rewards) > max(1, cfg.curve_window):
            rewards.pop(0)
        if len(rb) >= max(cfg.update_after, cfg.batch_size) and step % cfg.update_every == 0:
            metrics = agent.update(rb)
        if step % report_every == 0 or step == cfg.train_steps:
            row = {
                "step": step,
                "train_reward_mean_window": float(np.mean(rewards)) if rewards else 0.0,
                **metrics,
            }
            hist.append(row)
            render_progress(step, cfg.train_steps, f"{label} reward={row['train_reward_mean_window']:.4g}")
    return agent, hist


def save_agent(path: Path, algo: str, agent) -> None:
    if algo == "dqn":
        payload = {"q": agent.q.state_dict(), "target": agent.target.state_dict()}
    elif algo == "ppo":
        payload = {"net": agent.net.state_dict()}
    elif algo == "sac":
        payload = {
            "actor": agent.actor.state_dict(),
            "q": agent.q.state_dict(),
            "target": agent.target.state_dict(),
        }
    else:
        raise ValueError(algo)
    torch.save(payload, path)


def load_agent(path: Path, algo: str, problem, cfg: RLConfig, seed: int):
    payload = torch.load(path, map_location=cfg.device, weights_only=True)
    if algo == "dqn":
        agent = DQNAgent(problem, cfg, seed)
        agent.q.load_state_dict(payload["q"])
        agent.target.load_state_dict(payload["target"])
    elif algo == "ppo":
        agent = PPOAgent(problem, cfg, seed)
        agent.net.load_state_dict(payload["net"])
    elif algo == "sac":
        agent = SACAgent(problem, cfg, seed)
        agent.actor.load_state_dict(payload["actor"])
        agent.q.load_state_dict(payload["q"])
        agent.target.load_state_dict(payload["target"])
    else:
        raise ValueError(algo)
    return agent


def plot_results(case_rows: Sequence[Dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    topologies = sorted({str(row["test_topology"]) for row in case_rows})
    policies = ["gpah", "dqn", "ppo", "sac"]
    values = {
        (str(row["test_topology"]), str(row["policy"])): float(row["objective_mean"])
        for row in case_rows
    }
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    fig.patch.set_facecolor(PAPER_COLORS["background"])
    ax.set_facecolor(PAPER_COLORS["background"])
    x = np.arange(len(topologies))
    width = 0.8 / len(policies)
    for j, policy in enumerate(policies):
        vals = [values.get((topology, policy), np.nan) for topology in topologies]
        ax.bar(
            x + (j - (len(policies) - 1) / 2) * width,
            vals,
            width,
            label=RL_LABELS.get(policy, "gPAH"),
            color=RL_COLORS.get(policy, PAPER_COLORS["highlight"]),
            edgecolor="none",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(topologies, rotation=18, ha="right")
    ax.set_ylabel("Standardized objective")
    ax.grid(axis="y", color=PAPER_COLORS["grid"], alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(output_dir / "fig_rl_unseen_topology_objective.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incremental shared-policy RL test on topologies excluded from training."
    )
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--algorithms", default="dqn,ppo,sac")
    parser.add_argument("--train-topologies", default=",".join(DEFAULT_TRAIN_TOPOLOGIES))
    parser.add_argument("--test-topologies", default=",".join(DEFAULT_TEST_TOPOLOGIES))
    parser.add_argument("--budget-preset", choices=sorted(BUDGET_PRESETS), default="full")
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--train-seeds", type=int, default=None)
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=3000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20360527)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-train-progress", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    algorithms = parse_csv(args.algorithms)
    train_topologies = parse_csv(args.train_topologies)
    test_topologies = parse_csv(args.test_topologies)
    if set(train_topologies) & set(test_topologies):
        raise ValueError("training and held-out test topologies must be disjoint")

    all_cases = policy_cases(RunConfig(output_dir=str(output_dir), families=("A1_topology",)))
    cases_by_id = {case.case_id: case for case in all_cases}
    missing = sorted((set(train_topologies) | set(test_topologies)) - set(cases_by_id))
    if missing:
        raise ValueError(f"unknown topologies: {','.join(missing)}")
    case_order = {case.case_id: i for i, case in enumerate(all_cases, start=1)}
    problems = {
        topology: make_problem(cases_by_id[topology], args.seed + 1009 * case_order[topology])
        for topology in set(train_topologies) | set(test_topologies)
    }
    dims = {(problem.d, problem.K) for problem in problems.values()}
    if len(dims) != 1:
        raise ValueError("shared-policy topology test requires common d and K")

    preset = BUDGET_PRESETS[args.budget_preset]
    cfg = RLConfig(
        output_dir=str(output_dir),
        algorithms=algorithms,
        families=("A1_topology_generalization",),
        case_ids=test_topologies,
        replications=args.replications,
        train_seeds=int(args.train_seeds if args.train_seeds is not None else preset["train_seeds"]),
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        train_steps=int(args.train_steps if args.train_steps is not None else preset["train_steps"]),
        rollout_steps=args.rollout_steps,
        seed=args.seed,
        device=args.device,
        train_progress=not args.no_train_progress,
        budget_preset=args.budget_preset,
    )
    manifest = {
        **asdict(cfg),
        "algorithms": list(algorithms),
        "train_topologies": list(train_topologies),
        "heldout_test_topologies": list(test_topologies),
        "protocol": "one shared policy per algorithm and training seed; zero fine-tuning on held-out topologies",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rep_rows: List[Dict[str, object]] = []
    hist_rows: List[Dict[str, object]] = []
    train_problems = [problems[topology] for topology in train_topologies]
    total_agents = len(algorithms) * cfg.train_seeds
    completed_agents = 0
    for algo in algorithms:
        for ts in range(cfg.train_seeds):
            train_seed = cfg.seed + 10000 * (ts + 1) + ALGO_OFFSETS[algo]
            checkpoint = checkpoint_dir / f"{algo}_seed_{train_seed}.pt"
            label = f"unseen-topology train {algo} seed={train_seed}"
            if checkpoint.exists():
                agent = load_agent(checkpoint, algo, train_problems[0], cfg, train_seed)
                hist = []
                print(f"Reused checkpoint {checkpoint}", flush=True)
            elif algo == "dqn":
                agent, hist = train_multitopology_dqn(train_problems, cfg, train_seed, label)
                save_agent(checkpoint, algo, agent)
            elif algo == "ppo":
                agent, hist = train_multitopology_ppo(train_problems, cfg, train_seed, label)
                save_agent(checkpoint, algo, agent)
            elif algo == "sac":
                agent, hist = train_multitopology_sac(train_problems, cfg, train_seed, label)
                save_agent(checkpoint, algo, agent)
            else:
                raise ValueError(f"unknown algorithm {algo}")
            for row in hist:
                hist_rows.append({
                    "algorithm": algo,
                    "train_seed": train_seed,
                    "train_topologies": ",".join(train_topologies),
                    **row,
                })
            for ti, topology in enumerate(test_topologies, start=1):
                problem = problems[topology]
                set_agent_problem(agent, problem)
                for replication in range(cfg.replications):
                    eval_seed = cfg.seed + 800000 + 10000 * ti + replication
                    row = {
                        "family": "A1_unseen_topology",
                        "case_id": topology,
                        "test_topology": topology,
                        "train_topologies": ",".join(train_topologies),
                        "policy": algo,
                        "policy_label": RL_LABELS[algo],
                        "train_seed": train_seed,
                        "replication": replication,
                        "seed": eval_seed,
                    }
                    row.update(evaluate_agent(problem, agent, eval_seed, cfg.warmup_steps, cfg.eval_steps))
                    rep_rows.append(row)
            completed_agents += 1
            render_progress(completed_agents, total_agents, f"finished shared {algo} seed={train_seed}")

    for ti, topology in enumerate(test_topologies, start=1):
        problem = problems[topology]
        for replication in range(cfg.replications):
            eval_seed = cfg.seed + 800000 + 10000 * ti + replication
            row = {
                "family": "A1_unseen_topology",
                "case_id": topology,
                "test_topology": topology,
                "train_topologies": ",".join(train_topologies),
                "policy": "gpah",
                "policy_label": "gPAH",
                "train_seed": "",
                "replication": replication,
                "seed": eval_seed,
            }
            row.update(simulate_policy(problem, "gpah", eval_seed, cfg.warmup_steps, cfg.eval_steps))
            rep_rows.append(row)

    case_rows = summarize(rep_rows, ["family", "case_id", "test_topology", "policy", "policy_label"])
    policy_rows = summarize(rep_rows, ["policy", "policy_label"])
    write_csv(str(output_dir / "paper1_rl_unseen_topology_replications.csv"), rep_rows)
    write_csv(str(output_dir / "paper1_rl_unseen_topology_case_summary.csv"), case_rows)
    write_csv(str(output_dir / "paper1_rl_unseen_topology_policy_summary.csv"), policy_rows)
    write_csv(str(output_dir / "paper1_rl_unseen_topology_training_history.csv"), hist_rows)
    plot_results(case_rows, output_dir)
    print(f"Unseen-topology results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
