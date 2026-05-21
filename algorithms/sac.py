"""SAC — Haarnoja et al. (2018) "Soft Actor-Critic".

Supports both:
- Auto-tuned alpha (Haarnoja 2018b, default): set fixed_alpha=None.
- Fixed alpha (Haarnoja 2018a): pass fixed_alpha=0.2 (or any positive float).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from utils.networks import ContinuousQNetwork, GaussianActor
from utils.replay_buffer import ReplayBuffer
from utils.training import evaluate_policy, set_seed


@dataclass
class SACConfig:
    total_steps: int = 50_000
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    hidden: tuple = (256, 256)
    warmup_steps: int = 1_000
    train_every: int = 1
    updates_per_step: int = 1
    target_entropy: float = None      # default: -act_dim (only used when fixed_alpha is None)
    fixed_alpha: Optional[float] = None  # 2018a behaviour when set; 2018b auto-tuning when None
    eval_every: int = 2_000
    eval_episodes: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_sac(env_name: str, seed: int, cfg: SACConfig):
    set_seed(seed)
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    actor = GaussianActor(obs_dim, act_dim, act_limit, cfg.hidden).to(cfg.device)
    q1 = ContinuousQNetwork(obs_dim, act_dim, cfg.hidden).to(cfg.device)
    q2 = ContinuousQNetwork(obs_dim, act_dim, cfg.hidden).to(cfg.device)
    q1_target = ContinuousQNetwork(obs_dim, act_dim, cfg.hidden).to(cfg.device)
    q2_target = ContinuousQNetwork(obs_dim, act_dim, cfg.hidden).to(cfg.device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    for p in q1_target.parameters():
        p.requires_grad_(False)
    for p in q2_target.parameters():
        p.requires_grad_(False)

    optim_pi = optim.Adam(actor.parameters(), lr=cfg.lr)
    optim_q = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=cfg.lr)

    # ----- Temperature setup -----
    auto_alpha = cfg.fixed_alpha is None
    if auto_alpha:
        # 2018b: learn log_alpha to satisfy entropy constraint.
        target_entropy = -float(act_dim) if cfg.target_entropy is None else cfg.target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=cfg.device)
        optim_alpha = optim.Adam([log_alpha], lr=cfg.lr)
    else:
        # 2018a: fixed alpha. Store log_alpha as a non-trainable tensor so the
        # same .exp() interface works downstream without branching.
        log_alpha = torch.log(torch.tensor(float(cfg.fixed_alpha), device=cfg.device))
        log_alpha.requires_grad_(False)
        optim_alpha = None
        target_entropy = None  # unused

    buffer = ReplayBuffer(obs_dim, act_dim, cfg.buffer_size, cfg.device)

    eval_steps: List[int] = []
    eval_returns: List[float] = []
    episode_returns: List[float] = []

    obs, _ = env.reset(seed=seed)
    episode_return = 0.0

    for step in range(1, cfg.total_steps + 1):
        if step <= cfg.warmup_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                a, _ = actor(torch.as_tensor(obs, dtype=torch.float32,
                                             device=cfg.device).unsqueeze(0),
                             deterministic=False, with_logprob=False)
            action = a.squeeze(0).cpu().numpy()

        next_obs, r, done, truncated, _ = env.step(action)
        buffer.add(obs, action, r, next_obs, float(done))
        episode_return += r
        obs = next_obs
        if done or truncated:
            episode_returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()

        if step > cfg.warmup_steps and step % cfg.train_every == 0:
            for _ in range(cfg.updates_per_step):
                batch = buffer.sample(cfg.batch_size)
                obs_b = batch["obs"]
                act_b = batch["actions"]
                r_b = batch["rewards"]
                next_obs_b = batch["next_obs"]
                done_b = batch["dones"]

                # ----- Q update -----
                with torch.no_grad():
                    next_a, next_logp = actor(next_obs_b)
                    target_q1 = q1_target(next_obs_b, next_a)
                    target_q2 = q2_target(next_obs_b, next_a)
                    target_q = torch.min(target_q1, target_q2) - log_alpha.exp() * next_logp
                    y = r_b + cfg.gamma * (1.0 - done_b) * target_q

                q1_pred = q1(obs_b, act_b)
                q2_pred = q2(obs_b, act_b)
                q_loss = F.mse_loss(q1_pred, y) + F.mse_loss(q2_pred, y)
                optim_q.zero_grad()
                q_loss.backward()
                optim_q.step()

                # ----- Policy update -----
                a_new, logp_new = actor(obs_b)
                q1_pi = q1(obs_b, a_new)
                q2_pi = q2(obs_b, a_new)
                q_pi = torch.min(q1_pi, q2_pi)
                pi_loss = (log_alpha.exp().detach() * logp_new - q_pi).mean()
                optim_pi.zero_grad()
                pi_loss.backward()
                optim_pi.step()

                # ----- alpha update (skipped in 2018a / fixed-alpha mode) -----
                if auto_alpha:
                    alpha_loss = -(log_alpha * (logp_new.detach() + target_entropy)).mean()
                    optim_alpha.zero_grad()
                    alpha_loss.backward()
                    optim_alpha.step()

                # ----- Target soft update -----
                with torch.no_grad():
                    for p, tp in zip(q1.parameters(), q1_target.parameters()):
                        tp.data.mul_(1.0 - cfg.tau)
                        tp.data.add_(cfg.tau * p.data)
                    for p, tp in zip(q2.parameters(), q2_target.parameters()):
                        tp.data.mul_(1.0 - cfg.tau)
                        tp.data.add_(cfg.tau * p.data)

        if step % cfg.eval_every == 0:
            def policy(o):
                with torch.no_grad():
                    a, _ = actor(torch.as_tensor(o, dtype=torch.float32,
                                                 device=cfg.device).unsqueeze(0),
                                 deterministic=True, with_logprob=False)
                return a.squeeze(0).cpu().numpy()
            mean_ret = evaluate_policy(env_name, policy, cfg.eval_episodes,
                                       seed=seed + 10_000)
            eval_steps.append(step)
            eval_returns.append(mean_ret)

    env.close()
    return {
        "eval_steps": np.array(eval_steps),
        "eval_returns": np.array(eval_returns),
        "episode_returns": np.array(episode_returns),
        "algorithm": "SAC",
        "env": env_name,
        "seed": seed,
    }