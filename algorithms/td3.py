"""TD3 — Fujimoto et al. (2018) "Addressing Function Approximation Error in
Actor-Critic Methods".

Implementation notes / lessons learned:
- Three tricks distinguishing TD3 from DDPG:
    1. Clipped double-Q (twin critics, min of the two for the target).
    2. Delayed policy updates (update actor and target nets every `policy_delay`
       critic updates).
    3. Target policy smoothing (add clipped Gaussian noise to the target action).
- Without trick (1), Q values explode on MountainCarContinuous.
- Without trick (3), the policy becomes brittle and exploits sharp peaks of the
  critic.
- Exploration: simple Gaussian action noise (σ ≈ 0.1) works fine for the
  small-scale envs here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from utils.networks import ContinuousQNetwork, DeterministicActor
from utils.replay_buffer import ReplayBuffer
from utils.training import evaluate_policy, set_seed


@dataclass
class TD3Config:
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
    policy_delay: int = 2
    action_noise: float = 0.1
    target_noise: float = 0.2
    noise_clip: float = 0.5
    eval_every: int = 2_000
    eval_episodes: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_td3(env_name: str, seed: int, cfg: TD3Config):
    set_seed(seed)
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    actor = DeterministicActor(obs_dim, act_dim, act_limit, cfg.hidden).to(cfg.device)
    actor_target = DeterministicActor(obs_dim, act_dim, act_limit, cfg.hidden).to(cfg.device)
    actor_target.load_state_dict(actor.state_dict())
    for p in actor_target.parameters():
        p.requires_grad_(False)

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

    buffer = ReplayBuffer(obs_dim, act_dim, cfg.buffer_size, cfg.device)

    eval_steps: List[int] = []
    eval_returns: List[float] = []
    episode_returns: List[float] = []

    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    grad_steps = 0

    for step in range(1, cfg.total_steps + 1):
        if step <= cfg.warmup_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                a = actor(torch.as_tensor(obs, dtype=torch.float32,
                                          device=cfg.device).unsqueeze(0)).squeeze(0).cpu().numpy()
            a = a + np.random.normal(0, cfg.action_noise * act_limit, size=act_dim)
            action = np.clip(a, -act_limit, act_limit)

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

                with torch.no_grad():
                    # Target policy smoothing.
                    noise = (torch.randn_like(act_b) * cfg.target_noise * act_limit
                             ).clamp(-cfg.noise_clip * act_limit, cfg.noise_clip * act_limit)
                    next_a = (actor_target(next_obs_b) + noise).clamp(-act_limit, act_limit)
                    target_q = torch.min(q1_target(next_obs_b, next_a),
                                         q2_target(next_obs_b, next_a))
                    y = r_b + cfg.gamma * (1.0 - done_b) * target_q

                q1_pred = q1(obs_b, act_b)
                q2_pred = q2(obs_b, act_b)
                q_loss = F.mse_loss(q1_pred, y) + F.mse_loss(q2_pred, y)
                optim_q.zero_grad()
                q_loss.backward()
                optim_q.step()
                grad_steps += 1

                # Delayed policy + target updates.
                if grad_steps % cfg.policy_delay == 0:
                    pi_loss = -q1(obs_b, actor(obs_b)).mean()
                    optim_pi.zero_grad()
                    pi_loss.backward()
                    optim_pi.step()

                    with torch.no_grad():
                        for p, tp in zip(actor.parameters(), actor_target.parameters()):
                            tp.data.mul_(1.0 - cfg.tau)
                            tp.data.add_(cfg.tau * p.data)
                        for p, tp in zip(q1.parameters(), q1_target.parameters()):
                            tp.data.mul_(1.0 - cfg.tau)
                            tp.data.add_(cfg.tau * p.data)
                        for p, tp in zip(q2.parameters(), q2_target.parameters()):
                            tp.data.mul_(1.0 - cfg.tau)
                            tp.data.add_(cfg.tau * p.data)

        if step % cfg.eval_every == 0:
            def policy(o):
                with torch.no_grad():
                    a = actor(torch.as_tensor(o, dtype=torch.float32,
                                              device=cfg.device).unsqueeze(0)).squeeze(0).cpu().numpy()
                return np.clip(a, -act_limit, act_limit)
            mean_ret = evaluate_policy(env_name, policy, cfg.eval_episodes,
                                       seed=seed + 10_000)
            eval_steps.append(step)
            eval_returns.append(mean_ret)

    env.close()
    return {
        "eval_steps": np.array(eval_steps),
        "eval_returns": np.array(eval_returns),
        "episode_returns": np.array(episode_returns),
        "algorithm": "TD3",
        "env": env_name,
        "seed": seed,
    }
