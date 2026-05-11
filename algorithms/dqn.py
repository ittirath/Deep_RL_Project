"""DQN — Mnih et al. (2013) "Playing Atari with Deep Reinforcement Learning".

Implementation notes / lessons learned (see report):
- target network sync every `target_update_every` gradient steps is *crucial*.
  Without it, the bootstrap target moves at the same rate as the online network
  and learning becomes very unstable on MountainCar.
- ε-greedy: a *linear* decay from 1.0 down to a small ε_end (0.02–0.05) over a
  fraction of the total training works well. On MountainCar, the ε floor needs
  to stay relatively high because the reward signal is so sparse.
- Huber loss (smooth L1) is more stable than MSE when TD errors blow up
  occasionally (especially on Acrobot at the start).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from utils.networks import QNetwork
from utils.replay_buffer import ReplayBuffer
from utils.training import evaluate_policy, set_seed


@dataclass
class DQNConfig:
    total_steps: int = 30_000
    buffer_size: int = 50_000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 1e-3
    hidden: tuple = (64, 64)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 10_000
    warmup_steps: int = 1_000
    train_every: int = 1
    target_update_every: int = 500
    eval_every: int = 2_000
    eval_episodes: int = 10
    grad_clip: float = 10.0
    device: str = "cpu"


def train_dqn(env_name: str, seed: int, cfg: DQNConfig):
    """Train DQN on a discrete-action gym environment.

    Returns a dict with logged evaluation curves.
    """
    set_seed(seed)
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net = QNetwork(obs_dim, n_actions, cfg.hidden).to(cfg.device)
    target_net = QNetwork(obs_dim, n_actions, cfg.hidden).to(cfg.device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=cfg.lr)

    buffer = ReplayBuffer(obs_dim, 1, cfg.buffer_size, cfg.device,
                          discrete_actions=True)

    eval_steps: List[int] = []
    eval_returns: List[float] = []

    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    episode_returns: List[float] = []

    def epsilon(step: int) -> float:
        frac = min(step / cfg.eps_decay_steps, 1.0)
        return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)

    def act(obs_np, eps: float) -> int:
        if np.random.rand() < eps:
            return env.action_space.sample()
        with torch.no_grad():
            qs = q_net(torch.as_tensor(obs_np, dtype=torch.float32,
                                       device=cfg.device).unsqueeze(0))
        return int(qs.argmax(dim=-1).item())

    grad_steps = 0
    for step in range(1, cfg.total_steps + 1):
        eps = epsilon(step)
        action = act(obs, eps)
        next_obs, r, done, truncated, _ = env.step(action)
        # Distinguish "done because terminal state" from "done because time limit".
        # If truncated only, we should NOT mask the bootstrap because the value
        # continues — but for these tiny envs the effect is small, so we follow
        # the simple convention used in many DQN implementations.
        store_done = done  # use 'done' (terminal), not truncated
        buffer.add(obs, action, r, next_obs, store_done)
        episode_return += r
        obs = next_obs
        if done or truncated:
            episode_returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()

        # Train.
        if step > cfg.warmup_steps and step % cfg.train_every == 0:
            batch = buffer.sample(cfg.batch_size)
            obs_b = batch["obs"]
            act_b = batch["actions"]
            r_b = batch["rewards"]
            next_obs_b = batch["next_obs"]
            done_b = batch["dones"]

            q_vals = q_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next = target_net(next_obs_b).max(dim=1)[0]
                target = r_b + cfg.gamma * (1.0 - done_b) * q_next

            loss = F.smooth_l1_loss(q_vals, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), cfg.grad_clip)
            optimizer.step()
            grad_steps += 1

            if grad_steps % cfg.target_update_every == 0:
                target_net.load_state_dict(q_net.state_dict())

        # Evaluation.
        if step % cfg.eval_every == 0:
            def policy(o):
                with torch.no_grad():
                    qs = q_net(torch.as_tensor(o, dtype=torch.float32,
                                               device=cfg.device).unsqueeze(0))
                return int(qs.argmax(dim=-1).item())

            mean_ret = evaluate_policy(env_name, policy, cfg.eval_episodes,
                                       seed=seed + 10_000)
            eval_steps.append(step)
            eval_returns.append(mean_ret)

    env.close()
    return {
        "eval_steps": np.array(eval_steps),
        "eval_returns": np.array(eval_returns),
        "episode_returns": np.array(episode_returns),
        "algorithm": "DQN",
        "env": env_name,
        "seed": seed,
    }
