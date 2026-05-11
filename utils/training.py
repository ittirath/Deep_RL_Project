"""Utilities for evaluation, seeding, and logging."""
from __future__ import annotations

import random
from typing import Callable, List

import gymnasium as gym
import numpy as np
import torch


def set_seed(seed: int):
    """Seed numpy / torch / random for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_policy(env_name: str, act_fn: Callable, n_episodes: int = 5,
                    seed: int = 1000) -> float:
    """Run `act_fn` in `env_name` for `n_episodes`, return the mean return."""
    env = gym.make(env_name)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_ret = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            action = act_fn(obs)
            obs, r, done, truncated, _ = env.step(action)
            ep_ret += float(r)
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))


def smooth(y: List[float], k: int = 10) -> np.ndarray:
    """Simple moving-average smoothing for plotting noisy curves."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) < k:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="valid")
