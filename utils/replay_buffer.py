"""Simple ring-buffer replay buffer used by DQN, SAC, and TD3."""
from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size FIFO replay buffer storing (s, a, r, s', done) tuples."""

    def __init__(self, obs_dim, act_dim, size: int, device: str = "cpu",
                 discrete_actions: bool = False):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        if discrete_actions:
            self.actions = np.zeros(size, dtype=np.int64)
        else:
            self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size
        self.device = device
        self.discrete_actions = discrete_actions

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=torch.as_tensor(self.obs[idx], device=self.device),
            next_obs=torch.as_tensor(self.next_obs[idx], device=self.device),
            rewards=torch.as_tensor(self.rewards[idx], device=self.device),
            dones=torch.as_tensor(self.dones[idx], device=self.device),
        )
        batch["actions"] = torch.as_tensor(self.actions[idx], device=self.device)
        return batch

    def __len__(self):
        return self.size
