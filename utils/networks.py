"""Shared neural network architectures used by the four RL algorithms.

Keeps networks simple: 2 hidden-layer MLPs. This keeps comparisons fair across
algorithms (same capacity for the underlying function approximators).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


def mlp(sizes, activation=nn.ReLU, output_activation=None):
    """Build an MLP with the given layer sizes."""
    layers = []
    for j in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[j], sizes[j + 1]))
        if j < len(sizes) - 2:
            layers.append(activation())
        elif output_activation is not None:
            layers.append(output_activation())
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Q-network for DQN: outputs one value per discrete action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple = (128, 128)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, n_actions])

    def forward(self, obs):
        return self.net(obs)


class ContinuousQNetwork(nn.Module):
    """Critic Q(s, a) for SAC and TD3."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple = (256, 256)):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *hidden, 1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)


class DeterministicActor(nn.Module):
    """Deterministic policy for TD3: outputs an action in [-act_limit, act_limit]."""

    def __init__(self, obs_dim: int, act_dim: int, act_limit: float,
                 hidden: tuple = (256, 256)):
        super().__init__()
        self.act_limit = act_limit
        self.net = mlp([obs_dim, *hidden, act_dim], output_activation=nn.Tanh)

    def forward(self, obs):
        return self.act_limit * self.net(obs)


class GaussianActor(nn.Module):
    """Squashed Gaussian policy for SAC."""

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int, act_limit: float,
                 hidden: tuple = (256, 256)):
        super().__init__()
        self.act_limit = act_limit
        self.trunk = mlp([obs_dim, *hidden])
        # The "trunk" includes the last activation; we add the mean/log_std heads.
        self.mu_head = nn.Linear(hidden[-1], act_dim)
        self.log_std_head = nn.Linear(hidden[-1], act_dim)

    def forward(self, obs, deterministic: bool = False, with_logprob: bool = True):
        h = self.trunk(obs)
        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std_head(h), self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mu, std)
        if deterministic:
            u = mu
        else:
            u = dist.rsample()  # reparameterized sample
        if with_logprob:
            # Log-prob with tanh-squashing correction.
            log_prob = dist.log_prob(u).sum(-1)
            log_prob -= (2 * (torch.log(torch.tensor(2.0)) - u - F.softplus(-2 * u))).sum(-1)
        else:
            log_prob = None
        a = torch.tanh(u) * self.act_limit
        return a, log_prob


class CategoricalActor(nn.Module):
    """Categorical (discrete) policy head for PPO."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple = (64, 64)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, n_actions])

    def distribution(self, obs):
        return Categorical(logits=self.net(obs))

    def forward(self, obs, act=None):
        dist = self.distribution(obs)
        if act is None:
            act = dist.sample()
        log_prob = dist.log_prob(act)
        return act, log_prob, dist.entropy()


class GaussianActorPPO(nn.Module):
    """Diagonal Gaussian policy with state-independent log_std for PPO continuous control."""

    def __init__(self, obs_dim: int, act_dim: int, act_limit: float,
                 hidden: tuple = (64, 64), log_std_init: float = -0.5):
        super().__init__()
        self.act_limit = act_limit
        self.mu_net = mlp([obs_dim, *hidden, act_dim])
        # log_std is a learnable parameter independent of state (standard PPO choice).
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def distribution(self, obs):
        mu = self.mu_net(obs)
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)

    def forward(self, obs, act=None):
        dist = self.distribution(obs)
        if act is None:
            act = dist.sample()
        log_prob = dist.log_prob(act).sum(-1)
        entropy = dist.entropy().sum(-1)
        # Clip the *output* action to env bounds. We keep the raw action for logprob,
        # which is a standard (if slightly imperfect) PPO approximation.
        return act, log_prob, entropy


class ValueNetwork(nn.Module):
    """State value V(s) for PPO."""

    def __init__(self, obs_dim: int, hidden: tuple = (64, 64)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, 1])

    def forward(self, obs):
        return self.net(obs).squeeze(-1)
