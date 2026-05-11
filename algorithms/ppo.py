"""PPO — Schulman et al. (2017) "Proximal Policy Optimization Algorithms".

Implementation notes / lessons learned:
- GAE (Generalized Advantage Estimation) with λ ≈ 0.95 is what makes PPO actually
  learn smoothly. Plain TD-residual advantages produce a very noisy signal.
- Normalising advantages per minibatch (mean 0, std 1) is one of those "small
  details" that makes a big difference in stability.
- Value loss clipping (clipping V also, like in the original paper appendix and
  Stable-Baselines) gives a small improvement but is not strictly necessary on
  these small environments.
- For continuous control, a state-independent learnable log_std works better
  here than a state-dependent one — the latter introduces too many params for
  what these environments need.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from utils.networks import CategoricalActor, GaussianActorPPO, ValueNetwork
from utils.training import evaluate_policy, set_seed


@dataclass
class PPOConfig:
    total_steps: int = 100_000
    steps_per_rollout: int = 2_048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr_pi: float = 3e-4
    lr_v: float = 1e-3
    epochs: int = 10
    minibatch_size: int = 64
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: tuple = (64, 64)
    eval_every_rollouts: int = 5
    eval_episodes: int = 5
    device: str = "cpu"
    target_kl: float = 0.02  # early-stop on KL divergence


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """Generalized advantage estimation."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_v = last_value
            next_nonterminal = 1.0 - dones[t]
        else:
            next_v = values[t + 1]
            next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * next_nonterminal - values[t]
        last_adv = delta + gamma * lam * next_nonterminal * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


def train_ppo(env_name: str, seed: int, cfg: PPOConfig):
    """Train PPO on either discrete- or continuous-action gym environment."""
    set_seed(seed)
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    discrete = hasattr(env.action_space, "n")

    if discrete:
        actor = CategoricalActor(obs_dim, env.action_space.n, cfg.hidden).to(cfg.device)
    else:
        act_dim = env.action_space.shape[0]
        act_limit = float(env.action_space.high[0])
        actor = GaussianActorPPO(obs_dim, act_dim, act_limit, cfg.hidden).to(cfg.device)

    critic = ValueNetwork(obs_dim, cfg.hidden).to(cfg.device)
    optim_pi = optim.Adam(actor.parameters(), lr=cfg.lr_pi)
    optim_v = optim.Adam(critic.parameters(), lr=cfg.lr_v)

    eval_steps: List[int] = []
    eval_returns: List[float] = []

    obs, _ = env.reset(seed=seed)
    total_steps_done = 0
    rollout_idx = 0
    episode_return = 0.0
    episode_returns: List[float] = []

    while total_steps_done < cfg.total_steps:
        # Collect a rollout.
        obs_buf = np.zeros((cfg.steps_per_rollout, obs_dim), dtype=np.float32)
        if discrete:
            act_buf = np.zeros(cfg.steps_per_rollout, dtype=np.int64)
        else:
            act_buf = np.zeros((cfg.steps_per_rollout, env.action_space.shape[0]),
                               dtype=np.float32)
        logp_buf = np.zeros(cfg.steps_per_rollout, dtype=np.float32)
        rew_buf = np.zeros(cfg.steps_per_rollout, dtype=np.float32)
        val_buf = np.zeros(cfg.steps_per_rollout, dtype=np.float32)
        done_buf = np.zeros(cfg.steps_per_rollout, dtype=np.float32)

        for t in range(cfg.steps_per_rollout):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32,
                                         device=cfg.device).unsqueeze(0)
            with torch.no_grad():
                if discrete:
                    a, logp, _ = actor(obs_tensor)
                    a_np = int(a.item())
                else:
                    a, logp, _ = actor(obs_tensor)
                    a_np = a.squeeze(0).cpu().numpy()
                v = critic(obs_tensor).item()

            obs_buf[t] = obs
            act_buf[t] = a_np
            logp_buf[t] = logp.item()
            val_buf[t] = v

            # Clip continuous actions to env range for the step.
            if discrete:
                env_action = a_np
            else:
                env_action = np.clip(a_np, env.action_space.low,
                                     env.action_space.high)

            next_obs, r, done, truncated, _ = env.step(env_action)
            rew_buf[t] = r
            episode_return += r
            obs = next_obs

            # For GAE we treat truncation differently from termination:
            # if truncated, we still bootstrap from V(s_next); if done, we don't.
            if done or truncated:
                done_buf[t] = 1.0 if done else 0.0  # only terminal masks the bootstrap
                # If truncated (not done), we bootstrap with V(next_obs).
                if truncated and not done:
                    with torch.no_grad():
                        v_next = critic(torch.as_tensor(next_obs, dtype=torch.float32,
                                                       device=cfg.device).unsqueeze(0)).item()
                    rew_buf[t] = r + cfg.gamma * v_next  # absorb the tail into r
                episode_returns.append(episode_return)
                episode_return = 0.0
                obs, _ = env.reset()

            total_steps_done += 1

        # Compute GAE.
        with torch.no_grad():
            last_v = critic(torch.as_tensor(obs, dtype=torch.float32,
                                            device=cfg.device).unsqueeze(0)).item()
        advantages, returns = compute_gae(rew_buf, val_buf, done_buf, last_v,
                                          cfg.gamma, cfg.gae_lambda)

        # Convert to tensors.
        obs_t = torch.as_tensor(obs_buf, device=cfg.device)
        act_t = torch.as_tensor(act_buf, device=cfg.device)
        old_logp_t = torch.as_tensor(logp_buf, device=cfg.device)
        adv_t = torch.as_tensor(advantages, device=cfg.device)
        ret_t = torch.as_tensor(returns, device=cfg.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # PPO update epochs over minibatches.
        n = cfg.steps_per_rollout
        idxs = np.arange(n)
        early_stop = False
        for epoch in range(cfg.epochs):
            if early_stop:
                break
            np.random.shuffle(idxs)
            for start in range(0, n, cfg.minibatch_size):
                mb_idx = idxs[start:start + cfg.minibatch_size]
                mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.long, device=cfg.device)
                obs_mb = obs_t[mb_idx_t]
                act_mb = act_t[mb_idx_t]
                old_logp_mb = old_logp_t[mb_idx_t]
                adv_mb = adv_t[mb_idx_t]
                ret_mb = ret_t[mb_idx_t]

                # Re-evaluate log-prob and entropy.
                if discrete:
                    dist = actor.distribution(obs_mb)
                    new_logp = dist.log_prob(act_mb)
                    entropy = dist.entropy()
                else:
                    dist = actor.distribution(obs_mb)
                    new_logp = dist.log_prob(act_mb).sum(-1)
                    entropy = dist.entropy().sum(-1)

                ratio = torch.exp(new_logp - old_logp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio,
                                    1.0 + cfg.clip_ratio) * adv_mb
                pi_loss = -torch.min(surr1, surr2).mean()
                ent_loss = -entropy.mean()
                total_pi_loss = pi_loss + cfg.ent_coef * ent_loss

                optim_pi.zero_grad()
                total_pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                optim_pi.step()

                # Value loss.
                v_pred = critic(obs_mb)
                v_loss = F.mse_loss(v_pred, ret_mb)
                optim_v.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                optim_v.step()

            # KL early-stop check.
            with torch.no_grad():
                if discrete:
                    dist = actor.distribution(obs_t)
                    new_logp_full = dist.log_prob(act_t)
                else:
                    dist = actor.distribution(obs_t)
                    new_logp_full = dist.log_prob(act_t).sum(-1)
                kl = (old_logp_t - new_logp_full).mean().item()
            if kl > 1.5 * cfg.target_kl:
                early_stop = True

        rollout_idx += 1
        if rollout_idx % cfg.eval_every_rollouts == 0:
            def policy(o):
                ot = torch.as_tensor(o, dtype=torch.float32,
                                     device=cfg.device).unsqueeze(0)
                with torch.no_grad():
                    if discrete:
                        dist = actor.distribution(ot)
                        a_eval = dist.probs.argmax(dim=-1)  # greedy
                        return int(a_eval.item())
                    else:
                        dist = actor.distribution(ot)
                        a_eval = dist.mean.squeeze(0).cpu().numpy()
                        return np.clip(a_eval, env.action_space.low,
                                       env.action_space.high)
            mean_ret = evaluate_policy(env_name, policy, cfg.eval_episodes,
                                       seed=seed + 10_000)
            eval_steps.append(total_steps_done)
            eval_returns.append(mean_ret)

    env.close()
    return {
        "eval_steps": np.array(eval_steps),
        "eval_returns": np.array(eval_returns),
        "episode_returns": np.array(episode_returns),
        "algorithm": "PPO",
        "env": env_name,
        "seed": seed,
    }
