"""Run the main experiments and save results.

Each (algorithm, env) pair is averaged over `N_SEEDS` random seeds.
Results are saved to results/<algo>_<env>_seed<seed>.pkl.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from algorithms import (
    train_dqn, DQNConfig,
    train_ppo, PPOConfig,
    train_sac, SACConfig,
    train_td3, TD3Config,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
N_SEEDS = 3
SEEDS = [0, 1, 2]


def save_result(result: dict):
    name = f"{result['algorithm']}_{result['env']}_seed{result['seed']}.pkl"
    path = RESULTS_DIR / name
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"  -> saved {path.name}")


def already_done(algo: str, env: str, seed: int) -> bool:
    return (RESULTS_DIR / f"{algo}_{env}_seed{seed}.pkl").exists()


# -------------------- Experiment table --------------------
EXPERIMENTS = [
    # (label, algorithm fn, config, env_name, steps)
    ("DQN", train_dqn,
     lambda: DQNConfig(total_steps=15_000, eps_decay_steps=8_000,
                       eval_every=1_000, target_update_every=500),
     "CartPole-v1"),
    ("DQN", train_dqn,
     lambda: DQNConfig(total_steps=30_000, eps_decay_steps=15_000,
                       eval_every=2_000, target_update_every=500),
     "Acrobot-v1"),
    ("DQN", train_dqn,
     lambda: DQNConfig(total_steps=40_000, eps_decay_steps=20_000,
                       eps_end=0.1, eval_every=2_500,
                       target_update_every=1000),
     "MountainCar-v0"),

    ("PPO", train_ppo,
     lambda: PPOConfig(total_steps=30_000, steps_per_rollout=1024,
                       eval_every_rollouts=2),
     "CartPole-v1"),
    ("PPO", train_ppo,
     lambda: PPOConfig(total_steps=50_000, steps_per_rollout=2048,
                       eval_every_rollouts=2),
     "Acrobot-v1"),
    ("PPO", train_ppo,
     lambda: PPOConfig(total_steps=60_000, steps_per_rollout=2048,
                       eval_every_rollouts=2),
     "Pendulum-v1"),

    ("SAC", train_sac,
     lambda: SACConfig(total_steps=12_000, eval_every=1_000),
     "Pendulum-v1"),
    ("SAC", train_sac,
     lambda: SACConfig(total_steps=15_000, eval_every=1_500),
     "MountainCarContinuous-v0"),

    ("TD3", train_td3,
     lambda: TD3Config(total_steps=12_000, eval_every=1_000),
     "Pendulum-v1"),
    ("TD3", train_td3,
     lambda: TD3Config(total_steps=15_000, eval_every=1_500),
     "MountainCarContinuous-v0"),
]


def main():
    overall_start = time.time()
    for label, train_fn, make_cfg, env_name in EXPERIMENTS:
        for seed in SEEDS:
            if already_done(label, env_name, seed):
                print(f"[skip] {label} on {env_name} seed={seed} (cached)")
                continue
            cfg = make_cfg()
            t0 = time.time()
            print(f"[run] {label} on {env_name} seed={seed} ...")
            result = train_fn(env_name, seed=seed, cfg=cfg)
            dt = time.time() - t0
            print(f"       done in {dt:.1f}s, final eval = {result['eval_returns'][-1]:.2f}")
            save_result(result)
    print(f"\nAll done in {time.time() - overall_start:.1f}s.")


if __name__ == "__main__":
    main()
