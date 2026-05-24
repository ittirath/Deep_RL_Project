"""Run the main experiments and save results.

Each (algorithm, env) pair is averaged over `N_SEEDS` random seeds.
Results are saved to results/<algo>_<env>_seed<seed>.pkl.

Experiment design principle
---------------------------
All algorithms evaluated on the same environment use the same total_steps so
comparisons reflect algorithmic differences, not training-budget differences.
Eval frequency is calibrated per env to produce ~15 data points per curve.
"""
from __future__ import annotations

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


def save_result(result: dict, label: str):
    # Use the experiment label (not result['algorithm']) so that variants like
    # SAC_fixed_alpha get their own files instead of overwriting SAC results.
    name = f"{label}_{result['env']}_seed{result['seed']}.pkl"
    path = RESULTS_DIR / name
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"  -> saved {path.name}")


def already_done(algo: str, env: str, seed: int) -> bool:
    return (RESULTS_DIR / f"{algo}_{env}_seed{seed}.pkl").exists()


# -----------------------------------------------------------------------
# Experiment table
# Each entry: (label, train_fn, config_factory, env_name)
#
# Environments and their step budgets:
#   CartPole-v1            30 000  — DQN vs PPO
#   Acrobot-v1             50 000  — DQN vs PPO
#   MountainCar-v0         40 000  — DQN only (discrete; documented hard-exploration case)
#   Pendulum-v1            30 000  — PPO vs SAC vs SAC_fixed_alpha vs TD3
#   MountainCarContinuous  30 000  — SAC vs SAC_fixed_alpha vs TD3
# -----------------------------------------------------------------------
EXPERIMENTS = [

    # ---- CartPole-v1 — 30 000 steps (DQN vs PPO) -------------------------
    # ("DQN", train_dqn,
    #  lambda: DQNConfig(total_steps=30_000, eps_decay_steps=15_000,
    #                    eval_every=2_000, target_update_every=500),
    #  "CartPole-v1"),

    # ("PPO", train_ppo,
    #  lambda: PPOConfig(total_steps=30_000, steps_per_rollout=1_024,
    #                    eval_every_rollouts=2),
    #  "CartPole-v1"),

    # ---- Acrobot-v1 — 50 000 steps (DQN vs PPO) --------------------------
    # ("DQN", train_dqn,
    #  lambda: DQNConfig(total_steps=50_000, eps_decay_steps=25_000,
    #                    eval_every=3_000, target_update_every=500),
    #  "Acrobot-v1"),

    # ("PPO", train_ppo,
    #  lambda: PPOConfig(total_steps=50_000, steps_per_rollout=2_048,
    #                    eval_every_rollouts=2, ent_coef=0.02),
    #  "Acrobot-v1"),

    # ---- MountainCar-v0 — 40 000 steps (DQN vs PPO) ------------------------
    # ("DQN", train_dqn,
    #  lambda: DQNConfig(total_steps=40_000, eps_decay_steps=20_000,
    #                    eps_end=0.1, eval_every=2_500,
    #                    target_update_every=1_000),
    #  "MountainCar-v0"),
    # ("PPO", train_ppo,
    # lambda: PPOConfig(total_steps=40_000, steps_per_rollout=2_048,
    #                 eval_every_rollouts=2, ent_coef=0.02),
    # "MountainCar-v0"),

    # # ---- Pendulum-v1 — 30 000 steps (PPO vs SAC vs SAC_fixed_alpha vs TD3)
    # ("PPO", train_ppo,
    #  lambda: PPOConfig(total_steps=30_000, steps_per_rollout=1_024,
    #                    eval_every_rollouts=2, gamma=0.9, lr_pi=1e-3, lr_v=1e-3),
    #  "Pendulum-v1"),

    ("SAC", train_sac,
     lambda: SACConfig(total_steps=30_000, eval_every=2_000),
     "Pendulum-v1"),

    # ("SAC_fixed_alpha", train_sac,
    #  lambda: SACConfig(total_steps=30_000, eval_every=2_000, fixed_alpha=0.2),
    #  "Pendulum-v1"),

    ("TD3", train_td3,
     lambda: TD3Config(total_steps=30_000, eval_every=2_000),
     "Pendulum-v1"),

    # ---- MountainCarContinuous-v0 — 30 000 steps (SAC vs SAC_fixed_alpha vs TD3)
    # ("PPO", train_ppo,
    # lambda: PPOConfig(total_steps=30_000, steps_per_rollout=2_048,
    #                 eval_every_rollouts=2, ent_coef=0.02),
    # "MountainCarContinuous-v0"),

    # ("SAC", train_sac,
    #  lambda: SACConfig(total_steps=30_000, eval_every=2_000),
    #  "MountainCarContinuous-v0"),

    # ("SAC_fixed_alpha", train_sac,
    #  lambda: SACConfig(total_steps=30_000, eval_every=2_000, fixed_alpha=0.2),
    #  "MountainCarContinuous-v0"),

    # ("TD3", train_td3,
    #  lambda: TD3Config(total_steps=30_000, eval_every=2_000),
    #  "MountainCarContinuous-v0"),

    # ---- SAC fixed-α sweep on Pendulum-v1 — NEW --------------------------
    # Same budget as the existing runs for a fair comparison.
    # To re-run all experiments above, uncomment those blocks.
    # ("SAC_fixed_alpha_005", train_sac,
    #  lambda: SACConfig(total_steps=30_000, eval_every=2_000, fixed_alpha=0.05),
    #  "Pendulum-v1"),

    # ("SAC_fixed_alpha_050", train_sac,
    #  lambda: SACConfig(total_steps=30_000, eval_every=2_000, fixed_alpha=0.5),
    #  "Pendulum-v1"),

    # ("SAC_fixed_alpha_100", train_sac,
    #  lambda: SACConfig(total_steps=30_000, eval_every=2_000, fixed_alpha=1.0),
    #  "Pendulum-v1"),
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
            print(f"[run]  {label} on {env_name} seed={seed} ...")
            result = train_fn(env_name, seed=seed, cfg=cfg)
            dt = time.time() - t0
            print(f"       done in {dt:.1f}s, "
                  f"final eval = {result['eval_returns'][-1]:.2f}")
            save_result(result, label)
    print(f"\nAll done in {time.time() - overall_start:.1f}s.")


if __name__ == "__main__":
    main()
