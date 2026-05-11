# Deep RL Project — DQN · PPO · SAC · TD3

Re-implementations of four deep RL algorithms with empirical comparison on classic Gym control tasks. Built as a *high-level notebook + low-level Python modules* layout.

## Layout

```
deep_rl_project/
├── notebook.ipynb         # main entry point — graphs, comparisons, discussion
├── run_experiments.py     # CLI batch runner (multi-seed, multi-env)
├── algorithms/
│   ├── dqn.py             # Mnih et al. 2013
│   ├── ppo.py             # Schulman et al. 2017
│   ├── sac.py             # Haarnoja et al. 2018
│   └── td3.py             # Fujimoto et al. 2018
├── utils/
│   ├── networks.py        # shared MLP heads (Q, V, π — discrete & continuous)
│   ├── replay_buffer.py   # ring buffer for off-policy algorithms
│   ├── training.py        # set_seed, evaluate_policy, smoothing helper
│   └── plotting.py        # load pickles, aggregate across seeds, plot grids
└── results/               # pickled per-seed results (one file per (algo, env, seed))
```

Each algorithm exposes a single training function and a dataclass config:

```python
from algorithms import train_dqn, DQNConfig
result = train_dqn("CartPole-v1", seed=0, cfg=DQNConfig(total_steps=15_000))
# result is a dict with eval_steps, eval_returns, episode_returns, ...
```

## Setup

```bash
pip install torch gymnasium[classic-control] numpy matplotlib pandas tqdm
```

(No GPU needed for classic control. CPU is fine.)

## How to use

**Quick start.** Open `notebook.ipynb`. Every section has a smoke-test cell that trains a single seed in ~30 s to a few minutes; the comparison plots (§6) load any saved `results/*.pkl` and aggregate across seeds.

**Producing more results.**

```bash
python run_experiments.py
```

runs the matrix in `run_experiments.py` (one row per (algo, env)) for `SEEDS = [0, 1, 2]`. Already-saved results are skipped — so this is restart-safe. Adjust `EXPERIMENTS` and `SEEDS` at the top of the file.

**Adding hyperparameter sweeps.** Section 7 of the notebook has commented-out templates. Uncomment, adjust the range, and run. Each sweep saves nothing by default — modify if you want them persisted.

## What's working in the supplied baseline

The `results/` directory ships with smoke-run pickles to verify the plotting works end-to-end:

| Algorithm | Environment | Seeds | Notes |
|---|---|---|---|
| DQN | CartPole-v1 | 3 | reaches 500 peak but oscillates — vanilla DQN is unstable |
| DQN | Acrobot-v1 | 3 | reaches ~-130 peak then collapses to -500 (forgetting) |
| DQN | MountainCar-v0 | 3 | never reaches goal — sparse-reward exploration failure |
| PPO | CartPole-v1 | 3 | solves cleanly (500/500) |
| PPO | Acrobot-v1 | 3 | 2/3 seeds solve, 1 stuck (exploration variance) |
| PPO | Pendulum-v1 | 3 | learns with γ=0.9 (see notebook §10 for why γ=0.99 fails) |
| SAC | Pendulum-v1 | 3 | converges to ~-100 in 10k steps, best behaved |
| TD3 | Pendulum-v1 | 2 | converges to ~-100 in 10k steps |

These are baselines for the notebook to plot — not the final report-quality numbers. Re-run `run_experiments.py` with more seeds and longer budgets for tight curves.

## Known limitations / honest notes

- **MountainCarContinuous** is not in the baseline. SAC and TD3 both fell into the "don't act, save energy" local optimum in 15k steps. Solving it requires either heavier exploration (intrinsic rewards, NoisyNets) or much longer training.
- **DQN** is vanilla (2013 paper): no Double-DQN, no Dueling, no Prioritized Replay. The instability you see on CartPole and Acrobot is genuine and expected.
- **Each plot is averaged over the seeds present in `results/`** — typically 3, but two for TD3. For a final report, re-run with at least 5 seeds.
- **Wall-clock** numbers in §8 are CPU-only and approximate.

## License / credit

Reference papers:

- Mnih et al., *Playing Atari with Deep Reinforcement Learning*, arXiv:1312.5602 (2013)
- Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347 (2017)
- Haarnoja et al., *Soft Actor-Critic*, ICML 2018
- Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic Methods*, ICML 2018
