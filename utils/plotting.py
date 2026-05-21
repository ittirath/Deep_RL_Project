"""Plotting helpers for the project notebook.

Reads pickled results from `results/` (saved by algorithm training functions)
and produces learning-curve plots with mean ± std across seeds.

Result pickle schema (per file):
    {
        "algorithm":      str   (e.g. "DQN", "PPO", "SAC", "TD3")
        "env":            str   (e.g. "CartPole-v1")
        "seed":           int
        "eval_steps":     np.ndarray of env-step counts at which eval was run
        "eval_returns":   np.ndarray of mean return across eval episodes
        "episode_returns": np.ndarray of returns of each training episode
    }
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).parent.parent / "results"
ALGO_COLORS = {
    "DQN":             "#d62728",
    "PPO":             "#1f77b4",
    "SAC":             "#2ca02c",
    "SAC_fixed_alpha": "#17becf",
    "TD3":             "#9467bd",
}


def load_runs(algo: str, env: str,
              results_dir: Optional[Path] = None) -> List[Dict]:
    """Load all pickled runs for a (algorithm, env) pair."""
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    runs = []
    for p in sorted(results_dir.glob(f"{algo}_{env}_seed*.pkl")):
        with open(p, "rb") as f:
            runs.append(pickle.load(f))
    return runs


def aggregate(runs: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack eval curves across seeds. Returns (steps, mean, std).

    Assumes all runs share the same eval_steps grid; if not, this aligns to
    the shortest grid (truncates).
    """
    if not runs:
        return np.array([]), np.array([]), np.array([])
    n = min(len(r["eval_steps"]) for r in runs)
    steps = runs[0]["eval_steps"][:n]
    returns = np.stack([r["eval_returns"][:n] for r in runs])
    return steps, returns.mean(axis=0), returns.std(axis=0)


def plot_learning_curves(algos: List[str], env: str,
                          ax: Optional[plt.Axes] = None,
                          title: Optional[str] = None,
                          results_dir: Optional[Path] = None) -> plt.Axes:
    """Plot mean ± std learning curves for several algorithms on one env."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for algo in algos:
        runs = load_runs(algo, env, results_dir)
        if not runs:
            continue
        steps, mean, std = aggregate(runs)
        color = ALGO_COLORS.get(algo, None)
        ax.plot(steps, mean, label=f"{algo} (n={len(runs)})", color=color, linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Evaluation return")
    ax.set_title(title or env)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def plot_grid(layout: List[Tuple[str, List[str]]],
              ncols: int = 2,
              figsize: Optional[Tuple[float, float]] = None,
              results_dir: Optional[Path] = None) -> plt.Figure:
    """Plot a grid of (env, [algos]) panels.

    Example:
        layout = [
            ("CartPole-v1", ["DQN", "PPO"]),
            ("Pendulum-v1", ["PPO", "SAC", "TD3"]),
        ]
    """
    n = len(layout)
    nrows = (n + ncols - 1) // ncols
    figsize = figsize or (6.5 * ncols, 3.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    for i, (env, algos) in enumerate(layout):
        plot_learning_curves(algos, env, ax=axes[i], results_dir=results_dir)
    for j in range(len(layout), len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    return fig


def summary_table(algos_envs: List[Tuple[str, str]],
                  results_dir: Optional[Path] = None) -> "pandas.DataFrame":
    """Build a small dataframe summarising final performance per (algo, env)."""
    import pandas as pd
    rows = []
    for algo, env in algos_envs:
        runs = load_runs(algo, env, results_dir)
        if not runs:
            rows.append({"algo": algo, "env": env, "n_seeds": 0,
                         "final_mean": np.nan, "final_std": np.nan,
                         "peak_mean": np.nan})
            continue
        finals = np.array([r["eval_returns"][-1] for r in runs])
        peaks = np.array([r["eval_returns"].max() for r in runs])
        rows.append({
            "algo": algo,
            "env": env,
            "n_seeds": len(runs),
            "final_mean": finals.mean(),
            "final_std": finals.std(),
            "peak_mean": peaks.mean(),
            "peak_std": peaks.std(),
        })
    return pd.DataFrame(rows)


def smooth(y: np.ndarray, k: int = 10) -> np.ndarray:
    """Moving-average smoothing for noisy curves (e.g. episode returns)."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) < k:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="valid")
