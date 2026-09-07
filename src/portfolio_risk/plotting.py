"""Matplotlib plotting utilities: VaR method comparison and backtest exceedances."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe default; import pyplot only after this

import matplotlib.pyplot as plt
import numpy as np


def plot_var_comparison(returns: np.ndarray, var_results: dict[str, float], alpha: float = 0.95, ax=None):
    """Plot the loss histogram with a vertical line per method's VaR estimate.

    ``var_results`` maps a method name (e.g. "Historical") to its VaR
    estimate, expressed as a positive loss.
    """
    losses = -np.asarray(returns, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.hist(losses, bins=60, density=True, alpha=0.6, color="steelblue", label="Loss distribution")
    colors = plt.get_cmap("tab10").colors
    for i, (name, var_value) in enumerate(var_results.items()):
        ax.axvline(
            var_value,
            color=colors[i % len(colors)],
            linestyle="--",
            linewidth=1.5,
            label=f"{name} VaR ({alpha:.0%})",
        )
    ax.set_xlabel("Loss")
    ax.set_ylabel("Density")
    ax.set_title(f"VaR method comparison at {alpha:.0%} confidence")
    ax.legend()
    return ax


def plot_exceedances(returns: np.ndarray, var_estimates: np.ndarray, ax=None):
    """Plot realized losses against a VaR estimate, marking exceedance points."""
    losses = -np.asarray(returns, dtype=float)
    var_array = np.broadcast_to(np.asarray(var_estimates, dtype=float), losses.shape)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(losses, color="steelblue", linewidth=0.8, label="Realized loss")
    ax.plot(var_array, color="crimson", linewidth=1.2, label="VaR estimate")
    exceed_idx = np.where(losses > var_array)[0]
    ax.scatter(exceed_idx, losses[exceed_idx], color="crimson", zorder=3, s=15, label="Exceedance")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Loss")
    ax.set_title(f"VaR exceedances ({exceed_idx.size} of {losses.size})")
    ax.legend()
    return ax
