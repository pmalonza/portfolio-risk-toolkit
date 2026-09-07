"""Synthetic multi-asset return generation and portfolio aggregation."""

from __future__ import annotations

import numpy as np


def generate_synthetic_returns(
    n_assets: int = 5,
    n_days: int = 1500,
    annual_vol: float = 0.20,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate correlated daily asset returns via a multivariate normal.

    Builds a random (but valid, positive semi-definite) correlation matrix,
    scales it by per-asset volatilities derived from ``annual_vol``, and
    draws ``n_days`` samples.

    Returns (returns, mu, cov) where ``returns`` has shape (n_days, n_assets).
    """
    if n_assets < 1:
        raise ValueError("n_assets must be >= 1")
    if n_days < 1:
        raise ValueError("n_days must be >= 1")

    rng = np.random.default_rng(seed)

    a = rng.normal(size=(n_assets, n_assets))
    cov_base = a @ a.T
    d = np.sqrt(np.diag(cov_base))
    corr = cov_base / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)

    daily_vol = annual_vol / np.sqrt(252.0)
    per_asset_vol = rng.uniform(0.7, 1.3, size=n_assets) * daily_vol
    cov = np.outer(per_asset_vol, per_asset_vol) * corr

    mu = rng.uniform(-0.0002, 0.0005, size=n_assets)

    returns = rng.multivariate_normal(mean=mu, cov=cov, size=n_days)
    return returns, mu, cov


def equal_weights(n_assets: int) -> np.ndarray:
    """Return an equal-weight allocation vector of length ``n_assets``."""
    if n_assets < 1:
        raise ValueError("n_assets must be >= 1")
    return np.full(n_assets, 1.0 / n_assets)


def aggregate_portfolio_returns(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Combine per-asset return series into a single portfolio return series."""
    returns = np.asarray(returns)
    weights = np.asarray(weights, dtype=float)
    if returns.ndim != 2:
        raise ValueError("returns must be a 2D array of shape (n_days, n_assets)")
    if weights.shape[0] != returns.shape[1]:
        raise ValueError("weights length must match number of assets")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must sum to 1.0")
    return returns @ weights
