"""Monte Carlo VaR and CVaR via simulated multi-asset scenarios.

Draws correlated asset-return scenarios from a multivariate Normal fitted
to (or supplied for) the portfolio's constituent assets, aggregates each
scenario into a portfolio return with the given weights, and then applies
the historical-simulation estimator to the simulated sample.
"""

from __future__ import annotations

import numpy as np

from .data import aggregate_portfolio_returns
from .historical import historical_cvar, historical_var


def simulate_portfolio_returns(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    n_sims: int = 100_000,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate ``n_sims`` portfolio return scenarios from a Normal asset model."""
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if n_sims < 1:
        raise ValueError("n_sims must be >= 1")
    if mu.shape[0] != cov.shape[0]:
        raise ValueError("mu and cov must describe the same number of assets")

    rng = np.random.default_rng(seed)
    asset_scenarios = rng.multivariate_normal(mean=mu, cov=cov, size=n_sims)
    return aggregate_portfolio_returns(asset_scenarios, weights)


def monte_carlo_var(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.95,
    n_sims: int = 100_000,
    seed: int | None = None,
) -> float:
    """Monte Carlo VaR: simulate scenarios, then take the empirical quantile."""
    sims = simulate_portfolio_returns(mu, cov, weights, n_sims=n_sims, seed=seed)
    return historical_var(sims, alpha)


def monte_carlo_cvar(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.95,
    n_sims: int = 100_000,
    seed: int | None = None,
) -> float:
    """Monte Carlo CVaR: simulate scenarios, then take the empirical tail mean."""
    sims = simulate_portfolio_returns(mu, cov, weights, n_sims=n_sims, seed=seed)
    return historical_cvar(sims, alpha)
