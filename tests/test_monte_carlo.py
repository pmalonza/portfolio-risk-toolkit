import numpy as np
import pytest

from portfolio_risk.monte_carlo import (
    monte_carlo_cvar,
    monte_carlo_var,
    simulate_portfolio_returns,
)
from portfolio_risk.parametric import fit_normal, normal_cvar, normal_var


def test_simulate_portfolio_returns_shape():
    mu = np.array([0.0002, 0.0001])
    cov = np.array([[1e-4, 2e-5], [2e-5, 8e-5]])
    weights = np.array([0.6, 0.4])
    sims = simulate_portfolio_returns(mu, cov, weights, n_sims=1000, seed=0)
    assert sims.shape == (1000,)


def test_simulate_portfolio_returns_is_reproducible_with_seed():
    mu = np.array([0.0, 0.0])
    cov = np.eye(2) * 1e-4
    weights = np.array([0.5, 0.5])
    s1 = simulate_portfolio_returns(mu, cov, weights, n_sims=500, seed=42)
    s2 = simulate_portfolio_returns(mu, cov, weights, n_sims=500, seed=42)
    assert np.array_equal(s1, s2)


def test_simulate_portfolio_returns_rejects_mismatched_dimensions():
    mu = np.array([0.0, 0.0, 0.0])
    cov = np.eye(2)
    weights = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        simulate_portfolio_returns(mu, cov, weights, n_sims=10)


def test_monte_carlo_var_converges_to_parametric_normal_var():
    mu = np.array([0.0003, 0.0001, 0.0002])
    cov = np.array(
        [
            [1.0e-4, 1.0e-5, 0.5e-5],
            [1.0e-5, 8.0e-5, 1.0e-5],
            [0.5e-5, 1.0e-5, 6.0e-5],
        ]
    )
    weights = np.array([0.4, 0.35, 0.25])

    port_mu = float(weights @ mu)
    port_var_theoretical = float(weights @ cov @ weights)
    port_sigma = port_var_theoretical**0.5

    from portfolio_risk.parametric import NormalFit

    fit = NormalFit(mu=port_mu, sigma=port_sigma)
    expected_var = normal_var(fit, alpha=0.95)
    expected_cvar = normal_cvar(fit, alpha=0.95)

    mc_var = monte_carlo_var(mu, cov, weights, alpha=0.95, n_sims=300_000, seed=7)
    mc_cvar = monte_carlo_cvar(mu, cov, weights, alpha=0.95, n_sims=300_000, seed=7)

    assert mc_var == pytest.approx(expected_var, abs=5e-4)
    assert mc_cvar == pytest.approx(expected_cvar, abs=8e-4)


def test_monte_carlo_cvar_never_less_than_var():
    mu = np.array([0.0, 0.0])
    cov = np.array([[2e-4, 5e-5], [5e-5, 1.5e-4]])
    weights = np.array([0.5, 0.5])
    var = monte_carlo_var(mu, cov, weights, alpha=0.95, n_sims=50_000, seed=3)
    cvar = monte_carlo_cvar(mu, cov, weights, alpha=0.95, n_sims=50_000, seed=3)
    assert cvar >= var


def test_rejects_invalid_n_sims():
    mu = np.array([0.0])
    cov = np.array([[1e-4]])
    weights = np.array([1.0])
    with pytest.raises(ValueError):
        simulate_portfolio_returns(mu, cov, weights, n_sims=0)
