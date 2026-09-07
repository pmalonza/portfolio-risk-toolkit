import numpy as np
import pytest

from portfolio_risk.historical import historical_cvar, historical_var


def test_historical_var_matches_manual_quantile():
    returns = np.array([-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.04])
    losses = -returns
    expected = np.quantile(losses, 0.90)
    assert historical_var(returns, alpha=0.90) == pytest.approx(expected)


def test_historical_cvar_is_mean_of_tail_beyond_var():
    returns = np.array([-0.10, -0.05, -0.02, 0.0, 0.01, 0.03, 0.06, 0.08])
    alpha = 0.75
    var = historical_var(returns, alpha)
    cvar = historical_cvar(returns, alpha)
    losses = -returns
    expected_tail_mean = losses[losses >= var].mean()
    assert cvar == pytest.approx(expected_tail_mean)


def test_cvar_is_never_less_than_var():
    rng = np.random.default_rng(3)
    returns = rng.normal(loc=0.0005, scale=0.01, size=2000)
    for alpha in (0.90, 0.95, 0.99):
        var = historical_var(returns, alpha)
        cvar = historical_cvar(returns, alpha)
        assert cvar >= var


def test_higher_confidence_gives_larger_var():
    rng = np.random.default_rng(11)
    returns = rng.normal(loc=0.0, scale=0.02, size=5000)
    var_95 = historical_var(returns, 0.95)
    var_99 = historical_var(returns, 0.99)
    assert var_99 > var_95


def test_rejects_invalid_alpha():
    returns = np.array([0.01, -0.01, 0.02])
    with pytest.raises(ValueError):
        historical_var(returns, alpha=0.0)
    with pytest.raises(ValueError):
        historical_var(returns, alpha=1.0)
    with pytest.raises(ValueError):
        historical_cvar(returns, alpha=1.5)
