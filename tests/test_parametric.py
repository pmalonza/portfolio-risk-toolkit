import numpy as np
import pytest
from scipy import stats

from portfolio_risk.parametric import (
    NormalFit,
    StudentTFit,
    fit_normal,
    fit_student_t,
    normal_cvar,
    normal_var,
    parametric_cvar,
    parametric_var,
    student_t_cvar,
    student_t_var,
)


def test_fit_normal_recovers_known_parameters():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.001, scale=0.02, size=100000)
    fit = fit_normal(x)
    assert fit.mu == pytest.approx(0.001, abs=2e-4)
    assert fit.sigma == pytest.approx(0.02, abs=2e-4)


def test_normal_var_matches_empirical_quantile_on_large_sample():
    rng = np.random.default_rng(1)
    x = rng.normal(loc=0.0002, scale=0.01, size=200000)
    fit = fit_normal(x)
    var = normal_var(fit, alpha=0.95)
    empirical = np.quantile(-x, 0.95)
    assert var == pytest.approx(empirical, abs=5e-4)


def test_normal_cvar_matches_empirical_tail_mean_on_large_sample():
    rng = np.random.default_rng(2)
    x = rng.normal(loc=0.0, scale=0.015, size=300000)
    fit = fit_normal(x)
    cvar = normal_cvar(fit, alpha=0.95)
    losses = -x
    empirical = losses[losses >= np.quantile(losses, 0.95)].mean()
    assert cvar == pytest.approx(empirical, abs=1e-3)


def test_normal_cvar_never_less_than_var():
    fit = NormalFit(mu=0.0003, sigma=0.012)
    for alpha in (0.90, 0.95, 0.99):
        assert normal_cvar(fit, alpha) >= normal_var(fit, alpha)


def test_student_t_var_matches_scipy_ppf_directly():
    fit = StudentTFit(dof=6.0, loc=0.0004, scale=0.011)
    z = stats.t.ppf(0.95, fit.dof)
    expected = -fit.loc + fit.scale * z
    assert student_t_var(fit, 0.95) == pytest.approx(expected)


def test_student_t_cvar_matches_empirical_tail_mean_on_large_sample():
    rng = np.random.default_rng(4)
    raw = rng.standard_t(df=5, size=300000) * 0.01 + 0.0002
    fit = fit_student_t(raw)
    cvar = student_t_cvar(fit, alpha=0.95)
    losses = -raw
    empirical = losses[losses >= np.quantile(losses, 0.95)].mean()
    assert cvar == pytest.approx(empirical, rel=0.05)


def test_student_t_cvar_rejects_dof_at_or_below_one():
    fit = StudentTFit(dof=1.0, loc=0.0, scale=0.01)
    with pytest.raises(ValueError):
        student_t_cvar(fit, 0.95)


def test_heavier_tails_increase_var_at_matched_variance():
    rng = np.random.default_rng(5)
    normal_returns = rng.normal(loc=0.0, scale=0.01, size=200000)
    t_returns = rng.standard_t(df=4, size=200000) * 0.01 * np.sqrt(2 / 4)

    normal_var_est = parametric_var(normal_returns, alpha=0.99, dist="normal")
    t_var_est = parametric_var(t_returns, alpha=0.99, dist="t")
    assert t_var_est > normal_var_est


def test_parametric_wrappers_reject_unknown_distribution():
    x = np.array([0.01, -0.01, 0.02, -0.02])
    with pytest.raises(ValueError):
        parametric_var(x, dist="lognormal")
    with pytest.raises(ValueError):
        parametric_cvar(x, dist="lognormal")
