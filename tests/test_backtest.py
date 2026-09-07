import numpy as np
import pytest

from portfolio_risk.backtest import count_exceedances, kupiec_pof_test


def test_count_exceedances_with_scalar_var():
    returns = np.array([0.01, -0.05, -0.02, 0.03, -0.10])
    # losses: -0.01, 0.05, 0.02, -0.03, 0.10 -> exceed 0.04 at indices 1 and 4
    assert count_exceedances(returns, var_estimates=0.04) == 2


def test_count_exceedances_with_array_var():
    returns = np.array([-0.05, -0.01, -0.08])
    var_estimates = np.array([0.04, 0.02, 0.05])
    # losses: 0.05, 0.01, 0.08 -> exceed at index 0 and 2
    assert count_exceedances(returns, var_estimates) == 2


def test_well_calibrated_model_is_not_rejected():
    rng = np.random.default_rng(42)
    returns = rng.normal(loc=0.0, scale=0.01, size=2000)
    from portfolio_risk.parametric import fit_normal, normal_var

    fit = fit_normal(returns)
    var95 = normal_var(fit, alpha=0.95)

    out_of_sample = rng.normal(loc=0.0, scale=0.01, size=2000)
    result = kupiec_pof_test(out_of_sample, var95, alpha=0.95)

    assert result.n_obs == 2000
    assert result.expected_exceedances == pytest.approx(100.0)
    assert not result.reject_null
    assert result.p_value > 0.05


def test_badly_miscalibrated_model_is_rejected():
    rng = np.random.default_rng(7)
    returns = rng.normal(loc=0.0, scale=0.01, size=1500)
    understated_var = 0.001  # far too small for this distribution
    result = kupiec_pof_test(returns, understated_var, alpha=0.95)
    assert result.reject_null
    assert result.lr_statistic > result.critical_value


def test_lr_statistic_is_zero_when_exceedance_rate_matches_nominal_exactly():
    n = 1000
    p = 0.05
    returns = np.zeros(n)
    var_estimates = np.zeros(n)
    var_estimates[: int(n * p)] = -1.0  # first 50 obs "exceed" (loss 0 > var -1)
    result = kupiec_pof_test(returns, var_estimates, alpha=1 - p)
    assert result.n_exceedances == int(n * p)
    assert result.lr_statistic == pytest.approx(0.0, abs=1e-8)
    assert not result.reject_null


def test_handles_zero_exceedances_without_error():
    returns = np.array([0.01, 0.02, 0.03, 0.01])
    result = kupiec_pof_test(returns, var_estimates=10.0, alpha=0.95)
    assert result.n_exceedances == 0
    assert np.isfinite(result.lr_statistic)


def test_rejects_invalid_alpha_and_test_confidence():
    returns = np.array([0.01, -0.01])
    with pytest.raises(ValueError):
        kupiec_pof_test(returns, 0.02, alpha=1.0)
    with pytest.raises(ValueError):
        kupiec_pof_test(returns, 0.02, alpha=0.95, test_confidence=0.0)


def test_rejects_empty_returns():
    with pytest.raises(ValueError):
        kupiec_pof_test(np.array([]), var_estimates=0.01, alpha=0.95)
