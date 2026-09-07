import sympy as sp

from portfolio_risk.symbolic import (
    a,
    alpha,
    derive_es_multiplier,
    derive_normal_cvar_formula,
    derive_tail_expectation,
    evaluate_normal_cvar_formula,
    mu,
    sigma,
    standard_normal_pdf,
    verify_against_scipy,
    z,
)


def test_tail_expectation_equals_density_at_cutoff():
    """integral_a^inf z*phi(z) dz should collapse exactly to phi(a)."""
    tail_expectation = derive_tail_expectation()
    phi_at_a = standard_normal_pdf().subs(z, a)
    assert sp.simplify(tail_expectation - phi_at_a) == 0


def test_es_multiplier_is_tail_expectation_over_tail_probability():
    es_multiplier = derive_es_multiplier()
    tail_expectation = derive_tail_expectation()
    expected = tail_expectation / (1 - alpha)
    assert sp.simplify(es_multiplier - expected) == 0


def test_cvar_formula_contains_expected_symbols():
    formula = derive_normal_cvar_formula()
    free_symbols = formula.free_symbols
    assert {mu, sigma, a, alpha}.issubset(free_symbols)


def test_symbolic_cvar_matches_scipy_exactly_at_multiple_confidence_levels():
    for alpha_value in (0.90, 0.95, 0.975, 0.99, 0.999):
        symbolic_value, scipy_value = verify_against_scipy(0.0004, 0.015, alpha_value)
        assert abs(symbolic_value - scipy_value) < 1e-9


def test_symbolic_cvar_matches_scipy_across_mu_sigma_combinations():
    for mu_value in (-0.001, 0.0, 0.0005):
        for sigma_value in (0.005, 0.02, 0.05):
            symbolic_value, scipy_value = verify_against_scipy(mu_value, sigma_value, 0.95)
            assert abs(symbolic_value - scipy_value) < 1e-9


def test_evaluate_normal_cvar_formula_rejects_invalid_alpha():
    import pytest

    with pytest.raises(ValueError):
        evaluate_normal_cvar_formula(0.0, 0.01, 0.0)
    with pytest.raises(ValueError):
        evaluate_normal_cvar_formula(0.0, 0.01, 1.0)


def test_cvar_exceeds_var_component_since_es_multiplier_exceeds_z_alpha():
    """Sanity check the derived formula against the VaR formula: the ES
    multiplier phi(a)/(1-alpha) must exceed a itself, since CVaR >= VaR
    for any confidence level below 1."""
    from scipy import stats

    for alpha_value in (0.90, 0.95, 0.99):
        z_alpha = stats.norm.ppf(alpha_value)
        es_multiplier_value = float(
            derive_es_multiplier().subs({a: z_alpha, alpha: alpha_value})
        )
        assert es_multiplier_value > z_alpha
