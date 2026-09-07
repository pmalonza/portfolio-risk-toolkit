"""Symbolic derivation of the parametric Normal CVaR (expected shortfall) formula.

Derives, from first principles via SymPy integration, the closed form used
by ``portfolio_risk.parametric.normal_cvar``:

    CVaR_alpha = -mu + sigma * phi(z_alpha) / (1 - alpha)

where ``z_alpha`` is the standard-normal alpha-quantile and ``phi`` is the
standard normal density. The key symbolic fact being derived is:

    integral_{a}^{infinity} z * phi(z) dz = phi(a)

i.e. that the tail-expectation numerator of a standard normal collapses to
the density evaluated at the cutoff. This module derives that integral
symbolically and combines it with the tail probability (1 - alpha) to build
the full CVaR formula, then lets callers verify the result numerically
against SciPy.

Note on a real SymPy pitfall: symbols are matched by name *and* assumptions,
so ``sympy.symbols('a')`` and ``sympy.symbols('a', real=True)`` are distinct
symbols even though they print identically -- substituting into an
expression built from one using the other silently does nothing (no error,
the substitution is just a no-op). To avoid that trap, this module defines
its symbols once at module scope and every function reuses those same
objects rather than re-declaring "a" or "alpha" locally.
"""

from __future__ import annotations

import sympy as sp
from scipy import stats

from .parametric import NormalFit, normal_cvar

# Canonical symbols, defined once and reused everywhere in this module so
# that `.subs(...)` calls always match (see module docstring).
z = sp.symbols("z", real=True)
a = sp.symbols("a", real=True)
alpha = sp.symbols("alpha", positive=True)
mu = sp.symbols("mu", real=True)
sigma = sp.symbols("sigma", positive=True)

# Standard normal density, as a function of z.
_STANDARD_NORMAL_PDF = sp.exp(-(z**2) / 2) / sp.sqrt(2 * sp.pi)


def standard_normal_pdf() -> sp.Expr:
    """Return the standard normal density phi(z) as a SymPy expression in ``z``."""
    return _STANDARD_NORMAL_PDF


def derive_tail_expectation() -> sp.Expr:
    """Symbolically derive integral_{a}^{oo} z * phi(z) dz.

    This is the numerator of E[Z | Z >= a] for a standard normal Z. The
    closed form is phi(a), which falls straight out of the fact that phi'(z)
    = -z * phi(z), making z*phi(z) an exact derivative.
    """
    integral = sp.integrate(z * _STANDARD_NORMAL_PDF, (z, a, sp.oo))
    return sp.simplify(integral)


def derive_es_multiplier() -> sp.Expr:
    """Symbolically derive E[Z | Z >= a] = phi(a) / (1 - alpha) for standard normal Z."""
    tail_expectation = derive_tail_expectation()
    return tail_expectation / (1 - alpha)


def derive_normal_cvar_formula() -> sp.Expr:
    """Symbolically derive the full location-scale Normal CVaR formula.

    For loss L = -mu + sigma * Z with Z standard normal, this returns
    CVaR_alpha(L) = -mu + sigma * E[Z | Z >= a] as a SymPy expression in
    mu, sigma, a, and alpha.
    """
    return -mu + sigma * derive_es_multiplier()


def evaluate_normal_cvar_formula(mu_value: float, sigma_value: float, alpha_value: float) -> float:
    """Numerically evaluate the symbolically-derived CVaR formula at concrete inputs.

    ``a`` (the alpha-quantile of the standard normal) is supplied via SciPy's
    ``norm.ppf``, since SymPy has no closed form for the inverse CDF.
    """
    if not (0.0 < alpha_value < 1.0):
        raise ValueError("alpha_value must be strictly between 0 and 1")
    formula = derive_normal_cvar_formula()
    z_alpha = float(stats.norm.ppf(alpha_value))
    value = formula.subs({mu: mu_value, sigma: sigma_value, a: z_alpha, alpha: alpha_value})
    return float(value)


def verify_against_scipy(mu_value: float, sigma_value: float, alpha_value: float) -> tuple[float, float]:
    """Cross-check the symbolic derivation against ``portfolio_risk.parametric.normal_cvar``.

    Returns (symbolic_value, scipy_value); callers can assert these are close.
    """
    symbolic_value = evaluate_normal_cvar_formula(mu_value, sigma_value, alpha_value)
    scipy_value = normal_cvar(NormalFit(mu=mu_value, sigma=sigma_value), alpha_value)
    return symbolic_value, scipy_value
