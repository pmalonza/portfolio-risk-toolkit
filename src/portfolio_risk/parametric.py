"""Parametric (variance-covariance) VaR and CVaR using SciPy distributions.

Supports a fitted Normal model and a fitted Student-t model (heavier tails).
Both VaR and CVaR (expected shortfall) are computed from closed-form
expressions rather than simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def _validate_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be strictly between 0 and 1")


@dataclass(frozen=True)
class NormalFit:
    mu: float
    sigma: float


@dataclass(frozen=True)
class StudentTFit:
    dof: float
    loc: float
    scale: float


def fit_normal(returns: np.ndarray) -> NormalFit:
    """Fit a Normal distribution to a return series via sample mean/std."""
    returns = np.asarray(returns, dtype=float)
    return NormalFit(mu=float(returns.mean()), sigma=float(returns.std(ddof=1)))


def fit_student_t(returns: np.ndarray) -> StudentTFit:
    """Fit a Student-t distribution to a return series via maximum likelihood."""
    returns = np.asarray(returns, dtype=float)
    dof, loc, scale = stats.t.fit(returns)
    return StudentTFit(dof=float(dof), loc=float(loc), scale=float(scale))


def normal_var(fit: NormalFit, alpha: float = 0.95) -> float:
    """Closed-form Normal VaR: -mu + sigma * Phi^-1(alpha)."""
    _validate_alpha(alpha)
    z = stats.norm.ppf(alpha)
    return float(-fit.mu + fit.sigma * z)


def normal_cvar(fit: NormalFit, alpha: float = 0.95) -> float:
    """Closed-form Normal CVaR: -mu + sigma * phi(z_alpha) / (1 - alpha)."""
    _validate_alpha(alpha)
    z = stats.norm.ppf(alpha)
    es_multiplier = stats.norm.pdf(z) / (1.0 - alpha)
    return float(-fit.mu + fit.sigma * es_multiplier)


def student_t_var(fit: StudentTFit, alpha: float = 0.95) -> float:
    """Closed-form Student-t VaR: -loc + scale * t.ppf(alpha, dof)."""
    _validate_alpha(alpha)
    z = stats.t.ppf(alpha, fit.dof)
    return float(-fit.loc + fit.scale * z)


def student_t_cvar(fit: StudentTFit, alpha: float = 0.95) -> float:
    """Closed-form Student-t CVaR (expected shortfall).

    Uses the standard result for the expected shortfall of a location-scale
    Student-t distribution (requires dof > 1 for a finite mean):

        ES(alpha) = t.pdf(z_alpha, dof) / (1 - alpha) * (dof + z_alpha^2) / (dof - 1)
    """
    _validate_alpha(alpha)
    if fit.dof <= 1:
        raise ValueError("Student-t CVaR requires dof > 1 for a finite mean")
    z = stats.t.ppf(alpha, fit.dof)
    es_multiplier = (
        stats.t.pdf(z, fit.dof) / (1.0 - alpha) * (fit.dof + z**2) / (fit.dof - 1.0)
    )
    return float(-fit.loc + fit.scale * es_multiplier)


def parametric_var(returns: np.ndarray, alpha: float = 0.95, dist: str = "normal") -> float:
    """Convenience wrapper: fit ``dist`` to ``returns`` and return its VaR."""
    if dist == "normal":
        return normal_var(fit_normal(returns), alpha)
    if dist == "t":
        return student_t_var(fit_student_t(returns), alpha)
    raise ValueError("dist must be 'normal' or 't'")


def parametric_cvar(returns: np.ndarray, alpha: float = 0.95, dist: str = "normal") -> float:
    """Convenience wrapper: fit ``dist`` to ``returns`` and return its CVaR."""
    if dist == "normal":
        return normal_cvar(fit_normal(returns), alpha)
    if dist == "t":
        return student_t_cvar(fit_student_t(returns), alpha)
    raise ValueError("dist must be 'normal' or 't'")
