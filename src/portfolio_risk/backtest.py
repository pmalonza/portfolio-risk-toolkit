"""Kupiec (1995) proportion-of-failures backtest for VaR model calibration.

Given a series of realized returns and a VaR estimate (constant, or one
value per observation), counts how often the realized loss exceeded the
VaR estimate and tests whether that exceedance rate is statistically
consistent with the nominal (1 - alpha) exceedance probability, via a
likelihood-ratio test that is chi-squared distributed with 1 degree of
freedom under the null hypothesis of correct calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class KupiecResult:
    n_obs: int
    n_exceedances: int
    expected_exceedances: float
    exceedance_rate: float
    lr_statistic: float
    p_value: float
    critical_value: float
    reject_null: bool


def count_exceedances(returns: np.ndarray, var_estimates: np.ndarray | float) -> int:
    """Count observations where the realized loss exceeds the VaR estimate."""
    losses = -np.asarray(returns, dtype=float)
    var_array = np.broadcast_to(np.asarray(var_estimates, dtype=float), losses.shape)
    return int((losses > var_array).sum())


def _bernoulli_log_likelihood(rate: float, n_exceedances: int, n_obs: int) -> float:
    """Log-likelihood of observing ``n_exceedances`` successes in ``n_obs``
    Bernoulli(rate) trials, using the 0-successes / all-successes limits at
    the boundaries so that a rate of exactly 0 or 1 doesn't hit log(0)."""
    if n_exceedances == 0:
        return n_obs * np.log(1.0 - rate)
    if n_exceedances == n_obs:
        return n_obs * np.log(rate)
    return (n_obs - n_exceedances) * np.log(1.0 - rate) + n_exceedances * np.log(rate)


def kupiec_pof_test(
    returns: np.ndarray,
    var_estimates: np.ndarray | float,
    alpha: float = 0.95,
    test_confidence: float = 0.95,
) -> KupiecResult:
    """Run the Kupiec proportion-of-failures test.

    ``alpha`` is the VaR model's confidence level (e.g. 0.95 means a nominal
    5% exceedance rate is expected). ``test_confidence`` sets the
    significance threshold for the chi-squared test itself (e.g. 0.95 means
    reject calibration at the 5% test significance level).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be strictly between 0 and 1")
    if not (0.0 < test_confidence < 1.0):
        raise ValueError("test_confidence must be strictly between 0 and 1")

    returns = np.asarray(returns, dtype=float)
    n_obs = returns.shape[0]
    if n_obs < 1:
        raise ValueError("returns must be non-empty")

    n_exceedances = count_exceedances(returns, var_estimates)
    nominal_rate = 1.0 - alpha
    observed_rate = n_exceedances / n_obs

    log_l_null = _bernoulli_log_likelihood(nominal_rate, n_exceedances, n_obs)
    log_l_alt = _bernoulli_log_likelihood(observed_rate, n_exceedances, n_obs)
    lr_statistic = -2.0 * (log_l_null - log_l_alt)

    p_value = float(1.0 - stats.chi2.cdf(lr_statistic, df=1))
    critical_value = float(stats.chi2.ppf(test_confidence, df=1))

    return KupiecResult(
        n_obs=n_obs,
        n_exceedances=n_exceedances,
        expected_exceedances=nominal_rate * n_obs,
        exceedance_rate=observed_rate,
        lr_statistic=float(lr_statistic),
        p_value=p_value,
        critical_value=critical_value,
        reject_null=bool(lr_statistic > critical_value),
    )
