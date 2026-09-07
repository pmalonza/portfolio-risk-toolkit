"""Historical simulation VaR and CVaR.

Makes no distributional assumption: VaR is the empirical quantile of the
realized loss distribution, and CVaR is the mean loss beyond that quantile.
"""

from __future__ import annotations

import numpy as np


def _validate_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be strictly between 0 and 1")


def historical_var(returns: np.ndarray, alpha: float = 0.95) -> float:
    """Empirical Value-at-Risk at confidence level ``alpha``, as a positive loss."""
    _validate_alpha(alpha)
    losses = -np.asarray(returns, dtype=float)
    return float(np.quantile(losses, alpha))


def historical_cvar(returns: np.ndarray, alpha: float = 0.95) -> float:
    """Empirical Conditional VaR (expected shortfall) at confidence level ``alpha``."""
    _validate_alpha(alpha)
    losses = -np.asarray(returns, dtype=float)
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    if tail.size == 0:
        return var
    return float(tail.mean())
