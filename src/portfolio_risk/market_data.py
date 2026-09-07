"""Load real historical market data for use with the same VaR/CVaR pipeline
used for synthetic data (see ``data.py``).

This is the only module in the toolkit that touches the network or has a
non-core dependency: it requires the optional ``yfinance`` package. Nothing
else imports this module, so the rest of the toolkit works fully offline.
"""

from __future__ import annotations

import numpy as np


class MarketDataError(RuntimeError):
    """Raised when live market data cannot be downloaded or is unusable."""


def download_adjusted_close(
    tickers: list[str],
    period: str = "2y",
    start: str | None = None,
    end: str | None = None,
):
    """Download adjusted close prices for two or more tickers via yfinance.

    Returns a pandas DataFrame indexed by date, one column per ticker,
    aligned so every row has a price for every ticker (rows with any
    missing value -- e.g. from exchange holidays that don't line up across
    markets -- are dropped).
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for real market data. Install it with: "
            "pip install 'portfolio-risk-toolkit[market-data]'"
        ) from exc

    tickers = list(tickers)
    if len(tickers) < 2:
        raise ValueError("provide at least 2 tickers to form a portfolio")

    kwargs = {"auto_adjust": True, "progress": False}
    if start is not None or end is not None:
        kwargs["start"] = start
        kwargs["end"] = end
    else:
        kwargs["period"] = period

    raw = yf.download(tickers, **kwargs)
    if raw.empty:
        raise MarketDataError(f"no data returned for tickers {tickers}")

    prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    prices = prices[tickers].dropna(how="any")
    if prices.shape[0] < 30:
        raise MarketDataError(
            f"only {prices.shape[0]} aligned trading days available for {tickers}; need at least 30"
        )
    return prices


def compute_returns_from_prices(prices):
    """Convert a price DataFrame into a (returns, dates) pair of numpy arrays."""
    returns_df = prices.pct_change().dropna(how="any")
    return returns_df.to_numpy(), returns_df.index


def load_market_returns(
    tickers: list[str],
    period: str = "2y",
    start: str | None = None,
    end: str | None = None,
) -> tuple[np.ndarray, object, list[str]]:
    """Download real prices and convert directly to a portfolio-ready return series.

    Returns ``(returns, dates, tickers)`` where ``returns`` has shape
    ``(n_days, n_assets)`` -- the same shape ``generate_synthetic_returns``
    produces, so it can be dropped straight into the rest of the toolkit
    (historical/parametric/Monte Carlo VaR, the CLI, etc.).
    """
    prices = download_adjusted_close(tickers, period=period, start=start, end=end)
    returns, dates = compute_returns_from_prices(prices)
    return returns, dates, list(prices.columns)


def fit_moments(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample mean vector and covariance matrix of a real return series.

    Mirrors the (mu, cov) pair ``generate_synthetic_returns`` returns, so
    Monte Carlo simulation can be driven off real data the same way it's
    driven off the synthetic generator.
    """
    returns = np.asarray(returns, dtype=float)
    mu = returns.mean(axis=0)
    cov = np.cov(returns, rowvar=False)
    return mu, cov
