import numpy as np
import pytest

pd = pytest.importorskip("pandas")
yfinance = pytest.importorskip("yfinance")

from portfolio_risk.market_data import (  # noqa: E402
    MarketDataError,
    compute_returns_from_prices,
    download_adjusted_close,
    fit_moments,
    load_market_returns,
)


def _fake_multiindex_close(tickers, n_days=40, with_gap=False):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    close = pd.DataFrame(
        {ticker: np.linspace(100 + i * 10, 110 + i * 10, n_days) for i, ticker in enumerate(tickers)},
        index=dates,
    )
    if with_gap:
        close.iloc[5, 0] = np.nan  # simulate a missing print for the first ticker
    return pd.concat({"Close": close}, axis=1)


def test_download_adjusted_close_rejects_single_ticker():
    with pytest.raises(ValueError):
        download_adjusted_close(["AAPL"])


def test_download_adjusted_close_parses_multiindex_and_aligns_dates(monkeypatch):
    tickers = ["AAPL", "MSFT"]
    fake_raw = _fake_multiindex_close(tickers, n_days=40, with_gap=True)
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: fake_raw)

    prices = download_adjusted_close(tickers, period="2y")

    assert list(prices.columns) == tickers
    assert prices.shape[0] == 39  # the gapped row was dropped
    assert not prices.isna().any().any()


def test_download_adjusted_close_raises_on_empty_response(monkeypatch):
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(MarketDataError):
        download_adjusted_close(["AAPL", "MSFT"])


def test_download_adjusted_close_raises_when_too_few_aligned_days(monkeypatch):
    fake_raw = _fake_multiindex_close(["AAPL", "MSFT"], n_days=10)
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: fake_raw)
    with pytest.raises(MarketDataError):
        download_adjusted_close(["AAPL", "MSFT"])


def test_compute_returns_from_prices_matches_manual_pct_change():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"A": [100.0, 102.0, 101.0, 105.0, 104.0]}, index=dates)

    returns, out_dates = compute_returns_from_prices(prices)

    expected = np.array([[0.02], [-1 / 102], [4 / 101], [-1 / 105]])
    assert returns.shape == (4, 1)
    assert np.allclose(returns, expected)
    assert len(out_dates) == 4


def test_load_market_returns_end_to_end_with_mocked_download(monkeypatch):
    tickers = ["AAPL", "MSFT", "GOOG"]
    fake_raw = _fake_multiindex_close(tickers, n_days=60)
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: fake_raw)

    returns, dates, resolved_tickers = load_market_returns(tickers, period="2y")

    assert returns.shape == (59, 3)
    assert resolved_tickers == tickers
    assert len(dates) == 59


def test_fit_moments_matches_numpy_mean_and_cov():
    rng = np.random.default_rng(0)
    returns = rng.normal(size=(500, 3))
    mu, cov = fit_moments(returns)
    assert np.allclose(mu, returns.mean(axis=0))
    assert np.allclose(cov, np.cov(returns, rowvar=False))


@pytest.mark.network
def test_load_market_returns_live_download_smoke():
    """Real network call against Yahoo Finance -- excluded from the default
    test run (see pyproject.toml markers); run explicitly with:
        pytest -m network tests/test_market_data.py
    """
    returns, dates, tickers = load_market_returns(["AAPL", "MSFT"], period="3mo")
    assert returns.shape[1] == 2
    assert returns.shape[0] >= 30
    assert np.isfinite(returns).all()
    assert tickers == ["AAPL", "MSFT"]
