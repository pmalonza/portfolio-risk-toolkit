# Portfolio Risk Toolkit

[![CI](https://github.com/pmalonza/portfolio-risk-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/pmalonza/portfolio-risk-toolkit/actions/workflows/ci.yml)

A small toolkit for computing **Value-at-Risk (VaR)** and **Conditional
VaR / Expected Shortfall (CVaR)** for a multi-asset portfolio using three
independent methods, a from-first-principles symbolic derivation of the
parametric formula, and a statistical backtest of model calibration.

## Why three methods

Each method makes different assumptions and fails in different ways, which
is exactly why practitioners run all three side by side:

| Method | Assumption | Module |
|---|---|---|
| **Historical simulation** | None — uses the empirical loss quantile directly | `historical.py` |
| **Parametric (variance-covariance)** | Losses follow a fitted Normal or Student-t distribution | `parametric.py` |
| **Monte Carlo** | Assets follow a fitted multivariate Normal; simulate scenarios, then aggregate | `monte_carlo.py` |

On the synthetic near-normal data this toolkit generates, all three agree
closely — which is itself a useful sanity check: if they diverge sharply,
that's usually a sign of fat tails, skew, or a bug. Real market data makes
that divergence concrete: fitting a 4-stock portfolio (AAPL/MSFT/GOOG/AMZN,
2 years of history) gives a fitted Student-t of **~4.15 degrees of
freedom** — real markets have visibly fatter tails than the Normal
assumption, which is exactly the kind of thing this comparison is meant to
surface. See [Real market data](#real-market-data) below.

## Symbolic derivation

`symbolic.py` derives the parametric Normal CVaR formula from scratch with
SymPy rather than just citing it. The key step is the tail-expectation
integral for a standard normal:

```
integral_{a}^{infinity} z * phi(z) dz = phi(a)
```

which SymPy confirms directly (`phi' (z) = -z * phi(z)`, so `z*phi(z)` is an
exact derivative). Combining that with the tail probability `(1 - alpha)`
gives the closed-form expected-shortfall multiplier, and the full CVaR
formula for a location-scale Normal loss:

```
CVaR_alpha = -mu + sigma * phi(z_alpha) / (1 - alpha)
```

The derivation is cross-checked numerically against `scipy.stats.norm`
across multiple confidence levels and `(mu, sigma)` combinations in
`tests/test_symbolic.py` — see `verify_against_scipy()`.

## Backtesting

`backtest.py` implements the Kupiec (1995) proportion-of-failures test: it
counts how often realized losses exceeded the VaR estimate and runs a
likelihood-ratio test (chi-squared, 1 degree of freedom) of whether that
exceedance rate is statistically consistent with the model's nominal
`(1 - alpha)` rate — i.e. whether the VaR model is well-calibrated rather
than systematically too tight or too loose.

## Installation

```bash
git clone https://github.com/pmalonza/portfolio-risk-toolkit.git
cd portfolio-risk-toolkit
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

To also run on real market data, install the optional extra:

```bash
pip install -e ".[dev,market-data]"
```

## Usage

### CLI

```bash
portfolio-risk --n-assets 5 --n-days 1500 --alpha 0.95 --n-sims 100000 --plot-dir output/
```

Add `--dist t` to fit a Student-t distribution for the parametric method
instead of Normal (fatter tails, usually a better fit for real returns):

```
Portfolio Risk Toolkit -- 5 synthetic assets, 1500 days, alpha=95%

Method                           VaR        CVaR
------------------------------------------------
Historical                   0.8293%     1.0376%
Parametric (Normal)          0.8218%     1.0367%
Monte Carlo                  0.8066%     1.0149%

Kupiec backtest (out-of-sample):
  observations:  600
  exceedances:   41 (expected 30.0)
  LR statistic:  3.8284 (critical value 3.8415)
  p-value:       0.0504
  verdict:       not rejected - well-calibrated
```

### As a library

```python
from portfolio_risk.data import generate_synthetic_returns, aggregate_portfolio_returns
from portfolio_risk.historical import historical_var, historical_cvar
from portfolio_risk.parametric import fit_normal, normal_var, normal_cvar
from portfolio_risk.monte_carlo import monte_carlo_var, monte_carlo_cvar

returns, mu, cov = generate_synthetic_returns(n_assets=4, n_days=2000, seed=123)
weights = [0.40, 0.30, 0.20, 0.10]
portfolio_returns = aggregate_portfolio_returns(returns, weights)

var_hist = historical_var(portfolio_returns, alpha=0.99)
fit = fit_normal(portfolio_returns)
var_param = normal_var(fit, alpha=0.99)
var_mc = monte_carlo_var(mu, cov, weights, alpha=0.99, n_sims=200_000, seed=123)
```

See [`examples/example_run.py`](examples/example_run.py) for a complete
walkthrough that also runs the symbolic cross-check and the backtest.

## Real market data

Everything above also runs on real historical prices via the optional
`market-data` extra (`yfinance`) — swap `--n-assets`/`--n-days` for
`--tickers`:

```bash
portfolio-risk --tickers AAPL MSFT GOOG AMZN --period 2y --alpha 0.95
```

```
Portfolio Risk Toolkit -- AAPL, MSFT, GOOG, AMZN (501 real trading days), alpha=95%

Method                           VaR        CVaR
------------------------------------------------
Historical                   2.3009%     3.1426%
Parametric (Normal)          2.2762%     2.8810%
Monte Carlo                  2.2869%     2.8841%

Kupiec backtest (out-of-sample):
  observations:  201
  exceedances:   7 (expected 10.1)
  LR statistic:  1.0852 (critical value 3.8415)
  p-value:       0.2975
  verdict:       not rejected - well-calibrated
```

`--weights` sets a custom allocation (must match `--tickers` order and sum
to 1); it defaults to equal-weight. As a library:

```python
from portfolio_risk.market_data import load_market_returns, fit_moments
from portfolio_risk.data import aggregate_portfolio_returns, equal_weights

returns, dates, tickers = load_market_returns(["AAPL", "MSFT", "GOOG"], period="2y")
weights = equal_weights(len(tickers))
portfolio_returns = aggregate_portfolio_returns(returns, weights)
mu, cov = fit_moments(returns)  # feed straight into monte_carlo_var/cvar
```

See [`examples/real_portfolio_example.py`](examples/real_portfolio_example.py)
for a full walkthrough, including the Normal-vs-Student-t comparison that
surfaces real fat tails.

## Project structure

```
src/portfolio_risk/
    data.py         synthetic multi-asset return generation + portfolio aggregation
    market_data.py  real historical prices via yfinance (optional; network required)
    historical.py   historical simulation VaR/CVaR
    parametric.py   parametric variance-covariance VaR/CVaR (Normal, Student-t)
    monte_carlo.py  Monte Carlo VaR/CVaR via simulated scenarios
    symbolic.py      SymPy derivation of the parametric CVaR formula
    backtest.py      Kupiec proportion-of-failures backtest
    plotting.py      Matplotlib comparison and exceedance plots
    cli.py           command-line entry point
tests/               pytest suite, one file per module
examples/            example_run.py, real_portfolio_example.py
```

## Running tests

```bash
pytest -q
```

`tests/test_market_data.py` includes one test marked `network` that makes a
real call to Yahoo Finance; it's excluded by default (see the `markers`
config in `pyproject.toml`) so the suite stays fast and CI-safe. Run it
explicitly with:

```bash
pytest -m network tests/test_market_data.py
```

## License

MIT — see [LICENSE](LICENSE).
