"""Example: run the full VaR/CVaR pipeline on real historical stock prices
instead of synthetic data.

Requires the optional 'market-data' extra and network access:
    pip install -e ".[market-data]"
    python examples/real_portfolio_example.py
"""

from __future__ import annotations

from portfolio_risk.backtest import kupiec_pof_test
from portfolio_risk.data import aggregate_portfolio_returns, equal_weights
from portfolio_risk.historical import historical_cvar, historical_var
from portfolio_risk.market_data import fit_moments, load_market_returns
from portfolio_risk.monte_carlo import monte_carlo_cvar, monte_carlo_var
from portfolio_risk.parametric import fit_normal, fit_student_t, normal_cvar, normal_var, student_t_cvar, student_t_var


def main() -> None:
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
    alpha = 0.95

    returns, dates, tickers = load_market_returns(tickers, period="2y")
    weights = equal_weights(len(tickers))
    portfolio_returns = aggregate_portfolio_returns(returns, weights)
    mu, cov = fit_moments(returns)

    print(f"Portfolio: {tickers} (equal-weight)")
    print(f"History:   {dates[0].date()} to {dates[-1].date()} ({len(dates)} trading days)\n")

    hist_var = historical_var(portfolio_returns, alpha)
    hist_cvar = historical_cvar(portfolio_returns, alpha)
    print(f"Historical:    VaR={hist_var:.4%}  CVaR={hist_cvar:.4%}")

    normal_fit = fit_normal(portfolio_returns)
    print(f"Parametric-N:  VaR={normal_var(normal_fit, alpha):.4%}  CVaR={normal_cvar(normal_fit, alpha):.4%}")

    t_fit = fit_student_t(portfolio_returns)
    print(
        f"Parametric-t:  VaR={student_t_var(t_fit, alpha):.4%}  CVaR={student_t_cvar(t_fit, alpha):.4%}  "
        f"(fitted dof={t_fit.dof:.2f} -- real markets have fatter tails than Normal)"
    )

    mc_var = monte_carlo_var(mu, cov, weights, alpha=alpha, n_sims=200_000, seed=1)
    mc_cvar = monte_carlo_cvar(mu, cov, weights, alpha=alpha, n_sims=200_000, seed=1)
    print(f"Monte Carlo:   VaR={mc_var:.4%}  CVaR={mc_cvar:.4%}")

    split = int(len(portfolio_returns) * 0.6)
    in_sample, out_of_sample = portfolio_returns[:split], portfolio_returns[split:]
    backtest_var = historical_var(in_sample, alpha)
    result = kupiec_pof_test(out_of_sample, backtest_var, alpha=alpha)
    print(
        f"\nKupiec backtest: {result.n_exceedances}/{result.n_obs} exceedances "
        f"(expected {result.expected_exceedances:.1f}), "
        f"{'REJECTED' if result.reject_null else 'not rejected'} (p={result.p_value:.3f})"
    )


if __name__ == "__main__":
    main()
