"""Example: use the library directly (rather than the CLI) to compute and
compare VaR/CVaR for a 4-asset portfolio, then backtest the historical model.

Run with:
    python examples/example_run.py
"""

from __future__ import annotations

from portfolio_risk.backtest import kupiec_pof_test
from portfolio_risk.data import aggregate_portfolio_returns, generate_synthetic_returns
from portfolio_risk.historical import historical_cvar, historical_var
from portfolio_risk.monte_carlo import monte_carlo_cvar, monte_carlo_var
from portfolio_risk.parametric import fit_normal, normal_cvar, normal_var
from portfolio_risk.plotting import plot_exceedances, plot_var_comparison
from portfolio_risk.symbolic import verify_against_scipy


def main() -> None:
    weights = [0.40, 0.30, 0.20, 0.10]  # a hand-picked allocation, not equal-weight
    returns, mu, cov = generate_synthetic_returns(n_assets=4, n_days=2000, seed=123)
    portfolio_returns = aggregate_portfolio_returns(returns, weights)

    alpha = 0.99  # a stricter confidence level than the CLI default
    print(f"4-asset portfolio, weights={weights}, alpha={alpha:.0%}\n")

    hist_var = historical_var(portfolio_returns, alpha)
    hist_cvar = historical_cvar(portfolio_returns, alpha)
    print(f"Historical:   VaR={hist_var:.4%}  CVaR={hist_cvar:.4%}")

    fit = fit_normal(portfolio_returns)
    param_var = normal_var(fit, alpha)
    param_cvar = normal_cvar(fit, alpha)
    print(f"Parametric:   VaR={param_var:.4%}  CVaR={param_cvar:.4%}")

    mc_var = monte_carlo_var(mu, cov, weights, alpha=alpha, n_sims=200_000, seed=123)
    mc_cvar = monte_carlo_cvar(mu, cov, weights, alpha=alpha, n_sims=200_000, seed=123)
    print(f"Monte Carlo:  VaR={mc_var:.4%}  CVaR={mc_cvar:.4%}")

    symbolic_cvar, scipy_cvar = verify_against_scipy(fit.mu, fit.sigma, alpha)
    print(
        f"\nSymPy-derived CVaR formula vs. SciPy closed form: "
        f"{symbolic_cvar:.6f} vs {scipy_cvar:.6f} "
        f"(match: {abs(symbolic_cvar - scipy_cvar) < 1e-9})"
    )

    split = int(len(portfolio_returns) * 0.6)
    in_sample, out_of_sample = portfolio_returns[:split], portfolio_returns[split:]
    backtest_var = historical_var(in_sample, alpha)
    result = kupiec_pof_test(out_of_sample, backtest_var, alpha=alpha)
    print(
        f"\nKupiec backtest: {result.n_exceedances} exceedances in {result.n_obs} obs "
        f"(expected {result.expected_exceedances:.1f}), "
        f"{'REJECTED' if result.reject_null else 'not rejected'}"
    )

    plot_var_comparison(
        portfolio_returns, {"Historical": hist_var, "Parametric": param_var, "Monte Carlo": mc_var}, alpha=alpha
    ).figure.savefig("example_var_comparison.png")
    plot_exceedances(out_of_sample, backtest_var).figure.savefig("example_exceedances.png")
    print("\nSaved example_var_comparison.png and example_exceedances.png")


if __name__ == "__main__":
    main()
