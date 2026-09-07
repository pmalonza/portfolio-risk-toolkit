"""Command-line entry point tying together data generation, all three VaR/CVaR
methods, and a Kupiec backtest, with optional plot output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .backtest import kupiec_pof_test
from .data import aggregate_portfolio_returns, equal_weights, generate_synthetic_returns
from .historical import historical_cvar, historical_var
from .monte_carlo import monte_carlo_cvar, monte_carlo_var
from .parametric import fit_normal, normal_cvar, normal_var
from .plotting import plot_exceedances, plot_var_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="portfolio-risk",
        description="Compute portfolio VaR/CVaR via historical, parametric, and Monte Carlo methods, then backtest.",
    )
    parser.add_argument("--n-assets", type=int, default=5, help="Number of assets in the synthetic portfolio")
    parser.add_argument("--n-days", type=int, default=1500, help="Number of simulated daily return observations")
    parser.add_argument("--alpha", type=float, default=0.95, help="VaR/CVaR confidence level")
    parser.add_argument("--n-sims", type=int, default=100_000, help="Number of Monte Carlo scenarios")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--backtest-split",
        type=float,
        default=0.6,
        help="Fraction of days used in-sample to fit the historical VaR before backtesting on the remainder",
    )
    parser.add_argument("--plot-dir", type=Path, default=None, help="If set, save comparison plots to this directory")
    return parser


def run(args: argparse.Namespace) -> dict:
    returns, mu, cov = generate_synthetic_returns(
        n_assets=args.n_assets, n_days=args.n_days, seed=args.seed
    )
    weights = equal_weights(args.n_assets)
    portfolio_returns = aggregate_portfolio_returns(returns, weights)

    hist_var = historical_var(portfolio_returns, args.alpha)
    hist_cvar = historical_cvar(portfolio_returns, args.alpha)

    fit = fit_normal(portfolio_returns)
    param_var = normal_var(fit, args.alpha)
    param_cvar = normal_cvar(fit, args.alpha)

    mc_var = monte_carlo_var(mu, cov, weights, alpha=args.alpha, n_sims=args.n_sims, seed=args.seed)
    mc_cvar = monte_carlo_cvar(mu, cov, weights, alpha=args.alpha, n_sims=args.n_sims, seed=args.seed)

    split_idx = int(len(portfolio_returns) * args.backtest_split)
    in_sample, out_of_sample = portfolio_returns[:split_idx], portfolio_returns[split_idx:]
    backtest_var = historical_var(in_sample, args.alpha)
    backtest_result = kupiec_pof_test(out_of_sample, backtest_var, alpha=args.alpha)

    results = {
        "historical": {"var": hist_var, "cvar": hist_cvar},
        "parametric": {"var": param_var, "cvar": param_cvar},
        "monte_carlo": {"var": mc_var, "cvar": mc_cvar},
        "backtest": backtest_result,
    }

    if args.plot_dir is not None:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        comparison_ax = plot_var_comparison(
            portfolio_returns,
            {"Historical": hist_var, "Parametric": param_var, "Monte Carlo": mc_var},
            alpha=args.alpha,
        )
        comparison_ax.figure.savefig(args.plot_dir / "var_comparison.png")

        exceedance_ax = plot_exceedances(out_of_sample, backtest_var)
        exceedance_ax.figure.savefig(args.plot_dir / "exceedances.png")

    return results


def _print_results(args: argparse.Namespace, results: dict) -> None:
    print(f"Portfolio Risk Toolkit -- {args.n_assets} assets, {args.n_days} days, alpha={args.alpha:.0%}\n")
    header = f"{'Method':<14}{'VaR':>12}{'CVaR':>12}"
    print(header)
    print("-" * len(header))
    for name, key in (("Historical", "historical"), ("Parametric", "parametric"), ("Monte Carlo", "monte_carlo")):
        row = results[key]
        print(f"{name:<14}{row['var']:>12.4%}{row['cvar']:>12.4%}")

    bt = results["backtest"]
    print("\nKupiec backtest (out-of-sample):")
    print(f"  observations:  {bt.n_obs}")
    print(f"  exceedances:   {bt.n_exceedances} (expected {bt.expected_exceedances:.1f})")
    print(f"  LR statistic:  {bt.lr_statistic:.4f} (critical value {bt.critical_value:.4f})")
    print(f"  p-value:       {bt.p_value:.4f}")
    print(f"  verdict:       {'REJECTED - miscalibrated' if bt.reject_null else 'not rejected - well-calibrated'}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    results = run(args)
    _print_results(args, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
