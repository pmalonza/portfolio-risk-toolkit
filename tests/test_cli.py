import numpy as np
import pytest

from portfolio_risk.cli import build_parser, main, run


def test_build_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.n_assets == 5
    assert args.n_days == 1500
    assert args.alpha == 0.95
    assert args.plot_dir is None
    assert args.tickers is None


def test_run_returns_all_three_methods_and_backtest():
    parser = build_parser()
    args = parser.parse_args(["--n-days", "400", "--n-sims", "5000", "--seed", "1"])
    results = run(args)
    for key in ("historical", "parametric", "monte_carlo"):
        assert results[key]["var"] > 0
        assert results[key]["cvar"] >= results[key]["var"]
    assert results["backtest"].n_obs > 0


def test_run_writes_plots_when_plot_dir_given(tmp_path):
    parser = build_parser()
    plot_dir = tmp_path / "plots"
    args = parser.parse_args(
        ["--n-days", "300", "--n-sims", "2000", "--seed", "2", "--plot-dir", str(plot_dir)]
    )
    run(args)
    assert (plot_dir / "var_comparison.png").exists()
    assert (plot_dir / "exceedances.png").exists()


def test_main_prints_results_and_returns_zero(capsys):
    exit_code = main(["--n-days", "300", "--n-sims", "2000", "--seed", "3"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Portfolio Risk Toolkit" in captured.out
    assert "Kupiec backtest" in captured.out


def test_three_methods_agree_closely_on_near_normal_data():
    parser = build_parser()
    args = parser.parse_args(["--n-days", "2000", "--n-sims", "150000", "--seed", "9"])
    results = run(args)
    var_values = [results[k]["var"] for k in ("historical", "parametric", "monte_carlo")]
    assert max(var_values) - min(var_values) < 0.003


def _fake_load_market_returns(tickers, period="2y", start=None, end=None):
    rng = np.random.default_rng(0)
    n_days = 300
    returns = rng.normal(loc=0.0003, scale=0.015, size=(n_days, len(tickers)))
    dates = list(range(n_days))
    return returns, dates, list(tickers)


def test_run_with_tickers_uses_market_data(monkeypatch):
    monkeypatch.setattr("portfolio_risk.cli.load_market_returns", _fake_load_market_returns)
    parser = build_parser()
    args = parser.parse_args(["--tickers", "AAPL", "MSFT", "GOOG", "--n-sims", "5000"])
    results = run(args)
    assert results["meta"]["source"] == "market"
    assert results["meta"]["tickers"] == ["AAPL", "MSFT", "GOOG"]
    assert results["historical"]["var"] > 0


def test_run_rejects_mismatched_weights_length(monkeypatch):
    monkeypatch.setattr("portfolio_risk.cli.load_market_returns", _fake_load_market_returns)
    parser = build_parser()
    args = parser.parse_args(
        ["--tickers", "AAPL", "MSFT", "GOOG", "--weights", "0.5", "0.5", "--n-sims", "5000"]
    )
    with pytest.raises(ValueError):
        run(args)
