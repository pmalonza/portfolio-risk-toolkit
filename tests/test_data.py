import numpy as np
import pytest

from portfolio_risk.data import (
    aggregate_portfolio_returns,
    equal_weights,
    generate_synthetic_returns,
)


def test_generate_synthetic_returns_shapes():
    returns, mu, cov = generate_synthetic_returns(n_assets=4, n_days=250, seed=1)
    assert returns.shape == (250, 4)
    assert mu.shape == (4,)
    assert cov.shape == (4, 4)


def test_generate_synthetic_returns_cov_is_symmetric_and_psd():
    _, _, cov = generate_synthetic_returns(n_assets=6, n_days=100, seed=7)
    assert np.allclose(cov, cov.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert (eigenvalues >= -1e-10).all()


def test_generate_synthetic_returns_is_reproducible_with_seed():
    r1, _, _ = generate_synthetic_returns(n_assets=3, n_days=50, seed=99)
    r2, _, _ = generate_synthetic_returns(n_assets=3, n_days=50, seed=99)
    assert np.array_equal(r1, r2)


def test_generate_synthetic_returns_rejects_invalid_sizes():
    with pytest.raises(ValueError):
        generate_synthetic_returns(n_assets=0)
    with pytest.raises(ValueError):
        generate_synthetic_returns(n_days=0)


def test_equal_weights_sums_to_one():
    w = equal_weights(5)
    assert w.shape == (5,)
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w, 0.2)


def test_aggregate_portfolio_returns_matches_manual_computation():
    returns = np.array([[0.01, -0.02], [0.03, 0.01]])
    weights = np.array([0.5, 0.5])
    result = aggregate_portfolio_returns(returns, weights)
    expected = np.array([-0.005, 0.02])
    assert np.allclose(result, expected)


def test_aggregate_portfolio_returns_rejects_bad_weights():
    returns = np.zeros((10, 3))
    with pytest.raises(ValueError):
        aggregate_portfolio_returns(returns, np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        aggregate_portfolio_returns(returns, np.array([0.5, 0.5, 0.1]))
