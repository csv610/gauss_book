"""Tests for Gaussian Processes."""

import pytest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.gp import (
    rbf_kernel, matern_kernel, periodic_kernel, rational_quadratic_kernel,
    linear_kernel, white_kernel,
    GaussianProcess, simple_kriging, ordinary_kriging
)


class TestKernels:
    def test_rbf_identity(self):
        """RBF kernel with distance 0 returns variance."""
        x = np.array([[1.0]])
        K = rbf_kernel(x, x, length_scale=1.0, variance=1.0)
        assert K[0, 0] == 1.0

    def test_rbf_positive(self):
        """RBF kernel returns positive values."""
        x = np.array([[0.0], [1.0]])
        K = rbf_kernel(x, x, length_scale=1.0, variance=1.0)
        assert K[0, 0] == 1.0
        assert K[1, 1] == 1.0
        assert K[0, 1] > 0
        assert K[1, 0] == K[0, 1]

    def test_rbf_symmetric(self):
        x = np.array([[0.0], [1.0], [2.0]])
        K = rbf_kernel(x, x, length_scale=1.0, variance=1.0)
        assert np.allclose(K, K.T)

    def test_matern_1_5(self):
        x = np.array([[0.0], [1.0]])
        K = matern_kernel(x, x, length_scale=1.0, variance=1.0, nu=1.5)
        assert np.allclose(K[0, 0], 1.0, atol=1e-10)
        assert np.allclose(K[0, 1], K[1, 0])

    def test_linear_kernel(self):
        x = np.array([[1.0], [2.0]])
        K = linear_kernel(x, x, variance=1.0)
        assert K[0, 0] == 1.0
        assert K[0, 1] == 2.0

    def test_white_kernel(self):
        x = np.array([[0.0], [1.0], [2.0]])
        K = white_kernel(x, x, noise_level=0.1)
        assert K[0, 0] == 0.1
        assert K[1, 1] == 0.1
        assert K[0, 1] == 0.0


class TestGaussianProcess:
    def test_fit_predict(self):
        """Basic GP fit and predict."""
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 0.0])

        gp = GaussianProcess(rbf_kernel, noise=0.01)
        gp.fit(X, y, length_scale=1.0, variance=1.0)

        assert gp.X_train.shape == (3, 1)
        assert len(gp.y_train) == 3

    def test_predict_mean(self):
        """Predicted mean at training points should be close to observed."""
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 0.0])

        gp = GaussianProcess(rbf_kernel, noise=1e-6)
        gp.fit(X, y, length_scale=0.5, variance=1.0)

        mu, _ = gp.predict(X, return_std=False, length_scale=0.5, variance=1.0)
        assert np.allclose(mu, y, atol=0.1)

    def test_predict_std_positive(self):
        """Prediction standard deviation should be non-negative."""
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])

        gp = GaussianProcess(rbf_kernel, noise=0.1)
        gp.fit(X, y, length_scale=1.0, variance=1.0)

        _, std = gp.predict(X, return_std=True, length_scale=1.0, variance=1.0)
        assert np.all(std >= 0)

    def test_log_marginal_likelihood(self):
        """Log marginal likelihood should be a finite number."""
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 0.0])

        gp = GaussianProcess(rbf_kernel, noise=0.01)
        gp.fit(X, y, length_scale=1.0, variance=1.0)

        lml = gp.log_marginal_likelihood(length_scale=1.0, variance=1.0)
        assert np.isfinite(lml)
        assert lml < 0


class TestKriging:
    def test_simple_kriging(self):
        """Simple Kriging should produce finite predictions."""
        X_obs = np.array([[0.0], [1.0], [2.0]])
        y_obs = np.array([0.0, 1.0, 0.0])
        X_pred = np.array([[0.5], [1.5]])

        mu, std = simple_kriging(X_obs, y_obs, X_pred, rbf_kernel,
                                  length_scale=1.0, variance=1.0)
        assert len(mu) == 2
        assert len(std) == 2
        assert np.all(std >= 0)

    def test_ordinary_kriging(self):
        """Ordinary Kriging should produce finite predictions."""
        X_obs = np.array([[0.0], [1.0], [2.0]])
        y_obs = np.array([0.0, 1.0, 0.0])
        X_pred = np.array([[0.5], [1.5]])

        mu, std = ordinary_kriging(X_obs, y_obs, X_pred, rbf_kernel,
                                    length_scale=1.0, variance=1.0)
        assert len(mu) == 2
        assert len(std) == 2


class TestIntegration:
    def test_gp_regression_smooth_function(self):
        """GP should approximate sin function reasonably."""
        np.random.seed(42)
        X_train = np.linspace(0, 4, 10).reshape(-1, 1)
        y_train = np.sin(X_train.ravel())

        gp = GaussianProcess(rbf_kernel, noise=0.01)
        gp.fit(X_train, y_train, length_scale=0.5, variance=1.0)

        X_test = np.linspace(0, 4, 20).reshape(-1, 1)
        mu, _ = gp.predict(X_test, return_std=False, length_scale=0.5, variance=1.0)

        y_true = np.sin(X_test.ravel())
        mse = np.mean((mu - y_true)**2)
        assert mse < 0.1  # Should be reasonable fit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
