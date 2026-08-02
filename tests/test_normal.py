"""Tests for normal distribution functions."""

import pytest
import sys
import os
import math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.normal import (
    normal_pdf, normal_cdf, normal_sf, normal_ppf,
    normal_rvs, multivariate_normal_rvs,
    normal_mle, normal_mle_unbiased,
    confidence_interval_t, z_test, t_test,
    error_propagation,
    multivariate_normal_pdf, multivariate_normal_logpdf,
    box_muller, marsaglia_bray
)


class TestNormalPdf:
    def test_standard_normal_at_zero(self):
        assert abs(normal_pdf(0) - 1/math.sqrt(2*math.pi)) < 1e-10

    def test_symmetry(self):
        assert abs(normal_pdf(1) - normal_pdf(-1)) < 1e-10

    def test_positive(self):
        assert normal_pdf(0) > 0
        assert normal_pdf(1) > 0
        assert normal_pdf(-1) > 0

    def test_non_negative(self):
        for x in [-3, -2, -1, 0, 1, 2, 3]:
            assert normal_pdf(x) >= 0


class TestNormalCdf:
    def test_limits(self):
        assert abs(normal_cdf(-10) - 0) < 1e-10
        assert abs(normal_cdf(10) - 1) < 1e-10

    def test_median(self):
        assert abs(normal_cdf(0) - 0.5) < 1e-10

    def test_68_rule(self):
        # P(-1 < Z < 1) ≈ 0.68
        prob = normal_cdf(1) - normal_cdf(-1)
        assert abs(prob - 0.6827) < 0.01

    def test_95_rule(self):
        # P(-1.96 < Z < 1.96) ≈ 0.95
        prob = normal_cdf(1.96) - normal_cdf(-1.96)
        assert abs(prob - 0.95) < 0.01


class TestNormalSf:
    def test_sf_and_cdf(self):
        for x in [-2, -1, 0, 1, 2]:
            assert abs(normal_sf(x) + normal_cdf(x) - 1) < 1e-10

    def test_positive(self):
        assert normal_sf(0) == 0.5
        assert 0 < normal_sf(1) < 1


class TestNormalPpf:
    def test_inverse(self):
        assert abs(normal_ppf(0.5) - 0) < 1e-10
        assert abs(normal_ppf(0.975) - 1.96) < 0.01
        assert abs(normal_ppf(0.025) - (-1.96)) < 0.01

    def test_boundary(self):
        assert normal_ppf(0) == -np.inf
        assert normal_ppf(1) == np.inf


class TestNormalRvs:
    def test_mean(self):
        np.random.seed(42)
        samples = normal_rvs(10000, mu=0, sigma=1)
        assert abs(samples.mean() - 0) < 0.1

    def test_std(self):
        np.random.seed(42)
        samples = normal_rvs(10000, mu=0, sigma=1)
        assert abs(samples.std() - 1) < 0.1

    def test_shape(self):
        samples = normal_rvs(100, mu=5, sigma=2)
        assert samples.shape == (100,)

    def test_reproducible(self):
        np.random.seed(42)
        s1 = normal_rvs(10, mu=0, sigma=1)
        np.random.seed(42)
        s2 = normal_rvs(10, mu=0, sigma=1)
        assert np.allclose(s1, s2)


class TestMultivariateNormalRvs:
    def test_shape(self):
        mu = np.array([0.0, 0.0])
        Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
        samples = multivariate_normal_rvs(mu, Sigma, size=100)
        assert samples.shape == (100, 2)

    def test_mean(self):
        mu = np.array([1.0, 2.0])
        Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
        samples = multivariate_normal_rvs(mu, Sigma, size=1000)
        assert abs(samples[:, 0].mean() - 1.0) < 0.2
        assert abs(samples[:, 1].mean() - 2.0) < 0.2


class TestNormalMle:
    def test_known_data(self):
        data = [2.0, 4.0, 6.0]
        mu, sigma = normal_mle(data)
        assert abs(mu - 4.0) < 1e-10
        assert abs(sigma - math.sqrt(8/3)) < 1e-10


class TestNormalMleUnbiased:
    def test_known_data(self):
        data = [2.0, 4.0, 6.0]
        mu, sigma = normal_mle_unbiased(data)
        assert abs(mu - 4.0) < 1e-10
        # Sample std with n-1: sqrt(((2-4)^2 + (4-4)^2 + (6-4)^2) / (3-1)) = sqrt(8/2) = 2.0
        assert abs(sigma - 2.0) < 1e-10


class TestConfidenceIntervalT:
    def test_basic(self):
        ci = confidence_interval_t(mean=10.0, std=2.0, n=25, confidence=0.95)
        assert ci[0] < 10.0 < ci[1]
        assert ci[1] - ci[0] > 0

    def test_symmetric(self):
        ci = confidence_interval_t(mean=10.0, std=2.0, n=25, confidence=0.95)
        margin = (ci[1] - ci[0]) / 2
        assert abs(ci[0] - (10.0 - margin)) < 1e-10


class TestZTest:
    def test_known_mean(self):
        data = [2.1, 2.3, 1.9, 2.2, 2.0]
        z, p = z_test(data, 2.0, sigma=0.1)
        assert z > 0
        assert 0 < p < 1

    def test_extreme_case(self):
        data = [100.0] * 10
        z, p = z_test(data, 100.0, sigma=1.0)
        assert abs(z) < 0.1
        assert p > 0.5


class TestTTest:
    def test_known_mean(self):
        data = [2.1, 2.3, 1.9, 2.2, 2.0]
        t, p = t_test(data, 2.0)
        assert t > 0
        assert 0 < p < 1


class TestErrorPropagation:
    def test_division(self):
        f = lambda x, y: x / y
        mean, std = error_propagation(f, [(10.0, 0.1), (5.0, 0.05)])
        assert abs(mean - 2.0) < 0.01
        assert std > 0


class TestMultivariateNormalPdf:
    def test_2d_standard(self):
        mu = np.array([0.0, 0.0])
        Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
        x = np.array([0.0, 0.0])
        pdf_val = multivariate_normal_pdf(x, mu, Sigma)
        expected = 1 / (2 * math.pi)
        assert abs(pdf_val - expected) < 1e-10

    def test_positive(self):
        mu = np.array([0.0, 0.0])
        Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
        x = np.array([1.0, 0.0])
        assert multivariate_normal_pdf(x, mu, Sigma) > 0


class TestBoxMuller:
    def test_range(self):
        z0, z1 = box_muller(0.5, 0.5)
        assert isinstance(z0, float)
        assert isinstance(z1, float)

    def test_not_both_zero(self):
        z0, z1 = box_muller(0.5, 0.5)
        assert not (z0 == 0 and z1 == 0)


class TestMarsagliaBray:
    def test_range(self):
        z0, z1 = marsaglia_bray()
        assert isinstance(z0, float)
        assert isinstance(z1, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
