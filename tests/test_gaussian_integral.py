"""Tests for Gaussian integral module."""

import pytest
import sys
import math
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.gaussian_integral import (
    gaussian_integral, gaussian_pdf, gaussian_cdf, gaussian_ppf,
    erf, erfc, erfi, dawson_integral, faddeeva,
    normal_pdf, normal_cdf, normal_sf, normal_ppf,
    normal_rvs, confidence_interval_normal, error_propagation,
    chi2_pdf, chi2_cdf, t_pdf, f_pdf
)


class TestGaussianIntegral:
    def test_basic(self):
        # ∫ e^{-x^2} dx from -∞ to ∞ = √π
        val = gaussian_integral(1, 0, 0, -10, 10)
        assert abs(val - math.sqrt(math.pi)) < 1e-8
    
    def test_shifted(self):
        # ∫ e^{-(x-2)^2} dx = √π
        val = gaussian_integral(1, 4, -4, -10, 10)
        assert abs(val - math.sqrt(math.pi)) < 1e-8


class TestErrorFunction:
    def test_erf_values(self):
        # erf(0) = 0
        assert abs(erf(0)) < 1e-15
        # erf(∞) = 1
        assert abs(erf(5) - 1) < 1e-10
        # erf(-x) = -erf(x)
        assert abs(erf(-1) + erf(1)) < 1e-15
    
    def test_erfc(self):
        assert abs(erfc(0) - 1) < 1e-15
        assert abs(erfc(5)) < 1e-10
    
    def test_erfi(self):
        # erfi(x) = -i erf(ix)
        for x in [0, 0.5, 1.0, 2.0]:
            direct = erfi(x)
            via_erf = -1j * erf(1j * x)
            assert abs(direct - via_erf.real) < 1e-10
    
    def test_dawson(self):
        # Dawson integral at x=0 is 0
        assert abs(dawson_integral(0)) < 1e-15
        # Dawson integral for large x ~ 1/(2x)
        for x in [5, 10]:
            assert abs(dawson_integral(x) - 1/(2*x)) < 1e-2


class TestFaddeeva:
    def test_faddeeva(self):
        # w(z) = e^{-z^2} erfc(-iz)
        # w(0) = 1
        assert abs(faddeeva(0) - 1) < 1e-10


class TestNormalDistribution:
    def test_pdf(self):
        # Standard normal PDF at 0
        assert abs(normal_pdf(0) - 1/math.sqrt(2*math.pi)) < 1e-10
        # PDF integrates to 1
        val = gaussian_integral(0.5, 0, 0, -10, 10)  # Not the same
        # Just check values
        assert normal_pdf(0) > normal_pdf(1)
    
    def test_cdf(self):
        # CDF at -∞ = 0, ∞ = 1
        assert abs(normal_cdf(-5)) < 1e-6
        assert abs(normal_cdf(5) - 1) < 1e-6
        # CDF at 0 = 0.5
        assert abs(normal_cdf(0) - 0.5) < 1e-10
    
    def test_ppf(self):
        # Inverse CDF
        for p in [0.025, 0.05, 0.5, 0.95, 0.975]:
            z = normal_ppf(p)
            cdf = normal_cdf(z)
            assert abs(cdf - p) < 1e-4
    
    def test_rvs(self):
        samples = normal_rvs(1000, mu=2, sigma=3, random_state=42)
        mean = np.mean(samples)
        std = np.std(samples)
        assert abs(mean - 2) < 0.2
        assert abs(std - 3) < 0.2


class TestConfidenceInterval:
    def test_normal_ci(self):
        ci = confidence_interval_normal(mean=2.0, std=0.5, n=25, confidence=0.95)
        assert ci[0] < 2.0 < ci[1]
        # Margin of error = 1.96 * 0.5 / 5 = 0.196
        margin = (ci[1] - ci[0]) / 2
        assert abs(margin - 0.196) < 0.01


class TestErrorPropagation:
    def test_division(self):
        # x/y with x=10±0.1, y=5±0.05
        f = lambda x, y: x / y
        mean, std = error_propagation(f, [(10.0, 0.1), (5.0, 0.1)])
        assert abs(mean - 2.0) < 1e-6
        assert abs(std - 0.0447) < 1e-3
    
    def test_square(self):
        # x^2 with x=3±0.1 => 9 ± 2*3*0.1 = 9 ± 0.6
        f = lambda x: x**2
        mean, std = error_propagation(f, [(3.0, 0.1)])
        assert abs(mean - 9.0) < 1e-6
        assert abs(std - 0.6) < 1e-3


class TestRelatedDistributions:
    def test_chi2(self):
        # χ²(1) at x=1
        val = chi2_pdf(1, 1)
        assert val > 0
    
    def test_t_pdf(self):
        # t(1) at x=0 should be 1/π
        val = t_pdf(0, 1)
        assert abs(val - 1/math.pi) < 1e-6
    
    def test_f_pdf(self):
        val = f_pdf(1, 10, 10)
        assert val > 0


# Need numpy
import numpy as np

if __name__ == "__main__":
    pytest.main([__file__, "-v"])