"""Tests for Gaussian quadrature module."""

import pytest
import sys
import math
import numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.gauss_quadrature import (
    gauss_legendre, gauss_hermite, gauss_laguerre, gauss_chebyshev,
    golub_welsch, integrate, integrate_infinite,
    integrate_adaptive, gauss_kronrod, legendre_recurrence
)


class TestGaussLegendre:
    def test_known_values(self):
        # n=1
        nodes, weights = gauss_legendre(1)
        assert abs(nodes[0] - 0) < 1e-10
        assert abs(weights[0] - 2) < 1e-10
        
        # n=2
        nodes, weights = gauss_legendre(2)
        expected_nodes = [-1/math.sqrt(3), 1/math.sqrt(3)]
        np.testing.assert_allclose(nodes, expected_nodes, rtol=1e-10)
        np.testing.assert_allclose(weights, [1, 1], rtol=1e-10)
    
    def test_exact_polynomials(self):
        for n in range(1, 8):
            nodes, weights = gauss_legendre(n)
            # Should integrate x^{2n-1} exactly
            for deg in range(2*n):
                exact = 2 / (deg + 1) if deg % 2 == 0 else 0
                approx = np.sum(weights * nodes**deg)
                assert abs(approx - exact) < 1e-12, f"n={n}, deg={deg}: {approx} vs {exact}"


class TestGaussHermite:
    def test_basic(self):
        nodes, weights = gauss_hermite(2)
        # Nodes should be ±1/√2
        expected = [-1/math.sqrt(2), 1/math.sqrt(2)]
        np.testing.assert_allclose(nodes, expected, rtol=1e-10)
        np.testing.assert_allclose(weights, [math.sqrt(math.pi)/2, math.sqrt(math.pi)/2], rtol=1e-10)
    
    def test_weight_function(self):
        # ∫ x^2 e^{-x^2} dx = √π/2
        for n in [2, 3, 4, 5]:
            nodes, weights = gauss_hermite(n)
            approx = np.sum(weights * nodes**2)
            exact = math.sqrt(math.pi) / 2
            assert abs(approx - exact) < 1e-8, f"n={n}: {approx} vs {exact}"


class TestGaussLaguerre:
    def test_basic(self):
        nodes, weights = gauss_laguerre(1)
        assert abs(nodes[0] - 1) < 1e-10
        assert abs(weights[0] - 1) < 1e-10
    
    def test_weight_function(self):
        # ∫ x e^{-x} dx = 1
        for n in [2, 3, 4]:
            nodes, weights = gauss_laguerre(n)
            approx = np.sum(weights * nodes)
            assert abs(approx - 1) < 1e-8


class TestGaussChebyshev:
    def test_kind1(self):
        nodes, weights = gauss_chebyshev(4, kind=1)
        # Nodes: cos((2k-1)π/(2n))
        expected = [math.cos((2*k-1)*math.pi/8) for k in range(1, 5)]
        np.testing.assert_allclose(nodes, expected, rtol=1e-10)
        # Weights: π/n
        assert all(abs(w - math.pi/4) < 1e-10 for w in weights)
    
    def test_kind2(self):
        nodes, weights = gauss_chebyshev(4, kind=2)
        # Nodes: cos(kπ/(n+1))
        expected = [math.cos(k*math.pi/5) for k in range(1, 5)]
        np.testing.assert_allclose(nodes, expected, rtol=1e-10)


class TestIntegration:
    def test_finite_interval(self):
        # ∫ x^8 dx from -1 to 1 = 2/9
        val = integrate(lambda x: x**8, -1, 1, n=5)
        assert abs(val - 2/9) < 1e-12
    
    def test_infinite_interval(self):
        # ∫ e^{-x^2} dx from -∞ to ∞ = √π
        val = integrate_infinite(lambda x: math.exp(-x**2), n=10)
        assert abs(val - math.sqrt(math.pi)) < 1e-8
    
    def test_general_interval(self):
        # ∫ x^2 dx from 0 to 2 = 8/3
        val = integrate(lambda x: x**2, 0, 2, n=5)
        assert abs(val - 8/3) < 1e-12


class TestAdaptiveIntegration:
    def test_gaussian(self):
        val, err = integrate_adaptive(lambda x: math.exp(-x**2), 0, 1)
        exact = math.sqrt(math.pi)/2 * math.erf(1)
        assert abs(val - exact) < 1e-10
        assert err < 1e-8
    
    def test_singular(self):
        # ∫ 1/√x dx from 0 to 1 = 2
        val, err = integrate_adaptive(lambda x: 1/math.sqrt(x), 0, 1)
        assert abs(val - 2) < 1e-4


class TestGolubWelsch:
    def test_legendre_reproduction(self):
        alpha, beta = legendre_recurrence(5)
        beta[0] = 2.0
        nodes, weights = golub_welsch(alpha, beta)
        # Compare with direct gauss_legendre
        nodes2, weights2 = gauss_legendre(5)
        np.testing.assert_allclose(nodes, nodes2, rtol=1e-10)
        np.testing.assert_allclose(weights, weights2, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])