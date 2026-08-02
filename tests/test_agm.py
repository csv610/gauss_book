"""Tests for AGM module."""

import pytest
import sys
import math
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.agm import (
    agm, agm_sequence, gauss_constant,
    elliptic_K, elliptic_E, elliptic_Pi, ellippi,
    pi_brent_salamin, pi_borwein_quartic,
    lemniscate_constant, elliptic_perimeter,
    hypergeometric_2F1, agm_hypergeometric
)


class TestAGM:
    def test_basic(self):
        result = agm(1, 0.5)
        assert abs(result - 0.728395515523) < 1e-10
    
    def test_sequence(self):
        seq = agm_sequence(1, 0.5)
        assert len(seq) > 1
        # Check quadratic convergence
        for i in range(1, len(seq)-1):
            a, b = seq[i]
            diff = abs(a - b)
            assert diff < seq[i-1][0] - seq[i-1][1]
    
    def test_gauss_constant(self):
        G = gauss_constant()
        expected = 0.83462684167
        assert abs(G - expected) < 1e-8
        # Verify G = 1/M(1, sqrt(2))
        assert abs(G - 1/agm(1, math.sqrt(2))) < 1e-10


class TestEllipticIntegrals:
    def test_K(self):
        # K(0) = π/2
        assert abs(elliptic_K(0) - math.pi/2) < 1e-10
        # K(1/sqrt(2)) = Γ(1/4)²/(4√π)
        k = 1/math.sqrt(2)
        K = elliptic_K(k)
        expected = math.gamma(0.25)**2 / (4 * math.sqrt(math.pi))
        assert abs(K - expected) < 1e-8
    
    def test_E(self):
        # E(0) = π/2
        assert abs(elliptic_E(0) - math.pi/2) < 1e-10
        # E(1) = 1
        assert abs(elliptic_E(0.9999) - 1) < 1e-3


class TestEllipticThirdKind:
    def test_Pi_zero_n_equals_K(self):
        # Π(0, k) = K(k) — third kind reduces to first kind
        for k in [0.0, 0.3, 0.5, 0.7, 0.99]:
            assert abs(ellippi(0, k) - elliptic_K(k)) < 1e-8

    def test_Pi_zero_k(self):
        # Π(n, 0) = π / (2√(1-n)) — when modulus is zero
        for n in [-1.0, -0.5, 0.0, 0.3, 0.5, 0.9]:
            expected = math.pi / (2 * math.sqrt(1 - n))
            assert abs(ellippi(n, 0) - expected) < 1e-10

    def test_Pi_known_value(self):
        # Π(0.5, 0.5) ≈ 2.41367150420119 (verified via scipy.integrate.quad)
        result = ellippi(0.5, 0.5)
        assert abs(result - 2.41367150420119) < 1e-6

    def test_Pi_vs_wrapper(self):
        # elliptic_Pi delegates to ellippi
        for n, k in [(0.1, 0.3), (0.5, 0.7), (-0.2, 0.9)]:
            assert abs(elliptic_Pi(n, k) - ellippi(n, k)) < 1e-15

    def test_Pi_symmetry(self):
        # Π(n,k) should be real and smooth for |k| < 1, n < 1
        # Just verify smooth behavior across a range
        prev = ellippi(0.1, 0.1)
        for k in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            val = ellippi(0.1, k)
            assert val > 0
            # Π(n,k) increases with k for fixed n > 0
            assert val > prev
            prev = val


class TestPiAlgorithms:
    def test_brent_salamin(self):
        # Convergence is quadratic: check proper tolerance per iteration
        tolerances = {1: 0.4, 2: 1e-4, 3: 1e-8, 4: 1e-12, 5: 1e-12}
        for n in [1, 2, 3, 4, 5]:
            pi_approx = pi_brent_salamin(n)
            assert abs(pi_approx - math.pi) < tolerances[n]
    
    def test_borwein_quartic(self):
        # Convergence is quartic: check proper tolerance per iteration
        tolerances = {1: 1e-8, 2: 1e-12, 3: 1e-12, 4: 1e-12}
        for n in [1, 2, 3, 4]:
            pi_approx = pi_borwein_quartic(n)
            assert abs(pi_approx - math.pi) < tolerances[n]


class TestApplications:
    def test_lemniscate_constant(self):
        L = lemniscate_constant()
        G = gauss_constant()
        assert abs(L - math.pi*G) < 1e-10
    
    def test_ellipse_perimeter(self):
        # Circle: a=b=1 => P = 2π
        assert abs(elliptic_perimeter(1, 1) - 2*math.pi) < 1e-10
        # Ellipse with a=2, b=1
        P = elliptic_perimeter(2, 1)
        # Approximate value
        assert 9.5 < P < 9.8


class TestHypergeometric:
    def test_2F1(self):
        # ₂F₁(1,1;2;z) = -ln(1-z)/z
        for z in [0.1, 0.5, 0.9]:
            result = hypergeometric_2F1(1, 1, 2, z)
            expected = -math.log(1-z)/z
            assert abs(result - expected) < 1e-8
    
    def test_agm_hypergeometric(self):
        a, b = 1, 0.5
        agm_val = agm(a, b)
        hyp_val = agm_hypergeometric(a, b)
        assert abs(agm_val - hyp_val) < 1e-8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])