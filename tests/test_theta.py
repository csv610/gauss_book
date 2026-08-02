"""Tests for theta functions module."""

import pytest
import sys
import cmath
import math
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.theta import (
    theta1, theta2, theta3, theta4, theta_null,
    dedekind_eta, sum_of_squares, jacobi_triple_product,
    eisenstein_series, j_invariant
)


class TestThetaFunctions:
    def test_theta1(self):
        # θ₁(0|τ) = 0
        assert abs(theta1(0, 1j, 50)) < 1e-10
    
    def test_theta3_null(self):
        # θ₃(0|i) ≈ 1.086435
        val = theta3(0, 1j, 50)
        expected = 1.086435
        assert abs(val - expected) < 1e-5
    
    def test_theta_null_tuple(self):
        t1, t2, t3, t4 = theta_null(1j, 50)
        # Known values
        assert abs(t1) < 1e-10
        assert abs(t3 - 1.086435) < 1e-5
        assert abs(t2 - t4) < 1e-5  # For τ=i, θ₂ = θ₄


class TestDedekindEta:
    def test_eta_i(self):
        # η(i) = Γ(1/4) / (2π^{3/4}) ≈ 0.768225
        val = dedekind_eta(1j, 100)
        expected = math.gamma(0.25) / (2 * math.pi**0.75)
        assert abs(val - expected) < 1e-6
    
    def test_eta_2i(self):
        val = dedekind_eta(2j, 100)
        expected = 0.592383
        assert abs(val - expected) < 1e-6


class TestSumsOfSquares:
    def test_sum_of_4_squares(self):
        r4 = sum_of_squares(4, 20)
        # r_4(n) = 8 * sum_{d|n, 4∤d} d
        expected = {
            1: 8,   # 8*1
            2: 24,  # 8*(1+2)
            3: 32,  # 8*(1+3)
            4: 24,  # 8*(1+2) (4 not included)
            5: 48,  # 8*(1+5)
        }
        for n, v in expected.items():
            assert r4[n] == v, f"r_4({n}) = {r4[n]}, expected {v}"
    
    def test_sum_of_2_squares(self):
        r2 = sum_of_squares(2, 20)
        # r_2(n) = 4 * (d_1(n) - d_3(n)) where d_i = divisors ≡ i (mod 4)
        expected = {
            1: 4,   # (±1,0),(0,±1)
            2: 4,   # (±1,±1)
            3: 0,   # 3 not sum of 2 squares
            4: 4,   # (±2,0),(0,±2)
            5: 8,   # (±1,±2),(±2,±1)
        }
        for n, v in expected.items():
            assert r2[n] == v, f"r_2({n}) = {r2[n]}, expected {v}"


class TestModularForms:
    def test_eisenstein_series(self):
        # G₄(i) ≈ 3.151212
        G4 = eisenstein_series(4, 1j, 50)
        expected = 3.151212
        assert abs(G4 - expected) < 1e-3
    
    def test_j_invariant(self):
        # j(i) = 1728
        j = j_invariant(1j, 50)
        assert abs(j - 1728) < 1


class TestJacobiTripleProduct:
    def test_identity(self):
        q = cmath.exp(1j * math.pi * 1j)  # q = e^{-π}
        z = 0.5
        lhs = theta1(z, 1j, 50)
        rhs = jacobi_triple_product(q, z, 50)
        assert abs(lhs - rhs) < 1e-8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])