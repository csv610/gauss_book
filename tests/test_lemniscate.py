"""Tests for lemniscate functions using standard definitions (K = quarter period)."""

import pytest
import sys
import math
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.lemniscate import (
    sl, cl, sl_add, cl_add, lemniscate_division,
    K, GAUSS_CONSTANT, lemniscate_integral, lemniscate_integral_inverse,
    pythagorean_check
)


class TestLemniscateConstants:
    def test_constants(self):
        # K = π*G/2 (quarter period = π*G/2 where G is Gauss constant)
        assert abs(K - math.pi * GAUSS_CONSTANT / 2) < 1e-10
        assert abs(GAUSS_CONSTANT - 0.8346268416740732) < 1e-10


class TestLemniscateIntegral:
    def test_zero(self):
        assert abs(lemniscate_integral(0)) < 1e-10
    
    def test_one(self):
        # ∫₀¹ dt/√(1-t⁴) = K (quarter period)
        assert abs(lemniscate_integral(1) - K) < 1e-6
    
    def test_inverse(self):
        # sl(ω) = u where ∫₀ᵘ dt/√(1-t⁴) = ω
        u = 0.5
        omega = lemniscate_integral(u)
        u_recovered = lemniscate_integral_inverse(omega)
        assert abs(u - u_recovered) < 1e-10


class TestSlCl:
    def test_sl_zero(self):
        assert abs(sl(0)) < 1e-10
    
    def test_sl_K(self):
        # sl(K) = 1
        assert abs(sl(K) - 1) < 1e-6
    
    def test_sl_2K(self):
        # sl(2K) = 0
        assert abs(sl(2*K)) < 1e-6
    
    def test_sl_3K(self):
        # sl(3K) = -1
        assert abs(sl(3*K) + 1) < 1e-6
    
    def test_sl_4K(self):
        # sl(4K) = 0
        assert abs(sl(4*K)) < 1e-6
    
    def test_cl_zero(self):
        # cl(0) = 1
        assert abs(cl(0) - 1) < 1e-10
    
    def test_cl_K(self):
        # cl(K) = 0
        assert abs(cl(K)) < 1e-6
    
    def test_cl_2K(self):
        # cl(2K) = -1
        assert abs(cl(2*K) + 1) < 1e-6
    
    def test_sl_half_K(self):
        # sl(K/2) = √(√2 - 1) ≈ 0.64359
        expected = math.sqrt(math.sqrt(2) - 1)
        assert abs(sl(K/2) - expected) < 1e-6
    
    def test_cl_half_K(self):
        # cl(K/2) = sl(K/2) = √(√2 - 1)
        expected = math.sqrt(math.sqrt(2) - 1)
        assert abs(cl(K/2) - expected) < 1e-6


class TestPythagoreanIdentity:
    def test_identity(self):
        for omega in [0.1, 0.5, 1.0, 1.5, 2.0, K/2, K, 2*K, 3*K, 4*K]:
            s = sl(omega)
            c = cl(omega)
            lhs = s*s + c*c + s*s*c*c
            assert abs(lhs - 1) < 1e-5, f"Failed at ω={omega}: {lhs}"


class TestAdditionFormula:
    def test_sl_addition(self):
        for u, v in [(0.3, 0.5), (0.3, 1.2), (0.2, 0.8)]:
            s_uv = sl(u + v)
            s_add = sl_add(u, v)
            assert abs(s_uv - s_add) < 1e-5, f"Failed for u={u}, v={v}: {s_uv} vs {s_add}"
    
    def test_cl_addition(self):
        for u, v in [(0.3, 0.5), (0.3, 1.2), (0.2, 0.8)]:
            c_uv = cl(u + v)
            c_add = cl_add(u, v)
            assert abs(c_uv - c_add) < 1e-5, f"Failed for u={u}, v={v}: {c_uv} vs {c_add}"


class TestPeriodicity:
    def test_real_period(self):
        for omega in [0.5, 1.0, 1.5]:
            assert abs(sl(omega + 4*K) - sl(omega)) < 1e-5
            assert abs(cl(omega + 4*K) - cl(omega)) < 1e-5
    
    def test_antiperiod(self):
        for omega in [0.5, 1.0]:
            assert abs(sl(omega + 2*K) + sl(omega)) < 1e-5
            assert abs(cl(omega + 2*K) + cl(omega)) < 1e-5


class TestDivision:
    def test_constructible(self):
        # 3, 5, 17 should work (odd only)
        for n in [3, 5, 17]:
            points = lemniscate_division(n)
            assert len(points) == n
            # First point should be sl(K/(2n))
            assert abs(points[0] - sl(K/(2*n))) < 1e-6
    
    def test_non_constructible_raises(self):
        # Only even n should raise ValueError
        for n in [2, 4, 6, 8, 10, 12, 14, 16]:
            with pytest.raises(ValueError):
                lemniscate_division(n)


class TestSymmetry:
    def test_odd_even(self):
        for omega in [0.5, 1.0, 1.5]:
            assert abs(sl(-omega) + sl(omega)) < 1e-10
            assert abs(cl(-omega) - cl(omega)) < 1e-10
    
    def test_complementary(self):
        # cl(ω) = sl(K - ω)
        for omega in [0.5, 1.0, 1.5]:
            assert abs(cl(omega) - sl(K - omega)) < 1e-10
            # sl(ω) = cl(K - ω)
            assert abs(sl(omega) - cl(K - omega)) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])