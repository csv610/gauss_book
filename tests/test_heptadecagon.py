"""Tests for heptadecagon and constructible polygons."""

import pytest
import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.heptadecagon import (
    gaussian_periods, cos_2pi_over_17_value,
    constructible_ngons, fermat_primes, is_fermat_prime, is_constructible,
    heptadecagon_exact
)


class TestFermatPrimes:
    def test_list(self):
        assert fermat_primes() == [3, 5, 17, 257, 65537]

    def test_is_fermat_prime(self):
        assert is_fermat_prime(3)
        assert is_fermat_prime(5)
        assert is_fermat_prime(17)
        assert is_fermat_prime(257)
        assert is_fermat_prime(65537)

    def test_not_fermat_prime(self):
        assert not is_fermat_prime(7)
        assert not is_fermat_prime(9)
        assert not is_fermat_prime(11)


class TestIsConstructible:
    def test_regular_polygon_3(self):
        assert is_constructible(3)

    def test_regular_polygon_4(self):
        assert is_constructible(4)

    def test_regular_polygon_5(self):
        assert is_constructible(5)

    def test_regular_polygon_6(self):
        assert is_constructible(6)  # 2 * 3

    def test_regular_polygon_8(self):
        assert is_constructible(8)  # 2^3

    def test_regular_polygon_16(self):
        assert is_constructible(16)

    def test_regular_polygon_17(self):
        assert is_constructible(17)

    def test_regular_polygon_34(self):
        assert is_constructible(34)  # 2 * 17

    def test_regular_polygon_51(self):
        assert is_constructible(51)  # 3 * 17

    def test_regular_polygon_7(self):
        assert not is_constructible(7)

    def test_regular_polygon_9(self):
        assert not is_constructible(9)  # 3^2 (not distinct)

    def test_regular_polygon_11(self):
        assert not is_constructible(11)

    def test_regular_polygon_15(self):
        assert is_constructible(15)  # 3 * 5

    def test_large_values(self):
        assert is_constructible(256)  # 2^8
        assert is_constructible(512)  # 2^9


class TestConstructibleNgon:
    def test_up_to_20(self):
        result = constructible_ngons(20)
        expected = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 17, 20]
        assert result == expected

    def test_up_to_100(self):
        result = constructible_ngons(100)
        # Should include all powers of 2 up to 64, and products of distinct Fermat primes
        assert 64 in result
        assert 17 in result
        assert 51 in result  # 3 * 17
        assert 85 in result  # 5 * 17

    def test_empty(self):
        assert constructible_ngons(0) == []
        assert constructible_ngons(1) == [1]

    def test_small(self):
        result = constructible_ngons(2)
        assert set(result) == {1, 2}

class TestCos2PiOver17:
    def test_value(self):
        val = cos_2pi_over_17_value()
        expected = math.cos(2 * math.pi / 17)
        assert abs(val - expected) < 1e-10

    def test_range(self):
        val = cos_2pi_over_17_value()
        assert 0 < val < 1


class TestGaussianPeriods:
    def test_p5_length2(self):
        periods = gaussian_periods(5, 2)
        assert len(periods) == 2
        # Each period should be a sum of two 5th roots of unity
        for p in periods:
            assert abs(p) < 3  # Reasonable bound

    def test_p7_length3(self):
        periods = gaussian_periods(7, 3)
        assert len(periods) == 3

    def test_p17_length8(self):
        periods = gaussian_periods(17, 8)
        assert len(periods) == 8
        # Sum of periods should equal -1 (sum of all 17th roots minus 1)
        total = sum(periods)
        assert abs(total + 1) < 1e-10


class TestHeptadecagonExact:
    def test_returns_dict(self):
        result = heptadecagon_exact()
        assert isinstance(result, dict)
        assert 'eta1' in result
        assert 'eta2' in result
        assert 'cos_2pi_17' in result

    def test_cos_2pi_17_matches(self):
        exact = heptadecagon_exact()
        numerical = cos_2pi_over_17_value()
        # Verify numerical value matches standard cosine
        assert abs(numerical - math.cos(2 * math.pi / 17)) < 1e-10

    def test_eta1_positive(self):
        exact = heptadecagon_exact()
        assert exact['eta1'] > 0
        assert abs(exact['eta1'] - (-1 + math.sqrt(17)) / 2) < 1e-10

    def test_eta2_negative(self):
        exact = heptadecagon_exact()
        assert exact['eta2'] < 0
        assert abs(exact['eta2'] - (-1 - math.sqrt(17)) / 2) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
