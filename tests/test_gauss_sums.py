"""Tests for Gauss sums and Dirichlet characters."""

import pytest
import sys
import os
import math
import cmath
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.gauss_sums import (
    quadratic_gauss_sum, gauss_sum_sign, verify_gauss_sum,
    dirichlet_characters, general_gauss_sum, primitive_character,
    is_primitive_root, discrete_log, prime_factors,
    dirichlet_l_function, polya_vinogradov,
    quadratic_reciprocity_gauss_sum, legendre_symbol, fft_gauss_sum
)


class TestQuadraticGaussSum:
    def test_g3(self):
        g = quadratic_gauss_sum(3)
        expected = 1j * math.sqrt(3)
        assert abs(g - expected) < 1e-10

    def test_g5(self):
        g = quadratic_gauss_sum(5)
        expected = math.sqrt(5)
        assert abs(g - expected) < 1e-10

    def test_g7(self):
        g = quadratic_gauss_sum(7)
        expected = 1j * math.sqrt(7)
        assert abs(g - expected) < 1e-10

    def test_g11(self):
        g = quadratic_gauss_sum(11)
        expected = 1j * math.sqrt(11)
        assert abs(g - expected) < 1e-10

    def test_g13(self):
        g = quadratic_gauss_sum(13)
        expected = math.sqrt(13)
        assert abs(g - expected) < 1e-10

    def test_g2(self):
        g = quadratic_gauss_sum(2)
        assert abs(g - (1 + 1j)) < 1e-10


class TestGaussSumSign:
    def test_p_1_mod_4(self):
        assert gauss_sum_sign(5) == math.sqrt(5)
        assert gauss_sum_sign(13) == math.sqrt(13)

    def test_p_3_mod_4(self):
        assert gauss_sum_sign(3) == 1j * math.sqrt(3)
        assert gauss_sum_sign(7) == 1j * math.sqrt(7)


class TestVerifyGaussSum:
    def test_p3(self):
        direct, theory, error = verify_gauss_sum(3)
        assert error < 1e-10

    def test_p5(self):
        direct, theory, error = verify_gauss_sum(5)
        assert error < 1e-10

    def test_p17(self):
        direct, theory, error = verify_gauss_sum(17)
        assert error < 1e-10


class TestLegendreSymbol:
    def test_quadratic_residues_mod11(self):
        # Quadratic residues mod 11: 1, 3, 4, 5, 9
        residues = {1, 3, 4, 5, 9}
        for a in range(1, 11):
            expected = 1 if a in residues else -1
            assert legendre_symbol(a, 11) == expected

    def test_zero(self):
        assert legendre_symbol(0, 7) == 0

    def test_p_divides(self):
        assert legendre_symbol(11, 11) == 0


class TestPrimitiveRoot:
    def test_primitive_root_5(self):
        assert is_primitive_root(2, 5)
        assert is_primitive_root(3, 5)
        assert not is_primitive_root(1, 5)

    def test_primitive_root_7(self):
        # Primitive roots mod 7: 3, 5
        assert is_primitive_root(3, 7)
        assert is_primitive_root(5, 7)
        assert not is_primitive_root(1, 7)
        assert not is_primitive_root(2, 7)


class TestDirichletCharacters:
    def test_mod_3(self):
        chars = dirichlet_characters(3)
        assert len(chars) == 2  # φ(3) = 2

    def test_mod_5(self):
        chars = dirichlet_characters(5)
        assert len(chars) == 4  # φ(5) = 4

    def test_mod_7(self):
        chars = dirichlet_characters(7)
        assert len(chars) == 6  # φ(7) = 6

    def test_trivial_character(self):
        chars = dirichlet_characters(5)
        # First character should be trivial
        trivial = chars[0]
        assert abs(trivial[1] - 1) < 1e-10
        assert abs(trivial[2] - 1) < 1e-10
        assert abs(trivial[3] - 1) < 1e-10
        assert abs(trivial[4] - 1) < 1e-10


class TestGeneralGaussSum:
    def test_trivial_character(self):
        # Trivial character mod 4: chi(0)=0, chi(1)=1, chi(2)=0, chi(3)=1
        chi = [0, 1, 0, 1]  # Correct trivial character (zero on non-units)
        g = general_gauss_sum(chi, 4)
        # g = sum_{a=0}^{3} chi(a) * exp(2*pi*i*a/4) = 0 + 1*i + 0 + 1*(-i) = 0
        assert abs(g) < 1e-10


class TestPolyaVinogradov:
    def test_bound_positive(self):
        chi = [0, 1, 1j, -1j, -1]  # mod 5
        bound = polya_vinogradov(chi, 100)
        assert bound > 0
        assert bound == math.sqrt(5) * math.log(5)


class TestPrimeFactors:
    def test_factorization(self):
        assert prime_factors(12) == [2, 3]
        assert prime_factors(15) == [3, 5]
        assert prime_factors(7) == [7]
        assert prime_factors(1) == []


class TestIntegration:
    def test_quadratic_reciprocity_p5_q11(self):
        # (5/11) * (11/5) = (-1)^((5-1)(11-1)/4) = (-1)^10 = 1
        # So (5/11) = (11/5) = (1/5) = 1
        g5 = quadratic_gauss_sum(5)
        g11 = quadratic_gauss_sum(11)
        # |g(p)|^2 = p
        assert abs(abs(g5) ** 2 - 5) < 1e-6
        assert abs(abs(g11) ** 2 - 11) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
