"""Tests for quadratic reciprocity module."""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.quadratic import (
    legendre_symbol, legendre_symbol_gauss, jacobi_symbol,
    kronecker_symbol, quadratic_reciprocity, quadratic_residues,
    quadratic_nonresidues, tonelli_shanks, cipolla,
    solve_quadratic_congruence
)


class TestLegendreSymbol:
    def test_euler_criterion(self):
        for p in [7, 11, 13, 17, 19]:
            for a in range(1, p):
                expected = 1 if pow(a, (p-1)//2, p) == 1 else -1
                assert legendre_symbol(a, p) == expected, f"a={a}, p={p}"
    
    def test_gauss_lemma(self):
        for p in [7, 11, 13, 17]:
            for a in range(1, p):
                euler = legendre_symbol(a, p)
                gauss = legendre_symbol_gauss(a, p)
                assert euler == gauss, f"a={a}, p={p}: {euler} vs {gauss}"
    
    def test_zero(self):
        assert legendre_symbol(0, 7) == 0
        assert legendre_symbol(11, 11) == 0
    
    def test_known_values(self):
        # mod 7: residues are 1,2,4; non-residues 3,5,6
        assert legendre_symbol(1, 7) == 1
        assert legendre_symbol(2, 7) == 1
        assert legendre_symbol(3, 7) == -1
        assert legendre_symbol(4, 7) == 1
        assert legendre_symbol(5, 7) == -1
        assert legendre_symbol(6, 7) == -1


class TestJacobiSymbol:
    def test_basic(self):
        # (12/35) = (12/5)(12/7) = (2/5)(5/7) = (-1)(-1) = 1
        assert jacobi_symbol(12, 35) == 1
    
    def test_prime_modulus(self):
        # Jacobi symbol equals Legendre symbol for prime modulus
        for p in [7, 11, 13]:
            for a in range(p):
                assert jacobi_symbol(a, p) == legendre_symbol(a, p), f"a={a}, p={p}"
    
    def test_even_numerator(self):
        # (2/n) = (-1)^((n^2-1)/8)
        for n in [3, 5, 7, 9, 11, 13]:
            expected = -1 if (n * n - 1) // 8 % 2 else 1
            assert jacobi_symbol(2, n) == expected, f"n={n}"


class TestKroneckerSymbol:
    def test_odd_denominator(self):
        # For odd n, Kronecker = Jacobi
        for n in [3, 5, 7, 9, 11]:
            for a in range(n):
                assert kronecker_symbol(a, n) == jacobi_symbol(a, n), f"a={a}, n={n}"
    
    def test_even_denominator(self):
        # (a/2) = 0 if a even, 1 if a ≡ ±1 (mod 8), -1 if a ≡ ±3 (mod 8)
        assert kronecker_symbol(1, 2) == 1
        assert kronecker_symbol(3, 2) == -1
        assert kronecker_symbol(5, 2) == -1
        assert kronecker_symbol(7, 2) == 1
        assert kronecker_symbol(2, 2) == 0
    
    def test_negative_denominator(self):
        # (a/-n) = -(a/n) for a > 0
        for n in [3, 5, 7]:
            for a in range(1, n):
                assert kronecker_symbol(a, -n) == kronecker_symbol(a, n), f"a={a}, n={n}"


class TestQuadraticReciprocity:
    def test_pairs(self):
        for p, q in [(3, 5), (3, 7), (5, 13), (11, 17), (7, 11), (13, 17)]:
            lhs, rhs = quadratic_reciprocity(p, q)
            assert lhs == rhs, f"({p}/{q})*({q}/{p}) = {lhs} != {rhs}"
    
    def test_both_3_mod_4(self):
        # If both ≡ 3 (mod 4), product = -1
        p, q = 3, 7
        lhs, rhs = quadratic_reciprocity(p, q)
        assert lhs == -1
        assert rhs == -1


class TestQuadraticResidues:
    def test_prime_7(self):
        res = quadratic_residues(7)
        assert res == [1, 2, 4]
        nonres = quadratic_nonresidues(7)
        assert nonres == [3, 5, 6]
    
    def test_prime_11(self):
        res = quadratic_residues(11)
        assert len(res) == 5  # (11-1)/2 = 5
        assert set(res) == {1, 3, 4, 5, 9}
    
    def test_prime_2(self):
        res = quadratic_residues(2)
        assert res == [1]


class TestTonelliShanks:
    def test_simple(self):
        # sqrt(4) mod 11 = 2 or 9
        r = tonelli_shanks(4, 11)
        assert r in [2, 9]
        assert (r * r) % 11 == 4
    
    def test_p_3_mod_4(self):
        p = 19
        for a in range(1, p):
            if legendre_symbol(a, p) == 1:
                r = tonelli_shanks(a, p)
                assert r is not None
                assert (r * r) % p == a % p
    
    def test_p_1_mod_4(self):
        p = 13
        for a in range(1, p):
            if legendre_symbol(a, p) == 1:
                r = tonelli_shanks(a, p)
                assert r is not None
                assert (r * r) % p == a % p
    
    def test_non_residue(self):
        with pytest.raises(ValueError):
            tonelli_shanks(2, 11)
    
    def test_p_2(self):
        assert tonelli_shanks(1, 2) == 1
        assert tonelli_shanks(0, 2) == 0


class TestCipolla:
    def test_p_3_mod_4(self):
        p = 11
        for a in range(1, p):
            if legendre_symbol(a, p) == 1:
                r = cipolla(a, p)
                assert (r * r) % p == a % p
    
    def test_p_1_mod_4(self):
        p = 13
        for a in range(1, p):
            if legendre_symbol(a, p) == 1:
                r = cipolla(a, p)
                assert (r * r) % p == a % p
    
    def test_non_residue_raises(self):
        with pytest.raises(ValueError):
            cipolla(2, 11)  # 2 is non-residue mod 11


class TestSolveQuadraticCongruence:
    def test_linear(self):
        # 2x + 3 ≡ 0 (mod 11) => 2x ≡ -3 ≡ 8 => x ≡ 4
        sols = solve_quadratic_congruence(0, 2, 3, 11)
        assert sols == [4]
    
    def test_quadratic(self):
        # 2x^2 + 3x + 1 ≡ 0 (mod 11)
        sols = solve_quadratic_congruence(2, 3, 1, 11)
        for x in sols:
            assert (2*x*x + 3*x + 1) % 11 == 0
    
    def test_double_root(self):
        # x^2 ≡ 0 (mod 7) => x ≡ 0
        sols = solve_quadratic_congruence(1, 0, 0, 7)
        assert sols == [0]
    
    def test_no_solution(self):
        # x^2 ≡ 2 (mod 11) where 2 is non-residue
        sols = solve_quadratic_congruence(1, 0, -2, 11)
        assert sols == []
    
    def test_mod_2(self):
        # x^2 + x + 1 ≡ 0 (mod 2)
        sols = solve_quadratic_congruence(1, 1, 1, 2)
        # x=0: 1 ≡ 1, x=1: 1+1+1=3≡1 => no solutions
        assert sols == []


class TestIntegration:
    def test_all_primes_under_20(self):
        for p in [3, 5, 7, 11, 13, 17, 19]:
            res = quadratic_residues(p)
            nonres = quadratic_nonresidues(p)
            assert len(res) == (p - 1) // 2
            assert len(nonres) == (p - 1) // 2
            assert set(res).isdisjoint(nonres)
            assert set(res) | set(nonres) == set(range(1, p))
    
    def test_square_root_consistency(self):
        for p in [11, 13, 17, 19]:
            for a in range(1, p):
                if legendre_symbol(a, p) == 1:
                    r1 = tonelli_shanks(a, p)
                    r2 = cipolla(a, p)
                    assert r1 is not None
                    assert r2 is not None
                    assert (r1 * r1) % p == a
                    assert (r2 * r2) % p == a
                    # Both should be valid square roots (either same or negatives)
                    assert (r1 - r2) % p == 0 or (r1 + r2) % p == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])