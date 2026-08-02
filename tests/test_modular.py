"""Tests for modular arithmetic module."""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.modular import (
    mod_inv, chinese_remainder, primitive_root, discrete_log_gauss,
    euler_totient, multiplicative_order, all_primitive_roots,
    legendre_symbol, tonelli_shanks
)


class TestModularInverse:
    def test_basic_inverse(self):
        assert mod_inv(3, 11) == 4  # 3*4 = 12 ≡ 1 (mod 11)
        assert mod_inv(7, 11) == 8  # 7*8 = 56 ≡ 1 (mod 11)
    
    def test_self_inverse(self):
        assert mod_inv(1, 7) == 1
        assert mod_inv(6, 7) == 6  # 6*6 = 36 ≡ 1 (mod 7)
    
    def test_no_inverse_raises(self):
        with pytest.raises(ValueError):
            mod_inv(2, 4)  # gcd(2,4) = 2 ≠ 1
        with pytest.raises(ValueError):
            mod_inv(6, 9)  # gcd(6,9) = 3 ≠ 1


class TestChineseRemainder:
    def test_classic_example(self):
        # x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7) => x = 23
        assert chinese_remainder([2, 3, 2], [3, 5, 7]) == 23
    
    def test_two_congruences(self):
        assert chinese_remainder([1, 2], [2, 3]) == 5
    
    def test_large_moduli(self):
        x = chinese_remainder([1, 2], [997, 991])
        assert x % 997 == 1
        assert x % 991 == 2
    
    def test_non_coprime_raises(self):
        with pytest.raises(ValueError):
            chinese_remainder([1, 2], [4, 6])  # gcd(4,6) = 2
    
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            chinese_remainder([1, 2, 3], [2, 3])


class TestPrimitiveRoot:
    def test_prime_2(self):
        assert primitive_root(2) == 1
    
    def test_prime_3(self):
        assert primitive_root(3) in [2]
    
    def test_prime_5(self):
        assert primitive_root(5) in [2, 3]
    
    def test_prime_17(self):
        g = primitive_root(17)
        assert g in [3, 5, 6, 7, 10, 11, 12, 14]
    
    def test_all_primitive_roots(self):
        p = 11
        roots = all_primitive_roots(p)
        assert len(roots) == 4  # φ(10) = 4
        assert set(roots) == {2, 6, 7, 8}
    
    def test_primitive_root_property(self):
        p = 19
        g = primitive_root(p)
        generated = {pow(g, k, p) for k in range(1, p)}
        assert generated == set(range(1, p))


class TestDiscreteLog:
    def test_basic(self):
        x = discrete_log_gauss(3, 13, 17)
        assert x == 4
        assert pow(3, x, 17) == 13
    
    def test_generator(self):
        for a in range(1, 11):
            x = discrete_log_gauss(2, a, 11)
            assert x is not None
            assert pow(2, x, 11) == a
    
    def test_no_solution(self):
        assert discrete_log_gauss(2, 0, 7) is None
    
    def test_large_prime(self):
        p = 10007
        g = primitive_root(p)
        a = 1234
        x = discrete_log_gauss(g, a, p)
        assert x is not None
        assert pow(g, x, p) == a % p


class TestEulerTotient:
    def test_prime(self):
        for p in [2, 3, 5, 7, 11, 13, 17, 19]:
            assert euler_totient(p) == p - 1
    
    def test_prime_power(self):
        assert euler_totient(8) == 4   # φ(2^3) = 2^3 - 2^2 = 4
        assert euler_totient(9) == 6   # φ(3^2) = 3^2 - 3^1 = 6
        assert euler_totient(16) == 8  # φ(2^4) = 16 - 8 = 8
    
    def test_coprime_product(self):
        assert euler_totient(15) == 8  # φ(3)*φ(5) = 2*4 = 8
        assert euler_totient(30) == 8  # φ(2)*φ(3)*φ(5) = 1*2*4 = 8
    
    def test_known_values(self):
        assert euler_totient(1) == 1
        assert euler_totient(100) == 40
        assert euler_totient(1000) == 400


class TestMultiplicativeOrder:
    def test_basic(self):
        assert multiplicative_order(2, 7) == 3  # 2^3 = 8 ≡ 1 (mod 7)
    
    def test_generator(self):
        g = primitive_root(11)
        assert multiplicative_order(g, 11) == 10
    
    def test_not_coprime_raises(self):
        with pytest.raises(ValueError):
            multiplicative_order(2, 4)


class TestLegendreSymbol:
    def test_quadratic_residue(self):
        p = 11
        residues = {1, 3, 4, 5, 9}
        for a in range(1, p):
            expected = 1 if a in residues else -1
            assert legendre_symbol(a, p) == expected, f"a={a}"
    
    def test_zero(self):
        assert legendre_symbol(0, 7) == 0
        assert legendre_symbol(11, 11) == 0


class TestTonelliShanks:
    def test_simple_cases(self):
        r = tonelli_shanks(4, 11)
        assert r in [2, 9]
        assert (r * r) % 11 == 4
    
    def test_p_3_mod_4(self):
        p = 19  # 19 ≡ 3 (mod 4)
        for a in range(1, p):
            if legendre_symbol(a, p) == 1:
                r = tonelli_shanks(a, p)
                assert r is not None
                assert (r * r) % p == a % p
    
    def test_p_1_mod_4(self):
        p = 13  # 13 ≡ 1 (mod 4)
        for a in range(1, p):
            if legendre_symbol(a, p) == 1:
                r = tonelli_shanks(a, p)
                assert r is not None
                assert (r * r) % p == a % p
    
    def test_non_residue(self):
        assert tonelli_shanks(2, 11) is None  # 2 is non-residue mod 11
    
    def test_p_2(self):
        assert tonelli_shanks(1, 2) == 1
        assert tonelli_shanks(0, 2) == 0


class TestIntegration:
    def test_gauss_discrete_log_consistency(self):
        p = 31
        g = primitive_root(p)
        
        for a in range(1, p):
            x = discrete_log_gauss(g, a, p)
            assert x is not None
            assert pow(g, x, p) == a
    
    def test_crt_with_many_moduli(self):
        moduli = [3, 5, 7, 11, 13]
        residues = [2, 3, 1, 5, 7]
        x = chinese_remainder(residues, moduli)
        for a, m in zip(residues, moduli):
            assert x % m == a


if __name__ == "__main__":
    pytest.main([__file__, "-v"])