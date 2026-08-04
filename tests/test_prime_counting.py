"""Tests for the prime counting module."""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.prime_counting import (
    sieve_of_eratosthenes, is_prime, miller_rabin, prime_pi, nth_prime,
    prime_density, gauss_estimate, legendre_estimate, logarithmic_integral,
    riemann_r, prime_counting_table,
)


class TestSieve:
    def test_small(self):
        assert sieve_of_eratosthenes(10) == [2, 3, 5, 7]
        assert sieve_of_eratosthenes(2) == [2]
        assert sieve_of_eratosthenes(1) == []
        assert sieve_of_eratosthenes(0) == []

    def test_primes_under_100(self):
        primes = sieve_of_eratosthenes(100)
        assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
                          43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    def test_all_primes(self):
        n = 500
        primes = sieve_of_eratosthenes(n)
        assert all(is_prime(p) for p in primes)
        for m in range(2, n + 1):
            if m not in primes:
                assert not is_prime(m), f"{m} should be composite"


class TestPrimality:
    def test_known_primes(self):
        for p in [2, 3, 5, 17, 257, 65537, 7919, 104729, 15485863]:
            assert is_prime(p), f"{p} should be prime"

    def test_known_composites(self):
        for n in [1, 4, 6, 9, 91, 561, 1105, 1729, 65536, 4294967297]:
            assert not is_prime(n), f"{n} should be composite"

    def test_fermat_number(self):
        # 2^32 + 1 = 641 * 6700417 is composite (Euler's factorization)
        assert not is_prime(2 ** 32 + 1)

    def test_large_mersenne(self):
        assert is_prime(2 ** 61 - 1)
        assert not is_prime(2 ** 67 - 1)  # factored by Cole in 1903

    def test_miller_rabin_small(self):
        assert miller_rabin(2)
        assert miller_rabin(17)
        assert not miller_rabin(1)
        assert not miller_rabin(561)  # Carmichael number


class TestPrimePi:
    def test_known_values(self):
        for x, expected in [(10, 4), (100, 25), (1000, 168),
                            (10 ** 4, 1229), (10 ** 5, 9592),
                            (10 ** 6, 78498), (10 ** 7, 664579)]:
            assert prime_pi(x) == expected, f"pi({x})"

    def test_float_argument(self):
        assert prime_pi(10.0) == 4
        assert prime_pi(2.9) == 1

    def test_small_values(self):
        assert prime_pi(1) == 0
        assert prime_pi(2) == 1


class TestNthPrime:
    def test_known_values(self):
        for n, expected in [(1, 2), (2, 3), (10, 29), (100, 541),
                            (1000, 7919), (10 ** 4, 104729)]:
            assert nth_prime(n) == expected, f"p_{n}"

    def test_p_100000(self):
        assert nth_prime(10 ** 5) == 1299709

    def test_raises(self):
        with pytest.raises(ValueError):
            nth_prime(0)


class TestEstimates:
    def test_gauss_estimate(self):
        x = 10 ** 6
        assert abs(gauss_estimate(x) - 72382.41) < 0.1

    def test_legendre_estimate(self):
        x = 10 ** 6
        assert abs(legendre_estimate(x) - 78543.2) < 0.1

    def test_logarithmic_integral(self):
        x = 10 ** 6
        assert abs(logarithmic_integral(x) - 78627.549) < 1e-3

    def test_density(self):
        assert abs(prime_density(10 ** 6) - 1 / 13.815510557964274) < 1e-9

    def test_density_unit_interval(self):
        assert prime_density(10 ** 6) < 1
        assert prime_density(10 ** 6) > 0

    def test_invalid(self):
        with pytest.raises(ValueError):
            gauss_estimate(1)
        with pytest.raises(ValueError):
            logarithmic_integral(0)


class TestRiemannR:
    def test_known_values(self):
        assert abs(riemann_r(10 ** 6) - 78527.4) < 1.0
        assert abs(riemann_r(10 ** 9) - 50847455.1) < 2.0

    def test_converges_with_terms(self):
        for terms in (50, 100, 200):
            assert abs(riemann_r(10 ** 6, terms) - 78527.4) < 2.0

    def test_invalid(self):
        with pytest.raises(ValueError):
            riemann_r(1)
        with pytest.raises(ValueError):
            riemann_r(10, 0)


class TestTable:
    def test_rows(self):
        rows = prime_counting_table(10 ** 4)
        assert rows[0][0] == 10
        assert rows[0][1] == 4
        assert rows[-1][0] == 10 ** 4
        assert rows[-1][1] == 1229

    def test_ratio_column(self):
        rows = prime_counting_table(10 ** 6)
        x, pi_x, xlx, lix, ratio = rows[-1]
        assert ratio == pytest.approx(pi_x / xlx, rel=1e-12)

    def test_li_closer_than_x_log_x(self):
        for x, pi_x, xlx, lix, _ in prime_counting_table(10 ** 6):
            if x < 1000:
                continue
            assert abs(lix - pi_x) < abs(xlx - pi_x), f"x={x}"

    def test_too_small(self):
        with pytest.raises(ValueError):
            prime_counting_table(5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
