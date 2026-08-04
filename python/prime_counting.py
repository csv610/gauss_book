"""Prime counting: sieve of Eratosthenes and Gauss's conjectures on pi(x).

In 1792, at age fifteen, Gauss conjectured that the density of primes near
a large number x is approximately 1/log(x), so that pi(x) ~ x/log(x). He
recorded this in a table in his 1796 notebook and described it in his 1849
letter to Encke. This module implements the sieve of Eratosthenes, primality
tests, pi(x), the logarithmic integral Li(x), Legendre's improvement, and
Riemann's approximation R(x), together with the comparison tables that Gauss
himself tabulated for powers of ten.
"""

from math import exp, isqrt, log, floor
from typing import List, Optional, Tuple

from scipy.special import expi, zeta

__all__ = [
    'sieve_of_eratosthenes',
    'is_prime',
    'miller_rabin',
    'prime_pi',
    'nth_prime',
    'prime_density',
    'gauss_estimate',
    'legendre_estimate',
    'logarithmic_integral',
    'riemann_r',
    'prime_counting_table',
]


def sieve_of_eratosthenes(n: int) -> List[int]:
    """Return all primes up to n using the sieve of Eratosthenes.

    Uses a bytearray so that sieving to 10^7 uses only ~10 MB of memory.
    """
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = 0
    sieve[1] = 0
    for p in range(2, isqrt(n) + 1):
        if sieve[p]:
            sieve[p * p:n + 1:p] = b'\x00' * (((n - p * p) // p) + 1)
    return [i for i in range(2, n + 1) if sieve[i]]


def miller_rabin(n: int, bases: Optional[List[int]] = None) -> bool:
    """Miller-Rabin primality test with deterministic base set.

    With the default bases the test is deterministic for every
    n < 3.3 * 10^24, far beyond any value that occurs here.
    """
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in bases or list(small_primes):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def is_prime(n: int) -> bool:
    """Primality test, deterministic for n < 3.3 * 10^24."""
    return miller_rabin(n)


def prime_pi(x) -> int:
    """pi(x): the number of primes at most x.

    Computed by sieving up to floor(x); suitable for x up to ~10^7
    in interactive use.
    """
    if x < 2:
        return 0
    return len(sieve_of_eratosthenes(floor(x)))


def nth_prime(n: int) -> int:
    """Return the n-th prime, using n(log n + log log n) as a sieve bound."""
    if n < 1:
        raise ValueError("n must be positive")
    if n <= 4:
        x = 15
    else:
        x = int(n * (log(n) + log(log(n)))) + 10
    primes = sieve_of_eratosthenes(x)
    while len(primes) < n:
        x *= 2
        primes = sieve_of_eratosthenes(x)
    return primes[n - 1]


def prime_density(x) -> float:
    """Heuristic density of primes near x: 1/log(x) (Gauss's conjecture)."""
    if x <= 1:
        raise ValueError("x must be greater than 1")
    return 1.0 / log(x)


def gauss_estimate(x) -> float:
    """Gauss's estimate pi(x) ~ x / log(x)."""
    if x <= 1:
        raise ValueError("x must be greater than 1")
    return x / log(x)


def legendre_estimate(x) -> float:
    """Legendre's improvement pi(x) ~ x / (log(x) - 1.08366)."""
    if x <= 1:
        raise ValueError("x must be greater than 1")
    return x / (log(x) - 1.08366)


def logarithmic_integral(x) -> float:
    """Li(x) = Ei(log x), the logarithmic integral.

    Rounded values match the Li(x) column of Gauss's 1849 table, e.g.
    Li(10^6) = 78627.55.  The offset convention subtracts li(2) = 1.0452.
    """
    if x <= 0:
        raise ValueError("x must be positive")
    if x == 1:
        return -float('inf')
    return float(expi(log(x)))


def riemann_r(x, terms: int = 50) -> float:
    """Riemann's approximation R(x) via the Gram series.

    R(x) = 1 + sum_{k=1}^{inf} (log x)^k / (k * zeta(k+1) * k!).
    Closer to pi(x) than Li(x); for x <= 10^9, 50 terms suffice.
    """
    if x <= 1:
        raise ValueError("x must be greater than 1")
    if terms < 1:
        raise ValueError("terms must be positive")
    lx = log(x)
    total = 1.0
    term = lx / float(zeta(2))          # k = 1 term
    total += term
    for k in range(2, terms + 1):
        term *= (lx * (k - 1) / (k * k)) * float(zeta(k) / zeta(k + 1))
        total += term
    return total


def prime_counting_table(x_max: int) -> List[Tuple[int, int, float, float, float]]:
    """Compare pi(x), x/log(x) and Li(x) at powers of ten up to x_max.

    Each row is (x, pi(x), x/log(x), Li(x), pi(x) / (x/log(x))), the
    last column reproducing the ratio that convinced Gauss that the
    x/log(x) estimate improves as x grows.
    """
    if x_max < 10:
        raise ValueError("x_max must be at least 10")
    rows = []
    x = 10
    while x <= x_max:
        rows.append((x, prime_pi(x), gauss_estimate(x),
                     logarithmic_integral(x), prime_pi(x) / gauss_estimate(x)))
        x *= 10
    return rows


if __name__ == "__main__":
    print("=== Gauss's Prime Counting Demo ===")

    print("\n1. Table at powers of ten (Gauss's 1849 letter to Encke):")
    print(f"{'x':>12} {'pi(x)':>10} {'x/log x':>12} {'Li(x)':>12} {'pi/(x/log x)':>12}")
    for x, pi_x, xlx, lix, ratio in prime_counting_table(10**9):
        print(f"{x:>12,d} {pi_x:>10,d} {xlx:>12,.0f} {lix:>12,.0f} {ratio:>12.4f}")

    print("\n2. Legendre and Riemann approximations at x = 10^6:")
    x = 10**6
    print(f"   pi(10^6)             = {prime_pi(x):,}")
    print(f"   x/log x              = {gauss_estimate(x):,.1f}")
    print(f"   x/(log x - 1.08366)  = {legendre_estimate(x):,.1f}")
    print(f"   Li(10^6)             = {logarithmic_integral(x):,.1f}")
    print(f"   R(10^6)              = {riemann_r(x):,.1f}")

    print("\n3. Selected primes:")
    for n in (1, 10, 100, 1000, 10000):
        print(f"   p_{n} = {nth_prime(n):,}")
