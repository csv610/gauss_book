"""Modular arithmetic utilities implementing Gauss's algorithms."""

from math import gcd
from typing import List, Tuple, Optional

__all__ = [
    'mod_inv',
    'chinese_remainder',
    'primitive_root', 'all_primitive_roots', 'euler_totient', 'multiplicative_order',
    'discrete_log_gauss',
    'legendre_symbol', 'tonelli_shanks',
]


def mod_inv(a: int, n: int) -> int:
    """Modular inverse using extended Euclidean algorithm.
    
    Returns x such that a*x ≡ 1 (mod n).
    Raises ValueError if inverse doesn't exist.
    """
    def egcd(a: int, b: int) -> Tuple[int, int, int]:
        if b == 0:
            return (a, 1, 0)
        g, x1, y1 = egcd(b, a % b)
        return (g, y1, x1 - (a // b) * y1)
    
    g, x, _ = egcd(a, n)
    if g != 1:
        raise ValueError(f"No modular inverse for {a} mod {n}")
    return x % n


def chinese_remainder(residues: List[int], moduli: List[int]) -> int:
    """Solve system of congruences using Gauss's constructive proof.
    
    Args:
        residues: List of residues a_i
        moduli: List of pairwise coprime moduli n_i
        
    Returns:
        Solution x modulo product of moduli
    """
    if len(residues) != len(moduli):
        raise ValueError("Residues and moduli must have same length")
    
    # Verify pairwise coprime
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            if gcd(moduli[i], moduli[j]) != 1:
                raise ValueError(f"Moduli {moduli[i]} and {moduli[j]} are not coprime")
    
    N = 1
    for m in moduli:
        N *= m
    
    result = 0
    for a_i, n_i in zip(residues, moduli):
        N_i = N // n_i
        M_i = mod_inv(N_i, n_i)
        result += a_i * N_i * M_i
    
    return result % N


def primitive_root(p: int) -> int:
    """Find a primitive root modulo prime p.
    
    Uses Gauss's criterion: g is a primitive root iff
    g^((p-1)/q) ≠ 1 (mod p) for all prime factors q of p-1.
    """
    if p == 2:
        return 1
    
    # Factor p-1
    def prime_factors(n: int) -> List[int]:
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1 if d == 2 else 2
        if n > 1:
            factors.append(n)
        return list(set(factors))
    
    factors = prime_factors(p - 1)
    
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in factors):
            return g
    
    raise ValueError(f"No primitive root found for {p}")


def discrete_log_gauss(g: int, a: int, p: int) -> Optional[int]:
    """Gauss's algorithm for discrete logarithm (baby-step giant-step).
    
    Finds x such that g^x ≡ a (mod p), where g is a primitive root mod p.
    Returns None if no solution exists.
    """
    if a % p == 0:
        return None
    
    m = int((p - 1) ** 0.5) + 1
    
    # Baby steps: g^j for j = 0, 1, ..., m-1
    baby_steps = {}
    cur = 1
    for j in range(m):
        if cur not in baby_steps:
            baby_steps[cur] = j
        cur = (cur * g) % p
    
    # Giant steps: a * g^(-m*i)
    g_inv_m = pow(g, (p - 1) - m, p)  # g^(-m) mod p
    cur = a % p
    
    for i in range(m + 1):
        if cur in baby_steps:
            return i * m + baby_steps[cur]
        cur = (cur * g_inv_m) % p
    
    return None


def euler_totient(n: int) -> int:
    """Euler's totient function φ(n)."""
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1 if p == 2 else 2
    if n > 1:
        result -= result // n
    return result


def multiplicative_order(a: int, n: int) -> int:
    """Order of a modulo n (smallest k > 0 such that a^k ≡ 1 mod n)."""
    if gcd(a, n) != 1:
        raise ValueError("a and n must be coprime")
    
    phi = euler_totient(n)
    
    # Find all divisors of phi in O(sqrt(phi))
    divisors = []
    d = 1
    while d * d <= phi:
        if phi % d == 0:
            divisors.append(d)
            if d * d != phi:
                divisors.append(phi // d)
        d += 1
    divisors.sort()
    
    # Check each divisor
    for d in divisors:
        if pow(a, d, n) == 1:
            return d
    return phi


def all_primitive_roots(p: int) -> List[int]:
    """Return all primitive roots modulo prime p."""
    g = primitive_root(p)
    phi = p - 1
    roots = []
    for k in range(1, phi + 1):
        if gcd(k, phi) == 1:
            roots.append(pow(g, k, p))
    return sorted(roots)


def legendre_symbol(a: int, p: int) -> int:
    """Legendre symbol (a/p) for odd prime p.
    
    Returns 1 if a is a quadratic residue mod p,
    -1 if a is a non-residue, 0 if a ≡ 0 (mod p).
    """
    if a % p == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def tonelli_shanks(n: int, p: int) -> Optional[int]:
    """Tonelli-Shanks algorithm for square root modulo prime p.
    
    Returns r such that r^2 ≡ n (mod p), or None if no solution.
    """
    if n % p == 0:
        return 0
    if p == 2:
        return n % 2
    if legendre_symbol(n, p) != 1:
        return None
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)
    
    # Factor p-1 = Q * 2^S with Q odd
    Q = p - 1
    S = 0
    while Q % 2 == 0:
        Q //= 2
        S += 1
    
    # Find a quadratic non-residue z
    z = 2
    while legendre_symbol(z, p) != -1:
        z += 1
    
    M = S
    c = pow(z, Q, p)
    t = pow(n, Q, p)
    R = pow(n, (Q + 1) // 2, p)
    
    while t != 1:
        # Find least i such that t^(2^i) = 1
        i = 0
        temp = t
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
            if i == M:
                return None
        
        b = pow(c, 1 << (M - i - 1), p)
        M = i
        c = (b * b) % p
        t = (t * c) % p
        R = (R * b) % p
    
    return R


if __name__ == "__main__":
    # Demo
    print("=== Modular Arithmetic Demo ===")
    
    # Chinese Remainder Theorem
    print("\n1. Chinese Remainder Theorem:")
    x = chinese_remainder([2, 3, 2], [3, 5, 7])
    print(f"   x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7) => x = {x}")
    print(f"   Check: {x} % 3 = {x % 3}, {x} % 5 = {x % 5}, {x} % 7 = {x % 7}")
    
    # Primitive roots
    print("\n2. Primitive Roots:")
    p = 17
    g = primitive_root(p)
    roots = all_primitive_roots(p)
    print(f"   Primitive root of {p}: {g}")
    print(f"   All primitive roots mod {p}: {roots}")
    
    # Discrete log
    print("\n3. Discrete Logarithm (Gauss's baby-step giant-step):")
    g = 3
    a = 13
    p = 17
    x = discrete_log_gauss(g, a, p)
    print(f"   {g}^x ≡ {a} (mod {p}) => x = {x}")
    print(f"   Check: {g}^{x} % {p} = {pow(g, x, p)}")
    
    # Tonelli-Shanks
    print("\n4. Square Root Modulo Prime (Tonelli-Shanks):")
    n, p = 10, 13
    r = tonelli_shanks(n, p)
    print(f"   sqrt({n}) mod {p} = {r}")
    if r:
        print(f"   Check: {r}^2 % {p} = {r * r % p}")