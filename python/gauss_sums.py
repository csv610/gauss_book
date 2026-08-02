"""Gauss sums and Dirichlet characters."""

import cmath
import math
from typing import List, Dict, Tuple, Callable
import numpy as np

__all__ = [
    'quadratic_gauss_sum', 'gauss_sum_sign', 'verify_gauss_sum',
    'dirichlet_characters', 'general_gauss_sum', 'primitive_character',
    'is_primitive_root', 'discrete_log', 'prime_factors',
    'dirichlet_l_function', 'l_function_euler_product',
    'character_sum', 'polya_vinogradov',
    'fft_gauss_sum',
    'quadratic_reciprocity_gauss_sum', 'class_number_formula',
    'legendre_symbol',
]


# ======================================================================
# Quadratic Gauss sum
# ======================================================================

def quadratic_gauss_sum(p: int) -> complex:
    """Quadratic Gauss sum: g(p) = Σ_{a=0}^{p-1} e^{2πi a²/p}.
    
    For odd prime p:
    g(p) = √p if p ≡ 1 (mod 4)
    g(p) = i√p if p ≡ 3 (mod 4)
    """
    if p == 2:
        return 1 + 1j
    
    result = 0
    for a in range(p):
        result += cmath.exp(2j * math.pi * a * a / p)
    
    return result


def gauss_sum_sign(p: int) -> complex:
    """Gauss's evaluation of the quadratic Gauss sum."""
    sqrt_p = math.sqrt(p)
    if p % 4 == 1:
        return sqrt_p
    else:
        return 1j * sqrt_p


def verify_gauss_sum(p: int) -> Tuple[complex, complex, float]:
    """Compute g(p) directly and compare with theoretical value."""
    direct = quadratic_gauss_sum(p)
    theoretical = gauss_sum_sign(p)
    error = abs(direct - theoretical)
    return direct, theoretical, error


# ======================================================================
# General Gauss sums with Dirichlet characters
# ======================================================================

def dirichlet_characters(n: int) -> List[List[complex]]:
    """Generate all Dirichlet characters modulo n.
    
    Returns list of character values [χ(0), χ(1), ..., χ(n-1)] for each character.
    Handles prime n where (Z/nZ)^× is cyclic.
    """
    units = [a for a in range(n) if math.gcd(a, n) == 1]
    phi = len(units)
    
    if phi == 0:
        return []
    
    if n == 2:
        return [[1, 1], [1, -1]]
    
    # Find a generator g of (Z/nZ)^×
    g = None
    for gen in range(2, n):
        if math.gcd(gen, n) != 1:
            continue
        seen = set()
        cur = 1
        for _ in range(phi):
            seen.add(cur)
            cur = (cur * gen) % n
        if len(seen) == phi:
            g = gen
            break
    
    if g is None:
        raise ValueError(f"(Z/{n}Z)^× is not cyclic, only prime n supported")
    
    # Build index table: a -> discrete log base g
    index = {}
    cur = 1
    for k in range(phi):
        index[cur] = k
        cur = (cur * g) % n
    
    chars = []
    for k in range(phi):
        chi = [0] * n
        for a in units:
            chi[a] = cmath.exp(2j * math.pi * k * index[a] / phi)
        chars.append(chi)
    
    return chars


def general_gauss_sum(chi: List[int], n: int) -> complex:
    """Gauss sum τ(χ) = Σ χ(a) e^{2πi a/n}."""
    result = 0
    for a in range(n):
        if chi[a] != 0:
            result += chi[a] * cmath.exp(2j * math.pi * a / n)
    return result


def primitive_character(n: int) -> List[complex]:
    """Find a primitive character modulo n (if exists)."""
    chars = dirichlet_characters(n)
    if not chars:
        return None
    for chi in chars:
        # A character is primitive if it doesn't factor through a proper divisor
        # For prime n, every non-trivial character is primitive
        if any(abs(v) > 1e-10 for v in chi[1:]):
            return chi
    return None


def is_primitive_root(g: int, p: int) -> bool:
    """Check if g is a primitive root modulo p."""
    if math.gcd(g, p) != 1:
        return False
    phi = p - 1
    for q in prime_factors(phi):
        if pow(g, phi // q, p) == 1:
            return False
    return True


def discrete_log(g: int, a: int, p: int) -> int:
    """Find x such that g^x ≡ a (mod p)."""
    for x in range(p-1):
        if pow(g, x, p) == a:
            return x
    return None


def prime_factors(n: int) -> List[int]:
    """Prime factors of n."""
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


# ======================================================================
# Dirichlet L-functions
# ======================================================================

def dirichlet_l_function(chi: List[int], s: complex, terms: int = 10000) -> complex:
    """Dirichlet L-function L(s, χ) = Σ χ(n)/n^s."""
    result = 0
    for n in range(1, terms + 1):
        if chi[n % len(chi)] != 0:
            result += chi[n % len(chi)] / (n ** s)
    return result


def l_function_euler_product(chi: List[int], s: complex, primes: List[int]) -> complex:
    """L(s, χ) = ∏_p (1 - χ(p) p^{-s})^{-1}."""
    result = 1
    for p in primes:
        chi_p = chi[p % len(chi)]
        if chi_p != 0:
            result *= 1 / (1 - chi_p * (p ** (-s)))
    return result


# ======================================================================
# Character sums
# ======================================================================

def character_sum(chi: List[int], N: int, M: int = 0) -> complex:
    """Sum of character values: Σ_{n=M}^{N} χ(n)."""
    result = 0
    for n in range(M, N + 1):
        result += chi[n % len(chi)]
    return result


def polya_vinogradov(chi: List[int], N: int) -> float:
    """Pólya-Vinogradov inequality bound: |Σ_{n≤N} χ(n)| ≤ C √q log q."""
    q = len(chi)
    return math.sqrt(q) * math.log(q)


# ======================================================================
# Fast Gauss sum via FFT
# ======================================================================

def fft_gauss_sum(chi: List[int]) -> complex:
    """Compute Gauss sum using FFT for large n."""
    n = len(chi)
    # Pad to power of 2
    m = 1
    while m < n:
        m *= 2
    
    # χ values
    a = [chi[i] if i < n else 0 for i in range(m)]
    # Exponential values
    b = [cmath.exp(2j * math.pi * i / n) for i in range(m)]
    
    # FFT convolution
    A = np.fft.fft(a)
    B = np.fft.fft(b)
    C = A * B
    c = np.fft.ifft(C)
    
    return c[0]


# ======================================================================
# Applications
# ======================================================================

def quadratic_reciprocity_gauss_sum(p: int, q: int) -> int:
    """Check quadratic reciprocity via Gauss sums: g(p)^{q-1} = (q/p)^{(q-1)/2} ..."""
    g_p = quadratic_gauss_sum(p)
    g_q = quadratic_gauss_sum(q)
    lhs = (g_p ** (q - 1)).real
    rhs = (g_q ** (p - 1)).real if (p % 4 == 1 or q % 4 == 1) else -(g_q ** (p - 1)).real
    return 1 if abs(lhs - rhs) < 1e-10 else -1


def class_number_formula(D: int) -> float:
    """Class number formula using L(1, χ)."""
    raise NotImplementedError("Class number formula requires quadratic character and L(1,χ) computation")


# ======================================================================
# Demo
# ======================================================================

if __name__ == "__main__":
    print("=== Gauss Sums Demo ===")
    
    # Quadratic Gauss sums
    print("\n1. Quadratic Gauss sums:")
    for p in [3, 5, 7, 11, 13, 17, 19]:
        direct, theory, err = verify_gauss_sum(p)
        print(f"  g({p}) = {direct:.6f} (theory: {theory:.6f}, err: {err:.2e})")
    
    # Dirichlet characters
    print("\n2. Dirichlet characters modulo 7:")
    # Primitive character mod 7
    # (Z/7Z)^× is cyclic of order 6, generated by 3
    chi = [0, 1, 1, 1, 1, 1, 1]  # trivial
    print(f"  Trivial: {chi}")
    
    # Character sum
    print("\n3. Character sums:")
    for n in [10, 100, 1000]:
        # Sum of Legendre symbol mod p
        p = 7
        total = 0
        for a in range(1, n+1):
            total += legendre_symbol(a, p)
        print(f"  Σ_{{a=1}}^{n} (a/{p}) = {total}, bound: {polya_vinogradov([0]*7, n):.2f}")
    
    # L-function
    print("\n4. L-function values:")
    p = 5
    # Non-trivial character mod 5
    chi = [0, 1, 1j, -1j, -1]  # χ(2)=i
    for s in [2, 3, 4]:
        L = dirichlet_l_function(chi, s, 10000)
        print(f"  L({s}, χ) = {L:.6f}")


def legendre_symbol(a: int, p: int) -> int:
    """Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    return 1 if pow(a, (p-1)//2, p) == 1 else -1