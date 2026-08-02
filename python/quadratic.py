"""Quadratic reciprocity and Legendre/Jacobi/Kronecker symbols."""

from math import gcd
from typing import List, Tuple
from python.modular import mod_inv

__all__ = [
    'legendre_symbol', 'legendre_symbol_gauss',
    'jacobi_symbol', 'kronecker_symbol',
    'quadratic_reciprocity', 'quadratic_residues', 'quadratic_nonresidues',
    'tonelli_shanks', 'cipolla',
    'solve_quadratic_congruence',
    'gauss_sum', 'verify_gauss_sum',
]


def legendre_symbol(a: int, p: int) -> int:
    """Legendre symbol (a/p) for odd prime p using Euler's criterion.
    
    Returns 1 if a is a quadratic residue mod p,
    -1 if non-residue, 0 if a ≡ 0 (mod p).
    """
    if p == 2:
        return a % 2
    a = a % p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def legendre_symbol_gauss(a: int, p: int) -> int:
    """Legendre symbol using Gauss's Lemma.
    
    Counts how many of {a, 2a, ..., (p-1)/2 * a} have residue > p/2.
    """
    if p == 2:
        return a % 2
    a = a % p
    if a == 0:
        return 0
    
    mu = 0
    for k in range(1, (p - 1) // 2 + 1):
        residue = (k * a) % p
        if residue > p // 2:
            mu += 1
    return -1 if mu % 2 else 1


def jacobi_symbol(a: int, n: int) -> int:
    """Jacobi symbol (a/n) for odd positive n.
    
    Uses quadratic reciprocity law for Jacobi symbol.
    """
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be odd positive integer")
    
    a = a % n
    if a == 0:
        return 0 if n > 1 else 1
    if a == 1:
        return 1
    
    # Factor out powers of 2
    t = 0
    while a % 2 == 0:
        a //= 2
        t += 1
    
    # Apply (2/n)^t
    result = 1
    if t % 2 == 1:
        # (2/n) = (-1)^((n^2-1)/8)
        result *= -1 if (n * n - 1) // 8 % 2 else 1
    
    if a == 1:
        return result
    
    # Apply reciprocity: (a/n) = (n/a) * (-1)^((a-1)(n-1)/4)
    if (a - 1) // 2 % 2 == 1 and (n - 1) // 2 % 2 == 1:
        result *= -1
    
    return result * jacobi_symbol(n % a, a)


def kronecker_symbol(a: int, n: int) -> int:
    """Kronecker symbol (a/n) for any integer n.
    
    Extension of Jacobi symbol to all integers n.
    """
    if n == 0:
        return 1 if abs(a) == 1 else 0
    
    # Factor n into u * 2^e * m where m is odd positive
    u = -1 if n < 0 else 1
    n_abs = abs(n)
    
    e = 0
    while n_abs % 2 == 0:
        n_abs //= 2
        e += 1
    
    # (a/u) factor
    result = 1
    if u == -1:
        if a < 0:
            result = -1
    
    # (a/2)^e factor
    if e > 0:
        if a % 2 == 0:
            return 0
        if e % 2 == 1:
            result *= 1 if a % 8 in (1, 7) else -1
            
    # (a/m) factor
    return result * jacobi_symbol(a, n_abs)


def quadratic_reciprocity(p: int, q: int) -> Tuple[int, int]:
    """Verify quadratic reciprocity for odd primes p, q.
    
    Returns (LHS, RHS) where LHS = (p/q)*(q/p) and RHS = (-1)^((p-1)(q-1)/4).
    """
    if p == q or p % 2 == 0 or q % 2 == 0:
        raise ValueError("p and q must be distinct odd primes")
    
    lhs = legendre_symbol(p, q) * legendre_symbol(q, p)
    rhs = -1 if ((p - 1) // 2) % 2 == 1 and ((q - 1) // 2) % 2 == 1 else 1
    return (lhs, rhs)


def quadratic_residues(p: int) -> List[int]:
    """Return all quadratic residues modulo odd prime p."""
    if p == 2:
        return [1]
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    return sorted(residues)


def quadratic_nonresidues(p: int) -> List[int]:
    """Return all quadratic non-residues modulo odd prime p."""
    residues = set(quadratic_residues(p))
    return [a for a in range(1, p) if a not in residues]


def tonelli_shanks(n: int, p: int) -> int:
    """Tonelli-Shanks algorithm for sqrt(n) mod p.
    
    Returns r such that r^2 ≡ n (mod p).
    Raises ValueError if no solution exists.
    """
    if n % p == 0:
        return 0
    if legendre_symbol(n, p) != 1:
        raise ValueError(f"{n} is not a quadratic residue mod {p}")
    
    if p == 2:
        return n % 2
    
    # Simple case: p ≡ 3 (mod 4)
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
                raise ValueError("No square root found (algorithm failed)")
        
        b = pow(c, 1 << (M - i - 1), p)
        M = i
        c = (b * b) % p
        t = (t * c) % p
        R = (R * b) % p
    
    return R


def cipolla(a: int, p: int) -> int:
    """Cipolla's algorithm for square root modulo prime p.
    
    Finds r such that r^2 ≡ a (mod p).
    """
    if legendre_symbol(a, p) != 1:
        raise ValueError(f"{a} is not a quadratic residue mod {p}")
    
    if p == 2:
        return a
    
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    
    # Find t such that t^2 - a is a non-residue
    for t in range(p):
        d = (t * t - a) % p
        if legendre_symbol(d, p) == -1:
            break
    else:
        raise ValueError("No suitable t found")
    
    # Work in F_p[sqrt(d)] = {x + y*sqrt(d) | x,y in F_p}
    # Compute (t + sqrt(d))^((p+1)/2)
    exp = (p + 1) // 2
    
    # Represent as (x, y) = x + y*sqrt(d)
    def mul(u, v):
        return ((u[0]*v[0] + u[1]*v[1]*d) % p,
                (u[0]*v[1] + u[1]*v[0]) % p)
    
    result = (1, 0)
    base = (t, 1)
    
    while exp > 0:
        if exp & 1:
            result = mul(result, base)
        base = mul(base, base)
        exp >>= 1
    
    return result[0]


def solve_quadratic_congruence(a: int, b: int, c: int, p: int) -> List[int]:
    """Solve ax^2 + bx + c ≡ 0 (mod p) for prime p.
    
    Returns list of solutions modulo p.
    """
    if p == 2:
        return [x for x in range(2) if (a*x*x + b*x + c) % 2 == 0]
    
    a, b, c = a % p, b % p, c % p
    
    if a == 0:
        # Linear: bx + c ≡ 0
        if b == 0:
            return [] if c != 0 else list(range(p))
        return [(-c * mod_inv(b, p)) % p]
    
    # Discriminant
    D = (b * b - 4 * a * c) % p
    
    if D == 0:
        # Double root
        return [(-b * mod_inv(2 * a % p, p)) % p]
    
    if legendre_symbol(D, p) == -1:
        return []
    
    sqrt_D = tonelli_shanks(D, p)
    inv_2a = mod_inv(2 * a % p, p)
    
    r1 = (-b + sqrt_D) * inv_2a % p
    r2 = (-b - sqrt_D) * inv_2a % p
    
    return [r1, r2] if r1 != r2 else [r1]


def gauss_sum(p: int) -> complex:
    """Quadratic Gauss sum g(p) = sum_{a=0}^{p-1} e^(2πi a^2/p).
    
    Returns g(p) = sqrt(p) if p ≡ 1 (mod 4), i*sqrt(p) if p ≡ 3 (mod 4).
    """
    import cmath
    total = 0
    for a in range(p):
        total += cmath.exp(2j * cmath.pi * a * a / p)
    return total


def verify_gauss_sum(p: int) -> bool:
    """Verify Gauss's evaluation of the quadratic Gauss sum."""
    g = gauss_sum(p)
    expected = (1 if p % 4 == 1 else 1j) * (p ** 0.5)
    return abs(g - expected) < 1e-10


if __name__ == "__main__":
    print("=== Quadratic Reciprocity Demo ===")
    
    # Legendre symbols
    print("\n1. Legendre Symbols:")
    for p in [7, 11, 13]:
        residues = quadratic_residues(p)
        nonres = quadratic_nonresidues(p)
        print(f"   p={p}: residues={residues}, non-residues={nonres}")
    
    # Quadratic reciprocity
    print("\n2. Quadratic Reciprocity Verification:")
    for p, q in [(3, 5), (3, 7), (5, 13), (11, 17), (7, 11)]:
        lhs, rhs = quadratic_reciprocity(p, q)
        print(f"   ({p}/{q})*({q}/{p}) = {lhs} (expected {rhs}) {'✓' if lhs == rhs else '✗'}")
    
    # Jacobi symbol
    print("\n3. Jacobi Symbol:")
    print(f"   (12/35) = {jacobi_symbol(12, 35)}")
    print(f"   (5/21) = {jacobi_symbol(5, 21)}")
    print(f"   (7/15) = {jacobi_symbol(7, 15)}")
    
    # Kronecker symbol
    print("\n4. Kronecker Symbol:")
    print(f"   (5/12) = {kronecker_symbol(5, 12)}")
    print(f"   (-1/8) = {kronecker_symbol(-1, 8)}")
    print(f"   (3/8) = {kronecker_symbol(3, 8)}")
    
    # Square roots mod p
    print("\n5. Square Roots (Tonelli-Shanks):")
    for n, p in [(10, 13), (5, 11), (6, 19), (2, 17)]:
        try:
            r = tonelli_shanks(n, p)
            print(f"   sqrt({n}) mod {p} = {r} (check: {r}^2 = {r*r % p})")
        except ValueError as e:
            print(f"   sqrt({n}) mod {p}: {e}")
    
    # Quadratic congruence
    print("\n6. Quadratic Congruence: 2x^2 + 3x + 1 ≡ 0 (mod 11)")
    sols = solve_quadratic_congruence(2, 3, 1, 11)
    print(f"   Solutions: {sols}")
    for x in sols:
        val = (2*x*x + 3*x + 1) % 11
        print(f"   Check: 2*{x}^2 + 3*{x} + 1 = {2*x*x + 3*x + 1} ≡ {val} (mod 11)")
    
    # Gauss sum
    print("\n7. Quadratic Gauss Sum:")
    for p in [3, 5, 7, 11]:
        g = gauss_sum(p)
        expected = (1 if p % 4 == 1 else 1j) * (p ** 0.5)
        print(f"   g({p}) = {g:.6f} (expected {expected:.6f}) diff={abs(g-expected):.2e}")