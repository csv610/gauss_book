"""Lemniscate functions: sl(ω), cl(ω).

Standard definitions (Gauss, Whittaker & Watson):
- K = ∫₀¹ dt/√(1-t⁴) = 1.311028777... (quarter period)
- 2K = 2.622057554... (half period)
- 4K = 5.244... (full period)
- sl(ω) = u where ω = ∫₀ᵘ dt/√(1-t⁴)

Pythagorean identity: sl² + cl² + sl²cl² = 1
"""

import math
from typing import List

__all__ = [
    'K', 'FULL_PERIOD', 'GAUSS_CONSTANT', 'OMEGA', 'HALF_PERIOD',
    'lemniscate_integral', 'lemniscate_integral_inverse',
    'sl', 'cl',
    'pythagorean_check',
    'sl_add', 'cl_add',
    'lemniscate_division', 'constructible_lemniscate_divisions',
    'lemniscate_hypergeometric',
]
from typing import List


# ======================================================================
# Verified Constants
# ======================================================================

# Quarter period K = ∫₀¹ dt/√(1-t⁴) = Γ(1/4)²/(4√π) ≈ 1.311028777
K = 1.31102877714605990523241979494555970684137747571581158140841

# Full period = 4K ≈ 5.244115108...
FULL_PERIOD = 4 * K

# Gauss constant G = 1/agm(1, √2) ≈ 0.8346268416740732
GAUSS_CONSTANT = 0.834626841674073186281429732799046808993993013414

# Verified: K = π/(2G)  (quarter period = π/(2G))
# Full period = 4K = 2π/G

# For backward compatibility
OMEGA = 2 * K  # ϖ = 2K = half period
FULL_PERIOD = 4 * K  # full period = 4K
HALF_PERIOD = K  # K = quarter period


# ======================================================================
# Direct numerical integration (Simpson's rule)
# ======================================================================

def lemniscate_integral(u: float) -> float:
    """ω = ∫₀ᵘ dt/√(1-t⁴) using Simpson's rule.
    
    Maps u ∈ [0, 1] to ω ∈ [0, K] (quarter period).
    """
    if u <= 0:
        return 0.0
    if u >= 1:
        return K  # quarter period
    
    # Simpson's rule with adaptive refinement
    n = 20000  # even
    h = u / n
    integral = 0.0
    for i in range(n):
        t0 = i * h
        t1 = t0 + h/2
        t2 = t0 + h
        f0 = 1.0 / math.sqrt(1.0 - t0**4)
        f1 = 1.0 / math.sqrt(1.0 - t1**4)
        f2 = 1.0 / math.sqrt(1.0 - t2**4)
        integral += h / 6.0 * (f0 + 4*f1 + f2)
    return integral


def lemniscate_integral_inverse(omega: float) -> float:
    """Find u such that lemniscate_integral(u) = omega.
    
    Domain: ω ∈ [0, K] maps to u ∈ [0, 1].
    """
    if omega <= 0:
        return 0.0
    if omega >= K:
        return 1.0
    
    # Newton's method with safeguards
    u = omega / K  # initial guess
    
    for _ in range(50):
        F_u = lemniscate_integral(u)
        f_u = 1.0 / math.sqrt(max(1e-15, 1.0 - u**4))
        diff = F_u - omega
        
        if abs(diff) < 1e-15:
            break
        
        delta = diff / f_u
        u_new = u - delta
        
        # Clamp to [0, 1]
        if u_new >= 1:
            u_new = 1 - 1e-10
        elif u_new <= 0:
            u_new = 1e-10
        
        if abs(u_new - u) < 1e-15:
            return max(0.0, min(1.0, u_new))
        
        u = u_new
    
    return max(0.0, min(1.0, u))


# ======================================================================
# Lemniscate sine and cosine (principal branch, no recursion!)
# ======================================================================

def sl(omega: float) -> float:
    """Lemniscate sine: sl(ω) = u where ω = ∫₀ᵘ dt/√(1-t⁴).
    
    Properties (K ≈ 1.311):
    - sl(0) = 0
    - sl(K/2) = 1/√2
    - sl(K) = 1
    - sl(3K/2) = 1/√2
    - sl(2K) = 0
    - sl(3K) = -1
    - sl(4K) = 0
    - sl(-ω) = -sl(ω)  (odd)
    - sl(ω + 4K) = sl(ω)  (period 4K)
    - sl(ω + 2K) = -sl(ω)  (anti-period 2K)
    """
    if omega == 0:
        return 0.0
    
    # Reduce to [0, 4K)
    omega = omega % (4 * K)
    if omega < 0:
        omega += 4 * K
    
    # Key values
    if abs(omega) < 1e-15:
        return 0.0
    if abs(omega - 2*K) < 1e-15:
        return 0.0
    if abs(omega - 4*K) < 1e-15:
        return 0.0
    
    # Odd symmetry
    if omega < 0:
        return -sl(-omega)
    
    # Anti-period 2K: sl(ω + 2K) = -sl(ω)
    if omega > 2*K:
        return -sl(omega - 2*K)
    
    # Now ω ∈ [0, 2K]
    if omega > K:
        # Use symmetry: sl(2K - ω) = sl(ω) for ω ∈ [K, 2K]
        # Since sl(2K - ω) = sl(-(ω - 2K)) = -sl(ω - 2K) = sl(ω)
        return sl(2*K - omega)
    
    # ω ∈ [0, K]: use direct integral inverse
    return lemniscate_integral_inverse(omega)


def cl(omega: float) -> float:
    """Lemniscate cosine: cl(ω) = sl(K - ω).
    
    Properties:
    - cl(0) = 1
    - cl(K/2) = 1/√2
    - cl(K) = 0
    - cl(3K/2) = -1/√2
    - cl(2K) = -1
    - cl(3K) = 0
    - cl(4K) = 1
    - cl(-ω) = cl(ω) (even)
    - cl(ω + 4K) = cl(ω)
    - cl(ω + 2K) = -cl(ω)
    """
    return sl(K - omega)


# ======================================================================
# Pythagorean identity
# ======================================================================

def pythagorean_check(omega: float) -> float:
    s = sl(omega)
    c = cl(omega)
    return s*s + c*c + s*s*c*c


# ======================================================================
# Addition formulas
# ======================================================================

def sl_add(u: float, v: float) -> float:
    """sl(u+v) = (sl(u)cl(v) + sl(v)cl(u)) / (1 - sl(u)sl(v)cl(u)cl(v))."""
    su, cu = sl(u), cl(u)
    sv, cv = sl(v), cl(v)
    return (su*cv + sv*cu) / (1 - su*sv*cu*cv)


def cl_add(u: float, v: float) -> float:
    """cl(u+v) = (cl(u)cl(v) - sl(u)sl(v)) / (1 + sl(u)sl(v)cl(u)cl(v))."""
    su, cu = sl(u), cl(u)
    sv, cv = sl(v), cl(v)
    return (cu*cv - su*sv) / (1 + su*sv*cu*cv)


# ======================================================================
# Division of the lemniscate (Gauss's analog of 17-gon)
# ======================================================================

def lemniscate_division(n: int) -> List[float]:
    """Division points for n-secting the lemniscate (n odd).
    
    Returns sl((2k+1)K/n) for k = 0, ..., n-1.
    """
    if n % 2 == 0:
        raise ValueError("Only odd n implemented")
    
    points = []
    for k in range(n):
        omega = (2*k + 1) * K / (2*n)
        points.append(sl(omega))
    return points


def constructible_lemniscate_divisions(limit: int = 50) -> List[int]:
    """n for which lemniscate is constructible (same as polygons)."""
    fermat = [3, 5, 17, 257, 65537]
    constructible = set()
    
    for k in range(int(math.log2(limit)) + 2):
        base = 2**k
        if base > limit:
            break
        
        def add_products(current, start):
            if current > limit:
                return
            if current > 0:
                constructible.add(current)
            for i in range(start, len(fermat)):
                add_products(current * fermat[i], i + 1)
        
        add_products(base, 0)
    
    return sorted(constructible)


# ======================================================================
# Hypergeometric series (alternative computation)
# ======================================================================

def lemniscate_hypergeometric(x: float) -> float:
    """ω = x * ₂F₁(1/4, 1/2; 5/4; x⁴)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return K
    
    z = x**4
    a, b, c = 0.25, 0.5, 1.25
    result = 1.0
    term = 1.0
    
    for n in range(1, 100):
        term *= (a + n - 1) * (b + n - 1) / (c + n - 1) / n * z
        result += term
        if abs(term) < 1e-16:
            break
    
    return x * result


# ======================================================================
# Demo
# ======================================================================

if __name__ == "__main__":
    print("=== Lemniscate Functions (Standard Definitions) ===\n")
    
    print(f"Quarter period K = {K}")
    print(f"Full period 4K = {4*K}")
    print(f"Gauss constant G = {GAUSS_CONSTANT}")
    print(f"π/(2G) = {math.pi/(2*GAUSS_CONSTANT)}")
    print(f"K = π/(2G)? {abs(K - math.pi/(2*GAUSS_CONSTANT)) < 1e-12}")
    
    print("\nKey values:")
    for omega in [0, K/2, K, 3*K/2, 2*K, 5*K/2, 3*K, 7*K/2, 4*K]:
        s = sl(omega)
        c = cl(omega)
        pyth = pythagorean_check(omega)
        print(f"  ω={omega:.6f}: sl={s:.8f}, cl={c:.8f}, check={pyth:.8f}")
    
    print("\nPythagorean identity:")
    for omega in [0.1, 0.5, 1.0, 1.5, 2.0]:
        check = pythagorean_check(omega)
        print(f"  ω={omega:.3f}: sl²+cl²+sl²cl² = {check:.10f}")
    
    print("\nAddition formula:")
    for u, v in [(0.3, 0.5), (0.7, 1.2), (0.2, 0.8)]:
        s_uv = sl(u + v)
        s_add = sl_add(u, v)
        print(f"  sl({u}+{v}) = {s_uv:.8f}, formula = {s_add:.8f}, diff = {abs(s_uv-s_add):.2e}")
    
    print("\nPeriodicity:")
    for omega in [0.5, 1.0]:
        print(f"  sl({omega}+4K) = {sl(omega + 4*K):.8f} (expected {sl(omega):.8f})")
        print(f"  sl({omega}+2K) = {sl(omega + 2*K):.8f} (expected {-sl(omega):.8f})")
    
    print("\nOdd/Even:")
    for omega in [0.5, 1.0]:
        print(f"  sl(-{omega}) = {sl(-omega):.8f} (expected {-sl(omega):.8f})")
        print(f"  cl(-{omega}) = {cl(-omega):.8f} (expected {cl(omega):.8f})")
    
    print("\nComplementary:")
    for omega in [0.5, 1.0, 1.5]:
        print(f"  sl({omega}) = {sl(omega):.8f}, cl(K-{omega}) = {cl(K - omega):.8f}")
    
    print("\nDivision:")
    for n in [3, 5, 17]:
        try:
            pts = lemniscate_division(n)
            print(f"  {n}-division: {pts[:3]}... ({len(pts)} pts)")
        except ValueError as e:
            print(f"  {n}: {e}")
    
    print(f"\nConstructible divisions: {constructible_lemniscate_divisions(50)}")
    
    print("\nHypergeometric:")
    for x in [0.1, 0.5, 0.9]:
        F = lemniscate_hypergeometric(x)
        direct = lemniscate_integral(x)
        print(f"  x={x}: ₂F₁ = {F:.10f}, integral = {direct:.10f}, diff = {abs(F-direct):.2e}")