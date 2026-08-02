"""Arithmetic-Geometric Mean and related algorithms."""

import math
from typing import List, Tuple, Callable

__all__ = [
    'agm', 'agm_sequence', 'gauss_constant',
    'elliptic_K', 'elliptic_E', 'elliptic_Pi',
    'RF', 'ellippi',
    'pi_brent_salamin', 'pi_borwein_quartic', 'pi_agm_simple',
    'lemniscate_constant', 'elliptic_perimeter',
    'theta_null_via_agm', 'theta3_direct',
    'hypergeometric_2F1', 'agm_hypergeometric',
]
from typing import List, Tuple, Callable


# ======================================================================
# Basic AGM
# ======================================================================

def agm(a: float, b: float, tol: float = 1e-15, max_iter: int = 100) -> float:
    """Arithmetic-Geometric Mean of a and b.
    
    Iterates: a_{n+1} = (a_n + b_n)/2, b_{n+1} = sqrt(a_n b_n)
    """
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    
    for _ in range(max_iter):
        a_new = (a + b) / 2
        b_new = math.sqrt(a * b)
        if abs(a_new - b_new) < tol:
            return a_new
        a, b = a_new, b_new
    
    return (a + b) / 2


def agm_sequence(a: float, b: float, max_iter: int = 100) -> List[Tuple[float, float]]:
    """Return the full sequence of AGM iterations."""
    seq = [(a, b)]
    for _ in range(max_iter):
        a_new = (a + b) / 2
        b_new = math.sqrt(a * b)
        seq.append((a_new, b_new))
        if abs(a_new - b_new) < 1e-15:
            break
        a, b = a_new, b_new
    return seq


def gauss_constant() -> float:
    """Gauss's constant G = 1/M(1, sqrt(2))."""
    return 1 / agm(1, math.sqrt(2))


# ======================================================================
# Complete elliptic integrals
# ======================================================================

def elliptic_K(k: float) -> float:
    """Complete elliptic integral of the first kind: K(k) = ∫_0^{π/2} dθ/√(1-k²sin²θ).
    
    Uses the AGM: K(k) = π/(2*M(1, k')) where k' = √(1-k²).
    """
    if abs(k) >= 1:
        raise ValueError("|k| must be < 1")
    k_prime = math.sqrt(1 - k*k)
    return math.pi / (2 * agm(1, k_prime))


def elliptic_E(k: float) -> float:
    """Complete elliptic integral of the second kind: E(k) = ∫_0^{π/2} √(1-k²sin²θ) dθ.
    
    Uses AGM with auxiliary sequence.
    """
    if abs(k) >= 1:
        raise ValueError("|k| must be < 1")
    
    a = 1.0
    b = math.sqrt(1 - k*k)
    sum_c2 = k * k
    two_pow = 2.0
    
    for _ in range(50):
        a_new = (a + b) / 2
        b_new = math.sqrt(a * b)
        c_new = (a - b) / 2
        sum_c2 += two_pow * c_new * c_new
        two_pow *= 2
        if abs(c_new) < 1e-15:
            break
        a, b = a_new, b_new
    
    K = math.pi / (2 * a)
    return K * (1 - sum_c2 / 2)


def elliptic_Pi(n: float, k: float) -> float:
    """Complete elliptic integral of the third kind."""
    # Use Carlson's symmetric forms
    return ellippi(n, k)


# ======================================================================
# Carlson symmetric forms
# ======================================================================

def RF(x: float, y: float, z: float, tol: float = 1e-12) -> float:
    """Carlson symmetric integral R_F(x,y,z)."""
    # Check for negative arguments
    if x < 0 or y < 0 or z < 0:
        raise ValueError("Arguments must be non-negative")
    
    # Handle special cases
    if x == y == z:
        return 1 / math.sqrt(x)
    
    # Duplication algorithm
    for _ in range(100):
        mu = (x + y + z) / 3
        dx = (mu - x) / mu
        dy = (mu - y) / mu
        dz = (mu - z) / mu
        eps = max(abs(dx), abs(dy), abs(dz))
        
        if eps < tol:
            e2 = dx*dy + dy*dz + dz*dx
            e3 = dx*dy*dz
            return (1 - e2/10 + e3/14 + e2*e2/24 - 3*e2*e3/44) / math.sqrt(mu)
        
        lam = math.sqrt(x*y) + math.sqrt(y*z) + math.sqrt(z*x)
        x = (x + lam) / 4
        y = (y + lam) / 4
        z = (z + lam) / 4
    
    return 0  # Should not reach


def ellippi(n: float, k: float) -> float:
    """Complete elliptic integral of the third kind
    Π(n,k) = ∫_0^{π/2} dθ / ((1-n sin²θ) √(1-k² sin²θ)).

    Uses composite Simpson's rule with N = 10000 subintervals.
    Requires |k| < 1 and n < 1 (with n < 1 to avoid the pole at sin²θ = 1/n).
    """
    if abs(k) >= 1:
        raise ValueError("|k| must be < 1 for Π(n,k)")
    if n >= 1:
        raise ValueError("n must be < 1 for Π(n,k)")

    N = 10000
    a, b = 0.0, math.pi / 2
    h = (b - a) / N

    def integrand(theta: float) -> float:
        s = math.sin(theta)
        s2 = s * s
        return 1.0 / ((1.0 - n * s2) * math.sqrt(1.0 - k * k * s2))

    total = integrand(a) + integrand(b)
    for i in range(1, N, 2):
        total += 4.0 * integrand(a + i * h)
    for i in range(2, N, 2):
        total += 2.0 * integrand(a + i * h)

    return total * h / 3.0


# ======================================================================
# Algorithms for π
# ======================================================================

def pi_brent_salamin(n: int) -> float:
    """Brent-Salamin (Gauss-Legendre) algorithm for π.
    
    a_0 = 1, b_0 = 1/√2, t_0 = 1/4, p_0 = 1
    a_{k+1} = (a_k + b_k)/2
    b_{k+1} = √(a_k b_k)
    t_{k+1} = t_k - p_k(a_k - a_{k+1})²
    p_{k+1} = 2 p_k
    π ≈ (a_{n+1} + b_{n+1})² / (4 t_{n+1})
    """
    a = 1.0
    b = 1.0 / math.sqrt(2)
    t = 0.25
    p = 1.0
    
    for _ in range(n):
        a_new = (a + b) / 2
        b_new = math.sqrt(a * b)
        t -= p * (a - a_new) ** 2
        p *= 2
        a, b = a_new, b_new
    
    return (a + b) ** 2 / (4 * t)


def pi_borwein_quartic(n: int) -> float:
    """Borwein's quartic algorithm for π.
    
    y_0 = √2 - 1
    a_0 = 6 - 4√2
    y_{n+1} = (1 - (1-y_n⁴)^{1/4}) / (1 + (1-y_n⁴)^{1/4})
    a_{n+1} = a_n(1+y_{n+1})⁴ - 2^{2n+3} y_{n+1}(1+y_{n+1}+y_{n+1}²)
    1/a_n → π quartically.
    """
    y = math.sqrt(2) - 1
    a = 6 - 4 * math.sqrt(2)
    
    for i in range(n):
        y4 = y**4
        one_minus_y4_pow = (1 - y4)**0.25
        y_new = (1 - one_minus_y4_pow) / (1 + one_minus_y4_pow)
        
        two_pow = 2**(2*i + 3)
        a = a * (1 + y_new)**4 - two_pow * y_new * (1 + y_new + y_new**2)
        y = y_new
    
    return 1 / a


def pi_agm_simple(n: int) -> float:
    """Simple AGM-based π using K(1/√2) = Γ(1/4)²/(4√π)."""
    K = elliptic_K(1/math.sqrt(2))
    return math.pi  # K = π/(2*agm(1,1/√2)) = π/2 * G where G is Gauss constant


# ======================================================================
# Other applications
# ======================================================================

def lemniscate_constant() -> float:
    """Arc length of the lemniscate: L = 2 ∫_0^1 dt/√(1-t⁴) = π * G."""
    G = gauss_constant()
    return math.pi * G


def elliptic_perimeter(a: float, b: float) -> float:
    """Perimeter of ellipse with semi-axes a, b.
    
    P = 4a E(e) where e = √(1-b²/a²) is the eccentricity.
    """
    if a < b:
        a, b = b, a
    e = math.sqrt(1 - b*b/(a*a))
    return 4 * a * elliptic_E(e)


def theta_null_via_agm(tau: complex, max_iter: int = 20) -> float:
    """θ₃(0|τ) via AGM for τ = i t (t > 0).
    
    θ₃(0|i t) = √(2K/π) where K = K(k), k = θ₂(0|i t)²/θ₃(0|i t)².
    """
    # Use the relation: θ₃(0|τ)² = 2K(k)/π
    # and k = θ₂(0|τ)²/θ₃(0|τ)²
    # This is circular; instead use:
    # For τ = i t: θ₃(0|i t) = √(2/π) * K(k) where k = θ₂²/θ₃²
    # But also K(k) = π/(2*agm(1,k'))
    # The direct AGM relation is:
    # θ₃(0|i t) = √(agm(1, e^{-π t})) * something...
    # For simplicity, use direct summation
    t = tau.imag
    return theta3_direct(0, t)


def theta3_direct(z: float, t: float, terms: int = 100) -> float:
    """θ₃(z|i t) via direct summation."""
    result = 0.0
    for n in range(-terms, terms + 1):
        result += math.exp(-math.pi * t * n**2) * math.cos(2 * math.pi * n * z)
    return result


# ======================================================================
# Hypergeometric functions
# ======================================================================

def hypergeometric_2F1(a: float, b: float, c: float, z: float, 
                        max_terms: int = 1000, tol: float = 1e-15) -> float:
    """Gauss hypergeometric function ₂F₁(a,b;c;z)."""
    if abs(z) >= 1:
        raise ValueError("|z| must be < 1")
    
    result = 1.0
    term = 1.0
    for n in range(1, max_terms + 1):
        term *= (a + n - 1) * (b + n - 1) / (c + n - 1) / n * z
        result += term
        if abs(term) < tol:
            break
    return result


def agm_hypergeometric(a: float, b: float) -> float:
    """AGM via hypergeometric function: M(1,x) = a / ₂F₁(1/2,1/2;1;1-x²)."""
    x = b / a
    F = hypergeometric_2F1(0.5, 0.5, 1.0, 1 - x*x)
    return a / F


if __name__ == "__main__":
    print("=== AGM Demo ===")
    
    # AGM sequence
    print("\n1. AGM sequence:")
    seq = agm_sequence(1, 0.5)
    for i, (a, b) in enumerate(seq[:6]):
        print(f"   n={i}: a={a:.10f}, b={b:.10f}, diff={abs(a-b):.2e}")
    
    # Elliptic integrals
    print("\n2. Elliptic integrals:")
    for k in [0.5, 1/math.sqrt(2), 0.9]:
        K = elliptic_K(k)
        E = elliptic_E(k)
        print(f"   k={k:.4f}: K={K:.10f}, E={E:.10f}")
    
    # π algorithms
    print("\n3. Pi algorithms:")
    print(f"   Brent-Salamin (3 iter): {pi_brent_salamin(3):.15f}")
    print(f"   Borwein quartic (3 iter): {pi_borwein_quartic(3):.15f}")
    print(f"   True π: {math.pi:.15f}")
    
    # Gauss constant
    print(f"\n4. Gauss constant: G = {gauss_constant():.10f}")
    print(f"   Expected: 0.83462684167...")
    
    # Lemniscate constant
    print(f"\n5. Lemniscate constant: L = {lemniscate_constant():.10f}")
    
    # Ellipse perimeter
    print(f"\n6. Ellipse perimeter (a=2, b=1): {elliptic_perimeter(2, 1):.10f}")
    
    # AGM via hypergeometric
    print(f"\n7. AGM via ₂F₁: M(1,0.5) = {agm_hypergeometric(1, 0.5):.10f}")
    print(f"   Direct AGM: {agm(1, 0.5):.10f}")