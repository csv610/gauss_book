"""Theta functions and modular forms."""

import cmath
import math
from typing import List, Tuple, Callable
import numpy as np

__all__ = [
    'theta1', 'theta2', 'theta3', 'theta4', 'theta_null',
    'modular_transform_sl2z', 'theta_modular_transform',
    'dedekind_eta',
    'theta_quadratic_form', 'sum_of_squares',
    'jacobi_triple_product',
    'eisenstein_series', 'j_invariant',
    'theta_null_via_agm',
]


# ======================================================================
# Jacobi theta functions
# ======================================================================

def theta1(z: complex, tau: complex, terms: int = 50) -> complex:
    """Jacobi theta function θ₁(z|τ).
    
    θ₁(z|τ) = -i Σ (-1)^n q^{(n+1/2)^2} e^{2πi(n+1/2)z}
    """
    q = cmath.exp(1j * math.pi * tau)
    result = 0
    for n in range(-terms, terms + 1):
        n_half = n + 0.5
        result += (-1)**n * q**(n_half**2) * cmath.exp(2j * math.pi * n_half * z)
    return -1j * result


def theta2(z: complex, tau: complex, terms: int = 50) -> complex:
    """Jacobi theta function θ₂(z|τ).
    
    θ₂(z|τ) = Σ q^{(n+1/2)^2} e^{2πi(n+1/2)z}
    """
    q = cmath.exp(1j * math.pi * tau)
    result = 0
    for n in range(-terms, terms + 1):
        n_half = n + 0.5
        result += q**(n_half**2) * cmath.exp(2j * math.pi * n_half * z)
    return result


def theta3(z: complex, tau: complex, terms: int = 50) -> complex:
    """Jacobi theta function θ₃(z|τ).
    
    θ₃(z|τ) = Σ q^{n^2} e^{2πi n z}
    """
    q = cmath.exp(1j * math.pi * tau)
    result = 0
    for n in range(-terms, terms + 1):
        result += q**(n**2) * cmath.exp(2j * math.pi * n * z)
    return result


def theta4(z: complex, tau: complex, terms: int = 50) -> complex:
    """Jacobi theta function θ₄(z|τ).
    
    θ₄(z|τ) = Σ (-1)^n q^{n^2} e^{2πi n z}
    """
    q = cmath.exp(1j * math.pi * tau)
    result = 0
    for n in range(-terms, terms + 1):
        result += (-1)**n * q**(n**2) * cmath.exp(2j * math.pi * n * z)
    return result


def theta_null(tau: complex, terms: int = 50) -> Tuple[complex, complex, complex, complex]:
    """Return θ₁(0|τ), θ₂(0|τ), θ₃(0|τ), θ₄(0|τ)."""
    return (theta1(0, tau, terms), theta2(0, tau, terms),
            theta3(0, tau, terms), theta4(0, tau, terms))


# ======================================================================
# Modular transformations
# ======================================================================

def modular_transform_sl2z(tau: complex, a: int, b: int, c: int, d: int) -> complex:
    """Apply SL(2,Z) transformation: τ' = (aτ + b)/(cτ + d)."""
    if abs(c * tau + d) < 1e-15:
        raise ValueError("Transformation undefined")
    return (a * tau + b) / (c * tau + d)


def theta_modular_transform(theta_func, z: complex, tau: complex, 
                             a: int, b: int, c: int, d: int) -> complex:
    """Transform theta function under SL(2,Z).
    
    This is a simplified version; full transformation requires
    multiplier systems and square root branches.
    """
    tau_prime = modular_transform_sl2z(tau, a, b, c, d)
    # Simplified: just evaluate at transformed tau
    # Full transformation involves sqrt(c*tau+d) factors
    return theta_func(z, tau_prime)


# ======================================================================
# Dedekind eta function
# ======================================================================

def dedekind_eta(tau: complex, terms: int = 100) -> complex:
    """Dedekind eta function: η(τ) = q^{1/24} ∏ (1 - q^n)."""
    q = cmath.exp(2j * math.pi * tau)
    result = q**(1/24)
    for n in range(1, terms + 1):
        result *= (1 - q**n)
    return result


# ======================================================================
# Theta functions of quadratic forms
# ======================================================================

def theta_quadratic_form(A: np.ndarray, z: complex, terms: int = 10) -> complex:
    """Theta function of quadratic form Q(x) = x^T A x.
    
    Θ_A(z) = Σ_{x∈Z^n} e^{πi z Q(x)}
    """
    n = A.shape[0]
    result = 0
    
    # Sum over a box
    for x_int in np.ndindex(*([2*terms+1]*n)):
        x = np.array([xi - terms for xi in x_int])
        q = x @ A @ x
        result += cmath.exp(1j * math.pi * z * q)
    
    return result


def sum_of_squares(k: int, N: int) -> List[int]:
    """Number of ways to write n as sum of k squares, for n=0..N."""
    r = [0] * (N + 1)
    r[0] = 1  # 0 = 0^2 + ... + 0^2
    
    # Dynamic programming
    for _ in range(k):
        new_r = [0] * (N + 1)
        for n in range(N + 1):
            if r[n] > 0:
                # x = 0
                new_r[n] += r[n]
                # x > 0 (each non-zero square has 2 choices: +x and -x)
                for x in range(1, int(math.sqrt(N - n)) + 1):
                    if n + x*x <= N:
                        new_r[n + x*x] += 2 * r[n]
        r = new_r
    
    return r


def jacobi_triple_product(q: complex, z: complex, terms: int = 50) -> complex:
    """Jacobi triple product identity: θ₁(z|τ) = 2q^{1/4} sin(πz) ∏ (1 - q^{2n})..."""
    result = 2 * q**(1/4) * cmath.sin(math.pi * z)
    for n in range(1, terms + 1):
        q2n = q**(2*n)
        cos_term = cmath.cos(2 * math.pi * z)
        result *= (1 - q2n) * (1 - 2*q2n*cos_term + q2n**2)
    return result


# ======================================================================
# Modular forms
# ======================================================================

def eisenstein_series(k: int, tau: complex, terms: int = 100) -> complex:
    """Eisenstein series G_k(τ) for even k >= 4."""
    if k % 2 != 0 or k < 4:
        raise ValueError("k must be even >= 4")
    
    result = 0
    for m in range(-terms, terms + 1):
        for n in range(-terms, terms + 1):
            if m == 0 and n == 0:
                continue
            result += 1 / (m + n * tau)**k
    return result


def j_invariant(tau: complex, terms: int = 50) -> complex:
    """j-invariant: j(τ) = 1728 G₄(τ)³ / (G₄(τ)³ - 27 G₆(τ)²)."""
    G4 = eisenstein_series(4, tau, terms)
    G6 = eisenstein_series(6, tau, terms)
    return 1728 * G4**3 / (G4**3 - 27 * G6**2)


# ======================================================================
# Fast evaluation using AGM
# ======================================================================

def theta_null_via_agm(tau: complex, max_iter: int = 20) -> complex:
    """Compute θ₃(0|τ) using AGM.
    
    For τ = i t (t > 0): θ₃(0|i t) = √(2K/π) where K = K(k) with k = θ₂²/θ₃².
    """
    # This is a placeholder for the AGM-based computation
    # The full implementation uses the AGM to compute elliptic integrals
    return theta3(0, tau, 100)


if __name__ == "__main__":
    print("=== Theta Functions Demo ===")
    
    # Theta null values
    print("\n1. Theta null values:")
    for tau in [1j, 2j, 0.5 + 1j]:
        t1, t2, t3, t4 = theta_null(tau)
        print(f"   τ={tau}: θ₁(0)={t1:.6f}, θ₂(0)={t2:.6f}, θ₃(0)={t3:.6f}, θ₄(0)={t4:.6f}")
    
    # Sum of squares
    print("\n2. Sums of 4 squares:")
    r4 = sum_of_squares(4, 20)
    for n in range(1, 11):
        print(f"   r₄({n}) = {r4[n]}")
    
    # Dedekind eta
    print("\n3. Dedekind eta:")
    for tau in [1j, 2j]:
        eta = dedekind_eta(tau)
        print(f"   η({tau}) = {eta:.6f}")
    
    # Eisenstein series
    print("\n4. Eisenstein series:")
    for tau in [1j, 2j]:
        G4 = eisenstein_series(4, tau, 50)
        G6 = eisenstein_series(6, tau, 50)
        j = j_invariant(tau, 50)
        print(f"   τ={tau}: G₄={G4:.6f}, G₆={G6:.6f}, j={j:.6f}")