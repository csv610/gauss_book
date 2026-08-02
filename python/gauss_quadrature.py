"""Gaussian quadrature rules."""

import numpy as np
from typing import Tuple, List, Callable
import math

__all__ = [
    'legendre_recurrence', 'hermite_recurrence', 'laguerre_recurrence', 'chebyshev_recurrence',
    'golub_welsch',
    'gauss_legendre', 'gauss_hermite', 'gauss_laguerre', 'gauss_chebyshev',
    'integrate', 'integrate_infinite',
    'gauss_kronrod', 'integrate_adaptive',
]


# ======================================================================
# Classical orthogonal polynomial recurrence coefficients
# ======================================================================

def legendre_recurrence(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Recurrence coefficients for monic Legendre polynomials.
    
    Returns alpha (diagonal) and beta (sub-diagonal) for Jacobi matrix.
    """
    alpha = np.zeros(n)
    beta = np.zeros(n)
    beta[0] = 2.0  # mu_0
    for k in range(1, n):
        beta[k] = k**2 / (4*k**2 - 1)
    return alpha, beta


def hermite_recurrence(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Recurrence coefficients for monic Hermite polynomials."""
    alpha = np.zeros(n)
    beta = np.zeros(n)
    beta[0] = np.sqrt(np.pi)  # mu_0
    for k in range(1, n):
        beta[k] = k / 2
    return alpha, beta


def laguerre_recurrence(n: int, alpha: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Recurrence coefficients for monic generalized Laguerre polynomials."""
    a = np.zeros(n)
    b = np.zeros(n)
    b[0] = math.gamma(alpha + 1)  # mu_0
    for k in range(n):
        a[k] = 2*k + 1 + alpha
    for k in range(1, n):
        b[k] = k * (k + alpha)
    return a, b


def chebyshev_recurrence(n: int, kind: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Recurrence coefficients for monic Chebyshev polynomials.
    
    kind=1: T_n (weight 1/sqrt(1-x^2))
    kind=2: U_n (weight sqrt(1-x^2))
    """
    a = np.zeros(n)
    b = np.zeros(n)
    
    if kind == 1:
        b[0] = np.pi  # mu_0
        if n > 1:
            b[1] = 0.5
        for k in range(2, n):
            b[k] = 0.25
    elif kind == 2:
        b[0] = np.pi / 2  # mu_0
        for k in range(1, n):
            b[k] = 0.25
            
    return a, b


# ======================================================================
# Golub-Welsch algorithm
# ======================================================================

def golub_welsch(alpha: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute nodes and weights for Gaussian quadrature from recurrence coefficients.
    
    Args:
        alpha: Diagonal of Jacobi matrix
        beta: Sub-diagonal squared (beta[0] = int w(x) dx)
        
    Returns:
        nodes: Quadrature points
        weights: Quadrature weights
    """
    n = len(alpha)
    # Build Jacobi matrix
    J = np.diag(alpha) + np.diag(np.sqrt(beta[1:]), 1) + np.diag(np.sqrt(beta[1:]), -1)
    
    # Eigenvalues = nodes, first components of eigenvectors^2 * mu0 = weights
    eigvals, eigvecs = np.linalg.eigh(J)
    mu0 = beta[0]  # Integral of weight function
    
    nodes = eigvals
    weights = mu0 * eigvecs[0, :]**2
    
    return nodes, weights


# ======================================================================
# Specific quadrature rules
# ======================================================================

def gauss_legendre(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre quadrature on [-1, 1] with weight 1."""
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    elif n == 2:
        return np.array([-1/np.sqrt(3), 1/np.sqrt(3)]), np.array([1.0, 1.0])
    elif n == 3:
        return np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)]), np.array([5/9, 8/9, 5/9])
    else:
        # Use Golub-Welsch for larger n
        alpha, beta = legendre_recurrence(n)
        return golub_welsch(alpha, beta)


def gauss_hermite(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite quadrature on (-∞, ∞) with weight e^{-x^2}."""
    if n == 1:
        return np.array([0.0]), np.array([np.sqrt(np.pi)])
    elif n == 2:
        return np.array([-1/np.sqrt(2), 1/np.sqrt(2)]), np.array([np.sqrt(np.pi)/2, np.sqrt(np.pi)/2])
    else:
        alpha, beta = hermite_recurrence(n)
        return golub_welsch(alpha, beta)


def gauss_laguerre(n: int, alpha: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Laguerre quadrature on [0, ∞) with weight x^alpha e^{-x}."""
    a, b = laguerre_recurrence(n, alpha)
    return golub_welsch(a, b)


def gauss_chebyshev(n: int, kind: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Chebyshev quadrature on [-1, 1].
    
    kind=1: weight 1/sqrt(1-x^2)
    kind=2: weight sqrt(1-x^2)
    """
    if kind == 1:
        # Nodes: cos((2k-1)π/(2n))
        nodes = np.cos((2*np.arange(1, n+1) - 1) * np.pi / (2*n))
        weights = np.pi / n * np.ones(n)
        return nodes, weights
    else:
        # Nodes: cos(kπ/(n+1))
        nodes = np.cos(np.arange(1, n+1) * np.pi / (n + 1))
        weights = np.pi / (n + 1) * np.sin(np.arange(1, n+1) * np.pi / (n + 1))**2
        return nodes, weights


# ======================================================================
# Integration routines
# ======================================================================

def integrate(f: Callable[[float], float], a: float, b: float, 
              n: int = 10, rule: str = 'legendre') -> float:
    """Gaussian quadrature on [a, b]."""
    if rule == 'legendre':
        nodes, weights = gauss_legendre(n)
    elif rule == 'chebyshev':
        nodes, weights = gauss_chebyshev(n, kind=1)
    else:
        raise ValueError(f"Unknown rule: {rule}")
    
    # Map from [-1, 1] to [a, b]
    x = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    w = 0.5 * (b - a) * weights
    
    return np.sum(w * np.array([f(xi) for xi in x]))


def integrate_infinite(f: Callable[[float], float], 
                       n: int = 10, rule: str = 'hermite') -> float:
    """Gaussian quadrature on (-∞, ∞) for an arbitrary function f(x)."""
    if rule == 'hermite':
        nodes, weights = gauss_hermite(n)
        return np.sum(weights * np.array([f(xi) * math.exp(xi**2) for xi in nodes]))
    else:
        raise ValueError(f"Unknown rule for infinite interval: {rule}")


# ======================================================================
# Gauss-Kronrod extension (error estimation)
# ======================================================================

def gauss_kronrod(n: int = 7) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Gauss-Legendre and Kronrod nodes/weights for error estimation.
    
    Returns:
        g_nodes, g_weights, k_nodes, k_weights
    """
    # 7-point Gauss + 15-point Kronrod (standard)
    # Hardcoded for common case
    if n == 7:
        g_nodes = np.array([
            -0.9491079123427585, -0.7415311855993944, -0.4058451513773972, 0.0,
            0.4058451513773972, 0.7415311855993944, 0.9491079123427585
        ])
        g_weights = np.array([
            0.1294849661688697, 0.2797053914892767, 0.3818300505051189, 0.4179591836734694,
            0.3818300505051189, 0.2797053914892767, 0.1294849661688697
        ])
        k_nodes = np.array([
            -0.9914553711208126, -0.9491079123427585, -0.8648644233597691, -0.7415311855993944,
            -0.5860872354676911, -0.4058451513773972, -0.2077849550078985, 0.0,
            0.2077849550078985, 0.4058451513773972, 0.5860872354676911, 0.7415311855993944,
            0.8648644233597691, 0.9491079123427585, 0.9914553711208126
        ])
        k_weights = np.array([
            0.0229353220105292, 0.0630920926299786, 0.1047900103222502, 0.1406532597155259,
            0.1690047266392679, 0.1903505780647854, 0.2044329400752989, 0.2094821410847278,
            0.2044329400752989, 0.1903505780647854, 0.1690047266392679, 0.1406532597155259,
            0.1047900103222502, 0.0630920926299786, 0.0229353220105292
        ])
        return g_nodes, g_weights, k_nodes, k_weights
    else:
        raise NotImplementedError("Only n=7 Gauss-Kronrod implemented")


def integrate_adaptive(f: Callable[[float], float], a: float, b: float,
                       tol: float = 1e-10, max_depth: int = 20) -> Tuple[float, float]:
    """Adaptive Gauss-Kronrod quadrature with error estimate."""
    
    def quad_recursive(f, a, b, whole, depth):
        g_nodes, g_weights, k_nodes, k_weights = gauss_kronrod()
        
        # Evaluate Gauss rule
        g_x = 0.5 * (b - a) * g_nodes + 0.5 * (b + a)
        g_val = np.sum(g_weights * np.array([f(xi) for xi in g_x])) * 0.5 * (b - a)
        
        # Evaluate Kronrod rule
        k_x = 0.5 * (b - a) * k_nodes + 0.5 * (b + a)
        k_val = np.sum(k_weights * np.array([f(xi) for xi in k_x])) * 0.5 * (b - a)
        
        error = abs(k_val - g_val)
        
        if depth <= 0 or error < tol:
            return k_val, error
        
        mid = (a + b) / 2
        left_val, left_err = quad_recursive(f, a, mid, g_val/2, depth - 1)
        right_val, right_err = quad_recursive(f, mid, b, g_val/2, depth - 1)
        
        return left_val + right_val, left_err + right_err
    
    g_nodes, g_weights, _, _ = gauss_kronrod()
    g_x = 0.5 * (b - a) * g_nodes + 0.5 * (b + a)
    whole = np.sum(g_weights * np.array([f(xi) for xi in g_x])) * 0.5 * (b - a)
    
    return quad_recursive(f, a, b, whole, max_depth)


if __name__ == "__main__":
    print("=== Gaussian Quadrature Demo ===")
    
    # Gauss-Legendre
    print("\nGauss-Legendre nodes and weights:")
    for n in range(1, 6):
        nodes, weights = gauss_legendre(n)
        print(f"  n={n}: nodes={nodes}, weights={weights}")
    
    # Test integration
    print("\nIntegration tests:")
    exact = 2.0
    for n in [2, 3, 4, 5, 10]:
        nodes, weights = gauss_legendre(n)
        x = nodes
        approx = np.sum(weights * x**8)
        print(f"  n={n}: int x^8 dx = {approx:.10f} (exact={exact}) err={abs(approx-exact):.2e}")
    
    # Gauss-Hermite
    print("\nGauss-Hermite (weight e^{-x^2}):")
    for n in [2, 3, 4]:
        nodes, weights = gauss_hermite(n)
        # Test: int x^2 e^{-x^2} dx = sqrt(pi)/2
        approx = np.sum(weights * nodes**2)
        exact = np.sqrt(np.pi)/2
        print(f"  n={n}: int x^2 e^(-x^2) dx = {approx:.10f} (exact={exact:.10f})")
    
    # Adaptive integration
    print("\nAdaptive integration:")
    val, err = integrate_adaptive(lambda x: np.exp(-x**2), 0, 1)
    print(f"  int_0^1 exp(-x^2) dx = {val:.10f} ± {err:.2e}")
    print(f"  Exact: {np.sqrt(np.pi)/2 * math.erf(1):.10f}")