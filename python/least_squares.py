"""Least squares and Gauss-Newton optimization."""

import numpy as np
import math
from typing import Callable, Tuple, List
from python.gaussian_elim import qr_solve, ridge_regression

__all__ = [
    'least_squares_normal', 'least_squares_qr', 'weighted_least_squares', 'ridge_regression',
    'gauss_newton', 'levenberg_marquardt',
    'fit_exponential',
    'polyfit', 'polyval',
    'ceres_orbit',
]


# ======================================================================
# Linear Least Squares
# ======================================================================

def least_squares_normal(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve min ||Ax - b|| using normal equations."""
    return np.linalg.solve(A.T @ A, A.T @ b)


def least_squares_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve min ||Ax - b|| using QR decomposition."""
    return qr_solve(A, b)


def weighted_least_squares(A: np.ndarray, b: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Solve min ||W^{1/2}(Ax - b)||^2."""
    sqrtW = np.sqrt(W)
    return qr_solve(sqrtW[:, np.newaxis] * A, sqrtW * b)


def ridge_regression(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    """Tikhonov regularization: min ||Ax - b||^2 + λ||x||^2."""
    n = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(n), A.T @ b)


# ======================================================================
# Nonlinear Least Squares
# ======================================================================

def gauss_newton(
    f: Callable[[np.ndarray], np.ndarray],
    jac: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10
) -> Tuple[np.ndarray, List[float]]:
    """Gauss-Newton algorithm for nonlinear least squares.
    
    Minimize ||f(x)||^2.
    
    Returns:
        x_opt: Optimal parameters
        history: Residual norms at each iteration
    """
    x = x0.copy()
    history = []
    
    for i in range(max_iter):
        fx = f(x)
        J = jac(x)
        
        # Solve J^T J dx = -J^T f
        dx = np.linalg.lstsq(J, -fx, rcond=None)[0]
        x = x + dx
        
        residual_norm = np.linalg.norm(fx)
        history.append(residual_norm)
        
        if np.linalg.norm(dx) < tol:
            break
    
    return x, history


def levenberg_marquardt(
    f: Callable[[np.ndarray], np.ndarray],
    jac: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lam: float = 1e-3,
    max_iter: int = 100,
    tol: float = 1e-10
) -> Tuple[np.ndarray, List[float]]:
    """Levenberg-Marquardt algorithm.
    
    Minimize ||f(x)||^2.
    """
    x = x0.copy()
    history = []
    
    for i in range(max_iter):
        fx = f(x)
        J = jac(x)
        
        # (J^T J + λ diag(J^T J)) dx = -J^T f
        JTJ = J.T @ J
        Jtf = J.T @ fx
        
        diag_JTJ = np.diag(JTJ)
        A = JTJ + lam * np.diag(diag_JTJ)
        
        dx = np.linalg.solve(A, -Jtf)
        x_new = x + dx
        
        fx_new = f(x_new)
        residual_old = np.linalg.norm(fx)**2
        residual_new = np.linalg.norm(fx_new)**2
        
        if residual_new < residual_old:
            x = x_new
            lam *= 0.1
        else:
            lam *= 10
        
        history.append(math.sqrt(residual_old))
        
        if np.linalg.norm(dx) < tol:
            break
    
    return x, history


# ======================================================================
# Example: Exponential fitting
# ======================================================================

def fit_exponential(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit y = a * exp(b*x) + c using Levenberg-Marquardt."""
    def f(params):
        a, b, c = params
        return a * np.exp(b * x) + c - y
    
    def jac(params):
        a, b, c = params
        J = np.zeros((len(x), 3))
        J[:, 0] = np.exp(b * x)
        J[:, 1] = a * x * np.exp(b * x)
        J[:, 2] = 1
        return J
    
    x0 = np.array([1.0, 0.1, 0.0])
    params, _ = levenberg_marquardt(f, jac, x0)
    return params[0], params[1], params[2]


# ======================================================================
# Polynomial fitting
# ======================================================================

def polyfit(x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
    """Polynomial fit using QR (more stable than normal equations)."""
    A = np.vander(x, deg + 1)
    return qr_solve(A, y)


def polyval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate polynomial."""
    return np.polyval(coeffs, x)


# ======================================================================
# Gauss's Ceres orbit determination (placeholder)
# ======================================================================

def ceres_orbit(observations: List[Tuple[float, float, float]]) -> dict:
    """Gauss's method for orbit determination from 3 observations.
    
    Args:
        observations: List of (RA, Dec, time) tuples
        
    Returns:
        Dictionary with orbital elements
    """
    raise NotImplementedError("Full Gauss orbit determination requires spherical astronomy")


if __name__ == "__main__":
    print("=== Least Squares Demo ===")
    
    # Linear least squares
    print("\n1. Linear least squares:")
    A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=float)
    b = np.array([2.1, 3.9, 6.2, 7.8], dtype=float)
    x = least_squares_qr(A, b)
    print(f"   y = {x[0]:.4f} + {x[1]:.4f} * x")
    
    # Ridge regression
    print("\n2. Ridge regression (λ=0.1):")
    x_ridge = ridge_regression(A, b, 0.1)
    print(f"   y = {x_ridge[0]:.4f} + {x_ridge[1]:.4f} * x")
    
    # Exponential fitting
    print("\n3. Exponential fitting:")
    x_data = np.array([0, 1, 2, 3, 4])
    y_data = np.array([1.0, 2.7, 7.4, 20.1, 54.6])  # ~ exp(x)
    a, b, c = fit_exponential(x_data, y_data)
    print(f"   y = {a:.4f} * exp({b:.4f} * x) + {c:.4f}")
    
    # Polynomial fitting
    print("\n4. Polynomial fitting:")
    x = np.linspace(0, 10, 10)
    y = 2*x**2 + 3*x + 1 + 0.1*np.random.randn(10)
    coeffs = polyfit(x, y, 2)
    print(f"   Coeffs: {coeffs}")