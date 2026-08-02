"""Gaussian elimination with LU decomposition and linear system solving."""

import numpy as np
from typing import Tuple, List, Optional
import math

__all__ = [
    'gaussian_elimination',
    'lu_decomposition', 'lu_solve',
    'cholesky', 'cholesky_solve',
    'qr_decomposition', 'qr_solve',
    'matrix_inverse', 'condition_number',
    'least_squares_normal', 'least_squares_qr', 'ridge_regression',
    'bareiss',
]


# ======================================================================
# Gaussian Elimination with Partial Pivoting
# ======================================================================

def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b using Gaussian elimination with partial pivoting.
    
    Modifies A and b in place. Returns solution x.
    """
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    n = A.shape[0]
    
    # Forward elimination
    for k in range(n - 1):
        # Partial pivoting
        pivot_row = k + np.argmax(np.abs(A[k:, k]))
        if pivot_row != k:
            A[[k, pivot_row]] = A[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]
        
        if abs(A[k, k]) < 1e-15:
            raise np.linalg.LinAlgError("Matrix is singular or nearly singular")
        
        # Eliminate below
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
    
    # Check for singularity
    for i in range(n):
        if abs(A[i, i]) < 1e-15:
            raise np.linalg.LinAlgError("Matrix is singular or nearly singular")
            
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    
    return x


# ======================================================================
# LU Decomposition with Partial Pivoting
# ======================================================================

def lu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LU decomposition with partial pivoting: PA = LU.
    
    Returns:
        P: Permutation matrix
        L: Lower triangular with unit diagonal
        U: Upper triangular
    """
    A = A.copy().astype(float)
    n = A.shape[0]
    P = np.eye(n)
    L = np.eye(n)
    U = A.copy()
    
    for k in range(n - 1):
        # Partial pivoting
        pivot_row = k + np.argmax(np.abs(U[k:, k]))
        if pivot_row != k:
            U[[k, pivot_row]] = U[[pivot_row, k]]
            P[[k, pivot_row]] = P[[pivot_row, k]]
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
        
        if abs(U[k, k]) < 1e-15:
            raise np.linalg.LinAlgError("Matrix is singular")
        
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]
    
    return P, L, U


def lu_solve(P: np.ndarray, L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve PAx = b given PA = LU."""
    n = L.shape[0]
    
    # Forward substitution: Ly = Pb
    Pb = P @ b
    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])
    
    # Back substitution: Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    return x


# ======================================================================
# Cholesky Decomposition
# ======================================================================

def cholesky(A: np.ndarray) -> np.ndarray:
    """Cholesky decomposition A = LL^T for symmetric positive definite A.
    
    Returns lower triangular L.
    """
    A = A.copy().astype(float)
    n = A.shape[0]
    L = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                val = A[i, i] - s
                if val <= 0:
                    raise np.linalg.LinAlgError("Matrix not positive definite")
                L[i, j] = math.sqrt(val)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    
    return L


def cholesky_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve LL^T x = b."""
    n = L.shape[0]
    
    # Forward: Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    
    # Back: L^T x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(L[i+1:, i], x[i+1:])) / L[i, i]
    
    return x


# ======================================================================
# QR Decomposition (Householder)
# ======================================================================

def qr_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """QR decomposition using Householder reflections: A = QR.
    
    Returns:
        Q: m x m orthogonal matrix
        R: m x n upper triangular (or n x n if m >= n)
    """
    A = A.copy().astype(float)
    m, n = A.shape
    Q = np.eye(m)
    
    for k in range(min(m, n)):
        x = A[k:, k]
        norm_x = np.linalg.norm(x)
        if norm_x == 0:
            continue
        
        sign = -1 if x[0] >= 0 else 1
        v = x.copy()
        v[0] += sign * norm_x
        v = v / np.linalg.norm(v)
        
        # Apply to A
        A[k:, k:] -= 2 * np.outer(v, v @ A[k:, k:])
        # Apply to Q
        Q[k:, :] -= 2 * np.outer(v, v @ Q[k:, :])
    
    R = np.triu(A)
    return Q.T, R


def qr_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve min ||Ax - b|| using QR decomposition."""
    Q, R = qr_decomposition(A)
    m, n = A.shape
    Qb = Q.T @ b
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Qb[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
    return x


# ======================================================================
# Matrix Inverse
# ======================================================================

def matrix_inverse(A: np.ndarray) -> np.ndarray:
    """Invert matrix using LU decomposition."""
    n = A.shape[0]
    P, L, U = lu_decomposition(A)
    inv = np.zeros((n, n))
    
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        inv[:, i] = lu_solve(P, L, U, e)
    
    return inv


# ======================================================================
# Condition Number
# ======================================================================

def condition_number(A: np.ndarray, norm: str = '2') -> float:
    """Estimate condition number."""
    if norm == '2':
        s = np.linalg.svd(A, compute_uv=False)
        return s[0] / s[-1] if s[-1] > 0 else float('inf')
    elif norm == '1':
        return np.linalg.norm(A, 1) * np.linalg.norm(matrix_inverse(A), 1)
    elif norm == 'inf':
        return np.linalg.norm(A, np.inf) * np.linalg.norm(matrix_inverse(A), np.inf)
    else:
        raise ValueError("Norm must be '1', '2', or 'inf'")


# ======================================================================
# Least Squares
# ======================================================================

def least_squares_normal(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve min ||Ax - b|| using normal equations."""
    return np.linalg.solve(A.T @ A, A.T @ b)


def least_squares_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve min ||Ax - b|| using QR decomposition."""
    return qr_solve(A, b)


def ridge_regression(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    """Tikhonov regularization: min ||Ax - b||^2 + λ||x||^2."""
    n = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(n), A.T @ b)


# ======================================================================
# Bareiss Algorithm (Fraction-Free Gaussian Elimination)
# ======================================================================

def bareiss(A: np.ndarray) -> Tuple[np.ndarray, float]:
    """Bareiss algorithm: fraction-free Gaussian elimination.
    
    Returns upper triangular matrix and determinant.
    """
    A = A.copy().astype(float)
    n = A.shape[0]
    det = 1.0
    
    for k in range(n - 1):
        if A[k, k] == 0:
            # Find non-zero pivot
            for i in range(k + 1, n):
                if A[i, k] != 0:
                    A[[k, i]] = A[[i, k]]
                    det = -det
                    break
        
        if A[k, k] == 0:
            return A, 0.0
        
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i, j] = (A[i, j] * A[k, k] - A[i, k] * A[k, j])
                if k > 0:
                    A[i, j] /= A[k-1, k-1]
    
    det = A[n-1, n-1] if n > 0 else 1.0
    return A, det


if __name__ == "__main__":
    print("=== Gaussian Elimination Demo ===")
    
    # Test Gaussian elimination
    A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)
    x = gaussian_elimination(A, b)
    print(f"Solution: {x}")
    print(f"Check A*x = {A @ x}")
    
    # Test LU
    print("\nLU Decomposition:")
    P, L, U = lu_decomposition(A)
    print(f"P:\n{P}")
    print(f"L:\n{L}")
    print(f"U:\n{U}")
    print(f"P*A - L*U:\n{P @ A - L @ U}")
    
    # Test Cholesky
    print("\nCholesky Decomposition:")
    A_spd = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=float)
    L = cholesky(A_spd)
    print(f"L:\n{L}")
    print(f"L @ L.T:\n{L @ L.T}")
    
    # Test QR
    print("\nQR Decomposition:")
    A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    Q, R = qr_decomposition(A)
    print(f"Q:\n{Q}")
    print(f"R:\n{R}")
    print(f"Q @ R:\n{Q @ R}")
    
    # Test least squares
    print("\nLeast Squares:")
    A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=float)
    b = np.array([2.1, 3.9, 6.2, 7.8], dtype=float)
    x_qr = least_squares_qr(A, b)
    x_normal = least_squares_normal(A, b)
    print(f"QR solution: {x_qr}")
    print(f"Normal eq solution: {x_normal}")
    
    # Condition number
    print(f"\nCondition number: {condition_number(A):.2f}")