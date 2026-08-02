"""Tests for Gaussian elimination module."""

import pytest
import sys
import numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.gaussian_elim import (
    gaussian_elimination, lu_decomposition, lu_solve,
    cholesky, cholesky_solve, qr_decomposition, qr_solve,
    matrix_inverse, condition_number,
    least_squares_normal, least_squares_qr, ridge_regression,
    bareiss
)


class TestGaussianElimination:
    def test_simple_system(self):
        A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
        b = np.array([8, -11, -3], dtype=float)
        x = gaussian_elimination(A, b)
        expected = np.array([2, 3, -1])
        np.testing.assert_allclose(x, expected, rtol=1e-10)
    
    def test_singular_raises(self):
        A = np.array([[1, 2], [2, 4]], dtype=float)
        b = np.array([1, 2], dtype=float)
        with pytest.raises(np.linalg.LinAlgError):
            gaussian_elimination(A, b)


class TestLUDecomposition:
    def test_decomposition(self):
        A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
        P, L, U = lu_decomposition(A)
        np.testing.assert_allclose(P @ A, L @ U, rtol=1e-10)
        # Check L is unit lower triangular
        np.testing.assert_allclose(np.diag(L), 1)
        # Check U is upper triangular
        assert np.allclose(U[np.tril_indices_from(U, -1)], 0)
    
    def test_solve(self):
        A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
        b = np.array([8, -11, -3], dtype=float)
        P, L, U = lu_decomposition(A)
        x = lu_solve(P, L, U, b)
        expected = np.array([2, 3, -1])
        np.testing.assert_allclose(x, expected, rtol=1e-10)


class TestCholesky:
    def test_spd(self):
        A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=float)
        L = cholesky(A)
        np.testing.assert_allclose(L @ L.T, A, rtol=1e-10)
        # L should be lower triangular
        assert np.allclose(L[np.triu_indices_from(L, 1)], 0)
    
    def test_solve(self):
        A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=float)
        b = np.array([1, 2, 3], dtype=float)
        L = cholesky(A)
        x = cholesky_solve(L, b)
        expected = np.linalg.solve(A, b)
        np.testing.assert_allclose(x, expected, rtol=1e-10)
    
    def test_non_spd_raises(self):
        A = np.array([[1, 2], [2, 1]], dtype=float)
        with pytest.raises(np.linalg.LinAlgError):
            cholesky(A)


class TestQR:
    def test_decomposition(self):
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        Q, R = qr_decomposition(A)
        np.testing.assert_allclose(Q @ R, A, rtol=1e-10)
        # Q orthogonal
        np.testing.assert_allclose(Q.T @ Q, np.eye(3), rtol=1e-10, atol=1e-10)
        # R upper triangular
        assert np.allclose(R[np.tril_indices_from(R, -1)], 0)
    
    def test_solve(self):
        A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=float)
        b = np.array([2.1, 3.9, 6.2, 7.8], dtype=float)
        x = qr_solve(A, b)
        expected = np.array([0.15, 1.94])
        np.testing.assert_allclose(x, expected, rtol=1e-4)


class TestMatrixInverse:
    def test_inverse(self):
        A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
        Ainv = matrix_inverse(A)
        np.testing.assert_allclose(A @ Ainv, np.eye(3), rtol=1e-10, atol=1e-10)


class TestConditionNumber:
    def test_identity(self):
        A = np.eye(3)
        assert condition_number(A) == 1.0
    
    def test_ill_conditioned(self):
        A = np.array([[1, 1], [1, 1.0001]], dtype=float)
        cond = condition_number(A)
        assert cond > 1e4


class TestLeastSquares:
    def test_normal_vs_qr(self):
        A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=float)
        b = np.array([2.1, 3.9, 6.2, 7.8], dtype=float)
        x_normal = least_squares_normal(A, b)
        x_qr = least_squares_qr(A, b)
        np.testing.assert_allclose(x_normal, x_qr, rtol=1e-6)
    
    def test_ridge_regression(self):
        A = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
        b = np.array([1, 2, 3], dtype=float)
        x = ridge_regression(A, b, 0.1)
        assert len(x) == 2


class TestBareiss:
    def test_integer_matrix(self):
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)
        U, det = bareiss(A)
        # Should give upper triangular with integer entries
        assert abs(det - (-3)) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])