"""Tests for least squares module."""

import pytest
import sys
import numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.least_squares import (
    least_squares_normal, least_squares_qr,
    weighted_least_squares, ridge_regression,
    gauss_newton, levenberg_marquardt,
    fit_exponential, polyfit, polyval
)


class TestLinearLeastSquares:
    def test_normal_equations(self):
        A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=float)
        b = np.array([2.1, 3.9, 6.2, 7.8], dtype=float)
        x = least_squares_normal(A, b)
        # Actual fit: y = 0.15 + 1.94x
        assert abs(x[0] - 0.15) < 0.1
        assert abs(x[1] - 1.94) < 0.1
    
    def test_qr(self):
        A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=float)
        b = np.array([2.1, 3.9, 6.2, 7.8], dtype=float)
        x = least_squares_qr(A, b)
        # Actual fit: y = 0.15 + 1.94x
        assert abs(x[0] - 0.15) < 0.1
        assert abs(x[1] - 1.94) < 0.1
    
    def test_weighted(self):
        A = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
        b = np.array([2, 4, 6], dtype=float)
        W = np.array([1, 1, 100])  # Heavy weight on third point
        x = weighted_least_squares(A, b, W)
        # Should pass close to third point
        assert abs(A[2] @ x - 6) < 0.1
    
    def test_ridge(self):
        A = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
        b = np.array([2, 4, 6], dtype=float)
        x = ridge_regression(A, b, 1.0)
        assert len(x) == 2


class TestNonlinearLeastSquares:
    def test_gauss_newton(self):
        # Fit line: f(x) = a*x + b - y
        x_data = np.array([1, 2, 3, 4])
        y_data = np.array([2.1, 3.9, 6.2, 7.8])
        
        def f(params):
            a, b = params
            return a * x_data + b - y_data
        
        def jac(params):
            a, b = params
            J = np.column_stack([x_data, np.ones_like(x_data)])
            return J
        
        x0 = np.array([0.0, 0.0])
        x_opt, history = gauss_newton(f, jac, x0)
        
        # Actual fit: slope=1.94, intercept=0.15
        assert abs(x_opt[0] - 1.94) < 0.1
        assert abs(x_opt[1] - 0.15) < 0.1
        assert len(history) > 0
    
    def test_levenberg_marquardt(self):
        # Same test as Gauss-Newton
        x_data = np.array([1, 2, 3, 4])
        y_data = np.array([2.1, 3.9, 6.2, 7.8])
        
        def f(params):
            a, b = params
            return a * x_data + b - y_data
        
        def jac(params):
            a, b = params
            return np.column_stack([x_data, np.ones_like(x_data)])
        
        x0 = np.array([0.0, 0.0])
        x_opt, history = levenberg_marquardt(f, jac, x0)
        
        assert abs(x_opt[0] - 1.94) < 0.1
        assert abs(x_opt[1] - 0.15) < 0.1


class TestExponentialFit:
    def test_fit_exponential(self):
        x = np.array([0, 1, 2, 3, 4])
        y = np.exp(x)  # Perfect exponential
        a, b, c = fit_exponential(x, y)
        assert abs(a - 1.0) < 0.1
        assert abs(b - 1.0) < 0.1
        assert abs(c - 0.0) < 0.1


class TestPolyfit:
    def test_polyfit(self):
        x = np.linspace(0, 10, 20)
        y = 2*x**2 + 3*x + 1 + 0.1*np.random.randn(20)
        coeffs = polyfit(x, y, 2)
        # Should be close to [2, 3, 1]
        assert abs(coeffs[0] - 2) < 0.2
        assert abs(coeffs[1] - 3) < 0.2
        assert abs(coeffs[2] - 1) < 0.2
    
    def test_polyval(self):
        coeffs = np.array([2, 3, 1])
        x = np.array([0, 1, 2])
        y = polyval(coeffs, x)
        expected = 2*x**2 + 3*x + 1
        np.testing.assert_allclose(y, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])