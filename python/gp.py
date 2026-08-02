"""Gaussian Processes and Kriging."""

import numpy as np
from typing import Callable, Tuple, List, Optional

__all__ = [
    'rbf_kernel', 'matern_kernel', 'periodic_kernel', 'rational_quadratic_kernel',
    'linear_kernel', 'white_kernel',
    'GaussianProcess',
    'simple_kriging', 'ordinary_kriging',
]


# ======================================================================
# Covariance functions (kernels)
# ======================================================================

def rbf_kernel(x1: np.ndarray, x2: np.ndarray, 
               length_scale: float = 1.0, variance: float = 1.0) -> np.ndarray:
    """Squared Exponential (RBF) kernel.
    
    k(x, x') = variance * exp(-||x - x'||^2 / (2 * length_scale^2))
    """
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    
    dist_sq = np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :])**2, axis=2)
    return variance * np.exp(-0.5 * dist_sq / length_scale**2)


def matern_kernel(x1: np.ndarray, x2: np.ndarray,
                  length_scale: float = 1.0, variance: float = 1.0,
                  nu: float = 1.5) -> np.ndarray:
    """Matérn kernel.
    
    nu = 0.5: Exponential
    nu = 1.5: Matern 3/2
    nu = 2.5: Matern 5/2
    nu -> inf: RBF
    """
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    
    dist = np.sqrt(np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :])**2, axis=2))
    dist = np.maximum(dist, 1e-10)
    r = np.sqrt(2 * nu) * dist / length_scale
    
    if nu == 0.5:
        # Exponential kernel
        return variance * np.exp(-r)
    elif nu == 1.5:
        # Matérn 3/2
        return variance * (1 + r) * np.exp(-r)
    elif nu == 2.5:
        # Matérn 5/2
        return variance * (1 + r + r**2 / 3) * np.exp(-r)
    else:
        # General case using modified Bessel function
        from scipy.special import kv, gamma
        coeff = variance * (2**(1-nu)) / gamma(nu)
        return coeff * r**nu * kv(nu, r)


def periodic_kernel(x1: np.ndarray, x2: np.ndarray,
                    length_scale: float = 1.0, variance: float = 1.0,
                    period: float = 1.0) -> np.ndarray:
    """Periodic kernel."""
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    
    dist = np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :])**2, axis=2)
    return variance * np.exp(-2 * np.sin(np.pi * np.sqrt(dist) / period)**2 / length_scale**2)


def rational_quadratic_kernel(x1: np.ndarray, x2: np.ndarray,
                              length_scale: float = 1.0, variance: float = 1.0,
                              alpha: float = 1.0) -> np.ndarray:
    """Rational Quadratic kernel."""
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    
    dist_sq = np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :])**2, axis=2)
    return variance * (1 + dist_sq / (2 * alpha * length_scale**2))**(-alpha)


def linear_kernel(x1: np.ndarray, x2: np.ndarray,
                  variance: float = 1.0, c: float = 0.0) -> np.ndarray:
    """Linear kernel."""
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    return variance * x1 @ x2.T + c


def white_kernel(x1: np.ndarray, x2: np.ndarray,
                 noise_level: float = 1.0) -> np.ndarray:
    """White noise kernel."""
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    return noise_level * (x1[:, np.newaxis, :] == x2[np.newaxis, :, :]).all(axis=2).astype(float)


# ======================================================================
# Gaussian Process Regression
# ======================================================================

class GaussianProcess:
    """Gaussian Process for regression."""
    
    def __init__(self, kernel: Callable, noise: float = 1e-6):
        """
        Args:
            kernel: Covariance function k(x1, x2, **params)
            noise: Observation noise standard deviation
        """
        self.kernel = kernel
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.L = None  # Cholesky factor
        self.alpha = None  # K^{-1} y
        self.kernel_params = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kernel_params) -> 'GaussianProcess':
        """Fit GP to training data."""
        self.X_train = np.atleast_2d(X)
        self.y_train = np.ravel(y)
        self.kernel_params = kernel_params
        
        K = self.kernel(self.X_train, self.X_train, **kernel_params)
        K = K + self.noise**2 * np.eye(len(self.X_train))
        
        try:
            self.L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Add jitter
            K = K + 1e-6 * np.eye(len(self.X_train))
            self.L = np.linalg.cholesky(K)
        
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_train))
        return self
    
    def predict(self, X_test: np.ndarray, return_std: bool = True, 
                **kernel_params) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict at test points."""
        X_test = np.atleast_2d(X_test)
        
        K_star = self.kernel(X_test, self.X_train, **kernel_params)
        mu = K_star @ self.alpha
        
        if return_std:
            v = np.linalg.solve(self.L, K_star.T)
            K_star_star = self.kernel(X_test, X_test, **kernel_params)
            var = np.diag(K_star_star) - np.sum(v**2, axis=0)
            var = np.maximum(var, 0)
            std = np.sqrt(var)
            return mu, std
        return mu, None
    
    def log_marginal_likelihood(self, **kernel_params) -> float:
        """Compute log marginal likelihood."""
        if self.L is None:
            return -np.inf
        
        n = len(self.y_train)
        log_det = 2 * np.sum(np.log(np.diag(self.L)))
        return -0.5 * self.y_train @ self.alpha - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                                  initial_params: dict,
                                  bounds: dict,
                                  method: str = 'L-BFGS-B') -> dict:
        """Optimize hyperparameters by maximizing log marginal likelihood."""
        from scipy.optimize import minimize
        
        def neg_log_lik(params_dict):
            try:
                self.fit(X, y, **params_dict)
                return -self.log_marginal_likelihood(**params_dict)
            except Exception:
                return 1e10
        
        # Convert to flat array for optimizer
        param_names = list(initial_params.keys())
        x0 = np.array([initial_params[name] for name in param_names])
        bnds = [bounds[name] for name in param_names]
        
        def wrapper(x):
            params = dict(zip(param_names, x))
            return neg_log_lik(params)
        
        res = minimize(wrapper, x0, bounds=bnds, method=method)
        optimized_params = dict(zip(param_names, res.x))
        
        # Refit with optimized params
        self.fit(X, y, **optimized_params)
        return optimized_params


# ======================================================================
# Simple Kriging
# ======================================================================

def simple_kriging(x_obs: np.ndarray, y_obs: np.ndarray,
                   x_pred: np.ndarray,
                   kernel: Callable,
                   noise: float = 1e-6,
                   **kernel_params) -> Tuple[np.ndarray, np.ndarray]:
    """Simple Kriging with known mean (zero)."""
    x_obs = np.atleast_2d(x_obs)
    y_obs = np.ravel(y_obs)
    x_pred = np.atleast_2d(x_pred)
    
    K = kernel(x_obs, x_obs, **kernel_params) + noise**2 * np.eye(len(x_obs))
    K_star = kernel(x_pred, x_obs, **kernel_params)
    K_star_star = kernel(x_pred, x_pred, **kernel_params)
    
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
    
    mu = K_star @ alpha
    
    v = np.linalg.solve(L, K_star.T)
    var = np.diag(K_star_star) - np.sum(v**2, axis=0)
    var = np.maximum(var, 0)
    std = np.sqrt(var)
    
    return mu, std


def ordinary_kriging(x_obs: np.ndarray, y_obs: np.ndarray,
                     x_pred: np.ndarray,
                     kernel: Callable,
                     noise: float = 1e-6,
                     **kernel_params) -> Tuple[np.ndarray, np.ndarray]:
    """Ordinary Kriging with unknown constant mean."""
    x_obs = np.atleast_2d(x_obs)
    y_obs = np.ravel(y_obs)
    x_pred = np.atleast_2d(x_pred)
    n = len(x_obs)
    
    K = kernel(x_obs, x_obs, **kernel_params) + noise**2 * np.eye(n)
    K_star = kernel(x_pred, x_obs, **kernel_params)
    K_star_star = kernel(x_pred, x_pred, **kernel_params)
    
    # Ordinary kriging system
    # [K  1][λ] = [y]
    # [1^T 0][μ]   [0]
    A = np.block([[K, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    b = np.concatenate([y_obs, [0]])
    
    sol = np.linalg.solve(A, b)
    lam = sol[:n]
    mu = sol[n]
    
    # Prediction
    pred = K_star @ lam + mu
    
    # Variance (simplified - doesn't account for mean estimation uncertainty)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K = K + 1e-6 * np.eye(n)
        L = np.linalg.cholesky(K)
    v = np.linalg.solve(L, K_star.T)
    var = np.diag(K_star_star) - np.sum(v**2, axis=0)
    var = np.maximum(var, 0)
    std = np.sqrt(var)
    
    return pred, std


# ======================================================================
# Example usage
# ======================================================================

if __name__ == "__main__":
    print("=== Gaussian Process Demo ===")
    
    # Generate data
    np.random.seed(42)
    X = np.array([0, 1, 2, 3, 4]).reshape(-1, 1)
    y = np.sin(X.ravel()) + 0.1 * np.random.randn(5)
    
    # Fit GP
    gp = GaussianProcess(rbf_kernel, noise=0.01)
    gp.fit(X, y, length_scale=1.0, variance=1.0)
    
    # Predict
    X_test = np.linspace(0, 4, 50).reshape(-1, 1)
    mu, std = gp.predict(X_test, return_std=True, length_scale=1.0, variance=1.0)
    
    print(f"Training points: {X.ravel()}")
    print(f"Training values: {y}")
    print(f"Log marginal likelihood: {gp.log_marginal_likelihood(length_scale=1.0, variance=1.0):.4f}")
    
    # Test simple kriging
    print("\nSimple Kriging:")
    mu_k, std_k = simple_kriging(X, y, X_test, rbf_kernel, noise=0.01, length_scale=1.0, variance=1.0)
    print(f"Mean at test points: {mu_k[:5]}")
    print(f"Std at test points: {std_k[:5]}")