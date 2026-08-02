"""Normal distribution and statistical functions."""

import math
import random
from typing import List, Tuple, Callable
import numpy as np
from python.gaussian_integral import (
    normal_pdf as _normal_pdf, normal_cdf as _normal_cdf,
    normal_sf as _normal_sf, normal_ppf as _normal_ppf,
    confidence_interval_normal as _ci_normal,
    error_propagation as _error_propagation,
    chi2_pdf as _chi2_pdf, chi2_cdf as _chi2_cdf,
    t_pdf as _t_pdf, f_pdf as _f_pdf
)

__all__ = [
    'normal_pdf', 'normal_cdf', 'normal_sf', 'normal_ppf',
    'normal_rvs', 'multivariate_normal_rvs',
    'normal_mle', 'normal_mle_unbiased',
    'confidence_interval_normal', 'confidence_interval_t',
    'z_test', 't_test',
    'error_propagation',
    'multivariate_normal_pdf', 'multivariate_normal_logpdf',
    'chi2_pdf', 'chi2_cdf', 't_pdf', 'f_pdf',
    'box_muller', 'marsaglia_bray',
]


normal_pdf = _normal_pdf
normal_cdf = _normal_cdf
normal_sf = _normal_sf
normal_ppf = _normal_ppf
confidence_interval_normal = _ci_normal
error_propagation = _error_propagation
chi2_pdf = _chi2_pdf
chi2_cdf = _chi2_cdf
t_pdf = _t_pdf
f_pdf = _f_pdf


# ======================================================================
# Random variates
# ======================================================================

def normal_rvs(size: int = 1, mu: float = 0.0, sigma: float = 1.0, 
               random_state: int = None) -> np.ndarray:
    """Normal random variates using Box-Muller transform."""
    if random_state is not None:
        np.random.seed(random_state)
    return mu + sigma * np.random.randn(size)


def multivariate_normal_rvs(mu: np.ndarray, Sigma: np.ndarray, 
                            size: int = 1) -> np.ndarray:
    """Multivariate normal random variates."""
    L = np.linalg.cholesky(Sigma)
    n = len(mu)
    Z = np.random.randn(size, n)
    return mu + (L @ Z.T).T


# ======================================================================
# MLE and inference
# ======================================================================

def normal_mle(data: List[float]) -> Tuple[float, float]:
    """MLE for normal mean and variance."""
    n = len(data)
    mu = sum(data) / n
    sigma2 = sum((x - mu)**2 for x in data) / n
    return mu, math.sqrt(sigma2)


def normal_mle_unbiased(data: List[float]) -> Tuple[float, float]:
    """Unbiased estimators for normal mean and variance."""
    n = len(data)
    mu = sum(data) / n
    sigma2 = sum((x - mu)**2 for x in data) / (n - 1)
    return mu, math.sqrt(sigma2)


def confidence_interval_t(mean: float, std: float, n: int,
                          confidence: float = 0.95) -> Tuple[float, float]:
    """Confidence interval for normal mean with unknown variance (t-distribution)."""
    from scipy import stats
    alpha = 1 - confidence
    t = stats.t.ppf(1 - alpha/2, n - 1)
    margin = t * std / math.sqrt(n)
    return mean - margin, mean + margin


# ======================================================================
# Hypothesis testing
# ======================================================================

def z_test(data: List[float], mu0: float, sigma: float = None) -> Tuple[float, float]:
    """One-sample z-test for normal mean."""
    n = len(data)
    sample_mean = sum(data) / n
    
    if sigma is None:
        sigma = math.sqrt(sum((x - sample_mean)**2 for x in data) / n)
    
    z = (sample_mean - mu0) / (sigma / math.sqrt(n))
    p = 2 * normal_sf(abs(z))
    return z, p


def t_test(data: List[float], mu0: float) -> Tuple[float, float]:
    """One-sample t-test for normal mean."""
    from scipy import stats
    n = len(data)
    sample_mean = sum(data) / n
    sample_std = math.sqrt(sum((x - sample_mean)**2 for x in data) / (n - 1))
    t = (sample_mean - mu0) / (sample_std / math.sqrt(n))
    p = 2 * stats.t.sf(abs(t), n - 1)
    return t, p


# ======================================================================
# Multivariate normal
# ======================================================================

def multivariate_normal_pdf(x: np.ndarray, mu: np.ndarray, 
                            Sigma: np.ndarray) -> float:
    """Multivariate normal PDF."""
    n = len(mu)
    diff = x - mu
    det = np.linalg.det(Sigma)
    inv = np.linalg.inv(Sigma)
    exponent = -0.5 * diff @ inv @ diff
    return math.exp(exponent) / ((2*math.pi)**(n/2) * math.sqrt(det))


def multivariate_normal_logpdf(x: np.ndarray, mu: np.ndarray, 
                               Sigma: np.ndarray) -> float:
    """Multivariate normal log PDF."""
    n = len(mu)
    diff = x - mu
    sign, logdet = np.linalg.slogdet(Sigma)
    inv = np.linalg.inv(Sigma)
    exponent = -0.5 * diff @ inv @ diff
    return exponent - 0.5 * n * math.log(2*math.pi) - 0.5 * logdet


# ======================================================================
# Box-Muller and other generators
# ======================================================================

def box_muller(u1: float, u2: float) -> Tuple[float, float]:
    """Box-Muller transform: two independent uniform -> two independent normal."""
    z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
    return z0, z1


def marsaglia_bray() -> Tuple[float, float]:
    """Marsaglia-Bray (polar) method for normal generation."""
    while True:
        u1 = 2 * random.random() - 1
        u2 = 2 * random.random() - 1
        s = u1*u1 + u2*u2
        if 0 < s < 1:
            break
    factor = math.sqrt(-2 * math.log(s) / s)
    return u1 * factor, u2 * factor


if __name__ == "__main__":
    print("=== Normal Distribution Demo ===")
    
    # PDF and CDF
    print("\n1. Standard normal:")
    for x in [-2, -1, 0, 1, 2]:
        print(f"  phi({x}) = {normal_pdf(x):.6f}, Phi({x}) = {normal_cdf(x):.6f}")
    
    # Quantiles
    print("\n2. Quantiles:")
    for p in [0.025, 0.05, 0.5, 0.95, 0.975]:
        z = normal_ppf(p)
        print(f"  Phi^(-1)({p}) = {z:.6f}")
    
    # MLE
    print("\n3. MLE:")
    data = [1.2, 2.3, 1.8, 2.1, 1.9, 2.2, 1.7, 2.0]
    mu, sigma = normal_mle(data)
    print(f"  Data: {data}")
    print(f"  MLE: mu={mu:.4f}, sigma={sigma:.4f}")
    
    # Error propagation
    print("\n4. Error propagation:")
    f = lambda x, y: x / y
    mean, std = error_propagation(f, [(10.0, 0.1), (5.0, 0.05)])
    print(f"  x/y with x=10±0.1, y=5±0.05 => {mean:.6f} ± {std:.6f}")
    
    # Confidence interval
    print("\n5. Confidence interval (95%):")
    ci = confidence_interval_normal(mean=2.0, std=0.5, n=25)
    print(f"  CI = [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    # Z-test
    print("\n6. Z-test:")
    data = [2.1, 2.3, 1.9, 2.2, 2.0]
    z, p = z_test(data, 2.0, sigma=0.1)
    print(f"  z = {z:.4f}, p = {p:.4f}")