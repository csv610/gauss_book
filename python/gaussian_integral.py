"""Gaussian integral, error function, and related special functions."""

import math
from typing import Callable, Tuple, List
import cmath

__all__ = [
    'gaussian_integral', 'gaussian_pdf', 'gaussian_cdf', 'gaussian_ppf',
    'erf_series', 'erf_asymptotic', 'erfc_continued_fraction',
    'erf_chebyshev', 'erfc_asymptotic',
    'erf', 'erfc', 'erfi',
    'dawson_integral', 'dawson_continued_fraction', 'faddeeva',
    'normal_pdf', 'normal_cdf', 'normal_sf', 'normal_ppf',
    'normal_rvs',
    'confidence_interval_normal', 'error_propagation',
    'chi2_pdf', 'chi2_cdf', 't_pdf', 'f_pdf',
]


# ======================================================================
# Gaussian integral
# ======================================================================

def gaussian_integral(a: float, b: float, c: float, 
                      x1: float = -float('inf'), x2: float = float('inf')) -> float:
    """Integral of exp(-a*x^2 + b*x + c) from x1 to x2.
    
    Uses the formula: integral = sqrt(pi/a) * exp(c + b^2/(4a)) * 
                       [erf(sqrt(a)*x2 - b/(2*sqrt(a))) - erf(sqrt(a)*x1 - b/(2*sqrt(a)))] / 2
    """
    if a <= 0:
        raise ValueError("a must be positive")
    
    sqrt_a = math.sqrt(a)
    b_over_2a = b / (2 * a)
    exp_factor = math.exp(c + b*b / (4*a))
    prefactor = math.sqrt(math.pi / a) * exp_factor / 2
    
    def erf_term(x):
        if x == float('inf'):
            return 1.0
        if x == -float('inf'):
            return -1.0
        return math.erf(sqrt_a * x - b_over_2a)
    
    return prefactor * (erf_term(x2) - erf_term(x1))


def gaussian_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal probability density function."""
    return math.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * math.sqrt(2 * math.pi))


def gaussian_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal cumulative distribution function."""
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def gaussian_ppf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal percent point function (inverse CDF)."""
    try:
        import scipy.special as special
        return special.ndtri(p) * sigma + mu
    except ImportError:
        pass
        
    if p <= 0:
        return -float('inf')
    if p >= 1:
        return float('inf')
    if p == 0.5:
        return mu
        
    p_tail = p if p < 0.5 else 1.0 - p
    t = math.sqrt(-2.0 * math.log(p_tail))
    
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    
    num = c[0] + c[1] * t + c[2] * t * t
    den = 1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t
    x = t - num / den
    
    if p < 0.5:
        x = -x
        
    return mu + sigma * x


# ======================================================================
# Error function and related
# ======================================================================

def erf_series(x: float, terms: int = 20) -> float:
    """Error function via Taylor series: erf(x) = 2/sqrt(pi) sum (-1)^n x^(2n+1)/(n!(2n+1))."""
    result = 0.0
    x2 = x * x
    term = x
    for n in range(terms):
        result += term / (2*n + 1)
        term *= -x2 / (n + 1)
    return 2 * result / math.sqrt(math.pi)


def erf_asymptotic(x: float, terms: int = 10) -> float:
    """Complementary error function via asymptotic expansion for large x."""
    if x < 4:
        raise ValueError("Use series for small x")
    
    x2 = x * x
    inv_x2 = 1.0 / x2
    sum_term = 1.0
    term = 1.0
    sign = -1
    fact = 1
    for n in range(1, terms + 1):
        fact *= (2*n - 1)
        term = fact * (inv_x2 ** n) * sign
        sum_term += term
        sign = -sign
    
    return math.exp(-x2) / (x * math.sqrt(math.pi)) * sum_term


def erfc_continued_fraction(x: float, max_terms: int = 50) -> float:
    """erfc(x) via continued fraction (Lentz's method)."""
    # Using the CF: erfc(z) = e^{-z^2}/(sqrt(pi) (z + 1/(2z + 2/(2z + 3/(2z + ...)))))
    # Lentz's algorithm
    f = 0.0
    C = f
    D = 0.0
    tiny = 1e-30
    
    a = 0.5
    b = x
    
    for i in range(max_terms):
        if i > 0:
            a = i * (i - 0.5)
            b = x
        
        D = b + a * D
        if abs(D) < tiny:
            D = tiny
        C = b + a / C
        if abs(C) < tiny:
            C = tiny
        D = 1.0 / D
        delta = C * D
        f *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    
    return math.exp(-x*x) / (x * math.sqrt(math.pi)) * f


def erf_chebyshev(x: float) -> float:
    """Error function using Cody's rational Chebyshev approximation (from SciPy)."""
    # This is a simplified version; full implementation uses multiple ranges
    ax = abs(x)
    if ax < 0.85:
        # Rational approximation for |x| < 0.85
        p = [1.26551223, 1.00002368, 0.37409196, 0.09678418, 0.01061120, 0.00043596]
        q = [1.0, 2.824624, 2.57070, 1.24599, 0.30896, 0.04152]
        
        x2 = x * x
        num = p[5]
        den = q[5]
        for i in range(4, -1, -1):
            num = num * x2 + p[i]
            den = den * x2 + q[i]
        return x * num / den
    else:
        # Use complementary error function
        return math.copysign(1.0, x) * (1 - erfc_asymptotic(ax))


def erfc_asymptotic(x: float) -> float:
    """Complementary error function for x >= 0.85."""
    # Continued fraction for erfc
    t = 1.0 / (1.0 + 0.5 * x)
    # Horner scheme for polynomial
    poly = (((((
        0.00000643 * t + 0.00004677578) * t + 0.01629) * t + 0.1685) * t + 
        0.7728) * t + 1.668) * t + 2.5066
    return t * poly * math.exp(-x*x)


def erf(x: float) -> float:
    """Error function - uses math.erf, fallback to scipy for complex."""
    if isinstance(x, complex):
        import scipy.special as special
        return special.erf(x)
    return math.erf(x)


def erfc(x: float) -> float:
    """Complementary error function."""
    if isinstance(x, complex):
        import scipy.special as special
        return special.erfc(x)
    return math.erfc(x)


def erfi(x: float) -> float:
    """Imaginary error function: erfi(x) = -i erf(ix) = 2/sqrt(pi) int_0^x e^{t^2} dt."""
    # Use Dawson's integral: erfi(x) = 2/sqrt(pi) e^{x^2} D(x)
    return 2 / math.sqrt(math.pi) * math.exp(x*x) * dawson_integral(x)


def dawson_integral(x: float) -> float:
    """Dawson's integral: F(x) = e^{-x^2} int_0^x e^{t^2} dt."""
    if abs(x) < 0.2:
        # Taylor series
        return x * (1 - 2*x*x/3 + 4*x*x*x*x/15 - 8*x**6/105 + 16*x**8/945)
    elif abs(x) > 6:
        # Asymptotic expansion
        x2 = x * x
        return 1/(2*x) * (1 + 1/(2*x2) + 3/(4*x2*x2) + 15/(8*x2**3))
    else:
        # Use continued fraction
        return dawson_continued_fraction(x)


def dawson_continued_fraction(x: float) -> float:
    """Dawson's integral via continued fraction (McCabe's expansion)."""
    tiny = 1e-30
    x2 = x * x
    f = 1.0 + 2 * x2
    if abs(f) < tiny:
        f = tiny
    C = f
    D = 0.0
    
    for n in range(1, 100):
        a = -4 * n * x2
        b = 2 * n + 1 + 2 * x2
        
        D = b + a * D
        if abs(D) < tiny:
            D = tiny
        C = b + a / C
        if abs(C) < tiny:
            C = tiny
        D = 1.0 / D
        delta = C * D
        f *= delta
        if abs(delta - 1.0) < 1e-15:
            break
            
    return x / f


def faddeeva(z: complex) -> complex:
    """Faddeeva function: w(z) = e^{-z^2} erfc(-iz)."""
    import scipy.special as special
    return special.wofz(z)


# ======================================================================
# Normal distribution utilities
# ======================================================================

def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal PDF."""
    return math.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * math.sqrt(2 * math.pi))


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal CDF."""
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def normal_sf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal survival function (1 - CDF)."""
    return 1 - normal_cdf(x, mu, sigma)


def normal_ppf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal percent point function (inverse CDF)."""
    return gaussian_ppf(p, mu, sigma)


def normal_rvs(size: int = 1, mu: float = 0.0, sigma: float = 1.0, 
               random_state: int = None) -> list:
    """Normal random variates using Box-Muller transform."""
    import random
    if random_state is not None:
        random.seed(random_state)
    
    result = []
    for _ in range(size):
        u1 = random.random()
        u2 = random.random()
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        result.append(mu + sigma * z0)
    return result


def confidence_interval_normal(mean: float, std: float, n: int, 
                                confidence: float = 0.95) -> Tuple[float, float]:
    """Confidence interval for normal mean with known variance."""
    alpha = 1 - confidence
    z = normal_ppf(1 - alpha/2)
    margin = z * std / math.sqrt(n)
    return mean - margin, mean + margin


def error_propagation(f: Callable, vars: list[Tuple[float, float]]) -> Tuple[float, float]:
    """Error propagation using the delta method.
    
    Args:
        f: function of multiple variables
        vars: list of (mean, std) for each variable
        
    Returns:
        (mean, std) of f
    """
    # Use finite differences for gradient
    means = [v[0] for v in vars]
    stds = [v[1] for v in vars]
    
    f0 = f(*means)
    
    # Compute gradient numerically
    eps = 1e-6
    grad = []
    for i in range(len(means)):
        x_plus = means.copy()
        x_minus = means.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad_i = (f(*x_plus) - f(*x_minus)) / (2 * eps)
        grad.append(grad_i)
    
    # Variance = sum (grad_i * std_i)^2
    var = sum(g**2 * s**2 for g, s in zip(grad, stds))
    return f0, math.sqrt(var)


def chi2_pdf(x: float, df: int) -> float:
    """Chi-squared PDF."""
    if x <= 0:
        return 0.0
    return (x**(df/2 - 1) * math.exp(-x/2)) / (2**(df/2) * math.gamma(df/2))


def chi2_cdf(x: float, df: int) -> float:
    """Chi-squared CDF using incomplete gamma."""
    if x <= 0:
        return 0.0
    import scipy.special as special
    return special.gammainc(df/2, x/2)


def t_pdf(x: float, df: int) -> float:
    """Student's t PDF."""
    return (math.gamma((df+1)/2) / (math.sqrt(df * math.pi) * math.gamma(df/2))) * \
           (1 + x*x/df)**(-(df+1)/2)


def f_pdf(x: float, df1: int, df2: int) -> float:
    """F-distribution PDF."""
    if x <= 0:
        return 0.0
    return (math.gamma((df1+df2)/2) / (math.gamma(df1/2) * math.gamma(df2/2))) * \
           (df1/df2)**(df1/2) * x**(df1/2 - 1) * (1 + df1*x/df2)**(-(df1+df2)/2)


if __name__ == "__main__":
    print("=== Gaussian Integral Demo ===")
    
    # Gaussian integral
    val = gaussian_integral(1, 0, 0, -10, 10)
    print(f"int exp(-x^2) dx from -10 to 10 = {val}")
    print(f"sqrt(pi) = {math.sqrt(math.pi)}")
    
    # Error function
    print("\nError function values:")
    for x in [0.0, 0.5, 1.0, 2.0, 3.0]:
        print(f"  erf({x}) = {math.erf(x):.10f}")
    
    # Normal distribution
    print("\nNormal distribution:")
    for x in [-2, -1, 0, 1, 2]:
        print(f"  phi({x}) = {normal_pdf(x):.6f}, Phi({x}) = {normal_cdf(x):.6f}")
    
    # Quantiles
    print("\nNormal quantiles:")
    for p in [0.025, 0.05, 0.5, 0.95, 0.975]:
        z = normal_ppf(p)
        print(f"  Phi^{-1}({p}) = {z:.6f}")
    
    # Error propagation
    print("\nError propagation example:")
    f = lambda x, y: x / y
    mean, std = error_propagation(f, [(10.0, 0.1), (5.0, 0.05)])
    print(f"  x/y with x=10±0.1, y=5±0.05 => {mean:.6f} ± {std:.6f}")