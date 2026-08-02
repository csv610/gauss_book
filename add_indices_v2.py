#!/usr/bin/env python3
"""Add comprehensive \index{} commands to all chapter files."""
import re
from pathlib import Path

CHAPTERS_DIR = Path("/Users/csv610/Projects/MyBooks/Gauss/chapters")

# Define indices for each chapter
INDEX_DATA = {
    "ch01_modular.tex": [
        ("modular arithmetic", "modular arithmetic"),
        ("Chinese Remainder Theorem", "Chinese Remainder Theorem"),
        ("\\textit{primitive root modulo $p$}", "primitive root"),
        ("discrete logarithm", "discrete logarithm"),
        ("residue classes", "residue class"),
        ("primitive roots exist", "primitive root!existence"),
        ("pairwise coprime", "coprimality"),
        ("linear congruences", "linear congruence"),
    ],
    "ch02_quadratic.tex": [
        ("\\textit{Legendre symbol}", "Legendre symbol"),
        ("law of quadratic reciprocity", "quadratic reciprocity"),
        ("Gauss's Lemma", "Gauss's Lemma"),
        ("First Supplement", "quadratic reciprocity!supplements"),
        ("Second Supplement", "quadratic reciprocity!supplements"),
        ("Euler's criterion", "Euler's criterion"),
        ("Jacobi symbol", "Jacobi symbol"),
        ("Kronecker symbol", "Kronecker symbol"),
        ("Tonelli-Shanks", "Tonelli-Shanks algorithm"),
        ("Cipolla's algorithm", "Cipolla's algorithm"),
        ("quadratic residue", "quadratic residue"),
        ("quadratic non-residue", "quadratic non-residue"),
        ("golden theorem", "quadratic reciprocity!golden theorem"),
        ("Carmichael numbers", "Carmichael number"),
    ],
    "ch03_gauss_sums.tex": [
        ("Gauss sums", "Gauss sum"),
        ("quadratic Gauss sum", "Gauss sum!quadratic"),
        ("Dirichlet characters", "Dirichlet character"),
        ("primitive character", "Dirichlet character!primitive"),
        ("discrete Fourier transform", "discrete Fourier transform"),
        ("Dirichlet L-functions", "Dirichlet L-function"),
        ("Pólya-Vinogradov", "Pólya-Vinogradov inequality"),
        ("Dirichlet's theorem", "Dirichlet's theorem on primes"),
        ("orthogonality relations", "orthogonality relations"),
    ],
    "ch04_disquisitiones.tex": [
        ("Binary Quadratic Form", "binary quadratic form"),
        ("\\textit{discriminant}", "discriminant!quadratic form"),
        ("Gauss's Composition Law", "composition law!quadratic forms"),
        ("class group", "class group!quadratic forms"),
        ("class number", "class number"),
        ("ideal class group", "ideal class group"),
        ("Fermat prime", "Fermat prime"),
        ("constructible polygon", "constructible polygon"),
        ("roots of unity", "roots of unity"),
        ("cyclotomic", "cyclotomy"),
    ],
    "ch05_gaussian_integral.tex": [
        ("Gaussian integral", "Gaussian integral"),
        ("error function", "error function"),
        ("complementary error function", "complementary error function"),
        ("Dawson's integral", "Dawson's integral"),
        ("Faddeeva function", "Faddeeva function"),
        ("normal distribution", "normal distribution"),
        ("heat equation", "heat equation"),
        ("Mills ratio", "Mills ratio"),
    ],
    "ch06_theta.tex": [
        ("Theta functions", "theta function"),
        ("Jacobi theta functions", "Jacobi theta function"),
        ("modular forms", "modular form"),
        ("modular group", "modular group"),
        ("Dedekind eta function", "Dedekind eta function"),
        ("sums of squares", "sums of squares"),
        ("partition functions", "partition function"),
    ],
    "ch07_agm.tex": [
        ("arithmetic-geometric mean", "arithmetic-geometric mean"),
        ("elliptic integrals", "elliptic integral"),
        ("complete elliptic integral", "elliptic integral!complete"),
        ("Gauss's constant", "Gauss's constant"),
        ("lemniscate", "lemniscate"),
        ("Brent-Salamin algorithm", "Brent-Salamin algorithm"),
        ("hypergeometric function", "hypergeometric function"),
    ],
    "ch08_gaussian_elim.tex": [
        ("Gaussian elimination", "Gaussian elimination"),
        ("LU decomposition", "LU decomposition"),
        ("Cholesky decomposition", "Cholesky decomposition"),
        ("QR decomposition", "QR decomposition"),
        ("partial pivoting", "partial pivoting"),
        ("Householder reflections", "Householder reflection"),
        ("condition number", "condition number"),
        ("numerical stability", "numerical stability"),
        ("back substitution", "back substitution"),
        ("Bareiss algorithm", "Bareiss algorithm"),
    ],
    "ch09_least_squares.tex": [
        ("least squares", "least squares"),
        ("Normal Equations", "normal equations"),
        ("Gauss-Markov theorem", "Gauss-Markov theorem"),
        ("weighted least squares", "weighted least squares"),
        ("Tikhonov regularization", "Tikhonov regularization"),
        ("ridge regression", "ridge regression"),
        ("Gauss-Newton method", "Gauss-Newton method"),
        ("Levenberg-Marquardt", "Levenberg-Marquardt algorithm"),
        ("orbit determination", "orbit determination"),
    ],
    "ch10_gauss_quadrature.tex": [
        ("Gaussian quadrature", "Gaussian quadrature"),
        ("orthogonal polynomials", "orthogonal polynomial"),
        ("Gauss-Legendre", "Gauss-Legendre quadrature"),
        ("Gauss-Hermite", "Gauss-Hermite quadrature"),
        ("Gauss-Laguerre", "Gauss-Laguerre quadrature"),
        ("Gauss-Chebyshev", "Gauss-Chebyshev quadrature"),
        ("Golub-Welsch algorithm", "Golub-Welsch algorithm"),
        ("Kronrod extension", "Kronrod quadrature"),
        ("three-term recurrence", "three-term recurrence"),
    ],
    "ch11_normal.tex": [
        ("normal (Gaussian) distribution", "normal distribution"),
        ("Central Limit Theorem", "Central Limit Theorem"),
        ("maximum likelihood", "maximum likelihood estimation"),
        ("characteristic function", "characteristic function"),
        ("entropy", "Shannon entropy"),
        ("covariance matrix", "covariance matrix"),
        ("multivariate normal", "multivariate normal distribution"),
        ("confidence interval", "confidence interval"),
    ],
    "ch12_gp.tex": [
        ("Gaussian process", "Gaussian process"),
        ("covariance function", "covariance function"),
        ("kernel", "kernel (Gaussian process)"),
        ("Kriging", "Kriging"),
        ("squared exponential", "kernel!squared exponential"),
        ("Matérn", "kernel!Matérn"),
        ("log marginal likelihood", "marginal likelihood"),
        ("hyperparameter optimization", "hyperparameter optimization"),
    ],
    "ch13_heptadecagon.tex": [
        ("heptadecagon", "heptadecagon"),
        ("17-gon", "heptadecagon"),
        ("Fermat prime", "Fermat prime"),
        ("constructible", "constructible polygon"),
        ("Gaussian periods", "Gaussian period"),
        ("ruler and compass", "ruler-and-compass construction"),
        ("Eisenstein's criterion", "Eisenstein's criterion"),
        ("nested square roots", "nested radical"),
    ],
    "ch14_theorema.tex": [
        ("Theorema Egregium", "Theorema Egregium"),
        ("Gaussian curvature", "Gaussian curvature"),
        ("mean curvature", "mean curvature"),
        ("fundamental form", "fundamental form (differential geometry)"),
        ("Christoffel symbols", "Christoffel symbol"),
        ("Riemann curvature tensor", "Riemann curvature tensor"),
        ("geodesic", "geodesic"),
        ("Gauss-Bonnet", "Gauss-Bonnet theorem"),
        ("parametrized surface", "surface (differential geometry)"),
        ("unit normal", "normal vector"),
    ],
    "ch15_orbital.tex": [
        ("two-body problem", "two-body problem"),
        ("Kepler's laws", "Kepler's laws"),
        ("orbital elements", "orbital element"),
        ("semi-major axis", "semi-major axis"),
        ("eccentricity", "eccentricity"),
        ("Gauss's method", "Gauss's method (orbit determination)"),
        ("Kepler's equation", "Kepler's equation"),
        ("Laplace-Runge-Lenz", "Laplace-Runge-Lenz vector"),
        ("state vector", "state vector (orbital mechanics)"),
    ],
}


def add_index_safe(content, search, index_name):
    """Safely add an index command."""
    if f'\\index{{{index_name}}}' in content:
        return content, False
    if search not in content:
        return content, False
    idx = content.find(search)
    pos = idx + len(search)
    return content[:pos] + f' \\index{{{index_name}}}' + content[pos:], True


def main():
    total_added = 0
    for fname, entries in INDEX_DATA.items():
        fpath = CHAPTERS_DIR / fname
        if not fpath.exists():
            print(f"WARNING: {fname} not found")
            continue
        
        with open(fpath, 'r') as f:
            content = f.read()
        
        added = 0
        for search, index_name in entries:
            content, was_added = add_index_safe(content, search, index_name)
            added += was_added
        
        with open(fpath, 'w') as f:
            f.write(content)
        
        print(f"✓ {fname}: {added} indices")
        total_added += added
    
    print(f"\nTotal added: {total_added}")


if __name__ == "__main__":
    main()
