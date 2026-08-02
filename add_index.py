"""Add index entries only to chapter .tex files, not Python files."""

import os
import re

chapters_dir = "/Users/csv610/Projects/MyBooks/Gauss/chapters"

# All index terms to add
index_terms = [
    (r"congruence", r"congruence\index{congruence}"),
    (r"residue class", r"residue class\index{residue class}"),
    (r"Chinese Remainder Theorem", r"Chinese Remainder Theorem\index{Chinese Remainder Theorem}"),
    (r"primitive root", r"primitive root\index{primitive root}"),
    (r"discrete logarithm", r"discrete logarithm\index{discrete logarithm}"),
    (r"modular arithmetic", r"modular arithmetic\index{modular arithmetic}"),
    (r"Euler's totient", r"Euler's totient function\index{Euler's totient function}"),
    (r"Fermat's little theorem", r"Fermat's little theorem\index{Fermat's little theorem}"),
    (r"extended Euclidean", r"Euclidean algorithm\index{Euclidean algorithm}"),
    (r"Legendre symbol", r"Legendre symbol\index{Legendre symbol}"),
    (r"quadratic reciprocity", r"quadratic reciprocity\index{quadratic reciprocity}"),
    (r"Gauss's Lemma", r"Gauss's Lemma\index{Gauss's Lemma}"),
    (r"Jacobi symbol", r"Jacobi symbol\index{Jacobi symbol}"),
    (r"Euler's criterion", r"Euler's criterion\index{Euler's criterion}"),
    (r"quadratic residue", r"quadratic residue\index{quadratic residue}"),
    (r"Tonelli-Shanks", r"Tonelli-Shanks algorithm\index{Tonelli-Shanks algorithm}"),
    (r"Cipolla", r"Cipolla's algorithm\index{Cipolla's algorithm}"),
    (r"Gauss sum", r"Gauss sum\index{Gauss sum}"),
    (r"Dirichlet character", r"Dirichlet character\index{Dirichlet character}"),
    (r"L-function", r"L-function\index{L-function}"),
    (r"orthogonality relation", r"orthogonality relation\index{orthogonality relation}"),
    (r"Disquisitiones Arithmeticae", r"Disquisitiones Arithmeticae\index{Disquisitiones Arithmeticae}"),
    (r"binary quadratic form", r"binary quadratic form\index{binary quadratic form}"),
    (r"class group", r"class group\index{class group}"),
    (r"cyclotomic polynomial", r"cyclotomic polynomial\index{cyclotomic polynomial}"),
    (r"Fermat prime", r"Fermat prime\index{Fermat prime}"),
    (r"constructible", r"constructible polygon\index{constructible polygon}"),
    (r"Gaussian integral", r"Gaussian integral\index{Gaussian integral}"),
    (r"error function", r"error function\index{error function}"),
    (r"Faddeeva function", r"Faddeeva function\index{Faddeeva function}"),
    (r"Dawson integral", r"Dawson integral\index{Dawson integral}"),
    (r"theta function", r"theta function\index{theta function}"),
    (r"Dedekind eta", r"Dedekind eta function\index{Dedekind eta function}"),
    (r"Eisenstein series", r"Eisenstein series\index{Eisenstein series}"),
    (r"modular form", r"modular form\index{modular form}"),
    (r"arithmetic-geometric mean", r"arithmetic-geometric mean\index{arithmetic-geometric mean}"),
    (r"Brent-Salamin", r"Brent-Salamin algorithm\index{Brent-Salamin algorithm}"),
    (r"elliptic integral", r"elliptic integral\index{elliptic integral}"),
    (r"Gauss's constant", r"Gauss's constant\index{Gauss's constant}"),
    (r"Gaussian elimination", r"Gaussian elimination\index{Gaussian elimination}"),
    (r"LU decomposition", r"LU decomposition\index{LU decomposition}"),
    (r"Cholesky decomposition", r"Cholesky decomposition\index{Cholesky decomposition}"),
    (r"QR decomposition", r"QR decomposition\index{QR decomposition}"),
    (r"condition number", r"condition number\index{condition number}"),
    (r"partial pivoting", r"partial pivoting\index{partial pivoting}"),
    (r"Householder", r"Householder reflection\index{Householder reflection}"),
    (r"numerical stability", r"numerical stability\index{numerical stability}"),
    (r"least squares", r"least squares\index{least squares}"),
    (r"normal equations", r"normal equations\index{normal equations}"),
    (r"Gauss-Newton", r"Gauss-Newton method\index{Gauss-Newton method}"),
    (r"Levenberg-Marquardt", r"Levenberg-Marquardt algorithm\index{Levenberg-Marquardt algorithm}"),
    (r"Gaussian quadrature", r"Gaussian quadrature\index{Gaussian quadrature}"),
    (r"Legendre polynomial", r"Legendre polynomial\index{Legendre polynomial}"),
    (r"Chebyshev", r"Chebyshev polynomial\index{Chebyshev polynomial}"),
    (r"normal distribution", r"normal distribution\index{normal distribution}"),
    (r"central limit theorem", r"Central Limit Theorem\index{Central Limit Theorem}"),
    (r"confidence interval", r"confidence interval\index{confidence interval}"),
    (r"maximum likelihood", r"maximum likelihood estimation\index{maximum likelihood estimation}"),
    (r"error propagation", r"error propagation\index{error propagation}"),
    (r"Gaussian process", r"Gaussian process\index{Gaussian process}"),
    (r"Kriging", r"Kriging\index{Kriging}"),
    (r"heptadecagon", r"heptadecagon\index{heptadecagon}"),
    (r"regular 17-gon", r"regular 17-gon\index{regular 17-gon}"),
    (r"Gauss periods", r"Gauss period\index{Gauss period}"),
    (r"Theorema Egregium", r"Theorema Egregium\index{Theorema Egregium}"),
    (r"Gaussian curvature", r"Gaussian curvature\index{Gaussian curvature}"),
    (r"Christoffel", r"Christoffel symbol\index{Christoffel symbol}"),
    (r"Riemann tensor", r"Riemann curvature tensor\index{Riemann curvature tensor}"),
    (r"geodesic", r"geodesic\index{geodesic}"),
    (r"Gauss-Bonnet", r"Gauss-Bonnet theorem\index{Gauss-Bonnet theorem}"),
    (r"fundamental form", r"fundamental form\index{fundamental form}"),
    (r"orbital mechanics", r"orbital mechanics\index{orbital mechanics}"),
    (r"Kepler's equation", r"Kepler's equation\index{Kepler's equation}"),
    (r"semi-major axis", r"semi-major axis\index{semi-major axis}"),
    (r"eccentricity", r"eccentricity\index{eccentricity (orbit)}"),
]

count = 0
for filename in sorted(os.listdir(chapters_dir)):
    if not filename.endswith('.tex') or filename.startswith('appendix'):
        continue
    
    filepath = os.path.join(chapters_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    for pattern, replacement in index_terms:
        # Use simple string replacement to avoid regex issues
        content = content.replace(pattern, replacement)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {filename}")
        count += content.count('\index{') - original.count('\index{')

print(f"\nTotal new index entries added: {count}")
