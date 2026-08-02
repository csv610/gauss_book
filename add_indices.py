#!/usr/bin/env python3
"""Add \index{} commands to key mathematical terms in chapter files."""

import re
from pathlib import Path

CHAPTERS_DIR = Path("/Users/csv610/Projects/MyBooks/Gauss/chapters")

# Define index entries for each chapter file with their contexts
# Format: (chapter_file, [(pattern_to_find, replacement_text, context_lines_before)])

index_entries = {
    "ch01_modular.tex": [
        # Basic definitions and concepts
        (r"\\textit{congruent} to \\$b\\$", r"\\textit{congruent} to \\$b\\$ \\index{congruence|see also residue class}", None),
        (r"\\textit{residue classes modulo", r"\\textit{residue classes modulo $n$} \\index{residue class}", None),
        (r"\\textit{primitive root modulo $p$}", r"\\textit{primitive root modulo $p$} \\index{primitive root}", None),
        (r"\\textit{Disquisitiones Arithmeticae}", r"\\textit{Disquisitiones Arithmeticae} \\index{Disquisitiones Arithmeticae}", None),
        (r"Chinese Remainder Theorem", r"Chinese Remainder Theorem \\index{Chinese Remainder Theorem}", None),
        (r"Gauss's algorithm for discrete logarithms", r"Gauss's algorithm for discrete logarithms \\index{discrete logarithm}", None),
        (r"modular arithmetic---working", r"modular arithmetic \\index{modular arithmetic}---working", None),
        # Keep Legendre symbol separate (it's defined in ch02)
        (r"\\textit{primitive root modulo $p$} if the order", r"\\textit{primitive root modulo $p$} if the order \\index{primitive root!order}", None),
        (r"order of $g$ modulo $p$ is $p-1$", r"order of $g$ modulo $p$ is $p-1$ \\index{primitive root}, i.e., the smallest", None),
        (r"Gauss proved that primitive roots exist", r"Gauss proved that primitive roots exist \\index{primitive root!existence}", None),
        (r"discrete logarithm (index) calculus", r"discrete logarithm (index) calculus \\index{discrete logarithm!Gauss algorithm}", None),
        (r"congruence $ax \\equiv b \\pmod{n}$", r"congruence $ax \\equiv b \\pmod{n}$ \\index{linear congruence}", None),
        (r"pairwise coprime positive integers", r"pairwise coprime positive integers \\index{coprimality}", None),
        (r"unique solution modulo", r"unique solution modulo \\index{Chinese Remainder Theorem!uniqueness}", None),
    ],
    "ch02_quadratic.tex": [
        # Legendre symbol and quadratic reciprocity
        (r"\\textit{Legendre symbol} is defined", r"\\textit{Legendre symbol} is defined \\index{Legendre symbol}", None),
        (r"\\textit{discrete logarithm} modulo $p$", r"\\textit{discrete logarithm} modulo $p$ \\index{Legendre symbol}", None),
        (r"law of quadratic reciprocity", r"law of quadratic reciprocity \\index{quadratic reciprocity}", None),
        (r"Gauss's Lemma", r"Gauss's Lemma \\index{Gauss's Lemma}", None),
        (r"First Supplement", r"First Supplement \\index{quadratic reciprocity!supplements}", None),
        (r"Second Supplement", r"Second Supplement \\index{quadratic reciprocity!supplements}", None),
        (r"Euler's criterion", r"Euler's criterion \\index{Euler's criterion}", None),
        (r"Jacobi symbol", r"Jacobi symbol \\index{Jacobi symbol}", None),
        (r"Kronecker symbol", r"Kronecker symbol \\index{Kronecker symbol}", None),
        (r"Tonelli-Shanks", r"Tonelli-Shanks \\index{Tonelli-Shanks algorithm}", None),
        (r"Cipolla's algorithm", r"Cipolla's algorithm \\index{Cipolla's algorithm}", None),
        (r"quadratic residue", r"quadratic residue \\index{quadratic residue}", None),
        (r"quadratic non-residue", r"quadratic non-residue \\index{quadratic non-residue}", None),
        (r"Article 131", r"Article 131 \\index{quadratic reciprocity!fundamental theorem}", None),
        (r"``fundamental theorem,''", r"``fundamental theorem,'' \\index{quadratic reciprocity!fundamental theorem}", None),
        (r"``golden theorem''", r"``golden theorem'' \\index{quadratic reciprocity!golden theorem}", None),
        (r"solovay-strassen", r"Solovay-Strassen \\index{primality test!Solovay-Strassen}", None),
        (r"Atkin-Morain", r"Atkin-Morain \\index{primality test!Atkin-Morain}", None),
        (r"Carmichael numbers", r"Carmichael numbers \\index{Carmichael number}", None),
        (r"elliptic curves", r"elliptic curves \\index{elliptic curve}", None),
    ],
    "ch03_gauss_sums.tex": [
        # Gauss sums and Dirichlet characters
        (r"Gauss sums", r"Gauss sums \\index{Gauss sum}", None),
        (r"quadratic Gauss sum", r"quadratic Gauss sum \\index{Gauss sum!quadratic}", None),
        (r"Dirichlet characters", r"Dirichlet characters \\index{Dirichlet character}", None),
        (r"trivial character", r"trivial character \\index{Dirichlet character!trivial}", None),
        (r"primitive character", r"primitive character \\index{Dirichlet character!primitive}", None),
        (r"discrete Fourier transform", r"discrete Fourier transform \\index{discrete Fourier transform}", None),
        (r"Dirichlet L-functions", r"Dirichlet L-functions \\index{Dirichlet L-function}", None),
        (r"functional equation", r"functional equation \\index{L-function!functional equation}", None),
        (r"Pólya-Vinogradov", r"Pólya-Vinogradov \\index{Pólya-Vinogradov inequality}", None),
        (r"Dirichlet's theorem", r"Dirichlet's theorem \\index{Dirichlet's theorem on primes}", None),
        (r"orthogonality relations", r"Orthogonality Relations \\index{orthogonality relations!Dirichlet characters}", None),
        (r"Euler's totient", r"Euler's totient \\index{Euler's totient function}", None),
        (r"g(p)^2", r"g(p)^2 \\index{Gauss sum!evaluation}", None),
    ],
    "ch04_disquisitiones.tex": [
        # Disquisitiones Arithmeticae topics
        (r"\\textit{Disquisitiones Arithmeticae}", r"\\textit{Disquisitiones Arithmeticae} \\index{Disquisitiones Arithmeticae}", None),
        (r"Binary Quadratic Form", r"Binary Quadratic Form \\index{binary quadratic form}", None),
        (r"discriminant", r"\\textit{discriminant} is \\index{discriminant!quadratic form}", None),
        (r"Gauss's Composition Law", r"Gauss's Composition Law \\index{composition law!quadratic forms}", None),
        (r"class group", r"class group \\index{class group!quadratic forms}", None),
        (r"class number", r"class number \\index{class number}", None),
        (r"ideal class group", r"ideal class group \\index{ideal class group}", None),
        (r"Fermat prime", r"Fermat prime \\index{Fermat prime}", None),
        (r"constructible polygon", r"constructible polygon \\index{constructible polygon}", None),
        (r"roots of unity", r"roots of unity \\index{roots of unity}", None),
        (r"cyclotomic", r"\\textit{Cyclotomy} \\index{cyclotomy}", None),
    ],
    "ch05_gaussian_integral.tex": [
        # Gaussian integral and error function
        (r"Gaussian integral", r"Gaussian integral \\index{Gaussian integral}", None),
        (r"error function", r"error function \\index{error function}", None),
        (r"complementary error function", r"complementary error function \\index{complementary error function}", None),
        (r"Dawson's integral", r"Dawson's integral \\index{Dawson's integral}", None),
        (r"Faddeeva function", r"Faddeeva function \\index{Faddeeva function}", None),
        (r"imaginary error function", r"imaginary error function \\index{imaginary error function}", None),
        (r"Fresnel integrals", r"Fresnel integrals \\index{Fresnel integrals}", None),
        (r"normal distribution", r"normal distribution \\index{normal distribution}", None),
        (r"probability density function", r"probability density function \\index{probability density function}", None),
        (r"cumulative distribution function", r"cumulative distribution function \\index{cumulative distribution function}", None),
        (r"heat equation", r"heat equation \\index{heat equation}", None),
        (r"Mills ratio", r"Mills ratio \\index{Mills ratio}", None),
    ],
    "ch06_theta.tex": [
        # Theta functions
        (r"Theta functions", r"Theta functions \\index{theta function}", None),
        (r"Jacobi theta functions", r"Jacobi theta functions \\index{Jacobi theta function}", None),
        (r"modular forms", r"modular forms \\index{modular form}", None),
        (r"elliptic functions", r"elliptic functions \\index{elliptic function}", None),
        (r"abelian varieties", r"abelian varieties \\index{abelian variety}", None),
        (r"modular group", r"modular group \\index{modular group}", None),
        (r"SL(2,\\Z)", r"SL(2,\\Z) \\index{SL(2,Z)}", None),
        (r"Triple Product Identity", r"Triple Product Identity \\index{Jacobi triple product}", None),
        (r"Dedekind eta function", r"Dedekind eta function \\index{Dedekind eta function}", None),
        (r"sums of squares", r"sums of squares \\index{sums of squares}", None),
        (r"partition functions", r"partition functions \\index{partition function}", None),
        (r"modular transformation", r"modular transformation \\index{modular transformation}", None),
    ],
    "ch07_agm.tex": [
        # Arithmetic-geometric mean
        (r"arithmetic-geometric mean", r"arithmetic-geometric mean \\index{arithmetic-geometric mean}", None),
        (r"elliptic integrals", r"elliptic integrals \\index{elliptic integral}", None),
        (r"complete elliptic integral", r"complete elliptic integral \\index{elliptic integral!complete}", None),
        (r"Gauss's constant", r"Gauss's constant \\index{Gauss's constant}", None),
        (r"lemniscate", r"lemniscate \\index{lemniscate}", None),
        (r"Brent-Salamin algorithm", r"Brent-Salamin algorithm \\index{Brent-Salamin algorithm}", None),
        (r"Borwein quartic algorithm", r"Borwein quartic algorithm \\index{Borwein algorithm}", None),
        (r"quadratic convergence", r"quadratic convergence \\index{quadratic convergence}", None),
        (r"hypergeometric function", r"hypergeometric function \\index{hypergeometric function}", None),
        (r"lemniscate constant", r"lemniscate constant \\index{lemniscate constant}", None),
    ],
    "ch08_gaussian_elim.tex": [
        # Gaussian elimination
        (r"Gaussian elimination", r"Gaussian elimination \\index{Gaussian elimination}", None),
        (r"LU decomposition", r"LU decomposition \\index{LU decomposition}", None),
        (r"Cholesky decomposition", r"Cholesky decomposition \\index{Cholesky decomposition}", None),
        (r"QR decomposition", r"QR decomposition \\index{QR decomposition}", None),
        (r"partial pivoting", r"partial pivoting \\index{partial pivoting}", None),
        (r"Householder reflections", r"Householder reflections \\index{Householder reflection}", None),
        (r"condition number", r"condition number \\index{condition number}", None),
        (r"numerical stability", r"numerical stability \\index{numerical stability}", None),
        (r"back substitution", r"back substitution \\index{back substitution}", None),
        (r"Forward elimination", r"Forward elimination \\index{forward elimination}", None),
        (r" Wilkinson's matrix", r" Wilkinson's matrix \\index{Wilkinson's matrix}", None),
        (r"Bareiss algorithm", r"Bareiss algorithm \\index{Bareiss algorithm}", None),
        (r"fraction-free", r"fraction-free \\index{fraction-free elimination}", None),
        (r" SVD", r" SVD \\index{SVD}", None),
        (r" Golub-Reinsch", r" Golub-Reinsch \\index{Golub-Reinsch algorithm}", None),
    ],
    "ch09_least_squares.tex": [
        # Least squares
        (r"least squares", r"least squares \\index{least squares}", None),
        (r"Normal Equations", r"Normal Equations \\index{normal equations}", None),
        (r"Gauss-Markov theorem", r"Gauss-Markov theorem \\index{Gauss-Markov theorem}", None),
        (r"weighted least squares", r"weighted least squares \\index{weighted least squares}", None),
        (r"Tikhonov regularization", r"Tikhonov regularization \\index{Tikhonov regularization}", None),
        (r"ridge regression", r"ridge regression \\index{ridge regression}", None),
        (r"Gauss-Newton method", r"Gauss-Newton method \\index{Gauss-Newton method}", None),
        (r"Levenberg-Marquardt", r"Levenberg-Marquardt \\index{Levenberg-Marquardt algorithm}", None),
        (r"Ceres", r"Ceres \\index{Ceres (asteroid)}", None),
        (r"orbit determination", r"orbit determination \\index{orbit determination}", None),
        (r"nonlinear least squares", r"nonlinear least squares \\index{nonlinear least squares}", None),
        (r"Jacobian", r"Jacobian \\index{Jacobian (mathematics)}", None),
        (r"residuals", r"residuals \\index{residual (statistics)}", None),
    ],
    "ch10_gauss_quadrature.tex": [
        # Gaussian quadrature
        (r"Gaussian quadrature", r"Gaussian quadrature \\index{Gaussian quadrature}", None),
        (r"orthogonal polynomials", r"orthogonal polynomials \\index{orthogonal polynomial}", None),
        (r"Gauss-Legendre", r"Gauss-Legendre \\index{Gauss-Legendre quadrature}", None),
        (r"Gauss-Hermite", r"Gauss-Hermite \\index{Gauss-Hermite quadrature}", None),
        (r"Gauss-Laguerre", r"Gauss-Laguerre \\index{Gauss-Laguerre quadrature}", None),
        (r"Gauss-Chebyshev", r"Gauss-Chebyshev \\index{Gauss-Chebyshev quadrature}", None),
        (r"Legendre", r"Legendre \\index{Legendre polynomial}", None),
        (r"Hermite", r"Hermite \\index{Hermite polynomial}", None),
        (r"Laguerre", r"Laguerre \\index{Laguerre polynomial}", None),
        (r"Chebyshev", r"Chebyshev \\index{Chebyshev polynomial}", None),
        (r"Golub-Welsch algorithm", r"Golub-Welsch algorithm \\index{Golub-Welsch algorithm}", None),
        (r"Kronrod extension", r"Kronrod extension \\index{Kronrod quadrature}", None),
        (r"Clenshaw-Curtis", r"Clenshaw-Curtis \\index{Clenshaw-Curtis quadrature}", None),
        (r"Gauss-Radau", r"Gauss-Radau \\index{Gauss-Radau quadrature}", None),
        (r"Gauss-Lobatto", r"Gauss-Lobatto \\index{Gauss-Lobatto quadrature}", None),
        (r"three-term recurrence", r"three-term recurrence \\index{three-term recurrence}", None),
    ],
    "ch11_normal.tex": [
        # Normal distribution
        (r"normal (Gaussian) distribution", r"normal (Gaussian) distribution \\index{normal distribution}", None),
        (r"Central Limit Theorem", r"Central Limit Theorem \\index{Central Limit Theorem}", None),
        (r"maximum likelihood", r"maximum likelihood \\index{maximum likelihood estimation}", None),
        (r"MLE", r"MLE \\index{maximum likelihood estimation}", None),
        (r"characteristic function", r"characteristic function \\index{characteristic function}", None),
        (r"moment generating function", r"MGF \\index{moment generating function}", None),
        (r"Basu's theorem", r"Basu's theorem \\index{Basu's theorem}", None),
        (r"Box-Muller", r"Box-Muller \\index{Box-Muller transform}", None),
        (r"entropy", r"entropy \\index{Shannon entropy}", None),
        (r"regression", r"regression \\index{linear regression}", None),
        (r"covariance matrix", r"covariance matrix \\index{covariance matrix}", None),
        (r"multivariate normal", r"multivariate normal \\index{multivariate normal distribution}", None),
        (r"confidence interval", r"confidence interval \\index{confidence interval}", None),
        (r"uncertainty propagation", r"uncertainty propagation \\index{propagation of uncertainty}", None),
    ],
    "ch12_gp.tex": [
        # Gaussian processes
        (r"Gaussian process", r"Gaussian process \\index{Gaussian process}", None),
        (r"Gaussian processes", r"Gaussian processes \\index{Gaussian process}", None),
        (r"covariance function", r"covariance function \\index{covariance function}", None),
        (r"kernel", r"kernel \\index{kernel (Gaussian process)}", None),
        (r"Kriging", r"Kriging \\index{Kriging}", None),
        (r"squared exponential", r"squared exponential \\index{kernel!squared exponential}", None),
        (r"RBF", r"RBF \\index{kernel!RBF}", None),
        (r"Matérn", r"Matérn \\index{kernel!Matérn}", None),
        (r"log marginal likelihood", r"log marginal likelihood \\index{marginal likelihood}", None),
        (r"hyperparameter optimization", r"hyperparameter optimization \\index{hyperparameter optimization}", None),
        (r"mean function", r"mean function \\index{mean function (Gaussian process)}", None),
        (r"non-parametric", r"non-parametric \\index{non-parametric method}", None),
        (r"spatial interpolation", r"spatial interpolation \\index{spatial interpolation}", None),
        (r"time series", r"time series \\index{time series!Gaussian process}", None),
        (r"FITC", r"FITC \\index{FITC approximation}", None),
        (r"VFE", r"VFE \\index{VFE approximation}", None),
    ],
    "ch13_heptadecagon.tex": [
        # Heptadecagon and constructibility
        (r"heptadecagon", r"heptadecagon \\index{heptadecagon}", None),
        (r"17-gon", r"17-gon \\index{heptadecagon}", None),
        (r"cyclotomic polynomial", r"\\textit{Cyclotomic Polynomial} \\index{cyclotomic polynomial}", None),
        (r"Fermat prime", r"Fermat prime \\index{Fermat prime}", None),
        (r"constructible", r"constructible \\index{constructible polygon}", None),
        (r"Gaussian periods", r"Gaussian periods \\index{Gaussian period}", None),
        (r"ruler and compass", r"ruler and compass \\index{ruler-and-compass construction}", None),
        (r"Eisenstein's criterion", r"Eisenstein's criterion \\index{Eisenstein's criterion}", None),
        (r"field extension", r"field extension \\index{field extension}", None),
        (r"Galois theory", r"Galois theory \\index{Galois theory}", None),
        (r"Wantzel", r"Wantzel \\index{Wantzel's theorem}", None),
        (r"nested square roots", r"nested square roots \\index{nested radical}", None),
    ],
    "ch14_theorema.tex": [
        # Differential geometry
        (r"Theorema Egregium", r"Theorema Egregium \\index{Theorema Egregium}", None),
        (r"Gaussian curvature", r"Gaussian curvature \\index{Gaussian curvature}", None),
        (r"mean curvature", r"mean curvature \\index{mean curvature}", None),
        (r"fundamental form", r"fundamental form \\index{fundamental form (differential geometry)}", None),
        (r"Christoffel symbols", r"Christoffel symbols \\index{Christoffel symbol}", None),
        (r"Riemann curvature tensor", r"Riemann curvature tensor \\index{Riemann curvature tensor}", None),
        (r"geodesic", r"geodesic \\index{geodesic}", None),
        (r"Gauss-Bonnet", r"Gauss-Bonnet \\index{Gauss-Bonnet theorem}", None),
        (r"intrinsic invariant", r"intrinsic invariant \\index{intrinsic geometry}", None),
        (r"differential geometry", r"differential geometry \\index{differential geometry}", None),
        (r"parametrized surface", r"parametrized surface \\index{surface (differential geometry)}", None),
        (r"unit normal", r"unit normal \\index{normal vector}", None),
        (r"curvature", r"curvature \\index{curvature!Gaussian}", None),
        (r"sphere", r"sphere \\index{sphere!geometry}", None),
        (r"cylinder", r"cylinder \\index{cylinder!geometry}", None),
        (r"pseudosphere", r"pseudosphere \\index{pseudosphere}", None),
        (r"triangle mesh", r"triangle mesh \\index{mesh!triangle}", None),
    ],
    "ch15_orbital.tex": [
        # Orbital mechanics
        (r"two-body problem", r"two-body problem \\index{two-body problem}", None),
        (r"Kepler's laws", r"Kepler's laws \\index{Kepler's laws}", None),
        (r"orbital elements", r"orbital elements \\index{orbital element}", None),
        (r"semi-major axis", r"semi-major axis \\index{semi-major axis}", None),
        (r"eccentricity", r"eccentricity \\index{eccentricity}", None),
        (r"inclination", r"inclination \\index{inclination (orbit)}", None),
        (r"longitude of ascending node", r"longitude of ascending node \\index{longitude of ascending node}", None),
        (r"argument of periapsis", r"argument of periapsis \\index{argument of periapsis}", None),
        (r"mean anomaly", r"mean anomaly \\index{mean anomaly}", None),
        (r"true anomaly", r"true anomaly \\index{true anomaly}", None),
        (r"Gauss's method", r"Gauss's method \\index{Gauss's method (orbit determination)}", None),
        (r"Kepler's equation", r"Kepler's equation \\index{Kepler's equation}", None),
        (r"state vector", r"state vector \\index{state vector (orbital mechanics)}", None),
        (r"Laplace-Runge-Lenz", r"Laplace-Runge-Lenz \\index{Laplace-Runge-Lenz vector}", None),
        (r"differential corrections", r"differential corrections \\index{differential correction (orbital mechanics)}", None),
        (r"universal variable", r"universal variable \\index{universal variable (orbital mechanics)}", None),
    ],
}


def add_index_entries(content: str, entries: list) -> str:
    """Add index entries to content based on patterns."""
    modified = content
    for pattern, replacement, context in entries:
        if re.search(pattern, modified):
            modified = re.sub(pattern, replacement, modified)
    return modified


def main():
    total_entries = 0
    for chapter_file, entries in index_entries.items():
        file_path = CHAPTERS_DIR / chapter_file
        if not file_path.exists():
            print(f"WARNING: {chapter_file} not found")
            continue
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        new_content = add_index_entries(content, entries)
        
        if new_content != content:
            with open(file_path, 'w') as f:
                f.write(new_content)
            count = len(entries)
            print(f"Updated {chapter_file}: {count} index entries added")
            total_entries += count
        else:
            print(f"No changes needed for {chapter_file}")
    
    print(f"\nTotal: {total_entries} index entries added across all chapters")


if __name__ == "__main__":
    main()
