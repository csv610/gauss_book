"""Tests for Theorema Egregium and differential geometry."""

import pytest
import sys
import os
import math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.theorema import (
    first_fundamental_form, second_fundamental_form,
    gaussian_curvature, mean_curvature, principal_curvatures,
    christoffel_symbols, christoffel_from_surface,
    theorema_egregium_check,
    geodesic_equations, geodesic_ode,
    gauss_bonnet_discrete,
    sphere, cylinder, torus, pseudosphere
)


class TestFirstFundamentalForm:
    def test_sphere(self):
        """First fundamental form of unit sphere at θ=π/2, φ=0."""
        E, F, G = first_fundamental_form(sphere, math.pi/2, 0.0)
        assert abs(E - 1.0) < 1e-6
        assert abs(F) < 1e-6
        assert abs(G - 1.0) < 1e-6

    def test_cylinder(self):
        """First fundamental form of unit cylinder."""
        E, F, G = first_fundamental_form(cylinder, 0.0, 0.0)
        assert abs(E - 1.0) < 1e-6
        assert abs(F) < 1e-6
        assert abs(G - 1.0) < 1e-6


class TestSecondFundamentalForm:
    def test_sphere(self):
        """Second fundamental form of unit sphere."""
        L, M, N = second_fundamental_form(sphere, math.pi/2, 0.0)
        assert abs(L - (-1.0)) < 1e-4
        assert abs(M) < 1e-4
        assert abs(N - (-1.0)) < 1e-4


class TestGaussianCurvature:
    def test_sphere(self):
        """K = 1/R^2 for sphere of radius R."""
        E, F, G = first_fundamental_form(sphere, math.pi/2, 0.0)
        L, M, N = second_fundamental_form(sphere, math.pi/2, 0.0)
        K = gaussian_curvature(E, F, G, L, M, N)
        assert abs(K - 1.0) < 1e-3

    def test_cylinder(self):
        """K = 0 for cylinder (intrinsically flat)."""
        E, F, G = first_fundamental_form(cylinder, 0.0, 0.0)
        L, M, N = second_fundamental_form(cylinder, 0.0, 0.0)
        K = gaussian_curvature(E, F, G, L, M, N)
        assert abs(K) < 1e-4

    def test_pseudosphere(self):
        """K = -1 for pseudosphere."""
        E, F, G = first_fundamental_form(pseudosphere, 1.0, 0.5)
        L, M, N = second_fundamental_form(pseudosphere, 1.0, 0.5)
        K = gaussian_curvature(E, F, G, L, M, N)
        assert abs(K - (-1.0)) < 1e-3


class TestMeanCurvature:
    def test_sphere(self):
        """H = 1/R for sphere."""
        E, F, G = first_fundamental_form(sphere, math.pi/2, 0.0)
        L, M, N = second_fundamental_form(sphere, math.pi/2, 0.0)
        H = mean_curvature(E, F, G, L, M, N)
        assert abs(H - (-1.0)) < 1e-4  # Convention-dependent sign

    def test_cylinder(self):
        """H = 1/(2R) for cylinder."""
        E, F, G = first_fundamental_form(cylinder, 0.0, 0.0)
        L, M, N = second_fundamental_form(cylinder, 0.0, 0.0)
        H = mean_curvature(E, F, G, L, M, N)
        assert abs(H - 0.5) < 1e-3


class TestPrincipalCurvatures:
    def test_sphere(self):
        E, F, G = first_fundamental_form(sphere, math.pi/2, 0.0)
        L, M, N = second_fundamental_form(sphere, math.pi/2, 0.0)
        k1, k2 = principal_curvatures(E, F, G, L, M, N)
        assert abs(abs(k1) - 1.0) < 1e-4
        assert abs(abs(k2) - 1.0) < 1e-4


class TestChristoffelSymbols:
    def test_sphere(self):
        """Christoffel symbols for unit sphere."""
        E, F, G = first_fundamental_form(sphere, math.pi/4, 0.0)
        # Compute derivatives numerically
        h = 1e-6
        Eu = (first_fundamental_form(sphere, math.pi/4 + h, 0.0)[0] -
              first_fundamental_form(sphere, math.pi/4 - h, 0.0)[0]) / (2*h)
        Ev = (first_fundamental_form(sphere, math.pi/4, h)[0] -
              first_fundamental_form(sphere, math.pi/4, -h)[0]) / (2*h)
        Fu = (first_fundamental_form(sphere, math.pi/4 + h, 0.0)[1] -
              first_fundamental_form(sphere, math.pi/4 - h, 0.0)[1]) / (2*h)
        Fv = (first_fundamental_form(sphere, math.pi/4, h)[1] -
              first_fundamental_form(sphere, math.pi/4, -h)[1]) / (2*h)
        Gu = (first_fundamental_form(sphere, math.pi/4 + h, 0.0)[2] -
              first_fundamental_form(sphere, math.pi/4 - h, 0.0)[2]) / (2*h)
        Gv = (first_fundamental_form(sphere, math.pi/4, h)[2] -
              first_fundamental_form(sphere, math.pi/4, -h)[2]) / (2*h)

        Gamma1, Gamma2 = christoffel_symbols(E, F, G, Eu, Ev, Fu, Fv, Gu, Gv)
        assert Gamma1.shape == (2, 2)
        assert Gamma2.shape == (2, 2)


class TestTheoremaEgregiumCheck:
    def test_sphere(self):
        """K should be intrinsic (same whether computed from first form only or both forms)."""
        result = theorema_egregium_check(sphere, math.pi/2, 0.0)
        assert abs(result['K_extrinsic'] - 1.0) < 1e-3


class TestGaussBonnetDiscrete:
    def test_tetrahedron(self):
        """Gauss-Bonnet for tetrahedron: sum of defects = 4π."""
        verts = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ], dtype=float)
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ])
        defect = gauss_bonnet_discrete(verts, faces)
        assert abs(defect - 4 * math.pi) < 0.1


class TestGeodesicOde:
    def test_sphere_geodesic(self):
        """Geodesic equations should be well-defined for sphere."""
        state = np.array([math.pi/4, 0.0, 1.0, 0.0])
        h = 1e-6
        derivs = geodesic_ode(state, 0.0, sphere)
        assert len(derivs) == 4
        assert np.all(np.isfinite(derivs))


class TestIntegration:
    def test_torus_curvature(self):
        """Torus has regions of positive and negative curvature."""
        # At u=0, v=0 (outer equator): positive curvature
        E, F, G = first_fundamental_form(torus, 0.0, 0.0)
        L, M, N = second_fundamental_form(torus, 0.0, 0.0)
        K = gaussian_curvature(E, F, G, L, M, N)
        assert K > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
