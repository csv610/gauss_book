"""Tests for orbital mechanics."""

import pytest
import sys
import os
import math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.orbital import (
    MU_SUN, AU, DAY,
    solve_kepler, mean_anomaly_from_true, true_anomaly_from_mean,
    state_to_elements, elements_to_state, rotation_matrix,
    fg_functions, stumpff_C, stumpff_S,
    propagate_universal, ceres_example
)


class TestConstants:
    def test_mu_sun(self):
        assert abs(MU_SUN - 1.32712440018e20) < 1e10

    def test_au(self):
        assert abs(AU - 149597870700) < 1

    def test_day(self):
        assert DAY == 86400


class TestKeplerEquation:
    def test_e_zero(self):
        """For e=0, E = M."""
        E = solve_kepler(1.0, 0.0)
        assert abs(E - 1.0) < 1e-12

    def test_small_e(self):
        """For small e, should converge quickly."""
        E = solve_kepler(1.0, 0.01)
        assert abs(E - 1.0) < 0.01

    def test_e_near_one(self):
        """For e close to 1, should still converge to correct solution."""
        M = 1.0
        e = 0.99
        E = solve_kepler(M, e)
        # Verify Kepler equation is satisfied
        assert abs(E - e * math.sin(E) - M) < 1e-10

    def test_check_kepler(self):
        """Verify that E - e*sin(E) = M."""
        M = 1.5
        e = 0.5
        E = solve_kepler(M, e)
        assert abs(E - e * math.sin(E) - M) < 1e-10


class TestMeanTrueAnomaly:
    def test_conversion_roundtrip(self):
        """M -> f -> M should be identity."""
        M = 1.0
        e = 0.3
        f = true_anomaly_from_mean(M, e)
        M_back = mean_anomaly_from_true(f, e)
        assert abs(M - M_back) < 1e-10


class TestStateToElements:
    def test_circular_orbit(self):
        """Circular orbit at 1 AU."""
        r = np.array([AU, 0, 0])
        v = np.array([0, math.sqrt(MU_SUN/AU), 0])
        elems = state_to_elements(r, v)
        assert abs(elems['a'] - AU) < AU * 0.01
        assert elems['e'] < 0.01

    def test_elements_structure(self):
        r = np.array([AU, 0, 0])
        v = np.array([0, math.sqrt(MU_SUN/AU), 0])
        elems = state_to_elements(r, v)
        assert 'a' in elems
        assert 'e' in elems
        assert 'i' in elems
        assert 'Omega' in elems
        assert 'omega' in elems
        assert 'M0' in elems


class TestElementsToState:
    def test_circular_orbit(self):
        """Reconstruct state from circular orbit elements."""
        elems = {'a': AU, 'e': 0.0, 'i': 0.0,
                 'Omega': 0.0, 'omega': 0.0, 'M0': 0.0}
        r, v = elements_to_state(elems)
        assert abs(r[0] - AU) < 0.01 * AU
        assert abs(r[1]) < 0.01 * AU


class TestRotationMatrix:
    def test_identity(self):
        """Identity rotation."""
        R = rotation_matrix(0, 0, 0)
        assert np.allclose(R, np.eye(3))

    def test_orthogonality(self):
        """Rotation matrix should be orthogonal."""
        R = rotation_matrix(0.5, 0.3, 0.2)
        assert np.allclose(R @ R.T, np.eye(3))


class TestStumpffFunctions:
    def test_C_zero(self):
        assert abs(stumpff_C(0) - 0.5) < 1e-10

    def test_S_zero(self):
        assert abs(stumpff_S(0) - 1/6) < 1e-10

    def test_C_positive(self):
        """For z > 0, C(z) = (1 - cos(sqrt(z))) / z"""
        z = 1.0
        expected = (1 - math.cos(math.sqrt(z))) / z
        assert abs(stumpff_C(z) - expected) < 1e-10


class TestPropagateUniversal:
    def test_circular_orbit(self):
        """Propagate circular orbit for one period."""
        r0 = np.array([AU, 0, 0])
        v0 = np.array([0, math.sqrt(MU_SUN/AU), 0])
        period = 2 * math.pi * math.sqrt(AU**3 / MU_SUN)
        r, v = propagate_universal(r0, v0, period)
        # Should be back near starting point
        assert abs(np.linalg.norm(r) - AU) < 0.1 * AU


class TestIntegration:
    def test_kepler_with_high_eccentricity(self):
        """Test Kepler equation solver with high eccentricity."""
        for e in [0.1, 0.5, 0.9, 0.99]:
            M = 1.0
            E = solve_kepler(M, e)
            assert abs(E - e * math.sin(E) - M) < 1e-8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
