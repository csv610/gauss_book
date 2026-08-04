"""Tests for the geodesy module."""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math import hypot, isclose, pi, radians, sin, sqrt

from python.geodesy import (
    haversine_distance, vincenty_distance, spherical_excess,
    forward_triangulation, intersection, resection,
    adjust_triangle_angles, conditional_adjustment, meridian_arc_degree,
    _angle_at,
)

A, B, C = 6378137.0, 1.0 / 298.257223563, 6371008.8


def assert_close(a, b, tol=1e-6):
    assert isclose(a, b, abs_tol=tol), f"{a} != {b}"


class TestHaversine:
    def test_same_point(self):
        assert haversine_distance(0, 0, 0, 0) == 0
        assert haversine_distance(40.5, -73.9, 40.5, -73.9) == 0

    def test_equator_degree(self):
        # 1 degree of longitude on the equator
        assert_close(haversine_distance(0, 0, 0, 1), 2 * pi * C / 360, tol=1.0)

    def test_known_distance(self):
        # Paris (48.8566, 2.3522) to London (51.5074, -0.1278): ~344 km
        d = haversine_distance(48.8566, 2.3522, 51.5074, -0.1278)
        assert 340000 < d < 350000

    def test_symmetric(self):
        d1 = haversine_distance(30, 40, -10, 60)
        d2 = haversine_distance(-10, 60, 30, 40)
        assert_close(d1, d2)


class TestVincenty:
    def test_quarter_equator(self):
        assert_close(vincenty_distance(0, 0, 0, 90), pi * A / 2, tol=0.1)

    def test_meridian_equator_degree(self):
        assert_close(vincenty_distance(0, 0, 1, 0), 110574.39, tol=0.1)

    def test_known_distance(self):
        # same Paris-London pair on the ellipsoid
        d = vincenty_distance(48.8566, 2.3522, 51.5074, -0.1278)
        assert 340000 < d < 350000

    def test_symmetric(self):
        d1 = vincenty_distance(30, 40, -10, 60)
        d2 = vincenty_distance(-10, 60, 30, 40)
        assert_close(d1, d2, tol=1e-3)

    def test_short_distance(self):
        d = vincenty_distance(0, 0, 0, 0.001)
        assert 100 < d < 120  # ~111 m


class TestSphericalExcess:
    def test_octant(self):
        # Spherical triangle (0,0), (0,90), (90,0) covers 1/8 of the sphere
        e, area = spherical_excess(0, 0, 0, 90, 90, 0)
        assert_close(e, pi / 2, tol=1e-6)
        assert_close(area, pi * C ** 2 / 2, tol=1e3)

    def test_small_triangle(self):
        # A tiny triangle has nearly zero excess
        e, area = spherical_excess(0, 0, 0, 0.1, 0.1, 0)
        assert 0 < e < 1e-4

    def test_angle_sum(self):
        # Sides of the octant triangle are quarters of great circles
        assert spherical_excess(0, 0, 0, 90, 90, 0)[0] > 0


class TestIntersection:
    def test_right_angle_rays(self):
        p = intersection((0, 0), (1, 0), radians(45), radians(135))
        point, t1, t2 = p
        assert_close(point[0], 0.5, tol=1e-9)
        assert_close(point[1], 0.5, tol=1e-9)

    def test_parallel_raises(self):
        with pytest.raises(ValueError):
            intersection((0, 0), (0, 1), radians(90), radians(90))


class TestForwardTriangulation:
    def test_equilateral(self):
        pt, d1, d2 = forward_triangulation((0, 0), (1, 0), radians(60), radians(60))
        assert_close(pt[0], 0.5, tol=1e-9)
        assert_close(pt[1], sqrt(3) / 2, tol=1e-9)
        assert_close(d1, 1.0, tol=1e-9)
        assert_close(d2, 1.0, tol=1e-9)

    def test_scaled_baseline(self):
        scale = 123.456
        pt, d1, d2 = forward_triangulation((0, 0), (scale, 0), radians(60), radians(60))
        assert_close(pt[0], scale / 2, tol=1e-9)
        assert_close(pt[1], scale * sqrt(3) / 2, tol=1e-9)
        assert_close(d1, scale, tol=1e-9)

    def test_chain(self):
        # Build a chain of two equilateral triangles on one baseline; the
        # second triangle is again equilateral on the slanted baseline.
        pt1, _, _ = forward_triangulation((0, 0), (1, 0), radians(60), radians(60))
        pt2, d1, d2 = forward_triangulation(pt1, (1, 0), radians(60), radians(60))
        assert_close(d1, 1.0, tol=1e-9)
        assert_close(d2, 1.0, tol=1e-9)
        assert_close(hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]), 1.0, tol=1e-9)

    def test_does_not_close(self):
        with pytest.raises(ValueError):
            forward_triangulation((0, 0), (1, 0), radians(100), radians(100))


class TestResection:
    def test_recovers_point(self):
        stations = [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0)]
        for target in [(0.7, 0.9), (1.2, 1.4), (0.4, 0.2)]:
            alpha = _angle_at(target, stations[0], stations[1])
            beta = _angle_at(target, stations[1], stations[2])
            solved = resection(*stations, alpha, beta)
            assert_close(solved[0], target[0], tol=1e-8)
            assert_close(solved[1], target[1], tol=1e-8)

    def test_recovers_other_orientation(self):
        # Mirror configuration on the other side of the baseline
        stations = [(0.0, 0.0), (2.0, 0.0), (3.0, 1.5)]
        target = (1.0, -0.7)
        alpha = _angle_at(target, stations[0], stations[1])
        beta = _angle_at(target, stations[1], stations[2])
        solved = resection(*stations, alpha, beta)
        assert_close(solved[0], target[0], tol=1e-8)
        assert_close(solved[1], target[1], tol=1e-8)

    def test_equilateral_configuration(self):
        stations = [(0.0, 0.0), (1.0, 0.0), (0.5, sqrt(3) / 2)]
        target = (0.5, 0.3)
        alpha = _angle_at(target, stations[0], stations[1])
        beta = _angle_at(target, stations[1], stations[2])
        solved = resection(*stations, alpha, beta)
        assert_close(solved[0], target[0], tol=1e-8)
        assert_close(solved[1], target[1], tol=1e-8)

    def test_invalid_angles(self):
        with pytest.raises(ValueError):
            resection((0, 0), (1, 0), (2, 1), 0.0, 0.0)
        with pytest.raises(ValueError):
            resection((0, 0), (1, 0), (2, 1), pi, pi)


class TestAdjustTriangle:
    def test_sums_to_180(self):
        for angles in [[59.95, 60.05, 60.03], [60.0, 60.0, 60.5], [30.0, 70.0, 80.5]]:
            adjusted = adjust_triangle_angles(angles)
            assert_close(sum(adjusted), 180.0, tol=1e-9)
            for a, b in zip(angles, adjusted):
                assert_close(a - b, (sum(angles) - 180.0) / 3, tol=1e-9)

    def test_equal_correction(self):
        adjusted = adjust_triangle_angles([59.9, 60.0, 60.4])
        assert_close(adjusted[0], 59.8, tol=1e-9)
        assert_close(adjusted[1], 59.9, tol=1e-9)
        assert_close(adjusted[2], 60.3, tol=1e-9)

    def test_bad_input(self):
        with pytest.raises(ValueError):
            adjust_triangle_angles([60, 60])


class TestConditionalAdjustment:
    def test_single_constraint(self):
        # Three angles of one triangle, equal weights
        A = np.array([[1.0, 1.0, 1.0]])
        w = np.array([0.03])
        c = conditional_adjustment(A, w)
        assert_close(c[0], -0.01)
        assert_close(c[1], -0.01)
        assert_close(c[2], -0.01)
        assert np.allclose(A @ c, -w)

    def test_two_triangles(self):
        A = np.zeros((2, 6))
        A[0, :3] = 1
        A[1, 3:] = 1
        w = np.array([0.002, 0.001])
        c = conditional_adjustment(A, w)
        assert np.allclose(A @ c, -w)
        assert np.allclose(c[:3], -0.002 / 3)
        assert np.allclose(c[3:], -0.001 / 3)

    def test_weighted(self):
        # A less precise angle (smaller weight) receives a larger correction
        A = np.array([[1.0, 1.0, 1.0]])
        w = np.array([0.03])
        weights = np.array([1.0, 1.0, 0.1])
        c = conditional_adjustment(A, w, weights)
        assert abs(c[2]) > abs(c[0])
        assert np.allclose(A @ c, -w)

    def test_bad_shapes(self):
        with pytest.raises(ValueError):
            conditional_adjustment(np.ones((1, 3)), np.ones(2))


class TestMeridianArc:
    def test_equator(self):
        assert 110570 < meridian_arc_degree(0) < 110580

    def test_45(self):
        assert 111135 < meridian_arc_degree(45) < 111150

    def test_increases_toward_pole(self):
        assert meridian_arc_degree(80) > meridian_arc_degree(0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
