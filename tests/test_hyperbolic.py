"""Tests for the hyperbolic geometry module."""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math import cos, cosh, exp, isclose, pi, sin, sinh

from python.hyperbolic import (
    half_plane_to_disk, disk_to_half_plane, half_plane_distance,
    disk_distance, geodesic_circle, angle_of_parallelism, hyperbolic_angles,
    hyperbolic_law_of_cosines, hyperbolic_side_from_sas, hyperbolic_law_of_sines,
    verify_hyperbolic_law_of_sines, hyperbolic_triangle_area,
    circle_circumference, circle_area,
)


def assert_close(a, b, tol=1e-9):
    assert isclose(a, b, abs_tol=tol), f"{a} != {b}"


class TestCayleyMap:
    def test_round_trip(self):
        for z in [0.2 + 0.7j, 1.5j, 2.0 + 3.0j, 0.1 + 0.1j]:
            w = half_plane_to_disk(z)
            assert abs(disk_to_half_plane(w) - z) < 1e-12

    def test_i_maps_to_origin(self):
        assert abs(half_plane_to_disk(1j)) < 1e-15

    def test_disk_inside_unit(self):
        for z in [1j, 1 + 1j, 0.5 + 0.1j, 10j]:
            assert abs(half_plane_to_disk(z)) < 1 - 1e-12

    def test_boundary_of_disk_is_real_axis(self):
        for x in [0, 1, 5, -3]:
            assert abs(abs(half_plane_to_disk(x + 0j)) - 1) < 1e-12


class TestDistances:
    def test_models_agree(self):
        z1, z2 = 0.3 + 0.9j, 1.2 + 0.4j
        d1 = half_plane_distance(z1, z2)
        d2 = disk_distance(half_plane_to_disk(z1), half_plane_to_disk(z2))
        assert_close(d1, d2)

    def test_vertical_ray(self):
        for t in [0.1, 0.5, 1.0, 2.0, 3.5]:
            assert_close(half_plane_distance(1j, 1j * exp(t)), t)

    def test_symmetric(self):
        z1, z2 = 0.2 + 0.5j, 1.0 + 2.0j
        assert_close(half_plane_distance(z1, z2), half_plane_distance(z2, z1))

    def test_zero(self):
        z = 0.5 + 0.5j
        assert_close(half_plane_distance(z, z), 0)

    def test_triangle_inequality(self):
        z1, z2, z3 = 1j, 0.5 + 1.5j, 2j
        assert (half_plane_distance(z1, z3) <=
                half_plane_distance(z1, z2) + half_plane_distance(z2, z3) + 1e-9)

    def test_outside_half_plane(self):
        with pytest.raises(ValueError):
            half_plane_distance(0j, 1j)
        with pytest.raises(ValueError):
            disk_distance(1.0 + 0j, 0j)


class TestGeodesicCircle:
    def test_orthogonal_to_unit_circle(self):
        for p, q in [(0.2 + 0.3j, -0.3 + 0.5j),
                     (0.6j, 0.4 + 0.2j),
                     (-0.5 - 0.3j, 0.2 + 0.8j)]:
            c, r = geodesic_circle(p, q)
            assert_close(abs(c) ** 2 - r ** 2, 1.0, tol=1e-9)

    def test_passes_through_points(self):
        for p, q in [(0.2 + 0.3j, -0.3 + 0.5j), (0.6j, 0.4 + 0.2j)]:
            c, r = geodesic_circle(p, q)
            assert_close(abs(p - c), r, tol=1e-9)
            assert_close(abs(q - c), r, tol=1e-9)


class TestAngleOfParallelism:
    def test_values(self):
        assert_close(angle_of_parallelism(0), pi / 2)
        assert_close(angle_of_parallelism(1), 0.70502684, tol=1e-6)
        assert_close(angle_of_parallelism(5), 0.01347569, tol=1e-6)

    def test_limits(self):
        assert angle_of_parallelism(0.0001) < pi / 2
        assert angle_of_parallelism(0.0001) > angle_of_parallelism(1)
        assert angle_of_parallelism(100) < 1e-40


class TestHyperbolicTriangle:
    def test_equilateral(self):
        A, B, C = hyperbolic_angles(1.5, 1.5, 1.5)
        assert_close(A, B)
        assert_close(B, C)
        assert A < pi / 3
        assert_close(A + B + C, 2.37901683, tol=1e-6)

    def test_angle_sum_less_than_pi(self):
        for a, b, c in [(1.0, 1.0, 1.0), (2.0, 1.5, 1.0), (0.5, 0.7, 0.9)]:
            A, B, C = hyperbolic_angles(a, b, c)
            assert A + B + C < pi, f"({a},{b},{c})"

    def test_area_is_defect(self):
        for sides in [(1.5, 1.5, 1.5), (2.0, 1.5, 1.0), (0.5, 0.7, 0.9)]:
            A, B, C = hyperbolic_angles(*sides)
            assert_close(hyperbolic_triangle_area(*sides), pi - (A + B + C))

    def test_small_triangle_approaches_euclidean(self):
        # very small sides -> angles sum approaches pi, area -> 0
        A, B, C = hyperbolic_angles(0.001, 0.001, 0.001)
        assert_close(A + B + C, pi, tol=1e-6)
        assert hyperbolic_triangle_area(0.001, 0.001, 0.001) < 1e-6

    def test_law_of_cosines(self):
        a, b, c = 1.2, 1.5, 0.8
        A, B, C = hyperbolic_law_of_cosines(a, b, c)
        # verify each cosine law directly
        assert_close(cosh(c), cosh(a) * cosh(b) - sinh(a) * sinh(b) * cos(C))
        assert_close(cosh(a), cosh(b) * cosh(c) - sinh(b) * sinh(c) * cos(A))
        assert_close(cosh(b), cosh(a) * cosh(c) - sinh(a) * sinh(c) * cos(B))

    def test_sas_round_trip(self):
        a, b, C = 1.2, 1.5, 0.9
        c = hyperbolic_side_from_sas(a, b, C)
        A2, B2, C2 = hyperbolic_angles(a, b, c)
        assert_close(C2, C)

    def test_law_of_sines(self):
        for sides in [(1.2, 1.5, 0.8), (1.0, 1.0, 1.0), (2.0, 1.7, 1.3)]:
            assert verify_hyperbolic_law_of_sines(*sides)
            r = hyperbolic_law_of_sines(*sides)
            A, B, C = hyperbolic_angles(*sides)
            assert_close(r, sinh(sides[0]) / sin(A))


class TestCircleGrowth:
    def test_formulas(self):
        for r in [0.1, 0.5, 1.0, 2.5]:
            assert_close(circle_circumference(r), 2 * pi * sinh(r))
            assert_close(circle_area(r), 2 * pi * (cosh(r) - 1))

    def test_exceeds_euclidean(self):
        for r in [0.5, 1.0, 2.0]:
            assert circle_circumference(r) > 2 * pi * r
            assert circle_area(r) > pi * r * r

    def test_small_circle(self):
        r = 0.001
        assert abs(circle_circumference(r) / (2 * pi * r) - 1) < 1e-6
        assert abs(circle_area(r) / (pi * r * r) - 1) < 1e-6

    def test_negative_radius(self):
        with pytest.raises(ValueError):
            circle_circumference(-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
