"""Hyperbolic (non-Euclidean) geometry computations in the Poincare models.

Gauss developed non-Euclidean geometry privately from the 1790s onward but,
fearing the controversy that had engulfed Saccheri and others, published
nothing. After the publications of Lobachevsky (1829) and Bolyai (1832) he
wrote to Schumacher in 1846 that he had "long possessed" the ideas, and his
Werke contain fragments on the "non-Euclidean plane". This module computes
in the two conformal models that Beltrami and Poincare made famous: the
upper half-plane and the unit disk, both of constant curvature -1.
"""

from math import acosh, acos, asin, atan, cos, cosh, exp, pi, sin, sinh, sqrt, tan
from typing import Tuple

__all__ = [
    'half_plane_to_disk',
    'disk_to_half_plane',
    'half_plane_distance',
    'disk_distance',
    'geodesic_circle',
    'angle_of_parallelism',
    'hyperbolic_angles',
    'hyperbolic_law_of_cosines',
    'hyperbolic_side_from_sas',
    'hyperbolic_law_of_sines',
    'verify_hyperbolic_law_of_sines',
    'hyperbolic_triangle_area',
    'circle_circumference',
    'circle_area',
]


def half_plane_to_disk(z: complex) -> complex:
    """Cayley map from the upper half-plane to the unit disk: z -> (z - i)/(z + i)."""
    return (z - 1j) / (z + 1j)


def disk_to_half_plane(z: complex) -> complex:
    """Inverse Cayley map from the unit disk to the upper half-plane: z -> i(1+z)/(1-z)."""
    return 1j * (1 + z) / (1 - z)


def half_plane_distance(z1: complex, z2: complex) -> float:
    """Hyperbolic distance in the upper half-plane model.

    d(z1, z2) = arcosh(1 + |z1 - z2|^2 / (2 * Im(z1) * Im(z2))).
    """
    if z1.imag <= 0 or z2.imag <= 0:
        raise ValueError("both points must lie in the open upper half-plane")
    num = abs(z1 - z2) ** 2
    den = 2 * z1.imag * z2.imag
    return acosh(1 + num / den)


def disk_distance(z1: complex, z2: complex) -> float:
    """Hyperbolic distance in the Poincare disk model.

    d(z1, z2) = arcosh(1 + 2 * |z1 - z2|^2 / ((1 - |z1|^2)(1 - |z2|^2))).
    """
    if abs(z1) >= 1 or abs(z2) >= 1:
        raise ValueError("both points must lie inside the unit disk")
    num = abs(z1 - z2) ** 2
    den = (1 - abs(z1) ** 2) * (1 - abs(z2) ** 2)
    return acosh(1 + 2 * num / den)


def geodesic_circle(p: complex, q: complex) -> Tuple[complex, float]:
    """Euclidean circle carrying the geodesic through p and q in the disk.

    In the Poincare disk, geodesics are arcs of circles meeting the unit
    circle orthogonally.  A circle orthogonal to the unit circle has center
    C and radius R with |C|^2 - R^2 = 1.  Requiring it to pass through p
    and q gives two linear equations for C:
    2 Re(C conj(p)) = |p|^2 + 1,  2 Re(C conj(q)) = |q|^2 + 1.
    Returns (center, radius).  For a diameter (p, q collinear with the
    origin) returns (None, inf).
    """
    if abs(p) >= 1 or abs(q) >= 1:
        raise ValueError("points must lie inside the unit disk")
    b1 = (abs(p) ** 2 + 1) / 2.0
    b2 = (abs(q) ** 2 + 1) / 2.0
    det = p.real * q.imag - p.imag * q.real
    if abs(det) < 1e-12:
        return (None, float('inf'))
    x = (b1 * q.imag - p.imag * b2) / det
    y = (p.real * b2 - b1 * q.real) / det
    center = complex(x, y)
    radius = sqrt(abs(center) ** 2 - 1)
    return center, radius


def angle_of_parallelism(x: float) -> float:
    """Lobachevsky's angle of parallelism Pi(x) = 2 * arctan(e^{-x}).

    In the half-plane model, the distance from a point to the boundary line
    at which the boundary is first seen at angle Pi(x); Pi(x) tends to pi/2
    as x -> 0 (Euclidean limit) and to 0 as x -> infinity.
    """
    return 2 * atan(exp(-x))


def hyperbolic_angles(a: float, b: float, c: float) -> Tuple[float, float, float]:
    """Angles of a hyperbolic triangle from its side lengths.

    Applies the hyperbolic law of cosines
    cos C = (cosh a cosh b - cosh c) / (sinh a sinh b)
    at each vertex.  The angle sum is always less than pi.
    """
    if min(a, b, c) <= 0:
        raise ValueError("side lengths must be positive")
    A = acos((cosh(b) * cosh(c) - cosh(a)) / (sinh(b) * sinh(c)))
    B = acos((cosh(a) * cosh(c) - cosh(b)) / (sinh(a) * sinh(c)))
    C = acos((cosh(a) * cosh(b) - cosh(c)) / (sinh(a) * sinh(b)))
    return A, B, C


def hyperbolic_law_of_cosines(a: float, b: float, c: float) -> Tuple[float, float, float]:
    """Alias of hyperbolic_angles: angles (A, B, C) from sides (a, b, c)."""
    return hyperbolic_angles(a, b, c)


def hyperbolic_side_from_sas(a: float, b: float, C: float) -> float:
    """Third side c given two sides a, b and included angle C.

    c = arcosh(cosh a cosh b - sinh a sinh b cos C).
    """
    return acosh(cosh(a) * cosh(b) - sinh(a) * sinh(b) * cos(C))


def verify_hyperbolic_law_of_sines(a: float, b: float, c: float,
                                   tol: float = 1e-9) -> bool:
    """Check that sinh a / sin A = sinh b / sin B = sinh c / sin C."""
    A, B, C = hyperbolic_angles(a, b, c)
    r1 = sinh(a) / sin(A)
    r2 = sinh(b) / sin(B)
    r3 = sinh(c) / sin(C)
    return abs(r1 - r2) < tol and abs(r2 - r3) < tol


def hyperbolic_law_of_sines(a: float, b: float, c: float) -> float:
    """The common ratio sinh(a)/sin(A) = sinh(b)/sin(B) = sinh(c)/sin(C)."""
    A, B, C = hyperbolic_angles(a, b, c)
    r1 = sinh(a) / sin(A)
    r2 = sinh(b) / sin(B)
    r3 = sinh(c) / sin(C)
    if not (abs(r1 - r2) < 1e-8 and abs(r2 - r3) < 1e-8):
        raise ValueError("sides do not form a hyperbolic triangle")
    return r1


def hyperbolic_triangle_area(a: float, b: float, c: float) -> float:
    """Area of a hyperbolic triangle with curvature -1 (Gauss-Bonnet).

    Area = pi - (A + B + C), the angle defect.  In the Euclidean plane the
    defect is 0; here it is always positive and bounded above by pi.
    """
    A, B, C = hyperbolic_angles(a, b, c)
    return pi - (A + B + C)


def circle_circumference(r: float) -> float:
    """Circumference of a hyperbolic circle of radius r: 2 * pi * sinh(r)."""
    if r < 0:
        raise ValueError("radius must be non-negative")
    return 2 * pi * sinh(r)


def circle_area(r: float) -> float:
    """Area of a hyperbolic circle of radius r: 2 * pi * (cosh(r) - 1)."""
    if r < 0:
        raise ValueError("radius must be non-negative")
    return 2 * pi * (cosh(r) - 1)


if __name__ == "__main__":
    print("=== Hyperbolic Geometry Demo ===")

    print("\n1. Distance between models is invariant under the Cayley map:")
    i1, i2 = 0.5j, 0.3 + 0.7j
    d_half = half_plane_distance(i1, i2)
    d_disk = disk_distance(half_plane_to_disk(i1), half_plane_to_disk(i2))
    print(f"   half-plane distance = {d_half:.6f}")
    print(f"   disk distance       = {d_disk:.6f}")

    print("\n2. Geodesic: d(i, i e^t) = t for the vertical ray:")
    for t in (0.5, 1.0, 2.0):
        print(f"   t={t}: d = {half_plane_distance(1j, 1j * exp(t)):.6f}")

    print("\n3. Angle of parallelism Pi(x) = 2 arctan(e^{-x}):")
    for x in (0.0, 0.5, 1.0, 2.0, 5.0):
        print(f"   Pi({x}) = {angle_of_parallelism(x):.4f} rad = "
              f"{angle_of_parallelism(x) * 180 / pi:.2f} deg")

    print("\n4. Hyperbolic triangle with sides a = b = c = 1.5:")
    A, B, C = hyperbolic_angles(1.5, 1.5, 1.5)
    print(f"   angles = {A:.4f} rad each, sum = {A + B + C:.4f} "
          f"(< pi = {pi:.4f})")
    print(f"   area (defect) = {hyperbolic_triangle_area(1.5, 1.5, 1.5):.4f}")
    print(f"   law of sines holds: {verify_hyperbolic_law_of_sines(1.5, 1.5, 1.5)}")

    print("\n5. Circle growth in hyperbolic space:")
    for r in (0.5, 1.0, 2.0, 4.0):
        print(f"   r={r}: circumference = {circle_circumference(r):.4f} "
              f"(Euclidean: {2 * pi * r:.4f}), "
              f"area = {circle_area(r):.4f} (Euclidean: {pi * r * r:.4f})")
