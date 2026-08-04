"""Geodesy: triangulation, the three-point problem, and least-squares adjustment.

From 1818 to 1832 Gauss directed the triangulation of the Kingdom of
Hannover, inventing the heliotrope for measuring distant angles.  The survey
rested on the classical machinery of this chapter -- baseline measurement,
triangle chains solved by the sine rule, spherical excess, and the
adjustment of inconsistent angle observations by the method of least squares
(Gauss's principal innovation, developed precisely for this geodetic work).
Coordinates are given as (east, north) tuples; latitudes and longitudes in
degrees with WGS84 for the ellipsoidal routines.
"""

from math import acos, atan, atan2, cos, degrees, hypot, pi, radians, sin, sqrt, tan
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    'haversine_distance',
    'vincenty_distance',
    'spherical_excess',
    'forward_triangulation',
    'intersection',
    'resection',
    'adjust_triangle_angles',
    'conditional_adjustment',
    'meridian_arc_degree',
]

_EARTH_RADIUS = 6371008.8
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float,
                       radius: float = _EARTH_RADIUS) -> float:
    """Great-circle distance between two points (degrees) on a sphere."""
    p1, l1, p2, l2 = map(radians, (lat1, lon1, lat2, lon2))
    dphi = p2 - p1
    dlamb = l2 - l1
    a = sin(dphi / 2) ** 2 + cos(p1) * cos(p2) * sin(dlamb / 2) ** 2
    return 2 * radius * atan2(sqrt(a), sqrt(1 - a))


def vincenty_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Ellipsoidal distance (meters) on the WGS84 ellipsoid (Vincenty inverse).

    Accurate to sub-millimeter on non-antipodal pairs.
    """
    phi1, phi2 = radians(lat1), radians(lat2)
    L = radians(lon2 - lon1)
    U1 = atan((1 - _WGS84_F) * tan(phi1))
    U2 = atan((1 - _WGS84_F) * tan(phi2))
    sinU1, cosU1 = sin(U1), cos(U1)
    sinU2, cosU2 = sin(U2), cos(U2)
    lam = L
    for _ in range(200):
        sin_lam, cos_lam = sin(lam), cos(lam)
        sin_sigma = sqrt((cosU2 * sin_lam) ** 2 +
                         (cosU1 * sinU2 - sinU1 * cosU2 * cos_lam) ** 2)
        if sin_sigma == 0:
            return 0.0
        cos_sigma = sinU1 * sinU2 + cosU1 * cosU2 * cos_lam
        sigma = atan2(sin_sigma, cos_sigma)
        sin_alpha = cosU1 * cosU2 * sin_lam / sin_sigma
        cos_sq_alpha = 1 - sin_alpha ** 2
        cos_2sigma_m = (cos_sigma - 2 * sinU1 * sinU2 / cos_sq_alpha
                        if cos_sq_alpha else 0.0)
        C = _WGS84_F / 16 * cos_sq_alpha * (4 + _WGS84_F * (4 - 3 * cos_sq_alpha))
        lam_prev = lam
        lam = L + (1 - C) * _WGS84_F * sin_alpha * (
            sigma + C * sin_sigma * (cos_2sigma_m + C * cos_sigma *
                                     (-1 + 2 * cos_2sigma_m ** 2)))
        if abs(lam - lam_prev) < 1e-12:
            break
    else:
        raise ValueError("Vincenty formula failed to converge (antipodal points?)")
    u_sq = cos_sq_alpha * (_WGS84_A ** 2 - (_WGS84_A * (1 - _WGS84_F)) ** 2) / \
        (_WGS84_A * (1 - _WGS84_F)) ** 2
    A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
    cos_2sigma_m = (cos_sigma - 2 * sinU1 * sinU2 / cos_sq_alpha
                    if cos_sq_alpha else 0.0)
    delta_sigma = B * sin_sigma * (
        cos_2sigma_m + B / 4 * (cos_sigma * (-1 + 2 * cos_2sigma_m ** 2) -
                                B / 6 * cos_2sigma_m * (-3 + 4 * sin_sigma ** 2) *
                                (-3 + 4 * cos_2sigma_m ** 2)))
    s = _WGS84_A * (1 - _WGS84_F) * A * (sigma - delta_sigma)
    return s


def spherical_excess(lat1: float, lon1: float, lat2: float, lon2: float,
                     lat3: float, lon3: float,
                     radius: float = _EARTH_RADIUS) -> Tuple[float, float]:
    """Spherical excess E and area of the triangle with the given vertices.

    Uses l'Huilier's theorem on the sphere of the given radius:
    tan(E/4) = sqrt(tan(s/2) tan((s-a)/2) tan((s-b)/2) tan((s-c)/2)),
    where a, b, c are side lengths in radians.  Returns
    (excess in steradians, area in square meters = E * radius^2).
    """
    a = haversine_distance(lat1, lon1, lat2, lon2, radius) / radius
    b = haversine_distance(lat2, lon2, lat3, lon3, radius) / radius
    c = haversine_distance(lat3, lon3, lat1, lon1, radius) / radius
    s = (a + b + c) / 2
    t = sqrt(max(0.0, tan(s / 2) * tan((s - a) / 2) * tan((s - b) / 2) * tan((s - c) / 2)))
    E = 4 * atan(t)
    return E, E * radius ** 2


def intersection(p1: Tuple[float, float], p2: Tuple[float, float],
                 alpha1: float, alpha2: float) -> Tuple[Tuple[float, float], float, float]:
    """Intersect two rays from p1 and p2 with bearings alpha1, alpha2 (radians).

    Returns ((x, y), t1, t2) where the point is p1 + t1 * d1 = p2 + t2 * d2
    and d = (sin(alpha), cos(alpha)) points from north.
    """
    d1 = np.array([sin(alpha1), cos(alpha1)])
    d2 = np.array([sin(alpha2), cos(alpha2)])
    A = np.column_stack([d1, -d2])
    b = np.array(p2, dtype=float) - np.array(p1, dtype=float)
    if abs(np.linalg.det(A)) < 1e-14:
        raise ValueError("rays are parallel")
    t1, t2 = np.linalg.solve(A, b)
    point = tuple((np.array(p1, dtype=float) + t1 * d1).tolist())
    return point, float(t1), float(t2)


def forward_triangulation(p1: Tuple[float, float], p2: Tuple[float, float],
                          angle1: float, angle2: float) -> Tuple[Tuple[float, float], float, float]:
    """Locate a third vertex given a baseline p1-p2 and two interior angles.

    This is the core step of a triangulation chain: from a measured baseline
    and the angles at its endpoints, solve the triangle by the sine rule
    without measuring the long sides.  Returns ((x, y), dist1, dist2).
    """
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    theta = atan2(dx, dy)
    alpha1 = theta - angle1
    alpha2 = theta + pi + angle2
    point, t1, t2 = intersection(p1, p2, alpha1, alpha2)
    if t1 < 0 or t2 < 0:
        raise ValueError("baseline angles do not close toward the same side")
    return point, t1, t2


def _circle_from_chord(p: Tuple[float, float], q: Tuple[float, float],
                       theta: float) -> Tuple[Tuple[Tuple[float, float], float],
                                              Tuple[Tuple[float, float], float]]:
    """Circles through p and q from which chord pq subtends angle theta.

    Returns ((center1, r), (center2, r)) for the two possible circles.
    """
    dx, dy = q[0] - p[0], q[1] - p[1]
    c = hypot(dx, dy)
    if c == 0:
        raise ValueError("chord endpoints coincide")
    mid = (p[0] + dx / 2, p[1] + dy / 2)
    perp = (-dy / c, dx / c)
    r = (c / 2) / abs(sin(theta))
    h = (c / 2) / tan(theta)
    c1 = (mid[0] + h * perp[0], mid[1] + h * perp[1])
    c2 = (mid[0] - h * perp[0], mid[1] - h * perp[1])
    return (c1, r), (c2, r)


def _circle_intersection(c1: Tuple[float, float], r1: float,
                         c2: Tuple[float, float], r2: float) -> List[Tuple[float, float]]:
    """Intersection points of two Euclidean circles."""
    dx, dy = c2[0] - c1[0], c2[1] - c1[1]
    d = hypot(dx, dy)
    if d > r1 + r2 or d < abs(r1 - r2):
        return []
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h2 = r1 ** 2 - a ** 2
    h = sqrt(max(0.0, h2))
    px, py = c1[0] + a * dx / d, c1[1] + a * dy / d
    ox, oy = -dy / d * h, dx / d * h
    return [(px + ox, py + oy), (px - ox, py - oy)]


def _angle_at(p: Tuple[float, float], a: Tuple[float, float],
              b: Tuple[float, float]) -> float:
    """Angle a-p-b at p (radians)."""
    u = np.array([a[0] - p[0], a[1] - p[1]])
    v = np.array([b[0] - p[0], b[1] - p[1]])
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        raise ValueError("degenerate angle")
    return acos(max(-1.0, min(1.0, float(u @ v / (nu * nv)))))


def resection(p1: Tuple[float, float], p2: Tuple[float, float],
              p3: Tuple[float, float], alpha: float, beta: float) -> Tuple[float, float]:
    """Solve the three-point (Snellius-Pothenot) problem.

    Given three known stations p1, p2, p3 and the angles
    alpha = angle(p1, P, p2) and beta = angle(p2, P, p3) measured from the
    unknown point P, return the coordinates of P.
    """
    if not (0 < alpha < pi and 0 < beta < pi):
        raise ValueError("angles must be between 0 and pi")
    (ca, ra), (ca2, ra2) = _circle_from_chord(p1, p2, alpha)
    (cb, rb), (cb2, rb2) = _circle_from_chord(p2, p3, beta)
    best = None
    best_err = float('inf')
    for circle1, r1 in ((ca, ra), (ca2, ra2)):
        for circle2, r2 in ((cb, rb), (cb2, rb2)):
            for point in _circle_intersection(circle1, r1, circle2, r2):
                if abs(point[0] - p2[0]) < 1e-9 and abs(point[1] - p2[1]) < 1e-9:
                    continue
                err = (abs(_angle_at(point, p1, p2) - alpha) +
                       abs(_angle_at(point, p2, p3) - beta))
                if err < best_err:
                    best_err = err
                    best = point
    if best is not None and best_err < 1e-6:
        return best
    raise ValueError("no consistent solution found")


def adjust_triangle_angles(angles: List[float]) -> List[float]:
    """Correct triangle angles (degrees) to sum to 180.

    Gauss's practical corner adjustment distributes the closing error
    equally among the three angles.
    """
    if len(angles) != 3:
        raise ValueError("a triangle has exactly three angles")
    correction = (sum(angles) - 180.0) / 3.0
    return [a - correction for a in angles]


def conditional_adjustment(A: np.ndarray, w: np.ndarray,
                           weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Least-squares corrections c subject to the conditions A c = -w.

    Minimizes c' P c over corrections c closing the misclosures w
    (so that A c = -w).  The weight matrix P is diagonal by default.
    This is the conditional adjustment Gauss used for triangulation
    networks: c = -P^{-1} A' (A P^{-1} A')^{-1} w.
    """
    A = np.asarray(A, dtype=float)
    w = np.asarray(w, dtype=float)
    if A.ndim != 2 or w.ndim != 1 or A.shape[0] != w.shape[0]:
        raise ValueError("A must be (r, n) and w of length r")
    n = A.shape[1]
    if weights is None:
        Pinv = np.eye(n)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape == (n, n):
            Pinv = np.linalg.inv(weights)
        else:
            Pinv = np.diag(1.0 / weights)
    N = A @ Pinv @ A.T
    try:
        rhs = np.linalg.solve(N, w)
    except np.linalg.LinAlgError:
        raise ValueError("condition matrix has linearly dependent rows")
    return -Pinv @ A.T @ rhs


def meridian_arc_degree(lat: float) -> float:
    """Length in meters of one degree of latitude at the given latitude (WGS84)."""
    return vincenty_distance(lat, 0.0, lat + 1.0, 0.0)


if __name__ == "__main__":
    print("=== Geodesy Demo ===")

    print("\n1. Baseline triangulation (equilateral triangle on a 1 km baseline):")
    p1 = (0.0, 0.0)
    p2 = (1.0, 0.0)
    pt, d1, d2 = forward_triangulation(p1, p2, radians(60), radians(60))
    print(f"   third vertex = ({pt[0]:.6f}, {pt[1]:.6f})  (expected (0.5, 0.866))")
    print(f"   equal sides  = {d1:.6f}, {d2:.6f}")

    print("\n2. Three-point (resection) problem:")
    A, B, C = (0.0, 0.0), (1.0, 0.0), (2.0, 1.0)
    P = (0.7, 0.9)
    alpha = _angle_at(P, A, B)
    beta = _angle_at(P, B, C)
    solved = resection(A, B, C, alpha, beta)
    print(f"   observed angles = {degrees(alpha):.2f} deg, {degrees(beta):.2f} deg")
    print(f"   recovered P     = ({solved[0]:.6f}, {solved[1]:.6f})")

    print("\n3. Spherical excess of a large triangle:")
    e, area = spherical_excess(0, 0, 0, 10, 10, 0)
    print(f"   excess = {degrees(e):.6f} deg, area = {area:,.0f} m^2")

    print("\n4. Triangle angle adjustment:")
    raw = [59.95, 60.05, 60.03]
    print(f"   raw = {raw}, sum = {sum(raw):.4f}")
    print(f"   adjusted = {[round(a, 4) for a in adjust_triangle_angles(raw)]}, "
          f"sum = {sum(adjust_triangle_angles(raw)):.4f}")

    print("\n5. Conditional adjustment of two triangle sums (a small network):")
    obs = [60.001, 60.002, 59.999, 60.003, 59.998, 60.000]
    w = np.array([sum(obs[:3]) - 180.0, sum(obs[3:]) - 180.0])
    A = np.zeros((2, 6))
    A[0, :3] = 1
    A[1, 3:] = 1
    c = conditional_adjustment(A, w)
    print(f"   misclosures = {[round(x, 4) for x in w]}")
    print(f"   corrections = {[round(x, 5) for x in c]}")
    print(f"   closed: A c = {[round(x, 6) for x in (A @ c)]} "
          f"(should equal -w = {[round(-x, 6) for x in w]})")

    print("\n6. Meridian arc lengths (1 degree of latitude, WGS84):")
    for lat in (0, 45, 80):
        print(f"   lat {lat:3d} deg: {meridian_arc_degree(lat):,.1f} m")
