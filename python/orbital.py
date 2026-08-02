"""Orbital mechanics: Gauss's method for orbit determination."""

import numpy as np
from typing import List, Tuple, Callable
import math

__all__ = [
    'MU_SUN', 'AU', 'DAY',
    'solve_kepler', 'mean_anomaly_from_true', 'true_anomaly_from_mean',
    'state_to_elements', 'elements_to_state', 'rotation_matrix',
    'fg_functions', 'stumpff_C', 'stumpff_S', 'universal_anomaly', 'r_mag',
    'propagate_universal',
    'gauss_method', 'differential_correction',
    'ceres_example',
]


# ======================================================================
# Constants
# ======================================================================

MU_SUN = 1.32712440018e20  # m^3/s^2 (standard gravitational parameter)
AU = 149597870700  # m
DAY = 86400  # s

# Earth orbital elements (approximate)
EARTH_A = AU
EARTH_E = 0.0167
EARTH_I = 0.0
EARTH_OMEGA = 0.0
EARTH_W = 1.796595647  # rad
EARTH_M0 = 0.0


# ======================================================================
# Kepler's equation
# ======================================================================

def solve_kepler(M: float, e: float, tol: float = 1e-12, max_iter: int = 100) -> float:
    """Solve Kepler's equation: M = E - e sin E.
    
    Returns eccentric anomaly E.
    """
    if e == 0:
        return M
    
    # Initial guess
    E = M + e * math.sin(M)
    
    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        fp = 1 - e * math.cos(E)
        dE = f / fp
        E -= dE
        if abs(dE) < tol:
            break
    
    return E


def mean_anomaly_from_true(f: float, e: float) -> float:
    """Convert true anomaly to mean anomaly."""
    # tan(E/2) = sqrt((1-e)/(1+e)) * tan(f/2)
    E = 2 * math.atan(math.sqrt((1 - e) / (1 + e)) * math.tan(f / 2))
    return E - e * math.sin(E)


def true_anomaly_from_mean(M: float, e: float) -> float:
    """Convert mean anomaly to true anomaly."""
    E = solve_kepler(M, e)
    return 2 * math.atan(math.sqrt((1 + e) / (1 - e)) * math.tan(E / 2))


# ======================================================================
# State vector <-> Orbital elements
# ======================================================================

def state_to_elements(r: np.ndarray, v: np.ndarray, mu: float = MU_SUN) -> dict:
    """Convert position/velocity to orbital elements."""
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    
    # Angular momentum
    h_vec = np.cross(r, v)
    h = np.linalg.norm(h_vec)
    
    # Energy
    energy = 0.5 * v_norm**2 - mu / r_norm
    
    # Semi-major axis
    a = -mu / (2 * energy) if energy < 0 else float('inf')
    
    # Eccentricity vector
    e_vec = (np.cross(v, h_vec) / mu) - (r / r_norm)
    e = np.linalg.norm(e_vec)
    
    # Inclination
    i = math.acos(h_vec[2] / h)
    
    # Longitude of ascending node
    if abs(h_vec[0]) < 1e-12 and abs(h_vec[1]) < 1e-12:
        Omega = 0.0
    else:
        Omega = math.atan2(h_vec[0], -h_vec[1])
        if Omega < 0:
            Omega += 2 * math.pi
    
    # Argument of periapsis
    if e < 1e-12:
        omega = 0.0
    else:
        # Node vector
        n_vec = np.array([-h_vec[1], h_vec[0], 0])
        n = np.linalg.norm(n_vec)
        if n < 1e-12:
            omega = 0.0
        else:
            cos_omega = np.dot(n_vec, e_vec) / (n * e)
            omega = math.acos(np.clip(cos_omega, -1, 1))
            if e_vec[2] < 0:
                omega = 2 * math.pi - omega
    
    # True anomaly
    if e < 1e-12:
        f = 0.0
    else:
        cos_f = np.dot(r, e_vec) / (r_norm * e)
        f = math.acos(np.clip(cos_f, -1, 1))
        if np.dot(r, v) < 0:
            f = 2 * math.pi - f
    
    # Mean anomaly
    E = 2 * math.atan(math.sqrt((1 - e) / (1 + e)) * math.tan(f / 2))
    M = E - e * math.sin(E)
    
    return {
        'a': a, 'e': e, 'i': i, 'Omega': Omega, 'omega': omega, 'M0': M,
        'f': f, 'E': E
    }


def elements_to_state(elems: dict, mu: float = MU_SUN) -> Tuple[np.ndarray, np.ndarray]:
    """Convert orbital elements to position/velocity."""
    a = elems['a']
    e = elems['e']
    i = elems['i']
    Omega = elems['Omega']
    omega = elems['omega']
    M = elems.get('M0', elems.get('M', 0))
    
    # Eccentric anomaly
    E = solve_kepler(M, e)
    
    # True anomaly
    f = 2 * math.atan(math.sqrt((1 + e) / (1 - e)) * math.tan(E / 2))
    
    # Radius
    r_mag = a * (1 - e * math.cos(E))
    
    # Perifocal frame
    r_pf = np.array([r_mag * math.cos(f), r_mag * math.sin(f), 0])
    v_pf = np.array([-math.sqrt(mu * a) / r_mag * math.sin(E),
                      math.sqrt(mu * a) / r_mag * math.sqrt(1 - e**2) * math.cos(E),
                      0])
    
    # Rotation to inertial frame
    # R = R_z(-Omega) * R_x(-i) * R_z(-omega)
    R = rotation_matrix(Omega, i, omega)
    
    r = R @ r_pf
    v = R @ v_pf
    
    return r, v


def rotation_matrix(Omega: float, i: float, omega: float) -> np.ndarray:
    """Rotation matrix from perifocal to inertial frame."""
    cos_O, sin_O = math.cos(Omega), math.sin(Omega)
    cos_i, sin_i = math.cos(i), math.sin(i)
    cos_w, sin_w = math.cos(omega), math.sin(omega)
    
    R = np.array([
        [cos_O*cos_w - sin_O*sin_w*cos_i,
         -cos_O*sin_w - sin_O*cos_w*cos_i,
         sin_O*sin_i],
        [sin_O*cos_w + cos_O*sin_w*cos_i,
         -sin_O*sin_w + cos_O*cos_w*cos_i,
         -cos_O*sin_i],
        [sin_w*sin_i, cos_w*sin_i, cos_i]
    ])
    return R


# ======================================================================
# f and g functions (universal variable formulation)
# ======================================================================

def fg_functions(r0: np.ndarray, v0: np.ndarray, dt: float, 
                 mu: float = MU_SUN) -> Tuple[float, float, float, float]:
    """Universal f and g functions for orbit propagation.
    
    Returns: f, g, fdot, gdot
    """
    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)
    rv0 = np.dot(r0, v0)
    
    alpha = 2/r0_mag - v0_mag**2/mu  # 1/a
    
    # Universal anomaly chi
    chi = universal_anomaly(r0_mag, rv0, alpha, dt, mu)
    
    # Stumpff functions
    z = alpha * chi**2
    C = stumpff_C(z)
    S = stumpff_S(z)
    
    r = r_mag(chi, r0_mag, rv0, alpha, mu)
    
    f = 1 - chi**2 / r0_mag * C
    g = dt - chi**3 / math.sqrt(mu) * S
    fdot = math.sqrt(mu) / (r0_mag * r) * chi * (z*S - 1)
    gdot = 1 - chi**2 / r * C
    
    return f, g, fdot, gdot


def stumpff_C(z: float) -> float:
    """Stumpff function C(z)."""
    if z > 1e-6:
        return (1 - math.cos(math.sqrt(z))) / z
    elif z < -1e-6:
        return (math.cosh(math.sqrt(-z)) - 1) / (-z)
    else:
        return 0.5


def stumpff_S(z: float) -> float:
    """Stumpff function S(z)."""
    if z > 1e-6:
        return (math.sqrt(z) - math.sin(math.sqrt(z))) / (z**1.5)
    elif z < -1e-6:
        return (math.sinh(math.sqrt(-z)) - math.sqrt(-z)) / ((-z)**1.5)
    else:
        return 1/6


def universal_anomaly(r0: float, rv0: float, alpha: float, dt: float, mu: float) -> float:
    """Solve for universal anomaly chi using Newton's method."""
    if alpha == 0:
        # Parabolic
        return math.sqrt(mu) * dt / r0
    
    # Initial guess
    if alpha > 0:
        # Elliptic
        chi = math.sqrt(mu) * alpha * dt
    else:
        # Hyperbolic
        chi = math.log((-2*mu*alpha*dt + rv0 + math.sqrt(-2*mu*alpha)*r0) / 
                       (rv0 + math.sqrt(-2*mu*alpha)*r0)) / math.sqrt(-alpha)
    
    # Newton iteration
    for _ in range(50):
        z = alpha * chi**2
        C = stumpff_C(z)
        S = stumpff_S(z)
        
        # Universal Kepler equation: f(chi) = r0*chi + (rv0/sqrt(mu))*chi^2*C + (1-alpha*r0)*chi^3*S - sqrt(mu)*dt
        f = r0*chi + rv0/math.sqrt(mu)*chi*chi*C + (1 - alpha*r0)*chi*chi*chi*S - math.sqrt(mu)*dt
        
        # Simplified derivative (ignoring Stumpff function derivatives)
        df = r0 + rv0/math.sqrt(mu)*chi + (1 - alpha*r0)*chi*chi
        
        dchi = f / df
        chi -= dchi
        
        if abs(dchi) < 1e-12:
            break
    
    return chi


def r_mag(chi: float, r0: float, rv0: float, alpha: float, mu: float) -> float:
    """Radius magnitude from universal anomaly using Stumpff functions."""
    z = alpha * chi**2
    C = stumpff_C(z)
    return r0 + rv0/math.sqrt(mu)*chi + (1 - alpha*r0)*chi**2*C


def propagate_universal(r0: np.ndarray, v0: np.ndarray, dt: float,
                         mu: float = MU_SUN) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate orbit using universal variables."""
    f, g, fdot, gdot = fg_functions(r0, v0, dt, mu)
    r = f * r0 + g * v0
    v = fdot * r0 + gdot * v0
    return r, v


# ======================================================================
# Gauss's method (simplified placeholder)
# ======================================================================

def gauss_method(obs1: Tuple[float, float, float],
                 obs2: Tuple[float, float, float],
                 obs3: Tuple[float, float, float],
                 mu: float = MU_SUN) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss's method for preliminary orbit determination.
    
    Args:
        obs: (RA, Dec, time) in radians, radians, seconds
        
    Returns:
        r, v: Heliocentric position and velocity at middle observation
    """
    # This is a simplified placeholder
    # Full implementation requires:
    # 1. Convert RA/Dec to unit vectors
    # 2. Compute Earth's position at each time
    # 3. Solve for topocentric distances using coplanarity
    # 4. Compute heliocentric positions
    # 5. Use f,g functions to compute velocity
    raise NotImplementedError("Full Gauss method requires spherical astronomy")


# ======================================================================
# Differential correction
# ======================================================================

def differential_correction(observations: List[Tuple[float, float, float, float]],
                            initial_elements: dict,
                            mu: float = MU_SUN,
                            max_iter: int = 10) -> dict:
    """Batch least squares orbit refinement.
    
    observations: [(t, RA, Dec, weight), ...]
    """
    raise NotImplementedError("Full differential correction requires observation residuals and partial derivatives")


# ======================================================================
# Example: Ceres orbit (simplified)
# ======================================================================

def ceres_example():
    """Simulate Gauss's Ceres orbit determination."""
    # Historical data: three observations by Piazzi (1801)
    # Simplified: use known orbital elements
    ceres_elements = {
        'a': 2.767 * AU,
        'e': 0.076,
        'i': math.radians(10.59),
        'Omega': math.radians(80.3),
        'omega': math.radians(73.6),
        'M0': math.radians(180.0)  # arbitrary
    }
    
    r, v = elements_to_state(ceres_elements)
    print(f"Ceres position (AU): {r / AU}")
    print(f"Ceres velocity (AU/day): {v * DAY / AU}")
    
    # Propagate
    r_new, v_new = propagate_universal(r, v, 30 * DAY)
    print(f"After 30 days: {r_new / AU}")


if __name__ == "__main__":
    print("=== Orbital Mechanics Demo ===")
    
    # Test Kepler equation
    print("\n1. Kepler's equation:")
    for e in [0.0, 0.1, 0.5, 0.9]:
        M = 1.5
        E = solve_kepler(M, e)
        print(f"  e={e}: M={M:.3f} -> E={E:.6f} (check: {E - e*math.sin(E):.6f})")
    
    # State to elements
    print("\n2. State to elements:")
    # Circular orbit at 1 AU
    r = np.array([AU, 0, 0])
    v = np.array([0, math.sqrt(MU_SUN/AU), 0])
    elems = state_to_elements(r, v)
    print(f"  a={elems['a']/AU:.4f} AU, e={elems['e']:.4f}, i={math.degrees(elems['i']):.4f}°")
    
    # Elements to state
    r2, v2 = elements_to_state(elems)
    print(f"  Reconstructed r: {r2/AU}")
    print(f"  Reconstructed v: {v2*DAY/AU}")
    
    # Universal propagation
    print("\n3. Universal propagation:")
    r_prop, v_prop = propagate_universal(r, v, 100*DAY)
    print(f"  After 100 days: r={r_prop/AU}")
    
    # Ceres example
    print("\n4. Ceres example:")
    ceres_example()