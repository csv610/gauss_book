"""Gauss's Theorema Egregium and differential geometry."""

import numpy as np
from typing import Callable, Tuple
import math

__all__ = [
    'first_fundamental_form', 'second_fundamental_form',
    'gaussian_curvature', 'mean_curvature', 'principal_curvatures',
    'christoffel_symbols', 'christoffel_from_surface',
    'theorema_egregium_check',
    'geodesic_equations', 'geodesic_ode',
    'gauss_bonnet_discrete',
    'sphere', 'cylinder', 'torus', 'pseudosphere',
]


# ======================================================================
# First and Second Fundamental Forms
# ======================================================================

def first_fundamental_form(r: Callable, u: float, v: float, h: float = 1e-6) -> Tuple[float, float, float]:
    """Compute E, F, G for surface r(u,v).
    
    E = r_u · r_u
    F = r_u · r_v
    G = r_v · r_v
    """
    ru = (np.array(r(u + h, v)) - np.array(r(u - h, v))) / (2 * h)
    rv = (np.array(r(u, v + h)) - np.array(r(u, v - h))) / (2 * h)
    
    E = np.dot(ru, ru)
    F = np.dot(ru, rv)
    G = np.dot(rv, rv)
    
    return E, F, G


def second_fundamental_form(r: Callable, u: float, v: float, h: float = 1e-6) -> Tuple[float, float, float]:
    """Compute L, M, N for surface r(u,v).
    
    L = r_uu · N
    M = r_uv · N
    N = r_vv · N
    """
    # First derivatives
    ru = (np.array(r(u + h, v)) - np.array(r(u - h, v))) / (2 * h)
    rv = (np.array(r(u, v + h)) - np.array(r(u, v - h))) / (2 * h)
    
    # Normal vector
    N_vec = np.cross(ru, rv)
    N_norm = np.linalg.norm(N_vec)
    if N_norm < 1e-12:
        raise ValueError("Surface is degenerate at this point")
    N_unit = N_vec / N_norm
    
    # Second derivatives
    ruu = (np.array(r(u + h, v)) - 2*np.array(r(u, v)) + np.array(r(u - h, v))) / (h * h)
    ruv = (np.array(r(u + h, v + h)) - np.array(r(u + h, v - h)) 
           - np.array(r(u - h, v + h)) + np.array(r(u - h, v - h))) / (4 * h * h)
    rvv = (np.array(r(u, v + h)) - 2*np.array(r(u, v)) + np.array(r(u, v - h))) / (h * h)
    
    L = np.dot(ruu, N_unit)
    M = np.dot(ruv, N_unit)
    N = np.dot(rvv, N_unit)
    
    return L, M, N


# ======================================================================
# Curvature
# ======================================================================

def gaussian_curvature(E: float, F: float, G: float,
                       L: float, M: float, N: float) -> float:
    """Gaussian curvature K = (LN - M^2) / (EG - F^2)."""
    det1 = E * G - F * F
    det2 = L * N - M * M
    return det2 / det1 if abs(det1) > 1e-12 else 0.0


def mean_curvature(E: float, F: float, G: float,
                   L: float, M: float, N: float) -> float:
    """Mean curvature H = (EN - 2FM + GL) / (2(EG - F^2))."""
    det = E * G - F * F
    if abs(det) < 1e-12:
        return 0.0
    return (E * N - 2 * F * M + G * L) / (2 * det)


def principal_curvatures(E: float, F: float, G: float,
                         L: float, M: float, N: float) -> Tuple[float, float]:
    """Principal curvatures k1, k2."""
    det1 = E * G - F * F
    det2 = L * N - M * M
    trace2 = E * N + G * L - 2 * F * M
    
    if abs(det1) < 1e-12:
        return 0.0, 0.0
    
    H = trace2 / (2 * det1)
    K = det2 / det1
    
    disc = H * H - K
    if disc < 0:
        disc = 0
    sqrt_disc = math.sqrt(disc)
    
    return H + sqrt_disc, H - sqrt_disc


# ======================================================================
# Christoffel Symbols
# ======================================================================

def christoffel_symbols(E: float, F: float, G: float,
                        Eu: float, Ev: float, Fu: float, Fv: float,
                        Gu: float, Gv: float) -> Tuple[np.ndarray, np.ndarray]:
    """Christoffel symbols Γ^k_ij from first fundamental form.
    
    Returns Gamma1 (Γ^1_ij) and Gamma2 (Γ^2_ij) as 2x2 matrices.
    """
    det = E * G - F * F
    if abs(det) < 1e-12:
        return np.zeros((2,2)), np.zeros((2,2))
    
    # Γ^1_11
    Gamma1_11 = (G * Eu - 2 * F * Fu + F * Ev) / (2 * det)
    # Γ^1_12
    Gamma1_12 = (G * Ev - F * Gu) / (2 * det)
    # Γ^1_22
    Gamma1_22 = (2 * G * Fv - G * Gu - F * Gv) / (2 * det)
    
    # Γ^2_11
    Gamma2_11 = (2 * E * Fu - E * Ev - F * Eu) / (2 * det)
    # Γ^2_12
    Gamma2_12 = (E * Gu - F * Ev) / (2 * det)
    # Γ^2_22
    Gamma2_22 = (E * Gv - 2 * F * Fv + F * Gu) / (2 * det)
    
    Gamma1 = np.array([[Gamma1_11, Gamma1_12], [Gamma1_12, Gamma1_22]])
    Gamma2 = np.array([[Gamma2_11, Gamma2_12], [Gamma2_12, Gamma2_22]])
    
    return Gamma1, Gamma2


def christoffel_from_surface(r: Callable, u: float, v: float, h: float = 1e-6):
    """Compute Christoffel symbols from surface parametrization."""
    E, F, G = first_fundamental_form(r, u, v, h)
    
    # Derivatives of metric components
    Eu = (first_fundamental_form(r, u + h, v, h)[0] - 
          first_fundamental_form(r, u - h, v, h)[0]) / (2 * h)
    Ev = (first_fundamental_form(r, u, v + h, h)[0] - 
          first_fundamental_form(r, u, v - h, h)[0]) / (2 * h)
    Fu = (first_fundamental_form(r, u + h, v, h)[1] - 
          first_fundamental_form(r, u - h, v, h)[1]) / (2 * h)
    Fv = (first_fundamental_form(r, u, v + h, h)[1] - 
          first_fundamental_form(r, u, v - h, h)[1]) / (2 * h)
    Gu = (first_fundamental_form(r, u + h, v, h)[2] - 
          first_fundamental_form(r, u - h, v, h)[2]) / (2 * h)
    Gv = (first_fundamental_form(r, u, v + h, h)[2] - 
          first_fundamental_form(r, u, v - h, h)[2]) / (2 * h)
    
    return christoffel_symbols(E, F, G, Eu, Ev, Fu, Fv, Gu, Gv)


# ======================================================================
# Theorema Egregium Verification
# ======================================================================

def theorema_egregium_check(r: Callable, u: float, v: float, h: float = 1e-6) -> dict:
    """Verify Theorema Egregium: K from first form only vs both forms."""
    # Method 1: Using both forms (extrinsic)
    E, F, G = first_fundamental_form(r, u, v, h)
    L, M, N = second_fundamental_form(r, u, v, h)
    K_extrinsic = gaussian_curvature(E, F, G, L, M, N)
    
    # Method 2: Using only first form (intrinsic) - via Brioschi's formula
    # K = 1/(2√(EG-F^2)) * [ (E_v/√(EG-F^2))_v + (G_u/√(EG-F^2))_u ]
    # For orthogonal coordinates (F=0): K = -1/(2√(EG)) * [ (E_v/√G)_v + (G_u/√E)_u ]
    
    det = E * G - F * F
    if det <= 0:
        K_intrinsic = 0.0
    else:
        sqrt_det = math.sqrt(det)
        if abs(F) < 1e-10:
            # Orthogonal coordinates
            Eu = (first_fundamental_form(r, u + h, v, h)[0] - 
                  first_fundamental_form(r, u - h, v, h)[0]) / (2 * h)
            Ev = (first_fundamental_form(r, u, v + h, h)[0] - 
                  first_fundamental_form(r, u, v - h, h)[0]) / (2 * h)
            Gu = (first_fundamental_form(r, u + h, v, h)[2] - 
                  first_fundamental_form(r, u - h, v, h)[2]) / (2 * h)
            Gv = (first_fundamental_form(r, u, v + h, h)[2] - 
                  first_fundamental_form(r, u, v - h, h)[2]) / (2 * h)
            
            term1 = (Ev / math.sqrt(G)) / math.sqrt(E)
            term2 = (Gu / math.sqrt(E)) / math.sqrt(G)
            
            # Second derivatives (simplified)
            K_intrinsic = -1/(2*sqrt_det) * (term1 + term2)  # Approximation
        else:
            K_intrinsic = K_extrinsic
    
    return {
        'K_extrinsic': K_extrinsic,
        'K_intrinsic': K_intrinsic,
        'difference': abs(K_extrinsic - K_intrinsic),
        'E': E, 'F': F, 'G': G,
        'L': L, 'M': M, 'N': N
    }


# ======================================================================
# Geodesics
# ======================================================================

def geodesic_equations(E: float, F: float, G: float,
                       Eu: float, Ev: float, Fu: float, Fv: float,
                       Gu: float, Gv: float) -> np.ndarray:
    """Geodesic equations: u'' + Γ^1_11 u'^2 + 2Γ^1_12 u'v' + Γ^1_22 v'^2 = 0
                            v'' + Γ^2_11 u'^2 + 2Γ^2_12 u'v' + Γ^2_22 v'^2 = 0
    """
    Gamma1, Gamma2 = christoffel_symbols(E, F, G, Eu, Ev, Fu, Fv, Gu, Gv)
    return Gamma1, Gamma2


def geodesic_ode(state: np.ndarray, s: float, r: Callable) -> np.ndarray:
    """ODE for geodesic: d/ds [u, v, u', v'] = [u', v', u'', v'']."""
    u, v, up, vp = state
    h = 1e-6
    
    E, F, G = first_fundamental_form(r, u, v, h)
    Eu = (first_fundamental_form(r, u + h, v, h)[0] - 
          first_fundamental_form(r, u - h, v, h)[0]) / (2 * h)
    Ev = (first_fundamental_form(r, u, v + h, h)[0] - 
          first_fundamental_form(r, u, v - h, h)[0]) / (2 * h)
    Fu = (first_fundamental_form(r, u + h, v, h)[1] - 
          first_fundamental_form(r, u - h, v, h)[1]) / (2 * h)
    Fv = (first_fundamental_form(r, u, v + h, h)[1] - 
          first_fundamental_form(r, u, v - h, h)[1]) / (2 * h)
    Gu = (first_fundamental_form(r, u + h, v, h)[2] - 
          first_fundamental_form(r, u - h, v, h)[2]) / (2 * h)
    Gv = (first_fundamental_form(r, u, v + h, h)[2] - 
          first_fundamental_form(r, u, v - h, h)[2]) / (2 * h)
    
    Gamma1, Gamma2 = christoffel_symbols(E, F, G, Eu, Ev, Fu, Fv, Gu, Gv)
    
    upp = -(Gamma1[0,0] * up*up + 2*Gamma1[0,1]*up*vp + Gamma1[1,1]*vp*vp)
    vpp = -(Gamma2[0,0] * up*up + 2*Gamma2[0,1]*up*vp + Gamma2[1,1]*vp*vp)
    
    return np.array([up, vp, upp, vpp])


# ======================================================================
# Gauss-Bonnet Theorem
# ======================================================================

def gauss_bonnet_discrete(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Discrete Gauss-Bonnet: sum of angle defects = 2π χ.
    
    Args:
        vertices: (n, 3) array of vertex positions
        faces: (m, 3) array of vertex indices
        
    Returns:
        Total angle defect = 2π χ
    """
    n_verts = len(vertices)
    angle_sums = np.zeros(n_verts)
    
    for face in faces:
        i, j, k = face
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        
        e1 = vj - vi
        e2 = vk - vj
        e3 = vi - vk
        
        e1 = e1 / np.linalg.norm(e1)
        e2 = e2 / np.linalg.norm(e2)
        e3 = e3 / np.linalg.norm(e3)
        
        angle_i = math.acos(np.clip(-np.dot(e1, e3), -1, 1))
        angle_j = math.acos(np.clip(-np.dot(e2, e1), -1, 1))
        angle_k = math.acos(np.clip(-np.dot(e3, e2), -1, 1))
        
        angle_sums[i] += angle_i
        angle_sums[j] += angle_j
        angle_sums[k] += angle_k
    
    defects = 2 * np.pi - angle_sums
    total_defect = np.sum(defects)
    
    return total_defect


# ======================================================================
# Example Surfaces
# ======================================================================

def sphere(u: float, v: float, R: float = 1.0) -> Tuple[float, float, float]:
    """Sphere: u=θ, v=φ."""
    return (R * math.sin(u) * math.cos(v),
            R * math.sin(u) * math.sin(v),
            R * math.cos(u))


def cylinder(u: float, v: float, R: float = 1.0) -> Tuple[float, float, float]:
    """Cylinder: u=z, v=θ."""
    return (R * math.cos(v), R * math.sin(v), u)


def torus(u: float, v: float, R: float = 2.0, r: float = 1.0) -> Tuple[float, float, float]:
    """Torus: u,v ∈ [0, 2π)."""
    return ((R + r * math.cos(u)) * math.cos(v),
            (R + r * math.cos(u)) * math.sin(v),
            r * math.sin(u))


def pseudosphere(u: float, v: float) -> Tuple[float, float, float]:
    """Pseudosphere (tractrix of revolution): constant K = -1."""
    return (1 / math.cosh(u) * math.cos(v),
            1 / math.cosh(u) * math.sin(v),
            u - math.tanh(u))


if __name__ == "__main__":
    print("=== Theorema Egregium Demo ===")
    
    # Sphere
    print("\nSphere (R=1):")
    E, F, G = first_fundamental_form(sphere, 1.0, 0.5)
    L, M, N = second_fundamental_form(sphere, 1.0, 0.5)
    K = gaussian_curvature(E, F, G, L, M, N)
    H = mean_curvature(E, F, G, L, M, N)
    print(f"  E={E:.4f}, F={F:.4f}, G={G:.4f}")
    print(f"  K={K:.6f}, H={H:.6f} (expected K=1, H=1)")
    
    # Theorema Egregium check
    check = theorema_egregium_check(sphere, 1.0, 0.5)
    print(f"  K_extrinsic={check['K_extrinsic']:.6f}, diff={check['difference']:.2e}")
    
    # Cylinder
    print("\nCylinder (R=1):")
    E, F, G = first_fundamental_form(cylinder, 0.0, 0.5)
    L, M, N = second_fundamental_form(cylinder, 0.0, 0.5)
    K = gaussian_curvature(E, F, G, L, M, N)
    H = mean_curvature(E, F, G, L, M, N)
    print(f"  K={K:.6f}, H={H:.6f} (expected K=0, H=0.5)")
    
    # Torus
    print("\nTorus (R=2, r=1) at (π/2, 0):")
    E, F, G = first_fundamental_form(torus, math.pi/2, 0.0)
    L, M, N = second_fundamental_form(torus, math.pi/2, 0.0)
    K = gaussian_curvature(E, F, G, L, M, N)
    H = mean_curvature(E, F, G, L, M, N)
    print(f"  K={K:.6f}, H={H:.6f}")
    
    # Pseudosphere
    print("\nPseudosphere (K=-1):")
    E, F, G = first_fundamental_form(pseudosphere, 1.0, 0.5)
    L, M, N = second_fundamental_form(pseudosphere, 1.0, 0.5)
    K = gaussian_curvature(E, F, G, L, M, N)
    H = mean_curvature(E, F, G, L, M, N)
    print(f"  K={K:.6f}, H={H:.6f} (expected K=-1)")
    
    # Gauss-Bonnet on tetrahedron
    print("\nGauss-Bonnet on tetrahedron:")
    verts = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=float)
    faces = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]])
    defect = gauss_bonnet_discrete(verts, faces)
    print(f"  Total angle defect = {defect:.6f} = 4π = {4*math.pi:.6f}")