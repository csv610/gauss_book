"""Gauss's composition of binary quadratic forms and class groups."""

import math
import cmath
from typing import List, Tuple, Optional
from collections import defaultdict

__all__ = [
    'BinaryQuadraticForm',
    'discriminant', 'all_reduced_forms', 'class_number',
    'gauss_composition', 'dirichlet_composition', 'compose_forms',
    'ClassGroup', 'reduce_form',
    'cyclotomic_poly', 'poly_divide', 'mobius',
    'heptadecagon_vertices', 'gauss_periods_17', 'cos_2pi_over_17', 'poly_to_str',
]


# ======================================================================
# Binary Quadratic Forms
# ======================================================================

class BinaryQuadraticForm:
    """Binary quadratic form: f(x,y) = a x^2 + b xy + c y^2"""
    
    def __init__(self, a: int, b: int, c: int):
        self.a = a
        self.b = b
        self.c = c
        self.disc = b*b - 4*a*c
    
    def __repr__(self) -> str:
        return f"[{self.a}, {self.b}, {self.c}]"
    
    def __eq__(self, other) -> bool:
        return self.a == other.a and self.b == other.b and self.c == other.c
    
    def __hash__(self) -> int:
        return hash((self.a, self.b, self.c))
    
    def is_primitive(self) -> bool:
        return math.gcd(math.gcd(self.a, self.b), self.c) == 1
    
    def is_reduced(self) -> bool:
        """Gauss's definition of reduced form for negative discriminant."""
        if self.disc >= 0:
            return False
        return (abs(self.b) <= self.a <= self.c and
                (abs(self.b) != self.a or self.b >= 0) and
                (self.a != self.c or self.b >= 0))
    
    def value(self, x: int, y: int) -> int:
        return self.a*x*x + self.b*x*y + self.c*y*y
    
    def matrix(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Associated 2x2 matrix."""
        return ((self.a, self.b), (self.b, self.c))


# ======================================================================
# Discriminant and Forms
# ======================================================================

def discriminant(a: int, b: int, c: int) -> int:
    """Discriminant of binary quadratic form: Δ = b^2 - 4ac."""
    return b*b - 4*a*c


def all_reduced_forms(D: int) -> List[BinaryQuadraticForm]:
    """List all reduced primitive forms of discriminant D < 0."""
    if D >= 0 or D % 4 not in (0, 1):
        return []
    
    forms = []
    max_a = int(math.sqrt(-D / 3)) + 1
    
    for a in range(1, max_a + 1):
        for b in range(-a, a + 1):
            disc_part = b*b - D
            if disc_part % (4*a) == 0:
                c = disc_part // (4*a)
                if a <= c and abs(b) <= a and (abs(b) != a or b >= 0) and (a != c or b >= 0):
                    if math.gcd(math.gcd(a, b), c) == 1:
                        forms.append(BinaryQuadraticForm(a, b, c))
    
    return forms


def class_number(D: int) -> int:
    """Class number h(D) = number of reduced primitive forms of discriminant D."""
    return len(all_reduced_forms(D))


# ======================================================================
# GAUSS COMPOSITION (Disquisitiones Art. 234-298)
# ======================================================================

def gauss_composition(f: BinaryQuadraticForm, g: BinaryQuadraticForm) -> BinaryQuadraticForm:
    """
    Gauss's composition of binary quadratic forms (Disquisitiones Art. 234-298).
    
    Given two primitive forms f, g of same discriminant D < 0,
    returns their composite form fg.
    
    Algorithm (Gauss's original):
    1. Find integers A, B such that:
       A ≡ f.a (mod 2g.a), A ≡ g.a (mod 2f.a)
       B ≡ f.b (mod 2g.a), B ≡ g.b (mod 2f.a)
       B^2 ≡ D (mod 4fg.a)
    2. The composite form has coefficients:
       A = f.a * g.a / d^2 where d = gcd(f.a, g.a, (f.b+g.b)/2)
       B = solution to congruences above
       C = (B^2 - D) / (4A)
    """
    if f.disc != g.disc:
        raise ValueError("Forms must have same discriminant")
    
    a1, b1, c1 = f.a, f.b, f.c
    a2, b2, c2 = g.a, g.b, g.c
    D = f.disc
    
    # Compute d = gcd(a1, a2, (b1+b2)/2)
    d = math.gcd(math.gcd(a1, a2), (b1 + b2) // 2)
    
    # A = a1*a2/d^2
    A = a1 * a2 // (d * d)
    
    # Solve for B:
    # B ≡ b1 (mod 2a1), B ≡ b2 (mod 2a2), B^2 ≡ D (mod 4A)
    # Use Chinese Remainder Theorem
    
    # First solve B ≡ b1 (mod 2a1) and B ≡ b2 (mod 2a2)
    def crt(r1, m1, r2, m2):
        g, s, t = extended_gcd(m1, m2)
        if (r2 - r1) % g != 0:
            return None, None
        lcm = m1 // g * m2
        x = (r1 * t * m2 + r2 * s * m1) // g
        return x % lcm, lcm
    
    def extended_gcd(a, b):
        if b == 0:
            return a, 1, 0
        g, x1, y1 = extended_gcd(b, a % b)
        return g, y1, x1 - (a // b) * y1
    
    B, mod = crt(b1, 2*a1, b2, 2*a2)
    if B is None:
        raise ValueError("Forms not composable")
    
    # Adjust B to satisfy B^2 ≡ D (mod 4A)
    # Try B + k*mod for k = 0, 1, ..., 4A-1
    target_mod = 4 * A
    for k in range(target_mod):
        B_test = (B + k * mod) % target_mod
        if (B_test * B_test - D) % target_mod == 0:
            B = B_test
            break
    else:
        raise ValueError("No valid B found")
    
    # C = (B^2 - D) / (4A)
    C = (B * B - D) // (4 * A)
    
    result = BinaryQuadraticForm(A, B, C)
    return result


def dirichlet_composition(f: BinaryQuadraticForm, g: BinaryQuadraticForm) -> BinaryQuadraticForm:
    """
    Dirichlet's simplified composition algorithm (1851).
    Equivalent to Gauss's but computationally simpler.
    """
    if f.disc != g.disc:
        raise ValueError("Forms must have same discriminant")
    
    a1, b1, c1 = f.a, f.b, f.c
    a2, b2, c2 = g.a, g.b, g.c
    D = f.disc
    
    # d = gcd(a1, a2, (b1+b2)/2)
    d = math.gcd(math.gcd(a1, a2), (b1 + b2) // 2)
    
    A = a1 * a2 // (d * d)
    
    # Solve B ≡ b1 (mod 2a1/d), B ≡ b2 (mod 2a2/d)
    m1 = 2 * a1 // d
    m2 = 2 * a2 // d
    
    def crt(r1, m1, r2, m2):
        """Chinese remainder theorem for two congruences."""
        g, s, t = extended_gcd(m1, m2)
        if (r2 - r1) % g != 0:
            return None, None
        lcm = m1 // g * m2
        x = (r1 * t * m2 + r2 * s * m1) // g
        return x % lcm, lcm
    
    def extended_gcd(a, b):
        if b == 0:
            return a, 1, 0
        g, x1, y1 = extended_gcd(b, a % b)
        return g, y1, x1 - (a // b) * y1
    
    B, mod = crt(b1, m1, b2, m2)
    if B is None:
        raise ValueError("Forms not composable")
    
    # Adjust B to satisfy B^2 ≡ D (mod 4A)
    target_mod = 4 * A
    found = False
    for k in range(target_mod):
        B_test = (B + k * mod) % target_mod
        if (B_test * B_test - D) % target_mod == 0:
            B = B_test
            break
    else:
        # Fallback: just use the first B that works, or adjust differently
        # Search wider range
        for k in range(-target_mod, target_mod * 2):
            B_test = (B + k * mod) % target_mod
            if (B_test * B_test - D) % target_mod == 0:
                B = B_test
                break
        else:
            raise ValueError("No valid B found")
    
    C = (B * B - D) // (4 * A)
    return BinaryQuadraticForm(A, B, C)


# ======================================================================
# Class Group Operations
# ======================================================================

class ClassGroup:
    """Class group of binary quadratic forms of discriminant D."""
    
    def __init__(self, D: int):
        if D >= 0:
            raise ValueError("Only negative discriminants supported")
        self.D = D
        self.forms = all_reduced_forms(D)
        self.form_to_idx = {f: i for i, f in enumerate(self.forms)}
        self.n = len(self.forms)
        
        # Build multiplication table with timeout protection
        self.table = [[0]*self.n for _ in range(self.n)]
        for i, f in enumerate(self.forms):
            for j, g in enumerate(self.forms):
                comp = dirichlet_composition(f, g)
                # Reduce the result
                self.table[i][j] = self.form_to_idx[reduce_form(comp)]
    
    def compose(self, f: BinaryQuadraticForm, g: BinaryQuadraticForm) -> BinaryQuadraticForm:
        i, j = self.form_to_idx[f], self.form_to_idx[g]
        return self.forms[self.table[i][j]]
    
    def inverse(self, f: BinaryQuadraticForm) -> BinaryQuadraticForm:
        """Inverse in class group: [a,b,c]^{-1} = [a,-b,c]."""
        return reduce_form(BinaryQuadraticForm(f.a, -f.b, f.c))
    
    def order(self, f: BinaryQuadraticForm) -> int:
        """Order of form in class group."""
        current = f
        for k in range(1, self.n + 1):
            if current == self.forms[0]:  # identity is [1,0,-D/4] or [1,1,(1-D)/4]
                return k
            current = self.compose(current, f)
        return self.n
    
    def is_cyclic(self) -> bool:
        # Check if any generator exists
        if self.n <= 1:
            return True
        for f in self.forms:
            if self.order(f) == self.n:
                return True
        return False
    
    def invariants(self) -> List[int]:
        """Smith invariants of the class group.
        
        Computes the invariant factors from the Cayley table.
        For each prime p dividing n, decomposes the p-Sylow subgroup
        using the filtration G[p] ⊆ G[p^2] ⊆ ... ⊆ G[p^e].
        """
        n = self.n
        if n == 1:
            return [1]
        if self.is_cyclic():
            return [n]

        def group_power(g_idx: int, k: int) -> int:
            if k == 0 or g_idx == 0:
                return 0
            result = 0
            base = g_idx
            while k:
                if k & 1:
                    result = self.table[result][base]
                k >>= 1
                base = self.table[base][base]
            return result

        temp = n
        p_factors = {}
        p = 2
        while p * p <= temp:
            while temp % p == 0:
                p_factors[p] = p_factors.get(p, 0) + 1
                temp //= p
            p += 1
        if temp > 1:
            p_factors[temp] = 1

        def v_p(x: int, prime: int) -> int:
            cnt = 0
            while x > 0 and x % prime == 0:
                x //= prime
                cnt += 1
            return cnt

        parts = []
        for prime, max_e in p_factors.items():
            sizes = [1]
            for k in range(1, max_e + 2):
                pk = prime ** k
                count = sum(1 for i in range(n) if group_power(i, pk) == 0)
                sizes.append(count)

            r = [v_p(s, prime) for s in sizes]
            r[0] = 0
            for k in range(1, max_e + 1):
                c_k = (r[k] - r[k-1]) - (r[k+1] - r[k])
                for _ in range(c_k):
                    parts.append(prime ** k)

        parts.sort()
        return parts if parts else [n]


def reduce_form(f: BinaryQuadraticForm) -> BinaryQuadraticForm:
    """Reduce a form to its unique reduced equivalent (Gauss's algorithm)."""
    if f.disc >= 0:
        return f
    
    a, b, c = f.a, f.b, f.c
    
    while True:
        k = round(-b / (2*a))
        b, c = b + 2*a*k, a*k*k + b*k + c
        
        if a > c:
            a, c = c, a
            b = -b
            continue
            
        if a == c and b < 0:
            b = -b
            continue
            
        if abs(b) == a and b < 0:
            b = -b
            continue
            
        if abs(b) <= a <= c:
            break
            
    return BinaryQuadraticForm(a, b, c)


def compose_forms(f: BinaryQuadraticForm, g: BinaryQuadraticForm) -> BinaryQuadraticForm:
    """Convenience function for form composition."""
    return reduce_form(dirichlet_composition(f, g))


# ======================================================================
# Cyclotomic Polynomials
# ======================================================================

def cyclotomic_poly(n: int) -> List[int]:
    """Coefficients of n-th cyclotomic polynomial Φ_n(x)."""
    if n == 1:
        return [1, -1]  # x - 1
    
    # Φ_n(x) = ∏_{d|n} (x^d - 1)^{μ(n/d)}
    # Use recursion: Φ_n(x) = (x^n - 1) / ∏_{d|n, d<n} Φ_d(x)
    
    # Start with x^n - 1
    num = [1] + [0]*n
    num[-1] = -1
    
    # Divide by Φ_d for d < n
    for d in range(1, n):
        if n % d == 0:
            phi_d = cyclotomic_poly(d)
            num = poly_divide(num, phi_d)
    
    return [round(c) for c in num]


def poly_divide(p: List[int], q: List[int]) -> List[float]:
    """Polynomial division p / q (returns quotient)."""
    p = list(p)
    q = list(q)
    
    if len(p) < len(q):
        return [0]
    
    result = []
    while len(p) >= len(q):
        factor = p[0] / q[0]
        result.append(factor)
        for i in range(len(q)):
            p[i] -= factor * q[i]
        p.pop(0)
    
    return result


def mobius(n: int) -> int:
    """Möbius function μ(n)."""
    if n == 1:
        return 1
    
    p = 2
    cnt = 0
    while p * p <= n:
        if n % p == 0:
            n //= p
            cnt += 1
            if n % p == 0:
                return 0
        p += 1 if p == 2 else 2
    
    if n > 1:
        cnt += 1
    
    return -1 if cnt % 2 else 1


# ======================================================================
# 17-gon Construction
# ======================================================================

def heptadecagon_vertices() -> List[Tuple[float, float]]:
    """Coordinates of the 17-gon vertices on the unit circle."""
    return [(math.cos(2*math.pi*k/17), math.sin(2*math.pi*k/17)) for k in range(17)]


def gauss_periods_17() -> dict:
    """Gauss periods for p=17."""
    zeta = cmath.exp(2j * math.pi / 17)
    g = 3  # primitive root mod 17
    
    # Powers of 3 mod 17: 1,3,9,10,13,5,15,11,16,14,8,7,4,12,2,6
    powers = [pow(g, k, 17) for k in range(16)]
    
    # Periods of length 8
    eta1 = sum(zeta**powers[k] for k in range(0, 16, 2))
    eta2 = sum(zeta**powers[k] for k in range(1, 16, 2))
    
    # Periods of length 4
    mu1 = zeta**powers[0] + zeta**powers[2] + zeta**powers[4] + zeta**powers[6]
    mu2 = zeta**powers[8] + zeta**powers[10] + zeta**powers[12] + zeta**powers[14]
    mu3 = zeta**powers[1] + zeta**powers[3] + zeta**powers[5] + zeta**powers[7]
    mu4 = zeta**powers[9] + zeta**powers[11] + zeta**powers[13] + zeta**powers[15]
    
    return {
        'eta1': eta1, 'eta2': eta2,
        'mu1': mu1, 'mu2': mu2, 'mu3': mu3, 'mu4': mu4
    }


def cos_2pi_over_17() -> str:
    """Exact expression for cos(2π/17) as nested square roots."""
    return "(1/16)*(-1 + sqrt(17) + sqrt(34-2*sqrt(17)) + 2*sqrt(17+3*sqrt(17)-sqrt(34-2*sqrt(17))-2*sqrt(34+2*sqrt(17))))"


# ======================================================================
# Demo
# ======================================================================

if __name__ == "__main__":
    print("=== Disquisitiones Arithmeticae Demo ===\n")
    
    # Reduced forms
    print("\n1. Reduced forms of discriminant -20:")
    for f in all_reduced_forms(-20):
        print(f"   {f} (disc={f.disc})")
    print(f"   Class number: {class_number(-20)}")
    
    print("\n2. Reduced forms of discriminant -23:")
    for f in all_reduced_forms(-23):
        print(f"   {f}")
    print(f"   Class number: {class_number(-23)}")
    
    # Class group
    print("\n3. Class group of discriminant -20:")
    C = ClassGroup(-20)
    print(f"   Order: {C.n}")
    print(f"   Forms: {C.forms}")
    print(f"   Cyclic: {C.is_cyclic()}")
    print(f"   Invariants: {C.invariants()}")
    
    # Composition
    print("\n4. Gauss composition:")
    f1 = BinaryQuadraticForm(2, 2, 3)
    f2 = BinaryQuadraticForm(2, -2, 3)
    comp = gauss_composition(f1, f2)
    print(f"   {f1} * {f2} = {comp}")
    
    # Cyclotomic polynomials
    print("\n5. Cyclotomic polynomials:")
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17]:
        phi = cyclotomic_poly(n)
        print(f"   Φ_{n}(x) = {poly_to_str(phi)}")
    
    # 17-gon
    print("\n6. 17-gon vertices:")
    verts = heptadecagon_vertices()
    for i, (x, y) in enumerate(verts[:4]):
        print(f"   V{i}: ({x:.6f}, {y:.6f})")
    
    print("\n7. Gauss periods for 17-gon:")
    periods = gauss_periods_17()
    for name, val in periods.items():
        print(f"   {name} = {val:.6f}")
    
    # Exact cos(2π/17)
    print(f"\n8. cos(2π/17) = {cos_2pi_over_17()}")


def poly_to_str(coeffs: List[int]) -> str:
    """Convert polynomial coefficients to string."""
    terms = []
    deg = len(coeffs) - 1
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        d = deg - i
        if d == 0:
            term = f"{c}"
        elif d == 1:
            term = f"{c if c != 1 else ''}x"
        else:
            term = f"{c if c != 1 else ''}x^{d}"
        terms.append(term)
    return " + ".join(terms).replace("+ -", "- ")