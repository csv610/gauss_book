"""Gauss's composition of binary quadratic forms and class groups."""

import math
from typing import List, Tuple, Dict, Set
from collections import defaultdict

__all__ = [
    'BinaryQuadraticForm',
    'reduce_form', 'all_reduced_forms', 'class_number',
    'gauss_composition', 'compose_forms',
    'ClassGroup',
    'gauss_class_numbers', 'class_number_one_discriminants',
    'verify_gauss_class_numbers', 'dirichlet_composition',
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
    
    def __repr__(self):
        return f"[{self.a}, {self.b}, {self.c}]"
    
    def __eq__(self, other):
        return self.a == other.a and self.b == other.b and self.c == other.c
    
    def __hash__(self):
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
# Reduction Algorithm (Gauss, Disquisitiones Art. 171-198)
# ======================================================================

def reduce_form(f: BinaryQuadraticForm) -> BinaryQuadraticForm:
    """Reduce a positive definite form to its unique reduced equivalent."""
    a, b, c = f.a, f.b, f.c
    
    while True:
        # Step 1: Make |b| <= a
        if abs(b) > a:
            # Apply transformation (x,y) -> (x + ky, y) to reduce b
            k = -round(b / (2*a))
            b = b + 2*a*k
            c = a*k*k + b*k + c
            continue
        
        # Step 2: Make a <= c
        if a > c:
            # Swap a and c, change sign of b
            a, c = c, a
            b = -b
            continue
        
        # Step 3: If a == c, make b >= 0
        if a == c and b < 0:
            b = -b
            continue
        
        # Step 4: If |b| == a, make b >= 0
        if abs(b) == a and b < 0:
            b = -b
            continue
        
        break
    
    return BinaryQuadraticForm(a, b, c)


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
                if a <= c and (abs(b) < a or (abs(b) == a and b >= 0) or (a == c and b >= 0)):
                    if math.gcd(math.gcd(a, b), c) == 1:
                        forms.append(BinaryQuadraticForm(a, b, c))
    
    return forms


def class_number(D: int) -> int:
    """Class number h(D) = number of reduced primitive forms of discriminant D."""
    return len(all_reduced_forms(D))


# ======================================================================
# GAUSS'S COMPOSITION (Disquisitiones Art. 234-298)
# ======================================================================

def gauss_composition(f: BinaryQuadraticForm, g: BinaryQuadraticForm) -> BinaryQuadraticForm:
    """
    Gauss's composition of two binary quadratic forms of the same discriminant.
    
    This implements Dirichlet's simplified version (1851) of Gauss's original
    algorithm from Art. 234-298 of Disquisitiones.
    """
    if f.disc != g.disc:
        raise ValueError("Forms must have same discriminant")
    
    D = f.disc
    a1, b1, c1 = f.a, f.b, f.c
    a2, b2, c2 = g.a, g.b, g.c
    
    # Find B such that B ≡ b1 (mod 2a1), B ≡ b2 (mod 2a2), B^2 ≡ D (mod 4a1a2)
    # Using Chinese Remainder Theorem
    
    # Solve for B modulo 2*lcm(a1, a2)
    # B = b1 + 2a1 * k = b2 + 2a2 * l
    # => 2a1*k - 2a2*l = b2 - b1
    
    # Use extended Euclidean algorithm
    def egcd(a, b):
        if b == 0:
            return (a, 1, 0)
        g, x1, y1 = egcd(b, a % b)
        return (g, y1, x1 - (a // b) * y1)
    
    # Find solution to 2a1*k ≡ b2 - b1 (mod 2a2)
    A = 2 * a1
    B = 2 * a2
    C = b2 - b1
    
    g, k0, l0 = egcd(A, B)
    if C % g != 0:
        raise ValueError("Forms not composable: no solution to B congruence")
    
    k = k0 * (C // g)
    l = l0 * (C // g)
    
    B = b1 + A * k
    
    # Normalize B modulo 2*a1*a2
    modulus = 2 * a1 * a2
    B = B % modulus
    if B < 0:
        B += modulus
    
    # Now A = a1*a2 / gcd(a1,a2)^2 ? 
    # Actually: composed form has A = (a1*a2) / g^2 where g = gcd(a1, a2, (b1+b2)/2)
    
    # Standard Gauss composition:
    # A = a1 * a2 / d^2
    # where d = gcd(a1, a2, (b1+b2)/2)
    
    d = math.gcd(math.gcd(a1, a2), (b1 + b2) // 2)
    A = a1 * a2 // (d * d)
    
    # Find B such that B ≡ b1 (mod 2a1), B ≡ b2 (mod 2a2), B^2 ≡ D (mod 4A)
    # The B we found above needs adjustment
    
    # Use the formula: B = (b1 * a2 * s2 + b2 * a1 * s1) / d  (mod 2A)
    # where s1, s2 are Bezout coefficients: a1*s1 + a2*s2 = d
    _, s1, s2 = egcd(a1, a2)
    
    B = (b1 * a2 * s2 + b2 * a1 * s1) // d if (b1 * a2 * s2 + b2 * a1 * s1) >= 0 else -((-b1 * a2 * s2 - b2 * a1 * s1) // d)
    B = B % (2 * A)
    
    # C = (B^2 - D) / (4A)
    C = (B * B - D) // (4 * A)
    
    # Result might not be reduced
    result = BinaryQuadraticForm(A, B, C)
    return reduce_form(result)


def compose_forms(f: BinaryQuadraticForm, g: BinaryQuadraticForm) -> BinaryQuadraticForm:
    """Alias for gauss_composition."""
    return gauss_composition(f, g)


# ======================================================================
# Class Group Operations
# ======================================================================

class ClassGroup:
    """Class group of binary quadratic forms of discriminant D."""
    
    def __init__(self, D: int):
        self.D = D
        self.forms = all_reduced_forms(D)
        self.identity = BinaryQuadraticForm(1, 0, -D//4) if D % 4 == 0 else BinaryQuadraticForm(1, 1, (1-D)//4)
        # Ensure identity is reduced
        self.identity = reduce_form(self.identity)
        
        # Build multiplication table
        self.form_to_idx = {f: i for i, f in enumerate(self.forms)}
        self.idx_to_form = self.forms
        self.n = len(self.forms)
        
        self.table = [[0]*self.n for _ in range(self.n)]
        for i, f in enumerate(self.forms):
            for j, g in enumerate(self.forms):
                comp = gauss_composition(f, g)
                self.table[i][j] = self.form_to_idx[comp]
    
    def compose(self, f: BinaryQuadraticForm, g: BinaryQuadraticForm) -> BinaryQuadraticForm:
        i, j = self.form_to_idx[f], self.form_to_idx[g]
        return self.idx_to_form[self.table[i][j]]
    
    def inverse(self, f: BinaryQuadraticForm) -> BinaryQuadraticForm:
        """Inverse in class group: [a,b,c]^{-1} = [a,-b,c]."""
        return reduce_form(BinaryQuadraticForm(f.a, -f.b, f.c))
    
    def order(self, f: BinaryQuadraticForm) -> int:
        """Order of form in class group."""
        current = f
        for k in range(1, self.n + 1):
            if current == self.identity:
                return k
            current = self.compose(current, f)
        return self.n
    
    def is_cyclic(self) -> bool:
        """Check if class group is cyclic."""
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


# ======================================================================
# Class Number Computations (Gauss's Tables)
# ======================================================================

def gauss_class_numbers(limit: int = 1000) -> Dict[int, int]:
    """Compute class numbers for discriminants -limit < D < 0."""
    result = {}
    for D in range(-limit, 0):
        if D % 4 in (0, 1):
            h = class_number(D)
            if h > 0:
                result[D] = h
    return result


def class_number_one_discriminants() -> List[int]:
    """All negative discriminants with class number 1.
    
    Gauss conjectured these are the only ones (proved by Baker-Heegner-Stark).
    """
    return [-3, -4, -7, -8, -11, -19, -43, -67, -163]


def verify_gauss_class_numbers() -> Dict[int, Tuple[int, bool]]:
    """Verify Gauss's computed class numbers against modern computation."""
    # Gauss's table (Disquisitiones Art. 303)
    gauss_table = {
        -4: 1, -8: 1, -12: 1, -16: 1, -20: 2, -24: 2, -28: 1, -32: 2,
        -36: 2, -40: 2, -44: 3, -48: 2, -52: 2, -56: 4, -60: 2, -64: 2,
        -68: 4, -72: 2, -76: 3, -80: 4, -84: 4, -88: 2, -92: 3, -96: 4,
        -100: 2, -104: 6, -108: 3, -112: 2, -116: 6, -120: 4, -124: 5,
        -128: 4, -132: 4, -136: 4, -140: 6, -144: 4, -148: 3, -152: 6,
        -156: 4, -160: 6, -164: 8, -168: 4, -172: 7, -176: 6, -180: 4,
        -184: 10, -188: 5, -192: 4, -196: 4, -200: 6, -204: 8, -208: 6,
        -212: 6, -216: 6, -220: 10, -224: 6, -228: 4, -232: 6, -236: 11,
        -240: 8, -244: 8, -248: 6, -252: 8, -256: 4, -260: 6, -264: 8,
        -268: 12, -272: 8, -276: 8, -280: 8, -284: 8, -288: 4, -292: 6,
        -296: 6, -300: 4, -304: 12, -308: 8, -312: 8, -316: 9, -320: 8,
        -324: 6, -328: 6, -332: 14, -336: 8, -340: 8, -344: 10, -348: 12,
        -352: 8, -356: 11, -360: 8, -364: 12, -368: 12, -372: 8, -376: 10,
        -380: 10, -384: 8, -388: 12, -392: 8, -396: 8, -400: 4,
    }
    
    results = {}
    for D, gauss_h in gauss_table.items():
        computed_h = class_number(D)
        results[D] = (gauss_h, gauss_h == computed_h)
    
    return results


# ======================================================================
# Dirichlet Composition (Alternative to Gauss)
# ======================================================================

def dirichlet_composition(f: BinaryQuadraticForm, g: BinaryQuadraticForm) -> BinaryQuadraticForm:
    """Dirichlet's composition algorithm (simpler than Gauss's)."""
    a1, b1, c1 = f.a, f.b, f.c
    a2, b2, c2 = g.a, g.b, g.c
    D = f.disc
    
    # Find B such that B ≡ b1 (mod 2a1), B ≡ b2 (mod 2a2), B^2 ≡ D (mod 4a1a2)
    # Using CRT on the two congruences
    
    # First solve B ≡ b1 (mod 2a1), B ≡ b2 (mod 2a2)
    def crt(r1, m1, r2, m2):
        """Chinese remainder for two congruences."""
        g, s, t = egcd(m1, m2)
        if (r2 - r1) % g != 0:
            return None
        lcm = m1 // g * m2
        x = (r1 * t * m2 + r2 * s * m1) // g
        return x % lcm, lcm
    
    def egcd(a, b):
        if b == 0:
            return (a, 1, 0)
        g, x1, y1 = egcd(b, a % b)
        return (g, y1, x1 - (a // b) * y1)
    
    B, M = crt(b1, 2*a1, b2, 2*a2)
    if B is None:
        raise ValueError("Forms not composable")
    
    # Adjust B to satisfy B^2 ≡ D (mod 4a1a2)
    while (B*B - D) % (4*a1*a2) != 0:
        B += M
        if B >= 4*a1*a2:
            raise ValueError("No suitable B found")
    
    # A = a1*a2 / g^2 where g = gcd(a1, a2, (b1+b2)/2)
    g_val = math.gcd(math.gcd(a1, a2), (b1 + b2) // 2)
    A = a1 * a2 // (g_val * g_val)
    
    # C = (B^2 - D) / (4A)
    C = (B*B - D) // (4*A)
    
    return reduce_form(BinaryQuadraticForm(A, B, C))


# ======================================================================
# Demo
# ======================================================================

if __name__ == "__main__":
    print("=== Binary Quadratic Forms & Class Groups ===\n")
    
    # Reduced forms
    print("1. Reduced forms of discriminant -20:")
    for f in all_reduced_forms(-20):
        print(f"   {f} (primitive={f.is_primitive()}, reduced={f.is_reduced()})")
    
    print("\n2. Reduced forms of discriminant -23:")
    for f in all_reduced_forms(-23):
        print(f"   {f}")
    
    # Class group
    print("\n3. Class group of discriminant -20:")
    C = ClassGroup(-20)
    print(f"   Class number: {C.n}")
    print(f"   Forms: {C.forms}")
    print(f"   Cyclic: {C.is_cyclic()}")
    
    # Composition
    print("\n4. Gauss composition:")
    f1 = BinaryQuadraticForm(2, 2, 3)  # 2x^2 + 2xy + 3y^2
    f2 = BinaryQuadraticForm(2, -2, 3)  # 2x^2 - 2xy + 3y^2
    comp = gauss_composition(f1, f2)
    print(f"   {f1} * {f2} = {comp}")
    
    # Class number verification
    print("\n5. Class numbers (Gauss's table verification):")
    results = verify_gauss_class_numbers()
    errors = sum(1 for v in results.values() if not v[1])
    print(f"   Checked {len(results)} discriminants, {errors} discrepancies")
    
    # Class number 1 discriminants
    print("\n6. Class number 1 discriminants:")
    print(f"   {class_number_one_discriminants()}")
    
    # Group structure
    print("\n7. Class group structures:")
    for D in [-20, -23, -31, -47, -59, -83, -103, -107, -127]:
        C = ClassGroup(D)
        print(f"   D={D}: h({D}) = {C.n}, cyclic={C.is_cyclic()}")