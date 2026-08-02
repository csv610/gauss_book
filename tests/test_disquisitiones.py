"""Tests for Disquisitiones Arithmeticae module."""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.disquisitiones import (
    discriminant, all_reduced_forms, class_number,
    BinaryQuadraticForm, ClassGroup, dirichlet_composition, gauss_composition,
    cyclotomic_poly, mobius, heptadecagon_vertices, gauss_periods_17,
    cos_2pi_over_17, poly_to_str
)


class TestBinaryQuadraticForm:
    def test_initialization(self):
        f = BinaryQuadraticForm(2, 2, 3)
        assert f.a == 2
        assert f.b == 2
        assert f.c == 3
        assert f.disc == -20
    
    def test_equality(self):
        f1 = BinaryQuadraticForm(2, 2, 3)
        f2 = BinaryQuadraticForm(2, 2, 3)
        f3 = BinaryQuadraticForm(3, 4, 5)
        assert f1 == f2
        assert f1 != f3
    
    def test_hash(self):
        f1 = BinaryQuadraticForm(2, 2, 3)
        f2 = BinaryQuadraticForm(2, 2, 3)
        assert hash(f1) == hash(f2)
    
    def test_is_primitive(self):
        f1 = BinaryQuadraticForm(2, 2, 3)  # gcd(2,2,3)=1
        f2 = BinaryQuadraticForm(2, 4, 6)  # gcd(2,4,6)=2
        assert f1.is_primitive()
        assert not f2.is_primitive()
    
    def test_is_reduced(self):
        f1 = BinaryQuadraticForm(2, 2, 3)   # reduced: |2| <= 2 <= 3
        f2 = BinaryQuadraticForm(1, 0, 5)   # reduced: |0| <= 1 <= 5
        f3 = BinaryQuadraticForm(3, 4, 5)   # not reduced: |4| > 3
        assert f1.is_reduced()
        assert f2.is_reduced()
        assert not f3.is_reduced()
    
    def test_value(self):
        f = BinaryQuadraticForm(2, 2, 3)
        assert f.value(1, 0) == 2
        assert f.value(0, 1) == 3
        assert f.value(1, 1) == 7


class TestDiscriminant:
    def test_basic(self):
        assert discriminant(2, 2, 3) == -20
        assert discriminant(1, 1, 1) == -3
        assert discriminant(1, 0, 5) == -20


class TestReducedForms:
    def test_discriminant_minus_20(self):
        forms = all_reduced_forms(-20)
        assert len(forms) == 2
        assert BinaryQuadraticForm(1, 0, 5) in forms
        assert BinaryQuadraticForm(2, 2, 3) in forms
    
    def test_discriminant_minus_23(self):
        forms = all_reduced_forms(-23)
        assert len(forms) == 3
    
    def test_positive_discriminant(self):
        forms = all_reduced_forms(5)
        assert forms == []


class TestClassNumber:
    def test_known_values(self):
        assert class_number(-3) == 1
        assert class_number(-4) == 1
        assert class_number(-7) == 1
        assert class_number(-8) == 1
        assert class_number(-11) == 1
        assert class_number(-15) == 2
        assert class_number(-20) == 2
        assert class_number(-23) == 3
        assert class_number(-31) == 3
        assert class_number(-47) == 5


class TestClassGroup:
    def test_discriminant_20(self):
        C = ClassGroup(-20)
        assert C.n == 2
        assert C.is_cyclic()
        assert C.invariants() == [2]
    
    def test_discriminant_23(self):
        C = ClassGroup(-23)
        assert C.n == 3
        assert C.is_cyclic()
        assert C.invariants() == [3]
    
    def test_composition(self):
        C = ClassGroup(-20)
        f1 = BinaryQuadraticForm(2, 2, 3)
        f2 = BinaryQuadraticForm(2, 2, 3)
        comp = C.compose(f1, f2)
        # f1 * f2 = identity (principal form)
        assert comp == C.forms[0]
    
    def test_inverse(self):
        C = ClassGroup(-20)
        f = BinaryQuadraticForm(2, 2, 3)
        inv = C.inverse(f)
        assert inv == BinaryQuadraticForm(2, 2, 3)
    
    def test_order(self):
        C = ClassGroup(-23)
        for f in C.forms:
            order = C.order(f)
            assert order in [1, 3]


class TestFormComposition:
    def test_dirichlet_composition(self):
        f1 = BinaryQuadraticForm(2, 2, 3)
        f2 = BinaryQuadraticForm(2, -2, 3)
        comp = dirichlet_composition(f1, f2)
        # f1 * f2 should be principal form [1, 0, 5]
        assert comp.a == 1
        assert comp.b == 0
        assert comp.c == 5
    
    def test_gauss_composition(self):
        f1 = BinaryQuadraticForm(2, 2, 3)
        f2 = BinaryQuadraticForm(2, -2, 3)
        # Composed form must be reduced before comparing
        from python.disquisitiones import reduce_form
        comp = reduce_form(gauss_composition(f1, f2))
        assert comp.a == 1
        assert comp.b == 0
        assert comp.c == 5


class TestCyclotomicPoly:
    def test_small_n(self):
        assert cyclotomic_poly(1) == [1, -1]  # x - 1
        assert cyclotomic_poly(2) == [1, 1]   # x + 1
        assert cyclotomic_poly(3) == [1, 1, 1]  # x^2 + x + 1
        assert cyclotomic_poly(4) == [1, 0, 1]  # x^2 + 1
        assert cyclotomic_poly(5) == [1, 1, 1, 1, 1]  # x^4 + x^3 + x^2 + x + 1
    
    def test_prime(self):
        p = 7
        phi = cyclotomic_poly(p)
        assert len(phi) == p
        assert all(c == 1 for c in phi)
    
    def test_17(self):
        phi = cyclotomic_poly(17)
        assert len(phi) == 17
        assert all(c == 1 for c in phi)


class TestMobius:
    def test_values(self):
        assert mobius(1) == 1
        assert mobius(2) == -1
        assert mobius(3) == -1
        assert mobius(4) == 0  # square factor
        assert mobius(6) == 1   # 2*3, two primes
        assert mobius(30) == -1 # 2*3*5, three primes


class TestHeptadecagon:
    def test_vertices(self):
        verts = heptadecagon_vertices()
        assert len(verts) == 17
        for x, y in verts:
            assert abs(x*x + y*y - 1) < 1e-10
    
    def test_gauss_periods(self):
        periods = gauss_periods_17()
        eta1, eta2 = periods['eta1'], periods['eta2']
        assert abs(eta1 + eta2 + 1) < 1e-10  # eta1 + eta2 = -1
        assert abs(eta1 * eta2 + 4) < 1e-10  # eta1 * eta2 = -4


class TestCos2PiOver17:
    def test_exact_expression(self):
        expr = cos_2pi_over_17()
        assert 'sqrt(17)' in expr
        assert 'sqrt(34' in expr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])