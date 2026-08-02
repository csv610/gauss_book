"""Test script for disquisitiones module - run with python3 test_fix.py"""

import math
from python.disquisitiones import *

# Test basic functions
print("Testing reduced forms...")
forms = all_reduced_forms(-20)
print(f'D=-20 reduced forms: {forms}')
print(f'Class number h(-20): {class_number(-20)}')

# Test class group
print('\nTesting class group D=-20...')
C = ClassGroup(-20)
print(f'Class group order: {C.n}')
print(f'Cyclic: {C.is_cyclic()}')

# Test composition
print('\nTesting composition...')
f1 = BinaryQuadraticForm(2, 2, 3)  # 2x^2 + 2xy + 3y^2
f2 = BinaryQuadraticForm(2, -2, 3)  # 2x^2 - 2xy + 3y^2
print(f'f1 = {f1}')
print(f'f2 = {f2}')

comp = dirichlet_composition(f1, f2)
print(f'f1 * f2 = {comp}')
print(f'Reduced: {reduce_form(comp)}')

# Test cyclotomic polynomials
print('\nCyclotomic polynomials:')
for n in [1, 2, 3, 4, 5, 17]:
    phi = cyclotomic_poly(n)
    print(f'  Phi_{n}(x) = {poly_to_str(phi)}')

# Test 17-gon
print('\n17-gon vertices:')
verts = heptadecagon_vertices()
for i, (x, y) in enumerate(verts[:4]):
    print(f'  V{i}: ({x:.6f}, {y:.6f})')

print('\nGauss periods for 17-gon:')
periods = gauss_periods_17()
for name, val in periods.items():
    print(f'  {name} = {val:.6f}')

print(f'\ncos(2π/17) = {cos_2pi_over_17()}')

print('\nAll tests completed!')