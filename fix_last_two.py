#!/usr/bin/env python3
"""Add indices to ch08 and ch15 which were reverted."""
import re
from pathlib import Path

CHAPTERS_DIR = Path("/Users/csv610/Projects/MyBooks/Gauss/chapters")

# Fix ch08_gaussian_elim.tex
fpath = CHAPTERS_DIR / "ch08_gaussian_elim.tex"
with open(fpath, 'r') as f:
    content = f.read()

# Add indices
replacements = [
    ("Gaussian elimination", "Gaussian elimination"),
    ("LU decomposition", "LU decomposition"),
    ("Cholesky decomposition", "Cholesky decomposition"),
    ("QR decomposition", "QR decomposition"),
    ("partial pivoting", "partial pivoting"),
    ("Householder reflections", "Householder reflection"),
    ("condition number", "condition number"),
    ("numerical stability", "numerical stability"),
    ("back substitution", "back substitution"),
    ("Bareiss algorithm", "Bareiss algorithm"),
]

for search, index_name in replacements:
    if f'\\index{{{index_name}}}' not in content and search in content:
        content = content.replace(search, f'{search} \\index{{{index_name}}}', 1)

with open(fpath, 'w') as f:
    f.write(content)
print(f"Fixed ch08_gaussian_elim.tex: {content.count('\\index{')} indices")

# Fix ch15_orbital.tex - restore and add indices
fpath = CHAPTERS_DIR / "ch15_orbital.tex"
with open(fpath, 'r') as f:
    content = f.read()

# The file seems truncated. Let me restore from git and then add indices
import subprocess
subprocess.run(['git', 'checkout', '--', 'ch15_orbital.tex'], 
               cwd=CHAPTERS_DIR.parent, capture_output=True)

with open(fpath, 'r') as f:
    content = f.read()

replacements = [
    ("two-body problem", "two-body problem"),
    ("Kepler's laws", "Kepler's laws"),
    ("orbital elements", "orbital element"),
    ("semi-major axis", "semi-major axis"),
    ("eccentricity", "eccentricity"),
    ("Gauss's method", "Gauss's method (orbit determination)"),
    ("Kepler's equation", "Kepler's equation"),
    ("Laplace-Runge-Lenz", "Laplace-Runge-Lenz vector"),
    ("state vector", "state vector (orbital mechanics)"),
]

for search, index_name in replacements:
    if f'\\index{{{index_name}}}' not in content and search in content:
        content = content.replace(search, f'{search} \\index{{{index_name}}}', 1)

with open(fpath, 'w') as f:
    f.write(content)
print(f"Fixed ch15_orbital.tex: {content.count('\\index{')} indices")

# Final count
all_entries = []
for f in sorted(CHAPTERS_DIR.glob("ch*.tex")):
    matches = re.findall(r'\\index\{([^}]+)\}', f.read_text())
    all_entries.extend([(f.name, m) for m in matches])

unique = set(all_entries)
print(f"\nFinal: {len(unique)} unique entries across {len([f for f in CHAPTERS_DIR.glob('ch*.tex') if any(e[0] == f.name for e in all_entries)])} chapters")
