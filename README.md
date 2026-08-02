# Gauss: Mathematics from Theory to Code

A textbook and Python library covering Gauss's contributions to mathematics, from number theory to orbital mechanics.

## What's Inside

- **15 chapters** covering number theory, analysis, linear algebra, statistics, and geometry
- **18 Python modules** with 5,500+ lines of production-ready code
- **193 tests** ensuring correctness
- **Index** with 353 terms
- **Glossary** with 67 definitions
- **Exercise solutions** for all chapters

## Quick Start

### Build the Book
```bash
make book        # Generate gauss_book.pdf
make all         # Build book and student guide
```

### Run Tests
```bash
make test        # Run all 193 tests
```

### Use the Code
```python
from python.modular import mod_inv, solve_congruence
from python.gaussian_elim import lu_decomposition, cholesky
from python.quadratic import legendre_symbol, quadratic_residue
```

## Chapters

| # | Topic | Module |
|---|-------|--------|
| 1 | Modular Arithmetic | modular.py |
| 2 | Quadratic Reciprocity | quadratic.py |
| 3 | Gauss Sums | gauss_sums.py |
| 4 | Disquisitiones Arithmeticae | disquisitiones.py |
| 5 | Gaussian Integral | gaussian_integral.py |
| 6 | Theta Functions | theta.py |
| 7 | Arithmetic-Geometric Mean | agm.py |
| 8 | Gaussian Elimination | gaussian_elim.py |
| 9 | Least Squares | least_squares.py |
| 10 | Gaussian Quadrature | gauss_quadrature.py |
| 11 | Normal Distribution | normal.py |
| 12 | Gaussian Processes | gp.py |
| 13 | Heptadecagon Construction | heptadecagon.py |
| 14 | Theorema Egregium | theorema.py |
| 15 | Orbital Mechanics | orbital.py |

## Requirements

- Python 3.10+
- NumPy, SciPy
- TeX Live (for PDF generation)

## Install

```bash
pip install -e ".[test]"
```

## License

Educational use. Based on works by Carl Friedrich Gauss (1777–1855).
