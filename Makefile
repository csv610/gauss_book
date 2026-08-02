# Gauss Book Project Makefile
#
# Targets:
#   book       - Build gauss_book.pdf (LaTeX -> PDF)
#   guide      - Build gauss_user_guide.pdf
#   all        - Build both PDFs
#   test       - Run Python test suite
#   test-fast  - Run tests (no cov), continue on error
#   clean      - Remove generated files
#   rebuild    - book + clean of aux files

LATEXMK = latexmk
LATEXMK_FLAGS = -pdf -silent -interaction=nonstopmode
PYTHON = python3
PYTEST = python3 -m pytest

.PHONY: all book guide test test-fast clean rebuild

all: book guide test

book: gauss_book.pdf

guide: gauss_user_guide.pdf

gauss_book.pdf: gauss_book.tex chapters/*.tex references.bib
	$(LATEXMK) $(LATEXMK_FLAGS) gauss_book.tex

gauss_user_guide.pdf: gauss_user_guide.tex chapters/*.tex
	$(LATEXMK) $(LATEXMK_FLAGS) gauss_user_guide.tex

test:
	$(PYTEST) tests/ -v

test-fast:
	$(PYTEST) tests/ --tb=short --no-header -q || true

clean:
	$(LATEXMK) -c
	rm -f *.out *.run.xml *.bbl *.bcf *.blg *.idx *.ilg *.ind *.lol
	rm -f gauss_book.pdf gauss_user_guide.pdf
	rm -rf __pycache__ .pytest_cache python/__pycache__ tests/__pycache__

rebuild: clean book
