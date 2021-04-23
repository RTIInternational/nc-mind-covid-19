SHELL := /bin/bash

.PHONY: build_odd clean

# to run with the right python, excute:
#     make build_all PYTHON_BIN=$(which python)
# with your python environment activated

compile_odd.md:
	@test -n "$(PYTHON_BIN)" || (echo "PYTHON_BIN not set" ; exit 1)
	${PYTHON_BIN} ./docs/odd/odd_compile.py --output_path=compile_odd.md

build_odd_pdf: compile_odd.md
	pandoc --filter pandoc-citeproc \
	  --bibliography=./docs/odd/citations.bib \
	  --csl=./docs/odd/american-medical-association.csl \
	  -s compile_odd.md \
	  -o paper.pdf

build_odd_docx: compile_odd.md
	pandoc --filter pandoc-citeproc \
	  --bibliography=./docs/odd/citations.bib \
	  --csl=./docs/odd/american-medical-association.csl \
	  -s compile_odd.md \
	  -o paper.docx

build_all: build_odd_pdf build_odd_docx
	rm compile_odd.md

clean:
	rm -f paper.pdf
	rm -f paper.docx
