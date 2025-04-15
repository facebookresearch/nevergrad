# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

PYTHON        = python3
VENV          = $(BUILDDIR)/venv
PIP_OPTIONS   =
HAS_BASH      = $(shell command -v bash >/dev/null 2>&1 && echo YES || echo NO)
HAS_PYTHON3   = $(shell command -v python3 >/dev/null 2>&1 && echo YES || echo NO)

$(VENV):
	@if [ "$(HAS_BASH)" = "NO" ] ; then echo "bash was not found at $(OSHELL)! Please use: $(MAKE) SHELL=/path/to/bash" 1>&2; exit 1; fi
	@if [ "$(HAS_PYTHON3)" = "NO" ] ; then echo "python3 was not found! Please see README for further instructions" 1>&2; exit 1; fi
	@( \
		$(PYTHON) -m venv $(VENV); \
		. $(VENV)/bin/activate; \
		pip $(PIP_OPTIONS) install --upgrade pip; \
		pip $(PIP_OPTIONS) install --upgrade wheel; \
		pip $(PIP_OPTIONS) install -r $(SOURCEDIR)/requirements.txt; \
		deactivate;\
	)

html: $(VENV) conf.py
	@(\
		. $(VENV)/bin/activate ;\
		PYTHONPATH=.. $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) ;\
		deactivate ;\
	)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@echo "Removing all generated documentation and virtualenv..."
	@rm -rf "$(BUILDDIR)" .doctrees .coverage htmlcov *.pyc *.pyo __pycache__

.PHONY: help html clean

