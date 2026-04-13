# Minimal Makefile for Sphinx documentation
SPHINXBUILD = uv run --group docs sphinx-build
SOURCEDIR = docs/source
BUILDDIR = docs/build

.PHONY: html live clean

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

live:
	uv run --group docs sphinx-autobuild $(SOURCEDIR) $(BUILDDIR)/html --port 8001

clean:
	rm -rf $(BUILDDIR)/*
