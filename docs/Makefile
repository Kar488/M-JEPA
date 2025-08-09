# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# === Convenience targets ===

# Build HTML into the default _build/html
html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

# Zip the built HTML so you can download it from Codex/CI
htmlzip: html
	@cd "$(BUILDDIR)" && rm -f html_docs.zip && zip -r html_docs.zip html
	@echo "Zipped HTML at $(BUILDDIR)/html_docs.zip"

# Serve locally (handy in Codespaces/Codex; then open the forwarded port)
serve: html
	@python -m http.server -d "$(BUILDDIR)/html" 8000

# Build directly into docs/ so GitHub Pages can publish from /docs
pages:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" docs $(SPHINXOPTS)
	@touch docs/.nojekyll
	@echo "Built Pages site in ./docs (commit & push, then enable Pages: Deploy from branch → /docs)"

# Clean
clean:
	@rm -rf "$(BUILDDIR)" docs/.doctrees docs/.buildinfo