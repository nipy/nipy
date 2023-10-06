# Automating common tasks for NIPY development

PYTHON ?= python
HTML_DIR = doc/build/html
LATEX_DIR = doc/build/latex
WWW_DIR = doc/dist
DOCSRC_DIR = doc
PROJECT = nipy

clean-pyc:
	find . -regex ".*\.pyc" -exec rm -rf "{}" \;

clean: clean-pyc
	find . -regex ".*\.so" -exec rm -rf "{}" \;
	find . -regex ".*\.pyd" -exec rm -rf "{}" \;
	find . -regex ".*~" -exec rm -rf "{}" \;
	find . -regex ".*#" -exec rm -rf "{}" \;
	rm -rf build
	$(MAKE) -C doc clean

clean-dev: clean dev

distclean: clean
	-rm MANIFEST
	-rm $(COVERAGE_REPORT)
	@find . -name '*.py[co]' \
		 -o -name '*.a' \
		 -o -name '*,cover' \
		 -o -name '.coverage' \
		 -o -iname '*~' \
		 -o -iname '*.kcache' \
		 -o -iname '*.pstats' \
		 -o -iname '*.prof' \
		 -o -iname '#*#' | xargs -L10 rm -f
	-rm -r dist
	-rm build-stamp
	-rm -r .tox
	-git clean -fxd

install:
	$(PYTHON) -m pip install .

editable:
	$(PYTHON) -m pip install --no-build-isolation --editable .

# Print out info for possible install methods
check-version-info:
	bash tools/show_version_info.sh

source-release: distclean
	$(PYTHON) -m build . --sdist

tox-fresh:
	# tox tests with fresh-installed virtualenvs.  Needs network.  And
	# pytox, obviously.
	tox -c tox.ini

# Website stuff
$(WWW_DIR):
	if [ ! -d $(WWW_DIR) ]; then mkdir -p $(WWW_DIR); fi

htmldoc:
	cd $(DOCSRC_DIR) && $(MAKE) html

pdfdoc:
	cd $(DOCSRC_DIR) && $(MAKE) latex
	cd $(LATEX_DIR) && $(MAKE) all-pdf

html: html-stamp
html-stamp: $(WWW_DIR) htmldoc
	cp -r $(HTML_DIR)/* $(WWW_DIR)
	touch $@

pdf: pdf-stamp
pdf-stamp: $(WWW_DIR) pdfdoc
	cp $(LATEX_DIR)/*.pdf $(WWW_DIR)
	touch $@

website: website-stamp
website-stamp: $(WWW_DIR) html-stamp pdf-stamp
	cp -r $(HTML_DIR)/* $(WWW_DIR)
	touch $@

upload-html: html-stamp
	./tools/upload-gh-pages.sh $(WWW_DIR) $(PROJECT)

refresh-readme:
	$(PYTHON) tools/refresh_readme.py nipy

.PHONY: orig-src pylint
