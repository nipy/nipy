# Automating common tasks for NIPY development

PYTHON = python
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

dev: cythonize
	$(PYTHON) setup.py build_ext --inplace

test:
	cd .. && $(PYTHON) -c 'import nipy; nipy.test()'

build:
	$(PYTHON) setup.py build

install:
	$(PYTHON) setup.py install

cythonize:
	$(PYTHON) tools/nicythize

bdist_rpm:
	$(PYTHON) setup.py bdist_rpm \
	  --doc-files "doc" \
	  --packager "nipy authors <https://mail.python.org/mailman/listinfo/neuroimaging>"
	  --vendor "nipy authors <https://mail.python.org/mailman/listinfo/neuroimaging>"

# build MacOS installer -- depends on patched bdist_mpkg for Leopard
bdist_mpkg:
	$(PYTHON) tools/mpkg_wrapper.py setup.py install

# Check for files not installed
check-files:
	$(PYTHON) -c 'from nisext.testers import check_files; check_files("nipy")'

# Print out info for possible install methods
check-version-info:
	$(PYTHON) -c 'from nisext.testers import info_from_here; info_from_here("nipy")'

# Run tests from installed code
installed-tests:
	$(PYTHON) -c 'from nisext.testers import tests_installed; tests_installed("nipy")'

# Run tests from sdist archive of code
sdist-tests:
	$(PYTHON) -c 'from nisext.testers import sdist_tests; sdist_tests("nipy")'

# Run tests from bdist egg of code
bdist-egg-tests:
	$(PYTHON) -c 'from nisext.testers import bdist_egg_tests; bdist_egg_tests("nipy")'

source-release: distclean
	python -m compileall .
	make distclean
	python setup.py sdist --formats=gztar,zip

venv-tests:
	# I use this for python2.5 because the sdist-tests target doesn't work
	# (the tester routine uses a 2.6 feature)
	make distclean
	- rm -rf $(VIRTUAL_ENV)/lib/python$(PYVER)/site-packages/nipy
	python setup.py install
	cd .. && nosetests $(VIRTUAL_ENV)/lib/python$(PYVER)/site-packages/nipy

tox-fresh:
	# tox tests with fresh-installed virtualenvs.  Needs network.  And
	# pytox, obviously.
	tox -c tox.ini

tox-stale:
	# tox tests with MB's already-installed virtualenvs (numpy and nose
	# installed)
	tox -e python25,python26,python27,python32,np-1.2.1

recythonize:
	# Recythonize all pyx files
	find . -name "*.pyx" -exec cython -I libcstat/wrapper -I lib/fff_python_wrapper {} \;

# Website stuff
$(WWW_DIR):
	if [ ! -d $(WWW_DIR) ]; then mkdir -p $(WWW_DIR); fi

htmldoc: build
	cd $(DOCSRC_DIR) && PYTHONPATH=$(CURDIR):$(PYTHONPATH) $(MAKE) html

pdfdoc: build
	cd $(DOCSRC_DIR) && PYTHONPATH=$(CURDIR):$(PYTHONPATH) $(MAKE) latex
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
