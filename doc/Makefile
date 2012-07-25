# Makefile for Sphinx documentation
#

SF_USER ?= matthewbrett
DIST_DIR = dist

# You can set these variables from the command line.
SPHINXOPTS    = #-q # suppress all output but warnings
SPHINXBUILD   = sphinx-build
PAPER         =

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d build/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

.PHONY: help clean pdf all dist public_html web web_public htmlonly api html pickle htmlhelp latex changes linkcheck doctest

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      	make HTML and API documents"
	@echo "  htmlonly  	make HTML documents only"
	@echo "  api       	make API documents only"
	@echo "  latex     	make LaTeX documents (you can set\
					PAPER=a4 or PAPER=letter)"
	@echo "  all		make HTML, API and PDF documents"
	@echo "  clean		remove all generated documents"
	@echo
	@echo "  linkcheck 	check all external links for integrity"
	@echo "  doctest   	run doctests in reST files"
	@echo "  pdf		make <latex> and run the PDF generation"
	@echo "  dist		make <all> and put results in $DIST_DIR/"
	@echo "  public_html	make <dist> and rsync with public website"
	@echo "  upload-stable-web	make <dist> and rsync with public stable website"
	@echo "  upload-devel-web	make <dist> and rsync with public devel website"
	@echo "  gitwash-update	update git workflow from source repo"

# Commented these out, wasn't clear if we'd use these targets or not.
# @echo "  pickle    to make pickle files (usable by e.g. sphinx-web)"
# @echo "  htmlhelp  to make HTML files and a HTML help project"
# @echo "  changes   to make an overview over all changed/added/deprecated items"

clean:
	-rm -rf build/* $(DIST_DIR)/* *~ api/generated labs/generated
	-rm -f manual

pdf: latex
	cd build/latex && make all-pdf

all: html pdf

dist: clean all
	mkdir -p $(DIST_DIR)
	ln build/latex/nipy*.pdf $(DIST_DIR)
	cp -a build/html/* $(DIST_DIR)
	@echo "Build finished.  Final docs are in $(DIST_DIR)"

# This one udates for the specific user named at the top of the makefile
upload-stable-web: upload-stable-web-$(SF_USER)

# This one updates according to passed user e.g
# make upload-stable-web-my_own_sf_user
upload-stable-web-%: dist
	rsync -rzhvp --delete --chmod=Dg+s,g+rw $(DIST_DIR)/* \
		$*,nipy@web.sourceforge.net:/home/groups/n/ni/nipy/htdocs/nipy/stable

# This one udates for the specific user named at the top of the makefile
upload-devel-web: upload-devel-web-$(SF_USER)

# This one updates according to passed user e.g
# make upload-devel-web-my_own_sf_user
upload-devel-web-%: dist
	rsync -rzhvp --delete --chmod=Dg+s,g+rw $(DIST_DIR)/* \
		$*,nipy@web.sourceforge.net:/home/groups/n/ni/nipy/htdocs/nipy/devel

htmlonly:
	mkdir -p build/html build/doctrees
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) build/html
	@echo
	@echo "Build finished. The HTML pages are in build/html."

api:
	python ../tools/build_modref_templates.py
	@echo "Build API docs finished."

html: api htmlonly
	-ln -s build manual
	@echo "Build HTML and API finished."

gitwash-update:
	python ../tools/gitwash_dumper.py devel/guidelines nipy \
		--github-user=nipy \
		--project-url=http://nipy.org/nipy \
		--project-ml-url=http://mail.scipy.org/mailman/listinfo/nipy-devel

pickle:
	mkdir -p build/pickle build/doctrees
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) build/pickle
	@echo
	@echo "Build finished; now you can process the pickle files or run"
	@echo "  sphinx-web build/pickle"
	@echo "to start the sphinx-web server."

htmlhelp:
	mkdir -p build/htmlhelp build/doctrees
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) build/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in build/htmlhelp."

latex: api
	mkdir -p build/latex build/doctrees
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) build/latex
	# Clear bug for longtable column output in sphinx
	python ../tools/fix_longtable.py build/latex/nipy.tex
	@echo
	@echo "Build finished; the LaTeX files are in build/latex."
	@echo "Run \`make all-pdf' or \`make all-ps' in that directory to" \
	      "run these through (pdf)latex."

changes:
	mkdir -p build/changes build/doctrees
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) build/changes
	@echo
	@echo "The overview file is in build/changes."

linkcheck:
	mkdir -p build/linkcheck build/doctrees
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) build/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in build/linkcheck/output.txt."

doctest:
	mkdir -p build/doctest build/doctrees
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) build/doctest
	@echo
	@echo "The overview file is in build/doctest."

