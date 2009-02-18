====================
 Nipy Documentation
====================

This is the top level build directory for the nipy documentation.  All
of the documentation is written using Sphinx_, a python documentation
system built on top of reST_.  In order to build the documentation,
you must have Sphinx v0.5 or greater installed.

This directory contains:

* Makefile - the build script to build the HTML or PDF docs. Type
  ``make help`` for a list of options.

* users - the user documentation.

* devel - documentation for developers.

* faq - frequently asked questions

* api - placeholders to automatically generate the api documentation

* www - source files for website only reST documentss which should not
  go in the generated PDF documentation.

* links_names.txt - reST document with hyperlink targets for common
  links used throughout the documentation

* .rst files - some top-level documentation source files

* conf.py - the sphinx configuration.

* sphinxext - some extensions to sphinx to handle math, ipython syntax
  highlighting, numpy_ docstring
  parsing, and autodocs.

* _static - used by the sphinx build system.

* _templates - used by the sphinx build system.


Building the documentation
--------------------------

Instructions for building the documentation are in the file:
``devel/guidelines/howto_document.rst``

.. Since this README.txt is not processed by Sphinx during the
.. documentation build, I've included the links directly so it is at
.. least a valid reST doc.

.. _Sphinx: http://sphinx.pocoo.org/
.. _reST: http://docutils.sourceforge.net/rst.html
.. _numpy: http://www.scipy.org/NumPy
