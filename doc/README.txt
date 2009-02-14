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

Once the documentation is built, the **Developer's Guide** has
sections on *How to write documentation* and a *Sphinx Cheat Sheet*.
You need to have Sphinx_ and graphviz_
installed in order to build the documentation.

The ``Makefile`` automates the generation of the documents.  To
make the HTML documents::

  make html

For PDF documentation do::

  make pdf

The built documentation is then placed in a ``build/html`` or
``build/latex`` subdirectories.


Viewing the documentation
-------------------------

We also build our website using sphinx_.  All of the documentation in
the ``docs`` directory is included on the website.  There are a few
files that are *website only* and these are placed in the ``www``
directory.  The easiest way to view the documentation while editing it
is to build the website and open the local build in your browser::

  make web

Then open ``www/build/html/index.html`` in your browser.


.. Since this README.txt is not processed by Sphinx during the
.. documentation build, I've included the links directly so it is at
.. least a valid reST doc.

.. _graphviz: http://www.graphviz.org/
.. _Sphinx: http://sphinx.pocoo.org/
.. _reST: http://docutils.sourceforge.net/rst.html
.. _numpy: http://www.scipy.org/NumPy
