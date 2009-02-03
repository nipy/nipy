====================
 Nipy Documentation
====================

This is the top level build directory for the sphinx nipy documentation.  All
of the documentation is written using sphinx_, a python documentation system
built on top of ReST_.  This directory contains

* users - the user documentation.

* devel - documentation for developers.

* faq - frequently asked questions

* api - placeholders to automatically generate the api documentation

* index.rst - the top level include document for sampledocs document.

* conf.py - the sphinx configuration.

* make.py - the build script to build the html or PDF docs.

* sphinxext - some extensions to sphinx to handle math, ipython syntax
  highlighting, numpy_ docstring parsing, and autodocs.

* _static - used by the sphinx build system.

* _templates - used by the sphinx build system.

Building the documentation
--------------------------

Once the documentation is built, the **Developer's Guide** has
sections on *How to write documentation* and a *Sphinx Cheat Sheet*.
You need to have sphinx_ and graphviz_ installed in order to build the
documentation.

The :file:`Makefile` automates the generation of the documents.  To
make the html documents::

  make html

For PDF documentation do::

  make pdf

The built documentation is then placed in a ``build/html`` or
``build/latex`` subdirectories.

.. include:: links_names.txt

