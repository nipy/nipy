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
  highlighting, autodocs.

* _static - used by the sphinx build system.

* _templates - used by the sphinx build system.

Building the documentation
--------------------------

Once the documentation is built, the **Developer's Guide** has
sections on *How to write documentation* and a *Sphinx Cheat Sheet*.
You need to have sphinx_ and graphviz_ installed in order to build the
documentation.

The ``make.py`` script is used to build all of the documentation. For
html documentation do::

  python make.py html

For PDF documentation do::

  python make.py latex

The built documentation is then placed in a ``build/html`` or
``build/latex`` subdirectories.

.. _graphviz: http://www.graphviz.org/
.. _sphinx: http://sphinx.pocoo.org/
.. _ReST: http://docutils.sourceforge.net/rst.html


