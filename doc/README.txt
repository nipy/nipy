====================
 nipy documentation
====================

This is the top level build directory for the sphinx nipy documentation.  All
of the documentation is written using sphinx, a python documentation system
built on top of ReST.  This directory contains

* users - the user documentation.

* devel - documentation for developers.

* faq - frequently asked questions

* api - placeholders to automatically generate the api documentation

* index.rst - the top level include document for sampledocs document.

* conf.py - the sphinx configuration.

* make.py - the build script to build the html or PDF docs.  Do
  `python make.py html` or `python make.py latex` for PDF.

* sphinxext - some extensions to sphinx to handle math, ipython syntax
  highlighting, autodocs.

* _static - used by the sphinx build system.

* _templates - used by the sphinx build system.
