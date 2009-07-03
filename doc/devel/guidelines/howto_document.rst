.. _howto_document:

============================
 How to write documentation
============================

Nipy_ uses the Sphinx_ documentation generating tool.  Sphinx
translates reST_ formatted documents into html and pdf documents.  All
our documents and docstrings are in reST format, this allows us to
have both human-readable docstrings when viewed in ipython_, and
web and print quality documentation.


Building the documentation
--------------------------

You need to have Sphinx_ (version 0.6.2 or above) and graphviz_ (version
2.20 or greater).

The ``Makefile`` (in the top-level doc directory) automates the
generation of the documents.  To make the HTML documents::

  make html

For PDF documentation do::

  make pdf

The built documentation is then placed in a ``build/html`` or
``build/latex`` subdirectories.

For more options, type::

  make help

Viewing the documentation
-------------------------

We also build our website using sphinx_.  All of the documentation in
the ``docs`` directory is included on the website.  There are a few
files that are website only and these are placed in the ``www``
directory.  The easiest way to view the documentation while editing
is to build the website and open the local build in your browser::

  make web

Then open ``www/build/html/index.html`` in your browser.


Syntax
------

Please have a look at our :ref:`sphinx_helpers` for examples on using
Sphinx_ and reST_ in our documentation.

The Sphinx website also has an excellent `sphinx rest`_ primer.

Additional reST references::
  - `reST primer <http://docutils.sourceforge.net/docs/user/rst/quickstart.html>`_
  - `reST quick reference <http://docutils.sourceforge.net/docs/user/rst/quickref.html>`_

Consider using emacs for editing rst files - see :ref:`rst_emacs`

Style
-----

Nipy has adopted the numpy_ documentation standards.  The `numpy
coding style guideline`_ is the main reference for how to format the
documentation in your code.  It's also useful to look at the `source
reST file
<http://svn.scipy.org/svn/numpy/trunk/doc/HOWTO_DOCUMENT.txt>`_ that
generates the coding style guideline.

Numpy has a `detailed example
<http://svn.scipy.org/svn/numpy/trunk/doc/EXAMPLE_DOCSTRING.txt>`_ for
writing docstrings.

.. _`numpy coding style guideline`:
   http://scipy.org/scipy/numpy/wiki/CodingStyleGuidelines

Documentation Problems
----------------------

See our :ref:`documentation_faq` if you are having problems building
or writing the documentation.

.. include:: ../../links_names.txt
