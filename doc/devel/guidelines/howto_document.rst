.. _howto_document:

============================
 How to write documentation
============================

Syntax
------

Nipy_ uses the Sphinx_ documentation generating tool.  Sphinx
translates reST_ formatted documents into html and pdf documents.  All
our documents and docstrings are in reST format, this allows us to
have both human-readable docstrings when viewed in ipython_, and
web and print quality documentation.

Please have a look at our :ref:`sphinx_helpers` for examples on using
Sphinx and reST in our documentation.

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

.. include:: ../../links_names.txt
