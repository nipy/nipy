============================
 How to write documentation
============================

Syntax
------

Nipy_ uses the Sphinx_ documentation generating tool.  Sphinx_
translates reST_ formatted documents into html and pdf documents.  All
our documents and docstrings are in reST_ format, this allows us to
have both human-readable docstrings when viewed in ipython_, and
web and print quality documentation.

Please have a look at :ref:`sphinx_helpers` for information on using
both Sphinx_ and reST_.  The Sphinx_ website also has an excellent
`sphinx rest`_ primer.

Additional reST_ references::
  - `reST primer <http://docutils.sourceforge.net/docs/user/rst/quickstart.html>`_
  - `reST quick reference <http://docutils.sourceforge.net/docs/user/rst/quickref.html>`_

Consider using emacs for editing rst files - see :ref:`rst_emacs`

Style
-----

Nipy_ has adopted the numpy_ documentation standards.  The `numpy
coding style guideline`_ is the main reference for how to format the
documentation in your code.  It's also useful to look at the `source
reST file
<http://svn.scipy.org/svn/numpy/trunk/doc/HOWTO_DOCUMENT.txt>`_ that
generates the coding style guideline.

.. _`numpy coding style guideline`:
   http://scipy.org/scipy/numpy/wiki/CodingStyleGuidelines

External hyperlinks
-------------------

We have created a ``links_names.txt`` file to reference common
external hyperlinks that will be used throughout the documentation.
Links in the ``links_names.txt`` file are created using the `reST
reference
<http://docutils.sourceforge.net/docs/user/rst/quickref.html#hyperlink-targets>`_
syntax::

	.. _targetname: http://www.external_website.org

To refer to the reference in a separate reST file, include the
``links_names.txt`` file and refer to the link through it's target name.  For example, put this include at the bottom of your reST document::
     
     .. include:: ../links_names.txt

and refer to the hyperlink target::

    blah blah blah targetname_ more blah


.. include:: ../../links_names.txt
