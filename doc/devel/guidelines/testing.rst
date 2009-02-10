.. _testing:

=========
 Testing
=========

Nipy uses the Numpy_ test framework which is based on nose_.  If you
plan to do much development you should familiarize yourself with nose
and read through the `numpy testing guidelines
<http://projects.scipy.org/scipy/numpy/wiki/TestingGuidelines>`_.

Running tests
-------------

For our tests, we have collected a set of fmri imaging data which are
required for the tests to run. The data must be downloaded separately
and installed in your home directory ``$HOME/.nipy/tests/data``.  From
your home directory::

    mkdir -p .nipy/tests/data
    svn co http://neuroimaging.scipy.org/svn/ni/data/trunk/fmri .nipy/tests/data

Tests can be run on the package::

    import neuroimaging as ni
    ni.test()

Or on individual modules and functions from the command line::

    nosetests test_image.py


.. include:: ../../links_names.txt
