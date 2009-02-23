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

You can also run nose from the command line with a variety of options.
To test an individual module::

    nosetests test_image.py

To test an individual function::

    nosetests test_module:test_function

To test a class::

    nosetests test_module:TestClass

To test a class method::

   nosetests test_module:TestClass.test_method

Verbose mode (*-v* option) will print out the function names as they
are executed.  Standard output is normally supressed by nose, to see
any print statements you must include the *-s* option.  In order to
get a "full verbose" output, call nose like this::

    nosetests -sv test_module.py

To include doctests in the nose test::

   nosetests -sv --with-doctest test_module.py

Nose will also investigate your test coverage. This requires `Ned
Batchelder's coverage module
<http://nedbatchelder.com/code/modules/coverage.html>`_ to be
installed::

    nosetests -sv --with-coverage test_module.py   

For details on all the command line options::

    nosetests --help


.. include:: ../../links_names.txt
