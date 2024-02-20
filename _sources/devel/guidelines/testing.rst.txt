.. _testing:

=======
Testing
=======

Nipy uses the the Pytest_ framework.  If you plan to do development on nipy
please have a look at the `Pytest docs <pytest>`_ and read through the `numpy
testing guidelines
<http://projects.scipy.org/scipy/numpy/wiki/TestingGuidelines>`_.

.. _automated-testing:

Automated testing
-----------------

We run the tests on every commit with travis-ci_ |--| see `nipy on travis`_.

We also have a farm of machines set up to run the tests on every commit to the
``main`` branch at `nipy buildbot`_.

Writing tests
-------------

Test files
^^^^^^^^^^

We like test modules to import their testing functions and classes from the
module in which they are defined.  For example, we might want to use the
``assert_array_equal``, ``assert_almost_equal`` functions defined by
``numpy``, and the ``funcfile, anatfile`` variables from ``nipy``::

    from numpy.testing import assert_array_equal, assert_almost_equal
    from nipy.testing import funcfile, anatfile

Please name your test file with the ``test_`` prefix followed by the module
name it tests.  This makes it obvious for other developers which modules are
tested, where to add tests, etc...  An example test file and module pairing::

      nipy/core/reference/coordinate_system.py
      nipy/core/reference/tests/test_coordinate_system.py

All tests go in a ``tests`` subdirectory for each package.

Temporary files
^^^^^^^^^^^^^^^

If you need to create a temporary file during your testing, you could
use one of these three methods, in order of convenience:

#. `StringIO <http://docs.python.org/library/stringio.html>`_

   StringIO creates an in memory file-like object. The memory buffer
   is freed when the file is closed.  This is the preferred method for
   temporary files in tests.

#. `in_tmp_path` Pytest fixture.

   This is a convenient way of putting you into a temporary directory so you can
   save anything you like into the current directory, and feel fine about it
   after.  Like this::

       def test_func(in_tmp_path):
           f = open('myfile', 'wt')
           f.write('Anything at all')
           f.close()

   One thing to be careful of is that you may need to delete objects holding
   onto the file before you exit the enclosing function, otherwise Windows may
   refuse to delete the file.

#. `tempfile.mkstemp <http://docs.python.org/library/tempfile.html>`_

   This will create a temporary file which can be used during testing.
   There are parameters for specifying the filename *prefix* and
   *suffix*.

   .. Note::

        The tempfile module includes a convenience function
        *NamedTemporaryFile* which deletes the file automatically when
        it is closed.  However, whether the files can be opened a
        second time varies across platforms and there are problems
        using this function on *Windows*.

   Example::

    from tempfile import mkstemp
    try:
        fd, name = mkstemp(suffix='.nii.gz')
        tmpfile = open(name)
        save_image(fake_image, tmpfile.name)
        tmpfile.close()
    finally:
        os.unlink(name)  # This deletes the temp file

Please don't just create a file in the test directory and then remove it with
a call to ``os.remove``.  For various reasons, sometimes ``os.remove`` doesn't
get called and temp files get left around.

Many tests in one test function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To keep tests organized, it's best to have one test function correspond to one
class method or module-level function.  Often though, you need many individual
tests to thoroughly cover the method/function.  For convenience, we often
write many tests in a single test function.  This has the disadvantage that if
one test fails, the testing framework will not run any of the subsequent tests
in the same function.  This isn't a big problem in practice, because we run
the tests so often (:ref:`automated-testing`) that we can quickly pick up and
fix the failures.

For axample, this test function executes four tests::

    def test_index():
        cs = CoordinateSystem('ijk')
        assert_equal(cs.index('i'), 0)
        assert_equal(cs.index('j'), 1)
        assert_equal(cs.index('k'), 2)
        assert_raises(ValueError, cs.index, 'x')

Suppress *warnings* on test output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to reduce noise when running the tests, consider suppressing
*warnings* in your test modules.  See the `pytest documentation
<https://docs.pytest.org/en/7.1.x/how-to/capture-warnings.html>`_ for various
ways to do that, or search our code for `pytest.mark` for examples.

Running tests
-------------

Running the full test suite
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run nipy's tests, you will need to pytest_ installed.  Then::

    pytest nipy

You can run the full tests, including doctests with::

    pip install pytest-doctestplus

    pytest --doctest-plus nipy

Install optional data packages for testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For our tests, we have collected a set of fmri imaging data which are
required for the tests to run.  To do this, download the latest example
data and template package files from `NIPY data packages`_. See
:ref:`data-files`.

Running individual tests
^^^^^^^^^^^^^^^^^^^^^^^^

You can also run the tests from the command line with a variety of options.

To test an individual module::

    pytest nipy/core/image/tests/test_image.py

To test an individual function::

    pytest nipy/core/image/tests/test_image.py::test_maxmin_values

To test a class::

    pytest nipy/algorithms/clustering/tests/test_clustering.py::TestClustering

To test a class method::

    pytest nipy/algorithms/clustering/tests/test_clustering.py::TestClustering.testkmeans1

Verbose mode (*-v* option) will print out the function names as they
are executed.  Standard output is normally suppressed by Pytest, to see
any print statements you must include the *-s* option.  In order to
get a "full verbose" output, call Pytest like this::

    pytest -sv nipy

To include doctests in the tests::

   pytest -sv --docest-plus nipy

.. _coverage:

.. include:: ./coverage_testing.rst

.. include:: ../../links_names.txt
