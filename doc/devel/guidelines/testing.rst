.. _testing:

=======
Testing
=======

Nipy uses the Numpy_ test framework which is based on nose_.  If you
plan to do development on nipy please have a look at the `nose docs <nose>`_
and read through the `numpy testing guidelines
<http://projects.scipy.org/scipy/numpy/wiki/TestingGuidelines>`_.

.. _automated-testing:

Automated testing
-----------------

We run the tests on every commit with travis-ci_ |--| see `nipy on travis`_.

We also have a farm of machines set up to run the tests on every commit to the
``master`` branch at `nipy buildbot`_.

Writing tests
-------------

Test files
^^^^^^^^^^

We like test modules to import their testing functions and classes from the
module in which they are defined.  For example, we might want to use the
``assert_true``, ``assert_equal`` functions defined by ``nose``, the
``assert_array_equal``, ``assert_almost_equal`` functions defined by
``numpy``, and the ``funcfile, anatfile`` variables from ``nipy``::

    from nose.tools import assert_true, assert_equal
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

#. `nibabel.tmpdirs.InTemporaryDirectory` context manager.

   This is a convenient way of putting you into a temporary directory so you can
   save anything you like into the current directory, and feel fine about it
   after.  Like this::

       from ..tmpdirs import InTemporaryDirectory

       with InTemporaryDirectory():
           f = open('myfile', 'wt')
           f.write('Anything at all')
           f.close()

   One thing to be careful of is that you may need to delete objects holding
   onto the file before you exit the ``with`` statement, otherwise Windows may
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

Please don't just create a file in the test directory and then remove it with a
call to ``os.remove``.  For various reasons, sometimes ``os.remove`` doesn't get
called and temp files get left around.

Many tests in one test function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To keep tests organized, it's best to have one test function
correspond to one class method or module-level function.  Often
though, you need many individual tests to thoroughly cover the
method/function.  For convenience, we often write many tests in a single test
function.  This has the disadvantage that if one test fails, nose will not run
any of the subsequent tests in the same function.  This isn't a big problem in
practice, because we run the tests so often (:ref:`automated-testing`) that we
can quickly pick up and fix the failures.

For axample, this test function executes four tests::

    def test_index():
        cs = CoordinateSystem('ijk')
        assert_equal(cs.index('i'), 0)
        assert_equal(cs.index('j'), 1)
        assert_equal(cs.index('k'), 2)
        assert_raises(ValueError, cs.index, 'x')

We used to use `nose test generators
<http://nose.readthedocs.org/en/latest/writing_tests.html#test-generators>`_
for multiple tests in one function.  Test generators are test functions that
return tests and parameters from ``yield`` statements.  You will still find
many examples of these in the nipy codebase, but they made test failures
rather hard to debug, so please don't use test generators in new tests.

Suppress *warnings* on test output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to reduce noise when running the tests, consider suppressing
*warnings* in your test modules.  This can be done in the module-level
setup and teardown functions::

      import warnings
      ...

      def setup():
          # Suppress warnings during tests to reduce noise
          warnings.simplefilter("ignore")

      def teardown():
          # Clear list of warning filters
          warnings.resetwarnings()


Running tests
-------------

Running the full test suite
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run nipy's tests, you will need to nose_ installed.  Then::

    python -c "import nipy; nipy.test()"

You can also run nipy's tests with the ``nipnost`` script in the ``tools``
directory of the nipy distribution::

    ./tools/nipnost nipy

``nipnost`` is a thin wrapper around the standard ``nosetests`` program that
is part of the nose package. The ``nipnost`` wrapper sets up some custom
doctest machinery and makes sure that `matplotlib`_ is using non-interactive
plots.  ``nipy.test()`` does the same thing.

Try ``nipnost --help`` to see a large number of command-line options.

Install optional data packages for testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For our tests, we have collected a set of fmri imaging data which are
required for the tests to run.  To do this, download the latest example
data and template package files from `NIPY data packages`_. See
:ref:`data-files`.

Running individual tests
^^^^^^^^^^^^^^^^^^^^^^^^

You can also run the tests from the command line with a variety of options.

See above for a description of the ``nipnost`` program.

To test an individual module::

    nipnost test_image.py

To test an individual function::

    nipnost test_module:test_function

To test a class::

    nipnost test_module:TestClass

To test a class method::

   nipnost test_module:TestClass.test_method

Verbose mode (*-v* option) will print out the function names as they
are executed.  Standard output is normally supressed by nose, to see
any print statements you must include the *-s* option.  In order to
get a "full verbose" output, call nose like this::

    nipnost -sv test_module.py

To include doctests in the nose test::

   nipnost -sv --with-doctest test_module.py

For details on all the command line options::

    nipnost --help

.. _coverage:

.. include:: ./coverage_testing.rst

.. include:: ../../links_names.txt
