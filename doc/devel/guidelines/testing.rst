.. _testing:

=========
 Testing
=========

Nipy uses the Numpy_ test framework which is based on nose_.  If you
plan to do much development you should familiarize yourself with nose
and read through the `numpy testing guidelines
<http://projects.scipy.org/scipy/numpy/wiki/TestingGuidelines>`_.

Writing tests
-------------

Test files
^^^^^^^^^^

The numpy testing framework and nipy extensions are imported with one
line in your test module::

     from nipy.testing import *

This imports all the ``assert_*`` functions you need like
``assert_equal``, ``assert_raises``, ``assert_array_almost_equal``
etc..., numpy's ``rand`` function, and the numpy test decorators:
``knownfailure``, ``slow``, ``skipif``, etc...

Please name your test file with the *test_* prefix followed by the
module name it tests.  This makes it obvious for other developers
which modules are tested, where to add tests, etc...  An example test
file and module pairing::

      nipy/core/reference/coordinate_system.py
      nipy/core/reference/tests/test_coordinate_system.py

All tests go in a test subdirectory for each package.

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
though, you need many individual tests to thoroughly cover (100%
coverage) the method/function.  This calls for a `generator function
<http://docs.python.org/tutorial/classes.html#generators>`_.  Use a
``yield`` statement to run each individual test, independent from the
other tests.  This prevents the case where the first test fails and as
a result the following tests don't get run.

This test function executes four independent tests::

    def test_index():
        cs = CoordinateSystem('ijk')
        yield assert_equal, cs.index('i'), 0
        yield assert_equal, cs.index('j'), 1
        yield assert_equal, cs.index('k'), 2
        yield assert_raises, ValueError, cs.index, 'x'


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

For our tests, we have collected a set of fmri imaging data which are
required for the tests to run.  To do this, download the latest example
data and template package files from `NIPY data packages`_. See
:ref:`data-files`.

Running individual tests
^^^^^^^^^^^^^^^^^^^^^^^^

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

For details on all the command line options::

    nosetests --help

.. _coverage:

.. include:: ./coverage_testing.rst

.. include:: ../../links_names.txt
