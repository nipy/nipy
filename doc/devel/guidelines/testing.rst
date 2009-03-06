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

     from neuroimaging.testing import *

This imports all the ``assert_*`` functions you need like
``assert_equal``, ``assert_raises``, ``assert_array_almost_equal``
etc..., numpy's ``rand`` function, and the numpy test decorators:
``knownfailure``, ``slow``, ``skipif``, etc...

Please name your test file with the *test_* prefix followed by the
module name it tests.  This makes it obvious for other developers
which modules are tested, where to add tests, etc...  An example test
file and module pairing::

      neuroimaging/core/reference/coordinate_system.py
      neuroimaging/core/reference/tests/test_coordinate_system.py

All tests go in a test subdirectory for each package.

Temporary files
^^^^^^^^^^^^^^^

If you need to create a temporary file during your testing, you should
use either of these two methods:

#. `StringIO <http://docs.python.org/library/stringio.html>`_ 

   StringIO creates an in memory file-like object. The memory buffer
   is freed when the file is closed.  This is the preferred method for
   temporary files in tests.

#. `tempfile.NamedTemporaryFile <http://docs.python.org/library/tempfile.html>`_

   This will create a temporary file which is deleted when the file is
   closed.  There are parameters for specifying the filename *prefix*
   and *suffix*.

Both of the above libraries are preferred over creating a file in the
test directory and then removing them with a call to ``os.remove``.
For various reasons, sometimes ``os.remove`` doesn't get called and
temp files get left around.


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

The coverage report will cover any python source module imported after
the start of the test.  This can be noisy and difficult to focus on
the specific module for which you are writing nosetests.  To focus the
coverage report, you can provide nose with the specific package you
would like output from using the ``--cover-package``.  For example, in
writing tests for the coordinate_map module::

    nosetests --with-coverage --cover-package=neuroimaging.core.reference.coordinate_map test_coordinate_map.py


For details on all the command line options::

    nosetests --help


.. include:: ../../links_names.txt
