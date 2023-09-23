
Coverage Testing
----------------

Coverage testing is a technique used to see how much of the code is
exercised by the unit tests. It is important to remember that a high
level of coverage is a necessary but not sufficient condition for
having effective tests. Coverage testing can be useful for identifying
whole functions or classes which are not tested, or for finding
certain conditions which are never tested.

This is an excellent task for nose_ - the automated test runner we are
using.  Nose can run the `python coverage tester`_.  First make sure
you have the coverage tester installed on your system.  Download the
tarball from the link, extract and install ``python setup.py
install``. Or on Ubuntu you can install from apt-get: ``sudo apt-get
install python-coverage``.

Run nose with coverage testing arguments::

   nosetests -sv --with-coverage path_to_code

For example, this command::

    nosetests -sv --with-coverage test_coordinate_map.py

will report the following::

 Name                                            Stmts   Exec  Cover   Missing
 -----------------------------------------------------------------------------
 nipy                                       21     14    66%   70-74, 88-89
 nipy.core                                   4      4   100%
 nipy.core.reference                         8      8   100%
 nipy.core.reference.array_coords          100     90    90%   133-134, 148-151, 220, 222, 235, 242
 nipy.core.reference.coordinate_map        188    187    99%   738
 nipy.core.reference.coordinate_system      61     61   100%
 nipy.core.reference.slices                 34     34   100%
 nipy.core.transforms                        0      0   100%
 nipy.core.transforms.affines               14     14   100%


The coverage report will cover any python source module imported after
the start of the test.  This can be noisy and difficult to focus on
the specific module for which you are writing nosetests.  For
instance, the above report also included coverage of most of
``numpy``.  To focus the coverage report, you can provide nose with
the specific package you would like output from using the
``--cover-package``.  For example, in writing tests for the
coordinate_map module::

    nosetests --with-coverage --cover-package=nipy.core.reference.coordinate_map test_coordinate_map.py

.. include:: ../../links_names.txt
