
Coverage Testing
----------------

Coverage testing is a technique used to see how much of the code is
exercised by the unit tests. It is important to remember that a high
level of coverage is a necessary but not sufficient condition for
having effective tests. Coverage testing can be useful for identifying
whole functions or classes which are not tested, or for finding
certain conditions which are never tested.

This is an excellent task for pytest_ - the automated test runner we are
using.  Pytest can run the `python coverage tester`_.  First make sure
you have the coverage test plugin installed on your system::

    pip install pytest-cov

Run Pytest with coverage testing arguments::

    pytest --cov=nipy --doctest-plus nipy

The coverage report will cover any python source module imported after
the start of the test.  This can be noisy and difficult to focus on
the specific module for which you are writing tests.  For
instance, the default report also includes coverage of most of
``numpy``.  To focus the coverage report, you can provide Pytest with
the specific package you would like output from using the
``--cov=nipy`` (the option above).

.. include:: ../../links_names.txt
