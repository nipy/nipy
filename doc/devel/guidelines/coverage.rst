.. _coverage:

==========
 Coverage
==========

**NOTE:** We should consider using the *coverage* functionality that
  currently exists in the scipy_ test framework.  These notes may be
  out of date.

Coverage testing is a technique used to see how much of the code is
exercised by the unit tests. It is important to remember that a high
level of coverage is a necessary but not sufficient condition for
having effective tests. Coverage testing can be useful for identifying
whole functions or classes which are not tested, or for finding
certain conditions which are never tested.

The *coverage.py* utility can be used for coverage testing. To install
it, simply download the file from the website and place it somewhere
on your path, ensuring it has execute rights. To run the coverage
testing there are three steps.

Delete the results from any previous runs::

   coverage.py -e

Execute the unit tests. The syntax for doing this exactly the same as
for running any of the unit tests as described [here] except with
``coverage.py -x`` in front of it::

   coverage.py -x ./test --slow

Examine the coverage results::
   
   PYTHON_PATH=lib
   FILTER='tests\|neuroimaging/utils\|sandbox'
   coverage.py -r -m $(find $PYTHON_PATH/neuroimaging -name "*py" | grep -v $FILTER)

.. include:: ../../links_names.txt
