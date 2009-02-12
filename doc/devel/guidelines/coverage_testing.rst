.. _coverage:

==================
 Coverage Testing
==================

Coverage testing is a technique used to see how much of the code is
exercised by the unit tests. It is important to remember that a high
level of coverage is a necessary but not sufficient condition for
having effective tests. Coverage testing can be useful for identifying
whole functions or classes which are not tested, or for finding
certain conditions which are never tested.

This is an excellent task for nose_ - the automated test runner we are
using.  Nose can run the `python coverage tester`_.  First make sure
you have the coverage tester installed on your system.  For me this
was ``sudo apt-get install python-coverage``.  Next, run nose with
coverage testing::

   nosetests -sv --with-coverage path_to_code

Running this on some code I had to hand, ``binaryformats`` gave output
like::

  Name                                      Stmts   Exec  Cover   Missing
  -----------------------------------------------------------------------
  binaryformats                                 0      0   100%   
  binaryformats.endiancodes                    18     14    77%   56, 58, 60, 66
  binaryformats.headers                        21     21   100%   
  ...


.. include:: ../../links_names.txt
