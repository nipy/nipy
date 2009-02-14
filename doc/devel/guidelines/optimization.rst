.. _optimization:

==============
 Optimization
==============

In the early stages of NIPY development, we are focusing on
functionality and usability.  In regards to optimization, we benefit
**significantly** from the optimized routines in scipy_ and numpy_.
As NIPY progresses it is likely we will spend more energy on
optimizing critical functions.  In our `py4science group at UC
Berkeley <https://cirl.berkeley.edu/view/Py4Science/WebHome>`_ we've
had several meetings on the various optimization options including
ctypes, weave and blitz, and cython.  It's clear there are many good
options, including standard C-extensions.  However, optimized code
tends to be less readable and more difficult to debug and maintain.
When we do optimize our code we will first profile the code to
determine the offending sections, then optimize those sections. Until
that need arises, we will follow the great advice from these fellow
programmers:


Kent Beck:
  "First make it work.  Then make it right.  Then make it fast."

`Donald Knuth on optimization
<http://en.wikipedia.org/wiki/Optimization_(computer_science)#When_to_optimize>`_:

  "We should forget about small efficiencies, say about 97% of the
  time: premature optimization is the root of all evil."


Tim Hochberg, from the Numpy list::

    0. Think about your algorithm.
    1. Vectorize your inner loop.
    2. Eliminate temporaries
    3. Ask for help
    4. Recode in C.
    5. Accept that your code will never be fast.
   
    Step zero should probably be repeated after every other step ;)


.. include:: ../../links_names.txt
