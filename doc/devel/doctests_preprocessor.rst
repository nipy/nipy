####################
Doctest preprocessor
####################

We oftentimes have output from doctests that we don't want to test explicitly
against the output string.

An obvious example is decimal precision.  For example::

    >>> import numpy as np
    >>> np.sqrt(2)
    1.4142135623730951

where the last few digits depend on the CPU and math libraries on the particular
platform.  Another example are sympy tests, because the order of symbols in an
expression can be difficult to predict::

    >>> from sympy import symbols, cos
    >>> a, b = symbols('a, b')
    >>> cos(b) + a
    a + cos(b)
    >>> c, d = symbols('c, d')
    >>> cos(c) + d
    d + cos(c)

It looks like the order is predictable for particular versions of sympy, but not
across versions.

A third is the classic of output dtype specific to byte ordering::

    >>> np.zeros((1,), dtype=[('f1', 'i')])

    array([(0,)], 
    dtype=[('f1', '<i4')])

*******
Options
*******

Standard doctest option flags
=============================

See: http://docs.python.org/library/doctest.html

Specifically::

    >>> import numpy as np
    >>> np.sqrt(2) #doctest +ELLIPSIS
    1.414213562373...

and::

    >>> np.zeros((1,), dtype=[('f1', 'i')])

    array([(0,)], 
    dtype=[('f1', '...i4')])

So - ugly - because the reader can't easily guess what you've elided, and so the
examples are hard to read.

And, it can't easily deal with the sympy case above.

In the cases below, I found the doctest machinery by looking for
``OutputChecker`` or ``DoctestParser``.

Numpy doctest machinery
=======================

We clearly depend on Numpy.  The doctest machinery is in
``numpy/testing/noseclasses.py``.

Around line 232 in my version, we see::

    test.globs = {'__builtins__':__builtins__,
                    '__file__':'__main__',
                    '__name__':'__main__',
                    'np':numpy}

and following.  Here the normal module execution context gets overwritten by
this minimal namespace, and hence, if we used this stuff directly, we'd have to
import a load of module level stuff in every doctest - which looks messy and
makes the examples harder to read.

Notice around line 144, there is the ``NumpyOutputChecker`` - there is a hackish
check for the string ``#random`` that appears to result in the check being
short-cut as passed, and attempts to deal with the byte order and 'i4' 'i8'
default integer for 32 and 64 bit.

The whole is rather difficult to work with because the class names are
hard-coded into the various method calls.

In summary, ``numpy.testing.nosetester.NoseTester.test`` initializes then calls
``NumpyTestProgram``.  ``NumpyTestProgram`` pulls the ``doctest`` nose plugin
out of the plugin list.

Meanwhile, in ``numpy.testing.nosetester.NoseTester.prepare_test_args`` (called
from the ``test`` method), ``--with-doctest`` becomes ``--with-numpydoctest``,
and the method stuffs the ``NumpyDoctest`` and ``KnownFailure`` plugins into the
list of plugins.

So, overriding this stuff means subclassing or otherwise replacing
``NumpyDoctest`` with our own doctest plugin.

Options are - rewrite for ourselves, or generalize numpy's machinery, propose
pull, and meanwhile use the rewrite for our own purposes.

Sympy doctest machinery
=======================

We depend on sympy, but not as fundamentally as we depend on numpy.

Sympy's test machinery is in ``sympy/utilities/runtest.py``.

They don't use nose_.

Sympy also clears the context so all names have to be imported specifically in
the doctests.  However, names imported in one doctest are available in the
others.

They initialize printing with::

    def setup_pprint():
        from sympy import pprint_use_unicode, init_printing

        # force pprint to be in ascii mode in doctests
        pprint_use_unicode(False)

        # hook our nice, hash-stable strprinter
        init_printing(pretty_print=False)

in that same file.  Quick tests suggested that the usual result of this is to
output strings via ``sympy.printing.sstrrepr``.  This doesn't seem to affect the
order of symbol output so doesn't solve our problem above.

IPython doctest machinery
=========================

``IPython/testing/plugin/ipdoctest.py``

This looks very similar to the numpy machinery.  Again, it's a nose plugin that
inherits from the nose ``Doctest`` class.

.. include:: ../links_names.txt


