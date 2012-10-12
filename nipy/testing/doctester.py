""" Custom doctester based on Numpy doctester

To run doctests via nose, you'll need ``nosetests nipy/testing/doctester.py
--doctest-test``, because this file will be identified as containing tests.
"""
import re
from doctest import register_optionflag
import numpy as np

from ..fixes.numpy.testing.noseclasses import (NumpyDoctest,
                                               NumpyOutputChecker,
                                               NumpyDocTestFinder)

IGNORE_OUTPUT = register_optionflag('IGNORE_OUTPUT')
SYMPY_EQUAL = register_optionflag('SYMPY_EQUAL')
STRIP_ARRAY_REPR = register_optionflag('STRIP_ARRAY_REPR')
IGNORE_DTYPE = register_optionflag('IGNORE_DTYPE')
NOT_EQUAL = register_optionflag('NOT_EQUAL')
FP_4DP =  register_optionflag('FP_4DP')
FP_6DP =  register_optionflag('FP_6DP')

FP_REG = re.compile(r'(?<![0-9a-zA-Z_.])'
                    r'(\d+\.\d+)'
                    r'(e[+-]?\d+)?'
                    r'(?![0-9a-zA-Z_.])')

def round_numbers(in_str, precision):
    """ Replace fp numbers in `in_str` with numbers rounded to `precision`

    Parameters
    ----------
    in_str : str
        string possibly containing floating point numbers
    precision : int
        number of decimal places to round to

    Returns
    -------
    out_str : str
        `in_str` with any floating point numbers replaced with same numbers
        rounded to `precision` decimal places.

    Examples
    --------
    >>> round_numbers('A=0.234, B=12.345', 2)
    'A=0.23, B=12.35'

    Rounds the floating point value as it finds it in the string.  This is even
    true for numbers with exponentials.  Remember that:

    >>> '%.3f' % 0.3339e-10
    '0.000'

    This routine will recognize an exponential as something to process, but only
    works on the decimal part (leaving the exponential part is it is):

    >>> round_numbers('(0.3339e-10, "string")', 3)
    '(0.334e-10, "string")'
    """
    fmt = '%%.%df' % precision
    def dorep(match):
        gs = match.groups()
        res = fmt % float(gs[0])
        if not gs[1] is None:
            res+=gs[1]
        return res
    return FP_REG.sub(dorep, in_str)

ARRAY_REG = re.compile('^\s*array\((.*)\)\s*$', re.DOTALL)
DTYPE_REG = re.compile('\s*,\s+dtype=.*', re.DOTALL)

def strip_array_repr(in_str):
    """ Removes array-specific part of repr from string `in_str`

    This parser only works on lines that contain *only* an array repr (and
    therefore start with ``array``, and end with a close parenthesis.  To remove
    dtypes in array reprs that may be somewhere within the line, use the
    ``IGNORE_DTYPE`` doctest option.

    Parameters
    ----------
    in_str : str
        String maybe containing a repr for an array

    Returns
    -------
    out_str : str
        String from which the array specific parts of the repr have been
        removed.

    Examples
    --------
    >>> arr = np.arange(5, dtype='i2')

    Here's the normal repr:

    >>> arr
    array([0, 1, 2, 3, 4], dtype=int16)

    The repr with the 'array' bits removed:

    >>> strip_array_repr(repr(arr))
    '[0, 1, 2, 3, 4]'
    """
    arr_match = ARRAY_REG.match(in_str)
    if arr_match is None:
        return in_str
    out_str = arr_match.groups()[0]
    return DTYPE_REG.sub('', out_str)

IGNORE_DTYPE_REG = re.compile(',\s+dtype=.*?(?=\))', re.DOTALL)

def ignore_dtype(in_str):
    """ Removes dtype=[dtype] from string `in_str`

    Parameters
    ----------
    in_str : str
        String maybe containing dtype specifier

    Returns
    -------
    out_str : str
        String from which the dtype specifier has been removed.

    Examples
    --------
    >>> arr = np.arange(5, dtype='i2')

    Here's the normal repr:

    >>> arr
    array([0, 1, 2, 3, 4], dtype=int16)

    The repr with the dtype bits removed

    >>> ignore_dtype(repr(arr))
    'array([0, 1, 2, 3, 4])'
    >>> ignore_dtype('something(again, dtype=something)')
    'something(again)'

    Even if there are more closed brackets after the dtype

    >>> ignore_dtype('something(again, dtype=something) (1, 2)')
    'something(again) (1, 2)'

    We need the close brackets to match

    >>> ignore_dtype('again, dtype=something')
    'again, dtype=something'
    """
    return IGNORE_DTYPE_REG.sub('', in_str)


class NipyOutputChecker(NumpyOutputChecker):
    def check_output(self, want, got, optionflags):
        if IGNORE_OUTPUT & optionflags:
            return True
        # When writing tests we sometimes want to assure ourselves that the
        # results are _not_ equal
        wanted_tf = not (NOT_EQUAL & optionflags)
        # Strip dtype
        if IGNORE_DTYPE & optionflags:
            want = ignore_dtype(want)
            got = ignore_dtype(got)
        # Strip array repr from got and want if requested
        if STRIP_ARRAY_REPR & optionflags:
            # STRIP_ARRAY_REPR only matches for a line containing *only* an
            # array repr.  Use IGNORE_DTYPE to ignore a dtype specifier embedded
            # within a more complex line.
            want = strip_array_repr(want)
            got = strip_array_repr(got)
        # If testing floating point, round to required number of digits
        if optionflags & (FP_4DP | FP_6DP):
            if optionflags & FP_4DP:
                dp = 4
            elif optionflags & FP_6DP:
                dp = 6
            want = round_numbers(want, dp)
            got = round_numbers(got, dp)
        # Are the strings equal when run through sympy?
        if SYMPY_EQUAL & optionflags:
            from sympy import sympify
            res = sympify(want) == sympify(got)
            return res == wanted_tf
        # Pass tests through two-pass numpy checker
        res = NumpyOutputChecker.check_output(self, want, got, optionflags)
        # Return True if we wanted True and got True, or if we wanted False and
        # got False
        return res == wanted_tf


class DocTestSkip(object):
    """Object wrapper for doctests to be skipped."""

    def __init__(self,obj):
        self.obj = obj

    def __getattribute__(self,key):
        if key == '__doc__':
            return None
        return getattr(object.__getattribute__(self, 'obj'), key)


class NipyDocTestFinder(NumpyDocTestFinder):

    def _find(self, tests, obj, name, module, source_lines, globs, seen):
        """
        Find tests for the given object and any contained objects, and
        add them to `tests`.

        Add ability to skip doctest
        """
        if hasattr(obj, "skip_doctest") and obj.skip_doctest:
            obj = DocTestSkip(obj)
        NumpyDocTestFinder._find(self,tests, obj, name, module,
                                 source_lines, globs, seen)


class NipyDoctest(NumpyDoctest):
    name = 'nipydoctest'   # call nosetests with --with-nipydoctest
    out_check_class = NipyOutputChecker
    test_finder_class = NipyDocTestFinder

    def set_test_context(self, test):
        # set namespace for tests
        test.globs['np'] = np
