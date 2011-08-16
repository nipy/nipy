""" Custom doctester based on Numpy doctester
"""
import re
from doctest import register_optionflag
import numpy as np

from ..fixes.numpy.testing.noseclasses import NumpyDoctest, NumpyOutputChecker

IGNORE_OUTPUT = register_optionflag('IGNORE_OUTPUT')
SYMPY_EQUAL = register_optionflag('SYMPY_EQUAL')
NOT_EQUAL = register_optionflag('NOT_EQUAL')
FP_4DP_EQUAL =  register_optionflag('FP_4DP_EQUAL')
FP_6DP_EQUAL =  register_optionflag('FP_6DP_EQUAL')

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


class NipyOutputChecker(NumpyOutputChecker):
    def check_output(self, want, got, optionflags):
        if IGNORE_OUTPUT & optionflags:
            return True
        # When writing tests we sometimes want to assure ourselves that the
        # results are _not_ equal
        wanted_tf = not (NOT_EQUAL & optionflags)
        # Are the strings equal when run through sympy?
        if SYMPY_EQUAL & optionflags:
            from sympy import sympify
            res = sympify(want) == sympify(got)
            return res == wanted_tf
        # If testing floating point, round to required number of digits
        if optionflags & (FP_4DP_EQUAL | FP_6DP_EQUAL):
            if optionflags & FP_4DP_EQUAL:
                dp = 4
            elif optionflags & FP_6DP_EQUAL:
                dp = 6
            want = round_numbers(want, dp)
            got = round_numbers(got, dp)
        # Pass tests through two-pass numpy checker
        res = NumpyOutputChecker.check_output(self, want, got, optionflags)
        # Return True if we wanted True and got True, or if we wanted False and
        # got False
        return res == wanted_tf


class NipyDoctest(NumpyDoctest):
    name = 'nipydoctest'   # call nosetests with --with-nipydoctest
    out_check_class = NipyOutputChecker

    def set_test_context(self, test):
        # set namespace for tests
        test.globs['np'] = np
