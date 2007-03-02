"""
Package containing generic algorithms such as registration, statistics,
simulation, etc.
"""
__docformat__ = 'restructuredtext'

import statistics
import fwhm, interpolation, kernel_smooth, onesample, regression

def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)
