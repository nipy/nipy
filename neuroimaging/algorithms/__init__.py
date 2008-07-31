"""
Package containing generic algorithms such as registration, statistics,
simulation, etc.
"""
__docformat__ = 'restructuredtext'

import statistics
import fwhm, interpolation, kernel_smooth, onesample, regression


def test(level=1, verbosity=1, flags=[]):
    from neuroimaging.utils.testutils import set_flags
    set_flags(flags)
    from neuroimaging.testing import *
    return NumpyTest().test(level, verbosity)
