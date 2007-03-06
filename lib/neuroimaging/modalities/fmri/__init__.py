"""
TODO
"""

__docformat__ = 'restructuredtext'

import filters, fmri, functions, hrf, pca, protocol, regression, utils
import fmristat

def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)
