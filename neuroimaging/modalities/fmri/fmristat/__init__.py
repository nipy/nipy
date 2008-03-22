"""
This module is meant to reproduce the GLM analysis of fmristat.

    Liao et al. (2002).
TODO fix reference here

"""
__docformat__ = 'restructuredtext'

def test(level=1, verbosity=1, flags=[]):
    from neuroimaging.utils.testutils import set_flags
    set_flags(flags)
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)
