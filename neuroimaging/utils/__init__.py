"""
Package containing both generic configuration and testing stuff as well as
general purpose functions that are useful to a broader community and not
restricted to the neuroimaging community. This package may contain
third-party software included here for convenience.
"""

def test(level=1, verbosity=1, flags=[]):
    from neuroimaging.utils.testutils import set_flags
    set_flags(flags)
    from neuroimaging.testing import *
    return NumpyTest().test(level, verbosity)
