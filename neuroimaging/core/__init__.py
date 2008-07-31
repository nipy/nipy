"""
Package containing core neuroimaging classes.
"""
__docformat__ = 'restructuredtext'


def test(level=1, verbosity=1, flags=[]):
    from neuroimaging.utils.testutils import set_flags
    set_flags(flags)
    from neuroimaging.testing import *
    return NumpyTest().test(level, verbosity)
