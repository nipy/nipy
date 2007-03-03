"""
Package containing modality-specific classes.
"""
__docformat__ = 'restructuredtext'

def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)
