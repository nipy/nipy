"""
Package containing core neuroimaging classes.
"""
import image, reference, api
__all__ = ["image", "reference, api"]
__docformat__ = 'restructuredtext'

def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)
