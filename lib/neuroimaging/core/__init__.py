"""
Package containing core neuroimaging classes.
"""
__docformat__ = 'restructuredtext'

#import image, reference, api
__all__ = ["image", "reference", "api"]


def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)
