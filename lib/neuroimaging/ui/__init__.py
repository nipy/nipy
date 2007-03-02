"""
Package containing both command-line and graphical user interfaces as well as
visualization tools.
"""
__docformat__ = 'restructuredtext'

def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)
