# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests for the nose-like unittest support."""

from nipy.testing.lightunit import as_unittest, ParametricTestCase, parametric

@as_unittest
def trivial():
    """A trivial test"""
    pass

# Some examples of parametric tests.

def is_smaller(i,j):
    assert i<j,"%s !< %s" % (i,j)

class Tester(ParametricTestCase):

    def test_parametric(self):
        yield is_smaller(3, 4)
        x, y = 1, 2
        yield is_smaller(x, y)

@parametric
def test_par_standalone():
    yield is_smaller(3, 4)
    x, y = 1, 2
    yield is_smaller(x, y)


#-----------------------------------------------------------------------------
# Broken tests - these are useful for debugging and understanding how certain
# things work.
#-----------------------------------------------------------------------------

@as_unittest
def broken():
    x, y = 1, 0  # broken
    x, y = 1, 1  # ok
    x/y

def test_par_nose():
    yield (is_smaller,3, 4)
    x, y = 2, 2  # broken
    x, y = 2, 3  # ok
    yield (is_smaller,x, y)


if __name__ == '__main__':
    import unittest
    unittest.main()
