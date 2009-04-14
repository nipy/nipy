#!/usr/bin/env python

#
# Test BLAS 1
#

from numpy.testing import assert_almost_equal, assert_equal
import numpy as np
import neuroimaging.neurospin.bindings as fb

n = 15


def test_dnrm2():
    x = np.random.rand(n)
    assert_almost_equal(np.sqrt(np.sum(x**2)), fb.blas_dnrm2(x))

def test_dasum():
    x = np.random.rand(n)
    assert_almost_equal(np.sum(np.abs(x)), fb.blas_dasum(x))

def test_ddot():
    x = np.random.rand(n)
    y = np.random.rand(n)
    assert_almost_equal(np.dot(x,y), fb.blas_ddot(x, y))

def test_daxpy():
    x = np.random.rand(n)
    y = np.random.rand(n)
    alpha = np.random.rand()
    assert_almost_equal(alpha*x+y, fb.blas_daxpy(alpha, x, y))

def test_dscal():
    x = np.random.rand(n)
    alpha = np.random.rand()
    assert_almost_equal(alpha*x, fb.blas_dscal(alpha, x))


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])

