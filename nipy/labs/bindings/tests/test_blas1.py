
#
# Test BLAS 1
#

import numpy as np
from numpy.testing import assert_almost_equal

from .. import blas_dasum, blas_daxpy, blas_ddot, blas_dnrm2, blas_dscal

n = 15


def test_dnrm2():
    x = np.random.rand(n)
    assert_almost_equal(np.sqrt(np.sum(x**2)), blas_dnrm2(x))

def test_dasum():
    x = np.random.rand(n)
    assert_almost_equal(np.sum(np.abs(x)), blas_dasum(x))

def test_ddot():
    x = np.random.rand(n)
    y = np.random.rand(n)
    assert_almost_equal(np.dot(x,y), blas_ddot(x, y))

def test_daxpy():
    x = np.random.rand(n)
    y = np.random.rand(n)
    alpha = np.random.rand()
    assert_almost_equal(alpha*x+y, blas_daxpy(alpha, x, y))

def test_dscal():
    x = np.random.rand(n)
    alpha = np.random.rand()
    assert_almost_equal(alpha*x, blas_dscal(alpha, x))
