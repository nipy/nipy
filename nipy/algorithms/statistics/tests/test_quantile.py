#!/usr/bin/env python

import numpy as np

from .._quantile import _quantile, _median
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal)
from numpy import median as np_median
from scipy import percentile as sp_percentile


def _test_median(dtype, shape):
    X = (100 * (np.random.random(shape) - .5)).astype(dtype)
    for a in range(X.ndim):
        assert_array_equal(_median(X, axis=a).squeeze(),
                           np_median(X, axis=a))
    

def _test_quantile(dtype, shape):
    X = (100 * (np.random.random(shape) - .5)).astype(dtype)
    for a in range(X.ndim):
        assert_array_almost_equal(\
            _quantile(X, .75, axis=a, interp=True).squeeze(),
            sp_percentile(X, 75, axis=a))


def test_median_2d_int32():
    _test_median('int32', (10, 11))


def test_median_3d_int32():
    _test_median('int32', (10, 11, 12))


def test_median_2d_uint8():
    _test_median('uint8', (10, 11))


def test_median_3d_uint8():
    _test_median('uint8', (10, 11, 12))


def test_median_2d_double():
    _test_median('double', (10, 11))


def test_median_3d_double():
    _test_median('double', (10, 11, 12))


def test_quantile_2d_int32():
    _test_quantile('int32', (10, 11))


def test_quantile_3d_int32():
    _test_quantile('int32', (10, 11, 12))


def test_quantile_2d_uint8():
    _test_quantile('uint8', (10, 11))


def test_quantile_3d_uint8():
    _test_quantile('uint8', (10, 11, 12))


def test_quantile_2d_double():
    _test_quantile('double', (10, 11))


def test_quantile_3d_double():
    _test_quantile('double', (10, 11, 12))


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
