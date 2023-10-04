""" Testing
"""

import numpy as np
from numpy.testing import assert_array_almost_equal

from .._registration import _cspline_sample1d, _cspline_sample4d, _cspline_transform


def test_sample1d():
    a = np.random.rand(100)
    c = _cspline_transform(a)
    x = np.arange(100)
    b = np.zeros(100)
    b = _cspline_sample1d(b, c, x)
    assert_array_almost_equal(a, b)
    b = _cspline_sample1d(b, c, x, mode='nearest')
    assert_array_almost_equal(a, b)


def test_sample4d():
    a = np.random.rand(4, 5, 6, 7)
    c = _cspline_transform(a)
    x = np.mgrid[0:4, 0:5, 0:6, 0:7]
    b = np.zeros(a.shape)
    args = list(x)
    b = _cspline_sample4d(b, c, *args)
    assert_array_almost_equal(a, b)
    args = list(x) + ['nearest' for i in range(4)]
    b = _cspline_sample4d(b, c, *args)
    assert_array_almost_equal(a, b)
