#!/usr/bin/env python

import numpy as np

from .._quantile import _quantile, _median
from numpy.testing import assert_array_equal

def test_median():
    X = np.array([np.arange(11) for i in range(20)])
    assert_array_equal(_median(X, axis=1).squeeze(),
                       5 * np.ones(20))
    assert_array_equal(_median(X.T.astype('double')).squeeze(),
                       5 * np.ones(20))

def test_quantile():
    X = np.array([np.arange(1, 10) for i in range(20)])
    assert_array_equal(_quantile(X, axis=1, ratio=.8).squeeze(),
                       9 * np.ones(20))
    assert_array_equal(_quantile(X.T.astype('double'), ratio=.8).squeeze(),
                       9 * np.ones(20))


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
