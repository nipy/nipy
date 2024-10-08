""" Test quartile functions
"""

from itertools import chain

import numpy as np
from numpy import median as np_median
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.stats import scoreatpercentile as sp_percentile

from nipy.utils import SCTYPES

from .._quantile import _median, _quantile

NUMERIC_TYPES = list(
    chain.from_iterable(
        SCTYPES[t] for t in ("int", "uint", "float", "complex")
    )
)


def another_percentile(arr, pct, axis):
    # numpy.percentile not available until after numpy 1.4.1
    return np.apply_along_axis(sp_percentile, axis, arr.astype(float), pct)


def test_median():
    for dtype in NUMERIC_TYPES:
        for shape in ((10,), (10, 11), (10, 11, 12)):
            X = (100 * (np.random.random(shape) - .5)).astype(dtype)
            for a in range(X.ndim):
                assert_array_equal(_median(X, axis=a).squeeze(),
                                   np_median(X.astype(np.float64), axis=a))


def test_quantile():
    for dtype in NUMERIC_TYPES:
        for shape in ((10,), (10, 11), (10, 11, 12)):
            X = (100 * (np.random.random(shape) - .5)).astype(dtype)
            for a in range(X.ndim):
                assert_array_almost_equal(
                    _quantile(X, .75, axis=a, interp=True).squeeze(),
                    another_percentile(X, 75, axis=a))
