""" Tests for regression module
"""

import numpy as np

from nipy.fixes.scipy.stats.models.regression import OLSModel

from ..regression import output_T

from nose.tools import assert_true, assert_equal, assert_raises

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)


N = 10
X = np.c_[np.linspace(-1,1,N), np.ones((N,))]
Y = np.random.normal(size=(10,1)) * 10 + 100
MODEL = OLSModel(X)
C1 = [1, 0]

def test_output_T():
    # Test output_T convenience function
    results = MODEL.fit(Y)
    # Check we fit the mean
    assert_array_almost_equal(results.theta[1], np.mean(Y))
    # Check we get required outputs
    t = results.t(0)
    assert_array_almost_equal([t], output_T(C1, results, t=True))


