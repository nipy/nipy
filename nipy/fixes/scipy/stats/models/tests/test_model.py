""" Testing models module
"""

import numpy as np

# In fact we're testing methods defined in model
from ..regression import OLSModel

from nose.tools import assert_true, assert_equal, assert_raises

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

N = 10
X = np.c_[np.linspace(-1,1,N), np.ones((N,))]
Y = np.random.normal(size=(10,1)) * 10 + 100
MODEL = OLSModel(X)
C1 = [1, 0]

def test_t_output():
    # Test t contrast output
    results = MODEL.fit(Y)
    # Check we fit the mean
    assert_array_almost_equal(results.theta[1], np.mean(Y))
    # Check we get required outputs
    exp_t = results.t(0)
    exp_effect = results.theta[0]
    exp_sd = exp_effect / exp_t
    res = results.Tcontrast(C1)
    assert_array_almost_equal(res.t, exp_t)
    assert_array_almost_equal(res.effect, exp_effect)
    assert_array_almost_equal(res.sd, exp_sd)
    res = results.Tcontrast(C1, store=('effect',))
    assert_equal(res.t, None)
    assert_array_almost_equal(res.effect, exp_effect)
    assert_equal(res.sd, None)
    res = results.Tcontrast(C1, store=('t',))
    assert_array_almost_equal(res.t, exp_t)
    assert_equal(res.effect, None)
    assert_equal(res.sd, None)
    res = results.Tcontrast(C1, store=('sd',))
    assert_equal(res.t, None)
    assert_equal(res.effect, None)
    assert_array_almost_equal(res.sd, exp_sd)
    res = results.Tcontrast(C1, store=('effect', 'sd'))
    assert_equal(res.t, None)
    assert_array_almost_equal(res.effect, exp_effect)
    assert_array_almost_equal(res.sd, exp_sd)

