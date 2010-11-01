""" Testing
"""

import numpy as np

from ..transform import Transform

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_equal, assert_raises

def test_transform():
    t = Transform(lambda x : x+1)
    pts = np.random.normal(size=(10,3))
    assert_array_equal(t.apply(pts), pts+1)
    assert_raises(AttributeError, getattr, t, 'param')
    tm1 = Transform(lambda x : x-1)
    assert_array_equal(tm1.apply(pts), pts-1)
    tctm1 = t.compose(tm1)
    assert_array_almost_equal(tctm1.apply(pts), pts)
