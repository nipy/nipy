""" Testing
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ..transform import Transform


def test_transform():
    t = Transform(lambda x : x+1)
    pts = np.random.normal(size=(10,3))
    assert_array_equal(t.apply(pts), pts+1)
    pytest.raises(AttributeError, getattr, t, 'param')
    tm1 = Transform(lambda x : x-1)
    assert_array_equal(tm1.apply(pts), pts-1)
    tctm1 = t.compose(tm1)
    assert_array_almost_equal(tctm1.apply(pts), pts)


def test_transform_other_init():
    # Test we can have another init for our transform

    class C(Transform):

        def __init__(self):
            self.func = lambda x : x + 1

    pts = np.random.normal(size=(10,3))
    assert_array_equal(C().apply(pts), pts+1)
