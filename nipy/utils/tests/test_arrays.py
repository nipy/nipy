""" Testing arrays module
"""

import numpy as np
from nose.tools import assert_equal, assert_raises, assert_true
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ..arrays import strides_from


def test_strides_from():
    for shape in ((3,), (2,3), (2,3,4), (5,4,3,2)):
        for order in 'FC':
            for dtype in sum(np.sctypes.values(), []):
                if dtype is bytes:
                    dtype = 'S3'
                elif dtype is str:
                    dtype = 'U4'
                elif dtype is np.void:
                    continue
                exp = np.empty(shape, dtype=dtype, order=order).strides
                assert_equal(strides_from(shape, dtype, order), exp)
            assert_raises(ValueError, strides_from, shape, np.void, order)
            assert_raises(ValueError, strides_from, shape, bytes, order)
            assert_raises(ValueError, strides_from, shape, str, order)
    assert_raises(ValueError, strides_from, (3,2), 'f8', 'G')
