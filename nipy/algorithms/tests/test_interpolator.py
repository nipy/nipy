""" Testing interpolation module
"""

import numpy as np

from nipy.core.api import Image, vox2mni

from ..interpolation import ImageInterpolator

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


def test_interp_obj():
    arr = np.arange(24).reshape((2, 3, 4))
    coordmap =  vox2mni(np.eye(4))
    img = Image(arr, coordmap)
    interp = ImageInterpolator(img)
    assert_equal(interp.mode, 'constant')
    assert_equal(interp.order, 3)
    # order is read-only
    assert_raises(AttributeError,
                  setattr,
                  interp,
                  'order',
                  1)
    interp = ImageInterpolator(img, mode='nearest')
    assert_equal(interp.mode, 'nearest')
    # mode is read-only
    assert_raises(AttributeError,
                  setattr,
                  interp,
                  'mode',
                  'reflect')


def test_interpolator():
    arr = np.arange(24).reshape((2, 3, 4))
    coordmap =  vox2mni(np.eye(4))
    img = Image(arr, coordmap)
    isx = np.indices(arr.shape)
    for order in range(5):
        interp = ImageInterpolator(img, mode='nearest', order=order)
        # Interpolate at existing points.
        assert_almost_equal(interp.evaluate(isx), arr)
        # Interpolate off top right corner with different modes
        assert_almost_equal(interp.evaluate([0, 0, 4]), arr[0, 0, -1])
        interp = ImageInterpolator(img, mode='constant', order=order, cval=0)
        assert_array_equal(interp.evaluate([0, 0, 4]), 0)
        interp = ImageInterpolator(img, mode='constant', order=order, cval=1)
        assert_array_equal(interp.evaluate([0, 0, 4]), 1)
