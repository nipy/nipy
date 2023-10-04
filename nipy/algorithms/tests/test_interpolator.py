""" Testing interpolation module
"""

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from scipy.ndimage import map_coordinates

from nipy.core.api import Image, vox2mni

from ..interpolation import ImageInterpolator


def test_interp_obj():
    arr = np.arange(24).reshape((2, 3, 4))
    coordmap =  vox2mni(np.eye(4))
    img = Image(arr, coordmap)
    interp = ImageInterpolator(img)
    assert interp.mode == 'constant'
    assert interp.order == 3
    # order is read-only
    pytest.raises(AttributeError,
                  setattr,
                  interp,
                  'order',
                  1)
    interp = ImageInterpolator(img, mode='nearest')
    assert interp.mode == 'nearest'
    # mode is read-only
    pytest.raises(AttributeError,
                  setattr,
                  interp,
                  'mode',
                  'reflect')


def test_interpolator():
    shape = (2, 3, 4)
    arr = np.arange(24).reshape(shape)
    coordmap =  vox2mni(np.eye(4))
    img = Image(arr, coordmap)
    ixs = np.indices(arr.shape).astype(float)
    for order in range(5):
        interp = ImageInterpolator(img, mode='nearest', order=order)
        # Interpolate at existing points.
        assert_almost_equal(interp.evaluate(ixs), arr)
        # Interpolate at half voxel shift
        ixs_x_shift = ixs.copy()
        # Interpolate inside and outside at knots
        ixs_x_shift[0] += 1
        res = interp.evaluate(ixs_x_shift)
        assert_almost_equal(res, np.tile(arr[1], (2, 1, 1)))
        ixs_x_shift[0] -= 2
        res = interp.evaluate(ixs_x_shift)
        assert_almost_equal(res, np.tile(arr[0], (2, 1, 1)))
        # Interpolate at mid-points inside and outside
        ixs_x_shift[0] += 0.5
        res = interp.evaluate(ixs_x_shift)
        # Check inside.
        mid_arr = np.mean(arr, axis=0) if order > 0 else arr[1]
        assert_almost_equal(res[1], mid_arr)
        # Interpolate off top right corner with different modes
        assert_almost_equal(interp.evaluate([0, 0, 4]), arr[0, 0, -1])
        interp = ImageInterpolator(img, mode='constant', order=order, cval=0)
        assert_array_equal(interp.evaluate([0, 0, 4]), 0)
        interp = ImageInterpolator(img, mode='constant', order=order, cval=1)
        assert_array_equal(interp.evaluate([0, 0, 4]), 1)
        # Check against direct ndimage interpolation
        # Need floating point input array to replicate
        # our floating point backing store.
        farr = arr.astype(float)
        for offset, axis, mode in product(np.linspace(-2, 2, 15),
                                          range(3),
                                          ('nearest', 'constant')):
            interp = ImageInterpolator(img, mode=mode, order=order)
            coords = ixs.copy()
            slicer = tuple(None if i == axis else 0 for i in range(3))
            coords[slicer] = coords[slicer] + offset
            actual = interp.evaluate(coords)
            expected = map_coordinates(farr, coords, mode=mode, order=order)
            assert_almost_equal(actual, expected)
            del interp
