""" Testing interpolation module
"""

import numpy as np

from nipy.core.api import Image, vox2mni

from ..interpolation import ImageInterpolator

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


def test_interpolator():
    arr = np.arange(24).reshape((2, 3, 4))
    coordmap =  vox2mni(np.eye(4))
    img = Image(arr, coordmap)
    # Interpolate off top right corner with different modes
    interp = ImageInterpolator(img, mode='nearest')
    assert_almost_equal(interp.evaluate([0, 0, 4]), arr[0, 0, -1])
    interp = ImageInterpolator(img, mode='constant', cval=0)
    assert_array_equal(interp.evaluate([0, 0, 4]), 0)
    interp = ImageInterpolator(img, mode='constant', cval=1)
    assert_array_equal(interp.evaluate([0, 0, 4]), 1)
