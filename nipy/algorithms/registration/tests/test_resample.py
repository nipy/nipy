""" Testing resample function
"""

import numpy as np

from ....core.api import Image, vox2mni
from ..resample import resample
from ..affine import Affine

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_equal, assert_raises


def test_resample():
    # Check basic cases of resampling
    arr = np.zeros((2, 3, 4))
    img = Image(arr, vox2mni(np.eye(4)))
    T = Affine()
    img2 = resample(img, T, interp_order=0)
    assert_array_almost_equal(img2.get_data(), img.get_data())
    img2 = resample(img, T, img)
    assert_array_almost_equal(img2.get_data(), img.get_data())
    in_arr = np.zeros((3, 4, 5))
    in_arr[1:, 1:, 1:] = arr
    mov = Image(in_arr, vox2mni(np.eye(4)))
    exp_arr = np.zeros((3, 4, 5))
    exp_arr[:-1, :-1, :-1] = arr
    ref2mov = Affine([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    img4 = resample(mov, ref2mov)
    assert_array_almost_equal(img4.get_data(), exp_arr)
