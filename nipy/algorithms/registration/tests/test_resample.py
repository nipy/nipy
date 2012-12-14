""" Testing resample function
"""

import numpy as np

from ....core.image.image_spaces import (as_xyz_image,
                                         xyz_affine)
from ....core.api import Image, vox2mni
from ..resample import resample
from ..affine import Affine

from numpy.testing import assert_array_almost_equal


def _test_resample(arr, interp_orders):
    # Check basic cases of resampling
    img = Image(arr, vox2mni(np.eye(4)))
    T = Affine()
    for i in interp_orders:
        img2 = resample(img, T, interp_order=i)
        assert_array_almost_equal(img2.get_data(), img.get_data())
        img_aff = as_xyz_image(img)
        img2 = resample(img, T, reference=(img_aff.shape, xyz_affine(img_aff)),
                        interp_order=i)
        assert_array_almost_equal(img2.get_data(), img.get_data())


def test_resample_float_data():
    arr = np.random.rand(10, 11, 12)
    _test_resample(arr, (0, 1, 3, 5))

def test_resample_int_data():
    arr = np.random.randint(100, size=(10, 11, 12)) - 50
    _test_resample(arr, (3,))

def test_resample_uint_data():
    arr = np.random.randint(100, size=(10, 11, 12)).astype('uint8')
    img = Image(arr, vox2mni(np.eye(4)))
    T = Affine((.5, .5, .5, .1, .1, .1, 0, 0, 0, 0, 0, 0))
    img2 = resample(img, T)
    assert(np.min(img2.get_data()) >= 0)
    assert(np.max(img2.get_data()) < 255)

