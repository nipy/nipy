""" Testing resample function
"""

import numpy as np

from ....core.image.image_spaces import (as_xyz_image,
                                         xyz_affine)
from ....core.api import Image, vox2mni
from ..resample import resample
from ..affine import Affine

from numpy.testing import assert_array_almost_equal


def test_resample():
    # Check basic cases of resampling
    arr = np.random.rand(10, 11, 12)
    img = Image(arr, vox2mni(np.eye(4)))
    T = Affine()
    img2 = resample(img, T, interp_order=0)
    assert_array_almost_equal(img2.get_data(), img.get_data())
    img2 = resample(img, T, img)
    assert_array_almost_equal(img2.get_data(), img.get_data())
    img_aff = as_xyz_image(img)
    img2 = resample(img, T, reference=(img_aff.shape, xyz_affine(img_aff)))
    assert_array_almost_equal(img2.get_data(), img.get_data())
