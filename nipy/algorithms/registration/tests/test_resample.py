""" Testing resample function
"""

import numpy as np

from nibabel.affines import apply_affine

from ....core.image.image_spaces import (as_xyz_image,
                                         xyz_affine)
from ....core.api import Image, vox2mni
from ..resample import resample
from ..transform import Transform
from ..affine import Affine

from numpy.testing import assert_array_almost_equal, assert_array_equal


def _test_resample(arr, T, interp_orders):
    # Check basic cases of resampling
    img = Image(arr, vox2mni(np.eye(4)))
    for i in interp_orders:
        img2 = resample(img, T, interp_order=i)
        assert_array_almost_equal(img2.get_data(), img.get_data())
        img_aff = as_xyz_image(img)
        img2 = resample(img, T, reference=(img_aff.shape, xyz_affine(img_aff)),
                        interp_order=i)
        assert_array_almost_equal(img2.get_data(), img.get_data())


def test_resample_dtypes():
    for arr in (np.random.rand(10, 11, 12),
                np.random.randint(100, size=(10, 11, 12)) - 50):
        _test_resample(arr, Affine(), (0, 1, 3, 5))
        _test_resample(arr, Transform(lambda x : x), (0, 1, 3, 5))


class ApplyAffine(Transform):
    """ Class implements Transform protocol for testing affine Transforms
    """
    def __init__(self, aff):
        self.func = lambda pts : apply_affine(aff, pts)


def test_resample_uint_data():
    arr = np.random.randint(100, size=(10, 11, 12)).astype('uint8')
    img = Image(arr, vox2mni(np.eye(4)))
    aff_obj = Affine((.5, .5, .5, .1, .1, .1, 0, 0, 0, 0, 0, 0))
    for transform in aff_obj, ApplyAffine(aff_obj.as_affine()):
        img2 = resample(img, transform)
        assert(np.min(img2.get_data()) >= 0)
        assert(np.max(img2.get_data()) < 255)


def test_resample_outvalue():
    arr = np.arange(3*3*3).reshape(3,3,3)

    img = Image(arr, vox2mni(np.eye(4)))
    aff = np.eye(4)
    aff[0,3] = 1.
    img2 = resample(img, aff)
    arr2 = img2.get_data()
    exp_arr = np.zeros_like(arr)
    exp_arr[:-1,:,:] = arr[1:,:,:]
    assert_array_equal(arr2, exp_arr)

    param = {'cval':1}
    img2 = resample(img, aff, interp_param=param)
    arr2 = img2.get_data()
    exp_arr = np.zeros_like(arr) + 1.
    exp_arr[:-1,:,:] = arr[1:,:,:]

    assert_array_equal(arr2, exp_arr)
