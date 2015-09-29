""" Testing resample function
"""
from __future__ import absolute_import

import numpy as np

from nibabel.affines import apply_affine

from ....core.image.image_spaces import (as_xyz_image,
                                         xyz_affine)
from ....core.api import Image, vox2mni
from ..resample import resample, cast_array
from ..transform import Transform
from ..affine import Affine

from numpy.testing import assert_array_almost_equal, assert_array_equal


AUX = np.array([-1.9, -1.2, -1, 2.3, 2.9, 19, 100, 258, 258.2, 258.8, 1e5])


def test_cast_array_float():
    assert_array_equal(cast_array(AUX, np.dtype(float)), AUX)


def test_cast_array_int8():
    assert_array_equal(cast_array(AUX, np.dtype('int8')),
                       [-2, -1, -1, 2, 3, 19, 100, 127, 127, 127, 127])


def test_cast_array_uint8():
    assert_array_equal(cast_array(AUX, np.dtype('uint8')),
                       [0, 0, 0, 2, 3, 19, 100, 255, 255, 255, 255])


def test_cast_array_int16():
    assert_array_equal(cast_array(AUX, np.dtype('int16')),
                       [-2, -1, -1, 2, 3, 19, 100, 258, 258, 259, 2**15 - 1])


def test_cast_array_uint16():
    assert_array_equal(cast_array(AUX, np.dtype('uint16')),
                       [0, 0, 0, 2, 3, 19, 100, 258, 258, 259, 2**16 - 1])


def test_cast_array_int32():
    assert_array_equal(cast_array(AUX, np.dtype('int32')),
                       np.round(AUX))


def test_cast_array_uint32():
    assert_array_equal(cast_array(AUX, np.dtype('uint32')),
                       np.maximum(np.round(AUX), 0))


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
    for transform in (aff, ApplyAffine(aff)):
        for order in (1, 3):
            # Default interpolation outside is constant == 0
            img2 = resample(img, transform, interp_order=order)
            arr2 = img2.get_data()
            exp_arr = np.zeros_like(arr)
            exp_arr[:-1,:,:] = arr[1:,:,:]
            assert_array_equal(arr2, exp_arr)
            # Test explicit constant value of 0
            img2 = resample(img, transform, interp_order=order,
                            mode='constant', cval=0.)
            exp_arr = np.zeros(arr.shape)
            exp_arr[:-1, :, :] = arr[1:, :, :]
            assert_array_almost_equal(img2.get_data(), exp_arr)
            # Test constant value of 1
            img2 = resample(img, transform, interp_order=order,
                            mode='constant', cval=1.)
            exp_arr[-1, :, :] = 1
            assert_array_almost_equal(img2.get_data(), exp_arr)
            # Test nearest neighbor
            img2 = resample(img, transform, interp_order=order,
                            mode='nearest')
            exp_arr[-1, :, :] = arr[-1, :, :]
            assert_array_almost_equal(img2.get_data(), exp_arr)
