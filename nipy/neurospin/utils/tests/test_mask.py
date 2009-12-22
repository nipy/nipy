"""
Test the mask-extracting utilities.
"""

from __future__ import with_statement

from tempfile import mkstemp

import numpy as np

import nipy.io.imageformats as nii
import nipy.neurospin.utils.mask as nnm
from nipy.neurospin.utils.mask import _largest_cc

from nipy.utils import InTemporaryDirectory

from nipy.testing import assert_equal, assert_true, \
    assert_array_almost_equal, assert_array_equal, funcfile, anatfile


def test_largest_cc():
    """ Check the extraction of the largest connected component.
    """
    a = np.zeros((6, 6, 6))
    a[1:3, 1:3, 1:3] = 1
    yield assert_equal, a, _largest_cc(a)
    b = a.copy()
    b[5, 5, 5] = 1
    yield assert_equal, a, _largest_cc(b)


def test_unscaled_data():
    img, unscaled_arr = nnm.get_unscaled_img(funcfile)
    scaled_arr = img.get_data()
    dt = img.get_data_dtype()
    yield assert_equal, dt.kind, 'i'
    yield assert_equal, unscaled_arr.dtype.kind, 'i'
    yield assert_equal, scaled_arr.dtype.kind, 'f'
    hdr = img.get_header()
    aff = img.get_affine()
    slope = hdr['scl_slope']
    inter = hdr['scl_inter']
    yield assert_array_almost_equal, np.mean(scaled_arr), \
        np.mean(unscaled_arr) * slope + inter


def test_mask_files():
    with InTemporaryDirectory():
        # Make a 4D file from the anatomical example
        img = nii.load(anatfile)
        arr = img.get_data()
        a2 = np.zeros(arr.shape + (2,))
        a2[:,:,:,0] = arr
        a2[:,:,:,1] = arr
        img = nii.Nifti1Image(a2, np.eye(4))
        a_fname = 'fourd_anat.nii'
        nii.save(img, a_fname)
        # check 4D mask
        msk1 = nnm.compute_mask_files(a_fname)
        # and mask from identical list of 3D files
        msk2 = nnm.compute_mask_files([anatfile, anatfile])
        yield assert_array_equal, msk1, msk2



