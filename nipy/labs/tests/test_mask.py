# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the mask-extracting utilities.
"""

from __future__ import with_statement

import numpy as np

import nibabel as nib
from nibabel.tmpdirs import InTemporaryDirectory

from .. import mask as nnm
from ..mask import (largest_cc, threshold_connect_components, series_from_mask)

from nipy.testing import (assert_equal, assert_true, assert_array_equal,
                          anatfile, assert_false)


def test_largest_cc():
    """ Check the extraction of the largest connected component.
    """
    a = np.zeros((6, 6, 6))
    a[1:3, 1:3, 1:3] = 1
    assert_equal(a, largest_cc(a))
    b = a.copy()
    b[5, 5, 5] = 1
    assert_equal(a, largest_cc(b))


def test_threshold_connect_components():
    a = np.zeros((10, 10))
    a[0, 0] = 1
    a[3, 4] = 1
    a = threshold_connect_components(a, 2)
    assert_true(np.all(a == 0))
    a[0, 0:3] = 1
    b = threshold_connect_components(a, 2)
    assert_true(np.all(a == b))


def test_mask():
    mean_image = np.ones((9, 9))
    mean_image[3:-3, 3:-3] = 10
    mean_image[5, 5] = 100
    mask1 = nnm.compute_mask(mean_image)
    mask2 = nnm.compute_mask(mean_image, exclude_zeros=True)
    # With an array with no zeros, exclude_zeros should not make
    # any difference
    assert_array_equal(mask1, mask2)
    # Check that padding with zeros does not change the extracted mask
    mean_image2 = np.zeros((30, 30))
    mean_image2[:9, :9] = mean_image
    mask3 = nnm.compute_mask(mean_image2, exclude_zeros=True)
    assert_array_equal(mask1, mask3[:9, :9])
    # However, without exclude_zeros, it does
    mask3 = nnm.compute_mask(mean_image2)
    assert_false(np.allclose(mask1, mask3[:9, :9]))
    # check that  opening is 2 by default
    mask4 = nnm.compute_mask(mean_image, exclude_zeros=True, opening=2)
    assert_array_equal(mask1, mask4)
    # check that opening has an effect
    mask5 = nnm.compute_mask(mean_image, exclude_zeros=True, opening=0)
    assert_true(mask5.sum() > mask4.sum())


def test_mask_files():
    with InTemporaryDirectory():
        # Make a 4D file from the anatomical example
        img = nib.load(anatfile)
        arr = img.get_data()
        a2 = np.zeros(arr.shape + (2, ))
        a2[:, :, :, 0] = arr
        a2[:, :, :, 1] = arr
        img = nib.Nifti1Image(a2, np.eye(4))
        a_fname = 'fourd_anat.nii'
        nib.save(img, a_fname)
        # check 4D mask
        msk1, mean1 = nnm.compute_mask_files(a_fname, return_mean=True)
        # and mask from identical list of 3D files
        msk2, mean2 = nnm.compute_mask_files([anatfile, anatfile],
                                             return_mean=True)
        assert_array_equal(msk1, msk2)
        assert_array_equal(mean1, mean2)


def test_series_from_mask():
    """ Test the smoothing of the timeseries extraction
    """
    # A delta in 3D
    data = np.zeros((40, 40, 40, 2))
    data[20, 20, 20] = 1
    mask = np.ones((40, 40, 40), dtype=np.bool)
    with InTemporaryDirectory():
        for affine in (np.eye(4), np.diag((1, 1, -1, 1)),
                        np.diag((.5, 1, .5, 1))):
            img = nib.Nifti1Image(data, affine)
            nib.save(img, 'testing.nii')
            series, header = series_from_mask('testing.nii', mask, smooth=9)
            series = np.reshape(series[:, 0], (40, 40, 40))
            vmax = series.max()
            # We are expecting a full-width at half maximum of
            # 9mm/voxel_size:
            above_half_max = series > .5*vmax
            for axis in (0, 1, 2):
                proj = np.any(np.any(np.rollaxis(above_half_max,
                                axis=axis), axis=-1), axis=-1)
                assert_equal(proj.sum(), 9/np.abs(affine[axis, axis]))

        # Check that NaNs in the data do not propagate
        data[10, 10, 10] = np.NaN
        img = nib.Nifti1Image(data, affine)
        nib.save(img, 'testing.nii')
        series, header = series_from_mask('testing.nii', mask, smooth=9)
        assert_true(np.all(np.isfinite(series)))

def test_compute_mask_sessions():
    """Test that the mask computes well on multiple sessions
    """
    with InTemporaryDirectory():
        # Make a 4D file from the anatomical example
        img = nib.load(anatfile)
        arr = img.get_data()
        a2 = np.zeros(arr.shape + (2, ))
        a2[:, :, :, 0] = arr
        a2[:, :, :, 1] = arr
        img = nib.Nifti1Image(a2, np.eye(4))
        a_fname = 'fourd_anat.nii'
        nib.save(img, a_fname)
        a3 = a2.copy()
        a3[:10, :10, :10] = 0
        img2 = nib.Nifti1Image(a3, np.eye(4))
        # check 4D mask
        msk1 = nnm.compute_mask_sessions([img2, img2])
        msk2 = nnm.compute_mask_sessions([img2, a_fname])
        assert_array_equal(msk1, msk2)
        msk3 = nnm.compute_mask_sessions([img2, a_fname], threshold=.9)
        msk4 = nnm.compute_mask_sessions([img2, a_fname], threshold=0)
        msk5 = nnm.compute_mask_sessions([a_fname, a_fname])
        assert_array_equal(msk1, msk3)
        assert_array_equal(msk4, msk5)

if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
