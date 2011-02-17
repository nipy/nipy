# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the mask-extracting utilities.
"""

from __future__ import with_statement

import numpy as np

import nibabel as nib

from .. import mask as nnm
from ..mask import largest_cc, threshold_connect_components

from nipy.utils import InTemporaryDirectory

from nipy.testing import assert_equal, assert_true, \
    assert_array_almost_equal, assert_array_equal, funcfile, anatfile


def test_largest_cc():
    """ Check the extraction of the largest connected component.
    """
    a = np.zeros((6, 6, 6))
    a[1:3, 1:3, 1:3] = 1
    yield assert_equal, a, largest_cc(a)
    b = a.copy()
    b[5, 5, 5] = 1
    yield assert_equal, a, largest_cc(b)


def test_threshold_connect_components():
    a = np.zeros((10, 10))
    a[0, 0] = 1
    a[3, 4] = 1
    a = threshold_connect_components(a, 2)
    yield assert_true, np.all(a == 0)
    a[0, 0:3] = 1
    b = threshold_connect_components(a, 2)
    yield assert_true, np.all(a == b)


def test_mask_files():
    with InTemporaryDirectory():
        # Make a 4D file from the anatomical example
        img = nib.load(anatfile)
        arr = img.get_data()
        a2 = np.zeros(arr.shape + (2,))
        a2[:,:,:,0] = arr
        a2[:,:,:,1] = arr
        img = nib.Nifti1Image(a2, np.eye(4))
        a_fname = 'fourd_anat.nii'
        nib.save(img, a_fname)
        # check 4D mask
        msk1 = nnm.compute_mask_files(a_fname)
        # and mask from identical list of 3D files
        msk2 = nnm.compute_mask_files([anatfile, anatfile])
        yield assert_array_equal, msk1, msk2




if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
