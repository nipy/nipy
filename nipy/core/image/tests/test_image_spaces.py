""" Testing Image spaces
"""

import numpy as np

import nibabel as nib

from ..image import Image, rollaxis as img_rollaxis
from ..image_spaces import is_xyz_affable, as_xyz_affable, xyz_affine
from ...transforms.affines import from_matrix_vector
from ...reference.coordinate_system import CoordinateSystem as CS
from ...reference.coordinate_map import AffineTransform
from ...reference.spaces import (vox2mni, AffineError, AxesError)

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_false, assert_equal, assert_raises

def test_image_xyz_affine():
    # Test getting the image xyz affines
    arr = np.arange(24).reshape((2,3,4))
    aff = np.diag([2,3,4,1])
    img = Image(arr, vox2mni(aff))
    assert_true(is_xyz_affable(img))
    assert_array_equal(xyz_affine(img), aff)
    arr4 = np.arange(24).reshape((1,2,3,4))
    img4 = Image(arr4, vox2mni(np.diag([2,3,4,5,1])))
    assert_true(is_xyz_affable(img4))
    img4_r = img4.reordered_axes([3,2,0,1])
    assert_false(is_xyz_affable(img4_r))
    assert_raises(AffineError, xyz_affine, img4_r)
    nimg = nib.Nifti1Image(arr, aff)
    assert_true(is_xyz_affable(nimg))
    assert_array_equal(xyz_affine(nimg), aff)
    # Any dimensions not spatial, AxesError
    d_cs = CS('ijk', 'array')
    r_cs = CS(('mni-x', 'mni-y', 'mni-q'), 'mni')
    cmap = AffineTransform(d_cs,r_cs, aff)
    img = Image(arr, cmap)
    assert_raises(AxesError, xyz_affine, img)
    # Can pass in own validator
    my_valtor = dict(blind='x', leading='y', ditch='z')
    r_cs = CS(('blind', 'leading', 'ditch'), 'fall')
    cmap = AffineTransform(d_cs, r_cs, aff)
    img = Image(arr, cmap)
    assert_raises(AxesError, xyz_affine, img)
    assert_array_equal(xyz_affine(img, my_valtor), aff)


def test_image_as_xyz_affable():
    # Test getting xyz affable version of the image
    arr = np.arange(24).reshape((1,2,3,4))
    aff = np.diag([2,3,4,5,1])
    img = Image(arr, vox2mni(aff))
    img_r = as_xyz_affable(img)
    assert_true(img is img_r)
    img_t0 = img_rollaxis(img, 't')
    assert_false(is_xyz_affable(img_t0))
    img_t0_r = as_xyz_affable(img_t0)
    assert_false(img_t0 is img_t0_r)
    assert_array_equal(img.get_data(), img_t0_r.get_data())
    assert_equal(img.coordmap, img_t0_r.coordmap)
    nimg = nib.Nifti1Image(arr, np.diag([2,3,4,1]))
    nimg_r = as_xyz_affable(nimg)
    assert_true(nimg is nimg_r)
    # It's sometimes impossible to make an xyz affable image
    # If the xyz coordinates depend on the time coordinate
    aff = from_matrix_vector(np.arange(16).reshape((4,4)), [20,21,22,23])
    img = Image(arr, vox2mni(aff))
    assert_raises(AffineError, as_xyz_affable, img)
    # If any dimensions not spatial, AxesError
    arr = np.arange(24).reshape((2,3,4))
    aff = np.diag([2,3,4,1])
    d_cs = CS('ijk', 'array')
    r_cs = CS(('mni-x', 'mni-y', 'mni-q'), 'mni')
    cmap = AffineTransform(d_cs, r_cs, aff)
    img = Image(arr, cmap)
    assert_raises(AxesError, as_xyz_affable, img)
    # Can pass in own validator
    my_valtor = dict(blind='x', leading='y', ditch='z')
    r_cs = CS(('blind', 'leading', 'ditch'), 'fall')
    cmap = AffineTransform(d_cs, r_cs, aff)
    img = Image(arr, cmap)
    assert_raises(AxesError, as_xyz_affable, img)
    assert_true(as_xyz_affable(img, my_valtor) is img)
