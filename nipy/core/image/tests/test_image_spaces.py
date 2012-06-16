""" Testing Image spaces
"""

import numpy as np

import nibabel as nib
from nibabel.affines import from_matvec

from ..image import Image, rollimg
from ..image_spaces import (is_xyz_affable, as_xyz_image, xyz_affine,
                            make_xyz_image)
from ...reference.coordinate_system import CoordinateSystem as CS
from ...reference.coordinate_map import AffineTransform
from ...reference.spaces import (vox2mni, vox2talairach, voxel_csm,
                                 mni_space, talairach_space,
                                 AffineError, AxesError,
                                 XYZSpace, SpaceError)

from numpy.testing import (assert_array_almost_equal, assert_array_equal)

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
    assert_raises(AxesError, xyz_affine, img4_r)
    nimg = nib.Nifti1Image(arr, aff)
    assert_true(is_xyz_affable(nimg))
    assert_array_equal(xyz_affine(nimg), aff)
    # Any dimensions not spatial, AxesError
    d_cs = CS('ijk', 'voxels')
    r_cs = CS(('mni-x=L->R', 'mni-y=P->A', 'mni-q'), 'mni')
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


def test_image_as_xyz_image():
    # Test getting xyz affable version of the image
    arr = np.arange(24).reshape((1,2,3,4))
    aff = np.diag([2,3,4,5,1])
    img = Image(arr, vox2mni(aff))
    img_r = as_xyz_image(img)
    assert_true(img is img_r)
    # Reorder, reverse reordering, test != and ==
    for order in ((3, 0, 1, 2), (0, 3, 1, 2)):
        img_ro_out = img.reordered_reference(order)
        img_ro_in = img.reordered_axes(order)
        img_ro_both = img_ro_out.reordered_axes(order)
        for tmap in (img_ro_out, img_ro_in, img_ro_both):
            assert_false(is_xyz_affable(tmap))
            img_r = as_xyz_image(tmap)
            assert_false(tmap is img_r)
            assert_equal(img, img_r)
            assert_array_equal(img.get_data(), img_r.get_data())
    img_t0 = rollimg(img, 't')
    assert_false(is_xyz_affable(img_t0))
    img_t0_r = as_xyz_image(img_t0)
    assert_false(img_t0 is img_t0_r)
    assert_array_equal(img.get_data(), img_t0_r.get_data())
    assert_equal(img.coordmap, img_t0_r.coordmap)
    # Test against nibabel image
    nimg = nib.Nifti1Image(arr, np.diag([2,3,4,1]))
    nimg_r = as_xyz_image(nimg)
    assert_true(nimg is nimg_r)
    # It's sometimes impossible to make an xyz affable image
    # If the xyz coordinates depend on the time coordinate
    aff = np.array([[2, 0, 0, 2, 20],
                    [0, 3, 0, 0, 21],
                    [0, 0, 4, 0, 22],
                    [0, 0, 0, 5, 23],
                    [0, 0, 0, 0, 1]])
    img = Image(arr, vox2mni(aff))
    assert_raises(AffineError, as_xyz_image, img)
    # If any dimensions not spatial, AxesError
    arr = np.arange(24).reshape((2,3,4))
    aff = np.diag([2,3,4,1])
    d_cs = CS('ijk', 'voxels')
    r_cs = CS(('mni-x=L->R', 'mni-y=P->A', 'mni-q'), 'mni')
    cmap = AffineTransform(d_cs, r_cs, aff)
    img = Image(arr, cmap)
    assert_raises(AxesError, as_xyz_image, img)
    # Can pass in own validator
    my_valtor = dict(blind='x', leading='y', ditch='z')
    r_cs = CS(('blind', 'leading', 'ditch'), 'fall')
    cmap = AffineTransform(d_cs, r_cs, aff)
    img = Image(arr, cmap)
    assert_raises(AxesError, as_xyz_image, img)
    assert_true(as_xyz_image(img, my_valtor) is img)


def test_image_xyza_slices():
    # Jonathan found some nastiness where xyz present in output but there was
    # not corresponding axis for x in the input
    arr = np.arange(24).reshape((1,2,3,4))
    aff = np.diag([2,3,4,5,1])
    img = Image(arr, vox2mni(aff))
    img0 = img[0] # slice in X
    # The result does not have an input axis corresponding to x, and should
    # raise an error
    assert_raises(AxesError, as_xyz_image, img0)
    img0r = img0.reordered_reference([1,0,2,3]).reordered_axes([2,0,1])
    assert_raises(AxesError, as_xyz_image, img0r)


def test_make_xyz_image():
    # Standard neuro image creator
    arr = np.arange(24).reshape((1,2,3,4))
    aff = np.diag([2,3,4,1])
    img = make_xyz_image(arr, aff, 'mni')
    assert_equal(img.coordmap, vox2mni(aff, 1.0))
    assert_array_equal(img.get_data(), arr)
    img = make_xyz_image(arr, aff, 'talairach')
    assert_equal(img.coordmap, vox2talairach(aff, 1.0))
    assert_array_equal(img.get_data(), arr)
    img = make_xyz_image(arr, aff, talairach_space)
    assert_equal(img.coordmap, vox2talairach(aff, 1.0))
    # Unknown space as string raises SpaceError
    assert_raises(SpaceError, make_xyz_image, arr, aff, 'unlikely space name')
    funky_space = XYZSpace('hija')
    img = make_xyz_image(arr, aff, funky_space)
    csm = funky_space.to_coordsys_maker('t')
    in_cs = CS('ijkl', 'voxels')
    exp_cmap = AffineTransform(in_cs, csm(4), np.diag([2, 3, 4, 1, 1]))
    assert_equal(img.coordmap, exp_cmap)
    # Affine must be 4, 4
    bad_aff = np.diag([2,3,4,2,1])
    assert_raises(ValueError, make_xyz_image, arr, bad_aff, 'mni')
    # Can pass added zooms
    img = make_xyz_image(arr, (aff, (2.,)), 'mni')
    assert_equal(img.coordmap, vox2mni(aff, 2.0))
    # Also as scalar
    img = make_xyz_image(arr, (aff, 2.), 'mni')
    assert_equal(img.coordmap, vox2mni(aff, 2.0))
    # Must match length of needed zooms
    arr5 = arr[...,None]
    assert_raises(ValueError, make_xyz_image, arr5, (aff, 2.), 'mni')
    img = make_xyz_image(arr5, (aff, (2., 3.)), 'mni')
    assert_equal(img.coordmap, vox2mni(aff, (2.0, 3.0)))
    # Always xyz affable after creation
    assert_array_equal(xyz_affine(img), aff)
    assert_true(is_xyz_affable(img))
    # Need at least 3 dimensions in data
    assert_raises(ValueError, make_xyz_image, np.zeros((2,3)), aff, 'mni')
    # Check affines don't round / floor floating point
    aff = np.diag([2.1, 3, 4, 1])
    img = make_xyz_image(np.zeros((2, 3, 4)), aff, 'scanner')
    assert_array_equal(img.coordmap.affine, aff)
    img = make_xyz_image(np.zeros((2, 3, 4, 5)), aff, 'scanner')
    assert_array_equal(img.coordmap.affine, np.diag([2.1, 3, 4, 1, 1]))
