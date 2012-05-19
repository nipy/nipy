""" Testing coordinate map defined spaces
"""

import numpy as np

from nibabel.affines import from_matvec

from ...image.image import Image
from ..coordinate_system import CoordinateSystem as CS
from ..coordinate_map import AffineTransform, CoordinateMap
from ..spaces import (vox2mni, vox2scanner, vox2talairach, xyz_affine,
                      xyz_order, SpaceTypeError, AxesError, AffineError)

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_false, assert_equal, assert_raises

VARS = {}


def setup():
    d_names = list('ijkl')
    r_names = ['mni-x', 'mni-y', 'mni-z', 't']
    d_cs_r3 = CS(d_names[:3], 'array')
    d_cs_r4 = CS(d_names[:4], 'array')
    r_cs_r3 = CS(r_names[:3], 'mni')
    r_cs_r4 = CS(r_names[:4], 'mni')
    VARS.update(locals())


def test_image_creation():
    # 3D image
    arr = np.arange(24).reshape(2,3,4)
    aff = np.diag([2,3,4,1])
    img = Image(arr, vox2mni(aff))
    assert_equal(img.shape, (2,3,4))
    assert_array_equal(img.affine, aff)
    assert_array_equal(img.coordmap,
                       AffineTransform(VARS['d_cs_r3'], VARS['r_cs_r3'], aff))
    # 4D image
    arr = np.arange(24).reshape(2,3,4,1)
    img = Image(arr, vox2mni(aff, 7))
    exp_aff = np.diag([2,3,4,7,1])
    assert_equal(img.shape, (2,3,4,1))
    exp_cmap = AffineTransform(VARS['d_cs_r4'], VARS['r_cs_r4'], exp_aff)
    assert_equal(img.coordmap, exp_cmap)


def test_default_makers():
    # Tests that the makers make expected coordinate maps
    for csm, r_names, r_name in (
        (vox2scanner, ('scanner-x', 'scanner-y', 'scanner-z', 't'), 'scanner'),
        (vox2mni, ('mni-x', 'mni-y', 'mni-z', 't'), 'mni'),
        (vox2talairach,('talairach-x', 'talairach-y', 'talairach-z', 't'),
         'talairach')):
        for i in range(1,5):
            dom_cs = CS('ijkl'[:i], 'array')
            ran_cs = CS(r_names[:i], r_name)
            aff = np.diag(range(i) + [1])
            assert_equal(csm(aff), AffineTransform(dom_cs, ran_cs, aff))


def test_xyz_affine():
    # Getting an xyz affine from coordmaps
    aff3d = from_matvec(np.arange(9).reshape((3,3)), [15,16,17])
    cmap3d = AffineTransform(VARS['d_cs_r3'], VARS['r_cs_r3'], aff3d)
    rzs = np.c_[np.arange(12).reshape((4,3)), [0,0,0,12]]
    aff4d = from_matvec(rzs, [15,16,17,18])
    cmap4d = AffineTransform(VARS['d_cs_r4'], VARS['r_cs_r4'], aff4d)
    # Simplest case of 3D affine -> affine unchanged
    assert_array_equal(xyz_affine(cmap3d), aff3d)
    # 4D (5, 5) affine -> 3D equivalent
    assert_array_equal(xyz_affine(cmap4d), aff3d)
    # Any dimensions not spatial, AxesError
    r_cs = CS(('mni-x', 'mni-y', 'mni-q'), 'mni')
    funny_cmap = AffineTransform(VARS['d_cs_r3'],r_cs, aff3d)
    assert_raises(AxesError, xyz_affine, funny_cmap)
    r_cs = CS(('mni-x', 'mni-q', 'mni-z'), 'mni')
    funny_cmap = AffineTransform(VARS['d_cs_r3'],r_cs, aff3d)
    assert_raises(AxesError, xyz_affine, funny_cmap)
    # We insist that the coordmap is in output xyz order
    permutations = (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)
    for perm in permutations:
        assert_raises(AxesError, xyz_affine, cmap3d.reordered_range(perm))
    # The input order doesn't matter, as long as the xyz axes map to the first
    # three input axes
    for perm in permutations:
        assert_array_equal(xyz_affine(
            cmap3d.reordered_domain(perm)), aff3d[:, perm + (-1,)])
    # But if the corresponding input axes not in the first three, an axis error
    wrong_inputs = cmap4d.reordered_domain([0, 1, 3, 2])
    assert_raises(AxesError, xyz_affine, wrong_inputs)
    # xyzs must be orthogonal to dropped axis
    for i in range(3):
        aff = aff4d.copy()
        aff[i,3] = 1
        cmap = AffineTransform(VARS['d_cs_r4'], VARS['r_cs_r4'], aff)
        assert_raises(AffineError, xyz_affine, cmap)
        # And if reordered
        assert_raises(AxesError, xyz_affine, cmap.reordered_range([2,0,1,3]))
    # Non-square goes to square
    aff54 = np.array([[0, 1, 2, 15],
                      [3, 4, 5, 16],
                      [6, 7, 8, 17],
                      [0, 0, 0, 18],
                      [0, 0, 0, 1]])
    cmap = AffineTransform(VARS['d_cs_r3'], VARS['r_cs_r4'], aff54)
    assert_array_equal(xyz_affine(cmap), aff3d)
    aff57 = np.array([[0, 1, 2, 0, 0, 0, 15],
                      [3, 4, 5, 0, 0, 0, 16],
                      [6, 7, 8, 0, 0, 0, 17],
                      [0, 0, 0, 0, 0, 0, 18],
                      [0, 0, 0, 0, 0, 0, 1]])
    d_cs_r6 = CS('ijklmn', 'array')
    cmap = AffineTransform(d_cs_r6, VARS['r_cs_r4'], aff57)
    assert_array_equal(xyz_affine(cmap), aff3d)
    # Non-affine raises SpaceTypeError
    cmap_cmap = CoordinateMap(VARS['d_cs_r4'], VARS['r_cs_r4'], lambda x:x*3)
    assert_raises(SpaceTypeError, xyz_affine, cmap_cmap)
    # Not enough dimensions - SpaceTypeError
    d_cs_r2 = CS('ij', 'array')
    r_cs_r2 = CS(VARS['r_names'][:2], 'mni')
    cmap = AffineTransform(d_cs_r2, r_cs_r2,
                           np.array([[2,0,10],[0,3,11],[0,0,1]]))
    assert_raises(AxesError, xyz_affine, cmap)
    # Can pass in own validator
    my_valtor = dict(blind='x', leading='y', ditch='z')
    r_cs = CS(('blind', 'leading', 'ditch'), 'fall')
    cmap = AffineTransform(VARS['d_cs_r3'],r_cs, aff3d)
    assert_raises(AxesError, xyz_affine, cmap)
    assert_array_equal(xyz_affine(cmap, my_valtor), aff3d)
    # Slices in x, y, z coordmaps raise error because of missing spatial
    # dimensions
    arr = np.arange(120).reshape((2, 3, 4, 5))
    aff = np.diag([2, 3, 4, 5, 1])
    img = Image(arr, vox2mni(aff))
    assert_raises(AxesError, xyz_affine, img[1].coordmap)
    assert_raises(AxesError, xyz_affine, img[:,1].coordmap)
    assert_raises(AxesError, xyz_affine, img[:,:,1].coordmap)


def test_xyz_order():
    # Getting xyz ordering from a coordinate system
    assert_array_equal(xyz_order(VARS['r_cs_r3']), [0,1,2])
    assert_array_equal(xyz_order(VARS['r_cs_r4']), [0,1,2,3])
    r_cs = CS(('mni-x', 'mni-y', 'mni-q'), 'mni')
    assert_raises(AxesError, xyz_order, r_cs)
    r_cs = CS(('t', 'mni-x', 'mni-z', 'mni-y'), 'mni')
    assert_array_equal(xyz_order(r_cs), [1, 3, 2, 0])
    # Can pass in own validator
    my_valtor = dict(ditch='x', leading='y', blind='z')
    r_cs = CS(('blind', 'leading', 'ditch'), 'fall')
    assert_raises(AxesError, xyz_order, r_cs)
    assert_array_equal(xyz_order(r_cs, my_valtor), [2,1,0])


def is_xyz_affable():
    # Whether there exists an xyz affine for this coordmap
    affine = np.diag([2,4,5,6,1])
    cmap = AffineTransform(VARS['d_cs_r4'], VARS['r_cs_r4'], affine)
    assert_true(is_xyz_affable(cmap))
    assert_true(is_xyz_affable(cmap.reordered_range([3,0,1,2])))
    assert_false(is_xyz_affable(cmap.reordered_domain([3,0,1,2])))
    # Can pass in own validator
    my_valtor = dict(blind='x', leading='y', ditch='z')
    r_cs = CS(('blind', 'leading', 'ditch'), 'fall')
    cmap = AffineTransform(VARS['d_cs_r3'],r_cs, affine)
    # No xyz affine if we don't use our custom dictionary
    assert_false(is_xyz_affable(cmap))
    # Is if we do
    assert_true(is_xyz_affable(cmap, my_valtor))
