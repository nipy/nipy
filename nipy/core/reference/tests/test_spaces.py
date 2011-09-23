""" Testing coordinate map defined spaces
"""

import numpy as np

from ...image.image import Image
from ..coordinate_system import CoordinateSystem as CS
from ..coordinate_map import AffineTransform, CoordinateMap
from ...transforms.affines import from_matrix_vector
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
    affine = from_matrix_vector(np.arange(9).reshape((3,3)), [15,16,17])
    cmap = AffineTransform(VARS['d_cs_r3'], VARS['r_cs_r3'], affine)
    assert_array_equal(xyz_affine(cmap), affine)
    # Affine always reordered in xyz order
    assert_array_equal(xyz_affine(cmap.reordered_range([2,0,1])), affine)
    assert_array_equal(xyz_affine(cmap.reordered_range([2,1,0])), affine)
    assert_array_equal(xyz_affine(cmap.reordered_range([1,2,0])), affine)
    assert_array_equal(xyz_affine(cmap.reordered_range([1,0,2])), affine)
    assert_array_equal(xyz_affine(cmap.reordered_range([0,2,1])), affine)
    # 5x5 affine is shrunk
    rzs = np.c_[np.arange(12).reshape((4,3)), [0,0,0,12]]
    aff55 = from_matrix_vector(rzs, [15,16,17,18])
    cmap = AffineTransform(VARS['d_cs_r4'], VARS['r_cs_r4'], aff55)
    assert_array_equal(xyz_affine(cmap), affine)
    # Affine always reordered in xyz order
    assert_array_equal(xyz_affine(cmap.reordered_range([3,2,1,0])), affine)
    assert_array_equal(xyz_affine(cmap.reordered_range([2,0,1,3])), affine)
    # xyzs must be orthogonal to dropped axis
    for i in range(3):
        aff = aff55.copy()
        aff[i,3] = 1
        cmap = AffineTransform(VARS['d_cs_r4'], VARS['r_cs_r4'], aff)
        assert_raises(AffineError, xyz_affine, cmap)
        # And if reordered
        assert_raises(AffineError, xyz_affine, cmap.reordered_range([2,0,1,3]))
    # Non-square goes to square
    rzs = np.arange(12).reshape((4,3))
    aff54 = from_matrix_vector(rzs, [15,16,17,18])
    cmap = AffineTransform(VARS['d_cs_r3'], VARS['r_cs_r4'], aff54)
    assert_array_equal(xyz_affine(cmap), affine)
    rzs = np.c_[np.arange(12).reshape((4,3)), np.zeros((4,3))]
    aff57 = from_matrix_vector(rzs, [15,16,17,18])
    d_cs_r6 = CS('ijklmn', 'array')
    cmap = AffineTransform(d_cs_r6, VARS['r_cs_r4'], aff57)
    assert_array_equal(xyz_affine(cmap), affine)
    # Non-affine raises SpaceTypeError
    cmap_cmap = CoordinateMap(VARS['d_cs_r4'], VARS['r_cs_r4'], lambda x:x*3)
    assert_raises(SpaceTypeError, xyz_affine, cmap_cmap)
    # Not enough dimensions - SpaceTypeError
    d_cs_r2 = CS('ij', 'array')
    r_cs_r2 = CS(VARS['r_names'][:2], 'mni')
    cmap = AffineTransform(d_cs_r2, r_cs_r2,
                           np.array([[2,0,10],[0,3,11],[0,0,1]]))
    assert_raises(AxesError, xyz_affine, cmap)
    # Any dimensions not spatial, AxesError
    r_cs = CS(('mni-x', 'mni-y', 'mni-q'), 'mni')
    cmap = AffineTransform(VARS['d_cs_r3'],r_cs, affine)
    assert_raises(AxesError, xyz_affine, cmap)
    # Can pass in own validator
    my_valtor = dict(blind='x', leading='y', ditch='z')
    r_cs = CS(('blind', 'leading', 'ditch'), 'fall')
    cmap = AffineTransform(VARS['d_cs_r3'],r_cs, affine)
    assert_raises(AxesError, xyz_affine, cmap)
    assert_array_equal(xyz_affine(cmap, my_valtor), affine)


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
