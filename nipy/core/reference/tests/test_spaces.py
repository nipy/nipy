""" Testing coordinate map defined spaces
"""
from __future__ import absolute_import

import numpy as np

from nibabel.affines import from_matvec

from ...image.image import Image
from ..coordinate_system import CoordinateSystem as CS, CoordSysMakerError
from ..coordinate_map import AffineTransform, CoordinateMap
from ..spaces import (vox2mni, vox2scanner, vox2talairach, vox2unknown,
                      vox2aligned, xyz_affine, xyz_order, SpaceTypeError,
                      AxesError, AffineError, XYZSpace, known_space,
                      known_spaces, is_xyz_space, SpaceError,
                      is_xyz_affable,
                      get_world_cs, mni_csm, mni_space)

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_equal, assert_raises,
                        assert_not_equal)

VARS = {}


def setup():
    d_names = list('ijkl')
    xyzs = 'x=L->R', 'y=P->A', 'z=I->S'
    mni_xyzs = ['mni-' + suff for suff in xyzs]
    scanner_xyzs = ['scanner-' + suff for suff in xyzs]
    unknown_xyzs = ['unknown-' + suff for suff in xyzs]
    aligned_xyzs = ['aligned-' + suff for suff in xyzs]
    talairach_xyzs = ['talairach-' + suff for suff in xyzs]
    r_names = mni_xyzs + ['t']
    d_cs_r3 = CS(d_names[:3], 'voxels')
    d_cs_r4 = CS(d_names[:4], 'voxels')
    r_cs_r3 = CS(r_names[:3], 'mni')
    r_cs_r4 = CS(r_names[:4], 'mni')
    VARS.update(locals())


def test_xyz_space():
    # Space objects
    sp = XYZSpace('hijo')
    assert_equal(sp.name, 'hijo')
    exp_labels = ['hijo-' + L for L in ('x=L->R', 'y=P->A', 'z=I->S')]
    exp_map = dict(zip('xyz', exp_labels))
    assert_equal([sp.x, sp.y, sp.z], exp_labels)
    assert_equal(sp.as_tuple(), tuple(exp_labels))
    assert_equal(sp.as_map(), exp_map)
    known = {}
    sp.register_to(known)
    assert_equal(known, dict(zip(exp_labels, 'xyz')))
    # Coordinate system making, and __contains__ tests
    csm = sp.to_coordsys_maker()
    cs = csm(2)
    assert_equal(cs, CS(exp_labels[:2], 'hijo'))
    # This is only 2 dimensions, not fully in space
    assert_false(cs in sp)
    cs = csm(3)
    assert_equal(cs, CS(exp_labels, 'hijo'))
    # We now have all 3, this in in the space
    assert_true(cs in sp)
    # More dimensions than default, error
    assert_raises(CoordSysMakerError, csm, 4)
    # But we can pass in names for further dimensions
    csm = sp.to_coordsys_maker('tuv')
    cs = csm(6)
    assert_equal(cs, CS(exp_labels + list('tuv'), 'hijo'))
    # These are also in the space, because they contain xyz
    assert_true(cs in sp)
    # The axes can be in any order as long as they are a subset
    cs = CS(exp_labels, 'hijo')
    assert_true(cs in sp)
    cs = CS(exp_labels[::-1], 'hijo')
    assert_true(cs in sp)
    cs = CS(['t'] + exp_labels, 'hijo')
    assert_true(cs in sp)
    # The coordinate system name doesn't matter
    cs = CS(exp_labels, 'hija')
    assert_true(cs in sp)
    # Images, and coordinate maps, also work
    cmap = AffineTransform('ijk', cs, np.eye(4))
    assert_true(cmap in sp)
    img = Image(np.zeros((2,3,4)), cmap)
    assert_true(img in sp)
    # equality
    assert_equal(XYZSpace('hijo'), XYZSpace('hijo'))
    assert_not_equal(XYZSpace('hijo'), XYZSpace('hija'))


def test_is_xyz_space():
    # test test for xyz space
    assert_true(is_xyz_space(XYZSpace('hijo')))
    for sp in known_spaces:
        assert_true(is_xyz_space(sp))
    for obj in ([], {}, object(), CS('xyz')):
        assert_false(is_xyz_space(obj))


def test_known_space():
    # Known space utility routine
    for sp in known_spaces:
        cs = sp.to_coordsys_maker()(3)
        assert_equal(known_space(cs), sp)
    cs = CS('xyz')
    assert_equal(known_space(cs), None)
    sp0 = XYZSpace('hijo')
    sp1 = XYZSpace('hija')
    custom_spaces = (sp0, sp1)
    for sp in custom_spaces:
        cs = sp.to_coordsys_maker()(3)
        assert_equal(known_space(cs, custom_spaces), sp)


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
        (vox2scanner, VARS['scanner_xyzs'] + ['t'], 'scanner'),
        (vox2unknown, VARS['unknown_xyzs'] + ['t'], 'unknown'),
        (vox2aligned, VARS['aligned_xyzs'] + ['t'], 'aligned'),
        (vox2mni, VARS['mni_xyzs'] + ['t'], 'mni'),
        (vox2talairach, VARS['talairach_xyzs'] + ['t'], 'talairach')):
        for i in range(1,5):
            dom_cs = CS('ijkl'[:i], 'voxels')
            ran_cs = CS(r_names[:i], r_name)
            aff = np.diag(list(range(i)) + [1])
            assert_equal(csm(aff), AffineTransform(dom_cs, ran_cs, aff))


def test_get_world_cs():
    # Utility to get world from a variety of inputs
    assert_equal(get_world_cs('mni'), mni_csm(3))
    mnit = mni_space.to_coordsys_maker('t')(4)
    assert_equal(get_world_cs(mni_space, 4), mnit)
    assert_equal(get_world_cs(mni_csm, 4), mni_csm(4))
    assert_equal(get_world_cs(CS('xyz')), CS('xyz'))
    hija = XYZSpace('hija')
    maker = hija.to_coordsys_maker('qrs')
    assert_equal(get_world_cs('hija', ndim = 5, extras='qrs', spaces=[hija]),
                 maker(5))
    assert_raises(SpaceError, get_world_cs, 'hijo')
    assert_raises(SpaceError, get_world_cs, 'hijo', spaces=[hija])
    assert_raises(ValueError, get_world_cs, 0)


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
    d_cs_r6 = CS('ijklmn', 'voxels')
    cmap = AffineTransform(d_cs_r6, VARS['r_cs_r4'], aff57)
    assert_array_equal(xyz_affine(cmap), aff3d)
    # Non-affine raises SpaceTypeError
    cmap_cmap = CoordinateMap(VARS['d_cs_r4'], VARS['r_cs_r4'], lambda x:x*3)
    assert_raises(SpaceTypeError, xyz_affine, cmap_cmap)
    # Not enough dimensions - SpaceTypeError
    d_cs_r2 = CS('ij', 'voxels')
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
    r_cs = CS(('mni-x=L->R', 'mni-y=P->A', 'mni-q'), 'mni')
    assert_raises(AxesError, xyz_order, r_cs)
    r_cs = CS(('t', 'mni-x=L->R', 'mni-z=I->S', 'mni-y=P->A'), 'mni')
    assert_array_equal(xyz_order(r_cs), [1, 3, 2, 0])
    # Can pass in own validator
    my_valtor = dict(ditch='x', leading='y', blind='z')
    r_cs = CS(('blind', 'leading', 'ditch'), 'fall')
    assert_raises(AxesError, xyz_order, r_cs)
    assert_array_equal(xyz_order(r_cs, my_valtor), [2,1,0])


def test_is_xyz_affable():
    # Whether there exists an xyz affine for this coordmap
    affine = np.diag([2,4,5,6,1])
    cmap = AffineTransform(VARS['d_cs_r4'], VARS['r_cs_r4'], affine)
    assert_true(is_xyz_affable(cmap))
    assert_false(is_xyz_affable(cmap.reordered_range([3,0,1,2])))
    assert_false(is_xyz_affable(cmap.reordered_domain([3,0,1,2])))
    # Can pass in own validator
    my_valtor = dict(blind='x', leading='y', ditch='z')
    r_cs = CS(('blind', 'leading', 'ditch'), 'fall')
    affine = from_matvec(np.arange(9).reshape((3, 3)), [11, 12, 13])
    cmap = AffineTransform(VARS['d_cs_r3'], r_cs, affine)
    # No xyz affine if we don't use our custom dictionary
    assert_false(is_xyz_affable(cmap))
    # Is if we do
    assert_true(is_xyz_affable(cmap, my_valtor))
