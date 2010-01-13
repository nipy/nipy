""" Testing array coords

"""

import numpy as np

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nipy.testing import parametric

from nipy.core.api import Affine

import nipy.core.reference.array_coords as acs


@parametric
def test_array_coord_map():
    xz = 1.1; yz = 2.3; zz = 3.5
    xt = 10.0; yt = 11; zt = 12
    aff = np.diag([xz, yz, zz, 1])
    aff[:3,3] = [xt, yt, zt]
    shape = (2,3,4)
    cmap = Affine.from_params('ijk', 'xyz', aff)
    acm = acs.ArrayCoordMap(cmap, shape)
    sacm = acm[1]
    yield assert_array_almost_equal(sacm.coordmap.affine,
                             np.array([
                [0, 0, xz+xt],
                [yz, 0, yt],
                [0, zz, zt],
                [0, 0, 1]]))
    sacm = acm[:,1]
    yield assert_array_almost_equal(sacm.coordmap.affine,
                             np.array([
                [xz, 0, xt],
                [0, 0, yz+yt],
                [0, zz, zt],
                [0, 0, 1]]))
    sacm = acm[:,:,2]
    yield assert_array_almost_equal(sacm.coordmap.affine,
                             np.array([
                [xz, 0, xt],
                [0, yz, yt],
                [0, 0, 2*zz+zt],
                [0, 0, 1]]))
    sacm = acm[...,2]
    yield assert_array_almost_equal(sacm.coordmap.affine,
                             np.array([
                [xz, 0, xt],
                [0, yz, yt],
                [0, 0, 2*zz+zt],
                [0, 0, 1]]))
    sacm = acm[:,...,2]
    yield assert_array_almost_equal(sacm.coordmap.affine,
                             np.array([
                [xz, 0, xt],
                [0, yz, yt],
                [0, 0, 2*zz+zt],
                [0, 0, 1]]))
    sacm = acm[1,0,2]
    yield assert_array_almost_equal(sacm.coordmap.affine,
                             np.array([
                [xz+xt],
                [yt],
                [2*zz+zt],
                [1]]))
