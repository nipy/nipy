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
    # array coord map recreates the affine when you slice an image.  In
    # general, if you take an integer slice in some dimension, the
    # corresponding column of the affine will go, leaving a row for the
    # lost dimension, with all zeros, execpt for the translation in the
    # now-removed dimension, encoding the position of that particular
    # slice
    xz = 1.1; yz = 2.3; zz = 3.5
    xt = 10.0; yt = 11; zt = 12
    aff = np.diag([xz, yz, zz, 1])
    aff[:3,3] = [xt, yt, zt]
    shape = (2,3,4)
    cmap = Affine.from_params('ijk', 'xyz', aff)
    acm = acs.ArrayCoordMap(cmap, shape)
    # slice the coordinate map for the first axis
    sacm = acm[1]
    # The affine has lost the first column, but has a remaining row (the
    # first) encoding the translation to get to this slice
    yield assert_array_almost_equal(sacm.coordmap.affine,
                             np.array([
                [0, 0, xz+xt],
                [yz, 0, yt],
                [0, zz, zt],
                [0, 0, 1]]))
    sacm = acm[:,1]
    # lost second column, remaining second row with translation
    yield assert_array_almost_equal(sacm.coordmap.affine,
                             np.array([
                [xz, 0, xt],
                [0, 0, yz+yt],
                [0, zz, zt],
                [0, 0, 1]]))
    sacm = acm[:,:,2]
    # ditto third column and row
    yield assert_array_almost_equal(sacm.coordmap.affine,
                             np.array([
                [xz, 0, xt],
                [0, yz, yt],
                [0, 0, 2*zz+zt],
                [0, 0, 1]]))
    # check ellipsis slicing is the same as [:,: ...
    sacm = acm[...,2]
    yield assert_array_almost_equal(sacm.coordmap.affine,
                             np.array([
                [xz, 0, xt],
                [0, yz, yt],
                [0, 0, 2*zz+zt],
                [0, 0, 1]]))
    # that ellipsis can follow other slice types
    sacm = acm[:,...,2]
    yield assert_array_almost_equal(sacm.coordmap.affine,
                             np.array([
                [xz, 0, xt],
                [0, yz, yt],
                [0, 0, 2*zz+zt],
                [0, 0, 1]]))
    # that there can be only one ellipsis
    yield assert_raises(ValueError, acm.__getitem__, (
            (Ellipsis,Ellipsis,2)))
    # that you can integer slice in all three dimensions, leaving only
    # the translation column
    sacm = acm[1,0,2]
    yield assert_array_almost_equal(sacm.coordmap.affine,
                             np.array([
                [xz+xt],
                [yt],
                [2*zz+zt],
                [1]]))
    
