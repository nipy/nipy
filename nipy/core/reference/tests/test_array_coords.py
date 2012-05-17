# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Testing array coords

"""

import numpy as np

from nipy.core.api import (AffineTransform, CoordinateSystem,
                           CoordinateMap, Grid, ArrayCoordMap)

import nipy.core.reference.array_coords as acs

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal



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
    cmap = AffineTransform.from_params('ijk', 'xyz', aff)
    acm = acs.ArrayCoordMap(cmap, shape)
    # slice the coordinate map for the first axis
    sacm = acm[1]
    # The affine has lost the first column, but has a remaining row (the
    # first) encoding the translation to get to this slice
    assert_array_almost_equal(sacm.coordmap.affine,
                              np.array([
                                  [0, 0, xz+xt],
                                  [yz, 0, yt],
                                  [0, zz, zt],
                                  [0, 0, 1]]))
    sacm = acm[:,1]
    # lost second column, remaining second row with translation
    assert_array_almost_equal(sacm.coordmap.affine,
                              np.array([
                                  [xz, 0, xt],
                                  [0, 0, yz+yt],
                                  [0, zz, zt],
                                  [0, 0, 1]]))
    sacm = acm[:,:,2]
    # ditto third column and row
    assert_array_almost_equal(sacm.coordmap.affine,
                              np.array([
                                  [xz, 0, xt],
                                  [0, yz, yt],
                                  [0, 0, 2*zz+zt],
                                  [0, 0, 1]]))
    # check ellipsis slicing is the same as [:,: ...
    sacm = acm[...,2]
    assert_array_almost_equal(sacm.coordmap.affine,
                              np.array([
                                  [xz, 0, xt],
                                  [0, yz, yt],
                                  [0, 0, 2*zz+zt],
                                  [0, 0, 1]]))
    # that ellipsis can follow other slice types
    sacm = acm[:,...,2]
    assert_array_almost_equal(sacm.coordmap.affine,
                              np.array([
                                  [xz, 0, xt],
                                  [0, yz, yt],
                                  [0, 0, 2*zz+zt],
                                  [0, 0, 1]]))
    # that there can be only one ellipsis
    assert_raises(ValueError, acm.__getitem__, (
        (Ellipsis, Ellipsis,2)))
    # that you can integer slice in all three dimensions, leaving only
    # the translation column
    sacm = acm[1,0,2]
    assert_array_almost_equal(sacm.coordmap.affine,
                              np.array([
                                  [xz+xt],
                                  [yt],
                                  [2*zz+zt],
                                  [1]]))
    # that anything other than an int, slice or Ellipsis is an error
    assert_raises(ValueError, acm.__getitem__, ([0,2],))
    assert_raises(ValueError, acm.__getitem__, (np.array([0,2]),))


def test_grid():
    input = CoordinateSystem('ij', 'input')
    output = CoordinateSystem('xy', 'output')
    def f(ij):
        i = ij[:,0]
        j = ij[:,1]
        return np.array([i**2+j,j**3+i]).T
    cmap = CoordinateMap(input, output, f)
    grid = Grid(cmap)
    eval = ArrayCoordMap.from_shape(cmap, (50,40))
    assert_true(np.allclose(grid[0:50,0:40].values, eval.values))


def test_eval_slice():
    input = CoordinateSystem('ij', 'input')
    output = CoordinateSystem('xy', 'output')
    def f(ij):
        i = ij[:,0]
        j = ij[:,1]
        return np.array([i**2+j,j**3+i]).T

    cmap = CoordinateMap(input, output, f)

    cmap = CoordinateMap(input, output, f)
    grid = Grid(cmap)
    e = grid[0:50,0:40]
    ee = e[0:20:3]

    yield assert_equal, ee.shape, (7,40)
    yield assert_equal, ee.values.shape, (280,2)
    yield assert_equal, ee.transposed_values.shape, (2,7,40)

    ee = e[0:20:2,3]
    yield assert_equal, ee.values.shape, (10,2)
    yield assert_equal, ee.transposed_values.shape, (2,10)
    yield assert_equal, ee.shape, (10,)
