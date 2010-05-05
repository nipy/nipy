# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import warnings
import numpy as np

from nipy.core.api import CoordinateMap, AffineTransform, CoordinateSystem, \
    lps_output_coordnames, ras_output_coordnames

import nipy.io.nifti_ref as niref
from numpy.testing import assert_almost_equal

from nipy.testing import assert_equal, assert_true, \
    assert_raises, assert_array_equal


shape = range(1,8)
step = np.arange(1,8)

output_axes = lps_output_coordnames + tuple('tuvw')
input_axes = 'ijktuvw'
lps = lps_output_coordnames # shorthand

def setup():
    # Suppress warnings during tests
    warnings.simplefilter("ignore")


def teardown():
    # Clear list of warning filters
    warnings.resetwarnings()


def test_affine_transform_from_array():
    X = np.random.standard_normal((4,4))
    X[-1] = [0,0,0,1]
    function_domain = CoordinateSystem('ijk', 'input')
    function_range = CoordinateSystem(lps, 'output')
    cmap = AffineTransform(function_domain, function_range, X)

    A, _ = niref.affine_transform_from_array(X, 'ijk', [])

    yield assert_almost_equal, X, A.affine
    yield assert_raises, ValueError, niref.affine_transform_from_array, X[:,1:], 'ijk', []
    yield assert_raises, ValueError, niref.affine_transform_from_array, X, 'ij', []

    threed, fourd = niref.affine_transform_from_array(X, 'ijk', [3.5])
    yield assert_almost_equal, X, threed.affine
    yield assert_almost_equal, fourd.affine[3:,3:], np.diag([3.5,1])

    # get the pixdim back out
    A, p = niref.ni_affine_pixdim_from_affine(fourd)

    yield assert_almost_equal, X, A.affine
    yield assert_almost_equal, p, 3.5
    
    # try strict
    A, p = niref.ni_affine_pixdim_from_affine(fourd, strict=True)

    # try using RAS

    cmap = fourd.renamed_range(dict(zip(lps, ras_output_coordnames)))
    A, p = niref.ni_affine_pixdim_from_affine(cmap, strict=True)

    # will have been flipped to LPS

    yield assert_almost_equal, A.affine, np.dot(np.diag([-1,-1,1,1]),X)
    yield assert_equal, A.function_range.coord_names, lps

    # use coordinates that aren't OK and strict raises an exception

    cmap = fourd.renamed_range(dict(zip(lps, 'xyz')))
    yield assert_raises, ValueError, niref.ni_affine_pixdim_from_affine, cmap, \
        True

    # use coordinates that aren't OK and not strict just guesses LPS

    cmap4 = fourd.renamed_range(dict(zip(lps, 'xyz')))
    A, p =  niref.ni_affine_pixdim_from_affine(cmap4, False)

    yield assert_almost_equal, A.affine, X
    yield assert_equal, A.function_range.coord_names, lps
    yield assert_almost_equal, p, 3.5


    # non-square affine fails

    Z = np.random.standard_normal((5,4))
    Z[-1] = [0,0,0,1]
    affine = AffineTransform.from_params('ijk', 'xyzt', Z)
    yield assert_raises, ValueError, niref.ni_affine_pixdim_from_affine, affine

    # CoordinateMap fails
    
    ijk = CoordinateSystem('ijk')
    xyz = CoordinateSystem('xzy')
    cmap = CoordinateMap(ijk, xyz, np.exp)

    yield assert_raises, ValueError, niref.ni_affine_pixdim_from_affine, cmap, True

    # non-diagonal above 3rd dimension, with strict True raises an exception

    cmap5 = cmap4.renamed_range(dict(zip('xyz', lps)))
    cmap5.affine[3,-1] = 4.

    yield assert_raises, ValueError, niref.ni_affine_pixdim_from_affine, cmap5, True
    B, p = niref.ni_affine_pixdim_from_affine(cmap5)
    yield assert_equal, p, 3.5
