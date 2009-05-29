import warnings
import numpy as np

from nipy.testing import assert_equal, assert_true, assert_false, \
    assert_raises, assert_array_equal, dec

from nipy.core.api import CoordinateMap, Affine, CoordinateSystem

import nipy.io.nifti_ref as niref

shape = range(1,8)
step = np.arange(1,8)

output_axes = 'xyztuvw'
input_axes = 'ijklmno'


def setup():
    # Suppress warnings during tests
    warnings.simplefilter("ignore")


def teardown():
    # Clear list of warning filters
    warnings.resetwarnings()


def test_iljk_to_xyzt():
    # this should raise a warning about the first three input
    # coordinates, and one about the last axis not being in the
    # correct order. this also will give a warning about the pixdim.
    # ninput_axes = list('iljk')
    input_coords = CoordinateSystem('iljk', 'input')
    output_coords = CoordinateSystem('xyzt', 'output')
    cmap = Affine(np.identity(5), input_coords, output_coords)
    newcmap, order = niref.coerce_coordmap(cmap)
    aff = np.array([[ 1,  0,  0,  0,  0,],
                    [ 0,  0,  1,  0,  0,],
                    [ 0,  0,  0,  1,  0,],
                    [ 0,  1,  0,  0,  0,],
                    [ 0,  0,  0,  0,  1,]])
    yield assert_equal, newcmap.affine, aff
    yield assert_equal, newcmap.input_coords.name, 'input-reordered'
    # order should match a reorder to 'ijkl'
    yield assert_equal, order, (0,2,3,1)


def test_ijkn_to_xyzt():
    # This should raise an exception about not having axis names
    # ['ijkl'].  Some warnings are printed during the try/except

    # ninput_axes = list('ijkn')
    ninput_axes = [input_axes[0], input_axes[1], input_axes[2], input_axes[5]]
    input_coords = CoordinateSystem('ijkn', 'input')
    output_coords = CoordinateSystem('xyzt', 'output')
    cmap = Affine(np.identity(5), input_coords, output_coords)
    assert_raises, ValueError, niref.coerce_coordmap, cmap


def test_ijkml_to_xyztu():
    # This should raise a warning about the last 2 axes not being in
    # order, and one about the loss of information from a non-diagonal
    # matrix. This also means that the pixdim will be wrong
    input_coords = CoordinateSystem('ijkml', 'input')
    output_coords = CoordinateSystem('xyztu', 'output')
    cmap = Affine(np.identity(6), input_coords, output_coords)
    newcmap, order = niref.coerce_coordmap(cmap)
    yield assert_equal, newcmap.input_coords.name, 'input-reordered'
    yield assert_equal, order, (0,1,2,4,3)
    # build an affine xform that should match newcmap.affine
    ndim = cmap.ndim[0]
    perm = np.zeros((ndim+1,ndim+1))
    perm[-1,-1] = 1
    for i, j in enumerate(order):
        perm[i,j] = 1
    B = np.dot(np.identity(6), perm)
    yield assert_true, np.allclose(newcmap.affine, B)
    # Compare applying the original cmap to a vector against applying
    # the reordered cmap to a reordered vector.
    X = np.arange(10, 15)
    Xr = [X[i] for i in order]
    yield assert_true, np.allclose(newcmap(Xr), cmap(X))


def test_ijkml_to_utzyx():
    # This should raise a warning about the last 2 axes not being in
    # order, and one about the loss of information from a non-diagonal
    # matrix, and also one about the nifti output coordinates.  
    input_coords = CoordinateSystem('ijkml', 'input')
    output_coords = CoordinateSystem('utzyx', 'output')
    cmap = Affine(np.identity(6), input_coords, output_coords)
    newcmap, order = niref.coerce_coordmap(cmap)
    yield assert_equal, newcmap.input_coords.name, 'input-reordered'
    yield assert_equal, newcmap.output_coords.name, 'output-reordered'
    yield assert_equal, order, (0,1,2,4,3)
    # Generate an affine that matches the order of the input coords 'ijkml'
    ndim = cmap.ndim[0]
    perm = np.zeros((ndim+1,ndim+1))
    perm[-1,-1] = 1
    for i, j in enumerate(order):
        perm[i,j] = 1
    r = np.zeros_like(perm)
    r[5, 5] = 1.0
    # The rotation part (5x5) of the newcmap affine will be flipped
    # vertically to account for the reverse order of the output coords
    # 'utzyx'
    r[:5, :5] = np.flipud(perm[:5,:5])
    yield assert_true, np.allclose(newcmap.affine, r)
    X = np.arange(10, 15)
    Xr = [X[i] for i in order]
    yield assert_true, np.allclose(np.fliplr(newcmap(Xr)), cmap(X))


def test_ijk_from_fps():
    x = niref.fps_from_ijk('ijk')
    yield assert_equal, niref.ijk_from_fps(x), 'ijk'
    x = niref.fps_from_ijk('jki')
    yield assert_equal, niref.ijk_from_fps(x), 'jki'


def test_general_coercion():
    # General test of coormap coercion
    cmap = Affine(
        np.eye(4),
        CoordinateSystem('ijk'),
        CoordinateSystem('xyz'))
    ncmap, order = niref.coerce_coordmap(cmap)
    yield assert_equal, ncmap.output_coords, cmap.output_coords
    yield assert_equal, order, (0, 1, 2)
    cmap = CoordinateMap(
        lambda x : x,
        CoordinateSystem('ijk'),
        CoordinateSystem('xyz'))
    yield assert_raises, ValueError, niref.coerce_coordmap, cmap
    cmap = Affine(
        np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,1]]),
        CoordinateSystem('ij'),
        CoordinateSystem('xyz'))
    yield assert_raises, ValueError, niref.coerce_coordmap, cmap
    cmap = Affine(
        np.eye(4),
        CoordinateSystem('ijq'),
        CoordinateSystem('xyz'))
    yield assert_raises, ValueError, niref.coerce_coordmap, cmap
    cmap = Affine(
        np.eye(4),
        CoordinateSystem('ijk'),
        CoordinateSystem('xyq'))
    yield assert_raises, ValueError, niref.coerce_coordmap, cmap
    # Space order should not matter if no time reordering required
    cmap = Affine(
        np.eye(4),
        CoordinateSystem('kji'),
        CoordinateSystem('xyz'))
    ncmap, order = niref.coerce_coordmap(cmap)
    yield assert_equal, ncmap.output_coords, cmap.output_coords
    yield assert_equal, order, (0, 1, 2)
    yield assert_array_equal, np.eye(4), ncmap.affine
    # Even if time is present but in the right place
    cmap = Affine(
        np.eye(5),
        CoordinateSystem('kjil'),
        CoordinateSystem('xyzt'))
    ncmap, order = niref.coerce_coordmap(cmap)
    yield assert_equal, ncmap.output_coords, cmap.output_coords
    yield assert_equal, order, (0, 1, 2, 3)
    yield assert_array_equal, np.eye(5), ncmap.affine
