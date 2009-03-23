import warnings
import numpy as np

from neuroimaging.testing import *

from neuroimaging.core.api import Affine, CoordinateSystem
from neuroimaging.core.reference import coordinate_system
import neuroimaging.io.nifti_ref as nifti

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


def test_ijkl_to_xyzt():
    output_coords = CoordinateSystem(output_axes[:4], 'output')
    input_coords = CoordinateSystem(input_axes[:4], 'input')
    aff = np.diag(list(step[:4]) + [1])
    cmap = Affine(aff, input_coords, output_coords)
    newcmap, order, pixdim, diminfo = nifti.coordmap4io(cmap)
    yield assert_equal, newcmap.input_coords.name, 'input'
    yield assert_equal, newcmap.output_coords.name, 'output'
    yield assert_equal, order, (0,1,2,3)
    yield assert_equal, aff, newcmap.affine


@dec.knownfailure
def test_kji_to_xyz():
    output_coords = CoordinateSystem(output_axes[:3], 'output')
    input_coords = CoordinateSystem(input_axes[:3][::-1], 'input')
    cmap = Affine(np.diag(list(step[:3]) + [1]), input_coords, output_coords)
    newcmap, order, pixdim, diminfo = nifti.coordmap4io(cmap)
    yield assert_equal, newcmap.input_coords.name, 'input-reordered'
    yield assert_equal, newcmap.output_coords.name, 'output'
    aff = np.array([[ 0,  0,  1,  0,],
                    [ 0,  2,  0,  0,],
                    [ 3,  0,  0,  0,],
                    [ 0,  0,  0,  1,]])
    yield assert_equal, newcmap.affine, aff
    yield assert_equal, order, (2, 1, 0)

    # FIXME: Should we be testing the returned diminfo also?

    # One last test to remind us of the FIXME in pixdims
    yield assert_true, np.allclose(pixdim, np.arange(3)+1)

@dec.needs_review('Needs review of coordmap4io function')
def test_kijl_to_xyzt():
    output_coords = CoordinateSystem(output_axes[:4], 'output')
    input_coords = CoordinateSystem([input_axes[2],
                                                  input_axes[0],
                                                  input_axes[1],
                                                  input_axes[3]], 'input')
    cmap = Affine(np.identity(5), input_coords, output_coords)
    newcmap, order, pixdim, diminfo = nifti.coordmap4io(cmap)
    # the affine should come back reordered so the input coords map to
    # output coords
    aff = np.array([[ 0,  1,  0,  0,  0,],
                    [ 0,  0,  1,  0,  0,],
                    [ 1,  0,  0,  0,  0,],
                    [ 0,  0,  0,  1,  0,],
                    [ 0,  0,  0,  0,  1,]])

    yield assert_equal, newcmap.input_coords.name, 'input-reordered'
    yield assert_equal, newcmap.output_coords.name, 'output'
    yield assert_equal, order, (1, 2, 0, 3)
    yield assert_equal, newcmap.affine, aff


@dec.needs_review('Needs review of coordmap4io function')
def test_iljk_to_xyzt():
    # this should raise a warning about the first three input
    # coordinates, and one about the last axis not being in the
    # correct order. this also will give a warning about the pixdim.

    # ninput_axes = list('iljk')
    ninput_axes = [input_axes[0], input_axes[3], input_axes[1], input_axes[2]]
    input_coords = CoordinateSystem(ninput_axes, 'input')
    output_coords = CoordinateSystem(output_axes[:4], 'output')
    cmap = Affine(np.identity(5), input_coords, output_coords)
    newcmap, order, pixdim, diminfo = nifti.coordmap4io(cmap)
    aff = np.array([[ 1,  0,  0,  0,  0,],
                    [ 0,  0,  1,  0,  0,],
                    [ 0,  0,  0,  1,  0,],
                    [ 0,  1,  0,  0,  0,],
                    [ 0,  0,  0,  0,  1,]])
    yield assert_equal, newcmap.affine, aff
    yield assert_equal, newcmap.input_coords.name, 'input-reordered'
    # order should match a reorder to 'ijkl'
    yield assert_equal, order, (0,2,3,1)


@dec.needs_review('Needs review of coordmap4io function')
def test_ijkn_to_xyzt():
    # This should raise an exception about not having axis names
    # ['ijkl'].  Some warnings are printed during the try/except

    # ninput_axes = list('ijkn')
    ninput_axes = [input_axes[0], input_axes[1], input_axes[2], input_axes[5]]
    input_coords = CoordinateSystem(ninput_axes, 'input')
    output_coords = CoordinateSystem(output_axes[:4], 'output')
    cmap = Affine(np.identity(5), input_coords, output_coords)
    assert_raises, ValueError, nifti.coordmap4io, cmap


@dec.needs_review('Needs review of coordmap4io function')
def test_ijkml_to_xyztu():
    # This should raise a warning about the last 2 axes not being in
    # order, and one about the loss of information from a non-diagonal
    # matrix. This also means that the pixdim will be wrong
    
    # ninput_axes = list('ijkml')
    ninput_axes = [input_axes[0], input_axes[1], input_axes[2], input_axes[4],
                   input_axes[3]]
    input_coords = CoordinateSystem(ninput_axes, 'input')
    output_coords = CoordinateSystem(output_axes[:5], 'output')
    cmap = Affine(np.identity(6), input_coords, output_coords)
    newcmap, order, pixdim, diminfo = nifti.coordmap4io(cmap)
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


@dec.needs_review('Needs review of coordmap4io function')
def test_ijkml_to_utzyx():
    # This should raise a warning about the last 2 axes not being in
    # order, and one about the loss of information from a non-diagonal
    # matrix, and also one about the nifti output coordinates.  Again,
    # this will have a pixdim warning

    # ninput_axes = list('ijkml')
    ninput_axes = [input_axes[0], input_axes[1], input_axes[2], input_axes[4],
                   input_axes[3]]
    input_coords = CoordinateSystem(ninput_axes, 'input')
    output_coords = CoordinateSystem(output_axes[:5][::-1], 'output')
    cmap = Affine(np.identity(6), input_coords, output_coords)
    newcmap, order, pixdim, diminfo = nifti.coordmap4io(cmap)
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


@dec.needs_review('Needs review of coordmap4io function')
def test_jkil_to_xyzt():
    # This will issue a warning about the pixdims

    # ninput_axes = list('jkil')
    ninput_axes = [input_axes[1], input_axes[2], input_axes[0], input_axes[3]]
    input_coords = CoordinateSystem(ninput_axes, 'input')
    output_coords = CoordinateSystem(output_axes[:4], 'output')
    cmap = Affine(np.diag(range(1,5) + [1]), input_coords, output_coords)
    newcmap, order, pixdim, diminfo = nifti.coordmap4io(cmap)
    yield assert_equal, newcmap.input_coords.name, 'input-reordered'
    yield assert_equal, newcmap.output_coords.name, 'output'
    yield assert_equal, order, (2,0,1,3)
    yield assert_equal, newcmap.input_coords.coord_names, tuple('ijkl')
    aff = np.array([[ 0,  0,  1,  0,  0,],
                    [ 2,  0,  0,  0,  0,],
                    [ 0,  3,  0,  0,  0,],
                    [ 0,  0,  0,  4,  0,],
                    [ 0,  0,  0,  0,  1,]])
    yield assert_equal, newcmap.affine, aff


def test_ijk_from_diminfo():
    x = nifti._diminfo_from_fps(-1,-1,-1)
    yield assert_equal, nifti.ijk_from_diminfo(x), list('ijk')
    x = nifti._diminfo_from_fps(2,-1,-1)
    yield assert_equal, nifti.ijk_from_diminfo(x), list('jki')


def test_phase_freq_slice_axes():
    # Test that the phase, freq, time, slice axes work for valid NIFTI headers
    ninput_axes = [input_axes[1], input_axes[2], input_axes[0], input_axes[3]]
    input_coords = CoordinateSystem(ninput_axes, 'input')
    output_coords = CoordinateSystem(output_axes[:4], 'output')
    cmap = Affine(np.identity(5), input_coords, output_coords)

    yield assert_equal, nifti.get_time_axis(cmap), 3
    yield assert_equal, nifti.get_freq_axis(cmap), 0
    yield assert_equal, nifti.get_slice_axis(cmap), 1
    yield assert_equal, nifti.get_phase_axis(cmap), 2


def test_phase_freq_slice_axes_rev():
    # Same test_phase_freq_slice_axes, but the order of the output
    # coordinates is reversed

    ninput_axes = [input_axes[1], input_axes[2], input_axes[0], input_axes[3]]
    input_coords = CoordinateSystem(ninput_axes, 'input')
    output_coords = CoordinateSystem(output_axes[:4][::-1], 'output')
    cmap = Affine(np.identity(5), input_coords, output_coords)

    yield assert_equal, nifti.get_time_axis(cmap), 3
    yield assert_equal, nifti.get_freq_axis(cmap), 0
    yield assert_equal, nifti.get_slice_axis(cmap), 1
    yield assert_equal, nifti.get_phase_axis(cmap), 2


def test_phase_freq_slice_axes_coerce():
    # Test that the phase, freq, time, slice axes work for coercable
    # NIFTI headers

    ninput_axes = [input_axes[0], input_axes[1], input_axes[2], input_axes[4],
                   input_axes[3]]
    input_coords = CoordinateSystem(ninput_axes, 'input')
    output_coords = CoordinateSystem(output_axes[:5][::-1], 'output')
    cmap = Affine(np.identity(6), input_coords, output_coords)

    cmap = Affine(np.identity(6), input_coords, output_coords)

    yield assert_equal, nifti.get_time_axis(cmap), 4
    yield assert_equal, nifti.get_freq_axis(cmap), 1
    yield assert_equal, nifti.get_slice_axis(cmap), 2
    yield assert_equal, nifti.get_phase_axis(cmap), 0

    newcmap, _ = nifti.coerce_coordmap(cmap)

    yield assert_equal, nifti.get_time_axis(newcmap), 3
    yield assert_equal, nifti.get_freq_axis(newcmap), 1
    yield assert_equal, nifti.get_slice_axis(newcmap), 2
    yield assert_equal, nifti.get_phase_axis(newcmap), 0
