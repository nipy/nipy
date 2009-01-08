import numpy as np
import nose.tools

import neuroimaging.core.api as api
from neuroimaging.core.reference import nifti, coordinate_system
reload(nifti)

shape = range(1,8)
step = np.arange(1,8)

output_axes = [coordinate_system.Coordinate(s) for i, s in enumerate('xyztuvw')]
input_axes = [coordinate_system.Coordinate(s) for i, s in enumerate('ijklmno')]
input_coords = api.CoordinateSystem('input', input_axes)

def test_validate1():

    """
    this should work without any warnings
    """

    output_coords = api.CoordinateSystem('output', output_axes[:4])
    input_coords = api.CoordinateSystem('input', input_axes[:4])
    cmap = api.Affine(np.diag(list(step[:4]) + [1]), input_coords, output_coords)
    newcmap, order, pixdim, diminfo = nifti.coordmap4io(cmap)
    nose.tools.assert_true(newcmap.input_coords.name == 'input')
    nose.tools.assert_true(newcmap.output_coords.name == 'output')
    nose.tools.assert_true(order == (0,1,2,3))

@np.testing.dec.knownfailureif(True)
def test_validate1a():

    """
    this should work without any warnings, except PIXDIM will fail
    """

    output_coords = api.CoordinateSystem('output', output_axes[:3])
    input_coords = api.CoordinateSystem('input', input_axes[:3][::-1])
    cmap = api.Affine(np.diag(list(step[:3]) + [1]), input_coords, output_coords)
    newcmap, order, pixdim, diminfo = nifti.coordmap4io(cmap)
    nose.tools.assert_true(newcmap.input_coords.name == 'input')
    nose.tools.assert_true(newcmap.output_coords.name == 'output')
    nose.tools.assert_true(order == (0,1,2))

    # One last test to remind us of the FIXME in pixdims

    nose.tools.assert_true(np.allclose(pixdim, np.arange(3)+1))

def test_validate1b():

    """
    this should work without any warnings
    """

    output_coords = api.CoordinateSystem('output', output_axes[:4])
    input_coords = api.CoordinateSystem('input', [input_axes[2],
                                                  input_axes[0],
                                                  input_axes[1],
                                                  input_axes[3]])
    cmap = api.Affine(np.identity(5), input_coords, output_coords)
    newcmap, order, pixdim, diminfo = nifti.coordmap4io(cmap)
    nose.tools.assert_true(newcmap.input_coords.name == 'input')
    nose.tools.assert_true(newcmap.output_coords.name == 'output')
    nose.tools.assert_true(order == (0,1,2,3))

def test_validate2():
    """
    this should raise a warning about the first three input coordinates,
    and one about the last axis not being in the correct order. this also
    will give a warning about the pixdim.
    """

    ninput_axes = [input_axes[0], input_axes[3], input_axes[1], input_axes[2]]
    input_coords = api.CoordinateSystem('input', ninput_axes)
    output_coords = api.CoordinateSystem('output', output_axes[:4])
    cmap = api.Affine(np.identity(5), input_coords, output_coords)
    newcmap, order, pixdim, diminfo = nifti.coordmap4io(cmap)
    nose.tools.assert_true(newcmap.input_coords.name == 'input-reordered')
    nose.tools.assert_true(order == (0,2,3,1))
    return newcmap, order, diminfo

def test_validate3():
    """
    this should raise an exception about
    not having axis names ['ijkl'].

    some warnings are printed during the try/except
    """

    ninput_axes = [input_axes[0], input_axes[1], input_axes[2], input_axes[5]]
    input_coords = api.CoordinateSystem('input', ninput_axes)
    output_coords = api.CoordinateSystem('output', output_axes[:4])
    cmap = api.Affine(np.identity(5), input_coords, output_coords)
    try:
        nifti.coordmap4io(cmap)
    except:
        return
    raise ValueError, 'an exception should have been raised earlier' 

def test_validate4():
    """
    this should raise a warning about the last 2 axes not being in order,
    and one about the loss of information from a non-diagonal
    matrix. this also means that the pixdim will be wrong
    """

    ninput_axes = [input_axes[0], input_axes[1], input_axes[2], input_axes[4],
                   input_axes[3]]
    input_coords = api.CoordinateSystem('input', ninput_axes)
    output_coords = api.CoordinateSystem('output', output_axes[:5])
    cmap = api.Affine(np.identity(6), input_coords, output_coords)

    newcmap, order, pixdim, diminfo = nifti.coordmap4io(cmap)
    nose.tools.assert_true(newcmap.input_coords.name == 'input-reordered')
    nose.tools.assert_true(order == (0,1,2,4,3))

    ndim = cmap.ndim[0]
    perm = np.zeros((ndim+1,ndim+1))
    perm[-1,-1] = 1
    for i, j in enumerate(order):
        perm[i,j] = 1
    B = np.dot(np.identity(6), perm)

    nose.tools.assert_true(np.allclose(newcmap.affine, B))
    X = np.random.standard_normal((5,))
    Xr = [X[i] for i in order]
    nose.tools.assert_true(np.allclose(newcmap(Xr), cmap(X)))
    return newcmap, order, pixdim, diminfo

def test_validate5():
    """
    this should raise a warning about the last 2 axes not being in order,
    and one about the loss of information from a non-diagonal
    matrix, and also one about the nifti output coordinates. 
    again, this will have a pixdim warning like test_validate4
    """

    ninput_axes = [input_axes[0], input_axes[1], input_axes[2], input_axes[4],
                   input_axes[3]]
    input_coords = api.CoordinateSystem('input', ninput_axes)
    output_coords = api.CoordinateSystem('output', output_axes[:5][::-1])
    cmap = api.Affine(np.identity(6), input_coords, output_coords)
    newcmap, order, pixdim, diminfo = nifti.coordmap4io(cmap)
    nose.tools.assert_true(newcmap.input_coords.name == 'input-reordered')
    nose.tools.assert_true(newcmap.output_coords.name == 'output-reordered')
    nose.tools.assert_true(order == (0,1,2,4,3))

    ndim = cmap.ndim[0]
    perm = np.zeros((ndim+1,ndim+1))
    perm[-1,-1] = 1
    for i, j in enumerate(order):
        perm[i,j] = 1
    B = np.dot(np.identity(6), perm)

    r = np.zeros((6,6))
    r[5,5] =1.
    for i in range(5):
        r[i, 4-i] = 1.

    nose.tools.assert_true(np.allclose(newcmap.affine, 
                                       np.dot(r, B)))
    X = np.random.standard_normal((5,))
    Xr = [X[i] for i in order]
    nose.tools.assert_true(np.allclose(newcmap(Xr)[::-1], cmap(X)))


def test_validate6():
    """
    this should not raise any warnings
    """


    ninput_axes = [input_axes[1], input_axes[2], input_axes[0], input_axes[3]]
    input_coords = api.CoordinateSystem('input', ninput_axes)
    output_coords = api.CoordinateSystem('output', output_axes[:4])
    cmap = api.Affine(np.identity(5), input_coords, output_coords)
    newcmap, order, pixdim, diminfo = nifti.coordmap4io(cmap)
    nose.tools.assert_true(newcmap.input_coords.name == 'input')
    nose.tools.assert_true(newcmap.output_coords.name == 'output')
    nose.tools.assert_true(order == (0,1,2,3))

    nose.tools.assert_true(newcmap.input_coords.axisnames == ['j','k','i','l'])


def test_validate7():
    """
    same as test_validate6, but should raise
    a warning about negative pixdim
    """

    output_axes = [coordinate_system.Coordinate(s) for i, s in enumerate('xyztuvw')]
    ninput_axes = [input_axes[1], input_axes[2], input_axes[0], input_axes[3]]
    input_coords = api.CoordinateSystem('input', ninput_axes)
    output_coords = api.CoordinateSystem('output', output_axes[:4])
    cmap = api.Affine(np.diag(list(step[:4]) + [1]), input_coords, output_coords)
    newcmap, order, pixdim, diminfo = nifti.coordmap4io(cmap)
    nose.tools.assert_true(newcmap.input_coords.name == 'input')
    nose.tools.assert_true(newcmap.output_coords.name == 'output')
    nose.tools.assert_true(order == (0,1,2,3))
    nose.tools.assert_true(newcmap.input_coords.axisnames == ['j','k','i','l'])

def test_ijk1():
    assert(nifti.ijk_from_diminfo(nifti._diminfo_from_fps(-1,-1,-1)) == list('ijk'))
    assert(nifti.ijk_from_diminfo(nifti._diminfo_from_fps(2,-1,-1)) == list('jki'))

def test_ijk2():
    """
    Test that the phase, freq, time, slice axes work for valid NIFTI headers
    """


    ninput_axes = [input_axes[1], input_axes[2], input_axes[0], input_axes[3]]
    input_coords = api.CoordinateSystem('input', ninput_axes)
    output_coords = api.CoordinateSystem('output', output_axes[:4])
    cmap = api.Affine(np.identity(5), input_coords, output_coords)

    nose.tools.assert_true(nifti.get_time_axis(cmap) == 3)
    nose.tools.assert_true(nifti.get_freq_axis(cmap) == 0)
    nose.tools.assert_true(nifti.get_slice_axis(cmap) == 1)
    nose.tools.assert_true(nifti.get_phase_axis(cmap) == 2)

def test_ijk3():
    '''
    Same as test_ijk2, but the order of the output coordinates is reversed
    '''

    ninput_axes = [input_axes[1], input_axes[2], input_axes[0], input_axes[3]]
    input_coords = api.CoordinateSystem('input', ninput_axes)
    output_coords = api.CoordinateSystem('output', output_axes[:4][::-1])
    cmap = api.Affine(np.identity(5), input_coords, output_coords)

    nose.tools.assert_true(nifti.get_time_axis(cmap) == 3)
    nose.tools.assert_true(nifti.get_freq_axis(cmap) == 0)
    nose.tools.assert_true(nifti.get_slice_axis(cmap) == 1)
    nose.tools.assert_true(nifti.get_phase_axis(cmap) == 2)

def test_ijk4():
    """
    Test that the phase, freq, time, slice axes work for coercable NIFTI headers
    """

    ninput_axes = [input_axes[0], input_axes[1], input_axes[2], input_axes[4],
                   input_axes[3]]
    input_coords = api.CoordinateSystem('input', ninput_axes)
    output_coords = api.CoordinateSystem('output', output_axes[:5][::-1])
    cmap = api.Affine(np.identity(6), input_coords, output_coords)

    cmap = api.Affine(np.identity(6), input_coords, output_coords)

    nose.tools.assert_true(nifti.get_time_axis(cmap) == 4)
    nose.tools.assert_true(nifti.get_freq_axis(cmap) == 1)
    nose.tools.assert_true(nifti.get_slice_axis(cmap) == 2)
    nose.tools.assert_true(nifti.get_phase_axis(cmap) == 0)

    newcmap, _ = nifti.coerce_coordmap(cmap)

    nose.tools.assert_true(nifti.get_time_axis(newcmap) == 3)
    nose.tools.assert_true(nifti.get_freq_axis(newcmap) == 1)
    nose.tools.assert_true(nifti.get_slice_axis(newcmap) == 2)
    nose.tools.assert_true(nifti.get_phase_axis(newcmap) == 0)
