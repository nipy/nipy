import os
import numpy as np

from neuroimaging.testing import funcfile
from neuroimaging.core.api import load_image, save_image
from neuroimaging.core import api
from neuroimaging.core.reference import nifti

def test_save1():
    """
    A test to ensure that when a file is saved, the affine
    and the data agree. This image comes from a NIFTI file
    """
    img = load_image(funcfile)
    save_image(img, 'tmp.nii')
    img2 = load_image('tmp.nii')
    assert np.allclose(img.affine, img2.affine)
    assert img.shape == img2.shape
    assert np.allclose(np.asarray(img2), np.asarray(img))

def test_save2():
    """
    A test to ensure that when a file is saved, the affine
    and the data agree. This image comes from a NIFTI file

    The axes have to be reordered because save_image
    first does the usual 'python2matlab' reorder
    """

    shape = (13,5,7,3)
    step = np.array([3.45,2.3,4.5,6.93])

    output_axes = [api.RegularAxis(s, step=step[::-1][i]) for i, s in enumerate('xyzt')][::-1]
    output_coords = api.DiagonalCoordinateSystem('output', output_axes)

    input_axes = [api.VoxelAxis(s, length=shape[i]) for i, s in enumerate('ijkl')][::-1]
    input_coords = api.VoxelCoordinateSystem('input', input_axes)
    print 'here'
    cmap = api.CoordinateMap(api.Affine(output_coords.affine), input_coords, output_coords)
    print 'now'

    data = np.random.standard_normal(shape)
    img = api.Image(data, cmap)
    save_image(img, 'tmp.nii')
    img2 = load_image('tmp.nii')
    assert np.allclose(img.affine, img2.affine)
    assert img.shape == img2.shape
    assert np.allclose(np.asarray(img2), np.asarray(img))
    return img, img2

def test_save2a():
    """
    A test to ensure that when a file is saved, the affine
    and the data agree. This image comes from a NIFTI file

    This example has a non-diagonal affine matrix for the
    spatial part, but is 'diagonal' for the space part.

    this should raise no warnings.
    """

    # make a 5x5 transformatio
    step = np.array([3.45,2.3,4.5,6.9])
    A = np.random.standard_normal((3,3))
    B = np.diag(list(step)+[1])
    B[1:4,1:4] = A

    shape = (13,5,7,3)
    output_axes = [api.RegularAxis(s, step=step[::-1][i]) for i, s in enumerate('xyzt')][::-1]
    output_coords = api.DiagonalCoordinateSystem('output', output_axes)

    input_axes = [api.VoxelAxis(s, length=shape[i]) for i, s in enumerate('ijkl')][::-1]
    input_coords = api.VoxelCoordinateSystem('input', input_axes)

    cmap = api.CoordinateMap(api.Affine(B), input_coords, output_coords)

    data = np.random.standard_normal(shape)
    img = api.Image(data, cmap)
    save_image(img, 'tmp.nii')
    img2 = load_image('tmp.nii')
    assert np.allclose(img.affine, img2.affine)
    assert img.shape == img2.shape
    assert np.allclose(np.asarray(img2), np.asarray(img))
    return img, img2

def test_save3():
    """
    A test to ensure that when a file is saved, the affine
    and the data agree. In this case, things don't agree:
    i) the pixdim is off
    ii) makes the affine off

    """

    3,13,5,7
    shape = (13,5,7,3)
    output_axes = [api.RegularAxis(s, step=i+1) for i, s in enumerate('tzyx')]
    output_coords = api.DiagonalCoordinateSystem('output', output_axes)

    input_axes = [api.VoxelAxis(s, length=shape[i]) for i, s in enumerate('jkli')]
    input_coords = api.VoxelCoordinateSystem('input', input_axes)
    cmap = api.CoordinateMap(api.Affine(output_coords.affine), input_coords, output_coords)

    data = np.random.standard_normal(shape)
    img = api.Image(data, cmap)
    save_image(img, 'tmp.nii')
    img2 = load_image('tmp.nii')
    assert tuple([img.shape[l] for l in [2,1,0,3]]) == img2.shape
    a = np.transpose(np.asarray(img), [2,1,0,3])
    assert not np.allclose(img.affine, img2.affine)
    assert not np.allclose(nifti.get_pixdim(img.coordmap),
                           nifti.get_pixdim(img2.coordmap))
    assert np.allclose(a, np.asarray(img2))


def teardown():
    os.remove('tmp.nii')

def test_save4():
    """
    Failing (at least partly) because diminfo is not being set properly.
    Same as test_save3 except we have reordered the 'ijk' input axes.

    """

    shape = (13,5,7,3)
    output_axes = [api.RegularAxis(s, step=i+1) for i, s in enumerate('xyzt')][::-1]
    output_coords = api.DiagonalCoordinateSystem('output', output_axes)

    input_axes = [api.VoxelAxis(s, length=shape[i]) for i, s in enumerate('ikjl')][::-1]
    input_coords = api.VoxelCoordinateSystem('input', input_axes)
    cmap = api.CoordinateMap(api.Affine(output_coords.affine), input_coords, output_coords)

    data = np.random.standard_normal(shape)
    img = api.Image(data, cmap)
    save_image(img, 'tmp.nii')
    img2 = load_image('tmp.nii')
    assert np.allclose(img.affine, img2.affine)
    assert img.shape == img2.shape
    assert np.allclose(np.asarray(img2), np.asarray(img))
    print img2.coordmap.input_coords.axisnames(), img.coordmap.input_coords.axisnames()
    print nifti.get_diminfo(img.coordmap), nifti.get_diminfo(img2.coordmap)
    print img2.header['dim_info']
    assert img2.coordmap.input_coords.axisnames() == img.coordmap.input_coords.axisnames()
    assert img2.coordmap.input_coords.axisnames() == ['l', 'j', 'k', 'i']


test_save2()
