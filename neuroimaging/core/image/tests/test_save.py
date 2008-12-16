import os
import numpy as np

import nose.tools

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
    nose.tools.assert_true(np.allclose(img.affine, img2.affine))
    nose.tools.assert_true(img.shape == img2.shape)
    nose.tools.assert_true(np.allclose(np.asarray(img2), np.asarray(img)))

def test_save2():
    """
    A test to ensure that when a file is saved, the affine
    and the data agree. This image comes from a NIFTI file

    The axes have to be reordered because save_image
    first does the usual 'python2matlab' reorder
    """

    shape = (13,5,7,3)
    step = np.array([3.45,2.3,4.5,6.93])

    cmap = api.Affine.from_start_step('lkji', 'tzyx', [0]*4, step)

    data = np.random.standard_normal(shape)
    img = api.Image(data, cmap)
    save_image(img, 'tmp.nii')
    img2 = load_image('tmp.nii')
    nose.tools.assert_true(np.allclose(img.affine, img2.affine))
    nose.tools.assert_true(img.shape == img2.shape)
    nose.tools.assert_true(np.allclose(np.asarray(img2), np.asarray(img)))
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
    cmap = api.Affine.from_start_step('lkji', 'tzyx', [0]*4, step)

    data = np.random.standard_normal(shape)
    img = api.Image(data, cmap)
    save_image(img, 'tmp.nii')
    img2 = load_image('tmp.nii')
    nose.tools.assert_true(np.allclose(img.affine, img2.affine))
    nose.tools.assert_true(img.shape == img2.shape)
    nose.tools.assert_true(np.allclose(np.asarray(img2), np.asarray(img)))
    return img, img2

def test_save2b():
    """
    A test to ensure that when a file is saved, the affine
    and the data agree. This image comes from a NIFTI file

    This example has a non-diagonal affine matrix for the
    spatial part, but is 'diagonal' for the space part.

    this should raise a warnings about 'non-diagonal' affine matrix
    """

    # make a 5x5 transformatio
    step = np.array([3.45,2.3,4.5,6.9])
    A = np.random.standard_normal((4,4))
    B = np.diag(list(step)+[1])
    B[:4,:4] = A

    shape = (13,5,7,3)
    cmap = api.Affine.from_start_step('lkji', 'tzyx', [0]*4, step)
    cmap.affine = B
    data = np.random.standard_normal(shape)

    img = api.Image(data, cmap)

    save_image(img, 'tmp.nii')
    img2 = load_image('tmp.nii')
    nose.tools.assert_true(not np.allclose(img.affine, img2.affine))
    nose.tools.assert_true(img.shape == img2.shape)
    nose.tools.assert_true(np.allclose(np.asarray(img2), np.asarray(img)))
    return img, img2

def test_save3():
    """
    A test to ensure that when a file is saved, the affine
    and the data agree. In this case, things don't agree:
    i) the pixdim is off
    ii) makes the affine off

    """

    step = np.array([3.45,2.3,4.5,6.9])
    shape = (13,5,7,3)
    cmap = api.Affine.from_start_step('jkli', 'tzyx', [0]*4, step)

    data = np.random.standard_normal(shape)
    img = api.Image(data, cmap)
    save_image(img, 'tmp.nii')
    img2 = load_image('tmp.nii')
    nose.tools.assert_true(tuple([img.shape[l] for l in [2,1,0,3]]) == img2.shape)
    a = np.transpose(np.asarray(img), [2,1,0,3])
    nose.tools.assert_true(not np.allclose(img.affine, img2.affine))
    nose.tools.assert_true(np.allclose(a, np.asarray(img2)))


def teardown():
    os.remove('tmp.nii')

def test_save4():
    """
    Same as test_save3 except we have reordered the 'ijk' input axes.

    """

    shape = (13,5,7,3)
    step = np.array([3.45,2.3,4.5,6.9])
    cmap = api.Affine.from_start_step('ljki', 'tzyx', [0]*4, step)

    data = np.random.standard_normal(shape)

    img = api.Image(data, cmap)
    save_image(img, 'tmp.nii')
    img2 = load_image('tmp.nii')
    nose.tools.assert_true(np.allclose(img.affine, img2.affine))
    nose.tools.assert_true(img.shape == img2.shape)
    nose.tools.assert_true(np.allclose(np.asarray(img2), np.asarray(img)))
    print img2.coordmap.input_coords.axisnames, img.coordmap.input_coords.axisnames
    print nifti.get_diminfo(img.coordmap), nifti.get_diminfo(img2.coordmap)
    print img2.header['dim_info']
    nose.tools.assert_true(img2.coordmap.input_coords.axisnames == img.coordmap.input_coords.axisnames)
    nose.tools.assert_true(img2.coordmap.input_coords.axisnames == ['l', 'j', 'k', 'i'])


