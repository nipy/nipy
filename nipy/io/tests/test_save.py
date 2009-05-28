import os
from tempfile import mkstemp
import numpy as np

from nipy.testing import assert_true, assert_false, assert_equal, \
    assert_array_almost_equal, funcfile


from nipy.io.api import load_image, save_image
from nipy.core import api

class Tempfile():
    file = None

tmpfile = Tempfile()

def setup():
    fd, tmpfile.name = mkstemp(suffix='.nii')
    tmpfile.file = open(tmpfile.name)

def teardown():
    tmpfile.file.close()
    os.unlink(tmpfile.name)


def test_save1():
    # A test to ensure that when a file is saved, the affine and the
    # data agree. This image comes from a NIFTI file

    img = load_image(funcfile)
    save_image(img, tmpfile.name)
    img2 = load_image(tmpfile.name)
    yield assert_true, np.allclose(img.affine, img2.affine)
    yield assert_equal, img.shape, img2.shape
    yield assert_true, np.allclose(np.asarray(img2), np.asarray(img))


def test_save2():
    # A test to ensure that when a file is saved, the affine and the
    # data agree. This image comes from a NIFTI file 

    shape = (13,5,7,3)
    step = np.array([3.45,2.3,4.5,6.93])

    cmap = api.Affine.from_start_step('ijkl', 'xyzt', [1,3,5,0], step)

    data = np.random.standard_normal(shape)
    img = api.Image(data, cmap)
    save_image(img, tmpfile.name)
    img2 = load_image(tmpfile.name)
    yield assert_true, np.allclose(img.affine, img2.affine)
    yield assert_equal, img.shape, img2.shape
    yield assert_true, np.allclose(np.asarray(img2), np.asarray(img))


def test_save2b():
    # A test to ensure that when a file is saved, the affine and the
    # data agree. This image comes from a NIFTI file This example has
    # a non-diagonal affine matrix for the spatial part, but is
    # 'diagonal' for the space part.  this should raise a warnings
    # about 'non-diagonal' affine matrix

    # make a 5x5 transformatio
    step = np.array([3.45,2.3,4.5,6.9])
    A = np.random.standard_normal((4,4))
    B = np.diag(list(step)+[1])
    B[:4,:4] = A

    shape = (13,5,7,3)
    cmap = api.Affine.from_params('ijkl', 'xyzt', B)

    data = np.random.standard_normal(shape)

    img = api.Image(data, cmap)

    save_image(img, tmpfile.name)
    img2 = load_image(tmpfile.name)
    yield assert_false, np.allclose(img.affine, img2.affine)
    yield assert_true, np.allclose(img.affine[:3,:3], img2.affine[:3,:3])
    yield assert_equal, img.shape, img2.shape
    yield assert_true, np.allclose(np.asarray(img2), np.asarray(img))


def test_save3():
    # A test to ensure that when a file is saved, the affine
    # and the data agree. In this case, things don't agree:
    # i) the pixdim is off
    # ii) makes the affine off

    step = np.array([3.45,2.3,4.5,6.9])
    shape = (13,5,7,3)
    cmap = api.Affine.from_start_step('jkli', 'tzyx', [0,3,5,1], step)

    data = np.random.standard_normal(shape)
    img = api.Image(data, cmap)
    save_image(img, tmpfile.name)
    img2 = load_image(tmpfile.name)

    yield assert_equal, tuple([img.shape[l] for l in [3,0,1,2]]), img2.shape
    a = np.transpose(np.asarray(img), [3,0,1,2])
    yield assert_false, np.allclose(img.affine, img2.affine)
    yield assert_true, np.allclose(a, np.asarray(img2))


def test_save4():
    # Same as test_save3 except we have reordered the 'ijk' input axes.
    shape = (13,5,7,3)
    step = np.array([3.45,2.3,4.5,6.9])
    # When the input coords are in the 'ljki' order, the affines get
    # rearranged.  Note that the 'start' below, must be 0 for
    # non-spatial dimensions, because we have no way to store them in
    # most cases.  For example, a 'start' of [1,5,3,1] would be lost on
    # reload
    cmap = api.Affine.from_start_step('lkji', 'tzyx', [0,5,3,1], step)
    data = np.random.standard_normal(shape)
    img = api.Image(data, cmap)
    save_image(img, tmpfile.name)
    img2 = load_image(tmpfile.name)
    P = np.array([[0,0,0,1,0],
                  [0,0,1,0,0],
                  [0,1,0,0,0],
                  [1,0,0,0,0],
                  [0,0,0,0,1]])
    res = np.dot(P, np.dot(img.affine, P.T))
    yield assert_array_almost_equal, res, img2.affine
    yield assert_equal, img.shape[::-1], img2.shape
    yield (assert_array_almost_equal, 
           np.transpose(np.asarray(img2),[3,2,1,0]),
           np.asarray(img))
    #print img2.coordmap.input_coords.coord_names, img.coordmap.input_coords.coord_names
    #print nifti.get_diminfo(img.coordmap), nifti.get_diminfo(img2.coordmap)
    #print img2.header['dim_info']
    yield assert_equal, img2.coordmap.input_coords.coord_names, \
        img.coordmap.input_coords.coord_names[::-1]
    yield assert_equal, img2.coordmap.input_coords.coord_names, \
        ['i', 'j', 'k', 'l']
