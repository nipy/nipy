import gc, os
from tempfile import mkstemp
import warnings

import numpy as np

from nipy.modalities.fmri.api import fmri_generator, FmriImageList
from nipy.core.api import parcels, fromarray
from nipy.io.api import  load_image, save_image

from nose.tools import assert_equal, assert_true

from nipy.testing import funcfile, parametric


def setup():
    # Suppress warnings during tests to reduce noise
    warnings.simplefilter("ignore")


def teardown():
    # Clear list of warning filters
    warnings.resetwarnings()


def test_write():
    fp, fname = mkstemp('.nii')
    img = load_image(funcfile)
    save_image(img, fname)
    test = FmriImageList.from_image(load_image(fname))
    yield assert_equal, test[0].affine.shape, (4,4)
    yield assert_equal, img[0].affine.shape, (5,4)

    # Check the affine...
    A = np.identity(4)
    A[:3,:3] = img[:,:,:,0].affine[:3,:3]
    A[:3,-1] = img[:,:,:,0].affine[:3,-1]
    yield assert_true, np.allclose(test[0].affine, A)

    # Under windows, if you don't close before delete, you get a
    # locking error.
    os.close(fp)
    os.remove(fname)


@parametric
def test_iter():
    img = load_image(funcfile)
    img_shape = img.shape
    exp_shape = (img_shape[0],) + img_shape[2:]
    j = 0
    for i, d in fmri_generator(img):
        j += 1
        yield assert_equal(d.shape, exp_shape)
        del(i); gc.collect()
    yield assert_equal(j, img_shape[1])


def test_subcoordmap():
    img = load_image(funcfile)
    subcoordmap = img[3].coordmap
    xform = img.affine[:,1:]
    assert_true(np.allclose(subcoordmap.affine[1:], xform[1:]))
    assert_true(np.allclose(subcoordmap.affine[0], [0,0,0,img.coordmap([3,0,0,0])[0]]))
        

def test_labels1():
    img = load_image(funcfile)
    parcelmap = fromarray(np.asarray(img[0]), 'kji', 'zyx')    
    parcelmap = (np.asarray(parcelmap) * 100).astype(np.int32)
    v = 0
    for i, d in fmri_generator(img, parcels(parcelmap)):
        v += d.shape[1]
    assert_equal(v, parcelmap.size)
