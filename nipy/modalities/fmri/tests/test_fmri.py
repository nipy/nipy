import gc, os
from tempfile import mkstemp
import warnings

import numpy as np

import nose.tools

from nipy.modalities.fmri.api import fmri_generator, FmriImageList
from nipy.core.api import parcels, fromarray
from nipy.io.api import  load_image, save_image
from nipy.testing import funcfile


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
    yield nose.tools.assert_equal, test[0].affine.shape, (4,4)
    yield nose.tools.assert_equal, img[0].affine.shape, (5,4)
    yield nose.tools.assert_true, np.allclose(test[0].affine, img[0].affine[1:])
    # Under windows, if you don't close before delete, you get a
    # locking error.
    os.close(fp)
    os.remove(fname)


def test_iter():
    img = load_image(funcfile)

    j = 0
    for i, d in fmri_generator(img):
        j += 1
        nose.tools.assert_equal(d.shape, (20,20,20))
        del(i); gc.collect()
    nose.tools.assert_equal(j, 2)


def test_subcoordmap():
    img = load_image(funcfile)
    subcoordmap = img[3].coordmap
    xform = img.coordmap.affine[:,1:]
    nose.tools.assert_true(np.allclose(subcoordmap.affine[1:], xform[1:]))
    ## XXX FIXME: why is it [0,0] entry instead of [0] below?
    nose.tools.assert_true(np.allclose(subcoordmap.affine[0], [0,0,0,img.coordmap([3,0,0,0])[0,0]]))
        

def test_labels1():
    img = load_image(funcfile)
    parcelmap = fromarray(np.asarray(img[0]), 'kji', 'zyx')    
    parcelmap = (np.asarray(parcelmap) * 100).astype(np.int32)
    v = 0
    for i, d in fmri_generator(img, parcels(parcelmap)):
        v += d.shape[1]
    nose.tools.assert_equal(v, parcelmap.size)
