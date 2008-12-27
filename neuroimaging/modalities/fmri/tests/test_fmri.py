import gc, os
from tempfile import mkstemp
import numpy as np

import nose.tools

import neuroimaging.core.reference.coordinate_map as coordinate_map
from neuroimaging.modalities.fmri.api import FmriImageList, fmri_generator, fromimage
from neuroimaging.core.api import Image, load_image, data_generator, parcels, save_image
from neuroimaging.testing import anatfile, funcfile

edict = {}

def setup():
    edict['parcelmap'] = load_image(funcfile)[0]
    edict['img'] = load_image(funcfile)
    edict['fmri'] = fromimage(edict['img'])

def test_write():
    fp, fname = mkstemp('.nii')
    img = edict['img']
    save_image(img, fname)
    test = fromimage(load_image(fname))
    yield nose.tools.assert_equal, test[0].affine.shape, (4,4)
    yield nose.tools.assert_equal, img[0].affine.shape, (5,4)
    yield nose.tools.assert_true, np.allclose(test[0].affine, img[0].affine[1:])
    os.remove(fname)

def test_iter():
    img = edict['img']
    j = 0
    for i, d in fmri_generator(img):
        j += 1
        nose.tools.assert_equal(d.shape, (20,2,20))
        del(i); gc.collect()
    nose.tools.assert_equal(j, 20)

def test_subcoordmap():
    img = edict['img']
    subcoordmap = img[3].coordmap
        
    xform = np.array([[ 0., 0., 0., 10.35363007],
                      [-7.,  0., 0., 0.],
                      [ 0.,  -2.34375, 0., 0.],
                      [ 0.,  0., -2.34375, 0.],
                      [ 0.,  0., 0., 1.]])
        
    print img.affine
    nose.tools.assert_true(np.allclose(subcoordmap.affine, xform))
        
def test_labels1():
    parcelmap = edict['parcelmap']
    img = edict['img']
    parcelmap = (np.asarray(parcelmap) * 100).astype(np.int32)
        
    v = 0
    for i, d in fmri_generator(img, parcels(parcelmap)):
        v += d.shape[1]
    nose.tools.assert_equal(v, parcelmap.size)








