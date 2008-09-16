import gc, os
from tempfile import mkstemp
import numpy as np
from neuroimaging.testing import *

import neuroimaging.core.reference.axis as axis
import neuroimaging.core.reference.coordinate_map as coordinate_map

from neuroimaging.modalities.fmri.api import FmriImageList, fmri_generator, fromimage
from neuroimaging.core.api import Image, load_image, data_generator, parcels, save_image
from neuroimaging.testing import anatfile, funcfile


# not a test until test data is found
class test_fMRI(TestCase):

    def setUp(self):
        self.parcels = load_image(funcfile)[0]
        self.img = load_image(funcfile)
        self.fmri = fromimage(self.img)

    def data_setUp(self):
        self.img = load_image(funcfile)

    def test_write(self):
        #self.fail('this is a problem with reading/writing so-called "mat" files -- all the functions for these have been moved from core.reference.mapping to data_io.formats.analyze -- and they need to be fixed because they do not work. the names of the functions are: matfromstr, matfromfile, matfrombin, matfromxfm, mattofile')
        fp, fname = mkstemp('.nii')
        save_image(self.img, fname)
        test = fromimage(load_image(fname))
        self.assertEquals(test[0].coordmap.shape, self.img[0].coordmap.shape)
        os.remove(fname)

    def test_iter(self):
        j = 0
        for i, d in fmri_generator(self.img):
            j += 1
            self.assertEquals(d.shape, (20,2,20))
            del(i); gc.collect()
        self.assertEquals(j, 20)

    def test_subcoordmap(self):
        subcoordmap = self.img.coordmap[3]

        xform = np.array([[ 0., 0., 0., 10.35363007],
                          [-7.,  0., 0., 7.],
                          [ 0.,  -2.34375, 0., 53.90625],
                          [ 0.,  0., -2.34375, 53.90625],
                          [ 0.,  0., 0., 1.]])
        
        assert_almost_equal(subcoordmap.mapping.transform, xform)
        
    def test_labels1(self):
        parcelmap = (np.asarray(self.parcels) * 100).astype(np.int32)
        
        v = 0
        for i, d in fmri_generator(self.img, parcels(parcelmap)):
            v += d.shape[1]
        self.assertEquals(v, parcelmap.size)








