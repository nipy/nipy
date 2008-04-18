import gc, os
import numpy as N
from numpy.testing import NumpyTest, NumpyTestCase

import neuroimaging.core.reference.axis as axis
import neuroimaging.core.reference.grid as grid

from neuroimaging.utils.test_decorators import slow, data

from neuroimaging.modalities.fmri.api import FmriImage, fmri_generator, fromimage
from neuroimaging.core.api import Image, load_image, data_generator, parcels, save_image
from neuroimaging.testing import anatfile, funcfile
from neuroimaging.data_io.api import Analyze


# not a test until test data is found
class test_fMRI(NumpyTestCase):

    def setUp(self):
        self.parcels = load_image(funcfile)[0]
        self.img = load_image(funcfile)
        self.fmri = fromimage(self.img)

    def data_setUp(self):
        self.img = load_image(funcfile)

    def test_write(self):
        self.fail('this is a problem with reading/writing so-called "mat" files -- all the functions for these have been moved from core.reference.mapping to data_io.formats.analyze -- and they need to be fixed because they do not work. the names of the functions are: matfromstr, matfromfile, matfrombin, matfromxfm, mattofile')
        save_image(self.img, 'tmpfmri.hdr', format=Analyze)
        test = fromimage(load_image('tmpfmri.hdr', format=Analyze))
        self.assertEquals(test[0].grid.shape, self.img[0].grid.shape)
        os.remove('tmpfmri.img')
        os.remove('tmpfmri.hdr')

    def test_iter(self):
        j = 0
        for i, d in fmri_generator(self.img):
            j += 1
            self.assertEquals(d.shape, (20,20,20))
            del(i); gc.collect()
        self.assertEquals(j, 2)

    def test_subgrid(self):
        subgrid = self.img.grid[3]
        N.testing.assert_almost_equal(subgrid.mapping.transform,
                                      [[0., 0., 0., -49.21875],
                                       [-7., 0., 0., 7.],
                                       [0., -2.34375, 0., 53.90625],
                                       [0., 0., -2.34375, 53.90625],
                                       [0, 0, 0, 1]])

    def test_labels1(self):
        parcelmap = (N.asarray(self.parcels) * 100).astype(N.int32)
        
        v = 0
        for i, d in fmri_generator(self.img, parcels(parcelmap)):
            v += d.shape[1]
        self.assertEquals(v, parcelmap.size)


from neuroimaging.utils.testutils import make_doctest_suite
test_suite = make_doctest_suite('neuroimaging.modalities.fmri.fmri')


if __name__ == '__main__':
    NumpyTest.run()
