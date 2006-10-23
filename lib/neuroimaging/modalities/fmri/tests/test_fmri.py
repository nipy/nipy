import unittest, gc, os
import numpy as N

from neuroimaging.modalities.fmri import fMRIImage
from neuroimaging.core.image.image import Image
from neuroimaging.utils.tests.data import repository
from neuroimaging.data_io.formats.analyze import Analyze

# not a test until test data is found
class fMRITest(unittest.TestCase):

    def setUp(self):
        self.rho = Image(repository.filename('rho.hdr'))
        self.img = fMRIImage("test_fmri.hdr", datasource=repository)

    #def test_TR(self):
    #    tmp = N.around(self.rho.readall() * (self.nmax / 2.)) / (self.nmax / 2.)
    #    tmp.shape = tmp.size
    #    tmp = N.com
    #    x = self.img.frametimes

    def test_write(self):
        self.img.tofile('tmpfmri.hdr', format=Analyze)
        test = fMRIImage('tmpfmri.hdr', format=Analyze)
        self.assertEquals(test.grid.shape, self.img.grid.shape)
        os.remove('tmpfmri.img')
        os.remove('tmpfmri.hdr')

    def test_iter(self):
        j = 0
        for i in iter(self.img):
            j += 1
            self.assertEquals(i.shape, (120,1,128,128))
            del(i); gc.collect()
        self.assertEquals(j, 13)

    def test_subgrid(self):
        subgrid = self.img.grid.subgrid(3)
        N.testing.assert_almost_equal(subgrid.mapping.transform,
                                          self.img.grid.mapping.transform[1:,1:])

    def test_labels1(self):
        parcelmap = (self.rho.readall() * 100).astype(N.int32)
        
        self.img.grid.set_iter_param("itertype", "parcel")
        self.img.grid.set_iter_param("parcelmap", parcelmap)
        parcelmap.shape = parcelmap.size
        self.img.grid._parcelseq = N.unique(parcelmap)

        v = 0
        for t in self.img:
            v += t.shape[1]
        self.assertEquals(v, parcelmap.size)

    def test_labels2(self):
        parcelmap = (self.rho.readall() * 100).astype(N.int32)

        self.rho.grid.set_iter_param("itertype", "parcel")
        self.rho.grid.set_iter_param("parcelmap", parcelmap)
        parcelmap.shape = parcelmap.size
        self.rho.grid._parcelseq = N.unique(parcelmap)

        v = 0
        for t in self.rho:
            v += t.shape[0]
        self.assertEquals(v, parcelmap.size)

if __name__ == '__main__':
    unittest.main()
