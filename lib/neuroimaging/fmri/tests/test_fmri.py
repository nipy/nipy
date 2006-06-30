import unittest, gc, os
import numpy as N

from neuroimaging.fmri import fMRIImage
from neuroimaging.image import Image
from neuroimaging.tests.data import repository

# not a test until test data is found
class fMRITest(unittest.TestCase):

    def setUp(self):
        self.rho = Image(repository.filename('rho.img'))
        self.img = fMRIImage("test_fmri.img", datasource=repository)

    #def test_TR(self):
    #    tmp = N.around(self.rho.readall() * (self.nmax / 2.)) / (self.nmax / 2.)
    #    tmp.shape = N.product(tmp.shape)
    #    tmp = N.com
    #    x = self.img.frametimes

    def test_write(self):
        self.img.tofile('tmpfmri.img')
        os.remove('tmpfmri.img')
        os.remove('tmpfmri.hdr')

    def test_iter(self):
        j = 0
        for i in iter(self.img):
            j += 1
            self.assertEquals(i.shape, (120,128,128))
            del(i); gc.collect()
        self.assertEquals(j, 13)

    def test_subgrid(self):
        subgrid = self.img.grid.subgrid(3)
        N.testing.assert_almost_equal(subgrid.mapping.transform,
                                          self.img.grid.mapping.transform[1:,1:])

    def test_labels1(self):
        parcelmap = (self.rho.readall() * 100).astype(N.int32)

        self.img.grid.itertype = 'parcel'
        self.img.grid.parcelmap = parcelmap
        parcelmap.shape = N.product(parcelmap.shape)
        self.img.grid.parcelseq = N.unique(parcelmap)

        v = 0
        for t in self.img:
            v += t.shape[1]
        self.assertEquals(v, N.product(parcelmap.shape))

    def test_labels2(self):
        parcelmap = (self.rho.readall() * 100).astype(N.int32)

        self.rho.grid.itertype = 'parcel'
        self.rho.grid.parcelmap = parcelmap
        parcelmap.shape = N.product(parcelmap.shape)
        self.rho.grid.parcelseq = N.unique(parcelmap)

        v = 0
        for t in self.rho:
            v += t.shape[0]
        self.assertEquals(v, N.product(parcelmap.shape))

if __name__ == '__main__':
    unittest.main()
