import unittest, gc, os
import numpy as N

from neuroimaging.utils.test_decorators import slow, data

from neuroimaging.modalities.fmri import fMRIImage
from neuroimaging.core.api import Image
from neuroimaging.utils.tests.data import repository
from neuroimaging.data_io.formats.analyze import Analyze

from neuroimaging.core.reference.iterators import ParcelIterator, fMRIParcelIterator

# not a test until test data is found
class fMRITest(unittest.TestCase):

    def setUp(self):
        self.rho = Image(repository.filename('rho.hdr'))
        #self.img = fMRIImage("test_fmri.hdr", datasource=repository)

    def data_setUp(self):
        self.img = fMRIImage("test_fmri.hdr", datasource=repository)

    #def test_TR(self):
    #    tmp = N.around(self.rho.readall() * (self.nmax / 2.)) / (self.nmax / 2.)
    #    tmp.shape = tmp.size
    #    tmp = N.com
    #    x = self.img.frametimes

    @slow
    @data
    def test_write(self):
        self.img.tofile('tmpfmri.hdr', format=Analyze)
        test = fMRIImage('tmpfmri.hdr', format=Analyze)
        self.assertEquals(test.grid.shape, self.img.grid.shape)
        os.remove('tmpfmri.img')
        os.remove('tmpfmri.hdr')

    @data
    def test_iter(self):
        j = 0
        for i in self.img.slice_iterator():
            j += 1
            self.assertEquals(i.shape, (120,128,128))
            del(i); gc.collect()
        self.assertEquals(j, 13)

    @data
    def test_subgrid(self):
        subgrid = self.img.grid.subgrid(3)
        N.testing.assert_almost_equal(subgrid.mapping.transform,
                                          self.img.grid.mapping.transform[1:,1:])

    @slow
    @data
    def test_labels1(self):
        parcelmap = (self.rho.readall() * 100).astype(N.int32)
        
        it = fMRIParcelIterator(self.img, parcelmap)
        v = 0
        for t in it:
            v += t.shape[1]
        self.assertEquals(v, parcelmap.size)

    def test_labels2(self):
        parcelmap = (self.rho.readall() * 100).astype(N.int32)

        it = ParcelIterator(self.rho, parcelmap)
        v = 0
        for t in it:
            v += t.shape[0]

        self.assertEquals(v, parcelmap.size)

if __name__ == '__main__':
    unittest.main()
