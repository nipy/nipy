import unittest, gc, os

import numpy as N
import scipy

from neuroimaging import fmri
from neuroimaging.image import Image
from neuroimaging.reference import grid
from neuroimaging.tests.data import repository

# not a test until test data is found
class fMRITest(unittest.TestCase):

    def setUp(self):
        self.url = 'http://kff.stanford.edu/BrainSTAT/testdata/test_fmri.img'
        self.img = fmri.fMRIImage(self.url)

    def test_TR(self):
        x = self.img.frametimes

    def test_write(self):
        self.img.tofile('tmpfmri.img')
        os.remove('tmpfmri.img')
        os.remove('tmpfmri.hdr')

    def test_iter(self):
        I = iter(self.img)
        j = 0
        for i in I:
            j += 1
            self.assertEquals(i.shape, (120,128,128))
            del(i); gc.collect()
        self.assertEquals(j, 13)

    def test_subgrid(self):
        subgrid = self.img.grid.subgrid(3)
        matlabgrid = grid.python2matlab(subgrid)
        scipy.testing.assert_almost_equal(matlabgrid.mapping.transform,
                                          N.diag([2.34375,2.34375,7,1]))


    def test_labels1(self):
        rho = Image('rho.img', datasource=repository)
        labels = (rho.readall() * 100).astype(N.Int)

        self.img.grid.itertype = 'parcel'
        self.img.grid.labels = labels
        labels.shape = N.product(labels.shape)
        self.img.grid.labelset = set(N.unique(labels))

        v = 0
        for t in self.img:
            v += t.shape[1]
        self.assertEquals(v, N.product(labels.shape))


if __name__ == '__main__':
    unittest.main()
