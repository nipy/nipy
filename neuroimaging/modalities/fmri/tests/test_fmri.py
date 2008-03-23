import gc, os
import numpy as N
from numpy.testing import NumpyTest, NumpyTestCase

import neuroimaging.core.reference.axis as axis
import neuroimaging.core.reference.grid as grid

from neuroimaging.utils.test_decorators import slow, data

from neuroimaging.modalities.fmri.api import FmriImage
from neuroimaging.core.api import Image, load_image, parcel_iterator
from neuroimaging.utils.tests.data import repository
from neuroimaging.data_io.api import Analyze


# not a test until test data is found
class test_fMRI(NumpyTestCase):

    def setUp(self):
        self.rho = load_image(repository.filename('rho.hdr'), format=Analyze)
        self.img = load_image(repository.filename("test_fmri.hdr"))

    def data_setUp(self):
        self.img = load_image(repository.filename("test_fmri.hdr"))

    #def test_TR(self):
    #    tmp = N.around(self.rho.readall() * (self.nmax / 2.)) / (self.nmax / 2.)
    #    tmp.shape = tmp.size
    #    tmp = N.com
    #    x = self.img.frametimes

    @slow
    @data
    def test_write(self):
        self.img.tofile('tmpfmri.hdr', format=Analyze)
        test = FmriImage('tmpfmri.hdr', format=Analyze)
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
        parcelmap = (self.rho[:] * 100).astype(N.int32)
        
        v = 0
        for t in parcel_iterator(self.rho, parcelmap):
            v += t.shape[1]
        self.assertEquals(v, parcelmap.size)

    def test_labels2(self):
        parcelmap = (self.rho[:] * 100).astype(N.int32)

        v = 0
        for t in parcel_iterator(self.rho, parcelmap):
            v += t.shape[0]

        self.assertEquals(v, parcelmap.size)

from neuroimaging.utils.testutils import make_doctest_suite
test_suite = make_doctest_suite('neuroimaging.modalities.fmri.fmri')


if __name__ == '__main__':
    NumpyTest.run()
