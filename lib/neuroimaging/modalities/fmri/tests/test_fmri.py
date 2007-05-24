import gc, os
import numpy as N
from numpy.testing import NumpyTest, NumpyTestCase

import neuroimaging.core.reference.axis as axis
import neuroimaging.core.reference.grid as grid

from neuroimaging.utils.test_decorators import slow, data

from neuroimaging.modalities.fmri.api import fMRIImage, fMRIParcelIterator, \
   fMRISliceParcelIterator
from neuroimaging.core.api import Image, ParcelIterator
from neuroimaging.utils.tests.data import repository
from neuroimaging.data_io.api import Analyze


# not a test until test data is found
class test_fMRI(NumpyTestCase):

    def setUp(self):
        self.rho = Image(repository.filename('rho.hdr'), format=Analyze)
        #self.img = fMRIImage("test_fmri.hdr", datasource=repository)

    def data_setUp(self):
        self.img = fMRIImage("test_fmri.hdr", datasource=repository, format=Analyze)

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


class test_Iterators(NumpyTestCase):

    def setUp(self):
        self.img = fMRIImage(N.zeros((3, 4, 5, 6)), grid = grid.SamplingGrid.identity((3,4,5,6), axis.spacetime))

    def test_fmri_parcel(self):
        parcelmap = N.zeros(self.img.shape[1:])
        parcelmap[0,0,0] = 1
        parcelmap[1,1,1] = 1
        parcelmap[2,2,2] = 1
        parcelmap[1,2,1] = 2
        parcelmap[2,3,2] = 2
        parcelmap[0,1,0] = 2
        parcelseq = (0, 1, 2, 3)
        
        expected = [N.product(self.img.shape[1:]) - 6, 3, 3, 0]
        iterator = fMRIParcelIterator(self.img, parcelmap, parcelseq)

        for i, slice_ in enumerate(iterator):
            self.assertEqual((self.img.shape[0], expected[i],), slice_.shape)

        iterator = fMRIParcelIterator(self.img, parcelmap)
        for i, slice_ in enumerate(iterator):
            self.assertEqual((self.img.shape[0], expected[i],), slice_.shape)

    def test_fmri_parcel_write(self):
        parcelmap = N.zeros(self.img.shape[1:])
        parcelmap[0,0,0] = 1
        parcelmap[1,1,1] = 1
        parcelmap[2,2,2] = 1
        parcelmap[1,2,1] = 2
        parcelmap[2,3,2] = 2
        parcelmap[0,1,0] = 2
        parcelseq = (0, 1, 2, 3)
        expected = [N.product(self.img.shape[1:]) - 6, 3, 3, 0]
        iterator = fMRIParcelIterator(self.img, parcelmap, parcelseq, mode='w')

        for i, slice_ in enumerate(iterator):
            value = N.asarray([N.arange(expected[i]) for _ in range(self.img.shape[0])])
            slice_.set(value)

        iterator = fMRIParcelIterator(self.img, parcelmap, parcelseq)
        for i, slice_ in enumerate(iterator):
            self.assertEqual((self.img.shape[0], expected[i],), slice_.shape)
            N.testing.assert_equal(slice_, N.asarray([N.arange(expected[i]) for _ in range(self.img.shape[0])]))


        iterator = fMRIParcelIterator(self.img, parcelmap, mode='w')
        for i, slice_ in enumerate(iterator):
            value = N.asarray([N.arange(expected[i]) for _ in range(self.img.shape[0])])
            slice_.set(value)

        iterator = fMRIParcelIterator(self.img, parcelmap)
        for i, slice_ in enumerate(iterator):
            self.assertEqual((self.img.shape[0], expected[i],), slice_.shape)
            N.testing.assert_equal(slice_, N.asarray([N.arange(expected[i]) for _ in range(self.img.shape[0])]))


    def test_fmri_parcel_copy(self):
        parcelmap = N.zeros(self.img.shape[1:])
        parcelmap[0,0,0] = 1
        parcelmap[1,1,1] = 1
        parcelmap[2,2,2] = 1
        parcelmap[1,2,1] = 2
        parcelmap[2,3,2] = 2
        parcelmap[0,1,0] = 2
        parcelseq = (0, 1, 2, 3)
        expected = [N.product(self.img.shape[1:]) - 6, 3, 3, 0]
        iterator = fMRIParcelIterator(self.img, parcelmap, parcelseq)
        tmp = fMRIImage(self.img)

        new_iterator = iterator.copy(tmp)

        for i, slice_ in enumerate(new_iterator):
            self.assertEqual((self.img.shape[0], expected[i],), slice_.shape)

        iterator = fMRIParcelIterator(self.img, parcelmap)
        for i, slice_ in enumerate(new_iterator):
            self.assertEqual((self.img.shape[0], expected[i],), slice_.shape)

    def test_fmri_sliceparcel(self):
        parcelmap = N.asarray([[[0,0,0,1,2,2]]*5,
                               [[0,0,1,1,2,2]]*5,
                               [[0,0,0,0,2,2]]*5])
        parcelseq = ((1, 2), 0, 2)
        iterator = fMRISliceParcelIterator(self.img, parcelmap, parcelseq)
        for i, slice_ in enumerate(iterator):
            pm = parcelmap[i]
            ps = parcelseq[i]
            try:
                x = len([n for n in pm.flat if n in ps])
            except TypeError:
                x = len([n for n in pm.flat if n == ps])
            self.assertEqual(x, slice_.shape[1])
            self.assertEqual(self.img.shape[0], slice_.shape[0])

    def test_fmri_sliceparcel_write(self):
        parcelmap = N.asarray([[[0,0,0,1,2,2]]*5,
                               [[0,0,1,1,2,2]]*5,
                               [[0,0,0,0,2,2]]*5])
        parcelseq = ((1, 2), 0, 2)
        iterator = fMRISliceParcelIterator(self.img, parcelmap, parcelseq, mode='w')

        for i, slice_ in enumerate(iterator):
            pm = parcelmap[i]
            ps = parcelseq[i]
            try:
                x = len([n for n in pm.flat if n in ps])
            except TypeError:
                x = len([n for n in pm.flat if n == ps])
            value = [i*N.arange(x) for i in range(self.img.shape[0])]
            slice_.set(value)

        iterator = fMRISliceParcelIterator(self.img, parcelmap, parcelseq)
        for i, slice_ in enumerate(iterator):
            pm = parcelmap[i]
            ps = parcelseq[i]
            try:
                x = len([n for n in pm.flat if n in ps])
            except TypeError:
                x = len([n for n in pm.flat if n == ps])
            value = [i*N.arange(x) for i in range(self.img.shape[0])]
            self.assertEqual(x, slice_.shape[1])
            self.assertEqual(self.img.shape[0], slice_.shape[0])
            N.testing.assert_equal(slice_, value)


from neuroimaging.utils.testutils import make_doctest_suite
test_suite = make_doctest_suite('neuroimaging.modalities.fmri.fmri')


if __name__ == '__main__':
    NumpyTest.run()
