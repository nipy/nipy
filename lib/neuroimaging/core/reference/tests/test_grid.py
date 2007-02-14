import numpy as N
from numpy.testing import NumpyTest, NumpyTestCase

from neuroimaging.core.reference.axis import space
from neuroimaging.core.reference.grid import SamplingGrid, ConcatenatedGrids, \
     ConcatenatedIdenticalGrids
from neuroimaging.core.image.iterators import ParcelIterator, SliceParcelIterator
from neuroimaging.core.reference.mapping import Affine

from neuroimaging.data_io.formats.analyze import Analyze
from neuroimaging.utils.tests.data import repository

from neuroimaging.core.api import Image


class test_Grid(NumpyTestCase):

    def setUp(self):
        self.img = Image("avg152T1", datasource=repository, format=Analyze)

    def test_concat(self):
        grids = ConcatenatedGrids([self.img.grid]*5)
        self.assertEquals(tuple(grids.shape), (5,) + tuple(self.img.grid.shape))
        z = grids.mapping([4,5,6,7])
        a = grids.subgrid(0)
        x = a.mapping([5,6,7])

    def test_replicate(self):
        grids = self.img.grid.replicate(4)
        self.assertEquals(tuple(grids.shape), (4,) + tuple(self.img.grid.shape))
        z = grids.mapping([2,5,6,7])
        a = grids.subgrid(0)
        x = a.mapping([5,6,7])

    def test_replicate2(self):
        """
        Test passing
        """
        grids = self.img.grid.replicate(4)
        grids.python2matlab()

    def test_concat2(self):
        """
        Test passing
        """
        grids = ConcatenatedIdenticalGrids(self.img.grid, 4)
        grids.python2matlab()

    def test_concat3(self):
        """
        Test failing
        """
        grids = ConcatenatedGrids([self.img.grid]*4)
        grids.python2matlab()

    def test_identity(self):
        shape = (30,40,50)
        i = SamplingGrid.identity(shape=shape, names=space)
        self.assertEquals(tuple(i.shape), shape)
        y = i.mapping([3,4,5])
        N.testing.assert_almost_equal(y, N.array([3,4,5]))

    def test_identity2(self):
        shape = (30, 40)
        self.assertRaises(ValueError, SamplingGrid.identity, shape, space)

    def test_allslice(self):
        shape = (30,40,50)
        i = SamplingGrid.identity(shape=shape, names=space)
        i.allslice()
        
    def test_iterslices(self):
        for i in range(3):
            self.assertEqual(len(list(self.img.slice_iterator(axis=i))), self.img.grid.shape[i])
        
        parcelmap = N.zeros(self.img.grid.shape)
        parcelmap[:3,:5,:4] = 1
        parcelmap[3:10,5:10,4:10] = 2
        parcelseq = (1, (0,2))
        for i in ParcelIterator(self.img, parcelmap, parcelseq):
            pass

        parcelseq = (1, (1,2), 0) + (0,)*(len(parcelmap)-3)
        for i in SliceParcelIterator(self.img, parcelmap, parcelseq):
            pass

    def test_from_affine(self):
        a = Affine.identity()
        g = SamplingGrid.from_affine(a)

if __name__ == '__main__':
    NumpyTest.run()
