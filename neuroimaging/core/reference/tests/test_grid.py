import numpy as N
from numpy.testing import NumpyTest, NumpyTestCase

from neuroimaging.core.reference.grid import SamplingGrid, ConcatenatedGrids, \
     ConcatenatedIdenticalGrids
from neuroimaging.core.reference.mapping import Affine

from neuroimaging.data_io.api import Analyze
from neuroimaging.testing import anatfile, funcfile

from neuroimaging.core.api import load_image

from neuroimaging.utils.test_decorators import slow

class test_Grid(NumpyTestCase):

    def setUp(self):
        self.img = load_image(anatfile)

    def test_concat(self):
        self.fail("concatenated and replicated SamplingGrids need to be fixed")

        grids = ConcatenatedGrids([self.img.grid]*5)
        self.assertEquals(tuple(grids.shape), (5,) + tuple(self.img.grid.shape))
        z = grids.mapping([4,5,6,7])
        a = grids.subgrid(0)
        x = a.mapping([5,6,7])

    def test_replicate(self):
        self.fail("concatenated and replicated SamplingGrids need to be fixed")

        grids = self.img.grid.replicate(4)
        self.assertEquals(tuple(grids.shape), (4,) + tuple(self.img.grid.shape))
        z = grids.mapping([2,5,6,7])
        a = grids.subgrid(0)
        x = a.mapping([5,6,7])

    def test_replicate2(self):
        """
        Test passing
        """
        self.fail("concatenated and replicated SamplingGrids need to be fixed")
        grids = self.img.grid.replicate(4)
        grids.python2matlab()

    def test_concat2(self):
        """
        Test passing
        """
        self.fail("concatenated and replicated SamplingGrids need to be fixed")
        
        grids = ConcatenatedIdenticalGrids(self.img.grid, 4)
        grids.python2matlab()

    def test_concat3(self):
        """
        Test failing
        """
        self.fail("concatenated and replicated SamplingGrids need to be fixed")
        grids = ConcatenatedGrids([self.img.grid]*4)
        grids.python2matlab()

    def test_identity(self):
        shape = (30,40,50)
        i = SamplingGrid.identity(['zspace', 'yspace', 'xshape'], shape=shape)
        self.assertEquals(tuple(i.shape), shape)
        y = i.mapping([3,4,5])
        N.testing.assert_almost_equal(y, N.array([3,4,5]))

    def test_identity2(self):
        shape = (30, 40)
        self.assertRaises(ValueError, SamplingGrid.identity, ['zspace', 'yspace', 'xspace'], shape)


    def test_from_affine(self):
        a = Affine.identity(2)
        g = SamplingGrid.from_affine(a, ['zspace', 'xspace'], (20,30))


from neuroimaging.utils.testutils import make_doctest_suite
test_suite = make_doctest_suite('neuroimaging.core.reference.grid')

if __name__ == '__main__':
    NumpyTest.run()
