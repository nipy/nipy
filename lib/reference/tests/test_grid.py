import unittest, scipy
from neuroimaging.reference import axis, grid, coordinate_system
from neuroimaging.image.formats import analyze
import numpy as N

class GridTest(unittest.TestCase):

    def _open(self):
        imgname = '/usr/share/BrainSTAT/repository/kff.stanford.edu/~jtaylo/BrainSTAT/avg152T1.img'
        self.img = analyze.ANALYZE(filename=imgname)

    def test_concat(self):
        self._open()
        grids = grid.ConcatenatedGrids([self.img.grid]*5)
        self.assertEquals(tuple(grids.shape), (5,) + tuple(self.img.grid.shape))
        z = grids.warp(N.transpose([4,5,6,7]))
        a = grids.subgrid(0)
        x = a.warp(N.transpose([5,6,7]))

    def test_duplicate(self):
        self._open()
        grids = grid.DuplicatedGrids(self.img.grid,4)
        self.assertEquals(tuple(grids.shape), (4,) + tuple(self.img.grid.shape))
        z = grids.warp(N.transpose([2,5,6,7]))
        a = grids.subgrid(0)
        x = a.warp(N.transpose([5,6,7]))

    def test_identity(self):
        shape = (30,40,50)
        i = grid.IdentityGrid(shape=shape, names=axis.space)
        self.assertEquals(tuple(i.shape), shape)
        y = i.warp.map([3,4,5])
        scipy.testing.assert_almost_equal(y, N.array([3,4,5]))

if __name__ == '__main__':
    unittest.main()
