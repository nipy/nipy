import unittest, gc, scipy
import neuroimaging.fmri as fmri
import neuroimaging.image as image
import neuroimaging.reference.grid as grid
import numpy as N

class fMRITest(unittest.TestCase):

    def setUp(self):
        self.url = 'http://kff.stanford.edu/BrainSTAT/testdata/test_fmri.img'
        self.img = fmri.fMRIImage(self.url)

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
        scipy.testing.assert_almost_equal(matlabgrid.warp.transform,
                                          N.diag([2.34375,2.34375,7,1]))


    def test_labels1(self):
        rho = image.Image('http://kff.stanford.edu/~jtaylo/BrainSTAT/rho.img')
        labels = (rho.readall() * 100).astype(N.Int)
        L = image.Image(labels, grid=rho.grid)

        self.img.grid.itertype = 'parcel'
        self.img.grid.labels = L
        v = 0
        for t in self.img:
            v += t.shape[1]
        self.assertEquals(v, N.product(L.grid.shape))



if __name__ == '__main__':
    unittest.main()
