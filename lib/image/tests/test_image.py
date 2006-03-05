import unittest, os, scipy
import numpy as N
from neuroimaging.image import Image

class AnalyzeImageTest(unittest.TestCase):

    def setUp(self):
        imgname = '/usr/share/BrainSTAT/repository/kff.stanford.edu/~jtaylo/BrainSTAT/avg152T1.img'
        self.img = Image(imgname)

    def test_analyze(self):
        y = self.img.readall()
        self.assertEquals(y.shape, tuple(self.img.grid.shape))
        y.shape = N.product(y.shape)
        self.assertEquals(N.maximum.reduce(y), 437336.375)
        self.assertEquals(N.minimum.reduce(y), 0.)

    def test_slice1(self):
        x = self.img.getslice(3)
        self.assertEquals(x.shape, tuple(self.img.grid.shape[1:]))
        
    def test_slice2(self):
        x = self.img.getslice(slice(3,5))
        self.assertEquals(x.shape, (2,) + tuple(self.img.grid.shape[1:]))

    def test_slice3(self):
        s = slice(0,20,2)
        x = self.img.getslice(s)
        self.assertEquals(x.shape, (10,) + tuple(self.img.grid.shape[1:]))

    def test_slice4(self):
        s = slice(0,self.img.grid.shape[0])
        x = self.img.getslice(s)
        self.assertEquals(x.shape, tuple((self.img.grid.shape)))

    def test_array(self):
        x = self.img.toarray()
        
    def test_file(self):
        try:
            os.remove('tmp.img')
            os.remove('tmp.hdr')
        except:
            pass
        x = self.img.tofile('tmp.img')
        os.remove('tmp.img')
        os.remove('tmp.hdr')

    def test_nondiag(self):
        try:
            os.remove('tmp.img')
            os.remove('tmp.hdr')
            os.remove('tmp.mat')
        except:
            pass
        self.img.grid.warp.transform[0,1] = 3.0
        x = self.img.tofile('tmp.img')
        scipy.testing.assert_almost_equal(x.grid.warp.transform, self.img.grid.warp.transform)
        os.remove('tmp.img')
        os.remove('tmp.hdr')
        os.remove('tmp.mat')

        
    def test_clobber(self):
        try:
            os.remove('tmp.img')
            os.remove('tmp.hdr')
        except:
            pass
        x = self.img.tofile('tmp.img', clobber=True)
        a = Image('tmp.img')
        A = a.readall()
        I = self.img.readall()
        z = N.add.reduce(((A-I)**2).flat)
        self.assertEquals(z, 0.)
        t = a.grid.warp.transform
        b = self.img.grid.warp.transform
        os.remove('tmp.img')
        os.remove('tmp.hdr')
        scipy.testing.assert_almost_equal(b, t)

        
    def test_iter(self):
        I = iter(self.img)
        for i in I:
            self.assertEquals(i.shape, (109,91))


if __name__ == '__main__':
    unittest.main()
