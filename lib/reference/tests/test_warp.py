import unittest, scipy
import numpy.random as R
import numpy as N
from neuroimaging.reference import warp

class WarpTest(unittest.TestCase):

    def _init(self):
        a = warp.IdentityWarp()
        A = N.identity(4, N.Float)
        A[0:3] = R.standard_normal((3,4))
        self.warp = warp.Affine(a.input_coords, a.output_coords, A)

    def test_python2matlab1(self):
        self._init()
        v = R.standard_normal((3,))
        z = self.warp.map(v)
        p = warp.python2matlab(self.warp)
        z_ = p.map(N.array(v[::-1])+1)[::-1]
        scipy.testing.assert_almost_equal(z, z_)
        
    def test_python2matlab2(self):
        self._init()
        p = warp.python2matlab(self.warp)
        q = warp.matlab2python(p)
        scipy.testing.assert_almost_equal(q.transform, self.warp.transform)
        
    def test_python2matlab3(self):
        self._init()
        p = warp.matlab2python(self.warp)
        q = warp.python2matlab(p)
        scipy.testing.assert_almost_equal(q.transform, self.warp.transform)
        
if __name__ == '__main__':
    unittest.main()
