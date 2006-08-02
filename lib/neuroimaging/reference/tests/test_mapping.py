import unittest
import numpy.random as R
import numpy as N
from neuroimaging.reference import mapping

class MappingTest(unittest.TestCase):

    def _init(self):
        a = mapping.Mapping.identity()
        A = N.identity(4, N.float64)
        A[0:3] = R.standard_normal((3,4))
        self.mapping = mapping.Affine(a.input_coords, a.output_coords, A)

    def test_python2matlab1(self):
        self._init()
        v = R.standard_normal((3,))
        z = self.mapping(v)
        p = self.mapping.python2matlab()
        z_ = p(N.array(v[::-1])+1)[::-1]
        N.testing.assert_almost_equal(z, z_)
        
    def test_python2matlab2(self):
        self._init()
        p = self.mapping.python2matlab()
        q = p.matlab2python()
        N.testing.assert_almost_equal(q.transform, self.mapping.transform)
        
    def test_python2matlab3(self):
        self._init()
        p = self.mapping.matlab2python()
        q = p.python2matlab()
        N.testing.assert_almost_equal(q.transform, self.mapping.transform)
        
if __name__ == '__main__':
    unittest.main()
