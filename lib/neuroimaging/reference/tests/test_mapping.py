import unittest
import numpy.random as R
import numpy as N

import urllib

from neuroimaging.reference import mapping

class MappingTest(unittest.TestCase):

    def setUp(self):
        a = mapping.Mapping.identity()
        A = N.identity(4, N.float64)
        A[0:3] = R.standard_normal((3,4))
        self.mapping = mapping.Affine(a.input_coords, a.output_coords, A)

    def test_python2matlab1(self):
        v = R.standard_normal((3,))
        z = self.mapping(v)
        p = self.mapping.python2matlab()
        z_ = p(N.array(v[::-1])+1)[::-1]
        N.testing.assert_almost_equal(z, z_)
        
    def test_python2matlab2(self):
        p = self.mapping.python2matlab()
        q = p.matlab2python()
        N.testing.assert_almost_equal(q.transform, self.mapping.transform)
        
    def test_python2matlab3(self):
        p = self.mapping.matlab2python()
        q = p.python2matlab()
        N.testing.assert_almost_equal(q.transform, self.mapping.transform)


    def test_isdiagonal(self):
        m = N.array([[1,0,0],
                     [0,1,0],
                     [0,0,1]])
        self.assertTrue(mapping.isdiagonal(m))

                      
    def test_frombin(self):
        # FIXME: this will only work on my (Tim) system. Need to sort out getting these
        # test files either up on the web or into a standard place
        #mat = urllib.urlopen('ftp://ftp.cea.fr/pub/dsv/madic/FIAC/fiac3/fiac3_fonc3.txt')
        f = open('/home/timl/src/ni/ni/trunk/lib/neuroimaging/reference/tests/fiac3_fonc3_0089.mat')
        tstr = f.read()
        mapping.frombin(tstr)

    def test_matfromstr(self):
        # FIXME: as above
        t1 = open('/home/timl/src/ni/ni/trunk/lib/neuroimaging/reference/tests/fiac3_fonc3_0089.mat').read()
        t2 = open('/home/timl/src/ni/ni/trunk/lib/neuroimaging/reference/tests/fiac3_fonc3.txt').read()
        a1 = mapping.matfromstr(t1)
        a2 = mapping.matfromstr(t2)
        N.testing.assert_almost_equal(a1, a2, 1e-6)

    def test_tofromfile(self):
        # FIXME: This will only work on linux (at a guess)
        self.mapping.tofile("/tmp/mapping.csv")
        a = mapping.Mapping.fromfile("/tmp/mapping.csv")
        N.testing.assert_almost_equal(self.mapping.transform, a.transform)

    def test___str__(self):
        print self.mapping

    def test___eq__(self):
        self.assertTrue(self.mapping == self.mapping)
        self.assertTrue(not self.mapping != self.mapping)

    def test_translation_transform(self):
        a = mapping.translation_transform([1,2,3], 3)
        b = N.array([[1,0,0,1],
                     [0,1,0,2],
                     [0,0,1,3],
                     [0,0,0,1]], dtype=N.float64)
        N.testing.assert_equal(a, b)

    def test_permutation_transform(self):
        order = [2,0,1]
        a = mapping.permutation_transform(order)
        b = N.array([[ 0.,  0.,  1.,  0.,],
                     [ 1.,  0.,  0.,  0.,],
                     [ 0.,  1.,  0.,  0.,],
                     [ 0.,  0.,  0.,  1.,]])
        N.testing.assert_equal(a, b)

        self.assertRaises(ValueError, mapping.permutation_transform, [3,0,1])

    def test_reslice(self):
        print self.mapping.input_coords.shape
        
if __name__ == '__main__':
    unittest.main()
