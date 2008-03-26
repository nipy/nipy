import numpy.random as R
import numpy as N
from numpy.testing import NumpyTest, NumpyTestCase

import urllib, os
from tempfile import mkstemp

from neuroimaging.core.reference import mapping, mni
from neuroimaging.utils.test_decorators import slow

class test_Mapping2(NumpyTestCase):
    def setUp(self):
        def f(x):
            return 2*x
        def g(x):
            return x/2.0
        self.a = mapping.Mapping(f)
        self.b = mapping.Mapping(f, g)
        self.c = mapping.Mapping(g)        
        self.d = mapping.Mapping(g, f)        

    def test_call(self):
        value = N.array([1., 2., 3.])
        result_a = self.a(value)
        result_b = self.b(value)
        result_c = self.c(value)
        result_d = self.c(value)        
        N.testing.assert_almost_equal(result_a, 2*value)
        N.testing.assert_almost_equal(result_b, 2*value)
        N.testing.assert_almost_equal(result_c, value/2)
        N.testing.assert_almost_equal(result_d, value/2)        
        
    def test_str(self):
        s_a = str(self.a)
        s_b = str(self.b)
        s_c = str(self.c)
        s_d = str(self.d)                
        
    def test_eq(self):
        eq = lambda a, b: a == b
        neq = lambda a, b: a != b
        self.assertRaises(NotImplementedError, eq, self.a, self.b)
        self.assertRaises(NotImplementedError, neq, self.a, self.b)

    def test_mul(self):
        value = N.array([1., 2., 3.])

        aa = self.a*self.a
        self.assertFalse(aa.isinvertible())
        N.testing.assert_almost_equal(aa(value), 4*value)

        ab = self.a*self.b
        self.assertFalse(ab.isinvertible())
        N.testing.assert_almost_equal(ab(value), 4*value)

        ac = self.a*self.c
        self.assertFalse(ac.isinvertible())
        N.testing.assert_almost_equal(ac(value), value)

        ad = self.a*self.d
        self.assertFalse(ad.isinvertible())
        N.testing.assert_almost_equal(ad(value), value)

        ba = self.b*self.a
        self.assertFalse(ba.isinvertible())
        N.testing.assert_almost_equal(ba(value), 4*value)

        bb = self.b*self.b
        self.assertTrue(bb.isinvertible())
        N.testing.assert_almost_equal(bb(value), 4*value)
        N.testing.assert_almost_equal(bb.inverse()(value), value/4)        

        bc = self.b*self.c
        self.assertFalse(bc.isinvertible())
        N.testing.assert_almost_equal(bc(value), value)

        bd = self.b*self.d
        self.assertTrue(bd.isinvertible())
        N.testing.assert_almost_equal(bd(value), value)
        N.testing.assert_almost_equal(bd.inverse()(value), value)        

        ca = self.c*self.a
        self.assertFalse(ca.isinvertible())
        N.testing.assert_almost_equal(ca(value), value)

        cb = self.c*self.b
        self.assertFalse(cb.isinvertible())
        N.testing.assert_almost_equal(cb(value), value)

        cc = self.c*self.c
        self.assertFalse(cc.isinvertible())
        N.testing.assert_almost_equal(cc(value), value/4)

        cd = self.c*self.d
        self.assertFalse(cd.isinvertible())
        N.testing.assert_almost_equal(cd(value), value/4)

        da = self.d*self.a
        self.assertFalse(da.isinvertible())
        N.testing.assert_almost_equal(da(value), value)

        db = self.d*self.b
        self.assertTrue(db.isinvertible())
        N.testing.assert_almost_equal(db(value), value)
        N.testing.assert_almost_equal(db.inverse()(value), value)        

        dc = self.d*self.c
        self.assertFalse(dc.isinvertible())
        N.testing.assert_almost_equal(dc(value), value/4)

        dd = self.d*self.d
        self.assertTrue(dd.isinvertible())
        N.testing.assert_almost_equal(dd(value), value/4)
        N.testing.assert_almost_equal(dd.inverse()(value), 4*value)        

    def test_isinvertible(self):
        self.assertFalse(self.a.isinvertible())
        self.assertTrue(self.b.isinvertible())
        self.assertFalse(self.c.isinvertible())
        self.assertTrue(self.d.isinvertible())
        
    def test_inverse(self):
        inv = lambda a: a.inverse()
        self.assertRaises(AttributeError, inv, self.a)
        self.assertRaises(AttributeError, inv, self.c)
        inv_b = self.b.inverse()
        inv_d = self.d.inverse()
        ident_b = inv_b*self.b
        ident_d = inv_d*self.d
        value = N.array([1., 2., 3.])
        N.testing.assert_almost_equal(ident_b(value), value)
        N.testing.assert_almost_equal(ident_d(value), value)
        
      
        
    def test_tovoxel(self):
        value = N.array([2., 4, 6.])
        value2 = N.array([[2., 4, 6.], [2, 4, 6]])
        tovox = lambda a: a.tovoxel(value)
        self.assertRaises(AttributeError, tovox, self.a)
        self.assertRaises(AttributeError, tovox, self.c)        
        
        vox_b = self.b.tovoxel(value)
        N.testing.assert_almost_equal(vox_b, [1., 2., 3.])
        vox_d = self.d.tovoxel(value)
        N.testing.assert_almost_equal(vox_d, [4., 8., 12.])        

        vox_b = self.b.tovoxel(value2)
        N.testing.assert_almost_equal(vox_b, [[1., 2., 3.], [1,2,3]])
        vox_d = self.d.tovoxel(value2)
        N.testing.assert_almost_equal(vox_d, [[4., 8., 12.], [4,8,12]])        

    def test_python2matlab1(self):
        v = R.standard_normal((3,))
        z = self.a(v)
        p = self.a.python2matlab()
        z_ = p(N.array(v[::-1])+1)[::-1]
        N.testing.assert_almost_equal(z, z_)
        
    def test_python2matlab2(self):
        value = N.array([1., 2., 3.])
        for mat_ in [self.a, self.b, self.c, self.d]:
            p = mat_.python2matlab()
            q = p.matlab2python()
            N.testing.assert_almost_equal(q(value), mat_(value))
        
    def test_python2matlab3(self):
        value = N.array([1., 2., 3.])
        for mat_ in [self.a, self.b, self.c, self.d]:
            p = mat_.matlab2python()
            q = p.python2matlab()
            N.testing.assert_almost_equal(q(value), mat_(value))


class test_Identity(NumpyTestCase):
    def setUp(self):
        self.a = mapping.Affine.identity()
        
    def test_call(self):
        value = N.array([1., 2., 3.])
        N.testing.assert_almost_equal(self.a(value), value)
    
    def test_eq(self):
        self.assertTrue(self.a == mapping.Affine.identity())
        
    def test_mul(self):
        value = N.array([1., 2., 3.])
        b = self.a * self.a
        N.testing.assert_almost_equal(b(value), value)

    def test_str(self):
        s = str(self.a)
        
    def test_invertable(self):
        self.assertTrue(self.a.isinvertible())
        
    def test_inverse(self):
        self.assertTrue(self.a == self.a.inverse().inverse())
        
    def test_tofile(self):
        self.a.tofile("tmp.mat")
        b = mapping.Affine.fromfile("tmp.mat")
        self.assertTrue(self.a == b)
        os.remove("tmp.mat")
        
class test_Affine(NumpyTestCase):
    def setUp(self):    
        a = mapping.Affine.identity()
        A = N.identity(4)
        A[0:3] = R.standard_normal((3,4))
        self.mapping = mapping.Affine(a.input_coords, a.output_coords, A)


class test_Mapping(NumpyTestCase):

    def setUp(self):

        def f(x):
            return 2*x
        self.a = mapping.Mapping(f)

        a = mapping.Affine.identity()
        A = N.identity(4)
        A[0:3] = R.standard_normal((3,4))
        self.mapping = mapping.Affine(A)

        self.singular = mapping.Affine(N.array([[ 0,  1,  2,  3],
                                                [ 4,  5,  6,  7],
                                                [ 8,  9, 10, 11],
                                                [ 8,  9, 10, 11]]))
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

    def test_matvec_trasform(self):
        m1 = R.standard_normal((3, 3))
        v1 = R.standard_normal((3,))
        m2, v2 = mapping._2matvec(mapping._2transform(m1, v1))
        N.testing.assert_almost_equal(m1, m2)
        N.testing.assert_almost_equal(v1, v2)        
        
    @slow
    def test_fromurl(self):
        x = mapping.fromurl('http://kff.stanford.edu/nipy/testdata/fiac3_fonc1.txt')
        y = mapping.fromurl('http://kff.stanford.edu/nipy/testdata/fiac3_fonc1_0089.mat')
        N.testing.assert_almost_equal(x, y, decimal=5)

        z = mapping.fromurl('http://kff.stanford.edu/FIAC/fiac0/fonc1/fsl/example_func2highres.xfm')
        N.testing.assert_almost_equal(z[0], [[-0.058193, -3.47651, 0.827575, 225.012],
                                             [0.0612378, 0.658445, 4.01176, 37.1965],
                                             [-2.11749, -0.0100964, 0.0288848, 129.901],
                                             [0, 0, 0, 1]])
        N.testing.assert_almost_equal(z[1], [[0.898438, 0, 0, 0],
                                             [0, -1, 0, 256],
                                             [0, 0, 1.42188, 0],
                                             [0, 0, 0, 1]])

    def test_tofromfile(self):
        (_, filename) = mkstemp(suffix='.csv')
        self.mapping.tofile(filename)
        a = mapping.Affine.fromfile(filename)
        os.remove(filename)        
        N.testing.assert_almost_equal(self.mapping.transform, a.transform)

    def test___str__(self):
        s = str(self.mapping)

    def test___eq__(self):
        self.assertTrue(self.mapping == self.mapping)
        self.assertTrue(not self.mapping != self.mapping)
        self.assertTrue(not self.a == self.mapping)

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

    def test_tovoxel(self):
        voxel = [1,2,3]
        real = self.mapping(voxel)
        v = self.mapping.tovoxel(real)
        N.testing.assert_almost_equal(v, voxel)

        tovox = lambda a: a.tovoxel(real)
        self.assertRaises(N.linalg.LinAlgError, tovox, self.singular)
        

    def test_isinvertable(self):
        self.assertTrue(self.mapping.isinvertible())
        self.assertTrue(not self.singular.isinvertible())

from neuroimaging.utils.testutils import make_doctest_suite
test_suite = make_doctest_suite('neuroimaging.core.reference.mapping')        
        
if __name__ == '__main__':
    NumpyTest.run()
