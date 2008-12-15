import numpy.random as R
import nose.tools
import numpy as np

import urllib, os
from tempfile import mkstemp

from neuroimaging.core.reference import mapping, mni


class empty:
    pass

E = empty()

def setup():
    def f(x):
        return 2*x
    def g(x):
        return x/2.0
    E.a = mapping.Mapping(f)
    E.b = mapping.Mapping(f, g)
    E.c = mapping.Mapping(g)        
    E.d = mapping.Mapping(g, f)        
    E.e = mapping.Affine.identity(3)

    A = np.identity(4)
    A[0:3] = R.standard_normal((3,4))
    E.mapping = mapping.Affine(A)
    
    E.singular = mapping.Affine(np.array([[ 0,  1,  2,  3],
                                          [ 4,  5,  6,  7],
                                          [ 8,  9, 10, 11],
                                          [ 8,  9, 10, 11]]))



def test_call():
    value = np.array([1., 2., 3.])
    result_a = E.a(value)
    result_b = E.b(value)
    result_c = E.c(value)
    result_d = E.c(value)        
    nose.tools.assert_true(np.allclose(result_a, 2*value))
    nose.tools.assert_true(np.allclose(result_b, 2*value))
    nose.tools.assert_true(np.allclose(result_c, value/2))
    nose.tools.assert_true(np.allclose(result_d, value/2) )       
        
def test_str():
    s_a = str(E.a)
    s_b = str(E.b)
    s_c = str(E.c)
    s_d = str(E.d)                
        
def test_eq():
    eq = lambda a, b: a == b
    neq = lambda a, b: a != b
    nose.tools.assert_raises(NotImplementedError, eq, a, b)
    nose.tools.assert_raises(NotImplementedError, neq, a, b)

def test_compose():
    value = np.array([1., 2., 3.])
    
    aa = mapping.compose(E.a, E.a)
    nose.tools.assert_false(aa.isinvertible)
    nose.tools.assert_true(np.allclose(aa(value), 4*value))

    ab = mapping.compose(E.a,E.b)
    nose.tools.assert_false(ab.isinvertible)
    nose.tools.assert_true(np.allclose(ab(value), 4*value))

    ac = mapping.compose(E.a,E.c)
    nose.tools.assert_false(ac.isinvertible)
    nose.tools.assert_true(np.allclose(ac(value), value))


    bb = mapping.compose(E.b,E.b)
    nose.tools.assert_true(bb.isinvertible)
    nose.tools.assert_true(np.allclose(bb(value), 4*value))
    nose.tools.assert_true(np.allclose(bb.inverse(value), value/4)        )

  

def test_isinvertible():
    nose.tools.assert_false(E.a.isinvertible)
    nose.tools.assert_true(E.b.isinvertible)
    nose.tools.assert_false(E.c.isinvertible)
    nose.tools.assert_true(E.d.isinvertible)
        
def test_inverse():
    inv = lambda a: a.inverse
    nose.tools.assert_raises(AttributeError, inv, a)
    nose.tools.assert_raises(AttributeError, inv, c)
    inv_b = b.inverse
    inv_d = d.inverse
    ident_b = mapping.compose(inv_b,b)
    ident_d = mapping.compose(inv_d,d)
    value = np.array([1., 2., 3.])
    nose.tools.assert_true(np.allclose(ident_b(value), value))
    nose.tools.assert_true(np.allclose(ident_d(value), value))
        
      

def test_call():
    value = np.array([1., 2., 3.])
    nose.tools.assert_true(np.allclose(E.e(value), value))
    
def test_eq():
    nose.tools.assert_true(E.e == mapping.Affine.identity(3))
        
def test_mul():
    value = np.array([1., 2., 3.])
    b = mapping.compose(E.e, E.e)
    nose.tools.assert_true(np.allclose(b(value), value))

def test_str():
    s = str(E.e)
    
def test_invertible():
    nose.tools.assert_true(E.e.isinvertible)
    
def test_inverse():
    nose.tools.assert_true(E.e == E.e.inverse.inverse)
        
       
# def test_Affine():
#     a = mapping.Affine.identity(3)
#     A = np.identity(4)
#     A[0:3] = R.standard_normal((3,4))
#     mx = mapping.Affine(a.input_coords, a.output_coords, A)





def test_matvec_trasform():
    m1 = R.standard_normal((3, 3))
    v1 = R.standard_normal((3,))
    m2, v2 = mapping.matvec_from_transform(mapping.transform_from_matvec(m1, v1))
    nose.tools.assert_true(np.allclose(m1, m2))
    nose.tools.assert_true(np.allclose(v1, v2))


def test___eq__():
    nose.tools.assert_true(E.mapping == E.mapping)
    nose.tools.assert_false(E.a == E.mapping)


def test_isinvertible():
    nose.tools.assert_true(E.mapping.isinvertible)
    nose.tools.assert_false(E.singular.isinvertible)



        


