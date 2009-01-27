import numpy as np
from neuroimaging.testing import *

import nose.tools

from neuroimaging.core.reference.coordinate_map import CoordinateMap, Affine, compose, CoordinateSystem, Coordinate
from neuroimaging.core.reference.coordinate_map import matvec_from_transform, transform_from_matvec
from neuroimaging.testing import anatfile, funcfile
from neuroimaging.core.api import load_image

def test_identity():
    i = Affine.identity(['zspace', 'yspace', 'xshape'])
    y = i.mapping([3,4,5])
    nose.tools.assert_true(np.allclose(y, np.array([3,4,5])))


def test_from_affine():
    a = Affine.identity('ij')
    nose.tools.assert_equals(a.input_coords, a.output_coords)

def test_start_step():
    ''' Test from_start_step '''
    dcs = Affine.from_start_step('ijk', 'xyz', [5,5,5],[2,2,2])
    nose.tools.assert_true(np.allclose(dcs.affine, [[2,0,0,5],
                                                    [0,2,0,5],
                                                    [0,0,2,5],
                                                    [0,0,0,1]]))


class empty:
    pass

E = empty()

def setup():
    def f(x):
        return 2*x
    def g(x):
        return x/2.0
    x = CoordinateSystem('x', [Coordinate('x')])
    E.a = CoordinateMap(f, x, x)
    E.b = CoordinateMap(f, x, x, inverse=g)
    E.c = CoordinateMap(g, x, x)        
    E.d = CoordinateMap(g, x, x, inverse=f)        
    E.e = Affine.identity('ijk')

    A = np.identity(4)
    A[0:3] = np.random.standard_normal((3,4))
    E.mapping = Affine.from_params('ijk' ,'xyz', A)
    
    E.singular = Affine.from_params('ijk', 'xyzt',
                                    np.array([[ 0,  1,  2,  3],
                                              [ 4,  5,  6,  7],
                                              [ 8,  9, 10, 11],
                                              [ 8,  9, 10, 11],
                                              [ 0,  0,  0,  1]]))



def test_call():
    value = np.array([1., 2., 3.])
    result_a = E.a(value)
    result_b = E.b(value)
    result_c = E.c(value)
    result_d = E.d(value)        
    nose.tools.assert_true(np.allclose(result_a, 2*value))
    nose.tools.assert_true(np.allclose(result_b, 2*value))
    nose.tools.assert_true(np.allclose(result_c, value/2))
    nose.tools.assert_true(np.allclose(result_d, value/2) )       
        
def test_str():
    s_a = str(E.a)
    s_b = str(E.b)
    s_c = str(E.c)
    s_d = str(E.d)                
        
def test_compose():
    value = np.array([1., 2., 3.])
    
    aa = compose(E.a, E.a)
    nose.tools.assert_false(aa.inverse)
    nose.tools.assert_true(np.allclose(aa(value), 4*value))

    ab = compose(E.a,E.b)
    nose.tools.assert_false(ab.inverse)
    nose.tools.assert_true(np.allclose(ab(value), 4*value))

    ac = compose(E.a,E.c)
    nose.tools.assert_false(ac.inverse)
    nose.tools.assert_true(np.allclose(ac(value), value))


    bb = compose(E.b,E.b)
    nose.tools.assert_true(bb.inverse)

  

def test_isinvertible():
    nose.tools.assert_false(E.a.inverse)
    nose.tools.assert_true(E.b.inverse)
    nose.tools.assert_false(E.c.inverse)
    nose.tools.assert_true(E.d.inverse)
        
def test_inverse1():
    inv = lambda a: a.inverse
    nose.tools.assert_false(inv(E.a))
    nose.tools.assert_false(inv(E.c))
    inv_b = E.b.inverse
    inv_d = E.d.inverse
    ident_b = compose(inv_b,E.b)
    ident_d = compose(inv_d,E.d)
    value = np.array([1., 2., 3.])
    print ident_d(value)
    nose.tools.assert_true(np.allclose(ident_b(value), value))
    nose.tools.assert_true(np.allclose(ident_d(value), value))
        
      

def test_call():
    value = np.array([1., 2., 3.])
    nose.tools.assert_true(np.allclose(E.e(value), value))
    
def test_mul():
    value = np.array([1., 2., 3.])
    b = compose(E.e, E.e)
    nose.tools.assert_true(np.allclose(b(value), value))

def test_str():
    s = str(E.e)
    
def test_invertible():
    nose.tools.assert_true(E.e.inverse)
    
def test_inverse2():
    nose.tools.assert_true(np.allclose(E.e.affine, E.e.inverse.inverse.affine))
        
       

def test_matvec_trasform():
    m1 = np.random.standard_normal((3, 3))
    v1 = np.random.standard_normal((3,))
    m2, v2 = matvec_from_transform(transform_from_matvec(m1, v1))
    nose.tools.assert_true(np.allclose(m1, m2))
    nose.tools.assert_true(np.allclose(v1, v2))



def test_isinvertible():
    nose.tools.assert_true(E.mapping.inverse)
    nose.tools.assert_false(E.singular.inverse)



        


