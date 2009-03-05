import numpy as np
from neuroimaging.testing import *

from neuroimaging.core.reference.coordinate_map import CoordinateMap, Affine, \
    compose, CoordinateSystem, reorder_input, reorder_output, product, \
    replicate, linearize


class empty:
    pass

E = empty()

def setup():
    def f(x):
        return 2*x
    def g(x):
        return x/2.0
    x = CoordinateSystem('x', 'x')
    E.a = CoordinateMap(f, x, x)
    E.b = CoordinateMap(f, x, x, inverse_mapping=g)
    E.c = CoordinateMap(g, x, x)        
    E.d = CoordinateMap(g, x, x, inverse_mapping=f)        
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
    yield assert_true, np.allclose(result_a, 2*value)
    yield assert_true, np.allclose(result_b, 2*value)
    yield assert_true, np.allclose(result_c, value/2)
    yield assert_true, np.allclose(result_d, value/2)
        
def test_str():
    s_a = str(E.a)
    s_b = str(E.b)
    s_c = str(E.c)
    s_d = str(E.d)                


def test_compose():
    value = np.array([[1., 2., 3.]]).T
    aa = compose(E.a, E.a)
    yield assert_true, aa.inverse is None
    yield assert_true, np.allclose(aa(value), 4*value)
    ab = compose(E.a,E.b)
    yield assert_true, ab.inverse is None
    assert_true, np.allclose(ab(value), 4*value)
    ac = compose(E.a,E.c)
    yield assert_true, ac.inverse is None
    yield assert_true, np.allclose(ac(value), value)
    bb = compose(E.b,E.b)
    yield assert_true, bb.inverse is not None
  

def test_isinvertible():
    assert_false(E.a.inverse)
    assert_true(E.b.inverse)
    assert_false(E.c.inverse)
    assert_true(E.d.inverse)
        
def test_inverse1():
    inv = lambda a: a.inverse
    yield assert_true, inv(E.a) is None
    yield assert_true, inv(E.c) is None
    inv_b = E.b.inverse
    inv_d = E.d.inverse
    ident_b = compose(inv_b,E.b)
    ident_d = compose(inv_d,E.d)
    value = np.array([[1., 2., 3.]]).T    
    yield assert_true, np.allclose(ident_b(value), value)
    yield assert_true, np.allclose(ident_d(value), value)
        
      

def test_call():
    value = np.array([1., 2., 3.])
    assert_true(np.allclose(E.e(value), value))
    
def test_mul():
    value = np.array([1., 2., 3.])
    b = compose(E.e, E.e)
    assert_true(np.allclose(b(value), value))

def test_str():
    s = str(E.e)
    
def test_invertible():
    assert_true(E.e.inverse)
    
def test_inverse2():
    assert_true(np.allclose(E.e.affine, E.e.inverse.inverse.affine))


def test_isinvertible():
    yield assert_true, E.mapping.inverse
    yield assert_false, E.singular.inverse


def voxel_to_world():
    # utility function for generating trivial CoordinateMap
    incs = CoordinateSystem('ijk', 'voxels')
    outcs = CoordinateSystem('xyz', 'world')
    map = lambda x: x + 1
    inv = lambda x: x - 1
    return incs, outcs, map, inv

def test_comap_init():
    # Test mapping and non-mapping functions
    incs, outcs, map, inv = voxel_to_world()
    cm = CoordinateMap(map, incs, outcs, inv)
    yield assert_equal, cm.mapping, map
    yield assert_equal, cm.input_coords, incs
    yield assert_equal, cm.output_coords, outcs
    yield assert_equal, cm.inverse_mapping, inv
    yield assert_raises, ValueError, CoordinateMap, 'foo', incs, outcs, inv
    yield assert_raises, ValueError, CoordinateMap, map, incs, outcs, 'bar'


def test_comap_copy():
    incs, outcs, map, inv = voxel_to_world()
    cm = CoordinateMap(map, incs, outcs, inv)
    cmcp = cm.copy()
    yield assert_equal, cmcp.mapping, cm.mapping
    yield assert_equal, cmcp.input_coords, cm.input_coords
    yield assert_equal, cmcp.output_coords, cm.output_coords
    yield assert_equal, cmcp.inverse_mapping, cm.inverse_mapping


#
# Affine tests
#

def affine_v2w():
    # utility function
    incs = CoordinateSystem('ijk', 'voxels')
    outcs = CoordinateSystem('xyz', 'world')
    aff = np.diag([1, 2, 4, 1])
    aff[:3, 3] = [11, 12, 13]
    """array([[ 1,  0,  0, 11],
       [ 0,  2,  0, 12],
       [ 0,  0,  4, 13],
       [ 0,  0,  0,  1]])
    """
    return incs, outcs, aff


def test_affine_init():
    incs, outcs, aff = affine_v2w()
    cm = Affine(aff, incs, outcs)
    yield assert_equal, cm.input_coords, incs
    yield assert_equal, cm.output_coords, outcs
    yield assert_equal, cm.affine, aff
    badaff = np.diag([1,2])
    yield assert_raises, ValueError, Affine, badaff, incs, outcs


def test_affine_inverse():
    incs, outcs, aff = affine_v2w()
    inv = np.linalg.inv(aff)
    cm = Affine(aff, incs, outcs)
    invmap = cm.inverse_mapping
    x = np.array([10, 20, 30])
    x_roundtrip = cm.mapping(invmap(x))
    yield assert_equal, x_roundtrip, x
    badaff = np.array([[1,2,3],[4,5,6]])
    badcm = Affine(aff, incs, outcs)
    badcm._affine = badaff
    yield assert_raises, ValueError, getattr, badcm, 'inverse_mapping'


def test_affine_from_params():
    incs, outcs, aff = affine_v2w()
    cm = Affine.from_params('ijk', 'xyz', aff)
    yield assert_equal, cm.affine, aff
    badaff = np.array([[1,2,3],[4,5,6]])
    yield assert_raises, ValueError, Affine.from_params, 'ijk', 'xyz', badaff


def test_affine_start_step():
    incs, outcs, aff = affine_v2w()
    start = aff[:3, 3]
    step = aff.diagonal()[:3]
    cm = Affine.from_start_step(incs.coord_names, outcs.coord_names,
                                start, step)
    yield assert_equal, cm.affine, aff
    yield assert_raises, ValueError, Affine.from_start_step, 'ijk', 'xy', \
        start, step


def test_affine_identity():
    aff = Affine.identity('ijk')
    yield assert_equal, aff.affine, np.eye(4)
    yield assert_equal, aff.input_coords, aff.output_coords
    x = np.array([3, 4, 5])
    y = aff.mapping(x)
    yield assert_equal, y, x


def test_affine_copy():
    incs, outcs, aff = affine_v2w()
    cm = Affine(aff, incs, outcs)
    cmcp = cm.copy()
    yield assert_equal, cmcp.affine, cm.affine
    yield assert_equal, cmcp.input_coords, cm.input_coords
    yield assert_equal, cmcp.output_coords, cm.output_coords


#
# Module level functions
#

def test_reorder_input():
    incs, outcs, map, inv = voxel_to_world()
    cm = CoordinateMap(map, incs, outcs, inv)
    recm = reorder_input(cm, 'jki')
    yield assert_equal, recm.input_coords.coord_names, ('j', 'k', 'i')
    yield assert_equal, recm.output_coords.coord_names, outcs.coord_names
    yield assert_equal, recm.input_coords.name, incs.name+'-reordered'
    yield assert_equal, recm.output_coords.name, outcs.name
    # default reverse reorder
    recm = reorder_input(cm)
    yield assert_equal, recm.input_coords.coord_names, ('k', 'j', 'i')
    # reorder with order as indices
    recm = reorder_input(cm, [2,0,1])
    yield assert_equal, recm.input_coords.coord_names, ('k', 'i', 'j')


def test_reorder_output():
    incs, outcs, map, inv = voxel_to_world()
    cm = CoordinateMap(map, incs, outcs, inv)
    recm = reorder_output(cm, 'yzx')
    yield assert_equal, recm.input_coords.coord_names, incs.coord_names
    yield assert_equal, recm.output_coords.coord_names, ('y', 'z', 'x')
    yield assert_equal, recm.input_coords.name, incs.name
    yield assert_equal, recm.output_coords.name, outcs.name+'-reordered'
    # default reverse order
    recm = reorder_output(cm)
    yield assert_equal, recm.output_coords.coord_names, ('z', 'y', 'x')
    # reorder with indicies
    recm = reorder_output(cm, [2,0,1])
    yield assert_equal, recm.output_coords.coord_names, ('z', 'x', 'y')    


def test_product():
    cm1 = Affine.from_params('i', 'x', np.diag([2, 1]))
    cm2 = Affine.from_params('j', 'y', np.diag([3, 1]))
    cm = product(cm1, cm2)
    yield assert_equal, cm.input_coords.coord_names, ('i', 'j')
    yield assert_equal, cm.output_coords.coord_names, ('x', 'y')
    yield assert_equal, cm.affine, np.diag([2, 3, 1])


def test_compose():
    aff1 = np.diag([1,2,3,1])
    cm1 = Affine.from_params('ijk', 'xyz', aff1)
    aff2 = np.diag([4,5,6,1])
    cm2 = Affine.from_params('xyz', 'abc', aff2)
    # compose mapping from 'ijk' to 'abc'
    compcm = compose(cm2, cm1)
    yield assert_equal, compcm.input_coords.coord_names, ('i', 'j', 'k')
    yield assert_equal, compcm.output_coords.coord_names, ('a', 'b', 'c')
    yield assert_equal, compcm.affine, np.dot(aff2, aff1)
    # check invalid coordinate mappings
    yield assert_raises, ValueError, compose, cm1, cm2


def test_replicate():
    # FIXME: Implement test when this function works
    yield assert_raises, NotImplementedError, replicate, 'foo', 'bar'


def test_linearize():
    aff = np.diag([1,2,3,1])
    cm = Affine.from_params('ijk', 'xyz', aff)
    lincm = linearize(cm.mapping, cm.ndim[0])
    yield assert_equal, lincm, aff
    origin = np.array([10, 20, 30], dtype=cm.input_coords.coord_dtype)
    lincm = linearize(cm.mapping, cm.ndim[0], origin=origin)
    xform = np.array([[  1.,   0.,   0.,  10.],
                      [  0.,   2.,   0.,  40.],
                      [  0.,   0.,   3.,  90.],
                      [  0.,   0.,   0.,   1.]])
    yield assert_equal, lincm, xform
    # dtype mismatch
    #origin = np.array([10, 20, 30], dtype=np.int16)
    #yield assert_raises, UserWarning, linearize, cm.mapping, cm.ndim[0], \
    #    1, origin
