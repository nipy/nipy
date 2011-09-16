# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np

# this import line is a little ridiculous...
from ..coordinate_map import (CoordinateMap, AffineTransform, compose, product,
                              append_io_dim, drop_io_dim, equivalent,
                              shifted_domain_origin, shifted_range_origin,
                              CoordMapMaker, CoordMapMakerError,
                              _as_coordinate_map)

from ..coordinate_system import (CoordinateSystem, CoordinateSystemError,
                                 CoordSysMaker, CoordSysMakerError)

# shortcut
CS = CoordinateSystem

from nose.tools import (assert_true, assert_equal, assert_raises,
                        assert_false)

from numpy.testing import (assert_array_equal, assert_almost_equal)


class empty(object):
    pass

# object to hold module global setup
E = empty()


def setup():
    def f(x):
        return 2*x
    def g(x):
        return x/2.0
    x = CoordinateSystem('x', 'x')
    E.a = CoordinateMap(x, x, f)
    E.b = CoordinateMap(x, x, f, inverse_function=g)
    E.c = CoordinateMap(x, x, g)
    E.d = CoordinateMap(x, x, g, inverse_function=f)
    E.e = AffineTransform.identity('ijk')
    A = np.identity(4)
    A[0:3] = np.random.standard_normal((3,4))
    E.mapping = AffineTransform.from_params('ijk' ,'xyz', A)
    E.singular = AffineTransform.from_params('ijk', 'xyzt',
                                    np.array([[ 0,  1,  2,  3],
                                              [ 4,  5,  6,  7],
                                              [ 8,  9, 10, 11],
                                              [ 8,  9, 10, 11],
                                              [ 0,  0,  0,  1]]))


def test_shift_origin():
    CS = CoordinateSystem

    A = np.random.standard_normal((5,6))
    A[-1] = [0,0,0,0,0,1]

    aff1 = AffineTransform(CS('ijklm', 'oldorigin'), CS('xyzt'), A)
    difference = np.random.standard_normal(5)
    point_in_old_basis = np.random.standard_normal(5)

    for aff in [aff1, _as_coordinate_map(aff1)]:
        # The same affine transformation with a different origin for its domain
        shifted_aff = shifted_domain_origin(aff, difference, 'neworigin')
        # This is the relationship between coordinates in old and new origins
        assert_almost_equal(shifted_aff(point_in_old_basis),
                            aff(point_in_old_basis+difference))
        assert_almost_equal(shifted_aff(point_in_old_basis-difference),
                            aff(point_in_old_basis))
    # OK, now for the range
    A = np.random.standard_normal((5,6))
    A[-1] = [0,0,0,0,0,1]
    aff2 = AffineTransform(CS('ijklm', 'oldorigin'), CS('xyzt'), A)
    difference = np.random.standard_normal(4)
    for aff in [aff2, _as_coordinate_map(aff2)]:
    # The same affine transformation with a different origin for its domain
        shifted_aff = shifted_range_origin(aff, difference, 'neworigin')
        # Let's check that things work
        point_in_old_basis = np.random.standard_normal(5)
        # This is the relation ship between coordinates in old and new origins
        assert_almost_equal(shifted_aff(point_in_old_basis),
                            aff(point_in_old_basis)-difference)
        assert_almost_equal(shifted_aff(point_in_old_basis)+difference,
                            aff(point_in_old_basis))


def test_renamed():
    # Renaming domain and range
    A = AffineTransform.from_params('ijk', 'xyz', np.identity(4))
    ijk = CoordinateSystem('ijk')
    xyz = CoordinateSystem('xyz')
    C = CoordinateMap(ijk, xyz, np.log)
    for B in [A,C]:
        B_re = B.renamed_domain({'i':'foo'})
        assert_equal(B_re.function_domain.coord_names, ('foo', 'j', 'k'))
        B_re = B.renamed_domain({'i':'foo','j':'bar'})
        assert_equal(B_re.function_domain.coord_names, ('foo', 'bar', 'k'))
        B_re = B.renamed_range({'y':'foo'})
        assert_equal(B_re.function_range.coord_names, ('x', 'foo', 'z'))
        B_re = B.renamed_range({0:'foo',1:'bar'})
        assert_equal(B_re.function_range.coord_names, ('foo', 'bar', 'z'))
        B_re = B.renamed_domain({0:'foo',1:'bar'})
        assert_equal(B_re.function_domain.coord_names, ('foo', 'bar', 'k'))
        B_re = B.renamed_range({'y':'foo','x':'bar'})
        assert_equal(B_re.function_range.coord_names, ('bar', 'foo', 'z'))
        assert_raises(ValueError, B.renamed_range, {'foo':'y'})
        assert_raises(ValueError, B.renamed_domain, {'foo':'y'})


def test_calling_shapes():
    cs2d = CS('ij')
    cs1d = CS('i')
    cm2d = CoordinateMap(cs2d, cs2d, lambda x : x+1)
    cm1d2d = CoordinateMap(cs1d, cs2d,
                           lambda x : np.concatenate((x, x), axis=-1))
    at2d = AffineTransform(cs2d, cs2d, np.array([[1, 0, 1],
                                                 [0, 1, 1],
                                                 [0, 0, 1]]))
    at1d2d = AffineTransform(cs1d, cs2d, np.array([[1,0],
                                                   [0,1],
                                                   [0,1]]))
    # test coordinate maps and affine transforms
    for xfm2d, xfm1d2d in ((cm2d, cm1d2d), (at2d, at1d2d)):
        arr = np.array([0, 1])
        assert_array_equal(xfm2d(arr), [1, 2])
        # test lists work too
        res = xfm2d([0, 1])
        assert_array_equal(res, [1, 2])
        # and return arrays (by checking shape attribute)
        assert_equal(res.shape, (2,))
        # maintaining input shape
        arr_long = arr[None, None, :]
        assert_array_equal(xfm2d(arr_long), arr_long + 1)
        # wrong shape array raises error
        assert_raises(CoordinateSystemError, xfm2d, np.zeros((3,)))
        assert_raises(CoordinateSystemError, xfm2d, np.zeros((3,3)))
        # 1d to 2d
        arr = np.array(1)
        assert_array_equal(xfm1d2d(arr), [1,1] )
        arr_long = arr[None, None, None]
        assert_array_equal(xfm1d2d(arr_long), np.ones((1,1,2)))
        # wrong shape array raises error.  Note 1d input requires size 1
        # as final axis
        assert_raises(CoordinateSystemError, xfm1d2d, np.zeros((3,)))
        assert_raises(CoordinateSystemError, xfm1d2d, np.zeros((3,2)))


def test_call():
    value = 10
    assert_true(np.allclose(E.a(value), 2*value))
    assert_true(np.allclose(E.b(value), 2*value))
    # FIXME: this shape just below is not
    # really expected for a CoordinateMap
    assert_true(np.allclose(E.b([value]), 2*value))
    assert_true(np.allclose(E.c(value), value/2))
    assert_true(np.allclose(E.d(value), value/2))
    value = np.array([1., 2., 3.])
    assert_true(np.allclose(E.e(value), value))
    # check that error raised for wrong shape
    value = np.array([1., 2.,])
    assert_raises(CoordinateSystemError, E.e, value)


def test_compose():
    value = np.array([[1., 2., 3.]]).T
    aa = compose(E.a, E.a)
    assert_true(aa.inverse() is None)
    assert_almost_equal(aa(value), 4*value)
    ab = compose(E.a,E.b)
    assert_true(ab.inverse() is None)
    assert_almost_equal(ab(value), 4*value)
    ac = compose(E.a,E.c)
    assert_true(ac.inverse() is None)
    assert_almost_equal(ac(value), value)
    bb = compose(E.b,E.b)
    #    yield assert_true, bb.inverse() is not None
    aff1 = np.diag([1,2,3,1])
    affine1 = AffineTransform.from_params('ijk', 'xyz', aff1)
    aff2 = np.diag([4,5,6,1])
    affine2 = AffineTransform.from_params('xyz', 'abc', aff2)
    # compose mapping from 'ijk' to 'abc'
    compcm = compose(affine2, affine1)
    assert_equal(compcm.function_domain.coord_names, ('i', 'j', 'k'))
    assert_equal(compcm.function_range.coord_names, ('a', 'b', 'c'))
    assert_almost_equal(compcm.affine, np.dot(aff2, aff1))
    # check invalid coordinate mappings
    assert_raises(ValueError, compose, affine1, affine2)
    assert_raises(ValueError, compose, affine1, 'foo')
    cm1 = CoordinateMap(CoordinateSystem('ijk'),
                        CoordinateSystem('xyz'), np.log)
    cm2 = CoordinateMap(CoordinateSystem('xyz'),
                        CoordinateSystem('abc'), np.exp)
    assert_raises(ValueError, compose, cm1, cm2)


def test__eq__():
    yield assert_true, E.a == E.a
    yield assert_false, E.a != E.a

    yield assert_false, E.a == E.b
    yield assert_true, E.a != E.b

    yield assert_true, E.singular == E.singular
    yield assert_false, E.singular != E.singular

    A = AffineTransform.from_params('ijk', 'xyz', np.diag([4,3,2,1]))
    B = AffineTransform.from_params('ijk', 'xyz', np.diag([4,3,2,1]))

    yield assert_true, A == B
    yield assert_false, A != B


def test_isinvertible():
    yield assert_false, E.a.inverse()
    yield assert_true, E.b.inverse()
    yield assert_false, E.c.inverse()
    yield assert_true, E.d.inverse()
    yield assert_true, E.e.inverse()
    yield assert_true, E.mapping.inverse()
    yield assert_false, E.singular.inverse()


def test_inverse1():
    inv = lambda a: a.inverse()
    yield assert_true, inv(E.a) is None
    yield assert_true, inv(E.c) is None
    inv_b = E.b.inverse()
    inv_d = E.d.inverse()
    ident_b = compose(inv_b,E.b)
    ident_d = compose(inv_d,E.d)
    value = np.array([[1., 2., 3.]]).T
    yield assert_true, np.allclose(ident_b(value), value)
    yield assert_true, np.allclose(ident_d(value), value)


def test_compose_cmap():
    value = np.array([1., 2., 3.])
    b = compose(E.e, E.e)
    assert_true(np.allclose(b(value), value))


def test_inverse2():
    assert_true(np.allclose(E.e.affine, E.e.inverse().inverse().affine))


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
    cm = CoordinateMap(incs, outcs, map, inv)
    yield assert_equal, cm.function, map
    yield assert_equal, cm.function_domain, incs
    yield assert_equal, cm.function_range, outcs
    yield assert_equal, cm.inverse_function, inv
    yield assert_raises, ValueError, CoordinateMap, incs, outcs, 'foo', inv
    yield assert_raises, ValueError, CoordinateMap, incs, outcs, map, 'bar'


def test_comap_copy():
    import copy
    incs, outcs, map, inv = voxel_to_world()
    cm = CoordinateMap(incs, outcs, inv, map)
    cmcp = copy.copy(cm)
    yield assert_equal, cmcp.function, cm.function
    yield assert_equal, cmcp.function_domain, cm.function_domain
    yield assert_equal, cmcp.function_range, cm.function_range
    yield assert_equal, cmcp.inverse_function, cm.inverse_function


#
# AffineTransform tests
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
    cm = AffineTransform(incs, outcs, aff)
    assert_equal(cm.function_domain, incs)
    assert_equal(cm.function_range, outcs)
    assert_array_equal(cm.affine, aff)
    badaff = np.diag([1,2])
    assert_raises(ValueError, AffineTransform, incs, outcs, badaff)


def test_affine_bottom_row():
    # homogeneous transformations have bottom rows all zero
    # except the last one
    assert_raises(ValueError, AffineTransform.from_params,
                  'ij',  'x', np.array([[3,4,5],[1,1,1]]))


def test_affine_inverse():
    incs, outcs, aff = affine_v2w()
    inv = np.linalg.inv(aff)
    cm = AffineTransform(incs, outcs, aff)
    x = np.array([10, 20, 30], np.float)
    x_roundtrip = cm(cm.inverse()(x))
    assert_almost_equal(x_roundtrip, x)
    badaff = np.array([[1,2,3],[0,0,1]])
    badcm = AffineTransform(CoordinateSystem('ij'),
                            CoordinateSystem('x'),
                            badaff)
    assert_equal(badcm.inverse(), None)


def test_affine_from_params():
    incs, outcs, aff = affine_v2w()
    cm = AffineTransform.from_params('ijk', 'xyz', aff)
    assert_array_equal(cm.affine, aff)
    badaff = np.array([[1,2,3],[4,5,6]])
    assert_raises(ValueError,
                  AffineTransform.from_params, 'ijk', 'xyz', badaff)


def test_affine_start_step():
    incs, outcs, aff = affine_v2w()
    start = aff[:3, 3]
    step = aff.diagonal()[:3]
    cm = AffineTransform.from_start_step(incs.coord_names, outcs.coord_names,
                                start, step)
    assert_array_equal(cm.affine, aff)
    assert_raises(ValueError, AffineTransform.from_start_step, 'ijk', 'xy',
                  start, step)


def test_affine_identity():
    aff = AffineTransform.identity('ijk')
    assert_array_equal(aff.affine, np.eye(4))
    assert_equal(aff.function_domain, aff.function_range)
    # AffineTransform's aren't CoordinateMaps, so
    # they don't have "function" attributes
    assert_false(hasattr(aff, 'function'))


def test_affine_copy():
    incs, outcs, aff = affine_v2w()
    cm = AffineTransform(incs, outcs, aff)
    import copy
    cmcp = copy.copy(cm)
    assert_array_equal(cmcp.affine, cm.affine)
    assert_equal(cmcp.function_domain, cm.function_domain)
    assert_equal(cmcp.function_range, cm.function_range)


#
# Module level functions
#

def test_reordered_domain():
    incs, outcs, map, inv = voxel_to_world()
    cm = CoordinateMap(incs, outcs, map, inv)
    recm = cm.reordered_domain('jki')
    yield assert_equal, recm.function_domain.coord_names, ('j', 'k', 'i')
    yield assert_equal, recm.function_range.coord_names, outcs.coord_names
    yield assert_equal, recm.function_domain.name, incs.name
    yield assert_equal, recm.function_range.name, outcs.name
    # default reverse reorder
    recm = cm.reordered_domain()
    yield assert_equal, recm.function_domain.coord_names, ('k', 'j', 'i')
    # reorder with order as indices
    recm = cm.reordered_domain([2,0,1])
    yield assert_equal, recm.function_domain.coord_names, ('k', 'i', 'j')


def test_str():
    result = """AffineTransform(
   function_domain=CoordinateSystem(coord_names=('i', 'j', 'k'), name='', coord_dtype=float64),
   function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='', coord_dtype=float64),
   affine=array([[ 1.,  0.,  0.,  0.],
                 [ 0.,  1.,  0.,  0.],
                 [ 0.,  0.,  1.,  0.],
                 [ 0.,  0.,  0.,  1.]])
)"""
    domain = CoordinateSystem('ijk')
    range = CoordinateSystem('xyz')
    affine = np.identity(4)
    affine_mapping = AffineTransform(domain, range, affine)
    assert_equal(result, str(affine_mapping))

    cmap = CoordinateMap(domain, range, np.exp, np.log)
    result="""CoordinateMap(
   function_domain=CoordinateSystem(coord_names=('i', 'j', 'k'), name='', coord_dtype=float64),
   function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='', coord_dtype=float64),
   function=<ufunc 'exp'>,
   inverse_function=<ufunc 'log'>
  )"""
    cmap = CoordinateMap(domain, range, np.exp)
    result="""CoordinateMap(
   function_domain=CoordinateSystem(coord_names=('i', 'j', 'k'), name='', coord_dtype=float64),
   function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='', coord_dtype=float64),
   function=<ufunc 'exp'>
  )"""
    assert_equal(result, repr(cmap))


def test_reordered_range():
    incs, outcs, map, inv = voxel_to_world()
    cm = CoordinateMap(incs, outcs, inv, map)
    recm = cm.reordered_range('yzx')
    yield assert_equal, recm.function_domain.coord_names, incs.coord_names
    yield assert_equal, recm.function_range.coord_names, ('y', 'z', 'x')
    yield assert_equal, recm.function_domain.name, incs.name
    yield assert_equal, recm.function_range.name, outcs.name
    # default reverse order
    recm = cm.reordered_range()
    yield assert_equal, recm.function_range.coord_names, ('z', 'y', 'x')
    # reorder with indicies
    recm = cm.reordered_range([2,0,1])
    yield assert_equal, recm.function_range.coord_names, ('z', 'x', 'y')


def test_product():
    affine1 = AffineTransform.from_params('i', 'x', np.diag([2, 1]))
    affine2 = AffineTransform.from_params('j', 'y', np.diag([3, 1]))
    affine = product(affine1, affine2)
    cm1 = CoordinateMap(CoordinateSystem('i'),
                        CoordinateSystem('x'),
                        np.log)
    cm2 = CoordinateMap(CoordinateSystem('j'),
                        CoordinateSystem('y'),
                        np.log)
    cm = product(cm1, cm2)
    assert_equal(affine.function_domain.coord_names, ('i', 'j'))
    assert_equal(affine.function_range.coord_names, ('x', 'y'))
    assert_almost_equal(cm([3,4]), np.log([3,4]))
    assert_almost_equal(cm.function([[3,4],[5,6]]), np.log([[3,4],[5,6]]))
    assert_equal(affine.function_domain.coord_names, ('i', 'j'))
    assert_equal(affine.function_range.coord_names, ('x', 'y'))
    assert_array_equal(affine.affine, np.diag([2, 3, 1]))


def test_equivalent():
    ijk = CoordinateSystem('ijk')
    xyz = CoordinateSystem('xyz')
    T = np.random.standard_normal((4,4))
    T[-1] = [0,0,0,1]
    A = AffineTransform(ijk, xyz, T)

    # now, cycle through
    # all possible permutations of
    # 'ijk' and 'xyz' and confirm that
    # the mapping is equivalent

    yield assert_false, equivalent(A, A.renamed_domain({'i':'foo'}))

    try:
        import itertools
        for pijk in itertools.permutations('ijk'):
            for pxyz in itertools.permutations('xyz'):
                B = A.reordered_domain(pijk).reordered_range(pxyz)
                yield assert_true, equivalent(A, B)
    except (ImportError, AttributeError):
        # just do some if we can't find itertools, or if itertools
        # doesn't have permutations
        for pijk in ['ikj', 'kij']:
            for pxyz in ['xzy', 'yxz']:
                B = A.reordered_domain(pijk).reordered_range(pxyz)
                yield assert_true, equivalent(A, B)


def test_as_coordinate_map():

    ijk = CoordinateSystem('ijk')
    xyz = CoordinateSystem('xyz')

    A = np.random.standard_normal((4,4))

    # bottom row of A is not [0,0,0,1]
    yield assert_raises, ValueError, AffineTransform, ijk, xyz, A

    A[-1] = [0,0,0,1]

    aff = AffineTransform(ijk, xyz, A)
    _cmapA = _as_coordinate_map(aff)
    yield assert_true, isinstance(_cmapA, CoordinateMap)
    yield assert_true, _cmapA.inverse_function != None

    # a non-invertible one

    B = A[1:]
    xy = CoordinateSystem('xy')
    affB = AffineTransform(ijk, xy, B)
    _cmapB = _as_coordinate_map(affB)

    yield assert_true, isinstance(_cmapB, CoordinateMap)
    yield assert_true, _cmapB.inverse_function == None


def test_cm__setattr__raise_error():
    # CoordinateMap has all read-only attributes

    # AffineTransform has some properties and it seems
    # the same __setattr__ doesn't work for it.
    ijk = CoordinateSystem('ijk')
    xyz = CoordinateSystem('xyz')

    cm = CoordinateMap(ijk, xyz, np.exp)

    yield assert_raises, AttributeError, cm.__setattr__, "function_range", xyz


def test_append_io_dim():
    aff = np.diag([1,2,3,1])
    in_dims = tuple('ijk')
    out_dims = tuple('xyz')
    cm = AffineTransform.from_params(in_dims, out_dims, aff)
    cm2 = append_io_dim(cm, 'l', 't')
    assert_array_equal(cm2.affine, np.diag([1,2,3,1,1]))
    assert_equal(cm2.function_range.coord_names, out_dims + ('t',))
    assert_equal(cm2.function_domain.coord_names, in_dims + ('l',))
    cm2 = append_io_dim(cm, 'l', 't', 9, 5)
    a2 = np.diag([1,2,3,5,1])
    a2[3,4] = 9
    assert_array_equal(cm2.affine, a2)
    assert_equal(cm2.function_range.coord_names, out_dims + ('t',))
    assert_equal(cm2.function_domain.coord_names, in_dims + ('l',))
    # non square case
    aff = np.array([[2,0,0],
                    [0,3,0],
                    [0,0,1],
                    [0,0,1]])
    cm = AffineTransform.from_params('ij', 'xyz', aff)
    cm2 = append_io_dim(cm, 'q', 't', 9, 5)
    a2 = np.array([[2,0,0,0],
                   [0,3,0,0],
                   [0,0,0,1],
                   [0,0,5,9],
                   [0,0,0,1]])
    assert_array_equal(cm2.affine, a2)
    assert_equal(cm2.function_range.coord_names, tuple('xyzt'))
    assert_equal(cm2.function_domain.coord_names, tuple('ijq'))


def test_mod():
    from nipy.core.reference.coordinate_map import _matching_orth_dim
    aff = np.diag([1,2,3,1])
    for i in range(3):
        assert_equal(_matching_orth_dim(i, aff), (i, ''))
    aff = np.diag([-1,-2,-3,1])
    for i in range(3):
        assert_equal(_matching_orth_dim(i, aff), (i, ''))
    aff = np.ones((4,4))
    for i in range(3):
        val, msg = _matching_orth_dim(i, aff)
        assert_equal(val, None)
    aff = np.zeros((3,3))
    for i in range(2):
        val, msg = _matching_orth_dim(i, aff)
        assert_equal(val, None)
    aff = np.array([[1, 0, 0, 1],
                    [0, 0, 2, 1],
                    [0, 3, 0, 1],
                    [0, 0, 0, 1]])
    val, msg = _matching_orth_dim(1, aff)
    assert_equal(_matching_orth_dim(0, aff), (0, ''))
    assert_equal(_matching_orth_dim(1, aff), (2, ''))
    assert_equal(_matching_orth_dim(2, aff), (1, ''))
    aff = np.diag([1, 2, 0, 1])
    aff[:,3] = 1
    assert_equal(_matching_orth_dim(2, aff), (2, ''))


def test_drop_io_dim():
    # test ordinary case of 4d to 3d
    cm4d = AffineTransform.from_params('ijkl', 'xyzt', np.diag([1,2,3,4,1]))
    cm3d = drop_io_dim(cm4d, 't')
    assert_array_equal(cm3d.affine, np.diag([1, 2, 3, 1]))
    # 3d to 2d
    cm3d = AffineTransform.from_params('ijk', 'xyz', np.diag([1,2,3,1]))
    cm2d = drop_io_dim(cm3d, 'z')
    assert_array_equal(cm2d.affine, np.diag([1, 2, 1]))
    # test zero scaling for dropped dimension
    cm3d = AffineTransform.from_params('ijk', 'xyz', np.diag([1, 2, 0, 1]))
    cm2d = drop_io_dim(cm3d, 'z')
    assert_array_equal(cm2d.affine, np.diag([1, 2, 1]))
    # test not diagonal but orthogonal
    aff = np.array([[1, 0, 0, 0],
                    [0, 0, 2, 0],
                    [0, 3, 0, 0],
                    [0, 0, 0, 1]])
    cm3d = AffineTransform.from_params('ijk', 'xyz', aff)
    cm2d = drop_io_dim(cm3d, 'z')
    assert_array_equal(cm2d.affine, np.diag([1, 2, 1]))
    cm2d = drop_io_dim(cm3d, 'k')
    assert_array_equal(cm2d.affine, np.diag([1, 3, 1]))
    # and with zeros scaling for orthogonal dropped dimension
    aff[2] = 0
    cm3d = AffineTransform.from_params('ijk', 'xyz', aff)
    cm2d = drop_io_dim(cm3d, 'z')
    assert_array_equal(cm2d.affine, np.diag([1, 2, 1]))


def test_make_cmap():
    # Routine to put the guessing back into making coordinate maps
    d_names = list('ijklm')
    r_names = list('xyztu')
    domain_maker = CoordSysMaker(d_names, 'array')
    range_maker = CoordSysMaker(r_names, 'world')
    cmm = CoordMapMaker(domain_maker, range_maker)
    # Making with generic functions and with affines
    xform = lambda x : x+1
    inv_xform = lambda x : x-1
    diag_vals = range(2,8)
    for i in range(1, 6):
        dcs = CS(d_names[:i], 'array')
        rcs = CS(r_names[:i], 'world')
        # Generic
        assert_equal(cmm.make_cmap(i, xform, inv_xform),
                     CoordinateMap(dcs, rcs, xform, inv_xform))
        assert_equal(cmm.make_cmap(i, xform), CoordinateMap(dcs, rcs, xform))
        # Affines
        aff = np.diag(diag_vals[:i] + [1])
        assert_equal(cmm.make_affine(aff), AffineTransform(dcs, rcs, aff))
        # Test that the call method selects what it got correctly
        assert_equal(cmm(i, xform, inv_xform),
                     CoordinateMap(dcs, rcs, xform, inv_xform))
        assert_equal(cmm(i, xform), CoordinateMap(dcs, rcs, xform))
        assert_equal(cmm(aff), AffineTransform(dcs, rcs, aff))
    # For affines, we can append dimensions by adding on the diagonal
    aff = np.diag([2,3,4,1])
    dcs = CS(d_names[:4], 'array')
    rcs = CS(r_names[:4], 'world')
    assert_equal(cmm.make_affine(aff, 5),
                 AffineTransform(CS(d_names[:4], 'array'),
                                 CS(r_names[:4], 'world'),
                                 np.diag([2,3,4,5,1])))
    assert_equal(cmm.make_affine(aff, [5,6]),
                 AffineTransform(CS(d_names[:5], 'array'),
                                 CS(r_names[:5], 'world'),
                                 np.diag([2,3,4,5,6,1])))
    # we can add offsets too
    exp_aff = np.diag([2,3,4,5,6,1])
    exp_aff[3:5,-1] = [7,8]
    assert_equal(cmm.make_affine(aff, [5,6],[7,8]),
                 AffineTransform(CS(d_names[:5], 'array'),
                                 CS(r_names[:5], 'world'),
                                 exp_aff))
    # The zooms (diagonal elements) and offsets must match in length
    assert_raises(CoordMapMakerError, cmm.make_affine, aff, [5,6], 7)
    # Check non-square affines
    aff = np.array([[2,0,0],
                    [0,3,0],
                    [0,0,1],
                    [0,0,1]])
    dcs = CS(d_names[:2], 'array')
    rcs = CS(r_names[:3], 'world')
    assert_equal(cmm.make_affine(aff), AffineTransform(dcs, rcs, aff))
    dcs = CS(d_names[:3], 'array')
    rcs = CS(r_names[:4], 'world')
    exp_aff = np.array([[2,0,0,0],
                        [0,3,0,0],
                        [0,0,0,1],
                        [0,0,4,0],
                        [0,0,0,1]])
    assert_equal(cmm.make_affine(aff, 4), AffineTransform(dcs, rcs, exp_aff))
