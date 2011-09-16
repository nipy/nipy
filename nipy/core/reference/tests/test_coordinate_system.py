
import numpy as np

from ..coordinate_system import (CoordinateSystem, CoordinateSystemError,
                                 product, safe_dtype)

from nose.tools import assert_true, assert_equal, assert_raises


class empty(object):
    pass

E = empty()

def setup():
    E.name = "test"
    E.axes = ('i', 'j', 'k')
    E.coord_dtype = np.float32
    E.cs = CoordinateSystem(E.axes, E.name, E.coord_dtype)


def test_CoordinateSystem():
    assert_equal(E.cs.name, E.name)
    assert_equal(E.cs.coord_names, E.axes)
    assert_equal(E.cs.coord_dtype, E.coord_dtype)


def test_iterator_coordinate():
    def gen():
        yield 'i'
        yield 'j'
        yield 'k'
    coordsys = CoordinateSystem(gen(), name='test_iter')
    assert_equal(coordsys.coord_names, ('i','j','k'))


def test_ndim():
    cs = CoordinateSystem('xy')
    yield assert_equal, cs.ndim, 2
    cs = CoordinateSystem('ijk')
    yield assert_equal, cs.ndim, 3


def test_unique_coord_names():
    unique = ('i','j','k')
    notuniq = ('i','i','k')
    coordsys = CoordinateSystem(unique)
    assert_equal(coordsys.coord_names, unique)
    assert_raises(ValueError, CoordinateSystem, notuniq)


def test_dtypes():
    # invalid dtypes
    dtypes = np.sctypes['others']
    for dt in dtypes:
        yield assert_raises, ValueError, CoordinateSystem, 'ijk', 'test', dt
    # compound dtype
    dtype = np.dtype([('field1', '<f8'), ('field2', '<i4')])
    yield assert_raises, ValueError, CoordinateSystem, 'ijk', 'test', dtype
    # valid dtype
    dtypes = np.sctypes['int']
    for dt in dtypes:
        cs = CoordinateSystem('ijk', coord_dtype=dt)
        yield assert_equal, cs.coord_dtype, dt
        cs_dt = [(f, dt) for f in 'ijk']
        yield assert_equal, cs.dtype, np.dtype(cs_dt)
    # verify assignment fails
    yield assert_raises, AttributeError, setattr, cs, 'dtype', np.dtype(cs_dt)
    yield assert_raises, AttributeError, setattr, cs, 'coord_dtype', np.float


def test_readonly_attrs():
    cs = E.cs
    yield (assert_raises, AttributeError, setattr, cs, 'coord_dtype', 
           np.dtype(np.int32))
    yield (assert_raises, AttributeError, setattr, cs, 'coord_names',
           ['a','b','c'])
    yield (assert_raises, AttributeError, setattr, cs, 'dtype',
           np.dtype([('i', '<f4'), ('j', '<f4'), ('k', '<f4')]))
    yield assert_raises, AttributeError, setattr, cs, 'ndim', 4


def test_index():
    cs = CoordinateSystem('ijk')
    yield assert_equal, cs.index('i'), 0
    yield assert_equal, cs.index('j'), 1
    yield assert_equal, cs.index('k'), 2
    yield assert_raises, ValueError, cs.index, 'x'


def test__ne__():
    cs1 = CoordinateSystem('ijk')
    cs2 = CoordinateSystem('xyz')
    yield assert_true, cs1 != cs2
    cs1 = CoordinateSystem('ijk', coord_dtype='float')
    cs2 = CoordinateSystem('ijk', coord_dtype='int')
    yield assert_true, cs1 != cs2


def test___eq__():
    c1 = CoordinateSystem(E.cs.coord_names, E.cs.name, E.coord_dtype)
    assert_equal(c1, E.cs)


def test___str__():
    s = str(E.cs)
    assert_equal(s, "CoordinateSystem(coord_names=('i', 'j', 'k'), name='test', coord_dtype=float32)")


def test_checked_values():
    cs = CoordinateSystem('ijk', name='voxels', coord_dtype=np.float32)
    x = np.array([1, 2, 3], dtype=np.int16)
    xc = cs._checked_values(x)
    np.allclose(xc, x)
    # wrong shape
    assert_raises(CoordinateSystemError, cs._checked_values, x.reshape(3,1))
    # wrong length
    assert_raises(CoordinateSystemError, cs._checked_values, x[0:2])
    # wrong dtype
    x = np.array([1,2,3], dtype=np.float64)
    assert_raises(CoordinateSystemError, cs._checked_values, x)


def test_safe_dtype():
    yield assert_raises, TypeError, safe_dtype, type('foo')
    yield assert_raises, TypeError, safe_dtype, type('foo'), np.float64
    yield assert_raises, TypeError, safe_dtype, [('x', 'f8')]
    valid_dtypes = []
    valid_dtypes.extend(np.sctypes['complex'])
    valid_dtypes.extend(np.sctypes['float'])
    valid_dtypes.extend(np.sctypes['int'])
    valid_dtypes.extend(np.sctypes['uint'])
    for dt in valid_dtypes:
        sdt = safe_dtype(dt)
        yield assert_equal, sdt, dt
    # test a few upcastings
    dt = safe_dtype(np.float32, np.int16, np.bool)
    yield assert_equal, dt, np.float32
    dt = safe_dtype(np.float32, np.int64, np.uint32)
    yield assert_equal, dt, np.float64
    # test byteswapped types - isbuiltin will be false
    orig_dt = np.dtype('f')
    dt = safe_dtype(orig_dt, np.int64, np.uint32)
    yield assert_equal, dt, np.float64
    swapped_dt = orig_dt.newbyteorder()
    dt = safe_dtype(swapped_dt, np.int64, np.uint32)
    yield assert_equal, dt, np.float64


def test_product():
    # int32 + int64 => int64
    ax1 = CoordinateSystem('x', coord_dtype=np.int32)
    ax2 = CoordinateSystem('y', coord_dtype=np.int64)
    cs = product(ax1, ax2)
    # assert up-casting of dtype
    yield assert_equal, cs.coord_dtype, np.dtype(np.int64)
    # assert composed dtype 
    yield assert_equal, cs.dtype, np.dtype([('x', np.int64), ('y', np.int64)])
    # the axes should be typecast in the CoordinateSystem but
    # uneffected themselves
    yield assert_equal, ax1.dtype, np.dtype([('x', np.int32)])
    yield assert_equal, ax2.dtype, np.dtype([('y', np.int64)])

    # float32 + int64 => float64
    ax1 = CoordinateSystem('x', coord_dtype=np.float32)
    cs = product(ax1, ax2)
    yield assert_equal, cs.coord_dtype, np.dtype(np.float64)
    yield assert_equal, cs.dtype, np.dtype([('x', np.float64), 
                                            ('y', np.float64)])

    # int16 + complex64 => complex64
    ax1 = CoordinateSystem('x', coord_dtype=np.int16)
    ax2 = CoordinateSystem('y', coord_dtype=np.complex64)
    # Order of the params effects order of dtype but not resulting value type
    cs = product(ax2, ax1)
    yield assert_equal, cs.coord_dtype, np.complex64
    yield assert_equal, cs.dtype, np.dtype([('y', np.complex64),
                                            ('x', np.complex64)])


from ..coordinate_system import CoordSysMaker, CoordSysMakerError

def test_coordsys_maker():
    # Things that help making coordinate maps
    ax_names = list('ijklm')
    nl = len(ax_names)
    cs_maker = CoordSysMaker(ax_names, 'myname')
    for i in range(1,nl+1):
        assert_equal(cs_maker(i),
                     CoordinateSystem(ax_names[:i], 'myname', np.float))
    assert_raises(CoordSysMakerError, cs_maker, nl+1)
    # You can pass in your own name
    assert_equal(cs_maker(i, 'anothername'),
                 CoordinateSystem(ax_names[:i+1], 'anothername', np.float))
    # And your own dtype if you really want
    assert_equal(cs_maker(i, coord_dtype=np.int32),
                 CoordinateSystem(ax_names[:i+1], 'myname', np.int32))

