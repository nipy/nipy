""" Tests for coordinate_system module
"""

from types import SimpleNamespace

import numpy as np
import pytest

from ..coordinate_system import (
    CoordinateSystem,
    CoordinateSystemError,
    CoordSysMaker,
    CoordSysMakerError,
    is_coordsys,
    is_coordsys_maker,
    product,
    safe_dtype,
)


@pytest.fixture
def eg_cs():
    E = SimpleNamespace()
    E.name = "test"
    E.axes = ('i', 'j', 'k')
    E.coord_dtype = np.float32
    E.cs = CoordinateSystem(E.axes, E.name, E.coord_dtype)
    return E


def test_CoordinateSystem(eg_cs):
    assert eg_cs.cs.name == eg_cs.name
    assert eg_cs.cs.coord_names == eg_cs.axes
    assert eg_cs.cs.coord_dtype == eg_cs.coord_dtype


def test_iterator_coordinate():
    def gen():
        yield 'i'
        yield 'j'
        yield 'k'
    coordsys = CoordinateSystem(gen(), name='test_iter')
    assert coordsys.coord_names == ('i','j','k')


def test_ndim():
    cs = CoordinateSystem('xy')
    assert cs.ndim == 2
    cs = CoordinateSystem('ijk')
    assert cs.ndim == 3


def test_unique_coord_names():
    unique = ('i','j','k')
    notuniq = ('i','i','k')
    coordsys = CoordinateSystem(unique)
    assert coordsys.coord_names == unique
    pytest.raises(ValueError, CoordinateSystem, notuniq)


def test_dtypes():
    # invalid dtypes
    dtypes = np.sctypes['others']
    dtypes.remove(object)
    for dt in dtypes:
        pytest.raises(ValueError, CoordinateSystem, 'ijk', 'test', dt)
    # compound dtype
    dtype = np.dtype([('field1', '<f8'), ('field2', '<i4')])
    pytest.raises(ValueError, CoordinateSystem, 'ijk', 'test', dtype)
    # valid dtypes
    dtypes = (np.sctypes['int'] + np.sctypes['float'] + np.sctypes['complex'] +
              [object])
    for dt in dtypes:
        cs = CoordinateSystem('ij', coord_dtype=dt)
        assert cs.coord_dtype == dt
        cs_dt = [(f, dt) for f in 'ij']
        assert cs.dtype == np.dtype(cs_dt)
        # Check product too
        cs2 = CoordinateSystem('xy', coord_dtype=dt)
        assert (product(cs, cs2) ==
                     CoordinateSystem('ijxy', name='product', coord_dtype=dt))
    # verify assignment fails
    pytest.raises(AttributeError, setattr, cs, 'dtype', np.dtype(cs_dt))
    pytest.raises(AttributeError, setattr, cs, 'coord_dtype', np.float64)


def test_readonly_attrs(eg_cs):
    cs = eg_cs.cs
    pytest.raises(AttributeError, setattr, cs, 'coord_dtype',
                  np.dtype(np.int32))
    pytest.raises(AttributeError, setattr, cs, 'coord_names',
                  ['a','b','c'])
    pytest.raises(AttributeError, setattr, cs, 'dtype',
                  np.dtype([('i', '<f4'), ('j', '<f4'), ('k', '<f4')]))
    pytest.raises(AttributeError, setattr, cs, 'ndim', 4)


def test_index():
    cs = CoordinateSystem('ijk')
    assert cs.index('i') == 0
    assert cs.index('j') == 1
    assert cs.index('k') == 2
    pytest.raises(ValueError, cs.index, 'x')


def test__ne__():
    cs1 = CoordinateSystem('ijk')
    cs2 = CoordinateSystem('xyz')
    assert cs1 != cs2
    cs1 = CoordinateSystem('ijk', coord_dtype='float')
    cs2 = CoordinateSystem('ijk', coord_dtype='int')
    assert cs1 != cs2


def test___eq__():
    c0 = CoordinateSystem('ijk', 'my name', np.float32)
    c1 = CoordinateSystem('ijk', 'my name', np.float32)
    assert c0 == c1
    c2 = CoordinateSystem('ijk', 'another name', np.float32)
    assert c0 != c2
    c3 = CoordinateSystem('ijq', 'my name', np.float32)
    assert c0 != c3
    c4 = CoordinateSystem('ijk', 'my name', np.float64)
    assert c0 != c4


def test_similar_to():
    c0 = CoordinateSystem('ijk', 'my name', np.float32)
    c1 = CoordinateSystem('ijk', 'my name', np.float32)
    assert c0.similar_to(c1)
    c2 = CoordinateSystem('ijk', 'another name', np.float32)
    assert c0.similar_to(c2)
    c3 = CoordinateSystem('ijq', 'my name', np.float32)
    assert not c0.similar_to(c3)
    c4 = CoordinateSystem('ijk', 'my name', np.float64)
    assert not c0.similar_to(c4)


def test___str__(eg_cs):
    s = str(eg_cs.cs)
    assert s == "CoordinateSystem(coord_names=('i', 'j', 'k'), name='test', coord_dtype=float32)"


def test_is_coordsys():
    # Test coordinate system check
    csys = CoordinateSystem('ijk')
    assert is_coordsys(csys)
    class C: pass
    c = C()
    assert not is_coordsys(c)
    c.coord_names = []
    assert not is_coordsys(c)
    c.name = ''
    assert not is_coordsys(c)
    c.coord_dtype = np.float64
    assert is_coordsys(c)
    # Distinguish from CoordSysMaker
    class C:
        coord_names = []
        name = ''
        coord_dtype=np.float64
        def __call__(self):
            pass
    assert not is_coordsys(C())
    assert not is_coordsys(CoordSysMaker('xyz'))


def test_checked_values():
    cs = CoordinateSystem('ijk', name='voxels', coord_dtype=np.float32)
    x = np.array([1, 2, 3], dtype=np.int16)
    xc = cs._checked_values(x)
    np.allclose(xc, x)
    # wrong shape
    pytest.raises(CoordinateSystemError, cs._checked_values, x.reshape(3,1))
    # wrong length
    pytest.raises(CoordinateSystemError, cs._checked_values, x[0:2])
    # wrong dtype
    x = np.array([1,2,3], dtype=np.float64)
    pytest.raises(CoordinateSystemError, cs._checked_values, x)


def test_safe_dtype():
    pytest.raises(TypeError, safe_dtype, str)
    pytest.raises(TypeError, safe_dtype, str, np.float64)
    pytest.raises(TypeError, safe_dtype, [('x', 'f8')])
    valid_dtypes = []
    valid_dtypes.extend(np.sctypes['complex'])
    valid_dtypes.extend(np.sctypes['float'])
    valid_dtypes.extend(np.sctypes['int'])
    valid_dtypes.extend(np.sctypes['uint'])
    for dt in valid_dtypes:
        sdt = safe_dtype(dt)
        assert sdt == dt
    # test a few upcastings
    dt = safe_dtype(np.float32, np.int16, np.bool_)
    assert dt == np.float32
    dt = safe_dtype(np.float32, np.int64, np.uint32)
    assert dt == np.float64
    # test byteswapped types - isbuiltin will be false
    orig_dt = np.dtype('f')
    dt = safe_dtype(orig_dt, np.int64, np.uint32)
    assert dt == np.float64
    swapped_dt = orig_dt.newbyteorder()
    dt = safe_dtype(swapped_dt, np.int64, np.uint32)
    assert dt == np.float64


def test_product():
    # int32 + int64 => int64
    ax1 = CoordinateSystem('x', coord_dtype=np.int32)
    ax2 = CoordinateSystem('y', coord_dtype=np.int64)
    cs = product(ax1, ax2)
    # assert up-casting of dtype
    assert cs.coord_dtype == np.dtype(np.int64)
    # assert composed dtype
    assert cs.dtype == np.dtype([('x', np.int64), ('y', np.int64)])
    # the axes should be typecast in the CoordinateSystem but
    # uneffected themselves
    assert ax1.dtype == np.dtype([('x', np.int32)])
    assert ax2.dtype == np.dtype([('y', np.int64)])

    # float32 + int64 => float64
    ax1 = CoordinateSystem('x', coord_dtype=np.float32)
    cs = product(ax1, ax2)
    assert cs.coord_dtype == np.dtype(np.float64)
    assert cs.dtype == np.dtype([('x', np.float64),
                                     ('y', np.float64)])
    # int16 + complex64 => complex64
    ax1 = CoordinateSystem('x', coord_dtype=np.int16)
    ax2 = CoordinateSystem('y', coord_dtype=np.complex64)
    # Order of the params effects order of dtype but not resulting value type
    cs = product(ax2, ax1)
    assert cs.coord_dtype == np.complex64
    assert cs.dtype == np.dtype([('y', np.complex64),
                                     ('x', np.complex64)])
    # Passing name as argument
    cs = product(ax2, ax1, name='a name')
    assert cs.name == 'a name'
    # Anything else as kwarg -> error
    pytest.raises(TypeError, product, ax2, ax1, newarg='a name')


def test_coordsys_maker():
    # Things that help making coordinate maps
    ax_names = list('ijklm')
    nl = len(ax_names)
    cs_maker = CoordSysMaker(ax_names, 'myname')
    for i in range(1,nl+1):
        assert (cs_maker(i) ==
                     CoordinateSystem(ax_names[:i], 'myname', np.float64))
    pytest.raises(CoordSysMakerError, cs_maker, nl+1)
    # You can pass in your own name
    assert (cs_maker(i, 'anothername') ==
                 CoordinateSystem(ax_names[:i+1], 'anothername', np.float64))
    # And your own dtype if you really want
    assert (cs_maker(i, coord_dtype=np.int32) ==
                 CoordinateSystem(ax_names[:i+1], 'myname', np.int32))


def test_is_coordsys_maker():
    # Test coordinate system check
    cm = CoordSysMaker('xyz')
    assert is_coordsys_maker(cm)
    class C: pass
    c = C()
    assert not is_coordsys_maker(c)
    c.coord_names = []
    assert not is_coordsys_maker(c)
    c.name = ''
    assert not is_coordsys_maker(c)
    c.coord_dtype = np.float64
    assert not is_coordsys_maker(c)
    # Distinguish from CoordinateSystem
    class C:
        coord_names = []
        name = ''
        coord_dtype=np.float64
        def __call__(self):
            pass
    assert is_coordsys_maker(C())
    assert not is_coordsys_maker(CoordinateSystem('ijk'))
