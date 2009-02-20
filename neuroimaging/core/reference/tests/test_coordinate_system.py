
import numpy as np

from neuroimaging.testing import *
from neuroimaging.core.reference.coordinate_system import CoordinateSystem, product

class empty:
    pass

E = empty()

def setup():
    E.name = "test"
    E.axes = ['zspace', 'yspace', 'xspace']
    E.c = CoordinateSystem(E.axes, E.name)

def test_iterator_coordinate():
    def gen():
        yield 'i'
        yield 'j'
        yield 'k'
    coordsys = CoordinateSystem(gen(), name='test_iter')
    assert_equal(coordsys.coord_names, ['i','j','k'])


def test_unique_coord_names():
    unique = ['i','j','k']
    notuniq = ['i','i','k']
    coordsys = CoordinateSystem(unique)
    yield assert_equal, coordsys.coord_names, unique
    yield assert_raises, ValueError, CoordinateSystem, notuniq


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
    yield (assert_raises, AttributeError, setattr, cs, 'coord_dtype', np.float)


def test_index():
    cs = CoordinateSystem('ijk')
    yield assert_equal, cs.index('i'), 0
    yield assert_equal, cs.index('j'), 1
    yield assert_equal, cs.index('k'), 2
    yield assert_raises, ValueError, cs.index, 'x'


def test_CoordinateSystem():
    yield assert_equal, E.name, E.c.name
    yield assert_equal, E.c.coord_names, E.axes


def test___eq__():
    c1 = CoordinateSystem(E.c.coord_names, E.c.name)
    assert_true(c1 == E.c)


def test_reordered():
    new_order = [1, 2, 0]
    new_c = E.c.reordered("new", new_order)
    yield assert_equal, new_c.name, "new"
    print new_c.coord_names
    for i in range(3):
        yield assert_equal, E.c.index(new_c.coord_names[i]), new_order[i]

    new_c = E.c.reordered(None, new_order)
    yield assert_equal, new_c.name, E.c.name

def test___str__():
    s = str(E.c)

def test_dtype():

    ax1 = CoordinateSystem('x', coord_dtype=np.int32)
    ax2 = CoordinateSystem('y', coord_dtype=np.int64)

    cs = product(ax1, ax2)
    yield assert_equal, cs.coord_dtype, np.dtype(np.int64)
    yield assert_equal, cs.dtype, np.dtype([('x', np.int64), ('y', np.int64)])

    # the axes should be typecast in the CoordinateSystem
    yield assert_equal, ax1.dtype, np.dtype([('x', np.int32)])
