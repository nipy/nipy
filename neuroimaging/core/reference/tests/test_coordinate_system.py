import numpy as np
import nose.tools
from neuroimaging.testing import *
from neuroimaging.core.reference.coordinate_system import CoordinateSystem, Coordinate

class empty:
    pass

E = empty()

def setup():
    E.name = "test"
    E.axes = [Coordinate(n) for n in ['zspace', 'yspace', 'xspace']]
    E.c = CoordinateSystem(E.name, E.axes)

def test_CoordinateSystem():
    nose.tools.assert_equal(E.name, E.c.name)
    nose.tools.assert_equal([ax.name for ax in E.axes],
                            [ax.name for ax in E.c.axes])

def test_axisnames():
    nose.tools.assert_equal([ax.name for ax in E.axes],
                            E.c.axisnames)

def test___getitem__():
    for ax in E.axes:
        nose.tools.assert_equal(E.c[ax.name], ax)
    nose.tools.assert_raises(KeyError, E.c.__getitem__, "bad_name")

def test___setitem__():
    nose.tools.assert_raises(TypeError, E.c.__setitem__, E.c, "any_name", None)

def test___eq__():
    c1 = CoordinateSystem(E.c.name, E.c.axes)
    nose.tools.assert_true(c1 == E.c)

def test_reorder():
    new_order = [1, 2, 0]
    new_c = E.c.reorder("new", new_order)
    nose.tools.assert_equal(new_c.name, "new")
    generic = [Coordinate(n) for n in ['zspace', 'yspace', 'xspace']]
    print E.c
    print generic
    for i in range(3):
        nose.tools.assert_equal(E.c[generic[i].name],
                                new_c[generic[i].name])

    new_c = E.c.reorder(None, new_order)
    nose.tools.assert_equal(new_c.name, E.c.name)

def test___str__():
    s = str(E.c)

def test_dtype():

    ax1 = Coordinate('x', dtype=np.int32)
    ax2 = Coordinate('y', dtype=np.int64)

    cs = CoordinateSystem('input', [ax1, ax2])
    nose.tools.assert_equal(cs.builtin, np.dtype(np.int64))

    # the axes should be typecast in the CoordinateSystem

    nose.tools.assert_equal(cs['x'].dtype, np.dtype([('x', np.int64)]))
    nose.tools.assert_equal(ax1.dtype, np.dtype([('x', np.int32)]))





