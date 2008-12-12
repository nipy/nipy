import numpy as np

import nose.tools

from neuroimaging.testing import *

from neuroimaging.core.reference.coordinate_system import CoordinateSystem
from neuroimaging.core.reference.axis import Axis

class empty:
    pass

E = empty()

def setup():
    E.name = "test"
    E.axes = [Axis(n) for n in ['zspace', 'yspace', 'xspace']]
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
    nose.tools.assert_raises(TypeError, eval, 'E.c.__setitem__("any_name", None)')

def test___eq__():
    c1 = CoordinateSystem(E.c.name, E.c.axes)
    nose.tools.assertTrue(c1 == E.c)

def test_reorder():
    new_order = [1, 2, 0]
    new_c = E.c.reorder("new", new_order)
    nose.tools.assert_equal(new_c.name, "new")
    generic = [Axis(n) for n in ['zspace', 'yspace', 'xspace']]
    print E.c
    print generic
    for i in range(3):
        nose.tools.assert_equal(E.c[generic[i].name],
                                new_c[generic[[i]].name])

    new_c = E.c.reorder(None, new_order)
    nose.tools.assert_equal(new_c.name, E.c.name)

def test___str__():
    s = str(E.c)

        






