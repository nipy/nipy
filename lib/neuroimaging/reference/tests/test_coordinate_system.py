import unittest
import numpy as N

from neuroimaging.reference.coordinate_system import CoordinateSystem, \
     VoxelCoordinateSystem, DiagonalCoordinateSystem
from neuroimaging.reference.mni import generic

class CoordinateSystemTest(unittest.TestCase):

    def _init(self):
        self.name = "test"
        self.axes = generic
        self.c = CoordinateSystem(self.name, self.axes)

    def test_CoordinateSystem(self):
        self._init()
        self.assertEquals(self.name, self.c.name)
        self.assertEquals([ax.name for ax in self.axes],
                          [ax.name for ax in self.c.axes])

    def test_hasaxis(self):
        self._init()
        for ax in self.axes:
            self.assertTrue(self.c.hasaxis(ax.name))

    def test_getaxis(self):
        self._init()
        for ax in self.axes:
            self.assertEquals(self.c.getaxis(ax.name), ax)

    def test___getitem__(self):
        self._init()
        for ax in self.axes:
            self.assertEquals(self.c[ax.name], ax)

        # this is kinda ugly...
        f = lambda s: self.c[s]
        self.assertRaises(KeyError, f, "bad_name")

    def test___setitem__(self):
        self._init()
        # FIXME: how do we make something like this work?
        #self.assertRaises(TypeError, eval, 'self.c["any_name"] = 1')

    def test___eq__(self):
        self._init()
        c1 = CoordinateSystem(self.c.name, self.c.axes)
        self.assertTrue(c1 == self.c)

    def test_reorder(self):
        self._init()
        new_order = [1, 2, 0]
        new_c = self.c.reorder("new", new_order)
        self.assertEquals(new_c.name, "new")
        for i in range(3):
            self.assertEquals(self.c.getaxis(generic[i]),
                              new_c.getaxis(generic[new_order[i]]))


if __name__ == '__main__':
    unittest.main()
