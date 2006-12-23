import unittest

from neuroimaging.core.reference.coordinate_system import CoordinateSystem, \
     VoxelCoordinateSystem
from neuroimaging.core.reference.axis import generic

class CoordinateSystemTest(unittest.TestCase):

    def setUp(self):
        self.name = "test"
        self.axes = generic
        self.c = CoordinateSystem(self.name, self.axes)

    def test_CoordinateSystem(self):
        self.assertEquals(self.name, self.c.name)
        self.assertEquals([ax.name for ax in self.axes],
                          [ax.name for ax in self.c.axes()])

    def test_hasaxis(self):
        for ax in self.axes:
            self.assertTrue(self.c.hasaxis(ax.name))

    def test_getaxis(self):
        for ax in self.axes:
            self.assertEquals(self.c.getaxis(ax.name), ax)

    def test_axisnames(self):
        self.assertEquals([ax.name for ax in self.axes],
                            self.c.axisnames())

    def test___getitem__(self):
        for ax in self.axes:
            self.assertEquals(self.c[ax.name], ax)

        # this is kinda ugly...
        f = lambda s: self.c[s]
        self.assertRaises(KeyError, f, "bad_name")

    def test___setitem__(self):
        # FIXME: how do we make something like this work?
        #self.assertRaises(TypeError, eval, 'self.c["any_name"] = 1')
        self.assertRaises(TypeError, eval, 'self.c.__setitem__("any_name", None)')

    def test___eq__(self):
        c1 = CoordinateSystem(self.c.name, self.c.axes())
        self.assertTrue(c1 == self.c)

    def test_reorder(self):
        new_order = [1, 2, 0]
        new_c = self.c.reorder("new", new_order)
        self.assertEquals(new_c.name, "new")
        for i in range(3):
            self.assertEquals(self.c.getaxis(generic[i]),
                              new_c.getaxis(generic[new_order[i]]))

        new_c = self.c.reorder(None, new_order)
        self.assertEquals(new_c.name, self.c.name)

    def test___str__(self):
        s = str(self.c)

    def test_sub_coords(self):
        new_c = self.c.sub_coords()
        self.assertEquals(new_c.name, self.c.name + "-subgrid")
        self.assertEquals(new_c.axes(), self.c.axes()[1:])
        

class VoxelCoordinateSystemTest(unittest.TestCase):
    def setUp(self):
        self.name = "voxel_test"
        self.axes = generic
        self.shape = [3,4,5]
        self.v = VoxelCoordinateSystem(self.name, self.axes, self.shape)

    def test_VoxelCoordinateSystem(self):
        self.assertEqual(self.name, self.v.name)
        self.assertEquals([ax.name for ax in self.axes],
                          [ax.name for ax in self.v.axes()])
        self.assertEquals(self.shape, self.v.shape)

    def test_isvalid(self):
        self.assertTrue(self.v.isvalid([0,0,0]))
        self.assertTrue(not self.v.isvalid(self.shape))

if __name__ == '__main__':
    unittest.main()
