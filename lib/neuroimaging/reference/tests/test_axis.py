import unittest
import numpy as N
import scipy

from neuroimaging.reference.axis import Axis, VoxelAxis, RegularAxis

class AxisTest(unittest.TestCase):

    def testAxis(self):
        _axis = Axis(name='xspace')
        self.assertRaises(ValueError, Axis, name='bad_value')

    def testVoxelAxis(self):
        _axis = VoxelAxis(name='xspace', length=20)
        self.assertEquals(_axis.length, 20)
        self.assertEquals(len(_axis), 20)
        scipy.testing.assert_almost_equal(_axis.values(), N.arange(20))        

    def testRegularAxis(self):
        _axis = RegularAxis(name='yspace', length=20, start=1.0, step=2.0)
        v = N.arange(1,41,2).astype(N.float64)
        scipy.testing.assert_almost_equal(_axis.values(), v)
        print _axis

    def testRegularAxisEquality(self):
        a1 = RegularAxis(name='yspace', length=20, start=1.0, step=2.0)
        a2 = RegularAxis(name='yspace', length=20, start=1.0, step=2.0)
        self.assertEquals(a1, a2)

if __name__ == '__main__':
    unittest.main()
