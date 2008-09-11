from neuroimaging.testing import *

from neuroimaging.core.reference.slices import bounding_box, \
  zslice, yslice, xslice
from neuroimaging.core.reference.grid import CoordinateMap

# Names for a 3D axis set
names = ['xspace','yspace','zspace']

class test_Slice(TestCase):

    def test_bounding_box(self):
        shape = (10, 10, 10)
        grid = CoordinateMap.identity(names, shape)
        self.assertEqual(bounding_box(grid), [[0., 9.], [0, 9], [0, 9]])

    def test_box_slice(self):
        grid = CoordinateMap.identity(names, (10,10,10,))
        t = zslice(5, [0, 9], [0, 9], grid.output_coords, (10,10)).affine
        assert_almost_equal(t, [[ 0.,  0.,  5.],
                                [ 1.,  0.,  0.],
                                [ 0.,  1.,  0.],
                                [ 0.,  0.,  1.]])
        

        t = yslice(4, [0, 9], [0, 9], grid.output_coords, (10,10)).affine
        assert_almost_equal(t, [[ 1.,  0.,  0.],
                                [ 0.,  0.,  4.],
                                [ 0.,  1.,  0.],
                                [ 0.,  0.,  1.]])
        
        t = xslice(3, [0, 9], [0, 9], grid.output_coords, (10,10)).affine
        assert_almost_equal(t, [[ 1.,  0.,  0.],
                                [ 0.,  1.,  0.],
                                [ 0.,  0.,  3.],
                                [ 0.,  0.,  1.]])
        







