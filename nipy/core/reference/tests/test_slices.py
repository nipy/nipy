from nipy.testing import *

from nipy.core.reference.slices import bounding_box, \
  zslice, yslice, xslice
from nipy.core.reference.coordinate_map import AffineTransform

# Names for a 3D axis set
names = ['xspace','yspace','zspace']

class test_Slice(TestCase):

    def test_bounding_box(self):
        shape = (10, 14, 16)
        coordmap = AffineTransform.identity(names)
        #print coordmap.affine.dtype, 'affine'
        self.assertEqual(bounding_box(coordmap, shape), ([0., 9.], [0, 13], [0, 15]))

def test_box_slice():
    t = xslice(5, ([0, 9], 10), ([0, 9], 10))
    yield assert_almost_equal,t.affine, [[ 0.,  0.,  5.],
                                          [ 1.,  0.,  0.],
                                          [ 0.,  1.,  0.],
                                          [ 0.,  0.,  1.]]


    t = yslice(4, ([0, 9], 10), ([0, 9], 10))
    yield assert_almost_equal, t.affine, [[ 1.,  0.,  0.],
                                            [ 0.,  0.,  4.],
                                            [ 0.,  1.,  0.],
                                            [ 0.,  0.,  1.]]

    t = zslice(3, ([0, 9], 10), ([0, 9], 10))
    yield assert_almost_equal, t.affine, [[ 1.,  0.,  0.],
                                            [ 0.,  1.,  0.],
                                            [ 0.,  0.,  3.],
                                            [ 0.,  0.,  1.]]
        







