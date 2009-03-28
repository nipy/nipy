from neuroimaging.testing import *

from neuroimaging.core.reference.slices import bounding_box, \
  zslice, yslice, xslice
from neuroimaging.core.reference.coordinate_map import Affine

# Names for a 3D axis set
names = ['xspace','yspace','zspace']

class test_Slice(TestCase):

    def test_bounding_box(self):
        shape = (10, 10, 10)
        coordmap = Affine.identity(names)
        #print coordmap.affine.dtype, 'affine'
        self.assertEqual(bounding_box(coordmap, shape), [[0., 9.], [0, 9], [0, 9]])

    def test_box_slice(self):
        coordmap = Affine.identity(names)
        t = zslice(5, [0, 9], [0, 9], coordmap.output_coords, (10,10))
        assert_almost_equal(t.coordmap.affine, [[ 0.,  0.,  5.],
                                                [ 1.,  0.,  0.],
                                                [ 0.,  1.,  0.],
                                                [ 0.,  0.,  1.]])
        

        t = yslice(4, [0, 9], [0, 9], coordmap.output_coords, (10,10))
        assert_almost_equal(t.coordmap.affine, [[ 1.,  0.,  0.],
                                                [ 0.,  0.,  4.],
                                                [ 0.,  1.,  0.],
                                                [ 0.,  0.,  1.]])
        
        t = xslice(3, [0, 9], [0, 9], coordmap.output_coords, (10,10))
        assert_almost_equal(t.coordmap.affine, [[ 1.,  0.,  0.],
                                                [ 0.,  1.,  0.],
                                                [ 0.,  0.,  3.],
                                                [ 0.,  0.,  1.]])
        







