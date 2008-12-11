import numpy as np
from neuroimaging.testing import *

from neuroimaging.core.api import VoxelAxis, RegularAxis, VoxelCoordinateSystem, CoordinateSystem
from neuroimaging.core.reference.coordinate_map import CoordinateMap

from neuroimaging.core.reference.mapping import Affine

from neuroimaging.testing import anatfile, funcfile

from neuroimaging.core.api import load_image



class test_coordmap(TestCase):

    def setUp(self):
        self.img = load_image(anatfile)



    def test_identity(self):
        shape = (30,40,50)
        i = CoordinateMap.identity(['zspace', 'yspace', 'xshape'], shape=shape)
        self.assertEquals(tuple(i.shape), shape)
        y = i.mapping([3,4,5])
        assert_almost_equal(y, np.array([3,4,5]))

    def test_identity2(self):
        shape = (30, 40)
        self.assertRaises(ValueError, CoordinateMap.identity, ['zspace', 'yspace', 'xspace'], shape)


    def test_from_affine(self):
        a = Affine.identity(2)
        g = CoordinateMap.from_affine('ij', 'xy', a, (20,30))


