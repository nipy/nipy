import numpy as np
from neuroimaging.testing import *

from neuroimaging.core.reference.coordinate_map import CoordinateMap, ConcatenatedComaps, \
     ConcatenatedIdenticalComaps
from neuroimaging.core.reference.mapping import Affine

from neuroimaging.testing import anatfile, funcfile

from neuroimaging.core.api import load_image



class test_comap(TestCase):

    def setUp(self):
        self.img = load_image(anatfile)

    # FIXME: "concatenated and replicated CoordinateMaps need to be fixed"
    @dec.skipknownfailure
    def test_concat(self):
        self.fail("concatenated and replicated CoordinateMaps need to be fixed")

        comaps = ConcatenatedComaps([self.img.comap]*5)
        self.assertEquals(tuple(comaps.shape), (5,) + tuple(self.img.comap.shape))
        z = comaps.mapping([4,5,6,7])
        a = comaps.subcomap(0)
        x = a.mapping([5,6,7])

    # FIXME: "concatenated and replicated CoordinateMaps need to be fixed"
    @dec.skipknownfailure
    def test_replicate(self):
        self.fail("concatenated and replicated CoordinateMaps need to be fixed")

        comaps = self.img.comap.replicate(4)
        self.assertEquals(tuple(comaps.shape), (4,) + tuple(self.img.comap.shape))
        z = comaps.mapping([2,5,6,7])
        a = comaps.subcomap(0)
        x = a.mapping([5,6,7])

    # FIXME: "concatenated and replicated CoordinateMaps need to be fixed"
    @dec.skipknownfailure
    def test_replicate2(self):
        """
        Test passing
        """
        self.fail("concatenated and replicated CoordinateMaps need to be fixed")
        comaps = self.img.comap.replicate(4)
        comaps.python2matlab()

    # FIXME: "concatenated and replicated CoordinateMaps need to be fixed"
    @dec.skipknownfailure
    def test_concat2(self):
        """
        Test passing
        """
        self.fail("concatenated and replicated CoordinateMaps need to be fixed")
        
        comaps = ConcatenatedIdenticalComaps(self.img.comap, 4)
        comaps.python2matlab()

    # FIXME: "concatenated and replicated CoordinateMaps need to be fixed"
    @dec.skipknownfailure
    def test_concat3(self):
        """
        Test failing
        """
        self.fail("concatenated and replicated CoordinateMaps need to be fixed")
        comaps = ConcatenatedComaps([self.img.comap]*4)
        comaps.python2matlab()

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
        g = CoordinateMap.from_affine(a, ['zspace', 'xspace'], (20,30))







