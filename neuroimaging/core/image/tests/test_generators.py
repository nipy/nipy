import numpy as np
from neuroimaging.testing import *

from neuroimaging.core.api import data_generator, parcels, write_data, slice_generator
from neuroimaging.core.api import Image, load_image, save_image
import neuroimaging.core.reference.coordinate_map as coordinate_map
import neuroimaging.core.image.generators as g

class test_Generator(TestCase):

    def setUp(self):
        names = ['zspace', 'yspace', 'xspace']
        shape = (10,20,30)
        self.img = Image(np.zeros(shape), coordinate_map.Affine.from_start_step(names, names, (0,)*3, (1,)*3))
        self.img2 = Image(np.ones(shape), coordinate_map.Affine.from_start_step(names, names, (0,)*3, (1,)*3))
                       
        shape = (3,5,4)
        self.img3 = Image(np.zeros(shape), coordinate_map.Affine.from_start_step(names, names, (0,)*3, (1,)*3))
        self.img4 = Image(np.zeros(shape), coordinate_map.Affine.from_start_step(names, names, (0,)*3, (1,)*3))


    def test_read_slices(self):
        for _, d in slice_generator(self.img):
            self.assertEquals(d.shape, (20, 30))

        for _, d in slice_generator(self.img, axis=1):
            self.assertEquals(d.shape, (10, 30))

        for _, d in slice_generator(self.img, axis=2):
            self.assertEquals(d.shape, (10, 20))

    def test_write_slices(self):
        tmp = np.zeros(self.img.shape)
        write_data(tmp, slice_generator(self.img))
        assert_almost_equal(tmp, np.asarray(self.img))

        tmp = np.zeros(self.img.shape)
        write_data(tmp, slice_generator(self.img, axis=1))
        assert_almost_equal(tmp, np.asarray(self.img))

        tmp = np.zeros(self.img.shape)
        write_data(tmp, slice_generator(self.img, axis=2))
        assert_almost_equal(tmp, np.asarray(self.img))

    def test_multi_slice(self):
        for _, d in slice_generator(self.img, axis=[0, 1]):
            self.assertEquals(d.shape, (30,))

        for _, d in slice_generator(self.img, axis=[2, 1]):
            self.assertEquals(d.shape, (10,))

    def test_multi_slice_write(self):
        a = np.zeros(self.img.shape)
        write_data(a, slice_generator(self.img, axis=[0, 1]))
        assert_almost_equal(a, np.asarray(self.img))

    def test_parcel(self):
        parcelmap = np.zeros(self.img3.shape)
        parcelmap[0,0,0] = 1
        parcelmap[1,1,1] = 1
        parcelmap[2,2,2] = 1
        parcelmap[1,2,1] = 2
        parcelmap[2,3,2] = 2
        parcelmap[0,1,0] = 2
        parcelseq = (0, 1, 2, 3)
        expected = [np.product(self.img3.shape) - 6, 3, 3, 0]
        iterator = g.data_generator(self.img3, g.parcels(parcelmap, labels=parcelseq))

        for i, pair in enumerate(iterator):
            s, d = pair
            self.assertEqual((expected[i],), d.shape)

        iterator = g.data_generator(self.img3, g.parcels(parcelmap))
        for i, pair in enumerate(iterator):
            s, d = pair
            self.assertEqual((expected[i],), d.shape)

    def test_parcel_write(self):
        parcelmap = np.zeros(self.img3.shape)
        parcelmap[0,0,0] = 1
        parcelmap[1,1,1] = 1
        parcelmap[2,2,2] = 1
        parcelmap[1,2,1] = 2
        parcelmap[2,3,2] = 2
        parcelmap[0,1,0] = 2
        parcelseq = (0, 1, 2, 3)
        expected = [np.product(self.img3.shape) - 6, 3, 3, 0]
        iterator = g.parcels(parcelmap, labels=parcelseq)

        for i, s in enumerate(iterator):
            value = np.arange(expected[i])
            self.img3[s] = value

        iterator = g.parcels(parcelmap, labels=parcelseq)
        for i, pair in enumerate(g.data_generator(self.img3, iterator)):
            s, d = pair
            self.assertEqual((expected[i],), d.shape)
            assert_equal(d, np.arange(expected[i]))

        iterator = g.parcels(parcelmap)
        for i, s in enumerate(iterator):
            value = np.arange(expected[i])
            self.img3[s] = value

        iterator = g.parcels(parcelmap)
        for i, pair in enumerate(g.data_generator(self.img3, iterator)):
            s, d = pair
            self.assertEqual((expected[i],), d.shape)
            assert_equal(d, np.arange(expected[i]))

    def test_parcel_copy(self):
        parcelmap = np.zeros(self.img3.shape)
        parcelmap[0,0,0] = 1
        parcelmap[1,1,1] = 1
        parcelmap[2,2,2] = 1
        parcelmap[1,2,1] = 2
        parcelmap[2,3,2] = 2
        parcelmap[0,1,0] = 2
        parcelseq = (0, 1, 2, 3)
        expected = [np.product(self.img3.shape) - 6, 3, 3, 0]
        iterator = g.parcels(parcelmap, labels=parcelseq)
        tmp = Image(np.asarray(self.img3), self.img3.coordmap)

        new_iterator = g.data_generator(tmp, g.parcels(parcelmap, labels=parcelseq))

        for i, slice_ in enumerate(new_iterator):
            self.assertEqual((expected[i],), slice_[1].shape)


    def test_sliceparcel(self):
        parcelmap = np.asarray([[0,0,0,1,2],[0,0,1,1,2],[0,0,0,0,2]])
        parcelseq = ((1, 2), 0, 2)
        
        o = np.zeros(parcelmap.shape)
        iterator = g.slice_parcels(parcelmap, labels=parcelseq)

        for i, pair in enumerate(iterator):
            a, s = pair
            o[a][s] = i
        assert_equal(o,
                                np.array([[1,1,1,0,2],
                                          [4,4,3,3,5],
                                          [7,7,7,7,8]]))








