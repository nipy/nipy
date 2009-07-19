import numpy as np
import copy

from nipy.testing import TestCase, assert_almost_equal, assert_equal


from nipy.core import AffineImage
from .. import generators as gen

class TestGenerator(TestCase):

    def setUp(self):
        shape = (10,20,30)
        self.img  = AffineImage(data=np.zeros(shape), 
                                affine=np.eye(4),
                                world_space='test')
        self.img2 = AffineImage(np.ones(shape), 
                                affine=np.eye(4),
                                world_space='test')
                       
        shape = (3,5,4)
        self.img3 = AffineImage(np.zeros(shape), 
                                affine=np.eye(4),
                                world_space='test')
        self.img4 = AffineImage(np.zeros(shape), 
                                affine=np.eye(4),
                                world_space='test')


    def test_read_slices(self):
        for _, d in gen.slice_generator(self.img):
            self.assertEquals(d.shape, (20, 30))

        for _, d in gen.slice_generator(self.img, axis=1):
            self.assertEquals(d.shape, (10, 30))

        for _, d in gen.slice_generator(self.img, axis=2):
            self.assertEquals(d.shape, (10, 20))

    def test_write_slices(self):
        tmp = np.zeros(self.img.get_data().shape)
        gen.write_data(tmp, gen.slice_generator(self.img.get_data()))
        assert_almost_equal(tmp, self.img.get_data())

        tmp = np.zeros(self.img.get_data().shape)
        gen.write_data(tmp, gen.slice_generator(self.img.get_data(), axis=1))
        assert_almost_equal(tmp, self.img.get_data())

        tmp = np.zeros(self.img.get_data().shape)
        gen.write_data(tmp, gen.slice_generator(self.img.get_data(), axis=2))
        assert_almost_equal(tmp, self.img.get_data())

    def test_multi_slice(self):
        for _, d in gen.slice_generator(self.img, axis=[0, 1]):
            self.assertEquals(d.shape, (30,))

        for _, d in gen.slice_generator(self.img, axis=[2, 1]):
            self.assertEquals(d.shape, (10,))

    def test_multi_slice_write(self):
        a = np.zeros(self.img.get_data().shape)
        gen.write_data(a, gen.slice_generator(self.img.get_data(), axis=[0, 1]))
        assert_almost_equal(a, np.asarray(self.img))

    def test_parcel(self):
        parcelmap = np.zeros(self.img3.get_data().shape)
        parcelmap[0,0,0] = 1
        parcelmap[1,1,1] = 1
        parcelmap[2,2,2] = 1
        parcelmap[1,2,1] = 2
        parcelmap[2,3,2] = 2
        parcelmap[0,1,0] = 2
        parcelseq = (0, 1, 2, 3)
        expected = [np.product(self.img3.shape) - 6, 3, 3, 0]
        iterator = gen.data_generator(self.img3.get_data(), 
                                      gen.parcels(parcelmap, labels=parcelseq))

        for i, pair in enumerate(iterator):
            s, d = pair
            self.assertEqual((expected[i],), d.shape)

        iterator = gen.data_generator(self.img3.get_data(), 
                                    gen.parcels(parcelmap))
        for i, pair in enumerate(iterator):
            s, d = pair
            self.assertEqual((expected[i],), d.shape)

    def test_parcel_write(self):
        parcelmap = np.zeros(self.img3.get_data().shape)
        parcelmap[0,0,0] = 1
        parcelmap[1,1,1] = 1
        parcelmap[2,2,2] = 1
        parcelmap[1,2,1] = 2
        parcelmap[2,3,2] = 2
        parcelmap[0,1,0] = 2
        parcelseq = (0, 1, 2, 3)
        expected = [np.product(self.img3.get_data().shape) - 6, 3, 3, 0]
        iterator = gen.parcels(parcelmap, labels=parcelseq)

        for i, s in enumerate(iterator):
            value = np.arange(expected[i])
            self.img3[s] = value

        iterator = gen.parcels(parcelmap, labels=parcelseq)
        for i, pair in enumerate(gen.data_generator(self.img3.get_data(), 
                                                    iterator)):
            s, d = pair
            self.assertEqual((expected[i],), d.shape)
            assert_equal(d, np.arange(expected[i]))

        iterator = gen.parcels(parcelmap)
        for i, s in enumerate(iterator):
            value = np.arange(expected[i])
            self.img3[s] = value

        iterator = gen.parcels(parcelmap)
        for i, pair in enumerate(gen.data_generator(self.img3.get_data(), 
                                                    iterator)):
            s, d = pair
            self.assertEqual((expected[i],), d.shape)
            assert_equal(d, np.arange(expected[i]))

    def test_parcel_copy(self):
        parcelmap = np.zeros(self.img3.get_data().shape)
        parcelmap[0,0,0] = 1
        parcelmap[1,1,1] = 1
        parcelmap[2,2,2] = 1
        parcelmap[1,2,1] = 2
        parcelmap[2,3,2] = 2
        parcelmap[0,1,0] = 2
        parcelseq = (0, 1, 2, 3)
        expected = [np.product(self.img3.get_data().shape) - 6, 3, 3, 0]
        iterator = gen.parcels(parcelmap, labels=parcelseq)
        tmp = copy.copy(self.img3)

        gen_parcels = gen.parcels(parcelmap, labels=parcelseq)
        new_iterator = gen.data_generator(tmp, gen_parcels)

        for i, slice_ in enumerate(new_iterator):
            self.assertEqual((expected[i],), slice_[1].shape)


    def test_sliceparcel(self):
        parcelmap = np.asarray([[0,0,0,1,2],[0,0,1,1,2],[0,0,0,0,2]])
        parcelseq = ((1, 2), 0, 2)
        
        o = np.zeros(parcelmap.shape)
        iterator = gen.slice_parcels(parcelmap, labels=parcelseq)

        for i, pair in enumerate(iterator):
            a, s = pair
            o[a][s] = i
        assert_equal(o,
                     np.array([[1,1,1,0,2],
                               [4,4,3,3,5],
                               [7,7,7,7,8]]))





