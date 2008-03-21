import numpy as np
from numpy.testing import NumpyTest, NumpyTestCase

from neuroimaging.core.api import data_generator, parcels, write_data, slice_generator
from neuroimaging.core.api import Image, load_image, save_image
import neuroimaging.core.reference.axis as axis
import neuroimaging.core.reference.grid as grid

class test_Iterator(NumpyTestCase):

    def setUp(self):
        self.img = Image(np.zeros((10, 20, 30)), grid.SamplingGrid.from_start_step(shape=(10,20,30), step=(1,)*3, start=(0,)*3))
        self.img2 = Image(np.ones((10, 20, 30)), grid.SamplingGrid.from_start_step(shape=(10,20,30), step=(1,)*3, start=(0,)*3))
        self.img3 = Image(np.zeros((3, 5, 4)), grid.SamplingGrid.from_start_step(shape=(3,5,4), step=(1,)*3, start=(0,)*3))
        self.img4 = Image(np.ones((3, 5, 4)), grid.SamplingGrid.from_start_step(shape=(3,5,4), step=(1,)*3, start=(0,)*3))

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
        np.testing.assert_almost_equal(tmp, np.asarray(self.img))

        tmp = np.zeros(self.img.shape)
        write_data(tmp, slice_generator(self.img, axis=1))
        np.testing.assert_almost_equal(tmp, np.asarray(self.img))

        tmp = np.zeros(self.img.shape)
        write_data(tmp, slice_generator(self.img, axis=2))
        np.testing.assert_almost_equal(tmp, np.asarray(self.img))

    def test_multi_slice(self):
        for _, d in slice_generator(self.img, axis=[0, 1]):
            self.assertEquals(d.shape, (30,))

        for _, d in slice_generator(self.img, axis=[2, 1]):
            self.assertEquals(d.shape, (10,))

    def test_multi_slice_write(self):
        a = np.zeros(self.img.shape)
        write_data(a, slice_generator(self.img, axis=[0, 1]))
        np.testing.assert_almost_equal(a, np.asarray(self.img))

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
        iterator = ParcelIterator(self.img3, parcelmap, parcelseq)
        for i, slice_ in enumerate(iterator):
            self.assertEqual((expected[i],), slice_.shape)

        iterator = ParcelIterator(self.img3, parcelmap)
        for i, slice_ in enumerate(iterator):
            self.assertEqual((expected[i],), slice_.shape)

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
        iterator = ParcelIterator(self.img3, parcelmap, parcelseq, mode='w')

        for i, slice_ in enumerate(iterator):
            value = np.arange(expected[i])
            slice_.set(value)

        iterator = ParcelIterator(self.img3, parcelmap, parcelseq)
        for i, slice_ in enumerate(iterator):
            self.assertEqual((expected[i],), slice_.shape)
            np.testing.assert_equal(slice_, np.arange(expected[i]))

        iterator = ParcelIterator(self.img3, parcelmap, mode='w')
        for i, slice_ in enumerate(iterator):
            value = np.arange(expected[i])
            slice_.set(value)

        iterator = ParcelIterator(self.img3, parcelmap)
        for i, slice_ in enumerate(iterator):
            self.assertEqual((expected[i],), slice_.shape)
            np.testing.assert_equal(slice_, np.arange(expected[i]))



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
        iterator = ParcelIterator(self.img3, parcelmap, parcelseq)
        tmp = Image(np.array(self.img3), self.img3.grid)

        new_iterator = iterator.copy(tmp)

        for i, slice_ in enumerate(new_iterator):
            self.assertEqual((expected[i],), slice_.shape)

        iterator = ParcelIterator(self.img3, parcelmap)
        for i, slice_ in enumerate(new_iterator):
            self.assertEqual((expected[i],), slice_.shape)

    def test_sliceparcel(self):
        parcelmap = np.asarray([[0,0,0,1,2],[0,0,1,1,2],[0,0,0,0,2]])
        parcelseq = ((1, 2), 0, 2)
        iterator = SliceParcelIterator(self.img3, parcelmap, parcelseq)
        for i, slice_ in enumerate(iterator):
            pm = parcelmap[i]
            ps = parcelseq[i]
            try:
                x = len([n for n in pm if n in ps])
            except TypeError:
                x = len([n for n in pm if n == ps])
            self.assertEqual(x, slice_.shape[0])
            self.assertEqual(self.img3[:].shape[2:], slice_.shape[1:])

    def test_sliceparcel1(self):
        parcelmap = np.asarray([[0,0,0,1,2],[0,0,1,1,2],[0,0,0,0,2]])
        self.assertRaises(ValueError, SliceParcelIterator, self.img3, \
                          parcelmap, None)
        return

    def test_sliceparcel_copy(self):
        parcelmap = np.asarray([[0,0,0,1,2],[0,0,1,1,2],[0,0,0,0,2]])
        parcelseq = ((1, 2), 0, 2)
        iterator = SliceParcelIterator(self.img3, parcelmap, parcelseq)

        new_iterator = iterator.copy(self.img4)

        for i, slice_ in enumerate(new_iterator):
            pm = parcelmap[i]
            ps = parcelseq[i]
            try:
                x = len([n for n in pm if n in ps])
            except TypeError:
                x = len([n for n in pm if n == ps])
            self.assertEqual(x, slice_.shape[0])
            self.assertEqual(self.img4[:].shape[2:], slice_.shape[1:])

    def test_sliceparcel_write(self):
        parcelmap = np.asarray([[0,0,0,1,2],[0,0,1,1,2],[0,0,0,0,2]])
        parcelseq = ((1, 2), 0, 2)
        iterator = SliceParcelIterator(self.img3, parcelmap, parcelseq, mode='w')
        for i, slice_ in enumerate(iterator):
            pm = parcelmap[i]
            ps = parcelseq[i]
            try:
                x = len([n for n in pm if n in ps])
            except TypeError:
                x = len([n for n in pm if n == ps])
            shape = (x, self.img3.shape[2])
            slice_.set(np.ones(shape))

        iterator = SliceParcelIterator(self.img3, parcelmap, parcelseq)
        for i, slice_ in enumerate(iterator):
            pm = parcelmap[i]
            ps = parcelseq[i]
            try:
                x = len([n for n in pm if n in ps])
            except TypeError:
                x = len([n for n in pm if n == ps])
            self.assertEqual(x, slice_.shape[0])
            np.testing.assert_equal(np.ones((x ,self.img3.shape[2])), slice_)


from neuroimaging.utils.testutils import make_doctest_suite
test_suite = make_doctest_suite('neuroimaging.core.image.iterators')

if __name__ == '__main__':
    NumpyTest.run()
