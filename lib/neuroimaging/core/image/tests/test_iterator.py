import numpy as N
from numpy.testing import NumpyTest, NumpyTestCase

from neuroimaging.core.image.iterators import Iterator, IteratorItem, \
     ParcelIterator, SliceParcelIterator, fMRIParcelIterator, \
     fMRISliceParcelIterator

from neuroimaging.core.api import Image
from neuroimaging.modalities.fmri import fMRIImage
import neuroimaging.core.reference.axis as axis
import neuroimaging.core.reference.grid as grid

class test_Iterator(NumpyTestCase):

    def setUp(self):
        self.img = Image(N.zeros((10, 20, 30)))
        self.img2 = Image(N.ones((10, 20, 30)))
        self.img3 = Image(N.zeros((3, 5, 4)))
        self.img4 = Image(N.ones((3, 5, 4)))

        self.fmri_img = fMRIImage(N.zeros((3, 4, 5, 6)), grid = grid.SamplingGrid.identity((3,4,5,6), axis.spacetime))

    def test_base_class1(self):
        iterator = Iterator(self.img)
        self.assertRaises(NotImplementedError, iterator.next)

    def test_base_class2(self):
        item = IteratorItem(self.img, [slice(0, 10), slice(0, 20), slice(0,1)])
        self.assertEquals(item.get().shape, (10, 20, 1))

        value = N.ones((10, 20, 1))
        item.set(value)

        N.testing.assert_equal(item.get(), value)

    def test_base_class_copy(self):
        iterator = Iterator(self.img)
        new_iterator = iterator.copy(self.img)
        self.assertEquals(iterator.mode, new_iterator.mode)
        self.assertEquals(iterator.img, new_iterator.img)

    def test_read_slices(self):
        for slice_ in self.img.slice_iterator():
            self.assertEquals(slice_.shape, (20, 30))

        for slice_ in self.img.slice_iterator(axis=1):
            self.assertEquals(slice_.shape, (10, 30))

        for slice_ in self.img.slice_iterator(axis=2):
            self.assertEquals(slice_.shape, (10, 20))

    def test_write_slices(self):
        for slice_ in self.img.slice_iterator(mode='w'):
            slice_.set(N.ones((20, 30)))

        for slice_ in self.img.slice_iterator(axis=1, mode='w'):
            slice_.set(N.ones((10, 30)))

        for slice_ in self.img.slice_iterator(axis=2, mode='w'):
            slice_.set(N.ones((10, 20)))

    def test_copy(self):
        iterator = self.img.slice_iterator()
        iterator2 = iterator.copy(self.img2)
        for slice_ in iterator2:
            N.testing.assert_equal(slice_, N.ones((20, 30)))


    def test_multi_slice(self):
        for slice_ in self.img.slice_iterator(axis=[0, 1]):
            self.assertEquals(slice_.shape, (30,))

        for slice_ in self.img.slice_iterator(axis=[2, 1]):
            self.assertEquals(slice_.shape, (10,))

    def test_multi_slice_write(self):
        for slice_ in self.img.slice_iterator(axis = [0, 1], mode='w'):
            slice_.set(N.zeros((30,)))

    def test_parcel(self):
        parcelmap = N.zeros(self.img3.shape)
        parcelmap[0,0,0] = 1
        parcelmap[1,1,1] = 1
        parcelmap[2,2,2] = 1
        parcelmap[1,2,1] = 2
        parcelmap[2,3,2] = 2
        parcelmap[0,1,0] = 2
        parcelseq = (0, 1, 2, 3)
        expected = [N.product(self.img3.shape) - 6, 3, 3, 0]
        iterator = ParcelIterator(self.img3, parcelmap, parcelseq)
        for i, slice_ in enumerate(iterator):
            self.assertEqual((expected[i],), slice_.shape)

        iterator = ParcelIterator(self.img3, parcelmap)
        for i, slice_ in enumerate(iterator):
            self.assertEqual((expected[i],), slice_.shape)


    def test_parcel_write(self):
        parcelmap = N.zeros(self.img3.shape)
        parcelmap[0,0,0] = 1
        parcelmap[1,1,1] = 1
        parcelmap[2,2,2] = 1
        parcelmap[1,2,1] = 2
        parcelmap[2,3,2] = 2
        parcelmap[0,1,0] = 2
        parcelseq = (0, 1, 2, 3)
        expected = [N.product(self.img3.shape) - 6, 3, 3, 0]
        iterator = ParcelIterator(self.img3, parcelmap, parcelseq, mode='w')

        for i, slice_ in enumerate(iterator):
            value = N.arange(expected[i])
            slice_.set(value)

        iterator = ParcelIterator(self.img3, parcelmap, parcelseq)
        for i, slice_ in enumerate(iterator):
            self.assertEqual((expected[i],), slice_.shape)
            N.testing.assert_equal(slice_, N.arange(expected[i]))


        iterator = ParcelIterator(self.img3, parcelmap, mode='w')
        for i, slice_ in enumerate(iterator):
            value = N.arange(expected[i])
            slice_.set(value)

        iterator = ParcelIterator(self.img3, parcelmap)
        for i, slice_ in enumerate(iterator):
            self.assertEqual((expected[i],), slice_.shape)
            N.testing.assert_equal(slice_, N.arange(expected[i]))



    def test_parcel_copy(self):
        parcelmap = N.zeros(self.img3.shape)
        parcelmap[0,0,0] = 1
        parcelmap[1,1,1] = 1
        parcelmap[2,2,2] = 1
        parcelmap[1,2,1] = 2
        parcelmap[2,3,2] = 2
        parcelmap[0,1,0] = 2
        parcelseq = (0, 1, 2, 3)
        expected = [N.product(self.img3.shape) - 6, 3, 3, 0]
        iterator = ParcelIterator(self.img3, parcelmap, parcelseq)
        tmp = Image(self.img3)

        new_iterator = iterator.copy(tmp)

        for i, slice_ in enumerate(new_iterator):
            self.assertEqual((expected[i],), slice_.shape)

        iterator = ParcelIterator(self.img3, parcelmap)
        for i, slice_ in enumerate(new_iterator):
            self.assertEqual((expected[i],), slice_.shape)




    def test_fmri_parcel(self):
        parcelmap = N.zeros(self.fmri_img.shape[1:])
        parcelmap[0,0,0] = 1
        parcelmap[1,1,1] = 1
        parcelmap[2,2,2] = 1
        parcelmap[1,2,1] = 2
        parcelmap[2,3,2] = 2
        parcelmap[0,1,0] = 2
        parcelseq = (0, 1, 2, 3)
        
        expected = [N.product(self.fmri_img.shape[1:]) - 6, 3, 3, 0]
        iterator = fMRIParcelIterator(self.fmri_img, parcelmap, parcelseq)

        for i, slice_ in enumerate(iterator):
            self.assertEqual((self.fmri_img.shape[0], expected[i],), slice_.shape)

        iterator = fMRIParcelIterator(self.fmri_img, parcelmap)
        for i, slice_ in enumerate(iterator):
            self.assertEqual((self.fmri_img.shape[0], expected[i],), slice_.shape)

    def test_fmri_parcel_write(self):
        parcelmap = N.zeros(self.fmri_img.shape[1:])
        parcelmap[0,0,0] = 1
        parcelmap[1,1,1] = 1
        parcelmap[2,2,2] = 1
        parcelmap[1,2,1] = 2
        parcelmap[2,3,2] = 2
        parcelmap[0,1,0] = 2
        parcelseq = (0, 1, 2, 3)
        expected = [N.product(self.fmri_img.shape[1:]) - 6, 3, 3, 0]
        iterator = fMRIParcelIterator(self.fmri_img, parcelmap, parcelseq, mode='w')

        for i, slice_ in enumerate(iterator):
            value = N.asarray([N.arange(expected[i]) for _ in range(self.fmri_img.shape[0])])
            slice_.set(value)

        iterator = fMRIParcelIterator(self.fmri_img, parcelmap, parcelseq)
        for i, slice_ in enumerate(iterator):
            self.assertEqual((self.fmri_img.shape[0], expected[i],), slice_.shape)
            N.testing.assert_equal(slice_, N.asarray([N.arange(expected[i]) for _ in range(self.fmri_img.shape[0])]))


        iterator = fMRIParcelIterator(self.fmri_img, parcelmap, mode='w')
        for i, slice_ in enumerate(iterator):
            value = N.asarray([N.arange(expected[i]) for _ in range(self.fmri_img.shape[0])])
            slice_.set(value)

        iterator = fMRIParcelIterator(self.fmri_img, parcelmap)
        for i, slice_ in enumerate(iterator):
            self.assertEqual((self.fmri_img.shape[0], expected[i],), slice_.shape)
            N.testing.assert_equal(slice_, N.asarray([N.arange(expected[i]) for _ in range(self.fmri_img.shape[0])]))


    def test_fmri_parcel_copy(self):
        parcelmap = N.zeros(self.fmri_img.shape[1:])
        parcelmap[0,0,0] = 1
        parcelmap[1,1,1] = 1
        parcelmap[2,2,2] = 1
        parcelmap[1,2,1] = 2
        parcelmap[2,3,2] = 2
        parcelmap[0,1,0] = 2
        parcelseq = (0, 1, 2, 3)
        expected = [N.product(self.fmri_img.shape[1:]) - 6, 3, 3, 0]
        iterator = fMRIParcelIterator(self.fmri_img, parcelmap, parcelseq)
        tmp = fMRIImage(self.fmri_img)

        new_iterator = iterator.copy(tmp)

        for i, slice_ in enumerate(new_iterator):
            self.assertEqual((self.fmri_img.shape[0], expected[i],), slice_.shape)

        iterator = fMRIParcelIterator(self.fmri_img, parcelmap)
        for i, slice_ in enumerate(new_iterator):
            self.assertEqual((self.fmri_img.shape[0], expected[i],), slice_.shape)


    def test_sliceparcel(self):
        parcelmap = N.asarray([[0,0,0,1,2],[0,0,1,1,2],[0,0,0,0,2]])
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
            self.assertEqual(self.img3.shape[2:], list(slice_.shape[1:]))

    def test_sliceparcel1(self):
        parcelmap = N.asarray([[0,0,0,1,2],[0,0,1,1,2],[0,0,0,0,2]])
        self.assertRaises(ValueError, SliceParcelIterator, self.img3, \
                          parcelmap, None)
        return

    def test_sliceparcel_copy(self):
        parcelmap = N.asarray([[0,0,0,1,2],[0,0,1,1,2],[0,0,0,0,2]])
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
            self.assertEqual(self.img4.shape[2:], list(slice_.shape[1:]))

    def test_sliceparcel_write(self):
        parcelmap = N.asarray([[0,0,0,1,2],[0,0,1,1,2],[0,0,0,0,2]])
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
            slice_.set(N.ones(shape))

        iterator = SliceParcelIterator(self.img3, parcelmap, parcelseq)
        for i, slice_ in enumerate(iterator):
            pm = parcelmap[i]
            ps = parcelseq[i]
            try:
                x = len([n for n in pm if n in ps])
            except TypeError:
                x = len([n for n in pm if n == ps])
            self.assertEqual(x, slice_.shape[0])
            N.testing.assert_equal(N.ones((x ,self.img3.shape[2])), slice_)



    def test_fmri_sliceparcel(self):
        parcelmap = N.asarray([[[0,0,0,1,2,2]]*5,
                               [[0,0,1,1,2,2]]*5,
                               [[0,0,0,0,2,2]]*5])
        parcelseq = ((1, 2), 0, 2)
        iterator = fMRISliceParcelIterator(self.fmri_img, parcelmap, parcelseq)
        for i, slice_ in enumerate(iterator):
            pm = parcelmap[i]
            ps = parcelseq[i]
            try:
                x = len([n for n in pm.flat if n in ps])
            except TypeError:
                x = len([n for n in pm.flat if n == ps])
            self.assertEqual(x, slice_.shape[1])
            self.assertEqual(self.fmri_img.shape[0], slice_.shape[0])

    def test_fmri_sliceparcel_write(self):
        parcelmap = N.asarray([[[0,0,0,1,2,2]]*5,
                               [[0,0,1,1,2,2]]*5,
                               [[0,0,0,0,2,2]]*5])
        parcelseq = ((1, 2), 0, 2)
        iterator = fMRISliceParcelIterator(self.fmri_img, parcelmap, parcelseq, mode='w')

        for i, slice_ in enumerate(iterator):
            pm = parcelmap[i]
            ps = parcelseq[i]
            try:
                x = len([n for n in pm.flat if n in ps])
            except TypeError:
                x = len([n for n in pm.flat if n == ps])
            value = [i*N.arange(x) for i in range(self.fmri_img.shape[0])]
            slice_.set(value)

        iterator = fMRISliceParcelIterator(self.fmri_img, parcelmap, parcelseq)
        for i, slice_ in enumerate(iterator):
            pm = parcelmap[i]
            ps = parcelseq[i]
            try:
                x = len([n for n in pm.flat if n in ps])
            except TypeError:
                x = len([n for n in pm.flat if n == ps])
            value = [i*N.arange(x) for i in range(self.fmri_img.shape[0])]
            self.assertEqual(x, slice_.shape[1])
            self.assertEqual(self.fmri_img.shape[0], slice_.shape[0])
            N.testing.assert_equal(slice_, value)


if __name__ == '__main__':
    NumpyTest.run()
