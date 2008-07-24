#TODO the iterators are deprecated
import numpy as N
from numpy.testing import NumpyTest, NumpyTestCase

import neuroimaging.core.reference.axis as axis
from neuroimaging.core.api import Image
import neuroimaging.core.reference.grid as grid
from neuroimaging.modalities.fmri.api import FmriImage


class test_Iterators(NumpyTestCase):

    def setUp(self):
        im = Image(N.zeros((3,4,5,6)), grid = grid.SamplingGrid.identity((3,4,5,6), axis.spacetime))
        self.img = fromimage(im)

    def test_fmri_parcel(self):
        parcelmap = N.zeros(self.img.shape[1:])
        parcelmap[0,0,0] = 1
        parcelmap[1,1,1] = 1
        parcelmap[2,2,2] = 1
        parcelmap[1,2,1] = 2
        parcelmap[2,3,2] = 2
        parcelmap[0,1,0] = 2
        parcelseq = (0, 1, 2, 3)
        
        expected = [N.product(self.img.shape[1:]) - 6, 3, 3, 0]

        iterator = parcel_iterator(self.img, parcelmap, parcelseq)
        for i, slice_ in enumerate(iterator):
            self.assertEqual((self.img.shape[0], expected[i],), slice_.shape)

        iterator = parcel_iterator(self.img, parcelmap)
        for i, slice_ in enumerate(iterator):
            self.assertEqual((self.img.shape[0], expected[i],), slice_.shape)

    def test_fmri_parcel_write(self):
        parcelmap = N.zeros(self.img.shape[1:])
        parcelmap[0,0,0] = 1
        parcelmap[1,1,1] = 1
        parcelmap[2,2,2] = 1
        parcelmap[1,2,1] = 2
        parcelmap[2,3,2] = 2
        parcelmap[0,1,0] = 2
        parcelseq = (0, 1, 2, 3)
        expected = [N.product(self.img.shape[1:]) - 6, 3, 3, 0]

        iterator = parcel_iterator(self.img, parcelmap, parcelseq, mode='w')
        for i, slice_ in enumerate(iterator):
            value = N.asarray([N.arange(expected[i]) for _ in range(self.img.shape[0])])
            slice_.set(value)

        iterator = parcel_iterator(self.img, parcelmap, parcelseq)
        for i, slice_ in enumerate(iterator):
            self.assertEqual((self.img.shape[0], expected[i],), slice_.shape)
            N.testing.assert_equal(slice_, N.asarray([N.arange(expected[i]) for _ in range(self.img.shape[0])]))


        iterator = parcel_iterator(self.img, parcelmap, mode='w')
        for i, slice_ in enumerate(iterator):
            value = N.asarray([N.arange(expected[i]) for _ in range(self.img.shape[0])])
            slice_.set(value)

        iterator = parcel_iterator(self.img, parcelmap)
        for i, slice_ in enumerate(iterator):
            self.assertEqual((self.img.shape[0], expected[i],), slice_.shape)
            N.testing.assert_equal(slice_, N.asarray([N.arange(expected[i]) for _ in range(self.img.shape[0])]))


    def test_fmri_parcel_copy(self):
        parcelmap = N.zeros(self.img.shape[1:])
        parcelmap[0,0,0] = 1
        parcelmap[1,1,1] = 1
        parcelmap[2,2,2] = 1
        parcelmap[1,2,1] = 2
        parcelmap[2,3,2] = 2
        parcelmap[0,1,0] = 2
        parcelseq = (0, 1, 2, 3)
        expected = [N.product(self.img.shape[1:]) - 6, 3, 3, 0]
        iterator = parcel_iterator(self.img, parcelmap, parcelseq)
        tmp = FmriImage(self.img[:] * 1., self.img.grid)

        new_iterator = iterator.copy(tmp)

        for i, slice_ in enumerate(new_iterator):
            self.assertEqual((self.img.shape[0], expected[i],), slice_.shape)

        iterator = parcel_iterator(self.img, parcelmap)
        for i, slice_ in enumerate(new_iterator):
            self.assertEqual((self.img.shape[0], expected[i],), slice_.shape)

    def test_fmri_sliceparcel(self):
        parcelmap = N.asarray([[[0,0,0,1,2,2]]*5,
                               [[0,0,1,1,2,2]]*5,
                               [[0,0,0,0,2,2]]*5])
        parcelseq = ((1, 2), 0, 2)
        iterator = slice_parcel_iterator(self.img, parcelmap, parcelseq)
        for i, slice_ in enumerate(iterator):
            pm = parcelmap[i]
            ps = parcelseq[i]
            try:
                x = len([n for n in pm.flat if n in ps])
            except TypeError:
                x = len([n for n in pm.flat if n == ps])
            self.assertEqual(x, slice_.shape[1])
            self.assertEqual(self.img.shape[0], slice_.shape[0])

    def test_fmri_sliceparcel_write(self):
        parcelmap = N.asarray([[[0,0,0,1,2,2]]*5,
                               [[0,0,1,1,2,2]]*5,
                               [[0,0,0,0,2,2]]*5])
        parcelseq = ((1, 2), 0, 2)
        iterator = slice_parcel_iterator(self.img, parcelmap, parcelseq, mode='w')

        for i, slice_ in enumerate(iterator):
            pm = parcelmap[i]
            ps = parcelseq[i]
            try:
                x = len([n for n in pm.flat if n in ps])
            except TypeError:
                x = len([n for n in pm.flat if n == ps])
            value = [i*N.arange(x) for i in range(self.img.shape[0])]
            slice_.set(value)

        iterator = slice_parcel_iterator(self.img, parcelmap, parcelseq)
        for i, slice_ in enumerate(iterator):
            pm = parcelmap[i]
            ps = parcelseq[i]
            try:
                x = len([n for n in pm.flat if n in ps])
            except TypeError:
                x = len([n for n in pm.flat if n == ps])
            value = [i*N.arange(x) for i in range(self.img.shape[0])]
            self.assertEqual(x, slice_.shape[1])
            self.assertEqual(self.img.shape[0], slice_.shape[0])
            N.testing.assert_equal(slice_, value)


from neuroimaging.utils.testutils import make_doctest_suite
test_suite = make_doctest_suite('neuroimaging.modalities.fmri.iterators')

if __name__ == '__main__':
    NumpyTest().run()
