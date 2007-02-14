import os
import glob

import numpy as N
from numpy.testing import NumpyTest, NumpyTestCase

from neuroimaging.utils.test_decorators import slow

from neuroimaging.core.api import Image, ImageSequenceIterator
from neuroimaging.utils.tests.data import repository
from neuroimaging.data_io.formats.analyze import Analyze

from neuroimaging.core.image.iterators import ParcelIterator, \
     SliceParcelIterator, SliceIterator


class test_image(NumpyTestCase):

    def setUp(self):
        self.img = Image("avg152T1.img", repository, format=Analyze)

    def tearDown(self):
        tmpfiles = glob.glob('tmp.*')
        for tmpfile in tmpfiles:
            os.remove(tmpfile)
            
    def test_init(self):
        new = Image(self.img)
        N.testing.assert_equal(self.img[:], new[:])

        new = Image(self.img._source)
        N.testing.assert_equal(self.img[:], new[:])


        self.assertRaises(ValueError, Image, None)

    def test_badfile(self):
        filename = "bad.file"
        self.assertRaises(NotImplementedError, Image, filename)

    def test_analyze(self):
        y = self.img.readall()
        self.assertEquals(y.shape, tuple(self.img.grid.shape))
        y = y.flatten()
        self.assertEquals(N.maximum.reduce(y), 437336.375)
        self.assertEquals(N.minimum.reduce(y), 0.)

    def test_slice1(self):
        x = self.img[3]
        self.assertEquals(x.shape, tuple(self.img.grid.shape[1:]))
        
    def test_slice2(self):
        x = self.img[3:5]
        self.assertEquals(x.shape, (2,) + tuple(self.img.grid.shape[1:]))

    def test_slice3(self):
        s = slice(0,20,2)
        x = self.img[s]
        self.assertEquals(x.shape, (10,) + tuple(self.img.grid.shape[1:]))

    def test_slice4(self):
        s = slice(0,self.img.grid.shape[0])
        x = self.img[s]
        self.assertEquals(x.shape, tuple((self.img.grid.shape)))

    def test_slice5(self):
        slice_1 = slice(0,20,2)
        slice_2 = slice(0,50,5)
        x = self.img[[slice_1,slice_2]]
        self.assertEquals(x.shape, (10,10,self.img.grid.shape[2]))

    def test_array(self):
        x = self.img.toarray()
        
    def test_file(self):
        img2 = self.img.tofile('tmp.hdr')

        img2[0,0,0] = 370000
        img3 = Image(img2.asfile())
        img2[1,1,1] = 100000
        
        scale = img2._source.header['scale_factor']
        self.assertTrue(abs(370000 - img3[0,0,0]) < scale)
        self.assertTrue(abs(100000 - img3[1,1,1]) < scale)
        

    def test_nondiag(self):
        self.img.grid.mapping.transform[0,1] = 3.0
        x = self.img.tofile('tmp.hdr', usematfile=True)
        N.testing.assert_almost_equal(x.grid.mapping.transform, self.img.grid.mapping.transform)

    def test_clobber(self):
        x = self.img.tofile('tmp.hdr', format=Analyze, clobber=True)
        a = Image('tmp.hdr', format=Analyze)
        A = a.readall()
        I = self.img.readall()
        z = N.add.reduce(((A-I)**2).flat)
        self.assertEquals(z, 0.)

        t = a.grid.mapping.transform
        b = self.img.grid.mapping.transform
        N.testing.assert_almost_equal(b, t)


    def test_iter(self):
        for i in self.img.slice_iterator():
            self.assertEquals(i.shape, (109,91))

    def test_iter2(self):
        for i in self.img.iterate(SliceIterator(None, axis=0)):
            self.assertEquals(i.shape, (109,91))

    def test_iter3(self):
        self.assertRaises(NotImplementedError, iter, self.img)

    def test_iter4(self):
        tmp = Image(N.zeros(self.img.shape), mode='w')
        iterator = SliceIterator(self.img)
        tmp.from_slice_iterator(iterator)
        N.testing.assert_almost_equal(tmp[:], self.img[:])

    def test_iter5(self):
        tmp = Image(N.zeros(self.img.shape), mode='w')
        iterator1 = SliceIterator(self.img)
        iterator2 = SliceIterator(None)
        tmp.from_iterator(iterator1, iterator2)
        N.testing.assert_almost_equal(tmp[:], self.img[:])


    @slow
    def test_set_next(self):
        write_img = Image("test_write.hdr", repository, grid=self.img.grid, format=Analyze,
                          mode='w', clobber=True)
        I = write_img.slice_iterator('w')
        x = 0
        for slice in I:
            slice.set(N.ones((109, 91)))
            x += 1
        self.assertEquals(x, 91)

    def test_parcels1(self):
        rho = Image("rho.hdr", repository, format=Analyze)
        parcelmap = (rho.readall() * 100).astype(N.int32)
        test = Image(N.zeros(parcelmap.shape), grid=rho.grid)
        it = ParcelIterator(test, parcelmap)
        v = 0
        for i in it:
            v += i.shape[0]

        self.assertEquals(v, N.product(test.grid.shape))

       

    def test_parcels3(self):
        rho = Image("rho.hdr", repository, format=Analyze)
        parcelmap = (rho.readall() * 100).astype(N.int32)
        shape = parcelmap.shape
        parcelmap.shape = parcelmap.size
        parcelseq = N.unique(parcelmap)

        test = Image(N.zeros(shape), grid=rho.grid)

        it = ParcelIterator(test, parcelmap, parcelseq)
        v = 0
        for i in it:
            v += i.shape[0]

        self.assertEquals(v, N.product(test.grid.shape))

    @slow
    def test_parcels4(self):
        rho = Image("rho.hdr", repository, format=Analyze)
        parcelmap = (rho.readall() * 100).astype(N.int32)
        parcelseq = parcelmap
        
        test = Image(N.zeros(parcelmap.shape), grid=rho.grid)
        it = SliceParcelIterator(test, parcelmap, parcelseq)
        v = 0
        for i in it:
            v += 1
        self.assertEquals(v, test.grid.shape[0])

    def test_readall(self):
        a = self.img.readall(clean=False)
        b = self.img.readall(clean=True)
        N.testing.assert_equal(a, b)

    def test_badfile(self):
        # We shouldn't be able to find a reader for this file!
        filename = "test_image.py"
        self.assertRaises(IOError, Image, filename, repository, format=Analyze)

    def test_asfile(self):
        tmp_img = Image(self.img.asfile())
        N.testing.assert_almost_equal(tmp_img[:], self.img[:])

        array_img = Image(N.zeros((10, 10, 10)))
        tmp_img = Image(array_img.asfile())
        N.testing.assert_almost_equal(tmp_img[:], array_img[:])
        


class test_ImageSequenceIterator(NumpyTestCase):

    @slow
    def test_image_sequence_iterator(self):
        base_img = Image("avg152T1.img", repository, format=Analyze)
        imgs = [Image(base_img) for _ in range(10)]
        it = ImageSequenceIterator(imgs)
        for x in it:
            self.assertTrue(isinstance(x, N.ndarray))


if __name__ == '__main__':
    NumpyTest.run()
