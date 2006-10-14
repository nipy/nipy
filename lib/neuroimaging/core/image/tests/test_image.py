import unittest
import os
import glob

import numpy as N

from neuroimaging.core.image.image import Image, ImageSequenceIterator
#from neuroimaging.algorithms.onesample import ImageOneSample
from neuroimaging.utils.tests.data import repository
from neuroimaging.data_io.formats.analyze import Analyze



class ImageTest(unittest.TestCase):

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
        x = self.img.tofile('tmp.hdr')

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
        I = iter(self.img)
        for i in I:
            self.assertEquals(i.shape, (109,91))

    def test_parcels1(self):
        rho = Image("rho.hdr", repository, format=Analyze)
        parcelmap = (rho.readall() * 100).astype(N.int32)
        test = Image(N.zeros(parcelmap.shape), grid=rho.grid)
        test.grid.set_iter_param("itertype", 'parcel')
        test.grid.set_iter_param("parcelmap", parcelmap)
        v = 0
        for t in test:
            v += t.shape[0]
        self.assertEquals(v, N.product(test.grid.shape))

    def test_parcels2(self):
        rho = Image("rho.hdr", repository, format=Analyze)
        parcelmap = (rho.readall() * 100).astype(N.int32)
        test = Image(N.zeros(parcelmap.shape), grid=rho.grid)

        test.grid.set_iter_param("itertype", 'parcel')
        test.grid.set_iter_param("parcelmap", parcelmap)
        parcelmap.shape = N.product(parcelmap.shape)
       
        v = 0
        iter(test)
        while True:
            try:
                test.next(data=v)
                v += 1
            except StopIteration:
                break

    def test_parcels3(self):
        rho = Image("rho.hdr", repository, format=Analyze)
        parcelmap = (rho.readall() * 100).astype(N.int32)
        shape = parcelmap.shape
        parcelmap.shape = N.product(parcelmap.shape)
        parcelseq = N.unique(parcelmap)

        test = Image(N.zeros(shape), grid=rho.grid)
        test.grid.set_iter_param("itertype", 'parcel')
        test.grid.set_iter_param("parcelmap", parcelmap)
        test.grid.set_iter_param("parcelseq", parcelseq)
        v = 0

        for t in test:
            v += t.shape[0]
        self.assertEquals(v, N.product(test.grid.shape))


    def test_readall(self):
        a = self.img.readall(clean=False)
        b = self.img.readall(clean=True)
        N.testing.assert_equal(a, b)

class ImageSequenceIteratorTest(unittest.TestCase):

    def test_image_sequence_iterator(self):
        base_img = Image("avg152T1.img", repository, format=Analyze)
        imgs = [Image(base_img) for _ in range(10)]
        it = ImageSequenceIterator(imgs)
        for x in it:
            self.assertEquals(type(x), type(N.array([])))

        it = ImageSequenceIterator(imgs, grid=imgs[-1].grid)
        for x in it:
            self.assertEquals(type(x), type(N.array([])))


if __name__ == '__main__':
    unittest.main()
