import os
import glob

import numpy as np
#from numpy.testing import NumpyTest, NumpyTestCase
from neuroimaging.externals.scipy.testing import *
from neuroimaging.utils.test_decorators import slow

from neuroimaging.core.api import Image, load_image, save_image, fromarray
from neuroimaging.core.api import parcels, data_generator, write_data

from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.testing import anatfile, funcfile
from neuroimaging.data_io.api import Analyze


class test_image(TestCase):

    def setUp(self):
        self.img = load_image(anatfile)
        self.func = load_image(funcfile)

    def tearDown(self):
        tmpfiles = glob.glob('tmp.*')
        for tmpfile in tmpfiles:
            os.remove(tmpfile)
            
    def test_init(self):
        new = Image(np.asarray(self.img[:]), self.img.grid)
        np.testing.assert_equal(np.asarray(self.img)[:], np.asarray(new)[:])

        new = Image(self.img._data, self.img.grid)
        np.testing.assert_equal(np.asarray(self.img)[:], np.asarray(new)[:])

        self.assertRaises(ValueError, Image, None, None)

    def test_badfile(self):
        filename = "bad.file"
        self.assertRaises(NotImplementedError, Image, filename)

    def test_analyze(self):
        """
        TODO: 
        """
        self.fail("this maximum is wrong for the new anatfile -- value should be fixed")
        y = np.asarray(self.img)
        self.assertEquals(y.shape, tuple(self.img.grid.shape))
        y = y.flatten()
        self.assertEquals(np.maximum.reduce(y), 437336.375)
        self.assertEquals(np.minimum.reduce(y), 0.)

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
        x = self.func[[slice_1,slice(None,None), slice_2]]
        self.assertEquals(x.shape, (10,2,4,20))

    def test_array(self):
        x = np.asarray(self.img)
        
    def test_file(self):
        self.fail('this is a problem with reading/writing so-called "mat" files -- all the functions for these have been moved from core.reference.mapping to data_io.formats.analyze -- and they need to be fixed because they do not work. the names of the functions are: matfromstr, matfromfile, matfrombin, matfromxfm, mattofile')
        save_image(self.img, 'tmp.hdr', format=Analyze)
        
        # Analyze is broken
        img2 = load_image('tmp.hdr', format=Analyze)

        # This fails: saying array is not writeable
        
        try:
            img2[0,0,0] = 370000
        except RuntimeError:
            raise RuntimeError, 'this is a problem with the memmap of img2 -- seems not to be writeable'
        img3 = Image(img2.asfile(), use_memmap=True)
        img2[1,1,1] = 100000

        scale = img2._source.header['scale_factor']
        self.assertTrue(abs(370000 - img3[0,0,0]) < scale)
        self.assertTrue(abs(100000 - img3[1,1,1]) < scale)
        
    # TODO: This is a test for the SamplingGrid, not Image?
    def test_nondiag(self):
        """
        This test doesn't work, presumably something to do with the matfile.
        """
        self.fail('this is a problem with reading/writing so-called "mat" files -- all the functions for these have been moved from core.reference.mapping to data_io.formats.analyze -- and they need to be fixed because they do not work. the names of the functions are: matfromstr, matfromfile, matfrombin, matfromxfm, mattofile')
        self.img.grid.mapping.transform[0,1] = 3.0
        save_image(self.img, 'tmp.hdr', usematfile=True)
        try:
            x = load_image('tmp.hdr', usematfile=True, format=Analyze)
        except NotImplementedError:
            raise NotImplementedError, 'this is a problem with reading so-called "mat" files'
        np.testing.assert_almost_equal(x.grid.mapping.transform, self.img.grid.mapping.transform)

    def test_clobber(self):
        self.fail('this is a problem with reading/writing so-called "mat" files -- all the functions for these have been moved from core.reference.mapping to data_io.formats.analyze -- and they need to be fixed because they do not work. the names of the functions are: matfromstr, matfromfile, matfrombin, matfromxfm, mattofile')

        x = save_image(self.img, 'tmp.hdr', format=Analyze, clobber=True)
        a = load_image('tmp.hdr', format=Analyze)

        A = np.asarray(a)
        I = np.asarray(self.img)
        z = np.add.reduce(((A-I)**2).flat)
        self.assertEquals(z, 0.)

        t = a.grid.mapping.transform
        b = self.img.grid.mapping.transform
        np.testing.assert_almost_equal(b, t)


    # TODO: Should be in test_generator.py
    def test_iter(self):
        """
        """
        g = data_generator(self.func, range(self.func.shape[0]))
        for i, d in g:
            self.assertEquals(d.shape, (2,20,20))

    # TODO: Should be in test_generator.py
    def test_iter3(self):
        self.fail("this assertion is not raised -- python seems to give the default slice through data, i.e. iterating through the first indices")
        self.assertRaises(NotImplementedError, iter, self.img)

    # TODO: Should be in test_generator.py
    def test_iter4(self):
        tmp = Image(np.zeros(self.img.shape), self.img.grid)
        write_data(tmp, data_generator(self.img, range(self.img.shape[0])))
        np.testing.assert_almost_equal(np.asarray(tmp), np.asarray(self.img))

    
    # TODO: Should be in test_generator.py
    def test_iter5(self):
        """
        This next test seems like it could be deprecated with simplified iterator options
        """
        
        tmp = Image(np.zeros(self.img.shape), self.img.grid)
        g = data_generator(self.img)
        write_data(tmp, g)
        np.testing.assert_almost_equal(np.asarray(tmp), np.asarray(self.img))


    @slow
    def test_set_next(self):
        write_img = save_image("test_write.hdr", repository, grid=self.img.grid, format=Analyze,
                               clobber=True)
        I = write_img.slice_iterator(mode='w')
        x = 0
        for slice_ in I:
            slice_.set(np.ones((109, 91)))
            x += 1
        self.assertEquals(x, 91)

    # TODO: Should be in test_generator.py
    def test_parcels1(self):
        rho = load_image(anatfile)
        parcelmap = (np.asarray(rho)[:] * 100).astype(np.int32)
        
        test = np.zeros(parcelmap.shape)
        v = 0
        for i, d in data_generator(test, parcels(parcelmap)):
            v += d.shape[0]

        self.assertEquals(v, np.product(test.shape))

    # TODO: Should be in test_generator.py
    def test_parcels3(self):
        rho = load_image(anatfile)[0]
        parcelmap = (np.asarray(rho)[:] * 100).astype(np.int32)
        labels = np.unique(parcelmap)
        test = np.zeros(rho.shape)

        v = 0
        for i, d in data_generator(test, parcels(parcelmap, labels=labels)):
            v += d.shape[0]

        self.assertEquals(v, np.product(test.shape))

    # TODO: Should be in test_generator.py
    @slow
    def test_parcels4(self):
        """TODO: fix this
        """
        
        rho = load_image("rho.hdr", repository, format=Analyze)
        parcelmap = (rho[:] * 100).astype(np.int32)
        parcelseq = parcelmap
        
        test = Image(np.zeros(parcelmap.shape), grid=rho.grid)

        v = 0
        for i in test.slice_parcel_iterator(parcelmap, parcelseq):
            v += 1
        self.assertEquals(v, test.grid.shape[0])

    def test_badfile(self):
        # We shouldn't be able to find a reader for this file!
        filename = "test_image.py"
        self.assertRaises(NotImplementedError, load_image, filename)
    def test_tofile(self):
        save_image(self.img, "tmp.img")
        tmp_img = load_image("tmp.img")
        np.testing.assert_almost_equal(np.asarray(tmp_img)[:], np.asarray(self.img)[:])

        array_img = Image(np.zeros((10, 10, 10)), SamplingGrid.identity(['zspace', 'yspace', 'xspace'], (10,)*3))



def test_slicing():
    data = np.ones((2,3,4))
    img = fromarray(data)
    assert isinstance(img, Image)
    assert img.ndim == 3
    # 2D slice
    img2D = img[:,:,0]
    assert isinstance(img, Image)
    assert img.ndim == 2
    # 1D slice
    img1D = img[:,0,0]
    assert isinstance(img, Image)
    assert img.ndim == 1


class ImageInterface(object):
    def __init__(self):
        self.data = np.ones((2,3,4))
    
    def get_ndim(self):
        return self.data.ndim
    ndim = property(get_ndim)
        
    def get_shape(self):
        return self.data.shape
    shape = property(get_shape)

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __array__(self):
        return self.data

def test_ImageInterface():
    obj = ImageInterface()
    #assertRaises obj.ndim = 20
    img = image.fromarray(obj)
    assert img.ndim == 3
    assert img.shape == (2,3,4)
    assert np.allclose(np.asarray(img), 1)
    assert np.allclose(img[:], 1)
    img[:] = 4
    assert np.allclose(img[:], 4)


if __name__ == '__main__':
    nose.runmodule()

