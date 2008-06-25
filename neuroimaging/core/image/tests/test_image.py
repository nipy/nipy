import os
import glob

import numpy as np
#from numpy.testing import NumpyTest, NumpyTestCase
from neuroimaging.externals.scipy.testing import *
from neuroimaging.utils.test_decorators import slow

from neuroimaging.core.image import image

from neuroimaging.core.api import Image, load_image, save_image, fromarray
from neuroimaging.core.api import parcels, data_generator, write_data

from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.core.reference.mapping import Affine
from neuroimaging.testing import anatfile, funcfile
from neuroimaging.data_io.api import Analyze


class TestImage(TestCase):

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
        filename = "bad_file.foo"
        self.assertRaises(RuntimeError, load_image, filename)

    def test_maxmin_values(self):
        y = np.asarray(self.img)
        self.assertEquals(y.shape, tuple(self.img.grid.shape))
        self.assertEquals(y.max(), 7902.0)
        self.assertEquals(y.min(), 1910.0)

    def test_slice_plane(self):
        x = self.img[3]
        self.assertEquals(x.shape, self.img.shape[1:])
        self.assertEquals(x.shape, x.grid.shape)

    def test_slice_block(self):
        x = self.img[3:5]
        self.assertEquals(x.shape, (2,) + tuple(self.img.grid.shape[1:]))
        self.assertEquals(x.shape, x.grid.shape)

    def test_slice_step(self):
        s = slice(0,20,2)
        x = self.img[s]
        self.assertEquals(x.shape, (10,) + tuple(self.img.grid.shape[1:]))
        self.assertEquals(x.shape, x.grid.shape)

    def test_slice_type(self):
        s = slice(0,self.img.grid.shape[0])
        x = self.img[s]
        self.assertEquals(x.shape, tuple((self.img.grid.shape)))
        self.assertEquals(x.shape, x.grid.shape)

    def test_slice_steps(self):
        zdim, ydim, xdim = self.img.shape
        slice_z = slice(0, zdim, 2)
        slice_y = slice(0, ydim, 2)
        slice_x = slice(0, xdim, 2)
        x = self.img[slice_z, slice_y, slice_x]
        newshape = ((zdim/2)+1, (ydim/2)+1, (xdim/2)+1)
        self.assertEquals(x.shape, newshape)

    def test_array(self):
        x = np.asarray(self.img)
        assert isinstance(x, np.ndarray)

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

    def test_tofile(self):
        save_image(self.img, "tmp.img")
        tmp_img = load_image("tmp.img")
        np.testing.assert_almost_equal(np.asarray(tmp_img)[:], np.asarray(self.img)[:])

        array_img = Image(np.zeros((10, 10, 10)), SamplingGrid.identity(['zspace', 'yspace', 'xspace'], (10,)*3))



def test_slicing_returns_image():
    data = np.ones((2,3,4))
    img = fromarray(data)
    assert isinstance(img, Image)
    assert img.ndim == 3
    # 2D slice
    img2D = img[:,:,0]
    assert isinstance(img2D, Image)
    assert img2D.ndim == 2
    # 1D slice
    img1D = img[:,0,0]
    assert isinstance(img1D, Image)
    assert img1D.ndim == 1


class ArrayLikeObj(object):
    """The data attr in Image is an array-like object.
    Test the array-like interface that we'll expect to support."""
    def __init__(self):
        self._data = np.ones((2,3,4))
    
    def get_ndim(self):
        return self._data.ndim
    ndim = property(get_ndim)
        
    def get_shape(self):
        return self._data.shape
    shape = property(get_shape)

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    def __array__(self):
        return self._data

def test_ArrayLikeObj():
    obj = ArrayLikeObj()
    # create simple grid
    xform = np.eye(4)
    affine = Affine(xform)
    grid = SamplingGrid.from_affine(affine, ['zspace', 'yspace', 'xspace'],
                                    (2,3,4))
    # create image form array-like object and grid
    img = image.Image(obj, grid)
    assert img.ndim == 3
    assert img.shape == (2,3,4)
    assert np.allclose(np.asarray(img), 1)
    assert np.allclose(img[:], 1)
    img[:] = 4
    assert np.allclose(img[:], 4)

# Should test common image sizes 2D, 3D, 4D
class TestFromArray(TestCase):
    def setUp(self):
        self.array2D_shape = (2,3)
        self.array3D_shape = (2,3,4)
        self.array4D_shape = (2,3,4,5)

    def test_defaults_2D(self):
        data = np.ones(self.array2D_shape)
        img = image.fromarray(data, names=['yspace', 'xspace'])
        assert isinstance(img._data, np.ndarray)
        assert img.ndim == 2
        assert img.shape == self.array2D_shape
        self.assertRaises(AttributeError, getattr, img, 'header')
        assert img.affine.shape == (3,3)
        assert img.affine.diagonal().all() == 1
        
    def test_defaults_3D(self):
        img = image.fromarray(np.ones(self.array3D_shape))
        assert isinstance(img._data, np.ndarray)
        assert img.ndim == 3
        assert img.shape == self.array3D_shape
        # ndarray's do not have a header
        self.assertRaises(AttributeError, getattr, img, 'header')
        assert img.affine.shape == (4,4)
        assert img.affine.diagonal().all() == 1

    def test_defaults_4D(self):
        data = np.ones(self.array4D_shape)
        names = ['time', 'zspace', 'yspace', 'xspace']
        img = image.fromarray(data, names=names)
        assert isinstance(img._data, np.ndarray)
        assert img.ndim == 4
        assert img.shape == self.array4D_shape
        self.assertRaises(AttributeError, getattr, img, 'header')
        assert img.affine.shape == (5,5)
        assert img.affine.diagonal().all() == 1


# Nose also has assert_ functions, but they're hidden in tools
#nose.tools.assert_raises(AttributeError, getattr, img, 'header')

if __name__ == '__main__':
    nose.runmodule()

