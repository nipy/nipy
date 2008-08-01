from tempfile import NamedTemporaryFile

import numpy as np
from neuroimaging.testing import *


from neuroimaging.utils.tests.data import repository

from neuroimaging.core.image import image

from neuroimaging.core.api import Image, load_image, save_image, fromarray
from neuroimaging.core.api import parcels, data_generator, write_data

from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.core.reference.mapping import Affine

class TestImage(TestCase):

    def setUp(self):
        self.img = load_image(str(repository._fullpath('avg152T1.nii.gz')))
        self.tmpfile = NamedTemporaryFile(suffix='.nii.gz')
        
    def test_init(self):
        new = Image(np.asarray(self.img), self.img.grid)
        assert_equal(np.asarray(self.img)[:], np.asarray(new)[:])

        self.assertRaises(ValueError, Image, None, None)

    def test_badfile(self):
        filename = "bad_file.foo"
        self.assertRaises(RuntimeError, load_image, filename)

    def test_maxmin_values(self):
        y = np.asarray(self.img)
        self.assertEquals(y.shape, tuple(self.img.grid.shape))
        np.allclose(y.max(), 437336.36, rtol=1.0e-8)
        self.assertEquals(y.min(), 0.0)

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
        self.assertEquals(x.shape, self.img.shape)
        self.assertEquals(x.ndim, self.img.ndim)
        
    def test_file_roundtrip(self):
        save_image(self.img, self.tmpfile.name)
        img2 = load_image(self.tmpfile.name)
        data = np.asarray(self.img)
        data2 = np.asarray(img2)
        # verify data
        assert_almost_equal(data2, data)
        assert_almost_equal(data2.mean(), data.mean())
        assert_almost_equal(data2.min(), data.min())
        assert_almost_equal(data2.max(), data.max())
        # verify shape and ndims
        assert_equal(img2.shape, self.img.shape)
        assert_equal(img2.ndim, self.img.ndim)
        # verify affine
        assert_equal(img2.affine, self.img.affine)

        
    def test_nondiag(self):
        self.img.grid.mapping.transform[0,1] = 3.0
        save_image(self.img, self.tmpfile.name)
        img2 = load_image(self.tmpfile.name)
        assert_almost_equal(img2.grid.mapping.transform,
                                       self.img.grid.mapping.transform)

    def test_generator(self):
        gen = data_generator(self.img)
        for ind, data in gen:
            self.assertEquals(data.shape, (109,91))

    def test_iter(self):
        imgiter = iter(self.img)
        for data in imgiter:
            self.assertEquals(data.shape, (109,91))

    def test_iter4(self):
        tmp = Image(np.zeros(self.img.shape), self.img.grid)
        write_data(tmp, data_generator(self.img, range(self.img.shape[0])))
        assert_almost_equal(np.asarray(tmp), np.asarray(self.img))

    def test_iter5(self):
        #This next test seems like it could be deprecated with
        #simplified iterator options
        
        tmp = Image(np.zeros(self.img.shape), self.img.grid)
        g = data_generator(self.img)
        write_data(tmp, g)
        assert_almost_equal(np.asarray(tmp), np.asarray(self.img))

    def test_parcels1(self):
        rho = self.img
        parcelmap = (np.asarray(rho)[:] * 100).astype(np.int32)
        
        test = np.zeros(parcelmap.shape)
        v = 0
        for i, d in data_generator(test, parcels(parcelmap)):
            v += d.shape[0]

        self.assertEquals(v, np.product(test.shape))

    def test_parcels3(self):
        rho = self.img[0]
        parcelmap = (np.asarray(rho)[:] * 100).astype(np.int32)
        labels = np.unique(parcelmap)
        test = np.zeros(rho.shape)

        v = 0
        for i, d in data_generator(test, parcels(parcelmap, labels=labels)):
            v += d.shape[0]

        self.assertEquals(v, np.product(test.shape))

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




