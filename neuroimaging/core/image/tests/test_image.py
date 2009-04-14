import os
import warnings
from tempfile import mkstemp

import numpy as np
import numpy.testing as nptest

from nipy.testing import *

from nipy.core.image import image
from nipy.utils.tests.data import repository
from nipy.io.api import load_image, save_image
from nipy.core.api import Image, fromarray, merge_images
from nipy.core.api import parcels, data_generator, write_data

from nipy.core.reference.coordinate_map import CoordinateMap, Affine

def setup():
    # Suppress warnings during tests to reduce noise
    warnings.simplefilter("ignore")

def teardown():
    # Clear list of warning filters
    warnings.resetwarnings()


class TestImage(TestCase):

    def setUp(self):
        self.img = load_image(str(repository._fullpath('avg152T1.nii.gz')))
        fd, self.filename = mkstemp(suffix='.nii.gz')
        self.tmpfile = open(self.filename)
        
    def tearDown(self):
        self.tmpfile.close()
        os.unlink(self.filename)


    def test_init(self):
        new = Image(np.asarray(self.img), self.img.coordmap)
        yield assert_array_almost_equal, np.asarray(self.img), np.asarray(new)
        yield assert_raises, ValueError(Image, None, None)

    # This test causes output in the nifticlibs that we cannot suppress.
    # Comment out so there's less noise in test output
    #def test_badfile(self):
    #    filename = "bad_file.foo"
    #    assert_raises(RuntimeError, load_image, filename)

    def test_maxmin_values(self):
        y = np.asarray(self.img)
        assert_equal(y.shape, tuple(self.img.shape))
        assert np.allclose(y.max(), 437336.36, rtol=1.0e-8)
        assert_equal(y.min(), 0.0)

    def test_slice_plane(self):
        x = self.img[3]
        assert_equal(x.shape, self.img.shape[1:])

    def test_slice_block(self):
        x = self.img[3:5]
        assert_equal(x.shape, (2,) + tuple(self.img.shape[1:]))

    def test_slice_step(self):
        s = slice(0,20,2)
        x = self.img[s]
        assert_equal(x.shape, (10,) + tuple(self.img.shape[1:]))

    def test_slice_type(self):
        s = slice(0,self.img.shape[0])
        x = self.img[s]
        assert_equal(x.shape, self.img.shape)

    def test_slice_steps(self):
        zdim, ydim, xdim = self.img.shape
        slice_z = slice(0, zdim, 2)
        slice_y = slice(0, ydim, 2)
        slice_x = slice(0, xdim, 2)
        x = self.img[slice_z, slice_y, slice_x]
        newshape = ((zdim/2)+1, (ydim/2)+1, (xdim/2)+1)
        assert_equal(x.shape, newshape)

    def test_array(self):
        x = np.asarray(self.img)
        assert isinstance(x, np.ndarray)
        assert_equal(x.shape, self.img.shape)
        assert_equal(x.ndim, self.img.ndim)
        
    def test_nondiag(self):
        self.img.affine[0,1] = 3.0
        save_image(self.img, self.tmpfile.name)
        img2 = load_image(self.tmpfile.name)
        assert_true(np.allclose(img2.affine, self.img.affine))

    def test_generator(self):
        gen = data_generator(self.img)
        for ind, data in gen:
            assert_equal(data.shape, (109,91))

    def test_iter(self):
        imgiter = iter(self.img)
        for data in imgiter:
            assert_equal(data.shape, (109,91))

    def test_iter4(self):
        tmp = Image(np.zeros(self.img.shape), self.img.coordmap)
        write_data(tmp, data_generator(self.img, range(self.img.shape[0])))
        assert_true(np.allclose(np.asarray(tmp), np.asarray(self.img)))

    def test_iter5(self):
        
        tmp = Image(np.zeros(self.img.shape), self.img.coordmap)
        g = data_generator(self.img)
        write_data(tmp, g)
        assert_true(np.allclose(np.asarray(tmp), np.asarray(self.img)))

    def test_parcels1(self):
        rho = self.img
        parcelmap = (np.asarray(rho)[:] * 100).astype(np.int32)
        
        test = np.zeros(parcelmap.shape)
        v = 0
        for i, d in data_generator(test, parcels(parcelmap)):
            v += d.shape[0]

        assert_equal(v, np.product(test.shape))

    def test_parcels3(self):
        rho = self.img[0]
        parcelmap = (np.asarray(rho)[:] * 100).astype(np.int32)
        labels = np.unique(parcelmap)
        test = np.zeros(rho.shape)

        v = 0
        for i, d in data_generator(test, parcels(parcelmap, labels=labels)):
            v += d.shape[0]

        assert_equal(v, np.product(test.shape))

    def uint8_to_dtype(self, dtype, name):
        dtype = dtype
        shape = (2,3,4)
        dmax = np.iinfo(np.uint8).max
        data = np.random.randint(0, dmax, size=shape)
        data[0,0,0] = 0
        data[1,0,0] = dmax
        data = data.astype(np.uint8) # randint returns np.int32
        img = fromarray(data, 'kji', 'zxy')
        newimg = save_image(img, name, dtype=dtype)
        newdata = np.asarray(newimg)
        return newdata, data
        
    def test_scaling_uint8_to_uint8(self):
        dtype = np.uint8
        newdata, data = self.uint8_to_dtype(dtype, self.tmpfile.name)
        assert_true(np.allclose(newdata, data))

    def test_scaling_uint8_to_uint16(self):
        dtype = np.uint16
        newdata, data = self.uint8_to_dtype(dtype, self.tmpfile.name)
        assert_true(np.allclose(newdata, data))

    def test_scaling_uint8_to_float32(self):
        dtype = np.float32
        newdata, data = self.uint8_to_dtype(dtype, self.tmpfile.name)
        assert_true(np.allclose(newdata, data))

    def test_scaling_uint8_to_int32(self):
        dtype = np.int32
        newdata, data = self.uint8_to_dtype(dtype, self.tmpfile.name)
        assert_true(np.allclose(newdata, data))
    
    def float32_to_dtype(self, dtype):
        # Utility function for the float32_to_<dtype> functions
        # below. There is a lot of shared functionality, split up so
        # the function names are unique so it's clear which dtypes are
        # involved in a failure.
        dtype = dtype
        shape = (2,3,4)
        # set some value value for scaling our data
        scale = np.iinfo(np.uint16).max * 2.0
        data = np.random.normal(size=(2,3,4), scale=scale)
        data[0,0,0] = np.finfo(np.float32).max
        data[1,0,0] = np.finfo(np.float32).min
        # random.normal will return data as native machine type
        data = data.astype(np.float32)
        img = fromarray(data, 'kji', 'zyx')
        newimg = save_image(img, self.tmpfile.name, dtype=dtype)
        newdata = np.asarray(newimg)
        return newdata, data
        
    def test_scaling_float32_to_uint8(self):
        dtype = np.uint8
        newdata, data = self.float32_to_dtype(dtype)
        assert_true(np.allclose(newdata, data))

    def test_scaling_float32_to_uint16(self):
        dtype = np.uint16
        newdata, data = self.float32_to_dtype(dtype)
        assert_true(np.allclose(newdata, data))
        
    def test_scaling_float32_to_int16(self):
        dtype = np.int16
        newdata, data = self.float32_to_dtype(dtype)
        assert_true(np.allclose(newdata, data))

    def test_scaling_float32_to_float32(self):
        dtype = np.float32
        newdata, data = self.float32_to_dtype(dtype)
        assert_true(np.allclose(newdata, data))

def test_merge_images():
    """
    Check that merge_images works, and that the CoordinateMap instance is
    from the first one in the list.
    """
    img = load_image(str(repository._fullpath('avg152T1.nii.gz')))
    mimg = merge_images([img for i in range(4)])
    yield assert_equal, mimg.shape, (4,) + img.shape
    yield nptest.assert_almost_equal, mimg.affine[1:,1:], img.affine
    yield nptest.assert_almost_equal, mimg.affine[0], np.array([1,0,0,0,0])
    yield nptest.assert_almost_equal, mimg.affine[:,0], np.array([1,0,0,0,0])

    # A list with different CoordinateMaps -- the merged images
    # takes the first one

    naffine = Affine(np.diag([-2,-4,-6,1.]),
                     img.coordmap.input_coords,
                     img.coordmap.output_coords)
    nimg = Image(np.asarray(img), naffine)
    mimg = merge_images([nimg, img, img, img])
    yield assert_equal, mimg.shape, (4,) + img.shape
    yield nptest.assert_almost_equal, mimg.affine[1:,1:], nimg.affine
    yield nptest.assert_almost_equal, mimg.affine[0], np.array([1,0,0,0,0])
    yield nptest.assert_almost_equal, mimg.affine[:,0], np.array([1,0,0,0,0])

def test_slicing_returns_image():
    data = np.ones((2,3,4))
    img = fromarray(data, 'kji', 'zyx')
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


class ArrayLikeObj(TestCase):
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
    # create simple coordmap
    xform = np.eye(4)
    coordmap = Affine.from_params('xyz', 'ijk', xform)
    
    # create image form array-like object and coordmap
    img = image.Image(obj, coordmap)
    assert img.ndim == 3
    assert img.shape == (2,3,4)
    assert np.allclose(np.asarray(img), 1)
    assert np.allclose(img[:], 1)
    img[:] = 4
    assert np.allclose(img[:], 4)

# FIXME: 
# This test below fails because save_image does not
# deal with these fields of the header.

@dec.knownfailure
def test_header_roundtrip():
    # This test fails because save_image does not deal with all fields
    # of the header.

    img = load_image(str(repository._fullpath('avg152T1.nii.gz')))
    fd, name = mkstemp(suffix='.nii.gz')
    tmpfile = open(name)
    hdr = img.header
    # Update some header values and make sure they're saved
    hdr['slice_duration'] = 0.200
    hdr['intent_p1'] = 2.0
    hdr['descrip'] = 'descrip for TestImage:test_header_roundtrip'
    hdr['slice_end'] = 12
    img.header = hdr
    save_image(img, tmpfile.name)
    newimg = load_image(tmpfile.name)
    newhdr = newimg.header
    tmpfile.close()
    os.unlink(name)

    yield assert_true, np.allclose(newhdr['slice_duration'], hdr['slice_duration'])
    yield assert_equal, newhdr['intent_p1'], hdr['intent_p1']
    yield assert_equal, newhdr['descrip'], hdr['descrip']
    yield assert_equal, newhdr['slice_end'], hdr['slice_end']

def test_file_roundtrip():
    img = load_image(str(repository._fullpath('avg152T1.nii.gz')))
    fd, name = mkstemp(suffix='.nii.gz')
    tmpfile = open(name)
    save_image(img, tmpfile.name)
    img2 = load_image(tmpfile.name)
    data = np.asarray(img)
    data2 = np.asarray(img2)
    tmpfile.close()
    os.unlink(name)

    # verify data
    yield assert_true, np.allclose(data2, data)
    yield assert_true, np.allclose(data2.mean(), data.mean())
    yield assert_true, np.allclose(data2.min(), data.min())
    yield assert_true, np.allclose(data2.max(), data.max())
    
    # verify shape and ndims
    yield assert_equal, img2.shape, img.shape
    yield assert_equal, img2.ndim, img.ndim
    # verify affine
    yield assert_true, np.allclose(img2.affine, img.affine)

def test_roundtrip_fromarray():
    data = np.random.rand(10,20,30)
    img = fromarray(data, 'kji', 'zyx')
    fd, name = mkstemp(suffix='.nii.gz')
    tmpfile = open(name)
    save_image(img, tmpfile.name)
    img2 = load_image(tmpfile.name)
    data2 = np.asarray(img2)
    tmpfile.close()
    os.unlink(name)

    # verify data
    yield assert_true, np.allclose(data2, data)
    yield assert_true, np.allclose(data2.mean(), data.mean())
    yield assert_true, np.allclose(data2.min(), data.min())
    yield assert_true, np.allclose(data2.max(), data.max())

    # verify shape and ndims
    yield assert_equal, img2.shape, img.shape
    yield assert_equal, img2.ndim, img.ndim
    # verify affine
    yield assert_true, np.allclose(img2.affine, img.affine)


# Should test common image sizes 2D, 3D, 4D
class TestFromArray(TestCase):
    def setUp(self):
        self.array2D_shape = (2,3)
        self.array3D_shape = (2,3,4)
        self.array4D_shape = (2,3,4,5)

    def test_defaults_2D(self):
        data = np.ones(self.array2D_shape)
        img = image.fromarray(data, 'kj', 'yx')
        assert isinstance(img._data, np.ndarray)
        assert img.ndim == 2
        assert img.shape == self.array2D_shape
        assert_raises(AttributeError, getattr, img, 'header')
        assert img.affine.shape == (3,3)
        assert img.affine.diagonal().all() == 1
        
    def test_defaults_3D(self):
        img = image.fromarray(np.ones(self.array3D_shape), 'kji', 'zyx')
        assert isinstance(img._data, np.ndarray)
        assert img.ndim == 3
        assert img.shape == self.array3D_shape
        # ndarray's do not have a header
        assert_raises(AttributeError, getattr, img, 'header')
        assert img.affine.shape == (4,4)
        assert img.affine.diagonal().all() == 1

    def test_defaults_4D(self):
        data = np.ones(self.array4D_shape)
        names = ['time', 'zspace', 'yspace', 'xspace']
        img = image.fromarray(data, names, names)
        assert isinstance(img._data, np.ndarray)
        assert img.ndim == 4
        assert img.shape == self.array4D_shape
        assert_raises(AttributeError, getattr, img, 'header')
        assert img.affine.shape == (5,5)
        assert img.affine.diagonal().all() == 1
