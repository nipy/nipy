from tempfile import NamedTemporaryFile
from os import remove

import numpy as np

from neuroimaging.testing import dec
import nose.tools

from neuroimaging.core.image import image
from neuroimaging.utils.tests.data import repository
from neuroimaging.core.api import Image, load_image, save_image, fromarray
from neuroimaging.core.api import parcels, data_generator, write_data

from neuroimaging.core.reference.coordinate_map import CoordinateMap, Affine

class TestImage:

    def setUp(self):
        self.img = load_image(str(repository._fullpath('avg152T1.nii.gz')))
        self.tmpfile = NamedTemporaryFile(suffix='.nii.gz')
        
    def test_init(self):
        new = Image(np.asarray(self.img), self.img.coordmap)
        nose.tools.assert_true(np.allclose(np.asarray(self.img), np.asarray(new)))
        nose.tools.assert_raises(ValueError, Image, None, None)

    def test_badfile(self):
        filename = "bad_file.foo"
        nose.tools.assert_raises(RuntimeError, load_image, filename)

    def test_maxmin_values(self):
        y = np.asarray(self.img)
        nose.tools.assert_equal(y.shape, tuple(self.img.shape))
        nose.tools.assert_true(np.allclose(y.max(), 437336.36, rtol=1.0e-8))
        nose.tools.assert_equal(y.min(), 0.0)

    def test_slice_plane(self):
        x = self.img[3]
        nose.tools.assert_equal(x.shape, self.img.shape[1:])

    def test_slice_block(self):
        x = self.img[3:5]
        nose.tools.assert_equal(x.shape, (2,) + tuple(self.img.shape[1:]))

    def test_slice_step(self):
        s = slice(0,20,2)
        x = self.img[s]
        nose.tools.assert_equal(x.shape, (10,) + tuple(self.img.shape[1:]))

    def test_slice_type(self):
        s = slice(0,self.img.shape[0])
        x = self.img[s]
        nose.tools.assert_equal(x.shape, self.img.shape)

    def test_slice_steps(self):
        zdim, ydim, xdim = self.img.shape
        slice_z = slice(0, zdim, 2)
        slice_y = slice(0, ydim, 2)
        slice_x = slice(0, xdim, 2)
        x = self.img[slice_z, slice_y, slice_x]
        newshape = ((zdim/2)+1, (ydim/2)+1, (xdim/2)+1)
        nose.tools.assert_equal(x.shape, newshape)

    def test_array(self):
        x = np.asarray(self.img)
        assert isinstance(x, np.ndarray)
        nose.tools.assert_equal(x.shape, self.img.shape)
        nose.tools.assert_equal(x.ndim, self.img.ndim)
        
    def test_nondiag(self):
        self.img.affine[0,1] = 3.0
        save_image(self.img, self.tmpfile.name)
        img2 = load_image(self.tmpfile.name)
        print self.img.affine, img2.affine
        nose.tools.assert_true(np.allclose(img2.affine, self.img.affine))

    def test_generator(self):
        gen = data_generator(self.img)
        for ind, data in gen:
            nose.tools.assert_equal(data.shape, (109,91))

    def test_iter(self):
        imgiter = iter(self.img)
        for data in imgiter:
            nose.tools.assert_equal(data.shape, (109,91))

    def test_iter4(self):
        tmp = Image(np.zeros(self.img.shape), self.img.coordmap)
        write_data(tmp, data_generator(self.img, range(self.img.shape[0])))
        nose.tools.assert_true(np.allclose(np.asarray(tmp), np.asarray(self.img)))

    def test_iter5(self):
        
        tmp = Image(np.zeros(self.img.shape), self.img.coordmap)
        g = data_generator(self.img)
        write_data(tmp, g)
        nose.tools.assert_true(np.allclose(np.asarray(tmp), np.asarray(self.img)))

    def test_parcels1(self):
        rho = self.img
        parcelmap = (np.asarray(rho)[:] * 100).astype(np.int32)
        
        test = np.zeros(parcelmap.shape)
        v = 0
        for i, d in data_generator(test, parcels(parcelmap)):
            v += d.shape[0]

        nose.tools.assert_equal(v, np.product(test.shape))

    def test_parcels3(self):
        rho = self.img[0]
        parcelmap = (np.asarray(rho)[:] * 100).astype(np.int32)
        labels = np.unique(parcelmap)
        test = np.zeros(rho.shape)

        v = 0
        for i, d in data_generator(test, parcels(parcelmap, labels=labels)):
            v += d.shape[0]

        nose.tools.assert_equal(v, np.product(test.shape))

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
        nose.tools.assert_true(np.allclose(newdata, data))

    def test_scaling_uint8_to_uint16(self):
        dtype = np.uint16
        newdata, data = self.uint8_to_dtype(dtype, self.tmpfile.name)
        nose.tools.assert_true(np.allclose(newdata, data))

    def test_scaling_uint8_to_float32(self):
        dtype = np.float32
        newdata, data = self.uint8_to_dtype(dtype, self.tmpfile.name)
        nose.tools.assert_true(np.allclose(newdata, data))

    def test_scaling_uint8_to_int32(self):
        dtype = np.int32
        newdata, data = self.uint8_to_dtype(dtype, self.tmpfile.name)
        nose.tools.assert_true(np.allclose(newdata, data))
    
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
        nose.tools.assert_true(np.allclose(newdata, data))

    def test_scaling_float32_to_uint16(self):
        dtype = np.uint16
        newdata, data = self.float32_to_dtype(dtype)
        nose.tools.assert_true(np.allclose(newdata, data))
        
    def test_scaling_float32_to_int16(self):
        dtype = np.int16
        newdata, data = self.float32_to_dtype(dtype)
        nose.tools.assert_true(np.allclose(newdata, data))

    def test_scaling_float32_to_float32(self):
        dtype = np.float32
        newdata, data = self.float32_to_dtype(dtype)
        nose.tools.assert_true(np.allclose(newdata, data))

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
    """
    This test fails because save_image does not deal with all fields of the header.

    """
    img = load_image(str(repository._fullpath('avg152T1.nii.gz')))
    tmpfile = NamedTemporaryFile(suffix='.nii.gz')
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
    yield nose.tools.assert_true, np.allclose(newhdr['slice_duration'], hdr['slice_duration'])
    yield nose.tools.assert_equal, newhdr['intent_p1'], hdr['intent_p1']
    yield nose.tools.assert_equal, newhdr['descrip'], hdr['descrip']
    yield nose.tools.assert_equal, newhdr['slice_end'], hdr['slice_end']

def test_file_roundtrip():
    img = load_image(str(repository._fullpath('avg152T1.nii.gz')))
    tmpfile = NamedTemporaryFile(suffix='.nii.gz')
    save_image(img, tmpfile.name)
    img2 = load_image(tmpfile.name)
    data = np.asarray(img)
    data2 = np.asarray(img2)
    # verify data
    yield nose.tools.assert_true, np.allclose(data2, data)
    yield nose.tools.assert_true, np.allclose(data2.mean(), data.mean())
    yield nose.tools.assert_true, np.allclose(data2.min(), data.min())
    yield nose.tools.assert_true, np.allclose(data2.max(), data.max())
    
    # verify shape and ndims
    yield nose.tools.assert_equal, img2.shape, img.shape
    yield nose.tools.assert_equal, img2.ndim, img.ndim
    # verify affine
    yield nose.tools.assert_true, np.allclose(img2.affine, img.affine)

def test_roundtrip_fromarray():
    data = np.random.rand(10,20,30)
    img = fromarray(data, 'kji', 'zyx')
    tmpfile = NamedTemporaryFile(suffix='.nii.gz')
    save_image(img, tmpfile.name)
    img2 = load_image(tmpfile.name)
    data2 = np.asarray(img2)
    # verify data
    yield nose.tools.assert_true, np.allclose(data2, data)
    yield nose.tools.assert_true, np.allclose(data2.mean(), data.mean())
    yield nose.tools.assert_true, np.allclose(data2.min(), data.min())
    yield nose.tools.assert_true, np.allclose(data2.max(), data.max())

    # verify shape and ndims
    yield nose.tools.assert_equal, img2.shape, img.shape
    yield nose.tools.assert_equal, img2.ndim, img.ndim
    # verify affine
    yield nose.tools.assert_true, np.allclose(img2.affine, img.affine)


# Should test common image sizes 2D, 3D, 4D
class TestFromArray:
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
        nose.tools.assert_raises(AttributeError, getattr, img, 'header')
        assert img.affine.shape == (3,3)
        assert img.affine.diagonal().all() == 1
        
    def test_defaults_3D(self):
        img = image.fromarray(np.ones(self.array3D_shape), 'kji', 'zyx')
        assert isinstance(img._data, np.ndarray)
        assert img.ndim == 3
        assert img.shape == self.array3D_shape
        # ndarray's do not have a header
        nose.tools.assert_raises(AttributeError, getattr, img, 'header')
        assert img.affine.shape == (4,4)
        assert img.affine.diagonal().all() == 1

    def test_defaults_4D(self):
        data = np.ones(self.array4D_shape)
        names = ['time', 'zspace', 'yspace', 'xspace']
        img = image.fromarray(data, names, names)
        assert isinstance(img._data, np.ndarray)
        assert img.ndim == 4
        assert img.shape == self.array4D_shape
        nose.tools.assert_raises(AttributeError, getattr, img, 'header')
        assert img.affine.shape == (5,5)
        assert img.affine.diagonal().all() == 1




