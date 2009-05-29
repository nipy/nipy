import os
import warnings
from tempfile import mkstemp

import numpy as np
import numpy.testing as nptest

from nipy.testing import assert_true, assert_equal, assert_raises, \
    assert_array_equal, assert_array_almost_equal, TestCase

from nipy.core.image import image
from nipy.utils.tests.data import repository
from nipy.io.api import load_image, save_image
from nipy.core.api import Image, fromarray, merge_images
from nipy.core.api import parcels, data_generator, write_data

from nipy.core.reference.coordinate_map import Affine

'''
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

    def test_nondiag(self):
        self.img.affine[0,1] = 3.0
        save_image(self.img, self.tmpfile.name)
        img2 = load_image(self.tmpfile.name)
        assert_true(np.allclose(img2.affine, self.img.affine))

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

def test_scaling_uint8_to_uint8():
    dtype = np.uint8
    newdata, data = self.uint8_to_dtype(dtype, self.tmpfile.name)
    assert_true(np.allclose(newdata, data))

def test_scaling_uint8_to_uint16():
    dtype = np.uint16
    newdata, data = self.uint8_to_dtype(dtype, self.tmpfile.name)
    assert_true(np.allclose(newdata, data))

def test_scaling_uint8_to_float32():
    dtype = np.float32
    newdata, data = self.uint8_to_dtype(dtype, self.tmpfile.name)
    assert_true(np.allclose(newdata, data))

def test_scaling_uint8_to_int32():
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

def test_scaling_float32_to_uint8():
    dtype = np.uint8
    newdata, data = self.float32_to_dtype(dtype)
    assert_true(np.allclose(newdata, data))

def test_scaling_float32_to_uint16():
    dtype = np.uint16
    newdata, data = self.float32_to_dtype(dtype)
    assert_true(np.allclose(newdata, data))

def test_scaling_float32_to_int16():
    dtype = np.int16
    newdata, data = self.float32_to_dtype(dtype)
    assert_true(np.allclose(newdata, data))

def test_scaling_float32_to_float32():
    dtype = np.float32
    newdata, data = self.float32_to_dtype(dtype)
    assert_true(np.allclose(newdata, data))


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

'''
