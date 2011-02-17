# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
import warnings
from tempfile import mkstemp

import numpy as np

from nibabel.spatialimages import ImageFileError

from nipy.testing import assert_true, assert_equal, assert_raises, \
    assert_array_equal, assert_array_almost_equal, funcfile, parametric

from nipy.testing.decorators import if_templates

from nipy.utils import templates, DataError

from nipy.io.api import load_image, save_image, as_image
from nipy.core.api import fromarray


gfilename = ''
gtmpfile = None
gimg = None


def load_template_img():
    return load_image(
        templates.get_filename(
            'ICBM152', '2mm', 'T1.nii.gz'))


def setup_module():
    warnings.simplefilter("ignore")
    global gfilename, gtmpfile, gimg
    try:
        gimg = load_template_img()
    except DataError:
        pass
    fd, gfilename = mkstemp(suffix='.nii.gz')
    gtmpfile = open(gfilename)


def teardown_module():
    warnings.resetwarnings()
    global gtmpfile, gfilename
    gtmpfile.close()
    os.unlink(gfilename)


def test_badfile():
    filename = "bad_file.foo"
    assert_raises(ImageFileError, load_image, filename)


@if_templates
def test_maxminmean_values():
    # loaded array values from SPM
    y = np.asarray(gimg)
    yield assert_equal, y.shape, tuple(gimg.shape)
    yield assert_array_almost_equal, y.max(), 1.000000059
    yield assert_array_almost_equal, y.mean(), 0.273968048
    yield assert_equal, y.min(), 0.0


@if_templates
def test_nondiag():
    gimg.affine[0,1] = 3.0
    save_image(gimg, gtmpfile.name)
    img2 = load_image(gtmpfile.name)
    yield assert_true, np.allclose(img2.affine, gimg.affine)


def uint8_to_dtype(dtype, name):
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
    newdata, data = uint8_to_dtype(dtype, gtmpfile.name)
    yield assert_true, np.allclose(newdata, data)


def test_scaling_uint8_to_uint16():
    dtype = np.uint16
    newdata, data = uint8_to_dtype(dtype, gtmpfile.name)
    yield assert_true, np.allclose(newdata, data)


def test_scaling_uint8_to_float32():
    dtype = np.float32
    newdata, data = uint8_to_dtype(dtype, gtmpfile.name)
    yield assert_true, np.allclose(newdata, data)


def test_scaling_uint8_to_int32():
    dtype = np.int32
    newdata, data = uint8_to_dtype(dtype, gtmpfile.name)
    yield assert_true, np.allclose(newdata, data)


def float32_to_dtype(dtype):
    # Utility function for the scaling_float32 function
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
    newimg = save_image(img, gtmpfile.name, dtype=dtype)
    newdata = np.asarray(newimg)
    return newdata, data


def test_scaling_float32():
    for dtype in (np.uint8, np.uint16, np.int16, np.float32):
        newdata, data = float32_to_dtype(dtype)
        yield assert_array_almost_equal, newdata, data


@if_templates
def test_header_roundtrip():
    img = load_template_img()
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
    yield (assert_array_almost_equal,
           newhdr['slice_duration'],
           hdr['slice_duration'])
    yield assert_equal, newhdr['intent_p1'], hdr['intent_p1']
    yield assert_equal, newhdr['descrip'], hdr['descrip']
    yield assert_equal, newhdr['slice_end'], hdr['slice_end']


@if_templates
def test_file_roundtrip():
    img = load_template_img()
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
    img = fromarray(data, 'kji', 'xyz')
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


@parametric
def test_as_image():
    # test image creation / pass through function
    img = as_image(funcfile) # string filename
    img1 = as_image(unicode(funcfile))
    img2 = as_image(img)
    yield assert_equal(img.affine, img1.affine)
    yield assert_array_equal(np.asarray(img), np.asarray(img1))
    yield assert_true(img is img2)
