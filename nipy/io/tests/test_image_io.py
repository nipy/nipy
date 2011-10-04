# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np

from nibabel.spatialimages import ImageFileError

from ..api import load_image, save_image, as_image
from nipy.core.api import AffineTransform as AfT, Image

from nipy.testing import (assert_true, assert_equal, assert_raises,
                          assert_array_equal, assert_array_almost_equal,
                          assert_almost_equal, funcfile, anatfile)

from nibabel.tmpdirs import InTemporaryDirectory

from nipy.testing.decorators import if_templates
from nipy.utils import templates, DataError

gimg = None

def setup_module():
    global gimg
    try:
        gimg = load_template_img()
    except DataError:
        pass


def load_template_img():
    return load_image(
        templates.get_filename(
            'ICBM152', '2mm', 'T1.nii.gz'))


def test_badfile():
    filename = "bad_file.foo"
    assert_raises(ImageFileError, load_image, filename)


@if_templates
def test_maxminmean_values():
    # loaded array values from SPM
    y = gimg.get_data()
    yield assert_equal, y.shape, tuple(gimg.shape)
    yield assert_array_almost_equal, y.max(), 1.000000059
    yield assert_array_almost_equal, y.mean(), 0.273968048
    yield assert_equal, y.min(), 0.0


@if_templates
def test_nondiag():
    gimg.affine[0,1] = 3.0
    with InTemporaryDirectory():
        save_image(gimg, 'img.nii')
        img2 = load_image('img.nii')
        assert_almost_equal(img2.affine, gimg.affine)


def uint8_to_dtype(dtype, name):
    dtype = dtype
    shape = (2,3,4)
    dmax = np.iinfo(np.uint8).max
    data = np.random.randint(0, dmax, size=shape)
    data[0,0,0] = 0
    data[1,0,0] = dmax
    data = data.astype(np.uint8) # randint returns np.int32
    img = Image(data, AfT('kji', 'zxy', np.eye(4)))
    newimg = save_image(img, name, dtype=dtype)
    return newimg.get_data(), data


def float32_to_dtype(dtype, name):
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
    img = Image(data, AfT('kji', 'zyx', np.eye(4)))
    newimg = save_image(img, name, dtype=dtype)
    return newimg.get_data(), data


def test_scaling():
    with InTemporaryDirectory():
        for dtype_type in (np.uint8, np.uint16,
                           np.int16, np.int32,
                           np.float32):
            newdata, data = uint8_to_dtype(dtype_type, 'img.nii')
            assert_almost_equal(newdata, data)
            newdata, data = float32_to_dtype(dtype_type, 'img.nii')
            assert_almost_equal(newdata, data)


def test_header_roundtrip():
    img = load_image(anatfile)
    hdr = img.metadata['header']
    # Update some header values and make sure they're saved
    hdr['slice_duration'] = 0.200
    hdr['intent_p1'] = 2.0
    hdr['descrip'] = 'descrip for TestImage:test_header_roundtrip'
    hdr['slice_end'] = 12
    with InTemporaryDirectory():
        save_image(img, 'img.nii.gz')
        newimg = load_image('img.nii.gz')
    newhdr = newimg.metadata['header']
    assert_array_almost_equal(newhdr['slice_duration'],
                              hdr['slice_duration'])
    assert_equal(newhdr['intent_p1'], hdr['intent_p1'])
    assert_equal(newhdr['descrip'], hdr['descrip'])
    assert_equal(newhdr['slice_end'], hdr['slice_end'])


def test_file_roundtrip():
    img = load_image(anatfile)
    data = img.get_data()
    with InTemporaryDirectory():
        save_image(img, 'img.nii.gz')
        img2 = load_image('img.nii.gz')
        data2 = img2.get_data()
    # verify data
    assert_almost_equal(data2, data)
    assert_almost_equal(data2.mean(), data.mean())
    assert_almost_equal(data2.min(), data.min())
    assert_almost_equal(data2.max(), data.max())
    # verify shape and ndims
    assert_equal(img2.shape, img.shape)
    assert_equal(img2.ndim, img.ndim)
    # verify affine
    assert_almost_equal(img2.affine, img.affine)


def test_roundtrip_from_array():
    data = np.random.rand(10,20,30)
    img = Image(data, AfT('kji', 'xyz', np.eye(4)))
    with InTemporaryDirectory():
        save_image(img, 'img.nii.gz')
        img2 = load_image('img.nii.gz')
        data2 = img2.get_data()
    # verify data
    assert_almost_equal(data2, data)
    assert_almost_equal(data2.mean(), data.mean())
    assert_almost_equal(data2.min(), data.min())
    assert_almost_equal(data2.max(), data.max())
    # verify shape and ndims
    assert_equal(img2.shape, img.shape)
    assert_equal(img2.ndim, img.ndim)
    # verify affine
    assert_almost_equal(img2.affine, img.affine)


def test_as_image():
    # test image creation / pass through function
    img = as_image(funcfile) # string filename
    img1 = as_image(unicode(funcfile))
    img2 = as_image(img)
    assert_equal(img.affine, img1.affine)
    assert_array_equal(img.get_data(), img1.get_data())
    assert_true(img is img2)
