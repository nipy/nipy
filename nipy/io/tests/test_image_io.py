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
from nibabel.tests.test_round_trip import big_bad_ulp

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


def randimg_in2out(rng, in_dtype, out_dtype, name):
    in_dtype = np.dtype(in_dtype)
    out_dtype = np.dtype(out_dtype)
    shape = (2,3,4)
    if in_dtype.kind in 'iu':
        info = np.iinfo(in_dtype)
        dmin, dmax = info.min, info.max
        # Numpy bug for np < 1.6.0 allows overflow for range that does not fit
        # into C long int (int32 on 32-bit, int64 on 64-bit)
        try:
            data = rng.randint(dmin, dmax, size=shape)
        except ValueError:
            from random import randint
            vals = [randint(dmin, dmax) for v in range(np.prod(shape))]
            data = np.array(vals).astype(in_dtype).reshape(shape)
    elif in_dtype.kind == 'f':
        info = np.finfo(in_dtype)
        dmin, dmax = info.min, info.max
        # set some value for scaling our data
        scale = np.iinfo(np.uint16).max * 2.0
        data = rng.normal(size=shape, scale=scale)
    data[0,0,0] = dmin
    data[1,0,0] = dmax
    data = data.astype(in_dtype)
    img = Image(data, AfT('kji', 'zxy', np.eye(4)))
    # The io_dtype won't be visible until the image is loaded
    newimg = save_image(img, name, dtype_from=out_dtype)
    return newimg.get_data(), data


def test_scaling_io_dtype():
    # Does data dtype get set?
    # Is scaling correctly applied?
    rng = np.random.RandomState(19660520) # VBD
    ulp1_f32 = np.finfo(np.float32).eps
    types = (np.uint8, np.uint16, np.int16, np.int32, np.float32)
    with InTemporaryDirectory():
        for in_type in types:
            for out_type in types:
                data, _ = randimg_in2out(rng, in_type, out_type, 'img.nii')
                img = load_image('img.nii')
                # Check the output type is as expected
                hdr = img.metadata['header']
                assert_equal(hdr.get_data_dtype().type, out_type)
                # Check the data is within reasonable bounds. The exact bounds
                # are a little annoying to calculate - see
                # nibabel/tests/test_round_trip for inspiration
                data_back = img.get_data()
                top = np.abs(data - data_back)
                nzs = (top !=0) & (data !=0)
                abs_err = top[nzs]
                if abs_err.size != 0: # all exact, that's OK.
                    continue
                rel_err = abs_err / data[nzs]
                if np.dtype(out_type).kind in 'iu':
                    slope, inter = hdr.get_slope_inter()
                    abs_err_thresh = slope / 2.0
                    rel_err_thresh = ulp1_f32
                elif np.dtype(out_type).kind == 'f':
                    abs_err_thresh = big_bad_ulp(data.astype(out_type))[nzs]
                    rel_err_thresh = ulp1_f32
                assert_true(np.all(
                    (abs_err <= abs_err_thresh) |
                    (rel_err <= rel_err_thresh)))
        del img


def test_from_data():
    # Default data dtype comes from data
    pass


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
