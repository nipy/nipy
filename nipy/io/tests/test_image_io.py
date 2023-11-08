# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from os.path import dirname
from os.path import join as pjoin

import nibabel as nib
import numpy as np
import pytest
from nibabel import Nifti1Header
from nibabel.filebasedimages import ImageFileError
from nibabel.spatialimages import HeaderDataError
from nibabel.tests.test_round_trip import big_bad_ulp

from nipy.core.api import AffineTransform as AfT
from nipy.core.api import Image, vox2mni
from nipy.testing import (
    anatfile,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    funcfile,
)
from nipy.testing.decorators import if_templates
from nipy.utils import DataError, templates

from ..api import as_image, load_image, save_image


@pytest.fixture(scope='module')
def tpl_img():
    try:
        gimg = load_image(templates.get_filename(
            'ICBM152', '2mm', 'T1.nii.gz'))
    except DataError:
        gimg = None
    return gimg


def test_badfile():
    filename = "bad_file.foo"
    # nibabel prior 2.1.0 was throwing a ImageFileError for the not-recognized
    # file type.  >=2.1.0 give a FileNotFoundError.
    pytest.raises((ImageFileError, FileNotFoundError), load_image, filename)


@if_templates
def test_maxminmean_values(tpl_img):
    # loaded array values from SPM
    y = tpl_img.get_fdata()
    assert y.shape == tuple(tpl_img.shape)
    assert_array_almost_equal(y.max(), 1.000000059)
    assert_array_almost_equal(y.mean(), 0.273968048)
    assert y.min() == 0.0


@if_templates
def test_nondiag(tpl_img, in_tmp_path):
    tpl_img.affine[0,1] = 3.0
    save_image(tpl_img, 'img.nii')
    img2 = load_image('img.nii')
    assert_almost_equal(img2.affine, tpl_img.affine)


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
    img = Image(data, vox2mni(np.eye(4)))
    # The dtype_from dtype won't be visible until the image is loaded
    newimg = save_image(img, name, dtype_from=out_dtype)
    return newimg.get_fdata(), data


def test_scaling_io_dtype(in_tmp_path):
    # Does data dtype get set?
    # Is scaling correctly applied?
    rng = np.random.RandomState(19660520) # VBD
    ulp1_f32 = np.finfo(np.float32).eps
    types = (np.uint8, np.uint16, np.int16, np.int32, np.float32)
    for in_type in types:
        for out_type in types:
            data, _ = randimg_in2out(rng, in_type, out_type, 'img.nii')
            img = load_image('img.nii')
            # Check the output type is as expected
            hdr = img.metadata['header']
            assert hdr.get_data_dtype().type == out_type
            # Check the data is within reasonable bounds. The exact bounds
            # are a little annoying to calculate - see
            # nibabel/tests/test_round_trip for inspiration
            data_back = img.get_fdata().copy() # copy to detach from file
            del img
            top = np.abs(data - data_back)
            nzs = (top !=0) & (data !=0)
            abs_err = top[nzs]
            if abs_err.size != 0: # all exact, that's OK.
                continue
            rel_err = abs_err / data[nzs]
            if np.dtype(out_type).kind in 'iu':
                # Read slope from input header
                with open('img.nii', 'rb') as fobj:
                    orig_hdr = hdr.from_fileobj(fobj)
                abs_err_thresh = orig_hdr['scl_slope'] / 2.0
                rel_err_thresh = ulp1_f32
            elif np.dtype(out_type).kind == 'f':
                abs_err_thresh = big_bad_ulp(data.astype(out_type))[nzs]
                rel_err_thresh = ulp1_f32
            assert np.all(
                (abs_err <= abs_err_thresh) |
                (rel_err <= rel_err_thresh))


def assert_dt_no_end_equal(a, b):
    """ Assert two numpy dtype specifiers are equal apart from byte order

    Avoids failed comparison between int32 / int64 and intp
    """
    a = np.dtype(a).newbyteorder('=')
    b = np.dtype(b).newbyteorder('=')
    assert a.str == b.str


def test_output_dtypes(in_tmp_path):
    shape = (4, 2, 3)
    rng = np.random.RandomState(19441217) # IN-S BD
    data = rng.normal(4, 20, size=shape)
    aff = np.diag([2.2, 3.3, 4.1, 1])
    cmap = vox2mni(aff)
    img = Image(data, cmap)
    fname_root = 'my_file'
    for ext in 'img', 'nii':
        out_fname = fname_root + '.' + ext
        # Default is for data to come from data dtype
        save_image(img, out_fname)
        img_back = load_image(out_fname)
        hdr = img_back.metadata['header']
        assert_dt_no_end_equal(hdr.get_data_dtype(), np.float64)
        del img_back # lets window reuse the file
        # All these types are OK for both output formats
        for out_dt in 'i2', 'i4', np.int16, '<f4', '>f8':
            # Specified output dtype
            save_image(img, out_fname, out_dt)
            img_back = load_image(out_fname)
            hdr = img_back.metadata['header']
            assert_dt_no_end_equal(hdr.get_data_dtype(), out_dt)
            del img_back # windows file reuse
            # Output comes from data by default
            data_typed = data.astype(out_dt)
            img_again = Image(data_typed, cmap)
            save_image(img_again, out_fname)
            img_back = load_image(out_fname)
            hdr = img_back.metadata['header']
            assert_dt_no_end_equal(hdr.get_data_dtype(), out_dt)
            del img_back
            # Even if header specifies otherwise
            in_hdr = Nifti1Header()
            in_hdr.set_data_dtype(np.dtype('c8'))
            img_more = Image(data_typed, cmap, metadata={'header': in_hdr})
            save_image(img_more, out_fname)
            img_back = load_image(out_fname)
            hdr = img_back.metadata['header']
            assert_dt_no_end_equal(hdr.get_data_dtype(), out_dt)
            del img_back
            # But can come from header if specified
            save_image(img_more, out_fname, dtype_from='header')
            img_back = load_image(out_fname)
            hdr = img_back.metadata['header']
            assert_dt_no_end_equal(hdr.get_data_dtype(), 'c8')
            del img_back
    # u2 only OK for nifti
    save_image(img, 'my_file.nii', 'u2')
    img_back = load_image('my_file.nii')
    hdr = img_back.metadata['header']
    assert_dt_no_end_equal(hdr.get_data_dtype(), 'u2')
    # Check analyze can't save u2 datatype
    pytest.raises(HeaderDataError, save_image, img, 'my_file.img', 'u2')
    del img_back


def test_header_roundtrip(in_tmp_path):
    img = load_image(anatfile)
    hdr = img.metadata['header']
    # Update some header values and make sure they're saved
    hdr['slice_duration'] = 0.200
    hdr['intent_p1'] = 2.0
    hdr['descrip'] = 'descrip for TestImage:test_header_roundtrip'
    hdr['slice_end'] = 12
    save_image(img, 'img.nii.gz')
    newimg = load_image('img.nii.gz')
    newhdr = newimg.metadata['header']
    assert_array_almost_equal(newhdr['slice_duration'],
                              hdr['slice_duration'])
    assert newhdr['intent_p1'] == hdr['intent_p1']
    assert newhdr['descrip'] == hdr['descrip']
    assert newhdr['slice_end'] == hdr['slice_end']


def test_file_roundtrip(in_tmp_path):
    img = load_image(anatfile)
    data = img.get_fdata()
    save_image(img, 'img.nii.gz')
    img2 = load_image('img.nii.gz')
    data2 = img2.get_fdata()
    # verify data
    assert_almost_equal(data2, data)
    assert_almost_equal(data2.mean(), data.mean())
    assert_almost_equal(data2.min(), data.min())
    assert_almost_equal(data2.max(), data.max())
    # verify shape and ndims
    assert img2.shape == img.shape
    assert img2.ndim == img.ndim
    # verify affine
    assert_almost_equal(img2.affine, img.affine)
    # Test we can use Path objects
    out_path = 'path_img.nii'
    save_image(img, out_path)
    img2 = load_image(out_path)
    data2 = img2.get_fdata()
    # verify data
    assert_almost_equal(data2, data)


def test_roundtrip_from_array(in_tmp_path):
    data = np.random.rand(10,20,30)
    img = Image(data, AfT('kji', 'xyz', np.eye(4)))
    save_image(img, 'img.nii.gz')
    img2 = load_image('img.nii.gz')
    data2 = img2.get_fdata()
    # verify data
    assert_almost_equal(data2, data)
    assert_almost_equal(data2.mean(), data.mean())
    assert_almost_equal(data2.min(), data.min())
    assert_almost_equal(data2.max(), data.max())
    # verify shape and ndims
    assert img2.shape == img.shape
    assert img2.ndim == img.ndim
    # verify affine
    assert_almost_equal(img2.affine, img.affine)


def test_as_image():
    # test image creation / pass through function
    img = as_image(funcfile)  # string filename
    img1 = as_image(funcfile)  # unicode
    img2 = as_image(img)
    assert_array_equal(img.affine, img1.affine)
    assert_array_equal(img.get_fdata(), img1.get_fdata())
    assert img is img2


def test_no_minc():
    # We can't yet get good axis names for MINC files. Check we reject these
    pytest.raises(ValueError, load_image, 'nofile.mnc')
    data_path = pjoin(dirname(nib.__file__), 'tests', 'data')
    pytest.raises(ValueError, load_image, pjoin(data_path, 'tiny.mnc'))
