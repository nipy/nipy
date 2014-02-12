""" Testing diagnostics.command module
"""

from os.path import dirname, join as pjoin, isfile

import numpy as np

import nibabel as nib
from nibabel import AnalyzeImage, Spm2AnalyzeImage, Nifti1Pair, Nifti1Image
from nibabel.tmpdirs import InTemporaryDirectory

from ..commands import parse_fname_axes

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose import SkipTest
from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


def test_parse_fname_axes():
    # Test logic for setting time and slice axis defaults
    # We need real images for the tests because nipy will load them
    # For simplicity, we can create them
    shape = (4, 5, 6, 20)
    arr = np.arange(np.prod(shape), dtype=float).reshape(shape)
    zooms = (2., 3., 4., 2.1)
    with InTemporaryDirectory():
        for (img_class, ext) in ((AnalyzeImage, '.img'),
                                 (Spm2AnalyzeImage, '.img'),
                                 (Nifti1Pair, '.img'),
                                 (Nifti1Image, '.nii')):
            hdr = img_class.header_class()
            hdr.set_data_shape(shape)
            hdr.set_zooms(zooms)
            hdr.set_data_dtype(np.dtype(np.float64))
            nibabel_img = img_class(arr, None, hdr)
            # We so far haven't set any slice axis information
            for z_ext in ('', '.gz'):
                fname = 'image' + ext + z_ext
                nib.save(nibabel_img, fname)
                for in_time, in_sax, out_time, out_sax in (
                    (None, None, 't', 2),
                    (None, '0', 't', 0),
                    (None, 'i', 't', 'i'),
                    (None, '1', 't', 1),
                    (None, 'j', 't', 'j'),
                    ('k', 'j', 'k', 'j'),
                    ('k', None, 'k', 2)):
                    img, time_axis, slice_axis = parse_fname_axes(
                        fname,
                        in_time,
                        in_sax)
                    assert_equal(time_axis, out_time)
                    assert_equal(slice_axis, out_sax)
                    del img
            # For some images, we can set the slice dimension. This becomes the
            # default if input slice_axis is None
            if hasattr(hdr, 'set_dim_info'):
                for ax_no in range(3):
                    nibabel_img.get_header().set_dim_info(slice=ax_no)
                    nib.save(nibabel_img, fname)
                    img, time_axis, slice_axis = parse_fname_axes(fname,
                                                                  None,
                                                                  None)
                    assert_equal(time_axis, 't')
                    assert_equal(slice_axis, 'slice')
                    del img
            # Images other than 4D don't get the slice axis default
            for new_arr in (arr[..., 0], arr[..., None]):
                new_nib = img_class(new_arr, None, hdr)
                nib.save(new_nib, fname)
                assert_raises(ValueError, parse_fname_axes, fname, None, None)
                # But you can still set slice axis
                img, time_axis, slice_axis = parse_fname_axes(fname, None, 'j')
                assert_equal(time_axis, 't')
                assert_equal(slice_axis, 'j')
    # Non-analyze image types don't get the slice default
    nib_data = pjoin(dirname(nib.__file__), 'tests', 'data')
    mnc_4d_fname = pjoin(nib_data, 'minc1_4d.mnc')
    if isfile(mnc_4d_fname):
        assert_raises(ValueError, parse_fname_axes, mnc_4d_fname, None, None)
        # At the moment we can't even load these guys
        try:
            img, time_axis, slice_axis = parse_fname_axes(
                mnc_4d_fname, None, 'j')
        except ValueError: # failed load
            raise SkipTest('Hoping for a time when we can use MINC')
        # But you can still set slice axis (if we can load them)
        assert_equal(time_axis, 't')
        assert_equal(slice_axis, 'j')
