""" Testing diagnostics.command module
"""

import os
import shutil
from os.path import dirname, isfile
from os.path import join as pjoin

import nibabel as nib
import numpy as np
import pytest
from nibabel import AnalyzeImage, Nifti1Image, Nifti1Pair, Spm2AnalyzeImage
from numpy.testing import assert_array_equal

from nipy import load_image
from nipy.io.nibcompat import get_header
from nipy.io.nifti_ref import NiftiError
from nipy.testing import funcfile
from nipy.testing.decorators import needs_mpl_agg

from ..commands import diagnose, parse_fname_axes, tsdiffana
from ..timediff import time_slice_diffs_image


def test_parse_fname_axes(in_tmp_path):
    # Test logic for setting time and slice axis defaults
    # We need real images for the tests because nipy will load them
    # For simplicity, we can create them
    shape = (4, 5, 6, 20)
    arr = np.arange(np.prod(shape), dtype=float).reshape(shape)
    zooms = (2., 3., 4., 2.1)
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
                assert time_axis == out_time
                assert slice_axis == out_sax
                del img
        # For some images, we can set the slice dimension. This becomes the
        # default if input slice_axis is None
        if hasattr(hdr, 'set_dim_info'):
            for ax_no in range(3):
                get_header(nibabel_img).set_dim_info(slice=ax_no)
                nib.save(nibabel_img, fname)
                img, time_axis, slice_axis = parse_fname_axes(fname,
                                                              None,
                                                              None)
                assert time_axis == 't'
                assert slice_axis == 'slice'
                del img
        # Images other than 4D don't get the slice axis default
        for new_arr in (arr[..., 0], arr[..., None]):
            new_nib = img_class(new_arr, None, hdr)
            nib.save(new_nib, fname)
            pytest.raises(ValueError, parse_fname_axes, fname, None, None)
            # But you can still set slice axis
            img, time_axis, slice_axis = parse_fname_axes(fname, None, 'j')
            assert time_axis == 't'
            assert slice_axis == 'j'
    # Non-analyze image types don't get the slice default
    nib_data = pjoin(dirname(nib.__file__), 'tests', 'data')
    mnc_4d_fname = pjoin(nib_data, 'minc1_4d.mnc')
    if isfile(mnc_4d_fname):
        pytest.raises(ValueError, parse_fname_axes, mnc_4d_fname, None, None)
        # At the moment we can't even load these guys
        try:
            img, time_axis, slice_axis = parse_fname_axes(
                mnc_4d_fname, None, 'j')
        except ValueError: # failed load
            pytest.skip('Hoping for a time when we can use MINC')
        # But you can still set slice axis (if we can load them)
        assert time_axis == 't'
        assert slice_axis == 'j'


class Args: pass


def check_axes(axes, img_shape, time_axis, slice_axis):
    # Check axes as expected for plot
    assert len(axes) == 4
    # First x axis is time point differences
    assert_array_equal(axes[0].xaxis.get_data_interval(),
                       [0, img_shape[time_axis]-2])
    # Last x axis is over slices
    assert_array_equal(axes[-1].xaxis.get_data_interval(),
                       [0, img_shape[slice_axis]-1])


@pytest.mark.filterwarnings("ignore:"
                            "Default `strict` currently False:"
                            "FutureWarning")
@needs_mpl_agg
def test_tsdiffana(in_tmp_path):
    # Test tsdiffana command
    args = Args()
    img = load_image(funcfile)
    args.filename = funcfile
    args.time_axis = None
    args.slice_axis = None
    args.write_results = False
    args.out_path = None
    args.out_fname_label = None
    args.out_file = 'test.png'
    check_axes(tsdiffana(args), img.shape, -1, -2)
    assert isfile('test.png')
    args.time_axis = 't'
    check_axes(tsdiffana(args), img.shape, -1, -2)
    args.time_axis = '3'
    check_axes(tsdiffana(args), img.shape, -1, -2)
    args.slice_axis = 'k'
    check_axes(tsdiffana(args), img.shape, -1, -2)
    args.slice_axis = '2'
    check_axes(tsdiffana(args), img.shape, -1, -2)
    args.time_axis = '0'
    check_axes(tsdiffana(args), img.shape, 0, -2)
    args.slice_axis = 't'
    check_axes(tsdiffana(args), img.shape, 0, -1)
    # Check absolute path works
    args.slice_axis = 'j'
    args.time_axis = 't'
    args.out_file = in_tmp_path / 'test_again.png'
    check_axes(tsdiffana(args), img.shape, -1, -3)
    # Check that --out-images incompatible with --out-file
    args.write_results=True
    pytest.raises(ValueError, tsdiffana, args)
    args.out_file=None
    # Copy the functional file to a temporary writeable directory
    os.mkdir('mydata')
    tmp_funcfile = in_tmp_path / 'mydata' / 'myfunc.nii.gz'
    shutil.copy(funcfile, tmp_funcfile)
    args.filename = tmp_funcfile
    # Check write-results generates expected images
    check_axes(tsdiffana(args), img.shape, -1, -3)
    assert isfile(pjoin('mydata', 'tsdiff_myfunc.png'))
    max_img = load_image(pjoin('mydata', 'dv2_max_myfunc.nii.gz'))
    assert max_img.shape == img.shape[:-1]
    mean_img = load_image(pjoin('mydata', 'dv2_max_myfunc.nii.gz'))
    assert mean_img.shape == img.shape[:-1]
    exp_results = time_slice_diffs_image(img, 't', 'j')
    saved_results = np.load(pjoin('mydata', 'tsdiff_myfunc.npz'))
    for key in ('volume_means', 'slice_mean_diff2'):
        assert_array_equal(exp_results[key], saved_results[key])
    # That we can change out-path
    os.mkdir('myresults')
    args.out_path = 'myresults'
    check_axes(tsdiffana(args), img.shape, -1, -3)
    assert isfile(pjoin('myresults', 'tsdiff_myfunc.png'))
    max_img = load_image(pjoin('myresults', 'dv2_max_myfunc.nii.gz'))
    assert max_img.shape == img.shape[:-1]
    # And out-fname-label
    args.out_fname_label = 'vr2'
    check_axes(tsdiffana(args), img.shape, -1, -3)
    assert isfile(pjoin('myresults', 'tsdiff_vr2.png'))
    max_img = load_image(pjoin('myresults', 'dv2_max_vr2.nii.gz'))
    assert max_img.shape == img.shape[:-1]
    del max_img, mean_img, saved_results


def check_diag_results(results, img_shape,
                       time_axis, slice_axis, ncomps,
                       out_path, froot, ext='.nii.gz'):

    S = img_shape[slice_axis]
    T = img_shape[time_axis]
    pca_shape = list(img_shape)
    pca_shape[time_axis] = ncomps
    assert results['pca'].shape == tuple(pca_shape)
    assert (results['pca_res']['basis_projections'].shape ==
                 tuple(pca_shape))
    # Roll pca axis last to test shape of output image
    ax_order = list(range(4))
    ax_order.remove(time_axis)
    ax_order.append(time_axis)
    rolled_shape = tuple(pca_shape[i] for i in ax_order)
    pca_img = load_image(pjoin(out_path, 'pca_' + froot + ext))
    assert pca_img.shape == rolled_shape
    for prefix in ('mean', 'min', 'max', 'std'):
        fname = pjoin(out_path, prefix + '_' + froot + ext)
        img = load_image(fname)
        assert img.shape == rolled_shape[:-1]
    vars = np.load(pjoin(out_path, 'vectors_components_' + froot + '.npz'))
    assert (set(vars) ==
                 {'basis_vectors', 'pcnt_var', 'volume_means',
                      'slice_mean_diff2'})
    assert vars['volume_means'].shape == (T,)
    assert vars['basis_vectors'].shape == (T, T-1)
    assert vars['slice_mean_diff2'].shape == (T-1, S)


@pytest.mark.filterwarnings("ignore:"
                            "Default `strict` currently False:"
                            "FutureWarning")
@needs_mpl_agg
def test_diagnose(in_tmp_path):
    args = Args()
    img = load_image(funcfile)
    # Copy the functional file to a temporary writeable directory
    os.mkdir('mydata')
    tmp_funcfile = in_tmp_path / 'mydata' / 'myfunc.nii.gz'
    shutil.copy(funcfile, tmp_funcfile)
    args.filename = tmp_funcfile
    args.time_axis = None
    args.slice_axis = None
    args.out_path = None
    args.out_fname_label = None
    args.ncomponents = 10
    res = diagnose(args)
    check_diag_results(res, img.shape, 3, 2, 10, 'mydata', 'myfunc')
    args.slice_axis = 'j'
    res = diagnose(args)
    check_diag_results(res, img.shape, 3, 1, 10, 'mydata', 'myfunc')
    # Time axis is not going to work because we'd have to use up one of the
    # needed spatial axes
    args.time_axis = 'i'
    pytest.raises(NiftiError, diagnose, args)
    args.time_axis = 't'
    # Check that output works
    os.mkdir('myresults')
    args.out_path = 'myresults'
    args.out_fname_label = 'myana'
    res = diagnose(args)
    check_diag_results(res, img.shape, 3, 1, 10, 'myresults', 'myana')
