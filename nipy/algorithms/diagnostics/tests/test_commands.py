""" Testing diagnostics.command module
"""
from __future__ import absolute_import

import os
from os.path import dirname, join as pjoin, isfile
import shutil

import numpy as np

import nibabel as nib
from nibabel import AnalyzeImage, Spm2AnalyzeImage, Nifti1Pair, Nifti1Image
from nibabel.tmpdirs import InTemporaryDirectory

from nipy import load_image
from nipy.io.nifti_ref import NiftiError
from ..commands import parse_fname_axes, tsdiffana, diagnose
from ..timediff import time_slice_diffs_image

from numpy.testing import (assert_almost_equal, assert_array_equal)

from nose import SkipTest
from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)

from nipy.testing import funcfile
from nipy.testing.decorators import needs_mpl_agg


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


class Args(object): pass


def check_axes(axes, img_shape, time_axis, slice_axis):
    # Check axes as expected for plot
    assert_equal(len(axes), 4)
    # First x axis is time point differences
    assert_array_equal(axes[0].xaxis.get_data_interval(),
                       [0, img_shape[time_axis]-2])
    # Last x axis is over slices
    assert_array_equal(axes[-1].xaxis.get_data_interval(),
                       [0, img_shape[slice_axis]-1])


@needs_mpl_agg
def test_tsdiffana():
    # Test tsdiffana command
    args = Args()
    img = load_image(funcfile)
    with InTemporaryDirectory() as tmpdir:
        args.filename = funcfile
        args.time_axis = None
        args.slice_axis = None
        args.write_results = False
        args.out_path = None
        args.out_fname_label = None
        args.out_file = 'test.png'
        check_axes(tsdiffana(args), img.shape, -1, -2)
        assert_true(isfile('test.png'))
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
        args.out_file = pjoin(tmpdir, 'test_again.png')
        check_axes(tsdiffana(args), img.shape, -1, -3)
        # Check that --out-images incompatible with --out-file
        args.write_results=True
        assert_raises(ValueError, tsdiffana, args)
        args.out_file=None
        # Copy the functional file to a temporary writeable directory
        os.mkdir('mydata')
        tmp_funcfile = pjoin(tmpdir, 'mydata', 'myfunc.nii.gz')
        shutil.copy(funcfile, tmp_funcfile)
        args.filename = tmp_funcfile
        # Check write-results generates expected images
        check_axes(tsdiffana(args), img.shape, -1, -3)
        assert_true(isfile(pjoin('mydata', 'tsdiff_myfunc.png')))
        max_img = load_image(pjoin('mydata', 'dv2_max_myfunc.nii.gz'))
        assert_equal(max_img.shape, img.shape[:-1])
        mean_img = load_image(pjoin('mydata', 'dv2_max_myfunc.nii.gz'))
        assert_equal(mean_img.shape, img.shape[:-1])
        exp_results = time_slice_diffs_image(img, 't', 'j')
        saved_results = np.load(pjoin('mydata', 'tsdiff_myfunc.npz'))
        for key in ('volume_means', 'slice_mean_diff2'):
            assert_array_equal(exp_results[key], saved_results[key])
        # That we can change out-path
        os.mkdir('myresults')
        args.out_path = 'myresults'
        check_axes(tsdiffana(args), img.shape, -1, -3)
        assert_true(isfile(pjoin('myresults', 'tsdiff_myfunc.png')))
        max_img = load_image(pjoin('myresults', 'dv2_max_myfunc.nii.gz'))
        assert_equal(max_img.shape, img.shape[:-1])
        # And out-fname-label
        args.out_fname_label = 'vr2'
        check_axes(tsdiffana(args), img.shape, -1, -3)
        assert_true(isfile(pjoin('myresults', 'tsdiff_vr2.png')))
        max_img = load_image(pjoin('myresults', 'dv2_max_vr2.nii.gz'))
        assert_equal(max_img.shape, img.shape[:-1])
        del max_img, mean_img


def check_diag_results(results, img_shape,
                       time_axis, slice_axis, ncomps,
                       out_path, froot, ext='.nii.gz'):

    S = img_shape[slice_axis]
    T = img_shape[time_axis]
    pca_shape = list(img_shape)
    pca_shape[time_axis] = ncomps
    assert_equal(results['pca'].shape, tuple(pca_shape))
    assert_equal(results['pca_res']['basis_projections'].shape,
                 tuple(pca_shape))
    # Roll pca axis last to test shape of output image
    ax_order = list(range(4))
    ax_order.remove(time_axis)
    ax_order.append(time_axis)
    rolled_shape = tuple(pca_shape[i] for i in ax_order)
    pca_img = load_image(pjoin(out_path, 'pca_' + froot + ext))
    assert_equal(pca_img.shape, rolled_shape)
    for prefix in ('mean', 'min', 'max', 'std'):
        fname = pjoin(out_path, prefix + '_' + froot + ext)
        img = load_image(fname)
        assert_equal(img.shape, rolled_shape[:-1])
    vars = np.load(pjoin(out_path, 'vectors_components_' + froot + '.npz'))
    assert_equal(set(vars),
                 set(['basis_vectors', 'pcnt_var', 'volume_means',
                      'slice_mean_diff2']))
    assert_equal(vars['volume_means'].shape, (T,))
    assert_equal(vars['basis_vectors'].shape, (T, T-1))
    assert_equal(vars['slice_mean_diff2'].shape, (T-1, S))


@needs_mpl_agg
def test_diagnose():
    args = Args()
    img = load_image(funcfile)
    with InTemporaryDirectory() as tmpdir:
        # Copy the functional file to a temporary writeable directory
        os.mkdir('mydata')
        tmp_funcfile = pjoin(tmpdir, 'mydata', 'myfunc.nii.gz')
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
        assert_raises(NiftiError, diagnose, args)
        args.time_axis = 't'
        # Check that output works
        os.mkdir('myresults')
        args.out_path = 'myresults'
        args.out_fname_label = 'myana'
        res = diagnose(args)
        check_diag_results(res, img.shape, 3, 1, 10, 'myresults', 'myana')
