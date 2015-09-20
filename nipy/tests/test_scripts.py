# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test scripts

Run scripts and test output
"""
from __future__ import absolute_import

import os
from os.path import join as pjoin, isfile

import numpy as np

from nibabel.tmpdirs import InTemporaryDirectory

from nipy import load_image, save_image
from nipy.core.api import rollimg

from nose.tools import assert_true, assert_false, assert_equal, assert_raises

from ..testing import funcfile
from numpy.testing import decorators, assert_almost_equal

from nipy.testing.decorators import make_label_dec

from nibabel.optpkg import optional_package

matplotlib, HAVE_MPL, _ = optional_package('matplotlib')
needs_mpl = decorators.skipif(not HAVE_MPL, "Test needs matplotlib")
script_test = make_label_dec('script_test')

from .scriptrunner import ScriptRunner

runner = ScriptRunner(
    debug_print_var = 'NIPY_DEBUG_PRINT')
run_command = runner.run_command

@needs_mpl
@script_test
def test_nipy_diagnose():
    # Test nipy diagnose script
    fimg = load_image(funcfile)
    ncomps = 12
    with InTemporaryDirectory() as tmpdir:
        cmd = ['nipy_diagnose', funcfile,
               '--ncomponents={0}'.format(ncomps),
               '--out-path=' + tmpdir]
        run_command(cmd)
        for out_fname in ('components_functional.png',
                          'pcnt_var_functional.png',
                          'tsdiff_functional.png',
                          'vectors_components_functional.npz'):
            assert_true(isfile(out_fname))
        for out_img in ('max_functional.nii.gz',
                        'mean_functional.nii.gz',
                        'min_functional.nii.gz',
                        'std_functional.nii.gz'):
            img = load_image(out_img)
            assert_equal(img.shape, fimg.shape[:-1])
            del img
        pca_img = load_image('pca_functional.nii.gz')
        assert_equal(pca_img.shape, fimg.shape[:-1] + (ncomps,))
        vecs_comps = np.load('vectors_components_functional.npz')
        vec_diff = vecs_comps['slice_mean_diff2'].copy()# just in case
        assert_equal(vec_diff.shape, (fimg.shape[-1]-1, fimg.shape[2]))
        del pca_img, vecs_comps
    with InTemporaryDirectory() as tmpdir:
        # Check we can pass in slice and time flags
        s0_img = rollimg(fimg, 'k')
        save_image(s0_img, 'slice0.nii')
        cmd = ['nipy_diagnose', 'slice0.nii',
               '--ncomponents={0}'.format(ncomps),
               '--out-path=' + tmpdir,
               '--time-axis=t',
               '--slice-axis=0']
        run_command(cmd)
        pca_img = load_image('pca_slice0.nii')
        assert_equal(pca_img.shape, s0_img.shape[:-1] + (ncomps,))
        vecs_comps = np.load('vectors_components_slice0.npz')
        assert_almost_equal(vecs_comps['slice_mean_diff2'], vec_diff)
        del pca_img, vecs_comps


@needs_mpl
@script_test
def test_nipy_tsdiffana():
    # Test nipy_tsdiffana script
    out_png = 'ts_out.png'
    # Quotes in case of space in arguments
    with InTemporaryDirectory():
        for i, extras in enumerate(([],
                                    ['--time-axis=0'],
                                    ['--slice-axis=0'],
                                    ['--slice-axis=0', '--time-axis=1']
                                   )):
            out_png = 'ts_out{0}.png'.format(i)
            cmd = (['nipy_tsdiffana', funcfile] + extras +
                   ['--out-file=' + out_png])
            run_command(cmd)
            assert_true(isfile(out_png))
    # Out-file and write-results incompatible
    cmd = (['nipy_tsdiffana', funcfile, '--out-file=' + out_png,
            '--write-results'])
    assert_raises(RuntimeError,
                  run_command,
                  cmd)
    # Can save images
    cmd_root = ['nipy_tsdiffana', funcfile]
    with InTemporaryDirectory():
        os.mkdir('myresults')
        run_command(cmd_root + ['--out-path=myresults', '--write-results'])
        assert_true(isfile(pjoin('myresults', 'tsdiff_functional.png')))
        assert_true(isfile(pjoin('myresults', 'tsdiff_functional.npz')))
        assert_true(isfile(pjoin('myresults', 'dv2_max_functional.nii.gz')))
        assert_true(isfile(pjoin('myresults', 'dv2_mean_functional.nii.gz')))
        run_command(cmd_root + ['--out-path=myresults', '--write-results',
                                '--out-fname-label=vr2'])
        assert_true(isfile(pjoin('myresults', 'tsdiff_vr2.png')))
        assert_true(isfile(pjoin('myresults', 'tsdiff_vr2.npz')))
        assert_true(isfile(pjoin('myresults', 'dv2_max_vr2.nii.gz')))
        assert_true(isfile(pjoin('myresults', 'dv2_mean_vr2.nii.gz')))


@script_test
def test_nipy_3_4d():
    # Test nipy_3dto4d and nipy_4dto3d
    fimg = load_image(funcfile)
    N = fimg.shape[-1]
    out_4d = 'func4d.nii'
    with InTemporaryDirectory() as tmpdir:
        cmd = ['nipy_4dto3d', funcfile,  '--out-path=' + tmpdir]
        run_command(cmd)
        imgs_3d = ['functional_%04d.nii' % i for i in range(N)]
        for iname in imgs_3d:
            assert_true(isfile(iname))
        cmd = ['nipy_3dto4d'] + imgs_3d  + ['--out-4d=' + out_4d]
        run_command(cmd)
        fimg_back = load_image(out_4d)
        assert_almost_equal(fimg.get_data(), fimg_back.get_data())
        del fimg_back


@script_test
def test_nipy_4d_realign():
    # Test nipy_4d_realign script
    with InTemporaryDirectory():
        # Set matplotib agg backend
        with open("matplotlibrc", "wt") as fobj:
            fobj.write("backend : agg")
        cmd = ['nipy_4d_realign', '2.0', funcfile,
               '--slice_dim',  '2',  '--slice_dir', '-1', '--save_path', '.']
        run_command(cmd)
