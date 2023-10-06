# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test scripts

Run scripts and test output
"""

import os
from os.path import isfile
from os.path import join as pjoin
from unittest import skipIf

import numpy as np
import pytest
from nibabel.optpkg import optional_package
from numpy.testing import assert_almost_equal

from nipy import load_image, save_image
from nipy.core.api import rollimg
from nipy.testing.decorators import make_label_dec

from ..testing import funcfile

matplotlib, HAVE_MPL, _ = optional_package('matplotlib')
needs_mpl = skipIf(not HAVE_MPL, "Test needs matplotlib")
script_test = make_label_dec('script_test')

from .scriptrunner import ScriptRunner

runner = ScriptRunner(
    debug_print_var = 'NIPY_DEBUG_PRINT')
run_command = runner.run_command

@needs_mpl
@script_test
def test_nipy_diagnose(in_tmp_path):
    # Test nipy diagnose script
    fimg = load_image(funcfile)
    ncomps = 12
    cmd = ['nipy_diagnose', funcfile,
            f'--ncomponents={ncomps}',
            '--out-path=' + str(in_tmp_path)]
    run_command(cmd)
    for out_fname in ('components_functional.png',
                        'pcnt_var_functional.png',
                        'tsdiff_functional.png',
                        'vectors_components_functional.npz'):
        assert isfile(out_fname)
    for out_img in ('max_functional.nii.gz',
                    'mean_functional.nii.gz',
                    'min_functional.nii.gz',
                    'std_functional.nii.gz'):
        img = load_image(out_img)
        assert img.shape == fimg.shape[:-1]
        del img
    pca_img = load_image('pca_functional.nii.gz')
    assert pca_img.shape == fimg.shape[:-1] + (ncomps,)
    vecs_comps = np.load('vectors_components_functional.npz')
    vec_diff = vecs_comps['slice_mean_diff2'].copy()# just in case
    assert vec_diff.shape == (fimg.shape[-1]-1, fimg.shape[2])
    # Check we can pass in slice and time flags
    s0_img = rollimg(fimg, 'k')
    save_image(s0_img, 'slice0.nii')
    cmd = ['nipy_diagnose', 'slice0.nii',
            f'--ncomponents={ncomps}',
            '--out-path=' + str(in_tmp_path),
            '--time-axis=t',
            '--slice-axis=0']
    run_command(cmd)
    pca_img = load_image('pca_slice0.nii')
    assert pca_img.shape == s0_img.shape[:-1] + (ncomps,)
    vecs_comps = np.load('vectors_components_slice0.npz')
    assert_almost_equal(vecs_comps['slice_mean_diff2'], vec_diff)
    del pca_img, vecs_comps


@needs_mpl
@script_test
def test_nipy_tsdiffana(in_tmp_path):
    # Test nipy_tsdiffana script
    out_png = 'ts_out.png'
    # Quotes in case of space in arguments
    for i, extras in enumerate(([],
                                ['--time-axis=0'],
                                ['--slice-axis=0'],
                                ['--slice-axis=0', '--time-axis=1']
                                )):
        out_png = f'ts_out{i}.png'
        cmd = (['nipy_tsdiffana', funcfile] + extras +
                ['--out-file=' + out_png])
        run_command(cmd)
        assert isfile(out_png)
    # Out-file and write-results incompatible
    cmd = (['nipy_tsdiffana', funcfile, '--out-file=' + out_png,
            '--write-results'])
    pytest.raises(RuntimeError,
                  run_command,
                  cmd)
    # Can save images
    cmd_root = ['nipy_tsdiffana', funcfile]
    os.mkdir('myresults')
    run_command(cmd_root + ['--out-path=myresults', '--write-results'])
    assert isfile(pjoin('myresults', 'tsdiff_functional.png'))
    assert isfile(pjoin('myresults', 'tsdiff_functional.npz'))
    assert isfile(pjoin('myresults', 'dv2_max_functional.nii.gz'))
    assert isfile(pjoin('myresults', 'dv2_mean_functional.nii.gz'))
    run_command(cmd_root + ['--out-path=myresults', '--write-results',
                            '--out-fname-label=vr2'])
    assert isfile(pjoin('myresults', 'tsdiff_vr2.png'))
    assert isfile(pjoin('myresults', 'tsdiff_vr2.npz'))
    assert isfile(pjoin('myresults', 'dv2_max_vr2.nii.gz'))
    assert isfile(pjoin('myresults', 'dv2_mean_vr2.nii.gz'))


@script_test
def test_nipy_3_4d(in_tmp_path):
    # Test nipy_3dto4d and nipy_4dto3d
    fimg = load_image(funcfile)
    N = fimg.shape[-1]
    out_4d = 'func4d.nii'
    cmd = ['nipy_4dto3d', funcfile,  '--out-path=' + str(in_tmp_path)]
    run_command(cmd)
    imgs_3d = ['functional_%04d.nii' % i for i in range(N)]
    for iname in imgs_3d:
        assert isfile(iname)
    cmd = ['nipy_3dto4d'] + imgs_3d  + ['--out-4d=' + out_4d]
    run_command(cmd)
    fimg_back = load_image(out_4d)
    assert_almost_equal(fimg.get_fdata(), fimg_back.get_fdata())
    del fimg_back


@script_test
def test_nipy_4d_realign(in_tmp_path):
    # Test nipy_4d_realign script
    # Set matplotib agg backend
    with open("matplotlibrc", "w") as fobj:
        fobj.write("backend : agg")
    cmd = ['nipy_4d_realign', '2.0', funcfile,
           '--slice_dim',  '2',  '--slice_dir', '-1', '--save_path', '.']
    run_command(cmd)
