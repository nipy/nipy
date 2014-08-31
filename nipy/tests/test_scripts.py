# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test scripts

If we appear to be running from the development directory, use the scripts in
the top-level folder ``scripts``.  Otherwise try and get the scripts from the
path
"""
from __future__ import with_statement

import sys
import os
from os.path import dirname, join as pjoin, isfile, isdir, abspath, realpath, split

from subprocess import Popen, PIPE

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

# Need shell to get path to correct executables
USE_SHELL = True

DEBUG_PRINT = os.environ.get('NIPY_DEBUG_PRINT', False)

def local_script_dir():
    # Check for presence of scripts in development directory.  ``realpath``
    # checks for the situation where the development directory has been linked
    # into the path.
    below_nipy_dir = realpath(pjoin(dirname(__file__), '..', '..'))
    devel_script_dir = pjoin(below_nipy_dir, 'scripts')
    if isfile(pjoin(below_nipy_dir, 'setup.py')) and isdir(devel_script_dir):
        return devel_script_dir
    return None

LOCAL_SCRIPT_DIR = local_script_dir()

def run_command(cmd):
    if not LOCAL_SCRIPT_DIR is None:
        # Windows can't run script files without extensions natively so we need
        # to run local scripts (no extensions) via the Python interpreter.  On
        # Unix, we might have the wrong incantation for the Python interpreter
        # in the hash bang first line in the source file.  So, either way, run
        # the script through the Python interpreter
        cmd = "%s %s" % (sys.executable, pjoin(LOCAL_SCRIPT_DIR, cmd))
    if DEBUG_PRINT:
        print("Running command '%s'" % cmd)
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=USE_SHELL)
    stdout, stderr = proc.communicate()
    if proc.poll() == None:
        proc.terminate()
    if proc.returncode != 0:
        raise RuntimeError('Command "%s" failed with stdout\n%s\nstderr\n%s\n'
                           % (cmd, stdout, stderr))
    return proc.returncode


@needs_mpl
@script_test
def test_nipy_diagnose():
    # Test nipy diagnose script
    fimg = load_image(funcfile)
    ncomps = 12
    with InTemporaryDirectory() as tmpdir:
        # Need to quote out path in case it has spaces
        cmd = 'nipy_diagnose "%s" --ncomponents=%d --out-path="%s"' % (
            funcfile, ncomps, tmpdir)
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
        cmd = ('nipy_diagnose slice0.nii --ncomponents=%d --out-path="%s" '
               '--time-axis=t --slice-axis=0' % (ncomps, tmpdir))
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
    cmd_template = 'nipy_tsdiffana "{0}" --out-file="{1}"'
    with InTemporaryDirectory():
        for i, cmd in enumerate((cmd_template,
                                 cmd_template + ' --time-axis=0',
                                 cmd_template + ' --slice-axis=0',
                                 cmd_template + ' --slice-axis=0 ' +
                                 '--time-axis=1')):
            out_png = 'ts_out{0}.png'.format(i)
            run_command(cmd.format(funcfile, out_png))
            assert_true(isfile(out_png))
    # Out-file and write-results incompatible
    assert_raises(RuntimeError,
                  run_command,
                  cmd_template.format(funcfile, out_png) + '--write-results')
    # Can save images
    cmd_root = 'nipy_tsdiffana "{0}" '.format(funcfile)
    with InTemporaryDirectory():
        os.mkdir('myresults')
        run_command(cmd_root + '--out-path=myresults --write-results')
        assert_true(isfile(pjoin('myresults', 'tsdiff_functional.png')))
        assert_true(isfile(pjoin('myresults', 'tsdiff_functional.npz')))
        assert_true(isfile(pjoin('myresults', 'dv2_max_functional.nii.gz')))
        assert_true(isfile(pjoin('myresults', 'dv2_mean_functional.nii.gz')))
        run_command(cmd_root + '--out-path=myresults --write-results '
                    '--out-fname-label=vr2')
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
        # Quotes in case of space in arguments
        cmd = 'nipy_4dto3d "%s" --out-path="%s"' % (funcfile, tmpdir)
        run_command(cmd)
        imgs_3d = ['functional_%04d.nii' % i for i in range(N)]
        for iname in imgs_3d:
            assert_true(isfile(iname))
        cmd = 'nipy_3dto4d "%s" --out-4d="%s"' % ('" "'.join(imgs_3d), out_4d)
        run_command(cmd)
        fimg_back = load_image(out_4d)
        assert_almost_equal(fimg.get_data(), fimg_back.get_data())
        del fimg_back


@script_test
def test_nipy_4d_realign():
    # Test nipy_4d_realign script
    with InTemporaryDirectory() as tmpdir:
        # Set matplotib agg backend
        with open("matplotlibrc", "wt") as fobj:
            fobj.write("backend : agg")
        # Quotes in case of space in input filename
        cmd = ('nipy_4d_realign 2.0 "{0}" --slice_dim 2 --slice_dir -1 '
               '--save_path .'.format(funcfile))
        run_command(cmd)
