# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Testing diagnostic screen
"""

import os
from os.path import join as pjoin

from warnings import catch_warnings, simplefilter

import numpy as np

import nipy as ni
from nipy.core.api import rollimg
from ..screens import screen, write_screen_res
from ..timediff import time_slice_diffs
from ...utils.pca import pca
from ...utils.tests.test_pca import res2pos1

from nibabel.tmpdirs import InTemporaryDirectory

from nose.tools import (assert_true, assert_false, assert_equal, assert_raises)

from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal, decorators)

from nipy.testing import funcfile
from nipy.testing.decorators import needs_mpl_agg


def _check_pca(res, pca_res):
    # Standardize output vector signs
    screen_pca_res = res2pos1(res['pca_res'])
    for key in pca_res:
        assert_almost_equal(pca_res[key], screen_pca_res[key])


def _check_ts(res, data, time_axis, slice_axis):
    ts_res = time_slice_diffs(data, time_axis, slice_axis)
    for key in ts_res:
        assert_array_equal(ts_res[key], res['ts_res'][key])


def test_screen():
    img = ni.load_image(funcfile)
    # rename third axis to slice to match default of screen
    # This avoids warnings about future change in default; see the tests for
    # slice axis below
    img = img.renamed_axes(k='slice')
    res = screen(img)
    assert_equal(res['mean'].ndim, 3)
    assert_equal(res['pca'].ndim, 4)
    assert_equal(sorted(res.keys()),
                 ['max', 'mean', 'min',
                  'pca', 'pca_res',
                  'std', 'ts_res'])
    data = img.get_data()
    # Check summary images
    assert_array_equal(np.max(data, axis=-1), res['max'].get_data())
    assert_array_equal(np.mean(data, axis=-1), res['mean'].get_data())
    assert_array_equal(np.min(data, axis=-1), res['min'].get_data())
    assert_array_equal(np.std(data, axis=-1), res['std'].get_data())
    pca_res = pca(data, axis=-1, standardize=False, ncomp=10)
    # On windows, there seems to be some randomness in the PCA output vector
    # signs; this routine sets the basis vectors to have first value positive,
    # and therefore standardizes the signs
    pca_res = res2pos1(pca_res)
    _check_pca(res, pca_res)
    _check_ts(res, data, 3, 2)
    # Test that screens accepts and uses time axis
    data_mean = data.mean(axis=-1)
    res = screen(img, time_axis='t')
    assert_array_equal(data_mean, res['mean'].get_data())
    _check_pca(res, pca_res)
    _check_ts(res, data, 3, 2)
    res = screen(img, time_axis=-1)
    assert_array_equal(data_mean, res['mean'].get_data())
    _check_pca(res, pca_res)
    _check_ts(res, data, 3, 2)
    t0_img = rollimg(img, 't')
    t0_data = np.rollaxis(data, -1)
    res = screen(t0_img, time_axis='t')
    t0_pca_res = pca(t0_data, axis=0, standardize=False, ncomp=10)
    t0_pca_res = res2pos1(t0_pca_res)
    assert_array_equal(data_mean, res['mean'].get_data())
    _check_pca(res, t0_pca_res)
    _check_ts(res, t0_data, 0, 3)
    res = screen(t0_img, time_axis=0)
    assert_array_equal(data_mean, res['mean'].get_data())
    _check_pca(res, t0_pca_res)
    _check_ts(res, t0_data, 0, 3)
    # Check screens uses slice axis
    s0_img = rollimg(img, 2, 0)
    s0_data = np.rollaxis(data, 2, 0)
    res = screen(s0_img, slice_axis=0)
    _check_ts(res, s0_data, 3, 0)
    # And defaults to named slice axis
    # First re-show that when we don't specify, we get the default
    res = screen(img)
    _check_ts(res, data, 3, 2)
    assert_raises(AssertionError, _check_ts, res, data, 3, 0)
    # Then specify, get non-default
    slicey_img = img.renamed_axes(slice='k', i='slice')
    res = screen(slicey_img)
    _check_ts(res, data, 3, 0)
    assert_raises(AssertionError, _check_ts, res, data, 3, 2)


def pca_pos(data4d):
    """ Flips signs equal over volume for PCA

    Needed because Windows appears to generate random signs for PCA components
    across PCA runs on the same data.
    """
    signs = np.sign(data4d[0, 0, 0, :])
    return data4d * signs


def test_screen_slice_axis():
    img = ni.load_image(funcfile)
    # Default screen raises a FutureWarning because of the default slice_axis
    exp_res = screen(img, slice_axis='k')
    with catch_warnings():
        simplefilter('error')
        assert_raises(FutureWarning, screen, img)
        assert_raises(FutureWarning, screen, img, slice_axis=None)
        explicit_img = img.renamed_axes(k='slice')
        # Now the analysis works without warning
        res = screen(explicit_img)
        # And is the expected analysis
        # Very oddly on scipy 0.9 32 bit - at least - results differ between
        # runs, so we need assert_almost_equal
        assert_almost_equal(pca_pos(res['pca'].get_data()),
                            pca_pos(exp_res['pca'].get_data()))
        assert_array_equal(res['ts_res']['slice_mean_diff2'],
                           exp_res['ts_res']['slice_mean_diff2'])
        # Turn off warnings, also get expected analysis
        simplefilter('ignore')
        res = screen(img)
        assert_array_equal(res['ts_res']['slice_mean_diff2'],
                           exp_res['ts_res']['slice_mean_diff2'])


@needs_mpl_agg
def test_write_screen_res():
    img = ni.load_image(funcfile)
    with InTemporaryDirectory():
        res = screen(img)
        os.mkdir('myresults')
        write_screen_res(res, 'myresults', 'myana')
        pca_img = ni.load_image(pjoin('myresults', 'pca_myana.nii'))
        assert_equal(pca_img.shape, img.shape[:-1] + (10,))
        # Make sure we get the same output image even from rolled image
        # Do fancy roll to put time axis first, and slice axis last. This does
        # a stress test on the axis ordering, but also makes sure that we are
        # getting the number of components from the right place.  If we were
        # getting the number of components from the length of the last axis,
        # instead of the length of the 't' axis in the returned pca image, this
        # would be wrong (=21) which would also be more than the number of
        # basis vectors (19) so raise an error
        rimg = img.reordered_axes([3, 2, 0, 1])
        os.mkdir('rmyresults')
        rres = screen(rimg)
        write_screen_res(rres, 'rmyresults', 'myana')
        rpca_img = ni.load_image(pjoin('rmyresults', 'pca_myana.nii'))
        assert_equal(rpca_img.shape, img.shape[:-1] + (10,))
        del pca_img, rpca_img
