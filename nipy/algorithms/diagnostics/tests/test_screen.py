# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Testing diagnostic screen
"""

import numpy as np

import nipy as ni
from nipy.core.api import rollimg
from ..screens import screen
from ..timediff import time_slice_diffs
from ...utils.pca import pca
from ...utils.tests.test_pca import res2pos1

from nose.tools import (assert_true, assert_false, assert_equal, assert_raises)

from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)

from nipy.testing import funcfile, anatfile


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
    slicey_img = img.renamed_axes(i='slice')
    res = screen(slicey_img)
    _check_ts(res, data, 3, 0)
    assert_raises(AssertionError, _check_ts, res, data, 3, 2)
