# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Testing tsdiffana

"""

from os.path import dirname, join as pjoin

import numpy as np

import scipy.io as sio

from .. import timediff as tsd

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nipy import load_image
from nipy.testing import funcfile

TEST_DATA_PATH = pjoin(dirname(__file__), 'data')


def test_time_slice_diffs():
    n_tps = 10
    n_slices = 4
    slice_shape = (2,3)
    slice_size = np.prod(slice_shape)
    vol_shape = slice_shape + (n_slices,)
    vol_size = np.prod(vol_shape)
    ts = np.random.normal(size=vol_shape + (n_tps,)) * 100 + 10
    expected = {}
    expected['volume_means'] = ts.reshape((vol_size, -1)).mean(0)
    # difference over time ^2
    diffs2 = np.diff(ts, axis=-1)**2
    expected['volume_mean_diff2'] = np.mean(
        diffs2.reshape((vol_size, -1)), 0)
    expected['slice_mean_diff2'] = np.zeros((n_tps-1, n_slices))
    for s in range(n_slices):
        v = diffs2[:,:,s,:].reshape((slice_size, -1))
        expected['slice_mean_diff2'][:,s] = np.mean(v, 0)
    expected['diff2_mean_vol'] = np.mean(diffs2, -1)
    max_diff_is = np.argmax(expected['slice_mean_diff2'], 0)
    sdmv = np.empty(vol_shape)
    for si, dti in enumerate(max_diff_is):
        sdmv[:,:,si] = diffs2[:,:,si,dti]
    expected['slice_diff2_max_vol'] = sdmv
    results = tsd.time_slice_diffs(ts)
    for key in expected:
        assert_array_almost_equal(results[key], expected[key])
    # tranposes, reset axes, get the same result
    results = tsd.time_slice_diffs(ts.T, 0, 1)
    results['diff2_mean_vol'] = results['diff2_mean_vol'].T
    results['slice_diff2_max_vol'] = results['slice_diff2_max_vol'].T
    for key in expected:
        assert_array_almost_equal(results[key], expected[key])
    ts_t = ts.transpose((1, 3, 0, 2))
    results = tsd.time_slice_diffs(ts_t, 1, -1)
    results['diff2_mean_vol'] = results['diff2_mean_vol'].transpose(
        ((1,0,2)))
    results['slice_diff2_max_vol'] = results['slice_diff2_max_vol'].transpose(
        ((1,0,2)))
    for key in expected:
        assert_array_almost_equal(results[key], expected[key])


def test_against_matlab_results():
    fimg = load_image(funcfile)
    results = tsd.time_slice_diffs(fimg.get_data())
    # struct as record only to avoid deprecation warning
    tsd_results = sio.loadmat(pjoin(TEST_DATA_PATH, 'tsdiff_results.mat'),
                              struct_as_record=True, squeeze_me=True)
    assert_array_almost_equal(results['volume_means'], tsd_results['g'])
    assert_array_almost_equal(results['volume_mean_diff2'],
                              tsd_results['imgdiff'])
    assert_array_almost_equal(results['slice_mean_diff2'],
                              tsd_results['slicediff'])
    # next tests are from saved, reloaded volumes at 16 bit integer
    # precision, so are not exact, but very close, given that the mean
    # of this array is around 3200
    assert_array_almost_equal(results['diff2_mean_vol'],
                              tsd_results['diff2_mean_vol'],
                              decimal=1)
    assert_array_almost_equal(results['slice_diff2_max_vol'],
                              tsd_results['slice_diff2_max_vol'],
                              decimal=1)
