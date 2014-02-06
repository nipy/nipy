# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Testing tsdiffana

"""

from os.path import dirname, join as pjoin

import numpy as np

import scipy.io as sio

from ....core.api import rollimg
from ....core.reference.coordinate_map import AxisError

from .. import timediff as tsd

from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_not_equal, assert_raises)

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


def test_time_slice_axes():
    # Test time and slice axes work as expected
    fimg = load_image(funcfile)
    # Put into array
    data = fimg.get_data()
    orig_results = tsd.time_slice_diffs(data)
    t0_data = np.rollaxis(data, 3)
    t0_results = tsd.time_slice_diffs(t0_data, 0)
    for key in ('volume_means', 'slice_mean_diff2'):
        assert_array_almost_equal(orig_results[key], t0_results[key])
    s0_data = np.rollaxis(data, 2)
    s0_results = tsd.time_slice_diffs(s0_data, slice_axis=0)
    for key in ('volume_means', 'slice_mean_diff2'):
        assert_array_almost_equal(orig_results[key], s0_results[key])
    # Incorrect slice axis
    bad_s0_results = tsd.time_slice_diffs(s0_data)
    assert_not_equal(orig_results['slice_mean_diff2'].shape,
                     bad_s0_results['slice_mean_diff2'].shape)
    # Slice axis equal to time axis - ValueError
    assert_raises(ValueError, tsd.time_slice_diffs, data, -1, -1)
    assert_raises(ValueError, tsd.time_slice_diffs, data, -1, 3)
    assert_raises(ValueError, tsd.time_slice_diffs, data, 1, 1)
    assert_raises(ValueError, tsd.time_slice_diffs, data, 1, -3)


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


def assert_arr_img_res(arr_res, img_res):
    for key in ('volume_mean_diff2',
                'slice_mean_diff2',
                'volume_means'):
        assert_array_equal(arr_res[key], img_res[key])
    for key in ('slice_diff2_max_vol', 'diff2_mean_vol'):
        assert_array_almost_equal(arr_res[key], img_res[key].get_data())


def test_tsd_image():
    # Test image version of time slice diff
    fimg = load_image(funcfile)
    data = fimg.get_data()
    tsda = tsd.time_slice_diffs
    tsdi = tsd.time_slice_diffs_image
    arr_results = tsda(data)
    # image routine insists on named slice axis, no default
    assert_raises(AxisError, tsdi, fimg)
    # Works when specifying slice axis as keyword argument
    img_results = tsdi(fimg, slice_axis='k')
    assert_arr_img_res(arr_results, img_results)
    ax_names = fimg.coordmap.function_domain.coord_names
    # Test against array version
    for time_ax in range(4):
        time_name = ax_names[time_ax]
        for slice_ax in range(4):
            slice_name = ax_names[slice_ax]
            if time_ax == slice_ax:
                assert_raises(ValueError, tsda, data, time_ax, slice_ax)
                assert_raises(ValueError, tsdi, fimg, time_ax, slice_ax)
                assert_raises(ValueError, tsdi, fimg, time_name, slice_ax)
                assert_raises(ValueError, tsdi, fimg, time_ax, slice_name)
                assert_raises(ValueError, tsdi, fimg, time_name, slice_name)
                continue
            arr_res = tsda(data, time_ax, slice_ax)
            assert_arr_img_res(arr_res, tsdi(fimg, time_ax, slice_ax))
            assert_arr_img_res(arr_res, tsdi(fimg, time_name, slice_ax))
            assert_arr_img_res(arr_res, tsdi(fimg, time_ax, slice_name))
            img_results = tsdi(fimg, time_name, slice_name)
            assert_arr_img_res(arr_res, img_results)
            exp_ax_names = tuple(n for n in ax_names if n != time_name)
            for key in ('slice_diff2_max_vol', 'diff2_mean_vol'):
                img = img_results[key]
                assert_equal(img.coordmap.function_domain.coord_names,
                             exp_ax_names)
    # Test defaults on rolled image
    fimg_rolled = rollimg(fimg, 't')
    # Still don't have a slice axis specified
    assert_raises(AxisError, tsdi, fimg_rolled)
    # Test default time axis
    assert_arr_img_res(arr_results, tsdi(fimg_rolled, slice_axis='k'))
    # Test axis named slice overrides default guess
    time_ax = -1
    for sa_no, sa_name in ((0, 'i'), (1, 'j'), (2, 'k')):
        fimg_renamed = fimg.renamed_axes(**{sa_name: 'slice'})
        arr_res = tsda(data, time_ax, sa_no)
        assert_arr_img_res(arr_res, tsdi(fimg_renamed, time_ax))
