""" Testing tsdiffana 

"""

import numpy as np

import nipy.algorithms.diagnostics.tsdiffana as tsd

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nipy import load_image
from nipy.testing import parametric, funcfile


@parametric
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
    expected['volds'] = np.mean(diffs2.reshape((vol_size, -1)), 0)
    expected['sliceds'] = np.zeros((n_tps-1, n_slices))
    for s in range(n_slices):
        v = diffs2[:,:,s,:].reshape((slice_size, -1))
        expected['sliceds'][:,s] = np.mean(v, 0)
    expected['diff_mean_vol'] = np.mean(diffs2, -1)
    max_diff_is = np.argmax(expected['sliceds'], 0)
    sdmv = np.empty(vol_shape)
    for si, dti in enumerate(max_diff_is):
        sdmv[:,:,si] = diffs2[:,:,si,dti]
    expected['slice_diff_max_vol'] = sdmv
    results = tsd.time_slice_diffs(ts)
    for key in expected:
        yield assert_array_almost_equal(results[key], expected[key])
    # tranposes, reset axes, get the same result
    results = tsd.time_slice_diffs(ts.T, 0, 1)
    results['diff_mean_vol'] = results['diff_mean_vol'].T
    results['slice_diff_max_vol'] = results['slice_diff_max_vol'].T
    for key in expected:
        yield assert_array_almost_equal(results[key], expected[key])
    ts_t = ts.transpose((1, 3, 0, 2))
    results = tsd.time_slice_diffs(ts_t, 1, -1)
    results['diff_mean_vol'] = results['diff_mean_vol'].transpose(
        ((1,0,2)))
    results['slice_diff_max_vol'] = results['slice_diff_max_vol'].transpose(
        ((1,0,2)))
    for key in expected:
        yield assert_array_almost_equal(results[key], expected[key])
    

@parametric
def test_timeslice_fmri():
    fimg = load_image(funcfile)
    results = tsd.time_slice_diffs(fimg)
    
