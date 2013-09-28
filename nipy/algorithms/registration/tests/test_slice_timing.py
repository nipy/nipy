from __future__ import division

import numpy as np

from scipy.ndimage import gaussian_filter, gaussian_filter1d

from nose.tools import assert_true, assert_false
from numpy.testing import assert_almost_equal, assert_array_equal

from nipy.core.api import Image, vox2scanner
from ..groupwise_registration import SpaceTimeRealign


def check_stc(true_signal, corrected_signal, ref_slice=0,
              rtol=1e-5, atol=1e-5):
    n_slices = true_signal.shape[2]
    # The reference slice should be more or less perfect
    assert_almost_equal(
        corrected_signal[..., ref_slice, :],
        true_signal[..., ref_slice, :])
    # The other slices should be more or less right
    for sno in range(n_slices):
        if sno == ref_slice:
            continue # We checked this one
        arr0 = true_signal[..., sno, 1:-1]
        arr1 = corrected_signal[..., sno, 1:-1]
        # Intermediate test matrices for debugging
        abs_diff = np.abs(arr0 - arr1)
        rel_diff = np.abs((arr0 / arr1) - 1)
        abs_fails = abs_diff > atol
        rel_fails = rel_diff > rtol
        fails = abs_fails & rel_fails
        abs_only = abs_diff[fails]
        rel_only = rel_diff[fails]
        assert_true(np.allclose(arr0, arr1, rtol=rtol, atol=atol))


def test_slice_time_correction():
    # Make smooth time course at slice resolution
    TR = 2.
    n_vols = 25
    n_slices = 10
    # Create single volume
    shape_3d = (20, 30, n_slices)
    spatial_sigma = 4
    time_sigma = n_slices * 5 # time sigma in TRs
    one_vol = np.random.normal(100, 25, size=shape_3d)
    gaussian_filter(one_vol, spatial_sigma, output=one_vol)
    # Add smoothed time courses.  Time courses are at time resolution of one
    # slice time.  So, there are n_slices time points per TR.
    n_vol_slices = n_slices * n_vols
    time_courses = np.random.normal(0, 15, size=shape_3d + (n_vol_slices,))
    gaussian_filter1d(time_courses, time_sigma, output=time_courses)
    big_data = one_vol[..., None] + time_courses
    # Can the first time point be approximated from the later ones?
    first_signal = big_data[..., 0:n_vol_slices:n_slices]
    for name, time_to_slice in (
        ('ascending', list(range(n_slices))),
        ('descending', list(range(n_slices)[::-1])),
        ('asc_alt_2', (list(range(0, n_slices, 2)) +
                       list(range(1, n_slices, 2)))),
        ('desc_alt_2', (list(range(0, n_slices, 2)) +
                        list(range(1, n_slices, 2)))[::-1])
    ):
        slice_to_time = np.argsort(time_to_slice)
        acquired_signal = np.zeros_like(first_signal)
        for space_sno, time_sno in enumerate(slice_to_time):
            acquired_signal[..., space_sno, :] = \
                big_data[..., space_sno, time_sno:n_vol_slices:n_slices]
        # do STC - minimizer will fail
        acquired_image = Image(acquired_signal, vox2scanner(np.eye(5)))
        stc = SpaceTimeRealign(acquired_image, TR, name, 2)
        stc.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest')
        # Check no motion estimated
        assert_array_equal([t.param for t in stc._transforms[0]], 0)
        corrected = stc.resample()[0].get_data()
        # check we approximate first time slice with correction
        assert_false(np.allclose(acquired_signal, corrected, rtol=1e-3,
                                 atol=0.1))
        check_stc(first_signal, corrected, ref_slice=slice_to_time[0],
                  rtol=5e-4, atol=1e-6)
