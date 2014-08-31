# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import warnings

from nose.tools import assert_equal

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal,
                           assert_raises)
import numpy as np

from .... import load_image
from ....testing import funcfile
from ....fixes.nibabel import io_orientation
from ....core.image.image_spaces import (make_xyz_image, xyz_affine)

from ..groupwise_registration import (Image4d, resample4d, FmriRealign4d,
                                      SpaceTimeRealign, SpaceRealign, Realign4d)
from ...slicetiming.timefuncs import st_43210, st_02413, st_42031
from ..affine import Rigid

im = load_image(funcfile)

def test_futurewarning():
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always')
        FmriRealign4d([im], tr=2., slice_order='ascending')
        assert_equal(warns.pop(0).category, FutureWarning)


def test_scanner_time():
    im4d = Image4d(im.get_data(), im.affine, tr=3.,
                   slice_times=(0, 1, 2))
    assert_equal(im4d.scanner_time(0, 0), 0.)
    assert_equal(im4d.scanner_time(0, im4d.tr), 1.)


def test_slice_info():
    im4d = Image4d(im.get_data(), im.affine, tr=3.,
                   slice_times=(0, 1, 2), slice_info=(2, -1))
    assert_equal(im4d.slice_axis, 2)
    assert_equal(im4d.slice_direction, -1)


def test_slice_timing():
    affine = np.eye(4)
    affine[0:3, 0:3] = im.affine[0:3, 0:3]
    im4d = Image4d(im.get_data(), affine, tr=2., slice_times=0.0)
    x = resample4d(im4d, [Rigid() for i in range(im.shape[3])])
    assert_array_almost_equal(im4d.get_data(), x)


def test_realign4d_no_time_interp():
    runs = [im, im]
    R = FmriRealign4d(runs, time_interp=False)
    assert R.slice_times == 0


def test_realign4d_ascending():
    runs = [im, im]
    R = FmriRealign4d(runs, tr=3, slice_order='ascending')
    assert_array_equal(R.slice_times, (0, 1, 2))
    assert R.tr == 3


def test_realign4d_descending():
    runs = [im, im]
    R = FmriRealign4d(runs, tr=3, slice_order='descending')
    assert_array_equal(R.slice_times, (2, 1, 0))
    assert R.tr == 3


def test_realign4d_ascending_interleaved():
    runs = [im, im]
    R = FmriRealign4d(runs, tr=3, slice_order='ascending', interleaved=True)
    assert_array_equal(R.slice_times, (0, 2, 1))
    assert R.tr == 3


def test_realign4d_descending_interleaved():
    runs = [im, im]
    R = FmriRealign4d(runs, tr=3, slice_order='descending', interleaved=True)
    assert_array_equal(R.slice_times, (1, 2, 0))
    assert R.tr == 3


def wrong_call(slice_times=None, slice_order=None, tr_slices=None,
               interleaved=None, time_interp=None):
    runs = [im, im]
    return FmriRealign4d(runs, tr=3, slice_times=slice_times,
                         slice_order=slice_order,
                         tr_slices=tr_slices,
                         interleaved=interleaved,
                         time_interp=time_interp)


def test_realign4d_incompatible_args():
    assert_raises(ValueError, wrong_call, slice_order=(0, 1, 2),
                  interleaved=False)
    assert_raises(ValueError, wrong_call, slice_times=(0, 1, 2),
                  slice_order='ascending')
    assert_raises(ValueError, wrong_call, slice_times=(0, 1, 2),
                  slice_order=(0, 1, 2))
    assert_raises(ValueError, wrong_call, slice_times=(0, 1, 2),
                  time_interp=True)
    assert_raises(ValueError, wrong_call, slice_times=(0, 1, 2),
                  time_interp=False)
    assert_raises(ValueError, wrong_call, time_interp=True)
    assert_raises(ValueError, wrong_call, slice_times=(0, 1, 2),
                  tr_slices=1)


def test_realign4d():
    """
    This tests whether realign4d yields the same results depending on
    whether the slice order is input explicitely or as
    slice_times='ascending'.

    Due to the very small size of the image used for testing (only 3
    slices), optimization is numerically unstable. It seems to make
    the default optimizer, namely scipy.fmin.fmin_ncg, adopt a random
    behavior. To work around the resulting inconsistency in results,
    we use nipy.optimize.fmin_steepest as the optimizer, although it's
    generally not recommended in practice.
    """
    runs = [im, im]
    orient = io_orientation(im.affine)
    slice_axis = int(np.where(orient[:, 0] == 2)[0])
    R1 = SpaceTimeRealign(runs, tr=2., slice_times='ascending',
                          slice_info=slice_axis)
    R1.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest')
    nslices = im.shape[slice_axis]
    slice_times = (2. / float(nslices)) * np.arange(nslices)
    R2 = SpaceTimeRealign(runs, tr=2., slice_times=slice_times,
                          slice_info=slice_axis)
    R2.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest')
    for r in range(2):
        for i in range(im.shape[3]):
            assert_array_almost_equal(R1._transforms[r][i].translation,
                                      R2._transforms[r][i].translation)
            assert_array_almost_equal(R1._transforms[r][i].rotation,
                                      R2._transforms[r][i].rotation)
    for i in range(im.shape[3]):
            assert_array_almost_equal(R1._mean_transforms[r].translation,
                                      R2._mean_transforms[r].translation)
            assert_array_almost_equal(R1._mean_transforms[r].rotation,
                                      R2._mean_transforms[r].rotation)


def test_realign4d_runs_with_different_affines():
    aff = xyz_affine(im)
    aff2 = aff.copy()
    aff2[0:3, 3] += 5
    im2 = make_xyz_image(im.get_data(), aff2, 'scanner')
    runs = [im, im2]
    R = SpaceTimeRealign(runs, tr=2., slice_times='ascending', slice_info=2)
    R.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest')
    cor_im, cor_im2 = R.resample()
    assert_array_equal(xyz_affine(cor_im2), aff)


def test_realign4d_params():
    # Some tests for input parameters to realign4d
    R = Realign4d(im, 3, [0, 1, 2], None) # No slice_info - OK
    assert_equal(R.tr, 3)
    # TR cannot be None for set slice times
    assert_raises(ValueError, Realign4d, im, None, [0, 1, 2], None)
    # TR can be None if slice times are None
    R = Realign4d(im, None, None)
    assert_equal(R.tr, 1)


def test_spacetimerealign_params():
    runs = [im, im]
    for slice_times in ('descending', '43210', st_43210, [2, 1, 0]):
        R = SpaceTimeRealign(runs, tr=3, slice_times=slice_times, slice_info=2)
        assert_array_equal(R.slice_times, (2, 1, 0))
        assert_equal(R.tr, 3)
    for slice_times in ('asc_alt_2', '02413', st_02413, [0, 2, 1]):
        R = SpaceTimeRealign(runs, tr=3, slice_times=slice_times, slice_info=2)
        assert_array_equal(R.slice_times, (0, 2, 1))
        assert_equal(R.tr, 3)
    for slice_times in ('desc_alt_2', '42031', st_42031, [1, 2, 0]):
        R = SpaceTimeRealign(runs, tr=3, slice_times=slice_times, slice_info=2)
        assert_array_equal(R.slice_times, (1, 2, 0))
        assert_equal(R.tr, 3)
    # Check changing axis
    R = SpaceTimeRealign(runs, tr=21, slice_times='ascending', slice_info=1)
    assert_array_equal(R.slice_times, np.arange(21))
    # Check slice_times and slice_info and TR required
    R = SpaceTimeRealign(runs, 3, 'ascending', 2) # OK
    assert_raises(ValueError, SpaceTimeRealign, runs, 3, None, 2)
    assert_raises(ValueError, SpaceTimeRealign, runs, 3, 'ascending', None)
    assert_raises(ValueError, SpaceTimeRealign, runs, None, [0, 1, 2], 2)
    # Test when TR and nslices are not the same
    R1 = SpaceTimeRealign(runs, tr=2., slice_times='ascending', slice_info=2)
    assert_array_equal(R1.slice_times, np.arange(3) / 3. * 2.)
    # Smoke test run
    R1.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest')


def test_spacerealign():
    # Check space-only realigner
    runs = [im, im]
    R = SpaceRealign(runs)
    assert_equal(R.tr, 1)
    assert_equal(R.slice_times, 0.)
    # Smoke test run
    R.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest')


def test_single_image():
    # Check we can use a single image as argument
    R = SpaceTimeRealign(im, tr=3, slice_times='ascending', slice_info=2)
    R.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest')
    R = SpaceRealign(im)
    R.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest')
    R = Realign4d(im, 3, [0, 1, 2], (2, 1))
    R.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest')
