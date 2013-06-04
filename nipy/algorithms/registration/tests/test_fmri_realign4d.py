# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from nose.tools import assert_equal

from numpy.testing import assert_array_almost_equal, assert_array_equal
import numpy as np

from .... import load_image
from ....testing import funcfile
from ....fixes.nibabel import io_orientation
from ....core.image.image_spaces import (make_xyz_image,
                                        xyz_affine)

from ..groupwise_registration import Image4d, resample4d, SpaceTimeRealign
from ..affine import Rigid

im = load_image(funcfile)


def test_scanner_time():
    im4d = Image4d(im.get_data(), im.affine, tr=2.,
                   slice_times='ascending', interleaved=False)
    assert_equal(im4d.scanner_time(0, 0), 0.)
    assert_equal(im4d.scanner_time(0, im4d.tr), 1.)


def test_slice_info():
    im4d = Image4d(im.get_data(), im.affine, tr=2.,
                   slice_info=(1, -1))
    assert_equal(im4d.slice_axis, 1)
    assert_equal(im4d.slice_direction, -1)


def _test_image4d_init(nslices):
    data = np.zeros((3, 4, nslices, 6))
    aff = np.eye(4)
    tr = 2.0
    slice_times = (tr / float(nslices)) * np.arange(nslices)
    img4d = Image4d(data, aff, tr)
    assert_array_equal(img4d.slice_times, slice_times)
    img4d = Image4d(data, aff, tr, slice_times='ascending')
    assert_array_equal(img4d.slice_times, slice_times)
    img4d = Image4d(data, aff, tr, slice_times='descending')
    assert_array_equal(img4d.slice_times, slice_times[::-1])
    # test interleaved slice order
    interleaved_slice_times = (tr / float(nslices)) *\
        np.argsort(range(0, nslices, 2) + range(1, nslices, 2))
    img4d = Image4d(data, aff, tr, slice_times='ascending', interleaved=True)
    assert_array_equal(img4d.slice_times, interleaved_slice_times)
    img4d = Image4d(data, aff, tr, slice_times='descending', interleaved=True)
    assert_array_equal(img4d.slice_times, interleaved_slice_times[::-1])
    # can pass list
    img4d = Image4d(data, aff, tr, slice_times=list(slice_times))
    assert_array_equal(img4d.slice_times, slice_times)


def test_image4d_init_5slices():
    _test_image4d_init(5)


def test_image4d_init_6slices():
    _test_image4d_init(6)


def test_slice_timing():
    affine = np.eye(4)
    affine[0:3, 0:3] = im.affine[0:3, 0:3]
    im4d = Image4d(im.get_data(), affine, tr=2., slice_times=0.0)
    x = resample4d(im4d, [Rigid() for i in range(im.shape[3])])
    assert_array_almost_equal(im4d.get_data(), x)


def test_realign4d_no_time_interp():
    runs = [im, im]
    R = SpaceTimeRealign(runs, 1.0, slice_times=None)


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
    R1 = SpaceTimeRealign(runs, tr=2., slice_times='ascending')
    R1.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest')
    nslices = im.shape[slice_axis]
    slice_times = (2. / float(nslices)) * np.arange(nslices)
    R2 = SpaceTimeRealign(runs, tr=2., slice_times=slice_times)
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
    R = SpaceTimeRealign(runs, tr=2., slice_times='ascending')
    R.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest')
    cor_im, cor_im2 = R.resample()
    assert_array_equal(xyz_affine(cor_im2), aff)
