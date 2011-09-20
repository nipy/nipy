# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from nose.tools import assert_equal

from numpy.testing import assert_array_almost_equal
import numpy as np

from .... import load_image
from ....testing import funcfile

from ..groupwise_registration import Image4d, resample4d, FmriRealign4d
from ..affine import Rigid

im = load_image(funcfile)


def test_scanner_time():
    im4d = Image4d(im.get_data(), im.affine, tr=2.,
                   slice_order='ascending', interleaved=False)
    assert_equal(im4d.scanner_time(0, 0), 0.)
    assert_equal(im4d.scanner_time(0, im4d.tr), 1.)
    assert_equal(im4d.scanner_time(1, im4d.tr_slices), 0.)


def test_slice_timing():
    affine = np.eye(4)
    affine[0:3, 0:3] = im.affine[0:3, 0:3]
    im4d = Image4d(im.get_data(), affine, tr=2., tr_slices=0.0)
    x = resample4d(im4d, [Rigid() for i in range(im.shape[3])])
    assert_array_almost_equal(im4d.get_data(), x)


def test_realign4d_single_run():
    R = FmriRealign4d(im, tr=2., slice_order='ascending', interleaved=False)
    R.estimate(refscan=None, loops=(1, 0), between_loops=(1, 0))


def test_realign4d_two_runs():
    R = FmriRealign4d([im, im], tr=2., slice_order='ascending',
                      interleaved=False)
    R.estimate(refscan=None, loops=(1, 0), between_loops=(1, 0))
