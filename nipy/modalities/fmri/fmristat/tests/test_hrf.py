# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Testing fmristat hrf module

"""

import numpy as np

from ...utils import T, lambdify_t
from ... import hrf
from ..hrf import spectral_decomposition

from nose.tools import assert_true, assert_false, \
        assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_spectral_decomposition():
    # mainly to test that the second sign follows the first
    spectral, approx = spectral_decomposition(hrf.glover)
    val_makers = [lambdify_t(def_func(T)) for def_func in spectral]
    t = np.linspace(-15,50,3251)
    vals = [val_maker(t) for val_maker in val_makers]
    ind = np.argmax(vals[1])
    assert_true(vals[0][ind] > 0)
    # test that we can get several components
    spectral, approx = spectral_decomposition(hrf.glover, ncomp=5)
    assert_equal(len(spectral), 5)
