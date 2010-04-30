""" Testing fmristat hrf module

"""

import numpy as np

import nipy.modalities.fmri.hrf as hrf
from nipy.modalities.fmri.fmristat.hrf import (spectral_decomposition,
                                               taylor_approx)

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nipy.testing import parametric


@parametric
def test_spectral_decomposition():
    # mainly to test that the second sign follows the first
    spectral, approx = spectral_decomposition(hrf.glover)
    val_makers = [hrf.vectorize(def_func(hrf.t)) for def_func in spectral]
    t = np.linspace(-15,50,3251)
    vals = [val_maker(t) for val_maker in val_makers]
    ind = np.argmax(vals[1])
    yield assert_true(vals[0][ind] > 0)
    # test that we can get several components
    spectral, approx = spectral_decomposition(hrf.glover, ncomp=5)
    yield assert_equal(len(spectral), 5)
