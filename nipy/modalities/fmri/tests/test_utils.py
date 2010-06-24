# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Testing fmri utils

"""

import numpy as np

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nipy.testing import parametric

from nipy.modalities.fmri.formula import Term
from nipy.modalities.fmri.utils import events
from sympy import Symbol, Function, DiracDelta
import nipy.modalities.fmri.hrf as mfhrf


@parametric
def test_events():
    # test events utility function
    h = Function('hrf')
    t = Term('t')
    evs = events([3,6,9])
    yield assert_equal(DiracDelta(-9 + t) + DiracDelta(-6 + t) +
                       DiracDelta(-3 + t), evs)
    evs = events([3,6,9], f=h)
    yield assert_equal(h(-3 + t) + h(-6 + t) + h(-9 + t), evs)
    # test no error for numpy int arrays
    onsets = np.array([30, 70, 100], dtype=np.int64)
    evs = events(onsets, f=mfhrf.glover)


