# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test functions for models.utils
"""

import numpy as np

from .. import utils

from nose.tools import (assert_equal, assert_true, assert_raises)
from numpy.testing import (assert_array_equal, assert_array_almost_equal)


def test_StepFunction():
    x = np.arange(20)
    y = np.arange(20)
    f = utils.StepFunction(x, y)
    assert_array_almost_equal(f( np.array([[3.2,4.5],[24,-3.1]]) ), [[ 3, 4], [19, 0]])


def test_StepFunctionBadShape():
    x = np.arange(20)
    y = np.arange(21)
    assert_raises(ValueError, utils.StepFunction, x, y)
    x = np.zeros((2, 2))
    y = np.zeros((2, 2))
    assert_raises(ValueError, utils.StepFunction, x, y)
