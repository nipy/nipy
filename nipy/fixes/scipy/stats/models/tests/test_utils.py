# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test functions for models.utils
"""

import numpy as np
import numpy.random as R

from .. import utils
from ..utils import pos_recipr, recipr0

from nose.tools import (assert_equal, assert_true, assert_raises)
from numpy.testing import (assert_array_equal, assert_array_almost_equal)


def test_pos_recipr():
    X = np.array([2,1,-1,0], dtype=np.int8)
    eX = np.array([0.5,1,0,0])
    Y = pos_recipr(X)
    yield assert_array_almost_equal, Y, eX
    yield assert_equal, Y.dtype.type, np.float64
    X2 = X.reshape((2,2))
    Y2 = pos_recipr(X2)
    yield assert_array_almost_equal, Y2, eX.reshape((2,2))
    # check that lists have arrived
    XL = [0, 1, -1]
    yield assert_array_almost_equal, pos_recipr(XL), [0, 1, 0]
    # scalars
    yield assert_equal, pos_recipr(-1), 0
    yield assert_equal, pos_recipr(0), 0
    yield assert_equal, pos_recipr(2), 0.5


def test_recipr0():
    X = np.array([[2,1],[-4,0]])
    Y = recipr0(X)
    yield assert_array_almost_equal, Y, np.array([[0.5,1],[-0.25,0]])
    # check that lists have arrived
    XL = [0, 1, -1]
    yield assert_array_almost_equal, recipr0(XL), [0, 1, -1]
    # scalars
    yield assert_equal, recipr0(-1), -1
    yield assert_equal, recipr0(0), 0
    yield assert_equal, recipr0(2), 0.5


def test_rank():
    X = R.standard_normal((40,10))
    assert_equal(utils.rank(X), 10)
    X[:,0] = X[:,1] + X[:,2]
    assert_equal(utils.rank(X), 9)


def test_fullrank():
    X = R.standard_normal((40,10))
    X[:,0] = X[:,1] + X[:,2]

    Y = utils.fullrank(X)
    assert_equal(Y.shape, (40,9))
    assert_equal(utils.rank(Y), 9)

    X[:,5] = X[:,3] + X[:,4]
    Y = utils.fullrank(X)
    assert_equal(Y.shape, (40,8))
    assert_equal(utils.rank(Y), 8)


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
