# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test functions for utils.matrices """
from __future__ import print_function, absolute_import

import numpy as np

import scipy.linalg as spl

from ..matrices import (matrix_rank, full_rank, pos_recipr, recipr0)

from nose.tools import (assert_true, assert_equal, assert_false,
                        assert_raises)

from numpy.testing import (assert_almost_equal, assert_array_almost_equal)


def test_matrix_rank():
    # Full rank matrix
    assert_equal(4, matrix_rank(np.eye(4)))
    I=np.eye(4); I[-1,-1] = 0. # rank deficient matrix
    assert_equal(matrix_rank(I), 3)
    # All zeros - zero rank
    assert_equal(matrix_rank(np.zeros((4,4))), 0)
    # 1 dimension - rank 1 unless all 0
    assert_equal(matrix_rank(np.ones((4,))), 1)
    assert_equal(matrix_rank(np.zeros((4,))), 0)
    # accepts array-like
    assert_equal(matrix_rank([1]), 1)
    # Make rank deficient matrix
    rng = np.random.RandomState(20120613)
    X = rng.normal(size=(40, 10))
    X[:, 0] = X[:, 1] + X[:, 2]
    S = spl.svd(X, compute_uv=False)
    eps = np.finfo(X.dtype).eps
    assert_equal(matrix_rank(X, tol=0), 10)
    assert_equal(matrix_rank(X, tol=S.min() - eps), 10)
    assert_equal(matrix_rank(X, tol=S.min() + eps), 9)


def test_full_rank():
    rng = np.random.RandomState(20110831)
    X = rng.standard_normal((40,5))
    # A quick rank check
    assert_equal(matrix_rank(X), 5)
    X[:,0] = X[:,1] + X[:,2]
    assert_equal(matrix_rank(X), 4)
    Y1 = full_rank(X)
    assert_equal(Y1.shape, (40,4))
    Y2 = full_rank(X, r=3)
    assert_equal(Y2.shape, (40,3))
    Y3 = full_rank(X, r=4)
    assert_equal(Y3.shape, (40,4))
    # Windows - there seems to be some randomness in the SVD result; standardize
    # column signs before comparison
    flipper = np.sign(Y1[0]) * np.sign(Y3[0])
    assert_almost_equal(Y1, Y3 * flipper)


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
