""" Testing ``isestimable`` in regression module
"""

import numpy as np

from ..regression import isestimable

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


def test_estimable():
    rng = np.random.RandomState(20120713)
    N, P = (40, 10)
    X = rng.normal(size=(N, P))
    C = rng.normal(size=(1, P))
    assert_true(isestimable(C, X))
    assert_true(isestimable(np.eye(P), X))
    for row in np.eye(P):
        assert_true(isestimable(row, X))
    X = np.ones((40, 2))
    assert_true(isestimable([1, 1], X))
    assert_false(isestimable([1, 0], X))
    assert_false(isestimable([0, 1], X))
    assert_false(isestimable(np.eye(2), X))
    halfX = rng.normal(size=(N, 5))
    X = np.hstack([halfX, halfX])
    assert_false(isestimable(np.hstack([np.eye(5), np.zeros((5, 5))]), X))
    assert_false(isestimable(np.hstack([np.zeros((5, 5)), np.eye(5)]), X))
    assert_true(isestimable(np.hstack([np.eye(5), np.eye(5)]), X))
    # Test array-like for design
    XL = X.tolist()
    assert_true(isestimable(np.hstack([np.eye(5), np.eye(5)]), XL))
    # Test ValueError for incorrect number of columns
    X = rng.normal(size=(N, 5))
    for n in range(1, 4):
        assert_raises(ValueError, isestimable, np.ones((n,)), X)
    assert_raises(ValueError, isestimable, np.eye(4), X)
