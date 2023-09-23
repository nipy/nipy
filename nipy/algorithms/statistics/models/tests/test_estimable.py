""" Testing ``isestimable`` in regression module
"""

import numpy as np
import pytest

from ..regression import isestimable


def test_estimable():
    rng = np.random.RandomState(20120713)
    N, P = (40, 10)
    X = rng.normal(size=(N, P))
    C = rng.normal(size=(1, P))
    assert isestimable(C, X)
    assert isestimable(np.eye(P), X)
    for row in np.eye(P):
        assert isestimable(row, X)
    X = np.ones((40, 2))
    assert isestimable([1, 1], X)
    assert not isestimable([1, 0], X)
    assert not isestimable([0, 1], X)
    assert not isestimable(np.eye(2), X)
    halfX = rng.normal(size=(N, 5))
    X = np.hstack([halfX, halfX])
    assert not isestimable(np.hstack([np.eye(5), np.zeros((5, 5))]), X)
    assert not isestimable(np.hstack([np.zeros((5, 5)), np.eye(5)]), X)
    assert isestimable(np.hstack([np.eye(5), np.eye(5)]), X)
    # Test array-like for design
    XL = X.tolist()
    assert isestimable(np.hstack([np.eye(5), np.eye(5)]), XL)
    # Test ValueError for incorrect number of columns
    X = rng.normal(size=(N, 5))
    for n in range(1, 4):
        pytest.raises(ValueError, isestimable, np.ones((n,)), X)
    pytest.raises(ValueError, isestimable, np.eye(4), X)
