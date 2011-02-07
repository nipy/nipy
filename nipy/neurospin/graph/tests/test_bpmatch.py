# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the design_matrix utilities.

Note that the tests just looks whether the data produces has correct dimension,
not whether it is exact
"""

import numpy as np
from nipy.neurospin.graph.BPmatch import *
from numpy.testing import assert_almost_equal


def test_match_trivial_0():
    c1 = np.array([[0],[1],[2]])
    c2 = c1+0.7
    dmax = 1.0
    i, j, k = match_trivial(c1, c2, dmax)
    assert(len(i)==9)

def test_match_trivial_1():
    c1 = np.array([[0],[1],[2]])
    c2 = c1+0.7
    dmax = 1.0
    i, j, k = match_trivial(c1, c2, dmax)
    a = np.zeros((3,3))
    a[i,j]=k
    assert_almost_equal(a.sum(1),[1,1,1])

def test_match_trivial_2():
    c1 = np.array([[0],[1],[2]]).astype(np.float)
    c2 = c1+0.4
    dmax = 1.0
    i, j, k = match_trivial(c1, c2, dmax)
    a = np.zeros((3,3))
    a[i,j] = k
    assert_almost_equal(a.argmax(1),[0,1,2])

def test_match_trivial_3():
    c1 = np.array([[0],[1],[2]])
    c2 = c1+0.6
    dmax = 1.0
    i, j, k = match_trivial(c1, c2, dmax)
    a = np.zeros((3,3))
    a[i,j] = k
    assert_almost_equal(a.argmax(1),[0,0,1])

def test_match_2():
    c1 = np.array([[0],[1],[2]])
    c2 = c1+0.6
    adjacency = np.ones((3,3))-np.eye(3)
    adjacency[2,0] = 0
    adjacency[0,2] = 0
    dmax = 1.0
    i, j , k = BPmatch(c1, c2, adjacency, dmax)
    a = np.zeros((3,3))    
    a[i,j] = k
    assert_almost_equal(a.argmax(1),[0,1,2])

if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])


