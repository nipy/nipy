# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the fast distance estimator
"""
from __future__ import absolute_import
import numpy as np
from numpy.testing import assert_almost_equal

from ..fast_distance import euclidean_distance as ed 

def test_euclidean_1():
    """ test that the euclidean distance is as expected
    """  
    nx, ny = (10, 12)
    X = np.random.randn(nx, 2)
    Y = np.random.randn(ny, 2)
    ED = ed(X, Y)
    ref = np.zeros((nx, ny))
    for i in range(nx):
    	ref[i] = np.sqrt(np.sum((Y - X[i])**2, 1))
	
    assert_almost_equal(ED, ref)		

 
def test_euclidean_2():
    """ test that the euclidean distance is as expected
    """  
    nx = 10
    X = np.random.randn(nx, 2)
    ED = ed(X)
    ref = np.zeros((nx, nx))
    for i in range(nx):
    	ref[i] = np.sqrt(np.sum((X - X[i])**2, 1))
	
    assert_almost_equal(ED, ref) 

  
if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
