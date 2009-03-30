#!/usr/bin/env python

#
# Test fff linear algebra routines 
#

from numpy.testing import assert_almost_equal, assert_equal
import numpy as np
import neuroimaging.neurospin.bindings as fb

n = 15

def test_vector_get():
    x = np.random.rand(n)
    i = np.random.randint(n)
    xi = fb.vector_get(x, i)
    assert_equal(xi, x[i])

def test_vector_get_int32():
    x = (100*np.random.rand(n)).astype('int32')
    i = np.random.randint(n)
    xi = fb.vector_get(x, i)
    assert_equal(xi, x[i])

def test_vector_set():
    x = np.random.rand(n)
    i = np.random.randint(n)
    y = fb.vector_set(x, i, 3)
    assert_equal(3, y[i])
    
def test_vector_set_int32():
    x = (100*np.random.rand(n)).astype('int32')
    i = np.random.randint(n)
    y = fb.vector_set(x, i, 3)
    assert_equal(3, y[i])



if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])

