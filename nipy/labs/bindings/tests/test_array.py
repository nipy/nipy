from __future__ import absolute_import
#!/usr/bin/env python

#
# Test fff_array wrapping 
#

from numpy.testing import assert_almost_equal, assert_equal
import numpy as np
from .. import (array_get, array_get_block, array_add, 
                array_sub, array_mul, array_div) 


MAX_TEST_SIZE = 30
def random_shape(size):
    """
    Output random dimensions in the range (2, MAX_TEST_SIZE)
    """
    aux = np.random.randint(MAX_TEST_SIZE-1, size=size) + 2
    if size==1:
        return aux
    else:
        return tuple(aux)


def _test_array_get(x):
    pos = [s // 2 for s in x.shape]
    a = array_get(x, pos[0], pos[1], pos[2], pos[3])
    assert_equal(a, x[pos[0], pos[1], pos[2], pos[3]])

def test_array_get():
    d0, d1, d2, d3 = random_shape(4)
    x = np.random.rand(d0, d1, d2, d3)-.5
    _test_array_get(x)

def _test_array_get_block(x):
    b0 = array_get_block(x, 1, 8, 2, 1, 8, 2, 1, 8, 2, 1, 8, 2)
    b = x[1:8:2, 1:8:2, 1:8:2, 1:8:2]
    assert_equal(b0, b)

def test_array_get_block(): 
    x = np.random.rand(10, 10, 10, 10)-.5
    _test_array_get_block(x)

def _test_array_add(x, y): 
    z = array_add(x, y)
    assert_equal(z, x+y)

def test_array_add(): 
    d0, d1, d2, d3 = random_shape(4)
    x = np.random.rand(d0, d1, d2, d3)-.5
    y = (100*np.random.rand(d0, d1, d2, d3)).astype('uint8')
    _test_array_add(x, y)

def _test_array_mul(x, y): 
    z = array_mul(x, y)
    assert_equal(z, x*y)

def test_array_mul(): 
    d0, d1, d2, d3 = random_shape(4)
    x = np.random.rand(d0, d1, d2, d3)-.5
    y = (100*np.random.rand(d0, d1, d2, d3)).astype('uint8')
    _test_array_mul(x, y)

def _test_array_sub(x, y): 
    z = array_sub(x, y)
    assert_equal(z, x-y)

def test_array_sub(): 
    d0, d1, d2, d3 = random_shape(4)
    x = np.random.rand(d0, d1, d2, d3)-.5
    y = (100*np.random.rand(d0, d1, d2, d3)).astype('uint8')
    _test_array_sub(x, y)

def _test_array_div(x, y): 
    z = array_div(x, y)
    assert_almost_equal(z, x/y)

def test_array_div(): 
    d0, d1, d2, d3 = random_shape(4)
    x = np.random.rand(d0, d1, d2, d3)-.5
    y = np.random.rand(d0, d1, d2, d3)-.5
    _test_array_div(x, y)


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])

