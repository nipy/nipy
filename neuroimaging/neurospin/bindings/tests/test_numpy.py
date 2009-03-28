# Test numpy bindings

import time
from numpy.testing import assert_equal, assert_almost_equal
import numpy as np
import neuroimaging.neurospin.bindings as fb

def time_ratio(t0,t1):
    if t1==0:
        return np.inf
    else:
        return t0/t1


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

    

#
# Test type conversions
#

def test_type_conversions_to_fff():
    print('')
    print('Type conversions: numpy --> fff')
    for npy_t in np.typeDict.values():
        t, nbytes = fb.fff_type(np.dtype(npy_t))
        print('%s --> %s (bytes=%d)' %  (npy_t, t, int(nbytes)))
        if not t == 'unknown type': 
            assert_equal(nbytes, np.dtype(npy_t).itemsize)

def test_type_conversions_in_C():
    for t in fb.c_types:
        npy_t, nbytes = fb.npy_type(t)
        assert_equal(npy_t, t)


#
# Test bindings 
#

def _test_copy_vector(x): 
    t0 = time.clock()
    y0 = fb.copy_vector(x, 0) 
    dt0 = time.clock()-t0
    t1 = time.clock()
    y1 = fb.copy_vector(x, 1) 
    dt1 = time.clock()-t1
    assert_equal(y0, x)
    assert_equal(y1, x)
    ratio = time_ratio(dt0,dt1)
    print('  using fff_array: %f sec' % dt0)
    print('  using numpy C API: %f sec' % dt1)
    print('  ratio: %f' % ratio)

def test_copy_vector_contiguous(): 
    x = (1000*np.random.rand(1e6)).astype('int32')
    print('Contiguous buffer copy (int32-->double)')
    _test_copy_vector(x)

def test_copy_vector_strided(): 
    x0 = (1000*np.random.rand(2e6)).astype('int32')
    x = x0[::2]
    print('Non-contiguous buffer copy (int32-->double)')
    _test_copy_vector(x)

"""
def test_copy_vector_int32(): 
    x = np.random.rand(1e6).astype('int32')
    print('int32 buffer copy')
    _test_copy_vector(x)

def test_copy_vector_uint8(): 
    x = np.random.rand(1e6).astype('uint8')
    print('uint8 buffer copy')
    _test_copy_vector(x)
"""

def _test_pass_vector(x):
    y = fb.pass_vector(x)
    assert_equal(y, x)

def test_pass_vector():
    x = np.random.rand(random_shape(1))-.5
    _test_pass_vector(x)

def test_pass_vector_int32(): 
    x = (1000*(np.random.rand(random_shape(1))-.5)).astype('int32')
    _test_pass_vector(x)

def test_pass_vector_uint8(): 
    x = (256*(np.random.rand(random_shape(1)))).astype('uint8')
    _test_pass_vector(x)


def _test_pass_matrix(x):
    y = fb.pass_matrix(x)
    assert_equal(y, x)
    y = fb.pass_matrix(x.T)
    assert_equal(y, x.T)

def test_pass_matrix():
    d0, d1 = random_shape(2)
    x = np.random.rand(d0, d1)-.5
    _test_pass_matrix(x)

def test_pass_matrix_int32():
    d0, d1 = random_shape(2)
    x = (1000*(np.random.rand(d0, d1)-.5)).astype('int32')
    _test_pass_matrix(x)

def test_pass_matrix_uint8():
    d0, d1 = random_shape(2)
    x = (256*(np.random.rand(d0, d1))).astype('uint8')
    _test_pass_matrix(x)


def _test_pass_array(x):
    y = fb.pass_array(x)
    assert_equal(y, x)
    y = fb.pass_array(x.T)
    assert_equal(y, x.T)

def test_pass_array():
    d0, d1, d2, d3 = random_shape(4)
    x = np.random.rand(d0, d1, d2, d3)-.5
    _test_pass_array(x)

def test_pass_array_int32(): 
    d0, d1, d2, d3 = random_shape(4)
    x = (1000*(np.random.rand(d0, d1, d2, d3)-.5)).astype('int32')
    _test_pass_array(x)

def test_pass_array_uint8():
    d0, d1, d2, d3 = random_shape(4)
    x = (256*(np.random.rand(d0, d1, d2, d3))).astype('uint8')
    _test_pass_array(x)



#
# Multi-iterator testing 
#

def _test_pass_vector_via_iterator(X, pos=0):
    """
    Assume X.ndim == 2
    """
    # axis == 0 
    x = fb.pass_vector_via_iterator(X, axis=0, niters=pos)
    assert_equal(x, X[:, pos])
    # axis == 1
    x = fb.pass_vector_via_iterator(X, axis=1, niters=pos)
    assert_equal(x, X[pos, :])

def test_pass_vector_via_iterator():
    d0, d1 = random_shape(2)
    X = np.random.rand(d0, d1)-.5 
    _test_pass_vector_via_iterator(X)

def test_pass_vector_via_iterator_int32():
    d0, d1 = random_shape(2)
    X = (1000*(np.random.rand(d0, d1)-.5)).astype('int32')
    _test_pass_vector_via_iterator(X)

def test_pass_vector_via_iterator_uint8():
    d0, d1 = random_shape(2)
    X = (100*(np.random.rand(d0, d1))).astype('uint8')
    _test_pass_vector_via_iterator(X)

def test_pass_vector_via_iterator_shift():
    d0, d1 = random_shape(2)
    X = np.random.rand(d0, d1)-.5 
    _test_pass_vector_via_iterator(X, pos=1)

def test_pass_vector_via_iterator_shift_int32():
    d0, d1 = random_shape(2)
    X = (1000*(np.random.rand(d0, d1)-.5)).astype('int32')
    _test_pass_vector_via_iterator(X, pos=1)

def test_pass_vector_via_iterator_shift_uint8():
    d0, d1 = random_shape(2)
    X = (100*(np.random.rand(d0, d1))).astype('uint8')
    _test_pass_vector_via_iterator(X, pos=1)


def _test_copy_via_iterators(Y):
    for axis in range(4):
        Z = fb.copy_via_iterators(Y, axis)
        assert_equal(Z, Y) 
        ZT = fb.copy_via_iterators(Y.T, axis)
        assert_equal(ZT, Y.T) 

def test_copy_via_iterators():
    d0, d1, d2, d3 = random_shape(4)
    Y = np.random.rand(d0, d1, d2, d3)
    _test_copy_via_iterators(Y)

def test_copy_via_iterators_int32():
    d0, d1, d2, d3 = random_shape(4)
    Y = (1000*(np.random.rand(d0, d1, d2, d3)-.5)).astype('int32')
    _test_copy_via_iterators(Y)

def test_copy_via_iterators_uint8():
    d0, d1, d2, d3 = random_shape(4)
    Y = (256*(np.random.rand(d0, d1, d2, d3))).astype('uint8')
    _test_copy_via_iterators(Y)

def _test_sum_via_iterators(Y):
    for axis in range(4):
        Z = fb.sum_via_iterators(Y, axis)
        assert_almost_equal(Z, Y.sum(axis)) 
        ZT = fb.sum_via_iterators(Y.T, axis)
        assert_almost_equal(ZT, Y.T.sum(axis))

def test_sum_via_iterators():
    d0, d1, d2, d3 = random_shape(4)
    Y = np.random.rand(d0, d1, d2, d3)
    _test_sum_via_iterators(Y)

def test_sum_via_iterators_int32():
    d0, d1, d2, d3 = random_shape(4)
    Y = (1000*(np.random.rand(d0, d1, d2, d3)-.5)).astype('int32')
    _test_sum_via_iterators(Y)

def test_sum_via_iterators_uint8():
    d0, d1, d2, d3 = random_shape(4)
    Y = (256*(np.random.rand(d0, d1, d2, d3))).astype('uint8')
    _test_sum_via_iterators(Y)
    

if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])

