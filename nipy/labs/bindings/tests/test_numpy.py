# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# Test numpy bindings
from __future__ import absolute_import

import numpy as np

from .. import (c_types, fff_type, npy_type, copy_vector, pass_matrix,
                pass_vector, pass_array, pass_vector_via_iterator,
                sum_via_iterators, copy_via_iterators)


from nose.tools import assert_equal
from numpy.testing import assert_almost_equal, assert_array_equal

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
    # use np.sctypes for testing numpy types, np.typeDict.values
    # contains a lot of duplicates.  There are 140 values in
    # np.typeDict, but only 21 unique numpy types.  But only 11 fff
    # types in c_types.
    for type_key in np.sctypes:
        for npy_t in np.sctypes[type_key]:
            t, nbytes = fff_type(np.dtype(npy_t))
            if not t == 'unknown type':
                yield assert_equal, nbytes, np.dtype(npy_t).itemsize


def test_type_conversions_in_C():
    for t in c_types:
        npy_t, nbytes = npy_type(t)
        yield assert_equal, npy_t, t


#
# Test bindings 
#

def _test_copy_vector(x):
    # use fff
    y0 = copy_vector(x, 0)
    # use numpy
    y1 = copy_vector(x, 1) 
    yield assert_equal, y0, x
    yield assert_equal, y1, x


def test_copy_vector_contiguous():
    x = (1000*np.random.rand(int(1e6))).astype('int32')
    _test_copy_vector(x)

def test_copy_vector_strided():
    x0 = (1000*np.random.rand(int(2e6))).astype('int32')
    x = x0[::2]
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
    y = pass_vector(x)
    assert_array_equal(y, x)


def test_pass_vector():
    x = np.random.rand(int(random_shape(1)))-.5
    _test_pass_vector(x)


def test_pass_vector_int32():
    x = (1000*(np.random.rand(int(random_shape(1)))-.5)).astype('int32')
    _test_pass_vector(x)


def test_pass_vector_uint8():
    x = (256*(np.random.rand(int(random_shape(1))))).astype('uint8')
    _test_pass_vector(x)


def _test_pass_matrix(x):
    y = pass_matrix(x)
    yield assert_equal, y, x
    y = pass_matrix(x.T)
    yield assert_equal, y, x.T


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
    y = pass_array(x)
    yield assert_equal, y, x
    y = pass_array(x.T)
    yield assert_equal, y, x.T


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
    x = pass_vector_via_iterator(X, axis=0, niters=pos)
    yield assert_equal, x, X[:, pos]
    # axis == 1
    x = pass_vector_via_iterator(X, axis=1, niters=pos)
    yield assert_equal, x, X[pos, :]


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
        Z = copy_via_iterators(Y, axis)
        yield assert_equal, Z, Y
        ZT = copy_via_iterators(Y.T, axis)
        yield assert_equal, ZT, Y.T 


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
        Z = sum_via_iterators(Y, axis)
        yield assert_almost_equal, Z, Y.sum(axis)
        ZT = sum_via_iterators(Y.T, axis)
        yield assert_almost_equal, ZT, Y.T.sum(axis)


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
