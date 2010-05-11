# -*- Mode: Python -*-  Not really, but the syntax is close enough
"""
Basic ndarray routines for faster computations.

Author: Alexis Roche, 2008.
"""

__version__ = '0.1'

import numpy as np
cimport numpy as np

import cython

@cython.boundscheck(False)
def add_lines(np.ndarray[np.float_t, ndim=2] A, 
               np.ndarray[np.float_t, ndim=2] B, 
               np.ndarray[np.int_t,   ndim=1] I):
    """
    add(A, B, I)

    Add each line of A to a line of B indexed by I, where
    A and B are two-dimensional arrays and I is a
    one-dimensional array of indices.

    This is equivalent to: 

    for i in xrange(len(I)):
      B[I[i]] += A[i]

    """
    cdef int i, j, index
    cdef int i_max = I.shape[0]
    cdef int j_max = B.shape[1]

    for i in range(i_max):
        index = I[i]
        for j in range(j_max):
            B[index, j] = B[index, j] + A[i, j]

