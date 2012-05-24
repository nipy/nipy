# -*- Mode: Python -*-  Not really, but the syntax is close enough
"""
Author: Alexis Roche, 2012.
"""

import numpy as np
cimport numpy as np

np.import_array()

def histogram(x):
    """
    Fast histogram computation assuming input array is of uint data
    type.

    Parameters
    ----------
    x: array-like
      Assumed with uint dtype

    Output
    ------
    h: 1d array
      Histogram
    """
    if not x.dtype=='uint':
        raise ValueError('input array should have uint data type')

    cdef unsigned int xv
    cdef unsigned int nbins = <unsigned int>x.max() + 1
    cdef np.flatiter it = x.flat
    cdef np.ndarray h = np.zeros(nbins, dtype='uint')
    cdef unsigned int* hv

    while np.PyArray_ITER_NOTDONE(it):
        xv = (<unsigned int*>np.PyArray_ITER_DATA(it))[0]
        hv = <unsigned int*>np.PyArray_DATA(h) + xv
        hv[0] += 1
        np.PyArray_ITER_NEXT(it)
        
    return h
