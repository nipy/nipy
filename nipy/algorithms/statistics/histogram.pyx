# -*- Mode: Python -*-  Not really, but the syntax is close enough
"""
Author: Alexis Roche, 2012.
"""

import numpy as np
cimport numpy as np

np.import_array()

def histogram(x):
    """
    Fast histogram computation assuming input array is of uintp data
    type.

    Parameters
    ----------
    x: array-like
      Assumed with uintp dtype

    Output
    ------
    h: 1d array
      Histogram
    """
    if not x.dtype=='uintp':
        raise ValueError('input array should have uintp data type')

    cdef np.npy_uintp xv
    cdef np.npy_uintp nbins = <np.npy_uintp>x.max() + 1
    cdef np.flatiter it = x.flat
    cdef np.ndarray h = np.zeros(nbins, dtype='uintp')
    cdef np.npy_uintp* hv

    while np.PyArray_ITER_NOTDONE(it):
        xv = (<np.npy_uintp*>np.PyArray_ITER_DATA(it))[0]
        hv = <np.npy_uintp*>np.PyArray_DATA(h) + xv
        hv[0] += 1
        np.PyArray_ITER_NEXT(it)
        
    return h
