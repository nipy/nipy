# -*- Mode: Python -*-  Not really, but the syntax is close enough

"""
Very fast quantile computation using partial sorting.
Author: Alexis Roche.
"""

__version__ = '0.1'

cdef extern from "quantile.h":
    double quantile(double* data,
                    unsigned long size,
                    unsigned long stride,
                    double r,
                    int interp)

import numpy as np
cimport numpy as np

np.import_array()

# This is faster than scipy.stats.scoreatpercentile due to partial
# sorting
def _quantile(X, double ratio, int interp=False, int axis=0):
    """
    q = quantile(data, ratio, interp=False, axis=0).

    Partial sorting algorithm, very fast!!!
    """
    cdef double *x, *y
    cdef long int size, stride
    cdef np.flatiter itX, itY

    # Allocate output array Y
    dims = list(X.shape)
    dims[axis] = 1
    Y = np.zeros(dims)

    # Set size and stride along specified axis
    size = X.shape[axis]
    stride = X.strides[axis] / sizeof(double)

    # Create array iterators
    itX = np.PyArray_IterAllButAxis(X, &axis)
    itY = np.PyArray_IterAllButAxis(Y, &axis)

    # Loop 
    while np.PyArray_ITER_NOTDONE(itX):
        x = <double*>np.PyArray_ITER_DATA(itX)
        y = <double*>np.PyArray_ITER_DATA(itY)
        y[0] = quantile(x, size, stride, ratio, interp) 
        np.PyArray_ITER_NEXT(itX)
        np.PyArray_ITER_NEXT(itY)

    return Y

# This is faster than numpy.stats
# due to the underlying algorithm that relies on
# partial sorting as opposed to full sorting.
def _median(X, axis=0):
    """
    median(X, axis=0)
    """
    return _quantile(X, axis=axis, ratio=0.5, interp=True)

