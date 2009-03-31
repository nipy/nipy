# -*- Mode: Python -*-  

__version__ = '0.1'

import numpy as np
include "numpy.pxi"

def slice_time(Z, double tr_slices, slice_order):
    """
    Fast routine to compute the time when a slice is acquired given its index
    """
    cdef double *z, *t, *s
    cdef double weight, slice_interp
    cdef unsigned int zfloor, cycles, zl, zr
    cdef broadcast multi
    cdef unsigned int nslices

    Za = np.asarray(Z, dtype='double')
    T = np.zeros(Za.shape, dtype='double')
    S = np.ascontiguousarray(slice_order, dtype='double')
    s = <double*>PyArray_DATA(<ndarray>S)
    nslices = S.size
 
    multi = PyArray_MultiIterNew(2, <void*>Za, <void*>T)
    while(multi.index < multi.size):
        z = <double*>PyArray_MultiIter_DATA(multi, 0)
        t = <double*>PyArray_MultiIter_DATA(multi, 1)
        zfloor = <unsigned int>z[0]
        cycles = zfloor / nslices 
        zl = zfloor % nslices
        zr = zl + 1
        weight = z[0]-zfloor
        if zl < (nslices-1):
            slice_interp = (1-weight)*s[zl] + weight*s[zr]
        else:
            slice_interp = (1-weight)*s[zl] + weight*nslices
        t[0] = cycles + tr_slices*slice_interp
        PyArray_MultiIter_NEXT(multi)
    return T


