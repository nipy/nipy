# -*- Mode: Python -*-  

"""
Image processing routines: 
  * cubic spline sampling 
  * slice timing 
"""

__version__ = '0.2'


# Includes
include "numpy.pxi"

# Externals
cdef extern from "cubic_spline.h":
    
    void cubic_spline_import_array()
    void cubic_spline_transform(ndarray res, ndarray src)
    double cubic_spline_sample1d(double x, ndarray coef) 
    double cubic_spline_sample2d(double x, double y, ndarray coef) 
    double cubic_spline_sample3d(double x, double y, double z, ndarray coef) 
    double cubic_spline_sample4d(double x, double y, double z, double t, ndarray coef) 
    void cubic_spline_resample3d(ndarray im_resampled, ndarray im, 
                                 double* Tvox, int cast_integer)


# Initialize numpy
cubic_spline_import_array()
import_array()
import numpy as np
cimport numpy as np


def cspline_transform(ndarray x):
    c = np.zeros(x.shape)
    cubic_spline_transform(c, x)
    return c

def cspline_sample1d(ndarray R, ndarray C, X=0):
    cdef double *r, *x
    cdef broadcast multi
    Xa = np.reshape(X, R.shape).astype('double')
    multi = PyArray_MultiIterNew(2, <void*>R, <void*>Xa)
    while(multi.index < multi.size):
        r = <double*>PyArray_MultiIter_DATA(multi, 0)
        x = <double*>PyArray_MultiIter_DATA(multi, 1)
        r[0] = cubic_spline_sample1d(x[0], C)
        PyArray_MultiIter_NEXT(multi)
    return R

def cspline_sample2d(ndarray R, ndarray C, X=0, Y=0):
    cdef double *r, *x, *y
    cdef broadcast multi
    Xa = np.reshape(X, R.shape).astype('double')
    Ya = np.reshape(Y, R.shape).astype('double')
    multi = PyArray_MultiIterNew(3, <void*>R, <void*>Xa, <void*>Ya)
    while(multi.index < multi.size):
        r = <double*>PyArray_MultiIter_DATA(multi, 0)
        x = <double*>PyArray_MultiIter_DATA(multi, 1)
        y = <double*>PyArray_MultiIter_DATA(multi, 2)
        r[0] = cubic_spline_sample2d(x[0], y[0], C)
        PyArray_MultiIter_NEXT(multi)
    return R

def cspline_sample3d(ndarray R, ndarray C, X=0, Y=0, Z=0):
    cdef double *r, *x, *y, *z
    cdef broadcast multi
    Xa = np.reshape(X, R.shape).astype('double')
    Ya = np.reshape(Y, R.shape).astype('double')
    Za = np.reshape(Z, R.shape).astype('double')
    multi = PyArray_MultiIterNew(4, <void*>R, <void*>Xa, <void*>Ya, <void*>Za)
    while(multi.index < multi.size):
        r = <double*>PyArray_MultiIter_DATA(multi, 0)
        x = <double*>PyArray_MultiIter_DATA(multi, 1)
        y = <double*>PyArray_MultiIter_DATA(multi, 2)
        z = <double*>PyArray_MultiIter_DATA(multi, 3)
        r[0] = cubic_spline_sample3d(x[0], y[0], z[0], C)
        PyArray_MultiIter_NEXT(multi)
    return R


def cspline_sample4d(ndarray R, ndarray C, X=0, Y=0, Z=0, T=0):
    """
    cubic_spline_sample4d(R, C, X=0, Y=0, Z=0, T=0):

    In-place cubic spline sampling. R.dtype must be 'double'. 
    """
    cdef double *r, *x, *y, *z, *t
    cdef broadcast multi
    Xa = np.reshape(X, R.shape).astype('double')
    Ya = np.reshape(Y, R.shape).astype('double')
    Za = np.reshape(Z, R.shape).astype('double')
    Ta = np.reshape(T, R.shape).astype('double')
    multi = PyArray_MultiIterNew(5, <void*>R, <void*>Xa, <void*>Ya, <void*>Za, <void*>Ta)
    while(multi.index < multi.size):
        r = <double*>PyArray_MultiIter_DATA(multi, 0)
        x = <double*>PyArray_MultiIter_DATA(multi, 1)
        y = <double*>PyArray_MultiIter_DATA(multi, 2)
        z = <double*>PyArray_MultiIter_DATA(multi, 3)
        t = <double*>PyArray_MultiIter_DATA(multi, 4)
        r[0] = cubic_spline_sample4d(x[0], y[0], z[0], t[0], C)
        PyArray_MultiIter_NEXT(multi)
    return R


def cspline_resample3d(ndarray im, dims, ndarray Tvox, dtype=None):
    """
    cspline_resample3d(im, dims, Tvox, dtype=None)

    Note that the input transformation Tvox will be re-ordered in C
    convention if needed.
    """
    cdef double *tvox
    cdef int cast_integer

    # Create output array
    if dtype == None:
        dtype = im.dtype
    im_resampled = np.zeros(tuple(dims), dtype=dtype)

    # Ensure that the Tvox array is C-contiguous (required by the
    # underlying C routine)
    Tvox = np.asarray(Tvox, order='C')
    tvox = <double*>Tvox.data

    # Actual resampling 
    cast_integer = np.issubclass(dtype.type, np.integer)
    cubic_spline_resample3d(im_resampled, im, tvox, cast_integer)

    return im_resampled



