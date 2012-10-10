# -*- Mode: Python -*-  

"""
Bindings for various image registration routines written in C: joint
histogram computation, cubic spline interpolation, non-rigid
transformations. 
"""

__version__ = '0.3'


# Includes
from numpy cimport (import_array, ndarray, flatiter, broadcast, 
                    PyArray_MultiIterNew, PyArray_MultiIter_DATA, 
                    PyArray_MultiIter_NEXT)


# Externals
cdef extern from "joint_histogram.h":
    void joint_histogram_import_array()
    int joint_histogram(ndarray H, unsigned int clampI, unsigned int clampJ,  
                        flatiter iterI, ndarray imJ_padded, 
                        ndarray Tvox, int interp)
    int L1_moments(double* n, double* median, double* dev, ndarray H)

cdef extern from "cubic_spline.h":
    void cubic_spline_import_array()
    void cubic_spline_transform(ndarray res, ndarray src)
    double cubic_spline_sample1d(double x, ndarray coef, 
                                 int mode) 
    double cubic_spline_sample2d(double x, double y, ndarray coef, 
                                 int mode_x, int mode_y) 
    double cubic_spline_sample3d(double x, double y, double z, ndarray coef,
                                 int mode_x, int mode_y, int mode_z) 
    double cubic_spline_sample4d(double x, double y, double z, double t, ndarray coef, 
                                 int mode_x, int mode_y, int mode_z, int mode_t)
    void cubic_spline_resample3d(ndarray im_resampled, ndarray im, 
                                 double* Tvox, int cast_integer,
                                 int mode_x, int mode_y, int mode_z)

cdef extern from "polyaffine.h": 
    void polyaffine_import_array()
    void apply_polyaffine(ndarray XYZ, ndarray Centers, ndarray Affines, ndarray Sigma)


# Initialize numpy
joint_histogram_import_array()
cubic_spline_import_array()
polyaffine_import_array()
import_array()
import numpy as np

# Globals
modes = {'zero': 0, 'nearest': 1, 'reflect': 2}


def _joint_histogram(ndarray H, flatiter iterI, ndarray imJ, ndarray Tvox, long interp):
    """
    Compute the joint histogram given a transformation trial. 
    """
    cdef:
        double *h, *tvox
        unsigned int clampI, clampJ
        int ret

    # Views
    clampI = <unsigned int>H.shape[0]
    clampJ = <unsigned int>H.shape[1]    

    # Compute joint histogram 
    ret = joint_histogram(H, clampI, clampJ, iterI, imJ, Tvox, interp)
    if not ret == 0:
        raise RuntimeError('Joint histogram failed because of incorrect input arrays.')

    return 


def _L1_moments(ndarray H):
    """
    Compute L1 moments of order 0, 1 and 2 of a one-dimensional
    histogram.
    """
    cdef:
        double n[1], median[1], dev[1]
        int ret

    ret = L1_moments(n, median, dev, H)
    if not ret == 0:
        raise RuntimeError('L1_moments failed because input array is not double.')

    return n[0], median[0], dev[0]


def _cspline_transform(ndarray x):
    c = np.zeros([x.shape[i] for i in range(x.ndim)], dtype=np.double)
    cubic_spline_transform(c, x)
    return c

cdef ndarray _reshaped_double(object in_arr, ndarray sh_arr):
    shape = [sh_arr.shape[i] for i in range(sh_arr.ndim)]
    return np.reshape(in_arr, shape).astype(np.double)

def _cspline_sample1d(ndarray R, ndarray C, X=0, mode='zero'):
    cdef double *r, *x
    cdef broadcast multi
    Xa = _reshaped_double(X, R) 
    multi = PyArray_MultiIterNew(2, <void*>R, <void*>Xa)
    while(multi.index < multi.size):
        r = <double*>PyArray_MultiIter_DATA(multi, 0)
        x = <double*>PyArray_MultiIter_DATA(multi, 1)
        r[0] = cubic_spline_sample1d(x[0], C, modes[mode])
        PyArray_MultiIter_NEXT(multi)
    return R

def _cspline_sample2d(ndarray R, ndarray C, X=0, Y=0, 
                      mx='zero', my='zero'):
    cdef double *r, *x, *y
    cdef broadcast multi
    Xa = _reshaped_double(X, R)
    Ya = _reshaped_double(Y, R)
    multi = PyArray_MultiIterNew(3, <void*>R, <void*>Xa, <void*>Ya)
    while(multi.index < multi.size):
        r = <double*>PyArray_MultiIter_DATA(multi, 0)
        x = <double*>PyArray_MultiIter_DATA(multi, 1)
        y = <double*>PyArray_MultiIter_DATA(multi, 2)
        r[0] = cubic_spline_sample2d(x[0], y[0], C, modes[mx], modes[my])
        PyArray_MultiIter_NEXT(multi)
    return R

def _cspline_sample3d(ndarray R, ndarray C, X=0, Y=0, Z=0, 
                      mx='zero', my='zero', mz='zero'):
    cdef double *r, *x, *y, *z
    cdef broadcast multi
    Xa = _reshaped_double(X, R)
    Ya = _reshaped_double(Y, R)
    Za = _reshaped_double(Z, R)
    multi = PyArray_MultiIterNew(4, <void*>R, <void*>Xa, <void*>Ya, <void*>Za)
    while(multi.index < multi.size):
        r = <double*>PyArray_MultiIter_DATA(multi, 0)
        x = <double*>PyArray_MultiIter_DATA(multi, 1)
        y = <double*>PyArray_MultiIter_DATA(multi, 2)
        z = <double*>PyArray_MultiIter_DATA(multi, 3)
        r[0] = cubic_spline_sample3d(x[0], y[0], z[0], C, modes[mx], modes[my], modes[mz])
        PyArray_MultiIter_NEXT(multi)
    return R


def _cspline_sample4d(ndarray R, ndarray C, X=0, Y=0, Z=0, T=0, 
                      mx='zero', my='zero', mz='zero', mt='zero'):
    """
    In-place cubic spline sampling. R.dtype must be 'double'. 
    """
    cdef double *r, *x, *y, *z, *t
    cdef broadcast multi
    Xa = _reshaped_double(X, R)
    Ya = _reshaped_double(Y, R)
    Za = _reshaped_double(Z, R)
    Ta = _reshaped_double(T, R)
    multi = PyArray_MultiIterNew(5, <void*>R, <void*>Xa, <void*>Ya, <void*>Za, <void*>Ta)
    while(multi.index < multi.size):
        r = <double*>PyArray_MultiIter_DATA(multi, 0)
        x = <double*>PyArray_MultiIter_DATA(multi, 1)
        y = <double*>PyArray_MultiIter_DATA(multi, 2)
        z = <double*>PyArray_MultiIter_DATA(multi, 3)
        t = <double*>PyArray_MultiIter_DATA(multi, 4)
        r[0] = cubic_spline_sample4d(x[0], y[0], z[0], t[0], C, modes[mx], modes[my], modes[mz], modes[mt])
        PyArray_MultiIter_NEXT(multi)
    return R


def _cspline_resample3d(ndarray im, dims, ndarray Tvox, dtype=None,
                        mx='zero', my='zero', mz='zero'):
    """
    Perform cubic spline resampling of a 3d input image `im` into a
    grid with shape `dims` according to an affine transform
    represented by a 4x4 matrix `Tvox` that assumes voxel
    coordinates. Boundary conditions on each axis are determined by
    the keyword arguments `mx`, `my` and `mz`, respectively. Possible
    choices are:

    'zero': assume zero intensity outside the target grid
    'nearest': extrapolate intensity by the closest grid point along the axis
    'reflect': extrapolate intensity by mirroring the input image along the axis

    Note that `Tvox` will be re-ordered in C convention if needed.
    """
    cdef double *tvox
    cdef int cast_integer

    # Create output array
    if dtype == None:
        dtype = im.dtype
    im_resampled = np.zeros(tuple(dims), dtype=dtype)

    # Ensure that the Tvox array is C-contiguous (required by the
    # underlying C routine)
    Tvox = np.asarray(Tvox, dtype='double', order='C')
    tvox = <double*>Tvox.data

    # Actual resampling 
    if dtype.kind == 'i':
        cast_integer = 1
    elif dtype.kind == 'u':
        cast_integer = 2
    else:
        cast_integer = 0
    cubic_spline_resample3d(im_resampled, im, tvox, cast_integer,
                            modes[mx], modes[my], modes[mz])

    return im_resampled


def check_array(ndarray x, int dim, int exp_dim, xname): 
    if not x.flags['C_CONTIGUOUS'] or not x.dtype=='double':
        raise ValueError('%s array should be double C-contiguous' % xname)
    if not dim == exp_dim: 
        raise ValueError('%s has size %d in last dimension, %d expected' % (xname, dim, exp_dim))

def _apply_polyaffine(ndarray xyz, ndarray centers, ndarray affines, ndarray sigma): 

    check_array(xyz, xyz.shape[1], 3, 'xyz') 
    check_array(centers, centers.shape[1], 3, 'centers')
    check_array(affines, affines.shape[1], 12, 'affines')
    check_array(sigma, sigma.size, 3, 'sigma')
    if not centers.shape[0] == affines.shape[0]: 
        raise ValueError('centers and affines arrays should have same shape[0]')

    apply_polyaffine(xyz, centers, affines, sigma) 
