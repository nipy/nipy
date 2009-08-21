# -*- Mode: Python -*-  

"""
Fast registration routines module: joint histogram computation,
similarity measures, cubic spline resampling, affine transformation
parameterization.

Author: Alexis Roche, 2008.
"""

__version__ = '0.2'


# Includes
include "numpy.pxi"

# Externals
cdef extern from "iconic.h":

    void iconic_import_array()
    void histogram(double* H, unsigned int clampI, flatiter iterI)
    void joint_histogram(double* H, unsigned int clampI, unsigned int clampJ,  
                         flatiter iterI, ndarray imJ_padded, 
                         double* Tvox, int interp)
    double correlation_coefficient(double* H, unsigned int clampI, unsigned int clampJ, double* n)
    double correlation_ratio(double* H, unsigned int clampI, unsigned int clampJ, double* n) 
    double correlation_ratio_L1(double* H, double* hI, unsigned int clampI, unsigned int clampJ, double* n) 
    double joint_entropy(double* H, unsigned int clampI, unsigned int clampJ, double* n)
    double conditional_entropy(double* H, double* hJ, unsigned int clampI, unsigned int clampJ, double* n) 
    double mutual_information(double* H, 
                              double* hI, unsigned int clampI, 
                              double* hJ, unsigned int clampJ,
                              double* n)
    double normalized_mutual_information(double* H, 
                                         double* hI, unsigned int clampI, 
                                         double* hJ, unsigned int clampJ, 
                                         double* n) 
    double supervised_mutual_information(double* H, double* F, 
                                         double* fI, unsigned int clampI, 
                                         double* fJ, unsigned int clampJ,
                                         double* n) 
    void cubic_spline_resample(ndarray im_resampled, ndarray im, double* Tvox, int cast_integer)


cdef extern from "cubic_spline.h":
    
    void cubic_spline_import_array()
    void cubic_spline_transform(ndarray res, ndarray src)
    double cubic_spline_sample1d(double x, ndarray coef) 
    double cubic_spline_sample2d(double x, double y, ndarray coef) 
    double cubic_spline_sample3d(double x, double y, double z, ndarray coef) 
    double cubic_spline_sample4d(double x, double y, double z, double t, ndarray coef) 


# Initialize numpy
iconic_import_array()
cubic_spline_import_array()
import_array()
import numpy as np
cimport numpy as np


# Enumerate similarity measures
cdef enum similarity_measure:
    CORRELATION_COEFFICIENT,
    CORRELATION_RATIO,
    CORRELATION_RATIO_L1,
    JOINT_ENTROPY,
    CONDITIONAL_ENTROPY,
    MUTUAL_INFORMATION,
    NORMALIZED_MUTUAL_INFORMATION,
    SUPERVISED_MUTUAL_INFORMATION,

# Corresponding Python dictionary 
similarity_measures = {'cc': CORRELATION_COEFFICIENT,
                       'cr': CORRELATION_RATIO,
                       'crl1': CORRELATION_RATIO_L1, 
                       'mi': MUTUAL_INFORMATION, 
                       'je': JOINT_ENTROPY,
                       'ce': CONDITIONAL_ENTROPY,
                       'nmi': NORMALIZED_MUTUAL_INFORMATION,
                       'smi': SUPERVISED_MUTUAL_INFORMATION}


def _histogram(ndarray H, flatiter iterI):
    """
    _joint_histogram(H, iterI)
    Comments to follow.
    """
    cdef double *h
    cdef unsigned int clampI

    # Views
    clampI = <int>H.dimensions[0]
    h = <double*>H.data

    # Compute image histogram 
    histogram(h, clampI, iterI)

    return 


def _joint_histogram(ndarray H, flatiter iterI, ndarray imJ, ndarray Tvox, int interp):
    """
    _joint_histogram(H, iterI, imJ, Tvox, interp)
    Comments to follow.
    """
    cdef double *h, *tvox
    cdef unsigned int clampI, clampJ

    # Views
    clampI = <int>H.dimensions[0]
    clampJ = <int>H.dimensions[1]    
    h = <double*>H.data
    tvox = <double*>Tvox.data

    # Compute joint histogram 
    joint_histogram(h, clampI, clampJ, iterI, imJ, tvox, interp)

    return 


def _similarity(ndarray H, ndarray HI, ndarray HJ, int simitype, ndarray F=None):
    """
    _similarity(H, hI, hJ, simitype, ndarray F=None)
    Comments to follow
    """
    cdef int isF = 0
    cdef double *h, *hI, *hJ, *f=NULL
    cdef double simi=0.0, n
    cdef unsigned int clampI, clampJ

    # Array views
    clampI = <int>H.dimensions[0]
    clampJ = <int>H.dimensions[1]
    h = <double*>H.data
    hI = <double*>HI.data
    hJ = <double*>HJ.data
    if F != None:
        f = <double*>F.data
        isF = 1

    # Switch 
    if simitype == CORRELATION_COEFFICIENT:
        simi = correlation_coefficient(h, clampI, clampJ, &n)
    elif simitype == CORRELATION_RATIO: 
        simi = correlation_ratio(h, clampI, clampJ, &n) 
    elif simitype == CORRELATION_RATIO_L1:
        simi = correlation_ratio_L1(h, hI, clampI, clampJ, &n) 
    elif simitype == MUTUAL_INFORMATION: 
        simi = mutual_information(h, hI, clampI, hJ, clampJ, &n) 
    elif simitype == JOINT_ENTROPY:
        simi = joint_entropy(h, clampI, clampJ, &n) 
    elif simitype == CONDITIONAL_ENTROPY:
        simi = conditional_entropy(h, hJ, clampI, clampJ, &n) 
    elif simitype == NORMALIZED_MUTUAL_INFORMATION:
        simi = normalized_mutual_information(h, hI, clampI, hJ, clampJ, &n) 
    elif simitype == SUPERVISED_MUTUAL_INFORMATION:
        simi = supervised_mutual_information(h, f, hI, clampI, hJ, clampJ, &n)
    else:
        simi = 0.0
        
    return simi



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

    In-place cubic spline sampling. 
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


def cspline_resample(ndarray im, dims, ndarray Tvox, dtype=None):
    """
    cspline_resample(im, dims, Tvox, dtype=None)

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
    cast_integer = np.issubdtype(dtype, int)
    cubic_spline_resample(im_resampled, im, tvox, cast_integer)

    return im_resampled



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


# Enumerate transformation types
cdef enum transformation_type:
    RIGID2D=0, SIMILARITY2D=1, AFFINE2D=2,
    RIGID3D=3, SIMILARITY3D=4, AFFINE3D=5

# Corresponding Python constants 
affines = ['rigid', 'similarity', 'affine']

_rigid2d = [0,1,5]
_similarity2d = [0,1,5,6,7]
_affine2d = [0,1,5,6,7,11]
_rigid3d = range(6)
_similarity3d = range(9)
_affine3d = range(12)

_affines = [_rigid2d, _similarity2d[0:4], _affine2d, 
            _rigid3d, _similarity3d[0:7], _affine3d]



def rotation_vec2mat(r):
    """
    R = rotation_vec2mat(r)

    The rotation matrix is given by the Rodrigues formula:
    
    R = Id + sin(theta)*Sn + (1-cos(theta))*Sn^2  
    
    with:
    
           0  -nz  ny
    Sn =   nz   0 -nx
          -ny  nx   0
    
    where n = r / ||r||
    
    In case the angle ||r|| is very small, the above formula may lead
    to numerical instabilities. We instead use a Taylor expansion
    around theta=0:
    
    R = I + sin(theta)/tetha Sr + (1-cos(theta))/teta2 Sr^2
    
    leading to:
    
    R = I + (1-theta2/6)*Sr + (1/2-theta2/24)*Sr^2
    """
    cdef double theta, theta2
    theta = <double> np.linalg.norm(r)
    if theta > 1e-30:
        n = r/theta
        Sn = np.array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
        R = np.eye(3) + np.sin(theta)*Sn + (1-np.cos(theta))*np.dot(Sn,Sn)
    else:
        Sr = np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])
        theta2 = theta*theta
        R = np.eye(3) + (1-theta2/6.)*Sr + (.5-theta2/24.)*np.dot(Sr,Sr)
    return R




def param_to_vector12(ndarray param, ndarray t0, ndarray precond, int stamp=AFFINE3D):
    """
    t = param_to_vector12(p, t, precond, stamp=AFFINE3D).

    In-place modification of t. 

    p is a 1d-array of affine transformation parameters with size
    dependent on the transformation type, which is coded by the
    integer stamp.
    """
    t = t0

    # Switch on transformation type
    if stamp == RIGID3D:
        t[_rigid3d] = param*precond[_rigid3d]
    elif stamp == SIMILARITY3D:
        t[_similarity3d] = param[[0,1,2,3,4,5,6,6,6]]*precond[_similarity3d]
    elif stamp == AFFINE3D:
        t = param*precond
    elif stamp == RIGID2D:
        t[_rigid2d] = param*precond[_rigid2d]
    elif stamp == SIMILARITY2D:
        t[_similarity2d] = param[[0,1,2,3,3]]*precond[_similarity2d]
    elif stamp == AFFINE2D:
        t[_affine2d] = param*precond[_affine2d]
    
    return t
    

def matrix44(ndarray t, dtype):
    """
    T = matrix44(t)

    t is a vector of of affine transformation parameters with size at
    least 6.

    size < 6 ==> error
    size == 6 ==> t is interpreted as translation + rotation
    size == 7 ==> t is interpreted as translation + rotation + isotropic scaling
    7 < size < 12 ==> error
    size >= 12 ==> t is interpreted as translation + rotation + scaling + shearing 
    """
    cdef int size
    size = <int>PyArray_SIZE(t)
    T = np.eye(4, dtype=dtype)
    R = rotation_vec2mat(t[3:6])
    if size == 6:
        T[0:3,0:3] = R
    elif size == 7:
        T[0:3,0:3] = t[6]*R
    else:
        S = np.diag(t[6:9]) 
        Q = rotation_vec2mat(t[9:12]) 
        # Beware: R*s*Q
        T[0:3,0:3] = np.dot(R,np.dot(S,Q))
    T[0:3,3] = t[0:3] 
    return T 



