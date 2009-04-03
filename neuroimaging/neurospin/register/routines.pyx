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
    void joint_histogram(double* H, int clampI, int clampJ,  
                         flatiter iterI, ndarray imJ_padded, 
                         double* Tvox, int interp)
    double correlation_coefficient(double* H, int clampI, int clampJ)
    double correlation_ratio(double* H, int clampI, int clampJ) 
    double correlation_ratio_L1(double* H, double* hI, int clampI, int clampJ) 
    double joint_entropy(double* H, int clampI, int clampJ)
    double conditional_entropy(double* H, double* hJ, int clampI, int clampJ) 
    double mutual_information(double* H, 
                              double* hI, int clampI, 
                              double* hJ, int clampJ)
    double normalized_mutual_information(double* H, 
                                         double* hI, int clampI, 
                                         double* hJ, int clampJ) 
    double supervised_mutual_information(double* H, double* F, 
                                         double* fI, int clampI, 
                                         double* fJ, int clampJ) 
    void cubic_spline_resample(ndarray im_resampled, ndarray im, double* Tvox)


cdef extern from "cubic_spline.h":
    
    void cubic_spline_import_array()
    cubic_spline_transform(ndarray res, ndarray src)
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


def _joint_histogram(ndarray H, flatiter iterI, ndarray imJ, ndarray Tvox, int interp):

    """
    joint_hist(H, imI, imJ, Tvox, subsampling, corner, size)
    Comments to follow.
    """
    cdef double *h, *tvox
    cdef int clampI, clampJ

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
    similarity(H, hI, hJ).
    Comments to follow
    """
    cdef int isF = 0
    cdef double *h, *hI, *hJ, *f=NULL
    cdef double simi = 0.0
    cdef int clampI, clampJ

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
        simi = correlation_coefficient(h, clampI, clampJ)
    elif simitype == CORRELATION_RATIO: 
        simi = correlation_ratio(h, clampI, clampJ) 
    elif simitype == CORRELATION_RATIO_L1:
        simi = correlation_ratio_L1(h, hI, clampI, clampJ) 
    elif simitype == MUTUAL_INFORMATION: 
        simi = mutual_information(h, hI, clampI, hJ, clampJ) 
    elif simitype == JOINT_ENTROPY:
        simi = joint_entropy(h, clampI, clampJ) 
    elif simitype == CONDITIONAL_ENTROPY:
        simi = conditional_entropy(h, hJ, clampI, clampJ) 
    elif simitype == NORMALIZED_MUTUAL_INFORMATION:
        simi = normalized_mutual_information(h, hI, clampI, hJ, clampJ) 
    elif simitype == SUPERVISED_MUTUAL_INFORMATION:
        simi = supervised_mutual_information(h, f, hI, clampI, hJ, clampJ)
    else:
        simi = 0.0
        
    return simi




def resample(ndarray im, dims, ndarray Tvox, datatype=None):
    """
    Resample(im, dims, Tvox, datatype=None)

    Note that the input transformation Tvox will be re-ordered in C
    convention if needed.
    """
    cdef double *tvox
    
    # Create output array
    if datatype == None:
        datatype = im.dtype
    im_resampled = np.zeros(tuple(dims)).astype(datatype)

    # Ensure that the Tvox array is C-contiguous (required by the
    # underlying fff routine)
    Tvox = np.asarray(Tvox, order='C')
    tvox = <double*>Tvox.data

    # Actual resampling 
    cubic_spline_resample(im_resampled, im, tvox)

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
    RIGID2D, SIMILARITY2D, AFFINE2D,
    RIGID3D, SIMILARITY3D, AFFINE3D

# Corresponding Python constants 
transformation_types = {'rigid 2D': RIGID2D,
                        'similarity 2D': SIMILARITY2D,
                        'affine 2D': AFFINE2D, 
                        'rigid 3D': RIGID3D,
                        'similarity 3D': SIMILARITY3D,
                        'affine 3D': AFFINE3D}

def rotation_vector_to_matrix(r):

    """
    R = rotation_vector_to_matrix(r)

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
        t[0:6] = param*precond[0:6]
    elif stamp == SIMILARITY3D:
        t[0:9] = param[[0,1,2,3,4,5,6,6,6]]*precond[0:9]
    elif stamp == AFFINE3D:
        t = param*precond
    elif stamp == RIGID2D:
        t[[0,1,5]] = param*precond[[0,1,5]]
    elif stamp == SIMILARITY2D:
        t[[0,1,5,6,7]] = param[[0,1,2,3,3]]*precond[[0,1,5,6,7]]
    elif stamp == AFFINE2D:
        t[[0,1,5,6,7,11]] = param*precond[[0,1,5,6,7,11]]
    
    return t
    

def matrix44(ndarray t):
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

    T = np.eye(4)
    R = rotation_vector_to_matrix(t[3:6])
    if size == 6:
        T[0:3,0:3] = R
    elif size == 7:
        T[0:3,0:3] = t[6]*R
    else:
        S = np.diag(t[6:9]) 
        Q = rotation_vector_to_matrix(t[9:12]) 
        T[0:3,0:3] = np.dot(Q,np.dot(S,R))
    T[0:3,3] = t[0:3] 
    return T 



