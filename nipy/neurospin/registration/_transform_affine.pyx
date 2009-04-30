# -*- Mode: Python -*-  Not really, but the syntax is close enough


"""
Routines for affine transformation parameterization. 

Author: Alexis Roche, 2008.
"""

__version__ = '0.1'


# Includes
include "fff.pxi"

# Additional exports from fff_iconic_match.h
cdef extern from "fff_iconic_match.h":

    void fff_imatch_resample(fff_array* im_resampled, fff_array* im, double* Tvox) 


# Initialize numpy
fffpy_import_array()
import_array()
import numpy as np


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

def rvector_to_matrix(r):

    """
    R = rvector_to_matrix(r)

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
    R = rvector_to_matrix(t[3:6])
    if size == 6:
        T[0:3,0:3] = R
    elif size == 7:
        T[0:3,0:3] = t[6]*R
    else:
        S = np.diag(t[6:9]) 
        Q = rvector_to_matrix(t[9:12]) 
        T[0:3,0:3] = np.dot(Q,np.dot(S,R))
    T[0:3,3] = t[0:3] 
    return T 



def resample(ndarray Im, dims, ndarray Tvox, datatype=None):
    """
    Resample(im, dims, Tvox, datatype=None)

    Note that the input transformation Tvox will be re-ordered in C
    convention if needed.
    """
    cdef fff_array *im_resampled, *im
    cdef double *tvox
    
    # Create output array
    if datatype == None:
        datatype = Im.dtype
    Im_resampled = np.zeros(tuple(dims)).astype(datatype)

    # View on Python arrays 
    im_resampled = fff_array_fromPyArray(Im_resampled) 
    im = fff_array_fromPyArray(Im)

    # Ensure that the Tvox array is C-contiguous (required by the
    # underlying fff routine)
    Tvox = np.asarray(Tvox, order='C')
    tvox = <double*>Tvox.data

    # Actual resampling 
    fff_imatch_resample(im_resampled, im, tvox) 

    # Delete local structures 
    fff_array_delete(im_resampled) 
    fff_array_delete(im)

    return Im_resampled

