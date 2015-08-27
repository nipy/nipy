# -*- Mode: Python -*-  Not really, but the syntax is close enough


"""
Incremental (Kalman-like) filters for linear regression. 

Author: Alexis Roche, 2008.
"""

__version__ = '0.1'

# Includes
from fff cimport *

# Exports from fff_glm_kalman.h
cdef extern from "fff_glm_kalman.h":

    ctypedef struct fff_glm_KF:
        size_t t
        size_t dim
        fff_vector* b
        fff_matrix* Vb
        double ssd
        double s2
        double dof
        double s2_corr

    ctypedef struct fff_glm_RKF:
        size_t t
        size_t dim
        fff_glm_KF* Kfilt
        fff_vector* db
        fff_matrix* Hssd
        double spp
        fff_vector* Gspp
        fff_matrix* Hspp
        fff_vector* b
        fff_matrix* Vb
        double s2
        double a
        double dof
        double s2_cor
        fff_vector* vaux
        fff_matrix* Maux

    fff_glm_KF* fff_glm_KF_new(size_t dim)
    void fff_glm_KF_delete(fff_glm_KF* thisone)
    void fff_glm_KF_reset(fff_glm_KF* thisone)
    void fff_glm_KF_iterate(fff_glm_KF* thisone, double y, fff_vector* x)
    fff_glm_RKF* fff_glm_RKF_new(size_t dim)
    void fff_glm_RKF_delete(fff_glm_RKF* thisone)
    void fff_glm_RKF_reset(fff_glm_RKF* thisone)
    void fff_glm_RKF_iterate(fff_glm_RKF* thisone, unsigned int nloop, 
                             double y, fff_vector* x, 
                             double yy, fff_vector* xx)
    void fff_glm_KF_fit(fff_glm_KF* thisone, fff_vector* y, fff_matrix* X)
    void fff_glm_RKF_fit(fff_glm_RKF* thisone, unsigned int nloop, 
                         fff_vector* y, fff_matrix* X)


# Initialize numpy
fffpy_import_array()
import_array()
import numpy as np

# Standard Kalman filter

def ols(ndarray Y, ndarray X, int axis=0): 
    """
    (beta, norm_var_beta, s2, dof) = ols(Y, X, axis=0).

    Ordinary least-square multiple regression using the Kalman filter.
    Fit the N-dimensional array Y along the given axis in terms of the
    regressors in matrix X. The regressors must be stored columnwise.

    OUTPUT: a four-element tuple
    beta -- array of parameter estimates
    norm_var_beta -- normalized variance matrix of the parameter
    estimates (data independent)
    s2 -- array of squared scale
    parameters to multiply norm_var_beta for the variance matrix of
    beta.
    dof -- scalar degrees of freedom.

    REFERENCE:  Roche et al, ISBI 2004.
    """
    cdef fff_vector *y, *b, *s2
    cdef fff_matrix *x
    cdef fff_glm_KF *kfilt
    cdef size_t p
    cdef fffpy_multi_iterator* multi
    cdef double dof

    # View on design matrix
    x = fff_matrix_fromPyArray(X)

    # Number of regressors 
    p = x.size2
    
    # Allocate output arrays B and S2
    #
    # Using Cython cimport of numpy, Y.shape is a C array of npy_intp
    # type; see:
    # http://codespeak.net/pipermail/cython-dev/2009-April/005229.html
    dims = [Y.shape[i] for i in range(Y.ndim)]
    dims[axis] = p
    B = np.zeros(dims, dtype=np.double)
    dims[axis] = 1
    S2 = np.zeros(dims, dtype=np.double)

    # Allocate local structure
    kfilt = fff_glm_KF_new(p)

    # Create a new array iterator 
    multi = fffpy_multi_iterator_new(3, axis, <void*>Y, <void*>B, <void*>S2)

    # Create views
    y = multi.vector[0]
    b = multi.vector[1]
    s2 = multi.vector[2]

    # Loop 
    while(multi.index < multi.size):
        fff_glm_KF_fit(kfilt, y, x)
        fff_vector_memcpy(b, kfilt.b)
        s2.data[0] = kfilt.s2
        fffpy_multi_iterator_update(multi)

    # Normalized variance (computed from the last item)
    VB = fff_matrix_const_toPyArray(kfilt.Vb);    
    dof = kfilt.dof
    
    # Free memory
    fff_matrix_delete(x)
    fff_glm_KF_delete(kfilt)
    fffpy_multi_iterator_delete(multi)

    # Return
    return B, VB, S2, dof
    

def ar1(ndarray Y, ndarray X, int niter=2, int axis=0):
    """
    (beta, norm_var_beta, s2, dof, a) = ar1(Y, X, niter=2, axis=0)

    Refined Kalman filter -- enhanced Kalman filter to account for
    noise autocorrelation using an AR(1) model. Pseudo-likelihood
    multiple regression using the refined Kalman filter, a Kalman
    variant based on a AR(1) error model.  Fit the N-dimensional array
    Y along the given axis in terms of the regressors in matrix X. The
    regressors must be stored columnwise.

    OUTPUT: a five-element tuple
    beta -- array of parameter estimates
    norm_var_beta -- array of normalized variance matrices (which are data dependent
    unlike in standard OLS regression)
    s2 -- array of squared scale parameters to multiply norm_var_beta for the variance matrix of beta.
    dof -- scalar degrees of freedom
    a -- array of error autocorrelation estimates

    REFERENCE:
    Roche et al, MICCAI 2004.
    """
    cdef fff_vector *y, *b, *vb, *s2, *a
    cdef fff_vector Vb_flat
    cdef fff_matrix *x
    cdef fff_glm_RKF *rkfilt
    cdef size_t p, p2
    cdef fffpy_multi_iterator* multi
    cdef double dof

    # View on design matrix
    x = fff_matrix_fromPyArray(X)

    # Number of regressors 
    p = x.size2
    p2 = p*p
    
    # Allocate output arrays B and S2.
    #
    # Using Cython cimport of numpy, Y.shape is a C array of npy_intp
    # type; see:
    # http://codespeak.net/pipermail/cython-dev/2009-April/005229.html
    dims = [Y.shape[i] for i in range(Y.ndim)]
    dims[axis] = p
    B = np.zeros(dims, dtype=np.double)
    dims[axis] = p2
    VB = np.zeros(dims, dtype=np.double)
    dims[axis] = 1
    S2 = np.zeros(dims, dtype=np.double)
    A = np.zeros(dims, dtype=np.double)
 
    # Allocate local structure
    rkfilt = fff_glm_RKF_new(p)

    # Create a new array iterator 
    multi = fffpy_multi_iterator_new(5, axis, <void*>Y, <void*>B, <void*>VB, <void*>S2, <void*>A)

    # Create views
    y = multi.vector[0]
    b = multi.vector[1]
    vb = multi.vector[2]
    s2 = multi.vector[3]
    a = multi.vector[4]

    # Loop 
    while(multi.index < multi.size):
        fff_glm_RKF_fit(rkfilt, niter, y, x)
        fff_vector_memcpy(b, rkfilt.b)
        Vb_flat = fff_vector_view(rkfilt.Vb.data, p2, 1) # rkfilt.Vb contiguous by construction
        fff_vector_memcpy(vb, &Vb_flat)
        s2.data[0] = rkfilt.s2
        a.data[0] = rkfilt.a
        fffpy_multi_iterator_update(multi)

    # Dof 
    dof = rkfilt.dof
    
    # Free memory
    fff_matrix_delete(x)
    fff_glm_RKF_delete(rkfilt) 
    fffpy_multi_iterator_delete(multi)
    
    # Reshape variance array
    dims[axis] = p
    dims.insert(axis+1, p)
    VB = VB.reshape(dims)

    # Return
    return B, VB, S2, dof, A
    



