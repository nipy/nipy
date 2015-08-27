# -*- Mode: Python -*-  Not really, but the syntax is close enough
"""
Two-level general linear model for group analyses.

Author: Alexis Roche, 2008.
"""

__version__ = '0.1'

# Includes
from fff cimport *

# Exports from fff_glm_twolevel.h 
cdef extern from "fff_glm_twolevel.h":

    ctypedef struct fff_glm_twolevel_EM:
        fff_vector* b
        double s2
        fff_vector* z
        fff_vector* vz

    fff_glm_twolevel_EM* fff_glm_twolevel_EM_new(size_t n, size_t p)
    void fff_glm_twolevel_EM_delete(fff_glm_twolevel_EM* thisone)
    void fff_glm_twolevel_EM_init(fff_glm_twolevel_EM* em)
    void fff_glm_twolevel_EM_run(fff_glm_twolevel_EM* em,
                                 fff_vector* y,
                                 fff_vector* vy, 
                                 fff_matrix* X,
                                 fff_matrix* PpiX,
                                 unsigned int niter)
    double fff_glm_twolevel_log_likelihood(fff_vector* y,
                                           fff_vector* vy,
                                           fff_matrix* X, 
                                           fff_vector* b,
                                           double s2,
                                           fff_vector* tmp)

# Initialize numpy
fffpy_import_array()
import_array()
import numpy as np

# Constants
DEF_NITER = 2

def em(ndarray Y, ndarray VY, ndarray X, ndarray C=None, int axis=0, int niter=DEF_NITER):
    """
    b, s2 = em(y, vy, X, C=None, axis=0, niter=DEF_NITER).

    Maximum likelihood regression in a mixed-effect GLM using the
    EM algorithm.

    C is the contrast matrix. Conventionally, C is p x q where p
    is the number of regressors. 
    
    OUTPUT: beta, s2
    beta -- array of parameter estimates
    s2 -- array of squared scale parameters.
    
    REFERENCE:
    Keller and Roche, ISBI 2008.
    """
    cdef size_t n, p
    cdef fff_vector *y, *vy, *b, *s2
    cdef fff_matrix *x, *ppx
    cdef fff_glm_twolevel_EM *em
    cdef fffpy_multi_iterator* multi

    # View on design matrix
    x = fff_matrix_fromPyArray(X)

    # Number of observations / regressors 
    n = x.size1
    p = x.size2

    # Compute the projected pseudo-inverse matrix
    if C is None:
        PpX =  np.linalg.pinv(X)
    else:
        A = np.linalg.inv(np.dot(X.transpose(), X)) # (p,p)
        B = np.linalg.inv(np.dot(np.dot(C.transpose(), A), C)) # (q,q)
        P = np.eye(p) - np.dot(np.dot(np.dot(A, C), B), C.transpose()) # (p,p)
        PpX = np.dot(np.dot(P, A), X.transpose()) # (p,n)
    ppx = fff_matrix_fromPyArray(PpX)

    # Allocate output arrays
    dims = [Y.shape[i] for i in range(Y.ndim)]
    dims[axis] = p
    B = np.zeros(dims)
    dims[axis] = 1
    S2 = np.zeros(dims)

    # Local structs
    em = fff_glm_twolevel_EM_new(n, p)

    # Create a new array iterator 
    multi = fffpy_multi_iterator_new(4, axis, <void*>Y, <void*>VY,
                     <void*>B, <void*>S2)

    # Create views
    y = multi.vector[0]
    vy = multi.vector[1]
    b = multi.vector[2]
    s2 = multi.vector[3]

        # Loop 
    while(multi.index < multi.size):
        fff_glm_twolevel_EM_init(em)
        fff_glm_twolevel_EM_run(em, y, vy, x, ppx, niter)
        fff_vector_memcpy(b, em.b)
        s2.data[0] = em.s2
        fffpy_multi_iterator_update(multi)
    
    # Free memory
    fff_matrix_delete(x)
    fff_matrix_delete(ppx)
    fffpy_multi_iterator_delete(multi)
    fff_glm_twolevel_EM_delete(em)

    # Return
    return B, S2




def log_likelihood(Y, VY, X, B, S2, int axis=0):
    """
    ll = log_likelihood(y, vy, X, b, s2, axis=0)
    Log likelihood in a mixed-effect GLM.
    OUTPUT: array
    REFERENCE:
    Keller and Roche, ISBI 2008.
    """
    cdef fff_vector *y, *vy, *b, *s2, *ll, *tmp
    cdef fff_matrix *x
    cdef fffpy_multi_iterator* multi
    
    # Allocate output array
    dims = [Y.shape[i] for i in range(Y.ndim)]
    dims[axis] = 1
    LL = np.zeros(dims)

    # View on design matrix
    x = fff_matrix_fromPyArray(X)

    # Local structure
    tmp = fff_vector_new(x.size1)

    # Multi iterator 
    multi = fffpy_multi_iterator_new(5, axis, <void*>Y, <void*>VY,
                     <void*>B, <void*>S2, <void*>LL)

    # View on iterable arrays
    y = multi.vector[0]
    vy = multi.vector[1]
    b = multi.vector[2]
    s2 = multi.vector[3]
    ll = multi.vector[4]

    # Loop 
    while(multi.index < multi.size):
        ll.data[0] = fff_glm_twolevel_log_likelihood(y, vy, x, b, s2.data[0], tmp)
        fffpy_multi_iterator_update(multi)

    # Free memory
    fff_matrix_delete(x)
    fff_vector_delete(tmp)
    fffpy_multi_iterator_delete(multi)

    # Return
    return LL


def log_likelihood_ratio(Y, VY, X, C, int axis=0, int niter=DEF_NITER):
    """
    lda = em(y, vy, X, C, axis=0, niter=DEF_NITER).
    """

    # Constrained log-likelihood
    B, S2 = em(Y, VY, X, C, axis, niter)
    ll0 = log_likelihood(Y, VY, X, B, S2, axis)

    # Unconstrained log-likelihood
    B, S2 = em(Y, VY, X, None, axis, niter)
    ll = log_likelihood(Y, VY, X, B, S2, axis)
    
    # -2 log R = 2*(ll-ll0)
    lda = 2*(ll-ll0)
    return np.maximum(lda, 0.0)
