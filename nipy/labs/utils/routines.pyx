# -*- Mode: Python -*-  Not really, but the syntax is close enough

"""
Miscellaneous fff routines.

Author: Alexis Roche, 2008.
"""

__version__ = '0.1'

# Includes
from fff cimport *
cimport numpy as cnp

# Exports from fff_gen_stats.h
cdef extern from "fff_gen_stats.h":

    double fff_mahalanobis(fff_vector* x, fff_matrix* S, fff_matrix* Saux)
    void fff_permutation(unsigned int* x, unsigned int n,
                         unsigned long int magic)
    void fff_combination(unsigned int* x, unsigned int k, unsigned int n,
                         unsigned long magic)

# Exports from fff_specfun.h
cdef extern from "fff_specfun.h":

    extern double fff_gamln(double x)
    extern double fff_psi(double x)

    
# Exports from fff_lapack.h
cdef extern from "fff_lapack.h":

    extern int fff_lapack_dgesdd(fff_matrix* A, fff_vector* s, fff_matrix* U, fff_matrix* Vt, 
                                 fff_vector* work, fff_array* iwork, fff_matrix* Aux)


# Initialize numpy
fffpy_import_array()
cnp.import_array()
import numpy as np

# This is faster than scipy.stats.scoreatpercentile due to partial
# sorting
def quantile(X, double ratio, int interp=False, int axis=0):
    """
    q = quantile(data, ratio, interp=False, axis=0).

    Partial sorting algorithm, very fast!!!
    """
    cdef fff_vector *x, *y
    cdef fffpy_multi_iterator* multi

    # Allocate output array Y
    dims = list(X.shape)
    dims[axis] = 1
    Y = np.zeros(dims)

    # Create a new array iterator 
    multi = fffpy_multi_iterator_new(2, axis, <void*>X, <void*>Y)

    # Create vector views on both X and Y 
    x = multi.vector[0]
    y = multi.vector[1]

    # Loop 
    while(multi.index < multi.size):
        y.data[0] = fff_vector_quantile(x, ratio, interp) 
        fffpy_multi_iterator_update(multi)
       
    # Delete local structures
    fffpy_multi_iterator_delete(multi)
    return Y

# This is faster than numpy.stats
# due to the underlying algorithm that relies on
# partial sorting as opposed to full sorting.
def median(x, axis=0):
    """
    median(x, axis=0).
    Equivalent to: quantile(x, ratio=0.5, interp=True, axis=axis).
    """
    return quantile(x, axis=axis, ratio=0.5, interp=True)


def mahalanobis(X, VX):
    """
    d2 = mahalanobis(X, VX).

    ufunc-like function to compute Mahalanobis squared distances
    x'*inv(Vx)*x.  

    axis == 0 assumed. If X is shaped (d,K), VX must be shaped
    (d,d,K).
    """
    cdef fff_vector *x, *vx, *x_tmp, *vx_tmp, *d2
    cdef fff_matrix Sx 
    cdef fff_matrix *Sx_tmp
    cdef fffpy_multi_iterator* multi
    cdef int axis=0, n

    # Allocate output array
    dims = list(X.shape)
    dim = dims[0]
    dims[0] = 1
    D2 = np.zeros(dims)

    # Flatten input variance array
    VX_flat = VX.reshape( [dim*dim]+list(VX.shape[2:]) )

    # Create a new array iterator 
    multi = fffpy_multi_iterator_new(3, axis, <void*>X, <void*>VX_flat, <void*>D2)

    # Allocate local structures 
    n = <int> X.shape[axis]
    x_tmp = fff_vector_new(n)
    vx_tmp = fff_vector_new(n*n) 
    Sx_tmp = fff_matrix_new(n, n)

    # Create vector views on X, VX_flat and D2 
    x = multi.vector[0]
    vx = multi.vector[1] 
    d2 = multi.vector[2] 

    # Loop 
    while(multi.index < multi.size):
        fff_vector_memcpy(x_tmp, x)
        fff_vector_memcpy(vx_tmp, vx) 
        Sx = fff_matrix_view(vx_tmp.data, n, n, n) # OK because vx_tmp is contiguous  
        d2.data[0] = fff_mahalanobis(x_tmp, &Sx, Sx_tmp)
        fffpy_multi_iterator_update(multi)

    # Delete local structs and views
    fff_vector_delete(x_tmp)
    fff_vector_delete(vx_tmp) 
    fff_matrix_delete(Sx_tmp)
    fffpy_multi_iterator_delete(multi)

    # Return
    D2 = D2.reshape(VX.shape[2:])
    return D2 


def svd(X):
    """ Singular value decomposition of array `X`

    Y = svd(X)

    ufunc-like svd. Given an array X (m, n, K), perform an SV decomposition.

    Parameters
    ----------
    X : 2D array

    Returns
    -------
    S : (min(m,n), K)
    """
    cdef int axis=0
    cdef int m, n, dmin, dmax, lwork, liwork, info
    cdef fff_vector *work, *x_flat, *x_flat_tmp, *s, *s_tmp
    cdef fff_matrix x
    cdef fff_array *iwork
    cdef fff_matrix *Aux, *U, *Vt 
    cdef fffpy_multi_iterator* multi

    # Shape of matrices
    m = <int> X.shape[0]
    n = <int> X.shape[1]
    if m > n:
        dmin = n
        dmax = m
    else:
        dmin = m
        dmax = n

    # Create auxiliary arrays
    lwork = 4*dmin*(dmin+1)
    if dmax > lwork:
        lwork = dmax
    lwork = 2*(3*dmin*dmin + lwork)
    liwork = 8*dmin
    work = fff_vector_new(lwork)
    iwork = fff_array_new1d(FFF_INT, liwork)
    Aux = fff_matrix_new(dmax, dmax)
    U = fff_matrix_new(m, m)
    Vt = fff_matrix_new(n, n)
    x_flat_tmp = fff_vector_new(m*n)
    s_tmp = fff_vector_new(dmin)

    # Allocate output array
    endims = list(X.shape[2:])
    S = np.zeros([dmin]+endims)

    # Flatten input array
    X_flat = X.reshape([m*n]+endims)

    # Create a new array iterator
    multi = fffpy_multi_iterator_new(2, axis, <void*>X_flat, <void*>S)

    # Create vector views
    x_flat = multi.vector[0]
    s = multi.vector[1]

    # Loop
    while(multi.index < multi.size):
        fff_vector_memcpy(x_flat_tmp, x_flat)
        fff_vector_memcpy(s_tmp, s)
        x = fff_matrix_view(x_flat_tmp.data, m, n, n) # OK because x_flat_tmp is contiguous
        info = fff_lapack_dgesdd(&x, s_tmp, U, Vt, work, iwork, Aux )
        fff_vector_memcpy(s, s_tmp)
        fffpy_multi_iterator_update(multi)

    # Delete local structures
    fff_vector_delete(work)
    fff_vector_delete(x_flat_tmp)
    fff_vector_delete(s_tmp)
    fff_array_delete(iwork)
    fff_matrix_delete(Aux)
    fff_matrix_delete(U)
    fff_matrix_delete(Vt)
    fffpy_multi_iterator_delete(multi)

    # Return
    return S


def permutations(unsigned int n, unsigned int m=1, unsigned long magic=0):
    """
    P = permutations(n, m=1, magic=0).
    Generate m permutations from [0..n[.
    """
    cdef fff_array *p, *pi
    cdef fff_array pi_view
    cdef unsigned int i
    p = fff_array_new2d(FFF_UINT, n, m)
    pi = fff_array_new1d(FFF_UINT, n) ## contiguous, dims=(n,1,1,1)

    for i from 0 <= i < m:
        fff_permutation(<unsigned int*>pi.data, n, magic+i)
        pi_view = fff_array_get_block2d(p, 0, n-1, 1, i, i, 1) ## dims=(n,1,1,1)
        fff_array_copy(&pi_view, pi)

    P = fff_array_toPyArray(p)
    return P 


def combinations(unsigned int k, unsigned int n, unsigned int m=1, unsigned long magic=0):
    """
    P = combinations(k, n, m=1, magic=0).
    Generate m combinations of k elements  from [0..n[.
    """
    cdef fff_array *p, *pi
    cdef fff_array pi_view
    cdef unsigned int i
    p = fff_array_new2d(FFF_UINT, k, m)
    pi = fff_array_new1d(FFF_UINT, k) ## contiguous, dims=(n,1,1,1)

    for i from 0 <= i < m:
        fff_combination(<unsigned int*>pi.data, k, n, magic+i)
        pi_view = fff_array_get_block2d(p, 0, k-1, 1, i, i, 1) ## dims=(k,1,1,1)
        fff_array_copy(&pi_view, pi)

    C = fff_array_toPyArray(p)
    return C


def gamln(double x):
    """ Python bindings to log gamma. Do not use, this is there only for
        testing. Use scipy.special.gammaln.
    """
    cdef double y
    y = fff_gamln(x)
    return y 


def psi(double x):
    """ Python bindings to psi (d gamln(x)/dx. Do not use, this is there only 
        for testing. Use scipy.special.psi.
    """
    cdef double y
    y = fff_psi(x)
    return y

