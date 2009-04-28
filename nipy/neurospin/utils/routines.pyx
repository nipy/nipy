# -*- Mode: Python -*-  Not really, but the syntax is close enough

"""
Miscellaneous fff routines.

Author: Alexis Roche, 2008.
"""

__version__ = '0.1'

# Includes from the python headers
include "fff.pxi"

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

    

# Initialize numpy
fffpy_import_array()
import_array()
import numpy as np

# FIXME: Check that this is faster than scipy.stats.scoreatpercentile
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

# FIXME: Why this and not the numpy median? Check that this is faster.
def median(x, axis=0):
    """
    median(x, axis=0).
    Equivalent to: quantile(x, ratio=0.5, interp=True, axis=axis).
    """
    return quantile(x, axis=axis, ratio=0.5, interp=True)

    
def mahalanobis(X, VX):
    """
    d2 = mahalanobis(X, VX).
    axis == 0 assumed. If X is shaped (d,K), VX must be shaped (d,d,K). 
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

