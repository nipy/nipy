# -*- Mode: Python -*-  Not really, but the syntax is close enough

"""
Python access to core fff functions written in C. This module is
mainly used for unitary tests.

Author: Alexis Roche, 2008.
"""

__version__ = '0.1'

# Include fff
from fff cimport *

# Exports from fff_blas.h
cdef extern from "fff_blas.h":

    ctypedef enum CBLAS_TRANSPOSE_t:
        CblasNoTrans=111
        CblasTrans=112
        CblasConjTrans=113
            
    ctypedef enum CBLAS_UPLO_t:
        CblasUpper=121
        CblasLower=122
                
    ctypedef enum CBLAS_DIAG_t:
        CblasNonUnit=131
        CblasUnit=132
                    
    ctypedef enum CBLAS_SIDE_t:
        CblasLeft=141
        CblasRight=142

    ## BLAS level 1
    double fff_blas_ddot(fff_vector * x, fff_vector * y) 
    double fff_blas_dnrm2(fff_vector * x) 
    double fff_blas_dasum(fff_vector * x)
    size_t fff_blas_idamax(fff_vector * x)
    int fff_blas_dswap(fff_vector * x, fff_vector * y) 
    fff_blas_dcopy(fff_vector * x, fff_vector * y) 
    int fff_blas_daxpy(double alpha, fff_vector * x, fff_vector * y) 
    int fff_blas_dscal(double alpha, fff_vector * x) 
    int fff_blas_drot(fff_vector * x, fff_vector * y, double c, double s) 

    ## BLAS level 2
    int fff_blas_dgemv(CBLAS_TRANSPOSE_t TransA, double alpha, 
                       fff_matrix * A,  fff_vector * x, double beta, fff_vector * y) 
    int fff_blas_dtrmv(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA, CBLAS_DIAG_t Diag, 
                       fff_matrix * A, fff_vector * x) 
    int fff_blas_dtrsv(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA, CBLAS_DIAG_t Diag,
                       fff_matrix * A, fff_vector * x) 
    int fff_blas_dsymv(CBLAS_UPLO_t Uplo, 
                       double alpha,  fff_matrix * A, 
                       fff_vector * x, double beta, fff_vector * y) 
    int fff_blas_dger(double alpha,  fff_vector * x,  fff_vector * y, fff_matrix * A)
    int fff_blas_dsyr(CBLAS_UPLO_t Uplo, double alpha,  fff_vector * x, fff_matrix * A) 
    int fff_blas_dsyr2(CBLAS_UPLO_t Uplo, double alpha, 
                       fff_vector * x,  fff_vector * y, fff_matrix * A) 

    ## BLAS level 3
    int fff_blas_dgemm(CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB, 
                       double alpha,  fff_matrix * A, 
                       fff_matrix * B, double beta, 
                       fff_matrix * C) 
    int fff_blas_dsymm(CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, 
                       double alpha,  fff_matrix * A, 
                       fff_matrix * B, double beta,
                       fff_matrix * C)
    int fff_blas_dtrmm(CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, 
                       CBLAS_TRANSPOSE_t TransA, CBLAS_DIAG_t Diag, 
                       double alpha,  fff_matrix * A, fff_matrix * B)
    int fff_blas_dtrsm(CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, 
                       CBLAS_TRANSPOSE_t TransA, CBLAS_DIAG_t Diag, 
                       double alpha,  fff_matrix * A, fff_matrix * B) 
    int fff_blas_dsyrk(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t Trans, 
                       double alpha,  fff_matrix * A, double beta, fff_matrix * C) 
    int fff_blas_dsyr2k(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t Trans, 
                        double alpha,  fff_matrix * A,  fff_matrix * B, 
                        double beta, fff_matrix * C) 


# Initialize numpy
fffpy_import_array()
import_array()
import numpy as np

# Binded routines

## fff_vector.h 
def vector_get(X, size_t i):
    """
    Get i-th element.
    xi = vector_get(x, i)
    """
    cdef fff_vector* x
    cdef double xi
    x = fff_vector_fromPyArray(X)
    xi = fff_vector_get(x, i)
    fff_vector_delete(x)
    return xi

def vector_set(X, size_t i, double a):
    """
    Set i-th element.
    vector_set(x, i, a)
    """
    cdef fff_vector *x, *y
    x = fff_vector_fromPyArray(X)
    y = fff_vector_new(x.size)
    fff_vector_memcpy(y, x)
    fff_vector_set(y, i, a)
    fff_vector_delete(x)
    Y = fff_vector_toPyArray(y)
    return Y

def vector_set_all(X, double a):
    """
    Set to a constant value.
    vector_set_all(x, a)
    """
    cdef fff_vector *x, *y
    x = fff_vector_fromPyArray(X)
    y = fff_vector_new(x.size)
    fff_vector_memcpy(y, x)
    fff_vector_set_all(y, a)
    fff_vector_delete(x)
    Y = fff_vector_toPyArray(y)
    return Y

def vector_scale(X, double a):
    """
    Multiply by a constant value.
    y = vector_scale(x, a)
    """
    cdef fff_vector *x, *y
    x = fff_vector_fromPyArray(X)
    y = fff_vector_new(x.size)
    fff_vector_memcpy(y, x)
    fff_vector_scale(y, a)
    fff_vector_delete(x)
    Y = fff_vector_toPyArray(y)
    return Y

def vector_add_constant(X, double a):
    """
    Add a constant value.
    y = vector_add_constant(x, a)
    """
    cdef fff_vector *x, *y
    x = fff_vector_fromPyArray(X)
    y = fff_vector_new(x.size)
    fff_vector_memcpy(y, x)
    fff_vector_add_constant(y, a)
    fff_vector_delete(x)
    Y = fff_vector_toPyArray(y)
    return Y

def vector_add(X, Y):
    """
    Add two vectors.
    z = vector_add(x, y)
    """
    cdef fff_vector *x, *y, *z
    x = fff_vector_fromPyArray(X)
    y = fff_vector_fromPyArray(Y)
    z = fff_vector_new(x.size)
    fff_vector_memcpy(z, x)
    fff_vector_add(z, y)
    fff_vector_delete(x)
    fff_vector_delete(y)
    Z = fff_vector_toPyArray(z)
    return Z

def vector_sub(X, Y):
    """
    Substract two vectors: x - y
    z = vector_sub(x, y)
    """
    cdef fff_vector *x, *y, *z
    x = fff_vector_fromPyArray(X)
    y = fff_vector_fromPyArray(Y)
    z = fff_vector_new(x.size)
    fff_vector_memcpy(z, x)
    fff_vector_sub(z, y)
    fff_vector_delete(x)
    fff_vector_delete(y)
    Z = fff_vector_toPyArray(z)
    return Z

def vector_mul(X, Y):
    """
    Element-wise multiplication.
    z = vector_mul(x, y)
    """
    cdef fff_vector *x, *y, *z
    x = fff_vector_fromPyArray(X)
    y = fff_vector_fromPyArray(Y)
    z = fff_vector_new(x.size)
    fff_vector_memcpy(z, x)
    fff_vector_mul(z, y)
    fff_vector_delete(x)
    fff_vector_delete(y)
    Z = fff_vector_toPyArray(z)
    return Z

def vector_div(X, Y):
    """
    Element-wise division.
    z = vector_div(x, y)
    """
    cdef fff_vector *x, *y, *z
    x = fff_vector_fromPyArray(X)
    y = fff_vector_fromPyArray(Y)
    z = fff_vector_new(x.size)
    fff_vector_memcpy(z, x)
    fff_vector_mul(z, y)
    fff_vector_delete(x)
    fff_vector_delete(y)
    Z = fff_vector_toPyArray(z)
    return Z


def vector_sum(X):
    """
    Sum up array elements.
    s = vector_sum(x)
    """
    cdef fff_vector* x
    cdef long double s
    x = fff_vector_fromPyArray(X)
    s = fff_vector_sum(x)
    fff_vector_delete(x)
    return s

def vector_ssd(X, double m=0, int fixed=1):
    """
    (Minimal) sum of squared differences.
    s = vector_ssd(x, m=0, fixed=1)
    """
    cdef fff_vector* x
    cdef long double s
    x = fff_vector_fromPyArray(X)
    s = fff_vector_ssd(x, &m, fixed)
    fff_vector_delete(x)
    return s

def vector_sad(X, double m=0):
    """
    Sum of absolute differences.
    s = vector_sad(x, m=0)
    """
    cdef fff_vector* x
    cdef long double s
    x = fff_vector_fromPyArray(X)
    s = fff_vector_sad(x, m)
    fff_vector_delete(x)
    return s

def vector_median(X):
    """
    Median.
    m = vector_median(x)
    """
    cdef fff_vector* x
    cdef double m
    x = fff_vector_fromPyArray(X)
    m = fff_vector_median(x)
    fff_vector_delete(x)
    return m

def vector_quantile(X, double r, int interp):
    """
    Quantile.
    q = vector_quantile(x, r=0.5, interp=1)
    """
    cdef fff_vector* x
    cdef double q
    x = fff_vector_fromPyArray(X)
    q = fff_vector_quantile(x, r, interp)
    fff_vector_delete(x)
    return q


## fff_matrix.h 
def matrix_get(A, size_t i, size_t j):
    """
    Get (i,j) element.
    aij = matrix_get(A, i, j)
    """
    cdef fff_matrix* a
    cdef double aij
    a = fff_matrix_fromPyArray(A)
    aij = fff_matrix_get(a, i, j)
    fff_matrix_delete(a)
    return aij

def matrix_transpose(A):
    """
    Transpose a matrix.
    B = matrix_transpose(A)
    """
    cdef fff_matrix *a, *b
    a = fff_matrix_fromPyArray(A)
    b = fff_matrix_new(a.size2, a.size1)
    fff_matrix_transpose(b, a)
    fff_matrix_delete(a)
    B = fff_matrix_toPyArray(b)
    return B

def matrix_add(A, B):
    """
    C = matrix_add(A, B)
    """
    cdef fff_matrix *a, *b, *c
    a = fff_matrix_fromPyArray(A)
    b = fff_matrix_fromPyArray(B)
    c = fff_matrix_new(a.size1, a.size2)
    fff_matrix_memcpy(c, a)
    fff_matrix_add(c, b)
    C = fff_matrix_toPyArray(c)
    return C


## fff_blas.h
cdef CBLAS_TRANSPOSE_t flag_transpose( int flag ):
    cdef CBLAS_TRANSPOSE_t x
    if flag <= 0:
        x = CblasNoTrans
    else:
        x = CblasTrans
    return x

cdef CBLAS_UPLO_t flag_uplo( int flag ):
    cdef CBLAS_UPLO_t x
    if flag <= 0:
        x = CblasUpper
    else:
        x = CblasLower
    return x

cdef CBLAS_DIAG_t flag_diag( int flag ):
    cdef CBLAS_DIAG_t x
    if flag <= 0:
        x = CblasNonUnit
    else:
        x = CblasUnit
    return x
                    
cdef CBLAS_SIDE_t flag_side( int flag ):
    cdef CBLAS_SIDE_t x
    if flag <= 0:
        x = CblasLeft
    else:
        x = CblasRight
    return x


### BLAS 1
def blas_dnrm2(X): 
    cdef fff_vector *x
    x = fff_vector_fromPyArray(X)
    return fff_blas_dnrm2(x)

def blas_dasum(X):
    cdef fff_vector *x
    x = fff_vector_fromPyArray(X)
    return fff_blas_dasum(x)

def blas_ddot(X, Y):
    cdef fff_vector *x, *y
    x = fff_vector_fromPyArray(X)
    y = fff_vector_fromPyArray(Y)
    return fff_blas_ddot(x, y)

def blas_daxpy(double alpha, X, Y): 
    cdef fff_vector *x, *y, *z
    x = fff_vector_fromPyArray(X)
    y = fff_vector_fromPyArray(Y)
    z = fff_vector_new(y.size)
    fff_vector_memcpy(z, y)
    fff_blas_daxpy(alpha, x, z) 
    Z = fff_vector_toPyArray(z)
    return Z

def blas_dscal(double alpha, X):
    cdef fff_vector *x, *y
    x = fff_vector_fromPyArray(X)
    y = fff_vector_new(x.size)
    fff_vector_memcpy(y, x)
    fff_blas_dscal(alpha, y) 
    Y = fff_vector_toPyArray(y)
    return Y
    


### BLAS 3   
def blas_dgemm(int TransA, int TransB, double alpha, A, B, double beta, C):
    """
    D = blas_dgemm(int TransA, int TransB, double alpha, A, B, double beta, C).
    
    Compute the matrix-matrix product and sum D = alpha op(A) op(B) +
    beta C where op(A) = A, A^T, A^H for TransA = CblasNoTrans,
    CblasTrans, CblasConjTrans and similarly for the parameter TransB.
    """
    cdef fff_matrix *a, *b, *c, *d
    a = fff_matrix_fromPyArray(A)
    b = fff_matrix_fromPyArray(B)
    c = fff_matrix_fromPyArray(C)
    d = fff_matrix_new(c.size1, c.size2)
    fff_matrix_memcpy(d, c)
    fff_blas_dgemm(flag_transpose(TransA), flag_transpose(TransB), alpha, a, b, beta, d)
    fff_matrix_delete(a)
    fff_matrix_delete(b)
    fff_matrix_delete(c)
    D = fff_matrix_toPyArray(d)
    return D


def blas_dsymm(int Side, int Uplo, double alpha, A, B, beta, C):
    """
    D = blas_dsymm(int Side, int Uplo, double alpha, A, B, beta, C).
    
    Compute the matrix-matrix product and sum C = \alpha A B + \beta C
    for Side is CblasLeft and C = \alpha B A + \beta C for Side is
    CblasRight, where the matrix A is symmetric. When Uplo is
    CblasUpper then the upper triangle and diagonal of A are used, and
    when Uplo is CblasLower then the lower triangle and diagonal of A
    are used.
    """
    cdef fff_matrix *a, *b, *c, *d
    a = fff_matrix_fromPyArray(A)
    b = fff_matrix_fromPyArray(B)
    c = fff_matrix_fromPyArray(C)
    d = fff_matrix_new(c.size1, c.size2)
    fff_matrix_memcpy(d, c)
    fff_blas_dsymm(flag_side(Side), flag_uplo(Uplo), alpha, a, b, beta, d)
    fff_matrix_delete(a)
    fff_matrix_delete(b)
    fff_matrix_delete(c)
    D = fff_matrix_toPyArray(d)
    return D

def blas_dtrmm(int Side, int Uplo, int TransA, int Diag, double alpha, A, B):
    """
    C = blas_dtrmm(int Side, int Uplo, int TransA, int Diag, double alpha, A, B).
    
    Compute the matrix-matrix product B = \alpha op(A) B for Side
    is CblasLeft and B = \alpha B op(A) for Side is CblasRight. The
    matrix A is triangular and op(A) = A, A^T, A^H for TransA =
    CblasNoTrans, CblasTrans, CblasConjTrans. When Uplo is CblasUpper
    then the upper triangle of A is used, and when Uplo is CblasLower
    then the lower triangle of A is used. If Diag is CblasNonUnit then
    the diagonal of A is used, but if Diag is CblasUnit then the
    diagonal elements of the matrix A are taken as unity and are not
    referenced.
    """
    cdef fff_matrix *a, *b, *c
    a = fff_matrix_fromPyArray(A)
    b = fff_matrix_fromPyArray(B)
    c = fff_matrix_new(a.size1, a.size2)
    fff_matrix_memcpy(c, b)
    fff_blas_dtrmm(flag_side(Side), flag_uplo(Uplo), flag_transpose(TransA), flag_diag(Diag),
                   alpha, a, c)
    fff_matrix_delete(a)
    fff_matrix_delete(b)
    C = fff_matrix_toPyArray(c)
    return C


def blas_dtrsm(int Side, int Uplo, int TransA, int Diag, double alpha, A, B):
    """
    blas_dtrsm(int Side, int Uplo, int TransA, int Diag, double alpha, A, B).
    
    Compute the inverse-matrix matrix product B = \alpha
    op(inv(A))B for Side is CblasLeft and B = \alpha B op(inv(A)) for
    Side is CblasRight. The matrix A is triangular and op(A) = A, A^T,
    A^H for TransA = CblasNoTrans, CblasTrans, CblasConjTrans. When
    Uplo is CblasUpper then the upper triangle of A is used, and when
    Uplo is CblasLower then the lower triangle of A is used. If Diag
    is CblasNonUnit then the diagonal of A is used, but if Diag is
    CblasUnit then the diagonal elements of the matrix A are taken as
    unity and are not referenced.
    """
    cdef fff_matrix *a, *b, *c
    a = fff_matrix_fromPyArray(A)
    b = fff_matrix_fromPyArray(B)
    c = fff_matrix_new(a.size1, a.size2)
    fff_matrix_memcpy(c, b)
    fff_blas_dtrsm(flag_side(Side), flag_uplo(Uplo), flag_transpose(TransA), flag_diag(Diag),
                   alpha, a, c)
    fff_matrix_delete(a)
    fff_matrix_delete(b)
    C = fff_matrix_toPyArray(c)
    return C


def blas_dsyrk(int Uplo, int Trans, double alpha, A, double beta, C):
    """
    D = blas_dsyrk(int Uplo, int Trans, double alpha, A, double beta, C).
    
    Compute a rank-k update of the symmetric matrix C, C = \alpha A
    A^T + \beta C when Trans is CblasNoTrans and C = \alpha A^T A +
    \beta C when Trans is CblasTrans. Since the matrix C is symmetric
    only its upper half or lower half need to be stored. When Uplo is
    CblasUpper then the upper triangle and diagonal of C are used, and
    when Uplo is CblasLower then the lower triangle and diagonal of C
    are used.
    """
    cdef fff_matrix *a, *c, *d
    a = fff_matrix_fromPyArray(A)
    c = fff_matrix_fromPyArray(C)
    d = fff_matrix_new(a.size1, a.size2)
    fff_matrix_memcpy(d, c)
    fff_blas_dsyrk(flag_uplo(Uplo), flag_transpose(Trans), alpha, a, beta, d)
    fff_matrix_delete(a)
    fff_matrix_delete(c)
    D = fff_matrix_toPyArray(d)
    return D


def blas_dsyr2k(int Uplo, int Trans, double alpha, A, B, double beta, C):
    """
    Compute a rank-2k update of the symmetric matrix C, C = \alpha A B^T +
    \alpha B A^T + \beta C when Trans is CblasNoTrans and C = \alpha A^T B
    + \alpha B^T A + \beta C when Trans is CblasTrans. Since the matrix C
    is symmetric only its upper half or lower half need to be stored. When
    Uplo is CblasUpper then the upper triangle and diagonal of C are used,
    and when Uplo is CblasLower then the lower triangle and diagonal of C
    are used.
    """
    cdef fff_matrix *a, *b, *c, *d
    a = fff_matrix_fromPyArray(A)
    b = fff_matrix_fromPyArray(B)
    c = fff_matrix_fromPyArray(C)
    d = fff_matrix_new(a.size1, a.size2)
    fff_matrix_memcpy(d, c)
    fff_blas_dsyr2k(flag_uplo(Uplo), flag_transpose(Trans), alpha, a, b, beta, d)
    fff_matrix_delete(a)
    fff_matrix_delete(b)
    fff_matrix_delete(c)
    D = fff_matrix_toPyArray(d)
    return D



