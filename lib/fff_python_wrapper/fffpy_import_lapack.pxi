# -*- Mode: Python -*-  Not really, but the syntax is close enough
from scipy.linalg._fblas import (ddot, dnrm2, dasum, idamax, dswap,
                                 dcopy, daxpy, dscal, drot, drotg,
                                 drotmg, drotm, dgemv, dtrmv, dsymv,
                                 dger, dgemm, dsymm, dsyrk, dsyr2k)
from scipy.linalg._flapack import (dgetrf, dpotrf, dpotrs, dgesdd, dgeqrf)

cdef extern from "fff_blas.h":
    ctypedef enum fff_blas_func_key:
        FFF_BLAS_DDOT = 0,
        FFF_BLAS_DNRM2 = 1,
        FFF_BLAS_DASUM = 2,
        FFF_BLAS_IDAMAX = 3,
        FFF_BLAS_DSWAP = 4,
        FFF_BLAS_DCOPY = 5,
        FFF_BLAS_DAXPY = 6,
        FFF_BLAS_DSCAL = 7,
        FFF_BLAS_DROT = 8,
        FFF_BLAS_DROTG = 9,
        FFF_BLAS_DROTMG = 10,
        FFF_BLAS_DROTM = 11,
        FFF_BLAS_DGEMV = 12,
        FFF_BLAS_DTRMV = 13,
        FFF_BLAS_DTRSV = 14,
        FFF_BLAS_DSYMV = 15,
        FFF_BLAS_DGER = 16,
        FFF_BLAS_DSYR = 17,
        FFF_BLAS_DSYR2 = 18,
        FFF_BLAS_DGEMM = 19,
        FFF_BLAS_DSYMM = 20,
        FFF_BLAS_DTRMM = 21,
        FFF_BLAS_DTRSM = 22,
        FFF_BLAS_DSYRK = 23,
        FFF_BLAS_DSYR2K = 24


cdef extern from "fff_lapack.h":
    ctypedef enum fff_lapack_func_key:
        FFF_LAPACK_DGETRF = 0,
        FFF_LAPACK_DPOTRF = 1,
        FFF_LAPACK_DPOTRS = 2,
        FFF_LAPACK_DGESDD = 3,
        FFF_LAPACK_DGEQRF = 4


cdef extern from "fffpy.h":
    void fffpy_import_blas_func(object ptr, int key)
    void fffpy_import_lapack_func(object ptr, int key)

def fffpy_import_lapack():
    fffpy_import_blas_func(ddot._cpointer, FFF_BLAS_DDOT)
    fffpy_import_blas_func(dnrm2._cpointer, FFF_BLAS_DNRM2)
    fffpy_import_blas_func(dasum._cpointer, FFF_BLAS_DASUM)
    fffpy_import_blas_func(idamax._cpointer, FFF_BLAS_IDAMAX)
    fffpy_import_blas_func(dswap._cpointer, FFF_BLAS_DSWAP)
    fffpy_import_blas_func(dcopy._cpointer, FFF_BLAS_DCOPY)
    fffpy_import_blas_func(daxpy._cpointer, FFF_BLAS_DAXPY)
    fffpy_import_blas_func(dscal._cpointer, FFF_BLAS_DSCAL)
    fffpy_import_blas_func(drot._cpointer, FFF_BLAS_DROT)
    fffpy_import_blas_func(drotg._cpointer, FFF_BLAS_DROTG)
    fffpy_import_blas_func(drotmg._cpointer, FFF_BLAS_DROTMG)
    fffpy_import_blas_func(drotm._cpointer, FFF_BLAS_DROTM)
    fffpy_import_blas_func(dgemv._cpointer, FFF_BLAS_DGEMV)
    fffpy_import_blas_func(dtrmv._cpointer, FFF_BLAS_DTRMV)
    fffpy_import_blas_func(dsymv._cpointer, FFF_BLAS_DSYMV)
    fffpy_import_blas_func(dger._cpointer, FFF_BLAS_DGER)
    fffpy_import_blas_func(dgemm._cpointer, FFF_BLAS_DGEMM)
    fffpy_import_blas_func(dsymm._cpointer, FFF_BLAS_DSYMM)
    fffpy_import_blas_func(dsyrk._cpointer, FFF_BLAS_DSYRK)
    fffpy_import_blas_func(dsyr2k._cpointer, FFF_BLAS_DSYR2K)
    fffpy_import_lapack_func(dgetrf._cpointer, FFF_LAPACK_DGETRF)
    fffpy_import_lapack_func(dpotrf._cpointer, FFF_LAPACK_DPOTRF)
    fffpy_import_lapack_func(dpotrs._cpointer, FFF_LAPACK_DPOTRS)
    fffpy_import_lapack_func(dgesdd._cpointer, FFF_LAPACK_DGESDD)
    fffpy_import_lapack_func(dgeqrf._cpointer, FFF_LAPACK_DGEQRF)
