cimport numpy as cnp
cimport cython
ctypedef cnp.float64_t DOUBLE
ctypedef cnp.int_t INT
ctypedef cnp.intp_t INT


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def dilation(cnp.ndarray[DOUBLE, ndim=2] field,\
             cnp.ndarray[INT, ndim=1] idx,\
             cnp.ndarray[INT, ndim=1] neighb):
    cdef int size_max = field.shape[0]
    cdef int dim = field.shape[1]
    cdef int i, j, d
    cdef DOUBLE fmax
    cdef cnp.ndarray[DOUBLE, ndim=1] res = 0 * field[:, 0]
    for d in range(dim):
        for i in range(size_max):
            fmax = field[i, d]
            for j in range(idx[i], idx[i + 1]):
                if field[neighb[j], d] > fmax:
                    fmax = field[neighb[j], d]
            res[i] = fmax
        for i in range(size_max):
            field[i, d] = res[i]
    return res
