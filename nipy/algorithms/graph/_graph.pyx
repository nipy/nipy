import numpy as np
cimport numpy as np
cimport cython
ctypedef np.float64_t DOUBLE
ctypedef np.int_t INT


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def dilation(np.ndarray[DOUBLE, ndim=2] field,\
             np.ndarray[INT, ndim=1] idx,\
             np.ndarray[INT, ndim=1] neighb):    
    cdef int size_max = field.shape[0]
    cdef int dim = field.shape[1]
    cdef int i, j, d
    cdef DOUBLE fmax
    cdef np.ndarray[DOUBLE, ndim=1] res = 0 * field[:, 0]
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
