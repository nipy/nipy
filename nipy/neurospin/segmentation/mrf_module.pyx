# -*- Mode: Python -*-  

"""
Markov random field utils. 
"""

__version__ = '0.0'

import numpy as np
cimport numpy as np

import cython 

# Globals 
cdef np.ndarray ngb26 = np.array(( 
        (1,0,0),(-1,0,0),(0,1,0),(0,-1,0), 
        (1,1,0),(-1,1,0),(-1,1,0),(-1,-1,0), 
        (1,0,1),(-1,0,1),(0,1,1),(0,-1,1), 
        (1,1,1),(-1,1,1),(-1,1,1),(-1,-1,1), 
        (1,0,-1),(-1,0,-1),(0,1,-1),(0,-1,-1), 
        (1,1,-1),(-1,1,-1),(-1,1,-1),(-1,-1,-1), 
        (0,0,1),(0,0,-1)))

"""
def neighbor_average(np.ndarray[np.double_t, ndim=4] im, 
                     np.ndarray[np.uint_t, ndim=1] pt):
"""
@cython.boundscheck(False)
cdef neighbor_average(np.ndarray res, 
                      np.ndarray ppm, 
                      unsigned int x,
                      unsigned int y, 
                      unsigned int z):
    """
    Compute the mean value of a vector image in the 26-neighborhood
    of a given element with indices (x,y,z). 
    """
    cdef unsigned int j = 0
    cdef unsigned int xn, yn, zn
    cdef np.ndarray ngb 

    # Choose neighborhood system 
    # FIXME: should be an input  
    ngb = ngb26
    
    # Re-initialize output array 
    res[:] = 0.0

    # Loop over neighbors
    while j < ngb.shape[0]: 
        xn = x + ngb[j,0]
        yn = y + ngb[j,1]
        zn = z + ngb[j,2]
        res += ppm[xn,yn,zn,:]
        j += 1
    


@cython.boundscheck(False)
def ve_step(np.ndarray[np.double_t, ndim=4] ppm, 
            np.ndarray[np.double_t, ndim=1] lik, 
            np.ndarray[np.uint_t, ndim=1] X,
            np.ndarray[np.uint_t, ndim=1] Y,
            np.ndarray[np.uint_t, ndim=1] Z):
    
    cdef unsigned int npts = X.size, K = ppm.shape[3]
    cdef unsigned int i, x, y, z
    cdef np.ndarray p = np.zeros(K)
    cdef double psum

    cdef double beta = 1.0
    cdef TINY = 1e-20

    for i from 0 <= i < npts:
        x = X[i]
        y = Y[i]
        z = Z[i]
        neighbor_average(p, ppm, x, y, z)
        p = np.exp(beta*p)
        p *= lik[i]
        psum = p.sum()
        if psum > TINY: 
            p /= psum
        ppm[x,y,z,:] = p     
                
    return ppm

