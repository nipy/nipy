# -*- Mode: Python -*-  

"""
Markov random field utils. 
"""

__version__ = '0.0'

import numpy as np
cimport numpy as np

import cython 

@cython.boundscheck(False)
def neighbor_average(np.ndarray[np.double_t, ndim=4] im, 
                     np.ndarray[np.uint_t, ndim=1] pt):
    """
    pt must be 
    """
    cdef unsigned int j=0, dim
    cdef unsigned int x0, y0, z0, x, y, z
    cdef np.ndarray res

    cdef np.ndarray nb26 = np.array(((1,0,0),(-1,0,0),(0,1,0),(0,-1,0), 
                                     (1,1,0),(-1,1,0),(-1,1,0),(-1,-1,0), 
                                     (1,0,1),(-1,0,1),(0,1,1),(0,-1,1), 
                                     (1,1,1),(-1,1,1),(-1,1,1),(-1,-1,1), 
                                     (1,0,-1),(-1,0,-1),(0,1,-1),(0,-1,-1), 
                                     (1,1,-1),(-1,1,-1),(-1,1,-1),(-1,-1,-1), 
                                     (0,0,1),(0,0,-1)))

    x0 = <unsigned int>pt[0]
    y0 = <unsigned int>pt[1]
    z0 = <unsigned int>pt[2]
    dim = <unsigned int>im.shape[3]
    res = np.zeros(dim)

    # Loop over neighbors
    while j < nb26.shape[0]: 
        x = x0 + nb26[j,0]
        y = y0 + nb26[j,1]
        z = z0 + nb26[j,2]
        res += im[x,y,z,:]
        j += 1

    return res




"""
def _texture(ndarray im, ndarray H, Size, int texture, method=None): 

    cdef double *res, *h
    cdef double moments[5]
    cdef unsigned int clamp
    cdef unsigned int coords[3], size[3]
    cdef broadcast multi
    cdef flatiter im_iter

    # Views
    clamp = <unsigned int>H.dimensions[0]
    h = <double*>H.data
    
    # Copy size parameters
    size[0] = <unsigned int>Size[0]
    size[1] = <unsigned int>Size[1]
    size[2] = <unsigned int>Size[2]

    # Allocate output 
    imtext = np.zeros(im.shape, dtype='double')

    # Loop over input and output images
    multi = PyArray_MultiIterNew(2, <void*>imtext, <void*>im)
    while(multi.index < multi.size):
        res = <double*>PyArray_MultiIter_DATA(multi, 0)
        im_iter = <flatiter>multi.iters[1]
        # Compute local image histogram
        local_histogram(h, clamp, im_iter, size)
        # Switch 
        if texture == MIN:
            drange(h, clamp, moments)
            res[0] = moments[0]
        elif texture == MAX:
            drange(h, clamp, moments)
            res[0] = moments[1]
        elif texture == DRANGE:
            drange(h, clamp, moments)
            res[0] = moments[1]-moments[0]
        elif texture == MEAN: 
            L2_moments(h, clamp, moments)
            res[0] = moments[1]
        elif texture == MEAN: 
            L2_moments(h, clamp, moments)
            res[0] = moments[2]
        elif texture == MEDIAN:
            L1_moments(h, clamp, moments)
            res[0] = moments[1] 
        elif texture == L1DEV: 
            L1_moments(h, clamp, moments)
            res[0] = moments[2] 
        elif texture == ENTROPY: 
            res[0] = entropy(h, clamp, moments)
        else: # CUSTOM
            res[0] = method(H)
        # Next voxel please
        PyArray_MultiIter_NEXT(multi)
   
    return imtext
"""
