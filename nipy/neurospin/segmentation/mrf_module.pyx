# -*- Mode: Python -*-  

"""
Markov random field utils. 
"""

__version__ = '0.0'

# Includes
include "numpy.pxi"

# Externals
cdef extern from "mrf.h":
    
    void mrf_import_array()
    void smooth_ppm(ndarray ppm, 
                    ndarray lik,
                    ndarray XYZ, 
                    double beta)

# Initialize numpy
mrf_import_array()
import_array()
import numpy as np
#cimport numpy as np
#import cython 


def finalize_ve_step(ppm, lik, XYZ, double beta):
    
    if not ppm.flags['C_CONTIGUOUS'] or not ppm.dtype=='double':
        raise ValueError('ppm array should be double C-contiguous')

    if not lik.flags['C_CONTIGUOUS'] or not lik.dtype=='double':
        raise ValueError('lik array should be double C-contiguous')
    
    XYZ = np.asarray(XYZ, dtype='int')
    
    smooth_ppm(<ndarray>ppm, <ndarray>lik, <ndarray>XYZ, beta)
    return ppm 


