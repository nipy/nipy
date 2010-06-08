# -*- Mode: Python -*-  

"""
Markov random field utils. 
"""

__version__ = '0.0'

# Includes
from numpy cimport import_array, ndarray

# Externals
cdef extern from "mrf.h":
    
    void mrf_import_array()
    void ve_step(ndarray ppm, 
                 ndarray ref,
                 ndarray XYZ, 
                 ndarray mix, 
                 double beta, 
                 int copy, 
                 int hard)
    double concensus(ndarray ppm, 
		     ndarray XYZ, 
                     ndarray mix)


# Initialize numpy
mrf_import_array()
import_array()
import numpy as np


def _ve_step(ppm, ref, XYZ, double beta, int copy, int hard, mix=None):
    
    if not ppm.flags['C_CONTIGUOUS'] or not ppm.dtype=='double':
        raise ValueError('ppm array should be double C-contiguous')

    if not ref.flags['C_CONTIGUOUS'] or not ref.dtype=='double':
        raise ValueError('ref array should be double C-contiguous')
    
    XYZ = np.asarray(XYZ, dtype='int')
    
    ve_step(<ndarray>ppm, <ndarray>ref, <ndarray>XYZ, <ndarray>mix, 
             beta, copy, hard)
    return ppm 


def _concensus(ppm, XYZ, mix=None): 
    return concensus(<ndarray>ppm, <ndarray>XYZ, <ndarray>mix) 

