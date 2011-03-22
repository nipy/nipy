# -*- Mode: Python -*-  

"""
Markov random field utils. 

Author: Alexis Roche, 2010.
"""

__version__ = '0.2'

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
    double interaction_energy(ndarray ppm, 
                              ndarray XYZ, 
                              ndarray mix)


# Initialize numpy
mrf_import_array()
import_array()
import numpy as np
import warnings

def _ve_step(ppm, ref, XYZ, double beta, int copy, int hard, mix=None):

    if not XYZ.shape[1] == 3: 
        warnings.warn('MRF regularization only implemented in 3D, doing nothing')
        return ppm
    if not ppm.flags['C_CONTIGUOUS'] or not ppm.dtype=='double':
        raise ValueError('ppm array should be double C-contiguous')
    if not ref.flags['C_CONTIGUOUS'] or not ref.dtype=='double':
        raise ValueError('ref array should be double C-contiguous')
    if not XYZ.flags['C_CONTIGUOUS'] or not XYZ.dtype=='int':
        raise ValueError('XYZ array should be int C-contiguous')
    if not mix==None: 
        if not mix.flags['C_CONTIGUOUS'] or not mix.dtype=='double':
            raise ValueError('mix array should be double C-contiguous')

    ve_step(<ndarray>ppm, <ndarray>ref, <ndarray>XYZ, <ndarray>mix, 
             beta, copy, hard)

    return ppm 


def _interaction_energy(ppm, XYZ, mix=None): 

    if not XYZ.shape[1] == 3: 
        warnings.warn('MRF regularization only implemented in 3D, doing nothing')
        return ppm
    if not ppm.flags['C_CONTIGUOUS'] or not ppm.dtype=='double':
        raise ValueError('ppm array should be double C-contiguous')
    if not XYZ.flags['C_CONTIGUOUS'] or not XYZ.dtype=='int':
        raise ValueError('XYZ array should be int C-contiguous')
    if not mix==None: 
        if not mix.flags['C_CONTIGUOUS'] or not mix.dtype=='double':
            raise ValueError('mix array should be double C-contiguous')

    return interaction_energy(<ndarray>ppm, <ndarray>XYZ, <ndarray>mix) 

