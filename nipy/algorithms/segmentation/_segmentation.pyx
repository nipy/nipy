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
                 ndarray U,
                 int ngb_size, 
                 double beta)
    ndarray make_edges(ndarray mask,
                       int ngb_size)
    double interaction_energy(ndarray ppm, 
                              ndarray XYZ,
                              ndarray U,
                              int ngb_size)



# Initialize numpy
mrf_import_array()
import_array()
import numpy as np



def _ve_step(ppm, ref, XYZ, U, int ngb_size, double beta):

    if not ppm.flags['C_CONTIGUOUS'] or not ppm.dtype=='double':
        raise ValueError('ppm array should be double C-contiguous')
    if not ref.flags['C_CONTIGUOUS'] or not ref.dtype=='double':
        raise ValueError('ref array should be double C-contiguous')
    if not XYZ.flags['C_CONTIGUOUS'] or not XYZ.dtype=='intp':
        raise ValueError('XYZ array should be intp C-contiguous')
    if not XYZ.shape[1] == 3: 
        raise ValueError('XYZ array should be 3D')
    if not U.flags['C_CONTIGUOUS'] or not U.dtype=='double':
        raise ValueError('U array should be double C-contiguous')
    if not ppm.shape[-1] == ref.shape[-1]:
        raise ValueError('Inconsistent shapes for ppm and ref arrays')

    ve_step(<ndarray>ppm, <ndarray>ref, <ndarray>XYZ, <ndarray>U, 
             ngb_size, beta)
    return ppm 


def _make_edges(mask, int ngb_size):
    
    if not mask.flags['C_CONTIGUOUS'] or not mask.dtype=='intp':
        raise ValueError('mask array should be intp and C-contiguous')

    return make_edges(mask, ngb_size)


def _interaction_energy(ppm, XYZ, U, int ngb_size):

    if not ppm.flags['C_CONTIGUOUS'] or not ppm.dtype=='double':
        raise ValueError('ppm array should be double C-contiguous')
    if not XYZ.flags['C_CONTIGUOUS'] or not XYZ.dtype=='intp':
        raise ValueError('XYZ array should be intp C-contiguous')
    if not XYZ.shape[1] == 3: 
        raise ValueError('XYZ array should be 3D')
    if not U.flags['C_CONTIGUOUS'] or not U.dtype=='double':
        raise ValueError('U array should be double C-contiguous')

    return interaction_energy(<ndarray>ppm, <ndarray>XYZ, <ndarray>U,
                               ngb_size)
