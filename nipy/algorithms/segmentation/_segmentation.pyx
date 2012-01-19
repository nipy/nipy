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
                 int ngb_size,
                 double beta,
                 int synchronous,
                 int scheme)
    double interaction_energy(ndarray ppm,
                              ndarray XYZ,                 
                              int ngb_size)

    void gen_ve_step(ndarray ppm, 
                     ndarray ref,
                     ndarray XYZ, 
                     ndarray U,
                     int ngb_size, 
                     double beta)
    

# Initialize numpy
mrf_import_array()
import_array()
import numpy as np


def _ve_step(ppm, ref, XYZ, int ngb_size, double beta, int synchronous, int scheme):

    if not ppm.flags['C_CONTIGUOUS'] or not ppm.dtype=='double':
        raise ValueError('ppm array should be double C-contiguous')
    if not ref.flags['C_CONTIGUOUS'] or not ref.dtype=='double':
        raise ValueError('ref array should be double C-contiguous')
    if not XYZ.flags['C_CONTIGUOUS'] or not XYZ.dtype=='int':
        raise ValueError('XYZ array should be int C-contiguous')
    if not XYZ.shape[1] == 3: 
        raise ValueError('XYZ array should be 3D')

    ve_step(<ndarray>ppm, <ndarray>ref, <ndarray>XYZ, 
             ngb_size, beta, synchronous, scheme)
    return ppm 


def _interaction_energy(ppm, XYZ, int ngb_size): 

    if not ppm.flags['C_CONTIGUOUS'] or not ppm.dtype=='double':
        raise ValueError('ppm array should be double C-contiguous')
    if not XYZ.flags['C_CONTIGUOUS'] or not XYZ.dtype=='int':
        raise ValueError('XYZ array should be int C-contiguous')
    if not XYZ.shape[1] == 3: 
        raise ValueError('XYZ array should be 3D')

    return interaction_energy(<ndarray>ppm, <ndarray>XYZ, ngb_size) 


def _gen_ve_step(ppm, ref, XYZ, U, int ngb_size, double beta):

    if not ppm.flags['C_CONTIGUOUS'] or not ppm.dtype=='double':
        raise ValueError('ppm array should be double C-contiguous')
    if not ref.flags['C_CONTIGUOUS'] or not ref.dtype=='double':
        raise ValueError('ref array should be double C-contiguous')
    if not XYZ.flags['C_CONTIGUOUS'] or not XYZ.dtype=='int':
        raise ValueError('XYZ array should be int C-contiguous')
    if not XYZ.shape[1] == 3: 
        raise ValueError('XYZ array should be 3D')

    gen_ve_step(<ndarray>ppm, <ndarray>ref, <ndarray>XYZ, <ndarray>U, 
                 ngb_size, beta)
    return ppm 

