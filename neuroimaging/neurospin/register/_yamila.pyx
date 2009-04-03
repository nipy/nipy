# -*- Mode: Python -*-  Not really, but the syntax is close enough


"""
Joint histogram computation, similarity measures.

Author: Alexis Roche, 2008.
"""

__version__ = '0.1'


# Includes
include "numpy.pxi"

# Externals
cdef extern from "yamila.h":

    void yamila_import_array()
    void joint_histogram(double* H, int clampI, int clampJ,  
                         flatiter iterI, ndarray imJ_padded, 
                         double* Tvox, int interp)
    double correlation_coefficient(double* H, int clampI, int clampJ)
    double correlation_ratio(double* H, int clampI, int clampJ) 
    double correlation_ratio_L1(double* H, double* hI, int clampI, int clampJ) 
    double joint_entropy(double* H, int clampI, int clampJ)
    double conditional_entropy(double* H, double* hJ, int clampI, int clampJ) 
    double mutual_information(double* H, 
                              double* hI, int clampI, 
                              double* hJ, int clampJ)
    double normalized_mutual_information(double* H, 
                                         double* hI, int clampI, 
                                         double* hJ, int clampJ) 
    double supervised_mutual_information(double* H, double* F, 
                                         double* fI, int clampI, 
                                         double* fJ, int clampJ) 




# Initialize numpy
yamila_import_array()
import_array()
import numpy as np
cimport numpy as np


# Enumerate similarity measures
cdef enum similarity_measure:
    CORRELATION_COEFFICIENT,
    CORRELATION_RATIO,
    CORRELATION_RATIO_L1,
    JOINT_ENTROPY,
    CONDITIONAL_ENTROPY,
    MUTUAL_INFORMATION,
    NORMALIZED_MUTUAL_INFORMATION,
    SUPERVISED_MUTUAL_INFORMATION,

# Corresponding Python dictionary 
similarity_measures = {'cc': CORRELATION_COEFFICIENT,
                       'cr': CORRELATION_RATIO,
                       'crl1': CORRELATION_RATIO_L1, 
                       'mi': MUTUAL_INFORMATION, 
                       'je': JOINT_ENTROPY,
                       'ce': CONDITIONAL_ENTROPY,
                       'nmi': NORMALIZED_MUTUAL_INFORMATION,
                       'smi': SUPERVISED_MUTUAL_INFORMATION}


def _joint_histogram(ndarray H, flatiter iterI, ndarray imJ, ndarray Tvox, int interp):

    """
    joint_hist(H, imI, imJ, Tvox, subsampling, corner, size)
    Comments to follow.
    """
    cdef double *h, *tvox
    cdef int clampI, clampJ

    # Views
    clampI = <int>H.dimensions[0]
    clampJ = <int>H.dimensions[1]    
    h = <double*>H.data
    tvox = <double*>Tvox.data

    # Compute joint histogram 
    joint_histogram(h, clampI, clampJ, iterI, imJ, tvox, interp)

    return 


def _similarity(ndarray H, ndarray HI, ndarray HJ, int simitype, ndarray F=None):
    """
    similarity(H, hI, hJ).
    Comments to follow
    """
    cdef int isF = 0
    cdef double *h, *hI, *hJ, *f=NULL
    cdef double simi = 0.0
    cdef int clampI, clampJ

    # Array views
    clampI = <int>H.dimensions[0]
    clampJ = <int>H.dimensions[1]
    h = <double*>H.data
    hI = <double*>HI.data
    hJ = <double*>HJ.data
    if F != None:
        f = <double*>F.data
        isF = 1

    # Switch 
    if simitype == CORRELATION_COEFFICIENT:
        simi = correlation_coefficient(h, clampI, clampJ)
    elif simitype == CORRELATION_RATIO: 
        simi = correlation_ratio(h, clampI, clampJ) 
    elif simitype == CORRELATION_RATIO_L1:
        simi = correlation_ratio_L1(h, hI, clampI, clampJ) 
    elif simitype == MUTUAL_INFORMATION: 
        simi = mutual_information(h, hI, clampI, hJ, clampJ) 
    elif simitype == JOINT_ENTROPY:
        simi = joint_entropy(h, clampI, clampJ) 
    elif simitype == CONDITIONAL_ENTROPY:
        simi = conditional_entropy(h, hJ, clampI, clampJ) 
    elif simitype == NORMALIZED_MUTUAL_INFORMATION:
        simi = normalized_mutual_information(h, hI, clampI, hJ, clampJ) 
    elif simitype == SUPERVISED_MUTUAL_INFORMATION:
        simi = supervised_mutual_information(h, f, hI, clampI, hJ, clampJ)
    else:
        simi = 0.0
        
    return simi


