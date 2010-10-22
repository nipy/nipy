# -*- Mode: Python -*-  

"""
Fast registration routines module: joint histogram computation,
similarity measures, affine transformation parameterization.

Author: Alexis Roche, 2008.
"""

__version__ = '0.2'


# Includes
from numpy cimport import_array, ndarray, flatiter, broadcast, PyArray_SIZE, PyArray_MultiIterNew, PyArray_MultiIter_DATA, PyArray_MultiIter_NEXT


# Externals
cdef extern from "math.h":
 
   double log(double)


cdef extern from "joint_histogram.h":

    void joint_histogram_import_array()
    void L2_moments(double* h, unsigned int size, double* res)
    void L1_moments(double * h, unsigned int size, double *res)
    double entropy(double* h, unsigned int size, double* n)
    int joint_histogram(ndarray H, unsigned int clampI, unsigned int clampJ,  
                        flatiter iterI, ndarray imJ_padded, 
                        ndarray Tvox, int interp)
    double correlation_coefficient(double* H, unsigned int clampI, unsigned int clampJ, double* n)
    double correlation_ratio(double* H, unsigned int clampI, unsigned int clampJ, double* n) 
    double correlation_ratio_L1(double* H, double* hI, unsigned int clampI, unsigned int clampJ, double* n) 
    double joint_entropy(double* H, unsigned int clampI, unsigned int clampJ)
    double conditional_entropy(double* H, double* hJ, unsigned int clampI, unsigned int clampJ) 
    double mutual_information(double* H, 
                              double* hI, unsigned int clampI, 
                              double* hJ, unsigned int clampJ,
                              double* n)
    double normalized_mutual_information(double* H, 
                                         double* hI, unsigned int clampI, 
                                         double* hJ, unsigned int clampJ, 
                                         double* n) 
    double supervised_mutual_information(double* H, double* F, 
                                         double* fI, unsigned int clampI, 
                                         double* fJ, unsigned int clampJ,
                                         double* n) 



# Initialize numpy
joint_histogram_import_array()
import_array()
import numpy as np


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
    LLR_CORRELATION_COEFFICIENT,
    LLR_CORRELATION_RATIO,
    LLR_CORRELATION_RATIO_L1,
    LLR_MUTUAL_INFORMATION,
    LLR_SUPERVISED_MUTUAL_INFORMATION, 
    CUSTOM_SIMILARITY

# Corresponding Python dictionary 
builtin_similarities = {
    'cc': CORRELATION_COEFFICIENT,
    'cr': CORRELATION_RATIO,
    'crl1': CORRELATION_RATIO_L1, 
    'mi': MUTUAL_INFORMATION, 
    'je': JOINT_ENTROPY,
    'ce': CONDITIONAL_ENTROPY,
    'nmi': NORMALIZED_MUTUAL_INFORMATION,
    'smi': SUPERVISED_MUTUAL_INFORMATION,
    'llr_cc': LLR_CORRELATION_COEFFICIENT,
    'llr_cr': LLR_CORRELATION_RATIO,
    'llr_crl1': LLR_CORRELATION_RATIO_L1,
    'llr_mi': LLR_MUTUAL_INFORMATION,
    'llr_smi': LLR_SUPERVISED_MUTUAL_INFORMATION,  
    'custom': CUSTOM_SIMILARITY}

def _joint_histogram(ndarray H, flatiter iterI, ndarray imJ, ndarray Tvox, int interp):
    """
    _joint_histogram(H, iterI, imJ, Tvox, interp)
    Comments to follow.
    """
    cdef:
        double *h, *tvox
        unsigned int clampI, clampJ
        int ret

    # Views
    clampI = <unsigned int>H.shape[0]
    clampJ = <unsigned int>H.shape[1]    

    # Compute joint histogram 
    ret = joint_histogram(H, clampI, clampJ, iterI, imJ, Tvox, interp)
    if not ret == 0:
        raise RuntimeError('Joint histogram failed, which is impossible.')

    return 


cdef cc2llr(double x, double n):
    cdef double y = 1-x
    if y < 0.0:
        y = 0.0 
    return -.5 * n * log(y)


def _similarity(ndarray H, ndarray HI, ndarray HJ, int simitype, 
                ndarray F=None, method=None):
    """
    _similarity(H, hI, hJ, simitype, ndarray F=None)
    Comments to follow
    """
    cdef int isF = 0
    cdef double *h, *hI, *hJ, *f=NULL
    cdef double simi=0.0, n
    cdef unsigned int clampI, clampJ

    # Array views
    clampI = <unsigned int>H.shape[0]
    clampJ = <unsigned int>H.shape[1]
    h = <double*>H.data
    hI = <double*>HI.data
    hJ = <double*>HJ.data
    if F != None:
        f = <double*>F.data
        isF = 1

    # Switch 
    if simitype == CORRELATION_COEFFICIENT:
        simi = correlation_coefficient(h, clampI, clampJ, &n)
    elif simitype == CORRELATION_RATIO: 
        simi = correlation_ratio(h, clampI, clampJ, &n) 
    elif simitype == CORRELATION_RATIO_L1:
        simi = correlation_ratio_L1(h, hI, clampI, clampJ, &n) 
    elif simitype == MUTUAL_INFORMATION: 
        simi = mutual_information(h, hI, clampI, hJ, clampJ, &n) 
    elif simitype == JOINT_ENTROPY:
        simi = joint_entropy(h, clampI, clampJ) 
    elif simitype == CONDITIONAL_ENTROPY:
        simi = conditional_entropy(h, hJ, clampI, clampJ) 
    elif simitype == NORMALIZED_MUTUAL_INFORMATION:
        simi = normalized_mutual_information(h, hI, clampI, hJ, clampJ, &n) 
    elif simitype == SUPERVISED_MUTUAL_INFORMATION:
        simi = supervised_mutual_information(h, f, hI, clampI, hJ, clampJ, &n)
    elif simitype == LLR_CORRELATION_COEFFICIENT:
        simi = correlation_coefficient(h, clampI, clampJ, &n)
        simi = cc2llr(simi, n)
    elif simitype == LLR_CORRELATION_RATIO: 
        simi = correlation_ratio(h, clampI, clampJ, &n) 
        simi = cc2llr(simi, n)
    elif simitype == LLR_CORRELATION_RATIO_L1:
        simi = correlation_ratio_L1(h, hI, clampI, clampJ, &n) 
        simi = cc2llr(simi, n)
    elif simitype == LLR_MUTUAL_INFORMATION: 
        simi = mutual_information(h, hI, clampI, hJ, clampJ, &n) 
        simi = n*simi
    elif simitype == LLR_SUPERVISED_MUTUAL_INFORMATION:
        simi = supervised_mutual_information(h, f, hI, clampI, hJ, clampJ, &n)
        simi = n*simi
    else: # CUSTOM 
        simi = method(H)
        
    return simi


