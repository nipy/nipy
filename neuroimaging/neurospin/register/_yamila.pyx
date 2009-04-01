# -*- Mode: Python -*-  Not really, but the syntax is close enough


"""
Image registration routines. Joint histogram computation, similarity
measures.

Author: Alexis Roche, 2008.
"""

__version__ = '0.1'


# Includes
include "numpy.pxi"

# Additional exports from fff_iconic_match.h
cdef extern from "fff_iconic_match.h":

    ctypedef struct fff_imatch:
        fff_array* imI
        fff_array* imJ
        fff_array* imJ_padded
        int clampI
        int clampJ
        double* H
        double* hI
        double* hJ
        int owner_images
        int owner_histograms
        

    fff_imatch* fff_imatch_new(fff_array* imI, fff_array* imJ,
                               double thI, double thJ, int clampI, int clampJ)
    void fff_imatch_delete(fff_imatch* imatch)
    unsigned int fff_imatch_source_npoints( fff_array* imI )
    void fff_imatch_joint_hist(double* H, int clampI, int clampJ,
                               fff_array* imI, fff_array* imJ_padded, double* Tvox,
                               int interp) 
    double fff_imatch_cc(double* H, int clampI, int clampJ)
    double fff_imatch_cr(double* H, int clampI, int clampJ) 
    double fff_imatch_crL1(double* H, double* hI, int clampI, int clampJ) 
    double fff_imatch_mi(double* H, double* hI, int clampI, double* hJ, int clampJ)
    double fff_imatch_joint_ent(double* H, int clampI, int clampJ)
    double fff_imatch_cond_ent(double* H, double* hJ, int clampI, int clampJ) 
    double fff_imatch_norma_mi(double* H, double* hI, int clampI, double* hJ, int clampJ) 
    double fff_imatch_n_cc(double* H, int clampI, int clampJ, double norma)
    double fff_imatch_n_cr(double* H, int clampI, int clampJ, double norma) 
    double fff_imatch_n_crL1(double* H, double* hI, int clampI, int clampJ, double norma) 
    double fff_imatch_n_mi(double* H, double* hI, int clampI, double* hJ, int clampJ, double norma)
    double fff_imatch_supervised_mi(double* H, double* F, 
                                    double* fI, int clampI, double* fJ, int clampJ)
    double fff_imatch_n_supervised_mi(double* H, double* F, 
                                      double* fI, int clampI, double* fJ, int clampJ, double norma) 




# Initialize numpy
fffpy_import_array()
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
    N_CORRELATION_COEFFICIENT,
    N_CORRELATION_RATIO,
    N_CORRELATION_RATIO_L1,
    N_MUTUAL_INFORMATION,
    N_SUPERVISED_MUTUAL_INFORMATION

# Corresponding Python dictionary 
similarity_measures = {'correlation coefficient': [CORRELATION_COEFFICIENT, N_CORRELATION_COEFFICIENT],
                       'correlation ratio': [CORRELATION_RATIO, N_CORRELATION_RATIO],  
                       'correlation ratio L1': [CORRELATION_RATIO_L1, N_CORRELATION_RATIO_L1],
                       'mutual information': [MUTUAL_INFORMATION, N_MUTUAL_INFORMATION],
                       'joint entropy': [JOINT_ENTROPY],
                       'conditional entropy': [CONDITIONAL_ENTROPY],
                       'normalized mutual information': [NORMALIZED_MUTUAL_INFORMATION],
                       'supervised mutual information': [SUPERVISED_MUTUAL_INFORMATION, N_SUPERVISED_MUTUAL_INFORMATION]}



def joint_hist(ndarray H, flatiter iterI, ndarray imJ, ndarray Tvox, int interp):

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


def similarity(ndarray H, ndarray HI, ndarray HJ, int simitype=MUTUAL_INFORMATION, double norma=1.0, ndarray F=None):
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
        simi = fff_imatch_cc(h, clampI, clampJ)
    elif simitype == N_CORRELATION_COEFFICIENT:
        simi = fff_imatch_n_cc(h, clampI, clampJ, norma) 
    elif simitype == CORRELATION_RATIO: 
        simi = fff_imatch_cr(h, clampI, clampJ) 
    elif simitype == N_CORRELATION_RATIO: 
        simi = fff_imatch_n_cr(h, clampI, clampJ, norma)  
    elif simitype == CORRELATION_RATIO_L1:
        simi = fff_imatch_crL1(h, hI, clampI, clampJ) 
    elif simitype == N_CORRELATION_RATIO_L1:
        simi = fff_imatch_n_crL1(h, hI, clampI, clampJ, norma)  
    elif simitype == MUTUAL_INFORMATION: 
        simi = fff_imatch_mi(h, hI, clampI, hJ, clampJ) 
    elif simitype == N_MUTUAL_INFORMATION: 
        simi = fff_imatch_n_mi(h, hI, clampI, hJ, clampJ, norma) 
    elif simitype == JOINT_ENTROPY:
        simi = fff_imatch_joint_ent(h, clampI, clampJ) 
    elif simitype == CONDITIONAL_ENTROPY:
        simi = fff_imatch_cond_ent(h, hJ, clampI, clampJ) 
    elif simitype == NORMALIZED_MUTUAL_INFORMATION:
        simi = fff_imatch_norma_mi(h, hI, clampI, hJ, clampJ) 
    elif simitype == SUPERVISED_MUTUAL_INFORMATION:
        simi = fff_imatch_supervised_mi(h, f, hI, clampI, hJ, clampJ)
    elif simitype == N_SUPERVISED_MUTUAL_INFORMATION:
        simi = fff_imatch_n_supervised_mi(h, f, hI, clampI, hJ, clampJ, norma) 
    else:
        simi = 0.0
        
    return simi


