# -*- Mode: Python -*-  

"""
Fast registration routines module: joint histogram computation,
similarity measures, affine transformation parameterization.

Author: Alexis Roche, 2008.
"""

__version__ = '0.2'


# Includes
from numpy cimport import_array, ndarray, flatiter

# Externals
cdef extern from "math.h":
   double log(double)

cdef extern from "joint_histogram.h":
    void joint_histogram_import_array()
    int joint_histogram(ndarray H, unsigned int clampI, unsigned int clampJ,  
                        flatiter iterI, ndarray imJ_padded, 
                        ndarray Tvox, int interp)
    int L1_moments(double* n, double* median, double* dev, ndarray H)


# Initialize numpy
joint_histogram_import_array()
import_array()
import numpy as np


def _joint_histogram(ndarray H, flatiter iterI, ndarray imJ, ndarray Tvox, int interp):
    """
    Compute the joint histogram given a transformation trial. 
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
        raise RuntimeError('Joint histogram failed because of incorrect input arrays.')

    return 


def _L1_moments(ndarray H):
    """
    Compute L1 moments of order 0, 1 and 2 of a one-dimensional
    histogram.
    """
    cdef:
        double n[1], median[1], dev[1]
        int ret

    ret = L1_moments(n, median, dev, H)
    if not ret == 0:
        raise RuntimeError('L1_moments failed because input array is not double.')

    return n[0], median[0], dev[0]

