# -*- Mode: Python -*-  Not really, but the syntax is close enough
"""
Routines for massively univariate random-effect and mixed-effect
analysis. Two-sample case.

Author: Alexis Roche, 2008.
"""

__version__ = '0.1'


# Includes
from fff cimport *

# Exports from fff_twosample_stat.h
cdef extern from "fff_twosample_stat.h":

  ctypedef enum fff_twosample_stat_flag:
    FFF_TWOSAMPLE_STUDENT = 0,
    FFF_TWOSAMPLE_WILCOXON = 1,
    FFF_TWOSAMPLE_STUDENT_MFX = 10

  ctypedef struct fff_twosample_stat:
    pass

  ctypedef struct fff_twosample_stat_mfx:
    unsigned int niter

  fff_twosample_stat* fff_twosample_stat_new(unsigned int n1, unsigned int n2, fff_twosample_stat_flag flag)
  void fff_twosample_stat_delete(fff_twosample_stat* thisone) 
  double fff_twosample_stat_eval(fff_twosample_stat* thisone, fff_vector* x)

  fff_twosample_stat_mfx* fff_twosample_stat_mfx_new(unsigned int n1, unsigned int n2,
                                                     fff_twosample_stat_flag flag)
  void fff_twosample_stat_mfx_delete(fff_twosample_stat_mfx* thisone) 
  double fff_twosample_stat_mfx_eval(fff_twosample_stat_mfx* thisone, 
                                     fff_vector* x, fff_vector* vx)
  
  unsigned int fff_twosample_permutation(unsigned int* idx1, unsigned int* idx2, 
                                         unsigned int n1, unsigned int n2, double* magic)

  void fff_twosample_apply_permutation(fff_vector* px, fff_vector* pv, 
                                       fff_vector* x1, fff_vector* v1, 
                                       fff_vector* x2,  fff_vector* v2,
                                       unsigned int i, unsigned int* idx1, unsigned int* idx2)
   

# Initialize numpy
fffpy_import_array()
import_array()
import numpy as np


# Stat dictionary
stats = {'student': FFF_TWOSAMPLE_STUDENT,
         'wilcoxon': FFF_TWOSAMPLE_WILCOXON,
         'student_mfx': FFF_TWOSAMPLE_STUDENT_MFX}


def count_permutations(unsigned int n1, unsigned int n2):
  cdef double  n
  fff_twosample_permutation(NULL, NULL, n1, n2, &n)
  return int(n)
  

def stat(ndarray Y1, ndarray Y2, id='student', int axis=0, ndarray Magics=None):
  """
  T = stat(Y1, Y2, id='student', axis=0, magics=None).
  
  Compute a two-sample test statistic (Y1>Y2) over a number of
  deterministic or random permutations.
  """
  cdef fff_vector *y1, *y2, *t, *yp, *magics
  cdef fff_array *idx1, *idx2
  cdef unsigned int n, n1, n2, nex
  cdef unsigned long int simu, nsimu, idx
  cdef fff_twosample_stat* stat
  cdef fff_twosample_stat_flag flag_stat = stats[id]
  cdef double magic
  cdef fffpy_multi_iterator* multi

  # Get number of observations
  n1 = <unsigned int>Y1.shape[axis]
  n2 = <unsigned int>Y2.shape[axis]
  n = n1 + n2

  # Read out magic numbers
  if Magics is None:
    magics = fff_vector_new(1)
    magics.data[0] = 0 ## Just to make sure
  else:
    magics = fff_vector_fromPyArray(Magics)

  # Create output array
  nsimu = magics.size
  dims = [Y1.shape[i] for i in range(Y1.ndim)]
  dims[axis] = nsimu
  T = np.zeros(dims)

  # Create local structure
  yp = fff_vector_new(n)
  idx1 = fff_array_new1d(FFF_UINT, n1)
  idx2 = fff_array_new1d(FFF_UINT, n2)
  stat = fff_twosample_stat_new(n1, n2, flag_stat)

  # Multi-iterator 
  multi = fffpy_multi_iterator_new(3, axis, <void*>Y1, <void*>Y2, <void*>T)

  # Vector views
  y1 = multi.vector[0]
  y2 = multi.vector[1]
  t = multi.vector[2]
  
  # Loop
  for simu from 0 <= simu < nsimu:
    
    # Set the magic number
    magic = magics.data[simu*magics.stride]

    # Generate permutation 
    nex = fff_twosample_permutation(<unsigned int*>idx1.data,
                                    <unsigned int*>idx2.data,
                                    n1, n2, &magic)
   
    # Reset the multi-iterator
    fffpy_multi_iterator_reset(multi)
    
    # Perform the loop 
    idx = simu*t.stride
    while(multi.index < multi.size):
      fff_twosample_apply_permutation(yp, NULL, y1, NULL, y2, NULL, nex,
                                      <unsigned int*>idx1.data,
                                      <unsigned int*>idx2.data)
      t.data[idx] = fff_twosample_stat_eval(stat, yp)
      fffpy_multi_iterator_update(multi)

  # Delete local structures
  fffpy_multi_iterator_delete(multi)
  fff_vector_delete(magics)
  fff_vector_delete(yp)
  fff_array_delete(idx1)
  fff_array_delete(idx2)
  fff_twosample_stat_delete(stat)

  # Return
  return T


def stat_mfx(ndarray Y1, ndarray V1, ndarray Y2, ndarray V2,
             id='student_mfx', int axis=0, ndarray Magics=None,
             unsigned int niter=5):
  """
  T = stat(Y1, V1, Y2, V2, id='student', axis=0, magics=None, niter=5).
  
  Compute a two-sample test statistic (Y1>Y2) over a number of
  deterministic or random permutations.
  """
  cdef fff_vector *y1, *y2, *v1, *v2, *t, *yp, *vp, *magics
  cdef fff_array *idx1, *idx2
  cdef unsigned int n, n1, n2, nex
  cdef unsigned long int simu, nsimu, idx
  cdef fff_twosample_stat_mfx* stat
  cdef fff_twosample_stat_flag flag_stat = stats[id]
  cdef double magic
  cdef fffpy_multi_iterator* multi

  # Get number of observations
  n1 = <unsigned int>Y1.shape[axis]
  n2 = <unsigned int>Y2.shape[axis]
  n = n1 + n2

  # Read out magic numbers
  if Magics is None:
    magics = fff_vector_new(1)
    magics.data[0] = 0 ## Just to make sure
  else:
    magics = fff_vector_fromPyArray(Magics)

  # Create output array
  nsimu = magics.size
  dims = [Y1.shape[i] for i in range(Y1.ndim)]
  dims[axis] = nsimu 
  T = np.zeros(dims)

  # Create local structure
  yp = fff_vector_new(n)
  vp = fff_vector_new(n)
  idx1 = fff_array_new1d(FFF_UINT, n1)
  idx2 = fff_array_new1d(FFF_UINT, n2)
  stat = fff_twosample_stat_mfx_new(n1, n2, flag_stat)
  stat.niter = niter

  # Multi-iterator 
  multi = fffpy_multi_iterator_new(5, axis,
                                   <void*>Y1, <void*>V1,
                                   <void*>Y2, <void*>V2,
                                   <void*>T)

  # Vector views
  y1 = multi.vector[0]
  v1 = multi.vector[1]
  y2 = multi.vector[2]
  v2 = multi.vector[3]
  t = multi.vector[4]
  
  # Loop
  for simu from 0 <= simu < nsimu:
    
    # Set the magic number
    magic = magics.data[simu*magics.stride]

    # Generate permutation 
    nex = fff_twosample_permutation(<unsigned int*>idx1.data,
                                    <unsigned int*>idx2.data,
                                    n1, n2, &magic)
   
    # Reset the multi-iterator      
    fffpy_multi_iterator_reset(multi)
    
    # Perform the loop 
    idx = simu*t.stride
    while(multi.index < multi.size):
      fff_twosample_apply_permutation(yp, vp, y1, v1, y2, v2, nex,
                                      <unsigned int*>idx1.data,
                                      <unsigned int*>idx2.data)
      t.data[idx] = fff_twosample_stat_mfx_eval(stat, yp, vp)
      fffpy_multi_iterator_update(multi)

  # Delete local structures
  fffpy_multi_iterator_delete(multi)
  fff_vector_delete(magics)
  fff_vector_delete(yp)
  fff_vector_delete(vp)
  fff_array_delete(idx1)
  fff_array_delete(idx2)
  fff_twosample_stat_mfx_delete(stat)

  # Return
  return T
