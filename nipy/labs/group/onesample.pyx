# -*- Mode: Python -*-  Not really, but the syntax is close enough


"""
Routines for massively univariate random-effect and mixed-effect analysis.

Author: Alexis Roche, 2008.
"""

__version__ = '0.1'


# Includes
from fff cimport *

# Exports from fff_onesample_stat.h
cdef extern from "fff_onesample_stat.h":

  ctypedef enum fff_onesample_stat_flag:
    FFF_ONESAMPLE_EMPIRICAL_MEAN = 0
    FFF_ONESAMPLE_EMPIRICAL_MEDIAN = 1
    FFF_ONESAMPLE_STUDENT = 2
    FFF_ONESAMPLE_LAPLACE = 3
    FFF_ONESAMPLE_TUKEY = 4
    FFF_ONESAMPLE_SIGN_STAT = 5
    FFF_ONESAMPLE_WILCOXON = 6
    FFF_ONESAMPLE_ELR = 7 
    FFF_ONESAMPLE_GRUBB = 8
    FFF_ONESAMPLE_EMPIRICAL_MEAN_MFX = 10,
    FFF_ONESAMPLE_EMPIRICAL_MEDIAN_MFX = 11,
    FFF_ONESAMPLE_STUDENT_MFX = 12,
    FFF_ONESAMPLE_SIGN_STAT_MFX = 15,
    FFF_ONESAMPLE_WILCOXON_MFX = 16, 
    FFF_ONESAMPLE_ELR_MFX = 17, 
    FFF_ONESAMPLE_GAUSSIAN_MEAN_MFX = 19

  ctypedef struct fff_onesample_stat:
    pass

  ctypedef struct fff_onesample_stat_mfx:
    unsigned int niter
    unsigned int constraint
    

  fff_onesample_stat* fff_onesample_stat_new(size_t n, fff_onesample_stat_flag flag, double base)
  void fff_onesample_stat_delete(fff_onesample_stat* thisone)
  double fff_onesample_stat_eval(fff_onesample_stat* thisone, fff_vector* x)

  fff_onesample_stat_mfx* fff_onesample_stat_mfx_new(size_t n, fff_onesample_stat_flag flag, double base) 
  void fff_onesample_stat_mfx_delete(fff_onesample_stat_mfx* thisone) 
  double fff_onesample_stat_mfx_eval(fff_onesample_stat_mfx* thisone, fff_vector* x, fff_vector* vx)

  void fff_onesample_stat_mfx_pdf_fit(fff_vector* w, fff_vector* z,
                                      fff_onesample_stat_mfx* thisone, 
                                      fff_vector* x, fff_vector* vx)
  void fff_onesample_stat_gmfx_pdf_fit(double* mu, double* v, 
                                       fff_onesample_stat_mfx* thisone, 
                                       fff_vector* x, fff_vector* vx)

  void fff_onesample_permute_signs(fff_vector* xx, fff_vector* x, double magic)

# Initialize numpy
fffpy_import_array()
import_array()
import numpy as np


# Stat dictionary
stats = {'mean': FFF_ONESAMPLE_EMPIRICAL_MEAN,
         'median': FFF_ONESAMPLE_EMPIRICAL_MEDIAN,
         'student': FFF_ONESAMPLE_STUDENT,
         'laplace': FFF_ONESAMPLE_LAPLACE,
         'tukey': FFF_ONESAMPLE_TUKEY,
         'sign': FFF_ONESAMPLE_SIGN_STAT,
         'wilcoxon': FFF_ONESAMPLE_WILCOXON,
         'elr': FFF_ONESAMPLE_ELR,
         'grubb': FFF_ONESAMPLE_GRUBB,
         'mean_mfx': FFF_ONESAMPLE_EMPIRICAL_MEAN_MFX,
         'median_mfx': FFF_ONESAMPLE_EMPIRICAL_MEDIAN_MFX,
         'mean_gauss_mfx': FFF_ONESAMPLE_GAUSSIAN_MEAN_MFX,
         'student_mfx': FFF_ONESAMPLE_STUDENT_MFX, 
         'sign_mfx': FFF_ONESAMPLE_SIGN_STAT_MFX,
         'wilcoxon_mfx': FFF_ONESAMPLE_WILCOXON_MFX,
         'elr_mfx': FFF_ONESAMPLE_ELR_MFX}


# Test stat without mixed-effect correction
def stat(ndarray Y, id='student', double base=0.0,
         int axis=0, ndarray Magics=None):
  """
  T = stat(Y, id='student', base=0.0, axis=0, magics=None).
  
  Compute a one-sample test statistic over a number of deterministic
  or random permutations. 
  """
  cdef fff_vector *y, *t, *magics, *yp
  cdef fff_onesample_stat* stat
  cdef fff_onesample_stat_flag flag_stat = stats[id]
  cdef unsigned int n
  cdef unsigned long int simu, nsimu, idx
  cdef double magic
  cdef fffpy_multi_iterator* multi

  # Get number of observations
  n = <unsigned int>Y.shape[axis]

  # Read out magic numbers
  if Magics is None:
    magics = fff_vector_new(1)
    magics.data[0] = 0 ## Just to make sure
  else:
    magics = fff_vector_fromPyArray(Magics)
    
  # Create output array
  nsimu = magics.size
  dims = [Y.shape[i] for i in range(Y.ndim)]
  dims[axis] = nsimu 
  T = np.zeros(dims)

  # Create local structure
  stat = fff_onesample_stat_new(n, flag_stat, base)
  yp = fff_vector_new(n)

  # Multi-iterator 
  multi = fffpy_multi_iterator_new(2, axis, <void*>Y, <void*>T)

  # Vector views
  y = multi.vector[0]
  t = multi.vector[1]
  
  # Loop
  for simu from 0 <= simu < nsimu:
    
    # Set the magic number
    magic = magics.data[simu*magics.stride]
      
    # Reset the multi-iterator
    fffpy_multi_iterator_reset(multi); 
    
    # Perform the loop 
    idx = simu*t.stride
    while(multi.index < multi.size):
      fff_onesample_permute_signs(yp, y, magic)
      t.data[idx] = fff_onesample_stat_eval(stat, yp)
      fffpy_multi_iterator_update(multi)

  # Free memory 
  fffpy_multi_iterator_delete(multi)
  fff_vector_delete(yp)
  fff_vector_delete(magics)
  fff_onesample_stat_delete(stat)

  # Return
  return T


def stat_mfx(ndarray Y, ndarray V, id='student_mfx', double base=0.0,
             int axis=0, ndarray Magics=None, unsigned int niter=5):
  """
  T = stat_mfx(Y, V, id='student_mfx', base=0.0, axis=0, magics=None, niter=5).
  
  Compute a one-sample test statistic, with mixed-effect correction,
  over a number of deterministic or random permutations.
  """
  cdef fff_vector *y, *v, *t, *magics, *yp
  cdef fff_onesample_stat_mfx* stat
  cdef fff_onesample_stat_flag flag_stat = stats[id]
  cdef int n
  cdef unsigned long int nsimu_max, simu, idx
  cdef double magic
  cdef fffpy_multi_iterator* multi

  # Get number of observations
  n = <int>Y.shape[axis]

  # Read out magic numbers
  if Magics is None:
    magics = fff_vector_new(1)
    magics.data[0] = 0 ## Just to make sure
  else:
    magics = fff_vector_fromPyArray(Magics)
    
  # Create output array
  nsimu = magics.size
  dims = [Y.shape[i] for i in range(Y.ndim)]
  dims[axis] = nsimu 
  T = np.zeros(dims)

  # Create local structure
  stat = fff_onesample_stat_mfx_new(n, flag_stat, base)
  stat.niter = niter
  yp = fff_vector_new(n)

  # Multi-iterator 
  multi = fffpy_multi_iterator_new(3, axis, <void*>Y, <void*>V, <void*>T)

  # Vector views
  y = multi.vector[0]
  v = multi.vector[1]
  t = multi.vector[2]
  
  # Loop  
  for simu from 0 <= simu < nsimu:

    # Set the magic number
    magic = magics.data[simu*magics.stride]
    
    # Reset the multi-iterator      
    fffpy_multi_iterator_reset(multi)
    
    # Perform the loop 
    idx = simu*t.stride
    while(multi.index < multi.size):
      fff_onesample_permute_signs(yp, y, magic)
      t.data[idx] = fff_onesample_stat_mfx_eval(stat, yp, v)
      fffpy_multi_iterator_update(multi)
      

  # Free memory
  fffpy_multi_iterator_delete(multi)
  fff_vector_delete(yp)
  fff_vector_delete(magics)
  fff_onesample_stat_mfx_delete(stat)
  
  # Return
  return T



def pdf_fit_mfx(ndarray Y, ndarray V, int axis=0, int niter=5, int constraint=0, double base=0.0):
  """
  (W, Z) = pdf_fit_mfx(data=Y, vardata=V, axis=0, niter=5, constraint=False, base=0.0).
  
  Comments to follow.
  """
  cdef fff_vector *y, *v, *w, *z
  cdef fff_onesample_stat_mfx* stat
  cdef fffpy_multi_iterator* multi
  cdef int n = Y.shape[axis]

  # Create output array
  dims = [Y.shape[i] for i in range(Y.ndim)]
  W = np.zeros(dims)
  Z = np.zeros(dims)

  # Create local structure
  stat = fff_onesample_stat_mfx_new(n, FFF_ONESAMPLE_EMPIRICAL_MEAN_MFX, base)
  stat.niter = niter
  stat.constraint = constraint

  # Multi-iterator 
  multi = fffpy_multi_iterator_new(4, axis, <void*>Y, <void*>V, <void*>W, <void*>Z)

  # Create views on nd-arrays
  y = multi.vector[0]
  v = multi.vector[1]
  w = multi.vector[2]
  z = multi.vector[3]

  # Loop
  while(multi.index < multi.size):
    fff_onesample_stat_mfx_pdf_fit(w, z, stat, y, v)
    fffpy_multi_iterator_update(multi)
  

  # Delete local structures
  fffpy_multi_iterator_delete(multi)
  fff_onesample_stat_mfx_delete(stat)

  # Return
  return W, Z


def pdf_fit_gmfx(ndarray Y, ndarray V, int axis=0, int niter=5, int constraint=0, double base=0.0):
  """
  (MU, S2) = pdf_fit_gmfx(data=Y, vardata=V, axis=0, niter=5, constraint=False, base=0.0).
  
  Comments to follow.
  """
  cdef fff_vector *y, *v, *mu, *s2
  cdef fff_onesample_stat_mfx* stat
  cdef fffpy_multi_iterator* multi
  cdef int n = Y.shape[axis]
  
  # Create output array
  dims = [Y.shape[i] for i in range(Y.ndim)]
  dims[axis] = 1
  MU = np.zeros(dims)
  S2 = np.zeros(dims)

  # Create local structure
  stat = fff_onesample_stat_mfx_new(n, FFF_ONESAMPLE_STUDENT_MFX, base)
  stat.niter = niter
  stat.constraint = constraint

  # Multi-iterator 
  multi = fffpy_multi_iterator_new(4, axis, <void*>Y, <void*>V, <void*>MU, <void*>S2)

  # Create views on nd-arrays
  y = multi.vector[0]
  v = multi.vector[1]
  mu = multi.vector[2]
  s2 = multi.vector[3]

  # Loop
  while(multi.index < multi.size):
    fff_onesample_stat_gmfx_pdf_fit(mu.data, s2.data, stat, y, v)
    fffpy_multi_iterator_update(multi)
  

  # Delete local structures
  fffpy_multi_iterator_delete(multi)
  fff_onesample_stat_mfx_delete(stat)

  # Return
  return MU, S2

  
