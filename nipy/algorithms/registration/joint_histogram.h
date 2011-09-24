/*
  @author Alexis Roche
  @date 1997-2009
  
  Intensity-based texture analysis and image registration for 2D or 3D
  images [BETA VERSION].
 
  All computations are fed with the voxel-to-voxel transformation
  relating two images, so you do not need the voxel sizes.
*/

#ifndef JOINT_HISTOGRAM
#define JOINT_HISTOGRAM

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <numpy/arrayobject.h>


  /* Numpy import */
  extern void joint_histogram_import_array(void);


  /* 
     Update a pre-allocated joint histogram. Important notice: in all
     computations, H will be assumed C-contiguous.

     This means that it is contiguous and that, in C convention
     (row-major order, i.e. column indices are fastest):
     
     i (source intensities) are row indices 
     j (target intensities) are column indices

     interp: 
       0 - PV interpolation
       1 - TRILINEAR interpolation 
       <0 - RANDOM interpolation with seed=-interp
  */ 
  extern int joint_histogram(PyArrayObject* H, 
			     unsigned int clampI, 
			     unsigned int clampJ,  
			     PyArrayIterObject* iterI,
			     const PyArrayObject* imJ_padded, 
			     const PyArrayObject* Tvox, 
			     long interp); 

  extern int L1_moments(double* n_, double* median_, double* dev_, 
			const PyArrayObject* H);


#ifdef __cplusplus
}
#endif

#endif
