/*
  @author Alexis Roche
  @date 1997-2009
  
  Intensity-based image registration for 2D or 3D images
  [BETA VERSION].
 
  All computations are fed with the voxel-to-voxel transformation
  relating two images, so you do not need the voxel sizes.
*/

#ifndef ICONIC
#define ICONIC

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
  extern void joint_histogram(double* H, 
			      int clampI, 
			      int clampJ,  
			      PyArrayIterObject* iterI,
			      const PyArrayObject* imJ_padded, 
			      const double* Tvox, 
			      int interp); 


  extern double correlation_coefficient(const double* H, int clampI, int clampJ);
  extern double correlation_ratio(const double* H, int clampI, int clampJ); 
  extern double correlation_ratio_L1(const double* H, double* hI, int clampI, int clampJ); 
  extern double joint_entropy(const double* H, int clampI, int clampJ);
  extern double conditional_entropy(const double* H, double* hJ, int clampI, int clampJ); 
  extern double mutual_information(const double* H, double* hI, int clampI, double* hJ, int clampJ);
  extern double normalized_mutual_information(const double* H, double* hI, int clampI, double* hJ, int clampJ); 
  extern double supervised_mutual_information(const double* H, const double* F, 
					      double* fI, int clampI, double* fJ, int clampJ); 

  
  /*!
    \brief Apply a transformation to an image
    \param im_resampled output image
    \param im input image
    \param Tvox voxel transformation 

    If \a Tvox goes from source to target, use this function to
    resample the target.  Otherwise, pass the inverse of \a Tvox. Tvox
    assumed C-contiguous 16-sized. 
  */ 
  extern void cubic_spline_resample(PyArrayObject* im_resampled, 
				    const PyArrayObject* im, 
				    const double* Tvox); 



#ifdef __cplusplus
}
#endif

#endif
