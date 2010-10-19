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

#define L2_moments(h, size, res)		\
  L2_moments_with_stride(h, size, 1, res)
#define L1_moments(h, size, res)		\
  L1_moments_with_stride(h, size, 1, res)


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
  extern int joint_histogram(PyArrayIterObject* H, 
			     unsigned int clampI, 
			     unsigned int clampJ,  
			     PyArrayIterObject* iterI,
			     const PyArrayObject* imJ_padded, 
			     const PyArrayObject* Tvox, 
			     int interp); 


  extern double entropy(const double* h, unsigned int size, double* n); 
  extern void L2_moments_with_stride(const double * h, unsigned int size, 
				     unsigned int stride, double* res); 
  extern void L1_moments_with_stride(const double * h, unsigned int size, 
				     unsigned int stride, double* res);
  extern double correlation_coefficient(const double* H, 
					unsigned int clampI, 
					unsigned int clampJ, 
					double* n); 
  extern double correlation_ratio(const double* H, 
				  unsigned int clampI, 
				  unsigned int clampJ, 
				  double* n); 
  extern double correlation_ratio_L1(const double* H, 
				     double* hI, 
				     unsigned int clampI, 
				     unsigned int clampJ, 
				     double* n);
  extern double joint_entropy(const double* H,
                              unsigned int clampI,
                              unsigned int clampJ);
  extern double conditional_entropy(const double* H,
                                    double* hJ,
                                    unsigned int clampI,
                                    unsigned int clampJ);
  extern double mutual_information(const double* H, 
				   double* hI, 
				   unsigned int clampI, 
				   double* hJ,
				   unsigned int clampJ, 
				   double* n); 
  extern double normalized_mutual_information(const double* H, 
					      double* hI,
					      unsigned int clampI, 
					      double* hJ, 
					      unsigned int clampJ, 
					      double* n); 
  extern double supervised_mutual_information(const double* H, 
					      const double* F, 
					      double* fI, 
					      unsigned int clampI, 
					      double* fJ, 
					      unsigned int clampJ, 
					      double* n);
        

        



#ifdef __cplusplus
}
#endif

#endif
