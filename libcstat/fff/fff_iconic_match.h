/*!
  \file fff_iconic_match.h
  \brief Intensity-based image matching
  \bug This library is still under construction
  \author Alexis Roche
  \date 1997-2008
  
  Intensity-based image registration for 2D or 3D images [BETA VERSION]. 
 
  All computations in this library are fed with the voxel-to-voxel
  transformation relating two images, so you do not need the voxel
  sizes.  
*/

#ifndef FFF_ICONIC_MATCH
#define FFF_ICONIC_MATCH

#ifdef __cplusplus
extern "C" {
#endif

#include "fff_vector.h"
#include "fff_array.h"

  typedef struct fff_imatch{
    
    fff_array* imI; /* Source image in signed short format */  
    fff_array* imJ; /* Target image in signed short format */  
    fff_array* imJ_padded; /* Enlarged target image with borders padded with -1 */  
    
    int clampI; 
    int clampJ; 
    double* H;  /* Joint histogram */
    double* hI; /* In-overlap target histogram */
    double* hJ; /* In-overlap source histogram */

    int owner_images; 
    int owner_histograms; 
    
  } fff_imatch;

  

  /* 
     Create the joint histogram structure. Important notice: in all
     computations, H will be assumed C-contiguous. 

     This means that it is contiguous and that, in C convention
     (row-major order, i.e. column indices are fastest):
     
     i (source intensities) are row indices 
     j (target intensities) are column indices

     interp: 
       0 - PV interpolation
       1 - TRILINEAR interpolation 
       <0 - RANDOM interpolation with seed=-interp

       Returns the number of in-mask voxels in imI (transformation-independent). 

  */ 

  extern void fff_imatch_joint_hist( double* H, int clampI, int clampJ,  
				     const fff_array* imI,
				     const fff_array* imJ_padded, 
				     const double* Tvox, 
				     int interp ); 

  extern unsigned int fff_imatch_source_npoints( const fff_array* imI ); 


  extern double fff_imatch_cc( const double* H, int clampI, int clampJ );
  extern double fff_imatch_cr( const double* H, int clampI, int clampJ ); 
  extern double fff_imatch_crL1( const double* H, double* hI, int clampI, int clampJ ); 
  extern double fff_imatch_joint_ent( const double* H, int clampI, int clampJ );
  extern double fff_imatch_cond_ent( const double* H, double* hJ, int clampI, int clampJ ); 
  extern double fff_imatch_mi( const double* H, double* hI, int clampI, double* hJ, int clampJ );
  extern double fff_imatch_norma_mi( const double* H, double* hI, int clampI, double* hJ, int clampJ ); 
  extern double fff_imatch_n_cc( const double* H, int clampI, int clampJ, double norma );
  extern double fff_imatch_n_cr( const double* H, int clampI, int clampJ, double norma ); 
  extern double fff_imatch_n_crL1( const double* H, double* hI, int clampI, int clampJ, double norma ); 
  extern double fff_imatch_n_mi( const double* H, double* hI, int clampI, double* hJ, int clampJ, double norma );
  
  extern double fff_imatch_supervised_mi( const double* H, const double* F, 
					  double* fI, int clampI, double* fJ, int clampJ ); 
  extern double fff_imatch_n_supervised_mi( const double* H, const double* F, 
					    double* fI, int clampI, double* fJ, int clampJ, double norma ); 

  
  /*!
    \brief Apply a transformation to an image
    \param im_resampled output image
    \param im input image
    \param Tvox voxel transformation 

    If \a Tvox goes from source to target, use this function to
    resample the target.  Otherwise, pass the inverse of \a Tvox. Tvox
    assumed contiguous 16-sized. 
  */ 
  extern void fff_imatch_resample( fff_array* im_resampled, 
				   const fff_array* im, 
				   const double* Tvox ); 


  extern fff_imatch* fff_imatch_new ( const fff_array* imI,
				      const fff_array* imJ,
				      double thI,
				      double thJ,
				      int clampI, 
				      int clampJ );
  
  extern void fff_imatch_delete( fff_imatch* imatch );

#ifdef __cplusplus
}
#endif

#endif
