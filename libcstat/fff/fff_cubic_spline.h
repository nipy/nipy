/*!

  \file fff_cubic_spline.h
  \brief Cubic spline transformation and interpolation.
  \author Alexis Roche
  \date 2003

  Compute the cubic spline coefficients of regularly sampled signals
  (1d to 4d) and perform interpolation. The cubic spline transform is
  implemented from the recursive algorithm described in:

  M. Unser, "Splines : a perfect fit for signal/image processing ",
  IEEE Signal Processing Magazine, Nov. 1999.  Web page:
  http://bigwww.epfl.ch/publications/unser9902.html Please check the
  erratum.

*/


#ifndef FFF_CUBIC_SPLINE
#define FFF_CUBIC_SPLINE

#ifdef __cplusplus
extern "C" {
#endif

#include "fff_vector.h"
#include "fff_array.h"

  /*! 
    \brief Cubic spline basis function
    \param x input value 
  */
  extern double fff_cubic_spline_basis ( double x ); 
  /*! 
    \brief Cubic spline transform of a one-dimensional signal 
    \param src input signal 
    \param res output signal (same size)
  */
  extern void fff_cubic_spline_transform ( fff_vector* res, const fff_vector* src );
  /*! 
    \brief Cubic spline transform of an image
    \param src input image 
    \param res output image (same size), should be of data type FFF_DOUBLE
    \param work auxiliary vector

    The output image \a res must be the same size as \a src.  \a work
    must be allocated with size at least the maximum dimension of \a
    res or \a src.  While the input image may have any data type, the
    output image \a res has the mandatory data type FFF_DOUBLE, as
    spline coefficients are generally not discrete (even for discrete
    signals).
  */
  extern void fff_cubic_spline_transform_image ( fff_array* res, const fff_array* src, fff_vector* work );
  /*! 
    \brief Sample a one-dimensional cubic spline at a given location 
    \param x interpolation point 
    \param coef input spline coefficients
  */
  extern double fff_cubic_spline_sample ( double x, const fff_vector* coef );  
  /*! 
    \brief Sample a cubic spline image at a given spatial location 
    \param x interpolation point first coordinate
    \param y interpolation point second coordinate
    \param z interpolation point third coordinate
    \param t interpolation point fourth coordinate
    \param coef input spline coefficients

    This function works regardless of the actual dimension of the
    image. Only the coordinates that are required are taken into
    account. For instance, parameter \c t is not taken into account if
    \c coef is a 3D image.

  */
  extern double fff_cubic_spline_sample_image ( double x, double y, double z, double t, 
						const fff_array* coef );

    

#ifdef __cplusplus
}
#endif

#endif
